import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';

import { CronExpressionParser } from 'cron-parser';

import {
  ASSISTANT_NAME,
  DATA_DIR,
  GROUPS_DIR,
  IPC_POLL_INTERVAL,
  TIMEZONE,
} from './config.js';
import { sendPoolMessage } from './channels/telegram.js';
import { AvailableGroup } from './container-runner.js';
import {
  createTask,
  deleteAllSessions,
  deleteTask,
  getTaskById,
  storeMessage,
  updateTask,
} from './db.js';
import { isValidGroupFolder } from './group-folder.js';
import { logger } from './logger.js';
import { RegisteredGroup } from './types.js';

export interface IpcDeps {
  sendReaction?: (
    jid: string,
    messageId: string | undefined,
    emoji: string,
  ) => Promise<void>;
  sendMessage: (
    jid: string,
    text: string,
    replyToMessageId?: string,
  ) => Promise<string | void>;
  pinMessage?: (jid: string, messageId: string) => Promise<void>;
  sendFile?: (
    jid: string,
    filePath: string,
    caption?: string,
    replyToMessageId?: string,
  ) => Promise<void>;
  registeredGroups: () => Record<string, RegisteredGroup>;
  registerGroup: (jid: string, group: RegisteredGroup) => void;
  syncGroups: (force: boolean) => Promise<void>;
  getAvailableGroups: () => AvailableGroup[];
  writeGroupsSnapshot: (
    groupFolder: string,
    isMain: boolean,
    availableGroups: AvailableGroup[],
    registeredJids: Set<string>,
    isTrusted?: boolean,
  ) => void;
  onTasksChanged: () => void;
  nukeSession: (groupFolder: string) => void;
}

let ipcWatcherRunning = false;

export function startIpcWatcher(deps: IpcDeps): void {
  if (ipcWatcherRunning) {
    logger.debug('IPC watcher already running, skipping duplicate start');
    return;
  }
  ipcWatcherRunning = true;

  const ipcBaseDir = path.join(DATA_DIR, 'ipc');
  fs.mkdirSync(ipcBaseDir, { recursive: true });

  const processIpcFiles = async () => {
    // Scan all group IPC directories (identity determined by directory)
    let groupFolders: string[];
    try {
      groupFolders = fs.readdirSync(ipcBaseDir).filter((f) => {
        const stat = fs.statSync(path.join(ipcBaseDir, f));
        return stat.isDirectory() && f !== 'errors';
      });
    } catch (err) {
      logger.error({ err }, 'Error reading IPC base directory');
      setTimeout(processIpcFiles, IPC_POLL_INTERVAL);
      return;
    }

    const registeredGroups = deps.registeredGroups();

    // Build folder→isMain lookup from registered groups
    const folderIsMain = new Map<string, boolean>();
    for (const group of Object.values(registeredGroups)) {
      if (group.isMain) folderIsMain.set(group.folder, true);
    }

    for (const sourceGroup of groupFolders) {
      const isMain = folderIsMain.get(sourceGroup) === true;
      const messagesDir = path.join(ipcBaseDir, sourceGroup, 'messages');
      const tasksDir = path.join(ipcBaseDir, sourceGroup, 'tasks');

      // Process messages from this group's IPC directory
      try {
        if (fs.existsSync(messagesDir)) {
          const messageFiles = fs
            .readdirSync(messagesDir)
            .filter((f) => f.endsWith('.json'));
          for (const file of messageFiles) {
            const filePath = path.join(messagesDir, file);
            try {
              const stat = fs.statSync(filePath);
              if (stat.size > 1_048_576) {
                logger.warn(
                  { file, sourceGroup, size: stat.size },
                  'IPC file exceeds 1MB limit, moving to errors',
                );
                const errorDir = path.join(ipcBaseDir, 'errors');
                fs.mkdirSync(errorDir, { recursive: true });
                fs.renameSync(
                  filePath,
                  path.join(errorDir, `${sourceGroup}-${file}`),
                );
                continue;
              }
              const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
              if (
                data.type === 'react_to_message' &&
                data.chatJid &&
                data.emoji &&
                deps.sendReaction
              ) {
                const targetGroup = registeredGroups[data.chatJid];
                if (
                  isMain ||
                  (targetGroup && targetGroup.folder === sourceGroup)
                ) {
                  await deps.sendReaction(
                    data.chatJid,
                    data.messageId || undefined,
                    data.emoji,
                  );
                  logger.info(
                    {
                      chatJid: data.chatJid,
                      emoji: data.emoji,
                      sourceGroup,
                    },
                    'IPC reaction sent',
                  );
                } else {
                  logger.warn(
                    { chatJid: data.chatJid, sourceGroup },
                    'Unauthorized IPC reaction attempt blocked',
                  );
                }
              } else if (
                data.type === 'send_file' &&
                data.chatJid &&
                data.filePath &&
                deps.sendFile
              ) {
                const targetGroup = registeredGroups[data.chatJid];
                if (
                  isMain ||
                  (targetGroup && targetGroup.folder === sourceGroup)
                ) {
                  // Translate container path to host path
                  const containerPath: string = data.filePath;
                  let hostPath: string;
                  if (containerPath.startsWith('/workspace/group/')) {
                    hostPath = path.join(
                      GROUPS_DIR,
                      sourceGroup,
                      containerPath.replace('/workspace/group/', ''),
                    );
                  } else if (containerPath.startsWith('/workspace/trusted/')) {
                    hostPath = path.join(
                      process.cwd(),
                      'trusted',
                      containerPath.replace('/workspace/trusted/', ''),
                    );
                  } else {
                    logger.warn(
                      { containerPath, sourceGroup },
                      'send_file: path outside allowed mounts',
                    );
                    fs.unlinkSync(filePath);
                    continue;
                  }

                  if (fs.existsSync(hostPath)) {
                    await deps.sendFile(
                      data.chatJid,
                      hostPath,
                      data.caption,
                      data.replyToMessageId,
                    );
                    logger.info(
                      { chatJid: data.chatJid, hostPath, sourceGroup },
                      'IPC file sent',
                    );
                  } else {
                    logger.warn(
                      { hostPath, containerPath, sourceGroup },
                      'send_file: file not found on host',
                    );
                  }
                }
              } else if (data.type === 'message' && data.chatJid && data.text) {
                // Strip <internal> tags — if nothing remains, skip silently
                const cleanText = data.text
                  .replace(/<internal>[\s\S]*?<\/internal>/g, '')
                  .trim();
                if (!cleanText) {
                  logger.debug(
                    { sourceGroup },
                    'IPC message suppressed (all internal)',
                  );
                  fs.unlinkSync(filePath);
                  continue;
                }

                // Authorization: verify this group can send to this chatJid
                const targetGroup = registeredGroups[data.chatJid];
                if (
                  isMain ||
                  (targetGroup && targetGroup.folder === sourceGroup)
                ) {
                  if (data.sender && data.chatJid.startsWith('tg:')) {
                    await sendPoolMessage(
                      data.chatJid,
                      cleanText,
                      data.sender,
                      sourceGroup,
                    );
                  } else {
                    const sentMsgId = await deps.sendMessage(
                      data.chatJid,
                      cleanText,
                      data.replyToMessageId,
                    );
                    // Pin the message if requested
                    if (data.pin && sentMsgId && deps.pinMessage) {
                      await deps.pinMessage(data.chatJid, sentMsgId);
                    }
                  }
                  // Store bot response so heartbeat can track answered messages
                  storeMessage({
                    id: `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                    chat_jid: data.chatJid,
                    sender: data.sender || ASSISTANT_NAME,
                    sender_name: data.sender || ASSISTANT_NAME,
                    content: cleanText,
                    timestamp: new Date().toISOString(),
                    is_from_me: true,
                    is_bot_message: true,
                    reply_to_message_id: data.replyToMessageId,
                  });
                  logger.info(
                    { chatJid: data.chatJid, sourceGroup },
                    'IPC message sent',
                  );
                } else {
                  logger.warn(
                    { chatJid: data.chatJid, sourceGroup },
                    'Unauthorized IPC message attempt blocked',
                  );
                }
              }
              fs.unlinkSync(filePath);
            } catch (err) {
              logger.error(
                { file, sourceGroup, err },
                'Error processing IPC message',
              );
              const errorDir = path.join(ipcBaseDir, 'errors');
              fs.mkdirSync(errorDir, { recursive: true });
              fs.renameSync(
                filePath,
                path.join(errorDir, `${sourceGroup}-${file}`),
              );
            }
          }
        }
      } catch (err) {
        logger.error(
          { err, sourceGroup },
          'Error reading IPC messages directory',
        );
      }

      // Process tasks from this group's IPC directory
      try {
        if (fs.existsSync(tasksDir)) {
          const taskFiles = fs
            .readdirSync(tasksDir)
            .filter((f) => f.endsWith('.json'));
          for (const file of taskFiles) {
            const filePath = path.join(tasksDir, file);
            try {
              const stat = fs.statSync(filePath);
              if (stat.size > 1_048_576) {
                logger.warn(
                  { file, sourceGroup, size: stat.size },
                  'IPC task file exceeds 1MB limit, moving to errors',
                );
                const errorDir = path.join(ipcBaseDir, 'errors');
                fs.mkdirSync(errorDir, { recursive: true });
                fs.renameSync(
                  filePath,
                  path.join(errorDir, `${sourceGroup}-${file}`),
                );
                continue;
              }
              const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
              // Pass source group identity to processTaskIpc for authorization
              await processTaskIpc(data, sourceGroup, isMain, deps);
              fs.unlinkSync(filePath);
            } catch (err) {
              logger.error(
                { file, sourceGroup, err },
                'Error processing IPC task',
              );
              const errorDir = path.join(ipcBaseDir, 'errors');
              fs.mkdirSync(errorDir, { recursive: true });
              fs.renameSync(
                filePath,
                path.join(errorDir, `${sourceGroup}-${file}`),
              );
            }
          }
        }
      } catch (err) {
        logger.error({ err, sourceGroup }, 'Error reading IPC tasks directory');
      }
    }

    setTimeout(processIpcFiles, IPC_POLL_INTERVAL);
  };

  processIpcFiles();
  logger.info('IPC watcher started (per-group namespaces)');
}

export async function processTaskIpc(
  data: {
    type: string;
    taskId?: string;
    prompt?: string;
    schedule_type?: string;
    schedule_value?: string;
    context_mode?: string;
    script?: string;
    groupFolder?: string;
    chatJid?: string;
    targetJid?: string;
    // For register_group
    jid?: string;
    name?: string;
    folder?: string;
    trigger?: string;
    requiresTrigger?: boolean;
    containerConfig?: RegisteredGroup['containerConfig'];
    // For host operations / github_backup / promote_staging / sessionize
    requestId?: string;
    message?: string;
    tileName?: string;
    skillName?: string;
    slug?: string;
    filter?: Record<string, boolean>;
    dryRun?: boolean;
  },
  sourceGroup: string, // Verified identity from IPC directory
  isMain: boolean, // Verified from directory path
  deps: IpcDeps,
): Promise<void> {
  const registeredGroups = deps.registeredGroups();

  switch (data.type) {
    case 'schedule_task':
      if (
        data.prompt &&
        data.schedule_type &&
        data.schedule_value &&
        data.targetJid
      ) {
        // Resolve the target group from JID
        const targetJid = data.targetJid as string;
        const targetGroupEntry = registeredGroups[targetJid];

        if (!targetGroupEntry) {
          logger.warn(
            { targetJid },
            'Cannot schedule task: target group not registered',
          );
          break;
        }

        const targetFolder = targetGroupEntry.folder;

        // Authorization: non-main groups can only schedule for themselves
        if (!isMain && targetFolder !== sourceGroup) {
          logger.warn(
            { sourceGroup, targetFolder },
            'Unauthorized schedule_task attempt blocked',
          );
          break;
        }

        const scheduleType = data.schedule_type as 'cron' | 'interval' | 'once';

        let nextRun: string | null = null;
        if (scheduleType === 'cron') {
          try {
            const interval = CronExpressionParser.parse(data.schedule_value, {
              tz: TIMEZONE,
            });
            nextRun = interval.next().toISOString();
          } catch {
            logger.warn(
              { scheduleValue: data.schedule_value },
              'Invalid cron expression',
            );
            break;
          }
        } else if (scheduleType === 'interval') {
          const MIN_INTERVAL_MS = 60_000;
          const ms = parseInt(data.schedule_value, 10);
          if (isNaN(ms) || ms < MIN_INTERVAL_MS) {
            logger.warn(
              { scheduleValue: data.schedule_value, minMs: MIN_INTERVAL_MS },
              'Invalid interval: must be at least 60s',
            );
            break;
          }
          nextRun = new Date(Date.now() + ms).toISOString();
        } else if (scheduleType === 'once') {
          const date = new Date(data.schedule_value);
          if (isNaN(date.getTime())) {
            logger.warn(
              { scheduleValue: data.schedule_value },
              'Invalid timestamp',
            );
            break;
          }
          nextRun = date.toISOString();
        }

        const taskId =
          data.taskId ||
          `task-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const contextMode =
          data.context_mode === 'group' || data.context_mode === 'isolated'
            ? data.context_mode
            : 'isolated';
        createTask({
          id: taskId,
          group_folder: targetFolder,
          chat_jid: targetJid,
          prompt: data.prompt,
          script: data.script || null,
          schedule_type: scheduleType,
          schedule_value: data.schedule_value,
          context_mode: contextMode,
          next_run: nextRun,
          status: 'active',
          created_at: new Date().toISOString(),
        });
        logger.info(
          { taskId, sourceGroup, targetFolder, contextMode },
          'Task created via IPC',
        );
        deps.onTasksChanged();
      }
      break;

    case 'pause_task':
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (task && (isMain || task.group_folder === sourceGroup)) {
          updateTask(data.taskId, { status: 'paused' });
          logger.info(
            { taskId: data.taskId, sourceGroup },
            'Task paused via IPC',
          );
          deps.onTasksChanged();
        } else {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task pause attempt',
          );
        }
      }
      break;

    case 'resume_task':
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (task && (isMain || task.group_folder === sourceGroup)) {
          updateTask(data.taskId, { status: 'active' });
          logger.info(
            { taskId: data.taskId, sourceGroup },
            'Task resumed via IPC',
          );
          deps.onTasksChanged();
        } else {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task resume attempt',
          );
        }
      }
      break;

    case 'cancel_task':
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (task && (isMain || task.group_folder === sourceGroup)) {
          deleteTask(data.taskId);
          logger.info(
            { taskId: data.taskId, sourceGroup },
            'Task cancelled via IPC',
          );
          deps.onTasksChanged();
        } else {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task cancel attempt',
          );
        }
      }
      break;

    case 'update_task':
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (!task) {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Task not found for update',
          );
          break;
        }
        if (!isMain && task.group_folder !== sourceGroup) {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task update attempt',
          );
          break;
        }

        const updates: Parameters<typeof updateTask>[1] = {};
        if (data.prompt !== undefined) updates.prompt = data.prompt;
        if (data.script !== undefined) updates.script = data.script || null;
        if (data.schedule_type !== undefined)
          updates.schedule_type = data.schedule_type as
            | 'cron'
            | 'interval'
            | 'once';
        if (data.schedule_value !== undefined)
          updates.schedule_value = data.schedule_value;

        // Recompute next_run if schedule changed
        if (data.schedule_type || data.schedule_value) {
          const updatedTask = {
            ...task,
            ...updates,
          };
          if (updatedTask.schedule_type === 'cron') {
            try {
              const interval = CronExpressionParser.parse(
                updatedTask.schedule_value,
                { tz: TIMEZONE },
              );
              updates.next_run = interval.next().toISOString();
            } catch {
              logger.warn(
                { taskId: data.taskId, value: updatedTask.schedule_value },
                'Invalid cron in task update',
              );
              break;
            }
          } else if (updatedTask.schedule_type === 'interval') {
            const MIN_INTERVAL_MS = 60_000;
            const ms = parseInt(updatedTask.schedule_value, 10);
            if (!isNaN(ms) && ms >= MIN_INTERVAL_MS) {
              updates.next_run = new Date(Date.now() + ms).toISOString();
            } else if (!isNaN(ms)) {
              logger.warn(
                {
                  taskId: data.taskId,
                  value: updatedTask.schedule_value,
                  minMs: MIN_INTERVAL_MS,
                },
                'Invalid interval in task update: must be at least 60s',
              );
              break;
            }
          }
        }

        updateTask(data.taskId, updates);
        logger.info(
          { taskId: data.taskId, sourceGroup, updates },
          'Task updated via IPC',
        );
        deps.onTasksChanged();
      }
      break;

    case 'refresh_groups':
      // Only main group can request a refresh
      if (isMain) {
        logger.info(
          { sourceGroup },
          'Group metadata refresh requested via IPC',
        );
        await deps.syncGroups(true);
        // Write updated snapshot immediately
        const availableGroups = deps.getAvailableGroups();
        deps.writeGroupsSnapshot(
          sourceGroup,
          true,
          availableGroups,
          new Set(Object.keys(registeredGroups)),
        );
      } else {
        logger.warn(
          { sourceGroup },
          'Unauthorized refresh_groups attempt blocked',
        );
      }
      break;

    case 'register_group':
      // Only main group can register new groups
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized register_group attempt blocked',
        );
        break;
      }
      if (data.jid && data.name && data.folder && data.trigger) {
        if (!isValidGroupFolder(data.folder)) {
          logger.warn(
            { sourceGroup, folder: data.folder },
            'Invalid register_group request - unsafe folder name',
          );
          break;
        }
        // Defense in depth: agent cannot set isMain via IPC.
        // Preserve isMain from the existing registration so IPC config
        // updates (e.g. adding additionalMounts) don't strip the flag.
        const existingGroup = registeredGroups[data.jid];
        deps.registerGroup(data.jid, {
          name: data.name,
          folder: data.folder,
          trigger: data.trigger,
          added_at: new Date().toISOString(),
          containerConfig: data.containerConfig,
          requiresTrigger: data.requiresTrigger,
          isMain: existingGroup?.isMain,
        });
        // Refresh snapshot so available_groups.json reflects new trust config immediately
        const availableGroups = deps.getAvailableGroups();
        deps.writeGroupsSnapshot(
          sourceGroup,
          true,
          availableGroups,
          new Set(Object.keys(registeredGroups)),
        );
      } else {
        logger.warn(
          { data },
          'Invalid register_group request - missing required fields',
        );
      }
      break;

    case 'nuke_session':
      if (data.groupFolder) {
        logger.info(
          { groupFolder: data.groupFolder, sourceGroup },
          'Session nuke requested via IPC',
        );
        deps.nukeSession(sourceGroup);
      }
      break;

    // --- Named host operations ---

    case 'sync_tripit':
      if (data.requestId) {
        const groupDir = path.resolve(process.cwd(), 'groups', sourceGroup);
        const scriptPath = path.join(groupDir, 'scripts', 'sync-tripit.sh');
        if (!fs.existsSync(scriptPath)) {
          const errPath = path.join(
            DATA_DIR,
            'ipc',
            sourceGroup,
            'input',
            `_script_result_${data.requestId}.json`,
          );
          fs.writeFileSync(
            errPath,
            JSON.stringify({ error: 'sync-tripit.sh not found' }),
          );
          break;
        }

        logger.info({ sourceGroup }, 'Running sync_tripit');

        const { readEnvFile: readSyncEnv } = await import('./env.js');
        const syncVars = readSyncEnv([
          'TRIPIT_ICAL_URL',
          'TRIPIT_IGNORE_TRIPS',
          'TRIPIT_IGNORE_KEYWORDS',
          'RECLAIM_API_TOKEN',
          'GOOGLE_CLIENT_ID',
          'GOOGLE_CLIENT_SECRET',
          'GOOGLE_REFRESH_TOKEN',
        ]);
        const syncEnv: Record<string, string> = {
          PATH: process.env.PATH || '/usr/bin:/bin',
          HOME: process.env.HOME || '/root',
          TZ: process.env.TZ || 'UTC',
          ...Object.fromEntries(Object.entries(syncVars).filter(([, v]) => v)),
        };

        const scriptContent = fs.readFileSync(scriptPath, 'utf-8');
        const patchedContent = scriptContent.replace(
          /\/workspace\/group/g,
          groupDir,
        );
        const tmpScript = path.join(groupDir, '.tmp_host_sync-tripit.sh');
        fs.writeFileSync(tmpScript, patchedContent);

        execFile(
          'bash',
          [tmpScript],
          {
            cwd: groupDir,
            env: syncEnv,
            timeout: 120_000,
            maxBuffer: 1024 * 1024,
          },
          (error, stdout, stderr) => {
            const resultPath = path.join(
              DATA_DIR,
              'ipc',
              sourceGroup,
              'input',
              `_script_result_${data.requestId}.json`,
            );
            if (error) {
              logger.error(
                { sourceGroup, error: error.message, stderr },
                'sync_tripit failed',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
            } else {
              logger.info(
                { sourceGroup, stdoutLen: stdout.length },
                'sync_tripit completed',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({ stdout, stderr: stderr || undefined }),
              );
            }
            try {
              fs.unlinkSync(tmpScript);
            } catch {
              /* best effort */
            }
          },
        );
      }
      break;

    case 'fetch_trakt_history':
      if (data.requestId) {
        const groupDir = path.resolve(process.cwd(), 'groups', sourceGroup);
        const scriptPath = path.join(
          groupDir,
          'scripts',
          'trakt-watch-history.py',
        );
        if (!fs.existsSync(scriptPath)) {
          const errPath = path.join(
            DATA_DIR,
            'ipc',
            sourceGroup,
            'input',
            `_script_result_${data.requestId}.json`,
          );
          fs.writeFileSync(
            errPath,
            JSON.stringify({ error: 'trakt-watch-history.py not found' }),
          );
          break;
        }

        logger.info({ sourceGroup }, 'Running fetch_trakt_history');

        const { readEnvFile: readTraktEnv } = await import('./env.js');
        const traktVars = readTraktEnv([
          'TRAKT_CLIENT_ID',
          'TRAKT_ACCESS_TOKEN',
        ]);
        const traktEnv: Record<string, string> = {
          PATH: process.env.PATH || '/usr/bin:/bin',
          HOME: process.env.HOME || '/root',
          TZ: process.env.TZ || 'UTC',
          ...Object.fromEntries(Object.entries(traktVars).filter(([, v]) => v)),
        };

        const scriptContent = fs.readFileSync(scriptPath, 'utf-8');
        const patchedContent = scriptContent.replace(
          /\/workspace\/group/g,
          groupDir,
        );
        const tmpScript = path.join(
          groupDir,
          '.tmp_host_trakt-watch-history.py',
        );
        fs.writeFileSync(tmpScript, patchedContent);

        execFile(
          'python3',
          [tmpScript],
          {
            cwd: groupDir,
            env: traktEnv,
            timeout: 120_000,
            maxBuffer: 1024 * 1024,
          },
          (error, stdout, stderr) => {
            const resultPath = path.join(
              DATA_DIR,
              'ipc',
              sourceGroup,
              'input',
              `_script_result_${data.requestId}.json`,
            );
            if (error) {
              logger.error(
                { sourceGroup, error: error.message, stderr },
                'fetch_trakt_history failed',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
            } else {
              logger.info(
                { sourceGroup, stdoutLen: stdout.length },
                'fetch_trakt_history completed',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({ stdout, stderr: stderr || undefined }),
              );
            }
            try {
              fs.unlinkSync(tmpScript);
            } catch {
              /* best effort */
            }
          },
        );
      }
      break;

    case 'github_backup':
      if (data.requestId) {
        const backupDir = path.join(
          process.cwd(),
          'groups',
          sourceGroup,
          'backup-repo',
        );
        const resultPath = path.join(
          DATA_DIR,
          'ipc',
          sourceGroup,
          'input',
          `_script_result_${data.requestId}.json`,
        );

        if (!fs.existsSync(backupDir)) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: `backup-repo not found at ${backupDir}` }),
          );
          break;
        }

        const commitMsg =
          data.message || `backup: ${new Date().toISOString().split('T')[0]}`;
        logger.info(
          { sourceGroup, backupDir, commitMsg },
          'Running github_backup',
        );

        // Read GitHub token for push auth
        const { readEnvFile: readBackupEnv } = await import('./env.js');
        const backupEnvVars = readBackupEnv(['GITHUB_TOKEN']);
        const ghToken = backupEnvVars.GITHUB_TOKEN;

        execFile(
          'bash',
          [
            '-c',
            `cd "${backupDir}" && git add -A && git diff --cached --quiet && echo '{"stdout":"Nothing to commit."}' || (git commit -m "${commitMsg.replace(/"/g, '\\"')}" && git push && echo '{"stdout":"Committed and pushed."}')`,
          ],
          {
            timeout: 60_000,
            maxBuffer: 1024 * 1024,
            env: {
              ...process.env,
              ...(ghToken
                ? {
                    GIT_ASKPASS: 'echo',
                    GIT_TERMINAL_PROMPT: '0',
                    GITHUB_TOKEN: ghToken,
                    GIT_CONFIG_COUNT: '1',
                    GIT_CONFIG_KEY_0:
                      'url.https://x-access-token:' +
                      ghToken +
                      '@github.com/.insteadOf',
                    GIT_CONFIG_VALUE_0: 'https://github.com/',
                  }
                : {}),
            },
          },
          (error, stdout, stderr) => {
            if (error) {
              logger.error(
                { sourceGroup, error: error.message, stderr },
                'github_backup failed',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
            } else {
              // stdout is the JSON echo from the bash script
              try {
                const parsed = JSON.parse(stdout.trim().split('\n').pop()!);
                fs.writeFileSync(resultPath, JSON.stringify(parsed));
              } catch {
                fs.writeFileSync(
                  resultPath,
                  JSON.stringify({ stdout: stdout.trim() }),
                );
              }
              logger.info({ sourceGroup }, 'github_backup completed');
            }
          },
        );
      }
      break;

    case 'sessionize_get_event':
      if (data.requestId && data.slug) {
        const sessionizeResultPath = path.join(
          DATA_DIR,
          'ipc',
          sourceGroup,
          'input',
          `_script_result_${data.requestId}.json`,
        );

        const { readEnvFile: readSessionizeEnv } = await import('./env.js');
        const sessionizeVars = readSessionizeEnv(['SESSIONIZE_EVENT_API_KEY']);
        const apiKey = sessionizeVars.SESSIONIZE_EVENT_API_KEY;

        if (!apiKey) {
          fs.writeFileSync(
            sessionizeResultPath,
            JSON.stringify({
              error: 'SESSIONIZE_EVENT_API_KEY not set in .env',
            }),
          );
          break;
        }

        logger.info(
          { slug: data.slug, sourceGroup },
          'Fetching Sessionize event',
        );

        try {
          const url = `https://sessionize.com/api/universal/event?slug=${encodeURIComponent(data.slug)}`;
          const resp = await fetch(url, {
            headers: { 'X-API-KEY': apiKey },
            signal: AbortSignal.timeout(15_000),
          });

          if (!resp.ok) {
            fs.writeFileSync(
              sessionizeResultPath,
              JSON.stringify({
                error: `Sessionize API returned ${resp.status}: ${resp.statusText}`,
              }),
            );
            break;
          }

          const event = (await resp.json()) as Record<string, unknown>;
          const cfpDates = (event.cfpDates ?? {}) as Record<string, unknown>;
          const eventDates = (event.eventDates ?? {}) as Record<
            string,
            unknown
          >;
          const location = (event.location ?? {}) as Record<string, unknown>;
          const timezone = (event.timezone ?? {}) as Record<string, unknown>;
          const expenses = (event.expensesCovered ?? {}) as Record<
            string,
            unknown
          >;
          const normalized = {
            name: event.name,
            cfp_open:
              !!cfpDates.endUtc &&
              new Date(cfpDates.endUtc as string) > new Date(),
            cfp_start: cfpDates.startUtc,
            cfp_end: cfpDates.endUtc,
            cfp_start_local: cfpDates.start,
            cfp_end_local: cfpDates.end,
            conf_start: eventDates.start,
            conf_end: eventDates.end,
            location: location.full,
            city: location.city,
            country: location.country,
            timezone: timezone.iana,
            is_online: event.isOnline,
            website: event.website,
            cfp_url: event.cfpLink || `https://sessionize.com/${data.slug}/`,
            expenses_covered: expenses,
            organizer: event.organizer,
          };

          fs.writeFileSync(
            sessionizeResultPath,
            JSON.stringify({ data: normalized }),
          );
          logger.info({ slug: data.slug }, 'Sessionize event fetched');
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          logger.error(
            { slug: data.slug, error: errMsg },
            'Sessionize fetch failed',
          );
          fs.writeFileSync(
            sessionizeResultPath,
            JSON.stringify({ error: errMsg }),
          );
        }
      }
      break;

    case 'sessionize_open_cfps':
      if (data.requestId) {
        const cfpsResultPath = path.join(
          DATA_DIR,
          'ipc',
          sourceGroup,
          'input',
          `_script_result_${data.requestId}.json`,
        );

        const { readEnvFile: readCfpsEnv } = await import('./env.js');
        const cfpsVars = readCfpsEnv(['SESSIONIZE_SPEAKER_KEY']);
        const speakerKey = cfpsVars.SESSIONIZE_SPEAKER_KEY;

        if (!speakerKey) {
          fs.writeFileSync(
            cfpsResultPath,
            JSON.stringify({ error: 'SESSIONIZE_SPEAKER_KEY not set in .env' }),
          );
          break;
        }

        logger.info({ sourceGroup }, 'Fetching Sessionize open CFPs');

        try {
          const resp = await fetch(
            'https://sessionize.com/api/universal/open-cfps',
            {
              headers: { 'X-API-KEY': speakerKey },
              signal: AbortSignal.timeout(15_000),
            },
          );

          if (!resp.ok) {
            fs.writeFileSync(
              cfpsResultPath,
              JSON.stringify({
                error: `Sessionize API returned ${resp.status}: ${resp.statusText}`,
              }),
            );
            break;
          }

          let events = (await resp.json()) as Array<Record<string, unknown>>;
          const filter = (data.filter ?? {}) as Record<string, boolean>;

          // Apply filters — default: exclude online and user groups
          if (!filter.isOnline) {
            events = events.filter((e) => !e.isOnline);
          }
          if (!filter.isUserGroup) {
            events = events.filter((e) => !e.isUserGroup);
          }

          fs.writeFileSync(cfpsResultPath, JSON.stringify({ data: events }));
          logger.info({ count: events.length }, 'Sessionize open CFPs fetched');
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          logger.error({ error: errMsg }, 'Sessionize open CFPs fetch failed');
          fs.writeFileSync(cfpsResultPath, JSON.stringify({ error: errMsg }));
        }
      }
      break;

    case 'audible_backup':
      if (data.requestId) {
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized audible_backup attempt');
          break;
        }

        const audibleResultPath = path.join(
          DATA_DIR,
          'ipc',
          sourceGroup,
          'input',
          `_script_result_${data.requestId}.json`,
        );

        const dryRun = data.dryRun === true;
        logger.info({ sourceGroup, dryRun }, 'Running audible_backup');

        const dockerArgs = [
          'run',
          '--rm',
          '-v',
          `${path.dirname(process.env.HOST_PROJECT_ROOT || process.cwd())}/.audible:/root/.audible`,
          '-v',
          '/volume1/Google Drive/Audio Books:/library',
          'audible-backup:latest',
          '--json',
          ...(dryRun ? ['--dry-run'] : []),
        ];

        execFile(
          'docker',
          dockerArgs,
          {
            cwd: process.cwd(),
            env: {
              PATH: process.env.PATH || '/usr/bin:/bin',
              HOME: process.env.HOME || '/root',
            },
            timeout: 600_000,
            maxBuffer: 10 * 1024 * 1024,
          },
          (error, stdout, stderr) => {
            if (error) {
              logger.error(
                { sourceGroup, error: error.message, stderr },
                'audible_backup failed',
              );
              fs.writeFileSync(
                audibleResultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
            } else {
              logger.info(
                { sourceGroup, stdoutLen: stdout.length },
                'audible_backup completed',
              );
              // stdout is JSON from backup.py --json; merge stderr (progress logs) into it
              try {
                const parsed = JSON.parse(stdout);
                if (stderr) parsed.logs = stderr.slice(-2000);
                fs.writeFileSync(
                  audibleResultPath,
                  JSON.stringify(parsed),
                );
              } catch {
                fs.writeFileSync(
                  audibleResultPath,
                  JSON.stringify({ raw: stdout, logs: stderr?.slice(-2000) }),
                );
              }
            }
          },
        );
      }
      break;

    case 'promote_staging':
      if (data.requestId && data.tileName && data.skillName) {
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized promote_staging attempt');
          break;
        }

        const promoteResultPath = path.join(
          DATA_DIR,
          'ipc',
          sourceGroup,
          'input',
          `_script_result_${data.requestId}.json`,
        );

        const promoteScript = path.join(
          process.cwd(),
          'scripts',
          'promote-to-tile-repo.sh',
        );

        if (!fs.existsSync(promoteScript)) {
          fs.writeFileSync(
            promoteResultPath,
            JSON.stringify({
              error: 'promote-to-tile-repo.sh not found',
            }),
          );
          break;
        }

        const stagingDir = path.join(
          GROUPS_DIR,
          sourceGroup,
          'staging',
          data.tileName,
        );

        // Read credentials from .env for tile repo push
        const envPath = path.join(process.cwd(), '.env');
        const envContent = fs.existsSync(envPath)
          ? fs.readFileSync(envPath, 'utf-8')
          : '';
        const getEnv = (key: string) =>
          envContent
            .split('\n')
            .find((l) => l.startsWith(`${key}=`))
            ?.split('=')
            .slice(1)
            .join('=') || '';

        logger.info(
          { sourceGroup, tileName: data.tileName, skillName: data.skillName },
          'Running promote_staging',
        );

        execFile(
          'bash',
          [promoteScript, stagingDir, data.tileName, data.skillName],
          {
            timeout: 300_000,
            maxBuffer: 5 * 1024 * 1024,
            env: {
              ...process.env,
              GITHUB_TOKEN: getEnv('GITHUB_TOKEN'),
              TILE_OWNER: getEnv('TILE_OWNER') || 'jbaruch',
              ASSISTANT_NAME: getEnv('ASSISTANT_NAME') || 'AyeAye',
            },
          },
          (error, stdout, stderr) => {
            if (error) {
              logger.error(
                {
                  sourceGroup,
                  error: error.message,
                  stderr: stderr.slice(-500),
                },
                'promote_staging failed',
              );
              fs.writeFileSync(
                promoteResultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
            } else {
              logger.info(
                { sourceGroup },
                'promote_staging pushed to tile repo',
              );
              fs.writeFileSync(
                promoteResultPath,
                JSON.stringify({ stdout: stdout.trim() }),
              );

              // Schedule tessl update + session clear after GHA completes (~5 min)
              setTimeout(() => {
                logger.info('Running post-promote tessl update');
                execFile(
                  'bash',
                  [
                    '-c',
                    'cd /app/tessl-workspace && tessl update --yes --dangerously-ignore-security --agent claude-code 2>&1',
                  ],
                  { timeout: 120_000 },
                  (updateErr, updateStdout) => {
                    if (updateErr) {
                      logger.error(
                        { error: updateErr.message },
                        'Post-promote tessl update failed',
                      );
                    } else {
                      const cleared = deleteAllSessions();
                      logger.info(
                        {
                          sessionsCleared: cleared,
                          output: updateStdout.trim().slice(-200),
                        },
                        'Post-promote tessl update completed — sessions cleared',
                      );
                    }
                  },
                );
              }, 300_000); // 5 minutes
            }
          },
        );
      }
      break;

    default:
      logger.warn({ type: data.type }, 'Unknown IPC task type');
  }
}
