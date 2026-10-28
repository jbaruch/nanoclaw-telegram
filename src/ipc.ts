import {
  isFileSystemError,
  hasErrno,
  isRecoverableSqliteError,
  isTelegramError,
  IpcRouteUnavailableError,
} from './operational-errors.js';
import fs from 'fs';
import path from 'path';

import {
  createIpcInbox,
  IpcInboxHold,
  prepareIpcQuarantineRoot,
} from './ipc-inbox.js';

import { parseTaskSchedule } from './task-schedule.js';

import { DATA_DIR, IPC_POLL_INTERVAL } from './config.js';
import { AvailableGroup } from './container-runner.js';
import { createTask, deleteTask, getTaskById, updateTask } from './db.js';
import { isValidGroupFolder } from './group-folder.js';
import { logger } from './logger.js';
import { RegisteredGroup } from './types.js';

export { IpcRouteUnavailableError } from './operational-errors.js';

export interface IpcDeps {
  sendMessage: (jid: string, text: string) => Promise<void>;
  sendReaction?: (
    jid: string,
    messageId: string | undefined,
    emoji: string,
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
  ) => void;
  onTasksChanged: () => void;
}

/** Validate agent-written JSON before string operations or native effects. */
function validateIpcRequest(
  data: unknown,
  channel: 'messages' | 'tasks',
): void {
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new SyntaxError('IPC request must be an object');
  }
  const fields = data as Record<string, unknown>;
  if (fields.schema_version !== 1)
    throw new SyntaxError(
      'Unsupported IPC request schema; update the co-shipped writer/reader and inspect preserved evidence before any replay.',
    );
  if (typeof fields.type !== 'string')
    throw new SyntaxError('IPC type must be a string');
  const requirements: Record<string, readonly string[]> = {
    message: ['chatJid', 'text'],
    react_to_message: ['chatJid', 'emoji'],
    schedule_task: ['prompt', 'schedule_type', 'schedule_value', 'targetJid'],
    pause_task: ['taskId'],
    resume_task: ['taskId'],
    cancel_task: ['taskId'],
    update_task: ['taskId'],
    refresh_groups: [],
    register_group: ['jid', 'name', 'folder', 'trigger'],
  };
  if (!Object.hasOwn(requirements, fields.type))
    throw new SyntaxError('Unsupported IPC request type');
  const isMessage =
    fields.type === 'message' || fields.type === 'react_to_message';
  if (isMessage !== (channel === 'messages'))
    throw new SyntaxError('IPC request is in the wrong channel');
  for (const required of requirements[fields.type]) {
    if (typeof fields[required] !== 'string' || fields[required].length === 0)
      throw new SyntaxError(`IPC ${required} is required`);
  }
  if (
    fields.type === 'update_task' &&
    !['prompt', 'script', 'schedule_type', 'schedule_value'].some(
      (field) => fields[field] !== undefined,
    )
  )
    throw new SyntaxError('IPC task update requires a changed field');
  if (
    fields.schedule_type !== undefined &&
    !['cron', 'interval', 'once'].includes(String(fields.schedule_type))
  )
    throw new SyntaxError('Unsupported IPC schedule type');
  if (
    fields.context_mode !== undefined &&
    fields.context_mode !== 'group' &&
    fields.context_mode !== 'isolated'
  )
    throw new SyntaxError('Unsupported IPC context mode');

  for (const field of [
    'chatJid',
    'sender',
    'timestamp',
    'createdBy',
    'messageId',
    'emoji',
    'text',
    'taskId',
    'prompt',
    'schedule_type',
    'schedule_value',
    'context_mode',
    'groupFolder',
    'targetJid',
    'jid',
    'name',
    'folder',
    'trigger',
  ]) {
    if (fields[field] !== undefined && typeof fields[field] !== 'string') {
      throw new SyntaxError(`IPC ${field} must be a string`);
    }
  }
  if (
    fields.script !== undefined &&
    fields.script !== null &&
    typeof fields.script !== 'string'
  ) {
    throw new SyntaxError('IPC script must be a string or null');
  }
  if (
    fields.requiresTrigger !== undefined &&
    typeof fields.requiresTrigger !== 'boolean'
  ) {
    throw new SyntaxError('IPC requiresTrigger must be a boolean');
  }
  if (fields.containerConfig !== undefined) {
    const config = fields.containerConfig;
    if (!config || typeof config !== 'object' || Array.isArray(config))
      throw new SyntaxError('IPC containerConfig must be an object');
    if (
      'timeout' in config &&
      (typeof config.timeout !== 'number' || !Number.isFinite(config.timeout))
    )
      throw new SyntaxError('IPC timeout must be a finite number');
    if ('additionalMounts' in config) {
      if (!Array.isArray(config.additionalMounts))
        throw new SyntaxError('IPC additionalMounts must be an array');
      for (const mount of config.additionalMounts) {
        if (
          !mount ||
          typeof mount !== 'object' ||
          Array.isArray(mount) ||
          typeof mount.hostPath !== 'string' ||
          (mount.containerPath !== undefined &&
            typeof mount.containerPath !== 'string') ||
          (mount.readonly !== undefined && typeof mount.readonly !== 'boolean')
        )
          throw new SyntaxError('IPC mount fields have invalid types');
      }
    }
  }
}

function quarantineIpcFile(
  filePath: string,
  errorDir: string,
  sourceGroup: string,
  channel: 'messages' | 'tasks',
  file: string,
  error: unknown,
): void {
  try {
    prepareIpcQuarantineRoot(path.dirname(errorDir));
    const parent = fs.openSync(
      path.dirname(errorDir),
      fs.constants.O_RDONLY |
        fs.constants.O_DIRECTORY |
        fs.constants.O_NOFOLLOW,
    );
    using _parentLease = { [Symbol.dispose]: () => fs.closeSync(parent) };
    fs.fsyncSync(parent);

    // link() reserves the destination without overwriting earlier evidence.
    for (let sequence = 0; sequence < 1000; sequence++) {
      const destination = path.join(
        errorDir,
        `${sourceGroup}-${channel}-${sequence}-${file}`,
      );
      try {
        fs.linkSync(filePath, destination);
      } catch (collision) {
        if (hasErrno(collision, 'EEXIST')) continue;
        throw collision;
      }
      const destinationDirectory = fs.openSync(
        errorDir,
        fs.constants.O_RDONLY |
          fs.constants.O_DIRECTORY |
          fs.constants.O_NOFOLLOW,
      );
      try {
        fs.fsyncSync(destinationDirectory);
      } finally {
        fs.closeSync(destinationDirectory);
      }
      fs.unlinkSync(filePath);
      const sourceDirectory = fs.openSync(
        path.dirname(filePath),
        fs.constants.O_RDONLY |
          fs.constants.O_DIRECTORY |
          fs.constants.O_NOFOLLOW,
      );
      try {
        fs.fsyncSync(sourceDirectory);
      } finally {
        fs.closeSync(sourceDirectory);
      }
      return;
    }
    throw new Error(
      'IPC quarantine capacity exhausted; retain the source request and repair the error directory.',
    );
  } catch (quarantineError) {
    throw new AggregateError(
      [error, quarantineError],
      'IPC processing and quarantine failed; repair IPC storage before restarting the host.',
      { cause: quarantineError },
    );
  }
}

let ipcWatcherRunning = false;
let ipcLifecycle: Promise<void> | undefined;

export function startIpcWatcher(deps: IpcDeps): Promise<void> {
  if (ipcWatcherRunning) {
    logger.debug('IPC watcher already running, skipping duplicate start');
    return ipcLifecycle!;
  }
  ipcWatcherRunning = true;

  const ipcBaseDir = path.join(DATA_DIR, 'ipc');
  fs.mkdirSync(ipcBaseDir, { recursive: true, mode: 0o755 });

  let rejectLifecycle!: (error: unknown) => void;
  ipcLifecycle = new Promise<void>((_resolve, reject) => {
    rejectLifecycle = reject;
  });
  const poll = () => {
    void processIpcFiles().then(undefined, rejectLifecycle);
  };
  const processIpcFiles = async () => {
    try {
      // The base and group parents are host-owned; only channel entries are
      // agent-controlled. lstat never follows a group alias.
      const inbox = createIpcInbox(ipcBaseDir);
      const registeredGroups = deps.registeredGroups();
      const folderIsMain = new Map<string, boolean>();
      for (const group of Object.values(registeredGroups)) {
        if (group.isMain) folderIsMain.set(group.folder, true);
      }
      for (const sourceGroup of fs.readdirSync(ipcBaseDir)) {
        if (!isValidGroupFolder(sourceGroup)) continue;
        const stat = fs.lstatSync(path.join(ipcBaseDir, sourceGroup));
        if (!stat.isDirectory() || stat.isSymbolicLink()) {
          logger.error('IPC group alias refused; original entry retained.');
          continue;
        }
        try {
          inbox.capture(sourceGroup, 'messages');
          inbox.capture(sourceGroup, 'tasks');
        } catch (error) {
          if (!(error instanceof IpcInboxHold) && !isFileSystemError(error))
            throw error;
          logger.error(
            'IPC channel capture held; preserve its batch and inspect runtime-owned storage.',
          );
        }
      }
      const { claims, held } = inbox.pending();
      if (held)
        logger.error(
          { held },
          'IPC batches held for inspection; unknown dispatches are never replayed.',
        );
      for (const claim of claims) {
        const { sourceGroup, channel, file } = claim;
        const isMain = folderIsMain.get(sourceGroup) === true;
        try {
          const data = JSON.parse(claim.read());
          validateIpcRequest(data, channel);
          claim.dispatch();
          if (channel === 'messages') {
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
            } else if (data.type === 'message' && data.chatJid && data.text) {
              // Authorization: verify this group can send to this chatJid
              const targetGroup = registeredGroups[data.chatJid];
              if (
                isMain ||
                (targetGroup && targetGroup.folder === sourceGroup)
              ) {
                await deps.sendMessage(data.chatJid, data.text);
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
          } else {
            await processTaskIpc(data, sourceGroup, isMain, deps);
          }
          claim.complete();
        } catch (err) {
          quarantineIpcFile(
            claim.filePath,
            path.join(ipcBaseDir, 'errors'),
            sourceGroup,
            channel,
            file,
            err,
          );
          if (
            !(err instanceof SyntaxError) &&
            !(err instanceof IpcRouteUnavailableError) &&
            !isFileSystemError(err) &&
            !isTelegramError(err) &&
            !isRecoverableSqliteError(err)
          )
            throw err;
          logger.error(
            { sourceGroup, channel },
            'IPC request failed and was quarantined; inspect it before resubmitting.',
          );
        }
      }
    } catch (error) {
      if (!isFileSystemError(error)) throw error;
      logger.error(
        'IPC storage scan failed; requests are retained for a later scan.',
      );
    } finally {
      setTimeout(poll, IPC_POLL_INTERVAL);
    }
  };

  poll();
  logger.info('IPC watcher started (per-group namespaces)');
  return ipcLifecycle;
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

        const nextRun = parseTaskSchedule(
          scheduleType,
          data.schedule_value,
        ).nextRun;

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

        if (
          data.schedule_type !== undefined ||
          data.schedule_value !== undefined
        ) {
          const prospective = { ...task, ...updates };
          updates.next_run = parseTaskSchedule(
            prospective.schedule_type,
            prospective.schedule_value,
          ).nextRun;
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
      } else {
        logger.warn(
          { data },
          'Invalid register_group request - missing required fields',
        );
      }
      break;

    default:
      logger.warn({ type: data.type }, 'Unknown IPC task type');
  }
}
