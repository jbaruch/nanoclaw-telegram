import type { DraftStream } from './draft-stream.js';
import {
  isFileSystemError,
  isRecoverableSqliteError,
  isTelegramError,
  isOperationalRunError,
  TelegramPartialDeliveryError,
  TelegramPreviewUncertainError,
} from './operational-errors.js';
import fs from 'fs';
import path from 'path';

import { OneCLI } from '@onecli-sh/sdk';

import {
  ASSISTANT_NAME,
  DEFAULT_TRIGGER,
  getTriggerPattern,
  GROUPS_DIR,
  IDLE_TIMEOUT,
  MAX_MESSAGES_PER_PROMPT,
  ONECLI_URL,
  POLL_INTERVAL,
  TIMEZONE,
} from './config.js';
import './channels/index.js';
import {
  getChannelFactory,
  getRegisteredChannelNames,
} from './channels/registry.js';
import {
  ContainerOutput,
  runContainerAgent,
  writeGroupsSnapshot,
  writeTasksSnapshot,
} from './container-runner.js';
import {
  cleanupOrphans,
  ensureContainerRuntimeRunning,
} from './container-runtime.js';
import {
  getAllChats,
  getAllRegisteredGroups,
  getAllSessions,
  getAllTasks,
  getLastBotMessageTimestamp,
  getMessagesSince,
  getNewMessages,
  getRouterState,
  initDatabase,
  setRegisteredGroup,
  setRouterState,
  setSession,
  storeChatMetadata,
  storeMessage,
} from './db.js';
import { GroupQueue } from './group-queue.js';
import {
  InvalidGroupFolderError,
  resolveGroupFolderPath,
} from './group-folder.js';
import { startIpcWatcher, IpcRouteUnavailableError } from './ipc.js';
import { findChannel, formatMessages, formatOutbound } from './router.js';
import {
  restoreRemoteControl,
  startRemoteControl,
  stopRemoteControl,
} from './remote-control.js';
import {
  isSenderAllowed,
  isTriggerAllowed,
  loadSenderAllowlist,
  shouldDropMessage,
} from './sender-allowlist.js';
import { startSchedulerLoop } from './task-scheduler.js';
import { Channel, NewMessage, RegisteredGroup } from './types.js';
import { logger } from './logger.js';

// Re-export for backwards compatibility during refactor
export { escapeXml, formatMessages } from './router.js';

let lastTimestamp = '';
let sessions: Record<string, string> = {};
let registeredGroups: Record<string, RegisteredGroup> = {};
let lastAgentTimestamp: Record<string, string> = {};
let messageLoopRunning = false;

const channels: Channel[] = [];
const queue = new GroupQueue();

// --- Outbound message retry with exponential backoff ---
const SEND_RETRY_ATTEMPTS = 3;
const SEND_RETRY_BASE_DELAY_MS = 2000;

export async function sendWithRetry(
  ch: Channel,
  jid: string,
  text: string,
  replyToMessageId?: string,
): Promise<void> {
  for (let attempt = 0; attempt < SEND_RETRY_ATTEMPTS; attempt++) {
    try {
      await ch.sendMessage(jid, text, replyToMessageId);
      return;
    } catch (err: unknown) {
      const isLast = attempt === SEND_RETRY_ATTEMPTS - 1;
      const details = err !== null && typeof err === 'object' ? err : {};
      const nested =
        'error' in details &&
        details.error !== null &&
        typeof details.error === 'object'
          ? details.error
          : {};
      const code =
        ('code' in nested ? nested.code : undefined) ||
        ('code' in details ? details.code : '');
      const httpCode =
        'error_code' in details && typeof details.error_code === 'number'
          ? details.error_code
          : 0;
      const isTransient =
        /ETIMEDOUT|ECONNRESET|ECONNREFUSED|EAI_AGAIN|EPIPE/i.test(
          String(code),
        ) || httpCode >= 500;

      if (isLast || !isTransient) throw err;

      const delay = SEND_RETRY_BASE_DELAY_MS * (attempt + 1);
      logger.warn(
        { jid, attempt: attempt + 1, code, delay },
        'Transient send error, retrying',
      );
      await new Promise((r) => setTimeout(r, delay));
    }
  }
}

/** The IPC watcher owns expected route failures and quarantines their request. */
export async function sendIpcMessage(jid: string, text: string): Promise<void> {
  const channel = findChannel(channels, jid);
  if (!channel)
    throw new IpcRouteUnavailableError(
      'No active channel owns the IPC destination; verify the destination and channel connection before resubmitting.',
    );
  await sendDeduplicated(channel, jid, text);
}

// --- Outbound dedup guard ---
const DEDUP_WINDOW_MS = 10_000;
const recentSends = new Map<string, number>();
const pendingSends = new Map<string, Promise<void>>();

function isDuplicateSend(jid: string, text: string): boolean {
  const key = `${jid}::${text}`;
  const now = Date.now();

  // Evict stale entries
  for (const [k, ts] of recentSends) {
    if (now - ts > DEDUP_WINDOW_MS) recentSends.delete(k);
  }

  if (recentSends.has(key)) return true;
  return false;
}

async function sendDeduplicated(
  channel: Channel,
  jid: string,
  text: string,
): Promise<void> {
  if (isDuplicateSend(jid, text)) return;
  const key = `${jid}::${text}`;
  const existing = pendingSends.get(key);
  if (existing) return existing;
  const attempt = sendWithRetry(channel, jid, text);
  pendingSends.set(key, attempt);
  try {
    await attempt;
    recentSends.set(key, Date.now());
  } finally {
    pendingSends.delete(key);
  }
}

export async function sendScheduledMessage(
  jid: string,
  rawText: string,
): Promise<void> {
  const channel = findChannel(channels, jid);
  if (!channel)
    throw new IpcRouteUnavailableError(
      'No active channel owns the scheduled destination; restore its route before the next task run.',
    );
  const text = formatOutbound(rawText);
  if (text) await sendDeduplicated(channel, jid, text);
}

export async function sendIpcReaction(
  jid: string,
  messageId: string | undefined,
  emoji: string,
): Promise<void> {
  const channel = findChannel(channels, jid);
  if (!channel)
    throw new IpcRouteUnavailableError(
      'No active channel owns the reaction destination; verify its route before resubmitting.',
    );
  if (messageId) {
    if (!channel.sendReaction)
      throw new IpcRouteUnavailableError(
        'The destination channel does not support reactions to a native message.',
      );
    await channel.sendReaction(jid, messageId, emoji);
  } else {
    if (!channel.reactToLatestMessage)
      throw new IpcRouteUnavailableError(
        'The destination channel cannot select a native reaction target.',
      );
    await channel.reactToLatestMessage(jid, emoji);
  }
}

const onecli = new OneCLI({ url: ONECLI_URL });

function ensureOneCLIAgent(jid: string, group: RegisteredGroup): void {
  if (group.isMain) return;
  const identifier = group.folder.toLowerCase().replace(/_/g, '-');
  onecli.ensureAgent({ name: group.name, identifier }).then(
    (res) => {
      logger.info(
        { jid, identifier, created: res.created },
        'OneCLI agent ensured',
      );
    },
    (err) => {
      logger.debug(
        { jid, identifier, err: String(err) },
        'OneCLI agent ensure skipped',
      );
    },
  );
}

function loadState(): void {
  lastTimestamp = getRouterState('last_timestamp') || '';
  const agentTs = getRouterState('last_agent_timestamp');
  try {
    lastAgentTimestamp = agentTs ? JSON.parse(agentTs) : {};
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    logger.warn('Corrupted last_agent_timestamp in DB, resetting');
    lastAgentTimestamp = {};
  }
  sessions = getAllSessions();
  registeredGroups = getAllRegisteredGroups();
  logger.info(
    { groupCount: Object.keys(registeredGroups).length },
    'State loaded',
  );
}

/**
 * Return the message cursor for a group, recovering from the last bot reply
 * if lastAgentTimestamp is missing (new group, corrupted state, restart).
 */
function getOrRecoverCursor(chatJid: string): string {
  const existing = lastAgentTimestamp[chatJid];
  if (existing) return existing;

  const botTs = getLastBotMessageTimestamp(chatJid, ASSISTANT_NAME);
  if (botTs) {
    logger.info(
      { chatJid, recoveredFrom: botTs },
      'Recovered message cursor from last bot reply',
    );
    lastAgentTimestamp[chatJid] = botTs;
    saveState();
    return botTs;
  }
  return '';
}

function saveState(): void {
  setRouterState('last_timestamp', lastTimestamp);
  setRouterState('last_agent_timestamp', JSON.stringify(lastAgentTimestamp));
}

function registerGroup(jid: string, group: RegisteredGroup): void {
  let groupDir: string;
  try {
    groupDir = resolveGroupFolderPath(group.folder);
  } catch (err) {
    if (!(err instanceof InvalidGroupFolderError)) throw err;
    logger.warn(
      { jid, folder: group.folder, err },
      'Rejecting group registration with invalid folder',
    );
    return;
  }

  registeredGroups[jid] = group;
  setRegisteredGroup(jid, group);

  // Create group folder
  fs.mkdirSync(path.join(groupDir, 'logs'), { recursive: true });

  // Copy CLAUDE.md template into the new group folder so agents have
  // identity and instructions from the first run.  (Fixes #1391)
  const groupMdFile = path.join(groupDir, 'CLAUDE.md');
  if (!fs.existsSync(groupMdFile)) {
    const templateFile = path.join(
      GROUPS_DIR,
      group.isMain ? 'main' : 'global',
      'CLAUDE.md',
    );
    if (fs.existsSync(templateFile)) {
      let content = fs.readFileSync(templateFile, 'utf-8');
      if (ASSISTANT_NAME !== 'Andy') {
        content = content.replace(/^# Andy$/m, `# ${ASSISTANT_NAME}`);
        content = content.replace(/You are Andy/g, `You are ${ASSISTANT_NAME}`);
      }
      fs.writeFileSync(groupMdFile, content);
      logger.info({ folder: group.folder }, 'Created CLAUDE.md from template');
    }
  }

  // Ensure a corresponding OneCLI agent exists (best-effort, non-blocking)
  ensureOneCLIAgent(jid, group);

  logger.info(
    { jid, name: group.name, folder: group.folder },
    'Group registered',
  );
}

/**
 * Get available groups list for the agent.
 * Returns groups ordered by most recent activity.
 */
export function getAvailableGroups(): import('./container-runner.js').AvailableGroup[] {
  const chats = getAllChats();
  const registeredJids = new Set(Object.keys(registeredGroups));

  return chats
    .filter((c) => c.jid !== '__group_sync__' && c.is_group)
    .map((c) => ({
      jid: c.jid,
      name: c.name,
      lastActivity: c.last_message_time,
      isRegistered: registeredJids.has(c.jid),
    }));
}

/** @internal - exported for testing */
export function _setRegisteredGroups(
  groups: Record<string, RegisteredGroup>,
): void {
  registeredGroups = groups;
}

/**
 * Process all pending messages for a group.
 * Called by the GroupQueue when it's this group's turn.
 */
export async function processGroupMessages(chatJid: string): Promise<boolean> {
  const group = registeredGroups[chatJid];
  if (!group) return true;

  const channel = findChannel(channels, chatJid);
  if (!channel) {
    logger.warn({ chatJid }, 'No channel owns JID, skipping messages');
    return true;
  }

  const isMainGroup = group.isMain === true;

  const missedMessages = getMessagesSince(
    chatJid,
    getOrRecoverCursor(chatJid),
    ASSISTANT_NAME,
    MAX_MESSAGES_PER_PROMPT,
  );

  if (missedMessages.length === 0) return true;

  // For non-main groups, check if trigger is required and present
  if (!isMainGroup && group.requiresTrigger !== false) {
    const triggerPattern = getTriggerPattern(group.trigger);
    const allowlistCfg = loadSenderAllowlist();
    const hasTrigger = missedMessages.some(
      (m) =>
        triggerPattern.test(m.content.trim()) &&
        (m.is_from_me || isTriggerAllowed(chatJid, m.sender, allowlistCfg)),
    );
    if (!hasTrigger) return true;
  }

  const prompt = formatMessages(missedMessages, TIMEZONE);

  // Advance cursor so the piping path in startMessageLoop won't re-fetch
  // these messages. Save the old cursor so we can roll back on error.
  const previousCursor = lastAgentTimestamp[chatJid] || '';
  lastAgentTimestamp[chatJid] =
    missedMessages[missedMessages.length - 1].timestamp;
  saveState();

  logger.info(
    { group: group.name, messageCount: missedMessages.length },
    'Processing messages',
  );

  // Track idle timer for closing stdin when agent is idle
  let idleTimer: ReturnType<typeof setTimeout> | null = null;

  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      logger.debug(
        { group: group.name },
        'Idle timeout, closing container stdin',
      );
      queue.closeStdin(chatJid);
    }, IDLE_TIMEOUT);
  };

  let hadError = false;
  let outputSentToUser = false;

  // Progressive streaming: show live preview on edit-capable channels
  let draftStream: DraftStream | undefined;
  const restoreUndeliveredCursor = () => {
    if (!outputSentToUser) {
      lastAgentTimestamp[chatJid] = previousCursor;
      saveState();
    }
  };

  const [execution] = await Promise.allSettled([
    (async () => {
      await channel.setTyping?.(chatJid, true);
      draftStream = channel.createDraftStream?.(chatJid);
      return runAgent(
        group,
        prompt,
        chatJid,
        async (result) => {
          // Streaming preview — update draft with accumulated text
          if (result.streamText && draftStream) {
            const text = result.streamText
              .replace(/<internal>[\s\S]*?<\/internal>/g, '')
              .trim();
            if (text) await draftStream.update(text);
          }

          // Streaming output callback — called for each agent result
          if (result.result) {
            const raw =
              typeof result.result === 'string'
                ? result.result
                : JSON.stringify(result.result);
            // Strip <internal>...</internal> blocks — agent uses these for internal reasoning
            const text = raw
              .replace(/<internal>[\s\S]*?<\/internal>/g, '')
              .trim();
            logger.info(
              { group: group.name },
              `Agent output: ${raw.length} chars`,
            );
            if (text) {
              try {
                if (draftStream) {
                  const ok = await draftStream.finish(text);
                  if (!ok) {
                    // Text exceeded maxLength — fall back to regular send
                    await sendWithRetry(channel, chatJid, text);
                  }
                } else {
                  await sendWithRetry(channel, chatJid, text);
                }
                outputSentToUser = true;
              } catch (error) {
                if (
                  error instanceof TelegramPartialDeliveryError ||
                  error instanceof TelegramPreviewUncertainError
                )
                  outputSentToUser = true;
                throw error;
              }
            }
            // Only reset idle timer on actual results, not session-update markers (result: null)
            resetIdleTimer();
          }

          if (
            result.status === 'success' &&
            (result.completion !== undefined ||
              (result.result !== null && result.result.trim() !== ''))
          ) {
            queue.notifyIdle(chatJid);
          }

          if (result.status === 'error') {
            hadError = true;
          }
        },
        draftStream?.failure,
      );
    })(),
  ]);
  // Both cleanup actions run even when execution or another cleanup fails.
  if (idleTimer) clearTimeout(idleTimer);
  const cleanup = await Promise.allSettled([
    Promise.resolve().then(() => channel.setTyping?.(chatJid, false)),
    Promise.resolve().then(() =>
      !outputSentToUser ? draftStream?.cancel() : undefined,
    ),
  ]);
  const cleanupErrors = cleanup.flatMap((result) =>
    result.status === 'rejected' ? [result.reason as unknown] : [],
  );
  const uncertainPreview = cleanupErrors.some(
    (error) => error instanceof TelegramPreviewUncertainError,
  );
  if (uncertainPreview) outputSentToUser = true;
  if (execution.status === 'rejected' || cleanupErrors.length > 0) {
    const errors: unknown[] =
      execution.status === 'rejected'
        ? [execution.reason as unknown, ...cleanupErrors]
        : [...cleanupErrors];
    const rollback = await Promise.allSettled([
      Promise.resolve().then(restoreUndeliveredCursor),
    ]);
    errors.push(
      ...rollback.flatMap((result) =>
        result.status === 'rejected' ? [result.reason as unknown] : [],
      ),
    );
    if (uncertainPreview && errors.every(isOperationalRunError)) {
      logger.error(
        { group: group.name },
        'Acknowledged preview retirement is uncertain; preserving cursor without automatic replay. Inspect the preview before retrying.',
      );
      return true;
    }
    if (execution.status === 'rejected' && errors.length === 1) throw errors[0];
    throw new AggregateError(
      errors,
      'Agent execution or cleanup failed; inspect typing, draft and router state before retrying undelivered input.',
      { cause: errors[0] },
    );
  }
  const output = execution.value;

  if (output === 'error' || hadError) {
    // If we already sent output to the user, don't roll back the cursor —
    // the user got their response and re-processing would send duplicates.
    if (outputSentToUser) {
      logger.warn(
        { group: group.name },
        'Agent error after output was sent, skipping cursor rollback to prevent duplicates',
      );
      return true;
    }
    // Roll back cursor so retries can re-process these messages
    lastAgentTimestamp[chatJid] = previousCursor;
    saveState();
    logger.warn(
      { group: group.name },
      'Agent error, rolled back message cursor for retry',
    );
    return false;
  }

  return true;
}

async function runAgent(
  group: RegisteredGroup,
  prompt: string,
  chatJid: string,
  onOutput?: (output: ContainerOutput) => Promise<void>,
  outputFailure?: Promise<{ error: unknown }>,
): Promise<'success' | 'error'> {
  const isMain = group.isMain === true;
  const sessionId = sessions[group.folder];

  // Update tasks snapshot for container to read (filtered by group)
  const tasks = getAllTasks();
  writeTasksSnapshot(
    group.folder,
    isMain,
    tasks.map((t) => ({
      id: t.id,
      groupFolder: t.group_folder,
      prompt: t.prompt,
      script: t.script || undefined,
      schedule_type: t.schedule_type,
      schedule_value: t.schedule_value,
      status: t.status,
      next_run: t.next_run,
    })),
  );

  // Update available groups snapshot (main group only can see all groups)
  const availableGroups = getAvailableGroups();
  writeGroupsSnapshot(
    group.folder,
    isMain,
    availableGroups,
    new Set(Object.keys(registeredGroups)),
  );

  // Wrap onOutput to track session ID from streamed results
  const wrappedOnOutput = onOutput
    ? async (output: ContainerOutput) => {
        if (output.newSessionId) {
          sessions[group.folder] = output.newSessionId;
          setSession(group.folder, output.newSessionId);
        }
        await onOutput(output);
      }
    : undefined;

  try {
    const output = await runContainerAgent(
      group,
      {
        prompt,
        sessionId,
        groupFolder: group.folder,
        chatJid,
        isMain,
        assistantName: ASSISTANT_NAME,
      },
      (proc, containerName) =>
        queue.registerProcess(chatJid, proc, containerName, group.folder),
      wrappedOnOutput,
      outputFailure,
    );

    if (output.newSessionId) {
      sessions[group.folder] = output.newSessionId;
      setSession(group.folder, output.newSessionId);
    }

    if (output.status === 'error') {
      logger.error(
        { group: group.name, error: output.error },
        'Container agent error',
      );
      return 'error';
    }

    return 'success';
  } catch (err) {
    if (!isOperationalRunError(err)) throw err;
    logger.error({ group: group.name, err }, 'Agent error');
    return 'error';
  }
}

export async function startMessageLoop(): Promise<void> {
  if (messageLoopRunning) {
    logger.debug('Message loop already running, skipping duplicate start');
    return;
  }
  messageLoopRunning = true;

  logger.info(`NanoClaw running (default trigger: ${DEFAULT_TRIGGER})`);

  while (true) {
    try {
      const jids = Object.keys(registeredGroups);
      const { messages, newTimestamp } = getNewMessages(
        jids,
        lastTimestamp,
        ASSISTANT_NAME,
      );

      if (messages.length > 0) {
        logger.info({ count: messages.length }, 'New messages');

        // Advance the "seen" cursor for all messages immediately
        lastTimestamp = newTimestamp;
        saveState();

        // Deduplicate by group
        const messagesByGroup = new Map<string, NewMessage[]>();
        for (const msg of messages) {
          const existing = messagesByGroup.get(msg.chat_jid);
          if (existing) {
            existing.push(msg);
          } else {
            messagesByGroup.set(msg.chat_jid, [msg]);
          }
        }

        for (const [chatJid, groupMessages] of messagesByGroup) {
          const group = registeredGroups[chatJid];
          if (!group) continue;

          const channel = findChannel(channels, chatJid);
          if (!channel) {
            logger.warn({ chatJid }, 'No channel owns JID, skipping messages');
            continue;
          }

          const isMainGroup = group.isMain === true;
          const needsTrigger = !isMainGroup && group.requiresTrigger !== false;

          // For non-main groups, only act on trigger messages.
          // Non-trigger messages accumulate in DB and get pulled as
          // context when a trigger eventually arrives.
          if (needsTrigger) {
            const triggerPattern = getTriggerPattern(group.trigger);
            const allowlistCfg = loadSenderAllowlist();
            const hasTrigger = groupMessages.some(
              (m) =>
                triggerPattern.test(m.content.trim()) &&
                (m.is_from_me ||
                  isTriggerAllowed(chatJid, m.sender, allowlistCfg)),
            );
            if (!hasTrigger) continue;
          }

          // Pull all messages since lastAgentTimestamp so non-trigger
          // context that accumulated between triggers is included.
          const allPending = getMessagesSince(
            chatJid,
            getOrRecoverCursor(chatJid),
            ASSISTANT_NAME,
            MAX_MESSAGES_PER_PROMPT,
          );
          const messagesToSend =
            allPending.length > 0 ? allPending : groupMessages;
          const formatted = formatMessages(messagesToSend, TIMEZONE);

          // Typing must succeed before handing input to a live container or
          // committing its cursor; a defect leaves the database input pending.
          await channel.setTyping?.(chatJid, true);
          if (queue.sendMessage(chatJid, formatted)) {
            logger.debug(
              { chatJid, count: messagesToSend.length },
              'Piped messages to active container',
            );
            lastAgentTimestamp[chatJid] =
              messagesToSend[messagesToSend.length - 1].timestamp;
            saveState();
          } else {
            // No active container — enqueue for a new one
            queue.enqueueMessageCheck(chatJid);
          }
        }
      }
    } catch (err) {
      if (
        !isFileSystemError(err) &&
        !isTelegramError(err) &&
        !isRecoverableSqliteError(err)
      )
        throw err;
      logger.error({ err }, 'Error in message loop');
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
  }
}

/**
 * Startup recovery: check for unprocessed messages in registered groups.
 * Handles crash between advancing lastTimestamp and processing messages.
 */
function recoverPendingMessages(): void {
  for (const [chatJid, group] of Object.entries(registeredGroups)) {
    const pending = getMessagesSince(
      chatJid,
      getOrRecoverCursor(chatJid),
      ASSISTANT_NAME,
      MAX_MESSAGES_PER_PROMPT,
    );
    if (pending.length > 0) {
      logger.info(
        { group: group.name, pendingCount: pending.length },
        'Recovery: found unprocessed messages',
      );
      queue.enqueueMessageCheck(chatJid);
    }
  }
}

function ensureContainerSystemRunning(): void {
  ensureContainerRuntimeRunning();
  cleanupOrphans();
}

async function main(): Promise<void> {
  ensureContainerSystemRunning();
  initDatabase();
  logger.info('Database initialized');
  loadState();

  // Ensure OneCLI agents exist for all registered groups.
  // Recovers from missed creates (e.g. OneCLI was down at registration time).
  for (const [jid, group] of Object.entries(registeredGroups)) {
    ensureOneCLIAgent(jid, group);
  }

  restoreRemoteControl();

  // Graceful shutdown handlers
  const shutdown = async (signal: string) => {
    logger.info({ signal }, 'Shutdown signal received');
    await queue.shutdown(10000);
    for (const ch of channels) await ch.disconnect();
    process.exit(0);
  };
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  // Handle /remote-control and /remote-control-end commands
  async function handleRemoteControl(
    command: string,
    chatJid: string,
    msg: NewMessage,
  ): Promise<void> {
    const group = registeredGroups[chatJid];
    if (!group?.isMain) {
      logger.warn(
        { chatJid, sender: msg.sender },
        'Remote control rejected: not main group',
      );
      return;
    }

    const channel = findChannel(channels, chatJid);
    if (!channel) return;

    if (command === '/remote-control') {
      const result = await startRemoteControl(
        msg.sender,
        chatJid,
        process.cwd(),
      );
      if (result.ok) {
        await channel.sendMessage(chatJid, result.url);
      } else {
        await channel.sendMessage(
          chatJid,
          `Remote Control failed: ${result.error}`,
        );
      }
    } else {
      const result = stopRemoteControl();
      if (result.ok) {
        await channel.sendMessage(chatJid, 'Remote Control session ended.');
      } else {
        await channel.sendMessage(chatJid, result.error);
      }
    }
  }

  // Channel callbacks (shared by all channels)
  const channelOpts = {
    onMessage: (chatJid: string, msg: NewMessage) => {
      // Remote control commands — intercept before storage
      const trimmed = msg.content.trim();
      if (trimmed === '/remote-control' || trimmed === '/remote-control-end') {
        handleRemoteControl(trimmed, chatJid, msg).catch((err) =>
          logger.error({ err, chatJid }, 'Remote control command error'),
        );
        return;
      }

      // Sender allowlist drop mode: discard messages from denied senders before storing
      if (!msg.is_from_me && !msg.is_bot_message && registeredGroups[chatJid]) {
        const cfg = loadSenderAllowlist();
        if (
          shouldDropMessage(chatJid, cfg) &&
          !isSenderAllowed(chatJid, msg.sender, cfg)
        ) {
          if (cfg.logDenied) {
            logger.debug(
              { chatJid, sender: msg.sender },
              'sender-allowlist: dropping message (drop mode)',
            );
          }
          return;
        }
      }
      storeMessage(msg);
    },
    onChatMetadata: (
      chatJid: string,
      timestamp: string,
      name?: string,
      channel?: string,
      isGroup?: boolean,
    ) => storeChatMetadata(chatJid, timestamp, name, channel, isGroup),
    registeredGroups: () => registeredGroups,
  };

  // Create and connect all registered channels.
  // Each channel self-registers via the barrel import above.
  // Factories return null when credentials are missing, so unconfigured channels are skipped.
  for (const channelName of getRegisteredChannelNames()) {
    const factory = getChannelFactory(channelName)!;
    const channel = factory(channelOpts);
    if (!channel) {
      logger.warn(
        { channel: channelName },
        'Channel installed but credentials missing — skipping. Check .env or re-run the channel skill.',
      );
      continue;
    }
    channels.push(channel);
    await channel.connect();
  }
  if (channels.length === 0) {
    logger.fatal('No channels connected');
    process.exit(1);
  }

  // Start subsystems (independently of connection handler)
  const schedulerLifecycle = startSchedulerLoop({
    registeredGroups: () => registeredGroups,
    getSessions: () => sessions,
    queue,
    onProcess: (groupJid, proc, containerName, groupFolder) =>
      queue.registerProcess(groupJid, proc, containerName, groupFolder),
    sendMessage: sendScheduledMessage,
  });
  const ipcLifecycle = startIpcWatcher({
    sendMessage: sendIpcMessage,
    sendReaction: sendIpcReaction,
    registeredGroups: () => registeredGroups,
    registerGroup,
    syncGroups: async (force: boolean) => {
      await Promise.all(
        channels
          .filter((ch) => ch.syncGroups)
          .map((ch) => ch.syncGroups!(force)),
      );
    },
    getAvailableGroups,
    writeGroupsSnapshot: (gf, im, ag, rj) =>
      writeGroupsSnapshot(gf, im, ag, rj),
    onTasksChanged: () => {
      const tasks = getAllTasks();
      const taskRows = tasks.map((t) => ({
        id: t.id,
        groupFolder: t.group_folder,
        prompt: t.prompt,
        script: t.script || undefined,
        schedule_type: t.schedule_type,
        schedule_value: t.schedule_value,
        status: t.status,
        next_run: t.next_run,
      }));
      for (const group of Object.values(registeredGroups)) {
        writeTasksSnapshot(group.folder, group.isMain === true, taskRows);
      }
    },
  });
  queue.setProcessMessagesFn(processGroupMessages);
  recoverPendingMessages();
  // Background failures reach main's existing outer process boundary.
  await Promise.race([
    queue.waitForFailure(),
    schedulerLifecycle,
    ipcLifecycle,
    startMessageLoop(),
  ]);
}

// Guard: only run when executed directly, not when imported by tests
const isDirectRun =
  process.argv[1] &&
  new URL(import.meta.url).pathname ===
    new URL(`file://${process.argv[1]}`).pathname;

if (isDirectRun) {
  // outer-boundary-process-contract: the service owner detects failure through
  // stderr and a nonzero orchestrator exit. This handler logs and exits 1;
  // propagation alone reaches the legacy rejection reporter, which only logs
  // and could leave the process alive after a background lifecycle failed.
  main().catch((err) => {
    logger.error(
      { err },
      'NanoClaw runtime failed; inspect the error before restarting',
    );
    process.exit(1);
  });
}
