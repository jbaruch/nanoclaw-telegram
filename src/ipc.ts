import fs from 'fs';
import path from 'path';

import { DATA_DIR, IPC_POLL_INTERVAL } from './config.js';
import { SqliteError } from 'better-sqlite3';
import { GrammyError, HttpError } from 'grammy';

import { isFsErrorWithCode } from './fs-errors.js';
import { AvailableGroup } from './container-runner.js';
import type { ContainerStatus } from './group-queue.js';
import { registerCoreIpcHandlers } from './ipc-handlers/index.js';
import {
  dispatchIpcMessage,
  type IpcMessagePayload,
} from './ipc-message-registry.js';
import { dispatchIpcTask, type IpcTaskPayload } from './ipc-registry.js';
import { logger } from './logger.js';
import { RegisteredGroup } from './types.js';

// Errno codes the IPC poller's best-effort fs ops (readdir, stat, unlink,
// rename over group IPC dirs) may legitimately raise, incl. path-shape
// ENOTDIR/ELOOP/ENAMETOOLONG. Anything else is a real defect and propagates.
const IPC_FS_CODES = [
  'ENOENT',
  'EACCES',
  'EPERM',
  'EISDIR',
  'EBUSY',
  'EROFS',
  'ENOSPC',
  'ENOTDIR',
  'ELOOP',
  'ENAMETOOLONG',
];

/**
 * Recoverable failures when processing one IPC message/task file: a
 * malformed payload (SyntaxError), a persistence failure (SqliteError), a
 * Telegram send/transport failure (GrammyError / HttpError), or an fs errno.
 * Those quarantine the one bad file and let the poller continue. Anything
 * else (a TypeError or other programming defect) is not data-poison and
 * propagates so the bug surfaces.
 */
function isRecoverableIpcError(err: unknown): boolean {
  return (
    err instanceof SyntaxError ||
    // Only constraint-class SQLite errors are per-file data poison (a bad
    // payload violating a NOT NULL / UNIQUE / FK). Infrastructure faults
    // (SQLITE_CORRUPT / BUSY / LOCKED / READONLY / SCHEMA) are NOT per-file
    // recoverable and must propagate.
    (err instanceof SqliteError &&
      typeof err.code === 'string' &&
      err.code.startsWith('SQLITE_CONSTRAINT')) ||
    err instanceof GrammyError ||
    err instanceof HttpError ||
    isFsErrorWithCode(err, IPC_FS_CODES)
  );
}

/**
 * Best-effort quarantine of a bad IPC file into `errors/`. Wrapped so an fs
 * failure during the move can't break the per-file boundary — the file
 * simply stays and is retried next poll; a non-fs defect propagates.
 */
function moveIpcFileToErrors(
  ipcBaseDir: string,
  sourceGroup: string,
  file: string,
  filePath: string,
): void {
  try {
    const errorDir = path.join(ipcBaseDir, 'errors');
    fs.mkdirSync(errorDir, { recursive: true });
    fs.renameSync(filePath, path.join(errorDir, `${sourceGroup}-${file}`));
  } catch (err) {
    if (!isFsErrorWithCode(err, IPC_FS_CODES)) throw err;
    logger.error(
      { err, file, sourceGroup },
      '[ipc] Failed to move bad file to errors/ (will retry next poll)',
    );
  }
}

export interface IpcDeps {
  /**
   * Fired after a successful VISIBLE bot send from the IPC
   * `send_message` / `send_file` handlers (delivery confirmed by a
   * returned message id). `sourceGroupFolder` is the sending group;
   * the consumer decides whether the target chat is that group's own
   * chat before consuming the reply anchor (#722 — the anchor must be
   * released at the visible-send boundary, not only when the SDK
   * result later reaches the output callback).
   */
  onVisibleReply?: (chatJid: string, sourceGroupFolder: string) => void;
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
  ) => Promise<string | undefined>;
  registeredGroups: () => Record<string, RegisteredGroup>;
  registerGroup: (jid: string, group: RegisteredGroup) => void;
  /**
   * Inverse of `registerGroup` (#159). Removes the in-memory entry and
   * the DB row in one call. Returns false if the JID was not registered
   * — caller can use that to log a no-op or to surface "nothing to do"
   * to the requester. The caller is responsible for refreshing the
   * `available_groups.json` snapshot afterward (mirrors the
   * registerGroup contract).
   *
   * Out of scope: deleting the on-disk `groups/<folder>/` directory.
   * Operators delete that manually; auto-deletion would silently destroy
   * agent-curated state (CLAUDE.md, MEMORY.md, scheduled-task workspace)
   * on every churn of the registration.
   */
  unregisterGroup: (jid: string) => boolean;
  /** Partial update: flip `containerConfig.trusted` only. Returns false if the JID isn't registered. */
  setGroupTrusted: (jid: string, trusted: boolean) => boolean;
  /**
   * Partial update: change the trigger pattern (and optionally
   * `requiresTrigger`) only. Returns false if (a) the JID isn't
   * registered, or (b) the trigger fails the non-empty/whitespace
   * invariant enforced by `updateGroupTrigger`.
   */
  setGroupTrigger: (
    jid: string,
    trigger: string,
    requiresTrigger?: boolean,
  ) => boolean;
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
  nukeSession: (
    groupFolder: string,
    session: 'default' | 'maintenance' | 'all',
    options?: { skipReentry?: boolean },
  ) => void;
  /**
   * Signal every currently-active container across all groups and sessions
   * to wind down (write `_close` sentinels). Called by the `tessl_update`
   * IPC handler after the registry pulls in new tile content so running
   * containers respawn and pick up the fresh tiles on the next inbound
   * message — without this, a long-lived container keeps the old skills/
   * snapshot it copied at spawn time until its 30-min idle timeout.
   *
   * Implemented in `GroupQueue.closeAllActiveContainers()`. The dep is
   * injected (not imported directly) for the same reason `nukeSession`
   * is: ipc.ts mustn't reach into the orchestrator's queue singleton, and
   * tests need to substitute a stub.
   *
   * Returns the number of containers signaled.
   */
  closeAllActiveContainers: () => number;
  /**
   * Read the derived container status for a given (jid, sessionName)
   * slot. Used by `chat_status` to surface running/idle/cooling-down/
   * crashed/not-spawned without exposing the queue's internal state map.
   * Implementations must combine the GroupQueue slot state with the
   * orchestrator-side circuit breaker (per group folder).
   */
  getContainerStatus?: (
    chatJid: string,
    sessionName: 'default' | 'maintenance',
  ) => ContainerStatus;
}

let ipcWatcherRunning = false;

/**
 * Start the IPC polling loop. Returns a stop handle that halts the
 * loop and cancels the pending poll — production ignores it (the
 * watcher lives for the process), integration tests use it so a
 * finished suite doesn't leave a live timer polling a deleted tempdir.
 */
export function startIpcWatcher(deps: IpcDeps): () => void {
  if (ipcWatcherRunning) {
    logger.debug('IPC watcher already running, skipping duplicate start');
    return () => {};
  }
  ipcWatcherRunning = true;

  // Wire the command registries before the first poll. Pre-#878 only
  // `processTaskIpc` registered (the message path was an inline `else if`
  // chain needing no registry), so a watcher that saw a message file
  // before any task file would have found an empty message registry and
  // silently discarded it. Idempotent — `processTaskIpc` still calls it
  // for direct (test) invocations that bypass the watcher.
  registerCoreIpcHandlers();

  const ipcBaseDir = path.join(DATA_DIR, 'ipc');
  fs.mkdirSync(ipcBaseDir, { recursive: true });

  let stopped = false;
  let pollTimer: NodeJS.Timeout | undefined;

  const processIpcFiles = async () => {
    if (stopped) return;
    // Scan all group IPC directories (identity determined by directory)
    let groupFolders: string[];
    try {
      groupFolders = fs.readdirSync(ipcBaseDir).filter((f) => {
        const stat = fs.statSync(path.join(ipcBaseDir, f));
        return stat.isDirectory() && f !== 'errors';
      });
    } catch (err) {
      if (!isFsErrorWithCode(err, IPC_FS_CODES)) throw err;
      logger.error({ err }, 'Error reading IPC base directory');
      if (!stopped) pollTimer = setTimeout(processIpcFiles, IPC_POLL_INTERVAL);
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
            // Hoisted so the catch-block at the bottom can log which
            // message we were processing when things exploded. Without
            // this, a throw after `data = JSON.parse(...)` but before
            // the per-type branches would leave the error-log blind to
            // what content was in flight.
            let data: IpcMessagePayload | undefined;
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
              data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
              if (!data) {
                // JSON.parse produced null/undefined (e.g. the file
                // contained literal `null`). Skip to the next file.
                logger.warn(
                  { file, sourceGroup },
                  '[ipc] IPC file parsed to null/undefined — skipping',
                );
                fs.unlinkSync(filePath);
                continue;
              }
              await dispatchIpcMessage({
                data,
                sourceGroup,
                isMain,
                registeredGroups,
                deps,
                file,
              });
              fs.unlinkSync(filePath);
            } catch (err) {
              // Quarantine a bad file (recoverable data/transport failure) and
              // continue; a programming defect propagates.
              if (!isRecoverableIpcError(err)) throw err;
              // Catches anything thrown above (JSON.parse, auth, send, storeMessage).
              // If this fires AFTER the send already landed in Telegram, the
              // user sees a message with no DB row — exactly the ghost-heartbeat
              // symptom. Log the err, the data type, and a preview so the
              // operator can correlate with what appeared in the chat.
              logger.error(
                {
                  file,
                  sourceGroup,
                  err,
                  dataType: (data as { type?: string } | undefined)?.type,
                  dataChatJid: (data as { chatJid?: string } | undefined)
                    ?.chatJid,
                  dataTextPreview:
                    typeof (data as { text?: string } | undefined)?.text ===
                    'string'
                      ? (data as { text: string }).text.slice(0, 200)
                      : undefined,
                },
                '[ipc] Error processing IPC message — message may have been sent to the chat before the throw, in which case no DB row will exist',
              );
              moveIpcFileToErrors(ipcBaseDir, sourceGroup, file, filePath);
            }
          }
        }
      } catch (err) {
        if (!isFsErrorWithCode(err, IPC_FS_CODES)) throw err;
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
              // Quarantine a bad file (recoverable data/transport failure) and
              // continue; a programming defect propagates.
              if (!isRecoverableIpcError(err)) throw err;
              logger.error(
                { file, sourceGroup, err },
                'Error processing IPC task',
              );
              moveIpcFileToErrors(ipcBaseDir, sourceGroup, file, filePath);
            }
          }
        }
      } catch (err) {
        if (!isFsErrorWithCode(err, IPC_FS_CODES)) throw err;
        logger.error({ err, sourceGroup }, 'Error reading IPC tasks directory');
      }
    }

    if (!stopped) pollTimer = setTimeout(processIpcFiles, IPC_POLL_INTERVAL);
  };

  processIpcFiles();
  logger.info('IPC watcher started (per-group namespaces)');
  return () => {
    stopped = true;
    if (pollTimer) clearTimeout(pollTimer);
    ipcWatcherRunning = false;
  };
}

export async function processTaskIpc(
  data: IpcTaskPayload,
  sourceGroup: string, // Verified identity from IPC directory
  isMain: boolean, // Verified from directory path
  deps: IpcDeps,
): Promise<void> {
  // #845: every core command dispatches through the IPC registry
  // (host plugins register their own handlers at startup). An unknown
  // type is a wiring bug or a stale container payload.
  registerCoreIpcHandlers();
  if (await dispatchIpcTask({ data, sourceGroup, isMain, deps })) return;
  logger.warn({ type: data.type }, 'Unknown IPC task type');
}
