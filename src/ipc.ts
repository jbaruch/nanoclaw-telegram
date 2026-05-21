import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';

import { CronExpressionParser } from 'cron-parser';

import {
  ASSISTANT_NAME,
  DATA_DIR,
  GROUPS_DIR,
  HOST_PROJECT_ROOT,
  IPC_POLL_INTERVAL,
  STORE_DIR,
  TIMEZONE,
} from './config.js';
import { syncBackupRepo, type SyncResult } from './backup-sync.js';
import { sendPoolMessage } from './channels/telegram.js';
import { coerceTaskTextField } from './coerce-task-prompt.js';
import {
  buildSnitchmdFlags,
  formatSnitchmdHeader,
  parseFetchMarkdownUrl,
} from './fetch-markdown-args.js';
import {
  AvailableGroup,
  DEFAULT_SESSION_NAME,
  getInstalledTiles,
  resolveAgentModel,
  resolvePerGroupAgentModel,
  sessionInputDirName,
} from './container-runner.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';
import { hostLogsOrchestratorFile } from './host-logs.js';
import { findGateDecisions, readHostLog } from './host-log-parser.js';
import {
  applyTripitSegmentsToTzState,
  createTask,
  deleteAllSessions,
  deleteTask,
  getActivePendingRunAtNames,
  getCurrentTz,
  getLastFromMeMessages,
  getTaskById,
  getTasksForGroup,
  setTaskAgentModel,
  storeMessage,
  updateTask,
  type TripitSegment,
} from './db.js';
import type { ContainerStatus } from './group-queue.js';
import { isValidGroupFolder } from './group-folder.js';
import { logger } from './logger.js';
import { stripInternalTags } from './router.js';
import { recomputeLocalSchedules } from './task-scheduler.js';
import {
  isValidTimezone,
  normalizeScheduleTimezone,
  type ScheduleType,
} from './timezone.js';
import { RegisteredGroup, TriggerPattern } from './types.js';

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
 * Path to the `_script_result_<requestId>.json` reply file the host writes
 * for an IPC request. Must land in the SAME session's input dir that the
 * requesting container mounts at `/workspace/ipc/input/` — otherwise the
 * container polls forever and the IPC call times out.
 *
 * The container-side MCP server stamps `sessionName` onto every IPC
 * payload (both TASKS and MESSAGES — see
 * `container/agent-runner/src/ipc-mcp-stdio.ts`). Older containers that
 * predate that change (or any request where the field is missing) fall
 * back to the default session — matches pre-parallel behavior where
 * only one session existed.
 */
// Session names accepted on IPC requests: ONLY the two the orchestrator
// ever creates. A broader regex (e.g. `[A-Za-z0-9_-]+`) would let a
// container send distinct valid-looking names and force the host into
// unbounded `input-<session>/` dir creation below — an empty-dir DoS.
// Canonical enum is the right level of trust for payload-supplied values.
const KNOWN_SESSION_NAMES: ReadonlySet<string> = new Set([
  DEFAULT_SESSION_NAME,
  MAINTENANCE_SESSION_NAME,
]);
const VALID_REQUEST_ID_RE = /^[A-Za-z0-9_-]+$/;

// Host-side allowlist for the five tile-repo names the promote flow is
// wired against. The MCP tools' zod enums (ipc-mcp-stdio.ts::TILE_NAMES)
// mirror this list client-side for a clean schema error at tool-call
// time, but the IPC handler is reachable by any payload dropped into
// the tasks dir — a compromised container could skip the MCP path and
// write `{tileName: "../../etc"}` directly, escaping GROUPS_DIR via
// `path.join` or pointing the bash scripts at an attacker-controlled
// git URL. This set is the actual trust boundary; keeping it in sync
// with the client-side enum is a release-hygiene concern.
const KNOWN_TILE_NAMES: ReadonlySet<string> = new Set([
  'nanoclaw-admin',
  'nanoclaw-core',
  'nanoclaw-untrusted',
  'nanoclaw-trusted',
  'nanoclaw-host',
]);

/**
 * Resolve the IANA zone to pass to `cron-parser` when computing an
 * initial `next_run` for a stored task row. Centralised so
 * `schedule_task` and `update_task` produce values consistent with
 * `task-scheduler.ts:computeNextRunDetailed` at fire time.
 *
 * Three inputs:
 *   - `null`         → server-wide TIMEZONE fallback (pre-#102 behaviour
 *                       for rows without a per-task tz).
 *   - IANA name      → pass through unchanged (validated upstream by
 *                       `normalizeScheduleTimezone`).
 *   - `'local'`      → resolve via `getCurrentTz()` — the singleton
 *                       `tz_state.current_tz` written by `task-tz-sync`.
 *                       If the resolver returns null (no tz_state row
 *                       yet, or unfamiliar `schema_version`) OR returns
 *                       a value `Intl.DateTimeFormat` doesn't recognize
 *                       (corrupt write), fall back to TIMEZONE. Same
 *                       fallback the scheduler reaches via its
 *                       catch-and-retry path — narrowed here to a
 *                       sanity check up-front so a corrupt resolver
 *                       result doesn't brick `schedule_task` /
 *                       `update_task` with an "Invalid cron expression"
 *                       error that's actually a tz_state problem.
 */
function resolveCronTz(scheduleTimezone: string | null): string {
  if (scheduleTimezone === 'local') {
    const resolved = getCurrentTz();
    if (resolved && isValidTimezone(resolved)) return resolved;
    logger.warn(
      { resolved },
      'resolveCronTz: tz_state.current_tz unusable for cron-parser — falling back to TIMEZONE',
    );
    return TIMEZONE;
  }
  return scheduleTimezone || TIMEZONE;
}

/**
 * Compute the host path where an IPC response file should land.
 *
 * Both `data.sessionName` and `data.requestId` arrive from the container's
 * IPC payload — treat as untrusted. Without validation, crafted values
 * like `../default` or `../../etc/passwd` would make `path.join` escape
 * the expected `<DATA_DIR>/ipc/<sourceGroup>/input-<session>/` subtree.
 *
 * Fail-safe strategy, two independent fallbacks:
 * - Invalid `requestId` → fixed filename `_script_result_invalid.json`.
 *   Keeps path traversal out of the filename AND prevents a noisy/
 *   malicious container from filling disk by spamming unique ids —
 *   at most one orphan file per session's input dir, overwritten in
 *   place each time. The SESSION dir is still whatever was validated
 *   from the payload (the `sessionName` check is separate).
 * - Invalid `sessionName` → fall back to `DEFAULT_SESSION_NAME`. Blocks
 *   `..`-style path-segment escape into a different group's subtree.
 *
 * Both fallbacks log at warn level for auditing. The malformed request
 * effectively times out (its response lands where no container polls),
 * which is the correct outcome for a bad payload. This keeps every
 * caller's `fs.writeFileSync(resultPath, ...)` pattern intact (no null-
 * checking at 10+ call sites) while still blocking path traversal.
 */
function scriptResultPath(
  sourceGroup: string,
  data: { sessionName?: string; requestId?: string },
): string {
  let requestId: string;
  if (
    typeof data.requestId === 'string' &&
    VALID_REQUEST_ID_RE.test(data.requestId)
  ) {
    requestId = data.requestId;
  } else {
    logger.warn(
      { sourceGroup, requestId: data.requestId },
      'IPC request has missing or invalid requestId — routing response to orphan path',
    );
    // Fixed filename for all invalid requests so a noisy/malicious container
    // can't spam unique requestIds and fill disk with orphan replies. At
    // most one `_script_result_invalid.json` file exists per input dir, and
    // it gets overwritten on every subsequent malformed request.
    requestId = 'invalid';
  }
  let session = DEFAULT_SESSION_NAME;
  if (typeof data.sessionName === 'string' && data.sessionName) {
    if (KNOWN_SESSION_NAMES.has(data.sessionName)) {
      session = data.sessionName;
    } else {
      logger.warn(
        { sourceGroup, sessionName: data.sessionName },
        'IPC request has unknown sessionName — falling back to default',
      );
    }
  }
  const inputDir = path.join(
    DATA_DIR,
    'ipc',
    sourceGroup,
    sessionInputDirName(session),
  );
  // Ensure the session's input dir exists before the caller writes into it.
  // In the common path both sessions have already spawned at least once and
  // the dir exists — but a maintenance-only group (or a container that has
  // never gone through default) won't have `input-default/`, and our
  // fallback routes here for malformed payloads. Creating the dir
  // defensively keeps `fs.writeFileSync(resultPath, ...)` from throwing
  // ENOENT at every caller.
  fs.mkdirSync(inputDir, { recursive: true });
  return path.join(inputDir, `_script_result_${requestId}.json`);
}

// Prefix for outbound text emitted by the maintenance-session AyeAye.
// Without the prefix, a scheduled-task reply looks identical to a
// user-facing reply in the chat, which confused Baruch when he
// responded to `[heartbeat from maintenance]` messages as if they were
// live conversation. The prefix is applied BOTH to Telegram-bound text
// AND to the messages.db copy so the full trail shows provenance —
// heartbeat accounting, future message recap, etc.
const MAINTENANCE_MESSAGE_PREFIX = '[M] ';

/**
 * Prepend `[M] ` if the payload came from the maintenance session.
 * Idempotent — if the text already begins with the prefix (double-
 * hop case, agent that hand-typed it, whatever), we don't stack.
 * Exported for the unit test; the production caller is in the same
 * file so the public API is a single entry point.
 *
 * @internal — test-only export, should not be part of the public
 * `.d.ts` surface (we build with `stripInternal: true`).
 */
export function applyMaintenancePrefix(
  text: string,
  sessionName: string | undefined,
): string {
  if (sessionName !== MAINTENANCE_SESSION_NAME) return text;
  if (text.startsWith(MAINTENANCE_MESSAGE_PREFIX)) return text;
  return MAINTENANCE_MESSAGE_PREFIX + text;
}

/**
 * Decide whether to record a `bot-…` row in `messages.db` after the
 * IPC `send_message` handler dispatches a send. For Telegram we MUST
 * have a Telegram-native message id back from the channel — its
 * absence is the only reliable signal that the send was swallowed
 * (400 from a bad reply_to, network blip, malformed HTML even after
 * the plain-text fallback, blocked-by-user, rate-limit, etc.). A
 * row written without that id is a phantom: the heartbeat /
 * unanswered-cron treats it as evidence of a reply on a chat the
 * user never received anything in, and downstream agents quote-reply
 * to a message id Telegram has no record of.
 *
 * Non-Telegram channels are not gated — their `Channel.sendMessage`
 * contract permits returning `void` on success (see `src/types.ts`),
 * so absence of an id isn't a failure signal there. Until those
 * channels grow their own success-id surface, the gate would punish
 * a passing send.
 *
 * The undefined check is `!== undefined` rather than truthiness on
 * purpose, matching the comment on `sentMsgId` upstream: a future
 * Telegram id of `''` or `'0'` (we don't expect this today, but the
 * contract is `string | undefined`) must still record the row.
 *
 * @internal — test-only export, should not be part of the public
 * `.d.ts` surface (we build with `stripInternal: true`).
 */
export function shouldStoreBotMessage(
  chatJid: string,
  sentMsgId: string | undefined,
): boolean {
  const isTelegram = chatJid.startsWith('tg:');
  if (!isTelegram) return true;
  return sentMsgId !== undefined;
}

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
            // Hoisted so the catch-block at the bottom can log which
            // message we were processing when things exploded. Without
            // this, a throw after `data = JSON.parse(...)` but before
            // the per-type branches would leave the error-log blind to
            // what content was in flight.
            let data:
              | {
                  type?: string;
                  chatJid?: string;
                  text?: string;
                  sender?: string;
                  replyToMessageId?: string;
                  pin?: boolean;
                  emoji?: string;
                  messageId?: string;
                  filePath?: string;
                  caption?: string;
                  [key: string]: unknown;
                }
              | undefined;
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
                    // Strip <internal>…</internal> blocks from the caption
                    // so agent-written internal reasoning never leaks —
                    // neither to Telegram (display) nor to messages.db
                    // (which feeds heartbeat's answered-check accounting).
                    // Mirrors the message-payload stripping below. If the
                    // caption is fully internal, send the file with no
                    // caption; the file itself is still useful payload.
                    const strippedCaption = data.caption
                      ? stripInternalTags(data.caption)
                      : '';
                    // Tag maintenance-session captions so Baruch can
                    // tell a scheduled-task file-send from a live one.
                    // Skip the prefix entirely when the caption is
                    // empty — `[M] ` alone on a silent file-send is
                    // noise.
                    const cleanCaption = strippedCaption
                      ? applyMaintenancePrefix(
                          strippedCaption,
                          typeof data.sessionName === 'string'
                            ? data.sessionName
                            : undefined,
                        )
                      : '';
                    const sentFileMsgId = await deps.sendFile(
                      data.chatJid,
                      hostPath,
                      cleanCaption || undefined,
                      data.replyToMessageId,
                    );
                    // Store the cleaned caption (if any) so the message
                    // shows up in accounting the same as text messages.
                    // Without this, `send_file` is a bypass: captions
                    // reach Telegram but never hit messages.db, so
                    // heartbeat unanswered-checks think the agent never
                    // responded. Store the cleaned version — storing the
                    // raw caption would let a caption whose visible text
                    // was empty after stripping count as an "answered"
                    // response. Gate on `sentFileMsgId` (#428) — a
                    // failed send must not leave a phantom row that
                    // marks the user as answered when delivery never
                    // landed.
                    const captionDelivered = shouldStoreBotMessage(
                      data.chatJid,
                      sentFileMsgId,
                    );
                    if (cleanCaption && captionDelivered) {
                      storeMessage({
                        id: `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                        chat_jid: data.chatJid,
                        sender: ASSISTANT_NAME,
                        sender_name: ASSISTANT_NAME,
                        content: cleanCaption,
                        timestamp: new Date().toISOString(),
                        is_from_me: true,
                        is_bot_message: true,
                        reply_to_message_id: data.replyToMessageId,
                        // Stamp the Telegram message id so post-hoc
                        // "which bot send corresponds to Telegram
                        // message X" queries match the orchestrator
                        // text-reply path (`src/index.ts:1659`) and
                        // the `send_message` handler. Pre-#428
                        // sendFile returned void so this column was
                        // unavailable; now that we have the id, no
                        // reason not to record it.
                        telegram_message_id: sentFileMsgId,
                      });
                    } else if (cleanCaption && !captionDelivered) {
                      logger.warn(
                        {
                          chatJid: data.chatJid,
                          hostPath,
                          captionLen: cleanCaption.length,
                        },
                        'send_file: skipping caption storeMessage — sendFile returned no message id (delivery failed)',
                      );
                    }
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
                logger.debug(
                  {
                    sourceGroup,
                    chatJid: data.chatJid,
                    rawTextLen: data.text.length,
                    rawPreview: String(data.text).slice(0, 80),
                    hasSender: Boolean(data.sender),
                    senderValue: data.sender,
                    hasReplyTo: Boolean(data.replyToMessageId),
                    hasPin: Boolean(data.pin),
                    ipcFile: file,
                  },
                  '[ipc] Received send_message IPC',
                );
                // Strip <internal> tags via the shared helper so this
                // path can't drift from the send_file caption path
                // above. If nothing remains, skip silently.
                const strippedText = stripInternalTags(data.text);
                if (!strippedText) {
                  logger.debug(
                    { sourceGroup },
                    '[ipc] send_message suppressed (all internal)',
                  );
                  fs.unlinkSync(filePath);
                  continue;
                }
                // Tag maintenance-session text so Baruch can tell a
                // scheduled-task reply from a live conversational one.
                // Applied AFTER internal-tag stripping (no point
                // prefixing text we're about to suppress) and BEFORE
                // both the Telegram send and the messages.db store, so
                // the prefix flows through accounting uniformly.
                const cleanText = applyMaintenancePrefix(
                  strippedText,
                  typeof data.sessionName === 'string'
                    ? data.sessionName
                    : undefined,
                );
                logger.debug(
                  {
                    sourceGroup,
                    chatJid: data.chatJid,
                    cleanLen: cleanText.length,
                    cleanPreview: cleanText.slice(0, 80),
                  },
                  '[ipc] send_message after stripInternalTags + maintenance-prefix',
                );

                // Authorization: verify this group can send to this chatJid
                const targetGroup = registeredGroups[data.chatJid];
                const authOk =
                  isMain ||
                  Boolean(targetGroup && targetGroup.folder === sourceGroup);
                logger.debug(
                  {
                    sourceGroup,
                    chatJid: data.chatJid,
                    isMain,
                    targetGroupFolder: targetGroup?.folder,
                    authOk,
                  },
                  '[ipc] send_message auth check',
                );
                if (authOk) {
                  const usePool = Boolean(
                    data.sender && data.chatJid.startsWith('tg:'),
                  );
                  logger.debug(
                    {
                      sourceGroup,
                      chatJid: data.chatJid,
                      path: usePool ? 'pool' : 'direct',
                      sender: data.sender,
                    },
                    '[ipc] send_message path decision',
                  );
                  // Capture whichever send path's message ID applies. Both
                  // `sendPoolMessage` and `deps.sendMessage` return the
                  // Telegram-native message ID (or undefined if the send
                  // failed or the channel isn't Telegram). Stored on the
                  // messages row so "which bot send produced Telegram ID X"
                  // is queryable without log spelunking.
                  // Normalize immediately: both send paths can return
                  // `string | void | undefined`. Collapsing to the
                  // `string | undefined` domain up front keeps downstream
                  // uses (`pinMessage`, `storeMessage`) type-safe without
                  // truthiness checks that would also drop legitimate
                  // empty-string / '0' IDs if Telegram ever returns them.
                  let sentMsgId: string | undefined;
                  if (usePool) {
                    // `usePool` is only true when `data.sender` is a non-
                    // empty string — TS just can't re-narrow across the
                    // intermediate `Boolean(...)` boundary. The `!` is
                    // safe by the `usePool` definition directly above.
                    const poolResult = await sendPoolMessage(
                      data.chatJid,
                      cleanText,
                      data.sender!,
                      sourceGroup,
                    );
                    sentMsgId =
                      typeof poolResult === 'string' ? poolResult : undefined;
                    logger.debug(
                      {
                        sourceGroup,
                        chatJid: data.chatJid,
                        sentMsgId,
                      },
                      '[ipc] sendPoolMessage returned',
                    );
                  } else {
                    const directResult = await deps.sendMessage(
                      data.chatJid,
                      cleanText,
                      data.replyToMessageId,
                    );
                    sentMsgId =
                      typeof directResult === 'string'
                        ? directResult
                        : undefined;
                    logger.debug(
                      {
                        sourceGroup,
                        chatJid: data.chatJid,
                        sentMsgId,
                      },
                      '[ipc] deps.sendMessage returned',
                    );
                    // Pin the message if requested
                    if (data.pin && sentMsgId && deps.pinMessage) {
                      await deps.pinMessage(data.chatJid, sentMsgId);
                      logger.debug(
                        { sourceGroup, chatJid: data.chatJid, sentMsgId },
                        '[ipc] pinMessage returned',
                      );
                    }
                  }
                  // Gate the bot-row write on send success — see the
                  // `shouldStoreBotMessage` helper for the full rationale
                  // (phantom rows on swallowed Telegram sends would
                  // silence the heartbeat / unanswered-cron and feed
                  // cascading hallucinated quote-replies downstream).
                  if (shouldStoreBotMessage(data.chatJid, sentMsgId)) {
                    const botRowId = `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
                    storeMessage({
                      id: botRowId,
                      chat_jid: data.chatJid,
                      sender: data.sender || ASSISTANT_NAME,
                      sender_name: data.sender || ASSISTANT_NAME,
                      content: cleanText,
                      timestamp: new Date().toISOString(),
                      is_from_me: true,
                      is_bot_message: true,
                      reply_to_message_id: data.replyToMessageId,
                      telegram_message_id: sentMsgId,
                    });
                    logger.info(
                      {
                        chatJid: data.chatJid,
                        sourceGroup,
                        botRowId,
                        contentLen: cleanText.length,
                      },
                      '[ipc] send_message complete — DB row written',
                    );
                  } else {
                    logger.error(
                      {
                        chatJid: data.chatJid,
                        sourceGroup,
                        contentLen: cleanText.length,
                      },
                      '[ipc] send_message failed — Telegram returned no message id; skipping DB row to avoid phantom bot reply (would silence heartbeat / unanswered alerts)',
                    );
                  }
                } else {
                  logger.warn(
                    {
                      chatJid: data.chatJid,
                      sourceGroup,
                      targetGroupFolder: targetGroup?.folder,
                    },
                    '[ipc] Unauthorized IPC message attempt blocked',
                  );
                }
              }
              fs.unlinkSync(filePath);
            } catch (err) {
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
    /**
     * IANA timezone for cron expressions; #102.
     *
     * `null` is also accepted on the update path (where it means
     * "clear back to TIMEZONE default"). Must be `string | null` — not
     * just `string` — because IPC payloads arrive as raw JSON and the
     * caller can legitimately send `null` to unset; TS strict mode
     * would otherwise reject the `data.timezone === null` check.
     */
    timezone?: string | null;
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
    // For set_trusted
    trusted?: boolean;
    // For set_agent_model (#395). `string` = per-group override, `null` =
    // clear the override (fall back to global AGENT_MODEL).
    // `undefined` is rejected at the handler.
    agentModel?: string | null;
    // For set_maintenance_agent_model (#509). `string` = per-session-slot
    // override that applies only to the maintenance container slot for
    // this group; `null` = clear the override (maintenance falls back to
    // the per-group `agentModel` → global `AGENT_MODEL` ladder, which is
    // the pre-#509 behavior).
    maintenanceAgentModel?: string | null;
    // For set_additional_tiles (#305). Array of tile names from the
    // local registry to overlay on top of the trust-tier baseline.
    // `null` or `[]` clears the override. Anything else (string,
    // object, undefined) is rejected at the handler.
    additionalTiles?: string[] | null;
    // For host operations / github_backup / promote_staging / sessionize
    requestId?: string;
    message?: string;
    tileName?: string;
    skillName?: string;
    // push_staged_to_branch
    branch?: string;
    commitMessage?: string;
    slug?: string;
    filter?: Record<string, boolean>;
    dryRun?: boolean;
    command?: string;
    payload?: string | Record<string, unknown>;
    confirm?: boolean;
    // chat_status / nuke_chat / send_message_to_chat / inspect_gate_decisions
    chat_id?: string;
    chat_name?: string;
    session?: 'default' | 'maintenance' | 'all';
    // inspect_gate_decisions (#443)
    message_id?: string;
    limit?: number;
    // send_message_to_chat
    text?: string;
    pin?: boolean;
    sender?: string;
    /**
     * Continuation marker for self-resuming cycles (#93/#130). Set by the
     * resumable-cycle helper skill when scheduling the next link of a
     * chain via `schedule_task`. Persisted onto the scheduled_tasks row
     * verbatim; surfaced to the spawned container at fire time as
     * `NANOCLAW_CONTINUATION=1` + `NANOCLAW_CONTINUATION_CYCLE_ID=<value>`.
     * Free-form opaque slot key (UTC date / ISO week per the proposal),
     * but type-narrowed to string for safety; non-string values are
     * dropped at the handler.
     */
    continuation_cycle_id?: string;
    // For promote_learned_trigger (#451 item 1). Identifies a learned
    // proposal in registered_groups.trigger_pattern by its {kind,
    // pattern} tuple — the same identity the trigger-learner schema
    // uses to supersede prior versions.
    kind?: string;
    pattern?: string;
    // For fetch_markdown (#169). All optional except `url`. Mirrors the
    // snitchmd CLI flag set; see docs/fetch-tools.md for the decision
    // matrix and snitchmd's own README for flag semantics.
    url?: string;
    wait?: number;
    waitUntil?: string;
    waitForSelector?: string;
    favorPrecision?: boolean;
    favorRecall?: boolean;
    includeLinks?: boolean;
    includeImages?: boolean;
    maxChars?: number;
    noCache?: boolean;
    timeout?: number;
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
        // #512 — coerce `prompt` and `script` to text at the IPC
        // boundary. TS declares `string`, but the value is
        // deserialized from JSON: a non-string here would otherwise
        // bind as a BLOB via better-sqlite3 and surface later as
        // `t.prompt.slice is not a function` in `list_tasks`.
        // `coerceTaskTextField` returns:
        //   - the string itself when `data.prompt` is a string,
        //   - decoded UTF-8 when it's the JSON-Buffer shape
        //     (`{type:'Buffer',data:[...]}`),
        //   - `null` for any other shape — signal to reject the
        //     payload rather than persist a garbage row that would
        //     fire with `"[object Object]"` as its prompt.
        const promptStr = coerceTaskTextField(data.prompt);
        if (promptStr === null) {
          logger.warn(
            { sourceGroup, promptType: typeof data.prompt },
            'schedule_task: rejecting non-text prompt payload',
          );
          break;
        }
        let scriptStr: string | null = null;
        if (data.script != null) {
          const coerced = coerceTaskTextField(data.script);
          if (coerced === null) {
            logger.warn(
              { sourceGroup, scriptType: typeof data.script },
              'schedule_task: rejecting non-text script payload',
            );
            break;
          }
          // Treat empty-string script the same as null — same
          // semantics as the prior `data.script || null`.
          scriptStr = coerced || null;
        }
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

        // #102: optional timezone parameter. Validated up-front so a
        // typo fails the schedule call rather than silently falling
        // back to server-local at fire time. #456 extends the accepted
        // values with the literal token `'local'` — resolved at fire
        // time against `tz_state.current_tz` by
        // `task-scheduler.ts:computeNextRunDetailed`.
        //
        // Force `null` for non-cron types: the column has no effect on
        // `interval` (always elapsed-ms) or `once` (instant pinned at
        // schedule time). Persisting it for those types would be a
        // footgun if the task were later updated to `cron` without
        // explicitly passing `timezone` — an old, previously-ignored
        // value would silently start affecting cron evaluation.
        const tzOutcome = normalizeScheduleTimezone(
          data.timezone,
          scheduleType,
        );
        if (tzOutcome.action === 'reject-invalid') {
          logger.warn(
            { timezone: data.timezone },
            'Invalid IANA timezone for schedule_task',
          );
          break;
        }
        if (tzOutcome.action === 'ignore-non-cron') {
          logger.warn(
            { timezone: data.timezone, scheduleType },
            'schedule_task: timezone parameter is only meaningful for cron — ignoring',
          );
        }
        const scheduleTimezone: string | null =
          tzOutcome.action === 'accept' ? tzOutcome.value : null;

        let nextRun: string | null = null;
        if (scheduleType === 'cron') {
          try {
            // #456: resolve `'local'` against `tz_state.current_tz` to
            // mirror `computeNextRunDetailed`. The scheduler retries
            // with TIMEZONE on cron-parser failure; we narrow the
            // failure surface here by sanity-checking the resolved
            // value up-front so a corrupt `tz_state.current_tz` (e.g.
            // a future task-tz-sync bug writing garbage) doesn't brick
            // schedule_task — falls through to TIMEZONE just like NULL
            // `schedule_timezone` and like the scheduler's retry path.
            const cronTz = resolveCronTz(scheduleTimezone);
            const interval = CronExpressionParser.parse(data.schedule_value, {
              tz: cronTz,
            });
            nextRun = interval.next().toISOString();
          } catch (err) {
            // Bind + filter rather than catch-all per
            // `jbaruch/coding-policy: error-handling`. CronExpressionParser
            // throws plain Error instances on invalid syntax; anything
            // non-Error here is a bug somewhere else (e.g. a `throw "str"`
            // upstream) and should propagate.
            if (!(err instanceof Error)) throw err;
            logger.warn(
              { err: err.message, scheduleValue: data.schedule_value },
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
        // Provenance: derived from the VERIFIED source group's trust tier
        // (sourceGroup and isMain are set from the IPC directory path, not
        // from untrusted payload fields). The agent that scheduled the
        // task NEVER gets to claim its own role — this is the security
        // boundary that keeps an untrusted group from self-scheduling a
        // prompt that later fires unwrapped as if it were trusted.
        const sourceGroupEntry = Object.values(registeredGroups).find(
          (g) => g.folder === sourceGroup,
        );
        const createdByRole:
          | 'main_agent'
          | 'trusted_agent'
          | 'untrusted_agent' = isMain
          ? 'main_agent'
          : sourceGroupEntry?.containerConfig?.trusted
            ? 'trusted_agent'
            : 'untrusted_agent';
        // Optional continuation marker (#93/#130). Set by the
        // resumable-cycle helper skill when scheduling the next link of a
        // self-resuming cycle chain; the task-scheduler reads it at fire
        // time and plumbs the matching env vars onto the spawned
        // container. Untyped non-string values are dropped — the field is
        // a free-form opaque slot key (per the proposal: UTC date for
        // nightly/morning-brief, ISO week for weekly), but we never want
        // a stray number / object to land in the DB column.
        let continuationCycleId: string | null = null;
        if (
          typeof data.continuation_cycle_id === 'string' &&
          data.continuation_cycle_id.length > 0
        ) {
          continuationCycleId = data.continuation_cycle_id;
        }
        createTask({
          id: taskId,
          group_folder: targetFolder,
          chat_jid: targetJid,
          prompt: promptStr,
          script: scriptStr,
          schedule_type: scheduleType,
          schedule_value: data.schedule_value,
          schedule_timezone: scheduleTimezone,
          context_mode: contextMode,
          next_run: nextRun,
          status: 'active',
          created_at: new Date().toISOString(),
          created_by_role: createdByRole,
          continuation_cycle_id: continuationCycleId,
        });
        logger.info(
          {
            taskId,
            sourceGroup,
            targetFolder,
            contextMode,
            createdByRole,
            continuationCycleId,
          },
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
        // #512 — same boundary-coercion contract as schedule_task
        // above. `coerceTaskTextField` decodes the JSON-Buffer shape
        // so an older writer's BLOB-shaped payload still round-trips
        // to its real text; any other non-string shape is rejected
        // (we abort the whole update rather than write a garbage
        // column).
        if (data.prompt !== undefined) {
          const promptCoerced = coerceTaskTextField(data.prompt);
          if (promptCoerced === null) {
            logger.warn(
              {
                taskId: data.taskId,
                sourceGroup,
                promptType: typeof data.prompt,
              },
              'update_task: rejecting non-text prompt payload',
            );
            break;
          }
          updates.prompt = promptCoerced;
        }
        if (data.script !== undefined) {
          if (data.script == null) {
            updates.script = null;
          } else {
            const scriptCoerced = coerceTaskTextField(data.script);
            if (scriptCoerced === null) {
              logger.warn(
                {
                  taskId: data.taskId,
                  sourceGroup,
                  scriptType: typeof data.script,
                },
                'update_task: rejecting non-text script payload',
              );
              break;
            }
            // Empty string is the documented "clear back to no script"
            // signal — preserve that semantics.
            updates.script = scriptCoerced || null;
          }
        }
        if (data.schedule_type !== undefined)
          updates.schedule_type = data.schedule_type as
            | 'cron'
            | 'interval'
            | 'once';
        if (data.schedule_value !== undefined)
          updates.schedule_value = data.schedule_value;

        // #102: optional timezone update. `null`/empty-string clears
        // (back to TIMEZONE default); a non-null IANA string overrides.
        // Only meaningful for cron tasks: if the task IS a cron (or is
        // being changed to cron in this same update), accept and
        // persist; otherwise force the value to null so we don't store
        // a stray timezone that would silently start affecting cron
        // evaluation if the task were later switched to cron without
        // explicitly re-passing it.
        const effectiveScheduleType =
          updates.schedule_type ?? task.schedule_type;

        // If the schedule_type is being changed AWAY from cron AND the
        // existing row had a stored schedule_timezone, drop the stored
        // value too — even if the caller didn't explicitly pass
        // `timezone`. Otherwise a once/interval task can outlive a
        // previous cron incarnation with a stray timezone column that
        // would re-activate if the task were later flipped back to
        // cron without re-stating tz. (Copilot review round 2.)
        if (
          updates.schedule_type !== undefined &&
          updates.schedule_type !== 'cron' &&
          task.schedule_timezone &&
          data.timezone === undefined
        ) {
          updates.schedule_timezone = null;
        }

        if (data.timezone !== undefined) {
          // Same normalizer as schedule_task above — accepts null /
          // empty / IANA / `'local'` (#456); ignores tz on non-cron
          // (caller logs and forces null); rejects unrecognized
          // strings (caller logs and aborts).
          const updateTzOutcome = normalizeScheduleTimezone(
            data.timezone,
            effectiveScheduleType as ScheduleType,
          );
          if (updateTzOutcome.action === 'reject-invalid') {
            logger.warn(
              { taskId: data.taskId, timezone: data.timezone },
              'Invalid IANA timezone in task update',
            );
            break;
          }
          if (updateTzOutcome.action === 'ignore-non-cron') {
            logger.warn(
              {
                taskId: data.taskId,
                timezone: data.timezone,
                effectiveScheduleType,
              },
              'update_task: ignoring timezone — effective schedule_type is not cron',
            );
            updates.schedule_timezone = null;
          } else {
            updates.schedule_timezone = updateTzOutcome.value;
          }
        }

        // Recompute next_run if a recompute-relevant field changed.
        // Use `!== undefined` (not truthiness) for `schedule_value`
        // because an empty string IS a valid input on the wire (the
        // host catches it below as invalid) — truthy-skip would
        // silently leave next_run stale on a malformed update. For
        // `timezone`, only count it as a recompute trigger when the
        // (effective) schedule_type is cron — a timezone-only update
        // on a once/interval task has no effect on next_run.
        const triggerRecompute =
          data.schedule_type !== undefined ||
          data.schedule_value !== undefined ||
          (data.timezone !== undefined && effectiveScheduleType === 'cron');
        if (triggerRecompute) {
          const updatedTask = {
            ...task,
            ...updates,
          };
          if (updatedTask.schedule_type === 'cron') {
            try {
              // #456: same `'local'`-aware resolver as the schedule_task
              // path — sanity-checks `tz_state.current_tz` before
              // passing to cron-parser so a corrupt resolver result
              // falls back to TIMEZONE rather than aborting the update.
              const cronTz = resolveCronTz(
                updatedTask.schedule_timezone ?? null,
              );
              const interval = CronExpressionParser.parse(
                updatedTask.schedule_value,
                { tz: cronTz },
              );
              updates.next_run = interval.next().toISOString();
            } catch (err) {
              // See schedule_task above — same Error-or-rethrow pattern
              // per `jbaruch/coding-policy: error-handling`.
              if (!(err instanceof Error)) throw err;
              logger.warn(
                {
                  err: err.message,
                  taskId: data.taskId,
                  value: updatedTask.schedule_value,
                },
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
          } else if (updatedTask.schedule_type === 'once') {
            // #102 follow-up: if a once-task's schedule_value changes
            // (or the type flips to 'once'), recompute next_run from
            // the new timestamp. Without this branch the row would
            // keep its old `next_run` and fire incorrectly.
            const date = new Date(updatedTask.schedule_value);
            if (isNaN(date.getTime())) {
              logger.warn(
                { taskId: data.taskId, value: updatedTask.schedule_value },
                'Invalid once timestamp in task update',
              );
              break;
            }
            updates.next_run = date.toISOString();
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
      if (
        typeof data.jid === 'string' &&
        typeof data.name === 'string' &&
        typeof data.folder === 'string' &&
        typeof data.trigger === 'string' &&
        data.jid.length > 0 &&
        data.name.length > 0 &&
        data.folder.length > 0 &&
        data.trigger.length > 0
      ) {
        // `typeof === 'string'` guards BEFORE calling `.trim()` on
        // any field. IPC payloads are untrusted JSON: a malformed
        // request like `{jid: {}}` or `{name: 42}` would otherwise
        // throw a TypeError and route the task file to ipc/errors,
        // creating a low-effort log-spam / DoS vector. set_trusted /
        // set_trigger already follow this pattern; this reuses it.
        if (!isValidGroupFolder(data.folder)) {
          logger.warn(
            { sourceGroup, folder: data.folder },
            'Invalid register_group request - unsafe folder name',
          );
          break;
        }
        // Trim string fields so this IPC path can't leave a group
        // registered under a whitespace-padded key. set_trusted /
        // set_trigger trim before lookup; an untrimmed register would
        // otherwise produce a "ghost" registration the partial-update
        // tools can never match. Same normalization, same site of
        // truth.
        const trimmedJid = data.jid.trim();
        const trimmedName = data.name.trim();
        const trimmedTrigger = data.trigger.trim();
        if (
          trimmedJid.length === 0 ||
          trimmedName.length === 0 ||
          trimmedTrigger.length === 0
        ) {
          logger.warn(
            { data },
            'Invalid register_group request - empty/whitespace fields',
          );
          break;
        }
        // Defense in depth: agent cannot set isMain via IPC.
        // Preserve isMain from the existing registration so IPC config
        // updates (e.g. adding additionalMounts) don't strip the flag.
        const existingGroup = registeredGroups[trimmedJid];
        deps.registerGroup(trimmedJid, {
          name: trimmedName,
          folder: data.folder,
          trigger: trimmedTrigger,
          added_at: new Date().toISOString(),
          containerConfig: data.containerConfig,
          // Explicitly default to `false` when caller omits — matches
          // the MCP tool's documented default ("respond to all
          // messages"). setRegisteredGroup now preserves undefined as
          // SQL NULL, which is a distinct state from `false`, so we
          // must not pass undefined here or the new row would behave
          // differently than callers expect.
          requiresTrigger: data.requiresTrigger ?? false,
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

    case 'unregister_group': {
      // Inverse of register_group (#159). Same isMain gate — only the
      // main group can change the registry. The dormant-row problem
      // (#159 motivation) is exactly what happens when there is no
      // structured remove path: rows linger forever, the spawner
      // ignores them because the JSON snapshot doesn't list them, and
      // operators can't fix it from inside chat containers because
      // `/workspace/store/messages.db` is mounted read-only there.
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized unregister_group attempt blocked',
        );
        break;
      }
      if (typeof data.jid !== 'string' || data.jid.trim().length === 0) {
        logger.warn(
          { data },
          'Invalid unregister_group request - missing/empty jid',
        );
        break;
      }
      const trimmedJid = data.jid.trim();
      const target = registeredGroups[trimmedJid];
      if (!target) {
        logger.warn(
          { jid: trimmedJid },
          'unregister_group: group not registered (no-op)',
        );
        break;
      }
      // Refuse to unregister a main group via IPC. Losing the main
      // registration mid-runtime would leave the orchestrator without
      // any path that can re-create it (the same isMain gate above
      // would reject the corresponding register_group call). The
      // operator can flip `is_main` directly in the DB if they really
      // mean to, which is a deliberate destructive action rather than
      // a one-line MCP call.
      if (target.isMain) {
        logger.warn(
          { jid: trimmedJid, folder: target.folder },
          'unregister_group: refusing to unregister main group',
        );
        break;
      }
      // Cascade-delete scheduled_tasks tied to the unregistered folder
      // BEFORE we drop the registration. Without this, the scheduler
      // keeps firing the auto-created heartbeat (and any other tasks
      // bound to this folder) every cycle, logging "Group not found
      // for task" on each tick — exactly the noisy-orphan behaviour
      // Copilot flagged on PR #198. We do this before unregisterGroup
      // so a crash between the two leaves the registration alive (DB
      // delete is the authoritative atomic step); the inverse ordering
      // would orphan the registration with its tasks already gone,
      // which is the more confusing recovery path.
      const orphanTasks = getTasksForGroup(target.folder);
      for (const task of orphanTasks) {
        deleteTask(task.id);
      }
      if (orphanTasks.length > 0) {
        logger.info(
          {
            jid: trimmedJid,
            folder: target.folder,
            taskIds: orphanTasks.map((t) => t.id),
          },
          'unregister_group: cascade-deleted scheduled tasks for unregistered folder',
        );
        deps.onTasksChanged();
      }

      const removed = deps.unregisterGroup(trimmedJid);
      if (!removed) {
        // In-memory said yes but DB said no — possible if a parallel
        // path raced us. Log and fall through to snapshot refresh
        // anyway: the snapshot is derived state and a refresh is
        // always safe.
        logger.warn(
          { jid: trimmedJid, folder: target.folder },
          'unregister_group: in-memory entry present but DB delete reported no rows',
        );
      } else {
        logger.info(
          { jid: trimmedJid, folder: target.folder },
          'Group unregistered',
        );
      }
      // Refresh snapshot so available_groups.json no longer flags the
      // removed JID as registered. Same site-of-truth pattern as
      // register_group / set_trusted / set_trigger above.
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        true,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
      break;
    }

    case 'set_trusted':
      // Partial update: flip container_config.trusted only. Same isMain
      // gate as register_group — only the main group can change trust
      // state. See #105.
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized set_trusted attempt blocked',
        );
        break;
      }
      if (
        typeof data.jid === 'string' &&
        data.jid.trim().length > 0 &&
        typeof data.trusted === 'boolean'
      ) {
        // Trim JID for the same reason as set_trigger: avoids a
        // misleading "group not registered" warning when a caller
        // passes whitespace-padded JID.
        const trimmedJid = data.jid.trim();
        const ok = deps.setGroupTrusted(trimmedJid, data.trusted);
        if (!ok) {
          logger.warn(
            { jid: trimmedJid },
            'set_trusted: group not registered (use register_group first)',
          );
          break;
        }
        const availableGroups = deps.getAvailableGroups();
        deps.writeGroupsSnapshot(
          sourceGroup,
          true,
          availableGroups,
          new Set(Object.keys(registeredGroups)),
        );
        // setGroupTrusted may have reconciled the heartbeat task's
        // `script` field as a side effect of the trust flip — refresh
        // the per-group task snapshots so containers see the change
        // on their next read instead of waiting for the orchestrator
        // to write a snapshot for some other reason.
        deps.onTasksChanged();
      } else {
        logger.warn(
          { data },
          'Invalid set_trusted request - missing/empty jid or invalid trusted',
        );
      }
      break;

    case 'set_trigger':
      // Partial update: change trigger_pattern and optionally
      // requires_trigger. Same isMain gate as register_group. See #105.
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized set_trigger attempt blocked',
        );
        break;
      }
      if (
        typeof data.jid === 'string' &&
        data.jid.trim().length > 0 &&
        typeof data.trigger === 'string' &&
        data.trigger.trim().length > 0
      ) {
        // Reject empty/whitespace triggers + JIDs, then pass the
        // trimmed values downstream. `getTriggerPattern('')` trims and
        // falls back to `DEFAULT_TRIGGER`, so an empty trigger would
        // silently revert the group to the assistant's default trigger
        // word — not what the caller asked for. Trimming the JID
        // before lookup avoids a misleading "group not registered"
        // warning when a caller passes `' tg:-123 '` (whitespace would
        // never match the registry key).
        const trimmedJid = data.jid.trim();
        const trimmedTrigger = data.trigger.trim();
        const requiresTrigger =
          typeof data.requiresTrigger === 'boolean'
            ? data.requiresTrigger
            : undefined;
        const ok = deps.setGroupTrigger(
          trimmedJid,
          trimmedTrigger,
          requiresTrigger,
        );
        if (!ok) {
          logger.warn(
            { jid: trimmedJid },
            'set_trigger: group not registered (use register_group first)',
          );
          break;
        }
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
          'Invalid set_trigger request - missing/empty jid or trigger',
        );
      }
      break;

    case 'set_agent_model': {
      // Partial update: change `containerConfig.agentModel` only (#395).
      // Authorization mirrors schedule_task — main can target any
      // registered group; non-main can target only its own folder so an
      // untrusted agent can't quietly downgrade another group's model
      // (or escalate its own to a more expensive one for someone else's
      // bill). Sibling containerConfig fields (trusted, additionalMounts,
      // enableHeartbeat, timeout) are preserved verbatim — set_trusted /
      // set_trigger semantics, applied to a new column.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      if (!groupFolder) {
        logger.warn(
          { data },
          'Invalid set_agent_model request - missing/empty groupFolder',
        );
        break;
      }
      // `agentModel` accepts string (set/replace) or null (clear). Anything
      // else (number, object, undefined) is rejected — we don't want a
      // malformed payload to silently no-op.
      if (typeof data.agentModel !== 'string' && data.agentModel !== null) {
        logger.warn(
          { data },
          'Invalid set_agent_model request - agentModel must be string or null',
        );
        break;
      }
      // Locate the target group entry by folder. We need the JID to
      // call deps.registerGroup; iterate the in-memory registry rather
      // than touching the DB directly so the source-of-truth stays in
      // ipc.ts's existing pattern.
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'set_agent_model: group not registered (use register_group first)',
        );
        break;
      }
      // Authorization: non-main can only modify its own folder.
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized set_agent_model attempt blocked',
        );
        break;
      }
      const nextContainerConfig: RegisteredGroup['containerConfig'] = {
        ...(targetGroup.containerConfig ?? {}),
      };
      if (data.agentModel === null) {
        // Explicit clear — drop the field so it serialises as absent
        // (not as JSON null) and the runtime falls through to the
        // global AGENT_MODEL.
        delete nextContainerConfig.agentModel;
      } else {
        const trimmed = data.agentModel.trim();
        if (trimmed.length === 0) {
          // Treat empty/whitespace as a clear, same way
          // resolvePerGroupAgentModel folds empty into fallback.
          delete nextContainerConfig.agentModel;
        } else {
          nextContainerConfig.agentModel = trimmed;
        }
      }
      deps.registerGroup(targetJid, {
        ...targetGroup,
        containerConfig: nextContainerConfig,
      });
      logger.info(
        {
          groupFolder,
          agentModel: nextContainerConfig.agentModel ?? null,
          source: sourceGroup,
        },
        'set_agent_model: updated per-group AGENT_MODEL override',
      );
      // Refresh available_groups.json so containers see the new
      // override on their next read. `isMain` (not hardcoded true)
      // because a non-main source legally lands here when modifying
      // its own folder.
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
      break;
    }

    case 'set_maintenance_agent_model': {
      // Partial update: change `containerConfig.maintenanceAgentModel`
      // only (#509). Mirrors set_agent_model semantics — owner-of-the-bill
      // can change their own group's model knobs. Sibling containerConfig
      // fields are preserved verbatim.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      if (!groupFolder) {
        logger.warn(
          { data },
          'Invalid set_maintenance_agent_model request - missing/empty groupFolder',
        );
        break;
      }
      // `maintenanceAgentModel` accepts string (set/replace) or null
      // (clear). Anything else (number, object, undefined) is rejected.
      if (
        typeof data.maintenanceAgentModel !== 'string' &&
        data.maintenanceAgentModel !== null
      ) {
        logger.warn(
          { data },
          'Invalid set_maintenance_agent_model request - maintenanceAgentModel must be string or null',
        );
        break;
      }
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'set_maintenance_agent_model: group not registered (use register_group first)',
        );
        break;
      }
      // Authorization mirrors set_agent_model — non-main can only modify
      // its own folder.
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized set_maintenance_agent_model attempt blocked',
        );
        break;
      }
      const nextContainerConfig: RegisteredGroup['containerConfig'] = {
        ...(targetGroup.containerConfig ?? {}),
      };
      if (data.maintenanceAgentModel === null) {
        delete nextContainerConfig.maintenanceAgentModel;
      } else {
        const trimmed = data.maintenanceAgentModel.trim();
        if (trimmed.length === 0) {
          delete nextContainerConfig.maintenanceAgentModel;
        } else {
          nextContainerConfig.maintenanceAgentModel = trimmed;
        }
      }
      deps.registerGroup(targetJid, {
        ...targetGroup,
        containerConfig: nextContainerConfig,
      });
      logger.info(
        {
          groupFolder,
          maintenanceAgentModel:
            nextContainerConfig.maintenanceAgentModel ?? null,
          source: sourceGroup,
        },
        'set_maintenance_agent_model: updated per-group MAINTENANCE_AGENT_MODEL override',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
      break;
    }

    case 'set_task_agent_model': {
      // Per-task AGENT_MODEL override for #509 Phase 3. Writes
      // scheduled_tasks.agent_model directly; the row is the source of
      // truth, no in-memory registry to refresh. Authorization mirrors
      // set_maintenance_agent_model — owner-of-bill: a non-main caller
      // can only touch tasks belonging to its own group_folder.
      // Re-uses getTaskById's existing row shape rather than a
      // separate ownership-only query so a non-existent taskId surfaces
      // as a clean "not found" error to the caller instead of a 403.
      const taskId = typeof data.taskId === 'string' ? data.taskId.trim() : '';
      if (!taskId) {
        logger.warn(
          { data },
          'Invalid set_task_agent_model request - missing/empty taskId',
        );
        break;
      }
      // `agentModel` accepts string (set/replace) or null (clear). Anything
      // else (number, object, undefined) is rejected.
      if (typeof data.agentModel !== 'string' && data.agentModel !== null) {
        logger.warn(
          { data },
          'Invalid set_task_agent_model request - agentModel must be string or null',
        );
        break;
      }
      const task = getTaskById(taskId);
      if (!task) {
        logger.warn({ taskId }, 'set_task_agent_model: task not found');
        break;
      }
      // Owner-of-bill auth — non-main caller can only touch its own folder.
      if (!isMain && task.group_folder !== sourceGroup) {
        logger.warn(
          { sourceGroup, taskGroupFolder: task.group_folder, taskId },
          'Unauthorized set_task_agent_model attempt blocked',
        );
        break;
      }
      // Cadence-registry rows are declarative state — their shape
      // (including agent_model) is owned by the SKILL.md frontmatter
      // and reasserted on every per-spawn rebuild
      // (rebuildCadenceRegistry in src/cadence-registry.ts). An
      // imperative IPC write to such a row would silently revert on
      // the next tile-touching spawn, exactly the "looks fine, isn't"
      // failure mode this fleet's host-conventions and OpenAI's review
      // on PR #587 called out. Reject loudly and point the operator at
      // the durable surface (SKILL.md agentModel: frontmatter).
      // Non-cadence rows (source = 'schedule-task' — operator-
      // initiated reminders, ad-hoc monitors, the schedule-task IPC
      // surface) accept the imperative write because they have no
      // declarative source to argue with.
      const taskSource = (task as { source?: string }).source;
      if (taskSource === 'cadence-registry') {
        logger.warn(
          { taskId, source: taskSource },
          'set_task_agent_model: refusing to write to cadence-registry-owned row — modify the skill SKILL.md `agentModel:` frontmatter and republish the tile instead',
        );
        break;
      }
      // Normalise: `null` and empty-after-trim both mean "clear the
      // override" (fall back to the Phase 2 ladder). Trim non-empty
      // strings to match the existing knobs' shape.
      let nextValue: string | null;
      if (data.agentModel === null) {
        nextValue = null;
      } else {
        const trimmed = data.agentModel.trim();
        nextValue = trimmed.length === 0 ? null : trimmed;
      }
      const ok = setTaskAgentModel(taskId, nextValue);
      if (!ok) {
        // Race: row existed at getTaskById time but was deleted before
        // the UPDATE. Surfaces as a no-op for the operator.
        logger.warn(
          { taskId },
          'set_task_agent_model: task disappeared between check and update',
        );
        break;
      }
      logger.info(
        {
          taskId,
          groupFolder: task.group_folder,
          agentModel: nextValue,
          source: sourceGroup,
        },
        'set_task_agent_model: updated per-task agent_model override',
      );
      break;
    }

    case 'list_learned_triggers': {
      // Read-side surface for #451 item 3. Returns the learned-source
      // patterns with their observability fields (precision,
      // sample_count, last_matched_at, last_updated_at, pattern_version,
      // proposed_at, enabled, disabled). One group with `groupFolder`
      // set; all groups when omitted (main only). Authorization:
      // owner-of-bill — non-main can only inspect its own folder.
      // Result shape (matches list_installed_tiles convention):
      //   { stdout: JSON.stringify({ groups: [{folder, name, learned:[...]}] }) }
      const resultPath = scriptResultPath(sourceGroup, data);
      const filterFolder =
        typeof data.groupFolder === 'string' && data.groupFolder.trim()
          ? data.groupFolder.trim()
          : null;
      if (!isMain && filterFolder !== null && filterFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, requested: filterFolder },
          'Unauthorized list_learned_triggers attempt blocked',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: `list_learned_triggers: cross-folder read denied — non-main caller "${sourceGroup}" cannot inspect "${filterFolder}"`,
          }),
        );
        break;
      }
      // Non-main without filter is implicitly scoped to its own folder.
      const effectiveFolder =
        !isMain && filterFolder === null ? sourceGroup : filterFolder;
      const out: Array<{
        folder: string;
        name: string;
        learned: TriggerPattern[];
      }> = [];
      for (const [, g] of Object.entries(registeredGroups)) {
        if (effectiveFolder !== null && g.folder !== effectiveFolder) continue;
        const patterns = g.triggerPatterns?.patterns ?? [];
        const learned = patterns.filter((p) => p.source === 'learned');
        out.push({ folder: g.folder, name: g.name, learned });
      }
      fs.writeFileSync(
        resultPath,
        JSON.stringify({ stdout: JSON.stringify({ groups: out }) }),
      );
      logger.info(
        {
          sourceGroup,
          scope: effectiveFolder ?? 'all',
          group_count: out.length,
          learned_total: out.reduce((n, g) => n + g.learned.length, 0),
        },
        'list_learned_triggers served via IPC',
      );
      break;
    }

    case 'promote_learned_trigger': {
      // Promotion path for #451 item 1. The trigger-pattern learner
      // (#415, src/gates/trigger-learner.ts) writes proposals with
      // `source: 'learned'` and `enabled: false`. The trigger gate
      // skips `enabled: false` rows so proposals are inert until the
      // operator promotes them. This handler flips `enabled: false →
      // true` on a specific learned proposal identified by its
      // `{kind, pattern}` tuple. Demotion / re-enable, the per-group
      // dashboard, and producer-side enrichment of the haiku
      // classifier verdict are #451 items 2/3/4 — separate scopes.
      // Authorization mirrors set_agent_model: owner-of-bill —
      // non-main can only promote in its own folder.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      const kind = typeof data.kind === 'string' ? data.kind.trim() : '';
      // Pattern is the identity field used to match the row in
      // triggerPatterns — trimming would break lookups for patterns
      // legitimately stored with significant whitespace. Reject
      // whitespace-only / empty as a separate validation step rather
      // than mutating the value before the lookup.
      const pattern = typeof data.pattern === 'string' ? data.pattern : '';
      if (!groupFolder || !kind || !pattern || pattern.trim().length === 0) {
        logger.warn(
          { data },
          'Invalid promote_learned_trigger request - groupFolder, kind, and pattern are all required strings',
        );
        break;
      }
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'promote_learned_trigger: group not registered (use register_group first)',
        );
        break;
      }
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized promote_learned_trigger attempt blocked',
        );
        break;
      }
      const cfg = targetGroup.triggerPatterns;
      if (!cfg) {
        logger.warn(
          { groupFolder },
          'promote_learned_trigger: group has no triggerPatterns config (no learned proposals to promote)',
        );
        break;
      }
      const idx = cfg.patterns.findIndex(
        (p) =>
          p.source === 'learned' && p.kind === kind && p.pattern === pattern,
      );
      if (idx === -1) {
        logger.warn(
          { groupFolder, kind, pattern },
          'promote_learned_trigger: no matching learned pattern (check kind+pattern against registered_groups.trigger_pattern JSON)',
        );
        break;
      }
      const target = cfg.patterns[idx];
      if (target.disabled === true) {
        logger.warn(
          { groupFolder, kind, pattern },
          'promote_learned_trigger: pattern is auto-rolled-back (disabled=true) — re-enable via the demotion handler before promoting (item 2)',
        );
        break;
      }
      if (target.enabled === true) {
        logger.info(
          { groupFolder, kind, pattern, source: sourceGroup },
          'promote_learned_trigger: pattern already enabled, no-op',
        );
        break;
      }
      const updatedPatterns = cfg.patterns.map((p, i) =>
        i === idx ? { ...p, enabled: true } : p,
      );
      deps.registerGroup(targetJid, {
        ...targetGroup,
        triggerPatterns: { ...cfg, patterns: updatedPatterns },
      });
      logger.info(
        {
          groupFolder,
          kind,
          pattern,
          pattern_version: target.pattern_version ?? null,
          precision: target.precision,
          sample_count: target.sample_count,
          source: sourceGroup,
        },
        'promote_learned_trigger: flipped enabled false → true',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
      break;
    }

    case 'reenable_learned_trigger': {
      // Re-enable path for #451 item 2. Counterpart to the learner's
      // auto-rollback (`disabled: true`) — flips `disabled: true →
      // false` so the matcher resumes consuming the row. Operator's
      // remit: they've fixed the root cause (e.g. removed a noisy
      // synthetic-identity nickname) and want the proposal active
      // again WITHOUT bumping pattern_version. The learner is the
      // sole writer that ever sets `disabled: true`, so re-enable is
      // by definition operator-initiated. Same `{kind, pattern}`
      // identity + same authorization shape as promote.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      const kind = typeof data.kind === 'string' ? data.kind.trim() : '';
      // Pattern is the identity field used to match the row in
      // triggerPatterns — trimming would break lookups for patterns
      // legitimately stored with significant whitespace. Reject
      // whitespace-only / empty as a separate validation step rather
      // than mutating the value before the lookup.
      const pattern = typeof data.pattern === 'string' ? data.pattern : '';
      if (!groupFolder || !kind || !pattern || pattern.trim().length === 0) {
        logger.warn(
          { data },
          'Invalid reenable_learned_trigger request - groupFolder, kind, and pattern are all required strings',
        );
        break;
      }
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'reenable_learned_trigger: group not registered',
        );
        break;
      }
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized reenable_learned_trigger attempt blocked',
        );
        break;
      }
      const cfg = targetGroup.triggerPatterns;
      if (!cfg) {
        logger.warn(
          { groupFolder },
          'reenable_learned_trigger: group has no triggerPatterns config',
        );
        break;
      }
      const idx = cfg.patterns.findIndex(
        (p) =>
          p.source === 'learned' && p.kind === kind && p.pattern === pattern,
      );
      if (idx === -1) {
        logger.warn(
          { groupFolder, kind, pattern },
          'reenable_learned_trigger: no matching learned pattern',
        );
        break;
      }
      const target = cfg.patterns[idx];
      if (target.disabled !== true) {
        logger.info(
          { groupFolder, kind, pattern, source: sourceGroup },
          'reenable_learned_trigger: pattern was not disabled, no-op',
        );
        break;
      }
      const updatedPatterns = cfg.patterns.map((p, i) =>
        i === idx ? { ...p, disabled: false } : p,
      );
      deps.registerGroup(targetJid, {
        ...targetGroup,
        triggerPatterns: { ...cfg, patterns: updatedPatterns },
      });
      logger.info(
        {
          groupFolder,
          kind,
          pattern,
          pattern_version: target.pattern_version ?? null,
          source: sourceGroup,
        },
        'reenable_learned_trigger: flipped disabled true → false',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
      break;
    }

    case 'delete_learned_trigger': {
      // Permanent-delete path for #451 item 2 (second half). Removes
      // a learned proposal from the array entirely — for proposals
      // that should never come back regardless of root-cause fixes
      // (e.g. a pattern that overlaps with an owner-set entry, or a
      // demoted row whose precision is irrecoverable). Distinct from
      // re-enable: deletion is final and the learner can re-propose
      // the same body later as a fresh row with `pattern_version: 1`.
      // Same authorization shape as promote/reenable.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      const kind = typeof data.kind === 'string' ? data.kind.trim() : '';
      // Pattern is the identity field used to match the row in
      // triggerPatterns — trimming would break lookups for patterns
      // legitimately stored with significant whitespace. Reject
      // whitespace-only / empty as a separate validation step rather
      // than mutating the value before the lookup.
      const pattern = typeof data.pattern === 'string' ? data.pattern : '';
      if (!groupFolder || !kind || !pattern || pattern.trim().length === 0) {
        logger.warn(
          { data },
          'Invalid delete_learned_trigger request - groupFolder, kind, and pattern are all required strings',
        );
        break;
      }
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'delete_learned_trigger: group not registered',
        );
        break;
      }
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized delete_learned_trigger attempt blocked',
        );
        break;
      }
      const cfg = targetGroup.triggerPatterns;
      if (!cfg) {
        logger.warn(
          { groupFolder },
          'delete_learned_trigger: group has no triggerPatterns config',
        );
        break;
      }
      const before = cfg.patterns.length;
      const updatedPatterns = cfg.patterns.filter(
        (p) =>
          !(p.source === 'learned' && p.kind === kind && p.pattern === pattern),
      );
      if (updatedPatterns.length === before) {
        logger.warn(
          { groupFolder, kind, pattern },
          'delete_learned_trigger: no matching learned pattern',
        );
        break;
      }
      deps.registerGroup(targetJid, {
        ...targetGroup,
        triggerPatterns: { ...cfg, patterns: updatedPatterns },
      });
      logger.info(
        {
          groupFolder,
          kind,
          pattern,
          remaining_patterns: updatedPatterns.length,
          source: sourceGroup,
        },
        'delete_learned_trigger: removed learned proposal',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
      break;
    }

    case 'set_additional_tiles': {
      // Partial update: change `containerConfig.additionalTiles` only
      // (#305). Authorization: main-only — this is a trust-adjacent
      // capability (loading extra skill/rule tiles into the chat's
      // container) and a non-main agent must not be able to grant
      // itself capabilities its trust tier wasn't supposed to have.
      // Mirrors `set_trusted` semantics rather than `set_agent_model`'s
      // "owner can change own bill" semantics.
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized set_additional_tiles attempt blocked',
        );
        break;
      }
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      if (!groupFolder) {
        logger.warn(
          { data },
          'Invalid set_additional_tiles request - missing/empty groupFolder',
        );
        break;
      }
      // Accept array (set/replace) or null (clear). Reject anything
      // else so a malformed payload doesn't silently no-op the way an
      // `undefined` would. Element-level validation (string, non-empty,
      // installed) runs below.
      const raw = data.additionalTiles;
      if (raw !== null && !Array.isArray(raw)) {
        logger.warn(
          { data },
          'Invalid set_additional_tiles request - additionalTiles must be array or null',
        );
        break;
      }
      // Look up target group by folder. Same iteration pattern as
      // set_agent_model — the in-memory registry is the source of
      // truth and we want the JID to call deps.registerGroup.
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'set_additional_tiles: group not registered (use register_group first)',
        );
        break;
      }

      const nextContainerConfig: RegisteredGroup['containerConfig'] = {
        ...(targetGroup.containerConfig ?? {}),
      };

      if (raw === null || raw.length === 0) {
        // Explicit clear — drop the field so it serialises as absent
        // (not as JSON null / empty array) and `selectTiles` falls
        // through to the trust-tier baseline.
        delete nextContainerConfig.additionalTiles;
      } else {
        // Element validation: every entry must be a non-empty trimmed
        // string. Reject the whole write on the first malformed entry
        // — partial acceptance ("dropped 'foo' but kept the rest")
        // would silently lose capabilities the operator asked for.
        const cleaned: string[] = [];
        for (const entry of raw) {
          if (typeof entry !== 'string') {
            logger.warn(
              { groupFolder, entry },
              'set_additional_tiles: every entry must be a string',
            );
            return;
          }
          const trimmed = entry.trim();
          if (!trimmed) {
            logger.warn(
              { groupFolder },
              'set_additional_tiles: empty/whitespace tile name rejected',
            );
            return;
          }
          // De-dup at write time so the persisted value is clean and
          // selectTiles doesn't have to do the work on every spawn.
          if (!cleaned.includes(trimmed)) cleaned.push(trimmed);
        }

        // Registry validation: every entry must resolve to an
        // installed tile under `tessl-workspace/.tessl/tiles/<owner>/`.
        // `getInstalledTiles()` returns null when the registry
        // directory itself doesn't exist (cold start, never ran
        // `tessl install`) — treat as "nothing installed" so the
        // operator sees the failure now instead of on next spawn.
        const installed = new Set(getInstalledTiles() ?? []);
        const missing = cleaned.filter((t) => !installed.has(t));
        if (missing.length > 0) {
          logger.warn(
            { groupFolder, missing, requested: cleaned },
            `set_additional_tiles rejected: tile${missing.length === 1 ? '' : 's'} not in registry: ${missing.map((t) => `'${t}'`).join(', ')}`,
          );
          return;
        }

        nextContainerConfig.additionalTiles = cleaned;
      }

      deps.registerGroup(targetJid, {
        ...targetGroup,
        containerConfig: nextContainerConfig,
      });
      logger.info(
        {
          groupFolder,
          additionalTiles: nextContainerConfig.additionalTiles ?? null,
          source: sourceGroup,
        },
        'set_additional_tiles: updated per-group tile overlay',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        true,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
      break;
    }

    case 'nuke_session':
      if (data.groupFolder) {
        // Optional `session` arg narrows the nuke to one slot. Accepted
        // values: 'default', 'maintenance', 'all'. Anything else (or
        // missing) falls back to 'all' — the safe default that preserves
        // pre-parallel behaviour. The value comes from the container's
        // IPC payload so we cast from `unknown` and allowlist.
        const sessionArg = (data as Record<string, unknown>).session;
        const validSession: 'default' | 'maintenance' | 'all' =
          sessionArg === 'default' || sessionArg === 'maintenance'
            ? sessionArg
            : 'all';
        // Optional `skipReentry` (#127): when true, the dispatcher also
        // deletes `.checkpoints/default.md` + `previous.md` after the
        // standard wipe so the next spawn has no reentry Facts to load.
        // Strict boolean check — anything non-true (missing, null, the
        // string "true", etc.) falls back to the safe default of
        // preserving the checkpoint, so a malformed payload can't
        // accidentally erase reentry state.
        const skipReentryArg = (data as Record<string, unknown>).skipReentry;
        const skipReentry = skipReentryArg === true;
        // `sourceGroup` is authoritative (derived from the IPC dir the
        // request arrived in); `data.groupFolder` is only used as a
        // "yes-really-nuke" opt-in flag above and its value isn't honoured
        // downstream. Log sourceGroup to avoid misleading audit trails if
        // they ever differ.
        logger.info(
          { sourceGroup, session: validSession, skipReentry },
          'Session nuke requested via IPC',
        );
        deps.nukeSession(sourceGroup, validSession, { skipReentry });
      }
      break;

    case 'chat_status': {
      // Admin tile only. Returns a structured snapshot per chat: the
      // host-side state the admin needs to diagnose silent containers
      // (running / idle / cooling-down / crashed / not-spawned), tile
      // classification, trigger config, and the latest is_from_me=1
      // message recorded for the chat.
      const resultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized chat_status attempt blocked',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: 'chat_status is admin-tile only',
          }),
        );
        break;
      }

      // Resolve which chats to report on. Four cases:
      //   - both chat_id AND chat_name → reject. Two identifiers that
      //     might disagree is unsafe targeting; force the caller to
      //     pick one. Defense in depth — the MCP tool layer also
      //     blocks this, but a payload arriving directly via the IPC
      //     dir would otherwise let chat_id silently win.
      //   - chat_id provided → report only that one (must be registered).
      //   - chat_name provided → resolve via name match in
      //     registeredGroups (multiple matches → ambiguous error so the
      //     caller can pick the right JID rather than us guessing).
      //   - neither provided → all registered chats.
      const hasChatId =
        typeof data.chat_id === 'string' && data.chat_id.trim().length > 0;
      const hasChatName =
        typeof data.chat_name === 'string' && data.chat_name.trim().length > 0;
      if (hasChatId && hasChatName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'chat_status accepts chat_id OR chat_name, not both — they may disagree',
          }),
        );
        break;
      }
      const targets: string[] = [];
      if (hasChatId) {
        const trimmed = (data.chat_id as string).trim();
        if (!registeredGroups[trimmed]) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_id ${trimmed} not registered`,
            }),
          );
          break;
        }
        targets.push(trimmed);
      } else if (hasChatName) {
        const wanted = (data.chat_name as string).trim();
        const matches = Object.entries(registeredGroups).filter(
          ([, g]) => g.name === wanted,
        );
        if (matches.length === 0) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" did not match any registered chat`,
            }),
          );
          break;
        }
        if (matches.length > 1) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" is ambiguous — matches ${matches.length} chats`,
              candidates: matches.map(([jid]) => jid),
            }),
          );
          break;
        }
        targets.push(matches[0][0]);
      } else {
        targets.push(...Object.keys(registeredGroups));
      }

      // Batch the "latest is_from_me=1 message per chat" lookup into a
      // single grouped query (idx_messages_fromme_chat composite
      // index). Per-target getLastFromMeMessage calls were N
      // statement compilations + N scan-and-sort passes; this is one
      // query for any N.
      const lastMessages = getLastFromMeMessages(targets);

      // Resolve the orchestrator-wide default once. Each row then runs
      // the per-group resolver against this baseline so the value
      // reported here matches what `runContainerAgent` actually sets
      // as `AGENT_MODEL` on spawn — including the typo-fallback
      // behavior of `resolvePerGroupAgentModel`.
      const globalDefaultAgentModel = resolveAgentModel(
        process.env.AGENT_MODEL,
      );

      const rows = targets.map((jid) => {
        const group = registeredGroups[jid];
        const tile: 'admin' | 'trusted' | 'untrusted' = group.isMain
          ? 'admin'
          : group.containerConfig?.trusted
            ? 'trusted'
            : 'untrusted';
        // requiresTrigger defaults differ per tile: main groups bypass
        // the trigger entirely (privileged inbox), while non-main groups
        // require the trigger unless explicitly opted out. Mirror the
        // canSenderInteract logic so the reported value matches what
        // the orchestrator actually enforces.
        const triggered = group.isMain
          ? false
          : group.requiresTrigger !== false;
        const last = lastMessages.get(jid) ?? null;
        return {
          chat_id: jid,
          chat_name: group.name,
          trigger: triggered ? 'triggered' : 'untriggered',
          tile,
          last_ayeaye_message: last
            ? {
                timestamp: last.timestamp,
                // Truncate to keep the response small even if the
                // agent sent a multi-kilobyte reply. 200 chars matches
                // what fits comfortably in the admin's chat preview.
                content_snippet:
                  last.content.length > 200
                    ? last.content.slice(0, 200) + '…'
                    : last.content,
              }
            : null,
          containers: deps.getContainerStatus
            ? {
                default: deps.getContainerStatus(jid, 'default'),
                maintenance: deps.getContainerStatus(jid, 'maintenance'),
              }
            : { default: 'not-spawned', maintenance: 'not-spawned' },
          // Effective AGENT_MODEL for this group's next spawn — the
          // per-group `containerConfig.agentModel` override resolved
          // against the orchestrator-wide default. Surfaced for cost
          // attribution / audit so operators don't have to grep spawn
          // logs to learn which group runs which model (#395 follow-up).
          // Mirrors the resolver at the spawn site exactly: typo-bad
          // overrides resolve to the global default here too.
          effective_agent_model: resolvePerGroupAgentModel(
            group.containerConfig?.agentModel,
            globalDefaultAgentModel,
          ),
        };
      });

      logger.info(
        { sourceGroup, count: rows.length },
        'chat_status served via IPC',
      );
      fs.writeFileSync(
        resultPath,
        JSON.stringify({ stdout: JSON.stringify({ chats: rows }) }),
      );
      break;
    }

    case 'inspect_gate_decisions': {
      // Admin tile only. Returns the most-recent gate-decision records
      // for a chat — one row per `evaluateGateChain` per-message call,
      // captured from the canonical INFO line `'gate decision'` in
      // `data/host-logs/orchestrator.log` (#443). Logs are the
      // non-purgeable substrate the design landed on after rejecting a
      // SQLite table (every-message persistence on durable media isn't
      // worth the retention cost when logs already rotate via
      // `scripts/logrotate.sh`); the file is bind-mounted RO into the
      // admin agent container so the response payload is the same data
      // the agent could grep for itself, just structured.
      const resultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized inspect_gate_decisions attempt blocked',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: 'inspect_gate_decisions is admin-tile only',
          }),
        );
        break;
      }
      // chat_id is required — the canonical use case is "tell me about
      // chat X"; an unbounded scan over every chat is not what this
      // tool is for. Listing across chats would also require a much
      // larger limit and risk exposing cross-chat traffic in a single
      // response.
      const chatId =
        typeof data.chat_id === 'string' ? data.chat_id.trim() : '';
      if (chatId.length === 0) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: 'inspect_gate_decisions requires chat_id',
          }),
        );
        break;
      }
      // `limit` defaults to 10 — small enough that the response fits
      // comfortably in an MCP-tool text reply, large enough to cover
      // a recent burst when the user asks "what did the gate think
      // about the last few messages". Cap at 100 so a typo can't
      // request a multi-megabyte payload.
      let limit = 10;
      if (typeof data.limit === 'number' && Number.isInteger(data.limit)) {
        if (data.limit < 1) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: 'limit must be a positive integer (1–100)',
            }),
          );
          break;
        }
        limit = Math.min(data.limit, 100);
      }
      const messageId =
        typeof data.message_id === 'string' && data.message_id.length > 0
          ? data.message_id
          : undefined;
      const records = readHostLog(hostLogsOrchestratorFile());
      const hits = findGateDecisions(records, {
        chatJid: chatId,
        messageId,
        limit,
      });
      logger.info(
        {
          sourceGroup,
          chatId,
          messageId,
          limit,
          hitCount: hits.length,
        },
        'inspect_gate_decisions served via IPC',
      );
      fs.writeFileSync(
        resultPath,
        JSON.stringify({ stdout: JSON.stringify({ decisions: hits }) }),
      );
      break;
    }

    case 'nuke_chat': {
      // Admin tile only. Cross-chat nuke — looks up the target by
      // chat_id or chat_name and forwards to the same wipeSessionJsonl
      // path the per-chat nuke_session uses. Hard-fails when neither
      // identifier is provided so admin can never accidentally nuke
      // its own chat by omission (the nuke_session tool already does
      // "this chat" — nuke_chat is only useful when targeting another).
      const resultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn({ sourceGroup }, 'Unauthorized nuke_chat attempt blocked');
        fs.writeFileSync(
          resultPath,
          JSON.stringify({ error: 'nuke_chat is admin-tile only' }),
        );
        break;
      }

      const hasId =
        typeof data.chat_id === 'string' && data.chat_id.trim().length > 0;
      const hasName =
        typeof data.chat_name === 'string' && data.chat_name.trim().length > 0;
      if (!hasId && !hasName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'nuke_chat requires chat_id or chat_name — admin always operates cross-chat, never on the implicit current chat',
          }),
        );
        break;
      }
      // Two identifiers are an unsafe-targeting smell — if they
      // disagree, silently picking one is worse than refusing. Reject
      // here too (the MCP tool layer also blocks the same case).
      if (hasId && hasName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'nuke_chat accepts chat_id OR chat_name, not both — they may disagree',
          }),
        );
        break;
      }

      let targetJid = '';
      if (hasId) {
        const trimmed = (data.chat_id as string).trim();
        if (!registeredGroups[trimmed]) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: `chat_id ${trimmed} not registered` }),
          );
          break;
        }
        targetJid = trimmed;
      } else {
        const wanted = (data.chat_name as string).trim();
        const matches = Object.entries(registeredGroups).filter(
          ([, g]) => g.name === wanted,
        );
        if (matches.length === 0) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" did not match any registered chat`,
            }),
          );
          break;
        }
        if (matches.length > 1) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" is ambiguous — matches ${matches.length} chats`,
              candidates: matches.map(([jid]) => jid),
            }),
          );
          break;
        }
        targetJid = matches[0][0];
      }

      const targetGroup = registeredGroups[targetJid];
      const sessionArg = data.session;
      const validSession: 'default' | 'maintenance' | 'all' =
        sessionArg === 'default' ||
        sessionArg === 'maintenance' ||
        sessionArg === 'all'
          ? sessionArg
          : 'all';

      // Snapshot pre-nuke status to determine which slots actually had
      // a live container to kill. nukeSession ALWAYS wipes JSONL on
      // disk regardless of whether anything was running; the
      // user-visible status enum (per the issue spec) reports the
      // *live-container* outcome so admin can tell whether the call
      // actually freed any resources.
      const slotsRequested: Array<'default' | 'maintenance'> =
        validSession === 'all' ? ['default', 'maintenance'] : [validSession];
      const killedSessions: Array<'default' | 'maintenance'> = [];
      const getStatus = deps.getContainerStatus;
      for (const slot of slotsRequested) {
        const wasActive =
          getStatus &&
          (getStatus(targetJid, slot) === 'running' ||
            getStatus(targetJid, slot) === 'idle');
        if (wasActive) killedSessions.push(slot);
      }

      try {
        deps.nukeSession(targetGroup.folder, validSession);
        // Per the issue's status enum: 'success' when at least one
        // live container was killed; 'noop' when nothing was running
        // (even though the on-disk wipe still happened — see the
        // pre-snapshot comment above). 'partial' is reserved for a
        // future per-slot-failure signal from nukeSession; today
        // nukeSession is fire-and-forget per slot, so we can't
        // distinguish partial failure from full success without a
        // contract change. 'error' is reported only when nukeSession
        // throws — the catch branch below.
        const status: 'success' | 'noop' =
          killedSessions.length > 0 ? 'success' : 'noop';
        logger.info(
          {
            sourceGroup,
            targetJid,
            session: validSession,
            killedSessions,
            status,
          },
          'nuke_chat completed via IPC',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            stdout: JSON.stringify({
              chat_id: targetJid,
              chat_name: targetGroup.name,
              killed_sessions: killedSessions,
              status,
            }),
          }),
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error({ sourceGroup, targetJid, err }, 'nuke_chat failed');
        // Top-level `error` field — runHostOperation in the
        // agent-runner only treats `result.error` as a tool failure
        // and surfaces `isError: true` to the MCP caller. Burying the
        // failure inside `stdout` would make the call look like a
        // success to Claude, which would then move on as if the wipe
        // ran. Include the structured payload alongside so the admin
        // can still see what was attempted.
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: `nuke_chat failed for ${targetJid}: ${msg}`,
            chat_id: targetJid,
            chat_name: targetGroup.name,
            killed_sessions: [],
            status: 'error',
          }),
        );
      }
      break;
    }

    case 'send_message_to_chat': {
      // Admin tile only. Cross-chat broadcast — resolves chat_id or
      // chat_name against registeredGroups, then dispatches via the
      // same channel router used by the existing 'message' IPC path
      // (pool send for sender-tagged Telegram broadcasts, direct send
      // otherwise). Replaces the schedule_task + once: now+5s kludge
      // the agent used to reach when asked "post X to #other-chat".
      //
      // Bypassing the MESSAGES_DIR fire-and-forget path is deliberate:
      // the tool's contract surfaces failure (unknown JID, ambiguous
      // name, blocked-by-user, rate-limit) synchronously to the
      // calling agent so it can retry or explain. MESSAGES_DIR drops
      // failures into a log nobody reads from the agent's POV.
      const resultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized send_message_to_chat attempt blocked',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({ error: 'send_message_to_chat is admin-tile only' }),
        );
        break;
      }

      const rawText = typeof data.text === 'string' ? data.text : '';
      if (rawText.trim().length === 0) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: 'send_message_to_chat requires non-empty text',
          }),
        );
        break;
      }

      const hasId =
        typeof data.chat_id === 'string' && data.chat_id.trim().length > 0;
      const hasName =
        typeof data.chat_name === 'string' && data.chat_name.trim().length > 0;
      if (!hasId && !hasName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'send_message_to_chat requires chat_id or chat_name — admin always operates cross-chat. Use the regular send_message tool to reply in the current chat.',
          }),
        );
        break;
      }
      // Two identifiers are an unsafe-targeting smell — same rule as
      // chat_status / nuke_chat. Reject before resolving so the caller
      // can't get a silent JID-wins-over-name surprise.
      if (hasId && hasName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'send_message_to_chat accepts chat_id OR chat_name, not both — they may disagree',
          }),
        );
        break;
      }

      let targetJid = '';
      if (hasId) {
        const trimmed = (data.chat_id as string).trim();
        if (!registeredGroups[trimmed]) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: `chat_id ${trimmed} not registered` }),
          );
          break;
        }
        targetJid = trimmed;
      } else {
        const wanted = (data.chat_name as string).trim();
        const matches = Object.entries(registeredGroups).filter(
          ([, g]) => g.name === wanted,
        );
        if (matches.length === 0) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" did not match any registered chat`,
            }),
          );
          break;
        }
        if (matches.length > 1) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" is ambiguous — matches ${matches.length} chats`,
              candidates: matches.map(([jid]) => jid),
            }),
          );
          break;
        }
        targetJid = matches[0][0];
      }

      const targetGroup = registeredGroups[targetJid];
      // Trim before treating as present — a payload of `'   '` would
      // otherwise route through the pool path (Boolean(' ') is true)
      // and bind a pool bot to a whitespace identity. Empty-after-trim
      // collapses to undefined so routing matches the documented
      // contract ("named identity" → pool; nothing → direct).
      const senderRaw =
        typeof data.sender === 'string' ? data.sender.trim() : '';
      const sender = senderRaw.length > 0 ? senderRaw : undefined;
      const wantsPin = data.pin === true;

      // Strip <internal> tags for parity with the regular 'message'
      // handler — keeps agent reasoning out of cross-chat broadcasts.
      // We don't apply the maintenance-prefix here: this tool is an
      // explicit admin broadcast, and tagging it `[M]` would mislead
      // the recipient into thinking a scheduled-task heartbeat fired.
      const cleanText = stripInternalTags(rawText);
      if (!cleanText) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'send_message_to_chat: text was empty after stripping <internal> tags',
          }),
        );
        break;
      }

      try {
        // Mirror the routing decision in the 'message' IPC handler:
        // sender + Telegram → bot pool (named identity), else the
        // channel's default sendMessage. Pool sends don't expose a
        // pin hook, so pin is silently dropped on the pool path —
        // matching existing send_message behavior. The `pinned` field
        // in the success payload tells the agent what actually
        // happened so it can flag the asymmetry to the user.
        const usePool = Boolean(sender && targetJid.startsWith('tg:'));
        let sentMsgId: string | undefined;
        let pinned = false;
        if (usePool) {
          const poolResult = await sendPoolMessage(
            targetJid,
            cleanText,
            sender!,
            sourceGroup,
          );
          sentMsgId = typeof poolResult === 'string' ? poolResult : undefined;
        } else {
          const directResult = await deps.sendMessage(targetJid, cleanText);
          sentMsgId =
            typeof directResult === 'string' ? directResult : undefined;
          if (wantsPin && sentMsgId && deps.pinMessage) {
            await deps.pinMessage(targetJid, sentMsgId);
            pinned = true;
          }
        }

        // Gate the messages.db write on send success — see
        // shouldStoreBotMessage for the phantom-row rationale (#232).
        // A failed cross-chat send that wrote a bot- row would silence
        // the target chat's heartbeat / unanswered-cron on a chat the
        // recipient never received the message in. Critical here
        // because the entire purpose of this tool is sending into
        // chats the agent isn't watching.
        if (!shouldStoreBotMessage(targetJid, sentMsgId)) {
          logger.error(
            {
              sourceGroup,
              targetJid,
              contentLen: cleanText.length,
              usedPool: usePool,
            },
            'send_message_to_chat: target returned no message id; skipping DB row',
          );
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `send_message_to_chat: target ${targetJid} did not return a message id — likely blocked, rate-limited, or an invalid recipient. No DB row written.`,
              chat_id: targetJid,
              chat_name: targetGroup.name,
              status: 'failed',
            }),
          );
          break;
        }

        const botRowId = `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        // Persist the identity that ACTUALLY went out, not what the
        // caller asked for. `sender` only takes effect on the pool
        // path (Telegram + sender set). On the direct path (non-
        // Telegram, or Telegram without sender) the message goes from
        // the channel's default identity, so storing the caller's
        // `sender` would make the DB row claim a persona that never
        // touched the wire — misleading the heartbeat / unanswered-
        // cron / future audits about who replied.
        const effectiveSender = usePool && sender ? sender : ASSISTANT_NAME;
        storeMessage({
          id: botRowId,
          chat_jid: targetJid,
          sender: effectiveSender,
          sender_name: effectiveSender,
          content: cleanText,
          timestamp: new Date().toISOString(),
          is_from_me: true,
          is_bot_message: true,
          telegram_message_id: sentMsgId,
        });

        logger.info(
          {
            sourceGroup,
            targetJid,
            sentMsgId,
            usedPool: usePool,
            pinned,
          },
          'send_message_to_chat completed',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            stdout: JSON.stringify({
              chat_id: targetJid,
              chat_name: targetGroup.name,
              sent_message_id: sentMsgId,
              pinned,
              status: 'success',
            }),
          }),
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error(
          { sourceGroup, targetJid, err },
          'send_message_to_chat failed',
        );
        // Top-level `error` field — runHostOperation in the
        // agent-runner only treats `result.error` as a tool failure
        // and surfaces `isError: true` to the MCP caller. Same
        // contract as nuke_chat's catch branch.
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: `send_message_to_chat failed for ${targetJid}: ${msg}`,
            chat_id: targetJid,
            chat_name: targetGroup.name,
            status: 'error',
          }),
        );
      }
      break;
    }

    // --- Named host operations ---

    case 'sync_tripit':
      if (data.requestId) {
        const groupDir = path.resolve(process.cwd(), 'groups', sourceGroup);
        const scriptPath = path.join(groupDir, 'scripts', 'sync-tripit.sh');
        if (!fs.existsSync(scriptPath)) {
          const errPath = scriptResultPath(sourceGroup, data);
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
            const resultPath = scriptResultPath(sourceGroup, data);
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
              // #542 — Persist `segments[]` onto the singleton
              // `tz_state` row before writing the script-result file
              // so the agent caller's response and the host's
              // tz_state row are coherent on the same wall-clock
              // tick. The script invokes `node sync.mjs --output=json`
              // (silences regular `console.log`, emits a single JSON
              // payload); a malformed stdout is logged and skipped —
              // the user's sync still succeeded from the upstream's
              // POV, and the next heartbeat advisory will reconcile
              // current_tz once the cache is good. Failures from the
              // persistence helper itself (SqliteError, programming
              // bugs) propagate per `coding-policy: error-handling`
              // — they're separate from a parse failure and shouldn't
              // be silently downgraded to a parse warning.
              let parsed: { segments?: unknown } | null = null;
              try {
                parsed = JSON.parse(stdout) as { segments?: unknown };
              } catch (parseErr) {
                if (!(parseErr instanceof SyntaxError)) throw parseErr;
                // Per `coding-policy: no-secrets`: the script runs
                // with `TRIPIT_ICAL_URL` / `RECLAIM_API_TOKEN` /
                // Google OAuth in its environment, and a malformed-
                // stdout payload can carry credential-bearing URLs
                // or token fragments (e.g. an unhandled-error stack
                // that captured the full request context). Log only
                // non-sensitive shape diagnostics — length and the
                // SyntaxError's parser-reported position — so an
                // operator has enough to triage without paging the
                // script's raw bytes through structured logs.
                logger.warn(
                  {
                    sourceGroup,
                    err: parseErr.message,
                    stdoutLen: stdout.length,
                  },
                  'sync_tripit: stdout did not parse as JSON — segments not persisted (heartbeat advisory will recover on next good run)',
                );
              }
              if (parsed !== null) {
                const segments: TripitSegment[] = Array.isArray(parsed.segments)
                  ? (parsed.segments as TripitSegment[])
                  : [];
                // #584 — pass an `onTzFlipped` hook so a tz change
                // resolved from the TripIt segment walk also
                // invalidates cached `next_run` values on active
                // `schedule_timezone='local'` rows. The recompute is
                // wrapped inside the writer with a narrowed catch —
                // only transient SQLite contention (`SQLITE_BUSY` /
                // `SQLITE_LOCKED`) is swallowed-with-warn so the
                // canonical `tz_state` UPDATE that already landed
                // stays consistent; programming bugs, persistent DB
                // failures, and unexpected throws propagate back to
                // this IPC handler per `coding-policy: error-handling`.
                applyTripitSegmentsToTzState({ segments }, new Date(), () => {
                  recomputeLocalSchedules(getCurrentTz, new Date());
                });
              }
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
          const errPath = scriptResultPath(sourceGroup, data);
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
            // 4MB. Pre-#146 ceiling was 1MB, which silently truncated
            // realistic Trakt history payloads (~328 shows + 117
            // movies + 77 ratings = >1MB JSON) — execFile errored with
            // ENOBUFS and the agent's only signal was `Command
            // failed: python3 ...` because the empty-stderr /
            // dropped-stdout path ate the actual cause. 4MB headroom
            // covers years of library growth without inflating the
            // worst-case success payload to an unreasonable size: the
            // success path below writes raw stdout (truncating would
            // produce broken JSON the agent can't parse), so the
            // ceiling here directly bounds what the agent has to
            // ingest. ERROR-path stdout/stderr is head/tail-truncated
            // at ~64KB each below — that's safe to truncate because
            // the agent only uses it as diagnostic prose, not as
            // structured data.
            maxBuffer: 4 * 1024 * 1024,
          },
          (error, stdout, stderr) => {
            const resultPath = scriptResultPath(sourceGroup, data);
            // Cap each captured stream at ~64KB on the error path to
            // keep the script-result JSON reasonably small while still
            // preserving the most useful diagnostics. 64KB is
            // empirically enough to carry a full Python traceback plus
            // a few hundred lines of leading context — the truncation
            // issue the original `stderr.slice(-500)` caused was that
            // a long traceback was clipped to its tail, hiding the
            // most-informative top-of-stack lines. (Script-result
            // files don't go through `startIpcWatcher`'s 1MB inbound
            // task/message check, so the cap here is pragmatic
            // payload-size discipline, not a hard wall.)
            const ERROR_PAYLOAD_CAP = 64 * 1024;
            const headTail = (s: string) => {
              if (s.length <= ERROR_PAYLOAD_CAP) return s;
              const half = Math.floor(ERROR_PAYLOAD_CAP / 2);
              return (
                s.slice(0, half) +
                `\n…[${s.length - ERROR_PAYLOAD_CAP} bytes elided]…\n` +
                s.slice(s.length - half)
              );
            };
            if (error) {
              // ExecException-typed fields the SDK populates: `code`
              // is whatever Node attaches — for execFile that can be
              // a numeric process exit code (a script's `sys.exit(2)`
              // surfaces as `2`) OR a string error code for spawn-
              // side failures (`ERR_CHILD_PROCESS_STDIO_MAXBUFFER`
              // on stdout overflow, `ENOENT` if the binary is
              // missing, etc.). Surface it as-is and let the agent
              // treat it as a free-form indicator; trying to coerce
              // it to a single shape (always int / always string)
              // would lose information. `killed` is true on
              // timeout / SIGTERM — diagnostic gold the previous
              // `error.message`-only payload was throwing away,
              // because 401 (auth expired), ENOBUFS (stdout
              // overflow), and SIGTERM (timeout) all looked
              // identical at `Command failed: python3 ...`.
              const execErr = error as NodeJS.ErrnoException & {
                code?: string | number;
                killed?: boolean;
              };
              logger.error(
                {
                  sourceGroup,
                  error: error.message,
                  exitCode: execErr.code,
                  killed: execErr.killed,
                  stderrLen: stderr.length,
                  stdoutLen: stdout.length,
                },
                'fetch_trakt_history failed',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: error.message,
                  exit_code: execErr.code ?? null,
                  killed: execErr.killed ?? false,
                  // Both streams preserved (head + tail). Pre-fix the
                  // handler dropped stdout entirely on the error
                  // path, even though the Python script may have
                  // printed a partial JSON envelope or progress
                  // diagnostics there before crashing. And stderr
                  // was tail-only (-500), hiding the top-of-stack
                  // lines that actually identify the failure (auth,
                  // schema, network, etc.).
                  stdout: headTail(stdout),
                  stderr: headTail(stderr),
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

    case 'fetch_markdown':
      if (data.requestId) {
        const resultPath = scriptResultPath(sourceGroup, data);

        const parsed = parseFetchMarkdownUrl(data.url);
        if (!parsed.ok) {
          fs.writeFileSync(resultPath, JSON.stringify({ error: parsed.error }));
          break;
        }
        const parsedUrl = parsed.url;

        const flags = buildSnitchmdFlags(data);

        // Cache directory on the HOST filesystem — the snitchmd sibling
        // container is launched via the orchestrator's docker.sock, so
        // -v mount paths must reference the host's view (HOST_PROJECT_ROOT),
        // not the orchestrator's /app/store. Same convention as the agent
        // container mounts in container-runner.ts.
        const hostCacheDir = path.join(
          HOST_PROJECT_ROOT,
          'store',
          'snitchmd-cache',
        );
        // Also mkdir locally so the path is visible to the orchestrator
        // process (e.g. for size-rollup or cleanup tooling). The actual
        // write into the cache is performed by the sibling container as
        // its own user; we just want the directory to exist.
        fs.mkdirSync(path.join(STORE_DIR, 'snitchmd-cache'), {
          recursive: true,
        });

        // snitchmd is an app-level dependency (a renderer + content
        // extractor), not an API contract — we WANT to ride the latest
        // CloakBrowser fingerprint updates as anti-bot detection
        // evolves, and snitchmd's output shape is stable across point
        // releases. Operators who need a reproducible build can pin to
        // a specific tag or `sha256:…` digest via `SNITCHMD_IMAGE`
        // without a code change.
        const snitchmdImage =
          process.env.SNITCHMD_IMAGE || 'syabro/snitchmd:latest';

        logger.info(
          {
            sourceGroup,
            host: parsedUrl.host,
            flags: flags.filter((f) => !f.startsWith('http')),
          },
          'Running fetch_markdown',
        );

        execFile(
          'docker',
          [
            'run',
            '--rm',
            '-v',
            `${hostCacheDir}:/cache`,
            snitchmdImage,
            parsedUrl.toString(),
            ...flags,
          ],
          {
            // First call cold-pulls the image — give it room. Subsequent
            // calls are cached and complete in well under 30s.
            timeout: 240_000,
            // snitchmd's markdown output can run into the megabytes for
            // wiki/long articles; cap at 16 MiB so a runaway page doesn't
            // exhaust orchestrator memory before the agent's --max-chars
            // truncation kicks in.
            maxBuffer: 16 * 1024 * 1024,
          },
          (error, stdout, stderr) => {
            if (error) {
              // execFile attaches `code` (numeric exit code or string like
              // ETIMEDOUT) and `killed` (timeout signal). Surface both so
              // runHostOperation's diagnostic relay can show the agent
              // which failure mode hit (cf. #146 retry context).
              const execErr = error as NodeJS.ErrnoException & {
                code?: number | string;
                killed?: boolean;
              };
              // Per `jbaruch/coding-policy: no-secrets`: Node's `execFile`
              // packs the full command line — including the target URL —
              // into `error.message` as `Command failed: docker run ...
              // <url> --json ...`. A URL with query-string auth (session
              // token, signed-URL signature) would leak into both the
              // structured log AND the result-file the agent reads. Emit
              // a fixed-shape message that names only the host + exit
              // mode; the host attribute itself isn't a secret and the
              // operator can correlate with the structured log.
              const safeError = `fetch_markdown failed for ${parsedUrl.host} (exit_code: ${execErr.code ?? 'unknown'}${execErr.killed ? ', killed' : ''})`;
              // Defense-in-depth: snitchmd's stderr is normally just
              // `snitchmd: title=... quality=... chars=...`, but a
              // Playwright / Chromium fault could echo the input URL.
              // Scrub the exact URL we passed in before writing so a
              // query-string auth secret can't leak via stderr.
              const urlString = parsedUrl.toString();
              const safeStderr = stderr
                .slice(-2000)
                .split(urlString)
                .join('<URL>');
              logger.warn(
                {
                  sourceGroup,
                  host: parsedUrl.host,
                  exitCode: execErr.code,
                  killed: execErr.killed,
                  stderr: safeStderr.slice(-500),
                },
                'fetch_markdown failed',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: safeError,
                  exit_code: execErr.code,
                  killed: execErr.killed,
                  stderr: safeStderr,
                }),
              );
              return;
            }
            // snitchmd --json writes a single JSON object on stdout. Pass
            // the markdown body up to the agent as plain text (the
            // <untrusted-input> envelope is applied client-side via the
            // READ_TOOL_PATTERNS row in untrusted-input-wrap.ts — #321).
            let payload: {
              markdown?: string;
              title?: string;
              final_url?: string;
              quality?: number | null;
              chars?: number;
            } = {};
            try {
              payload = JSON.parse(stdout);
            } catch (parseErr) {
              if (!(parseErr instanceof SyntaxError)) throw parseErr;
              logger.warn(
                {
                  sourceGroup,
                  host: parsedUrl.host,
                  stdoutLen: stdout.length,
                },
                'fetch_markdown: stdout did not parse as JSON',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: 'fetch_markdown: snitchmd produced non-JSON output',
                  stderr: stderr.slice(-2000),
                  stdout: stdout.slice(-2000),
                }),
              );
              return;
            }
            const markdown = payload.markdown ?? '';
            const header = formatSnitchmdHeader(payload, parsedUrl.toString());
            logger.info(
              {
                sourceGroup,
                host: parsedUrl.host,
                chars: payload.chars ?? markdown.length,
                quality: payload.quality,
              },
              'fetch_markdown completed',
            );
            fs.writeFileSync(
              resultPath,
              JSON.stringify({
                stdout: header + markdown,
                stderr: stderr.slice(-500) || undefined,
              }),
            );
          },
        );
      }
      break;

    case 'github_backup':
      if (data.requestId) {
        // Authorization: github_backup performs a host-side filesystem
        // sync + `git push` using GITHUB_TOKEN — same privilege class
        // as `audible_backup`, `dominos_pizza`, `promote_staging`, all
        // of which gate on `isMain`. Untrusted-tier groups have a
        // separate `#324` token gate at the container layer (the
        // confirmation-tokens hook), but the host-side gate here
        // shrinks the blast radius further: a compromised non-main
        // container can't trigger the backup pipeline by writing an
        // IPC task file directly.
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized github_backup attempt');
          break;
        }

        const backupDir = path.join(GROUPS_DIR, sourceGroup, 'backup-repo');
        const dbPath = path.join(STORE_DIR, 'messages.db');
        const resultPath = scriptResultPath(sourceGroup, data);

        // Sync live group state into backup-repo BEFORE git plumbing.
        // The sync mirrors groups/global/ and every non-hidden
        // groups/<name>/ subdir into backup-repo/{global,groups/<name>}
        // with delete-on-missing semantics, and dumps the SQLite
        // state-table surface into backup-repo/state/<table>.sql.
        // Policy: back up everything under groups/ that the runtime
        // mutates and that ISN'T reproducible from deploy or
        // `tessl install` — see `src/backup-sync.ts` for the denylist
        // (.tessl, .claude, node_modules, dist, logs, tmp,
        // conversations, *.bak-*, etc.).
        // syncBackupRepo also validates groupsRoot / backupDir
        // existence and throws an actionable error if either is
        // missing — the catch block below converts that into a
        // structured `{ error, stage: 'sync' }` envelope.
        let syncSummary: SyncResult;
        try {
          syncSummary = syncBackupRepo({
            groupsRoot: GROUPS_DIR,
            backupDir,
            dbPath,
          });
        } catch (e) {
          // Non-Error throws (TypeScript allows `throw 42`) bubble up
          // — those indicate a bug, not an operational sync failure.
          if (!(e instanceof Error)) throw e;
          logger.error(
            {
              sourceGroup,
              groupsRoot: GROUPS_DIR,
              backupDir,
              dbPath,
              error: e.message,
            },
            'github_backup sync failed',
          );
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: e.message, stage: 'sync' }),
          );
          break;
        }

        const commitMsg =
          data.message || `backup: ${new Date().toISOString().split('T')[0]}`;
        logger.info(
          {
            sourceGroup,
            backupDir,
            commitMsg,
            copied: syncSummary.copied.length,
            removed: syncSummary.removed.length,
            dumped: syncSummary.dumped.length,
            skipped: syncSummary.skipped.length,
          },
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
            `cd "${backupDir}" && git add -A && (git diff --cached --quiet && echo '{"committed":false,"stdout":"Nothing to commit."}' || (git commit -m "${commitMsg.replace(/"/g, '\\"')}" && git push && echo '{"committed":true,"stdout":"Committed and pushed."}'))`,
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
                  stage: 'git',
                  sync_summary: syncSummary,
                }),
              );
            } else {
              // stdout is the JSON echo from the bash script. The
              // catch is narrowed to SyntaxError per
              // `jbaruch/coding-policy: error-handling` — only the
              // expected "malformed-JSON-from-bash" path falls back
              // to the raw-stdout shape; any other thrown class
              // (programmer error, OOM, runtime fault) propagates as
              // a real failure.
              const lastLine = stdout.trim().split('\n').pop() ?? '';
              let parsed: { committed?: boolean; stdout?: string };
              try {
                parsed = JSON.parse(lastLine) as typeof parsed;
              } catch (e) {
                if (!(e instanceof SyntaxError)) throw e;
                parsed = { stdout: stdout.trim() };
              }
              fs.writeFileSync(
                resultPath,
                JSON.stringify({ ...parsed, sync_summary: syncSummary }),
              );
              logger.info(
                { sourceGroup, committed: parsed.committed },
                'github_backup completed',
              );
            }
          },
        );
      }
      break;

    case 'sessionize_get_event':
      if (data.requestId && data.slug) {
        const sessionizeResultPath = scriptResultPath(sourceGroup, data);

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
        const cfpsResultPath = scriptResultPath(sourceGroup, data);

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

        const audibleResultPath = scriptResultPath(sourceGroup, data);

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
                fs.writeFileSync(audibleResultPath, JSON.stringify(parsed));
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

    case 'dominos_pizza':
      if (data.requestId) {
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized dominos_pizza attempt');
          break;
        }

        const dominosResultPath = scriptResultPath(sourceGroup, data);

        const dominosCommand = data.command || '';
        const dominosPayload = data.payload || '';
        const dominosConfirm = data.confirm === true;
        logger.info(
          { sourceGroup, command: dominosCommand, confirm: dominosConfirm },
          'Running dominos_pizza',
        );

        const payloadStr =
          typeof dominosPayload === 'string'
            ? dominosPayload
            : JSON.stringify(dominosPayload);

        const dominosArgs: string[] = [
          'run',
          '--rm',
          'dominos-order:latest',
          dominosCommand,
          ...(payloadStr ? [payloadStr] : []),
          ...(dominosConfirm ? ['--confirm'] : []),
        ];

        execFile(
          'docker',
          dominosArgs,
          {
            cwd: process.cwd(),
            env: {
              PATH: process.env.PATH || '/usr/bin:/bin',
              HOME: process.env.HOME || '/root',
            },
            timeout: 120_000,
            maxBuffer: 1024 * 1024,
          },
          (error, stdout, stderr) => {
            if (error) {
              logger.error(
                { sourceGroup, error: error.message, stderr },
                'dominos_pizza failed',
              );
              fs.writeFileSync(
                dominosResultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
            } else {
              logger.info(
                { sourceGroup, command: dominosCommand },
                'dominos_pizza completed',
              );
              try {
                const parsed = JSON.parse(stdout);
                if (stderr) parsed.logs = stderr.slice(-1000);
                fs.writeFileSync(dominosResultPath, JSON.stringify(parsed));
              } catch {
                fs.writeFileSync(
                  dominosResultPath,
                  JSON.stringify({ raw: stdout, logs: stderr?.slice(-1000) }),
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

        const promoteResultPath = scriptResultPath(sourceGroup, data);

        if (!KNOWN_TILE_NAMES.has(data.tileName)) {
          logger.warn(
            { sourceGroup, tileName: data.tileName },
            'promote_staging rejected: tileName not in allowlist',
          );
          fs.writeFileSync(
            promoteResultPath,
            JSON.stringify({
              error: `Invalid tileName "${data.tileName}". Allowed: ${[...KNOWN_TILE_NAMES].join(', ')}.`,
            }),
          );
          break;
        }

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
            // 15 minutes. promote-to-tile-repo.sh runs `tessl skill
            // review --optimize` on each staged skill, and tessl
            // itself tells you that each review "can take up to 1
            // minute." A bulk promote (`skillName=all`) against a tile
            // with 10+ staged skills easily blows past the old
            // 5-minute cap, which observably killed every bulk promote
            // AyeAye tried and returned a mid-run truncated error. 15
            // min fits the typical 10-15 skill worst case with
            // headroom; larger bulk promotes should split the staging
            // directory into smaller batches — every skill is reviewed
            // (no per-skill opt-out).
            timeout: 900_000,
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
                'promote_staging opened PR on tile repo',
              );
              fs.writeFileSync(
                promoteResultPath,
                JSON.stringify({ stdout: stdout.trim() }),
              );

              // Post-promote `tessl update` + session clear used to
              // run here on a 5-minute delay, predicated on the old
              // flow that pushed directly to tile main and triggered
              // GHA publish within ~5min. New flow opens a PR and
              // requests Copilot review; publish only happens after
              // the PR is merged (which could be minutes, hours, or
              // never if Copilot/human rejects it). The auto-update
              // would fire against a registry that hasn't changed
              // yet — at best a no-op, at worst tearing down sessions
              // for no reason. The agent now calls the `tessl_update`
              // MCP tool explicitly after the PR merges; a periodic
              // 15-min catch-up in index.ts covers missed invocations.
            }
          },
        );
      }
      break;

    case 'tessl_update':
      if (data.requestId) {
        const tesslResultPath = scriptResultPath(sourceGroup, data);
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized tessl_update attempt');
          fs.writeFileSync(
            tesslResultPath,
            JSON.stringify({
              error: 'Only the main group can trigger tessl_update.',
            }),
          );
          break;
        }

        // #496 — gate the on-demand update on the same fresh-lock check
        // the periodic catch-up uses. A scheduled task mid-flight has
        // its skill holding `follow_me_tasks.pending_run_at`; running
        // tessl_update now would force-close the maintenance container
        // and orphan the lock. Defer with a structured response so the
        // calling agent can reason about retrying. Threshold is the same
        // 1h freshness window documented at the constant's definition
        // in `src/index.ts` — duplicated here as a literal because
        // ipc.ts has no shared-config home for it and lifting the
        // constant would force a circular import via index.ts.
        const PENDING_RUN_AT_FRESHNESS_MS = 60 * 60 * 1000;
        let activeLocks: string[];
        try {
          activeLocks = getActivePendingRunAtNames(PENDING_RUN_AT_FRESHNESS_MS);
        } catch (lockErr) {
          if (!(lockErr instanceof Error)) throw lockErr;
          // Don't fail the update on a lock-check error — log and
          // proceed. Failing closed would block legitimate updates
          // forever if the DB or schema regressed.
          logger.warn(
            { sourceGroup, err: lockErr.message },
            'tessl_update: pending_run_at check failed — proceeding without deferral',
          );
          activeLocks = [];
        }
        if (activeLocks.length > 0) {
          logger.info(
            { sourceGroup, activeLocks },
            'tessl_update deferred — scheduled task(s) mid-flight with fresh pending_run_at lock (#496)',
          );
          fs.writeFileSync(
            tesslResultPath,
            JSON.stringify({
              stdout: `Deferred: scheduled task(s) currently mid-flight with fresh pending_run_at lock (${activeLocks.join(', ')}). Retry after the task(s) finish — running tessl_update now would force-close the maintenance container and orphan the lock (#496).`,
              deferred: true,
              activeLocks,
            }),
          );
          break;
        }

        logger.info({ sourceGroup }, 'Running tessl_update');

        execFile(
          'bash',
          [
            '-c',
            'cd /app/tessl-workspace && tessl update --yes --dangerously-ignore-security --agent claude-code 2>&1',
          ],
          { timeout: 150_000, maxBuffer: 2 * 1024 * 1024 },
          (error, stdout) => {
            if (error) {
              logger.error(
                {
                  sourceGroup,
                  error: error.message,
                  output: stdout.slice(-500),
                },
                'tessl_update failed',
              );
              fs.writeFileSync(
                tesslResultPath,
                JSON.stringify({
                  error: error.message,
                  stdout: stdout.slice(-2000),
                }),
              );
              return;
            }
            const output = stdout.trim();
            // `tessl update` prints "Updated ..." when a tile actually
            // moved forward. No-op runs don't, and clearing sessions on a
            // no-op would nuke conversation state for nothing — hence
            // the string check instead of an unconditional clear.
            if (/\bUpdated\b/.test(output)) {
              const cleared = deleteAllSessions();
              // Close every currently-running container so it respawns on
              // the next inbound message with the freshly-installed tile
              // content. `deleteAllSessions()` only resets the SDK
              // session-id mapping — without this companion call, a
              // container that's mid-idle-loop right now keeps its old
              // skills/.tessl/ snapshot until its 30-min idle timeout
              // fires (issue #64). The two operations are paired: clear
              // the on-disk session state AND signal the live containers
              // so neither lingers on stale state.
              //
              // Guarded because `closeAllActiveContainers()` rethrows
              // unexpected (non-fs) errors by contract. Without the
              // guard, a programming bug would skip the result-file
              // write below and leave the IPC requester hanging without
              // a structured payload. On failure we still write a
              // partial-success JSON so the caller can distinguish
              // "containers signaled" from "sessions cleared but
              // signaling failed."
              let closed = 0;
              let closeErr: Error | null = null;
              try {
                closed = deps.closeAllActiveContainers();
              } catch (e) {
                // See src/index.ts periodic update for the rationale —
                // outer-boundary guard around an async-callback
                // boundary. Narrow to Error so non-Error throws (a
                // programming bug throwing a non-Error value)
                // propagate per error-handling.md.
                if (!(e instanceof Error)) throw e;
                closeErr = e;
                logger.error(
                  { err: e, sourceGroup, sessionsCleared: cleared },
                  'closeAllActiveContainers threw an unexpected error during tessl_update — sessions still cleared, but live containers will not respawn until idle timeout',
                );
              }
              logger.info(
                {
                  sourceGroup,
                  sessionsCleared: cleared,
                  containersClosed: closed,
                  closeError: closeErr ? String(closeErr) : undefined,
                },
                'tessl_update found new tiles — sessions cleared and running containers signaled to restart',
              );
              fs.writeFileSync(
                tesslResultPath,
                JSON.stringify(
                  closeErr
                    ? {
                        stdout: `${output}\n\nSessions cleared: ${cleared}\nContainers signaled to restart: ${closed}`,
                        warning: `closeAllActiveContainers failed: ${String(closeErr)} — running containers will pick up new tiles on idle timeout (~30 min) instead of immediately`,
                      }
                    : {
                        stdout: `${output}\n\nSessions cleared: ${cleared}\nContainers signaled to restart: ${closed}`,
                      },
                ),
              );
            } else {
              logger.info(
                { sourceGroup },
                'tessl_update completed — no new tiles',
              );
              fs.writeFileSync(
                tesslResultPath,
                JSON.stringify({ stdout: output || '(no output)' }),
              );
            }
          },
        );
      }
      break;

    case 'list_installed_tiles': {
      // Admin tile only. Returns the names of every tile currently
      // installed in the local Tessl registry — the same set
      // `set_additional_tiles` validates against. The agent uses this
      // to surface "what overlay tiles can I set on this chat?" to
      // the operator before proposing a config change. Read-only;
      // never mutates the registry.
      //
      // The result includes BOTH overlay tiles and the trust-tier
      // baseline (`nanoclaw-core`, `nanoclaw-trusted`,
      // `nanoclaw-untrusted`, `nanoclaw-admin`). The agent presenting
      // this list is expected to call out the baseline names as
      // reserved — `selectTiles` dedupes them against the baseline so
      // configuring one as an overlay is a no-op, not an error, but
      // it's still a misuse of the surface. We don't filter here
      // because (a) the baseline set is policy that lives in
      // `selectTiles`, not registry truth, and (b) future trust-tier
      // changes shouldn't silently alter what this tool returns.
      const tilesResultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized list_installed_tiles attempt',
        );
        fs.writeFileSync(
          tilesResultPath,
          JSON.stringify({
            error: 'list_installed_tiles is admin-tile only',
          }),
        );
        break;
      }
      const tiles = getInstalledTiles();
      if (tiles === null) {
        // Registry directory doesn't exist (cold start, never ran
        // `tessl install`). Distinct from "registry exists but empty"
        // so the operator knows whether to run `tessl_update` first.
        logger.warn(
          { sourceGroup },
          'list_installed_tiles: registry directory absent — run tessl_update',
        );
        fs.writeFileSync(
          tilesResultPath,
          JSON.stringify({
            stdout: JSON.stringify({
              tiles: [],
              registryAbsent: true,
            }),
          }),
        );
        break;
      }
      logger.info(
        { sourceGroup, count: tiles.length },
        'list_installed_tiles served via IPC',
      );
      fs.writeFileSync(
        tilesResultPath,
        JSON.stringify({
          stdout: JSON.stringify({
            tiles,
            registryAbsent: false,
          }),
        }),
      );
      break;
    }

    case 'push_staged_to_branch':
      if (
        data.requestId &&
        data.tileName &&
        data.branch &&
        data.commitMessage
      ) {
        const pushResultPath = scriptResultPath(sourceGroup, data);
        if (!isMain) {
          logger.warn(
            { sourceGroup },
            'Unauthorized push_staged_to_branch attempt',
          );
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({
              error: 'Only the main group can push to tile branches.',
            }),
          );
          break;
        }

        if (!KNOWN_TILE_NAMES.has(data.tileName)) {
          logger.warn(
            { sourceGroup, tileName: data.tileName },
            'push_staged_to_branch rejected: tileName not in allowlist',
          );
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({
              error: `Invalid tileName "${data.tileName}". Allowed: ${[...KNOWN_TILE_NAMES].join(', ')}.`,
            }),
          );
          break;
        }

        const pushScript = path.join(
          process.cwd(),
          'scripts',
          'push-staged-to-branch.sh',
        );

        if (!fs.existsSync(pushScript)) {
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({ error: 'push-staged-to-branch.sh not found' }),
          );
          break;
        }

        const stagingDir = path.join(
          GROUPS_DIR,
          sourceGroup,
          'staging',
          data.tileName,
        );

        // Same .env reader pattern as promote_staging — the orchestrator
        // holds the tile-repo credentials, containers never see them.
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
          {
            sourceGroup,
            tileName: data.tileName,
            branch: data.branch,
            skillName: data.skillName || 'all',
          },
          'Running push_staged_to_branch',
        );

        execFile(
          'bash',
          [
            pushScript,
            stagingDir,
            data.tileName,
            data.branch,
            data.commitMessage,
            data.skillName || 'all',
          ],
          {
            // 5 min is enough for a clone-branch + copy + commit +
            // push. No tessl review loop here — fixups don't re-trigger
            // the local optimize pass.
            timeout: 300_000,
            maxBuffer: 2 * 1024 * 1024,
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
                'push_staged_to_branch failed',
              );
              fs.writeFileSync(
                pushResultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
              return;
            }
            logger.info(
              {
                sourceGroup,
                tileName: data.tileName,
                branch: data.branch,
              },
              'push_staged_to_branch pushed fixup',
            );
            fs.writeFileSync(
              pushResultPath,
              JSON.stringify({ stdout: stdout.trim() }),
            );
          },
        );
      }
      break;

    default:
      logger.warn({ type: data.type }, 'Unknown IPC task type');
  }
}
