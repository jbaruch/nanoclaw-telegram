import path from 'path';

import {
  ASSISTANT_NAME,
  ASSISTANT_OWNER_TG_USER_ID,
  DATA_DIR,
  DEFAULT_TRIGGER,
  ENABLE_THRESHOLD_NUKE,
  getTriggerPattern,
  MAX_MESSAGES_PER_PROMPT,
  MODEL_CONTEXT_WINDOW,
  POLL_INTERVAL,
  SESSION_TOKEN_CAP,
  SESSION_TURN_CAP,
  TIMEZONE,
} from './config.js';
import { writeCheckpoint } from './checkpoint.js';
import { computeThresholds } from './threshold.js';
import { emitSessionTokens } from './usage-telemetry.js';
import './channels/index.js';
import {
  ContainerAgentError,
  ContainerOutput,
  runContainerAgent,
  writeGroupsSnapshot,
  writeTasksSnapshot,
} from './container-runner.js';
import {
  getAllChats,
  getLastFromMeMessage,
  getMessagesSince,
  getNewMessages,
  shouldStoreBotMessage,
  storeMessage,
} from './db-messages.js';
import {
  consumeSessionReset,
  markSessionForReset,
  recordSessionTurn,
} from './db-session-length-cap.js';
import { deleteSessionName, setSession } from './db-sessions.js';
import { getAllTasks } from './db-tasks.js';
import {
  buildHandoffPrefix,
  buildResetNotification,
  resolveLiveSessionCaps,
  shouldMarkForReset,
} from './session-length-cap.js';
import { DEFAULT_SESSION_NAME } from './group-queue.js';
import { BEST_EFFORT_FS_CODES, isFsErrorWithCode } from './fs-errors.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { findChannel, formatMessages } from './router.js';
import {
  buildAgentContextFromDb,
  formatAgentContextTag,
} from './agent-context.js';
import {
  pruneSessionArtifacts,
  resolveSessionArtifactRetentionConfig,
} from './session-artifact-retention.js';
import {
  extractSessionCommand,
  handleSessionCommand,
  isSessionCommandAllowed,
} from './session-commands.js';
import { Channel, NewMessage, RegisteredGroup } from './types.js';
import { logger } from './logger.js';
import {
  resolveGatesForGroup,
  evaluateGateChain,
} from './gates/orchestrator.js';
import { isAddressedToUs } from './message-classify.js';
import {
  CIRCUIT_BREAKER_COOLDOWN_MINUTES,
  checkCircuitBreaker,
  recordFailure,
  recordSuccess,
} from './circuit-breaker.js';
import {
  getOrRecoverCursor,
  lastAgentTimestamp,
  lastTimestamp,
  registeredGroups,
  saveState,
  sessions,
  setLastTimestamp,
} from './orchestrator-state.js';
import {
  getActiveIdleTimer,
  installIdleTimerControl as installIdleTimerControlImpl,
  releaseIdleTimerControl,
} from './idle-timer.js';
import type { IdleTimerControl } from './idle-timer.js';
import {
  claimReplyAnchor,
  decideAgentOutputAction,
} from './agent-output-action.js';
import { wipeSessionJsonl } from './session-wipe.js';
import {
  channels,
  nukeTimestamps,
  pendingReplyTo,
  queue,
} from './orchestrator-runtime.js';

// Message pipeline extracted from src/index.ts (#749): the inbound
// poll -> gate -> spawn loop. Shared state comes from
// ./orchestrator-state.js and ./orchestrator-runtime.js.

let messageLoopRunning = false;

// Idle-close timer state lives in `./idle-timer.ts` so the install /
// release / lookup lifecycle can be unit-tested with fake timers
// without booting the full message loop (#506). The wrapper closes
// over `queue` so the pure module stays free of host-singleton imports.
function installIdleTimerControl(
  chatJid: string,
  group: RegisteredGroup,
): IdleTimerControl {
  return installIdleTimerControlImpl(chatJid, group, () =>
    queue.closeStdin(chatJid),
  );
}

/**
 * Typing indicators are cosmetic: a channel transport rejection must
 * never abort message processing or skip the post-run cleanup
 * (`releaseIdleTimerControl`) that follows the call sites (#826).
 * Failures log a warning — visible, never silent — and the pipeline
 * moves on. Exported for the unit tests in
 * `src/message-pipeline.test.ts`.
 */
export function setTypingBestEffort(
  channel: Channel,
  chatJid: string,
  typing: boolean,
): Promise<void> {
  // Promise-style .catch, matching the piped-message setTyping site in
  // startMessageLoop; the .then wrapper folds a synchronous throw from
  // a channel implementation into the same rejection path.
  return Promise.resolve()
    .then(() => channel.setTyping?.(chatJid, typing))
    .catch((err) => {
      logger.warn({ chatJid, typing, err }, 'Failed to set typing indicator');
    });
}

/**
 * Predicate: should the orchestrator clear the stored sessionId after
 * the agent-runner reported `output.error` and we have an active
 * `sessionId`?
 *
 * Trues on either of:
 * 1. Error strings the SDK / agent-runner produces when the JSONL
 *    transcript is missing or unloadable: `no conversation found`,
 *    `ENOENT.../<uuid>.jsonl`, `session ... not found`. These are
 *    the historical signals — kept verbatim to preserve the
 *    pre-existing recovery for crash-mid-write / disk-full cases.
 * 2. The token `error_during_execution` anywhere in the message —
 *    this is the SDK's result-message subtype that the agent-runner
 *    formats as `error_during_execution: <summary>` per #149's
 *    error-result recovery path. The previous regex missed this
 *    entirely, which is why #144's nuke-resurrected sessionId
 *    wedged chats: every spawn reproduced the same SDK error and
 *    nothing ever cleared the bad row from the DB.
 *
 * Exported for the unit test in `src/index.stale-session.test.ts`.
 *
 * @internal — call sites in this module are the only production
 *   consumers; the export is solely for test isolation.
 */
const STALE_SESSION_RE =
  /no conversation found|ENOENT.*\.jsonl|session.*not found|error_during_execution/i;
export function isStaleSessionError(errorMsg: string | undefined): boolean {
  if (!errorMsg) return false;
  return STALE_SESSION_RE.test(errorMsg);
}

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
      containerConfig: registeredGroups[c.jid]?.containerConfig,
      requiresTrigger: registeredGroups[c.jid]?.requiresTrigger,
    }));
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

  // Circuit breaker: skip groups that have failed too many times in a row
  const breakerStatus = checkCircuitBreaker(group.folder);
  if (breakerStatus === 'skip') {
    logger.warn({ group: group.name }, 'Circuit breaker active — skipping');
    return true;
  }
  if (breakerStatus === 'resumed') {
    logger.info(
      { group: group.name },
      'Circuit breaker cooldown expired — resuming',
    );
  }

  const missedMessages = getMessagesSince(
    chatJid,
    getOrRecoverCursor(chatJid),
    ASSISTANT_NAME,
    MAX_MESSAGES_PER_PROMPT,
  );

  if (missedMessages.length === 0) return true;

  // --- Session command interception (before trigger check) ---
  const cmdResult = await handleSessionCommand({
    missedMessages,
    isMainGroup,
    groupName: group.name,
    // null group.trigger → undefined → global @<assistant> default.
    // See deriveTriggerString in db.ts: null surfaces only for configs
    // with no keyword/mention entries; we want session-command parsing
    // to keep working in that case.
    triggerPattern: getTriggerPattern(group.trigger ?? undefined),
    timezone: TIMEZONE,
    deps: {
      sendMessage: async (text) => {
        await channel.sendMessage(chatJid, text);
      },
      setTyping: (typing) =>
        channel.setTyping?.(chatJid, typing) ?? Promise.resolve(),
      runAgent: (prompt, onOutput) =>
        // Session commands (`/compact`, etc.) are explicit
        // user-invoked operations — always addressed to us.
        runAgent(group, prompt, chatJid, onOutput, undefined, true),
      closeStdin: () => queue.closeStdin(chatJid),
      advanceCursor: (ts) => {
        lastAgentTimestamp[chatJid] = ts;
        saveState();
      },
      formatMessages,
      canSenderInteract: (msg) => {
        // Coerce null group.trigger → undefined so getTriggerPattern
        // falls back to @<assistant>. Same rationale as the
        // session-command call above.
        const hasTrigger = getTriggerPattern(group.trigger ?? undefined).test(
          msg.content.trim(),
        );
        const reqTrigger = !isMainGroup && group.requiresTrigger !== false;
        // Per #145: trigger-gate is now pattern-match-only. The
        // pre-removal sender-allowlist clause was unmanaged (host-only
        // JSON config, no MCP/skill UI, no audit), redundant with the
        // trigger pattern itself, and shipped with no chat overrides
        // — every consulting call read the default `allow: '*'`.
        return isMainGroup || !reqTrigger || hasTrigger;
      },
    },
  });
  if (cmdResult.handled) return cmdResult.success;
  // --- End session command interception ---

  // Host-side Stage 1 gates (#80). Run the configured gate chain over
  // every message in the batch; skip the spawn entirely if no message
  // clears the chain. Per #145, no pre-filter on sender — the gates
  // (specifically the trigger gate) decide pattern-match-only.
  //
  // `allowedMessageId` tracks which message produced the gate verdict —
  // structural support for downstream features (#108 reply context).
  // Post-gate 👀 emit (#104) is satisfied structurally by canonical's
  // #289 design: the agent-runner's react-first hook only fires when
  // a container is alive, which only happens after gates returned allow.
  const gateNames = resolveGatesForGroup(group);
  let allowedMessageId: string | undefined;
  if (gateNames.length > 0) {
    const gateResult = await evaluateGateChain(
      group,
      chatJid,
      missedMessages,
      gateNames,
    );
    if (!gateResult.allowed) {
      logger.info(
        { group: group.name, chatJid, gates: gateNames },
        'gate chain skipped spawn',
      );
      return true;
    }
    allowedMessageId = gateResult.allowedMessageId;
  } else {
    // Empty chain (typically main groups) — every message trivially
    // clears gating; latest message id is the implicit verdict target.
    allowedMessageId = missedMessages[missedMessages.length - 1]?.id;
  }

  // #576 — build the enriched `<context>` tag for this invocation
  // (local_datetime, weekday, location, timezone_source). Falls back
  // to container_default when tz_state isn't seeded or no segments /
  // location are available. The same resolved tz drives per-message
  // `time=` display so the header and message timestamps agree.
  const agentCtx = buildAgentContextFromDb({
    ownerSenderId: ASSISTANT_OWNER_TG_USER_ID,
    containerTimezone: TIMEZONE,
  });
  const prompt = formatMessages(missedMessages, {
    timezone: agentCtx.timezone,
    contextTag: formatAgentContextTag(agentCtx),
  });

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

  // Track idle timeout for closing stdin when the active user-facing
  // container is idle. Installing a new control clears any stale timer
  // from a previous container generation for this chat (#506).
  const idleTimerControl = installIdleTimerControl(chatJid, group);

  await setTypingBestEffort(channel, chatJid, true);
  let hadError = false;
  let outputSentToUser = false;

  // Progressive streaming disabled — causes message override bugs when
  // multiple messages are piped to the same container.

  // Track which message triggered the response — first reply quotes it.
  // Turn-start assignment is unconditional: this turn's trigger owns
  // the anchor. Mid-turn pipes may only claim it AFTER the first reply
  // consumes it (`claimReplyAnchor`, #722) — an unconditional overwrite
  // there made long turns quote the latest piped message instead of
  // the one they were answering.
  pendingReplyTo[chatJid] = missedMessages[missedMessages.length - 1]?.id;
  logger.info(
    {
      replyToMessageId: pendingReplyTo[chatJid],
      messageIds: missedMessages.map((m) => m.id),
      group: group.name,
    },
    'Reply-to tracking',
  );

  const output = await runAgent(
    group,
    prompt,
    chatJid,
    async (result) => {
      // Streaming output callback — called for each agent result.
      // Decision logic lives in `decideAgentOutputAction` so the four
      // branches (send / mark-displayed / reset-only / noop) stay
      // unit-testable without the channel + DB + queue machinery
      // around this callback. See `src/agent-output-action.ts`.
      const action = decideAgentOutputAction(result);
      if (action.kind !== 'noop') {
        // Telemetry: raw-length log fires on every non-noop event so
        // we keep visibility into work that produced only `<internal>`
        // content (reset-only) and full chat replies (send /
        // mark-displayed) alike.
        const rawLength =
          action.kind === 'reset-only'
            ? action.rawLength
            : action.textForLog.length;
        logger.info({ group: group.name }, `Agent output: ${rawLength} chars`);
      }
      if (action.kind === 'send') {
        const text = action.textForLog;
        const replyId = pendingReplyTo[chatJid];
        const sendResult = await channel.sendMessage(chatJid, text, replyId);
        // Normalize `string | void` to `string | undefined`; only
        // persist a telegram_message_id when we actually got one.
        const sentMsgId =
          typeof sendResult === 'string' ? sendResult : undefined;
        // Gate `storeMessage` on actual delivery (#428). Pre-#428,
        // we wrote a `bot-*` row even when `sendMessage` returned
        // undefined (transport-layer failure surfaced via the
        // outer Channel catch and #414's narrowed gate). Heartbeat
        // counts a `bot-*` row as evidence that the user message
        // was answered — recording one for a send that never
        // reached Telegram makes the user go silent-and-unanswered
        // with no operator-visible signal. The channel's outer
        // catch already logs `[send] Failed to send Telegram
        // message`, so the operator-visible signal is preserved
        // either way. Reuses the same `shouldStoreBotMessage`
        // predicate as the IPC `send_message` and
        // `send_message_to_chat` paths (`src/ipc.ts:686, 2254`)
        // so all four bot-row write sites apply the same gate.
        if (shouldStoreBotMessage(chatJid, sentMsgId)) {
          storeMessage({
            id: `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
            chat_jid: chatJid,
            sender: ASSISTANT_NAME,
            sender_name: ASSISTANT_NAME,
            content: text,
            timestamp: new Date().toISOString(),
            is_from_me: true,
            is_bot_message: true,
            reply_to_message_id: replyId,
            telegram_message_id: sentMsgId,
          });
        } else {
          logger.warn(
            {
              chatJid,
              contentLen: text.length,
            },
            '[send] Skipping bot-message storeMessage — channel returned no message id (delivery failed)',
          );
        }
        // Consume after first reply — prevents replying to the wrong message
        // when user sends follow-ups while background agent is working.
        pendingReplyTo[chatJid] = undefined;
        outputSentToUser = true;
      } else if (action.kind === 'mark-displayed') {
        // #581 — agent already delivered the reply via send_message;
        // result text is preserved for logs but not re-sent.
        logger.info(
          {
            group: group.name,
            chatJid,
            contentLen: action.textForLog.length,
          },
          '[output] chat_displayed set — skipping chat-echo (agent already sent via send_message)',
        );
        pendingReplyTo[chatJid] = undefined;
        outputSentToUser = true;
      }
      if (action.kind !== 'noop') {
        // Re-anchor idle close on any non-null agent result (tool calls,
        // thinking, user-visible output) — not session-update markers
        // where `result.result` is null. The reset fires even when
        // `text` is empty after stripping `<internal>` blocks because
        // the agent IS doing work, just not user-visible work.
        idleTimerControl.reset('agent-output');
      }

      if (result.status === 'success') {
        queue.notifyIdle(chatJid);
      }

      if (result.status === 'error') {
        hadError = true;
      }
    },
    pendingReplyTo[chatJid],
    isAddressedToUs(group, chatJid, missedMessages),
    allowedMessageId,
  );

  await setTypingBestEffort(channel, chatJid, false);
  releaseIdleTimerControl(chatJid, idleTimerControl);

  if (output === 'error' || hadError) {
    // Track consecutive failures for circuit breaker
    const { tripped, failures } = recordFailure(group.folder);
    if (tripped) {
      logger.error(
        { group: group.name, failures },
        `Circuit breaker tripped — pausing group for ${CIRCUIT_BREAKER_COOLDOWN_MINUTES} minutes`,
      );
      // Notify via main group if this isn't the main group
      if (!isMainGroup) {
        const mainJid = Object.keys(registeredGroups).find(
          (jid) => registeredGroups[jid].isMain,
        );
        if (mainJid) {
          const mainChannel = findChannel(channels, mainJid);
          // Fire-and-forget by design; without a .catch a transport
          // rejection here becomes an unhandled rejection in an
          // already-degraded state (#826).
          mainChannel
            ?.sendMessage(
              mainJid,
              `Circuit breaker tripped for "${group.name}" — ${failures} consecutive failures. Paused for ${CIRCUIT_BREAKER_COOLDOWN_MINUTES} minutes. Check logs.`,
            )
            .catch((err) =>
              logger.warn(
                { group: group.name, mainJid, err },
                'Failed to send circuit-breaker notification',
              ),
            );
        }
      }
    }

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

  // Reset failure counter on success
  recordSuccess(group.folder);
  return true;
}

export async function runAgent(
  group: RegisteredGroup,
  prompt: string,
  chatJid: string,
  onOutput?: (output: ContainerOutput) => Promise<void>,
  replyToMessageId?: string,
  addressedToUs?: boolean,
  triggerMessageId?: string,
): Promise<'success' | 'error'> {
  const isMain = group.isMain === true;
  // User-facing path always uses the `default` slot's session chain.
  let sessionId: string | undefined =
    sessions[group.folder]?.[DEFAULT_SESSION_NAME];

  // Session-length cap reset path (#413). If the prior turn marked
  // this slot for reset, consume the marker now — BEFORE the
  // container spawn — so we drop the stale `sessionId`, prepend a
  // brief context-handoff to the prompt, and queue an in-band user
  // notification. The reset itself is "fire-and-forget for this
  // turn"; the next assistant turn under a fresh `sessionId` will
  // re-INSERT the cap-state row via `recordSessionTurn`.
  //
  // Why consume here (not at threshold-cross): we never want to yank
  // a turn mid-flight. The threshold check that ran AFTER the prior
  // turn's `usage` payload set `marked_for_reset = 1` on the row;
  // the very next inbound (this call) is the safe boundary.
  let resetNotification: string | null = null;
  let handoffPrefix = '';
  const pendingReset = consumeSessionReset(group.folder, DEFAULT_SESSION_NAME);
  if (pendingReset) {
    // Build the handoff prefix from the most recent assistant turn —
    // continuity, not completeness, per the issue body. Using the
    // last `is_from_me=1` message means we get verbatim text the
    // agent already produced rather than synthesizing a summary
    // (the synthesis would itself be a model call we want to avoid
    // on the cap-trigger path). Last user instruction is omitted
    // here because the current `prompt` already contains the
    // user's just-arrived batch — duplicating it would burn budget.
    const lastAssistant = getLastFromMeMessage(chatJid);
    const built = buildHandoffPrefix({
      lastAssistantText: lastAssistant?.content,
      assistantName: ASSISTANT_NAME,
    });
    if (built) handoffPrefix = built;

    // Drop the stale sessionId from BOTH the in-memory cache and the
    // DB row so the container spawns a fresh SDK session. The
    // assistant's first turn under the new session will write back
    // its own `newSessionId` via the existing setSession path.
    if (sessions[group.folder]) {
      delete sessions[group.folder][DEFAULT_SESSION_NAME];
    }
    deleteSessionName(group.folder, DEFAULT_SESSION_NAME);
    sessionId = undefined;

    // Build the user-facing notification from the reason + cap that
    // were persisted at mark-time (`markSessionForReset`). The
    // value reflects the operator-configured cap at the moment the
    // threshold actually tripped — env changes between mark and
    // consume don't lie to the user.
    resetNotification = buildResetNotification(
      pendingReset.reason,
      pendingReset.cap,
    );
    logger.warn(
      {
        group: group.name,
        prevSessionId: pendingReset.sessionId,
        handoffPrefixLen: handoffPrefix.length,
        reason: pendingReset.reason,
        cap: pendingReset.cap,
      },
      'session_length_cap_reset_consumed',
    );
  }

  // Apply the handoff prefix to the prompt the container will see.
  // Keep the original `prompt` parameter immutable — the caller's
  // logging / error paths reference it by length, and rebinding the
  // prefixed value into a new local lets a future reviewer trace the
  // mutation in one diff.
  const promptForContainer = handoffPrefix
    ? `${handoffPrefix}${prompt}`
    : prompt;

  // Capture the spawn-start wall clock BEFORE any container work
  // begins. The two `setSession` writes below (the streaming
  // wrappedOnOutput and the post-completion handler) compare this
  // against `nukeTimestamps[group.folder]` to detect a nuke that
  // landed mid-spawn — the dying container's last SDK result still
  // carries the now-defunct `newSessionId`, and writing it back to
  // the DB would resurrect a row whose JSONL was just wiped. See
  // #144 bug 1 for the failure mode (every subsequent spawn reads
  // the resurrected row, fails to load the missing transcript, and
  // wedges the chat permanently).
  const spawnStart = Date.now();
  const wasNukedDuringSpawn = (): boolean =>
    (nukeTimestamps[group.folder] ?? 0) >= spawnStart;

  // Pre-resume artifact pruning (#538). Runs after spawnStart capture
  // so nuke detection covers the retention sweep too — the sweep
  // does local file I/O that's normally <100ms but a nuke landing
  // in that window must still be detected by the post-completion
  // handler.
  if (sessionId) {
    const retention = pruneSessionArtifacts({
      dataDir: DATA_DIR,
      groupFolder: group.folder,
      sessionName: DEFAULT_SESSION_NAME,
      sessionId,
      config: resolveSessionArtifactRetentionConfig(),
    });
    if (
      retention.imageBlocksReplaced > 0 ||
      retention.toolResultRefsReplaced > 0 ||
      retention.toolResultFilesDeleted > 0
    ) {
      logger.info(
        {
          group: group.name,
          groupFolder: group.folder,
          sessionId,
          transcriptPath: retention.transcriptPath,
          imageBlocksReplaced: retention.imageBlocksReplaced,
          toolResultRefsReplaced: retention.toolResultRefsReplaced,
          toolResultFilesDeleted: retention.toolResultFilesDeleted,
        },
        'session_artifact_retention_pruned',
      );
    }
  }

  // Update tasks snapshot for container to read (filtered by group)
  const isTrusted = !!group.containerConfig?.trusted;
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
    isTrusted,
  );

  // Update available groups snapshot (main group only can see all groups)
  const availableGroups = getAvailableGroups();
  writeGroupsSnapshot(
    group.folder,
    isMain,
    availableGroups,
    new Set(Object.keys(registeredGroups)),
    isTrusted,
  );

  // Kill-auto-compaction state for this runAgent invocation. Captured
  // off the stream callback below so the post-run handler (after
  // runContainerAgent resolves) can decide whether to write a
  // checkpoint and — if `ENABLE_THRESHOLD_NUKE` is on — nuke the
  // default slot. Always-on telemetry vs. flag-gated nuke is the
  // separation the design (issue #104, docs/proposals/
  // kill-auto-compaction.md) calls for: the flag-off period generates
  // calibration data and validates the `## Facts` file format on
  // real transcripts before the destructive flip.
  const thresholds = computeThresholds(MODEL_CONTEXT_WINDOW);
  let lastUsedTokens = 0;
  let thresholdReached: 'warn' | 'nuke' | null = null;

  // Session-length cap (#413) per-invocation latches. The handoff
  // prefix (when consumed above) is written into the cap-state row
  // on the FIRST turn of the new session; the latch is set to null
  // after that write so subsequent turns don't re-stamp it. The
  // mark-for-reset latch is set the moment the threshold check
  // verdict says reset, so multiple assistant messages within one
  // runQuery don't churn the row's reason/cap columns. Both reset
  // automatically on the next runAgent invocation.
  let snapshotHandoffPrefix: string | null = handoffPrefix || null;
  let sessionLengthMarked = false;

  // Wrap onOutput to track session ID from streamed results
  const wrappedOnOutput = onOutput
    ? async (output: ContainerOutput) => {
        if (output.newSessionId) {
          if (wasNukedDuringSpawn()) {
            logger.warn(
              {
                group: group.name,
                staleSessionId: output.newSessionId,
                nukeAt: nukeTimestamps[group.folder],
                spawnStart,
              },
              'Dropping streaming setSession write — nuke fired during spawn (#144)',
            );
          } else {
            if (!sessions[group.folder]) sessions[group.folder] = {};
            sessions[group.folder][DEFAULT_SESSION_NAME] = output.newSessionId;
            setSession(group.folder, DEFAULT_SESSION_NAME, output.newSessionId);
          }
        }

        // Kill-auto-compaction telemetry: log every per-turn usage
        // payload so the operator can curve "context size over time"
        // for any session via host-logs. The classifier raises the
        // log level at warn / nuke so a `grep -E "WARN|ERROR"` on
        // host-logs surfaces the rare events without scrolling
        // through the per-turn noise. Telemetry is always-on
        // (decoupled from ENABLE_THRESHOLD_NUKE) — collecting curves
        // before the flip is exactly the data the flip decision
        // needs. The emit/classify body lives in `usage-telemetry.ts`
        // (#349) so the scheduled-task path emits the same shape; the
        // post-emit `thresholdReached` latch stays here because it
        // drives the inbound-only kill-auto-compaction handshake
        // below — scheduled-task fires don't trigger that path.
        if (output.usage) {
          // Per-turn context size — sum of input_tokens delta plus
          // cached portions. Used for the checkpoint "Tokens used at
          // trigger" diagnostic and `threshold_nuke_fired`/
          // `threshold_nuke_inert` log lines, so a cache-heavy nuke
          // doesn't render "2 / 1,000,000" when real context was 750K
          // (#498).
          lastUsedTokens =
            output.usage.input_tokens +
            (output.usage.cache_read_input_tokens ?? 0) +
            (output.usage.cache_creation_input_tokens ?? 0);
          const state = emitSessionTokens(output.usage, {
            group: group.name,
            session: sessions[group.folder]?.[DEFAULT_SESSION_NAME],
            thresholds,
          });
          // Latch on first nuke crossing this turn — multiple
          // assistant messages in one runQuery can each emit usage
          // above the threshold, and we only want one checkpoint
          // write per turn.
          if (state === 'nuke' && thresholdReached !== 'nuke') {
            thresholdReached = 'nuke';
          } else if (state === 'warn' && thresholdReached === null) {
            thresholdReached = 'warn';
          }

          // Session-length cap (#413): cumulative accounting. Tracks
          // SUM of input_tokens and turn count across the entire
          // session — distinct from the per-turn nuke threshold
          // above. Only proceed once we know the active sessionId
          // (the very first turn of a fresh session emits `usage`
          // alongside `newSessionId`; both fields appear on the same
          // `output`, so reading `sessions[..][DEFAULT_SESSION_NAME]`
          // AFTER the setSession write at the top of this callback
          // is the right ordering).
          //
          // Skip when nuked-during-spawn — writing cap state for a
          // session that was just wiped would resurrect the very
          // accounting issue #144 fixed for the sessions table.
          const activeSessionId =
            sessions[group.folder]?.[DEFAULT_SESSION_NAME];
          if (activeSessionId && !wasNukedDuringSpawn()) {
            const snapshot = recordSessionTurn(
              group.folder,
              DEFAULT_SESSION_NAME,
              activeSessionId,
              output.usage.input_tokens,
              // `last_handoff_summary`: write once per session, on
              // the first turn after a reset. Subsequent turns pass
              // null and the COALESCE in the UPDATE preserves the
              // initial value.
              snapshotHandoffPrefix,
            );
            // Clear the latch after the first write so subsequent
            // turns of THIS session don't re-write the same prefix
            // (paranoia — the COALESCE already handles it, but the
            // latch keeps the diagnostic shape clean).
            if (snapshotHandoffPrefix) snapshotHandoffPrefix = null;

            // Per-group cap override (#561): a quiet group may pin a
            // tighter turn/token cap via containerConfig; absent/invalid
            // values inherit the global SESSION_*_CAP. Resolved from the
            // LIVE registry (by folder), not the spawn-captured `group`,
            // so a `set_session_caps` IPC that lands mid-session takes
            // effect on this container's next turn — not only on its next
            // spawn. Falls back to the captured config if the live entry
            // is gone (group unregistered mid-run).
            const effectiveCaps = resolveLiveSessionCaps(
              registeredGroups,
              group.folder,
              group.containerConfig,
              { tokenCap: SESSION_TOKEN_CAP, turnCap: SESSION_TURN_CAP },
            );
            const verdict = shouldMarkForReset(
              {
                totalInputTokens: snapshot.total_input_tokens,
                turnCount: snapshot.turn_count,
              },
              effectiveCaps,
            );
            if (verdict.reset && !sessionLengthMarked) {
              sessionLengthMarked = true;
              const changed = markSessionForReset(
                group.folder,
                DEFAULT_SESSION_NAME,
                verdict.reason,
                verdict.cap,
              );
              logger.warn(
                {
                  group: group.name,
                  session: activeSessionId,
                  reason: verdict.reason,
                  observed: verdict.observed,
                  cap: verdict.cap,
                  total_input_tokens: snapshot.total_input_tokens,
                  turn_count: snapshot.turn_count,
                  marked: changed === 1,
                },
                'session_length_cap_marked_for_reset',
              );
            }
          }
        }

        await onOutput(output);
      }
    : undefined;

  // Fire the in-band reset notification BEFORE the container spawn so
  // the user sees "session reset" before the agent's first reply
  // arrives. The Channel contract (`Promise<string | void>`) says
  // sendMessage absorbs transport failures internally and returns
  // void rather than throwing — Telegram's implementation logs
  // `[send] Failed to send Telegram message` and returns undefined
  // (see `src/channels/telegram.ts`). So a thrown exception here
  // would indicate a programming bug, not a transport failure;
  // per `rules/error-handling.md` we let it propagate rather than
  // catch it under a bare handler. The reset itself has already
  // happened (DB row deleted, sessionId cleared) — only the
  // user-facing notification is at risk if the channel returns
  // void, and the `[send] ...` error log already surfaces that.
  if (resetNotification) {
    const notifyChannel = findChannel(channels, chatJid);
    if (notifyChannel) {
      await notifyChannel.sendMessage(chatJid, resetNotification);
    } else {
      logger.warn(
        { group: group.name, chatJid },
        'session_length_cap_notification_no_channel',
      );
    }
  }

  try {
    const output = await runContainerAgent(
      group,
      {
        prompt: promptForContainer,
        sessionId,
        groupFolder: group.folder,
        chatJid,
        isMain,
        isTrusted: !!group.containerConfig?.trusted,
        assistantName: ASSISTANT_NAME,
        replyToMessageId,
        // #479 sub-#1: gate-verdict trigger message id for usage-log
        // attribution. Distinct from replyToMessageId (reply target).
        triggerMessageId,
        // User-facing path. Invariant: inbound messages always route to
        // `default`. `src/task-scheduler.ts` is the sole writer of
        // `'maintenance'` — maintenance-AyeAye never reaches this code path.
        sessionName: DEFAULT_SESSION_NAME,
        // Drives the agent-runner's react-first 👀 gate. See
        // `isAddressedToUs` in this file for the resolution rules.
        // `requires_trigger` is intentionally NOT consulted — it
        // governs answering mode, not addressed-ness.
        addressedToUs,
      },
      (proc, containerName) =>
        queue.registerProcess(
          chatJid,
          DEFAULT_SESSION_NAME,
          proc,
          containerName,
          group.folder,
        ),
      wrappedOnOutput,
    );

    if (output.newSessionId) {
      if (wasNukedDuringSpawn()) {
        logger.warn(
          {
            group: group.name,
            staleSessionId: output.newSessionId,
            nukeAt: nukeTimestamps[group.folder],
            spawnStart,
          },
          'Dropping post-completion setSession write — nuke fired during spawn (#144)',
        );
      } else {
        if (!sessions[group.folder]) sessions[group.folder] = {};
        sessions[group.folder][DEFAULT_SESSION_NAME] = output.newSessionId;
        setSession(group.folder, DEFAULT_SESSION_NAME, output.newSessionId);
      }
    }

    if (output.status === 'error') {
      // Detect stale/corrupt session — clear it so the next retry
      // starts fresh. The session .jsonl can go missing after a crash
      // mid-write, manual deletion, disk-full, or — most critically
      // for this codebase — after the #144 race wrote a stale
      // sessionId back to the DB pointing at a JSONL that nuke had
      // already wiped. The existing backoff in group-queue.ts handles
      // the retry; we just need to remove the broken session ID.
      //
      // The regex covers the SDK's two reporting shapes:
      // 1. Thrown / re-formatted into the error string by the SDK or
      //    the agent-runner (e.g. "no conversation found", "ENOENT
      //    /workspace/.claude/projects/.../<uuid>.jsonl", "session
      //    not found").
      // 2. SDK result-message error subtypes that the agent-runner
      //    formats as `<subtype>: <summary>` per #149's recovery path
      //    (e.g. "error_during_execution: ..."). The
      //    `error_during_execution` token is the dominant signal that
      //    a session pointer is broken — the SDK uses it whenever
      //    transcript load fails for ANY reason, and the previous
      //    regex missed it entirely. Per #144 bug 2.
      const isStaleSession = !!sessionId && isStaleSessionError(output.error);

      if (isStaleSession) {
        logger.warn(
          { group: group.name, staleSessionId: sessionId, error: output.error },
          'Stale session detected — clearing for next retry',
        );
        // Only clear the DEFAULT slot — this path runs the user-facing
        // container, so a stale session here is default's problem, not
        // maintenance's. Wiping both would force maintenance to restart
        // its own session chain for no reason.
        if (sessions[group.folder])
          delete sessions[group.folder][DEFAULT_SESSION_NAME];
        deleteSessionName(group.folder, DEFAULT_SESSION_NAME);
      } else if (sessionId && output.error) {
        // Drift surface (#155): we had a sessionId AND an error, but
        // the predicate didn't match. Either the error genuinely isn't
        // a stale-session signal (model rate limit, OAuth, etc.) — fine —
        // or the SDK changed its wording and the regex needs an update.
        // Debug-level so steady-state noise stays low; an operator who
        // sees recovery stop working can flip the log level and the
        // unmatched string surfaces immediately.
        logger.debug(
          { group: group.name, error: output.error },
          'Container error did not match stale-session predicate (no sessionId clear)',
        );
      }

      logger.error(
        { group: group.name, error: output.error },
        'Container agent error',
      );
      return 'error';
    }

    // Kill-auto-compaction threshold-cross handler (issue #104, design
    // at docs/proposals/kill-auto-compaction.md). Always-on checkpoint
    // write; flag-gated nuke. Runs on the success path only — error
    // turns either already cleared the session above or didn't reach
    // any model state worth checkpointing.
    if (thresholdReached === 'nuke' && !wasNukedDuringSpawn()) {
      const sessionForCheckpoint =
        sessions[group.folder]?.[DEFAULT_SESSION_NAME];
      if (sessionForCheckpoint) {
        const groupDir = resolveGroupFolderPath(group.folder);
        // Fast-path slug `-workspace-group` mirrors the convention
        // documented in `wipeSessionJsonl` (`src/session-wipe.ts` — see
        // the comment about the container's project dir layout).
        // Falls through to an empty Facts list if the slug differs;
        // the writer tolerates a missing JSONL via parseSessionTranscript.
        const jsonlPath = path.join(
          DATA_DIR,
          'sessions',
          group.folder,
          DEFAULT_SESSION_NAME,
          '.claude',
          'projects',
          '-workspace-group',
          `${sessionForCheckpoint}.jsonl`,
        );
        try {
          await writeCheckpoint({
            groupDir,
            jsonlPath,
            sessionId: sessionForCheckpoint,
            thresholds,
            usedTokens: lastUsedTokens,
            groupName: group.name,
          });
        } catch (err) {
          if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
          logger.error(
            { group: group.name, err },
            'Threshold-cross checkpoint write failed',
          );
        }

        if (ENABLE_THRESHOLD_NUKE) {
          // Mirror the deps.nukeSession 'default' branch in this
          // file. Kept intentionally inline (rather than refactored
          // into a shared helper) because the IPC nuke path has
          // additional bookkeeping for 'all' / 'maintenance' that
          // doesn't apply here, and a partial extraction would
          // create more drift surface than parallel logic. If a
          // third caller appears, refactor.
          nukeTimestamps[group.folder] = Date.now();
          queue.closeStdin(chatJid, DEFAULT_SESSION_NAME);
          if (sessions[group.folder]) {
            delete sessions[group.folder][DEFAULT_SESSION_NAME];
          }
          deleteSessionName(group.folder, DEFAULT_SESSION_NAME);
          const wiped = wipeSessionJsonl(
            group.folder,
            DEFAULT_SESSION_NAME,
            sessionForCheckpoint,
          );
          logger.error(
            {
              group: group.name,
              session: sessionForCheckpoint,
              used_tokens: lastUsedTokens,
              context_window: thresholds.contextWindow,
              wiped,
            },
            'threshold_nuke_fired',
          );
        } else {
          logger.warn(
            {
              group: group.name,
              session: sessionForCheckpoint,
              used_tokens: lastUsedTokens,
              context_window: thresholds.contextWindow,
            },
            'threshold_nuke_inert (ENABLE_THRESHOLD_NUKE=0)',
          );
        }
      }
    }

    return 'success';
  } catch (err) {
    // runAgent must resolve to a typed 'success' | 'error': the caller
    // (processGroupMessages) branches on 'error' for cursor-rollback (#428),
    // so an OPERATIONAL agent failure is reported as 'error' rather than
    // thrown. #784: runContainerAgent now raises a typed ContainerAgentError
    // for every infrastructure failure (container spawn, docker, fs, OneCLI
    // fail-closed), so this boundary narrows to that closed set instead of the
    // defect-blacklist it used to carry. Anything else — a programmer defect
    // (TypeError, …) or a plain-Error defect in runAgent's own post-processing
    // (checkpoint/threshold handling above, which the old blacklist masked) —
    // propagates to the queue's per-group boundary (group-queue.ts) so the bug
    // surfaces instead of being retried as a routine agent failure.
    if (!(err instanceof ContainerAgentError)) {
      throw err;
    }
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
        setLastTimestamp(newTimestamp);
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

          // --- Session command interception (message loop) ---
          // Scan ALL messages in the batch for a session command.
          const loopCmdMsg = groupMessages.find(
            (m) =>
              extractSessionCommand(
                m.content,
                // Coerce null → undefined: session commands are
                // matched against the @<assistant> global default
                // when the group has no keyword/mention pattern.
                getTriggerPattern(group.trigger ?? undefined),
              ) !== null,
          );

          if (loopCmdMsg) {
            // Only close active container if the sender is authorized — otherwise an
            // untrusted user could kill in-flight work by sending /compact (DoS).
            // closeStdin no-ops internally when no container is active.
            if (
              isSessionCommandAllowed(
                isMainGroup,
                loopCmdMsg.is_from_me === true,
              )
            ) {
              queue.closeStdin(chatJid);
            }
            // Enqueue so processGroupMessages handles auth + cursor advancement.
            // Don't pipe via IPC — slash commands need a fresh container with
            // string prompt (not MessageStream) for SDK recognition.
            queue.enqueueMessageCheck(chatJid);
            continue;
          }
          // --- End session command interception ---

          // Host-side Stage 1 gates (#80). For non-main groups, only
          // wake the bot on messages that clear the configured gate
          // chain. Non-trigger messages still accumulate in DB and get
          // pulled as context when a trigger eventually arrives. Per
          // #145, no pre-filter on sender — gates decide pattern-only.
          // `allowedMessageId` tracks which message produced the gate
          // verdict (structural support for #108 reply context).
          const gateNames = resolveGatesForGroup(group);
          let allowedMessageId: string | undefined;
          if (gateNames.length > 0) {
            const gateResult = await evaluateGateChain(
              group,
              chatJid,
              groupMessages,
              gateNames,
            );
            if (!gateResult.allowed) {
              logger.info(
                { group: group.name, chatJid, gates: gateNames },
                'gate chain skipped spawn',
              );
              continue;
            }
            allowedMessageId = gateResult.allowedMessageId;
          } else {
            // Empty chain (typically main groups) — every message
            // trivially clears gating; latest message id is the
            // implicit verdict target.
            allowedMessageId = groupMessages[groupMessages.length - 1]?.id;
          }
          void allowedMessageId;

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
          // #576 — enriched context tag, same shape as the main
          // pre-spawn path above.
          const pipeAgentCtx = buildAgentContextFromDb({
            ownerSenderId: ASSISTANT_OWNER_TG_USER_ID,
            containerTimezone: TIMEZONE,
          });
          const formatted = formatMessages(messagesToSend, {
            timezone: pipeAgentCtx.timezone,
            contextTag: formatAgentContextTag(pipeAgentCtx),
          });

          const lastMsgId = messagesToSend[messagesToSend.length - 1]?.id;
          // Per-pipe addressed-ness for the agent-runner's react-first
          // hook. Without this, a piped batch into a container that
          // was originally spawned for non-addressed traffic inherits
          // the stale spawn-time flag — a fresh `@AyeAye` reply
          // landing on an already-running container would otherwise
          // never get a 👀.
          const pipedAddressedToUs = isAddressedToUs(
            group,
            chatJid,
            messagesToSend,
          );
          if (
            queue.sendMessage(chatJid, formatted, lastMsgId, pipedAddressedToUs)
          ) {
            // Re-anchor the active container's idle close on user input,
            // not only on user-visible agent output. Tool-heavy or
            // internal-only turns can otherwise run past a timer that was
            // scheduled by the previous response and get closed mid-turn
            // (#506). Optional-chain is intentional: if the prior
            // `processGroupMessages` cycle has already released its
            // control, a fresh cycle will install one — no need to
            // synthesize a control here.
            getActiveIdleTimer(chatJid)?.reset('user-input');
            // #722: claim the reply anchor only when the in-flight
            // turn has already consumed it — an unconditional overwrite
            // here made the running turn's final response quote this
            // piped batch instead of the message it was answering. When
            // the anchor is free, this batch is the next turn's trigger
            // and the claim lands; when held, the piped messages are
            // still processed but the in-flight turn keeps its quote.
            const anchorClaimed = claimReplyAnchor(
              pendingReplyTo,
              chatJid,
              lastMsgId,
            );
            logger.debug(
              {
                chatJid,
                count: messagesToSend.length,
                replyToMessageId: lastMsgId,
                anchorClaimed,
              },
              'Piped messages to active container',
            );
            lastAgentTimestamp[chatJid] =
              messagesToSend[messagesToSend.length - 1].timestamp;
            saveState();
            // Note: an earlier draft swept stale IPC inputs here on every
            // confirmed cursor advance. That created a race (caught in
            // PR #288 review): the agent-runner only drains IPC between
            // queries, not during one. A long-running query (tool calls,
            // subagent work) means files written during that query sit
            // unread on disk; an age-based sweep at write time can then
            // unlink them before the agent ever sees them, dropping
            // messages while `lastAgentTimestamp` has already advanced.
            // The pre-spawn sweep in `buildVolumeMounts` handles the
            // actual #287 symptom — the cross-lifetime backlog (the
            // 1604-file pile that accumulated across many respawns).
            // Within-lifetime accumulation on an untrusted RO mount is
            // NOT zero — the agent can't unlink consumed files, so the
            // dir grows for as long as the container stays up. For a
            // continuously-busy group that keeps the idle timer reset,
            // that lifetime can be days. If `drainIpcInput`'s
            // `readdirSync(...).filter(...).sort()` cost ever shows up
            // in profiles, the right next step is an explicit ack
            // channel (agent writes consumed filenames to the writable
            // `messages/` mount, host sweeps acked files) — see #287
            // follow-up. The pre-spawn sweep keeps the upper bound
            // tied to "longest container lifetime" rather than
            // "lifetime of the install", which is the change that
            // actually unblocks untrusted containers today.
            // Show typing indicator while the container processes the piped message
            channel
              .setTyping?.(chatJid, true)
              ?.catch((err) =>
                logger.warn({ chatJid, err }, 'Failed to set typing indicator'),
              );
          } else {
            // No active container — enqueue for a new one
            queue.enqueueMessageCheck(chatJid);
          }
        }
      }
      // outer-boundary-process-contract (coding-policy: error-handling): the
      // message poll loop's must-not-die boundary.
      //   - Caller's silent-failure shape: an uncaught throw rejects the
      //     startMessageLoop promise and stops ALL message processing until
      //     the orchestrator restarts.
      //   - What the catch emits: an error log; the loop sleeps and polls again.
      //   - Why propagation breaks the contract: one tick's failure would take
      //     down message processing for every group.
      // eslint-disable-next-line no-catch-all/no-catch-all -- outer-boundary-process-contract
    } catch (err) {
      logger.error({ err }, 'Error in message loop');
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
  }
}
