import fs from 'fs';
import path from 'path';

import { SqliteError } from 'better-sqlite3';

import {
  ASSISTANT_NAME,
  ASSISTANT_OWNER_TG_USER_ID,
  CREDENTIAL_PROXY_PORT,
  DATA_DIR,
  DEFAULT_TRIGGER,
  ENABLE_THRESHOLD_NUKE,
  getTriggerPattern,
  GROUPS_DIR,
  HOST_GID,
  HOST_UID,
  MAX_MESSAGES_PER_PROMPT,
  MODEL_CONTEXT_WINDOW,
  POLL_INTERVAL,
  SESSION_TOKEN_CAP,
  SESSION_TURN_CAP,
  TELEGRAM_BOT_POOL,
  TIMEZONE,
} from './config.js';
import { clearCheckpoints, writeCheckpoint } from './checkpoint.js';
import { writeShutdownCheckpoints } from './shutdown-checkpoints.js';
import { computeThresholds } from './threshold.js';
import { emitSessionTokens } from './usage-telemetry.js';
import { startCredentialProxy } from './credential-proxy.js';
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
  PROXY_BIND_HOST,
} from './container-runtime.js';
import {
  markHandoffActive,
  readAndConsumeHandoffMarker,
  writeHandoffMarker,
} from './handoff.js';
import {
  clearStalePendingRunAt,
  clearTaskSessionIdsForGroup,
  clearSessionLengthStateForGroup,
  consumeSessionReset,
  getActivePendingRunAtNames,
  getAllChats,
  getAllRegisteredGroups,
  getAllSessions,
  deleteAllSessions,
  deleteRegisteredGroup,
  deleteSession,
  deleteSessionName,
  getAllTasks,
  getChatByJid,
  getLastBotMessageTimestamp,
  getLastFromMeMessage,
  getMessageById,
  getMessagesSince,
  getTaskById,
  markSessionForReset,
  recordSessionTurn,
  runTzHeartbeatAdvisory,
  TzAdvisoryResult,
  createTask,
  deleteTask,
  getNewMessages,
  getRouterState,
  initDatabase,
  setRegisteredGroup,
  updateGroupTrusted,
  updateGroupTrigger,
  setRouterState,
  setSession,
  storeChatMetadata,
  storeLocation,
  storeMessage,
} from './db.js';
import {
  buildHandoffPrefix,
  buildResetNotification,
  shouldMarkForReset,
} from './session-length-cap.js';
import {
  DEFAULT_SESSION_NAME,
  GroupQueue,
  MAINTENANCE_SESSION_NAME,
} from './group-queue.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { initBotPool } from './channels/telegram.js';
import { shouldStoreBotMessage, startIpcWatcher } from './ipc.js';
import { findChannel, formatMessages, formatOutbound } from './router.js';
import {
  buildAgentContextFromDb,
  formatAgentContextTag,
} from './agent-context.js';
import { ChannelType } from './text-styles.js';
import {
  restoreRemoteControl,
  startRemoteControl,
  stopRemoteControl,
} from './remote-control.js';
import { pruneOldContainerLogs } from './host-logs.js';
import { startSessionCleanup } from './session-cleanup.js';
import {
  pruneSessionArtifacts,
  resolveSessionArtifactRetentionConfig,
} from './session-artifact-retention.js';
import {
  extractSessionCommand,
  handleSessionCommand,
  isSessionCommandAllowed,
} from './session-commands.js';
import {
  startHubitatListener,
  stopHubitatListener,
} from './hubitat-listener.js';
import { startSchedulerLoop } from './task-scheduler.js';
import { startTriggerLearner } from './gates/trigger-learner-runtime.js';
import { installTelegramOutboundTap } from './telegram-outbound-tap.js';
import { Channel, NewMessage, RegisteredGroup } from './types.js';
import { logger } from './logger.js';
import { initObserver } from './observer.js';
import { runGateChain, GateContext } from './gates/index.js';
import { checkSilentZero } from './usage-log.js';
import {
  getActiveIdleTimer,
  installIdleTimerControl as installIdleTimerControlImpl,
  releaseIdleTimerControl,
} from './idle-timer.js';
import type { IdleTimerControl } from './idle-timer.js';

// Re-export for backwards compatibility during refactor
export { escapeXml, formatMessages } from './router.js';

/** Check if a message is a reply to or quote of a bot message. */
function isReplyToBot(msg: NewMessage): boolean {
  // Check content prefix — resolveReply adds [Replying to SenderName: "..."]
  if (msg.content.startsWith(`[Replying to ${ASSISTANT_NAME}:`)) return true;
  // Check reply_to_message_id in DB — covers cases where prefix format differs
  if (msg.reply_to_message_id) {
    const original = getMessageById(msg.reply_to_message_id, msg.chat_jid);
    if (original?.is_from_me) return true;
  }
  return false;
}

/**
 * Decide whether an inbound batch is "addressed to us" — drives the
 * agent-runner's react-first 👀 gate (#289). Independent of
 * `requires_trigger`, which governs whether the agent ANSWERS
 * deterministically vs. reasons about every inbound. The 👀 ack is
 * about whether the message was directed at us at all.
 *
 * Resolves true iff:
 *  - the chat is the main control group, OR
 *  - the chat is a 1:1 DM (`chats.is_group=0`) — every solo inbound is
 *    implicitly for us, OR
 *  - at least one message in the batch matches the trigger pattern, OR
 *  - at least one message replies to OUR bot (per `isReplyToBot`).
 *
 * Note: `requires_trigger=false` on a multi-bot group (e.g. `Old.wtf`)
 * does NOT short-circuit — that was the original bug.
 */
function isAddressedToUs(
  group: RegisteredGroup,
  chatJid: string,
  messages: NewMessage[],
): boolean {
  if (group.isMain === true) return true;
  const chat = getChatByJid(chatJid);
  if (chat && chat.is_group === 0) return true;
  const triggerPattern = getTriggerPattern(group.trigger ?? undefined);
  return messages.some(
    (m) => triggerPattern.test(m.content.trim()) || isReplyToBot(m),
  );
}

/**
 * Resolve which gates apply to a group for inbound-message gating (#80).
 *
 * Migration path A (locked in spec): groups with no `gates` configured
 * AND `requiresTrigger !== false` get the implicit `['trigger']` chain
 * so they keep their pre-#80 behaviour bit-for-bit. Main groups get an
 * empty chain (= always allow), matching pre-#80 semantics.
 *
 * Even when `requiresTrigger === false`, the deterministic `'trigger'`
 * gate is included whenever the group has trigger patterns configured
 * — running it is free (microseconds, $0) and short-circuits expensive
 * Stage 2 LLM calls when a deterministic match exists. The
 * `requires_trigger=false` semantic ("respond to all messages") is
 * preserved because (a) when no patterns match the trigger gate
 * returns `pass` and the chain falls through to whatever's next
 * (Stage 2 if enabled, fail-open default otherwise), and (b) groups
 * with no patterns at all still get an empty implicit chain — UNLESS
 * Stage 2 is enabled (see Stage 2 paragraph below), in which case the
 * appended `haiku-classifier` is the entire chain. Main groups with
 * Stage 2 enabled also pick up `haiku-classifier`; if you don't want
 * the classifier on a main group, set `stage2Enabled: false`
 * explicitly.
 *
 * Path B (one-shot DB migration to set `containerConfig.gates =
 * ['trigger']`) is a future cleanup — the column stays for now.
 *
 * Stage 2 (#83): when `containerConfig.stage2Enabled === true`
 * AND `requiresTrigger !== true` (#98), `'haiku-classifier'` is
 * APPENDED LAST so deterministic gates short-circuit before any API
 * call. New groups default `stage2Enabled` to `true` via
 * `applyNewGroupContainerConfigDefaults`; existing groups keep
 * whatever was previously persisted, and an explicit `false` always
 * disables.
 */
export function resolveGatesForGroup(group: RegisteredGroup): string[] {
  // `containerConfig` is JSON-parsed but not field-validated at the DB
  // layer (see db.ts), so a hand-edited row could carry
  // `gates: null` or `gates: 'trigger'` (string) and trigger a
  // `gateNames.length` throw at every call site downstream. Validate
  // shape here and treat anything malformed as "no opinion" — falls
  // through to the implicit-chain branch so the row keeps working.
  let chain: string[];
  const explicitGates = group.containerConfig?.gates;
  if (
    Array.isArray(explicitGates) &&
    explicitGates.every((g) => typeof g === 'string')
  ) {
    chain = [...explicitGates];
  } else {
    const isMainGroup = group.isMain === true;
    if (isMainGroup) {
      chain = [];
    } else if (group.requiresTrigger !== false) {
      chain = ['trigger'];
    } else {
      // requiresTrigger=false: still run the free deterministic
      // trigger gate when patterns exist, so a match short-circuits
      // any downstream paid gate. No patterns → empty chain (preserves
      // the "respond to all" semantic for brand-new groups).
      const hasPatterns = (group.triggerPatterns?.patterns?.length ?? 0) > 0;
      chain = hasPatterns ? ['trigger'] : [];
    }
  }
  // Stage 2 only adds value for permissive (non-strict) groups.
  // requiresTrigger=true means "respond only to deterministic matches" —
  // there's no grey zone for Haiku to adjudicate, and with the
  // last-gate-wins combinator (in-flight), putting Haiku after a
  // strict-trigger gate would cause a Stage 1 deny to fall through to
  // Haiku, silently breaking the strict-gating contract.
  //
  // The check is `requiresTrigger !== true` so that explicit-`false` and
  // unset-or-undefined both qualify as permissive. Existing groups with
  // requiresTrigger left at its DB default get Stage 2 if stage2Enabled
  // is true; only groups that have explicitly opted into strict-trigger
  // gating skip Stage 2.
  //
  // Explicit `containerConfig.gates` bypasses this predicate by virtue
  // of resolving the chain through the explicit branch above — an
  // operator who pins `gates: ['trigger', 'haiku-classifier']` knows
  // what they're asking for.
  if (
    group.containerConfig?.stage2Enabled === true &&
    group.requiresTrigger !== true &&
    !chain.includes('haiku-classifier')
  ) {
    chain.push('haiku-classifier');
  }
  return chain;
}

/**
 * Strip the inline `[Replying to <sender>: "<preview>"]\n` quote prefix
 * from a stored message's `content` to recover the user's actual body.
 * Returns the input unchanged when no prefix is detected.
 *
 * The Telegram channel bakes this prefix into `content` for the agent
 * prompt path (router.ts and the agent's /workspace/ipc/input view).
 * The gate-evaluation path needs a clean view so Stage 1's keyword /
 * mention / synthetic-identity matchers don't false-positive on
 * tokens inside the quoted preview (#107).
 *
 * Detection is structural: a leading `[Replying to <sender>: "..."]`
 * followed by a newline. We do NOT rely on matching the exact
 * `reply_to_sender_name` / `reply_to_message_content` because the
 * preview is truncated and may include re-escaped quotes.
 */
function stripReplyQuotePrefix(content: string): string {
  // Anchored at start. `[Replying to <name>: "<preview>"]\n` —
  // greedy-but-line-bounded so a stray `]\n` inside the quoted body
  // cannot prematurely terminate the prefix. Use `[\s\S]` for the
  // preview body so multi-line previews (rare but possible) are
  // captured. Single match, single newline terminator.
  const re = /^\[Replying to [^\n]+?: "[\s\S]*?"\]\n/;
  return content.replace(re, '');
}

/**
 * Build a per-message GateContext.
 *
 * `text` is the user's CLEAN body — the inline `[Replying to ...]`
 * quote prefix that the Telegram channel bakes into `content` is
 * stripped here so Stage 1 matchers (#107) see only the user's actual
 * message. Reply context, when present, is exposed structurally via
 * `replyTo` so Stage 2's Haiku classifier still gets the positive
 * signal it relies on for short reply-messages.
 */
function buildGateContext(
  group: RegisteredGroup,
  groupJid: string,
  msg: NewMessage,
): GateContext {
  const cleanText = stripReplyQuotePrefix(msg.content).trim();

  let replyTo: GateContext['message']['replyTo'];
  if (msg.reply_to_message_id) {
    const original = getMessageById(msg.reply_to_message_id, msg.chat_jid);
    const isAssistant = original?.is_from_me === true;
    const isBot = original?.is_bot_message === true || isAssistant;
    const senderName =
      original?.sender_name ?? msg.reply_to_sender_name ?? 'Unknown';
    const contentPreview =
      msg.reply_to_message_content ??
      (original?.content
        ? original.content.length > 200
          ? original.content.slice(0, 200) + '...'
          : original.content
        : '');
    replyTo = {
      messageId: msg.reply_to_message_id,
      senderName,
      isBot,
      isAssistant,
      contentPreview,
    };
  }

  return {
    groupJid,
    groupFolder: group.folder,
    message: {
      text: cleanText,
      senderJid: msg.sender,
      messageId: msg.id,
      // Legacy field kept for tests that pin the old behaviour. Only
      // populated when the reply target is the assistant — same
      // semantic as before. New code should read `replyTo.isAssistant`.
      replyToMessageId: replyTo?.isAssistant ? replyTo.messageId : undefined,
      replyTo,
      isFromMe: msg.is_from_me === true,
    },
    triggerPatterns: group.triggerPatterns ?? null,
  };
}

/**
 * Spawn-decision: run the gate chain over the candidate messages. The
 * chain is evaluated per-message and the first `allow` flips the
 * group-level decision to "spawn"; `deny` on every message means
 * skip. Per #145 there is no sender-allowlist pre-filter — gates
 * see every message in the batch and the trigger gate's pattern
 * match decides on its own.
 *
 * Returns `{ allowed, allowedMessageId }` — `allowedMessageId` is the
 * id of the message that produced the `allow` verdict (or, when
 * `gateNames` is empty and the chain short-circuits, the last
 * candidate's id). Used by #108's reply-context strip and other
 * downstream consumers that need to know which message produced the
 * verdict; canonical's #289 design keeps the 👀 emit at the
 * agent-runner (not the host).
 */
export async function evaluateGateChain(
  group: RegisteredGroup,
  groupJid: string,
  candidateMessages: NewMessage[],
  gateNames: string[],
): Promise<{ allowed: boolean; allowedMessageId?: string }> {
  if (gateNames.length === 0) {
    const last = candidateMessages[candidateMessages.length - 1];
    return { allowed: true, allowedMessageId: last?.id };
  }
  for (const m of candidateMessages) {
    const ctx = buildGateContext(group, groupJid, m);
    const result = await runGateChain(gateNames, ctx);
    // #443 — emit a single INFO line per per-message gate evaluation so
    // `inspect_gate_decisions` can answer "why didn't bot respond to
    // message X" without grepping the debug-tier per-gate trace
    // `runGateChain` already produces. The host log file is the
    // non-purgeable substrate (a SQLite table would re-introduce
    // retention concerns this design explicitly rejected); the
    // logger's `formatData` JSON-stringifies every value so `chain`
    // round-trips as a parseable array per the format pinned in
    // `src/host-log-parser.ts`.
    //
    // Log-volume tradeoff: one INFO line per inbound message (Stage 1
    // always runs; the line fires regardless of which gate decides).
    // The existing per-gate trace at `src/gates/index.ts` stays
    // debug-only to keep the hot path clean; this new chain-level
    // summary is the minimum producer surface
    // `inspect_gate_decisions` needs. `orchestrator.log` is rotation-
    // capped via `ORCHESTRATOR_LOG_MAX_BYTES` (10 MB) so the line's
    // contribution is bounded by the rotation, not unbounded growth.
    // If volume becomes a problem on a very high-traffic group a
    // future PR can sample by `finalDecision === 'deny'` only — the
    // diagnostic question is almost always about denies, never about
    // already-allowed traffic — without changing the parser
    // contract. The exact field set + message text below is the
    // contract `findGateDecisions` greps for; a producer-side
    // regression test in `src/inspect-gate-decisions.test.ts` pins
    // it so a silent rename here fails CI, not production triage.
    logger.info(
      {
        chatJid: groupJid,
        messageId: m.id,
        groupFolder: group.folder,
        finalDecision: result.finalDecision,
        reason: result.reason,
        chain: result.chain.map((r) => ({
          gate: r.gateName,
          decision: r.decision,
          reason: r.reason,
        })),
      },
      'gate decision',
    );
    if (result.finalDecision === 'allow') {
      return { allowed: true, allowedMessageId: m.id };
    }
  }
  return { allowed: false };
}

/**
 * Boolean wrapper around {@link evaluateGateChain} for callers that
 * only need the spawn decision.
 */
export async function gateAllowsSpawn(
  group: RegisteredGroup,
  groupJid: string,
  candidateMessages: NewMessage[],
  gateNames: string[],
): Promise<boolean> {
  const { allowed } = await evaluateGateChain(
    group,
    groupJid,
    candidateMessages,
    gateNames,
  );
  return allowed;
}

let lastTimestamp = '';
// Nested by groupFolder → sessionName → sessionId. Tracks the user-facing
// `default` slot's SDK session chain so consecutive inbound messages
// resume the prior turn. `maintenance` entries may still be present here
// (e.g. loaded from persisted session state at startup, or written by a
// pre-#193 build), but scheduled tasks no longer update or resume that
// slot: they always start a fresh SDK turn (#193) to prevent cross-task
// `last_result` bleed, and the scheduler wipes their on-disk session
// artifacts (JSONL transcript + tool-results dir) immediately after
// each run completes.
let sessions: Record<string, Record<string, string>> = {};
let registeredGroups: Record<string, RegisteredGroup> = {};
let lastAgentTimestamp: Record<string, string> = {};
// Per-chat reply-to tracking: updated when follow-up messages are piped,
// consumed by the output callback to quote-reply the latest message.
const pendingReplyTo: Record<string, string | undefined> = {};
let messageLoopRunning = false;

const channels: Channel[] = [];
const queue = new GroupQueue();

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

// Circuit breaker: pause groups that fail repeatedly to avoid burning credits.
const MAX_CONSECUTIVE_FAILURES = 5;
const CIRCUIT_BREAKER_COOLDOWN_MS = 30 * 60 * 1000; // 30 minutes
const consecutiveFailures: Record<string, number> = {};
const circuitBreakerUntil: Record<string, number> = {};

// #496 — `follow_me_tasks.pending_run_at` is treated as "fresh" (and
// therefore worth respecting) for this long after stamp-time. Anything
// older is presumed orphaned by a dead/killed container and gets
// reclaimed via `clearStalePendingRunAt`.
//
// Sized for the largest plausible follow-me task duration with margin:
// the morning-brief skill (the longest in the fleet) caps around
// 10–15 min of agent work; 1h gives 4× margin for an unusually slow
// run while still recovering well before the next-day fire. A shorter
// window (e.g. 10 min) would risk reclaiming an in-flight lock from a
// genuinely long but legitimate run; a longer window delays recovery
// past the daily-fire boundary, defeating the point.
const PENDING_RUN_AT_FRESHNESS_MS = 60 * 60 * 1000; // 1 hour

// Per-folder timestamp of the most recent `nukeSession` call. Used to
// gate the post-spawn `setSession` writes against a race where a nuke
// fires while a container is still being awaited: the dying container
// emits a final SDK result containing the same `newSessionId` it was
// processing, and the completion handler would otherwise resurrect
// that row in the DB right after nuke deleted it. Resurrected row
// points at the JSONL file that nuke just wiped — every subsequent
// spawn reads the resurrected sessionId, the SDK can't load the
// transcript, and the chat wedges permanently. See #144 bug 1.
//
// Compare against the spawn's start timestamp captured BEFORE
// `runContainerAgent` is invoked: if `nukeTimestamps[folder] >=
// spawnStart`, the nuke landed after the spawn began (or
// concurrently), so any session-id write coming back from this
// container is stale and must be dropped.
const nukeTimestamps: Record<string, number> = {};

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

function loadState(): void {
  lastTimestamp = getRouterState('last_timestamp') || '';
  const agentTs = getRouterState('last_agent_timestamp');
  try {
    lastAgentTimestamp = agentTs ? JSON.parse(agentTs) : {};
  } catch {
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

/**
 * Cleanup pass for the retired non-main heartbeat task (#453).
 *
 * The non-main heartbeat existed solely to drive `tessl__check-unanswered`
 * — a per-cycle SQL+LLM scan for unreplied messages. The skill was
 * retired in `jbaruch/nanoclaw-core#38` because the steady-state token
 * cost wasn't justified by the rate of genuinely-dropped messages it
 * caught. With no work for the task to do, this orchestrator no longer
 * creates `heartbeat-<folder>` rows for non-main groups, and any rows
 * left behind by older orchestrator versions are deleted on startup so
 * the scheduler stops firing them.
 *
 * `containerConfig.enableHeartbeat` stays in the schema for backwards
 * compatibility (operators upgrading from older orchestrator versions
 * may have it set), but on non-main groups it is now a no-op. The
 * main-group heartbeat (a separate `heartbeat-<folder>` row created
 * inline in `registerGroup` for `group.isMain`) is unaffected — different
 * code path, different prompt, different lifetime.
 *
 * Future feature work that wants non-main heartbeats lands as its own
 * issue with its own task definition.
 */
function cleanupOrphanNonMainHeartbeats(): void {
  for (const [jid, group] of Object.entries(registeredGroups)) {
    if (group.isMain) continue;
    const heartbeatId = `heartbeat-${group.folder}`;
    if (!getTaskById(heartbeatId)) continue;
    deleteTask(heartbeatId);
    logger.info(
      { jid, folder: group.folder, taskId: heartbeatId },
      'Deleted orphan non-main heartbeat task (retired in nanoclaw-core#38 / #453)',
    );
  }
}

/**
 * Drift detector (#159): log every `registered_groups` row whose JID has
 * no matching `chats` row. The spawner reads `available_groups.json`
 * (which is rebuilt from `getAllChats()`) so a row missing from `chats`
 * is silently ignored at runtime — exactly the dormant-row failure mode
 * #159's one-shot cleanup addressed for `tg:1698969` / `telegram_main`.
 *
 * Read-only by design: future operator-introduced drift gets surfaced
 * for review (operator can resolve it via `unregister_group`) instead
 * of being auto-deleted at startup. Auto-delete would make recovery
 * from a transient `chats` outage (e.g. a partial DB restore that
 * truncated `chats` but kept `registered_groups`) catastrophic — every
 * registered group would vanish on the next restart.
 */
function logRegisteredGroupOrphans(): void {
  const knownJids = new Set(getAllChats().map((c) => c.jid));
  const orphans: Array<{ jid: string; folder: string }> = [];
  for (const [jid, group] of Object.entries(registeredGroups)) {
    if (knownJids.has(jid)) continue;
    orphans.push({ jid, folder: group.folder });
  }
  if (orphans.length > 0) {
    logger.warn(
      { orphans },
      'registered_groups rows have no matching chats row — invisible to the spawner. Run unregister_group to clean up if intended.',
    );
  }
}

/**
 * Delete the on-disk session artifacts (JSONL transcript and per-session
 * tool-results directory) for a given session slot, given the SDK
 * sessionId. Returns the number of filesystem entries actually removed —
 * up to 2 per slug (1 transcript + 1 tool-results dir) summed across
 * every project-slug subdirectory found.
 *
 * Path layout (host side):
 *   ${DATA_DIR}/sessions/<groupFolder>/<sessionName>/.claude/projects/<project-slug>/<sessionId>.jsonl
 *   ${DATA_DIR}/sessions/<groupFolder>/<sessionName>/.claude/projects/<project-slug>/<sessionId>/
 *
 * The project-slug is `-workspace-group` for our containers (see
 * CLAUDE_PROJECT_SLUG in container-runner.ts). We glob the projects/
 * directory rather than hardcoding the slug so a future change to the
 * slug — or any operator who renamed the workspace path — doesn't
 * silently leave stale artifacts behind.
 *
 * Used by `nukeSession` (#100) to actually wipe transcript state, and
 * by the scheduler's per-run finally (#193) to wipe scheduled-task
 * artifacts that aren't tracked in the sessions cache. Without this,
 * the next container spawn re-reads the JSONL and the bad state
 * (poison, stuck plan, corrupt memory) is immediately back, AND
 * orphan tool-results directories accumulate forever under the
 * maintenance slot.
 *
 * **Security**: `sessionId` ultimately originates from container stdout
 * (parsed `newSessionId` from the SDK's stream), which is *untrusted*
 * for untrusted-tier groups. A crafted value containing path separators
 * or `..` segments would otherwise be interpolated into the artifact
 * paths and could escape `projectsDir/<slug>/` to delete arbitrary
 * files or directories anywhere the orchestrator process can write.
 * Defense in depth:
 *   1. Reject anything that isn't a strict UUID-or-token charset.
 *   2. After joining, assert the resolved path stays inside `projectsDir`.
 *   3. The tool-results-dir helper additionally relies on Node's
 *      `fs.rmSync` not following symlinks during recursive removal,
 *      so a malicious container that scattered host-pointing symlinks
 *      inside its own dir cannot redirect the wipe outward.
 */
const SESSION_ID_PATTERN = /^[A-Za-z0-9_-]+$/;

/**
 * Try to unlink `${slugPath}/${sessionId}.jsonl`. Returns 1 if the
 * filesystem entry was unlinked, 0 otherwise.
 *
 * Two paths depending on what `${sessionId}.jsonl` actually is:
 *
 *   - **Regular file**: dereference via `realpath` and verify it
 *     resolves inside `slugPath`'s realpath. This catches the TOCTOU
 *     case where a symlink ancestor of slugPath was swapped between
 *     the outer lstat and here, and would otherwise let an unlink
 *     escape the intended tree.
 *
 *   - **Symlink**: unlink the symlink itself. `fs.unlinkSync` on a
 *     symlink path removes the LINK, not the target — safe regardless
 *     of where the link points (including dangling). This is the
 *     "nuke really nukes" promise: if a compromised container makes
 *     the JSONL a symlink to dodge wipe, the symlink still goes away.
 *     Without this branch, the prior realpath-containment check would
 *     refuse to unlink a symlink-out-of-tree and leave the entry on
 *     disk — defeating the nuke entirely.
 *
 * Companion helper `removeToolResultsDirInSlug` mirrors this for the
 * sibling per-session tool-results directory at `${slugPath}/${sessionId}/`.
 */
function unlinkJsonlInSlug(
  slugPath: string,
  sessionId: string,
  groupFolder: string,
  sessionName: string,
): number {
  const jsonlPath = path.join(slugPath, `${sessionId}.jsonl`);

  // lstat first to learn what the entry actually is, without
  // following any symlink. This is the hinge for the two branches.
  let entryStat: fs.Stats;
  try {
    entryStat = fs.lstatSync(jsonlPath);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0; // no such jsonl — fine
    logger.warn(
      { err, groupFolder, sessionName, jsonlPath },
      'unlinkJsonlInSlug: lstat failed on jsonl — skipping',
    );
    return 0;
  }

  if (entryStat.isSymbolicLink()) {
    // Unlink the symlink itself. fs.unlinkSync removes the link
    // entry; it never deletes the target file the link points at.
    try {
      fs.unlinkSync(jsonlPath);
      logger.info(
        { groupFolder, sessionName, sessionId, jsonlPath },
        'unlinkJsonlInSlug: unlinked symlinked jsonl (target preserved)',
      );
      return 1;
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') return 0;
      logger.warn(
        { err, groupFolder, sessionName, sessionId, jsonlPath },
        'unlinkJsonlInSlug: unlink-of-symlink failed',
      );
      return 0;
    }
  }

  // Regular-file path: realpath containment check before unlink to
  // catch a slugPath ancestor symlink swap between the outer lstat
  // and here. `path.resolve` alone is string-based and wouldn't
  // notice such an escape.
  let realSlug: string;
  let realJsonl: string;
  try {
    realSlug = fs.realpathSync(slugPath);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, slugPath },
      'unlinkJsonlInSlug: realpath failed on slug — skipping',
    );
    return 0;
  }
  try {
    realJsonl = fs.realpathSync(jsonlPath);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, jsonlPath },
      'unlinkJsonlInSlug: realpath failed on jsonl — skipping',
    );
    return 0;
  }
  if (!realJsonl.startsWith(realSlug + path.sep)) {
    logger.warn(
      { groupFolder, sessionName, sessionId, jsonlPath, realSlug, realJsonl },
      'unlinkJsonlInSlug: refusing to unlink — realpath escapes slug directory',
    );
    return 0;
  }
  try {
    fs.unlinkSync(jsonlPath);
    return 1;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, sessionId, jsonlPath },
      'unlinkJsonlInSlug: unlink failed',
    );
    return 0;
  }
}

/**
 * Try to remove `${slugPath}/${sessionId}/` (the per-session tool-results
 * directory the SDK writes alongside `${sessionId}.jsonl`). Returns 1
 * if a filesystem entry was removed, 0 otherwise.
 *
 * Mirrors `unlinkJsonlInSlug` with the same lstat → branch on type →
 * realpath-containment discipline; only the leaf operation differs.
 *
 *   - **Symlink**: unlink the symlink itself. `fs.unlinkSync` removes
 *     the link entry without following it, so a compromised container
 *     can't redirect the wipe to walk into an arbitrary host directory
 *     and `recursive: true` it. Same "nuke really nukes" promise as
 *     the JSONL path.
 *
 *   - **Directory**: realpath the dir and the slug, verify the dir's
 *     real path is inside the slug's real path, then `fs.rmSync` with
 *     `recursive: true`. Node's `rmSync` does NOT traverse symlinks
 *     it encounters inside the tree — they're removed as link entries,
 *     never followed — so a malicious container that drops a symlink
 *     to `/etc` inside its own tool-results dir cannot trick us into
 *     deleting host files. The realpath check guards the parent path
 *     itself against ancestor-symlink swap (TOCTOU between the outer
 *     `wipeSessionJsonl` lstat and this call).
 *
 *   - **Regular file at the dir path**: not something the SDK writes,
 *     but if a compromised container plants one we leave it alone and
 *     log — wiping it would be outside the helper's contract (it's a
 *     directory remover) and could mask whatever produced the file.
 */
function removeToolResultsDirInSlug(
  slugPath: string,
  sessionId: string,
  groupFolder: string,
  sessionName: string,
): number {
  const dirPath = path.join(slugPath, sessionId);

  let entryStat: fs.Stats;
  try {
    entryStat = fs.lstatSync(dirPath);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, dirPath },
      'removeToolResultsDirInSlug: lstat failed — skipping',
    );
    return 0;
  }

  if (entryStat.isSymbolicLink()) {
    try {
      fs.unlinkSync(dirPath);
      logger.info(
        { groupFolder, sessionName, sessionId, dirPath },
        'removeToolResultsDirInSlug: unlinked symlinked tool-results dir (target preserved)',
      );
      return 1;
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') return 0;
      logger.warn(
        { err, groupFolder, sessionName, sessionId, dirPath },
        'removeToolResultsDirInSlug: unlink-of-symlink failed',
      );
      return 0;
    }
  }

  if (!entryStat.isDirectory()) {
    // The SDK only writes directories at this path. A regular file
    // here means something else put it there — leave it alone rather
    // than deleting state we can't account for.
    logger.warn(
      { groupFolder, sessionName, sessionId, dirPath },
      'removeToolResultsDirInSlug: refusing — entry exists but is neither symlink nor directory',
    );
    return 0;
  }

  // Directory path: realpath containment check before rm. Same TOCTOU
  // defense as the JSONL helper — a slugPath ancestor symlink swap
  // between the outer lstat and here would otherwise let `rmSync`
  // recurse into an unintended tree.
  let realSlug: string;
  let realDir: string;
  try {
    realSlug = fs.realpathSync(slugPath);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, slugPath },
      'removeToolResultsDirInSlug: realpath failed on slug — skipping',
    );
    return 0;
  }
  try {
    realDir = fs.realpathSync(dirPath);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, dirPath },
      'removeToolResultsDirInSlug: realpath failed on dir — skipping',
    );
    return 0;
  }
  if (!realDir.startsWith(realSlug + path.sep)) {
    logger.warn(
      { groupFolder, sessionName, sessionId, dirPath, realSlug, realDir },
      'removeToolResultsDirInSlug: refusing to remove — realpath escapes slug directory',
    );
    return 0;
  }
  try {
    // `recursive: true` walks the tree. Node never follows symlinks
    // inside — they're removed as entries — so a compromised container
    // that scattered symlinks to host paths in its own tool-results
    // tree cannot redirect the wipe.
    //
    // No `force: true`: we want ENOENT to surface as an error so the
    // returned count reflects actual removals. Without that distinction,
    // a concurrent cleanup that vanished the path between our lstat
    // and rmSync would still count as `1` here, inflating the caller's
    // "entries removed" total.
    fs.rmSync(dirPath, { recursive: true });
    return 1;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, sessionId, dirPath },
      'removeToolResultsDirInSlug: rmSync failed',
    );
    return 0;
  }
}

/**
 * Wipe the on-disk session artifacts (JSONL transcript and the
 * sibling per-session tool-results directory) for the given sessionId
 * across every project-slug subdirectory under the slot's `projects/`.
 * Returns the total number of filesystem entries removed.
 *
 * The function name retains the historical "Jsonl" suffix from when it
 * only unlinked transcripts; the contract is now a full session-artifact
 * wipe. Both artifact types share one realpath-containment regime, one
 * DoS-cap regime, and one slug-walk traversal — keeping them in a single
 * function avoids walking `projects/` twice for what is conceptually one
 * "wipe everything tied to this sessionId" operation.
 *
 * Production callers:
 *   1. `nukeSession` (#100) — owns the multi-step order-of-operations
 *      wipe (capture sessionIds → kill containers → drop DB rows →
 *      remove session artifacts).
 *   2. `startSchedulerLoop` (#193) — injects this as a dependency so
 *      `runTask`'s post-run finally can wipe the per-run artifacts the
 *      moment a scheduled run completes (its sessionId is never
 *      persisted to the DB, so the time-based `cleanup-sessions.sh`
 *      can't find them later).
 *
 * Tests also import this symbol directly to bypass the full
 * `nukeSession` path.
 *
 * @internal — the orchestrator builds with `tsconfig.stripInternal: true`,
 * so this tag keeps the symbol out of the emitted `.d.ts`. The two
 * production callers above are in-tree and don't need d.ts visibility;
 * tests reach the symbol through the source `.ts` import, not the d.ts.
 */
export function wipeSessionJsonl(
  groupFolder: string,
  sessionName: string,
  sessionId: string,
): number {
  if (!SESSION_ID_PATTERN.test(sessionId)) {
    logger.warn(
      { groupFolder, sessionName, sessionId },
      'wipeSessionJsonl: refusing to wipe — sessionId fails strict-charset check',
    );
    return 0;
  }

  const projectsDir = path.join(
    DATA_DIR,
    'sessions',
    groupFolder,
    sessionName,
    '.claude',
    'projects',
  );

  // Validate `projects/` BEFORE any unlink work — the fast path and
  // the slow walk both depend on it being a real directory inside
  // DATA_DIR, not a symlink swap pointing elsewhere. The per-session
  // `.claude` mount is writable from the container, so a compromised
  // container could replace `.claude/projects` with a symlink. We
  // refuse to traverse a symlinked `projects/` regardless of where
  // it points.
  let projectsLstat: fs.Stats;
  try {
    projectsLstat = fs.lstatSync(projectsDir);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, sessionId, projectsDir },
      'wipeSessionJsonl: lstat failed on projects directory',
    );
    return 0;
  }
  if (projectsLstat.isSymbolicLink()) {
    logger.error(
      { groupFolder, sessionName, sessionId, projectsDir },
      'wipeSessionJsonl: refusing to traverse — projects/ itself is a symlink (possible escape attempt)',
    );
    return 0;
  }
  if (!projectsLstat.isDirectory()) {
    return 0;
  }

  // Try the well-known project slug FIRST as a fast path. Two wins:
  //   1. Defeats flood attacks where a compromised container fills
  //      `projects/` with millions of decoy slug-named subdirs to
  //      push the legitimate one past any walk cap. We hit the
  //      legitimate path directly and, on success, return without
  //      walking the tree at all.
  //   2. ~constant-time in the steady state — no opendir+iterate
  //      overhead per nuke.
  // The slow path (full walk) below is the safety net for cases
  // where the slug naming convention drifts (CLAUDE_PROJECT_SLUG
  // changes, operator-renamed workspace, etc.).
  let deleted = 0;
  const fastPathSlug = path.join(projectsDir, '-workspace-group');
  // Same lstat-then-skip discipline as the slow-path loop below
  // (line ~605): without this, a compromised container could replace
  // `projects/-workspace-group` with a symlink to an arbitrary host
  // directory. `unlinkJsonlInSlug`'s realpath-containment check
  // resolves BOTH the slug and the jsonl through the same symlink,
  // so the containment passes and the unlink lands inside the
  // symlink target.
  let fastPathLstat: fs.Stats | undefined;
  try {
    fastPathLstat = fs.lstatSync(fastPathSlug);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code !== 'ENOENT') {
      logger.warn(
        { err, groupFolder, sessionName, sessionId, fastPathSlug },
        'wipeSessionJsonl: lstat failed on fast-path slug',
      );
    }
  }
  if (fastPathLstat?.isSymbolicLink()) {
    logger.error(
      { groupFolder, sessionName, sessionId, fastPathSlug },
      'wipeSessionJsonl: refusing fast path — slug is a symlink (possible escape attempt)',
    );
  } else if (fastPathLstat?.isDirectory()) {
    deleted += unlinkJsonlInSlug(
      fastPathSlug,
      sessionId,
      groupFolder,
      sessionName,
    );
    deleted += removeToolResultsDirInSlug(
      fastPathSlug,
      sessionId,
      groupFolder,
      sessionName,
    );
  }

  // Walk project-slug subdirectories with `opendirSync` — an
  // iterator-style API that does NOT materialize the full directory
  // listing up front, unlike `readdirSync`.
  //
  // Caps:
  //   - MAX_DIRS_VISITED bounds the slow-path search across many
  //     project slugs. Stray files don't count.
  //   - MAX_TOTAL_ENTRIES bounds total readSync iterations so a
  //     `projects/` filled with millions of stub FILES can't block
  //     the orchestrator event loop synchronously.
  const MAX_DIRS_VISITED = 10000;
  const MAX_TOTAL_ENTRIES = 100000;

  // TOCTOU defense for the parent dir: realpath after opendir.
  // `fs.Dir` doesn't expose its FD, so we can't fstat the open handle
  // — instead we resolve the path through the symlink chain at this
  // moment. If a compromised container swapped `projects/` to a
  // symlink between our lstat above and the opendirSync below, the
  // realpath result will land outside the expected `<DATA_DIR>/...`
  // tree and we abort. Residual race: a container would have to win
  // a sub-millisecond inode swap AND aim it inside DATA_DIR — at
  // which point it has already broken out of its sandbox and the
  // orchestrator has bigger problems. Per-slug realpath checks below
  // catch escape attempts at the leaf level regardless.
  let dir: fs.Dir;
  try {
    dir = fs.opendirSync(projectsDir);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return deleted;
    logger.warn(
      { err, groupFolder, sessionName, sessionId, projectsDir },
      'wipeSessionJsonl: failed to open projects directory',
    );
    return deleted;
  }
  try {
    const realProjects = fs.realpathSync(projectsDir);
    // Also realpath DATA_DIR to handle macOS where /var → /private/var
    // (or similar OS-level symlinks). Without this both sides could
    // dereference to different absolute prefixes and the prefix check
    // would false-positive even on a perfectly legitimate path.
    const realDataDir = fs.realpathSync(DATA_DIR);
    const expectedPrefix = realDataDir + path.sep;
    if (!realProjects.startsWith(expectedPrefix)) {
      logger.error(
        {
          groupFolder,
          sessionName,
          sessionId,
          projectsDir,
          realProjects,
          expectedPrefix,
        },
        'wipeSessionJsonl: projects/ realpath outside DATA_DIR — aborting (TOCTOU?)',
      );
      dir.closeSync();
      return deleted;
    }
  } catch (err) {
    logger.warn(
      { err, groupFolder, sessionName, sessionId, projectsDir },
      'wipeSessionJsonl: realpath on projects/ failed — aborting',
    );
    dir.closeSync();
    return deleted;
  }

  let dirsVisited = 0;
  let totalEntries = 0;
  let bailedOnLimit: 'total-entries' | 'dirs-visited' | null = null;
  try {
    let entry: fs.Dirent | null;
    while ((entry = dir.readSync()) !== null) {
      totalEntries++;
      if (totalEntries > MAX_TOTAL_ENTRIES) {
        bailedOnLimit = 'total-entries';
        break;
      }
      // Skip the slug we already tried in the fast path — would
      // double-count `deleted` if the file was already gone.
      if (entry.name === '-workspace-group') continue;

      const slugPath = path.join(projectsDir, entry.name);
      let linkStat: fs.Stats;
      try {
        linkStat = fs.lstatSync(slugPath);
      } catch (err) {
        const code = (err as NodeJS.ErrnoException).code;
        if (code === 'ENOENT') continue;
        logger.warn(
          { err, groupFolder, sessionName, slugPath },
          'wipeSessionJsonl: lstat failed on slug entry — skipping',
        );
        continue;
      }
      if (linkStat.isSymbolicLink()) {
        logger.warn(
          { groupFolder, sessionName, slugPath },
          'wipeSessionJsonl: refusing to traverse symlink under projects/',
        );
        continue;
      }
      if (!linkStat.isDirectory()) continue;

      dirsVisited++;
      if (dirsVisited > MAX_DIRS_VISITED) {
        bailedOnLimit = 'dirs-visited';
        break;
      }

      deleted += unlinkJsonlInSlug(
        slugPath,
        sessionId,
        groupFolder,
        sessionName,
      );
      deleted += removeToolResultsDirInSlug(
        slugPath,
        sessionId,
        groupFolder,
        sessionName,
      );
    }
  } finally {
    dir.closeSync();
  }

  if (bailedOnLimit === 'total-entries') {
    logger.error(
      {
        groupFolder,
        sessionName,
        sessionId,
        totalEntries,
        limit: MAX_TOTAL_ENTRIES,
        deleted,
      },
      'wipeSessionJsonl: stopped early — total readSync count exceeded MAX_TOTAL_ENTRIES (possible DoS via stub-file flood)',
    );
  } else if (bailedOnLimit === 'dirs-visited') {
    logger.error(
      {
        groupFolder,
        sessionName,
        sessionId,
        dirsVisited,
        limit: MAX_DIRS_VISITED,
        deleted,
      },
      'wipeSessionJsonl: stopped early — directory-traversal count exceeded MAX_DIRS_VISITED (possible DoS via slug-dir flood)',
    );
  }
  return deleted;
}

/**
 * Apply registration-time defaults to a group's containerConfig. New
 * groups get `stage2Enabled: true` unless the caller explicitly pinned
 * a value — caller-pinned (including `false`) wins. Existing groups
 * pass through unchanged so we never auto-flip a stored config.
 *
 * Exported for unit testing; callers should use `registerGroup`.
 */
export function applyNewGroupContainerConfigDefaults(
  group: RegisteredGroup,
  isNew: boolean,
): RegisteredGroup {
  if (!isNew) return group;
  if (group.containerConfig?.stage2Enabled !== undefined) return group;
  return {
    ...group,
    containerConfig: {
      ...(group.containerConfig ?? {}),
      stage2Enabled: true,
    },
  };
}

function registerGroup(jid: string, group: RegisteredGroup): void {
  let groupDir: string;
  try {
    groupDir = resolveGroupFolderPath(group.folder);
  } catch (err) {
    logger.warn(
      { jid, folder: group.folder, err },
      'Rejecting group registration with invalid folder',
    );
    return;
  }

  // Stage 2 default for NEW groups only: opt them into the Haiku
  // classifier unless the caller explicitly pinned `stage2Enabled`.
  // Existing groups keep whatever they already have on disk.
  group = applyNewGroupContainerConfigDefaults(group, !registeredGroups[jid]);

  registeredGroups[jid] = group;
  setRegisteredGroup(jid, group);

  // Create group folder
  fs.mkdirSync(path.join(groupDir, 'logs'), { recursive: true });

  // CLAUDE.md is no longer copied per-group — it's a thin trust-tier
  // pointer mounted readonly by container-runner.ts at spawn time, so
  // the trust flag at the moment of spawn picks the right template
  // every time (fixes #153 by construction). The agent's mutable
  // per-group memory lives in MEMORY.md; create an empty placeholder
  // here so the @import in CLAUDE.md resolves on the very first
  // message instead of the agent seeing a missing file.
  const memoryMdFile = path.join(groupDir, 'MEMORY.md');
  if (!fs.existsSync(memoryMdFile)) {
    fs.writeFileSync(
      memoryMdFile,
      `# Memory — ${group.name || group.folder}\n\n` +
        '_Persistent notes the agent has accumulated about this group. ' +
        'Append facts the agent should recall in future sessions._\n',
    );
    logger.info({ folder: group.folder }, 'Created empty MEMORY.md for group');
  }

  // Chown group folder to the container user so the agent can write to it.
  // In DooD the orchestrator runs as root — files it creates are root-owned.
  const effectiveUid = HOST_UID ?? process.getuid?.();
  const effectiveGid = HOST_GID ?? process.getgid?.();
  if (effectiveUid != null && effectiveUid !== 0) {
    try {
      chownRecursive(groupDir, effectiveUid, effectiveGid ?? effectiveUid);
    } catch (err) {
      logger.warn(
        { folder: group.folder, err },
        'Failed to chown group folder',
      );
    }
  }

  // Non-main heartbeat is retired (#453, see `cleanupOrphanNonMainHeartbeats`
  // above). The `containerConfig.enableHeartbeat` flag stays in the schema
  // for backwards compatibility with operators upgrading from older
  // orchestrator versions, but registerGroup no longer creates a
  // `heartbeat-<folder>` row for non-main groups. The startup cleanup
  // pass removes any rows left behind by older orchestrator versions.

  // Auto-create the parallel-maintenance heartbeat for every main group.
  // Mirrors the non-main auto-registration above, but runs in the
  // `maintenance` session slot so it doesn't block user-facing AyeAye.
  // The task-scheduler fires this every 15 minutes via
  // `MAINTENANCE_SESSION_NAME`; the prompt keeps the defensive preamble
  // as belt-and-suspenders against improvisation.
  if (group.isMain) {
    const heartbeatId = `heartbeat-${group.folder}`;
    if (!getTaskById(heartbeatId)) {
      createTask({
        id: heartbeatId,
        group_folder: group.folder,
        chat_jid: jid,
        prompt:
          'MANDATORY FIRST ACTION: Call Skill(skill: "tessl__heartbeat") BEFORE doing anything else. Do NOT improvise checks. Do NOT query databases. Do NOT invent thresholds. Load and execute the skill exactly as written.\n\n' +
          'This is a scheduled heartbeat — no ACK reaction, no reply_to.\n' +
          'Workspace: /workspace/group/\n' +
          'Telegram HTML ONLY: <b>, <i>, <code>, <a href="url">text</a>, • for bullets. NEVER Markdown.\n' +
          'CRITICAL: NEVER set the "sender" parameter on send_message. Always call send_message with only "text" and optionally "pin". The sender parameter routes through pool bots and bypasses the database — messages become ghosts.\n' +
          'If nothing actionable → produce NO output at all. Silence = success.',
        schedule_type: 'interval',
        schedule_value: '900000', // 15 minutes in ms
        // Heartbeats are stateless by design — every input is read from
        // external sources (messages.db, workspace, skills) on each tick,
        // so persisting the SDK session chain across runs is pure
        // liability. After 6 days of 15-min ticks the swarm group's
        // maintenance JSONL hit 187 MB and crossed the AUP-classifier
        // threshold, refusing every subsequent run (#114). A single
        // contaminated tick (poisoned tool_result, oversized image, hung
        // tool output) also got persisted forever and re-read on every
        // later tick. `'isolated'` makes each tick a fresh session —
        // manual recovery becomes unnecessary because there's no
        // accumulated state to wipe.
        context_mode: 'isolated',
        next_run: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
        status: 'active',
        created_at: new Date().toISOString(),
        created_by_role: 'owner',
      });
      logger.info(
        { jid, folder: group.folder },
        'Auto-created maintenance heartbeat for main group',
      );
    }
  }

  logger.info(
    { jid, name: group.name, folder: group.folder },
    'Group registered',
  );
}

function chownRecursive(dir: string, uid: number, gid: number): void {
  fs.chownSync(dir, uid, gid);
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    fs.chownSync(fullPath, uid, gid);
    if (entry.isDirectory()) {
      chownRecursive(fullPath, uid, gid);
    }
  }
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
      containerConfig: registeredGroups[c.jid]?.containerConfig,
      requiresTrigger: registeredGroups[c.jid]?.requiresTrigger,
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
async function processGroupMessages(chatJid: string): Promise<boolean> {
  let group = registeredGroups[chatJid];
  if (!group) return true;

  const channel = findChannel(channels, chatJid);
  if (!channel) {
    logger.warn({ chatJid }, 'No channel owns JID, skipping messages');
    return true;
  }

  const isMainGroup = group.isMain === true;

  // Circuit breaker: skip groups that have failed too many times in a row
  const breakerExpiry = circuitBreakerUntil[group.folder];
  if (breakerExpiry) {
    if (Date.now() < breakerExpiry) {
      logger.warn({ group: group.name }, 'Circuit breaker active — skipping');
      return true;
    }
    // Cooldown expired — reset and let the group try again
    delete circuitBreakerUntil[group.folder];
    consecutiveFailures[group.folder] = 0;
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

  await channel.setTyping?.(chatJid, true);
  let hadError = false;
  let outputSentToUser = false;

  // Progressive streaming disabled — causes message override bugs when
  // multiple messages are piped to the same container.

  // Track which message triggered the response — first reply quotes it.
  // Uses shared pendingReplyTo map so follow-up messages piped via
  // queue.sendMessage() can update the reply target for the output callback.
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
      // Streaming output callback — called for each agent result
      if (result.result) {
        const raw =
          typeof result.result === 'string'
            ? result.result
            : JSON.stringify(result.result);
        // Strip <internal>...</internal> blocks — agent uses these for internal reasoning
        const text = raw.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
        logger.info({ group: group.name }, `Agent output: ${raw.length} chars`);
        // #581 — when the agent already used send_message / send_file
        // successfully, the agent-runner sets `chat_displayed: true`.
        // The result text is still present (so the orchestrator can
        // log it / surface it), but we MUST NOT echo it back via
        // `channel.sendMessage` (would duplicate the user-visible
        // reply) and we MUST NOT call `storeMessage` (the IPC
        // `send_message` handler already wrote the bot row).
        if (text && !result.chat_displayed) {
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
        } else if (text && result.chat_displayed) {
          // #581 — agent already delivered the reply via send_message;
          // result text is preserved for logs but not re-sent.
          logger.info(
            {
              group: group.name,
              chatJid,
              contentLen: text.length,
            },
            '[output] chat_displayed set — skipping chat-echo (agent already sent via send_message)',
          );
          pendingReplyTo[chatJid] = undefined;
          outputSentToUser = true;
        }
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

  await channel.setTyping?.(chatJid, false);
  releaseIdleTimerControl(chatJid, idleTimerControl);

  if (output === 'error' || hadError) {
    // Track consecutive failures for circuit breaker
    consecutiveFailures[group.folder] =
      (consecutiveFailures[group.folder] || 0) + 1;
    if (consecutiveFailures[group.folder] >= MAX_CONSECUTIVE_FAILURES) {
      circuitBreakerUntil[group.folder] =
        Date.now() + CIRCUIT_BREAKER_COOLDOWN_MS;
      logger.error(
        { group: group.name, failures: consecutiveFailures[group.folder] },
        `Circuit breaker tripped — pausing group for ${CIRCUIT_BREAKER_COOLDOWN_MS / 60_000} minutes`,
      );
      // Notify via main group if this isn't the main group
      if (!isMainGroup) {
        const mainJid = Object.keys(registeredGroups).find(
          (jid) => registeredGroups[jid].isMain,
        );
        if (mainJid) {
          const mainChannel = findChannel(channels, mainJid);
          mainChannel?.sendMessage(
            mainJid,
            `Circuit breaker tripped for "${group.name}" — ${consecutiveFailures[group.folder]} consecutive failures. Paused for 30 minutes. Check logs.`,
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
  consecutiveFailures[group.folder] = 0;
  return true;
}

async function runAgent(
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

            const verdict = shouldMarkForReset(
              {
                totalInputTokens: snapshot.total_input_tokens,
                turnCount: snapshot.turn_count,
              },
              { tokenCap: SESSION_TOKEN_CAP, turnCap: SESSION_TURN_CAP },
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
        // documented in `wipeSessionJsonl` (src/index.ts:#581 — see
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
    logger.error({ group: group.name, err }, 'Agent error');
    return 'error';
  }
}

async function startMessageLoop(): Promise<void> {
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
            // Update shared reply-to so the output callback quotes this message
            pendingReplyTo[chatJid] = lastMsgId;
            logger.debug(
              {
                chatJid,
                count: messagesToSend.length,
                replyToMessageId: lastMsgId,
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
    } catch (err) {
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
  // #213: consult the graceful-shutdown handoff marker before killing
  // orphans. If the prior orchestrator exited cleanly (via the
  // SIGTERM handler below), it left a list of agent containers that
  // were still doing useful work — `cleanupOrphans` skips those and
  // only kills the rest. Without this, every `deploy.sh` cascades
  // 137 across every active conversation/heartbeat container, losing
  // the in-flight turn. A marker absent / stale / corrupt falls
  // through to the pre-#213 "kill all nanoclaw-* containers"
  // behaviour, which is the right default for genuine crash recovery
  // (a SIGKILL'd or hung orchestrator never wrote a marker).
  const handoff = readAndConsumeHandoffMarker();
  const skipNames = handoff
    ? new Set(handoff.containers.map((c) => c.name))
    : undefined;
  if (handoff) {
    logger.info(
      {
        shutdownAt: handoff.shutdown_at,
        containerCount: handoff.containers.length,
      },
      'Found graceful-shutdown handoff marker, adopting containers',
    );
    // Open the spawn-collision detection window (#213 Phase A) ONLY
    // when there's actually something to collide with. An empty
    // handoff (graceful shutdown with no active agents) means no
    // adopted containers exist — opening the window would cost a
    // `docker ps` per spawn for the next HANDOFF_TTL_MS and could
    // produce misleading WARNs against this orchestrator's own
    // freshly-spawned containers (the prefix check has no way to
    // distinguish "leftover from prior run" from "just spawned by
    // this run" once the prior run had nothing). Skip the window
    // when the container list is empty.
    if (handoff.containers.length > 0) {
      markHandoffActive();
    }
  }
  cleanupOrphans(skipNames);
}

async function main(): Promise<void> {
  // Install the outbound-Telegram HTTP tap FIRST — before any grammy Bot,
  // credential proxy, or channel loads. The tap wraps `fetch`, `http/https`
  // `request`, and `child_process.spawn/exec/execFile` to log any outbound
  // call to `api.telegram.org` regardless of which in-process code path
  // originates it. Gated on `LOG_LEVEL=debug` (same as #87's transformer);
  // no overhead at `info` or higher. See `telegram-outbound-tap.ts`.
  installTelegramOutboundTap();
  ensureContainerSystemRunning();
  initDatabase();
  logger.info('Database initialized');
  loadState();
  restoreRemoteControl();

  // Start credential proxy (containers route API calls through this)
  const proxyServer = await startCredentialProxy(
    CREDENTIAL_PROXY_PORT,
    PROXY_BIND_HOST,
  );

  // Graceful shutdown handlers
  const shutdown = async (signal: string) => {
    logger.info({ signal }, 'Shutdown signal received');
    // #213: snapshot active agent containers BEFORE `queue.shutdown`
    // clears their state, so the next startup can identify them as
    // intentional handoffs and skip the orphan-cleanup that
    // otherwise cascades 137 across every in-flight conversation /
    // heartbeat. Best-effort: a write failure here just means the
    // next startup will fall through to the pre-#213 cleanup
    // behaviour for these names — strictly worse than the new
    // path, but no worse than today.
    let active: ReturnType<typeof queue.getActiveContainersForHandoff> = [];
    try {
      active = queue.getActiveContainersForHandoff();
      writeHandoffMarker(active);
      logger.info(
        { count: active.length, names: active.map((c) => c.name) },
        'Wrote graceful-shutdown handoff marker',
      );
    } catch (err) {
      logger.warn({ err }, 'Failed to write handoff marker');
    }

    // #497: checkpoint every active default session BEFORE
    // `queue.shutdown` runs, so deploy SIGTERM → force-kill no longer
    // discards in-flight work. The existing SessionStart hook +
    // `session-reentry` skill pipeline picks up the checkpoint on the
    // next fresh spawn — no new reentry surface.
    const checkpointed = await writeShutdownCheckpoints({
      active,
      sessions,
      registeredGroups,
      thresholds: computeThresholds(MODEL_CONTEXT_WINDOW),
      defaultSessionName: DEFAULT_SESSION_NAME,
      dataDir: DATA_DIR,
      logger,
    });
    logger.info(
      { count: checkpointed },
      'Pre-shutdown checkpoint pass complete',
    );

    stopHubitatListener();
    proxyServer.close();
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

    if (!msg.is_from_me) {
      logger.warn(
        { chatJid, sender: msg.sender },
        'Remote control rejected: sender is not the account owner',
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

      storeMessage(msg);
    },
    onChatMetadata: (
      chatJid: string,
      timestamp: string,
      name?: string,
      channel?: string,
      isGroup?: boolean,
    ) => storeChatMetadata(chatJid, timestamp, name, channel, isGroup),
    onLocation: storeLocation,
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

  // Initialize Telegram bot pool for agent teams (swarm)
  if (TELEGRAM_BOT_POOL.length > 0) {
    await initBotPool(TELEGRAM_BOT_POOL);
  }

  // Cleanup MUST run before the scheduler starts. `startSchedulerLoop`
  // below kicks off its first `loop()` immediately, and if any orphan
  // non-main heartbeat row from an older orchestrator version has
  // `next_run <= now`, it would dispatch the (now-broken) old prompt
  // before cleanup gets a chance to delete the row. Running the cleanup
  // first guarantees the first post-deploy scheduler tick sees only
  // valid task rows.
  cleanupOrphanNonMainHeartbeats();

  // Surface registered-but-invisible-to-spawner rows (#159) at startup
  // so we notice future drift instead of growing dormant rows silently.
  logRegisteredGroupOrphans();

  // Initialize the optional observer channel — opt-in via OBSERVER_CHAT_JID
  // env var (no-op when unset). See src/observer.ts. Awaited so the
  // privacy-gate verification (warn loudly when the configured chat
  // is a multi-participant group, refuse only when no channel owns
  // the JID or chat-type lookup fails) finishes before we start
  // spawning queries that would feed the observer.
  await initObserver(channels, () => registeredGroups);

  // Start subsystems (independently of connection handler).
  // Scheduled tasks run through the shared queue under the parallel
  // `maintenance` slot, but they do NOT resume or persist an SDK session
  // chain across runs (#193). Each run gets a fresh sessionId; the
  // scheduler wipes the per-run on-disk artifacts (JSONL transcript +
  // tool-results dir) via `wipeSessionJsonl` once the run completes so
  // the per-slot `.claude/projects/` tree doesn't accumulate orphans.
  startSchedulerLoop({
    registeredGroups: () => registeredGroups,
    queue,
    onProcess: (groupJid, sessionName, proc, containerName, groupFolder) =>
      queue.registerProcess(
        groupJid,
        sessionName,
        proc,
        containerName,
        groupFolder,
      ),
    sendMessage: async (jid, rawText) => {
      const channel = findChannel(channels, jid);
      if (!channel) {
        logger.warn({ jid }, 'No channel owns JID, cannot send message');
        return;
      }
      const text = formatOutbound(rawText, channel.name as ChannelType);
      if (text) await channel.sendMessage(jid, text);
    },
    wipeSessionJsonl,
  });
  // Stage 1 trigger-pattern self-improvement loop (#82). Runs at a
  // configurable cadence (default 24h via TRIGGER_LEARNER_INTERVAL_MS),
  // emits PROPOSALS into `registered_groups.trigger_pattern` with
  // `enabled: false` (the trigger gate skips those rows). The owner
  // promotes via the existing admin path — there is no auto-apply.
  // Safety rails: cold-start floor, sender-tier weighting, auto-rollback
  // on FP-rate spike, and pattern versioning. See
  // `src/gates/trigger-learner.ts` for the full design.
  startTriggerLearner();
  startIpcWatcher({
    sendMessage: (jid, rawText, replyToMessageId) => {
      const channel = findChannel(channels, jid);
      if (!channel) throw new Error(`No channel for JID: ${jid}`);
      const text = formatOutbound(rawText, channel.name as ChannelType);
      if (!text) return Promise.resolve();
      return channel.sendMessage(jid, text, replyToMessageId);
    },
    sendReaction: async (jid, messageId, emoji) => {
      const channel = findChannel(channels, jid);
      if (!channel) return;
      if (messageId) {
        await channel.sendReaction?.(jid, messageId, emoji);
      } else {
        await channel.reactToLatestMessage?.(jid, emoji);
      }
    },
    pinMessage: async (jid, messageId) => {
      const channel = findChannel(channels, jid);
      if (!channel) return;
      await channel.pinMessage?.(jid, messageId);
    },
    sendFile: async (jid, filePath, caption, replyToMessageId) => {
      const channel = findChannel(channels, jid);
      if (!channel) return undefined;
      return channel.sendFile?.(jid, filePath, caption, replyToMessageId);
    },
    registeredGroups: () => registeredGroups,
    registerGroup,
    unregisterGroup: (jid) => {
      // Mirror DB delete into the in-memory registry so subsequent
      // routing decisions stop seeing the JID as registered before any
      // restart. Same site-of-truth pattern as setGroupTrusted /
      // setGroupTrigger above. Returns the DB delete's truthy-changes
      // result so the IPC handler can distinguish "actually removed"
      // from "wasn't there to begin with" — see #159.
      delete registeredGroups[jid];
      return deleteRegisteredGroup(jid);
    },
    setGroupTrusted: (jid, trusted) => {
      const updated = updateGroupTrusted(jid, trusted);
      if (!updated) return false;
      // Mirror DB change into the in-memory registry so subsequent
      // routing decisions see the new trust flag immediately, before any
      // restart. Without this, the agent would have to wait for the
      // orchestrator to reload from DB to see its own update.
      registeredGroups[jid] = updated;

      // Trust flips no longer touch heartbeat task state. The non-main
      // heartbeat was retired in #453 with the check-unanswered skill it
      // depended on; there's no per-trust-tier heartbeat script left to
      // reconcile. Main-group heartbeat is created and managed by
      // `registerGroup`'s `group.isMain` branch and isn't trust-tier
      // sensitive. If a future feature reintroduces a non-main
      // heartbeat with trust-tier-conditional behaviour, restore the
      // reconciliation here.
      return true;
    },
    setGroupTrigger: (jid, trigger, requiresTrigger) => {
      const updated = updateGroupTrigger(jid, trigger, requiresTrigger);
      if (!updated) return false;
      registeredGroups[jid] = updated;
      // Heartbeat lifecycle is intentionally NOT touched here. Pre-#158
      // a flip to `requiresTrigger=true` would auto-create a heartbeat
      // and the inverse flip would log a warning. With heartbeat opt-in
      // via `containerConfig.enableHeartbeat`, trigger config and
      // heartbeat are orthogonal — operators change each independently.
      return true;
    },
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
    nukeSession: (
      groupFolder: string,
      session: 'default' | 'maintenance' | 'all',
      options?: { skipReentry?: boolean },
    ) => {
      // Stamp the nuke's wall-clock timestamp BEFORE doing any of the
      // wipe work — the in-flight spawn handler (runAgent) compares
      // this against its own pre-spawn timestamp to decide whether
      // the SDK result it just got is from a session that's since
      // been nuked. The Date.now() resolution + the
      // setSession-before-nuke sequence is robust against the race
      // observed in #144 bug 1 (concurrent setSession resurrected
      // a row that the same call had just deleted).
      nukeTimestamps[groupFolder] = Date.now();
      // Granular nuke: `session` narrows which slot(s) to kill.
      //   'all'         → kill default + maintenance (pre-parallel default)
      //   'default'     → kill only user-facing container
      //   'maintenance' → kill only scheduled-task container
      // Useful when one session is wedged (e.g. a hung heartbeat in
      // maintenance) and we don't want to drop the user's default
      // conversation state as collateral damage.
      //
      // Per #100, the nuke runs in four steps, in order:
      //   1. Capture the SDK sessionIds we're about to drop (before
      //      clearing them — once they're gone we can't find the
      //      on-disk artifacts).
      //   2. Kill the running container(s) so nothing keeps writing.
      //   3. Delete the session rows from the DB and clear in-memory.
      //   4. Delete the on-disk session artifacts (JSONL transcript
      //      and the per-session tool-results directory beside it).
      //
      // Without step 4, the next container spawn re-reads whatever poison
      // / stuck plan / corrupt state put the session in a bad state and
      // we're right back where we started — see #100 for the Gmail
      // invisible-Unicode incident that motivated this.
      const slotsToWipe: Array<'default' | 'maintenance'> =
        session === 'all'
          ? ['default', 'maintenance']
          : [
              session === 'default'
                ? DEFAULT_SESSION_NAME
                : MAINTENANCE_SESSION_NAME,
            ];
      const sessionIdsToWipe = new Map<string, string>();
      for (const slot of slotsToWipe) {
        const sid = sessions[groupFolder]?.[slot];
        if (sid) sessionIdsToWipe.set(slot, sid);
      }

      const jid =
        Object.entries(registeredGroups).find(
          ([, g]) => g.folder === groupFolder,
        )?.[0] || '';
      if (jid) {
        if (session === 'default' || session === 'all') {
          queue.closeStdin(jid, DEFAULT_SESSION_NAME);
        }
        if (session === 'maintenance' || session === 'all') {
          queue.closeStdin(jid, MAINTENANCE_SESSION_NAME);
        }
      }
      // Clear stored sessionIds for the killed slot(s). `deleteSession`
      // removes every row for the folder — reuse for 'all'. For
      // single-slot nukes we use the new `deleteSessionName` helper so
      // the surviving slot keeps its session chain.
      if (session === 'all') {
        delete sessions[groupFolder];
        deleteSession(groupFolder);
      } else {
        const sessionName =
          session === 'default'
            ? DEFAULT_SESSION_NAME
            : MAINTENANCE_SESSION_NAME;
        if (sessions[groupFolder]) delete sessions[groupFolder][sessionName];
        deleteSessionName(groupFolder, sessionName);
      }

      // Step 4: wipe on-disk session artifacts (JSONL transcript +
      // per-session tool-results directory). Delete-while-open is
      // safe on POSIX (the container's open FD keeps writing to a
      // phantom inode that vanishes on close), so we don't have to wait
      // for closeStdin to actually terminate the process. The returned
      // `count` is the total number of filesystem entries removed: up
      // to 2 per slug (1 transcript + 1 tool-results dir), summed
      // across every project-slug subdirectory walked.
      for (const [slot, sessionId] of sessionIdsToWipe) {
        const wiped = wipeSessionJsonl(groupFolder, slot, sessionId);
        if (wiped > 0) {
          logger.info(
            { groupFolder, sessionName: slot, sessionId, count: wiped },
            'Wiped session artifacts (transcript + tool-results dir)',
          );
        }
      }

      // Step 4b (#336): clear per-task `session_id` columns for any
      // scheduled task in this group that referenced the just-wiped
      // maintenance transcripts. Without this, the next fire of a
      // recurring task would pass `resume:` an id whose JSONL is
      // gone — the SDK would 404 and start fresh anyway, just
      // noisily. Only clears when the maintenance slot was actually
      // touched: a 'default'-only nuke leaves scheduled-task sessions
      // untouched (they live in maintenance, with their own ids).
      // Fires before `skipReentry` so checkpoint state and per-task
      // session state both reach "clean slate" together.
      if (session === 'maintenance' || session === 'all') {
        const cleared = clearTaskSessionIdsForGroup(groupFolder);
        if (cleared > 0) {
          logger.info(
            { groupFolder, count: cleared },
            'Cleared per-task session_ids — next fire of each will start a fresh SDK session (#336)',
          );
        }
      }

      // Step 4c (#413): drop session-length-cap accounting for the
      // nuked group. Without this, a stale row could carry a
      // `marked_for_reset = 1` flag against a session_id that's
      // already gone, and the next inbound spawn would consume the
      // marker (harmless but noisy in logs). The cap state is
      // group-level; an 'all' nuke clears every slot, while a
      // single-slot nuke is rare enough that wiping both slots'
      // accounting is fine — the surviving slot's next turn
      // re-INSERTs cleanly.
      const clearedCapRows = clearSessionLengthStateForGroup(groupFolder);
      if (clearedCapRows > 0) {
        logger.info(
          { groupFolder, count: clearedCapRows },
          'Cleared session_length_state rows on nuke (#413)',
        );
      }

      // Step 5 (#127, optional): when `skipReentry` is set, also
      // delete the per-group checkpoint files so the next container
      // spawn has no Facts/Reasoning to load via the reentry skill.
      // Default behaviour (option absent or false) preserves the
      // checkpoint — the standard nuke is "fresh session, but the
      // reentry skill still runs" because checkpoints typically
      // outlive a single nuke (they're written by the threshold-cross
      // path). Skip-reentry exists for the case where the checkpoint
      // itself is the problem (poisoned plan, stale do-not-re-execute
      // list); without this, the operator's only workaround was a
      // manual `rm` from the host.
      //
      // Checkpoint files are per-group, NOT per-slot — there's one
      // pair under `<groupDir>/.checkpoints/` shared by both default
      // and maintenance. So skipReentry deletes the same files
      // regardless of which slot was nuked. That matches the design
      // doc (see `docs/proposals/kill-auto-compaction.md` §1, §2):
      // the Facts section is the orchestrator's view of "what just
      // happened in this group", not slot-specific.
      // Strict boolean check (defense in depth): the IPC layer
      // already filters non-true values, but a future direct caller
      // (test, new IPC handler, refactor) could pass a truthy
      // non-boolean and accidentally erase reentry state. The dispatcher
      // is the last gate before the disk operation, so it owns the
      // strictest check.
      if (options?.skipReentry === true) {
        let groupDir: string;
        try {
          groupDir = resolveGroupFolderPath(groupFolder);
        } catch (err) {
          // Per `jbaruch/coding-policy: error-handling`: only handle
          // the expected case (Error from path validation), let
          // anything else propagate. resolveGroupFolderPath
          // documents Error throws on path-traversal / invalid
          // segment; non-Error throws here would indicate a bug
          // upstream and should bubble up to the IPC dispatch
          // wrapper, which logs and keeps the orchestrator alive.
          if (!(err instanceof Error)) throw err;
          // The expected case: bad groupFolder. Log full error
          // object (logger handles `err` specially — preserves
          // stack, formats nicely) and skip the checkpoint clear
          // without blocking the rest of the nuke. Reentry skill
          // will find the checkpoint still on disk; operator can
          // rerun with a fixed group_folder.
          logger.error(
            { groupFolder, err },
            'skipReentry: cannot resolve group folder — checkpoint files left in place',
          );
          logger.info({ groupFolder, session }, 'Session nuked via IPC');
          return;
        }
        // Best-effort cleanup: if clearCheckpoints throws (e.g.
        // EACCES on unlink — a file was found but couldn't be
        // removed), log at error level and CONTINUE with the rest of
        // the nuke. The main session state (DB rows + JSONL) is
        // already wiped at this point; failing the whole IPC handler
        // would be noisier than helpful and contradicts the
        // best-effort framing the surrounding comments describe.
        // Non-Error throws still propagate as upstream bugs per
        // `jbaruch/coding-policy: error-handling`.
        try {
          const checkpointsDeleted = clearCheckpoints(groupDir);
          logger.info(
            { groupFolder, checkpointsDeleted },
            'Checkpoint files cleared (skipReentry=true)',
          );
        } catch (err) {
          if (!(err instanceof Error)) throw err;
          logger.error(
            { groupFolder, groupDir, session, err },
            'skipReentry: failed to clear checkpoint files — continuing with session nuke',
          );
        }
      }

      logger.info(
        { groupFolder, session, skipReentry: options?.skipReentry === true },
        'Session nuked via IPC',
      );
    },
    getContainerStatus: (chatJid, sessionName) => {
      // Combine the GroupQueue's per-slot signals (active/idleWaiting/
      // retryCount/lastExitStatus) with the long-term per-folder
      // circuit breaker. The breaker lives here, not in GroupQueue,
      // because it's keyed on group.folder and is set by message-loop
      // bookkeeping rather than queue lifecycle. Both signals are
      // cooldown windows from the chat_status caller's perspective.
      const group = registeredGroups[chatJid];
      const breakerExpiry = group ? circuitBreakerUntil[group.folder] : 0;
      const breakerActive = !!breakerExpiry && Date.now() < breakerExpiry;
      return queue.getStatus(chatJid, sessionName, breakerActive);
    },
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
        writeTasksSnapshot(
          group.folder,
          group.isMain === true,
          taskRows,
          !!group.containerConfig?.trusted,
        );
      }
    },
    closeAllActiveContainers: () => queue.closeAllActiveContainers(),
  });
  startSessionCleanup();
  queue.setProcessMessagesFn(processGroupMessages);
  recoverPendingMessages();

  // Per-container streaming logs grow with every spawn. Without
  // pruning, `data/host-logs/containers/<group>/<session>/*.log`
  // would accumulate indefinitely — a chatty trusted group spawning
  // many times a day fills the disk over a few months. Run prune at
  // startup AND once a day thereafter; both are safe and cheap.
  // Retention window is owned by host-logs.ts (currently 7 days).
  void (async () => {
    try {
      const deleted = pruneOldContainerLogs();
      if (deleted > 0) {
        logger.info(
          { deleted },
          'host-logs prune at startup removed expired per-spawn logs',
        );
      }
    } catch (err) {
      logger.warn({ err }, 'host-logs prune at startup failed');
    }
  })();
  // 24h interval, unref'd so the timer doesn't keep the orchestrator
  // alive past graceful shutdown. setInterval is fine even though the
  // logical schedule is "once per day" — the orchestrator process
  // typically lives for weeks, and crash recovery brings the timer
  // back on next start.
  const ONE_DAY_MS = 24 * 60 * 60 * 1000;
  setInterval(() => {
    try {
      const deleted = pruneOldContainerLogs();
      if (deleted > 0) {
        logger.info(
          { deleted },
          'host-logs daily prune removed expired per-spawn logs',
        );
      }
    } catch (err) {
      logger.warn({ err }, 'host-logs daily prune failed');
    }
  }, ONE_DAY_MS).unref();

  // Silent-zero-output guard for the credential-proxy usage log
  // (#479 sub-#3). Polls every minute; emits one ERROR if the proxy
  // has handled at least one /v1/messages POST but produced zero
  // JSONL records past the warmup grace period. Edge-triggered
  // against messagesSeen so a persistent broken state logs once,
  // not every tick — see `checkSilentZero` in src/usage-log.ts.
  const SILENT_ZERO_POLL_MS = 60 * 1000;
  setInterval(() => {
    const diag = checkSilentZero();
    if (diag) {
      logger.error(
        { ...diag },
        'usage-log: /v1/messages traffic seen but zero JSONL records — capture pipeline broken (see #479 sub-#3)',
      );
    }
  }, SILENT_ZERO_POLL_MS).unref();

  // Write available_groups.json for all main/trusted groups on startup.
  // Otherwise the snapshot only updates when a container spawns, which can
  // leave it weeks stale if the group doesn't get traffic.
  const startupGroups = getAvailableGroups();
  const startupRegisteredJids = new Set(Object.keys(registeredGroups));
  for (const [, group] of Object.entries(registeredGroups)) {
    if (group.isMain || group.containerConfig?.trusted) {
      writeGroupsSnapshot(
        group.folder,
        group.isMain === true,
        startupGroups,
        startupRegisteredJids,
        !!group.containerConfig?.trusted,
      );
    }
  }

  // Start Hubitat smart home listener (if configured)
  startHubitatListener();

  // #496 — stale-lock recovery. On startup, clear any
  // `follow_me_tasks.pending_run_at` older than the freshness window.
  // A pre-restart crash (or a `tessl_update`-killed run from before
  // this fix shipped) can leave dangling locks that block the next
  // scheduled fire on its Phase A gate. The host is a non-owner
  // reader of `follow_me_tasks` per
  // `coding-policy: stateful-artifacts`, so we don't migrate the
  // schema; we only NULL out value fields whose owners are demonstrably
  // dead. Threshold matches `PENDING_RUN_AT_FRESHNESS_MS` below — see
  // that constant's doc for the rationale.
  try {
    const cleared = clearStalePendingRunAt(PENDING_RUN_AT_FRESHNESS_MS);
    if (cleared.length > 0) {
      logger.warn(
        { tasks: cleared, ageThresholdMs: PENDING_RUN_AT_FRESHNESS_MS },
        'Cleared stale pending_run_at locks at startup (#496) — likely orphaned by a prior crash or a tessl_update mid-flight kill',
      );
    }
  } catch (err) {
    if (!(err instanceof Error)) throw err;
    logger.error(
      { err },
      'Startup pending_run_at cleanup failed — scheduled tasks with stale locks may refuse to fire on Phase A gate; investigate follow_me_tasks rows manually',
    );
  }

  // Periodic tile update from registry (every 15 min)
  // Heartbeat runs in the container and can't call tessl update.
  // This catches publishes that the post-promote timer missed.
  const { execFile: execTesslUpdate } = await import('child_process');
  setInterval(() => {
    // #496 — defer the periodic tessl_update if a scheduled task is
    // currently mid-flight (its skill has acquired a fresh
    // `follow_me_tasks.pending_run_at` lock). Running tessl_update now
    // would force-close the maintenance container via the
    // `closeAllActiveContainers` path and leave the lock dangling.
    // Skipping this tick is harmless — the next 15-min tick retries,
    // and the periodic catch-up isn't time-critical (the on-demand
    // `tessl_update` MCP tool is the load-bearing path for fresh
    // tile content; this loop is the safety net for missed
    // invocations).
    let activeLocks: string[];
    try {
      activeLocks = getActivePendingRunAtNames(PENDING_RUN_AT_FRESHNESS_MS);
    } catch (err) {
      if (!(err instanceof Error)) throw err;
      logger.warn(
        { err },
        'Periodic tessl update: pending_run_at check failed — proceeding with update (failing closed would skip every tick)',
      );
      activeLocks = [];
    }
    if (activeLocks.length > 0) {
      logger.info(
        { activeLocks },
        'Periodic tessl update deferred — scheduled task(s) mid-flight with fresh pending_run_at lock (#496); will retry on next 15-min tick',
      );
      return;
    }
    execTesslUpdate(
      'bash',
      [
        '-c',
        'cd /app/tessl-workspace && tessl update --yes --dangerously-ignore-security --agent claude-code 2>&1',
      ],
      { timeout: 120_000 },
      (err, stdout) => {
        if (err) {
          logger.warn({ error: err.message }, 'Periodic tessl update failed');
        } else if (stdout.includes('Updated')) {
          const cleared = deleteAllSessions();
          // Companion to the on-demand `tessl_update` IPC handler in
          // `ipc.ts`: any path that pulls new tile content into the
          // registry must also signal currently-running containers to
          // restart, otherwise they keep serving requests from the
          // skills/.tessl/ snapshot they copied at spawn time until
          // their 30-min idle timeout (issue #64).
          //
          // Wrapped because `closeAllActiveContainers()` rethrows
          // unexpected (non-fs) errors by contract — without this guard,
          // a programming bug surfacing through that path would propagate
          // out of the `setInterval` callback as an uncaught exception
          // and crash the orchestrator. Sessions stay cleared either way;
          // we degrade to "containers will pick up new tiles on idle
          // timeout" rather than taking the process down.
          let closed = 0;
          try {
            closed = queue.closeAllActiveContainers();
          } catch (closeErr) {
            // Narrow to Error instances (the only thing realistic
            // production code throws). Non-Error throws (a bare string,
            // `null`, etc.) are themselves a programming bug and
            // propagate as uncaught exceptions per error-handling.md
            // ("let unexpected propagate"). This catch is the
            // outer-boundary guard for an async callback — without it,
            // an Error from the close path would terminate the
            // orchestrator process; with it, sessions stay cleared and
            // we degrade to "containers refresh on idle timeout."
            if (!(closeErr instanceof Error)) throw closeErr;
            logger.error(
              { err: closeErr, sessionsCleared: cleared },
              'closeAllActiveContainers threw an unexpected error during periodic tessl update — sessions still cleared, but live containers will not respawn until idle timeout',
            );
          }
          logger.info(
            {
              sessionsCleared: cleared,
              containersClosed: closed,
              output: stdout.trim().slice(-200),
            },
            'Periodic tessl update found new tiles — sessions cleared and running containers signaled to restart',
          );
        }
      },
    );
  }, 900_000);

  // #542 — Heartbeat advisory walker. Re-walks the cached
  // `tz_state.segments` timeline every 30 min so a `from`/`to`
  // segment boundary crossing flips `current_tz` mid-day without
  // waiting for the next nightly `sync_tripit` run. Silent skip on
  // every path where there's nothing to do (no row, no segments,
  // computed tz matches current_tz, malformed cache); on flip,
  // notify the main group via its channel so the user sees the
  // change (matches the circuit-breaker notify pattern).
  const TZ_HEARTBEAT_INTERVAL_MS = 30 * 60 * 1000;
  setInterval(() => {
    let advisory: TzAdvisoryResult;
    try {
      advisory = runTzHeartbeatAdvisory(new Date(), ASSISTANT_OWNER_TG_USER_ID);
    } catch (err) {
      // Narrowed to transient SQLite contention codes only —
      // SQLITE_BUSY / SQLITE_LOCKED can fire under WAL contention
      // with the orchestrator's own writers and are genuinely
      // recoverable on the next 30-min tick. Every other SqliteError
      // (SQLITE_CORRUPT, SQLITE_SCHEMA, SQLITE_READONLY, missing
      // column from a botched migration, etc.) signals a
      // persistent state-layer problem that hiding behind a periodic
      // warn would silently freeze `current_tz` indefinitely; those
      // propagate per `coding-policy: error-handling`. The
      // malformed-JSON case is handled inside
      // `runTzHeartbeatAdvisory` (narrowed `SyntaxError`) and returns
      // null rather than throwing, so the only thing that reaches
      // this catch is a real DB-side failure.
      const TRANSIENT_SQLITE_CODES = new Set(['SQLITE_BUSY', 'SQLITE_LOCKED']);
      if (
        !(err instanceof SqliteError) ||
        !TRANSIENT_SQLITE_CODES.has(err.code)
      ) {
        throw err;
      }
      logger.warn(
        { err: err.message, code: err.code },
        'tz heartbeat advisory: transient SqliteError on read — will retry on next 30-min tick',
      );
      return;
    }
    if (!advisory.flip && !advisory.warningToFire) return;
    const mainJid = Object.keys(registeredGroups).find(
      (jid) => registeredGroups[jid].isMain,
    );
    if (!mainJid) {
      // No main group registered — the flip + cooldown stamp already
      // landed on the DB row; the next time a main group is
      // registered, the user will see the right `current_tz` without
      // a separate notification. Skipping the chat send here is the
      // right failure mode (chat delivery would have nothing to
      // target).
      return;
    }
    const mainChannel = findChannel(channels, mainJid);
    if (!mainChannel || !mainChannel.isConnected()) return;

    // Chat notify: best-effort. The DB row already reflects the
    // flip / cooldown stamp, so a chat-send failure means "user
    // doesn't see the message this tick" — non-fatal. Each notify
    // gets its own `.catch` to convert rejection into a structured
    // warn instead of bubbling as an unhandled-rejection warning
    // that newer Node versions can terminate the orchestrator over.
    if (advisory.flip) {
      const flipForLog = advisory.flip;
      mainChannel
        .sendMessage(
          mainJid,
          `📍 Timezone changed: ${flipForLog.prev} → ${flipForLog.next}`,
        )
        .catch((err: unknown) => {
          if (!(err instanceof Error)) throw err;
          logger.warn(
            {
              err: err.message,
              mainJid,
              prev: flipForLog.prev,
              next: flipForLog.next,
            },
            'tz heartbeat advisory: chat notify failed (DB row already updated)',
          );
        });
    }
    if (advisory.warningToFire === 'stale_no_share') {
      // Cooldown is already enforced inside runTzHeartbeatAdvisory
      // (state-015 column `last_stale_warning_at`); reaching here
      // means a real fire is due, not a per-tick spam.
      mainChannel
        .sendMessage(
          mainJid,
          `⚠️ I haven't seen your location in 12 h+. If you're traveling, share your live location so TZ tracking stays accurate (#574 Phase 2).`,
        )
        .catch((err: unknown) => {
          if (!(err instanceof Error)) throw err;
          logger.warn(
            { err: err.message, mainJid },
            'tz heartbeat advisory: stale-location nag failed (DB cooldown stamp already updated; next 12 h tick will retry)',
          );
        });
    }
  }, TZ_HEARTBEAT_INTERVAL_MS).unref();

  startMessageLoop().catch((err) => {
    logger.fatal({ err }, 'Message loop crashed unexpectedly');
    process.exit(1);
  });
}

// Guard: only run when executed directly, not when imported by tests
const isDirectRun =
  process.argv[1] &&
  new URL(import.meta.url).pathname ===
    new URL(`file://${process.argv[1]}`).pathname;

if (isDirectRun) {
  main().catch((err) => {
    logger.error({ err }, 'Failed to start NanoClaw');
    process.exit(1);
  });
}
