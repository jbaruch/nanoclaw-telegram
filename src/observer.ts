/**
 * Agent activity observer — optional forwarder that sends per-query summaries
 * and live error alerts to a dedicated Telegram "observer" chat, parsing the
 * agent-runner's stderr lines that container-runner already emits as debug logs.
 *
 * Enable by setting `OBSERVER_CHAT_JID=tg:-100...` in .env or the plist.
 */
import { readEnvFile } from './env.js';
import { logger } from './logger.js';
import type { Channel, RegisteredGroup } from './types.js';

// The configured observer chat JID. Resolved once at module load
// from process.env, falling back to the on-disk .env file. Tests
// override this via __enableObserverForTests, which sets
// `observerChatJidOverride` directly.
const OBSERVER_CHAT_JID_FROM_ENV =
  process.env.OBSERVER_CHAT_JID ||
  readEnvFile(['OBSERVER_CHAT_JID']).OBSERVER_CHAT_JID;
let observerChatJidOverride: string | undefined;
function getObserverChatJid(): string | undefined {
  return observerChatJidOverride ?? OBSERVER_CHAT_JID_FROM_ENV;
}

let channelsRef: Channel[] | null = null;
let registeredGroupsRef: (() => Record<string, RegisteredGroup>) | null = null;
// Toggled to true only after we've verified the configured JID points
// at a 1:1 / DM chat. Stays false on misconfiguration so onAgentLine
// becomes a no-op and no thinking content leaks into a wrong chat.
let observerEnabledFlag = false;

export async function initObserver(
  channels: Channel[],
  registeredGroups: () => Record<string, RegisteredGroup>,
): Promise<void> {
  channelsRef = channels;
  registeredGroupsRef = registeredGroups;
  const jid = getObserverChatJid();
  if (!jid) return;

  // Privacy gate: warn loudly when OBSERVER_CHAT_JID points at a
  // multi-participant chat, but enable anyway. The observer mirrors
  // *all* containers' thinking, tool-use, and partial output into
  // this chat, so accidentally pointing it at a group with external
  // members would be a wholesale leak — but some operators
  // intentionally run a "single-user private group" (just bot + self)
  // as their observer chat, which Telegram exposes as a group rather
  // than a 1:1 DM. Refusing all non-private chats would block that
  // legitimate setup; warning at startup makes a misconfiguration
  // visible without locking out the legitimate one. The env var IS
  // the credential — operators are expected to confirm membership
  // before setting it.
  const owner = channels.find((c) => c.ownsJid(jid) && c.isConnected());
  if (!owner) {
    logger.error(
      { jid },
      'Observer disabled: no connected channel owns the configured JID',
    );
    return;
  }
  if (!owner.isPrivateChat) {
    logger.error(
      { channel: owner.name, jid },
      'Observer disabled: channel cannot verify chat is private — refusing to enable',
    );
    return;
  }
  let isPrivate: boolean;
  try {
    isPrivate = await owner.isPrivateChat(jid);
  } catch (err) {
    // Fail-closed security boundary: any verification error refuses to enable
    // the observer. A non-Error throw is a defect and propagates.
    if (!(err instanceof Error)) throw err;
    logger.error(
      { err, jid },
      'Observer disabled: failed to verify chat type — refusing to enable',
    );
    return;
  }
  if (!isPrivate) {
    // Operator chose a non-private chat (group / channel). This IS a
    // leak surface — the observer mirrors thinking, tool-use, and
    // partial output from EVERY container, and any member of the
    // observer chat sees that stream. We allow it because some
    // operators run a deliberate "single-user private group" as the
    // observer (no third parties added). Loud warn at startup so the
    // misconfiguration is visible if it happens by accident.
    logger.warn(
      { jid },
      'Observer chat is a group / channel, NOT a 1:1 DM. Anyone in the chat will see all containers reasoning. Use a chat with only the bot + you, or switch OBSERVER_CHAT_JID to a private DM.',
    );
  }
  observerEnabledFlag = true;
  logger.info({ jid, isPrivate }, 'Observer chat enabled');
  armSelfTest();
}

// Per-(chat, message) dedupe map: remember the last emoji we set so
// we don't re-fire identical reactions to the Telegram API. The map
// is keyed `${chatJid}:${msgId}` because two in-flight messages in
// the same chat must not collapse onto a single key — that would
// cause one's emoji write to suppress the other's.
//
// `latestUserMessage` (host-seeded fallback) was removed in #289 once
// the observer started parsing `target_message_id` from every Query
// input log line — for channel-routed inbounds, the agent-runner
// always emits one, so the host fallback was strictly redundant.
// #104's intent (only emit 👀 after gate chain returns allow) is
// satisfied structurally in canonical: the agent-runner's react-first
// hook only fires when the container is alive, which only happens
// after the gate chain has already returned allow.
const lastReactionEmoji = new Map<string, string>(); // `${chatJid}:${msgId}` -> emoji

// Reverse map cache for folderToChatJid. Rebuilt only when the
// registeredGroups dict identity changes (the orchestrator hands us a
// closure over the live dict, so identity equality is the cheapest
// invalidation signal). Avoids the per-thinking-block O(N) scan.
let cachedGroupsDict: Record<string, RegisteredGroup> | null = null;
let cachedFolderToJid: Map<string, string> = new Map();

function folderToChatJid(folder: string): string | undefined {
  if (!registeredGroupsRef) return undefined;
  const groups = registeredGroupsRef();
  if (groups !== cachedGroupsDict) {
    cachedFolderToJid = new Map();
    for (const [jid, g] of Object.entries(groups)) {
      cachedFolderToJid.set(g.folder, jid);
    }
    cachedGroupsDict = groups;
  }
  return cachedFolderToJid.get(folder);
}

function updateReaction(folder: string, emoji: string): void {
  if (!channelsRef) return;
  const chatJid = folderToChatJid(folder);
  if (!chatJid) return;
  const state = states.get(folder);
  // #289 — addressed-ness gate. Suppress the entire reaction ladder
  // (🤔/⚡/✍/watchdog blinks/done) for explicitly non-addressed
  // queries. The agent reasons about every inbound in
  // `requires_trigger=false` rooms but should not produce any
  // visible reaction trail on bystander chatter — that was the
  // noise this whole gate exists to kill. `undefined` falls through
  // to the engagement-gate alone for backward compat with old log
  // lines that don't carry the field.
  if (state?.addressed === false) return;
  // The agent-runner emits `target_message_id=<id>` on every Query
  // input log line for channel-routed inbounds (the orchestrator
  // wraps each in `<message id="…">` and `extractLatestInboundId`
  // resolves it before logging). No fallback needed: scheduled
  // tasks and raw prompts emit `-`, leaving `targetMessageId`
  // undefined here — and those paths shouldn't fire reactions on
  // any chat message anyway.
  const msgId = state?.targetMessageId;
  if (!msgId) return;
  // Dedupe on (chat, msg) — using chat alone collapses across
  // different in-flight messages and would re-fire emojis on
  // each one even when the same emoji was just set on another
  // message in the same chat.
  const dedupeKey = `${chatJid}:${msgId}`;
  if (lastReactionEmoji.get(dedupeKey) === emoji) return;
  lastReactionEmoji.set(dedupeKey, emoji);
  const channel = channelsRef.find(
    (c) => c.ownsJid(chatJid) && c.isConnected() && c.sendReaction,
  );
  if (!channel?.sendReaction) return;
  channel.sendReaction(chatJid, msgId, emoji).catch((err: unknown) => {
    logger.debug(
      { err, chatJid, msgId, emoji },
      'Observer reaction update failed',
    );
  });
}

// Liveness watchdog: long queries with no chat output look like the bot
// hung. We blink the reaction emoji and, past a longer threshold, post a
// terse "still working" message to the user's chat so they know it's alive.
interface Watchdog {
  intervalId: NodeJS.Timeout;
  startedAt: number;
  pingsSent: number;
  lastBlinkEmoji: string;
}
const watchdogs = new Map<string, Watchdog>(); // source -> watchdog

const BLINK_INTERVAL_MS = 30_000;
const PING_AT_SECONDS = [60, 120, 300]; // 1m, 2m, 5m
// Liveness-only emojis — chosen so they DON'T overlap with the
// semantic states emitted from agent events: ⚡ (tool-use, except
// send_message) and ✍ (send_message in flight). Using ⚡ here would
// overwrite legitimate tool-fired state mid-query and leave the
// "tool" emoji stuck on the message after a long watchdog cycle even
// when the final state should have been ✍ (composed reply) or 🤔
// (thinking-only). 🫡 / 🤓 are both in TELEGRAM_ALLOWED_REACTIONS
// (see src/channels/telegram.ts) and read as "still on it" without
// claiming a particular semantic phase.
const BLINK_PAIR = ['🫡', '🤓'];
// Final reaction set when a query completes. The watchdog blink
// otherwise leaves whichever blink-emoji happened to be current on
// the user's message — unrelated to the actual end-state of the
// query. ✍ is what telegram.ts initially writes when send_message
// fires, but if no send_message ran (thinking-only or pure-text reply
// path) we still need a deterministic "done" emoji.
const DONE_REACTION = '🤝';

function chatChannel(chatJid: string): Channel | undefined {
  return channelsRef?.find((c) => c.ownsJid(chatJid) && c.isConnected());
}

function startWatchdog(source: string): void {
  if (watchdogs.has(source)) return;
  const startedAt = Date.now();
  const w: Watchdog = {
    startedAt,
    pingsSent: 0,
    lastBlinkEmoji: BLINK_PAIR[0],
    intervalId: setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - startedAt) / 1000);
      // Engagement gate: don't blink reactions on a message the
      // agent is silently ignoring. If the turn never commits to
      // a user-visible action (text / tool_use), no observer
      // reaction has fired yet — blinking 🫡/🤓 here would light
      // up the user's chat with bot activity for a message that's
      // about to receive zero response.
      if (!states.get(source)?.committed) return;
      // Blink reaction
      const next =
        w.lastBlinkEmoji === BLINK_PAIR[0] ? BLINK_PAIR[1] : BLINK_PAIR[0];
      w.lastBlinkEmoji = next;
      updateReaction(source, next);
      // Threshold pings — once each. Restricted to the main chat:
      // posting "Still working — 60s in" into a shared trusted /
      // untrusted group is conversational noise for everyone *else*
      // in the chat (the people who didn't ask the bot anything),
      // and in untrusted contexts it also leaks "I'm grinding on
      // your prompt" before the agent's bad-actor-disengage rule
      // had a say. The blink reaction above stays for ALL chats
      // (it's just a reaction on the user's own message — no
      // broadcast surface). Threshold pings broadcast a new
      // message, so they go only where conversation is owner-only.
      const nextThreshold = PING_AT_SECONDS[w.pingsSent];
      if (nextThreshold && elapsedSec >= nextThreshold) {
        w.pingsSent++;
        const chatJid = folderToChatJid(source);
        // Same per-state target as updateReaction — pin the ping
        // to the message this query is actually processing.
        const stateForPing = states.get(source);
        const msgId = stateForPing?.targetMessageId;
        const groups = registeredGroupsRef?.();
        const isMainChat = chatJid && groups?.[chatJid]?.isMain === true;
        if (chatJid && msgId && isMainChat) {
          const ch = chatChannel(chatJid);
          const state = states.get(source);
          const toolCount = state?.toolCalls.length ?? 0;
          const text = `<i>Still working — ${elapsedSec}s in${toolCount ? `, ${toolCount} tools so far` : ''}.</i>`;
          ch?.sendMessage(chatJid, text, msgId).catch((err) =>
            logger.debug({ err }, 'Watchdog ping failed'),
          );
        }
      }
    }, BLINK_INTERVAL_MS),
  };
  watchdogs.set(source, w);
}

function stopWatchdog(source: string, reactionOnStop?: string): void {
  const w = watchdogs.get(source);
  if (!w) return;
  clearInterval(w.intervalId);
  watchdogs.delete(source);
  // Sentinel: caller passed `'__skip__'` to mean "tear down the
  // watchdog interval but DO NOT touch the user's reaction." Used
  // when the turn ended without any agent commitment to engagement
  // — silent thinking-only turn — so the message stays untouched.
  if (reactionOnStop === '__skip__') return;
  // Reset the user's chat reaction so a stale 🫡/🤓 blink doesn't
  // outlive the watchdog. If the caller didn't pick a specific
  // emoji (e.g. mid-flight watchdog teardown to defuse a stale
  // entry), fall back to the deterministic done emoji.
  const target = reactionOnStop ?? DONE_REACTION;
  updateReaction(source, target);
}

export function observerEnabled(): boolean {
  return observerEnabledFlag;
}

/**
 * Test seam — reset all module-local state so independent tests
 * don't pollute each other through the per-source maps. Not exported
 * via the package surface; only the in-repo test file imports this.
 */
export function __resetObserverForTests(): void {
  channelsRef = null;
  registeredGroupsRef = null;
  observerEnabledFlag = false;
  observerChatJidOverride = undefined;
  lastReactionEmoji.clear();
  cachedGroupsDict = null;
  cachedFolderToJid = new Map();
  for (const w of watchdogs.values()) {
    clearInterval(w.intervalId);
  }
  watchdogs.clear();
  states.clear();
  if (selfTestTimer) {
    clearTimeout(selfTestTimer);
    selfTestTimer = null;
  }
  sawAnyQueryInput = false;
}

/**
 * Test seam — bypass the env / privacy-gate path of initObserver()
 * and mark the observer enabled with the supplied refs. The
 * production `initObserver` is awkward to drive in a unit test
 * because it both reads `OBSERVER_CHAT_JID` at module load time and
 * gates on `Channel.isPrivateChat`. Tests that exercise the parser
 * / state machine don't need that machinery.
 */
export function __enableObserverForTests(
  channels: Channel[],
  registeredGroups: () => Record<string, RegisteredGroup>,
  observerChatJid = 'tg:-100999000111',
): void {
  channelsRef = channels;
  registeredGroupsRef = registeredGroups;
  observerEnabledFlag = true;
  observerChatJidOverride = observerChatJid;
}

/** Test seam — read internal state for assertions. */
export function __getObserverInternalsForTests() {
  return { states, watchdogs, lastReactionEmoji };
}

// Self-test: if the observer is enabled but the agent-runner log
// format has drifted (or the orchestrator never spawned a query), we
// won't know — every onAgentLine call falls through silently.  Arm a
// one-shot watchdog after init: if no `Query input:` line lands
// within OBSERVER_SELF_TEST_MS, emit a single warning. That makes
// SDK-upgrade log-format breaks loud instead of letting the observer
// rot in place.
const OBSERVER_SELF_TEST_MS = 10 * 60 * 1000; // 10 minutes
let selfTestTimer: NodeJS.Timeout | null = null;
let sawAnyQueryInput = false;

function armSelfTest(): void {
  if (selfTestTimer) return;
  selfTestTimer = setTimeout(() => {
    if (!sawAnyQueryInput) {
      logger.warn(
        { graceMs: OBSERVER_SELF_TEST_MS },
        'Observer enabled but no `Query input:` lines parsed — agent-runner log format may have drifted; check container/agent-runner output',
      );
    }
    selfTestTimer = null;
  }, OBSERVER_SELF_TEST_MS);
  // Don't keep the event loop alive purely for this timer.
  selfTestTimer.unref?.();
}

interface QueryState {
  startTime: number;
  thinkingCount: number;
  toolCalls: string[]; // flattened list; we count duplicates at flush
  toolErrors: number;
  textSnippets: string[];
  stopReason?: string;
  // Engagement gate: stays false until the agent commits to a
  // user-visible action (text or tool_use). Thinking-only turns
  // (the agent reading a message and deciding to stay silent) leave
  // this false, and we suppress all reaction updates / watchdog
  // pings so the user sees no bot activity on irrelevant messages
  // in low-trust groups where every message spawns a container.
  // Once committed, every state transition through the rest of the
  // turn fires reactions normally.
  committed: boolean;
  // The message ID this query is processing, parsed from the
  // `target_message_id=` field on the Query input log line (resolved
  // by the agent-runner from the `<message id="N">` wrappers in the
  // prompt body). Reactions fire on THIS message specifically — a
  // fast-arriving newer message in the same chat would otherwise
  // race into the wrong target if we picked "the chat's latest
  // inbound" instead.
  targetMessageId?: string;
  // #289 — addressed-ness signal for this query, parsed from the
  // `addressed=true|false|-` field on the Query input log line.
  // The reaction ladder (🤔/⚡/✍/watchdog blinks/done) is suppressed
  // when this is `false` — the agent reasons about every inbound in
  // `requires_trigger=false` rooms but should not produce any visible
  // reaction trail on bystander chatter. `undefined` means "no signal
  // emitted by the agent-runner" and we treat that as "fall through
  // to the engagement gate alone" — keeps backward compatibility
  // with old log lines that didn't carry the field.
  addressed?: boolean;
}

// Keyed by container/group folder; one slot per concurrent query.
const states = new Map<string, QueryState>();

function newState(): QueryState {
  return {
    startTime: Date.now(),
    thinkingCount: 0,
    toolCalls: [],
    toolErrors: 0,
    textSnippets: [],
    committed: false,
  };
}

// Telegram caps messages at 4096 chars; leave headroom for any HTML the
// channel layer may add and for our own continuation marker.
const OBSERVER_CHUNK_SIZE = 3800;

// Exported for unit testing — internal callers continue to use the
// module-local symbol so the implementation stays inlinable.
export function chunkText(text: string, size: number): string[] {
  return _chunkText(text, size);
}

function _chunkText(text: string, size: number): string[] {
  if (text.length <= size) return [text];
  const chunks: string[] = [];
  let i = 0;
  while (i < text.length) {
    let end = Math.min(i + size, text.length);
    // Try to break at the nearest whitespace within the last 200 chars
    // so we don't cut a word in half. If no whitespace found, hard-cut.
    if (end < text.length) {
      const slack = text.lastIndexOf(' ', end);
      if (slack > i + size - 200) end = slack;
    }
    chunks.push(text.slice(i, end));
    i = end;
    while (i < text.length && text[i] === ' ') i++;
  }
  return chunks;
}

function send(text: string): void {
  const jid = getObserverChatJid();
  if (!jid || !channelsRef) return;
  const channel = channelsRef.find((c) => c.ownsJid(jid) && c.isConnected());
  if (!channel) {
    logger.warn({ jid }, 'Observer: no channel owns JID');
    return;
  }
  const parts = chunkText(text, OBSERVER_CHUNK_SIZE);
  // Send sequentially so chunks land in order. Failure on any chunk is
  // logged but does not abort the rest — partial visibility beats none.
  let chain: Promise<unknown> = Promise.resolve();
  parts.forEach((part, idx) => {
    const body =
      parts.length > 1 ? `${part} (${idx + 1}/${parts.length})` : part;
    chain = chain.then(() =>
      channel.sendMessage(jid, body).catch((err: unknown) => {
        logger.warn(
          { err, jid, chunk: idx + 1, of: parts.length },
          'Observer send failed',
        );
      }),
    );
  });
}

/**
 * Feed one stderr line from the agent-runner. Parses known patterns and
 * accumulates per-source state. Returns silently for unrecognized lines.
 */
export function onAgentLine(source: string, raw: string): void {
  if (!observerEnabled()) return;

  // Strip the "[agent-runner] " prefix if present — makes the regexes simpler.
  const line = raw.replace(/^\[agent-runner\]\s*/, '');

  // Query boundaries
  if (line.startsWith('Query input:')) {
    sawAnyQueryInput = true;
    const fresh = newState();
    // Read the explicit `target_message_id` field the agent-runner
    // emits on the Query input log line (see container/agent-runner/
    // src/index.ts). The agent-runner resolves this from the prompt
    // body before logging — observer.ts no longer has to recover it
    // from a length-capped prompt preview, which also avoided
    // logging raw prompt content per jbaruch/coding-policy:
    // no-secrets. `-` is the agent-runner's sentinel for "no inbound
    // id" (scheduled tasks, raw prompts) — treat it as undefined.
    const targetMatch = /target_message_id=([^,\s]+)/.exec(line);
    if (targetMatch && targetMatch[1] !== '-') {
      fresh.targetMessageId = targetMatch[1];
    }
    // #289 — addressed-ness signal. Sentinel `-` means the
    // agent-runner emitted "no signal" (legacy / scheduled paths);
    // leave fresh.addressed as undefined so the engagement-gate
    // alone governs reactions for backward compatibility.
    const addressedMatch = /addressed=(true|false|-)/.exec(line);
    if (addressedMatch && addressedMatch[1] !== '-') {
      fresh.addressed = addressedMatch[1] === 'true';
    }
    states.set(source, fresh);
    // Defuse any watchdog left over from a prior query that crashed
    // before emitting `Query done.` (SDK exception, agent-runner kill,
    // container OOM). Without this, the second query would see
    // `watchdogs.has(source) === true`, skip startWatchdog, and inherit
    // a stale `startedAt` / `pingsSent` — threshold pings would
    // misfire instantly. Pass undefined so the reset uses the
    // deterministic done emoji rather than the new query's not-yet-
    // established state.
    stopWatchdog(source);
    // A new query is starting. Don't pre-emptively send any
    // reaction here — the agent-runner's react-first hook fires 👀
    // from inside the container when the prompt is actually
    // submitted (see `decideReactFirst`); the first thinking/tool
    // event below will swap to 🤔/⚡ via the engagement-gated path.
    //
    // Don't arm the watchdog for scheduled tasks. Cron-driven queries
    // (SmartThings refresh, heartbeat, etc.) run in
    // maintenance containers but share the source folder with the user's
    // default container. Without this gate, a long cron task crossing 120s
    // fires "Still working — 120s in" into the user's chat — looking like
    // the user's last message is still being processed when actually the
    // user's query finished minutes ago and a separate cron is running.
    // The agent-runner emits an explicit `scheduled_task=true|false`
    // field on the Query input line so we don't have to grep the
    // prompt preview for `[SCHEDULED TASK` markers.
    const isScheduledTask = /scheduled_task=true/.test(line);
    // #289 — also skip the watchdog on non-addressed queries.
    // The blink emojis (🫡/🤓) and threshold pings ("Still working…")
    // are noise on bystander chatter the agent is just reasoning
    // about. `updateReaction` would no-op them anyway via the
    // addressed-gate above, but starting an interval just to have
    // every tick early-return wastes timers — gate at arm-time.
    if (!isScheduledTask && fresh.addressed !== false) {
      startWatchdog(source);
    }
    return;
  }

  const state = states.get(source);
  if (!state) return;

  // Thinking block — count AND live-stream the full content so you can
  // watch the agent's reasoning unfold in real time. The agent-runner emits
  // the full thinking text on a single log line (whitespace collapsed); the
  // chunker in send() splits anything over Telegram's 4096-char cap.
  const thinkingMatch = line.match(/^\[msg #\d+\] thinking="(.*)"$/);
  if (thinkingMatch) {
    state.thinkingCount++;
    send(`🧠 [${source}] ${thinkingMatch[1]}`);
    // Engagement gate: do NOT fire 🤔 reaction here. A thinking-only
    // turn ending in stop_reason=end_turn means the agent decided to
    // stay silent (irrelevant message, bad-actor disengage, etc.) —
    // showing 🤔 in that case lights up the user's chat with bot
    // activity for messages the bot is intentionally ignoring.
    // Reactions only fire once the agent commits via text/tool_use
    // below.
    if (state.committed) {
      updateReaction(source, '🤔');
    }
    return;
  }

  // Tool use
  const toolUse = line.match(/^\[msg #\d+\] tool_use=(\S+)/);
  if (toolUse) {
    // Normalize "mcp__onecli__gmail_search" → "gmail_search" for readability
    const name = toolUse[1].replace(/^mcp__[^_]+__/, '');
    state.toolCalls.push(name);
    // First non-thinking event marks engagement. Fire the
    // backlogged 🤔 so the user briefly sees the cycle, then the
    // tool emoji.
    const justCommitted = !state.committed;
    state.committed = true;
    if (justCommitted && state.thinkingCount > 0) {
      updateReaction(source, '🤔');
    }
    // mcp__nanoclaw__send_message means the agent is DELIVERING, not doing
    // more work — show the "composing" emoji instead of the "working"
    // emoji so the user sees progress: thinking → working → composing.
    //
    // IMPORTANT: these must all be in TELEGRAM_ALLOWED_REACTIONS in
    // src/channels/telegram.ts. Telegram limits bot reactions to a fixed
    // set; anything else silently falls back to 👍 and defeats the signal.
    // ⚡ is the closest "busy/working" emoji in the allowed set.
    updateReaction(
      source,
      toolUse[1] === 'mcp__nanoclaw__send_message' ? '✍' : '⚡',
    );
    return;
  }

  // Tool result — capture status + live-alert on errors
  const toolResult = line.match(
    /^\[msg #\d+\] tool_result id=\S+ (ok|error)(?: latency=(\d+)ms)?/,
  );
  if (toolResult) {
    if (toolResult[1] === 'error') {
      state.toolErrors++;
      // Live alert — unclipped line, truncated for Telegram sanity
      send(`❌ [${source}] ${line.slice(0, 800)}`);
    }
    return;
  }

  // Final user-facing text
  const textMatch = line.match(/^\[msg #\d+\] text="([^"]*)"/);
  if (textMatch) {
    state.textSnippets.push(textMatch[1]);
    // Text emission is a commit signal ONLY if there's user-visible
    // content. Text wrapped fully in `<internal>…</internal>` is the
    // agent's "I read this but I'm staying silent" pattern — the
    // host strips it before sending and counts it as a non-reply
    // for accounting purposes. Treating it as commitment fires
    // 🤝 (DONE_REACTION) on irrelevant messages, lighting up the
    // chat for things the bot is intentionally ignoring.
    const stripped = textMatch[1]
      .replace(/<internal>[\s\S]*?<\/internal>/g, '')
      .trim();
    if (stripped.length > 0) {
      const justCommitted = !state.committed;
      state.committed = true;
      if (justCommitted && state.thinkingCount > 0) {
        updateReaction(source, '🤔');
      }
    }
    return;
  }

  // Stop reason is on the "assistant blocks=..." header line
  const stopMatch = line.match(
    /^\[msg #\d+\] assistant blocks=\[[^\]]*\] stop=(\S+)/,
  );
  if (stopMatch) {
    state.stopReason = stopMatch[1];
    return;
  }

  // Query done — stop the liveness watchdog and flush summary
  if (line.startsWith('Query done.')) {
    // Pick the deterministic end-state emoji. ✍ if the agent ended on
    // a send_message (composed reply landed); otherwise fall through
    // to stopWatchdog's DONE_REACTION default. We deliberately don't
    // try to reproduce ⚡ — a watchdog blink could already have
    // overwritten that, and "tool fired" isn't a stable end-state.
    // toolCalls stores names with the `mcp__<server>__` prefix already
    // stripped (see normalization in the tool_use branch above).
    //
    // Engagement gate: if the turn never committed to a user-visible
    // action (silent thinking-only turn), don't set ANY done emoji.
    // Pass an explicit no-op skip so stopWatchdog clears its interval
    // without writing a reaction the user didn't earn through real
    // bot engagement.
    const lastTool = state.toolCalls[state.toolCalls.length - 1];
    const doneEmoji = !state.committed
      ? '__skip__'
      : lastTool === 'send_message'
        ? '✍'
        : undefined;
    stopWatchdog(source, doneEmoji);
    // Parse metrics emitted on the Query done. line by agent-runner.
    // Field name `wall_ms` (not `wall`) matches the explicit unit in
    // the agent-runner emit and avoids ambiguity with seconds.
    const wall = /wall_ms=(\d+)/.exec(line)?.[1];
    const tokIn = /tokens_in=(\d+)/.exec(line)?.[1];
    const tokOut = /tokens_out=(\d+)/.exec(line)?.[1];
    const cacheHit = /cache_hit_rate=([0-9.]+|n\/a)/.exec(line)?.[1];
    flushSummary(source, state, {
      wall: wall ? parseInt(wall, 10) : undefined,
      tokIn: tokIn ? parseInt(tokIn, 10) : undefined,
      tokOut: tokOut ? parseInt(tokOut, 10) : undefined,
      cacheHit,
    });
    // Drop reaction-state entries for this query so the map doesn't
    // grow unbounded over the lifetime of the orchestrator process.
    // Keys are composite `${chatJid}:${msgId}`; both the chat-jid
    // half (resolved from the source folder) and the msg-id half
    // (the query's targetMessageId) are known here, so we can purge
    // exactly the keys this query wrote without scanning every entry.
    const chatJid = folderToChatJid(source);
    if (chatJid && state.targetMessageId) {
      lastReactionEmoji.delete(`${chatJid}:${state.targetMessageId}`);
    }
    states.delete(source);
    return;
  }
}

function flushSummary(
  source: string,
  state: QueryState,
  metrics: {
    wall?: number;
    tokIn?: number;
    tokOut?: number;
    cacheHit?: string;
  },
): void {
  // Roll up tool calls: ["gcal_list", "gmail_search", "gmail_search"] → "gcal_list, gmail_search×2"
  const counts: Record<string, number> = {};
  for (const name of state.toolCalls) counts[name] = (counts[name] || 0) + 1;
  const toolLine =
    Object.keys(counts).length === 0
      ? '(none)'
      : Object.entries(counts)
          .map(([n, c]) => (c > 1 ? `${n}×${c}` : n))
          .join(', ');

  const wallSec = metrics.wall ? (metrics.wall / 1000).toFixed(1) : '?';
  const errPart = state.toolErrors > 0 ? ` | ❌ ${state.toolErrors} err` : '';
  const stopPart = state.stopReason ? ` stop=${state.stopReason}` : '';

  const lines = [
    `📊 [${source}]`,
    `🧠 ${state.thinkingCount} thinking | 🔧 ${state.toolCalls.length} tools: ${toolLine}${errPart}`,
    `⏱ ${wallSec}s | in=${metrics.tokIn ?? '?'} out=${metrics.tokOut ?? '?'} | cache=${metrics.cacheHit ?? '?'}%${stopPart}`,
  ];

  if (state.textSnippets.length > 0) {
    const last = state.textSnippets[state.textSnippets.length - 1];
    lines.push(`💬 "${last.slice(0, 160)}${last.length > 160 ? '…' : ''}"`);
  }

  send(lines.join('\n'));
}
