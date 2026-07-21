import { getMessageById } from '../db-messages.js';
import { logger } from '../logger.js';
import { NewMessage, RegisteredGroup } from '../types.js';
import { runGateChain, GateContext } from './index.js';

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
 * — running it is free (microseconds, $0) and short-circuits when a
 * deterministic match exists. The `requires_trigger=false` semantic
 * ("respond to all messages") is preserved because (a) when no patterns
 * match the trigger gate returns `pass` and the chain falls through to
 * the fail-open default, and (b) groups with no patterns at all get an
 * empty implicit chain.
 *
 * Path B (one-shot DB migration to set `containerConfig.gates =
 * ['trigger']`) is a future cleanup — the column stays for now.
 */
// Gate names removed from the registry entirely (the Stage-2 Haiku
// classifier was retired after the subscription-OAuth cutover). A
// persisted explicit `containerConfig.gates` row can still carry a
// removed name — the column is hand-editable JSON and survives code
// changes — and an unregistered name in the chain logs an error on
// every message in `runGateChain`. Filter them out here so stale rows
// keep working with the gates that still exist.
const REMOVED_GATE_NAMES: ReadonlySet<string> = new Set(['haiku-classifier']);

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
    chain = explicitGates.filter((g) => !REMOVED_GATE_NAMES.has(g));
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
 * `replyTo` so gates can inspect the reply target without re-parsing
 * the inline prefix.
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
