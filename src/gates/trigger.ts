/**
 * Built-in `trigger` gate.
 *
 * Replaces the legacy `requires_trigger` boolean check by walking the
 * group's `TriggerPatternConfig` (#81) and returning `allow` on the
 * first matching pattern.
 *
 * Pattern kinds handled in v1:
 *   - `keyword`: literal string, word-boundary match (mirrors
 *     `buildTriggerPattern` from src/config.ts so legacy semantics are
 *     preserved bit-for-bit during the transition).
 *   - `mention`: matches `@<pattern>` substring in message text OR an
 *     exact entry in `ctx.message.mentions[]` (channel-native list,
 *     when present).
 *   - `reply`: matches when the inbound message is a reply to a
 *     message the assistant itself sent. The gate consults
 *     `ctx.message.replyTo.isAssistant` (structured reply metadata
 *     populated by the call site). Replies to PEER bots or to humans
 *     do NOT match — the trigger semantic is specifically "reply to
 *     this assistant" (#107). Legacy `replyToMessageId` is also
 *     accepted for backward compat with callers that haven't been
 *     migrated to the structured form.
 *
 * `kind: 'sender_tier'` and `kind: 'regex'` are *unevaluatable* in v1
 * (reserved for #82 — sandboxed regex evaluator and the sender-tier
 * table are not yet wired). Unknown future kinds are also treated as
 * unevaluatable so that a forward-looking config entry never
 * black-holes traffic.
 *
 * Three-valued matcher semantics:
 *   - any pattern returns `match`        → gate decides `allow`
 *   - all patterns are `unevaluatable`   → gate decides `pass`
 *     (defers to the next gate / fail-open default)
 *   - at least one evaluatable kind, none matched → gate decides `deny`
 *
 * That means a config with **only** `sender_tier` or `regex` entries
 * is a true no-op pass-through. Mixed configs (e.g. `sender_tier` +
 * `keyword`) evaluate the implemented kinds normally — only the
 * evaluatable kinds count toward the deny accumulator.
 */
import type { GateContext, GateDecision } from './index.js';
import type { TriggerPattern } from '../types.js';
import { ASSISTANT_NAME, ASSISTANT_USERNAME } from '../config.js';

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Build the same word-boundary regex used by
 * `buildTriggerPattern` in src/config.ts. Keep them in lock-step —
 * any divergence here surfaces as a behaviour change for groups
 * still on the legacy path.
 */
function buildKeywordRegex(keyword: string): RegExp {
  return new RegExp(`(?:^|\\s)${escapeRegex(keyword.trim())}\\b`, 'i');
}

function matchKeyword(text: string, pattern: string): boolean {
  if (!pattern.trim()) return false;
  return buildKeywordRegex(pattern).test(text);
}

function matchMention(
  text: string,
  pattern: string,
  mentions: string[] | undefined,
): boolean {
  const normalized = pattern.startsWith('@') ? pattern.slice(1) : pattern;
  if (!normalized.trim()) return false;
  if (mentions && mentions.some((m) => m === normalized || m === pattern)) {
    return true;
  }
  // Plain substring with explicit `@` prefix. `\b` after `@<word>` —
  // word boundary keeps `@andy` from matching `@andybot` (which would
  // be a separate username).
  const re = new RegExp(`(^|\\s)@${escapeRegex(normalized)}\\b`, 'i');
  return re.test(text);
}

/**
 * Three-valued result distinguishes "evaluated and didn't match"
 * from "kind not implemented yet (or unknown)". Without this
 * distinction, a config consisting solely of forward-looking
 * `sender_tier`/`regex` entries would have every pattern report
 * `false` and the gate would deny — silently black-holing traffic.
 */
type MatchResult = 'match' | 'no-match' | 'unevaluatable';

function matchPattern(p: TriggerPattern, ctx: GateContext): MatchResult {
  switch (p.kind) {
    case 'keyword':
      return matchKeyword(ctx.message.text, p.pattern) ? 'match' : 'no-match';
    case 'mention':
      return matchMention(ctx.message.text, p.pattern, ctx.message.mentions)
        ? 'match'
        : 'no-match';
    case 'reply':
      // Match iff the reply target was sent by THIS assistant.
      // Replies to peer bots / humans don't fire this kind — that
      // would let any reply-prefix in a busy multi-bot group light up
      // the trigger gate (#107). Legacy `replyToMessageId` is accepted
      // for backward compat: the v1 wiring populated it only when the
      // reply pointed at a bot-emitted message of THIS assistant.
      if (ctx.message.replyTo) {
        return ctx.message.replyTo.isAssistant ? 'match' : 'no-match';
      }
      return ctx.message.replyToMessageId ? 'match' : 'no-match';
    case 'regex':
      // TODO(#82?): evaluate once a sandboxed runner exists. Today
      // we'd risk DoS via catastrophic backtracking on owner-supplied
      // patterns, so the matcher reports unevaluatable rather than
      // forcing a deny on configs that contain only regex entries.
      return 'unevaluatable';
    case 'sender_tier':
      // TODO(#82?): wired up by the self-improvement loop / sender
      // tier table. Until that lands, `sender_tier` entries are
      // unevaluatable — a config consisting solely of `sender_tier`
      // patterns yields gate decision `pass`, deferring to the next
      // gate or fail-open default.
      return 'unevaluatable';
    default:
      // Unknown future kind — fail-safe. Treat as unevaluatable so
      // forward-compat config entries don't deny traffic outright.
      return 'unevaluatable';
  }
}

/**
 * Build the synthetic identity patterns evaluated before any
 * operator-configured patterns. These are derived on every gate
 * invocation from `ASSISTANT_NAME` and `ASSISTANT_USERNAME` and are
 * NOT stored in the DB — they exist purely to short-circuit
 * direct-identity references at Stage 1 (microseconds, zero API
 * cost) instead of letting them fall through to Stage 2 Haiku.
 *
 * Operator-configured identity patterns (e.g. `@LoMBot`) remain in
 * the DB for backward compatibility. They match alongside the
 * synthetic identity patterns; the duplicate is harmless. Future
 * cleanup can strip operator-set identity patterns from group
 * configs once we're confident the auto-injection is bulletproof.
 */
function buildSyntheticIdentityPatterns(): TriggerPattern[] {
  const now: TriggerPattern[] = [];
  if (ASSISTANT_USERNAME && ASSISTANT_USERNAME.trim()) {
    now.push({
      pattern: ASSISTANT_USERNAME,
      kind: 'mention',
      source: 'owner-set',
      precision: 0,
      sample_count: 0,
      last_matched_at: null,
      last_updated_at: null,
    });
  }
  if (ASSISTANT_NAME && ASSISTANT_NAME.trim()) {
    now.push({
      pattern: ASSISTANT_NAME,
      kind: 'keyword',
      source: 'owner-set',
      precision: 0,
      sample_count: 0,
      last_matched_at: null,
      last_updated_at: null,
    });
  }
  return now;
}

// Annotated with the concrete sync return type rather than the broader
// `GateFn` (which widened to `GateDecision | Promise<GateDecision>` for
// Stage 2 async gates). Tests call `triggerGate(ctx)` directly and
// expect a sync `GateDecision`. Assigning into the registry via
// `registerGate('trigger', triggerGate)` still satisfies `GateFn` —
// the narrower sync signature is a subtype of the union.
export const triggerGate = (ctx: GateContext): GateDecision => {
  const cfg = ctx.triggerPatterns;
  const operatorPatterns = cfg?.patterns ?? [];

  // Synthetic identity patterns evaluated FIRST. Built per-call so
  // they pick up any test-time override of ASSISTANT_NAME /
  // ASSISTANT_USERNAME (the constants are imported once at module
  // load — but in practice the env is fixed for the host process
  // lifetime, so the perf cost is negligible).
  const syntheticPatterns = buildSyntheticIdentityPatterns();

  // Synthetic patterns short-circuit on match but are intentionally
  // NOT counted toward the deny accumulator on no-match. They're
  // auxiliary positive triggers — a non-match here must leave the
  // operator-pattern semantics (including the "all unevaluatable →
  // pass" rule) untouched.
  for (const p of syntheticPatterns) {
    const r = matchPattern(p, ctx);
    if (r === 'match') {
      return {
        decision: 'allow',
        reason: `assistant identity match (auto): kind=${p.kind} pattern=${p.pattern}`,
      };
    }
  }

  // No operator config AND no synthetic match — preserve the
  // pre-existing pass-through semantics for groups that opted out of
  // trigger patterns entirely.
  if (operatorPatterns.length === 0) {
    return { decision: 'pass', reason: 'no trigger patterns configured' };
  }

  let sawEvaluatable = false;
  for (const p of operatorPatterns) {
    const r = matchPattern(p, ctx);
    if (r === 'match') {
      return {
        decision: 'allow',
        reason: `trigger pattern matched: kind=${p.kind} pattern=${p.pattern}`,
      };
    }
    if (r === 'no-match') {
      sawEvaluatable = true;
    }
    // 'unevaluatable' — ignore for the deny accumulator.
  }
  if (sawEvaluatable) {
    return {
      decision: 'deny',
      reason: 'no trigger pattern matched',
    };
  }
  return {
    decision: 'pass',
    reason: 'no evaluatable patterns (only future-kind entries)',
  };
};
