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
 *   - `reply`: `replyToMessageId` is set. The "reply to BOT" check is
 *     not done inside the gate — the gate is pure, has no DB
 *     reference, and the call site is responsible for already having
 *     consulted `isReplyToBot` when building `replyToMessageId`. See
 *     the wiring in src/index.ts: only bot-replies are surfaced into
 *     the GateContext as a non-empty `replyToMessageId`.
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
import type { GateContext, GateDecision, GateFn } from './index.js';
import type { TriggerPattern } from '../types.js';

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
      // The call site only populates replyToMessageId when the reply
      // points at a bot-emitted message — see src/index.ts wiring.
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

export const triggerGate: GateFn = (ctx: GateContext): GateDecision => {
  const cfg = ctx.triggerPatterns;
  if (!cfg || !cfg.patterns || cfg.patterns.length === 0) {
    return { decision: 'pass', reason: 'no trigger patterns configured' };
  }

  let sawEvaluatable = false;
  for (const p of cfg.patterns) {
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
