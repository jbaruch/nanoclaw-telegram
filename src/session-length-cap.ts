/**
 * Session-length cap (#413). Cumulative thresholds — a session can
 * stay below the per-turn `ENABLE_THRESHOLD_NUKE` percentage for
 * hours while still accumulating past the cap defined here.
 *
 * Pure logic only — no DB, no logger. State persistence lives in
 * `src/db.ts` (`recordSessionTurn`, `markSessionForReset`,
 * `consumeSessionReset`). Integration lives in `src/index.ts`
 * `runAgent`. This module is the seam between the two so the
 * threshold formula and the handoff-prefix builder can be unit-tested
 * without touching SQLite or the message loop.
 *
 * Why a separate module rather than inlining in runAgent: the
 * "should we mark this session for reset?" call is invoked on every
 * assistant turn (hot path) and the "build handoff prefix" call is
 * invoked exactly once per reset (cold path). Both have well-defined
 * inputs and outputs and benefit from focused tests with fixed data.
 */

/**
 * Cumulative thresholds for the session-length cap. Resolved from
 * env at startup in `src/config.ts` (`SESSION_TOKEN_CAP`,
 * `SESSION_TURN_CAP`). A non-positive value disables that
 * specific threshold without disabling the other.
 */
export interface SessionLengthCaps {
  /** Sum of `usage.input_tokens` across all turns. <= 0 disables. */
  tokenCap: number;
  /** Number of assistant turns. <= 0 disables. */
  turnCap: number;
}

/**
 * Snapshot of a session's accumulated state at the moment of the
 * threshold check. Sourced from the `session_length_state` row.
 */
export interface SessionLengthSnapshot {
  totalInputTokens: number;
  turnCount: number;
}

/**
 * Verdict returned by `shouldMarkForReset`. The `reason` discriminator
 * lets the caller log a precise cause without recomputing the
 * comparisons.
 */
export type ResetVerdict =
  | { reset: false }
  | {
      reset: true;
      reason: 'token_cap' | 'turn_cap';
      observed: number;
      cap: number;
    };

/**
 * Decide whether the cumulative session state crosses a configured
 * cap. Token cap takes precedence when both fire simultaneously —
 * arbitrary but deterministic, and the log line carries `observed`
 * + `cap` either way.
 *
 * Caps <= 0 are skipped (disabled). Both caps disabled → never
 * reset.
 */
export function shouldMarkForReset(
  snapshot: SessionLengthSnapshot,
  caps: SessionLengthCaps,
): ResetVerdict {
  if (caps.tokenCap > 0 && snapshot.totalInputTokens >= caps.tokenCap) {
    return {
      reset: true,
      reason: 'token_cap',
      observed: snapshot.totalInputTokens,
      cap: caps.tokenCap,
    };
  }
  if (caps.turnCap > 0 && snapshot.turnCount >= caps.turnCap) {
    return {
      reset: true,
      reason: 'turn_cap',
      observed: snapshot.turnCount,
      cap: caps.turnCap,
    };
  }
  return { reset: false };
}

/**
 * Inputs for building the brief context handoff. The orchestrator
 * pulls these from the chat's recent message history at reset time
 * (last user message + last assistant reply). The point is
 * **continuity, not completeness** — the handoff prefix is a 1–2
 * sentence prompt prefix the new session sees on its very first
 * turn so a follow-up reference doesn't fail.
 */
export interface HandoffInputs {
  /** Most recent assistant turn's text content. May be empty. */
  lastAssistantText?: string;
  /** Most recent inbound user message text. May be empty. */
  lastUserText?: string;
  /**
   * Display name for the assistant in the handoff prefix
   * (`<name>'s last reply: ...`). Driven by `ASSISTANT_NAME` at the
   * call site; defaults to a neutral label when unset so the prefix
   * doesn't lie in non-default deployments.
   */
  assistantName?: string;
}

const HANDOFF_MAX_CHARS = 400;
const DEFAULT_ASSISTANT_LABEL = 'Assistant';

/**
 * Build the brief context-handoff prefix injected into the new
 * session's initial prompt.
 *
 * Per the issue body and task spec: do NOT synthesize from scratch.
 * Pull from the most recent assistant turn or last user instruction
 * verbatim and bound the length. The receiving agent sees a short
 * `<session-handoff>` block describing what the prior session was
 * working on.
 *
 * Truncation is simple character-cap with an ellipsis, not sentence
 * detection — sentence boundaries in chat text are noisy enough
 * (inline code, URLs, multi-line) that a regex would mis-classify
 * more often than help. The cap of 400 characters is well under any
 * channel's outbound limit and well under the SDK's prompt budget;
 * tuning is cheap if a future case shows it's too tight.
 *
 * Returns null when both inputs are empty/whitespace-only — caller
 * skips the handoff prefix entirely in that case so a fresh session
 * doesn't get a meaningless empty `<session-handoff>` block.
 */
export function buildHandoffPrefix(inputs: HandoffInputs): string | null {
  const assistant = (inputs.lastAssistantText ?? '').trim();
  const user = (inputs.lastUserText ?? '').trim();
  if (!assistant && !user) return null;

  const label = (inputs.assistantName ?? '').trim() || DEFAULT_ASSISTANT_LABEL;

  const lines: string[] = [];
  if (user) lines.push(`User just before reset: ${truncate(user)}`);
  if (assistant) lines.push(`${label}'s last reply: ${truncate(assistant)}`);

  return [
    '<session-handoff>',
    'The prior session was reset to keep context bounded. Continue the same line of work — no need to recap.',
    ...lines,
    '</session-handoff>',
    '',
  ].join('\n');
}

function truncate(text: string): string {
  // Collapse internal whitespace so a multi-line snippet doesn't
  // spend its budget on blank lines. \s covers tabs/newlines/etc.
  const collapsed = text.replace(/\s+/g, ' ').trim();
  if (collapsed.length <= HANDOFF_MAX_CHARS) return collapsed;
  return collapsed.slice(0, HANDOFF_MAX_CHARS - 1) + '…';
}

/**
 * One-line user-facing notification. Sent in-band via the channel
 * router BEFORE the container spawn for the post-reset turn so the
 * user sees "session reset" ahead of the agent's first reply, not
 * after. Intentionally short — operators reading a chat scrollback
 * should be able to spot it without parsing.
 *
 * The reason discriminator surfaces token vs. turn cap so an
 * operator who wants to retune one cap can grep for the right
 * line. The cap value is included so the line is self-describing
 * even when log lines aren't to hand.
 */
export function buildResetNotification(
  reason: 'token_cap' | 'turn_cap',
  cap: number,
): string {
  if (reason === 'token_cap') {
    return `Session reset: cumulative input tokens crossed ${cap.toLocaleString('en-US')} — starting fresh to keep context bounded.`;
  }
  return `Session reset: turn count crossed ${cap} — starting fresh to keep context bounded.`;
}
