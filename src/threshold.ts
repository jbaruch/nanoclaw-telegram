/**
 * Threshold formula for kill-auto-compaction (design doc §5,
 * `docs/proposals/kill-auto-compaction.md`):
 *
 *   threshold_warn = min(70% of context, context - 200K)
 *   threshold_nuke = min(80% of context, context - 100K)
 *
 * The percentage handles large-context models cleanly (Opus 4.7 1M:
 * warn at 700K, nuke at 800K). The headroom floor handles
 * smaller-context models where 70/80% would leave too little room to
 * write the checkpoint and reentry context.
 *
 * Per-group override: deferred. Until production data shows a group
 * needs a different value, both thresholds are derived purely from
 * MODEL_CONTEXT_WINDOW (config.ts). If a future group hits the seam
 * too often, add a `compaction_threshold_pct INTEGER` column to
 * `registered_groups` and override there.
 */

const WARN_HEADROOM_TOKENS = 200_000;
const NUKE_HEADROOM_TOKENS = 100_000;
const WARN_PERCENT = 0.7;
const NUKE_PERCENT = 0.8;

export interface Thresholds {
  /** Warn threshold in tokens. Crossing this surfaces an in-line note
   *  to the user that a reset is approaching. */
  warn: number;
  /** Nuke threshold in tokens. Crossing this fires the handshake
   *  (when ENABLE_THRESHOLD_NUKE=1) or just logs (when off). */
  nuke: number;
  /** Echo of the context window the thresholds were derived from, so
   *  callers can format diagnostics without re-reading config. */
  contextWindow: number;
}

/**
 * Compute warn/nuke thresholds for a given context-window size.
 *
 * Pure function — no env reads, no side effects. The orchestrator
 * passes `MODEL_CONTEXT_WINDOW` (from config.ts); tests pass
 * arbitrary values to verify the formula handles small contexts
 * (where the headroom floor dominates) and large contexts (where
 * the percentage dominates).
 */
export function computeThresholds(contextWindow: number): Thresholds {
  return {
    warn: Math.min(
      Math.floor(contextWindow * WARN_PERCENT),
      contextWindow - WARN_HEADROOM_TOKENS,
    ),
    nuke: Math.min(
      Math.floor(contextWindow * NUKE_PERCENT),
      contextWindow - NUKE_HEADROOM_TOKENS,
    ),
    contextWindow,
  };
}

export type ThresholdState = 'below_warn' | 'warn' | 'nuke';

/**
 * Classify a token-usage value against the warn/nuke thresholds.
 *
 * Caller invariant: `usedTokens` is the per-turn context size —
 * `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`
 * from the SDK `usage` payload. The bare `input_tokens` field is only
 * the delta added on this turn under prompt caching, so passing it
 * alone underreports context by 2–3 orders of magnitude on
 * cache-heavy turns (see #498).
 *
 * Returned state tells the orchestrator how to act (when
 * ENABLE_THRESHOLD_NUKE is on) or just what to label the log entry
 * (when off).
 */
export function classifyUsage(
  usedTokens: number,
  thresholds: Thresholds,
): ThresholdState {
  if (usedTokens >= thresholds.nuke) return 'nuke';
  if (usedTokens >= thresholds.warn) return 'warn';
  return 'below_warn';
}
