import { ContainerOutput } from './container-runner.js';
import { logger } from './logger.js';
import { classifyUsage, ThresholdState, Thresholds } from './threshold.js';

/**
 * Per-turn `usage` payload shape. Re-exported from `ContainerOutput` so
 * the helper has a stable input type independent of where the call
 * site sits — the inbound path (`src/index.ts`) and the scheduled-task
 * path (`src/task-scheduler.ts`) both receive the same SDK shape.
 */
export type SessionUsage = NonNullable<ContainerOutput['usage']>;

/**
 * Context passed alongside each `usage` payload. `group` and
 * `thresholds` are required so the log line carries the discriminator
 * (`group:`) and the classification context (`threshold_state`,
 * `threshold_warn`, `threshold_nuke`). `session` is optional because
 * scheduled-task fires never persist a sessionId (#193) — pass
 * `undefined` and the field reflects that. `extra` is the seam for
 * call-site-specific tags (e.g. `taskId` / `scheduleType` / `taskSkill`
 * on the scheduled-task variant) so a single `grep "session_tokens"`
 * over host-logs surfaces both code paths' lines and analyses can
 * bucket by recurring-task identity.
 */
export interface SessionTokensContext {
  group: string;
  session?: string;
  thresholds: Thresholds;
  extra?: Record<string, unknown>;
}

/**
 * Emit one `session_tokens*` log line per turn-`usage` payload.
 *
 * Same fields and same threshold-state-based log-key classification
 * on both code paths — extracted from the inbound block at
 * `src/index.ts` (issue #349) so the scheduled-task path can call
 * the same helper from `runTask`'s `onOutput`. Returns the classified
 * state so callers can drive subsequent threshold-cross handling
 * (e.g. the kill-auto-compaction nuke / checkpoint write); `null`
 * when `usage` is undefined (no-op).
 *
 * Log key/level mapping is preserved verbatim from the pre-extract
 * site — the trailing-state suffix on the log key lets a `grep -E
 * "session_tokens(_warn|_nuke)?"` cover all three at once or pick the
 * rare events directly:
 *   - `below_warn` → `logger.info(..., 'session_tokens')`
 *   - `warn`       → `logger.warn(..., 'session_tokens_warn')`
 *   - `nuke`       → `logger.error(..., 'session_tokens_nuke')`
 *
 * `extra` is spread last so a caller-supplied tag CAN override a
 * built-in field — kept this way to avoid silently dropping a
 * caller's tag if it ever happens to collide. Real callers don't
 * collide.
 */
export function emitSessionTokens(
  usage: SessionUsage | undefined,
  ctx: SessionTokensContext,
): ThresholdState | null {
  if (!usage) return null;
  const state = classifyUsage(usage.input_tokens, ctx.thresholds);
  const logFields = {
    group: ctx.group,
    session: ctx.session,
    input_tokens: usage.input_tokens,
    output_tokens: usage.output_tokens,
    cache_read: usage.cache_read_input_tokens,
    cache_creation: usage.cache_creation_input_tokens,
    percent: Number(
      ((usage.input_tokens / ctx.thresholds.contextWindow) * 100).toFixed(1),
    ),
    threshold_state: state,
    threshold_warn: ctx.thresholds.warn,
    threshold_nuke: ctx.thresholds.nuke,
    ...ctx.extra,
  };
  if (state === 'nuke') {
    logger.error(logFields, 'session_tokens_nuke');
  } else if (state === 'warn') {
    logger.warn(logFields, 'session_tokens_warn');
  } else {
    logger.info(logFields, 'session_tokens');
  }
  return state;
}
