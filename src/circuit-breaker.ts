/**
 * Per-group circuit breaker: pauses a group after repeated consecutive
 * agent failures so a wedged group stops burning credits, then
 * auto-resumes after a cooldown. State is keyed by `group.folder` and
 * lives here — out of the message loop — so the loop (check + record)
 * and the `chat_status` IPC (read-only status) share one source of
 * truth. Extracted from `src/index.ts` as #749 seam-3 prep.
 *
 * The breaker deliberately lives here rather than in `GroupQueue`: it is
 * keyed on `group.folder` and driven by message-loop bookkeeping, not by
 * queue lifecycle.
 */
const MAX_CONSECUTIVE_FAILURES = 5;
const CIRCUIT_BREAKER_COOLDOWN_MS = 30 * 60 * 1000; // 30 minutes

/** Cooldown length in minutes, for caller-side log / notify messages. */
export const CIRCUIT_BREAKER_COOLDOWN_MINUTES =
  CIRCUIT_BREAKER_COOLDOWN_MS / 60_000;

const consecutiveFailures: Record<string, number> = {};
const circuitBreakerUntil: Record<string, number> = {};

export type BreakerCheck = 'skip' | 'resumed' | 'ok';

/**
 * Pre-spawn gate for the message loop. Mutating: when a group's cooldown
 * has expired this resets its state and returns `'resumed'` so the caller
 * logs the resume; while still in cooldown it returns `'skip'` (caller
 * skips the group); otherwise `'ok'`.
 */
export function checkCircuitBreaker(folder: string): BreakerCheck {
  const until = circuitBreakerUntil[folder];
  if (until) {
    if (Date.now() < until) return 'skip';
    // Cooldown expired — reset and let the group try again.
    delete circuitBreakerUntil[folder];
    consecutiveFailures[folder] = 0;
    return 'resumed';
  }
  return 'ok';
}

/**
 * Read-only breaker status for the `chat_status` IPC. Never mutates, so a
 * status query cannot reset a live cooldown.
 */
export function isBreakerActive(folder: string): boolean {
  const until = circuitBreakerUntil[folder];
  return !!until && Date.now() < until;
}

/**
 * Record one agent failure. Returns the new consecutive-failure count and
 * whether this failure tripped the breaker (armed the cooldown) so the
 * caller can log + notify the main group.
 */
export function recordFailure(folder: string): {
  tripped: boolean;
  failures: number;
} {
  const failures = (consecutiveFailures[folder] || 0) + 1;
  consecutiveFailures[folder] = failures;
  if (failures >= MAX_CONSECUTIVE_FAILURES) {
    circuitBreakerUntil[folder] = Date.now() + CIRCUIT_BREAKER_COOLDOWN_MS;
    return { tripped: true, failures };
  }
  return { tripped: false, failures };
}

/** Reset a group's failure counter after a successful run. */
export function recordSuccess(folder: string): void {
  consecutiveFailures[folder] = 0;
}
