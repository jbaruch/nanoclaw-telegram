/**
 * #545 — Activity-aware hard-exit watchdog (#461 follow-up).
 *
 * After the agent-runner observes the `_close` IPC sentinel and calls
 * `stream.end()`, the SDK iterator is supposed to drain promptly so
 * `main()` can return and `process.exit(0)` fires naturally. If it
 * hangs (open keepalive, lingering Promise, pending tool_result it
 * expects to never arrive), the container would stay alive against
 * the host's hard timeout.
 *
 * Pre-#545 the watchdog was a flat 30s budget from close detection.
 * That killed legitimately-working scheduled tasks (morning-brief,
 * soul-searching, entertainment-sync) whose entire compose phase
 * happens AFTER `_close` is detected — for one-shot scheduled spawns
 * there's no follow-up IPC message expected, so close flips early in
 * the run, and the 30s post-close budget became the entire
 * compose-to-send window.
 *
 * Post-#545 the budget measures idleness (time since last SDK event)
 * rather than wall-clock from close. A stuck SDK iterator emits no
 * events — its idle window grows monotonically and trips the budget.
 * A working agent emits assistant turns, tool_use, tool_result events
 * every few seconds, each resets the activity timestamp, and the
 * watchdog re-arms with the remaining idle budget.
 */

export const HARD_EXIT_IDLE_BUDGET_MS = 30_000;

export type WatchdogDecision =
  | { action: 'exit'; idleMs: number }
  | { action: 'rearm'; rearmInMs: number; idleMs: number };

/**
 * Pure decision function for the post-close hard-exit watchdog.
 *
 * Given `now` and the timestamp of the last observed SDK event,
 * returns either:
 *   - `{ action: 'exit' }` — idle window >= budget, time to bail.
 *   - `{ action: 'rearm', rearmInMs: number }` — activity within
 *     the budget, re-schedule the timer to fire when the idle
 *     window would next elapse (`budget - idleMs`). The same idle
 *     budget applies to the next check; chatty agents that emit
 *     events at <= budget cadence keep re-arming and never trip,
 *     while a stuck SDK eventually idles past the threshold.
 *
 * Boundary: `idleMs == budgetMs` exits (>= comparison). A timer
 * fired exactly at the budget boundary should kill, not re-arm at
 * 0ms (which would create a tight loop).
 */
export function decideHardExitWatchdog(
  now: number,
  lastSdkActivityAt: number,
  budgetMs: number = HARD_EXIT_IDLE_BUDGET_MS,
): WatchdogDecision {
  const idleMs = now - lastSdkActivityAt;
  if (idleMs >= budgetMs) {
    return { action: 'exit', idleMs };
  }
  return { action: 'rearm', rearmInMs: budgetMs - idleMs, idleMs };
}
