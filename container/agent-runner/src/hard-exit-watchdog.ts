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
 *
 * #589 — three reinforcements after morning-brief continued tripping
 * the post-#545 watchdog mid-compose (Claude's thinking-then-send
 * turn after a heavy 33KB data fetch emits no intervening SDK event
 * for 30+ seconds; the activity-aware re-arm only sees one block of
 * silence and trips):
 *
 *   1. Default budget raised 30s → 90s. The SIGKILL deploy-kill
 *      cascade (#249/#250) still bounds truly-stuck cases at the
 *      host-side hard timeout, so trading 60s of grace for correct
 *      behavior on heavy maintenance turns is cheap.
 *   2. Per-skill override via SKILL.md `drain_timeout_ms` frontmatter.
 *      Known-heavy skills (morning-brief, soul-searching) can declare
 *      a longer budget without bumping the global default. Resolved
 *      at runQuery start from the prompt's `Skill(skill: "...")`
 *      invocation; missing or malformed values fall back.
 *   3. In-flight `tool_use` re-arms unconditionally. Between the
 *      `tool_use` SDK event and its matching `tool_result`, the agent
 *      is provably alive (the model emitted a tool call; tools are
 *      blocking). A Bash that legitimately runs 60s+ now keeps the
 *      watchdog re-armed instead of tripping on the silent idle
 *      window between those two events.
 *
 * #589 (reopened) — the budget bumps above (30s → activity-aware →
 * 90s) never closed the gap for ONE-SHOT MAINTENANCE spawns, because
 * the idle clock only advances on SDK message events. The window
 * between a `tool_result` and the NEXT assistant turn — an LLM
 * inference request in flight — emits no SDK event, and a compose turn
 * that stalls on a slow / failing LLM call (credential-proxy failover)
 * blows past any fixed budget before `send_message` ever fires. The
 * in-container watchdog is a tighter, redundant duplicate of the
 * host-side maintenance inactivity timeout
 * (`MAINTENANCE_CONTAINER_TIMEOUT`, src/container-runner.ts), so for
 * maintenance it is disabled (see `shouldArmHardExitWatchdog`): the
 * host timer is the single bound. The per-skill `drain_timeout_ms`
 * override and the in-flight `tool_use` re-arm still apply to
 * interactive / default sessions, where `_close` means the multi-turn
 * conversation genuinely idled and a long post-close compose is not
 * expected.
 */

import {
  readFrontmatterScalar,
  readSkillMdForPrompt,
} from './skill-frontmatter.js';

// Re-exported from `skill-frontmatter.ts` (#890), which now owns the
// prompt→SKILL.md resolution shared by all three per-skill overrides.
// Kept exported here so existing importers of the #589 surface keep
// resolving.
export {
  parseSkillNameFromPrompt,
  SAFE_SKILL_NAME_RE,
} from './skill-frontmatter.js';

export const HARD_EXIT_IDLE_BUDGET_MS = 90_000;

export type WatchdogDecision =
  | { action: 'exit'; idleMs: number }
  | { action: 'rearm'; rearmInMs: number; idleMs: number };

/**
 * Pure decision function for the post-close hard-exit watchdog.
 *
 * Given `now`, the timestamp of the last observed SDK event, and the
 * count of `tool_use` invocations still awaiting their `tool_result`,
 * returns either:
 *   - `{ action: 'exit' }` — idle window >= budget AND no in-flight
 *     tool calls; time to bail.
 *   - `{ action: 'rearm', rearmInMs }` — activity within the budget,
 *     OR a tool call is in flight. The same idle budget applies to
 *     the next check.
 *
 * `pendingToolCallCount > 0` re-arms with the full budget. The agent
 * emitted a `tool_use` event the SDK is blocking on; the matching
 * `tool_result` will arrive whenever the tool returns and will
 * refresh `lastSdkActivityAt`. The deploy-kill SIGKILL cascade
 * (#249/#250) catches a tool that genuinely never returns.
 *
 * Boundary: `idleMs == budgetMs` exits (>= comparison) when no tools
 * are in flight. A timer fired exactly at the budget boundary should
 * kill, not re-arm at 0ms (which would create a tight loop).
 */
export function decideHardExitWatchdog(
  now: number,
  lastSdkActivityAt: number,
  budgetMs: number = HARD_EXIT_IDLE_BUDGET_MS,
  pendingToolCallCount: number = 0,
): WatchdogDecision {
  const idleMs = now - lastSdkActivityAt;
  if (pendingToolCallCount > 0) {
    return { action: 'rearm', rearmInMs: budgetMs, idleMs };
  }
  if (idleMs >= budgetMs) {
    return { action: 'exit', idleMs };
  }
  return { action: 'rearm', rearmInMs: budgetMs - idleMs, idleMs };
}

/**
 * #589 (reopened) — decide whether to arm the in-container post-close
 * hard-exit watchdog at all. Maintenance one-shot spawns defer to the
 * host-side `MAINTENANCE_CONTAINER_TIMEOUT` inactivity bound: the
 * in-container watchdog is a tighter duplicate whose post-close idle
 * budget became the entire compose window (close flips early when no
 * follow-up IPC is expected), so a compose turn stalled on an LLM /
 * proxy blip tripped it before `send_message` fired. Interactive /
 * default sessions keep the watchdog — there `_close` means the
 * conversation genuinely idled and a long post-close compose is not
 * expected, so growing idleness is a real stuck-iterator signal.
 */
export function shouldArmHardExitWatchdog(
  isMaintenanceSession: boolean,
): boolean {
  return !isMaintenanceSession;
}

/**
 * Upper bound on `drain_timeout_ms` overrides. Node's `setTimeout`
 * clamps delays > `2_147_483_647` (Int32 max) to 1ms, which would
 * make the watchdog re-arm in a tight loop instead of waiting the
 * declared budget — silently disabling the override. Capping well
 * below the Int32 cliff also rejects implausibly-large values that
 * are almost certainly a tile-author typo (10× the intended budget,
 * a unit confusion `min` vs `ms`, etc.). 10 minutes is generous —
 * the heaviest known maintenance compose is morning-brief at ~150s
 * — and any genuine need for a longer budget should land here as a
 * deliberate constant bump, not as a tile-author one-off.
 */
export const DRAIN_TIMEOUT_MS_MAX = 600_000;

/**
 * Parse the `drain_timeout_ms` scalar from a SKILL.md's leading YAML
 * frontmatter. Returns `undefined` if the file has no frontmatter, no
 * `drain_timeout_ms` key, or a value that doesn't parse as a positive
 * integer within `(0, DRAIN_TIMEOUT_MS_MAX]`. Deliberately narrow —
 * pulling js-yaml just for one scalar would be unjustified weight;
 * the host-side cadence-registry's `parseSkillFrontmatter` is the
 * same shape and the same justification.
 */
export function parseDrainTimeoutMsFromFrontmatter(
  content: string,
): number | undefined {
  const value = readFrontmatterScalar(content, 'drain_timeout_ms');
  if (value === undefined) return undefined;
  if (!/^\d+$/.test(value)) return undefined;
  const n = Number.parseInt(value, 10);
  if (n <= 0) return undefined;
  if (n > DRAIN_TIMEOUT_MS_MAX) return undefined;
  return n;
}

/**
 * Resolve the effective idle budget for the current runQuery. Reads
 * the prompt's `Skill(skill: "...")` invocation, looks up the named
 * skill's SKILL.md under `skillsDir`, and returns the
 * `drain_timeout_ms` frontmatter override if valid. Falls back to
 * `defaultMs` on any miss — no skill invocation in the prompt, skill
 * name fails the safe-name regex, skill not installed, missing or
 * malformed frontmatter. Resolution and its ENOENT/ENOTDIR fallback
 * live in `readSkillMdForPrompt` (`skill-frontmatter.ts`).
 */
export function resolveDrainTimeoutMs(
  prompt: string,
  skillsDir: string,
  defaultMs: number = HARD_EXIT_IDLE_BUDGET_MS,
): number {
  const content = readSkillMdForPrompt(prompt, skillsDir);
  if (content === undefined) return defaultMs;
  return parseDrainTimeoutMsFromFrontmatter(content) ?? defaultMs;
}
