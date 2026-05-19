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
 */

import * as fs from 'fs';
import * as path from 'path';

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
 * Extract the first `Skill(skill: "...")` invocation name from a
 * prompt. Mirrors the host-side `parseTaskSkill` in
 * `src/task-scheduler.ts` so per-skill overrides resolve the same
 * way in both layers. The shape is fully enumerable — the literal
 * SDK skill-invocation syntax the orchestrator prepends — so a regex
 * is appropriate per `coding-policy: script-delegation`.
 */
export function parseSkillNameFromPrompt(prompt: string): string | undefined {
  const match = prompt.match(/Skill\(\s*skill:\s*["']([^"']+)["']/);
  return match?.[1];
}

/**
 * Parse the `drain_timeout_ms` scalar from a SKILL.md's leading YAML
 * frontmatter. Returns `undefined` if the file has no frontmatter, no
 * `drain_timeout_ms` key, or a value that doesn't parse as a positive
 * integer. Deliberately narrow — pulling js-yaml just for one scalar
 * would be unjustified weight; the host-side cadence-registry's
 * `parseSkillFrontmatter` is the same shape and the same justification.
 */
export function parseDrainTimeoutMsFromFrontmatter(
  content: string,
): number | undefined {
  // Trim a single leading BOM (U+FEFF) defensively — some editors
  // emit it on save. Matches the host-side `parseSkillFrontmatter` in
  // `src/cadence-registry.ts`.
  const stripped = content.replace(/^\uFEFF/, '');
  if (!stripped.startsWith('---\n') && !stripped.startsWith('---\r\n')) {
    return undefined;
  }
  const afterOpen = stripped.replace(/^---\r?\n/, '');
  const closeIdx = afterOpen.search(/^---\s*$/m);
  if (closeIdx < 0) return undefined;
  const body = afterOpen.slice(0, closeIdx);
  for (const rawLine of body.split(/\r?\n/)) {
    const line = rawLine.replace(/\s+$/, '');
    if (!line.trim() || line.trim().startsWith('#')) continue;
    const colonIdx = line.indexOf(':');
    if (colonIdx <= 0) continue;
    const key = line.slice(0, colonIdx).trim();
    if (key !== 'drain_timeout_ms') continue;
    let value = line.slice(colonIdx + 1).trim();
    if (
      value.length >= 2 &&
      ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'")))
    ) {
      value = value.slice(1, -1);
    }
    if (!/^\d+$/.test(value)) return undefined;
    const n = Number.parseInt(value, 10);
    return n > 0 ? n : undefined;
  }
  return undefined;
}

/**
 * Resolve the effective idle budget for the current runQuery. Reads
 * the prompt's `Skill(skill: "...")` invocation, looks up the named
 * skill's SKILL.md under `skillsDir`, and returns the
 * `drain_timeout_ms` frontmatter override if valid. Falls back to
 * `defaultMs` on any miss — no skill invocation in the prompt, skill
 * not installed, missing or malformed frontmatter, or I/O error.
 *
 * Read errors are swallowed deliberately: the watchdog is a last-
 * resort net and must never throw during budget resolution. A
 * missing SKILL.md means "use the default", not "abort the run".
 */
export function resolveDrainTimeoutMs(
  prompt: string,
  skillsDir: string,
  defaultMs: number = HARD_EXIT_IDLE_BUDGET_MS,
): number {
  const skillName = parseSkillNameFromPrompt(prompt);
  if (!skillName) return defaultMs;
  const skillPath = path.join(skillsDir, skillName, 'SKILL.md');
  let content: string;
  try {
    content = fs.readFileSync(skillPath, 'utf-8');
  } catch {
    return defaultMs;
  }
  const override = parseDrainTimeoutMsFromFrontmatter(content);
  return override ?? defaultMs;
}
