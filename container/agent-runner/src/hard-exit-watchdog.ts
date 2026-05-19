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
    // Strip an inline `# ...` comment from an unquoted value before
    // validating, matching the host-side `parseSkillFrontmatter`
    // semantics in `src/cadence-registry.ts`. Without this,
    // `drain_timeout_ms: 180000 # 3 minutes` would fail the digit
    // regex and silently fall back to the default — exactly the
    // shape tile authors will naturally write when annotating a
    // non-obvious value.
    if (!value.startsWith('"') && !value.startsWith("'")) {
      const inlineCommentIdx = value.search(/\s+#/);
      if (inlineCommentIdx >= 0) {
        value = value.slice(0, inlineCommentIdx).trimEnd();
      }
    }
    if (
      value.length >= 2 &&
      ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'")))
    ) {
      value = value.slice(1, -1);
    }
    if (!/^\d+$/.test(value)) return undefined;
    const n = Number.parseInt(value, 10);
    if (n <= 0) return undefined;
    if (n > DRAIN_TIMEOUT_MS_MAX) return undefined;
    return n;
  }
  return undefined;
}

/**
 * Skill names are directory names under the container's skills
 * mount. Allow ASCII letters, digits, `_`, `-`, and the `__`
 * namespace separator the tile installer uses (`tessl__<name>`).
 * Rejecting anything else — slashes, dots, leading hyphens, NUL,
 * empty — defends against a prompt that smuggles `..` or an
 * absolute path into a `Skill(skill: "...")` invocation and tries
 * to walk the budget resolver into reading a SKILL.md outside the
 * mount. The character class is deliberately tighter than
 * "anything `path.join` would accept" so an attacker can't slip a
 * dotted segment past the regex.
 */
const SAFE_SKILL_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

/**
 * Resolve the effective idle budget for the current runQuery. Reads
 * the prompt's `Skill(skill: "...")` invocation, looks up the named
 * skill's SKILL.md under `skillsDir`, and returns the
 * `drain_timeout_ms` frontmatter override if valid. Falls back to
 * `defaultMs` on any miss — no skill invocation in the prompt, skill
 * name fails the safe-name regex, skill not installed, missing or
 * malformed frontmatter.
 *
 * Filesystem misses (`ENOENT` / `ENOTDIR`) fall back to the default
 * silently — a missing SKILL.md means "use the default", not "abort
 * the run". Other I/O errors (permission, malformed path that
 * survives the regex, etc.) are unexpected and propagate so the
 * operator can diagnose; the existing `runQuery` error-handling
 * surface in `index.ts` writes them to the SDK result diagnostic
 * channel rather than silently disabling the override.
 */
export function resolveDrainTimeoutMs(
  prompt: string,
  skillsDir: string,
  defaultMs: number = HARD_EXIT_IDLE_BUDGET_MS,
): number {
  const skillName = parseSkillNameFromPrompt(prompt);
  if (!skillName) return defaultMs;
  if (!SAFE_SKILL_NAME_RE.test(skillName)) return defaultMs;
  const skillPath = path.join(skillsDir, skillName, 'SKILL.md');
  // Defence-in-depth: even with the safe-name regex above, verify
  // the resolved path is still under `skillsDir`. A future relaxation
  // of the regex (or a `skillsDir` that itself contains a symlink)
  // could otherwise widen the read surface.
  const resolvedSkill = path.resolve(skillPath);
  const resolvedRoot = path.resolve(skillsDir) + path.sep;
  if (!resolvedSkill.startsWith(resolvedRoot)) return defaultMs;
  let content: string;
  try {
    content = fs.readFileSync(skillPath, 'utf-8');
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT' || code === 'ENOTDIR') return defaultMs;
    throw err;
  }
  const override = parseDrainTimeoutMsFromFrontmatter(content);
  return override ?? defaultMs;
}
