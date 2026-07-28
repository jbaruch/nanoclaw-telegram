/**
 * #890 — per-skill scheduled-task precheck timeout override.
 *
 * The precheck kill used to be a flat 30s `SCRIPT_TIMEOUT_MS` applied
 * to every skill regardless of what it does. That budget throttled
 * `jbaruch/nanoclaw-travel`'s `drive-engine` reconcile sweep, whose
 * cold pass runs past 30s (travel-side fix: nanoclaw-travel#211).
 *
 * Exactly two timeouts bound a precheck now, and this module owns the
 * first:
 *
 *   1. What the skill declares for itself — `precheck_timeout_ms` in
 *      its SKILL.md frontmatter. Read here, honoured verbatim, no
 *      ceiling imposed on top of it.
 *   2. The host's container kill — `containerConfig.timeout`, falling
 *      back to `MAINTENANCE_CONTAINER_TIMEOUT`, in
 *      `src/container-runner.ts`. Always present, per-group settable.
 *
 * A skill that declares nothing gets NO in-container timer: bound (2)
 * is the only one, which is what it was already doing to any precheck
 * that outlived the old global. The flat 30s that used to sit between
 * the two is gone — it was a third timeout nobody chose.
 *
 * A precheck streams no stdout while it runs, so its whole duration
 * counts against the host's inactivity timer. Declaring more here than
 * the group allows means the container is killed first — the host's
 * bound to enforce and the operator's to raise; this module never
 * re-asserts it.
 *
 * Resolution mirrors `drain_timeout_ms` (#589) — same `Skill(skill:
 * "...")` prompt lookup, shared via `skill-frontmatter.ts`. For every
 * cadence task the prompt is literally `Skill(skill: "<name>")`
 * (`src/cadence-registry.ts`).
 */

import {
  readFrontmatterScalar,
  readSkillMdForPrompt,
} from './skill-frontmatter.js';

/**
 * Largest delay Node's `setTimeout` can represent (Int32 max, ~24.8
 * days). Above this Node does not wait longer — it warns
 * `TimeoutOverflowWarning` and substitutes **1ms**, so an over-large
 * declaration fires almost immediately and kills the precheck at once.
 *
 * This is NOT a policy ceiling on how long a precheck may run — the
 * skill's declaration and the container kill remain the only two
 * bounds. It is the boundary past which the platform cannot express
 * the declared value at all, and where honouring the literal number
 * would do the exact opposite of what it says.
 */
const NODE_MAX_TIMER_MS = 2_147_483_647;

/**
 * Parse the `precheck_timeout_ms` scalar from a SKILL.md's leading
 * YAML frontmatter. Returns the declared value when it is a positive
 * integer Node can actually represent as a timer delay, or `undefined`
 * when the file has no frontmatter, no `precheck_timeout_ms` key, or a
 * value that isn't such an integer (`0`, a negative, `90s`, `90.5`, an
 * empty scalar, anything above `NODE_MAX_TIMER_MS`).
 *
 * An over-`NODE_MAX_TIMER_MS` value resolves to "no declaration", so
 * the precheck runs bounded by the container kill — the same place a
 * skill that declares nothing lands. Returning the literal number
 * instead would hand `setTimeout` a delay it silently rewrites to 1ms,
 * inverting the widest possible declaration into the narrowest.
 */
export function parsePrecheckTimeoutMsFromFrontmatter(
  content: string,
): number | undefined {
  const value = readFrontmatterScalar(content, 'precheck_timeout_ms');
  if (value === undefined) return undefined;
  if (!/^\d+$/.test(value)) return undefined;
  const n = Number.parseInt(value, 10);
  if (n <= 0) return undefined;
  if (n > NODE_MAX_TIMER_MS) return undefined;
  return n;
}

/**
 * Resolve the precheck budget for a scheduled task: the invoked
 * skill's `precheck_timeout_ms` declaration, or `undefined` when it
 * declares none. `undefined` means "arm no in-container timer" —
 * `runScript` leaves the precheck bounded solely by the host's
 * container kill.
 *
 * Resolves to `undefined` on every miss — no skill invocation in the
 * prompt (an ad-hoc task whose script isn't skill-owned), unsafe skill
 * name, skill not installed, missing or malformed frontmatter.
 */
export function resolvePrecheckTimeoutMs(
  prompt: string,
  skillsDir: string,
): number | undefined {
  const content = readSkillMdForPrompt(prompt, skillsDir);
  if (content === undefined) return undefined;
  return parsePrecheckTimeoutMsFromFrontmatter(content);
}
