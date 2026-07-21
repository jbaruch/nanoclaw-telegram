/**
 * Pre-spawn eligibility gates (#754, #846).
 *
 * Some cadence skills only do useful work inside a bounded window, yet
 * the cadence registry fires them at a fixed high frequency (a windowed
 * skill firing every 2 minutes → ~30 spawns/hour). Off-window every fire
 * spawns a container just for a precheck that returns "nothing to do" —
 * the cost lives in the spawn, not the precheck. A pre-spawn gate lets
 * the host decide, WITHOUT spawning, whether a fire is eligible.
 *
 * The gate is keyed on the skill the task invokes (`parseTaskSkill`), so
 * it is source-agnostic — it applies whether the row is a declarative
 * `cadence-registry` fire or an imperative `schedule-task` one. Only
 * skills with a registered gate are gated; every other task spawns
 * unconditionally (the gate resolver returns `null`).
 *
 * Core owns only the MECHANISM (#846): the registry, the resolver, and
 * the shared fail-open/closed fs-errno narrowing helper. Gate POLICY —
 * which skill is gated, and what file it reads to rule on a fire —
 * belongs to the capability's own module under `src/host-plugins/`,
 * registered at startup via `registerSpawnGate`.
 */

export interface SpawnEligibility {
  /** `true` → spawn the container as normal; `false` → skip the spawn. */
  eligible: boolean;
  /** Human-readable rationale, surfaced in the skip log line. */
  reason: string;
}

/** A gate reads the group folder + current instant and rules on a fire. */
export type SpawnGate = (groupDir: string, now: Date) => SpawnEligibility;

/**
 * Filesystem errnos a gate treats as expected "file-state" conditions
 * (mapped to fail-open/closed). Anything else caught around a `fs` call is
 * a programming bug, not a file state, and must propagate — mirrors the
 * explicit-errno narrowing in `checkTaskEvidence` (`task-scheduler.ts`).
 * The narrowing helper is exported for gate implementations
 * (`src/host-plugins/`).
 */
const EXPECTED_FS_ERRNOS = new Set([
  'ENOENT',
  'EACCES',
  'EISDIR',
  'ENOTDIR',
  'ELOOP',
  'ENAMETOOLONG',
]);

export function isExpectedFsError(err: unknown): err is NodeJS.ErrnoException {
  const code = (err as NodeJS.ErrnoException).code;
  return (
    err instanceof Error &&
    typeof code === 'string' &&
    EXPECTED_FS_ERRNOS.has(code)
  );
}

/**
 * Registry of pre-spawn gates keyed by the exact skill identifier
 * `parseTaskSkill` returns (the value inside `Skill(skill: "…")`).
 */
const spawnGates = new Map<string, SpawnGate>();

/**
 * Register a pre-spawn gate for a skill (#846). Called by host-plugin
 * modules at startup — core never hard-codes a skill name here.
 * Duplicate names are a wiring bug (two modules claiming one skill's
 * gate) and fail loudly.
 */
export function registerSpawnGate(skillName: string, gate: SpawnGate): void {
  if (spawnGates.has(skillName)) {
    throw new Error(`Spawn gate already registered: ${skillName}`);
  }
  spawnGates.set(skillName, gate);
}

/**
 * Resolve and evaluate the pre-spawn gate for a task's skill.
 *
 * Returns `null` when the skill has no registered gate (→ spawn as
 * normal), otherwise the eligibility verdict. `groupDir` must be the
 * already-validated host path for the task's group folder.
 */
export function evaluateSpawnGate(
  skill: string | undefined,
  groupDir: string,
  now: Date,
): SpawnEligibility | null {
  if (!skill) return null;
  const gate = spawnGates.get(skill);
  if (!gate) return null;
  return gate(groupDir, now);
}
