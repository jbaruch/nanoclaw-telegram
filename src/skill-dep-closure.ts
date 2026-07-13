/**
 * #544 — Static dep extraction for the maintenance skill blocklist.
 *
 * Pre-#544 the blocklist was a flat name set: every member's prompt
 * was excluded from the maintenance container's agent context (token
 * savings, intended) AND its scripts were too (collateral, fixed by
 * #544a). The remaining gap was nested invocation: a non-blocklisted
 * skill A whose SKILL.md says `Skill(skill: "tessl__B")` cannot
 * actually invoke B if B is blocklisted — the agent loads A's prompt,
 * tries to invoke B, and the SDK fails with "Unknown skill: B".
 *
 * Reference incident: 2026-05-10 `wiki-lint` (not in blocklist) failed
 * with "Unknown skill: wiki" because `wiki` (built-in) was in the
 * blocklist and `wiki-lint` invokes it via `Skill(skill: "wiki")`.
 *
 * The fix: when computing the EFFECTIVE blocklist for a spawn, walk
 * the full transitive `Skill()` reference graph starting from every
 * non-blocklisted skill. Any reachable skill is exempted from the
 * blocklist for that spawn — its prompt loads alongside the loaded
 * surface so the nested invocation can resolve.
 *
 * The skill-authoring rule mandates typed `Skill(skill: "name")`
 * calls (no prose references), which makes the call shape statically
 * extractable with a fixed regex. Tile skills are invoked as
 * `tessl__<name>` (the agent-runner mounts them at
 * `.claude/skills/tessl__<name>/`); built-in skills (`wiki`,
 * `agent-browser`, `status`) are invoked bare. The regex handles both
 * by treating the `tessl__` prefix as optional.
 *
 * Over-inclusion is the safe failure mode: if the regex matches a
 * fenced code-block example or a string that happens to look like an
 * invocation, the worst case is one extra blocklist exemption — the
 * agent loads a skill it doesn't end up using. Under-inclusion is
 * the unsafe failure mode (exactly the wiki-lint outage), so the
 * regex stays permissive on whitespace and quote style.
 */

// Captures both `Skill(skill: "tessl__name")` and `Skill(skill: "name")`
// (built-ins). The non-capturing `(?:tessl__)?` swallows the prefix when
// present so the captured group is always the bare blocklist name.
// Allows single OR double quotes, whitespace anywhere except inside
// the name itself. The `[^)]*` suffix tolerates trailing named params
// after the skill name (`Skill(skill: "wiki", args: "lint")`) — the
// name is already captured, so anything up to the closing paren is
// skipped. Without it, every `args:`-bearing invocation is invisible
// to the closure and the rescue silently never fires (#652).
const SKILL_INVOCATION_PATTERN =
  /Skill\(\s*skill:\s*["'](?:tessl__)?([a-zA-Z0-9_-]+)["'][^)]*\)/g;

// #441/#439 — cross-skill dependencies that are NOT `Skill()` invocations.
// A skill can depend on another skill's *files* (not its prompt) by shelling
// out to a script at that skill's mount path — e.g. `morning-brief`'s
// `resolve-reminder-schedule.py` runs
// `.../skills/tessl__scheduler-timezone/scripts/compute-schedule-value.py`
// as a subprocess. The `Skill()` regex never sees that edge, so pre-#441 the
// closure left `scheduler-timezone` blocklisted in the maintenance container
// and the subprocess call hit a missing mount path (the 2026-07-12 morning-
// brief Step-9 failure). This pattern captures the depended-on skill from any
// `tessl__<name>/` mount-path reference — the trailing `/` anchors it to an
// actual path INTO the skill's dir, so a bare `tessl__<name>` mention in a
// doc-comment (no path) does not falsely rescue it. Extraction runs over the
// caller's SKILL.md AND its scripts/references text (see
// computeEffectiveSkillContextForSpawn), the only surfaces a mount-path
// reference appears on. Over-inclusion stays the safe failure mode.
const MOUNT_PATH_DEP_PATTERN = /tessl__([a-zA-Z0-9_-]+)\//g;

function matchAll(pattern: RegExp, content: string, into: Set<string>): void {
  // Build a fresh RegExp per call — a module-level shared instance with the
  // `g` flag carries lastIndex state across calls and would skip matches on
  // the second invocation.
  const re = new RegExp(pattern.source, 'g');
  let match: RegExpExecArray | null;
  while ((match = re.exec(content)) !== null) {
    into.add(match[1]);
  }
}

/**
 * Pure: extract every `Skill(skill: "...")` invocation target from a
 * SKILL.md content string. Returns the set of bare skill names
 * (without the `tessl__` prefix) so the caller can compare against a
 * blocklist whose entries are bare names.
 */
export function extractSkillDeps(skillMdContent: string): Set<string> {
  const deps = new Set<string>();
  matchAll(SKILL_INVOCATION_PATTERN, skillMdContent, deps);
  return deps;
}

/**
 * Pure: extract every `tessl__<name>/` mount-path dependency from a content
 * string (SKILL.md, a script, or a reference doc). These are cross-skill
 * *file* dependencies — one skill shelling out to another skill's script —
 * that the `Skill()` extractor cannot see (#441). Returns bare skill names.
 */
export function extractMountPathDeps(content: string): Set<string> {
  const deps = new Set<string>();
  matchAll(MOUNT_PATH_DEP_PATTERN, content, deps);
  return deps;
}

/**
 * Pure: compute the effective skill context for a spawn given the
 * original blocklist and a map of every skill's SKILL.md content.
 *
 * Returns both:
 *   - `effectiveBlocklist` — the subset of `originalBlocklist` that
 *     remains after exempting any skill reachable from a live root.
 *     This is what callers feed back into the per-tile copy loop.
 *   - `reachableSkills` — the names the agent's loaded skill graph
 *     reaches: live roots + transitively-referenced exemptions. Note
 *     that this also includes names referenced via `Skill()` from a
 *     loaded skill but absent from `skillSources` (typo, retired
 *     skill) — those names DON'T have a prompt to load, but they're
 *     reachable in the graph sense. The agent's runtime "Unknown
 *     skill" error surfaces the absence at invocation time.
 *     #552 uses this for rule `requires:` filtering — a rule loads
 *     iff at least one of its declared gating skills is in this set.
 *     A dead `requires: [retired-name]` reference would therefore
 *     match a `Skill("retired-name")` from a loaded skill; the
 *     publish-time tile lint catches that case so dead refs don't
 *     accumulate.
 *
 * Algorithm: BFS starting from every NON-blocklisted skill (the
 * "roots" — these load into the agent's context unconditionally).
 * Walk `Skill()` references from each root; any skill reachable via
 * the reference graph is exempted from the blocklist for this spawn,
 * because its prompt MUST be available for nested invocation to
 * resolve. Continues transitively — A → B → C all get exempted if A
 * is a root and B is referenced by A and C is referenced by B.
 *
 * Returns fresh sets; the input `originalBlocklist` is not mutated.
 *
 * Edge cases:
 *   - A skill referencing itself (`Skill(skill: "self")`) is a no-op
 *     (already in the reachable set; the BFS doesn't re-enqueue).
 *   - A reference to a name that doesn't exist in `skillSources`
 *     (typo, retired skill name) is added to `reachableSkills` (the
 *     agent WILL try to load it and fail loudly at invocation time;
 *     the closure mirrors that intent) but not enqueued for further
 *     BFS, since there's no content to walk. For #552 `requires:`
 *     filtering this means a rule declaring a dead skill name would
 *     still match — caught by the publish-time lint, not at runtime.
 *   - Empty `skillSources` produces an effective blocklist equal to
 *     the original (no roots, no closure) and an empty reachable set.
 */
export interface EffectiveSkillContext {
  effectiveBlocklist: Set<string>;
  reachableSkills: Set<string>;
}

export function computeEffectiveSkillContext(
  originalBlocklist: ReadonlySet<string>,
  skillSources: ReadonlyMap<string, string>,
): EffectiveSkillContext {
  const reachable = new Set<string>();
  const queue: string[] = [];

  // Roots: every skill present in `skillSources` that isn't in the
  // blocklist. These are the prompts the agent will load by default.
  for (const skill of skillSources.keys()) {
    if (!originalBlocklist.has(skill)) {
      reachable.add(skill);
      queue.push(skill);
    }
  }

  // BFS along the reference graph. Deps come from two surfaces: typed
  // `Skill(skill: "...")` invocations (nested prompt loads) AND
  // `tessl__<name>/` mount-path references (a skill shelling out to another
  // skill's script — #441). Both mean "that skill's files must be present in
  // this container", so both rescue the target from the blocklist.
  while (queue.length > 0) {
    const skill = queue.shift() as string;
    const md = skillSources.get(skill);
    if (md === undefined) continue;
    const refs = extractSkillDeps(md);
    for (const ref of extractMountPathDeps(md)) refs.add(ref);
    for (const ref of refs) {
      if (!reachable.has(ref)) {
        reachable.add(ref);
        // Only enqueue if we have its SKILL.md — otherwise there's
        // nothing to walk further (and the runtime will surface the
        // missing skill at invocation time).
        if (skillSources.has(ref)) queue.push(ref);
      }
    }
  }

  const effective = new Set<string>();
  for (const skill of originalBlocklist) {
    if (!reachable.has(skill)) effective.add(skill);
  }
  return { effectiveBlocklist: effective, reachableSkills: reachable };
}

/**
 * Back-compat wrapper around {@link computeEffectiveSkillContext} for
 * callers that only need the blocklist piece. New callers should
 * prefer the context-returning form so the positive presence set is
 * available for downstream filters (rule `requires:` per #552).
 */
export function computeEffectiveBlocklist(
  originalBlocklist: ReadonlySet<string>,
  skillSources: ReadonlyMap<string, string>,
): Set<string> {
  return computeEffectiveSkillContext(originalBlocklist, skillSources)
    .effectiveBlocklist;
}
