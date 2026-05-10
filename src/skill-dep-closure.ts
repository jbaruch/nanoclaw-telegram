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
// the name itself.
const SKILL_INVOCATION_PATTERN =
  /Skill\(\s*skill:\s*["'](?:tessl__)?([a-zA-Z0-9_-]+)["']\s*\)/g;

/**
 * Pure: extract every `Skill(skill: "...")` invocation target from a
 * SKILL.md content string. Returns the set of bare skill names
 * (without the `tessl__` prefix) so the caller can compare against a
 * blocklist whose entries are bare names.
 */
export function extractSkillDeps(skillMdContent: string): Set<string> {
  const deps = new Set<string>();
  // Build a fresh RegExp per call — a module-level shared instance
  // with the `g` flag carries lastIndex state across calls and would
  // skip matches on the second invocation.
  const re = new RegExp(SKILL_INVOCATION_PATTERN.source, 'g');
  let match: RegExpExecArray | null;
  while ((match = re.exec(skillMdContent)) !== null) {
    deps.add(match[1]);
  }
  return deps;
}

/**
 * Pure: compute the effective blocklist for a spawn given the
 * original blocklist and a map of every skill's SKILL.md content.
 *
 * Algorithm: BFS starting from every NON-blocklisted skill (the
 * "roots" — these load into the agent's context unconditionally).
 * Walk `Skill()` references from each root; any skill reachable via
 * the reference graph is exempted from the blocklist for this spawn,
 * because its prompt MUST be available for nested invocation to
 * resolve. Continues transitively — A → B → C all get exempted if A
 * is a root and B is referenced by A and C is referenced by B.
 *
 * Returns a new set; the input `originalBlocklist` is not mutated.
 *
 * Edge cases:
 *   - A skill referencing itself (`Skill(skill: "self")`) is a no-op
 *     (already in the reachable set; the BFS doesn't re-enqueue).
 *   - A reference to a name that doesn't exist in `skillSources`
 *     (typo, retired skill name) is silently dropped — the BFS
 *     can't traverse what it can't read. The agent's runtime
 *     "Unknown skill" error surfaces this case at invocation time;
 *     this helper doesn't try to validate references.
 *   - Empty `skillSources` produces an effective blocklist equal to
 *     the original (no roots, no closure).
 */
export function computeEffectiveBlocklist(
  originalBlocklist: ReadonlySet<string>,
  skillSources: ReadonlyMap<string, string>,
): Set<string> {
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

  // BFS along the reference graph.
  while (queue.length > 0) {
    const skill = queue.shift() as string;
    const md = skillSources.get(skill);
    if (md === undefined) continue;
    for (const ref of extractSkillDeps(md)) {
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
  return effective;
}
