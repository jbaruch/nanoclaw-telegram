/**
 * #552 — Conditional rule loading by `requires:` frontmatter.
 *
 * Pre-#552 every rule in every installed tile was unconditionally
 * concatenated into the per-group `/workspace/group/.tessl/RULES.md`.
 * The 932-line baseline (28 always-on rules across two tiles)
 * established by issue #552 is the bloat this filter exists to trim:
 * a rule that only matters when a specific skill is loaded should
 * not occupy context in spawns where that skill is absent.
 *
 * The mechanism mirrors `skill-dep-closure.ts` on the rule side. A
 * rule declares its gating skill(s) via an optional `requires:`
 * frontmatter field. The container-runner's rule-assembly loop reads
 * each rule's frontmatter, extracts `requires:`, and includes the
 * rule only if at least one listed skill is in the spawn's effective
 * skill set (any-of semantics). Rules without `requires:` load
 * unconditionally — that's the current behavior and the safe default.
 *
 * The filter is additive: existing rules without `requires:` work
 * unchanged, and rule authors opt in per-rule when the gating
 * relationship is clean (e.g. `no-orphan-tasks` requires
 * `schedule-task`, `messages-db-schema` requires `query-history`).
 *
 * Over-inclusion is the safe failure mode (a rule loads when its
 * gating skill is absent — same cost as today's unconditional
 * behavior, no semantic regression). Under-inclusion is the unsafe
 * failure mode (a rule fails to load when it should — the agent
 * makes the mistake the rule exists to prevent). The parser is
 * therefore permissive on whitespace / quoting and only treats a
 * well-formed `requires:` field as a constraint; any ambiguity falls
 * back to "no constraint declared, load unconditionally".
 */

/**
 * Pure: parse a rule .md file's leading YAML frontmatter and return
 * the `requires:` value as a list of skill names. Returns `null` if
 * the rule has no `requires:` declaration (= unconditional load).
 * Returns an empty array if `requires:` is present but empty (e.g.
 * `requires: []`) — semantically distinct from `null` (the author
 * declared a constraint, just an empty one; no skill satisfies, the
 * rule never loads).
 *
 * Supported shapes:
 *   - Inline list:   `requires: [skill-a, skill-b]`
 *   - Single value:  `requires: skill-a`
 *   - Block list:
 *       requires:
 *         - skill-a
 *         - skill-b
 *
 * Skill names match `[a-zA-Z0-9_-]+`; single or double quotes around
 * a name are stripped. Anything that doesn't parse cleanly returns
 * `null` (safe: "no constraint, load unconditionally") rather than
 * throwing — a malformed frontmatter shouldn't crash a spawn.
 */
export function parseRequiresFrontmatter(content: string): string[] | null {
  // Strip a single leading BOM (U+FEFF, written as the Unicode
  // escape rather than a literal so eslint's no-irregular-whitespace
  // rule doesn't trip on the source character). Matches the
  // convention in `cadence-registry.ts:parseSkillFrontmatter`.
  const stripped = content.replace(/^\uFEFF/, '');
  if (!stripped.startsWith('---\n') && !stripped.startsWith('---\r\n')) {
    return null;
  }
  const afterOpen = stripped.replace(/^---\r?\n/, '');
  const closeIdx = afterOpen.search(/^---\s*$/m);
  if (closeIdx < 0) return null;
  const body = afterOpen.slice(0, closeIdx);

  const lines = body.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Match the field name at the start of the line (no leading
    // whitespace — top-level frontmatter key). A nested `requires:`
    // inside a sub-mapping wouldn't gate-load anyway since we don't
    // support nested mappings here.
    const m = line.match(/^requires:\s*(.*?)\s*$/);
    if (!m) continue;
    const inline = m[1];

    // Inline list: requires: [a, b, c]
    if (inline.startsWith('[') && inline.endsWith(']')) {
      const inner = inline.slice(1, -1).trim();
      if (!inner) return [];
      return inner
        .split(',')
        .map((s) => stripQuotes(s.trim()))
        .filter((s) => s.length > 0);
    }

    // Single bare value: requires: foo
    if (inline.length > 0) {
      return [stripQuotes(inline)];
    }

    // Block list: requires: <empty>, followed by `  - foo` lines
    const items: string[] = [];
    for (let j = i + 1; j < lines.length; j++) {
      const sub = lines[j];
      const subMatch = sub.match(/^\s*-\s+(.+?)\s*$/);
      if (!subMatch) {
        // First non-list line ends the block. Empty lines also end it
        // — block-list YAML doesn't allow gaps.
        break;
      }
      items.push(stripQuotes(subMatch[1]));
    }
    return items;
  }
  return null;
}

/**
 * Strip a single layer of surrounding single or double quotes from a
 * scalar value. Matches `parseSkillFrontmatter` in cadence-registry —
 * we don't try to be YAML-perfect, just handle the common quoting
 * shapes a rule author might use.
 */
function stripQuotes(v: string): string {
  if (
    v.length >= 2 &&
    ((v.startsWith('"') && v.endsWith('"')) ||
      (v.startsWith("'") && v.endsWith("'")))
  ) {
    return v.slice(1, -1);
  }
  return v;
}

/**
 * Pure: decide whether a rule should be included in a spawn's
 * aggregated RULES.md given the rule's content and the set of
 * skills present in that spawn's effective skill context.
 *
 * Returns:
 *   - `{ include: true, reason: 'no-requires' }` when the rule has
 *     no `requires:` field (current default, always loads).
 *   - `{ include: true, reason: 'gating-skill-present' }` when at
 *     least one `requires:` entry resolves to a skill in
 *     `presentSkills`.
 *   - `{ include: false, reason: 'no-gating-skill' }` when
 *     `requires:` is non-empty but no entry resolves.
 *   - `{ include: false, reason: 'empty-requires' }` when the rule
 *     declared `requires: []` — author explicitly said "never load".
 *     Distinct from `no-requires` (= no declaration at all).
 *
 * The `requires` field on the return value carries the parsed list
 * (or `null` for the no-requires case) so callers can log filtered
 * rules with their declared gating skills for operator debugging.
 */
export type RuleFilterResult =
  | { include: true; reason: 'no-requires'; requires: null }
  | { include: true; reason: 'gating-skill-present'; requires: string[] }
  | { include: false; reason: 'no-gating-skill'; requires: string[] }
  | { include: false; reason: 'empty-requires'; requires: string[] };

export function shouldIncludeRule(
  ruleContent: string,
  presentSkills: ReadonlySet<string>,
): RuleFilterResult {
  const requires = parseRequiresFrontmatter(ruleContent);
  if (requires === null) {
    return { include: true, reason: 'no-requires', requires: null };
  }
  if (requires.length === 0) {
    return { include: false, reason: 'empty-requires', requires };
  }
  if (requires.some((s) => presentSkills.has(s))) {
    return { include: true, reason: 'gating-skill-present', requires };
  }
  return { include: false, reason: 'no-gating-skill', requires };
}
