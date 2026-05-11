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
    const raw = m[1];

    // Strip a YAML inline comment (`  # ...`) from a bare/unquoted
    // value before further parsing. Matches the standard YAML semantic
    // and the convention in `cadence-registry.ts:parseSkillFrontmatter`.
    // We only strip comments outside of a `[...]` inline-list block —
    // a literal `#` inside brackets is preserved by the inline-list
    // branch's split-on-comma logic, which is sufficient since skill
    // names match `[a-zA-Z0-9_-]+` and `#` can't appear in a real name.
    let inline = raw;
    if (!inline.startsWith('[')) {
      const commentIdx = inline.search(/\s+#/);
      if (commentIdx >= 0) inline = inline.slice(0, commentIdx).trimEnd();
    } else {
      // Inline-list form: only strip a comment that comes AFTER the
      // closing `]`. A `#` before the close bracket is treated as part
      // of the (rejected) name and filtered by the name-shape check.
      const closeIdx = inline.lastIndexOf(']');
      if (closeIdx >= 0) {
        const tail = inline.slice(closeIdx + 1);
        const commentInTail = tail.search(/\s*#/);
        if (commentInTail >= 0) {
          inline = inline.slice(0, closeIdx + 1 + commentInTail).trimEnd();
        }
      }
    }

    // Inline list: requires: [a, b, c]
    if (inline.startsWith('[') && inline.endsWith(']')) {
      const inner = inline.slice(1, -1).trim();
      // Literal `[]` (or `[   ]` whitespace-only inner) is the
      // author's explicit "never load" form — distinct from a list
      // with content that happened to parse to zero valid entries.
      if (!inner) return [];
      const candidates = inner.split(',').map((s) => stripQuotes(s.trim()));
      const valid = candidates.filter(
        (s) => s.length > 0 && SKILL_NAME_PATTERN.test(s),
      );
      // If the author wrote a bracket list with content (e.g.
      // `[bad#name]` or `[,,]`) but every entry was rejected by the
      // name-shape check or by the empty-entry filter, return null
      // (= no constraint, load unconditionally) rather than collapsing
      // to the explicit-empty-list semantic. The "ambiguity → load"
      // safety contract applies here: the author's intent was clearly
      // NOT `requires: []` literal — they just wrote something we
      // can't make sense of. Distinguishing these prevents a typo'd
      // requires list from silently parking a rule the same way an
      // explicit empty does.
      if (valid.length === 0) return null;
      return valid;
    }

    // Single bare value: requires: foo
    if (inline.length > 0) {
      const name = stripQuotes(inline);
      // Bare value must match the documented skill-name shape; an
      // unparseable value (e.g. accidental punctuation, a typo'd YAML
      // list) falls back to null = load unconditionally rather than
      // returning an unmatchable scalar that silently filters the rule.
      return SKILL_NAME_PATTERN.test(name) ? [name] : null;
    }

    // Block list: requires: <empty>, followed by `  - foo` lines.
    // If NO `- foo` lines follow (operator wrote `requires:` with no
    // value and no block items), fall back to null per the
    // "ambiguity → load unconditionally" safety contract. An author
    // wanting the never-load semantic uses the explicit `requires: []`
    // inline-empty form.
    const items: string[] = [];
    for (let j = i + 1; j < lines.length; j++) {
      const sub = lines[j];
      const subMatch = sub.match(/^\s*-\s+(.+?)\s*$/);
      if (!subMatch) break;
      const name = stripQuotes(subMatch[1]);
      if (SKILL_NAME_PATTERN.test(name)) items.push(name);
    }
    return items.length === 0 ? null : items;
  }
  return null;
}

// Skill names per the convention extracted in `skill-dep-closure.ts`'s
// `Skill()` invocation regex — kebab-case, snake_case, or alphanumeric.
const SKILL_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/;

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
