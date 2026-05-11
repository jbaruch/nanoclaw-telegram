import { describe, it, expect } from 'vitest';

import {
  parseRequiresFrontmatter,
  shouldIncludeRule,
} from './rule-requires-filter.js';

describe('parseRequiresFrontmatter (#552)', () => {
  // The parser's contract is asymmetric: parsing failures return
  // `null` ("no constraint, load unconditionally") so a malformed
  // rule never silently disappears from a spawn. Well-formed but
  // empty `requires: []` is a different shape — the author
  // explicitly said "no gating skill satisfies", which means the
  // rule never loads.

  it('returns null when the rule has no frontmatter at all', () => {
    expect(parseRequiresFrontmatter('# Just a Rule\n\nNo header.')).toBeNull();
  });

  it('returns null when frontmatter exists but has no requires field', () => {
    const md = '---\nalwaysApply: true\n---\n\n# Rule\n\nBody.\n';
    expect(parseRequiresFrontmatter(md)).toBeNull();
  });

  it('parses an inline list with multiple skills', () => {
    const md =
      '---\nalwaysApply: true\nrequires: [schedule-task, manage-tasks]\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual([
      'schedule-task',
      'manage-tasks',
    ]);
  });

  it('parses an inline list with a single skill', () => {
    const md =
      '---\nalwaysApply: true\nrequires: [schedule-task]\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual(['schedule-task']);
  });

  it('parses an inline empty list (semantic: never load)', () => {
    // `requires: []` is the author explicitly declaring no gating
    // skill satisfies. `shouldIncludeRule` treats this as
    // "never include" — distinct from `null` (no declaration).
    const md = '---\nalwaysApply: true\nrequires: []\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual([]);
  });

  it('parses a single bare value (no brackets)', () => {
    const md =
      '---\nalwaysApply: true\nrequires: schedule-task\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual(['schedule-task']);
  });

  it('parses a block list (dash-prefixed lines)', () => {
    const md = `---
alwaysApply: true
requires:
  - schedule-task
  - manage-tasks
---

# Rule
`;
    expect(parseRequiresFrontmatter(md)).toEqual([
      'schedule-task',
      'manage-tasks',
    ]);
  });

  it('strips double quotes from list entries', () => {
    const md =
      '---\nrequires: ["schedule-task", "manage-tasks"]\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual([
      'schedule-task',
      'manage-tasks',
    ]);
  });

  it('strips single quotes from list entries', () => {
    const md =
      "---\nrequires: ['schedule-task', 'manage-tasks']\n---\n\n# Rule\n";
    expect(parseRequiresFrontmatter(md)).toEqual([
      'schedule-task',
      'manage-tasks',
    ]);
  });

  it('handles whitespace inside an inline list', () => {
    const md =
      '---\nrequires: [  schedule-task ,  manage-tasks  ]\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual([
      'schedule-task',
      'manage-tasks',
    ]);
  });

  it('handles kebab-case and snake_case skill names', () => {
    const md = '---\nrequires: [task-tz-sync, some_skill]\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual([
      'task-tz-sync',
      'some_skill',
    ]);
  });

  it('returns null for an unterminated frontmatter block', () => {
    // Missing closing `---`. Treat as malformed; safe default is
    // "no constraint declared" so the rule still loads.
    const md = '---\nrequires: [foo]\n\n# Rule body without close\n';
    expect(parseRequiresFrontmatter(md)).toBeNull();
  });

  it('tolerates CRLF line endings', () => {
    const md = '---\r\nrequires: [foo]\r\n---\r\n\r\n# Rule\r\n';
    expect(parseRequiresFrontmatter(md)).toEqual(['foo']);
  });

  it('block-list mode stops at the first non-list line', () => {
    // The block-list terminates as soon as a non-`- ` line appears,
    // mirroring YAML's indentation-sensitive block semantics. The
    // trailing `otherField:` line is ignored for `requires:` purposes.
    const md = `---
requires:
  - foo
  - bar
otherField: x
---

# Rule
`;
    expect(parseRequiresFrontmatter(md)).toEqual(['foo', 'bar']);
  });

  it('strips an inline YAML comment after a bare value', () => {
    // `requires: schedule-task # optional` — the comment must not be
    // part of the parsed value. Without comment-stripping the bare
    // branch would return `['schedule-task # optional']` which never
    // matches a real skill and silently filters the rule. Matches the
    // standard YAML semantic and `cadence-registry.ts` convention.
    const md =
      '---\nrequires: schedule-task # optional, lifted out for #552\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual(['schedule-task']);
  });

  it('strips an inline YAML comment after an inline list', () => {
    // `requires: [foo] # comment` — same shape as the bare case but
    // the bracket-form parse must also tolerate trailing comments.
    const md = '---\nrequires: [foo, bar] # any-of\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual(['foo', 'bar']);
  });

  it('returns null for a bare value that fails the skill-name shape', () => {
    // Punctuation outside `[a-zA-Z0-9_-]` (e.g. dots, slashes,
    // accidentally-pasted text) shouldn't produce an unmatchable
    // scalar that silently filters the rule. Safety contract:
    // ambiguity → load unconditionally.
    const md = '---\nrequires: not.a.real.skill/name\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toBeNull();
  });

  it('filters bracket-list entries that fail the skill-name shape (keeps valid ones)', () => {
    // Mixed shape: one valid name, one with an embedded `#` (illegal
    // per the skill-name pattern). The valid entry survives; the
    // invalid one is dropped rather than producing an unmatchable
    // scalar inside the list.
    const md = '---\nrequires: [valid-name, bad#name]\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toEqual(['valid-name']);
  });

  it('returns null for a bracket list whose entries ALL fail the shape check', () => {
    // `requires: [bad#name]` has content but no valid entries. This
    // is distinct from the explicit `requires: []` literal-empty form
    // (which means "never load"). The author clearly intended to
    // declare a constraint; the content just doesn't parse. Per the
    // ambiguity-fallback contract, return null (= load unconditionally)
    // rather than collapsing to the explicit empty-list semantic and
    // silently parking the rule.
    const md = '---\nrequires: [bad#name]\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toBeNull();
  });

  it('returns null for a bracket list of only commas (operator typo)', () => {
    // `requires: [,,]` splits into three empty strings, all filtered
    // out. Same logic as the all-shape-invalid case: distinct from
    // the literal `[]` empty form; load unconditionally.
    const md = '---\nrequires: [,,]\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toBeNull();
  });

  it('preserves the literal `[]` never-load semantic (not affected by the fallback)', () => {
    // The literal empty list — whitespace-only inner content — is
    // the author's explicit "never load" form. Round-trips through
    // shouldIncludeRule as `empty-requires`, distinct from the
    // ambiguity-fallback `no-requires` of the malformed cases above.
    expect(
      parseRequiresFrontmatter('---\nrequires: []\n---\n\n# Rule\n'),
    ).toEqual([]);
    expect(
      parseRequiresFrontmatter('---\nrequires: [   ]\n---\n\n# Rule\n'),
    ).toEqual([]);
  });

  it('returns null for a block-list `requires:` with no items (operator typo)', () => {
    // `requires:` alone with no `- foo` lines is the most common
    // authoring mistake. Per the safety contract this is treated as
    // "no constraint, load unconditionally" rather than as an
    // explicit "never load" — that intent is reserved for the
    // explicit `requires: []` inline-empty form.
    const md =
      '---\nalwaysApply: true\nrequires:\notherField: x\n---\n\n# Rule\n';
    expect(parseRequiresFrontmatter(md)).toBeNull();
  });

  it('only matches a top-level requires: line (not nested mappings)', () => {
    // `nested.requires:` at sub-indentation isn't a top-level
    // constraint. The parser's regex anchors on line start with no
    // leading whitespace, so this returns null.
    const md = `---
other:
  requires: [foo]
---

# Rule
`;
    expect(parseRequiresFrontmatter(md)).toBeNull();
  });
});

describe('shouldIncludeRule (#552)', () => {
  it('includes a rule with no requires (default load)', () => {
    const md = '---\nalwaysApply: true\n---\n\n# Rule\n';
    expect(shouldIncludeRule(md, new Set(['anything']))).toEqual({
      include: true,
      reason: 'no-requires',
      requires: null,
    });
  });

  it('includes a rule whose gating skill is present', () => {
    const md = '---\nrequires: [schedule-task]\n---\n\n# Rule\n';
    expect(
      shouldIncludeRule(md, new Set(['schedule-task', 'other-skill'])),
    ).toEqual({
      include: true,
      reason: 'gating-skill-present',
      requires: ['schedule-task'],
    });
  });

  it('includes a rule whose gating list has ANY present skill (any-of)', () => {
    const md =
      '---\nrequires: [absent-a, present-b, absent-c]\n---\n\n# Rule\n';
    expect(shouldIncludeRule(md, new Set(['present-b']))).toEqual({
      include: true,
      reason: 'gating-skill-present',
      requires: ['absent-a', 'present-b', 'absent-c'],
    });
  });

  it('excludes a rule whose gating skill(s) are absent', () => {
    const md = '---\nrequires: [absent-skill]\n---\n\n# Rule\n';
    expect(shouldIncludeRule(md, new Set(['unrelated']))).toEqual({
      include: false,
      reason: 'no-gating-skill',
      requires: ['absent-skill'],
    });
  });

  it('excludes a rule with an explicitly empty requires list', () => {
    // `requires: []` is the author saying "never load". Distinct
    // from no declaration at all.
    const md = '---\nrequires: []\n---\n\n# Rule\n';
    expect(shouldIncludeRule(md, new Set(['anything']))).toEqual({
      include: false,
      reason: 'empty-requires',
      requires: [],
    });
  });

  it('treats an absent skill set the same as missing the gating skill', () => {
    const md = '---\nrequires: [foo]\n---\n\n# Rule\n';
    expect(shouldIncludeRule(md, new Set())).toEqual({
      include: false,
      reason: 'no-gating-skill',
      requires: ['foo'],
    });
  });
});
