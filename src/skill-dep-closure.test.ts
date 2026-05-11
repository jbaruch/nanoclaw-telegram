import { describe, it, expect } from 'vitest';

import {
  computeEffectiveBlocklist,
  computeEffectiveSkillContext,
  extractSkillDeps,
} from './skill-dep-closure.js';

describe('extractSkillDeps (#544)', () => {
  it('extracts a single tessl__-prefixed Skill() invocation', () => {
    const md = 'Step 1\n\n`Skill(skill: "tessl__heartbeat")`\n';
    expect(extractSkillDeps(md)).toEqual(new Set(['heartbeat']));
  });

  it('extracts a single bare-name Skill() invocation (built-in skill)', () => {
    // Built-in skills (`wiki`, `agent-browser`, `status`) are
    // installed at `.claude/skills/<name>/` without the `tessl__`
    // prefix that tile skills get. The agent invokes them by bare
    // name; the blocklist also stores bare names, so the regex
    // should match either form.
    const md = 'Step 2\n\n`Skill(skill: "wiki")`\n';
    expect(extractSkillDeps(md)).toEqual(new Set(['wiki']));
  });

  it('extracts multiple invocations from one SKILL.md', () => {
    const md = `
## Step 1

\`Skill(skill: "tessl__morning-brief")\`

## Step 2

\`Skill(skill: "tessl__check-calendar")\`

## Step 3

\`Skill(skill: "wiki")\`
`;
    expect(extractSkillDeps(md)).toEqual(
      new Set(['morning-brief', 'check-calendar', 'wiki']),
    );
  });

  it('deduplicates repeated invocations of the same skill', () => {
    // A skill that calls another skill twice (e.g. a router that
    // dispatches the same target on two branches) should not
    // produce duplicate entries.
    const md = `
On miss: \`Skill(skill: "tessl__check-cfps")\`
On hit:  \`Skill(skill: "tessl__check-cfps")\`
`;
    expect(extractSkillDeps(md)).toEqual(new Set(['check-cfps']));
  });

  it('accepts both single and double quotes', () => {
    const md = `
\`Skill(skill: "tessl__double")\`
\`Skill(skill: 'tessl__single')\`
`;
    expect(extractSkillDeps(md)).toEqual(new Set(['double', 'single']));
  });

  it('tolerates whitespace around the skill: argument', () => {
    const md = `
\`Skill( skill: "tessl__padded" )\`
\`Skill(skill:"tessl__tight")\`
`;
    expect(extractSkillDeps(md)).toEqual(new Set(['padded', 'tight']));
  });

  it('returns an empty set when SKILL.md has no invocations', () => {
    const md =
      '# A leaf skill with no nested invocations\n\nDo the thing. Done.';
    expect(extractSkillDeps(md)).toEqual(new Set());
  });

  it('handles underscores and hyphens in skill names', () => {
    // Real skill names use kebab-case (`task-tz-sync`) and snake_case
    // is also valid in identifiers. The regex should accept both.
    const md = `
\`Skill(skill: "tessl__task-tz-sync")\`
\`Skill(skill: "tessl__some_skill_name")\`
`;
    expect(extractSkillDeps(md)).toEqual(
      new Set(['task-tz-sync', 'some_skill_name']),
    );
  });
});

describe('computeEffectiveBlocklist (#544)', () => {
  // The blocklist algebra under different reference shapes. The
  // contract: any skill REACHABLE from a non-blocklisted root is
  // exempted from the blocklist for this spawn — its prompt loads
  // so nested `Skill()` invocations can resolve.

  it('returns the original blocklist when no skills reference any blocked skill', () => {
    const original = new Set(['blocked-1', 'blocked-2']);
    const sources = new Map([
      ['root-a', '# leaf\n'],
      ['root-b', '# leaf\n'],
      ['blocked-1', '# isolated\n'],
      ['blocked-2', '# isolated\n'],
    ]);
    expect(computeEffectiveBlocklist(original, sources)).toEqual(
      new Set(['blocked-1', 'blocked-2']),
    );
  });

  it('exempts a directly-referenced blocked skill (wiki-lint → wiki)', () => {
    // The reference incident: wiki-lint (root, not blocked) invokes
    // `Skill(skill: "wiki")`; wiki is in the blocklist but reachable
    // from a loaded root, so it must be exempted.
    const original = new Set(['wiki']);
    const sources = new Map([
      ['wiki-lint', '`Skill(skill: "wiki")`\n'],
      ['wiki', '# the wiki skill\n'],
    ]);
    expect(computeEffectiveBlocklist(original, sources)).toEqual(new Set());
  });

  it('exempts transitively-referenced blocked skills (A → B → C, all reachable)', () => {
    // A (root) calls B (blocked), B calls C (blocked). Both B and C
    // must be exempted — the agent loads A, A invokes B, B's prompt
    // now needs to be available, B invokes C, C's prompt needs to
    // be available. The closure walks the full graph.
    const original = new Set(['b', 'c']);
    const sources = new Map([
      ['a', '`Skill(skill: "tessl__b")`\n'],
      ['b', '`Skill(skill: "tessl__c")`\n'],
      ['c', '# leaf\n'],
    ]);
    expect(computeEffectiveBlocklist(original, sources)).toEqual(new Set());
  });

  it('keeps blocked skills that are unreachable even when other blocked skills are reachable', () => {
    // Mixed graph: root → blocked-reachable, but blocked-isolated is
    // not referenced by anyone loaded. The latter stays blocked.
    const original = new Set(['blocked-reachable', 'blocked-isolated']);
    const sources = new Map([
      ['root', '`Skill(skill: "tessl__blocked-reachable")`\n'],
      ['blocked-reachable', '# leaf\n'],
      ['blocked-isolated', '# truly unused this spawn\n'],
    ]);
    expect(computeEffectiveBlocklist(original, sources)).toEqual(
      new Set(['blocked-isolated']),
    );
  });

  it('handles a cycle in the reference graph without looping forever', () => {
    // Pathological case: A → B → A. The BFS must terminate — it
    // does, because the `reachable` set is checked before
    // re-enqueueing.
    const original = new Set<string>();
    const sources = new Map([
      ['a', '`Skill(skill: "tessl__b")`\n'],
      ['b', '`Skill(skill: "tessl__a")`\n'],
    ]);
    // Both are roots (neither blocked); both end up in `reachable`.
    // Empty effective blocklist (the input was empty too).
    expect(computeEffectiveBlocklist(original, sources)).toEqual(new Set());
  });

  it('does not exempt a blocked skill referenced ONLY by another blocked skill (no live root)', () => {
    // blocked-a (blocked) calls blocked-b (blocked). Neither is a
    // root because both are in the blocklist; neither prompt loads.
    // The reference from blocked-a to blocked-b doesn't matter
    // because blocked-a's prompt isn't available to issue the call.
    // Both stay blocked.
    const original = new Set(['blocked-a', 'blocked-b']);
    const sources = new Map([
      ['live-root', '# leaf\n'],
      ['blocked-a', '`Skill(skill: "tessl__blocked-b")`\n'],
      ['blocked-b', '# leaf\n'],
    ]);
    expect(computeEffectiveBlocklist(original, sources)).toEqual(
      new Set(['blocked-a', 'blocked-b']),
    );
  });

  it('tolerates references to unknown skill names (typo / retired skill)', () => {
    // A SKILL.md may reference a skill that doesn't exist anymore
    // (retired in a prior PR; SKILL.md not updated). The closure
    // can't walk what it can't read; the BFS silently drops the
    // unknown ref. The agent's runtime "Unknown skill" error
    // surfaces this case at invocation time — this helper isn't a
    // validator.
    const original = new Set(['real-blocked']);
    const sources = new Map([
      [
        'root',
        '`Skill(skill: "tessl__retired-name")`\n`Skill(skill: "tessl__real-blocked")`\n',
      ],
      ['real-blocked', '# leaf\n'],
    ]);
    expect(computeEffectiveBlocklist(original, sources)).toEqual(new Set());
  });

  it('returns the original blocklist when skillSources is empty (no roots, nothing to walk)', () => {
    const original = new Set(['x', 'y']);
    expect(computeEffectiveBlocklist(original, new Map())).toEqual(
      new Set(['x', 'y']),
    );
  });

  it('does not mutate the input blocklist', () => {
    const original = new Set(['blocked']);
    const sources = new Map([
      ['root', '`Skill(skill: "tessl__blocked")`\n'],
      ['blocked', '# leaf\n'],
    ]);
    const result = computeEffectiveBlocklist(original, sources);
    expect(result).toEqual(new Set());
    // Original must be untouched — caller may share the set across
    // multiple spawns and would not expect side effects.
    expect(original).toEqual(new Set(['blocked']));
  });
});

describe('computeEffectiveSkillContext (#552 — reachable surface)', () => {
  // The context-returning form exposes the positive presence set
  // (every skill whose prompt loads into the agent's context) so
  // the #552 rule `requires:` filter can decide whether a rule's
  // gating skill is actually available for this spawn.

  it('reachable set contains every non-blocklisted root with no references', () => {
    // No `Skill()` invocations anywhere — the reachable set is
    // exactly the non-blocklisted roots.
    const original = new Set(['blocked']);
    const sources = new Map([
      ['root-a', '# leaf\n'],
      ['root-b', '# leaf\n'],
      ['blocked', '# leaf\n'],
    ]);
    const ctx = computeEffectiveSkillContext(original, sources);
    expect(ctx.reachableSkills).toEqual(new Set(['root-a', 'root-b']));
    expect(ctx.effectiveBlocklist).toEqual(new Set(['blocked']));
  });

  it('reachable set includes transitively-referenced blocked skills', () => {
    // A (root) → B (blocked) → C (blocked). Both B and C are
    // reachable; the rule filter can therefore satisfy a rule that
    // requires B or C.
    const original = new Set(['b', 'c']);
    const sources = new Map([
      ['a', '`Skill(skill: "tessl__b")`\n'],
      ['b', '`Skill(skill: "tessl__c")`\n'],
      ['c', '# leaf\n'],
    ]);
    const ctx = computeEffectiveSkillContext(original, sources);
    expect(ctx.reachableSkills).toEqual(new Set(['a', 'b', 'c']));
    expect(ctx.effectiveBlocklist).toEqual(new Set());
  });

  it('reachable set excludes blocked skills with no live referrer', () => {
    // blocked-a → blocked-b — neither is a root, so blocked-a's
    // prompt never loads, so blocked-b is unreachable too. Live
    // root has no references; reachable = {live-root}.
    const original = new Set(['blocked-a', 'blocked-b']);
    const sources = new Map([
      ['live-root', '# leaf\n'],
      ['blocked-a', '`Skill(skill: "tessl__blocked-b")`\n'],
      ['blocked-b', '# leaf\n'],
    ]);
    const ctx = computeEffectiveSkillContext(original, sources);
    expect(ctx.reachableSkills).toEqual(new Set(['live-root']));
    expect(ctx.effectiveBlocklist).toEqual(new Set(['blocked-a', 'blocked-b']));
  });

  it('default-class spawn (empty blocklist) produces reachable = every present skill', () => {
    // The common case for #552: the spawn isn't maintenance, so no
    // blocklist applies, and every skill loads. The rule filter
    // operates against the full skill surface.
    const original = new Set<string>();
    const sources = new Map([
      ['skill-a', '# leaf\n'],
      ['skill-b', '`Skill(skill: "tessl__skill-c")`\n'],
      ['skill-c', '# leaf\n'],
    ]);
    const ctx = computeEffectiveSkillContext(original, sources);
    expect(ctx.reachableSkills).toEqual(
      new Set(['skill-a', 'skill-b', 'skill-c']),
    );
    expect(ctx.effectiveBlocklist).toEqual(new Set());
  });

  it('empty skillSources produces empty reachable and unchanged blocklist', () => {
    const original = new Set(['x', 'y']);
    const ctx = computeEffectiveSkillContext(original, new Map());
    expect(ctx.reachableSkills).toEqual(new Set());
    expect(ctx.effectiveBlocklist).toEqual(new Set(['x', 'y']));
  });
});
