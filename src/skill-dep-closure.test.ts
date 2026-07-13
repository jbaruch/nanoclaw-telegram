import { describe, it, expect } from 'vitest';

import {
  computeEffectiveBlocklist,
  computeEffectiveSkillContext,
  extractMountPathDeps,
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

  it('extracts an invocation carrying an args: param (#652)', () => {
    // The #652 outage: wiki-lint invokes `Skill(skill: "wiki",
    // args: "lint")`. The old regex required `)` immediately after
    // the closing quote, so the args form was invisible — wiki was
    // never exempted and the maintenance container failed with
    // "Unknown skill: wiki". The bare and args forms must both match.
    const md = '`Skill(skill: "wiki", args: "lint")`\n';
    expect(extractSkillDeps(md)).toEqual(new Set(['wiki']));
  });

  it('extracts a tessl__-prefixed invocation with args: (#652)', () => {
    const md = '`Skill(skill: "tessl__resumable-cycle", args: "check foo")`\n';
    expect(extractSkillDeps(md)).toEqual(new Set(['resumable-cycle']));
  });

  it('extracts both no-args and args forms in one SKILL.md (#652)', () => {
    // Mixed shapes in a single skill — morning-brief calls
    // resumable-cycle with args, then a leaf skill with none.
    const md = `
\`Skill(skill: "tessl__resumable-cycle", args: "check")\`
\`Skill(skill: "tessl__morning-brief")\`
`;
    expect(extractSkillDeps(md)).toEqual(
      new Set(['resumable-cycle', 'morning-brief']),
    );
  });
});

describe('extractMountPathDeps (#441)', () => {
  it('extracts a skill from a tessl__<name>/ mount-path reference', () => {
    const script =
      'subprocess.run(["python3", ' +
      '"/home/node/.claude/skills/tessl__scheduler-timezone/scripts/compute-schedule-value.py"])';
    expect(extractMountPathDeps(script)).toEqual(
      new Set(['scheduler-timezone']),
    );
  });

  it('extracts multiple distinct mount-path deps and dedups repeats', () => {
    const text = `
      .../tessl__scheduler-timezone/scripts/a.py
      .../tessl__scheduler-timezone/scripts/b.py
      .../tessl__check-calendar/references/x.md
    `;
    expect(extractMountPathDeps(text)).toEqual(
      new Set(['scheduler-timezone', 'check-calendar']),
    );
  });

  it('does NOT match a bare tessl__<name> mention without a trailing slash', () => {
    // A doc-comment listing another skill by name (no path into its dir)
    // must not falsely rescue it — the trailing `/` anchors the match to
    // an actual file reference (heartbeat-precheck.py mentions
    // `tessl__check-email` / `tessl__trusted-memory` in prose, #337 audit).
    const comment =
      'Other writers: tessl__check-email and tessl__trusted-memory';
    expect(extractMountPathDeps(comment)).toEqual(new Set());
  });

  it('returns an empty set when there are no mount-path references', () => {
    expect(extractMountPathDeps('plain text, no skill refs')).toEqual(
      new Set(),
    );
  });

  it('handles underscores and hyphens in the skill name', () => {
    expect(extractMountPathDeps('tessl__a_b-c/scripts/x')).toEqual(
      new Set(['a_b-c']),
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

  it('exempts a blocked skill referenced via a tessl__<name>/ mount path (#441 morning-brief → scheduler-timezone)', () => {
    // The 2026-07-12 failure shape: morning-brief (root, not blocked)
    // shells out to scheduler-timezone's compute-schedule-value.py by
    // mount path from a script — no `Skill()` call. scheduler-timezone is
    // on the maintenance blocklist; the closure must rescue it from the
    // mount-path reference folded into morning-brief's source blob.
    const original = new Set(['scheduler-timezone']);
    const sources = new Map([
      [
        'morning-brief',
        '# morning-brief\nsubprocess: .../tessl__scheduler-timezone/scripts/compute-schedule-value.py\n',
      ],
      ['scheduler-timezone', '# scheduler-timezone\n'],
    ]);
    expect(computeEffectiveBlocklist(original, sources)).toEqual(new Set());
  });

  it('does NOT rescue a blocked skill referenced only by a bare (slashless) tessl__ mention', () => {
    // A doc-comment naming another skill without a path into its dir is
    // not a real dependency — it must stay blocked.
    const original = new Set(['trusted-memory']);
    const sources = new Map([
      [
        'heartbeat',
        '# heartbeat\nother writer: tessl__trusted-memory (note)\n',
      ],
      ['trusted-memory', '# trusted-memory\n'],
    ]);
    expect(computeEffectiveBlocklist(original, sources)).toEqual(
      new Set(['trusted-memory']),
    );
  });

  it('exempts wiki when wiki-lint invokes it with args: (#652 outage)', () => {
    // The actual #652 failure shape: wiki-lint's SKILL.md invokes
    // `Skill(skill: "wiki", args: "lint")`. The old regex couldn't
    // see the args form, so wiki stayed on the maintenance blocklist
    // and the spawn failed with "Unknown skill: wiki". The closure
    // must exempt wiki from the args-bearing invocation too.
    const original = new Set(['wiki']);
    const sources = new Map([
      ['wiki-lint', '`Skill(skill: "wiki", args: "lint")`\n'],
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
