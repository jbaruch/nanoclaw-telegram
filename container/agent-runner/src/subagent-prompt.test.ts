import { describe, it, expect } from 'vitest';

import {
  buildSubagentRuleFilePaths,
  buildSubagentRulesDirective,
  shouldIncludeSubagentDefinitions,
} from './subagent-prompt.js';

const SOUL = '/workspace/global/SOUL.md';
const FORMATTING = '/workspace/global/FORMATTING.md';

describe('shouldIncludeSubagentDefinitions', () => {
  it('includes for a default (non-maintenance) session', () => {
    expect(shouldIncludeSubagentDefinitions(false, {})).toBe(true);
  });

  it('skips for a maintenance session by default', () => {
    expect(shouldIncludeSubagentDefinitions(true, {})).toBe(false);
  });

  it('force-includes for maintenance when MAINTENANCE_LOAD_SUBAGENTS=1', () => {
    expect(
      shouldIncludeSubagentDefinitions(true, {
        MAINTENANCE_LOAD_SUBAGENTS: '1',
      }),
    ).toBe(true);
  });

  it('only the exact "1" opt-in flips a maintenance session', () => {
    // Any other value (typo, "true", "0", empty) keeps the cost-saving
    // default of skipping — the escape hatch must be deliberate.
    for (const v of ['true', '0', 'yes', '', 'TRUE']) {
      expect(
        shouldIncludeSubagentDefinitions(true, {
          MAINTENANCE_LOAD_SUBAGENTS: v,
        }),
      ).toBe(false);
    }
  });

  it('ignores the opt-in flag for a non-maintenance session (already included)', () => {
    expect(
      shouldIncludeSubagentDefinitions(false, {
        MAINTENANCE_LOAD_SUBAGENTS: '0',
      }),
    ).toBe(true);
  });
});

describe('buildSubagentRuleFilePaths', () => {
  it('non-main subagents get SOUL + FORMATTING + per-group MEMORY + per-group rules', () => {
    const files = buildSubagentRuleFilePaths({
      isMain: false,
      soulMdPath: SOUL,
      formattingMdPath: FORMATTING,
    });
    expect(files).toEqual([
      SOUL,
      FORMATTING,
      '/workspace/group/MEMORY.md',
      '/workspace/group/.tessl/RULES.md',
    ]);
  });

  it('main subagents additionally load project-root RULES + ADMIN.md', () => {
    const files = buildSubagentRuleFilePaths({
      isMain: true,
      soulMdPath: SOUL,
      formattingMdPath: FORMATTING,
    });
    // Both project-root paths must be present — the per-group
    // tessl path lives elsewhere on main and ADMIN.md carries the
    // admin/runbook instructions main subagents need.
    expect(files).toContain('/workspace/project/.tessl/RULES.md');
    expect(files).toContain('/workspace/project/groups/main/ADMIN.md');
    // And main still gets the standard chain (SOUL + FORMATTING + MEMORY +
    // per-group RULES) — the main-only files are additive, not replacements.
    expect(files).toContain(SOUL);
    expect(files).toContain(FORMATTING);
    expect(files).toContain('/workspace/group/MEMORY.md');
    expect(files).toContain('/workspace/group/.tessl/RULES.md');
  });

  it('main-only paths are NOT included when isMain is false', () => {
    const files = buildSubagentRuleFilePaths({
      isMain: false,
      soulMdPath: SOUL,
      formattingMdPath: FORMATTING,
    });
    expect(files).not.toContain('/workspace/project/.tessl/RULES.md');
    expect(files).not.toContain('/workspace/project/groups/main/ADMIN.md');
  });
});

describe('buildSubagentRulesDirective (#696)', () => {
  const TILES = '/home/node/.claude/.tessl/tiles';

  it('names the passed tiles dir and the rules glob', () => {
    const d = buildSubagentRulesDirective(TILES);
    expect(d).toContain(TILES);
    expect(d).toContain('*/rules/*.md');
  });

  it('embeds a runnable enumeration command for the exact dir', () => {
    // The subagent must be able to find the files itself — the directive
    // carries the literal find command against the same dir.
    expect(buildSubagentRulesDirective(TILES)).toContain(
      `find ${TILES} -path '*/rules/*.md'`,
    );
  });

  it('instructs reading before acting and marks the rules mandatory', () => {
    // The whole point of externalizing the bulk is that the subagent
    // pulls it on spawn — so the directive must be an explicit
    // read-first / mandatory instruction, not a soft hint.
    const d = buildSubagentRulesDirective(TILES);
    expect(d).toMatch(/MANDATORY/);
    expect(d).toMatch(/before .*action/i);
    expect(d).toMatch(/\bRead\b/);
  });

  it('reflects a different tiles dir verbatim', () => {
    // No hardcoded path — the directive is parameterized so a relocated
    // mount (or a test fixture) is honored.
    const alt = '/somewhere/else/tiles';
    const d = buildSubagentRulesDirective(alt);
    expect(d).toContain(alt);
    expect(d).not.toContain(TILES);
  });
});
