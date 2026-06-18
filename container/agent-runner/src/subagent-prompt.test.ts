import { describe, it, expect } from 'vitest';

import {
  buildSubagentRuleFilePaths,
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
