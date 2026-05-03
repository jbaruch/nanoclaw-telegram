// Stage 1 trigger gate — multi-handle (#464) coverage. The default
// trigger.test.ts cases use the host's resolved ASSISTANT_NAME /
// ASSISTANT_USERNAME defaults; this file mounts an isolated mock of
// `../config.js` so we can inject a synthetic alias list and prove
// every alias short-circuits to `allow` on a direct @-mention.
//
// Lives in its own file because vi.mock applies to the whole module
// graph for that test file — combining it with trigger.test.ts would
// require mutating the mock between tests, which fights the
// once-per-suite caching of `triggerGate`'s import.
import { describe, it, expect, vi } from 'vitest';

// Literals are inlined into the vi.mock factory because vi.mock is
// hoisted above any top-level `const` and would otherwise see the
// fixtures as still-uninitialized.
vi.mock('../config.js', async () => {
  const actual =
    await vi.importActual<typeof import('../config.js')>('../config.js');
  return {
    ...actual,
    ASSISTANT_NAME: 'TestAssistant',
    ASSISTANT_USERNAME: 'testbot',
    ASSISTANT_USERNAMES: ['testbot', 'testbotsurebot'],
  };
});

const PRIMARY_USERNAME = 'testbot';
const ALIAS_USERNAME = 'testbotsurebot';

import { triggerGate } from './trigger.js';
import type { GateContext } from './index.js';

function ctx(text: string): GateContext {
  return {
    groupJid: 'g@g.us',
    groupFolder: 'g',
    message: { text, senderJid: 's@s.whatsapp.net' },
    triggerPatterns: { version: 1, patterns: [] },
  };
}

describe('triggerGate — multi-handle synthetic identity (#464)', () => {
  it('matches @<primary handle>', () => {
    const result = triggerGate(ctx(`@${PRIMARY_USERNAME} ping`));
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
    expect(result.reason).toContain(PRIMARY_USERNAME);
  });

  it('matches @<alias handle> — the bug reported in #464', () => {
    // Reproducer: bot's autocomplete-only handle differs from the
    // orchestrator-internal username. Without alias support the gate
    // returned `deny` ("no trigger pattern matched") and the message
    // fell through to Stage 2 (which had the same blind spot).
    const result = triggerGate(ctx(`@${ALIAS_USERNAME} ping`));
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
    expect(result.reason).toContain(ALIAS_USERNAME);
  });

  it('alias match is case-insensitive', () => {
    const result = triggerGate(
      ctx(`hey @${ALIAS_USERNAME.toUpperCase()} please`),
    );
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
  });

  it('does not deny when the message has no identity reference at all', () => {
    // A message with no identity markers and no operator patterns
    // must still pass (deferring to the next gate / fail-open
    // default). The alias change must not flip this.
    const result = triggerGate(ctx('just human chatter'));
    expect(result.decision).toBe('pass');
  });

  it('does not match an unrelated @-handle that shares a prefix with an alias', () => {
    // `@testbotsurebot` is an alias; `@testbotsurebother` must not
    // match — the synthetic-mention regex carries a `\b` after the
    // handle. Pin this so a future "prefix-aware" optimization
    // doesn't silently broaden the match. The empty-operator
    // path returns `pass` (no trigger patterns configured) when
    // the synthetic match misses, so the assertion is "definitely
    // not allowed" rather than a specific decision value.
    const result = triggerGate(ctx(`@${ALIAS_USERNAME}other ping`));
    expect(result.decision).not.toBe('allow');
  });
});
