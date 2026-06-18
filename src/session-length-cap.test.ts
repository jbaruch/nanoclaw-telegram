import { describe, expect, it } from 'vitest';

import {
  buildHandoffPrefix,
  buildResetNotification,
  resolveLiveSessionCaps,
  resolveSessionCaps,
  shouldMarkForReset,
} from './session-length-cap.js';

// Fixed-data tests per `rules/testing-standards.md`. No randomness, no
// shared mutable state — every case has its own inputs and asserts a
// deterministic outcome.

describe('shouldMarkForReset', () => {
  const CAPS = { tokenCap: 1_000_000, turnCap: 100 };

  it('returns reset:false when both totals are below their caps', () => {
    expect(
      shouldMarkForReset({ totalInputTokens: 500_000, turnCount: 50 }, CAPS),
    ).toEqual({ reset: false });
  });

  it('returns reset:true with token_cap reason at the boundary', () => {
    expect(
      shouldMarkForReset({ totalInputTokens: 1_000_000, turnCount: 1 }, CAPS),
    ).toEqual({
      reset: true,
      reason: 'token_cap',
      observed: 1_000_000,
      cap: 1_000_000,
    });
  });

  it('returns reset:true with turn_cap reason at the boundary', () => {
    expect(
      shouldMarkForReset({ totalInputTokens: 1, turnCount: 100 }, CAPS),
    ).toEqual({
      reset: true,
      reason: 'turn_cap',
      observed: 100,
      cap: 100,
    });
  });

  it('does NOT trip one token below the cap (boundary correctness)', () => {
    expect(
      shouldMarkForReset({ totalInputTokens: 999_999, turnCount: 99 }, CAPS),
    ).toEqual({ reset: false });
  });

  it('reports token_cap when both caps fire simultaneously (deterministic precedence)', () => {
    expect(
      shouldMarkForReset({ totalInputTokens: 2_000_000, turnCount: 200 }, CAPS),
    ).toEqual({
      reset: true,
      reason: 'token_cap',
      observed: 2_000_000,
      cap: 1_000_000,
    });
  });

  it('skips token cap when configured value is <= 0 (turn cap still active)', () => {
    expect(
      shouldMarkForReset(
        { totalInputTokens: 9_999_999, turnCount: 100 },
        { tokenCap: 0, turnCap: 100 },
      ),
    ).toEqual({
      reset: true,
      reason: 'turn_cap',
      observed: 100,
      cap: 100,
    });
  });

  it('skips turn cap when configured value is <= 0 (token cap still active)', () => {
    expect(
      shouldMarkForReset(
        { totalInputTokens: 1_000_000, turnCount: 9_999 },
        { tokenCap: 1_000_000, turnCap: -1 },
      ),
    ).toEqual({
      reset: true,
      reason: 'token_cap',
      observed: 1_000_000,
      cap: 1_000_000,
    });
  });

  it('never trips when both caps are disabled', () => {
    expect(
      shouldMarkForReset(
        { totalInputTokens: 1e9, turnCount: 1e6 },
        { tokenCap: 0, turnCap: 0 },
      ),
    ).toEqual({ reset: false });
  });
});

describe('resolveSessionCaps', () => {
  const DEFAULTS = { tokenCap: 1_000_000, turnCap: 100 };

  it('inherits both global caps when no override is present', () => {
    expect(resolveSessionCaps(undefined, DEFAULTS)).toEqual(DEFAULTS);
    expect(resolveSessionCaps({}, DEFAULTS)).toEqual(DEFAULTS);
  });

  it('applies a positive per-group turn cap, inherits the token cap', () => {
    expect(resolveSessionCaps({ sessionTurnCap: 40 }, DEFAULTS)).toEqual({
      tokenCap: 1_000_000,
      turnCap: 40,
    });
  });

  it('applies a positive per-group token cap, inherits the turn cap', () => {
    expect(resolveSessionCaps({ sessionTokenCap: 500_000 }, DEFAULTS)).toEqual({
      tokenCap: 500_000,
      turnCap: 100,
    });
  });

  it('applies both per-group overrides independently', () => {
    expect(
      resolveSessionCaps(
        { sessionTurnCap: 40, sessionTokenCap: 250_000 },
        DEFAULTS,
      ),
    ).toEqual({ tokenCap: 250_000, turnCap: 40 });
  });

  it('falls back to global on non-positive overrides (no silent disable)', () => {
    expect(
      resolveSessionCaps({ sessionTurnCap: 0, sessionTokenCap: -5 }, DEFAULTS),
    ).toEqual(DEFAULTS);
  });

  it('falls back to global on non-finite and non-number overrides', () => {
    expect(
      resolveSessionCaps(
        {
          sessionTurnCap: Number.NaN,
          sessionTokenCap: Number.POSITIVE_INFINITY,
        },
        DEFAULTS,
      ),
    ).toEqual(DEFAULTS);
    expect(
      resolveSessionCaps(
        // Malformed values from a hand-edited container_config row.
        { sessionTurnCap: '40', sessionTokenCap: null },
        DEFAULTS,
      ),
    ).toEqual(DEFAULTS);
  });

  it('passes a per-group cap through to a reset verdict end-to-end', () => {
    const caps = resolveSessionCaps({ sessionTurnCap: 40 }, DEFAULTS);
    // 40 turns with the tightened cap trips; it would not under the
    // global 100.
    expect(
      shouldMarkForReset({ totalInputTokens: 1, turnCount: 40 }, caps),
    ).toEqual({ reset: true, reason: 'turn_cap', observed: 40, cap: 40 });
    expect(
      shouldMarkForReset({ totalInputTokens: 1, turnCount: 40 }, DEFAULTS),
    ).toEqual({ reset: false });
  });
});

describe('resolveLiveSessionCaps', () => {
  const DEFAULTS = { tokenCap: 1_000_000, turnCap: 100 };

  it('resolves from the live registry entry matched by folder', () => {
    const registry = {
      'jid-a': { folder: 'group-a', containerConfig: { sessionTurnCap: 40 } },
      'jid-b': { folder: 'group-b' },
    };
    expect(
      resolveLiveSessionCaps(registry, 'group-a', undefined, DEFAULTS),
    ).toEqual({ tokenCap: 1_000_000, turnCap: 40 });
  });

  it('prefers the live registry config over the captured fallback', () => {
    // The captured (spawn-time) config still says 100; the live registry
    // entry has been updated to 30. The live value must win.
    const registry = {
      'jid-a': { folder: 'group-a', containerConfig: { sessionTurnCap: 30 } },
    };
    expect(
      resolveLiveSessionCaps(
        registry,
        'group-a',
        { sessionTurnCap: 100 },
        DEFAULTS,
      ),
    ).toEqual({ tokenCap: 1_000_000, turnCap: 30 });
  });

  it('picks up a cap change made mid-session on the next resolution (the no-respawn outcome)', () => {
    // Simulates set_session_caps landing while a default container is
    // still active: the per-turn check re-reads the registry, so the
    // very next turn sees the new cap without a respawn.
    const registry: Record<
      string,
      { folder: string; containerConfig?: { sessionTurnCap?: number } }
    > = {
      'jid-a': { folder: 'group-a' },
    };
    const captured = registry['jid-a'].containerConfig;

    expect(
      resolveLiveSessionCaps(registry, 'group-a', captured, DEFAULTS).turnCap,
    ).toBe(100); // inherits global before any override

    // set_session_caps replaces the registry entry (registerGroup writes
    // a fresh object — the captured `captured` reference is unchanged).
    registry['jid-a'] = {
      folder: 'group-a',
      containerConfig: { sessionTurnCap: 25 },
    };

    expect(
      resolveLiveSessionCaps(registry, 'group-a', captured, DEFAULTS).turnCap,
    ).toBe(25); // next turn honors the override with no respawn
  });

  it('falls back to the captured config when no live entry matches the folder', () => {
    // Group unregistered mid-run: the registry no longer has the folder,
    // so the check degrades to the spawn-time config rather than dropping
    // the cap entirely.
    const registry = {
      'jid-b': { folder: 'group-b', containerConfig: { sessionTurnCap: 10 } },
    };
    expect(
      resolveLiveSessionCaps(
        registry,
        'group-a',
        { sessionTurnCap: 50 },
        DEFAULTS,
      ),
    ).toEqual({ tokenCap: 1_000_000, turnCap: 50 });
  });

  it('inherits the global default when neither live nor fallback override is present', () => {
    expect(resolveLiveSessionCaps({}, 'group-a', undefined, DEFAULTS)).toEqual(
      DEFAULTS,
    );
  });
});

describe('buildHandoffPrefix', () => {
  it('returns null when both inputs are empty', () => {
    expect(buildHandoffPrefix({})).toBeNull();
    expect(
      buildHandoffPrefix({ lastAssistantText: '', lastUserText: '' }),
    ).toBeNull();
    expect(
      buildHandoffPrefix({ lastAssistantText: '   ', lastUserText: '\n\t' }),
    ).toBeNull();
  });

  it('builds a session-handoff block with both lines when both are present', () => {
    const out = buildHandoffPrefix({
      lastAssistantText: 'Done — the deploy script reports green.',
      lastUserText: 'ship it',
      assistantName: 'TestAssistant',
    });
    expect(out).toContain('<session-handoff>');
    expect(out).toContain('</session-handoff>');
    expect(out).toContain('User just before reset: ship it');
    expect(out).toContain(
      "TestAssistant's last reply: Done — the deploy script reports green.",
    );
  });

  it('omits the user line when only the assistant text is provided', () => {
    const out = buildHandoffPrefix({
      lastAssistantText: 'I rebuilt the container.',
      assistantName: 'TestAssistant',
    });
    expect(out).not.toBeNull();
    expect(out).not.toContain('User just before reset:');
    expect(out).toContain(
      "TestAssistant's last reply: I rebuilt the container.",
    );
  });

  it('uses the passed-in assistantName in the prefix label', () => {
    const out = buildHandoffPrefix({
      lastAssistantText: 'Container restarted.',
      assistantName: 'NanoClaw',
    });
    expect(out).toContain("NanoClaw's last reply: Container restarted.");
    expect(out).not.toContain("TestAssistant's last reply:");
  });

  it("falls back to neutral 'Assistant' label when assistantName is unset", () => {
    const out = buildHandoffPrefix({
      lastAssistantText: 'Working on it.',
    });
    expect(out).toContain("Assistant's last reply: Working on it.");
    expect(out).not.toContain("TestAssistant's last reply:");
  });

  it("falls back to 'Assistant' for whitespace-only assistantName", () => {
    // An env var like `ASSISTANT_NAME=  ` shouldn't produce a
    // `'s last reply:` prefix with no name.
    const out = buildHandoffPrefix({
      lastAssistantText: 'Working on it.',
      assistantName: '   ',
    });
    expect(out).toContain("Assistant's last reply: Working on it.");
  });

  it('truncates very long inputs with an ellipsis (cap fits well under prompt budget)', () => {
    const longText = 'a'.repeat(2_000);
    const out = buildHandoffPrefix({
      lastAssistantText: longText,
      assistantName: 'TestAssistant',
    });
    expect(out).not.toBeNull();
    // 400-char cap (399 chars + ellipsis).
    expect(out).toContain('a'.repeat(399) + '…');
    expect(out).not.toContain('a'.repeat(400));
  });

  it('collapses internal whitespace so multi-line snippets do not waste budget', () => {
    const out = buildHandoffPrefix({
      lastAssistantText: 'line1\n\n\nline2\t\tline3',
      assistantName: 'TestAssistant',
    });
    expect(out).toContain("TestAssistant's last reply: line1 line2 line3");
  });
});

describe('buildResetNotification', () => {
  it('formats a token_cap line that includes a thousands-separated cap value', () => {
    expect(buildResetNotification('token_cap', 1_000_000)).toBe(
      'Session reset: cumulative input tokens crossed 1,000,000 — starting fresh to keep context bounded.',
    );
  });

  it('formats a turn_cap line that includes the integer cap value', () => {
    expect(buildResetNotification('turn_cap', 100)).toBe(
      'Session reset: turn count crossed 100 — starting fresh to keep context bounded.',
    );
  });
});
