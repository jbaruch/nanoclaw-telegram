import { describe, expect, it } from 'vitest';

import {
  buildHandoffPrefix,
  buildResetNotification,
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
