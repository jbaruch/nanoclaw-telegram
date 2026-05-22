import { describe, it, expect } from 'vitest';
import { classifyResultIsError } from './classify-result-error.js';

describe('classifyResultIsError', () => {
  it('returns false for clean success results', () => {
    expect(
      classifyResultIsError({
        subtype: 'success',
        is_error: false,
        stop_reason: 'end_turn',
        terminal_reason: 'completed',
      }),
    ).toBe(false);
  });

  it('returns true on the classic error_during_execution shape', () => {
    expect(
      classifyResultIsError({
        subtype: 'error_during_execution',
        is_error: true,
        stop_reason: 'tool_use',
        terminal_reason: 'tool_use_limit_reached',
      }),
    ).toBe(true);
  });

  it('returns true when subtype is non-success and is_error is unset (legacy heuristic preserved)', () => {
    expect(
      classifyResultIsError({
        subtype: 'permission_denied',
        stop_reason: 'permissions',
      }),
    ).toBe(true);
  });

  it('returns false on the SDK contradiction shape — subtype=success + is_error=true + stop_sequence + completed (#619)', () => {
    // Exact production shape observed 2026-05-22T05:04Z on
    // tessl__heartbeat and 05:09Z on tessl__morning-brief. Three of
    // four classification signals say success; the SDK's `is_error`
    // flag is the outlier. Treat as success.
    expect(
      classifyResultIsError({
        subtype: 'success',
        is_error: true,
        stop_reason: 'stop_sequence',
        terminal_reason: 'completed',
      }),
    ).toBe(false);
  });

  it('returns false on the SDK contradiction shape when terminal_reason is absent', () => {
    // Some SDK versions omit `terminal_reason` entirely on clean
    // completions. The suppression treats absent terminal_reason the
    // same as `completed` because the other three signals already
    // agree on success.
    expect(
      classifyResultIsError({
        subtype: 'success',
        is_error: true,
        stop_reason: 'stop_sequence',
      }),
    ).toBe(false);
  });

  it('keeps tool_use_limit failures classified as errors even on subtype=success (preserves #582 prior art)', () => {
    // The #582 test pinned this exact shape as a real failure:
    // `subtype: 'success'`, `is_error: true`, `stop_reason: 'tool_use'`,
    // `terminal_reason: 'tool_use_limit_reached'`. The narrow #619
    // suppression must NOT consume it — only stop_sequence+completed
    // gets suppressed.
    expect(
      classifyResultIsError({
        subtype: 'success',
        is_error: true,
        stop_reason: 'tool_use',
        terminal_reason: 'tool_use_limit_reached',
      }),
    ).toBe(true);
  });

  it('keeps stop_sequence as error when subtype is not success', () => {
    // Defense against an SDK quirk where stop_reason=stop_sequence
    // happens to land on a genuinely-errored result. subtype=error
    // wins.
    expect(
      classifyResultIsError({
        subtype: 'error_during_execution',
        is_error: true,
        stop_reason: 'stop_sequence',
        terminal_reason: 'completed',
      }),
    ).toBe(true);
  });

  it('keeps stop_sequence as error when terminal_reason indicates failure', () => {
    // If terminal_reason is something other than 'completed' or
    // absent, the suppression does not apply — the SDK is telling us
    // the turn ended for a reason other than clean completion.
    expect(
      classifyResultIsError({
        subtype: 'success',
        is_error: true,
        stop_reason: 'stop_sequence',
        terminal_reason: 'cancelled',
      }),
    ).toBe(true);
  });

  it('treats missing subtype as unknown (matches legacy behaviour)', () => {
    // Per the #149 review note: undefined subtype defaults to
    // 'unknown' so the error string stays readable. `is_error` not
    // set + subtype = 'unknown' → not an error.
    expect(classifyResultIsError({})).toBe(false);
  });

  it('flags explicit is_error=true on unknown subtype', () => {
    expect(classifyResultIsError({ is_error: true })).toBe(true);
  });
});
