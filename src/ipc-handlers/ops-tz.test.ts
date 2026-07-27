import { describe, it, expect } from 'vitest';

import {
  MAX_TZ_SEGMENTS,
  classifyTzPersist,
  coerceTzSegments,
} from './ops-tz.js';

describe('coerceTzSegments', () => {
  it('passes a segment array through untouched', () => {
    const segments = [
      { timezone: 'America/New_York', from: '2025-08-01', to: '2025-08-05' },
      { timezone: 'Europe/London', from: '2025-09-10', to: '2025-09-14' },
    ];
    const result = coerceTzSegments(segments);
    expect(result.wasArray).toBe(true);
    expect(result.segments).toEqual(segments);
  });

  it('treats an empty array as a valid (cleared) segment set', () => {
    // A genuine no-active-trips sync emits `segments: []` — this must be an
    // array pass-through (`wasArray: true`), so a real "the owner has no
    // upcoming travel" result is the one and only way to clear tz_state.
    const result = coerceTzSegments([]);
    expect(result.wasArray).toBe(true);
    expect(result.segments).toEqual([]);
  });

  it.each([
    ['undefined', undefined],
    ['null', null],
    ['a plain object', { segments: [] }],
    ['a string', '[]'],
    ['a number', 3],
  ])('coerces %s to an empty array (wasArray=false)', (_label, raw) => {
    const result = coerceTzSegments(raw);
    expect(result.wasArray).toBe(false);
    expect(result.segments).toEqual([]);
  });
});

// -----------------------------------------------------------------
// classifyTzPersist — the pure deny/reject/persist decision behind the
// persist_tz_segments IPC handler (#748). The security-sensitive outcomes
// live here so they're testable without staging processTaskIpc: a non-main
// caller is denied (the isMain re-check that stops IPC-file poisoning), an
// over-cap payload is refused (not truncated), and a valid payload persists.
// -----------------------------------------------------------------
describe('classifyTzPersist', () => {
  const seg = {
    timezone: 'America/New_York',
    from: '2025-08-01',
    to: '2025-08-05',
  };

  it('denies a non-main caller regardless of payload', () => {
    // The MCP tool is isMain-gated, but the IPC task-file path is reachable by
    // any container — this deny is what actually blocks tz_state poisoning.
    expect(classifyTzPersist(false, [seg])).toEqual({
      action: 'deny',
      error: 'persist_tz_segments is main-group only',
    });
    expect(classifyTzPersist(false, undefined).action).toBe('deny');
  });

  it('persists a valid within-bounds array from the main caller', () => {
    expect(classifyTzPersist(true, [seg])).toEqual({
      action: 'persist',
      segments: [seg],
    });
  });

  it('persists an explicit empty array as the cleared (no-active-trips) state', () => {
    expect(classifyTzPersist(true, [])).toEqual({
      action: 'persist',
      segments: [],
    });
  });

  it.each([
    ['undefined', undefined],
    ['null', null],
    ['a plain object', { segments: [] }],
    ['a string', '[]'],
  ])(
    'rejects a non-array (%s) rather than clearing tz_state',
    (_label, raw) => {
      // A non-array is a caller bug, not a "no trips" signal. Persisting an empty
      // set would clear tz_state and could flip the owner tz — refuse instead, as
      // the removed host-op path did. Only an explicit [] clears.
      expect(classifyTzPersist(true, raw)).toEqual({
        action: 'reject',
        error: 'segments must be an array (send [] to explicitly clear)',
      });
    },
  );

  it('persists at exactly the cap (boundary is inclusive)', () => {
    const at = Array.from({ length: MAX_TZ_SEGMENTS }, () => seg);
    const decision = classifyTzPersist(true, at);
    expect(decision.action).toBe('persist');
  });

  it('rejects an over-cap payload without truncating it', () => {
    const over = Array.from({ length: MAX_TZ_SEGMENTS + 1 }, () => seg);
    expect(classifyTzPersist(true, over)).toEqual({
      action: 'reject',
      error: `too many segments (${MAX_TZ_SEGMENTS + 1} > ${MAX_TZ_SEGMENTS})`,
    });
  });
});
