import { describe, it, expect } from 'vitest';

import {
  formatLocalTime,
  isValidTimezone,
  normalizeScheduleTimezone,
  resolveTimezone,
} from './timezone.js';

// --- formatLocalTime ---

describe('formatLocalTime', () => {
  it('converts UTC to local time display', () => {
    // 2026-02-04T18:30:00Z in America/New_York (EST, UTC-5) = 1:30 PM
    const result = formatLocalTime(
      '2026-02-04T18:30:00.000Z',
      'America/New_York',
    );
    expect(result).toContain('1:30');
    expect(result).toContain('PM');
    expect(result).toContain('Feb');
    expect(result).toContain('2026');
  });

  it('handles different timezones', () => {
    // Same UTC time should produce different local times
    const utc = '2026-06-15T12:00:00.000Z';
    const ny = formatLocalTime(utc, 'America/New_York');
    const tokyo = formatLocalTime(utc, 'Asia/Tokyo');
    // NY is UTC-4 in summer (EDT), Tokyo is UTC+9
    expect(ny).toContain('8:00');
    expect(tokyo).toContain('9:00');
  });

  it('does not throw on invalid timezone, falls back to UTC', () => {
    expect(() =>
      formatLocalTime('2026-01-01T00:00:00.000Z', 'IST-2'),
    ).not.toThrow();
    const result = formatLocalTime('2026-01-01T12:00:00.000Z', 'IST-2');
    // Should format as UTC (noon UTC = 12:00 PM)
    expect(result).toContain('12:00');
    expect(result).toContain('PM');
  });
});

describe('isValidTimezone', () => {
  it('accepts valid IANA identifiers', () => {
    expect(isValidTimezone('America/New_York')).toBe(true);
    expect(isValidTimezone('UTC')).toBe(true);
    expect(isValidTimezone('Asia/Tokyo')).toBe(true);
    expect(isValidTimezone('Asia/Jerusalem')).toBe(true);
  });

  it('rejects invalid timezone strings', () => {
    expect(isValidTimezone('IST-2')).toBe(false);
    expect(isValidTimezone('XYZ+3')).toBe(false);
  });

  it('rejects empty and garbage strings', () => {
    expect(isValidTimezone('')).toBe(false);
    expect(isValidTimezone('NotATimezone')).toBe(false);
  });
});

describe('resolveTimezone', () => {
  it('returns the timezone if valid', () => {
    expect(resolveTimezone('America/New_York')).toBe('America/New_York');
  });

  it('falls back to UTC for invalid timezone', () => {
    expect(resolveTimezone('IST-2')).toBe('UTC');
    expect(resolveTimezone('')).toBe('UTC');
  });
});

describe('normalizeScheduleTimezone', () => {
  it('accepts undefined / null / empty as no per-task tz', () => {
    expect(normalizeScheduleTimezone(undefined, 'cron')).toEqual({
      action: 'accept',
      value: null,
    });
    expect(normalizeScheduleTimezone(null, 'cron')).toEqual({
      action: 'accept',
      value: null,
    });
    expect(normalizeScheduleTimezone('', 'cron')).toEqual({
      action: 'accept',
      value: null,
    });
  });

  it('accepts pinned IANA names on cron schedules', () => {
    expect(normalizeScheduleTimezone('America/New_York', 'cron')).toEqual({
      action: 'accept',
      value: 'America/New_York',
    });
    expect(normalizeScheduleTimezone('Asia/Tokyo', 'cron')).toEqual({
      action: 'accept',
      value: 'Asia/Tokyo',
    });
  });

  it("accepts the literal 'local' token on cron schedules (#456)", () => {
    // The token is what makes a row travel with the owner — the
    // scheduler resolves it against tz_state.current_tz at fire time.
    // Rejecting it at the IPC boundary (the pre-#456 behavior) made
    // the cadence-registry path the only writer that could produce
    // travel-anchored rows; agent-driven schedule_task calls that
    // followed the schedule-task SKILL classifier would 400 silently.
    expect(normalizeScheduleTimezone('local', 'cron')).toEqual({
      action: 'accept',
      value: 'local',
    });
  });

  it('rejects unknown strings on cron schedules', () => {
    expect(normalizeScheduleTimezone('NotATimezone', 'cron')).toEqual({
      action: 'reject-invalid',
    });
    expect(normalizeScheduleTimezone('IST-2', 'cron')).toEqual({
      action: 'reject-invalid',
    });
  });

  it('ignores any non-empty value on interval / once schedules', () => {
    // The column has no effect for non-cron rows; a stray value
    // would be a footgun if the row were later flipped to cron
    // without re-stating tz. Caller logs the warning and forces null.
    expect(normalizeScheduleTimezone('America/New_York', 'interval')).toEqual({
      action: 'ignore-non-cron',
    });
    expect(normalizeScheduleTimezone('local', 'once')).toEqual({
      action: 'ignore-non-cron',
    });
    expect(normalizeScheduleTimezone('NotATimezone', 'interval')).toEqual({
      action: 'ignore-non-cron',
    });
  });

  it('ignore takes precedence over reject-invalid on non-cron', () => {
    // A typo'd tz on a once/interval task should drop silently —
    // the field has no effect anyway, so failing the whole call would
    // be a pointless footgun.
    expect(normalizeScheduleTimezone('GarbageZone', 'once')).toEqual({
      action: 'ignore-non-cron',
    });
  });
});
