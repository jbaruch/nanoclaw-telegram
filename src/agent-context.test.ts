import { describe, it, expect } from 'vitest';

import type { LocationRecord } from './types.js';
import type { TripitSegment } from './db-tz.js';
import { buildAgentContext, formatAgentContextTag } from './agent-context.js';

const FIXED_NOW = new Date('2026-05-16T13:40:00.000Z');

function loc(overrides: Partial<LocationRecord> = {}): LocationRecord {
  return {
    chat_jid: 'group@g.us',
    sender: '12345',
    message_id: 'm1',
    latitude: 36.0234,
    longitude: -86.782,
    source: 'static',
    recorded_at: '2026-05-16T13:37:00.000Z', // 3 min ago vs FIXED_NOW
    ...overrides,
  };
}

function tripit(overrides: Partial<TripitSegment> = {}): TripitSegment {
  return {
    timezone: 'Europe/Warsaw',
    from: '2026-05-10T08:00:00Z',
    to: '2026-05-20T08:00:00Z',
    ...overrides,
  } as TripitSegment;
}

// --- buildAgentContext ---

describe('buildAgentContext — container_default fallback', () => {
  it('returns container_default when no location and no segments', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: null,
    });
    expect(ctx.timezone_source).toBe('container_default');
    expect(ctx.timezone).toBe('America/Chicago');
    // No location fields when source != location.
    expect(ctx.location_lat).toBeUndefined();
    expect(ctx.location_lng).toBeUndefined();
    expect(ctx.location_age_minutes).toBeUndefined();
  });

  it('treats empty segments array the same as null', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: [],
    });
    expect(ctx.timezone_source).toBe('container_default');
  });
});

describe('buildAgentContext — fresh location drives source=location', () => {
  it('fresh pin (<4h) marks location source and exposes coords', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: loc(), // Nashville, 3 min ago
      segments: null,
    });
    expect(ctx.timezone_source).toBe('location');
    expect(ctx.timezone).toBe('America/Chicago'); // Nashville is in Chicago tz
    expect(ctx.location_lat).toBe(36.0234);
    expect(ctx.location_lng).toBe(-86.782);
    expect(ctx.location_age_minutes).toBe(3);
  });

  it('fresh pin from a different tz produces that tz', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: loc({ latitude: 50.0647, longitude: 19.945 }), // Krakow
      segments: null,
    });
    expect(ctx.timezone_source).toBe('location');
    expect(ctx.timezone).toBe('Europe/Warsaw');
  });
});

describe('buildAgentContext — walker fallback', () => {
  it('stale location (>4h) + covering segment → source=segment', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: loc({
        recorded_at: '2026-05-15T00:00:00.000Z', // ~37h ago
        latitude: 36.0234,
        longitude: -86.782,
      }),
      segments: [tripit({ timezone: 'Europe/Warsaw' })],
    });
    expect(ctx.timezone).toBe('Europe/Warsaw');
    expect(ctx.timezone_source).toBe('segment');
    // location fields are NOT populated when source isn't `location`
    // — stale coords would be a misleading anchor for "here".
    expect(ctx.location_lat).toBeUndefined();
  });

  it('stale location + no covering segment → source=home_fallback', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: loc({
        recorded_at: '2026-05-15T00:00:00.000Z',
      }),
      segments: [
        tripit({
          // Segment ends before FIXED_NOW — walker falls back to home.
          from: '2026-05-01T00:00:00Z',
          to: '2026-05-10T00:00:00Z',
        }),
      ],
    });
    expect(ctx.timezone_source).toBe('home_fallback');
    expect(ctx.timezone).toBe('America/Chicago');
  });

  it('no location at all + covering segment → source=segment (no home equivalence)', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: [tripit({ timezone: 'Europe/Warsaw' })],
    });
    expect(ctx.timezone).toBe('Europe/Warsaw');
    expect(ctx.timezone_source).toBe('segment');
  });
});

describe('buildAgentContext — datetime fields', () => {
  it('utc_datetime is ISO-8601 with Z, no millis', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: null,
    });
    expect(ctx.utc_datetime).toBe('2026-05-16T13:40:00Z');
  });

  it('local_datetime is ISO-8601 with offset', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: null,
    });
    // 2026-05-16 13:40 UTC in Chicago (CDT, -05:00) = 08:40
    expect(ctx.local_datetime).toBe('2026-05-16T08:40:00-05:00');
    expect(ctx.local_date).toBe('2026-05-16');
  });

  it('local_date can differ from UTC date across midnight', () => {
    // 03:00 UTC on May 17 → still May 16 in Chicago (22:00 CDT)
    const earlyMorningUtc = new Date('2026-05-17T03:00:00Z');
    const ctx = buildAgentContext({
      now: earlyMorningUtc,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: null,
    });
    expect(ctx.local_date).toBe('2026-05-16');
    expect(ctx.weekday).toBe('Saturday');
  });

  it('weekday matches local frame', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: null,
    });
    expect(ctx.weekday).toBe('Saturday');
  });

  it('UTC timezone produces Z offset', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'UTC',
      containerTimezone: 'UTC',
      latestLocation: null,
      segments: null,
    });
    expect(ctx.local_datetime).toBe('2026-05-16T13:40:00Z');
    expect(ctx.timezone).toBe('UTC');
  });
});

// --- formatAgentContextTag ---

describe('formatAgentContextTag', () => {
  it('renders a self-closing <context> tag with all required attrs', () => {
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: null,
    });
    const tag = formatAgentContextTag(ctx);
    expect(tag.startsWith('<context ')).toBe(true);
    expect(tag.endsWith(' />')).toBe(true);
    expect(tag).toContain('utc_datetime="2026-05-16T13:40:00Z"');
    expect(tag).toContain('local_datetime="2026-05-16T08:40:00-05:00"');
    expect(tag).toContain('local_date="2026-05-16"');
    expect(tag).toContain('weekday="Saturday"');
    expect(tag).toContain('timezone="America/Chicago"');
    expect(tag).toContain('timezone_source="container_default"');
  });

  it('includes location_* attrs only when source=location', () => {
    const ctxLoc = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: loc(),
      segments: null,
    });
    const tagLoc = formatAgentContextTag(ctxLoc);
    expect(tagLoc).toContain('location_lat="36.0234"');
    expect(tagLoc).toContain('location_lng="-86.782"');
    expect(tagLoc).toContain('location_age_minutes="3"');

    const ctxFallback = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: null,
    });
    const tagFallback = formatAgentContextTag(ctxFallback);
    expect(tagFallback).not.toContain('location_lat');
    expect(tagFallback).not.toContain('location_lng');
    expect(tagFallback).not.toContain('location_age_minutes');
  });

  it('escapes XML special chars in tz / weekday values', () => {
    // Synthetic: a hostile-shaped timezone string should still emit
    // valid XML. The resolver/Intl path wouldn't produce these, but
    // the formatter is the wire-protocol boundary.
    const ctx = buildAgentContext({
      now: FIXED_NOW,
      homeTimezone: 'America/Chicago',
      containerTimezone: 'America/Chicago',
      latestLocation: null,
      segments: null,
    });
    const tampered = {
      ...ctx,
      timezone: 'a"b<c>d&e',
    };
    const tag = formatAgentContextTag(tampered);
    expect(tag).toContain('a&quot;b&lt;c&gt;d&amp;e');
    expect(tag).not.toContain('"a"b<c>d&e"');
  });
});
