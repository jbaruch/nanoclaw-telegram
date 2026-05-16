import { describe, it, expect } from 'vitest';

import {
  resolveCurrentTz,
  STALE_FOR_TZ_FALLBACK_HOURS,
  STALE_WARNING_HOURS,
} from './tz-resolver.js';
import { LocationRecord } from './types.js';

// #574 Phase 2 — pure-function tests for the resolver cascade. No DB
// access here; the resolver takes (now, latestLocation, segments,
// home_tz) and returns a deterministic envelope. The db.test.ts file
// covers the impure side (runTzHeartbeatAdvisory storing rows,
// cooldown stamps, etc.).

const NOW = new Date('2026-05-16T12:00:00Z');
const HOUR_MS = 60 * 60 * 1000;

// Krakow coordinates → Europe/Warsaw via geo-tz.
const KRAKOW: LocationRecord = {
  chat_jid: 'tg:-100',
  sender: '99001',
  message_id: 'm1',
  latitude: 50.0647,
  longitude: 19.945,
  source: 'live_update',
  recorded_at: '2026-05-16T12:00:00.000Z', // overridden per test via spread
  live_period: 28800,
};

function locationAtAge(
  hours: number,
  base: LocationRecord = KRAKOW,
): LocationRecord {
  return {
    ...base,
    recorded_at: new Date(NOW.getTime() - hours * HOUR_MS).toISOString(),
  };
}

describe('resolveCurrentTz — fresh-location path', () => {
  it('age < 4h: tz_from_coords drives the result', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: locationAtAge(1),
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.tz).toBe('Europe/Warsaw');
    expect(result.source).toBe('fresh_location');
    expect(result.warning).toBeNull();
    expect(result.latest_location_age_seconds).toBeCloseTo(3600, 0);
  });

  it('age just under 4h: still fresh path', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: locationAtAge(STALE_FOR_TZ_FALLBACK_HOURS - 0.01),
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.source).toBe('fresh_location');
    expect(result.tz).toBe('Europe/Warsaw');
  });

  it('age exactly STALE_FOR_TZ_FALLBACK_HOURS: walker path (strict-less-than boundary)', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: locationAtAge(STALE_FOR_TZ_FALLBACK_HOURS),
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.source).toBe('walker_stale_location');
    expect(result.tz).toBe('America/Chicago'); // walker fell back to home_tz
  });
});

describe('resolveCurrentTz — walker fallback paths', () => {
  it('no location ever → walker_no_location source', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: null,
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.source).toBe('walker_no_location');
    expect(result.tz).toBe('America/Chicago');
    expect(result.warning).toBeNull();
    expect(result.latest_location_age_seconds).toBeNull();
  });

  it('stale location + covering TripIt segment: walker resolves to segment tz', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: locationAtAge(6),
      segments: [
        { timezone: 'Europe/Berlin', from: '2026-05-12', to: '2026-05-19' },
      ],
      home_tz: 'America/Chicago',
    });
    expect(result.source).toBe('walker_stale_location');
    expect(result.tz).toBe('Europe/Berlin');
    expect(result.warning).toBeNull();
  });

  it('coord lookup failure on fresh pin: walker_coord_lookup_failed', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: locationAtAge(1, {
        ...KRAKOW,
        // Out-of-range latitude — tz_from_coords returns null, cascade
        // falls through to walker rather than propagating the failure.
        latitude: 999,
      }),
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.source).toBe('walker_coord_lookup_failed');
    expect(result.tz).toBe('America/Chicago'); // walker home_tz fallback
    expect(result.warning).toBeNull();
  });
});

describe('resolveCurrentTz — stale-warning surface', () => {
  it('age ≥ 12h: warning fires (resolver layer, cooldown applied by caller)', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: locationAtAge(13),
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.warning).toBe('stale_no_share');
    expect(result.source).toBe('walker_stale_location');
  });

  it('age 4-12h range: stale for cascade, no warning yet', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: locationAtAge(8),
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.warning).toBeNull();
    expect(result.source).toBe('walker_stale_location');
  });

  it('exactly STALE_WARNING_HOURS: warning fires (≥ boundary inclusive)', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: locationAtAge(STALE_WARNING_HOURS),
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.warning).toBe('stale_no_share');
  });

  it('no location ever → no warning (different from "owner went silent for 12h+")', () => {
    // Distinction matters: "we never saw the owner share" is a setup
    // state (cold orchestrator, first-deploy), not a we-lost-the-owner
    // alert worth nagging the chat about. The cascade falls to walker
    // silently.
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: null,
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.warning).toBeNull();
  });
});

describe('resolveCurrentTz — defensive parsing', () => {
  it('malformed recorded_at: treats as no-age, falls through to walker', () => {
    const result = resolveCurrentTz({
      now: NOW,
      latestLocation: { ...KRAKOW, recorded_at: 'not-a-date' },
      segments: null,
      home_tz: 'America/Chicago',
    });
    expect(result.latest_location_age_seconds).toBeNull();
    expect(result.source).toBe('walker_stale_location');
    expect(result.warning).toBeNull();
  });
});
