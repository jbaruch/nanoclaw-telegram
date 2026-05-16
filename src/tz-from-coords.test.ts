import { describe, it, expect } from 'vitest';

import { tzFromCoords } from './tz-from-coords.js';

// #574 Phase 2 — geo-tz wrapper tests. These exercise the validation
// + null-on-empty-result paths the resolver relies on for its
// `walker_coord_lookup_failed` fallback; the geo-tz data itself is
// upstream-tested and we don't re-verify every country here.

describe('tzFromCoords — happy paths', () => {
  it.each([
    [50.0647, 19.945, 'Europe/Warsaw'], // Krakow
    [36.0234, -86.782, 'America/Chicago'], // Nashville-area
    [51.5007, -0.1246, 'Europe/London'], // Big Ben
    [35.6762, 139.6503, 'Asia/Tokyo'], // Tokyo
    [-33.8688, 151.2093, 'Australia/Sydney'], // Sydney
  ])('resolves (%f, %f) to %s', (lat, lng, expected) => {
    expect(tzFromCoords(lat, lng)).toBe(expected);
  });
});

describe('tzFromCoords — defensive paths', () => {
  it('returns null for non-finite latitude', () => {
    expect(tzFromCoords(Number.NaN, 0)).toBeNull();
    expect(tzFromCoords(Number.POSITIVE_INFINITY, 0)).toBeNull();
  });

  it('returns null for non-finite longitude', () => {
    expect(tzFromCoords(0, Number.NaN)).toBeNull();
    expect(tzFromCoords(0, Number.NEGATIVE_INFINITY)).toBeNull();
  });

  it('returns null for latitude outside [-90, 90]', () => {
    expect(tzFromCoords(91, 0)).toBeNull();
    expect(tzFromCoords(-91, 0)).toBeNull();
    expect(tzFromCoords(999, 0)).toBeNull();
  });

  it('returns null for longitude outside [-180, 180]', () => {
    expect(tzFromCoords(0, 181)).toBeNull();
    expect(tzFromCoords(0, -181)).toBeNull();
    expect(tzFromCoords(0, 360)).toBeNull();
  });

  it('handles edge-of-range coordinates without rejection', () => {
    // Exactly 90 / -90 / 180 / -180 are valid WGS-84 limits, not
    // out-of-range. The validator's bounds-check is inclusive.
    // geo-tz may return a sea zone (Etc/GMT*) for these — we just
    // assert it returns SOMETHING (not null).
    expect(tzFromCoords(90, 0)).not.toBeNull();
    expect(tzFromCoords(-90, 0)).not.toBeNull();
    expect(tzFromCoords(0, 180)).not.toBeNull();
    expect(tzFromCoords(0, -180)).not.toBeNull();
  });
});
