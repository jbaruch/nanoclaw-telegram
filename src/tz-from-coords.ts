/**
 * #574 Phase 2 — IANA timezone lookup from a lat/lng coordinate pair.
 *
 * Wraps the `geo-tz` package (timezone-boundary-builder data) into a
 * single-result helper. We pick `geo-tz` over `tz-lookup` despite the
 * size difference (~900 kB vs ~72 kB) because the published error
 * rate for `tz-lookup` on inhabited points is ~10 % (per the
 * `@photostructure/tz-lookup` maintainer's own README), and a 10 %
 * miss rate on the owner's own city would be a visible regression
 * vs the TripIt-walker baseline. Server-side install size doesn't
 * matter here; correctness does.
 *
 * Multi-zone returns from `geo-tz` happen only when a coordinate is
 * literally on a timezone boundary (rare for a phone-shared
 * location, which has sub-meter accuracy). When that does happen,
 * pick the first entry — `geo-tz` orders the multi-result array
 * deterministically by zone-name ascending, so the choice is stable
 * across calls; the cascade in `tz-resolver` accepts that one
 * deterministic answer rather than encoding "pick the most-likely
 * one" heuristics that would drift over time.
 *
 * If `geo-tz` returns an empty array (coordinate falls outside every
 * known boundary — should never happen since the dataset covers
 * land + territorial waters globally, but defensive), the caller
 * gets `null` and the cascade falls through to the walker. Same
 * for malformed input (non-finite numbers, out-of-range lat/lng).
 */

import { find as findTz } from 'geo-tz';

import { logger } from './logger.js';

// Warm the `geo-tz` shapefile cache at module load. `geo-tz` lazily
// reads the timezone-boundary-builder dataset (~900 kB of GeoJSON +
// quadtree indices) on the FIRST `find()` call, blocking the event
// loop while it streams the file in. The resolver's first reach into
// this module is from the 30-min heartbeat advisory's setInterval —
// without the warm-up the first tick after process start pays a
// hundreds-of-ms-to-seconds stall on the main thread. A dummy
// `findTz(0, 0)` call here forces the load at orchestrator startup
// where the latency is invisible. The Atlantic-Ocean coords are
// chosen so the warm-up doesn't pretend a real lookup happened.
//
// If `geo-tz` ever changes to load eagerly on import, this becomes a
// harmless no-op; if it gains an explicit prewarm API, swap to that.
findTz(0, 0);

const VALID_LATITUDE_MIN = -90;
const VALID_LATITUDE_MAX = 90;
const VALID_LONGITUDE_MIN = -180;
const VALID_LONGITUDE_MAX = 180;

export function tzFromCoords(
  latitude: number,
  longitude: number,
): string | null {
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    logger.warn(
      { latitude, longitude },
      'tzFromCoords: non-finite coordinate — falling through to walker',
    );
    return null;
  }
  if (
    latitude < VALID_LATITUDE_MIN ||
    latitude > VALID_LATITUDE_MAX ||
    longitude < VALID_LONGITUDE_MIN ||
    longitude > VALID_LONGITUDE_MAX
  ) {
    logger.warn(
      { latitude, longitude },
      'tzFromCoords: coordinate out of WGS-84 range — falling through to walker',
    );
    return null;
  }
  const zones = findTz(latitude, longitude);
  if (!Array.isArray(zones) || zones.length === 0) {
    // Should be unreachable in practice — `geo-tz` includes a sea
    // fallback. Log defensively in case the dataset ever drifts.
    logger.warn(
      { latitude, longitude },
      'tzFromCoords: geo-tz returned no zones — falling through to walker',
    );
    return null;
  }
  return zones[0];
}
