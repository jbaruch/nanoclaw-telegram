/**
 * #574 Phase 2 — current-TZ resolver.
 *
 * Pure function: given a wall-clock `now`, the latest known owner
 * location, the cached TripIt segments timeline, and `home_tz`,
 * decide which IANA zone the owner is currently in.
 *
 * Cascade:
 *   age < STALE_FOR_TZ_FALLBACK_HOURS (4h)
 *     → tz_from_coords(location)               — fresh measurement
 *   else
 *     → walkTzSegments(segments, now, home_tz) — TripIt walker fallback
 *
 * When `tz_from_coords` returns null (out-of-range coords, missing
 * geo-tz data), we silently fall through to the walker rather than
 * propagate the failure — the resolver's job is to always return
 * SOMETHING usable, never to surface a tz-lookup miss as a TZ
 * regression for the user.
 *
 * Why we don't need IATA-airport lookup for mid-flight: when the
 * owner's coords are stale AND TripIt has an active flight segment
 * covering `now`, `walkTzSegments` already returns the arrival
 * segment's `timezone` field directly (#572 mid-transit fix). The
 * walker is the right answer for that case; no coords lookup
 * required.
 *
 * Warning is separate from the cascade choice — it fires whenever
 * the latest known location is older than STALE_WARNING_HOURS (12h),
 * regardless of which branch produced the tz. Caller decides whether
 * to actually surface the warning (cooldown / delivery channel).
 */

import { tzFromCoords } from './tz-from-coords.js';
import { TripitSegment, walkTzSegments } from './db-tz.js';
import { LocationRecord } from './types.js';

export const STALE_FOR_TZ_FALLBACK_HOURS = 4;
export const STALE_WARNING_HOURS = 12;

const HOUR_MS = 60 * 60 * 1000;

export type TzResolverSource =
  // age < 4h: fresh-pin path; `tz_from_coords` produced the answer
  | 'fresh_location'
  // 4h ≤ age: stale-pin path; walker produced the answer (mid-flight
  // returns arrival tz, covering segment returns segment tz, no
  // segment returns home_tz)
  | 'walker_stale_location'
  // No location ever recorded for this sender (or owner-id absent
  // from config); walker took over from the first tick
  | 'walker_no_location'
  // Coords were present and fresh but `tz_from_coords` couldn't
  // resolve them (out-of-range / data gap). Walker fallback fired.
  | 'walker_coord_lookup_failed';

export type TzWarningType =
  // Latest location is ≥12h old. Caller surfaces a one-shot chat
  // notification (with cooldown) asking the owner to re-share.
  'stale_no_share' | null;

export interface TzResolverInput {
  now: Date;
  latestLocation: LocationRecord | null;
  segments: readonly TripitSegment[] | null;
  home_tz: string;
}

export interface TzResolverResult {
  tz: string;
  source: TzResolverSource;
  // Seconds since `latestLocation.recorded_at`. `null` when no
  // location row exists for the sender, or its `recorded_at`
  // couldn't be parsed.
  latest_location_age_seconds: number | null;
  warning: TzWarningType;
}

export function resolveCurrentTz(input: TzResolverInput): TzResolverResult {
  const { now, latestLocation, segments, home_tz } = input;

  let ageSec: number | null = null;
  if (latestLocation) {
    const ts = Date.parse(latestLocation.recorded_at);
    if (Number.isFinite(ts)) {
      ageSec = (now.getTime() - ts) / 1000;
    }
  }

  const ageMs = ageSec !== null ? ageSec * 1000 : null;
  const isFresh =
    ageMs !== null && ageMs < STALE_FOR_TZ_FALLBACK_HOURS * HOUR_MS;
  const isStaleWarning =
    ageMs !== null && ageMs >= STALE_WARNING_HOURS * HOUR_MS;

  if (isFresh && latestLocation) {
    const coordTz = tzFromCoords(
      latestLocation.latitude,
      latestLocation.longitude,
    );
    if (coordTz !== null) {
      return {
        tz: coordTz,
        source: 'fresh_location',
        latest_location_age_seconds: ageSec,
        warning: null,
      };
    }
    // Coord lookup failed despite a fresh pin — fall through to walker
    // and record the path so the operator can investigate without
    // having to read warn-level logs.
    return {
      tz: walkTzSegments(segments, now, home_tz),
      source: 'walker_coord_lookup_failed',
      latest_location_age_seconds: ageSec,
      warning: null,
    };
  }

  return {
    tz: walkTzSegments(segments, now, home_tz),
    source: latestLocation ? 'walker_stale_location' : 'walker_no_location',
    latest_location_age_seconds: ageSec,
    warning: isStaleWarning ? 'stale_no_share' : null,
  };
}
