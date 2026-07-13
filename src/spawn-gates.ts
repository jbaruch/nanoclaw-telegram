import fs from 'fs';
import path from 'path';

import { logger } from './logger.js';

/**
 * Pre-spawn eligibility gates (#754).
 *
 * Some cadence skills only do useful work inside a bounded window, yet
 * the cadence registry fires them at a fixed high frequency (flight-assist
 * runs every 2 minutes → ~30 spawns/hour). Off-window every fire spawns a
 * container just for a precheck that returns "nothing to do" — the cost
 * lives in the spawn, not the precheck. A pre-spawn gate lets the host
 * decide, WITHOUT spawning, whether a fire is eligible.
 *
 * The gate is keyed on the skill the task invokes (`parseTaskSkill`), so
 * it is source-agnostic — it applies whether the row is a declarative
 * `cadence-registry` fire or an imperative `schedule-task` one. Only the
 * skills registered in `SPAWN_GATES` are gated; every other task spawns
 * unconditionally (the gate resolver returns `null`).
 */

export interface SpawnEligibility {
  /** `true` → spawn the container as normal; `false` → skip the spawn. */
  eligible: boolean;
  /** Human-readable rationale, surfaced in the skip log line. */
  reason: string;
}

/** A gate reads the group folder + current instant and rules on a fire. */
type SpawnGate = (groupDir: string, now: Date) => SpawnEligibility;

const HOUR_MS = 60 * 60 * 1000;
/** The window opens 24h before a trip's start date. */
const TRIP_WINDOW_LEAD_MS = 24 * HOUR_MS;
/**
 * A trip's `end` is a bare `YYYY-MM-DD` date; `Date.parse` pins it to
 * that day's UTC midnight. Add a day so the operator stays in-window
 * through the whole final day (v1 UTC date bounds — local end-of-day
 * can be layered in later if a timezone-sensitive tail matters).
 */
const TRIP_WINDOW_TRAIL_MS = 24 * HOUR_MS;

/** Basename of the travel state file the trip-window gate reads. */
const TRAVEL_DB_BASENAME = 'travel-db.json';

/**
 * Filesystem errnos the gate treats as expected "file-state" conditions
 * (mapped to fail-open/closed). Anything else caught around a `fs` call is
 * a programming bug, not a file state, and must propagate — mirrors the
 * explicit-errno narrowing in `checkTaskEvidence` (`task-scheduler.ts`).
 */
const EXPECTED_FS_ERRNOS = new Set([
  'ENOENT',
  'EACCES',
  'EISDIR',
  'ENOTDIR',
  'ELOOP',
  'ENAMETOOLONG',
]);

function isExpectedFsError(err: unknown): err is NodeJS.ErrnoException {
  const code = (err as NodeJS.ErrnoException).code;
  return (
    err instanceof Error &&
    typeof code === 'string' &&
    EXPECTED_FS_ERRNOS.has(code)
  );
}

/** Minimal shape of `travel-db.json` — only the fields the gate reads. */
interface TravelDbTrip {
  start?: unknown;
  end?: unknown;
}
interface TravelDbShape {
  // `unknown` on purpose: this is parsed from a container-writable file,
  // so the runtime value is validated (object vs array vs primitive)
  // before it is treated as a trip map.
  trips?: unknown;
}

/**
 * Resolve `<groupDir>/travel-db.json` with symlink containment. The group
 * folder is CONTAINER-writable while this runs HOST-side, so an agent
 * could plant a symlink at the travel-db path pointing outside the group
 * tree (the host `.env`, etc.). Resolve both ends through realpath and
 * require the target to stay inside the group folder — mirrors
 * `checkTaskEvidence` in `task-scheduler.ts`. Returns the safe absolute
 * path, or a typed failure the caller maps to a fail-open/closed branch.
 */
function resolveTravelDbPath(
  groupDir: string,
):
  | { ok: true; path: string }
  | { ok: false; kind: 'absent' | 'unreadable'; detail: string } {
  const candidate = path.join(groupDir, TRAVEL_DB_BASENAME);

  // Classify "truly absent" vs "present but unresolvable" WITHOUT
  // following the link: `realpathSync` on a broken symlink throws ENOENT,
  // the same code as a missing file, which would misclassify a planted
  // broken/looping symlink (the group folder is container-writable) as an
  // absent itinerary → fail CLOSED → suppress a possibly-active trip.
  // `lstatSync` sees the link node itself: ENOENT here means genuinely
  // nothing is there (→ absent, fail closed); a successful lstat means
  // SOMETHING exists, so any later resolution failure is "present but
  // unusable" (→ unreadable, fail open).
  try {
    fs.lstatSync(candidate);
  } catch (err: unknown) {
    // Only known filesystem errnos are file-state conditions; anything
    // else (a programming bug) propagates rather than being masked as a
    // file state.
    if (!isExpectedFsError(err)) throw err;
    if (err.code === 'ENOENT') {
      return {
        ok: false,
        kind: 'absent',
        detail: `${TRAVEL_DB_BASENAME} not found`,
      };
    }
    return {
      ok: false,
      kind: 'unreadable',
      detail: `cannot stat ${TRAVEL_DB_BASENAME} (${err.code})`,
    };
  }

  // Something exists at the path — resolve + contain it. Any failure now
  // (broken symlink target, loop, ...) is present-but-unusable, so it maps
  // to `unreadable` (fail open), never `absent`.
  try {
    const groupRoot = fs.realpathSync(groupDir);
    const resolved = fs.realpathSync(candidate);
    if (resolved !== groupRoot && !resolved.startsWith(groupRoot + path.sep)) {
      return {
        ok: false,
        kind: 'unreadable',
        detail: `${TRAVEL_DB_BASENAME} resolves outside the group folder (symlink?)`,
      };
    }
    return { ok: true, path: resolved };
  } catch (err: unknown) {
    // A broken symlink target / loop surfaces here as a known errno →
    // present-but-unusable (fail open). An unexpected error is a bug →
    // propagate.
    if (!isExpectedFsError(err)) throw err;
    return {
      ok: false,
      kind: 'unreadable',
      detail: `cannot resolve ${TRAVEL_DB_BASENAME} (${err.code})`,
    };
  }
}

/**
 * Does any trip in `travel-db.json` cover `now`, where the window runs
 * from 24h before `start` through the end of the `end` day?
 *
 * Fail-open vs fail-closed is deliberate and asymmetric, because the
 * gated skill (flight-assist) is safety-relevant — missing a gate change
 * mid-trip is worse than a few wasted off-trip spawns:
 *
 *   - File ABSENT → out of window. flight-assist derives its byAir polls
 *     from this same file; with no itinerary on disk there is nothing to
 *     act on, so suppressing the spawn is correct, not blinding.
 *   - File present but UNREADABLE / not JSON / wrong shape → in window
 *     (fail OPEN) + WARN. A corrupt file must never silently blind the
 *     operator while a trip is genuinely active.
 *   - File present and VALID → evaluate trips. In window iff a well-formed
 *     trip covers now. A trip whose dates don't parse is skipped (it can't
 *     be meaningfully "active") and logged.
 */
function tripWindowGate(groupDir: string, now: Date): SpawnEligibility {
  const resolved = resolveTravelDbPath(groupDir);
  if (!resolved.ok) {
    if (resolved.kind === 'absent') {
      return {
        eligible: false,
        reason: `no travel itinerary on disk (${resolved.detail}) — nothing for a windowed skill to act on`,
      };
    }
    logger.warn(
      { groupDir, detail: resolved.detail },
      '[spawn-gate] travel-db unreadable — failing OPEN so an active trip is never blinded (#754)',
    );
    return {
      eligible: true,
      reason: `travel-db unreadable (${resolved.detail}) — failing open`,
    };
  }

  let parsed: TravelDbShape;
  try {
    parsed = JSON.parse(
      fs.readFileSync(resolved.path, 'utf-8'),
    ) as TravelDbShape;
  } catch (err: unknown) {
    // Two expected shapes mean "can't trust this file" → fail open so an
    // active trip is never blinded: a JSON `SyntaxError`, or a known fs
    // errno from the read (perms/dir race after the realpath check). Any
    // other error is a programming bug and propagates.
    if (!(err instanceof SyntaxError) && !isExpectedFsError(err)) throw err;
    logger.warn(
      { path: resolved.path, detail: err.message },
      '[spawn-gate] travel-db unreadable/not JSON — failing OPEN (#754)',
    );
    return {
      eligible: true,
      reason: 'travel-db unreadable/not valid JSON — failing open',
    };
  }

  const trips: unknown = parsed?.trips;
  if (trips === undefined || trips === null) {
    // A well-formed travel-db always carries a `trips` object (empty when
    // there is no itinerary). A missing key is an unexpected/truncated
    // shape → fail OPEN rather than blind a possibly-active trip.
    logger.warn(
      { path: resolved.path },
      '[spawn-gate] travel-db has no `trips` key — failing OPEN (#754)',
    );
    return {
      eligible: true,
      reason: 'travel-db missing trips key — failing open',
    };
  }
  if (typeof trips !== 'object' || Array.isArray(trips)) {
    // `trips` present but the wrong type (array / primitive). Iterating a
    // non-record as trip records would be nonsense, so treat it as a
    // corrupt shape and fail OPEN.
    logger.warn(
      {
        path: resolved.path,
        tripsType: Array.isArray(trips) ? 'array' : typeof trips,
      },
      '[spawn-gate] travel-db `trips` has the wrong shape — failing OPEN (#754)',
    );
    return {
      eligible: true,
      reason: 'travel-db trips has wrong shape — failing open',
    };
  }

  // A valid (possibly empty) trip map. An empty map falls through the loop
  // to the out-of-window return below — the intended steady state when no
  // itinerary is active.
  const nowMs = now.getTime();
  let skipped = 0;
  for (const [tripId, trip] of Object.entries(
    trips as Record<string, TravelDbTrip>,
  )) {
    const startMs =
      typeof trip?.start === 'string' ? Date.parse(trip.start) : NaN;
    const endMs = typeof trip?.end === 'string' ? Date.parse(trip.end) : NaN;
    if (Number.isNaN(startMs) || Number.isNaN(endMs)) {
      skipped++;
      continue;
    }
    const windowStart = startMs - TRIP_WINDOW_LEAD_MS;
    const windowEnd = endMs + TRIP_WINDOW_TRAIL_MS;
    if (nowMs >= windowStart && nowMs < windowEnd) {
      return {
        eligible: true,
        reason: `in trip window "${tripId}" (${String(trip.start)}..${String(trip.end)}, 24h lead)`,
      };
    }
  }

  if (skipped > 0) {
    logger.warn(
      { path: resolved.path, skipped },
      '[spawn-gate] skipped trips with unparseable start/end dates while evaluating trip window (#754)',
    );
  }

  return {
    eligible: false,
    reason: 'no trip window covers now — out of window',
  };
}

/**
 * Registry of pre-spawn gates keyed by the exact skill identifier
 * `parseTaskSkill` returns (the value inside `Skill(skill: "…")`).
 */
const SPAWN_GATES: Record<string, SpawnGate> = {
  // `Skill(skill: "tessl__flight-assist")` → parseTaskSkill returns this.
  'tessl__flight-assist': tripWindowGate,
};

/**
 * Resolve and evaluate the pre-spawn gate for a task's skill.
 *
 * Returns `null` when the skill has no registered gate (→ spawn as
 * normal), otherwise the eligibility verdict. `groupDir` must be the
 * already-validated host path for the task's group folder.
 */
export function evaluateSpawnGate(
  skill: string | undefined,
  groupDir: string,
  now: Date,
): SpawnEligibility | null {
  if (!skill) return null;
  const gate = SPAWN_GATES[skill];
  if (!gate) return null;
  return gate(groupDir, now);
}
