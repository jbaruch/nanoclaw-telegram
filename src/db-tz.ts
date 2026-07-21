// Timezone and location accessors (#751 seam 8, extracted verbatim
// from src/db.ts).
//
// Covers the `tz_state`, `locations`, and `follow_me_tasks` tables:
// the #574 location cascade (`storeLocation` / owner-location lookup),
// the tz_state reader/writer pair (`getCurrentTz`, TripIt segment
// walking, `applyTripitSegmentsToTzState`), the heartbeat advisory
// (`runTzHeartbeatAdvisory`), and the follow-me pending-run lock
// helpers. Mutually recursive with `tz-resolver.ts` by design —
// `walkTzSegments` feeds the resolver, the resolver's
// `resolveCurrentTz` feeds the advisory; both references resolve at
// call time, never at module-load time.
import { SqliteError } from 'better-sqlite3';

import { db } from './db-connection.js';
import { logger } from './logger.js';
import { resolveCurrentTz, STALE_WARNING_HOURS } from './tz-resolver.js';
import { LocationRecord } from './types.js';

/**
 * #584 — SQLite error codes the `onTzFlipped` callback catches as
 * recoverable contention. SQLITE_BUSY / SQLITE_LOCKED can fire under
 * WAL contention with the orchestrator's other writers; the canonical
 * `tz_state` UPDATE has already landed in the same transaction, and
 * the next scheduler tick will retry the recompute against the
 * stored `current_tz`. Every other error (programming bug, persistent
 * DB failure, malformed schema) propagates. Mirrors the same set
 * used in `src/index.ts` around `runTzHeartbeatAdvisory`.
 */
const TRANSIENT_SQLITE_CODES: ReadonlySet<string> = new Set([
  'SQLITE_BUSY',
  'SQLITE_LOCKED',
]);

/**
 * Append a location row (#574 Phase 3). Always INSERT, never UPDATE —
 * live-location updates carry the same `message_id` as the initial
 * share, but each tick is a fresh observation worth persisting (the
 * Phase 2 resolver's "most-recent wins" rule operates on `recorded_at`,
 * not on per-message_id state).
 *
 * Callers don't need to deduplicate; Telegram itself only fires
 * `edited_message:location` when the location actually changed, so
 * the natural rate is bounded by movement, not by polling cadence.
 */
export function storeLocation(record: LocationRecord): void {
  db.prepare(
    `INSERT INTO locations (chat_jid, sender, message_id, latitude, longitude, accuracy_m, source, recorded_at, live_period) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    record.chat_jid,
    record.sender,
    record.message_id,
    record.latitude,
    record.longitude,
    record.accuracy_m ?? null,
    record.source,
    record.recorded_at,
    record.live_period ?? null,
  );
}

/**
 * Most-recent location for a given sender across every chat. Used by
 * the #574 Phase 2 TZ resolver as the canonical "where is the owner"
 * query — `sender` is the owner's channel-specific user id (Telegram
 * numeric id; the orchestrator resolves it once via
 * `ASSISTANT_OWNER_TG_USER_ID`).
 *
 * Returns `null` when no rows match (first deploy, owner-id wrong,
 * etc.) — caller falls through to the TripIt walker per the Phase 2
 * cascade.
 */
export function getLatestLocationForSender(
  sender: string,
): LocationRecord | null {
  // Stable tie-breaker on `id DESC` matters because `recorded_at` is
  // second-resolution (Telegram `date` / `edit_date` are both Unix
  // timestamps in seconds). When two ticks land in the same second
  // — common during a fast-moving live share — sorting only by
  // recorded_at gives SQLite implementation-defined ordering and the
  // resolver can flap between two coords for the same instant. `id`
  // is monotonic INTEGER PRIMARY KEY AUTOINCREMENT, so the composite
  // sort is deterministic; the `idx_locations_sender_time` index
  // still satisfies the leading columns and `id DESC` is a small
  // per-group sort after the index scan.
  const row = db
    .prepare(
      `SELECT chat_jid, sender, message_id, latitude, longitude, accuracy_m, source, recorded_at, live_period
       FROM locations WHERE sender = ?
       ORDER BY recorded_at DESC, id DESC LIMIT 1`,
    )
    .get(sender) as
    | {
        chat_jid: string;
        sender: string;
        message_id: string;
        latitude: number;
        longitude: number;
        accuracy_m: number | null;
        source: string;
        recorded_at: string;
        live_period: number | null;
      }
    | undefined;
  if (!row) return null;
  return {
    chat_jid: row.chat_jid,
    sender: row.sender,
    message_id: row.message_id,
    latitude: row.latitude,
    longitude: row.longitude,
    accuracy_m: row.accuracy_m,
    source: row.source as LocationRecord['source'],
    recorded_at: row.recorded_at,
    live_period: row.live_period,
  };
}

// Highest tz_state schema_version this reader knows how to interpret.
// Per `coding-policy: stateful-artifacts`, readers that observe a higher
// `schema_version` must treat the row as "no usable prior state" rather
// than guess. Bumped to 2 by state-012 (#542) when the host took over
// as the writer of `tz_state` and added the `segments` column; bumped
// to 3 by state-013 (jbaruch/nanoclaw-admin#229) when `walkTzSegments`
// gained per-segment ISO-datetime resolution (consuming the upstream
// `reclaim-tripit-timezones-sync#13` parser fix); bumped to 4 by
// state-015 (#574 Phase 2) when the row gained
// `last_stale_warning_at` for the stale-location warning cooldown.
// The state-NNN migrations run before this gate is consulted, so any
// row that existed at a prior version has already been bumped by the
// time `getCurrentTz` reads.
// Exported for the tz-state JSON importer in `db-json-migrations.ts` (#751).
export const SUPPORTED_TZ_STATE_SCHEMA_VERSION = 4;

/**
 * Test-only helper: seed a `follow_me_tasks` row directly. The
 * production writer is the agent-side `task-tz-sync` skill (and the
 * other follow-me skills' Phase C / Phase D updates); this shortcut
 * lets the host-side `clearStalePendingRunAt` /
 * `getActivePendingRunAtNames` tests exercise the cleanup helpers
 * without spinning up the full agent stack.
 */
export function _seedFollowMeTaskForTests(args: {
  name: string;
  localTime?: string;
  scheduleValue?: string;
  lastRunDate?: string | null;
  pendingRunAt?: string | null;
}): void {
  db.prepare(
    `INSERT INTO follow_me_tasks
       (name, local_time, schedule_value, last_run_date, pending_run_at)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(name) DO UPDATE SET
       local_time     = excluded.local_time,
       schedule_value = excluded.schedule_value,
       last_run_date  = excluded.last_run_date,
       pending_run_at = excluded.pending_run_at,
       updated_at     = CURRENT_TIMESTAMP`,
  ).run(
    args.name,
    args.localTime ?? '08:00',
    args.scheduleValue ?? '0 13 * * *',
    args.lastRunDate ?? null,
    args.pendingRunAt ?? null,
  );
}

/**
 * Test-only helper: seed the singleton `tz_state` row directly. The
 * production writer is the host-side `applyTripitSegmentsToTzState`
 * (after every `sync_tripit` run, see #542); this shortcut lets
 * `getCurrentTz` and the heartbeat-advisory walker tests exercise
 * read paths without spinning up the full TripIt sync.
 *
 * `segments` defaults to NULL because most read-path tests only care
 * about `current_tz`. Tests that exercise the heartbeat-advisory
 * walker opt in by passing a JSON-stringified payload. The default
 * `schemaVersion` matches `SUPPORTED_TZ_STATE_SCHEMA_VERSION` so
 * readers don't reject the seeded row as unfamiliar; tests that need
 * to verify the gate's "unfamiliar version" branch pass an explicit
 * higher value.
 */
export function _seedTzStateForTests(args: {
  currentTz: string;
  homeTz?: string;
  schedulerTz?: string | null;
  segments?: string | null;
  schemaVersion?: number;
  // ISO-8601 UTC; null clears the stamp (the resolver treats null as
  // "cooldown expired, may fire on next stale check"). Defaults to
  // null so existing tests that don't care about cooldown behave
  // as if no warning has ever fired.
  lastStaleWarningAt?: string | null;
}): void {
  db.prepare(
    `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, segments, schema_version, last_stale_warning_at)
       VALUES (1, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       current_tz            = excluded.current_tz,
       home_tz               = excluded.home_tz,
       scheduler_tz          = excluded.scheduler_tz,
       segments              = excluded.segments,
       schema_version        = excluded.schema_version,
       last_stale_warning_at = excluded.last_stale_warning_at`,
  ).run(
    args.currentTz,
    args.homeTz ?? args.currentTz,
    args.schedulerTz ?? null,
    args.segments ?? null,
    args.schemaVersion ?? SUPPORTED_TZ_STATE_SCHEMA_VERSION,
    args.lastStaleWarningAt ?? null,
  );
}

/**
 * Read `current_tz` from the singleton `tz_state` row. Returns null if
 * the row is absent (pre-state-010 install / migration not yet run) or
 * if `schema_version` is unfamiliar to this reader.
 *
 * Used by `task-scheduler.ts:computeNextRunDetailed` to resolve rows
 * declared with `schedule_timezone = 'local'` (#456): the cron is
 * evaluated against the owner's current zone at fire time without
 * mutating the row's `schedule_value`.
 */
export function getCurrentTz(): string | null {
  const row = db
    .prepare('SELECT current_tz, schema_version FROM tz_state WHERE id = 1')
    .get() as { current_tz: string; schema_version: number } | undefined;
  if (!row) return null;
  if (row.schema_version !== SUPPORTED_TZ_STATE_SCHEMA_VERSION) {
    logger.warn(
      {
        observed: row.schema_version,
        supported: SUPPORTED_TZ_STATE_SCHEMA_VERSION,
      },
      'tz_state schema_version unfamiliar — treating as no usable prior state',
    );
    return null;
  }
  return row.current_tz;
}

/**
 * Single segment shape as emitted by `sync_tripit`'s stdout JSON
 * (`result.segments` in `reclaim-tripit-timezones-sync/sync.mjs`).
 *
 * `from` / `to` are date-only `YYYY-MM-DD` strings — the original
 * shape from before reclaim-tripit-timezones-sync#13. Lexicographic
 * comparison of these strings is equivalent to date comparison, so
 * the walker can use plain string `<=`/`<` when datetime fields are
 * absent.
 *
 * `from_dt` / `to_dt` (jbaruch/nanoclaw-admin#229,
 * reclaim-tripit-timezones-sync#13) are ISO 8601 UTC strings derived
 * from the same underlying `Date` objects in `lib/tripit.mjs` — they
 * preserve flight arrival / lodging check-in wall-clock instead of
 * collapsing to UTC midnight. Optional because (a) a row written by
 * a pre-deploy `applyTripitSegmentsToTzState` carries the old shape
 * until the next `sync_tripit` rewrites it, and (b) the upstream
 * parser may legitimately omit them on edge paths.
 */
export interface TripitSegment {
  timezone: string;
  from: string;
  to: string;
  from_dt?: string;
  to_dt?: string;
  label?: string;
}

/**
 * Pure walker: pick the IANA zone name covering `now` from a segments
 * timeline, falling back to `homeTz` if no segment covers it.
 *
 * Match rule: per-segment, prefer ISO-datetime resolution when the
 * segment carries `from_dt` / `to_dt` (post-#229); fall through to
 * date-only `from` / `to` otherwise. The match keeps strict
 * inequality on the right edge in both shapes — a traveler whose
 * return-flight arrival is `to_dt` should see the next segment (or
 * `homeTz`) take over the instant the flight lands, not hold the
 * destination zone for the rest of that second.
 *
 * Why per-segment, not array-wide: a single segments array can mix
 * shapes across deploys — a row written by a pre-deploy
 * `applyTripitSegmentsToTzState` is date-only until the next
 * `sync_tripit` rewrites it; in the transient between the upstream
 * parser bump (rtts#13) and the next sync, an array can also carry
 * one shape exclusively. Per-segment fallback handles every case
 * with one walk.
 *
 * `from_dt` and `to_dt` are both required for the datetime path —
 * if only one is present (malformed payload), the segment falls
 * back to date-only on its own row rather than the walker emitting
 * a one-sided datetime compare against the missing end.
 *
 * String compare on ISO 8601 UTC strings is equivalent to chrono
 * compare, the same property the date-only path relies on. Both
 * shapes use `now.toISOString()` (`...T...Z`) or its 10-char prefix
 * for `today`, so each path is internally consistent.
 *
 * Edge cases (covered by unit tests):
 *   - Empty / non-array `segments` → `homeTz` (silent fallback).
 *   - Inter-segment gap, transit (the most-recent prev-ended segment
 *     is NOT `homeTz`) → the ARRIVAL segment's tz (#571). Segments
 *     are built from lodging / ground stays; flight legs are gaps by
 *     design. When the heartbeat fires mid-flight between two foreign
 *     stays, the user is travelling TOWARDS the next booked stay, so
 *     returning that segment's tz lets the morning brief / scheduler
 *     think in the arrival zone the user is about to land in.
 *     `home_tz` was wrong for this case: it's often physically
 *     impossible (mid-Atlantic on a Europe-bound leg) and the user
 *     can't act on a "you're home" signal while in the air.
 *   - Inter-segment gap, home-bounded (the most-recent prev-ended
 *     segment IS `homeTz`) → `homeTz` (#573). The #571 arrival-tz
 *     rule over-corrected this case: an inbound home segment IS the
 *     "you're home now" signal, so the gap that follows is the user
 *     sitting at home between trips, not in motion. Without this
 *     branch the walker ships a forward-looking foreign tz for the
 *     entire dwell-at-home window. Detection uses the MOST RECENT
 *     prev-ended segment's tz (chronological-order invariant: the
 *     last segment that classifies as prev-ended in iteration order
 *     wins, mirroring the first-wins rule the arrival branch uses).
 *     The symmetric case (`nextSegTz === homeTz` — foreign trip
 *     followed by a future home segment that hasn't started yet)
 *     needs no special branch: the arrival-tz fallback already
 *     returns `homeTz` because `nextSegTz` IS `homeTz`.
 *   - Before the first segment (all segments are future) → `homeTz`.
 *   - After the last segment (no future segment remaining) →
 *     `homeTz`. This is the post-trip case: the user has returned
 *     and the next `sync_tripit` will eventually clear the row, but
 *     until then home is the right answer.
 *   - `from === to` (date or datetime) → never matches (degenerate
 *     segment); skipped.
 *   - Empty / non-string `timezone` → segment skipped (the upstream
 *     drops these via `result.segments` already, but defend against
 *     a malformed payload anyway).
 *   - Overlapping segments → first match wins (matches the natural
 *     "segments are produced in chronological order, lodging-primary
 *     first" upstream invariant).
 *   - `to_dt === now.toISOString()` (or `to === todayUtc` on the
 *     date-only path) → segment ends now; the next segment (or the
 *     gap / home fallback) takes over.
 *   - Mixed array (some segments with `from_dt`/`to_dt`, some
 *     without) → each segment uses its own shape; the gap-fallback
 *     classification (ended-before-now / starts-after-now) also uses
 *     each segment's own shape against `nowIso` / `todayUtc`.
 */
export function walkTzSegments(
  segments: readonly TripitSegment[] | null | undefined,
  now: Date,
  homeTz: string,
): string {
  if (!Array.isArray(segments) || segments.length === 0) return homeTz;
  const nowIso = now.toISOString();
  const todayUtc = nowIso.slice(0, 10);

  // #571 / #573 — classify each non-covering segment as "ended
  // before now" or "starts after now". When BOTH classes appear in
  // the same walk, the user is in an inter-segment gap; the right
  // answer depends on what the user did last:
  //   - prev-ended segment was a foreign zone → mid-transit toward
  //     the next booked stay → return the arrival (first-future) tz
  //     (#571).
  //   - prev-ended segment was the home zone → user landed at home
  //     and is sitting between trips → return `homeTz` (#573).
  // Both branches rely on the lodging-primary chronological-order
  // invariant: `nextSegTz` keeps the FIRST future segment's tz
  // (first wins), and `prevEndedTz` keeps the LAST prev-ended
  // segment's tz (last wins = most recent under chronological
  // iteration). No cross-shape datetime-vs-date comparison is
  // needed.
  let prevEndedTz: string | null = null;
  let nextSegTz: string | null = null;

  for (const seg of segments) {
    if (
      typeof seg?.timezone !== 'string' ||
      seg.timezone.length === 0 ||
      typeof seg.from !== 'string' ||
      typeof seg.to !== 'string'
    ) {
      continue;
    }
    // Datetime path: both `from_dt` and `to_dt` must be present and
    // strings. Partial-shape segments fall through to the date-only
    // path on their own row.
    if (typeof seg.from_dt === 'string' && typeof seg.to_dt === 'string') {
      // Degenerate (`from_dt === to_dt`) segments are skipped per the
      // function's documented contract — the inside-check already
      // rejects them via `nowIso < seg.to_dt`, but the gap-fallback
      // classifier (#571) would otherwise feed them into
      // `hasPrevEndedSeg` / `nextSegTz` and let a malformed row drive
      // the resolution. Skip BEFORE the gap classifier to keep
      // degenerate segments invisible to every code path here.
      if (seg.from_dt === seg.to_dt) continue;
      if (seg.from_dt <= nowIso && nowIso < seg.to_dt) return seg.timezone;
      // Strict `<` for the prev-ended classification (mirror in the
      // date-only path below): a segment whose `to_dt` exactly
      // equals `now` is at the right-edge boundary and stays out of
      // the gap-fallback's "previous" pool. The mixed-shape case
      // (#229 regression guard) is the load-bearing reason: a
      // legacy date-only segment with `to: 2026-05-12` and a
      // datetime future segment at `from_dt: 2026-05-12T13:30Z`
      // would otherwise let the walker return the future zone at
      // `00:30Z` (departure morning) — same early-flip shape the
      // #229 per-segment-datetime fix existed to prevent.
      if (seg.to_dt < nowIso) {
        prevEndedTz = seg.timezone;
      } else if (nowIso < seg.from_dt) {
        if (nextSegTz === null) nextSegTz = seg.timezone;
      }
      continue;
    }
    // Date-only path — same degenerate-segment guard + strict-`<`
    // right edge for prev-ended classification. A date-only segment
    // whose `to` equals today's UTC date is "ending today" but the
    // user is still in it until UTC midnight rolls; treating it as
    // already-past would flip mid-day for any later segment.
    if (seg.from === seg.to) continue;
    if (seg.from <= todayUtc && todayUtc < seg.to) return seg.timezone;
    if (seg.to < todayUtc) {
      prevEndedTz = seg.timezone;
    } else if (todayUtc < seg.from) {
      if (nextSegTz === null) nextSegTz = seg.timezone;
    }
  }

  // In-gap: previous-ended AND future segment both exist. Distinguish
  // home-bounded gaps (user landed home, sitting between trips) from
  // transit gaps (user is mid-flight toward the next booked stay).
  if (prevEndedTz !== null && nextSegTz !== null) {
    return prevEndedTz === homeTz ? homeTz : nextSegTz;
  }
  // Before-first or after-last: fall back to home_tz.
  return homeTz;
}

/**
 * #542 — Persist `sync_tripit` stdout's `segments[]` onto the
 * singleton `tz_state` row and recompute `current_tz` from the
 * segment timeline (or fall back to `home_tz` for gaps between
 * trips).
 *
 * Writes `schema_version = SUPPORTED_TZ_STATE_SCHEMA_VERSION` (3
 * post-jbaruch/nanoclaw-admin#229; was 2 between #542 and #229).
 * The constant is the source of truth so every writer in this file
 * picks up future state-NNN bumps automatically. The row
 * MUST already exist — `home_tz` is NOT NULL on `tz_state` and isn't
 * derivable from the TripIt payload, so the host can't synthesize a
 * row from scratch. This is by design: `tz_state.home_tz` is set by
 * the historical JSON migration / first-time setup and is never
 * touched by `sync_tripit`. If the row is absent, the helper logs a
 * warning and exits without writing — the caller's run is still
 * counted as a successful TripIt sync (segments were fetched and
 * parsed) so the user gets the upstream success reaction; the
 * follow-up heartbeat advisory will pick up the segments on the next
 * tick once the row exists.
 *
 * Returns the resolved `{ prev, next, changed }` so the caller can
 * audit-log the flip — the `sync_tripit` call site logs at info /
 * warn levels on the orchestrator's structured logger; there is no
 * separate audit-log table per `coding-policy: stateful-artifacts`'s
 * "what counts" section (state files vs. orchestrator logs).
 */
export function applyTripitSegmentsToTzState(
  stdoutJson: { segments?: readonly TripitSegment[] | null } | null,
  now: Date = new Date(),
  onTzFlipped?: (prev: string, next: string) => void,
): { prev: string | null; next: string | null; changed: boolean } {
  const segments =
    stdoutJson && Array.isArray(stdoutJson.segments) ? stdoutJson.segments : [];

  const row = db
    .prepare('SELECT current_tz, home_tz FROM tz_state WHERE id = 1')
    .get() as { current_tz: string; home_tz: string } | undefined;
  if (!row) {
    logger.warn(
      { segmentCount: segments.length },
      'applyTripitSegmentsToTzState: tz_state row missing — skipping (home_tz must be seeded by JSON migration / first-time setup before sync_tripit can persist segments)',
    );
    return { prev: null, next: null, changed: false };
  }

  const next = walkTzSegments(segments, now, row.home_tz);
  const segmentsJson = JSON.stringify(segments);

  // Plain UPDATE rather than UPSERT: the row-existence check above
  // already proved the singleton is present, and an UPSERT's INSERT
  // arm would clobber `scheduler_tz` (which the JSON migration
  // legitimately populates as `informational only`) on a hypothetical
  // concurrent-delete race. Plain UPDATE on a missing row is a
  // no-op which is the safer failure mode.
  //
  // `schema_version` is bound through `SUPPORTED_TZ_STATE_SCHEMA_VERSION`
  // (rather than a SQL literal) so a future state-NNN bump only has to
  // touch the constant — every writer in this file picks up the new
  // value automatically.
  db.prepare(
    `UPDATE tz_state
        SET current_tz     = ?,
            segments       = ?,
            schema_version = ?
      WHERE id = 1`,
  ).run(next, segmentsJson, SUPPORTED_TZ_STATE_SCHEMA_VERSION);

  const changed = row.current_tz !== next;
  if (changed) {
    logger.info(
      { prev: row.current_tz, next, segmentCount: segments.length },
      'tz_state.current_tz flipped via sync_tripit segment walk (#542)',
    );
    // #584 — invoke the recompute hook AFTER the UPDATE landed, so a
    // reader picking up the new current_tz sees the new value. Narrow
    // the catch to transient SQLite contention codes only
    // (SQLITE_BUSY / SQLITE_LOCKED) — those are genuinely recoverable
    // against the orchestrator's other WAL writers and the next
    // scheduler tick will re-anchor `next_run` against the now-canonical
    // `current_tz`. Every other error (programming bug, persistent DB
    // failure, malformed schema) propagates out per
    // `coding-policy: error-handling`. Mirrors the narrowing pattern
    // in `src/index.ts` around `runTzHeartbeatAdvisory`.
    if (onTzFlipped) {
      try {
        onTzFlipped(row.current_tz, next);
      } catch (err) {
        if (
          !(err instanceof SqliteError) ||
          !TRANSIENT_SQLITE_CODES.has(err.code)
        ) {
          throw err;
        }
        logger.warn(
          {
            err: err.message,
            code: err.code,
            prev: row.current_tz,
            next,
          },
          'applyTripitSegmentsToTzState: onTzFlipped transient SQLite contention — tz_state write landed, next scheduler tick will recompute',
        );
      }
    }
  }
  return { prev: row.current_tz, next, changed };
}

/**
 * #542 / #574 Phase 2 — Heartbeat advisory. Re-evaluates `current_tz`
 * against (a) the owner's most-recent location row, falling through to
 * (b) the cached TripIt segments walker. Called from a 30-min
 * `setInterval` in `src/index.ts`; on flip, the caller sends a chat
 * message via the main group's channel. On stale-location warning,
 * the caller sends a "please re-share" notice — bounded by a 12h
 * cooldown stored in `tz_state.last_stale_warning_at`.
 *
 * `ownerSenderId` is the channel-specific owner identifier (Telegram
 * numeric user_id today; the orchestrator resolves it once from
 * `ASSISTANT_OWNER_TG_USER_ID` and threads it in). When null /
 * undefined / unset, the location-first path is skipped and the
 * function falls back to the pre-Phase-2 walker-only behaviour
 * (zero-config installs keep working unchanged).
 *
 * Returns `{ flip, warningToFire }`:
 *   - `flip` carries the `{ prev, next }` zone change when the
 *     computed tz differs from `current_tz`; null otherwise.
 *   - `warningToFire` is `'stale_no_share'` only when the resolver
 *     reports the latest location is ≥12 h old AND the cooldown
 *     window has elapsed since the last warning; null otherwise.
 *     The DB column `last_stale_warning_at` is updated atomically
 *     with the decision so a concurrent advisory can't double-fire.
 *
 * Skip paths (return `{ flip: null, warningToFire: null }`):
 *   - tz_state row missing
 *   - tz_state row at an unfamiliar schema_version (warned + skipped)
 *   - tz_state.segments malformed JSON (warned + skipped; recovers
 *     on next sync_tripit)
 *
 * Lock-step contract with state-015: this function reads / writes
 * `last_stale_warning_at`. A row that hasn't been migrated to v4
 * trips the schema_version gate above and skips, so the column
 * absence is structurally impossible inside the hot path.
 */
export interface TzAdvisoryResult {
  flip: { prev: string; next: string } | null;
  warningToFire: 'stale_no_share' | null;
}

/**
 * Read-only snapshot of `tz_state` for the `<context>`-tag builder
 * in `agent-context.ts`. Returns `home_tz` plus the decoded
 * `segments` array (or `null` when the column is empty / malformed
 * JSON). Does NOT mutate any state, NOT write warnings to the log,
 * and NOT run the resolver — those side effects belong to
 * `runTzHeartbeatAdvisory`, not the per-prompt context builder.
 *
 * Returns `null` when the singleton row is missing entirely (no
 * `tz_state` seeded yet) so the caller can fall through to the
 * container-default context shape per `agent-context.ts`'s "no
 * usable input" early-return.
 */
export interface TzStateForContext {
  home_tz: string;
  segments: readonly TripitSegment[] | null;
}

export function readTzStateForContext(): TzStateForContext | null {
  const row = db
    .prepare(
      'SELECT home_tz, segments, schema_version FROM tz_state WHERE id = 1',
    )
    .get() as
    | {
        home_tz: string;
        segments: string | null;
        schema_version: number;
      }
    | undefined;
  if (!row) return null;
  // Mirror `runTzHeartbeatAdvisory`'s schema-version gate. A row at
  // an unfamiliar version is "no usable prior state" — return null
  // here so the context builder falls back to container_default
  // rather than feeding stale-shape data into the resolver.
  if (row.schema_version !== SUPPORTED_TZ_STATE_SCHEMA_VERSION) return null;

  let segments: readonly TripitSegment[] | null = null;
  if (row.segments) {
    try {
      const decoded = JSON.parse(row.segments) as unknown;
      if (Array.isArray(decoded)) {
        segments = decoded as readonly TripitSegment[];
      }
    } catch (err) {
      if (!(err instanceof SyntaxError)) throw err;
      // Malformed segments JSON: fall back to null. The
      // `runTzHeartbeatAdvisory` walker fires every 30 min and will
      // log this case; no need to re-emit on every agent prompt.
      segments = null;
    }
  }
  return { home_tz: row.home_tz, segments };
}

export function runTzHeartbeatAdvisory(
  now: Date = new Date(),
  ownerSenderId?: string | null,
  onTzFlipped?: (prev: string, next: string) => void,
): TzAdvisoryResult {
  const row = db
    .prepare(
      'SELECT current_tz, home_tz, segments, schema_version, last_stale_warning_at FROM tz_state WHERE id = 1',
    )
    .get() as
    | {
        current_tz: string;
        home_tz: string;
        segments: string | null;
        schema_version: number;
        last_stale_warning_at: string | null;
      }
    | undefined;
  if (!row) return { flip: null, warningToFire: null };
  if (row.schema_version !== SUPPORTED_TZ_STATE_SCHEMA_VERSION) {
    // Same contract as `getCurrentTz` — an unfamiliar version is "no
    // usable prior state". The walker fires every 30 min, so this
    // emits one warn per tick until the operator drops the bad row;
    // there's no log-once dedup. Two tradeoffs in play: (a) a stuck
    // mid-migration row in production is a load-bearing alert that
    // should keep paging until cleared (silencing it would let an
    // operator forget about it for hours), and (b) a 30-min cadence
    // doesn't pollute the structured log meaningfully — twice an
    // hour is far below the heartbeat noise floor on this surface.
    logger.warn(
      {
        observed: row.schema_version,
        supported: SUPPORTED_TZ_STATE_SCHEMA_VERSION,
      },
      'runTzHeartbeatAdvisory: tz_state schema_version unfamiliar — skipping',
    );
    return { flip: null, warningToFire: null };
  }

  let parsed: readonly TripitSegment[] | null = null;
  let segmentsMalformed = false;
  if (row.segments) {
    try {
      const decoded = JSON.parse(row.segments) as unknown;
      if (Array.isArray(decoded)) {
        parsed = decoded as readonly TripitSegment[];
      }
    } catch (err) {
      if (!(err instanceof SyntaxError)) throw err;
      logger.warn(
        { err: err.message },
        'runTzHeartbeatAdvisory: tz_state.segments is malformed JSON — falling through (will recover on next sync_tripit run)',
      );
      segmentsMalformed = true;
    }
  }

  // Read the owner's most-recent location only when the orchestrator
  // told us who the owner is. Empty / null / undefined `ownerSenderId`
  // skips the location-first path entirely (walker-only, pre-Phase-2
  // behaviour).
  const latestLocation = ownerSenderId
    ? getLatestLocationForSender(ownerSenderId)
    : null;

  // Pre-Phase-2 contract preservation: when there's NO usable input
  // for the cascade (no segments OR malformed segments) AND no
  // location row, the advisory was a no-op (`current_tz` untouched,
  // null return). Phase 2's resolver would otherwise call
  // `walkTzSegments(null, ...)` and silently flip `current_tz` to
  // `home_tz`, which is a behavioural change for zero-config installs
  // where `ASSISTANT_OWNER_TG_USER_ID` is unset and the segments
  // cache hasn't been populated yet. Early-return preserves the
  // pre-Phase-2 no-op exactly. NOTE: this guard sits AFTER the
  // location read so an owner with stale-but-existent location data
  // can still drive the cascade through the walker-fallback path
  // (and possibly fire a stale_no_share warning) when segments are
  // broken.
  if (latestLocation === null && (parsed === null || segmentsMalformed)) {
    return { flip: null, warningToFire: null };
  }

  const resolved = resolveCurrentTz({
    now,
    latestLocation,
    segments: parsed,
    home_tz: row.home_tz,
  });

  // Cooldown for the stale-location warning. We fire `stale_no_share`
  // at most once per STALE_WARNING_HOURS window so an owner who travels
  // for a weekend doesn't get the same nag 48 times across 24 h. Reset
  // the cooldown stamp ONLY when the resolver reports
  // `source: 'fresh_location'` — that's the unambiguous signal that
  // the owner has shared again. Resetting on any `warning === null`
  // would erase the cooldown during the 4 h ≤ age < 12 h band (where
  // the location is stale-for-cascade but not yet warning-eligible)
  // and re-fire the nag on the first ≥ 12 h tick after — defeating
  // the cooldown's purpose.
  let warningToFire: 'stale_no_share' | null = null;
  let nextLastWarning: string | null | undefined = undefined;
  if (resolved.warning === 'stale_no_share') {
    const lastWarn = row.last_stale_warning_at
      ? Date.parse(row.last_stale_warning_at)
      : NaN;
    const cooldownExpired =
      !Number.isFinite(lastWarn) ||
      now.getTime() - lastWarn >= STALE_WARNING_HOURS * 60 * 60 * 1000;
    if (cooldownExpired) {
      warningToFire = 'stale_no_share';
      nextLastWarning = now.toISOString();
    }
  } else if (
    resolved.source === 'fresh_location' &&
    row.last_stale_warning_at !== null
  ) {
    // Owner has genuinely shared again — clear the cooldown so the
    // next stale window starts clean.
    nextLastWarning = null;
  }

  // Single UPDATE batches the current_tz flip and the cooldown stamp
  // so a concurrent advisory can't see a half-applied state. SQLite's
  // single-writer model already serialises this, but the batch keeps
  // the read-then-write window tight.
  const flip =
    resolved.tz !== row.current_tz
      ? { prev: row.current_tz, next: resolved.tz }
      : null;

  if (flip !== null || nextLastWarning !== undefined) {
    const setClauses: string[] = [];
    const bindings: (string | null)[] = [];
    if (flip !== null) {
      setClauses.push('current_tz = ?');
      bindings.push(resolved.tz);
    }
    if (nextLastWarning !== undefined) {
      setClauses.push('last_stale_warning_at = ?');
      bindings.push(nextLastWarning);
    }
    db.prepare(`UPDATE tz_state SET ${setClauses.join(', ')} WHERE id = 1`).run(
      ...bindings,
    );
  }

  if (flip) {
    logger.info(
      {
        prev: row.current_tz,
        next: resolved.tz,
        source: resolved.source,
        latest_location_age_seconds: resolved.latest_location_age_seconds,
      },
      'tz_state.current_tz flipped via heartbeat advisory (#574 Phase 2)',
    );
    // #584 — see `applyTripitSegmentsToTzState` for the rationale; same
    // narrowing contract on the heartbeat-advisory writer. Catch only
    // transient SQLite contention (SQLITE_BUSY / SQLITE_LOCKED); every
    // other error propagates so programming bugs / persistent DB
    // failures surface instead of getting swallowed as a warn.
    if (onTzFlipped) {
      try {
        onTzFlipped(flip.prev, flip.next);
      } catch (err) {
        if (
          !(err instanceof SqliteError) ||
          !TRANSIENT_SQLITE_CODES.has(err.code)
        ) {
          throw err;
        }
        logger.warn(
          {
            err: err.message,
            code: err.code,
            prev: flip.prev,
            next: flip.next,
          },
          'runTzHeartbeatAdvisory: onTzFlipped transient SQLite contention — tz_state write landed, next scheduler tick will recompute',
        );
      }
    }
  }

  return { flip, warningToFire };
}

/**
 * Stale-lock recovery for `follow_me_tasks.pending_run_at` (#496).
 *
 * Background: skills running inside agent containers acquire a
 * pending-run lock by setting `pending_run_at` mid-run, then clear it
 * post-run. If the host kills the container mid-run (e.g. the periodic
 * `tessl_update` writes `_close`, the agent-runner watchdog hits its
 * 30s timeout, and the container exits before the post-run clear), the
 * lock is left dangling. Tomorrow's scheduled fire then sees a stale
 * `pending_run_at` and refuses to run on the Phase A gate.
 *
 * This helper is the host-side recovery: any `pending_run_at` older
 * than `maxAgeMs` is treated as orphaned (its owning container is long
 * gone) and cleared. Per `coding-policy: stateful-artifacts`, the host
 * is a NON-OWNER reader of `follow_me_tasks` (the owning skill is
 * `nanoclaw-admin/skills/task-tz-sync`); non-owners must not migrate
 * the schema, but clearing a value field is lock-recovery, not
 * migration — it's the inverse of what the owner skill does on a
 * normal post-run.
 *
 * Returns the names of every task whose `pending_run_at` was cleared
 * so the caller can log the recovery.
 */
export function clearStalePendingRunAt(maxAgeMs: number): string[] {
  if (!Number.isFinite(maxAgeMs) || maxAgeMs <= 0) {
    throw new Error(
      `clearStalePendingRunAt: maxAgeMs must be a positive number (got ${maxAgeMs})`,
    );
  }
  const cutoffIso = new Date(Date.now() - maxAgeMs).toISOString();
  // Two-step (SELECT then UPDATE) so we can return the names of cleared
  // rows for logging. The window between SELECT and UPDATE is racy in
  // principle — a fresh skill could land a NEW `pending_run_at` on the
  // same row between the two — but the UPDATE's WHERE clause re-checks
  // the cutoff, so a row that gained a fresh lock won't be cleared.
  const rows = db
    .prepare(
      `SELECT name FROM follow_me_tasks
        WHERE pending_run_at IS NOT NULL
          AND pending_run_at < ?`,
    )
    .all(cutoffIso) as { name: string }[];
  if (rows.length === 0) return [];
  db.prepare(
    `UPDATE follow_me_tasks
        SET pending_run_at = NULL,
            updated_at     = CURRENT_TIMESTAMP
      WHERE pending_run_at IS NOT NULL
        AND pending_run_at < ?`,
  ).run(cutoffIso);
  return rows.map((r) => r.name);
}

/**
 * Returns the names of every `follow_me_tasks` row with a fresh
 * (within `maxAgeMs`) `pending_run_at` lock. Empty array means no
 * task is currently mid-run from the host's perspective.
 *
 * Used by the periodic `tessl_update` catch-up to skip the
 * session-clear / container-close path while a scheduled task is in
 * flight (#496 mitigation 1). The owning agent containers wouldn't
 * meaningfully observe new tile content until they finish anyway, so
 * deferring is harmless; the next 15-minute tick will retry.
 */
export function getActivePendingRunAtNames(maxAgeMs: number): string[] {
  if (!Number.isFinite(maxAgeMs) || maxAgeMs <= 0) {
    throw new Error(
      `getActivePendingRunAtNames: maxAgeMs must be a positive number (got ${maxAgeMs})`,
    );
  }
  const cutoffIso = new Date(Date.now() - maxAgeMs).toISOString();
  const rows = db
    .prepare(
      `SELECT name FROM follow_me_tasks
        WHERE pending_run_at IS NOT NULL
          AND pending_run_at >= ?`,
    )
    .all(cutoffIso) as { name: string }[];
  return rows.map((r) => r.name);
}
