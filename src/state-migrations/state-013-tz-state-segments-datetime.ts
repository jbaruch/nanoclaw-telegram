import type { StateMigration } from '../db.js';

/**
 * jbaruch/nanoclaw-admin#229 — Per-segment datetime resolution for the
 * `tz_state` walker.
 *
 * Background: state-012 (#542) handed timezone-state ownership to the
 * host orchestrator and stored the `sync_tripit` stdout `segments[]`
 * array on `tz_state.segments`. The walker compared each segment's
 * date-only `from` / `to` against `now.toISOString().slice(0, 10)`, so
 * it could only resolve at UTC-midnight granularity. The upstream
 * `reclaim-tripit-timezones-sync/lib/tripit.mjs::formatDate` flattened
 * every Date through `toISOString().slice(0, 10)` before the segment
 * left the parser, so the walker had no finer signal to act on even
 * when the underlying iCal feed carried full datetimes.
 *
 * Production incident on 2026-05-11 19:26 CT (00:26 UTC 2026-05-12):
 * `tz_state.current_tz` flipped `America/Chicago` → `America/New_York`
 * the moment UTC rolled over to the calendar day of the next morning's
 * BNA→ATL flight. The traveler was still physically in TN.
 *
 * Upstream fix in
 * https://github.com/jbaruch/reclaim-tripit-timezones-sync/pull/13:
 * every emitted segment now carries `startDateTime` / `endDateTime`
 * (ISO 8601 UTC) adjacent to the existing `startDate` / `endDate`, and
 * `sync.mjs`'s `--output=json` payload surfaces them as `from_dt` /
 * `to_dt` next to the unchanged `from` / `to`. Purely additive — old
 * date-only consumers keep working.
 *
 * This migration bumps `schema_version` 2 → 3 to signal the new
 * walker contract. There is NO DDL change: the `segments` column is
 * still `TEXT` (JSON-stringified), and the datetime fields live inside
 * each element's JSON object. A v2 row's existing JSON is forward-
 * compatible — the walker treats missing `from_dt` / `to_dt` as
 * "fall back to date-only comparison", which is the exact pre-#229
 * behavior. The first `sync_tripit` run after deploy (with the new
 * tile version installed via `tessl update`) rewrites the row with
 * the new shape; until then, the walker uses the legacy fields.
 *
 * Lock-step with the reader gate `SUPPORTED_TZ_STATE_SCHEMA_VERSION`
 * in `src/db.ts` per `coding-policy: stateful-artifacts` — the gate
 * advances 2 → 3 in the same commit so a row sitting at v2 (written
 * by a pre-deploy `applyTripitSegmentsToTzState`) is migrated up to
 * v3 by the UPDATE below before the reader consults the version. A
 * row that's already at v3 (somehow seeded by a future writer that
 * raced the migration) survives unchanged — the WHERE clause is
 * conservative.
 */
export const STATE_013_TZ_STATE_SEGMENTS_DATETIME: StateMigration = {
  version: 13,
  name: 'tz_state.schema_version 2 → 3 for per-segment datetime walker (jbaruch/nanoclaw-admin#229)',
  sql: `
    UPDATE tz_state SET schema_version = 3 WHERE id = 1 AND schema_version = 2;
  `,
};
