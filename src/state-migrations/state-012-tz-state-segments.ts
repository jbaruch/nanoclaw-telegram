import type { StateMigration } from '../db.js';

/**
 * #542 — Persist the `segments[]` array returned by `sync_tripit`'s
 * stdout (TripIt → Reclaim sync) onto the singleton `tz_state` row.
 *
 * Today `tz_state.current_tz` is written by the LLM-side `task-tz-sync`
 * skill, which derives TZ from a parallel pipeline (Flighty Google
 * Calendar via `composio-fetch` + `travel-schedule.json` fallback).
 * That pipeline disagreed with the host-side `sync_tripit` (the actual
 * canonical source) on 2026-05-09 and produced
 * jbaruch/nanoclaw-admin#224.
 *
 * The fix routes everything through one source: after every successful
 * `sync_tripit` run, the host persists the returned `segments[]` array
 * into a new `tz_state.segments TEXT` column and writes the computed
 * `current_tz`. A heartbeat advisory walker re-walks the cached
 * segments every 30 minutes for sub-day reactivity without re-fetching
 * iCal.
 *
 * Shape of `segments` (verified against
 * https://github.com/jbaruch/reclaim-tripit-timezones-sync/blob/v0.2.0/sync.mjs
 * line 148 `result.segments = segments.map(s => ({timezone, from, to,
 * label}))`):
 *
 *     [
 *       {"timezone": "Europe/Berlin",
 *        "from": "2026-05-12",
 *        "to":   "2026-05-19",
 *        "label": "Devoxx UK 2026 - London"},
 *       ...
 *     ]
 *
 *   - `from` / `to` are date-only `YYYY-MM-DD` strings (the upstream
 *     `formatDate` helper in `lib/tripit.mjs` slices `toISOString()` to
 *     10 chars). Lexicographic comparison of these strings is
 *     equivalent to date comparison, so the segment-walk can use plain
 *     string `<=`/`<` without parsing.
 *   - The column is `TEXT` (JSON-stringified) rather than a child
 *     table because every reader of `segments` walks the whole array
 *     anyway — there are no per-segment lookups, joins, or partial
 *     reads that would benefit from a row-per-segment shape. Reading
 *     the JSON blob and `JSON.parse()` once per heartbeat is cheaper
 *     than a `SELECT * FROM tz_segments WHERE …` followed by sort.
 *   - Nullable: `tz_state` may exist with `current_tz` set but
 *     `segments` NULL on the transient where the row was seeded by
 *     `_seedTzStateForTests` / `migrateTaskTzStateJsonFiles` before any
 *     `sync_tripit` run has populated the column. The heartbeat
 *     advisory walker treats null/empty `segments` as a silent skip.
 *
 * `schema_version` bumps 1 → 2. Per `coding-policy: stateful-artifacts`
 * the `tz_state` reader gate (`SUPPORTED_TZ_STATE_SCHEMA_VERSION` in
 * `src/db.ts`) advances in lock-step so the gate matches the new
 * shape. Owner skill is now the host orchestrator
 * (`applyTripitSegmentsToTzState` in `src/db.ts`); the prior owner
 * (`nanoclaw-admin/skills/task-tz-sync`) retires in
 * jbaruch/nanoclaw-admin#223 (the tile-side cleanup PR).
 *
 * The migration only adjusts the schema. The new writer is wired in
 * `src/ipc.ts` `sync_tripit` success-path; the heartbeat advisory
 * walker is added to the orchestrator's startup interval block in
 * `src/index.ts`. The two existing UPSERT call sites against
 * `tz_state` (`_seedTzStateForTests` and `migrateTaskTzStateJsonFiles`)
 * are extended to write the new column with NULL.
 */
export const STATE_012_TZ_STATE_SEGMENTS: StateMigration = {
  version: 12,
  name: 'tz_state.segments column + schema_version=2 (#542)',
  sql: `
    ALTER TABLE tz_state ADD COLUMN segments TEXT;
    UPDATE tz_state SET schema_version = 2 WHERE id = 1;
  `,
};
