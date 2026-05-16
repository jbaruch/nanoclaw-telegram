import type { StateMigration } from '../db.js';

/**
 * #574 Phase 2 — `tz_state.last_stale_warning_at` column for stale-
 * location warning cooldown.
 *
 * When the resolver detects that the owner's latest location is ≥12 h
 * old (`STALE_WARNING_HOURS`), `runTzHeartbeatAdvisory` emits a
 * `stale_no_share` signal so the orchestrator's heartbeat tick can
 * send one-shot chat notification to the main group asking for a
 * re-share. The advisory fires every 30 min — without a cooldown,
 * an owner who travels for a weekend without re-sharing would get
 * the same warning ~48 times across a Saturday. Track the last
 * fire time on `tz_state` so we can suppress repeats inside a 12 h
 * window from the last warning, AND reset the cooldown to NULL
 * once a fresh location lands (so the NEXT stale window gets one
 * notification at the start, not silently never).
 *
 * `tz_state` is a singleton row keyed on `id = 1`, so the column
 * lives on that row rather than its own table. Schema bump 3 → 4
 * keeps the reader gate (`SUPPORTED_TZ_STATE_SCHEMA_VERSION` in
 * `src/db.ts`) in lock-step with the new column — without the bump,
 * a v3 reader on a v4 row would silently ignore the new column,
 * and the runTzHeartbeatAdvisory cooldown check would always see
 * `undefined` and fire every tick.
 *
 * Conservative WHERE clauses on both statements: the column-add is
 * idempotent at the `state_migrations`-table level (the runner
 * tracks applied migrations and never re-runs), but the v3 → v4
 * UPDATE only fires on rows that are actually at v3 — a future
 * state-016 that re-bumped to v5 would leave a v5 row alone here,
 * and a hypothetical fresh row at some other version is left
 * untouched rather than corrupted with an unexpected version flip.
 */
export const STATE_015_TZ_STATE_STALE_WARNING: StateMigration = {
  version: 15,
  name: 'tz_state.last_stale_warning_at + schema_version 3 → 4 (#574 Phase 2)',
  sql: `
    ALTER TABLE tz_state ADD COLUMN last_stale_warning_at TEXT;
    UPDATE tz_state SET schema_version = 4 WHERE id = 1 AND schema_version = 3;
  `,
};
