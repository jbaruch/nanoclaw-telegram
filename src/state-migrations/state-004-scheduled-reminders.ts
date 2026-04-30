import type { StateMigration } from '../db.js';

/**
 * #296 — Migrate scheduled-reminders.json (per-group state file
 * under `/workspace/group/`) to a `scheduled_reminders` SQLite
 * table in messages.db.
 *
 * Schema rationale:
 *   - `event_id PRIMARY KEY` makes the dedup safety net in
 *     append-scheduled-reminders.py automatic — the previous
 *     re-check-at-append-time guard against pre-MCP dedup-snapshot
 *     staleness collapses to `INSERT ... ON CONFLICT DO NOTHING`,
 *     no read-modify-write needed.
 *   - `idx_scheduled_reminders_utc_time` accelerates nightly Step
 *     22's purge-stale sweep, which is the dominant DELETE path.
 *   - `utc_time` is stored as full ISO-8601 with `T...Z`. Writers
 *     vary in precision: the JSON-era format used second precision
 *     (`2026-04-30T15:00:00Z`), but JavaScript callers using
 *     `Date#toISOString()` emit millisecond precision
 *     (`2026-04-30T15:00:00.000Z`). The nightly purge query MUST
 *     account for both — `strftime('%Y-%m-%dT%H:%M:%SZ', 'now')`
 *     would lex-compare WRONG against a `.000Z`-suffixed row at the
 *     same wall-second (`.` < `Z`, so the row would be deleted as
 *     "past"). Use `%f` for fractional seconds instead:
 *       DELETE FROM scheduled_reminders
 *        WHERE utc_time < strftime('%Y-%m-%dT%H:%M:%fZ', 'now');
 *     `%f` emits `SS.SSS` so the comparison string is `...HH:MM:SS.SSSZ`
 *     — lex compare works correctly against either second-precision
 *     or millisecond-precision row values, and against
 *     `CURRENT_TIMESTAMP` it doesn't (different shape entirely).
 *   - `schema_version` is included from day one per
 *     `coding-policy: stateful-artifacts` (the gh-aw reviewer holds
 *     the line on per-record stamps; learnt from #295's two-PR
 *     restoration cycle, doing it right the first time saves the
 *     dance).
 *
 * Like prior #293-epic migrations, this only adds the schema. Data
 * import (existing JSON → table rows) and tile-side consumer
 * rewrites (`morning-brief`, `check-calendar`, `nightly-housekeeping`,
 * retiring `append-scheduled-reminders.py`) land in follow-up PRs
 * through the orchestrator-side `migrateJsonState` extension and
 * the tile staging→promote pipeline.
 */
export const STATE_004_SCHEDULED_REMINDERS: StateMigration = {
  version: 4,
  name: 'scheduled_reminders table (#296)',
  sql: `
    CREATE TABLE scheduled_reminders (
      event_id            TEXT PRIMARY KEY,
      title               TEXT NOT NULL,
      utc_time            TEXT NOT NULL,
      reminder_offset_min INTEGER NOT NULL,
      task_id             TEXT NOT NULL,
      created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      schema_version      INTEGER NOT NULL DEFAULT 1
    );
    CREATE INDEX idx_scheduled_reminders_utc_time
      ON scheduled_reminders(utc_time);
  `,
};
