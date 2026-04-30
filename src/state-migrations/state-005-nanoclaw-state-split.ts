import type { StateMigration } from '../db.js';

/**
 * #297 — Split the `nanoclaw-state.json` junk drawer into three
 * SQLite tables, each with a single owner skill (vs. the multi-
 * writer race that motivated the field-clobber bug class #293
 * targets).
 *
 *   - `email_state`  — singleton row, owned by tessl__check-email
 *                      (cursor fields: last_email_checked, date,
 *                      fetched_at).
 *   - `email_seen_ids` — append-mostly dedup set, owned by
 *                      tessl__check-email. Replaces the seen_email_ids
 *                      array that #273 Step 16 was patching around
 *                      with a "consolidate two files" dance — both
 *                      writers target this table now and trim-to-N
 *                      becomes a windowed DELETE.
 *   - `resumable_cycles` — owned by tessl__resumable-cycle, read
 *                      by nightly / weekly / morning-brief.
 *
 * Schema rationale:
 *   - `email_state` enforces single-row via `CHECK(id = 1)` so the
 *     PK can never be duplicated by a misbehaving writer; readers
 *     always SELECT … WHERE id=1, getting a clear NULL on missing
 *     rather than racing on "which row".
 *   - `email_seen_ids.seen_at` indexed for the trim-to-N sweep
 *     (`DELETE FROM email_seen_ids WHERE email_id NOT IN (SELECT
 *     email_id FROM email_seen_ids ORDER BY seen_at DESC LIMIT N)`).
 *     Default value uses `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`
 *     rather than `CURRENT_TIMESTAMP` because the trim-to-N
 *     contract relies on lex ordering and writers passing explicit
 *     `seen_at` use ISO-8601 with `T` and `Z` (JS
 *     `Date#toISOString()`). `CURRENT_TIMESTAMP` produces
 *     `YYYY-MM-DD HH:MM:SS` (no `T`, no `Z`); mixing the two
 *     shapes in the same column would break the index lookup AND
 *     the windowed DELETE. Same `%f` (fractional-seconds)
 *     reasoning applies as state-004's purge predicate — see #296.
 *   - `resumable_cycles.updated_at` uses the same
 *     `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')` default for the same
 *     lex-ordering rationale.
 *   - `resumable_cycles.skill_name` is the natural PK because each
 *     skill owns at most one in-flight cycle; the cleanup contract
 *     collapses to a single `DELETE WHERE skill_name=? AND cycle_id=?`.
 *   - `remaining_steps` is `TEXT` (JSON blob) because its shape is
 *     a per-skill payload, not a schema concern — different skills
 *     have different step lists, and the orchestrator never queries
 *     into them.
 *   - `schema_version` on every table per `coding-policy:
 *     stateful-artifacts` (the rule the gh-aw reviewer enforces
 *     literally — see #295's two-PR restoration cycle).
 *
 * The consolidate-email-dedup.py two-file dance retires entirely:
 * both writers target `email_seen_ids` directly, no more
 * "merge nanoclaw-state's seen_ids into session-state" indirection.
 */
export const STATE_005_NANOCLAW_STATE_SPLIT: StateMigration = {
  version: 5,
  name: 'nanoclaw-state.json split → email_state + email_seen_ids + resumable_cycles (#297)',
  sql: `
    CREATE TABLE email_state (
      id                 INTEGER PRIMARY KEY CHECK(id = 1),
      last_email_checked TEXT,
      date               TEXT,
      fetched_at         TEXT,
      schema_version     INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE email_seen_ids (
      email_id       TEXT PRIMARY KEY,
      seen_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
      schema_version INTEGER NOT NULL DEFAULT 1
    );
    CREATE INDEX idx_email_seen_ids_seen_at
      ON email_seen_ids(seen_at);

    CREATE TABLE resumable_cycles (
      skill_name      TEXT PRIMARY KEY,
      cycle_id        TEXT NOT NULL,
      slot_key        TEXT NOT NULL,
      continuation_n  INTEGER NOT NULL DEFAULT 0,
      remaining_steps TEXT,
      schema_version  INTEGER NOT NULL DEFAULT 1,
      updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    );
  `,
};
