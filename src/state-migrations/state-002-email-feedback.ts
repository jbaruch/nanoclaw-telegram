import type { StateMigration } from '../db.js';

/**
 * #295 — Migrate email-feedback.json (admin-only state file under
 * `/workspace/state/brief-cleanup/`) to an `email_feedback` SQLite
 * table in messages.db.
 *
 * Like #294, this migration adds the schema only. The data import
 * (existing JSON array → table rows) and tile-side consumer rewrite
 * (`nanoclaw-admin/skills/brief-cleanup` Step 6, retiring
 * `append-feedback.py`) land in follow-up PRs through the tile
 * staging→promote pipeline.
 *
 * Schema rationale:
 *   - `id INTEGER PRIMARY KEY AUTOINCREMENT` — append-only log,
 *     synthetic ids; matches the SQLite-idiomatic shape (the JSON
 *     records had no stable id, just array position).
 *   - `pattern` is stored lowercased at write time; case-folding the
 *     INDEX would force collation choices that the existing `pattern`
 *     contract already pre-normalises.
 *   - `label CHECK(...)` enforces the actionable/noise invariant at
 *     the DB layer instead of relying on every helper to validate.
 *     The state-schema doc currently lists those as the only valid
 *     values; expanding the set is a v3 schema bump.
 *   - `source DEFAULT 'baruch-response'` matches the JSON record
 *     default. New writers can override.
 *   - `created_at DEFAULT CURRENT_TIMESTAMP` — every INSERT picks up
 *     wall-clock automatically. SQLite returns ISO-ish format from
 *     CURRENT_TIMESTAMP (`YYYY-MM-DD HH:MM:SS`); writers wanting full
 *     ISO-8601 with `T` and timezone must pass it explicitly.
 *   - `idx_email_feedback_pattern` accelerates the brief-cleanup
 *     "have we seen this pattern before" lookup, which is the
 *     dominant read path (see brief-cleanup SKILL.md Step 6).
 *
 * Historical note: this migration originally dropped the per-record
 * `schema_version` field that the JSON shape carried, on the
 * argument that `PRAGMA user_version` on the DB was sufficient for
 * shape auditability. The `coding-policy: stateful-artifacts`
 * reviewer disagreed, and `state-003-email-feedback-schema-version`
 * (the next migration in this directory) restored the column. Every
 * row has both axes now: DB-level `user_version` for "which
 * migrations have run" and per-row `schema_version` for "which
 * record contract this row was written under".
 */
export const STATE_002_EMAIL_FEEDBACK: StateMigration = {
  version: 2,
  name: 'email_feedback table (#295)',
  sql: `
    CREATE TABLE email_feedback (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      pattern    TEXT NOT NULL,
      label      TEXT NOT NULL CHECK(label IN ('actionable', 'noise')),
      source     TEXT NOT NULL DEFAULT 'baruch-response',
      date       TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX idx_email_feedback_pattern ON email_feedback(pattern);
  `,
};
