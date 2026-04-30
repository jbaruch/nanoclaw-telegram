import type { StateMigration } from '../db.js';

/**
 * #295 follow-up — Add per-record `schema_version` column to
 * `email_feedback` to satisfy `coding-policy: stateful-artifacts`'s
 * "A `schema_version` field on every record so migrations are
 * auditable" requirement.
 *
 * The state-002 migration originally dropped the per-record
 * `schema_version` field that the JSON shape carried, on the
 * argument that DDL columns + `PRAGMA user_version` provide
 * stronger auditability than per-row stamps. The gh-aw policy
 * reviewer flagged that as a literal rule violation on the
 * brief-cleanup tile-side PR (jbaruch/nanoclaw-admin#116) and held
 * the line through multiple decline cycles. Restoring the column
 * is the simplest path to compliance.
 *
 * The column carries a per-row stamp the writer chooses deliberately
 * at INSERT time — it does NOT auto-increment with later migrations.
 * Today every writer stamps `1`; a future shape change adds a new
 * state-NNN migration AND updates the writer to stamp the new
 * value, so the column accurately records "which record contract
 * this row was written under" regardless of what migrations have
 * run since.
 *
 * DEFAULT 1 covers two cases: legacy rows that pre-dated this
 * migration (backfilled by ALTER TABLE … DEFAULT) and any future
 * writer that forgets to stamp explicitly (caught by the
 * `test_writer_stamps_schema_version_explicitly` regression in the
 * tile, but the DDL default keeps the column NOT NULL even on the
 * forget path).
 */
export const STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION: StateMigration = {
  version: 3,
  name: 'email_feedback.schema_version column (#295 / coding-policy: stateful-artifacts)',
  sql: `
    ALTER TABLE email_feedback
      ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1;
  `,
};
