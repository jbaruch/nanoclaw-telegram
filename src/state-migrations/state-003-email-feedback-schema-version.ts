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
 * is the simplest path to compliance — and harmless since the
 * column always equals the migration that wrote the row, which is
 * exactly the "audit trail per row" the rule asks for.
 *
 * Defaults to 1 — every row inserted before this migration ran was
 * written under state-002's shape, which is v1 of the email_feedback
 * record contract. Future shape changes bump the default in a
 * follow-up state-NNN migration AND update writers to stamp the new
 * value explicitly.
 */
export const STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION: StateMigration = {
  version: 3,
  name: 'email_feedback.schema_version column (#295 / coding-policy: stateful-artifacts)',
  sql: `
    ALTER TABLE email_feedback
      ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1;
  `,
};
