import Database from 'better-sqlite3';
import { describe, it, expect } from 'vitest';

import { applyStateMigrations } from '../db.js';

import { STATE_001_ORDERS } from './state-001-orders.js';
import { STATE_002_EMAIL_FEEDBACK } from './state-002-email-feedback.js';
import { STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION } from './state-003-email-feedback-schema-version.js';

describe('state-003-email-feedback-schema-version', () => {
  it('adds the schema_version column with NOT NULL DEFAULT 1', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [
        STATE_001_ORDERS,
        STATE_002_EMAIL_FEEDBACK,
        STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
      ]);

      expect(Number(database.pragma('user_version', { simple: true }))).toBe(3);

      const cols = database
        .prepare('PRAGMA table_info(email_feedback)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        dflt_value: string | null;
      }>;
      const versionCol = cols.find((c) => c.name === 'schema_version');
      expect(versionCol).toBeDefined();
      expect(versionCol!.type).toBe('INTEGER');
      expect(versionCol!.notnull).toBe(1);
      expect(versionCol!.dflt_value).toBe('1');
    } finally {
      database.close();
    }
  });

  it('backfills existing rows with schema_version = 1', () => {
    // Rows inserted under state-002's shape (no schema_version column)
    // must come back with schema_version = 1 after state-003 runs —
    // they were written by the v1 record contract.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [
        STATE_001_ORDERS,
        STATE_002_EMAIL_FEEDBACK,
      ]);

      database
        .prepare(
          `INSERT INTO email_feedback (pattern, label, date) VALUES (?, ?, ?)`,
        )
        .run('legacy@example.com', 'noise', '2026-04-29');

      // Now apply state-003 — adds the column without re-running 1+2.
      applyStateMigrations(database, [
        STATE_001_ORDERS,
        STATE_002_EMAIL_FEEDBACK,
        STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
      ]);

      const row = database
        .prepare(
          `SELECT pattern, schema_version FROM email_feedback WHERE id = 1`,
        )
        .get() as { pattern: string; schema_version: number };
      expect(row.pattern).toBe('legacy@example.com');
      expect(row.schema_version).toBe(1);
    } finally {
      database.close();
    }
  });

  it('lets writers stamp schema_version explicitly', () => {
    // Forward-compat: when a future shape change lands, the writer
    // bumps the value at INSERT time. Verify the column accepts an
    // explicit value (no CHECK constraint pinning it to 1).
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [
        STATE_001_ORDERS,
        STATE_002_EMAIL_FEEDBACK,
        STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
      ]);
      database
        .prepare(
          `INSERT INTO email_feedback (pattern, label, date, schema_version) VALUES (?, ?, ?, ?)`,
        )
        .run('future@example.com', 'actionable', '2026-05-01', 2);
      const row = database
        .prepare(
          `SELECT schema_version FROM email_feedback WHERE pattern = 'future@example.com'`,
        )
        .get() as { schema_version: number };
      expect(row.schema_version).toBe(2);
    } finally {
      database.close();
    }
  });
});
