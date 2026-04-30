import Database from 'better-sqlite3';
import { describe, it, expect } from 'vitest';

import { applyStateMigrations } from '../db.js';

import { STATE_001_ORDERS } from './state-001-orders.js';
import { STATE_002_EMAIL_FEEDBACK } from './state-002-email-feedback.js';

describe('state-002-email-feedback', () => {
  it('creates the email_feedback table and bumps user_version to 2', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [
        STATE_001_ORDERS,
        STATE_002_EMAIL_FEEDBACK,
      ]);

      expect(Number(database.pragma('user_version', { simple: true }))).toBe(2);

      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      expect(tables.map((t) => t.name)).toContain('email_feedback');
    } finally {
      database.close();
    }
  });

  it('declares every column from the issue spec', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [
        STATE_001_ORDERS,
        STATE_002_EMAIL_FEEDBACK,
      ]);

      const cols = database
        .prepare('PRAGMA table_info(email_feedback)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        dflt_value: string | null;
        pk: number;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));

      expect(byName['id']).toMatchObject({ type: 'INTEGER', pk: 1 });
      expect(byName['pattern']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['label']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['source']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['date']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['created_at']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      // Defaults — better-sqlite3 surfaces dflt_value as the literal
      // SQL text, including quote characters for string defaults.
      expect(byName['source'].dflt_value).toBe(`'baruch-response'`);
      expect(byName['created_at'].dflt_value).toBe('CURRENT_TIMESTAMP');
    } finally {
      database.close();
    }
  });

  it('enforces the label CHECK(actionable | noise) constraint', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [
        STATE_001_ORDERS,
        STATE_002_EMAIL_FEEDBACK,
      ]);

      const insert = database.prepare(
        `INSERT INTO email_feedback (pattern, label, date) VALUES (?, ?, ?)`,
      );
      // Valid labels accepted.
      expect(() => insert.run('foo', 'actionable', '2026-04-29')).not.toThrow();
      expect(() => insert.run('bar', 'noise', '2026-04-29')).not.toThrow();
      // Anything else rejected — keeps the brief-cleanup contract
      // honest at the DB layer rather than relying on every helper
      // to validate.
      expect(() => insert.run('baz', 'maybe-actionable', '2026-04-29')).toThrow(
        /CHECK constraint failed/,
      );
      expect(() => insert.run('qux', '', '2026-04-29')).toThrow(
        /CHECK constraint failed/,
      );
    } finally {
      database.close();
    }
  });

  it('applies defaults when source / created_at are omitted on INSERT', () => {
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
        .run('payment-confirmation', 'noise', '2026-04-29');
      const row = database
        .prepare(`SELECT source, created_at FROM email_feedback WHERE id = 1`)
        .get() as { source: string; created_at: string };
      expect(row.source).toBe('baruch-response');
      // CURRENT_TIMESTAMP has the SQLite shape `YYYY-MM-DD HH:MM:SS`
      // — assert the structural pattern, not a specific time value.
      expect(row.created_at).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/);
    } finally {
      database.close();
    }
  });

  it('creates the pattern lookup index', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [
        STATE_001_ORDERS,
        STATE_002_EMAIL_FEEDBACK,
      ]);

      const indexes = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index'")
        .all() as Array<{ name: string }>;
      expect(indexes.map((i) => i.name)).toContain(
        'idx_email_feedback_pattern',
      );
    } finally {
      database.close();
    }
  });
});
