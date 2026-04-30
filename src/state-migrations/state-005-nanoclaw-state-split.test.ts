import Database from 'better-sqlite3';
import { describe, it, expect } from 'vitest';

import { applyStateMigrations } from '../db.js';

import { STATE_001_ORDERS } from './state-001-orders.js';
import { STATE_002_EMAIL_FEEDBACK } from './state-002-email-feedback.js';
import { STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION } from './state-003-email-feedback-schema-version.js';
import { STATE_004_SCHEDULED_REMINDERS } from './state-004-scheduled-reminders.js';
import { STATE_005_NANOCLAW_STATE_SPLIT } from './state-005-nanoclaw-state-split.js';

const ALL = [
  STATE_001_ORDERS,
  STATE_002_EMAIL_FEEDBACK,
  STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
  STATE_004_SCHEDULED_REMINDERS,
  STATE_005_NANOCLAW_STATE_SPLIT,
];

describe('state-005-nanoclaw-state-split', () => {
  it('creates all three tables and bumps user_version to 5', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(5);
      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      const names = tables.map((t) => t.name);
      expect(names).toContain('email_state');
      expect(names).toContain('email_seen_ids');
      expect(names).toContain('resumable_cycles');
    } finally {
      database.close();
    }
  });

  it('email_state enforces single-row via CHECK(id = 1)', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      // Insert with id=1 succeeds.
      database
        .prepare(
          `INSERT INTO email_state (id, last_email_checked, date, fetched_at)
           VALUES (1, ?, ?, ?)`,
        )
        .run('msg-aaa', '2026-04-30', '2026-04-30T12:00:00Z');
      // Insert with id=2 violates the CHECK constraint.
      expect(() =>
        database
          .prepare(
            `INSERT INTO email_state (id, last_email_checked, date, fetched_at)
             VALUES (2, ?, ?, ?)`,
          )
          .run('msg-bbb', '2026-05-01', '2026-05-01T12:00:00Z'),
      ).toThrow(/CHECK constraint failed/);
      // Re-inserting id=1 violates the PRIMARY KEY.
      expect(() =>
        database
          .prepare(
            `INSERT INTO email_state (id, last_email_checked, date, fetched_at)
             VALUES (1, ?, ?, ?)`,
          )
          .run('msg-ccc', '2026-05-02', '2026-05-02T12:00:00Z'),
      ).toThrow(/UNIQUE constraint failed/);
    } finally {
      database.close();
    }
  });

  it('email_state singleton supports the SELECT-with-fallback pattern', () => {
    // Empty table: SELECT … WHERE id=1 returns no rows. Reader gets a
    // clear "no state yet" signal rather than racing on which row is
    // canonical (the failure mode the singleton CHECK exists to
    // prevent).
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const empty = database
        .prepare('SELECT last_email_checked FROM email_state WHERE id = 1')
        .get();
      expect(empty).toBeUndefined();
    } finally {
      database.close();
    }
  });

  it('email_seen_ids supports trim-to-N via windowed DELETE', () => {
    // The "consolidate seen_email_ids" two-file dance #273 Step 16 was
    // patching around becomes a single windowed DELETE here:
    //   DELETE FROM email_seen_ids WHERE email_id NOT IN
    //     (SELECT email_id FROM email_seen_ids
    //       ORDER BY seen_at DESC LIMIT N);
    // Lock that contract in.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const insert = database.prepare(
        `INSERT INTO email_seen_ids (email_id, seen_at) VALUES (?, ?)`,
      );
      insert.run('id-1', '2026-04-01T00:00:00Z');
      insert.run('id-2', '2026-04-02T00:00:00Z');
      insert.run('id-3', '2026-04-03T00:00:00Z');
      insert.run('id-4', '2026-04-04T00:00:00Z');
      insert.run('id-5', '2026-04-05T00:00:00Z');

      database
        .prepare(
          `DELETE FROM email_seen_ids
            WHERE email_id NOT IN
              (SELECT email_id FROM email_seen_ids
                ORDER BY seen_at DESC LIMIT 3)`,
        )
        .run();

      const survivors = database
        .prepare('SELECT email_id FROM email_seen_ids ORDER BY seen_at DESC')
        .all() as Array<{ email_id: string }>;
      expect(survivors.map((r) => r.email_id)).toEqual([
        'id-5',
        'id-4',
        'id-3',
      ]);
    } finally {
      database.close();
    }
  });

  it('email_seen_ids default seen_at produces ISO-8601 with T and Z (lex-safe with explicit writers)', () => {
    // The trim-to-N contract relies on lex compare. Default writers
    // (omitting seen_at) and explicit writers (passing ISO-8601)
    // must produce sortable strings in the same shape. Verify the
    // default's literal output matches the documented form
    // YYYY-MM-DDTHH:MM:SS.SSSZ — `CURRENT_TIMESTAMP` would have
    // produced `YYYY-MM-DD HH:MM:SS` (no T, no Z) and broken lex
    // compare against explicit ISO-8601 writers.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(`INSERT INTO email_seen_ids (email_id) VALUES (?)`)
        .run('id-default');
      const row = database
        .prepare(
          `SELECT seen_at FROM email_seen_ids WHERE email_id = 'id-default'`,
        )
        .get() as { seen_at: string };
      // ISO-8601 with T separator, fractional seconds, Z suffix.
      expect(row.seen_at).toMatch(
        /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$/,
      );
    } finally {
      database.close();
    }
  });

  it('email_seen_ids trim-to-N works correctly with mixed default + explicit writers', () => {
    // Combine an explicit-seen_at write (past) with a default-seen_at
    // write (now) — the explicit-past row must trim out, the
    // default-now row must survive. This is the exact scenario the
    // CURRENT_TIMESTAMP→strftime fix addresses.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(`INSERT INTO email_seen_ids (email_id, seen_at) VALUES (?, ?)`)
        .run('past', '2020-01-01T00:00:00.000Z');
      database
        .prepare(`INSERT INTO email_seen_ids (email_id) VALUES (?)`)
        .run('now-default');

      database
        .prepare(
          `DELETE FROM email_seen_ids
            WHERE email_id NOT IN
              (SELECT email_id FROM email_seen_ids
                ORDER BY seen_at DESC LIMIT 1)`,
        )
        .run();

      const survivors = database
        .prepare('SELECT email_id FROM email_seen_ids')
        .all() as Array<{ email_id: string }>;
      expect(survivors.map((r) => r.email_id)).toEqual(['now-default']);
    } finally {
      database.close();
    }
  });

  it('email_seen_ids has the seen_at index', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const indexes = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index'")
        .all() as Array<{ name: string }>;
      expect(indexes.map((i) => i.name)).toContain(
        'idx_email_seen_ids_seen_at',
      );
    } finally {
      database.close();
    }
  });

  it('resumable_cycles uses skill_name as PK so cleanup is one DELETE', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const insert = database.prepare(
        `INSERT INTO resumable_cycles
           (skill_name, cycle_id, slot_key, continuation_n, remaining_steps)
         VALUES (?, ?, ?, ?, ?)`,
      );
      insert.run(
        'tessl__nightly-housekeeping',
        'cycle-aaa',
        '2026-04-30',
        0,
        '["step-15","step-16"]',
      );
      // Re-inserting the same skill_name violates the PK — only one
      // in-flight cycle per skill is allowed.
      expect(() =>
        insert.run(
          'tessl__nightly-housekeeping',
          'cycle-bbb',
          '2026-05-01',
          0,
          '["step-1"]',
        ),
      ).toThrow(/UNIQUE constraint failed/);
      // Cleanup contract: one DELETE clears it.
      const result = database
        .prepare(
          `DELETE FROM resumable_cycles WHERE skill_name = ? AND cycle_id = ?`,
        )
        .run('tessl__nightly-housekeeping', 'cycle-aaa');
      expect(result.changes).toBe(1);
    } finally {
      database.close();
    }
  });

  it('every #297 table carries schema_version with default 1', () => {
    // Per-record schema_version is mandatory per
    // `coding-policy: stateful-artifacts`. Verify all three tables
    // got the column from day one.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      for (const table of [
        'email_state',
        'email_seen_ids',
        'resumable_cycles',
      ]) {
        const cols = database
          .prepare(`PRAGMA table_info(${table})`)
          .all() as Array<{
          name: string;
          notnull: number;
          dflt_value: string | null;
        }>;
        const versionCol = cols.find((c) => c.name === 'schema_version');
        expect(versionCol).toBeDefined();
        expect(versionCol!.notnull).toBe(1);
        expect(versionCol!.dflt_value).toBe('1');
      }
    } finally {
      database.close();
    }
  });
});
