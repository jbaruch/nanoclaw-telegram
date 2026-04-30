import Database from 'better-sqlite3';
import { describe, it, expect } from 'vitest';

import { applyStateMigrations } from '../db.js';

import { STATE_001_ORDERS } from './state-001-orders.js';
import { STATE_002_EMAIL_FEEDBACK } from './state-002-email-feedback.js';
import { STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION } from './state-003-email-feedback-schema-version.js';
import { STATE_004_SCHEDULED_REMINDERS } from './state-004-scheduled-reminders.js';

const ALL_MIGRATIONS = [
  STATE_001_ORDERS,
  STATE_002_EMAIL_FEEDBACK,
  STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
  STATE_004_SCHEDULED_REMINDERS,
];

describe('state-004-scheduled-reminders', () => {
  it('creates the scheduled_reminders table and bumps user_version to 4', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL_MIGRATIONS);

      expect(Number(database.pragma('user_version', { simple: true }))).toBe(4);

      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      expect(tables.map((t) => t.name)).toContain('scheduled_reminders');
    } finally {
      database.close();
    }
  });

  it('declares every column from the issue spec, plus schema_version', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL_MIGRATIONS);

      const cols = database
        .prepare('PRAGMA table_info(scheduled_reminders)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        dflt_value: string | null;
        pk: number;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));

      expect(byName['event_id']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['title']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['utc_time']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['reminder_offset_min']).toMatchObject({
        type: 'INTEGER',
        notnull: 1,
      });
      expect(byName['task_id']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['created_at']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      expect(byName['created_at'].dflt_value).toBe('CURRENT_TIMESTAMP');
      expect(byName['schema_version']).toMatchObject({
        type: 'INTEGER',
        notnull: 1,
      });
      expect(byName['schema_version'].dflt_value).toBe('1');
    } finally {
      database.close();
    }
  });

  it('event_id PRIMARY KEY enables INSERT ... ON CONFLICT DO NOTHING dedup', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL_MIGRATIONS);

      const insert = database.prepare(
        `INSERT INTO scheduled_reminders
           (event_id, title, utc_time, reminder_offset_min, task_id)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(event_id) DO NOTHING`,
      );
      const first = insert.run(
        'evt-1',
        'Meeting',
        '2026-04-30T15:00:00Z',
        15,
        'task-aaa',
      );
      const second = insert.run(
        'evt-1',
        'Meeting (overwrite attempt)',
        '2026-04-30T15:00:00Z',
        15,
        'task-bbb',
      );
      // First INSERT lands; second is a no-op via ON CONFLICT.
      expect(first.changes).toBe(1);
      expect(second.changes).toBe(0);
      const stored = database
        .prepare(
          'SELECT title, task_id FROM scheduled_reminders WHERE event_id = ?',
        )
        .get('evt-1') as { title: string; task_id: string };
      expect(stored.title).toBe('Meeting');
      expect(stored.task_id).toBe('task-aaa');
    } finally {
      database.close();
    }
  });

  it('creates the utc_time index for nightly purge sweeps', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL_MIGRATIONS);

      const indexes = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index'")
        .all() as Array<{ name: string }>;
      expect(indexes.map((i) => i.name)).toContain(
        'idx_scheduled_reminders_utc_time',
      );
    } finally {
      database.close();
    }
  });

  it('nightly purge with strftime-%f-now handles both second- and millisecond-precision utc_time', () => {
    // Writers vary in precision: JSON-era + `mcp__nanoclaw__schedule_task`
    // emit second precision; JS callers using Date#toISOString() emit
    // millisecond precision. The migration docstring tells consumers
    // to use `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')` (fractional
    // seconds) so the lex compare works against either shape.
    //
    // Past + far-future rows in BOTH precisions; only the two future
    // rows must survive.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL_MIGRATIONS);
      const insert = database.prepare(
        `INSERT INTO scheduled_reminders
           (event_id, title, utc_time, reminder_offset_min, task_id)
         VALUES (?, ?, ?, ?, ?)`,
      );
      insert.run('past-sec', 'Past S', '2020-01-01T00:00:00Z', 15, 'task');
      insert.run('past-ms', 'Past M', '2020-01-01T00:00:00.000Z', 15, 'task');
      insert.run('future-sec', 'Future S', '2099-01-01T00:00:00Z', 15, 'task');
      insert.run(
        'future-ms',
        'Future M',
        '2099-01-01T00:00:00.000Z',
        15,
        'task',
      );

      database
        .prepare(
          `DELETE FROM scheduled_reminders
             WHERE utc_time < strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`,
        )
        .run();

      const survivors = database
        .prepare('SELECT event_id FROM scheduled_reminders ORDER BY event_id')
        .all() as Array<{ event_id: string }>;
      expect(survivors.map((r) => r.event_id)).toEqual([
        'future-ms',
        'future-sec',
      ]);
    } finally {
      database.close();
    }
  });

  it('schema_version defaults to 1 when omitted on INSERT', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL_MIGRATIONS);
      database
        .prepare(
          `INSERT INTO scheduled_reminders
             (event_id, title, utc_time, reminder_offset_min, task_id)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .run('evt-default', 'Test', '2026-04-30T15:00:00Z', 30, 'task-x');
      const row = database
        .prepare(
          `SELECT schema_version FROM scheduled_reminders WHERE event_id = ?`,
        )
        .get('evt-default') as { schema_version: number };
      expect(row.schema_version).toBe(1);
    } finally {
      database.close();
    }
  });
});
