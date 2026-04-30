import Database from 'better-sqlite3';
import { describe, it, expect } from 'vitest';

import { applyStateMigrations } from '../db.js';

import { STATE_001_ORDERS } from './state-001-orders.js';
import { STATE_002_EMAIL_FEEDBACK } from './state-002-email-feedback.js';
import { STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION } from './state-003-email-feedback-schema-version.js';
import { STATE_004_SCHEDULED_REMINDERS } from './state-004-scheduled-reminders.js';
import { STATE_005_NANOCLAW_STATE_SPLIT } from './state-005-nanoclaw-state-split.js';
import { STATE_006_TRUSTED_SESSION_STATE } from './state-006-trusted-session-state.js';
import { STATE_007_MORNING_BRIEF_PENDING } from './state-007-morning-brief-pending.js';
import { STATE_008_CALENDAR_STATE } from './state-008-calendar-state.js';

const ALL = [
  STATE_001_ORDERS,
  STATE_002_EMAIL_FEEDBACK,
  STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
  STATE_004_SCHEDULED_REMINDERS,
  STATE_005_NANOCLAW_STATE_SPLIT,
  STATE_006_TRUSTED_SESSION_STATE,
  STATE_007_MORNING_BRIEF_PENDING,
  STATE_008_CALENDAR_STATE,
];

describe('state-008-calendar-state', () => {
  it('creates calendar_snapshots + calendar_events tables and bumps user_version to 8', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(8);
      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      const names = tables.map((t) => t.name);
      expect(names).toContain('calendar_snapshots');
      expect(names).toContain('calendar_events');
    } finally {
      database.close();
    }
  });

  it('calendar_snapshots has correct columns + nullability per issue spec', () => {
    // The issue spec calls for TEXT PK `date`, NOT NULL `fetched_at`.
    // Lock that down — and lock down the `schema_version` default
    // (`'1'`) from the start so a future "tighten the schema" refactor
    // that drops the default and turns every existing INSERT into a
    // NOT NULL constraint failure fails here instead of in production.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(calendar_snapshots)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
        dflt_value: string | null;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['date']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['fetched_at']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      expect(byName['schema_version']).toMatchObject({
        type: 'INTEGER',
        notnull: 1,
        dflt_value: '1',
      });
    } finally {
      database.close();
    }
  });

  it('calendar_events has correct columns + nullability per issue spec', () => {
    // The issue spec calls for TEXT PK `event_id`, NOT NULL `date` /
    // `title` / `start`, nullable `end` and `reminder_task_id`.
    // Lock the `schema_version` default (`'1'`) down at the same time
    // — same reasoning as on `calendar_snapshots`.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(calendar_events)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
        dflt_value: string | null;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['event_id']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['date']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['title']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['start']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['end']).toMatchObject({ type: 'TEXT', notnull: 0 });
      expect(byName['reminder_task_id']).toMatchObject({
        type: 'TEXT',
        notnull: 0,
      });
      expect(byName['schema_version']).toMatchObject({
        type: 'INTEGER',
        notnull: 1,
        dflt_value: '1',
      });
    } finally {
      database.close();
    }
  });

  it('calendar_events.title and start enforce NOT NULL', () => {
    // A calendar event without a title or start time is unusable in
    // the brief; DB-level NOT NULL means a malformed insert fails
    // loudly rather than landing a useless row that surfaces empty
    // tomorrow morning.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO calendar_snapshots (date, fetched_at)
           VALUES (?, ?)`,
        )
        .run('2026-04-30', '2026-04-30T07:00:00Z');
      expect(() =>
        database
          .prepare(
            `INSERT INTO calendar_events (event_id, date, title, start)
             VALUES (?, ?, NULL, ?)`,
          )
          .run('evt-aaa', '2026-04-30', '2026-04-30T09:00:00Z'),
      ).toThrow(/NOT NULL constraint failed/);
      expect(() =>
        database
          .prepare(
            `INSERT INTO calendar_events (event_id, date, title, start)
             VALUES (?, ?, ?, NULL)`,
          )
          .run('evt-bbb', '2026-04-30', 'Standup'),
      ).toThrow(/NOT NULL constraint failed/);
    } finally {
      database.close();
    }
  });

  it('foreign-key constraint on calendar_events.date references calendar_snapshots(date) with ON DELETE CASCADE', () => {
    // Introspect the FK declaration so the cascade-rationale comment
    // in the migration's doc-header isn't load-bearing prose. If a
    // future schema edit drops the FK or changes the on_delete action
    // to RESTRICT / NO ACTION, the daily-rotation cleanup
    // (`DELETE FROM calendar_snapshots WHERE date < ?`) silently stops
    // cascading and orphan events accumulate forever — that needs to
    // fail here, not in a 3am production stack trace.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const fks = database
        .prepare("PRAGMA foreign_key_list('calendar_events')")
        .all() as Array<{
        id: number;
        seq: number;
        table: string;
        from: string;
        to: string;
        on_update: string;
        on_delete: string;
        match: string;
      }>;
      expect(fks).toHaveLength(1);
      expect(fks[0]).toMatchObject({
        table: 'calendar_snapshots',
        from: 'date',
        to: 'date',
        on_delete: 'CASCADE',
      });
    } finally {
      database.close();
    }
  });

  it('idx_calendar_events_date index exists on calendar_events.date', () => {
    // The index is the read-path payoff: `check-calendar`'s
    // "give me today's events" query (`SELECT * FROM calendar_events
    // WHERE date = ?`) hits this index, and the daily-rotation
    // cascade (`DELETE FROM calendar_snapshots WHERE date < ?`) uses
    // it to find the events to delete on the child side. Lock down
    // its presence — a missing index would degrade these to full
    // table scans without any other test catching it.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const indexes = database
        .prepare("PRAGMA index_list('calendar_events')")
        .all() as Array<{
        seq: number;
        name: string;
        unique: number;
        origin: string;
        partial: number;
      }>;
      const dateIndex = indexes.find(
        (idx) => idx.name === 'idx_calendar_events_date',
      );
      expect(dateIndex).toBeDefined();
      const cols = database
        .prepare("PRAGMA index_info('idx_calendar_events_date')")
        .all() as Array<{ seqno: number; cid: number; name: string }>;
      expect(cols).toHaveLength(1);
      expect(cols[0].name).toBe('date');
    } finally {
      database.close();
    }
  });

  it('ON DELETE CASCADE removes events when their parent snapshot is deleted', () => {
    // The headline cleanup operation the issue spec calls out:
    // `DELETE FROM calendar_snapshots WHERE date < ?` cascading to
    // `calendar_events`. Enable `PRAGMA foreign_keys = ON` for this
    // test even though the orchestrator does not currently flip it —
    // the schema declaration's intent is what's under test, and a
    // future flip of the pragma should not silently break this
    // contract.
    const database = new Database(':memory:');
    try {
      database.pragma('foreign_keys = ON');
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO calendar_snapshots (date, fetched_at)
           VALUES (?, ?)`,
        )
        .run('2026-04-29', '2026-04-29T07:00:00Z');
      database
        .prepare(
          `INSERT INTO calendar_snapshots (date, fetched_at)
           VALUES (?, ?)`,
        )
        .run('2026-04-30', '2026-04-30T07:00:00Z');
      database
        .prepare(
          `INSERT INTO calendar_events (event_id, date, title, start)
           VALUES (?, ?, ?, ?)`,
        )
        .run(
          'evt-old-1',
          '2026-04-29',
          'Yesterday standup',
          '2026-04-29T09:00:00Z',
        );
      database
        .prepare(
          `INSERT INTO calendar_events (event_id, date, title, start)
           VALUES (?, ?, ?, ?)`,
        )
        .run(
          'evt-old-2',
          '2026-04-29',
          'Yesterday review',
          '2026-04-29T15:00:00Z',
        );
      database
        .prepare(
          `INSERT INTO calendar_events (event_id, date, title, start)
           VALUES (?, ?, ?, ?)`,
        )
        .run(
          'evt-today-1',
          '2026-04-30',
          'Today standup',
          '2026-04-30T09:00:00Z',
        );

      database
        .prepare('DELETE FROM calendar_snapshots WHERE date < ?')
        .run('2026-04-30');

      const remainingSnapshots = database
        .prepare('SELECT date FROM calendar_snapshots ORDER BY date')
        .all() as Array<{ date: string }>;
      expect(remainingSnapshots.map((s) => s.date)).toEqual(['2026-04-30']);

      const remainingEvents = database
        .prepare('SELECT event_id FROM calendar_events ORDER BY event_id')
        .all() as Array<{ event_id: string }>;
      expect(remainingEvents.map((e) => e.event_id)).toEqual(['evt-today-1']);
    } finally {
      database.close();
    }
  });

  it('UPDATE on calendar_events.reminder_task_id touches a single row by event_id', () => {
    // The issue's explicit acceptance criterion: `check-calendar`'s
    // `reminder_task_id` updates are single-row SQL UPDATEs. With
    // `event_id` as the PK, an `UPDATE ... WHERE event_id = ?`
    // touches exactly one row regardless of how many other events
    // share the same date. The JSON-era code path had to read the
    // whole snapshot, mutate one entry in the events array, and write
    // the whole snapshot back — racing on the sibling fields. This
    // test inserts three events on the same date with NULL
    // `reminder_task_id`, updates one, and asserts the other two are
    // untouched.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO calendar_snapshots (date, fetched_at)
           VALUES (?, ?)`,
        )
        .run('2026-04-30', '2026-04-30T07:00:00Z');
      for (const id of ['evt-a', 'evt-b', 'evt-c']) {
        database
          .prepare(
            `INSERT INTO calendar_events (event_id, date, title, start)
             VALUES (?, ?, ?, ?)`,
          )
          .run(id, '2026-04-30', `Event ${id}`, '2026-04-30T09:00:00Z');
      }

      const result = database
        .prepare(
          `UPDATE calendar_events SET reminder_task_id = ?
           WHERE event_id = ?`,
        )
        .run('task-xyz', 'evt-b');
      expect(result.changes).toBe(1);

      const rows = database
        .prepare(
          `SELECT event_id, reminder_task_id FROM calendar_events
           ORDER BY event_id`,
        )
        .all() as Array<{ event_id: string; reminder_task_id: string | null }>;
      expect(rows).toEqual([
        { event_id: 'evt-a', reminder_task_id: null },
        { event_id: 'evt-b', reminder_task_id: 'task-xyz' },
        { event_id: 'evt-c', reminder_task_id: null },
      ]);
    } finally {
      database.close();
    }
  });

  it('event_id PK uniqueness prevents duplicate inserts', () => {
    // Google Calendar event IDs are globally unique within an
    // account; PK uniqueness means a duplicate-id insert (e.g. a
    // re-fetch racing with itself) fails loudly rather than silently
    // shadowing the first row.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO calendar_snapshots (date, fetched_at)
           VALUES (?, ?)`,
        )
        .run('2026-04-30', '2026-04-30T07:00:00Z');
      database
        .prepare(
          `INSERT INTO calendar_events (event_id, date, title, start)
           VALUES (?, ?, ?, ?)`,
        )
        .run('evt-dup', '2026-04-30', 'First', '2026-04-30T09:00:00Z');
      expect(() =>
        database
          .prepare(
            `INSERT INTO calendar_events (event_id, date, title, start)
             VALUES (?, ?, ?, ?)`,
          )
          .run('evt-dup', '2026-04-30', 'Second', '2026-04-30T10:00:00Z'),
      ).toThrow(/UNIQUE constraint failed/);
    } finally {
      database.close();
    }
  });
});
