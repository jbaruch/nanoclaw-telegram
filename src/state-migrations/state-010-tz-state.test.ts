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
import { STATE_009_PHASE_COMPLETIONS } from './state-009-phase-completions.js';
import { STATE_010_TZ_STATE } from './state-010-tz-state.js';

const ALL = [
  STATE_001_ORDERS,
  STATE_002_EMAIL_FEEDBACK,
  STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
  STATE_004_SCHEDULED_REMINDERS,
  STATE_005_NANOCLAW_STATE_SPLIT,
  STATE_006_TRUSTED_SESSION_STATE,
  STATE_007_MORNING_BRIEF_PENDING,
  STATE_008_CALENDAR_STATE,
  STATE_009_PHASE_COMPLETIONS,
  STATE_010_TZ_STATE,
];

describe('state-010-tz-state', () => {
  it('creates tz_state + follow_me_tasks tables and bumps user_version to 10', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(
        10,
      );
      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      const names = tables.map((t) => t.name);
      expect(names).toContain('tz_state');
      expect(names).toContain('follow_me_tasks');
    } finally {
      database.close();
    }
  });

  it('tz_state has correct columns + nullability + defaults per issue spec', () => {
    // The issue spec calls for INTEGER PK `id` with CHECK(id = 1)
    // singleton, NOT NULL `current_tz` and `home_tz`, nullable
    // `scheduler_tz` (informational only; no longer load-bearing per
    // spec), and `schema_version INTEGER NOT NULL DEFAULT 1`. Lock
    // the `schema_version` default down to the literal `'1'` from
    // the start — a future "tighten the schema" refactor that drops
    // the default and turns every column-list-omitting INSERT into
    // a NOT NULL constraint failure must fail here, not in
    // production.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(tz_state)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
        dflt_value: string | null;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['id']).toMatchObject({ type: 'INTEGER', pk: 1 });
      expect(byName['current_tz']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      expect(byName['home_tz']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      expect(byName['scheduler_tz']).toMatchObject({
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

  it('tz_state singleton CHECK enforces id = 1', () => {
    // Mirrors the email_state singleton from state-005. The CHECK
    // constraint exists so a misbehaving writer that hand-crafts
    // an insert with id=2 (instead of UPSERT-ing onto id=1) fails
    // loudly with a CHECK constraint failure rather than silently
    // shadowing the canonical id=1 row, which would leave readers
    // racing on "which row is the real timezone configuration".
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      // Insert with id=1 succeeds.
      database
        .prepare(
          `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
           VALUES (1, ?, ?, ?)`,
        )
        .run('America/Chicago', 'America/Chicago', 'America/Chicago');
      // Insert with id=2 violates the CHECK constraint.
      expect(() =>
        database
          .prepare(
            `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
             VALUES (2, ?, ?, ?)`,
          )
          .run('Europe/Berlin', 'America/Chicago', 'Europe/Berlin'),
      ).toThrow(/CHECK constraint failed/);
      // Re-inserting id=1 violates the PRIMARY KEY.
      expect(() =>
        database
          .prepare(
            `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
             VALUES (1, ?, ?, ?)`,
          )
          .run('America/New_York', 'America/Chicago', null),
      ).toThrow(/UNIQUE constraint failed/);
    } finally {
      database.close();
    }
  });

  it('tz_state enforces NOT NULL on current_tz and home_tz', () => {
    // current_tz and home_tz are load-bearing for every reader
    // (morning-brief / nightly / weekly / check-calendar /
    // heartbeat-precheck). DB-level NOT NULL means a malformed
    // insert fails loudly rather than landing a row that every
    // reader then has to defend against. scheduler_tz is
    // intentionally not asserted NOT NULL — the spec calls it
    // "informational only; no longer load-bearing".
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(() =>
        database
          .prepare(
            `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
             VALUES (1, ?, ?, ?)`,
          )
          .run(null, 'America/Chicago', null),
      ).toThrow(/NOT NULL constraint failed/);
      expect(() =>
        database
          .prepare(
            `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
             VALUES (1, ?, ?, ?)`,
          )
          .run('America/Chicago', null, null),
      ).toThrow(/NOT NULL constraint failed/);
    } finally {
      database.close();
    }
  });

  it('follow_me_tasks has correct columns + nullability + defaults per issue spec', () => {
    // The issue spec calls for TEXT PK `name`, NOT NULL `local_time`
    // and `schedule_value`, nullable `last_run_date` /
    // `pending_run_at`, NOT NULL `updated_at` defaulting to
    // CURRENT_TIMESTAMP, and `schema_version INTEGER NOT NULL
    // DEFAULT 1`. Lock both default literals (`'CURRENT_TIMESTAMP'`
    // and `'1'`) down from the start — a future "tighten the
    // schema" refactor that drops a default and turns every
    // column-list-omitting INSERT into a NOT NULL constraint
    // failure must fail here, not in production.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(follow_me_tasks)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
        dflt_value: string | null;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['name']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['local_time']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      expect(byName['schedule_value']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      expect(byName['last_run_date']).toMatchObject({
        type: 'TEXT',
        notnull: 0,
      });
      expect(byName['pending_run_at']).toMatchObject({
        type: 'TEXT',
        notnull: 0,
      });
      expect(byName['updated_at']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
        dflt_value: 'CURRENT_TIMESTAMP',
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

  it('follow_me_tasks UPSERT updates existing row in place via ON CONFLICT(name)', () => {
    // The headline contract: cursor mutations on follow_me_tasks
    // (last_run_date / pending_run_at) flow through a single
    //   INSERT ... ON CONFLICT(name) DO UPDATE SET ...
    // The JSON-era array-rewrite + LOCK_EX disappear. Verify the
    // UPSERT updates the existing row in place, the row count
    // stays at N (here 3 — morning-brief / nightly-housekeeping /
    // weekly-housekeeping), and SELECT returns the new
    // schedule_value on the morning-brief row.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const upsert = database.prepare(
        `INSERT INTO follow_me_tasks
           (name, local_time, schedule_value, last_run_date, pending_run_at)
           VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(name) DO UPDATE SET
           local_time     = excluded.local_time,
           schedule_value = excluded.schedule_value,
           last_run_date  = excluded.last_run_date,
           pending_run_at = excluded.pending_run_at,
           updated_at     = CURRENT_TIMESTAMP`,
      );
      upsert.run('morning-brief', '08:00', '0 13 * * *', null, null);
      upsert.run('nightly-housekeeping', '02:00', '0 7 * * *', null, null);
      upsert.run('weekly-housekeeping', '02:00', '0 7 * * 0', null, null);

      // UPSERT the morning-brief row with a new schedule_value (the
      // user moved their wake time). Row count stays at 3.
      upsert.run('morning-brief', '07:30', '30 12 * * *', '2026-04-29', null);

      const count = database
        .prepare('SELECT COUNT(*) AS n FROM follow_me_tasks')
        .get() as { n: number };
      expect(count.n).toBe(3);

      const row = database
        .prepare(
          `SELECT local_time, schedule_value, last_run_date
             FROM follow_me_tasks WHERE name = ?`,
        )
        .get('morning-brief') as {
        local_time: string;
        schedule_value: string;
        last_run_date: string;
      };
      expect(row).toMatchObject({
        local_time: '07:30',
        schedule_value: '30 12 * * *',
        last_run_date: '2026-04-29',
      });
    } finally {
      database.close();
    }
  });

  it('SQL transaction replaces the LOCK_EX + Phase-C/Phase-D re-read pattern', () => {
    // The follow-me-two-phase-lock.md rule's "re-read in Phase C /
    // Phase D" pattern becomes a `BEGIN IMMEDIATE; SELECT; UPDATE;
    // COMMIT;` envelope as documented in the migration header. The
    // doc recommends IMMEDIATE specifically (RESERVED lock taken up
    // front) so a concurrent writer's `BEGIN IMMEDIATE` blocks
    // rather than racing the read-then-write window. To actually
    // exercise that flavour, the test uses raw `database.exec`
    // calls — `database.transaction(fn)` in better-sqlite3 emits a
    // plain (deferred) `BEGIN`, which would not exercise the
    // RESERVED-lock semantics the doc relies on.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO follow_me_tasks
             (name, local_time, schedule_value, last_run_date, pending_run_at)
             VALUES (?, ?, ?, ?, ?)`,
        )
        .run('morning-brief', '08:00', '0 13 * * *', '2026-04-28', null);

      database.exec('BEGIN IMMEDIATE');
      try {
        // Phase C: re-read inside the transaction.
        const before = database
          .prepare(`SELECT last_run_date FROM follow_me_tasks WHERE name = ?`)
          .get('morning-brief') as { last_run_date: string };
        // Reasoning step: did anything change since Phase A
        // observed the row? (In production this is the morning-
        // brief skill deciding whether to fire today.) Here we
        // just verify the read returned the seeded value.
        expect(before.last_run_date).toBe('2026-04-28');
        // Phase D: write the new cursor value inside the same
        // transaction — no LOCK_EX needed.
        database
          .prepare(
            `UPDATE follow_me_tasks
                SET last_run_date  = ?,
                    pending_run_at = NULL,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE name = ?`,
          )
          .run('2026-04-30', 'morning-brief');
        database.exec('COMMIT');
      } catch (err) {
        database.exec('ROLLBACK');
        throw err;
      }

      // Post-commit observability: the new cursor is visible on a
      // fresh SELECT outside the transaction.
      const after = database
        .prepare(
          `SELECT last_run_date, pending_run_at
             FROM follow_me_tasks WHERE name = ?`,
        )
        .get('morning-brief') as {
        last_run_date: string;
        pending_run_at: string | null;
      };
      expect(after.last_run_date).toBe('2026-04-30');
      expect(after.pending_run_at).toBeNull();
    } finally {
      database.close();
    }
  });

  it('follow_me_tasks enforces NOT NULL on local_time and schedule_value', () => {
    // A follow-me task without a wall-clock time or a UTC-cron
    // string is unschedulable — task-tz-sync's "compute the next
    // run" math would have nothing to compute against. DB-level
    // NOT NULL means a malformed insert fails loudly rather than
    // landing a row that silently breaks the scheduler. Both
    // last_run_date and pending_run_at are intentionally NOT
    // asserted NOT NULL — they're legitimately NULL on the
    // never-run-yet path and outside the Phase-C-to-Phase-D
    // window respectively.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(() =>
        database
          .prepare(
            `INSERT INTO follow_me_tasks
               (name, local_time, schedule_value, last_run_date, pending_run_at)
               VALUES (?, ?, ?, ?, ?)`,
          )
          .run('morning-brief', null, '0 13 * * *', null, null),
      ).toThrow(/NOT NULL constraint failed/);
      expect(() =>
        database
          .prepare(
            `INSERT INTO follow_me_tasks
               (name, local_time, schedule_value, last_run_date, pending_run_at)
               VALUES (?, ?, ?, ?, ?)`,
          )
          .run('morning-brief', '08:00', null, null, null),
      ).toThrow(/NOT NULL constraint failed/);
    } finally {
      database.close();
    }
  });

  it('follow_me_tasks PK uniqueness prevents duplicate plain INSERTs', () => {
    // The `name` string is the natural uniqueness contract — exactly
    // one row per follow-me task, ever. A plain duplicate INSERT
    // (without the ON CONFLICT clause) must fail loudly so a
    // writer that skips the UPSERT and falls back to a plain
    // INSERT can't silently shadow the existing row, splitting the
    // task's cursor across two rows.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO follow_me_tasks
             (name, local_time, schedule_value, last_run_date, pending_run_at)
             VALUES (?, ?, ?, ?, ?)`,
        )
        .run('morning-brief', '08:00', '0 13 * * *', null, null);
      expect(() =>
        database
          .prepare(
            `INSERT INTO follow_me_tasks
               (name, local_time, schedule_value, last_run_date, pending_run_at)
               VALUES (?, ?, ?, ?, ?)`,
          )
          .run('morning-brief', '07:00', '0 12 * * *', null, null),
      ).toThrow(/UNIQUE constraint failed/);
    } finally {
      database.close();
    }
  });
});
