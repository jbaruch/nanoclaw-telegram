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
];

describe('state-009-phase-completions', () => {
  it('creates phase_completions table and bumps user_version to 9', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(9);
      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      const names = tables.map((t) => t.name);
      expect(names).toContain('phase_completions');
    } finally {
      database.close();
    }
  });

  it('phase_completions has correct columns + nullability + defaults per issue spec', () => {
    // The issue spec calls for TEXT PK `phase`, NOT NULL
    // `last_completed`, nullable `metadata`, NOT NULL `updated_at`
    // defaulting to CURRENT_TIMESTAMP. Lock the `schema_version`
    // default (`'1'`) and `updated_at` default (`'CURRENT_TIMESTAMP'`)
    // down at the same time — a future "tighten the schema" refactor
    // that drops a default and turns every existing INSERT into a
    // NOT NULL constraint failure must fail here, not in production.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(phase_completions)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
        dflt_value: string | null;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['phase']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['last_completed']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      expect(byName['metadata']).toMatchObject({
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

  it('phase PK uniqueness prevents duplicate plain INSERTs', () => {
    // The `phase` string is the natural uniqueness contract — exactly
    // one row per phase, ever. A plain duplicate INSERT (without the
    // ON CONFLICT clause) must fail loudly so a writer that skips the
    // UPSERT and falls back to a plain INSERT can't silently shadow
    // the existing row.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO phase_completions (phase, last_completed)
           VALUES (?, ?)`,
        )
        .run('heartbeat', '2026-04-30T07:00:00Z');
      expect(() =>
        database
          .prepare(
            `INSERT INTO phase_completions (phase, last_completed)
             VALUES (?, ?)`,
          )
          .run('heartbeat', '2026-04-30T08:00:00Z'),
      ).toThrow(/UNIQUE constraint failed/);
    } finally {
      database.close();
    }
  });

  it('metadata accepts a JSON-string blob and reads it back verbatim', () => {
    // `metadata` is opaque TEXT to the schema — readers parse it as
    // JSON in their own code. Verify a JSON string round-trips byte-
    // for-byte (no SQLite-side mangling, no normalization). This is
    // the contract the `last_composio_check`-on-heartbeat writer
    // (#301 follow-up) relies on.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const blob = '{"last_composio_check":"2026-04-30T07:00:00Z"}';
      database
        .prepare(
          `INSERT INTO phase_completions (phase, last_completed, metadata)
           VALUES (?, ?, ?)`,
        )
        .run('heartbeat', '2026-04-30T07:00:00Z', blob);
      const row = database
        .prepare(`SELECT metadata FROM phase_completions WHERE phase = ?`)
        .get('heartbeat') as { metadata: string };
      expect(row.metadata).toBe(blob);
    } finally {
      database.close();
    }
  });

  it('UPSERT updates last_completed in place via ON CONFLICT(phase) DO UPDATE', () => {
    // The headline contract the issue spec calls out:
    // `mark-phase-complete.py` becomes a single
    //   INSERT ... ON CONFLICT(phase) DO UPDATE SET ...
    // The LOCK_EX ceremony disappears. Verify the UPSERT updates the
    // existing row in place, the row count stays at 1, and SELECT
    // returns the new last_completed.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO phase_completions (phase, last_completed)
           VALUES (?, ?)
           ON CONFLICT(phase) DO UPDATE SET
             last_completed = excluded.last_completed,
             metadata       = excluded.metadata,
             updated_at     = CURRENT_TIMESTAMP`,
        )
        .run('heartbeat', '2026-04-30T07:00:00Z');
      database
        .prepare(
          `INSERT INTO phase_completions (phase, last_completed)
           VALUES (?, ?)
           ON CONFLICT(phase) DO UPDATE SET
             last_completed = excluded.last_completed,
             metadata       = excluded.metadata,
             updated_at     = CURRENT_TIMESTAMP`,
        )
        .run('heartbeat', '2026-04-30T08:00:00Z');

      const count = database
        .prepare('SELECT COUNT(*) AS n FROM phase_completions')
        .get() as { n: number };
      expect(count.n).toBe(1);

      const row = database
        .prepare(`SELECT last_completed FROM phase_completions WHERE phase = ?`)
        .get('heartbeat') as { last_completed: string };
      expect(row.last_completed).toBe('2026-04-30T08:00:00Z');
    } finally {
      database.close();
    }
  });

  it('updated_at is non-decreasing across UPDATEs', () => {
    // `updated_at` is the row-mutation timestamp. SQLite's
    // CURRENT_TIMESTAMP is second-resolution, so back-to-back
    // UPSERTs in the same test run can produce equal timestamps —
    // assert non-decreasing (>=), not strictly increasing. The point
    // is that the UPDATE branch's `updated_at = CURRENT_TIMESTAMP`
    // clause re-stamps the column rather than carrying the original
    // INSERT-time value forward forever.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO phase_completions (phase, last_completed)
           VALUES (?, ?)`,
        )
        .run('heartbeat', '2026-04-30T07:00:00Z');
      const before = database
        .prepare(`SELECT updated_at FROM phase_completions WHERE phase = ?`)
        .get('heartbeat') as { updated_at: string };
      database
        .prepare(
          `INSERT INTO phase_completions (phase, last_completed)
           VALUES (?, ?)
           ON CONFLICT(phase) DO UPDATE SET
             last_completed = excluded.last_completed,
             metadata       = excluded.metadata,
             updated_at     = CURRENT_TIMESTAMP`,
        )
        .run('heartbeat', '2026-04-30T08:00:00Z');
      const after = database
        .prepare(`SELECT updated_at FROM phase_completions WHERE phase = ?`)
        .get('heartbeat') as { updated_at: string };
      // Lex compare on CURRENT_TIMESTAMP's `YYYY-MM-DD HH:MM:SS`
      // format is monotonic — non-decreasing is the correct contract.
      expect(after.updated_at >= before.updated_at).toBe(true);
    } finally {
      database.close();
    }
  });

  it('repeated UPSERTs of the same phase keep COUNT(*) at 1', () => {
    // The ON CONFLICT path must not create duplicate rows. With
    // `phase` as the PK and the conflict clause set to UPDATE, ten
    // back-to-back UPSERTs on the same phase must leave the table
    // with exactly one row — that's the structural property that
    // makes the JSON-era LOCK_EX unnecessary. Spans heartbeat /
    // nightly / weekly to exercise the three real-world phases.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const upsert = database.prepare(
        `INSERT INTO phase_completions (phase, last_completed)
         VALUES (?, ?)
         ON CONFLICT(phase) DO UPDATE SET
           last_completed = excluded.last_completed,
           metadata       = excluded.metadata,
           updated_at     = CURRENT_TIMESTAMP`,
      );
      for (let i = 0; i < 10; i++) {
        upsert.run('heartbeat', `2026-04-30T07:0${i}:00Z`);
        upsert.run('nightly', `2026-04-30T02:0${i}:00Z`);
        upsert.run('weekly', `2026-04-27T02:0${i}:00Z`);
      }
      const count = database
        .prepare('SELECT COUNT(*) AS n FROM phase_completions')
        .get() as { n: number };
      expect(count.n).toBe(3);
      const phases = database
        .prepare('SELECT phase FROM phase_completions ORDER BY phase')
        .all() as Array<{ phase: string }>;
      expect(phases.map((p) => p.phase)).toEqual([
        'heartbeat',
        'nightly',
        'weekly',
      ]);
    } finally {
      database.close();
    }
  });

  it('last_completed enforces NOT NULL', () => {
    // A phase row without a completion timestamp is meaningless —
    // heartbeat-precheck would have to defend against NULL on every
    // read. DB-level NOT NULL means a malformed insert fails loudly
    // rather than landing a useless row that silently breaks the
    // next phase's "should I run?" decision.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(() =>
        database
          .prepare(
            `INSERT INTO phase_completions (phase, last_completed)
             VALUES (?, ?)`,
          )
          .run('heartbeat', null),
      ).toThrow(/NOT NULL constraint failed/);
    } finally {
      database.close();
    }
  });
});
