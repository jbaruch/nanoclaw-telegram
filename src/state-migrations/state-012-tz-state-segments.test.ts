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
import { STATE_011_SESSION_LENGTH_CAP } from './state-011-session-length-cap.js';
import { STATE_012_TZ_STATE_SEGMENTS } from './state-012-tz-state-segments.js';

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
  STATE_011_SESSION_LENGTH_CAP,
  STATE_012_TZ_STATE_SEGMENTS,
];

const PRIOR = ALL.slice(0, ALL.length - 1);

describe('state-012-tz-state-segments', () => {
  it('adds segments column and bumps user_version to 12', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(
        12,
      );
      const cols = database
        .prepare('PRAGMA table_info(tz_state)')
        .all() as Array<{ name: string; type: string; notnull: number }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['segments']).toMatchObject({
        type: 'TEXT',
        notnull: 0,
      });
    } finally {
      database.close();
    }
  });

  it('preserves singleton CHECK(id = 1) and NOT NULL on current_tz / home_tz', () => {
    // ALTER TABLE ADD COLUMN must not regress the constraints declared
    // by state-010 — those guarantees (singleton, load-bearing
    // current_tz / home_tz) are still load-bearing for every reader.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(() =>
        database
          .prepare(
            `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
             VALUES (2, ?, ?, ?)`,
          )
          .run('Europe/Berlin', 'America/Chicago', null),
      ).toThrow(/CHECK constraint failed/);
      expect(() =>
        database
          .prepare(
            `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
             VALUES (1, ?, ?, ?)`,
          )
          .run(null, 'America/Chicago', null),
      ).toThrow(/NOT NULL constraint failed/);
    } finally {
      database.close();
    }
  });

  it('UPDATE bumps the existing row from schema_version=1 to schema_version=2', () => {
    // Mid-migration scenario: a deployment that ran state-010 + state-
    // 011 already has a tz_state row at schema_version=1 written by
    // _seedTzStateForTests / migrateTaskTzStateJsonFiles. The state-
    // 012 UPDATE must bump that row to 2 so the existing reader gate
    // (SUPPORTED_TZ_STATE_SCHEMA_VERSION = 2) recognizes it on first
    // read after the migration. Otherwise getCurrentTz() would return
    // null until the next sync_tripit run rewrote the row.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, PRIOR);
      database
        .prepare(
          `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
           VALUES (1, ?, ?, ?)`,
        )
        .run('America/Chicago', 'America/Chicago', null);
      // Before the new migration runs, the existing row sits at the
      // state-010 default of schema_version = 1.
      const before = database
        .prepare('SELECT schema_version FROM tz_state WHERE id = 1')
        .get() as { schema_version: number };
      expect(before.schema_version).toBe(1);

      applyStateMigrations(database, ALL);

      const after = database
        .prepare(
          'SELECT current_tz, home_tz, segments, schema_version FROM tz_state WHERE id = 1',
        )
        .get() as {
        current_tz: string;
        home_tz: string;
        segments: string | null;
        schema_version: number;
      };
      expect(after.schema_version).toBe(2);
      expect(after.current_tz).toBe('America/Chicago');
      expect(after.home_tz).toBe('America/Chicago');
      // The migration only adds the column with NULL on existing rows
      // — the writer (`applyTripitSegmentsToTzState`) is what
      // populates `segments` on the next `sync_tripit` run.
      expect(after.segments).toBeNull();
    } finally {
      database.close();
    }
  });

  it('UPDATE is a no-op on a fresh DB (no row yet) and does not error', () => {
    // First-deploy / fresh-test scenario: state-010 created the table
    // but no `_seedTzStateForTests` / `migrateTaskTzStateJsonFiles`
    // call has landed a row yet. The state-012 UPDATE must be safe —
    // SQLite's UPDATE on zero rows is a no-op, but verify
    // explicitly so a future operator who reorders the UPDATE
    // statement (e.g. into a runtime UPDATE … RETURNING) doesn't
    // ship a regression that breaks first-time installs.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const count = database
        .prepare('SELECT COUNT(*) AS n FROM tz_state')
        .get() as { n: number };
      expect(count.n).toBe(0);
      // A subsequent insert should land schema_version = 1 from the
      // state-010 column default — the state-012 migration didn't
      // change the default, only the existing row's value. Future
      // inserts go through `_seedTzStateForTests` (test path) or
      // `applyTripitSegmentsToTzState` (production path), both of
      // which write `schema_version = 2` explicitly.
      database
        .prepare(
          `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
           VALUES (1, ?, ?, ?)`,
        )
        .run('America/Chicago', 'America/Chicago', null);
      const row = database
        .prepare('SELECT schema_version, segments FROM tz_state WHERE id = 1')
        .get() as { schema_version: number; segments: string | null };
      expect(row.schema_version).toBe(1);
      expect(row.segments).toBeNull();
    } finally {
      database.close();
    }
  });

  it('segments column accepts JSON-serialized array text', () => {
    // The column is plain TEXT — SQLite has no JSON column type, but
    // the writer contract is "JSON.stringify the segments[] payload."
    // Spot-check that a representative payload round-trips through
    // INSERT/SELECT without truncation or encoding loss.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const segments = JSON.stringify([
        {
          timezone: 'Europe/Berlin',
          from: '2026-05-12',
          to: '2026-05-19',
          label: 'Devoxx UK 2026 - London',
        },
        {
          timezone: 'America/New_York',
          from: '2026-06-03',
          to: '2026-06-08',
          label: 'WeAreDevelopers - New York',
        },
      ]);
      database
        .prepare(
          `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, segments, schema_version)
           VALUES (1, ?, ?, ?, ?, 2)`,
        )
        .run('Europe/Berlin', 'America/Chicago', null, segments);
      const row = database
        .prepare('SELECT segments FROM tz_state WHERE id = 1')
        .get() as { segments: string };
      expect(row.segments).toBe(segments);
      const parsed = JSON.parse(row.segments) as Array<{
        timezone: string;
        from: string;
        to: string;
      }>;
      expect(parsed).toHaveLength(2);
      expect(parsed[0]?.timezone).toBe('Europe/Berlin');
      expect(parsed[1]?.from).toBe('2026-06-03');
    } finally {
      database.close();
    }
  });
});
