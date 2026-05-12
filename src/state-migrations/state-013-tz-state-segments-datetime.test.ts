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
import { STATE_013_TZ_STATE_SEGMENTS_DATETIME } from './state-013-tz-state-segments-datetime.js';

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
  STATE_013_TZ_STATE_SEGMENTS_DATETIME,
];

const PRIOR = ALL.slice(0, ALL.length - 1);

describe('state-013-tz-state-segments-datetime', () => {
  it('bumps user_version to 13 without altering the tz_state column shape', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(
        13,
      );
      // No DDL — the segments column stays TEXT, nullable. The new
      // datetime fields live INSIDE each segment's JSON object, not
      // as new SQL columns.
      const cols = database
        .prepare('PRAGMA table_info(tz_state)')
        .all() as Array<{ name: string; type: string; notnull: number }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['segments']).toMatchObject({ type: 'TEXT', notnull: 0 });
      // schema_version's column default stays at 1 (state-010); only
      // the existing row's value advances, controlled by the writer
      // gate constant.
      expect(byName['schema_version']).toMatchObject({
        type: 'INTEGER',
        notnull: 1,
      });
    } finally {
      database.close();
    }
  });

  it('UPDATE bumps an existing v2 row to v3 in-place', () => {
    // The realistic post-state-012 mid-migration shape: a deployment
    // ran state-010..012 and `_seedTzStateForTests` /
    // `applyTripitSegmentsToTzState` left a row at schema_version=2
    // with date-only segments. state-013 bumps that to 3 so the
    // new reader gate accepts it.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, PRIOR);
      const segmentsJson = JSON.stringify([
        {
          timezone: 'America/Chicago',
          from: '2026-05-12',
          to: '2026-05-19',
          label: 'pre-datetime',
        },
      ]);
      database
        .prepare(
          `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, segments, schema_version)
           VALUES (1, ?, ?, ?, ?, 2)`,
        )
        .run('America/Chicago', 'America/Chicago', null, segmentsJson);

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
      expect(after.schema_version).toBe(3);
      expect(after.current_tz).toBe('America/Chicago');
      expect(after.home_tz).toBe('America/Chicago');
      // The migration MUST NOT touch the JSON payload — segments
      // stays as-shipped by the prior writer. The walker handles
      // mixed shapes (date-only legacy vs. datetime post-deploy)
      // per-segment; rewriting on migrate would force a flag-day.
      expect(after.segments).toBe(segmentsJson);
    } finally {
      database.close();
    }
  });

  it('UPDATE is conservative: a row already at v3 stays at v3', () => {
    // Defense against an out-of-order replay: if a hypothetical
    // future writer raced ahead and wrote v3 directly before this
    // migration ran, the WHERE clause's `schema_version = 2` guard
    // skips the row rather than blindly clobbering it. v3 stays v3.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, PRIOR);
      database
        .prepare(
          `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, schema_version)
           VALUES (1, ?, ?, ?, 3)`,
        )
        .run('America/Chicago', 'America/Chicago', null);
      applyStateMigrations(database, ALL);
      const after = database
        .prepare('SELECT schema_version FROM tz_state WHERE id = 1')
        .get() as { schema_version: number };
      expect(after.schema_version).toBe(3);
    } finally {
      database.close();
    }
  });

  it('UPDATE is a no-op on a fresh DB with no row', () => {
    // First-deploy scenario: state-010 created the table but no row
    // exists yet. The state-013 UPDATE must not error.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const count = database
        .prepare('SELECT COUNT(*) AS n FROM tz_state')
        .get() as { n: number };
      expect(count.n).toBe(0);
    } finally {
      database.close();
    }
  });

  it('chains a v1 row through state-012 (v1 → v2) then state-013 (v2 → v3)', () => {
    // The realistic startup path for a deployment that was on v1
    // before state-012 ever ran: state-010 + state-011 created the
    // table and the row landed at the column default of 1. When
    // state-012 + state-013 land in the same orchestrator start,
    // state-012's `UPDATE WHERE id = 1` bumps v1 → v2 first, then
    // state-013's `UPDATE WHERE schema_version = 2` bumps v2 → v3.
    // The final value is 3 because BOTH migrations ran in order;
    // skipping either would leave the row at the prior version.
    const database = new Database(':memory:');
    try {
      // Apply every migration before state-012, leaving the schema
      // at the pre-segments-column state-010 + state-011 shape. The
      // earlier state-001..009 migrations apply unrelated tables;
      // what matters for this test is that the `tz_state` table
      // exists (from state-010) and has no `segments` column yet
      // (state-012 adds it).
      applyStateMigrations(database, PRIOR.slice(0, PRIOR.length - 1));
      database
        .prepare(
          `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, schema_version)
           VALUES (1, ?, ?, ?, 1)`,
        )
        .run('America/Chicago', 'America/Chicago', null);
      // Apply the remaining migrations — state-012 bumps the v1 row
      // to v2 (via `UPDATE WHERE id = 1`), then state-013 bumps the
      // v2 row to v3 (via `UPDATE WHERE schema_version = 2`).
      applyStateMigrations(database, ALL);
      const after = database
        .prepare('SELECT schema_version FROM tz_state WHERE id = 1')
        .get() as { schema_version: number };
      expect(after.schema_version).toBe(3);
    } finally {
      database.close();
    }
  });

  it("guards against leapfrog: state-013 alone won't bump a v1 row past v1", () => {
    // Defense-in-depth for an operator-introduced corruption: a v1
    // row that somehow survived state-012 (e.g. a manual rollback
    // that re-INSERTed at v1 between migrations) must NOT be
    // bumped to v3 by state-013. The `WHERE schema_version = 2`
    // guard is the safety here; a less-defensive `WHERE id = 1`
    // would clobber the v1 row to v3 and skip the segments-column
    // ADD COLUMN invariant state-012 carries.
    //
    // Setup: apply every migration through state-012 (PRIOR — the
    // full set of state-001..012) so the `tz_state.segments` column
    // exists. Then INSERT a v1 row directly, bypassing the writer's
    // normal path so we can pin the pre-state-013 version exactly.
    // Finally, exec state-013's SQL on its own — direct `exec`
    // rather than `applyStateMigrations` because we want to
    // exercise just this migration's WHERE clause against a v1 row,
    // without touching `user_version` bookkeeping.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, PRIOR);
      database
        .prepare(
          `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, schema_version)
           VALUES (1, ?, ?, ?, 1)`,
        )
        .run('America/Chicago', 'America/Chicago', null);

      database.exec(STATE_013_TZ_STATE_SEGMENTS_DATETIME.sql);

      const after = database
        .prepare('SELECT schema_version FROM tz_state WHERE id = 1')
        .get() as { schema_version: number };
      expect(after.schema_version).toBe(1);
    } finally {
      database.close();
    }
  });
});
