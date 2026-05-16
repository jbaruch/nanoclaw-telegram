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
import { STATE_014_LOCATIONS } from './state-014-locations.js';

const ALL_MIGRATIONS = [
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
  STATE_014_LOCATIONS,
];

describe('state-014-locations', () => {
  it('creates the locations table and bumps user_version to 14', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL_MIGRATIONS);

      expect(Number(database.pragma('user_version', { simple: true }))).toBe(
        14,
      );

      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      expect(tables.map((t) => t.name)).toContain('locations');
    } finally {
      database.close();
    }
  });

  it('declares all columns with correct types and constraints', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL_MIGRATIONS);

      const cols = database
        .prepare('PRAGMA table_info(locations)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));

      expect(byName['id']).toMatchObject({ type: 'INTEGER', pk: 1 });
      expect(byName['chat_jid']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['sender']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['message_id']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['latitude']).toMatchObject({ type: 'REAL', notnull: 1 });
      expect(byName['longitude']).toMatchObject({ type: 'REAL', notnull: 1 });
      expect(byName['accuracy_m']).toMatchObject({ notnull: 0 }); // nullable
      expect(byName['source']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['recorded_at']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['live_period']).toMatchObject({ notnull: 0 }); // nullable
    } finally {
      database.close();
    }
  });

  it('creates idx_locations_sender_time index', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL_MIGRATIONS);

      const indexes = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index'")
        .all() as Array<{ name: string }>;
      expect(indexes.map((i) => i.name)).toContain('idx_locations_sender_time');
    } finally {
      database.close();
    }
  });

  it('is idempotent when run against a v13 database (skips already-applied migrations)', () => {
    const database = new Database(':memory:');
    try {
      // Apply all migrations including 014.
      applyStateMigrations(database, ALL_MIGRATIONS);
      // Running again must not throw (validator skips applied versions).
      expect(() =>
        applyStateMigrations(database, ALL_MIGRATIONS),
      ).not.toThrow();
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(
        14,
      );
    } finally {
      database.close();
    }
  });
});
