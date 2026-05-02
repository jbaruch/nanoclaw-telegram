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
];

describe('state-011-session-length-cap', () => {
  it('creates the session_length_state table and bumps user_version to 11', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(
        11,
      );
      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      expect(tables.map((t) => t.name)).toContain('session_length_state');
    } finally {
      database.close();
    }
  });

  it('does NOT collide with the orchestrator-level `sessions` table', () => {
    // Sanity: the new cap-state table is namespaced separately from
    // the existing `sessions` table (SDK session-id mapping). A
    // composite-named table avoids the same collision risk that
    // state-006 documents — both tables coexist independently.
    const database = new Database(':memory:');
    try {
      database.exec(`
        CREATE TABLE sessions (
          group_folder TEXT NOT NULL,
          session_name TEXT NOT NULL DEFAULT 'default',
          session_id TEXT NOT NULL,
          PRIMARY KEY (group_folder, session_name)
        );
      `);
      applyStateMigrations(database, ALL);
      const tables = database
        .prepare(
          `SELECT name FROM sqlite_master
            WHERE type = 'table' AND name LIKE '%session%'
            ORDER BY name`,
        )
        .all() as Array<{ name: string }>;
      const names = tables.map((t) => t.name);
      expect(names).toContain('sessions');
      expect(names).toContain('session_length_state');
    } finally {
      database.close();
    }
  });

  it('declares schema_version with NOT NULL DEFAULT 1 per stateful-artifacts', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(session_length_state)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        dflt_value: unknown;
        pk: number;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['schema_version']).toMatchObject({
        type: 'INTEGER',
        notnull: 1,
      });
      expect(String(byName['schema_version'].dflt_value)).toBe('1');
    } finally {
      database.close();
    }
  });

  it('declares (group_folder, session_name) as the composite primary key', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(session_length_state)')
        .all() as Array<{ name: string; pk: number }>;
      const pkCols = cols
        .filter((c) => c.pk > 0)
        .map((c) => c.name)
        .sort();
      expect(pkCols).toEqual(['group_folder', 'session_name']);
    } finally {
      database.close();
    }
  });

  it('carries reset_reason and reset_cap nullable columns for the consume path', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(session_length_state)')
        .all() as Array<{ name: string; notnull: number; type: string }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['reset_reason']).toMatchObject({
        type: 'TEXT',
        notnull: 0,
      });
      expect(byName['reset_cap']).toMatchObject({
        type: 'INTEGER',
        notnull: 0,
      });
    } finally {
      database.close();
    }
  });
});
