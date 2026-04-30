import Database from 'better-sqlite3';
import { describe, it, expect } from 'vitest';

import { applyStateMigrations } from '../db.js';

import { STATE_001_ORDERS } from './state-001-orders.js';
import { STATE_002_EMAIL_FEEDBACK } from './state-002-email-feedback.js';
import { STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION } from './state-003-email-feedback-schema-version.js';
import { STATE_004_SCHEDULED_REMINDERS } from './state-004-scheduled-reminders.js';
import { STATE_005_NANOCLAW_STATE_SPLIT } from './state-005-nanoclaw-state-split.js';
import { STATE_006_TRUSTED_SESSION_STATE } from './state-006-trusted-session-state.js';

const ALL = [
  STATE_001_ORDERS,
  STATE_002_EMAIL_FEEDBACK,
  STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
  STATE_004_SCHEDULED_REMINDERS,
  STATE_005_NANOCLAW_STATE_SPLIT,
  STATE_006_TRUSTED_SESSION_STATE,
];

describe('state-006-trusted-session-state', () => {
  it('creates both tables and bumps user_version to 6', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(6);
      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      const names = tables.map((t) => t.name);
      expect(names).toContain('trusted_sessions');
      expect(names).toContain('trusted_session_singleton');
    } finally {
      database.close();
    }
  });

  it('does NOT collide with the orchestrator-level `sessions` table', () => {
    // The orchestrator's src/db.ts already declares a `sessions`
    // table for SDK session-id tracking per (group_folder,
    // session_name). The trusted-memory split was originally
    // proposed as `sessions` too, which would have errored loudly
    // at migration apply time ("table sessions already exists" —
    // SQLite's CREATE TABLE without IF NOT EXISTS fails on
    // duplicate names) and broken every orchestrator startup
    // post-deploy. The `trusted_` prefix is the namespace boundary
    // — verify both tables can coexist with no collision.
    const database = new Database(':memory:');
    try {
      // Orchestrator-style sessions DDL (matches src/db.ts shape).
      database.exec(`
        CREATE TABLE sessions (
          group_folder TEXT NOT NULL,
          session_name TEXT NOT NULL DEFAULT 'default',
          session_id TEXT NOT NULL,
          PRIMARY KEY (group_folder, session_name)
        );
      `);
      // Now run the state-NNN migrations on top — must not throw.
      applyStateMigrations(database, ALL);
      // Both tables exist independently.
      const tables = database
        .prepare(
          `SELECT name FROM sqlite_master
            WHERE type = 'table' AND name LIKE '%session%'
            ORDER BY name`,
        )
        .all() as Array<{ name: string }>;
      expect(tables.map((t) => t.name)).toEqual([
        'sessions',
        'trusted_session_singleton',
        'trusted_sessions',
      ]);
    } finally {
      database.close();
    }
  });

  it('trusted_sessions has session_name PK and nullable session_id', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(trusted_sessions)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['session_name']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['session_id']).toMatchObject({
        type: 'TEXT',
        notnull: 0,
      });
      expect(byName['started']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['epoch']).toMatchObject({
        type: 'INTEGER',
        notnull: 1,
      });
      expect(byName['last_seen']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
    } finally {
      database.close();
    }
  });

  it('trusted_sessions accepts session_id=NULL (sqlite-error fallback path)', () => {
    // The owner skill's writer (register-session.py) needs to be
    // able to register a session row before the SDK call that
    // produces session_id has succeeded — readers can then
    // distinguish "registered, session_id pending" from "never
    // registered" by checking IS NULL on session_id.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO trusted_sessions
             (session_name, session_id, started, epoch, last_seen)
           VALUES (?, NULL, ?, ?, ?)`,
        )
        .run(
          'default',
          '2026-04-30T17:00:00.000Z',
          1714498800,
          '2026-04-30T17:00:00.000Z',
        );
      const row = database
        .prepare(
          `SELECT session_id FROM trusted_sessions WHERE session_name = ?`,
        )
        .get('default') as { session_id: string | null };
      expect(row.session_id).toBeNull();
    } finally {
      database.close();
    }
  });

  it('trusted_session_singleton enforces single-row via CHECK(id = 1)', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO trusted_session_singleton
             (id, active_session_id, pending_response, muted_threads)
           VALUES (1, ?, ?, ?)`,
        )
        .run('sess-aaa', null, null);
      // id=2 violates CHECK.
      expect(() =>
        database
          .prepare(
            `INSERT INTO trusted_session_singleton
               (id, active_session_id) VALUES (2, ?)`,
          )
          .run('sess-bbb'),
      ).toThrow(/CHECK constraint failed/);
      // Re-inserting id=1 violates PK.
      expect(() =>
        database
          .prepare(
            `INSERT INTO trusted_session_singleton
               (id, active_session_id) VALUES (1, ?)`,
          )
          .run('sess-ccc'),
      ).toThrow(/UNIQUE constraint failed/);
    } finally {
      database.close();
    }
  });

  it('trusted_session_singleton stores JSON-blob payload columns verbatim', () => {
    // pending_response and muted_threads are TEXT (JSON blob —
    // payload, not schema). Verify the column accepts arbitrary
    // string content and round-trips it without any normalisation.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const pending = '{"id":"req-42","at":"2026-04-30T17:00:00.000Z"}';
      const muted = '["thread-aaa","thread-bbb"]';
      database
        .prepare(
          `INSERT INTO trusted_session_singleton
             (id, pending_response, muted_threads)
           VALUES (1, ?, ?)`,
        )
        .run(pending, muted);
      const row = database
        .prepare(
          `SELECT pending_response, muted_threads
             FROM trusted_session_singleton WHERE id = 1`,
        )
        .get() as { pending_response: string; muted_threads: string };
      expect(row.pending_response).toBe(pending);
      expect(row.muted_threads).toBe(muted);
    } finally {
      database.close();
    }
  });

  it('schema_version on both tables defaults to 1', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      for (const table of ['trusted_sessions', 'trusted_session_singleton']) {
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
