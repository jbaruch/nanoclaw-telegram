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

const ALL = [
  STATE_001_ORDERS,
  STATE_002_EMAIL_FEEDBACK,
  STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
  STATE_004_SCHEDULED_REMINDERS,
  STATE_005_NANOCLAW_STATE_SPLIT,
  STATE_006_TRUSTED_SESSION_STATE,
  STATE_007_MORNING_BRIEF_PENDING,
];

describe('state-007-morning-brief-pending', () => {
  it('creates all three queue tables and bumps user_version to 7', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      expect(Number(database.pragma('user_version', { simple: true }))).toBe(7);
      const tables = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .all() as Array<{ name: string }>;
      const names = tables.map((t) => t.name);
      expect(names).toContain('pending_cleanup_items');
      expect(names).toContain('pending_decisions');
      expect(names).toContain('pending_undated_tasks');
    } finally {
      database.close();
    }
  });

  it('pending_cleanup_items has correct columns + nullability per issue spec', () => {
    // The issue spec calls for TEXT PK `id`, NOT NULL `type`, and
    // nullable `question` / `subject` / `sender` (different
    // cleanup-item types populate different subsets, discriminated
    // by `type`). Lock that down so a future "tighten the schema"
    // refactor can't silently break the email-cleanup vs.
    // question-cleanup variants.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(pending_cleanup_items)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
        dflt_value: string | null;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['id']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['type']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['question']).toMatchObject({ type: 'TEXT', notnull: 0 });
      expect(byName['subject']).toMatchObject({ type: 'TEXT', notnull: 0 });
      expect(byName['sender']).toMatchObject({ type: 'TEXT', notnull: 0 });
      expect(byName['added']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
        dflt_value: 'CURRENT_TIMESTAMP',
      });
    } finally {
      database.close();
    }
  });

  it('pending_decisions enforces NOT NULL on question', () => {
    // The owner skill (`morning-brief`) cannot use a row without a
    // question — DB-level NOT NULL means a malformed insert fails
    // loudly rather than landing a useless row that surfaces empty
    // in the next brief.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(pending_decisions)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['id']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['question']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(() =>
        database
          .prepare(
            `INSERT INTO pending_decisions (id, question) VALUES (?, NULL)`,
          )
          .run('decision-aaa'),
      ).toThrow(/NOT NULL constraint failed/);
    } finally {
      database.close();
    }
  });

  it('pending_undated_tasks enforces NOT NULL on title and tasklist_id', () => {
    // Google Tasks rows missing a due date are appended here by
    // `nightly-housekeeping` Step 18; both `title` and `tasklist_id`
    // are required — without them the row can't be re-attached to
    // its source list when `morning-brief` consumes it.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      const cols = database
        .prepare('PRAGMA table_info(pending_undated_tasks)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        pk: number;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['id']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['title']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['tasklist_id']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      expect(() =>
        database
          .prepare(
            `INSERT INTO pending_undated_tasks (id, title, tasklist_id)
             VALUES (?, NULL, ?)`,
          )
          .run('task-aaa', 'list-default'),
      ).toThrow(/NOT NULL constraint failed/);
      expect(() =>
        database
          .prepare(
            `INSERT INTO pending_undated_tasks (id, title, tasklist_id)
             VALUES (?, ?, NULL)`,
          )
          .run('task-bbb', 'Buy milk'),
      ).toThrow(/NOT NULL constraint failed/);
    } finally {
      database.close();
    }
  });

  it('added defaults to CURRENT_TIMESTAMP on all three tables', () => {
    // The doc-comment header and the issue spec both promise that
    // every queue row gets an `added` timestamp at insert time
    // without the writer having to populate it. Lock the default
    // string ('CURRENT_TIMESTAMP', verbatim) into the test so a
    // future schema edit that drops the default — turning every
    // existing INSERT site into a NOT NULL constraint failure —
    // fails here instead of in a 3am production stack trace.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      for (const table of [
        'pending_cleanup_items',
        'pending_decisions',
        'pending_undated_tasks',
      ]) {
        const cols = database
          .prepare(`PRAGMA table_info(${table})`)
          .all() as Array<{
          name: string;
          dflt_value: string | null;
        }>;
        const addedCol = cols.find((c) => c.name === 'added');
        expect(addedCol).toBeDefined();
        expect(addedCol!.dflt_value).toBe('CURRENT_TIMESTAMP');
      }
    } finally {
      database.close();
    }
  });

  it('schema_version on all three tables defaults to 1', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      for (const table of [
        'pending_cleanup_items',
        'pending_decisions',
        'pending_undated_tasks',
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

  it('DELETE FROM pending_cleanup_items leaves the other two queues intact', () => {
    // The headline reason this migration exists: the JSON-era
    // `brief-cleanup` Step 9 ("set cleanup_items to []") used to
    // rewrite the whole JSON object, which clobbered sibling keys
    // (`pending_decisions`, `undated_tasks`) when a writer raced.
    // The acceptance criterion "sibling-key preservation impossible"
    // becomes a structural property: `DELETE FROM
    // pending_cleanup_items` cannot touch the other tables.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(
          `INSERT INTO pending_cleanup_items (id, type, subject, sender)
           VALUES (?, ?, ?, ?)`,
        )
        .run('cleanup-aaa', 'email', 'Re: lunch?', 'alice@example.com');
      database
        .prepare(
          `INSERT INTO pending_decisions (id, question)
           VALUES (?, ?)`,
        )
        .run('decision-aaa', 'Approve PR #123?');
      database
        .prepare(
          `INSERT INTO pending_undated_tasks (id, title, tasklist_id)
           VALUES (?, ?, ?)`,
        )
        .run('task-aaa', 'Buy milk', 'list-default');

      database.prepare('DELETE FROM pending_cleanup_items').run();

      expect(
        (
          database
            .prepare('SELECT COUNT(*) AS n FROM pending_cleanup_items')
            .get() as { n: number }
        ).n,
      ).toBe(0);
      expect(
        (
          database
            .prepare('SELECT COUNT(*) AS n FROM pending_decisions')
            .get() as { n: number }
        ).n,
      ).toBe(1);
      expect(
        (
          database
            .prepare('SELECT COUNT(*) AS n FROM pending_undated_tasks')
            .get() as { n: number }
        ).n,
      ).toBe(1);
    } finally {
      database.close();
    }
  });

  it('id PK uniqueness is enforced on each of the three tables', () => {
    // Each table's `id` is the natural cleanup contract (DELETE
    // WHERE id = ? for a targeted purge). PK uniqueness means a
    // duplicate-id insert fails loudly rather than silently
    // shadowing the first row.
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, ALL);
      database
        .prepare(`INSERT INTO pending_cleanup_items (id, type) VALUES (?, ?)`)
        .run('cleanup-dup', 'email');
      expect(() =>
        database
          .prepare(`INSERT INTO pending_cleanup_items (id, type) VALUES (?, ?)`)
          .run('cleanup-dup', 'question'),
      ).toThrow(/UNIQUE constraint failed/);

      database
        .prepare(`INSERT INTO pending_decisions (id, question) VALUES (?, ?)`)
        .run('decision-dup', 'q1');
      expect(() =>
        database
          .prepare(`INSERT INTO pending_decisions (id, question) VALUES (?, ?)`)
          .run('decision-dup', 'q2'),
      ).toThrow(/UNIQUE constraint failed/);

      database
        .prepare(
          `INSERT INTO pending_undated_tasks (id, title, tasklist_id)
           VALUES (?, ?, ?)`,
        )
        .run('task-dup', 'Title 1', 'list-default');
      expect(() =>
        database
          .prepare(
            `INSERT INTO pending_undated_tasks (id, title, tasklist_id)
             VALUES (?, ?, ?)`,
          )
          .run('task-dup', 'Title 2', 'list-other'),
      ).toThrow(/UNIQUE constraint failed/);
    } finally {
      database.close();
    }
  });
});
