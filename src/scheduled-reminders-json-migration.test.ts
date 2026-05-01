import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';

// Mirrors the per-group migration test pattern from
// `src/morning-brief-pending-json-migration.test.ts`: each scenario
// builds an isolated tempDir and chdirs into it so the CWD-rooted
// `STORE_DIR` and `GROUPS_DIR` constants in `src/config.ts` resolve
// under it.

interface ScheduledReminderRow {
  event_id: string;
  title: string;
  utc_time: string;
  reminder_offset_min: number;
  task_id: string;
}

function writeRemindersWrapped(
  tempDir: string,
  groupName: string,
  reminders: Array<Record<string, unknown>>,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'scheduled-reminders.json');
  fs.writeFileSync(filePath, JSON.stringify({ reminders }, null, 2));
  return filePath;
}

function writeRemindersBareArray(
  tempDir: string,
  groupName: string,
  reminders: Array<Record<string, unknown>>,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'scheduled-reminders.json');
  fs.writeFileSync(filePath, JSON.stringify(reminders, null, 2));
  return filePath;
}

function writeRemindersRaw(
  tempDir: string,
  groupName: string,
  raw: string,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'scheduled-reminders.json');
  fs.writeFileSync(filePath, raw);
  return filePath;
}

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-sr-'));
  try {
    process.chdir(tempDir);
    fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

describe('scheduled-reminders.json → SQLite migration (#296)', () => {
  it('imports wrapped-shape reminders and renames the source file', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeRemindersWrapped(tempDir, 'telegram_main', [
        {
          event_id: 'evt-001',
          title: 'Standup',
          utc_time: '2026-04-30T13:00:00Z',
          reminder_offset_min: 15,
          task_id: 'task-aaa',
        },
        {
          event_id: 'evt-002',
          title: 'Lunch',
          utc_time: '2026-04-30T16:30:00Z',
          reminder_offset_min: 5,
          task_id: 'task-bbb',
        },
      ]);

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const rows = db
            .prepare(
              'SELECT event_id, title, utc_time, reminder_offset_min, task_id FROM scheduled_reminders ORDER BY event_id',
            )
            .all() as ScheduledReminderRow[];
          expect(rows).toHaveLength(2);
          expect(rows[0]).toEqual({
            event_id: 'evt-001',
            title: 'Standup',
            utc_time: '2026-04-30T13:00:00Z',
            reminder_offset_min: 15,
            task_id: 'task-aaa',
          });
          expect(rows[1]).toEqual({
            event_id: 'evt-002',
            title: 'Lunch',
            utc_time: '2026-04-30T16:30:00Z',
            reminder_offset_min: 5,
            task_id: 'task-bbb',
          });
        } finally {
          db.close();
        }

        // Source renamed to .migrated-<YYYY-MM-DD>; the rename is what
        // makes a re-run a no-op.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('scheduled-reminders.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('imports bare-array legacy shape', async () => {
    // The earlier `append-scheduled-reminders.py` revisions wrote a
    // top-level array instead of `{ reminders: [...] }`. The migration
    // accepts both so an operator who never upgraded the writer doesn't
    // get a stuck file at startup.
    await runWithTempDir(async (tempDir) => {
      const filePath = writeRemindersBareArray(tempDir, 'telegram_main', [
        {
          event_id: 'evt-bare-1',
          title: 'Legacy reminder',
          utc_time: '2026-04-30T18:00:00Z',
          reminder_offset_min: 30,
          task_id: 'task-legacy',
        },
      ]);

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const row = db
            .prepare(
              'SELECT event_id, title FROM scheduled_reminders WHERE event_id = ?',
            )
            .get('evt-bare-1') as { event_id: string; title: string };
          expect(row).toEqual({
            event_id: 'evt-bare-1',
            title: 'Legacy reminder',
          });
        } finally {
          db.close();
        }
        expect(fs.existsSync(filePath)).toBe(false);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('renames the source file and leaves the table empty when reminders array is empty', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeRemindersWrapped(tempDir, 'telegram_main', []);

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const count = (
            db
              .prepare('SELECT COUNT(*) AS n FROM scheduled_reminders')
              .get() as { n: number }
          ).n;
          expect(count).toBe(0);
        } finally {
          db.close();
        }
        // Empty array is still a recognised array — file gets renamed
        // so the next boot doesn't keep re-reading it.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('scheduled-reminders.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('is a no-op for duplicate event_id pre-seeded into the table (ON CONFLICT(event_id) DO NOTHING)', async () => {
    // The migration uses `ON CONFLICT(event_id) DO NOTHING` so a
    // re-import where the operator copied a partial DB back over an
    // already-imported one silently absorbs the PK conflict. Pre-seed
    // a row with a different `title` than the source JSON, run the
    // migration, then verify the pre-seeded row is preserved (not
    // overwritten) and the import did NOT throw.
    await runWithTempDir(async (tempDir) => {
      writeRemindersWrapped(tempDir, 'telegram_main', [
        {
          event_id: 'evt-dup',
          title: 'Source JSON title',
          utc_time: '2026-04-30T13:00:00Z',
          reminder_offset_min: 15,
          task_id: 'task-source',
        },
      ]);

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        // Pre-seed a row with the same event_id but a different title
        // and task_id BEFORE the migration runs again. To do that we
        // need to inject the row before the source is consumed — but
        // the migration ran on initDatabase() above, so the source has
        // already been imported and renamed. Pre-seed AFTER the first
        // import: write a NEW source JSON with the same event_id and
        // re-init.
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          db.prepare(
            'UPDATE scheduled_reminders SET title = ?, task_id = ? WHERE event_id = ?',
          ).run('Pre-seeded title', 'task-preseeded', 'evt-dup');
        } finally {
          db.close();
        }
        _closeDatabase();

        // Write a second source file with the same event_id but a
        // different title. The migration must not overwrite the
        // pre-seeded row.
        writeRemindersWrapped(tempDir, 'telegram_main', [
          {
            event_id: 'evt-dup',
            title: 'Second source title',
            utc_time: '2026-04-30T13:00:00Z',
            reminder_offset_min: 15,
            task_id: 'task-second',
          },
        ]);

        vi.resetModules();
        const second = await import('./db.js');
        expect(() => second.initDatabase()).not.toThrow();
        try {
          const db2 = new Database(path.join(tempDir, 'store', 'messages.db'));
          try {
            const row = db2
              .prepare(
                'SELECT title, task_id FROM scheduled_reminders WHERE event_id = ?',
              )
              .get('evt-dup') as { title: string; task_id: string };
            // Pre-seeded values preserved; ON CONFLICT(event_id) DO
            // NOTHING absorbed the duplicate.
            expect(row).toEqual({
              title: 'Pre-seeded title',
              task_id: 'task-preseeded',
            });
          } finally {
            db2.close();
          }
        } finally {
          second._closeDatabase();
        }
      } finally {
        // Best-effort cleanup if an error path skipped one of the
        // closes above. Calling on an already-closed handle throws,
        // which is fine for this scenario's failure surface.
      }
    });
  });

  it('rolls back and leaves the source file in place when a row violates a NOT NULL constraint', async () => {
    // ON CONFLICT(event_id) DO NOTHING (NOT `INSERT OR IGNORE`)
    // intentionally lets NOT NULL violations throw inside the per-file
    // transaction. The transaction rolls back so no partial rows land,
    // the catch helper turns the throw into a warn, and the source
    // file stays in place for human triage.
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeRemindersWrapped(tempDir, 'good', [
        {
          event_id: 'evt-good',
          title: 'Standup',
          utc_time: '2026-04-30T13:00:00Z',
          reminder_offset_min: 15,
          task_id: 'task-good',
        },
      ]);
      // Bypass the schema's TS type to write a contract-violating
      // fixture: reminder row missing `task_id`. NOT NULL on `task_id`
      // throws inside the transaction.
      const badFile = writeRemindersWrapped(tempDir, 'bad', [
        {
          event_id: 'evt-bad',
          title: 'Broken',
          utc_time: '2026-04-30T13:00:00Z',
          reminder_offset_min: 15,
          // task_id intentionally missing
        },
      ]);

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // Bad file's transaction rolled back: zero rows for the bad
          // event_id.
          const badCount = (
            db
              .prepare(
                'SELECT COUNT(*) AS n FROM scheduled_reminders WHERE event_id = ?',
              )
              .get('evt-bad') as { n: number }
          ).n;
          expect(badCount).toBe(0);
          // Good file still imported.
          const goodCount = (
            db
              .prepare(
                'SELECT COUNT(*) AS n FROM scheduled_reminders WHERE event_id = ?',
              )
              .get('evt-good') as { n: number }
          ).n;
          expect(goodCount).toBe(1);
        } finally {
          db.close();
        }
        // Good file renamed; bad file left in place.
        expect(fs.existsSync(goodFile)).toBe(false);
        expect(fs.existsSync(badFile)).toBe(true);
        // Per-file warn fired with the constraint-violation message
        // (catch is narrowed to constraint-class SqliteError per
        // `coding-policy: error-handling`).
        const constraintWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('violated a DB constraint'));
        });
        expect(constraintWarnFired).toBe(true);
        const errCodeIsConstraint = warnSpy.mock.calls.some((call) => {
          const meta = call.find(
            (arg) =>
              typeof arg === 'object' &&
              arg !== null &&
              'errCode' in (arg as object),
          ) as { errCode?: string } | undefined;
          return Boolean(
            meta?.errCode && meta.errCode.startsWith('SQLITE_CONSTRAINT_'),
          );
        });
        expect(errCodeIsConstraint).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('skips a non-object reminder element with a warning', async () => {
    // Per-row object guard: a stale null/string/number element in the
    // reminders array would otherwise throw a TypeError inside the
    // transaction (`null.event_id` blows up before any INSERT runs),
    // and the narrowed catch would propagate that as "unexpected" and
    // halt orchestrator startup. Skip non-object elements with a warn
    // instead.
    await runWithTempDir(async (tempDir) => {
      writeRemindersWrapped(tempDir, 'telegram_main', [
        // Cast through unknown so TS lets us fixture a non-object
        // element. The whole point is to verify runtime guards.
        null as unknown as Record<string, unknown>,
        {
          event_id: 'evt-real',
          title: 'Real reminder',
          utc_time: '2026-04-30T13:00:00Z',
          reminder_offset_min: 15,
          task_id: 'task-real',
        },
      ]);

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        expect(() => initDatabase()).not.toThrow();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const ids = db
            .prepare('SELECT event_id FROM scheduled_reminders')
            .all() as Array<{ event_id: string }>;
          expect(ids.map((r) => r.event_id)).toEqual(['evt-real']);
        } finally {
          db.close();
        }
        const nonObjectWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('skipping non-object row'));
        });
        expect(nonObjectWarnFired).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('is idempotent: re-running after a successful import is a no-op', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeRemindersWrapped(tempDir, 'telegram_main', [
        {
          event_id: 'evt-once',
          title: 'Standup',
          utc_time: '2026-04-30T13:00:00Z',
          reminder_offset_min: 15,
          task_id: 'task-once',
        },
      ]);

      vi.resetModules();
      const first = await import('./db.js');
      first.initDatabase();
      first._closeDatabase();
      expect(fs.existsSync(filePath)).toBe(false);

      // Second initDatabase call sees no source file (it was renamed)
      // and is a clean no-op.
      vi.resetModules();
      const second = await import('./db.js');
      expect(() => second.initDatabase()).not.toThrow();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // Single row from the first run, no duplicates.
          const count = (
            db
              .prepare('SELECT COUNT(*) AS n FROM scheduled_reminders')
              .get() as { n: number }
          ).n;
          expect(count).toBe(1);
        } finally {
          db.close();
        }
      } finally {
        second._closeDatabase();
      }
    });
  });

  it('skips a malformed JSON file with a warning and leaves it in place', async () => {
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeRemindersWrapped(tempDir, 'good', [
        {
          event_id: 'evt-good',
          title: 'Good',
          utc_time: '2026-04-30T13:00:00Z',
          reminder_offset_min: 15,
          task_id: 'task-good',
        },
      ]);
      const badFile = writeRemindersRaw(
        tempDir,
        'bad',
        '{ not valid json at all',
      );

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const ids = db
            .prepare('SELECT event_id FROM scheduled_reminders')
            .all() as Array<{ event_id: string }>;
          expect(ids.map((r) => r.event_id)).toEqual(['evt-good']);
        } finally {
          db.close();
        }
        // Good file renamed; bad file left in place for triage.
        expect(fs.existsSync(goodFile)).toBe(false);
        expect(fs.existsSync(badFile)).toBe(true);
        const malformedWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('invalid JSON'));
        });
        expect(malformedWarnFired).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('skips a non-object JSON payload (null / number / string) with a warning', async () => {
    // `JSON.parse` returns null / numbers / strings for syntactically
    // valid but non-object/non-array payloads. The migration must
    // guard against those before binding to property accesses,
    // otherwise one bad file (a stale `null` written during some prior
    // incident) would throw and abort the whole pass.
    for (const payload of ['null', '42', '"hello"']) {
      await runWithTempDir(async (tempDir) => {
        const goodFile = writeRemindersWrapped(tempDir, 'good', [
          {
            event_id: `evt-${payload.replace(/\W/g, '')}`,
            title: 'Good',
            utc_time: '2026-04-30T13:00:00Z',
            reminder_offset_min: 15,
            task_id: 'task-good',
          },
        ]);
        const badFile = writeRemindersRaw(tempDir, 'bad', payload);

        vi.resetModules();
        const { initDatabase, _closeDatabase } = await import('./db.js');
        const { logger } = await import('./logger.js');
        const warnSpy = vi.spyOn(logger, 'warn');
        try {
          expect(() => initDatabase()).not.toThrow();
          const db = new Database(path.join(tempDir, 'store', 'messages.db'));
          try {
            const ids = db
              .prepare('SELECT event_id FROM scheduled_reminders')
              .all() as Array<{ event_id: string }>;
            expect(ids).toHaveLength(1);
          } finally {
            db.close();
          }
          expect(fs.existsSync(goodFile)).toBe(false);
          expect(fs.existsSync(badFile)).toBe(true);
          const nonObjectWarnFired = warnSpy.mock.calls.some((call) => {
            const msg = call.find((arg) => typeof arg === 'string') as
              | string
              | undefined;
            return Boolean(msg && msg.includes('payload is not an object'));
          });
          expect(nonObjectWarnFired).toBe(true);
        } finally {
          warnSpy.mockRestore();
          _closeDatabase();
        }
      });
    }
  });
});
