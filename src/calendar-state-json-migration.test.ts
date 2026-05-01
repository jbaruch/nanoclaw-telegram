import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';

// Each scenario builds an isolated tempDir + chdir so the
// CWD-rooted `STORE_DIR` and `GROUPS_DIR` constants in `src/config.ts`
// resolve under it. Mirrors the pattern already used in
// `src/morning-brief-pending-json-migration.test.ts` and
// `src/orders-json-migration.test.ts`.

interface CalendarStateJsonShape {
  date?: unknown;
  fetched_at?: unknown;
  events?: unknown;
}

function writeCalendarFile(
  tempDir: string,
  groupName: string,
  payload: CalendarStateJsonShape | unknown,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'calendar-state.json');
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  return filePath;
}

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-cal-'));
  try {
    process.chdir(tempDir);
    fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

describe('calendar-state.json → SQLite migration (#300)', () => {
  it('imports the snapshot row and every event row, then renames the source file', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeCalendarFile(tempDir, 'telegram_main', {
        date: '2026-04-29',
        fetched_at: '2026-04-29T08:00:00.000Z',
        events: [
          {
            event_id: 'evt-1',
            title: 'Standup',
            start: '2026-04-29T09:00:00.000Z',
            end: '2026-04-29T09:15:00.000Z',
            reminder_task_id: 'task-1',
          },
          {
            event_id: 'evt-2',
            title: 'All-day offsite',
            start: '2026-04-29',
            // end omitted — Google all-day events legitimately have no end.
            // reminder_task_id omitted — paired Task not yet created.
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const snapshots = db
            .prepare('SELECT * FROM calendar_snapshots ORDER BY date')
            .all() as Array<Record<string, unknown>>;
          expect(snapshots).toHaveLength(1);
          expect(snapshots[0]).toMatchObject({
            date: '2026-04-29',
            fetched_at: '2026-04-29T08:00:00.000Z',
          });

          const events = db
            .prepare('SELECT * FROM calendar_events ORDER BY event_id')
            .all() as Array<Record<string, unknown>>;
          expect(events).toHaveLength(2);
          expect(events[0]).toMatchObject({
            event_id: 'evt-1',
            date: '2026-04-29',
            title: 'Standup',
            start: '2026-04-29T09:00:00.000Z',
            end: '2026-04-29T09:15:00.000Z',
            reminder_task_id: 'task-1',
          });
          expect(events[1]).toMatchObject({
            event_id: 'evt-2',
            date: '2026-04-29',
            title: 'All-day offsite',
            start: '2026-04-29',
            end: null,
            reminder_task_id: null,
          });
        } finally {
          db.close();
        }

        // Source renamed to .migrated-<YYYY-MM-DD>; renaming is what
        // makes a re-run a no-op (no version-gate guard for data
        // backfill — the schema is already at v8).
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('calendar-state.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('imports the snapshot row and renames the source file even when the events array is empty', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeCalendarFile(tempDir, 'telegram_main', {
        date: '2026-04-30',
        fetched_at: '2026-04-30T08:00:00.000Z',
        events: [],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const snapshotCount = (
            db
              .prepare('SELECT COUNT(*) AS n FROM calendar_snapshots')
              .get() as {
              n: number;
            }
          ).n;
          expect(snapshotCount).toBe(1);
          const eventCount = (
            db.prepare('SELECT COUNT(*) AS n FROM calendar_events').get() as {
              n: number;
            }
          ).n;
          expect(eventCount).toBe(0);
        } finally {
          db.close();
        }
        // File got renamed even though no events landed — empty list
        // is a valid "fetched, no events that day" snapshot, not a
        // failure.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('calendar-state.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('skips the file with a warning and leaves it in place when date or fetched_at is missing', async () => {
    // Both `date` and `fetched_at` are NOT NULL on calendar_snapshots
    // (and `date` is the FK target for calendar_events). If either is
    // missing/non-string we cannot write a snapshot row at all, and
    // events would dangle — warn-and-skip per `coding-policy: error-
    // handling`.
    await runWithTempDir(async (tempDir) => {
      const noDate = writeCalendarFile(tempDir, 'no_date', {
        fetched_at: '2026-04-29T08:00:00.000Z',
        events: [],
      });
      const noFetchedAt = writeCalendarFile(tempDir, 'no_fetched_at', {
        date: '2026-04-29',
        events: [],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        expect(() => initDatabase()).not.toThrow();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const snapshotCount = (
            db
              .prepare('SELECT COUNT(*) AS n FROM calendar_snapshots')
              .get() as {
              n: number;
            }
          ).n;
          expect(snapshotCount).toBe(0);
        } finally {
          db.close();
        }
        // Both files stay in place for triage.
        expect(fs.existsSync(noDate)).toBe(true);
        expect(fs.existsSync(noFetchedAt)).toBe(true);
        const missingFieldsWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(
            msg && msg.includes('missing or non-string date / fetched_at'),
          );
        });
        expect(missingFieldsWarnFired).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('skips a non-object event element with a warning and still imports the sibling rows', async () => {
    await runWithTempDir(async (tempDir) => {
      writeCalendarFile(tempDir, 'telegram_main', {
        date: '2026-04-29',
        fetched_at: '2026-04-29T08:00:00.000Z',
        events: [
          {
            event_id: 'evt-good-1',
            title: 'Standup',
            start: '2026-04-29T09:00:00.000Z',
          },
          // Stale `null` element — the per-row object guard must skip
          // it without TypeError-ing inside the transaction. Lock down
          // the contract from the helper integration.
          null,
          {
            event_id: 'evt-good-2',
            title: 'Lunch',
            start: '2026-04-29T12:00:00.000Z',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        expect(() => initDatabase()).not.toThrow();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const events = db
            .prepare('SELECT event_id FROM calendar_events ORDER BY event_id')
            .all() as Array<{ event_id: string }>;
          // Both well-shaped sibling rows landed; the null was
          // skipped, not propagated.
          expect(events.map((r) => r.event_id)).toEqual([
            'evt-good-1',
            'evt-good-2',
          ]);
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

  it('rolls back and leaves the source file in place when a row violates a NOT NULL constraint', async () => {
    // The migration uses `ON CONFLICT(event_id) DO NOTHING` (not
    // `INSERT OR IGNORE`) precisely so the import doesn't silently
    // swallow malformed rows like an event missing `title`. The
    // contract: NOT NULL throws inside the per-file transaction, the
    // transaction rolls back so no partial rows land (snapshot row
    // included), the catch turns the throw into a warn, and the
    // source file stays put for human triage. Sibling good file still
    // imports.
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeCalendarFile(tempDir, 'good', {
        date: '2026-04-29',
        fetched_at: '2026-04-29T08:00:00.000Z',
        events: [
          {
            event_id: 'good-evt',
            title: 'Standup',
            start: '2026-04-29T09:00:00.000Z',
          },
        ],
      });
      const badFile = writeCalendarFile(tempDir, 'bad', {
        date: '2026-04-30',
        fetched_at: '2026-04-30T08:00:00.000Z',
        // `title` is NOT NULL on calendar_events — omit it on this row
        // to provoke SQLITE_CONSTRAINT_NOTNULL.
        events: [
          {
            event_id: 'bad-evt',
            // title intentionally missing
            start: '2026-04-30T09:00:00.000Z',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // Bad file's transaction rolled back: zero events AND zero
          // snapshot for the bad date. Both are inside the same
          // transaction, so the snapshot row must NOT land if the
          // event row throws.
          const badSnap = (
            db
              .prepare(
                `SELECT COUNT(*) AS n FROM calendar_snapshots WHERE date = ?`,
              )
              .get('2026-04-30') as { n: number }
          ).n;
          expect(badSnap).toBe(0);
          const badEvt = (
            db
              .prepare(
                `SELECT COUNT(*) AS n FROM calendar_events WHERE event_id = ?`,
              )
              .get('bad-evt') as { n: number }
          ).n;
          expect(badEvt).toBe(0);
          // Good file still imported.
          const goodSnap = (
            db
              .prepare(
                `SELECT COUNT(*) AS n FROM calendar_snapshots WHERE date = ?`,
              )
              .get('2026-04-29') as { n: number }
          ).n;
          expect(goodSnap).toBe(1);
          const goodEvt = (
            db
              .prepare(
                `SELECT COUNT(*) AS n FROM calendar_events WHERE event_id = ?`,
              )
              .get('good-evt') as { n: number }
          ).n;
          expect(goodEvt).toBe(1);
        } finally {
          db.close();
        }
        // Good file renamed; bad file left in place.
        expect(fs.existsSync(goodFile)).toBe(false);
        expect(fs.existsSync(badFile)).toBe(true);
        // Per-file warn fired with the constraint-violation message.
        const constraintWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('violated a DB constraint'));
        });
        expect(constraintWarnFired).toBe(true);
        // The warn metadata carries the SQLite constraint code so an
        // operator can grep for the exact failure class without
        // spelunking the full error object.
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

  it('skips a non-object JSON payload (null / number / string / array) with a warning and leaves it in place', async () => {
    // `JSON.parse` returns null / numbers / strings / arrays for
    // syntactically valid but non-object payloads. The shared helper
    // (`parseJsonObjectOrWarn`) guards against those before binding to
    // property accesses; this test verifies the integration: bad file
    // is warned-and-skipped, good file still imports, neither aborts
    // the pass.
    for (const payload of ['null', '42', '"x"', '[1, 2, 3]']) {
      await runWithTempDir(async (tempDir) => {
        const goodFile = writeCalendarFile(tempDir, 'good', {
          date: '2026-04-29',
          fetched_at: '2026-04-29T08:00:00.000Z',
          events: [
            {
              event_id: `good-${payload.replace(/\W/g, '')}`,
              title: 'Standup',
              start: '2026-04-29T09:00:00.000Z',
            },
          ],
        });
        const badFolder = path.join(tempDir, 'groups', 'bad');
        fs.mkdirSync(badFolder, { recursive: true });
        const badFile = path.join(badFolder, 'calendar-state.json');
        fs.writeFileSync(badFile, payload);

        vi.resetModules();
        const { initDatabase, _closeDatabase } = await import('./db.js');
        const { logger } = await import('./logger.js');
        const warnSpy = vi.spyOn(logger, 'warn');
        try {
          expect(() => initDatabase()).not.toThrow();
          const db = new Database(path.join(tempDir, 'store', 'messages.db'));
          try {
            const ids = db
              .prepare('SELECT event_id FROM calendar_events')
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

  it('skips a malformed JSON file with a warning and leaves it in place', async () => {
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeCalendarFile(tempDir, 'good', {
        date: '2026-04-29',
        fetched_at: '2026-04-29T08:00:00.000Z',
        events: [
          {
            event_id: 'good-evt',
            title: 'Standup',
            start: '2026-04-29T09:00:00.000Z',
          },
        ],
      });
      const badFolder = path.join(tempDir, 'groups', 'bad');
      fs.mkdirSync(badFolder, { recursive: true });
      const badFile = path.join(badFolder, 'calendar-state.json');
      fs.writeFileSync(badFile, '{ this is not valid json');

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const ids = db
            .prepare('SELECT event_id FROM calendar_events')
            .all() as Array<{ event_id: string }>;
          expect(ids.map((r) => r.event_id)).toEqual(['good-evt']);
        } finally {
          db.close();
        }
        // Good file renamed; bad file left in place for human triage.
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

  it('is a no-op when the source file has already been renamed (re-run)', async () => {
    await runWithTempDir(async (tempDir) => {
      // Simulate a successful prior run: only the renamed file exists.
      const folder = path.join(tempDir, 'groups', 'telegram_main');
      fs.mkdirSync(folder, { recursive: true });
      const renamedPath = path.join(
        folder,
        'calendar-state.json.migrated-2026-04-29',
      );
      fs.writeFileSync(
        renamedPath,
        JSON.stringify({
          date: '2026-04-29',
          fetched_at: '2026-04-29T08:00:00.000Z',
          events: [
            {
              event_id: 'already-imported',
              title: 'Standup',
              start: '2026-04-29T09:00:00.000Z',
            },
          ],
        }),
      );

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      // Should not throw — the migration just doesn't see a source
      // file to consume, so calendar_snapshots / calendar_events stay
      // empty.
      expect(() => initDatabase()).not.toThrow();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const snapCount = (
            db
              .prepare('SELECT COUNT(*) AS n FROM calendar_snapshots')
              .get() as {
              n: number;
            }
          ).n;
          expect(snapCount).toBe(0);
          const eventCount = (
            db.prepare('SELECT COUNT(*) AS n FROM calendar_events').get() as {
              n: number;
            }
          ).n;
          expect(eventCount).toBe(0);
        } finally {
          db.close();
        }
        // Renamed file still in place, untouched.
        expect(fs.existsSync(renamedPath)).toBe(true);
      } finally {
        _closeDatabase();
      }
    });
  });
});
