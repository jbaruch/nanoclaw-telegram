import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';

// Each scenario builds an isolated tempDir + chdir so the
// CWD-rooted `STORE_DIR` and `GROUPS_DIR` constants in `src/config.ts`
// resolve under it. Mirrors the pattern in
// `src/orders-json-migration.test.ts`.

interface MorningBriefPendingJsonShape {
  cleanup_items?: Array<Record<string, unknown>>;
  pending_decisions?: Array<Record<string, unknown>>;
  undated_tasks?: Array<Record<string, unknown>>;
}

function writeMorningBriefFile(
  tempDir: string,
  groupName: string,
  payload: MorningBriefPendingJsonShape,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'morning-brief-pending.json');
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  return filePath;
}

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-mbp-'));
  try {
    process.chdir(tempDir);
    fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

describe('morning-brief-pending.json → SQLite migration (#299)', () => {
  it('imports cleanup_items, pending_decisions, and undated_tasks rows; renames source file', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeMorningBriefFile(tempDir, 'telegram_main', {
        cleanup_items: [
          {
            id: 'cleanup-1',
            type: 'email',
            question: null,
            subject: 'Receipt for your order',
            sender: 'noreply@shop.example.com',
            added: '2026-04-20T10:00:00.000Z',
          },
          {
            id: 'cleanup-2',
            type: 'question',
            question: 'Did you intend to keep this thread?',
            subject: null,
            sender: null,
            added: '2026-04-20T10:05:00.000Z',
          },
        ],
        pending_decisions: [
          {
            id: 'dec-1',
            question: 'Approve the calendar move?',
            added: '2026-04-20T11:00:00.000Z',
          },
        ],
        undated_tasks: [
          {
            id: 'task-1',
            title: 'Schedule dentist',
            tasklist_id: 'tasklist-personal',
            added: '2026-04-19T09:00:00.000Z',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const cleanup = db
            .prepare('SELECT * FROM pending_cleanup_items ORDER BY id')
            .all() as Array<Record<string, unknown>>;
          expect(cleanup).toHaveLength(2);
          expect(cleanup[0]).toMatchObject({
            id: 'cleanup-1',
            type: 'email',
            subject: 'Receipt for your order',
            sender: 'noreply@shop.example.com',
            question: null,
            added: '2026-04-20T10:00:00.000Z',
          });
          expect(cleanup[1]).toMatchObject({
            id: 'cleanup-2',
            type: 'question',
            question: 'Did you intend to keep this thread?',
            subject: null,
            sender: null,
          });

          const decisions = db
            .prepare('SELECT * FROM pending_decisions ORDER BY id')
            .all() as Array<Record<string, unknown>>;
          expect(decisions).toHaveLength(1);
          expect(decisions[0]).toMatchObject({
            id: 'dec-1',
            question: 'Approve the calendar move?',
            added: '2026-04-20T11:00:00.000Z',
          });

          const tasks = db
            .prepare('SELECT * FROM pending_undated_tasks ORDER BY id')
            .all() as Array<Record<string, unknown>>;
          expect(tasks).toHaveLength(1);
          expect(tasks[0]).toMatchObject({
            id: 'task-1',
            title: 'Schedule dentist',
            tasklist_id: 'tasklist-personal',
            added: '2026-04-19T09:00:00.000Z',
          });
        } finally {
          db.close();
        }

        // Source file renamed to .migrated-<YYYY-MM-DD>; the rename is
        // what makes a re-run a no-op (no version-gate guard for data
        // backfill — the schema is already at v7).
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('morning-brief-pending.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('renames the source file and leaves tables empty when every array is empty', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeMorningBriefFile(tempDir, 'telegram_main', {
        cleanup_items: [],
        pending_decisions: [],
        undated_tasks: [],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          for (const table of [
            'pending_cleanup_items',
            'pending_decisions',
            'pending_undated_tasks',
          ]) {
            const count = (
              db.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get() as {
                n: number;
              }
            ).n;
            expect(count).toBe(0);
          }
        } finally {
          db.close();
        }

        // Empty-but-present arrays are still recognised arrays — the
        // file gets renamed even though no rows landed, so the next
        // boot doesn't keep re-reading it.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('morning-brief-pending.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('handles partially-shaped files (only one array key present) without crashing', async () => {
    await runWithTempDir(async (tempDir) => {
      // Only cleanup_items — the other two keys missing entirely.
      // Older morning-brief writers may have produced this shape.
      writeMorningBriefFile(tempDir, 'telegram_main', {
        cleanup_items: [
          {
            id: 'partial-1',
            type: 'email',
            subject: 'Hello',
            sender: 'a@b.com',
            added: '2026-04-20T08:00:00.000Z',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const cleanupCount = (
            db
              .prepare('SELECT COUNT(*) AS n FROM pending_cleanup_items')
              .get() as { n: number }
          ).n;
          expect(cleanupCount).toBe(1);
          const decisionCount = (
            db.prepare('SELECT COUNT(*) AS n FROM pending_decisions').get() as {
              n: number;
            }
          ).n;
          expect(decisionCount).toBe(0);
          const taskCount = (
            db
              .prepare('SELECT COUNT(*) AS n FROM pending_undated_tasks')
              .get() as { n: number }
          ).n;
          expect(taskCount).toBe(0);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('falls back to schema CURRENT_TIMESTAMP default when row omits the `added` field', async () => {
    await runWithTempDir(async (tempDir) => {
      // Capture the wall clock just before initDatabase runs so the
      // default-timestamp assertion has a concrete lower bound.
      const beforeMs = Date.now() - 1_000;
      writeMorningBriefFile(tempDir, 'telegram_main', {
        // No `added` on any row — exercise the schema default for all
        // three tables in one pass.
        cleanup_items: [
          {
            id: 'no-added-cleanup',
            type: 'question',
            question: 'No timestamp here',
          },
        ],
        pending_decisions: [
          {
            id: 'no-added-decision',
            question: 'Pick one',
          },
        ],
        undated_tasks: [
          {
            id: 'no-added-task',
            title: 'Do thing',
            tasklist_id: 'tasklist-x',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const cleanup = db
            .prepare('SELECT id, added FROM pending_cleanup_items')
            .get() as { id: string; added: string };
          expect(cleanup.added).toBeTruthy();
          // CURRENT_TIMESTAMP renders as 'YYYY-MM-DD HH:MM:SS' (UTC).
          // Verify it parses to a real date roughly "now".
          const parsedMs = Date.parse(cleanup.added.replace(' ', 'T') + 'Z');
          expect(parsedMs).not.toBeNaN();
          expect(parsedMs).toBeGreaterThanOrEqual(beforeMs);

          const decision = db
            .prepare('SELECT id, added FROM pending_decisions')
            .get() as { id: string; added: string };
          expect(decision.added).toBeTruthy();

          const task = db
            .prepare('SELECT id, added FROM pending_undated_tasks')
            .get() as { id: string; added: string };
          expect(task.added).toBeTruthy();
        } finally {
          db.close();
        }
      } finally {
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
        'morning-brief-pending.json.migrated-2026-04-20',
      );
      fs.writeFileSync(
        renamedPath,
        JSON.stringify({
          cleanup_items: [
            {
              id: 'already-imported',
              type: 'email',
              subject: 'old',
              sender: 's@e',
              added: '2026-04-20T08:00:00.000Z',
            },
          ],
        }),
      );

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      // Should not throw — the migration just doesn't see a source
      // file to consume, so all three tables stay empty.
      expect(() => initDatabase()).not.toThrow();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          for (const table of [
            'pending_cleanup_items',
            'pending_decisions',
            'pending_undated_tasks',
          ]) {
            const count = (
              db.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get() as {
                n: number;
              }
            ).n;
            expect(count).toBe(0);
          }
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

  it('rolls back and leaves the source file in place when a row violates a NOT NULL constraint', async () => {
    // The migration uses `ON CONFLICT(id) DO NOTHING` (not `INSERT OR
    // IGNORE`) precisely so the import doesn't silently swallow
    // malformed rows like a `pending_decisions` entry missing
    // `question`. The contract: NOT NULL throws inside the per-file
    // transaction, the transaction rolls back so no partial rows
    // land, the catch turns the throw into a warn, and the source
    // file stays put for human triage. Lock that down here.
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeMorningBriefFile(tempDir, 'good', {
        cleanup_items: [
          {
            id: 'good-cleanup',
            type: 'email',
            subject: 's',
            sender: 'g@e',
            added: '2026-04-20T08:00:00.000Z',
          },
        ],
      });
      // Bypass the schema's TS type to write a contract-violating
      // fixture (decision row with no `question`). That's the whole
      // point of this test.
      const badFile = writeMorningBriefFile(tempDir, 'bad', {
        pending_decisions: [{ id: 'bad-decision' }],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // Bad file's transaction rolled back: zero rows for the
          // bad row.
          const decisionCount = (
            db
              .prepare(
                `SELECT COUNT(*) AS n FROM pending_decisions WHERE id = ?`,
              )
              .get('bad-decision') as { n: number }
          ).n;
          expect(decisionCount).toBe(0);
          // Good file still imported.
          const cleanupCount = (
            db
              .prepare('SELECT COUNT(*) AS n FROM pending_cleanup_items')
              .get() as {
              n: number;
            }
          ).n;
          expect(cleanupCount).toBe(1);
        } finally {
          db.close();
        }
        // Good file renamed; bad file left in place.
        expect(fs.existsSync(goodFile)).toBe(false);
        expect(fs.existsSync(badFile)).toBe(true);
        // Per-file warn fired with the constraint-violation message.
        // The catch is narrowed to constraint-class SqliteError per
        // `coding-policy: error-handling` (no bare catch-all).
        const constraintWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('violated a DB constraint'));
        });
        expect(constraintWarnFired).toBe(true);
        // The warn metadata carries the SQLite constraint code (e.g.
        // SQLITE_CONSTRAINT_NOTNULL) so an operator can grep for the
        // exact failure class without spelunking the full error
        // object.
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

  it('skips a non-object JSON payload (null / number / array) with a warning and leaves it in place', async () => {
    // `JSON.parse` returns null / numbers / strings / arrays for
    // syntactically valid but non-object payloads. The migration
    // must guard against those before binding to property accesses,
    // otherwise one bad file (a stale empty `null` written during
    // some prior incident) would throw "cannot read properties of
    // null" and abort the whole pass per `coding-policy: error-
    // handling` ("try alternatives before failing"). Lock that
    // contract down for each non-object shape that JSON.parse can
    // produce.
    for (const payload of ['null', '42', '"hello"', '[1, 2, 3]']) {
      await runWithTempDir(async (tempDir) => {
        const goodFile = writeMorningBriefFile(tempDir, 'good', {
          cleanup_items: [
            {
              id: `good-${payload.replace(/\W/g, '')}`,
              type: 'email',
              subject: 's',
              sender: 'g@e',
              added: '2026-04-20T08:00:00.000Z',
            },
          ],
        });
        const badFolder = path.join(tempDir, 'groups', 'bad');
        fs.mkdirSync(badFolder, { recursive: true });
        const badFile = path.join(badFolder, 'morning-brief-pending.json');
        fs.writeFileSync(badFile, payload);

        vi.resetModules();
        const { initDatabase, _closeDatabase } = await import('./db.js');
        const { logger } = await import('./logger.js');
        const warnSpy = vi.spyOn(logger, 'warn');
        try {
          // Must not throw — the bad file is warned-and-skipped, the
          // good file still imports.
          expect(() => initDatabase()).not.toThrow();
          const db = new Database(path.join(tempDir, 'store', 'messages.db'));
          try {
            const ids = db
              .prepare('SELECT id FROM pending_cleanup_items')
              .all() as Array<{ id: string }>;
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
      // Mix one valid file and one malformed file. Malformed must NOT
      // abort the pass — other groups still need to migrate. Mirrors
      // the orders test for the same case.
      const goodFile = writeMorningBriefFile(tempDir, 'good', {
        cleanup_items: [
          {
            id: 'good-cleanup',
            type: 'email',
            subject: 'good',
            sender: 'g@e',
            added: '2026-04-20T08:00:00.000Z',
          },
        ],
      });
      const badFolder = path.join(tempDir, 'groups', 'bad');
      fs.mkdirSync(badFolder, { recursive: true });
      const badFile = path.join(badFolder, 'morning-brief-pending.json');
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
            .prepare('SELECT id FROM pending_cleanup_items')
            .all() as Array<{ id: string }>;
          expect(ids.map((r) => r.id)).toEqual(['good-cleanup']);
        } finally {
          db.close();
        }
        // Good file renamed; bad file left in place for human triage.
        expect(fs.existsSync(goodFile)).toBe(false);
        expect(fs.existsSync(badFile)).toBe(true);
        // A warn was logged for the malformed file. Match on message
        // substring rather than exact shape so the test doesn't break
        // on logger formatting tweaks.
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
});
