import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';

// Each scenario builds an isolated tempDir + chdir so the
// CWD-rooted `STORE_DIR` and `GROUPS_DIR` constants in `src/config.ts`
// resolve under it. Mirrors the pattern in
// `src/morning-brief-pending-json-migration.test.ts`.

interface TaskTzStateFollowMeTaskJson {
  name?: unknown;
  local_time?: unknown;
  schedule_value?: unknown;
  last_run_date?: unknown;
  pending_run_at?: unknown;
}

interface TaskTzStateJsonShape {
  current_tz?: unknown;
  home_tz?: unknown;
  scheduler_tz?: unknown;
  follow_me_tasks?: unknown;
}

function writeTaskTzStateFile(
  tempDir: string,
  groupName: string,
  payload: TaskTzStateJsonShape | string,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'task-tz-state.json');
  if (typeof payload === 'string') {
    fs.writeFileSync(filePath, payload);
  } else {
    fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  }
  return filePath;
}

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-tztz-'));
  try {
    process.chdir(tempDir);
    fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

describe('task-tz-state.json → SQLite migration (#302)', () => {
  it('imports the tz_state singleton + N follow_me_tasks rows; renames source', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeTaskTzStateFile(tempDir, 'telegram_main', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        scheduler_tz: 'America/Chicago',
        follow_me_tasks: [
          {
            name: 'morning-brief',
            local_time: '08:00',
            schedule_value: '0 13 * * *',
            last_run_date: '2026-04-29',
            pending_run_at: null,
          },
          {
            name: 'nightly-housekeeping',
            local_time: '22:00',
            schedule_value: '0 3 * * *',
            last_run_date: '2026-04-29',
            pending_run_at: '2026-04-30T03:00:00Z',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const tzRows = db
            .prepare(
              'SELECT id, current_tz, home_tz, scheduler_tz FROM tz_state',
            )
            .all() as Array<Record<string, unknown>>;
          expect(tzRows).toEqual([
            {
              id: 1,
              current_tz: 'America/Chicago',
              home_tz: 'America/Chicago',
              scheduler_tz: 'America/Chicago',
            },
          ]);

          const tasks = db
            .prepare(
              'SELECT name, local_time, schedule_value, last_run_date, pending_run_at FROM follow_me_tasks ORDER BY name',
            )
            .all() as Array<Record<string, unknown>>;
          expect(tasks).toHaveLength(2);
          expect(tasks[0]).toMatchObject({
            name: 'morning-brief',
            local_time: '08:00',
            schedule_value: '0 13 * * *',
            last_run_date: '2026-04-29',
            pending_run_at: null,
          });
          expect(tasks[1]).toMatchObject({
            name: 'nightly-housekeeping',
            local_time: '22:00',
            schedule_value: '0 3 * * *',
            last_run_date: '2026-04-29',
            pending_run_at: '2026-04-30T03:00:00Z',
          });
        } finally {
          db.close();
        }

        // Source file renamed — the rename is what makes a re-run a
        // no-op (the schema migration is independent of the data
        // import).
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('task-tz-state.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('round-trips NULL last_run_date and NULL pending_run_at (explicit-null and key-absent shapes)', async () => {
    // Both columns are nullable on the schema; readers (morning-brief
    // / nightly / weekly) tolerate NULL as "task hasn't run yet" or
    // "no pending run scheduled". The migration must preserve both
    // shapes (`"key": null` and `"key" missing`) as SQL NULL.
    await runWithTempDir(async (tempDir) => {
      writeTaskTzStateFile(tempDir, 'telegram_main', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        follow_me_tasks: [
          // Explicit nulls.
          {
            name: 'task-explicit-null',
            local_time: '06:00',
            schedule_value: '0 11 * * *',
            last_run_date: null,
            pending_run_at: null,
          },
          // Keys absent entirely.
          {
            name: 'task-key-absent',
            local_time: '07:00',
            schedule_value: '0 12 * * *',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const tasks = db
            .prepare(
              'SELECT name, last_run_date, pending_run_at FROM follow_me_tasks ORDER BY name',
            )
            .all() as Array<Record<string, unknown>>;
          expect(tasks).toEqual([
            {
              name: 'task-explicit-null',
              last_run_date: null,
              pending_run_at: null,
            },
            {
              name: 'task-key-absent',
              last_run_date: null,
              pending_run_at: null,
            },
          ]);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('skips when current_tz or home_tz is missing/empty; warns; file left in place; no rows inserted', async () => {
    // Both columns are NOT NULL on tz_state. An empty string would
    // satisfy NOT NULL but break every reader (IANA zone name
    // expected). The migration treats missing/non-string/empty as
    // "malformed envelope": warn, file in place, and DON'T import the
    // sibling follow_me_tasks rows (a partial import would silently
    // ship N task rows without their tz context).
    const cases: Array<{ label: string; payload: TaskTzStateJsonShape }> = [
      {
        label: 'missing current_tz',
        payload: {
          home_tz: 'America/Chicago',
          follow_me_tasks: [
            {
              name: 'morning-brief',
              local_time: '08:00',
              schedule_value: '0 13 * * *',
            },
          ],
        },
      },
      {
        label: 'missing home_tz',
        payload: {
          current_tz: 'America/Chicago',
          follow_me_tasks: [
            {
              name: 'morning-brief',
              local_time: '08:00',
              schedule_value: '0 13 * * *',
            },
          ],
        },
      },
      {
        label: 'empty current_tz',
        payload: {
          current_tz: '',
          home_tz: 'America/Chicago',
        },
      },
      {
        label: 'empty home_tz',
        payload: {
          current_tz: 'America/Chicago',
          home_tz: '',
        },
      },
    ];
    for (const { payload } of cases) {
      await runWithTempDir(async (tempDir) => {
        const filePath = writeTaskTzStateFile(
          tempDir,
          'telegram_main',
          payload,
        );

        vi.resetModules();
        const { initDatabase, _closeDatabase } = await import('./db.js');
        const { logger } = await import('./logger.js');
        const warnSpy = vi.spyOn(logger, 'warn');
        try {
          initDatabase();
          const db = new Database(path.join(tempDir, 'store', 'messages.db'));
          try {
            const tzCount = (
              db.prepare('SELECT COUNT(*) AS n FROM tz_state').get() as {
                n: number;
              }
            ).n;
            expect(tzCount).toBe(0);
            const taskCount = (
              db.prepare('SELECT COUNT(*) AS n FROM follow_me_tasks').get() as {
                n: number;
              }
            ).n;
            expect(taskCount).toBe(0);
          } finally {
            db.close();
          }
          expect(fs.existsSync(filePath)).toBe(true);
          const skipWarnFired = warnSpy.mock.calls.some((call) => {
            const msg = call.find((arg) => typeof arg === 'string') as
              | string
              | undefined;
            return Boolean(
              msg && msg.includes('missing or empty current_tz / home_tz'),
            );
          });
          expect(skipWarnFired).toBe(true);
        } finally {
          warnSpy.mockRestore();
          _closeDatabase();
        }
      });
    }
  });

  it('lands tz_state but leaves follow_me_tasks empty when the array is empty; renames source', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeTaskTzStateFile(tempDir, 'telegram_main', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        scheduler_tz: 'America/Chicago',
        follow_me_tasks: [],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const tzCount = (
            db.prepare('SELECT COUNT(*) AS n FROM tz_state').get() as {
              n: number;
            }
          ).n;
          expect(tzCount).toBe(1);
          const taskCount = (
            db.prepare('SELECT COUNT(*) AS n FROM follow_me_tasks').get() as {
              n: number;
            }
          ).n;
          expect(taskCount).toBe(0);
        } finally {
          db.close();
        }
        // Source still renamed even with zero tasks — the tz_state row
        // is the meaningful import.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('task-tz-state.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('UPSERT preserves schema_version on the second group import (multi-group, NOT INSERT OR REPLACE)', async () => {
    // Load-bearing assertion. Two groups each carry a task-tz-state.
    // json. With the schema's `CHECK(id = 1)` enforcing a singleton
    // tz_state, the second group's UPSERT must update the same row
    // in place — row count stays at 1. AND because the writer uses
    // `ON CONFLICT(id) DO UPDATE` (NOT `INSERT OR REPLACE`, which is
    // delete+insert in SQLite and resets defaulted columns), an
    // already-bumped `schema_version` survives the second group's
    // import. We pin that down by manually bumping schema_version to
    // 5 between the two imports and asserting it's still 5 after the
    // second group lands.
    //
    // To stage the manual bump in the middle of the migration pass,
    // we run the migration once with only the first group's file
    // present, perform the manual UPDATE, drop the second group's
    // file, and re-run initDatabase. That mirrors how the rename-on-
    // success makes a re-run idempotent: the first folder's file is
    // already `.migrated-...`, only the new file gets imported on
    // pass two.
    await runWithTempDir(async (tempDir) => {
      writeTaskTzStateFile(tempDir, 'group_a', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        follow_me_tasks: [],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const dbPath = path.join(tempDir, 'store', 'messages.db');
        // Bump schema_version manually between imports — emulates a
        // future state-NNN migration that bumped the version on
        // existing rows. The second group's UPSERT must NOT clobber
        // it.
        {
          const db = new Database(dbPath);
          try {
            db.prepare(
              'UPDATE tz_state SET schema_version = 5 WHERE id = 1',
            ).run();
            const before = db
              .prepare('SELECT schema_version FROM tz_state WHERE id = 1')
              .get() as { schema_version: number };
            expect(before.schema_version).toBe(5);
          } finally {
            db.close();
          }
        }
      } finally {
        _closeDatabase();
      }

      // Drop a second group's file in place; re-run initDatabase to
      // drive the second import pass.
      writeTaskTzStateFile(tempDir, 'group_b', {
        current_tz: 'Europe/Berlin',
        home_tz: 'Europe/Berlin',
        follow_me_tasks: [],
      });

      vi.resetModules();
      const { initDatabase: initDatabase2, _closeDatabase: close2 } =
        await import('./db.js');
      initDatabase2();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // Singleton guarantee: still exactly one row.
          const tzRows = db
            .prepare(
              'SELECT id, current_tz, home_tz, schema_version FROM tz_state',
            )
            .all() as Array<Record<string, unknown>>;
          expect(tzRows).toHaveLength(1);
          expect(tzRows[0]).toMatchObject({
            id: 1,
            current_tz: 'Europe/Berlin',
            home_tz: 'Europe/Berlin',
          });
          // Load-bearing: schema_version preserved through UPSERT. If
          // the writer ever drifts to `INSERT OR REPLACE`, this
          // assertion catches it because that statement is delete+
          // insert in SQLite and would reset schema_version to the
          // schema's `DEFAULT 1`.
          expect(tzRows[0].schema_version).toBe(5);
        } finally {
          db.close();
        }
      } finally {
        close2();
      }
    });
  });

  it('skips a non-object element in follow_me_tasks with a warn; sibling rows still import', async () => {
    // A stale non-object element (null / number / string) in the
    // follow_me_tasks array would otherwise throw a TypeError before
    // any UPSERT runs (e.g. `null.name` blows up), and the narrowed
    // catch on constraint-class SqliteError would propagate that as
    // "unexpected" and halt orchestrator startup. Per `coding-policy:
    // error-handling`, the per-row guard skips the bad element with
    // a warn and lets sibling rows import normally.
    await runWithTempDir(async (tempDir) => {
      writeTaskTzStateFile(tempDir, 'telegram_main', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        follow_me_tasks: [
          {
            name: 'good-task',
            local_time: '08:00',
            schedule_value: '0 13 * * *',
          },
          // Non-object stragglers — JSON is permissive, so any of
          // these shapes can appear in a stale envelope.
          null as unknown as TaskTzStateFollowMeTaskJson,
          42 as unknown as TaskTzStateFollowMeTaskJson,
          'not-a-task' as unknown as TaskTzStateFollowMeTaskJson,
          {
            name: 'good-task-2',
            local_time: '09:00',
            schedule_value: '0 14 * * *',
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
          const tasks = db
            .prepare('SELECT name FROM follow_me_tasks ORDER BY name')
            .all() as Array<{ name: string }>;
          expect(tasks.map((t) => t.name)).toEqual([
            'good-task',
            'good-task-2',
          ]);
        } finally {
          db.close();
        }
        // Per non-object element fired a warn carrying the
        // `queue: 'follow_me_tasks'` discriminator so triage knows
        // exactly which sub-array was malformed.
        const nonObjectWarns = warnSpy.mock.calls.filter((call) => {
          const meta = call.find(
            (arg) =>
              typeof arg === 'object' &&
              arg !== null &&
              'queue' in (arg as object),
          ) as { queue?: string } | undefined;
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(
            meta?.queue === 'follow_me_tasks' &&
            msg &&
            msg.includes('non-object row'),
          );
        });
        expect(nonObjectWarns.length).toBe(3);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('is a no-op when the source file has already been renamed (re-run idempotent)', async () => {
    await runWithTempDir(async (tempDir) => {
      // Simulate a successful prior run: only the renamed file
      // exists. The migration loop's existsSync gate at the top
      // skips over folders without a live `task-tz-state.json`.
      const folder = path.join(tempDir, 'groups', 'telegram_main');
      fs.mkdirSync(folder, { recursive: true });
      const renamedPath = path.join(
        folder,
        'task-tz-state.json.migrated-2026-04-20',
      );
      fs.writeFileSync(
        renamedPath,
        JSON.stringify({
          current_tz: 'America/Chicago',
          home_tz: 'America/Chicago',
          follow_me_tasks: [
            {
              name: 'already-imported',
              local_time: '08:00',
              schedule_value: '0 13 * * *',
            },
          ],
        }),
      );

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      expect(() => initDatabase()).not.toThrow();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const tzCount = (
            db.prepare('SELECT COUNT(*) AS n FROM tz_state').get() as {
              n: number;
            }
          ).n;
          expect(tzCount).toBe(0);
          const taskCount = (
            db.prepare('SELECT COUNT(*) AS n FROM follow_me_tasks').get() as {
              n: number;
            }
          ).n;
          expect(taskCount).toBe(0);
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

  it('skips a malformed JSON file with a warn and leaves it in place', async () => {
    await runWithTempDir(async (tempDir) => {
      // Mix one valid file and one malformed file. The malformed
      // file must NOT abort the whole migration pass — other groups
      // still need to import. Helper-driven via parseJsonObjectOrWarn.
      const goodFile = writeTaskTzStateFile(tempDir, 'good', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        follow_me_tasks: [
          {
            name: 'good-task',
            local_time: '08:00',
            schedule_value: '0 13 * * *',
          },
        ],
      });
      const badFolder = path.join(tempDir, 'groups', 'bad');
      fs.mkdirSync(badFolder, { recursive: true });
      const badFile = path.join(badFolder, 'task-tz-state.json');
      fs.writeFileSync(badFile, '{ this is not valid json');

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const tasks = db
            .prepare('SELECT name FROM follow_me_tasks')
            .all() as Array<{ name: string }>;
          expect(tasks.map((r) => r.name)).toEqual(['good-task']);
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

  it('skips a non-object JSON payload (null / number / array) with a warning and leaves it in place', async () => {
    // JSON.parse returns null / numbers / strings / arrays for
    // syntactically valid but non-object payloads. The migration
    // must guard against those before binding to property accesses,
    // otherwise one bad file would throw "cannot read properties of
    // null" and abort the whole pass. Helper-driven via
    // parseJsonObjectOrWarn.
    for (const payload of ['null', '42', '"hello"', '[1, 2, 3]']) {
      await runWithTempDir(async (tempDir) => {
        const goodFile = writeTaskTzStateFile(tempDir, 'good', {
          current_tz: 'America/Chicago',
          home_tz: 'America/Chicago',
          follow_me_tasks: [
            {
              name: `good-${payload.replace(/\W/g, '')}`,
              local_time: '08:00',
              schedule_value: '0 13 * * *',
            },
          ],
        });
        const badFolder = path.join(tempDir, 'groups', 'bad');
        fs.mkdirSync(badFolder, { recursive: true });
        const badFile = path.join(badFolder, 'task-tz-state.json');
        fs.writeFileSync(badFile, payload);

        vi.resetModules();
        const { initDatabase, _closeDatabase } = await import('./db.js');
        const { logger } = await import('./logger.js');
        const warnSpy = vi.spyOn(logger, 'warn');
        try {
          expect(() => initDatabase()).not.toThrow();
          const db = new Database(path.join(tempDir, 'store', 'messages.db'));
          try {
            const tasks = db
              .prepare('SELECT name FROM follow_me_tasks')
              .all() as Array<{ name: string }>;
            expect(tasks).toHaveLength(1);
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
