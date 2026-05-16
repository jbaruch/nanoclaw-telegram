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
  // Legacy fields per #431 — pre-state-010 writer shape.
  task_id?: unknown;
  local_hour?: unknown;
  local_minute?: unknown;
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

  it('UPSERT writes the writer-known schema_version on the second group import (singleton, NOT INSERT OR REPLACE)', async () => {
    // Load-bearing assertion (post-#542). Two groups each carry a
    // task-tz-state.json. With the schema's `CHECK(id = 1)` enforcing
    // a singleton tz_state, the second group's UPSERT must update the
    // same row in place — row count stays at 1.
    //
    // Pre-#542 the writer omitted `schema_version` from its column
    // list, so an already-bumped value survived. Post-#542 the writer
    // is expected to STAY in sync with the reader gate
    // (`SUPPORTED_TZ_STATE_SCHEMA_VERSION` in src/db.ts — currently 3
    // post-jbaruch/nanoclaw-admin#229; was 2 between #542 and #229)
    // by writing the constant's current value explicitly. The thing
    // we're still catching: a regression to `INSERT OR REPLACE`,
    // which is delete+insert in SQLite and would reset schema_version
    // to the schema's `DEFAULT 1`. So we manually bump to 5 between
    // imports, run the second import, and assert the result equals
    // the writer's known shape (the constant's current value) —
    // anything else (5, 1, or undefined) signals either a delete+
    // insert regression (1) or that the manual UPDATE isn't being
    // touched at all (5).
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
          // Load-bearing: schema_version equals the writer's known
          // shape (4 post-#574 Phase 2 in lock-step with the reader
          // gate; was 3 between #229 and #574 Phase 2, 2 between #542
          // and #229). If the writer ever drifts to `INSERT OR
          // REPLACE`, the result would be `1` (delete+insert in
          // SQLite resets to the schema's `DEFAULT 1`); if the writer
          // ever drops the explicit column, the result would be `5`
          // (the manual bump survives). Either drift breaks the
          // singleton's shape contract — assert exactly the current
          // writer's shape value.
          expect(tzRows[0].schema_version).toBe(4);
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

  // #431 — Legacy follow_me shape: pre-state-010 task-tz-sync writer
  // emitted `local_hour: N, local_minute: N` integers and keyed the
  // cron via `task_id` on the sibling `scheduled_tasks` row instead of
  // duplicating it on the follow_me entry. The migration must
  // synthesize `local_time = "HH:MM"` and look the cron up by task_id;
  // otherwise the file silently fails the constraint and stays in
  // place every boot.
  it('synthesizes local_time from local_hour/local_minute and looks up schedule_value by task_id (legacy shape, #431)', async () => {
    await runWithTempDir(async (tempDir) => {
      // Pass 1: prime the DB so scheduled_tasks rows exist before the
      // JSON migration runs. initDatabase creates schema and runs
      // state migrations; we close, drop the JSON, and re-init in
      // pass 2 to actually exercise the legacy-shape lookup.
      vi.resetModules();
      {
        const { initDatabase, _closeDatabase } = await import('./db.js');
        initDatabase();
        try {
          const db = new Database(path.join(tempDir, 'store', 'messages.db'));
          try {
            const insertCron = db.prepare(
              `INSERT INTO scheduled_tasks
                 (id, group_folder, chat_jid, prompt, schedule_type, schedule_value, created_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
            );
            insertCron.run(
              'task-1776347641153-dexq4p',
              'telegram_main',
              'tg:1',
              'morning-brief',
              'cron',
              '0 7 * * *',
              '2026-04-25T00:00:00Z',
              'active',
            );
            insertCron.run(
              'task-1776347643027-tlz3uc',
              'telegram_main',
              'tg:1',
              'nightly-housekeeping',
              'cron',
              '0 3 * * *',
              '2026-04-25T00:00:00Z',
              'active',
            );
          } finally {
            db.close();
          }
        } finally {
          _closeDatabase();
        }
      }

      const filePath = writeTaskTzStateFile(tempDir, 'telegram_main', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        scheduler_tz: 'America/Chicago',
        follow_me_tasks: [
          {
            task_id: 'task-1776347641153-dexq4p',
            name: 'morning-brief',
            local_hour: 7,
            local_minute: 0,
            last_run_date: '2026-05-01',
            pending_run_at: null,
          },
          {
            task_id: 'task-1776347643027-tlz3uc',
            name: 'nightly-housekeeping',
            local_hour: 3,
            local_minute: 0,
            last_run_date: '2026-05-01',
            pending_run_at: null,
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
              'SELECT name, local_time, schedule_value, last_run_date FROM follow_me_tasks ORDER BY name',
            )
            .all() as Array<Record<string, unknown>>;
          expect(tasks).toEqual([
            {
              name: 'morning-brief',
              local_time: '07:00',
              schedule_value: '0 7 * * *',
              last_run_date: '2026-05-01',
            },
            {
              name: 'nightly-housekeeping',
              local_time: '03:00',
              schedule_value: '0 3 * * *',
              last_run_date: '2026-05-01',
            },
          ]);
        } finally {
          db.close();
        }
        // Source renamed — the constraint failure that #431 was about
        // is gone; subsequent boots are no-op idempotent.
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

  it('zero-pads single-digit local_hour / local_minute correctly (boundary: 00:00, 09:05)', async () => {
    await runWithTempDir(async (tempDir) => {
      // First-pass init to create scheduled_tasks rows the legacy
      // lookup needs.
      vi.resetModules();
      {
        const { initDatabase, _closeDatabase } = await import('./db.js');
        initDatabase();
        try {
          const db = new Database(path.join(tempDir, 'store', 'messages.db'));
          try {
            const insertCron = db.prepare(
              `INSERT INTO scheduled_tasks
                 (id, group_folder, chat_jid, prompt, schedule_type, schedule_value, created_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
            );
            insertCron.run(
              'task-zero',
              'telegram_main',
              'tg:1',
              'midnight',
              'cron',
              '0 0 * * *',
              '2026-04-25T00:00:00Z',
              'active',
            );
            insertCron.run(
              'task-single-digit',
              'telegram_main',
              'tg:1',
              'morning',
              'cron',
              '5 9 * * *',
              '2026-04-25T00:00:00Z',
              'active',
            );
          } finally {
            db.close();
          }
        } finally {
          _closeDatabase();
        }
      }

      writeTaskTzStateFile(tempDir, 'telegram_main', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        follow_me_tasks: [
          {
            task_id: 'task-zero',
            name: 'midnight',
            local_hour: 0,
            local_minute: 0,
          },
          {
            task_id: 'task-single-digit',
            name: 'morning',
            local_hour: 9,
            local_minute: 5,
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
              'SELECT name, local_time FROM follow_me_tasks ORDER BY name',
            )
            .all() as Array<Record<string, unknown>>;
          expect(tasks).toEqual([
            { name: 'midnight', local_time: '00:00' },
            { name: 'morning', local_time: '09:05' },
          ]);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  // PR #434 review feedback (Copilot): a follow_me row with a missing
  // or non-string `name` would otherwise hit the `name TEXT PRIMARY KEY`
  // constraint at upsert and roll back the entire group's transaction
  // — defeating the per-row-skip purpose of the rest of the fix. The
  // `name` validator runs before the local_time / schedule_value
  // resolver and produces a distinct warn message so triage can tell
  // the two skip paths apart.
  it('skips a follow_me row with missing or non-string name (PK); sibling rows still land; file renamed', async () => {
    await runWithTempDir(async (tempDir) => {
      writeTaskTzStateFile(tempDir, 'telegram_main', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        follow_me_tasks: [
          // name is missing entirely — would fail `name TEXT PRIMARY KEY`
          // and roll back the whole group's transaction without the
          // pre-upsert validator.
          {
            local_time: '08:00',
            schedule_value: '0 13 * * *',
          },
          // name is the wrong type (number) — JSON.stringify would
          // happily emit this and the unchecked cast would coerce.
          {
            name: 42,
            local_time: '09:00',
            schedule_value: '0 14 * * *',
          },
          // name is the empty string — satisfies `typeof === 'string'`
          // but not the not-empty contract.
          {
            name: '',
            local_time: '10:00',
            schedule_value: '0 15 * * *',
          },
          // Healthy sibling row; must still land despite the three above.
          {
            name: 'healthy-sibling',
            local_time: '06:00',
            schedule_value: '0 11 * * *',
          },
        ],
      });

      const filePath = path.join(
        tempDir,
        'groups',
        'telegram_main',
        'task-tz-state.json',
      );

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
            .all() as Array<Record<string, unknown>>;
          expect(tasks).toEqual([{ name: 'healthy-sibling' }]);
        } finally {
          db.close();
        }
        const nameSkipWarns = warnSpy.mock.calls.filter((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(
            msg &&
            msg.includes(
              'skipping follow_me row with missing or non-string name',
            ),
          );
        });
        // One warn per malformed row (3 total).
        expect(nameSkipWarns.length).toBe(3);
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('task-tz-state.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('skips a legacy follow_me row whose task_id has no matching scheduled_tasks row; sibling modern row still lands; file renamed', async () => {
    await runWithTempDir(async (tempDir) => {
      // Don't pre-populate scheduled_tasks. The legacy entry with
      // an unresolvable task_id should skip-and-warn while the
      // sibling modern-shape row imports normally and the source
      // file gets renamed (per-row skip, not per-file rollback).
      writeTaskTzStateFile(tempDir, 'telegram_main', {
        current_tz: 'America/Chicago',
        home_tz: 'America/Chicago',
        follow_me_tasks: [
          {
            task_id: 'task-orphan',
            name: 'orphan-legacy',
            local_hour: 5,
            local_minute: 30,
          },
          {
            name: 'modern-sibling',
            local_time: '06:00',
            schedule_value: '0 11 * * *',
          },
        ],
      });

      const filePath = path.join(
        tempDir,
        'groups',
        'telegram_main',
        'task-tz-state.json',
      );

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
            .all() as Array<Record<string, unknown>>;
          expect(tasks).toEqual([{ name: 'modern-sibling' }]);
        } finally {
          db.close();
        }
        const skipWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(
            msg &&
            msg.includes(
              'cannot resolve local_time / schedule_value for follow_me row',
            ),
          );
        });
        expect(skipWarnFired).toBe(true);
        // File still renamed — sibling row succeeded so the per-file
        // import counts as a partial success rather than a full skip.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('task-tz-state.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });
});
