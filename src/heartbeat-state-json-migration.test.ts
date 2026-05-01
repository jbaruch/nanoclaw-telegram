import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';

// Mirrors the temp-dir + chdir pattern in
// `morning-brief-pending-json-migration.test.ts` so the CWD-rooted
// `STORE_DIR` and `GROUPS_DIR` constants in `src/config.ts` resolve
// under an isolated tempDir per scenario.

interface HeartbeatStateJsonShape {
  heartbeat_last_completed?: string;
  nightly_last_completed?: string;
  weekly_last_completed?: string;
  last_composio_check?: string;
}

function writeHeartbeatFile(
  tempDir: string,
  groupName: string,
  payload: HeartbeatStateJsonShape,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'heartbeat-state.json');
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  return filePath;
}

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-heartbeat-'));
  try {
    process.chdir(tempDir);
    fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

describe('heartbeat-state.json -> phase_completions migration (#301)', () => {
  it('imports all three phase rows, lands last_composio_check in heartbeat metadata only, renames source', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeHeartbeatFile(tempDir, 'telegram_swarm', {
        heartbeat_last_completed: '2026-04-29T08:00:00.000Z',
        nightly_last_completed: '2026-04-28T22:00:00.000Z',
        weekly_last_completed: '2026-04-26T22:00:00.000Z',
        last_composio_check: '2026-04-29T08:00:00.000Z',
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const rows = db
            .prepare(
              'SELECT phase, last_completed, metadata FROM phase_completions ORDER BY phase',
            )
            .all() as Array<{
            phase: string;
            last_completed: string;
            metadata: string | null;
          }>;
          expect(rows).toHaveLength(3);
          expect(rows[0]).toMatchObject({
            phase: 'heartbeat',
            last_completed: '2026-04-29T08:00:00.000Z',
          });
          expect(rows[1]).toMatchObject({
            phase: 'nightly',
            last_completed: '2026-04-28T22:00:00.000Z',
          });
          expect(rows[2]).toMatchObject({
            phase: 'weekly',
            last_completed: '2026-04-26T22:00:00.000Z',
          });

          // metadata is opaque TEXT in the schema; readers parse it as
          // JSON themselves. The test pins the writer's contract: the
          // heartbeat row carries `{last_composio_check: ...}`, the
          // nightly + weekly rows are NULL.
          expect(rows[0].metadata).not.toBeNull();
          expect(JSON.parse(rows[0].metadata as string)).toEqual({
            last_composio_check: '2026-04-29T08:00:00.000Z',
          });
          expect(rows[1].metadata).toBeNull();
          expect(rows[2].metadata).toBeNull();
        } finally {
          db.close();
        }

        // Source renamed to .migrated-<YYYY-MM-DD>; the rename is what
        // makes a re-run a no-op.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('heartbeat-state.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('imports only the phases present in the source — others stay absent (count = 0)', async () => {
    // Source carries only `heartbeat_last_completed`. The schema
    // doesn't enforce a CHECK on the phase string, so "no row yet"
    // is the legitimate steady state for a phase that never ran.
    // Readers (heartbeat-precheck.py) tolerate this.
    await runWithTempDir(async (tempDir) => {
      writeHeartbeatFile(tempDir, 'telegram_swarm', {
        heartbeat_last_completed: '2026-04-29T08:00:00.000Z',
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const heartbeatCount = (
            db
              .prepare(
                "SELECT count(*) AS n FROM phase_completions WHERE phase = 'heartbeat'",
              )
              .get() as { n: number }
          ).n;
          expect(heartbeatCount).toBe(1);

          const nightlyCount = (
            db
              .prepare(
                "SELECT count(*) AS n FROM phase_completions WHERE phase = 'nightly'",
              )
              .get() as { n: number }
          ).n;
          expect(nightlyCount).toBe(0);

          const weeklyCount = (
            db
              .prepare(
                "SELECT count(*) AS n FROM phase_completions WHERE phase = 'weekly'",
              )
              .get() as { n: number }
          ).n;
          expect(weeklyCount).toBe(0);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('UPSERT semantics on re-run: newer last_completed wins; schema_version is NOT reset to default', async () => {
    // Pre-seed the table with a stale `last_completed` AND a bumped
    // `schema_version`, then run import. Locks down two contracts at
    // once:
    //   1. The new `last_completed` wins (UPSERT vs. nothing).
    //   2. The bumped `schema_version` survives — `INSERT OR REPLACE`
    //      would delete+insert the row, defaulting `schema_version`
    //      back to 1; the explicit `DO UPDATE SET (last_completed,
    //      metadata, updated_at)` UPSERT touches only those three
    //      columns and leaves `schema_version` alone.
    await runWithTempDir(async (tempDir) => {
      writeHeartbeatFile(tempDir, 'telegram_swarm', {
        heartbeat_last_completed: '2026-04-29T08:00:00.000Z',
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');

      // First boot: seed table with a stale row carrying a bumped
      // schema_version, then move the source file back into place so
      // the second initDatabase call re-runs the import on top of the
      // seeded row.
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          db.prepare(
            `INSERT OR REPLACE INTO phase_completions
               (phase, last_completed, metadata, schema_version)
             VALUES ('heartbeat', '2025-01-01T00:00:00.000Z', NULL, 99)`,
          ).run();
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }

      // Restore the source file (the first init renamed it). A second
      // initDatabase pass should UPSERT the stale row.
      const folder = path.join(tempDir, 'groups', 'telegram_swarm');
      const renamed = fs
        .readdirSync(folder)
        .filter((f) => f.startsWith('heartbeat-state.json.migrated-'))[0];
      fs.renameSync(
        path.join(folder, renamed),
        path.join(folder, 'heartbeat-state.json'),
      );

      vi.resetModules();
      const second = await import('./db.js');
      second.initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const row = db
            .prepare(
              "SELECT last_completed, schema_version FROM phase_completions WHERE phase = 'heartbeat'",
            )
            .get() as { last_completed: string; schema_version: number };
          // The new value won — UPSERT, not OR REPLACE-as-no-op.
          expect(row.last_completed).toBe('2026-04-29T08:00:00.000Z');
          // The bumped schema_version survives — `INSERT OR REPLACE`
          // would have reset it to the default 1.
          expect(row.schema_version).toBe(99);
        } finally {
          db.close();
        }
      } finally {
        second._closeDatabase();
      }
    });
  });

  it('multi-group: alphabetically-last folder wins the UPSERT; total row count stays at 3', async () => {
    // Two groups each with `heartbeat-state.json`, deliberately picked
    // so plain-codepoint sort puts `aaa-first` before `zzz-last`. The
    // first group's UPSERTs land, then the second group's UPSERTs
    // update-in-place — the row count for each phase stays at 1 (3
    // total), and the second (alphabetically-last) folder's
    // last_completed values win. The `listGroupFoldersForMigration`
    // helper sorts by code-point, NOT by locale, so this is stable
    // across filesystems and Node versions.
    await runWithTempDir(async (tempDir) => {
      writeHeartbeatFile(tempDir, 'aaa-first', {
        heartbeat_last_completed: '2026-04-01T00:00:00.000Z',
        nightly_last_completed: '2026-04-01T00:00:00.000Z',
        weekly_last_completed: '2026-04-01T00:00:00.000Z',
      });
      writeHeartbeatFile(tempDir, 'zzz-last', {
        heartbeat_last_completed: '2026-04-29T08:00:00.000Z',
        nightly_last_completed: '2026-04-28T22:00:00.000Z',
        weekly_last_completed: '2026-04-26T22:00:00.000Z',
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const total = (
            db.prepare('SELECT count(*) AS n FROM phase_completions').get() as {
              n: number;
            }
          ).n;
          // Three rows total — `phase` is the PK, so two groups each
          // contributing the same three phase strings collapse to
          // three rows, not six.
          expect(total).toBe(3);

          const rows = db
            .prepare(
              'SELECT phase, last_completed FROM phase_completions ORDER BY phase',
            )
            .all() as Array<{ phase: string; last_completed: string }>;
          // `zzz-last` is alphabetically last, so its UPSERT runs
          // second and wins on every phase.
          expect(rows).toEqual([
            { phase: 'heartbeat', last_completed: '2026-04-29T08:00:00.000Z' },
            { phase: 'nightly', last_completed: '2026-04-28T22:00:00.000Z' },
            { phase: 'weekly', last_completed: '2026-04-26T22:00:00.000Z' },
          ]);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('is a no-op once the source file is renamed to .migrated-<stamp>', async () => {
    // After a successful import, the source carries the
    // `.migrated-YYYY-MM-DD` suffix. A subsequent boot must NOT
    // re-import the renamed file; only `heartbeat-state.json` (no
    // suffix) is eligible.
    await runWithTempDir(async (tempDir) => {
      writeHeartbeatFile(tempDir, 'telegram_swarm', {
        heartbeat_last_completed: '2026-04-29T08:00:00.000Z',
        nightly_last_completed: '2026-04-28T22:00:00.000Z',
      });

      vi.resetModules();
      const first = await import('./db.js');
      first.initDatabase();
      first._closeDatabase();

      const folder = path.join(tempDir, 'groups', 'telegram_swarm');
      // Confirm the rename happened.
      expect(
        fs
          .readdirSync(folder)
          .filter((f) => f.startsWith('heartbeat-state.json.migrated-')),
      ).toHaveLength(1);
      expect(fs.existsSync(path.join(folder, 'heartbeat-state.json'))).toBe(
        false,
      );

      // Second boot: tamper with the renamed file's contents. If the
      // migration mistakenly read it, the `last_completed` would
      // change. It must NOT.
      const renamed = fs
        .readdirSync(folder)
        .filter((f) => f.startsWith('heartbeat-state.json.migrated-'))[0];
      fs.writeFileSync(
        path.join(folder, renamed),
        JSON.stringify({
          heartbeat_last_completed: '2099-12-31T23:59:59.000Z',
        }),
      );

      vi.resetModules();
      const second = await import('./db.js');
      second.initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const row = db
            .prepare(
              "SELECT last_completed FROM phase_completions WHERE phase = 'heartbeat'",
            )
            .get() as { last_completed: string };
          // First-boot value preserved; renamed file ignored.
          expect(row.last_completed).toBe('2026-04-29T08:00:00.000Z');
        } finally {
          db.close();
        }
      } finally {
        second._closeDatabase();
      }
    });
  });

  it('skips a non-object JSON payload with a warning and leaves it in place (uses parseJsonObjectOrWarn)', async () => {
    // `JSON.parse` returns null / numbers / strings / arrays for
    // syntactically valid but non-object payloads. The migration
    // delegates to `parseJsonObjectOrWarn` which warns-and-skips on
    // each non-object shape so one bad file doesn't abort the pass
    // (per `coding-policy: error-handling`, "try alternatives before
    // failing").
    for (const payload of ['null', '42', '"hello"', '[1, 2, 3]']) {
      await runWithTempDir(async (tempDir) => {
        const goodFile = writeHeartbeatFile(tempDir, 'good', {
          heartbeat_last_completed: '2026-04-29T08:00:00.000Z',
        });
        const badFolder = path.join(tempDir, 'groups', 'bad');
        fs.mkdirSync(badFolder, { recursive: true });
        const badFile = path.join(badFolder, 'heartbeat-state.json');
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
            const total = (
              db
                .prepare('SELECT count(*) AS n FROM phase_completions')
                .get() as { n: number }
            ).n;
            expect(total).toBe(1);
          } finally {
            db.close();
          }
          // Good file renamed; bad file left in place for triage.
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

  it('skips a malformed JSON file with a warning and leaves it in place (uses parseJsonObjectOrWarn)', async () => {
    // Mix one valid and one malformed file. A malformed file must
    // NOT abort the rest of the pass — other groups still need to
    // migrate. The malformed file is left in place for human triage.
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeHeartbeatFile(tempDir, 'good', {
        heartbeat_last_completed: '2026-04-29T08:00:00.000Z',
      });
      const badFolder = path.join(tempDir, 'groups', 'bad');
      fs.mkdirSync(badFolder, { recursive: true });
      const badFile = path.join(badFolder, 'heartbeat-state.json');
      fs.writeFileSync(badFile, '{ this is not valid json');

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const phases = db
            .prepare('SELECT phase FROM phase_completions')
            .all() as Array<{ phase: string }>;
          expect(phases.map((r) => r.phase)).toEqual(['heartbeat']);
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
});
