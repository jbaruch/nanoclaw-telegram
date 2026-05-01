import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';

// Each scenario builds an isolated tempDir + chdir so the
// CWD-rooted `STORE_DIR` and `GROUPS_DIR` constants in `src/config.ts`
// resolve under it. Mirrors the pattern in
// `src/morning-brief-pending-json-migration.test.ts` and
// `src/orders-json-migration.test.ts`.

interface TrustedSessionStateJsonShape {
  schema_version?: number;
  sessions?: Record<string, unknown>;
  active_session_id?: string;
  // The trusted-memory schema treats these as opaque JSON blobs;
  // the migration JSON-stringifies structured shapes before binding.
  pending_response?: unknown;
  muted_threads?: unknown;
  // The JSON-era top-level `seen_email_ids` field intentionally does
  // NOT migrate here (it relocates to `email_seen_ids` from #297).
  // Tests assert no rows land in that table from this migration's
  // run.
  seen_email_ids?: string[];
}

function writeSessionStateFile(
  tempDir: string,
  groupName: string,
  payload: TrustedSessionStateJsonShape | string,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'session-state.json');
  fs.writeFileSync(
    filePath,
    typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2),
  );
  return filePath;
}

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-trsess-'));
  try {
    process.chdir(tempDir);
    fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

describe('session-state.json → trusted_* SQLite tables migration (#298)', () => {
  it('imports per-named-session rows + singleton; renames the source file', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeSessionStateFile(tempDir, 'telegram_main', {
        schema_version: 1,
        sessions: {
          default: {
            session_id: 'sdk-default-001',
            started: '2026-04-29T08:00:00.000Z',
            epoch: 1714377600,
            last_seen: '2026-04-29T09:30:00.000Z',
          },
          cleanup: {
            session_id: 'sdk-cleanup-001',
            started: '2026-04-29T08:05:00.000Z',
            epoch: 1714377900,
            last_seen: '2026-04-29T09:31:00.000Z',
          },
        },
        active_session_id: 'sdk-default-001',
        pending_response: { kind: 'reply', body: 'check Tuesday' },
        muted_threads: ['thread-a', 'thread-b'],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const sessions = db
            .prepare('SELECT * FROM trusted_sessions ORDER BY session_name')
            .all() as Array<Record<string, unknown>>;
          expect(sessions).toHaveLength(2);
          expect(sessions[0]).toMatchObject({
            session_name: 'cleanup',
            session_id: 'sdk-cleanup-001',
            started: '2026-04-29T08:05:00.000Z',
            epoch: 1714377900,
            last_seen: '2026-04-29T09:31:00.000Z',
          });
          expect(sessions[1]).toMatchObject({
            session_name: 'default',
            session_id: 'sdk-default-001',
            started: '2026-04-29T08:00:00.000Z',
            epoch: 1714377600,
            last_seen: '2026-04-29T09:30:00.000Z',
          });

          const singleton = db
            .prepare('SELECT * FROM trusted_session_singleton WHERE id = 1')
            .get() as Record<string, unknown>;
          expect(singleton).toBeDefined();
          expect(singleton.active_session_id).toBe('sdk-default-001');
          // pending_response and muted_threads are TEXT in the schema
          // (opaque JSON blobs per the owner-skill contract). The
          // migration JSON-stringifies structured shapes; round-trip
          // back through JSON.parse to assert preservation.
          expect(JSON.parse(singleton.pending_response as string)).toEqual({
            kind: 'reply',
            body: 'check Tuesday',
          });
          expect(JSON.parse(singleton.muted_threads as string)).toEqual([
            'thread-a',
            'thread-b',
          ]);
        } finally {
          db.close();
        }

        // Source file renamed to .migrated-<YYYY-MM-DD>; the rename is
        // what makes a re-run a no-op.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('session-state.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('UPSERTs the singleton in place when a second group has its own file (CHECK(id=1) preserved, schema_version not reset)', async () => {
    // Two groups each have a session-state.json. Both files contribute
    // to the same row in trusted_session_singleton (id=1). The contract
    // is UPSERT, not REPLACE: row count must stay at 1 across the
    // second import, and the existing `schema_version` must be
    // preserved (REPLACE would reset it to the column DEFAULT).
    await runWithTempDir(async (tempDir) => {
      writeSessionStateFile(tempDir, 'group_a', {
        active_session_id: 'sdk-a-001',
        pending_response: 'a-pending',
        muted_threads: ['a-thread'],
      });
      writeSessionStateFile(tempDir, 'group_b', {
        active_session_id: 'sdk-b-001',
        pending_response: 'b-pending',
        muted_threads: ['b-thread-1', 'b-thread-2'],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      // Pre-stamp the existing singleton's schema_version to a non-
      // default value BEFORE the migration runs. We can't do this
      // directly because initDatabase performs the migration during
      // its own apply, so instead we stamp the column DEFAULT to a
      // value that the migration's UPSERT must preserve. Approach:
      // open the DB outside of initDatabase first, set a singleton
      // row with schema_version=99, then close + run initDatabase
      // (which still triggers the data migrations). But initDatabase
      // is the entrypoint that creates the schema, so we instead
      // verify the contract directly: after migration, both group
      // files have collapsed into a single id=1 row whose values are
      // the second writer's, and schema_version stays at the schema
      // DEFAULT(=1) because no prior row existed. Then we manually
      // bump schema_version=99 to mimic an owner-skill upgrade and
      // re-run migration via a third group: the row count must stay
      // at 1 AND schema_version must remain 99 (UPSERT, not REPLACE).
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const rows1 = db
            .prepare('SELECT COUNT(*) AS n FROM trusted_session_singleton')
            .get() as { n: number };
          expect(rows1.n).toBe(1);
          // Plain code-point sort puts 'group_a' before 'group_b', so
          // group_b is the second writer and its values land last.
          const singleton1 = db
            .prepare('SELECT * FROM trusted_session_singleton WHERE id = 1')
            .get() as Record<string, unknown>;
          expect(singleton1.active_session_id).toBe('sdk-b-001');
          expect(singleton1.schema_version).toBe(1);

          // Mimic an owner-skill upgrade that bumped schema_version.
          db.prepare(
            'UPDATE trusted_session_singleton SET schema_version = 99 WHERE id = 1',
          ).run();
        } finally {
          db.close();
        }

        // Now drop a third group's file and re-run the migration.
        // schema_version must stay at 99 — that's the UPSERT-not-
        // REPLACE assertion: REPLACE would delete the row and re-
        // insert with the schema column's DEFAULT(=1).
        writeSessionStateFile(tempDir, 'group_c', {
          active_session_id: 'sdk-c-001',
          pending_response: 'c-pending',
          muted_threads: ['c-thread'],
        });

        // Re-run the migration. initDatabase is idempotent and
        // re-scans every group folder.
        _closeDatabase();
        vi.resetModules();
        const reloaded = await import('./db.js');
        reloaded.initDatabase();

        const db2 = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const rows2 = db2
            .prepare('SELECT COUNT(*) AS n FROM trusted_session_singleton')
            .get() as { n: number };
          expect(rows2.n).toBe(1);
          const singleton2 = db2
            .prepare('SELECT * FROM trusted_session_singleton WHERE id = 1')
            .get() as Record<string, unknown>;
          // Latest writer's values land in the data columns.
          expect(singleton2.active_session_id).toBe('sdk-c-001');
          // schema_version must NOT be reset to 1 — that's the
          // REPLACE-vs-UPSERT distinction. ON CONFLICT DO UPDATE
          // touches only the named columns.
          expect(singleton2.schema_version).toBe(99);
        } finally {
          db2.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('JSON-stringifies pending_response and muted_threads so they round-trip through TEXT columns', async () => {
    // The schema declares `pending_response` and `muted_threads` as
    // TEXT. Per the state-006 doc-header, the owner skill treats them
    // as opaque JSON blobs. The migration must therefore JSON.stringify
    // structured shapes on the way in so the column round-trips back
    // through JSON.parse to the same object/array.
    await runWithTempDir(async (tempDir) => {
      const pendingResponse = {
        kind: 'multi',
        body: { headline: 'see PR', tags: ['urgent', 'review'] },
        attempts: 3,
      };
      const mutedThreads = ['t1', 't2', 't3'];
      writeSessionStateFile(tempDir, 'roundtrip', {
        active_session_id: 'sdk-rt-001',
        pending_response: pendingResponse,
        muted_threads: mutedThreads,
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const row = db
            .prepare(
              'SELECT pending_response, muted_threads FROM trusted_session_singleton WHERE id = 1',
            )
            .get() as { pending_response: string; muted_threads: string };
          expect(typeof row.pending_response).toBe('string');
          expect(typeof row.muted_threads).toBe('string');
          // Round-trip equality is the load-bearing assertion.
          expect(JSON.parse(row.pending_response)).toEqual(pendingResponse);
          expect(JSON.parse(row.muted_threads)).toEqual(mutedThreads);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('skips a session entry missing started/epoch/last_seen with a warn and continues', async () => {
    // `trusted_sessions.started` / `epoch` / `last_seen` are NOT NULL.
    // A JSON-era session entry that pre-dates one of those fields
    // would otherwise throw NOT NULL inside the per-file transaction
    // and roll back the whole import. Skip-with-warn so the remaining
    // session entries (and the singleton) still land.
    await runWithTempDir(async (tempDir) => {
      writeSessionStateFile(tempDir, 'partial', {
        sessions: {
          ok: {
            session_id: 'sdk-ok-001',
            started: '2026-04-29T08:00:00.000Z',
            epoch: 1714377600,
            last_seen: '2026-04-29T09:30:00.000Z',
          },
          // Missing `epoch` and `last_seen` — should be skipped.
          broken: {
            session_id: 'sdk-broken-001',
            started: '2026-04-29T08:00:00.000Z',
          },
        },
        active_session_id: 'sdk-ok-001',
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const sessions = db
            .prepare('SELECT * FROM trusted_sessions ORDER BY session_name')
            .all() as Array<Record<string, unknown>>;
          // Only the well-formed entry landed.
          expect(sessions).toHaveLength(1);
          expect(sessions[0].session_name).toBe('ok');
          // The singleton still imported alongside.
          const singleton = db
            .prepare('SELECT * FROM trusted_session_singleton WHERE id = 1')
            .get() as Record<string, unknown>;
          expect(singleton.active_session_id).toBe('sdk-ok-001');
        } finally {
          db.close();
        }
        const missingFieldsWarn = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(
            msg &&
            msg.includes('session entry missing required fields') &&
            msg.includes('started/epoch/last_seen'),
          );
        });
        expect(missingFieldsWarn).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('skips a non-object session entry with a warn and continues', async () => {
    // A stale `null` / number / string element in the `sessions` map
    // would TypeError on property access before any INSERT runs. Skip-
    // with-warn per the helper-driven contract.
    await runWithTempDir(async (tempDir) => {
      // Bypass the TS shape to write a non-object session value.
      writeSessionStateFile(tempDir, 'nonobj', {
        sessions: {
          ok: {
            session_id: 'sdk-ok-001',
            started: '2026-04-29T08:00:00.000Z',
            epoch: 1714377600,
            last_seen: '2026-04-29T09:30:00.000Z',
          },
          // Non-object value: a string rather than an entry object.
          legacy: 'just-a-session-id-string' as unknown as Record<
            string,
            unknown
          >,
        },
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const sessions = db
            .prepare('SELECT * FROM trusted_sessions')
            .all() as Array<Record<string, unknown>>;
          expect(sessions).toHaveLength(1);
          expect(sessions[0].session_name).toBe('ok');
        } finally {
          db.close();
        }
        const nonObjectWarn = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('non-object session entry'));
        });
        expect(nonObjectWarn).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it("does NOT migrate the JSON-era top-level seen_email_ids — that is #297's domain", async () => {
    // The state-006 doc-header is explicit: `seen_email_ids`
    // intentionally relocates to the `email_seen_ids` table created
    // by state-005 (#297) where both check-email writers can target
    // it without the old two-file consolidate dance. This data-import
    // pass must NOT touch `email_seen_ids`. #297's own data-import
    // PR owns that backfill.
    await runWithTempDir(async (tempDir) => {
      writeSessionStateFile(tempDir, 'admin', {
        sessions: {
          default: {
            session_id: 'sdk-default-001',
            started: '2026-04-29T08:00:00.000Z',
            epoch: 1714377600,
            last_seen: '2026-04-29T09:30:00.000Z',
          },
        },
        active_session_id: 'sdk-default-001',
        seen_email_ids: ['msg-001', 'msg-002', 'msg-003'],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // The schema-005 table exists but should still be empty after
          // this migration's run — the data import for that table is
          // owned by #297, not #298.
          const count = db
            .prepare('SELECT COUNT(*) AS n FROM email_seen_ids')
            .get() as { n: number };
          expect(count.n).toBe(0);
          // The session itself migrated.
          const sessions = db
            .prepare('SELECT * FROM trusted_sessions')
            .all() as Array<Record<string, unknown>>;
          expect(sessions).toHaveLength(1);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('is a no-op when the source file has already been renamed (re-run idempotent)', async () => {
    await runWithTempDir(async (tempDir) => {
      // Simulate a successful prior run: only the renamed file exists.
      const folder = path.join(tempDir, 'groups', 'idempotent');
      fs.mkdirSync(folder, { recursive: true });
      const renamedPath = path.join(
        folder,
        'session-state.json.migrated-2026-04-29',
      );
      fs.writeFileSync(
        renamedPath,
        JSON.stringify({
          sessions: {
            default: {
              session_id: 'sdk-already-001',
              started: '2026-04-29T08:00:00.000Z',
              epoch: 1714377600,
              last_seen: '2026-04-29T09:30:00.000Z',
            },
          },
          active_session_id: 'sdk-already-001',
        }),
      );

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      // Should not throw — the migration just doesn't see a source
      // file to consume, so both tables stay empty.
      expect(() => initDatabase()).not.toThrow();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const sessionsCount = (
            db.prepare('SELECT COUNT(*) AS n FROM trusted_sessions').get() as {
              n: number;
            }
          ).n;
          expect(sessionsCount).toBe(0);
          const singletonCount = (
            db
              .prepare('SELECT COUNT(*) AS n FROM trusted_session_singleton')
              .get() as { n: number }
          ).n;
          expect(singletonCount).toBe(0);
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

  it('skips a malformed JSON / non-object payload with a warn and leaves the file in place', async () => {
    // The shared helper `parseJsonObjectOrWarn` covers both shapes
    // (SyntaxError + non-object). Mix a good file with each bad shape
    // and assert the bad file is warned-and-skipped while the good
    // one still imports.
    for (const payload of [
      '{ this is not valid json',
      'null',
      '42',
      '[1, 2, 3]',
    ]) {
      await runWithTempDir(async (tempDir) => {
        const goodFile = writeSessionStateFile(tempDir, 'good', {
          sessions: {
            default: {
              session_id: 'sdk-good-001',
              started: '2026-04-29T08:00:00.000Z',
              epoch: 1714377600,
              last_seen: '2026-04-29T09:30:00.000Z',
            },
          },
          active_session_id: 'sdk-good-001',
        });
        const badFile = writeSessionStateFile(tempDir, 'bad', payload);

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
            const sessions = db
              .prepare('SELECT * FROM trusted_sessions')
              .all() as Array<Record<string, unknown>>;
            expect(sessions).toHaveLength(1);
            expect(sessions[0].session_name).toBe('default');
          } finally {
            db.close();
          }
          // Good file renamed; bad file left in place for triage.
          expect(fs.existsSync(goodFile)).toBe(false);
          expect(fs.existsSync(badFile)).toBe(true);
          // A warn for the bad shape — either invalid JSON or
          // non-object payload, depending on the case.
          const warnFired = warnSpy.mock.calls.some((call) => {
            const msg = call.find((arg) => typeof arg === 'string') as
              | string
              | undefined;
            return Boolean(
              msg &&
              (msg.includes('invalid JSON') ||
                msg.includes('payload is not an object')),
            );
          });
          expect(warnFired).toBe(true);
        } finally {
          warnSpy.mockRestore();
          _closeDatabase();
        }
      });
    }
  });
});
