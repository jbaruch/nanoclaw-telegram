import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';

// Each scenario builds an isolated tempDir + chdir so the
// CWD-rooted `STORE_DIR` and `GROUPS_DIR` constants in `src/config.ts`
// resolve under it. Mirrors the pattern in
// `src/morning-brief-pending-json-migration.test.ts`.

interface NanoclawStateJsonShape {
  last_email_checked?: unknown;
  date?: unknown;
  fetched_at?: unknown;
  seen_email_ids?: unknown;
  resumable_cycles?: unknown;
}

function writeNanoclawStateFile(
  tempDir: string,
  groupName: string,
  payload: NanoclawStateJsonShape,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'nanoclaw-state.json');
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  return filePath;
}

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-ns-'));
  try {
    process.chdir(tempDir);
    fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

describe('nanoclaw-state.json → SQLite migration (#297)', () => {
  it('imports email_state + email_seen_ids + resumable_cycles rows; renames source file', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeNanoclawStateFile(tempDir, 'telegram_main', {
        last_email_checked: '2026-04-29T10:00:00.000Z',
        date: '2026-04-29',
        fetched_at: '2026-04-29T10:00:05.000Z',
        seen_email_ids: ['msg-aaa', 'msg-bbb', 'msg-ccc'],
        resumable_cycles: {
          'tessl__nightly-housekeeping': {
            cycle_id: 'cycle-001',
            slot_key: 'nightly-2026-04-29',
            continuation_n: 2,
            remaining_steps: ['step-3', 'step-4'],
          },
          'tessl__weekly-housekeeping': {
            cycle_id: 'cycle-002',
            slot_key: 'weekly-2026-W17',
            continuation_n: 0,
            remaining_steps: null,
          },
        },
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // email_state singleton row
          const emailState = db
            .prepare('SELECT * FROM email_state WHERE id = 1')
            .get() as Record<string, unknown> | undefined;
          expect(emailState).toBeDefined();
          expect(emailState).toMatchObject({
            id: 1,
            last_email_checked: '2026-04-29T10:00:00.000Z',
            date: '2026-04-29',
            fetched_at: '2026-04-29T10:00:05.000Z',
          });
          // schema_version preserved at default (1) under UPSERT.
          expect(emailState!.schema_version).toBe(1);

          // email_seen_ids — one row per id
          const seen = db
            .prepare('SELECT email_id FROM email_seen_ids ORDER BY email_id')
            .all() as Array<{ email_id: string }>;
          expect(seen.map((r) => r.email_id)).toEqual([
            'msg-aaa',
            'msg-bbb',
            'msg-ccc',
          ]);

          // resumable_cycles — one row per skill
          const cycles = db
            .prepare('SELECT * FROM resumable_cycles ORDER BY skill_name')
            .all() as Array<Record<string, unknown>>;
          expect(cycles).toHaveLength(2);
          expect(cycles[0]).toMatchObject({
            skill_name: 'tessl__nightly-housekeeping',
            cycle_id: 'cycle-001',
            slot_key: 'nightly-2026-04-29',
            continuation_n: 2,
          });
          // remaining_steps stored as JSON string
          expect(JSON.parse(cycles[0].remaining_steps as string)).toEqual([
            'step-3',
            'step-4',
          ]);
          expect(cycles[1]).toMatchObject({
            skill_name: 'tessl__weekly-housekeeping',
            cycle_id: 'cycle-002',
            slot_key: 'weekly-2026-W17',
            continuation_n: 0,
            remaining_steps: null,
          });
        } finally {
          db.close();
        }

        // Source file renamed to .migrated-<YYYY-MM-DD>; the rename is
        // what makes a re-run a no-op (no version-gate guard for data
        // backfill — the schema is already at v5).
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('nanoclaw-state.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('UPSERTs email_state singleton across multiple groups WITHOUT clobbering schema_version', async () => {
    // Two groups' files — each contributes the same singleton row
    // (id=1). UPSERT (NOT INSERT OR REPLACE) preserves schema_version
    // across the second writer. INSERT OR REPLACE would delete and
    // re-insert, resetting schema_version to its DEFAULT (which would
    // be a no-op today since both default to 1, but a future bump
    // would silently undo the existing value). This test locks down
    // the contract.
    await runWithTempDir(async (tempDir) => {
      // Pre-stage state-005 schema and bump schema_version on the
      // singleton row before the importer runs, by writing a sentinel
      // value into the DB through a separate Database handle. The
      // importer's UPSERT must preserve this bump.
      // Create the DB and run the schema migrations once first — we
      // need state-005 in place to seed the row.
      vi.resetModules();
      const { initDatabase: initDatabasePre, _closeDatabase: closePre } =
        await import('./db.js');
      initDatabasePre();
      const dbPre = new Database(path.join(tempDir, 'store', 'messages.db'));
      try {
        dbPre
          .prepare(
            `INSERT INTO email_state (id, last_email_checked, schema_version)
             VALUES (1, 'pre-existing', 99)`,
          )
          .run();
      } finally {
        dbPre.close();
        closePre();
      }

      // Now drop two source files for two groups. The migrate runs
      // again on next initDatabase, sees the source files, and
      // UPSERTs both groups' rows.
      // Folders are sorted by code-point comparison so 'a-group'
      // imports before 'b-group'; b-group's cursor fields win.
      writeNanoclawStateFile(tempDir, 'a-group', {
        last_email_checked: '2026-04-28T08:00:00.000Z',
        seen_email_ids: ['a-1'],
      });
      writeNanoclawStateFile(tempDir, 'b-group', {
        last_email_checked: '2026-04-29T10:00:00.000Z',
        seen_email_ids: ['b-1'],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const row = db
            .prepare('SELECT * FROM email_state WHERE id = 1')
            .get() as Record<string, unknown>;
          // Last-writer-wins (sorted folder order) for the cursor field.
          expect(row.last_email_checked).toBe('2026-04-29T10:00:00.000Z');
          // schema_version preserved at the pre-existing 99 — UPSERT
          // does not touch this column, INSERT OR REPLACE would have
          // reset it to the schema default.
          expect(row.schema_version).toBe(99);

          // Both groups' seen_email_ids appended.
          const seen = db
            .prepare('SELECT email_id FROM email_seen_ids ORDER BY email_id')
            .all() as Array<{ email_id: string }>;
          expect(seen.map((r) => r.email_id)).toEqual(['a-1', 'b-1']);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('email_seen_ids ON CONFLICT(email_id) DO NOTHING — duplicate ids are silent no-ops on re-run', async () => {
    // Two groups with overlapping ids. The second group's file
    // contributes a duplicate that must NOT throw and must NOT bump
    // any existing row (we don't have a per-id timestamp from the
    // source so the DEFAULT seen_at fires once and stays put).
    await runWithTempDir(async (tempDir) => {
      writeNanoclawStateFile(tempDir, 'a-group', {
        seen_email_ids: ['shared-id', 'a-only'],
      });
      writeNanoclawStateFile(tempDir, 'b-group', {
        seen_email_ids: ['shared-id', 'b-only'],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const seen = db
            .prepare('SELECT email_id FROM email_seen_ids ORDER BY email_id')
            .all() as Array<{ email_id: string }>;
          // 3 distinct ids; the duplicate 'shared-id' was a no-op.
          expect(seen.map((r) => r.email_id)).toEqual([
            'a-only',
            'b-only',
            'shared-id',
          ]);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('warns for missing top-level keys but imports the present sections', async () => {
    // Source has only seen_email_ids — no email cursor fields, no
    // resumable_cycles. The migration should warn for the two missing
    // sections and still import the present one.
    await runWithTempDir(async (tempDir) => {
      const filePath = writeNanoclawStateFile(tempDir, 'partial-group', {
        seen_email_ids: ['only-id'],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // email_state has no row.
          const emailRow = db
            .prepare('SELECT COUNT(*) AS n FROM email_state')
            .get() as { n: number };
          expect(emailRow.n).toBe(0);
          // resumable_cycles empty.
          const cyclesRow = db
            .prepare('SELECT COUNT(*) AS n FROM resumable_cycles')
            .get() as { n: number };
          expect(cyclesRow.n).toBe(0);
          // seen_email_ids row present.
          const seen = db
            .prepare('SELECT email_id FROM email_seen_ids')
            .all() as Array<{ email_id: string }>;
          expect(seen.map((r) => r.email_id)).toEqual(['only-id']);
        } finally {
          db.close();
        }
        // File renamed even though only one section landed.
        expect(fs.existsSync(filePath)).toBe(false);

        const messages = warnSpy.mock.calls
          .map(
            (c) => c.find((a) => typeof a === 'string') as string | undefined,
          )
          .filter((m): m is string => m !== undefined);
        expect(messages.some((m) => m.includes('email cursor fields'))).toBe(
          true,
        );
        expect(
          messages.some((m) => m.includes('resumable_cycles missing')),
        ).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('skips non-string entries in seen_email_ids with a warn', async () => {
    // The email_id PK column is TEXT NOT NULL. A stale array element
    // that is null / number / object would either coerce silently
    // (number → text) or throw on the writer. The migration filters
    // non-string entries before binding so the dedup set keeps its
    // string-only contract.
    await runWithTempDir(async (tempDir) => {
      writeNanoclawStateFile(tempDir, 'mixed-group', {
        seen_email_ids: [
          'good-1',
          // Cast to bypass TS — that's the whole point of this test.
          null as unknown as string,
          42 as unknown as string,
          'good-2',
          { foo: 'bar' } as unknown as string,
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
          const seen = db
            .prepare('SELECT email_id FROM email_seen_ids ORDER BY email_id')
            .all() as Array<{ email_id: string }>;
          expect(seen.map((r) => r.email_id)).toEqual(['good-1', 'good-2']);
        } finally {
          db.close();
        }
        // Per-non-string-entry warn fired for each skipped row.
        const skippedWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(
            msg && msg.includes('skipping non-string entry in seen_email_ids'),
          );
        });
        expect(skippedWarnFired).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('rolls back and leaves the source file in place when a row violates a NOT NULL constraint', async () => {
    // resumable_cycles.cycle_id and .slot_key are NOT NULL on the
    // schema. A source file whose subtree omits cycle_id should
    // throw inside the per-file transaction; the transaction rolls
    // back so no partial rows land in any of the three tables; the
    // narrowed catch turns the throw into a warn; the source file
    // stays put for triage.
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeNanoclawStateFile(tempDir, 'a-good', {
        last_email_checked: '2026-04-29T10:00:00.000Z',
        seen_email_ids: ['good-1'],
      });
      const badFile = writeNanoclawStateFile(tempDir, 'b-bad', {
        last_email_checked: '2026-04-29T11:00:00.000Z',
        resumable_cycles: {
          'tessl__nightly-housekeeping': {
            // Missing cycle_id — NOT NULL violation expected.
            slot_key: 'nightly-2026-04-29',
            continuation_n: 0,
          },
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
          // Bad file's transaction rolled back: no resumable_cycles row,
          // no email_state row from b-bad either (transaction
          // atomicity). Because a-good already committed FIRST (sorted
          // folder order), email_state from a-good remains.
          const emailState = db
            .prepare('SELECT * FROM email_state WHERE id = 1')
            .get() as Record<string, unknown> | undefined;
          expect(emailState).toBeDefined();
          // a-good's value (b-bad rolled back).
          expect(emailState!.last_email_checked).toBe(
            '2026-04-29T10:00:00.000Z',
          );

          const cyclesCount = (
            db.prepare('SELECT COUNT(*) AS n FROM resumable_cycles').get() as {
              n: number;
            }
          ).n;
          expect(cyclesCount).toBe(0);

          // a-good's seen_id still landed.
          const seen = db
            .prepare('SELECT email_id FROM email_seen_ids')
            .all() as Array<{ email_id: string }>;
          expect(seen.map((r) => r.email_id)).toEqual(['good-1']);
        } finally {
          db.close();
        }
        // Good file renamed; bad file left in place.
        expect(fs.existsSync(goodFile)).toBe(false);
        expect(fs.existsSync(badFile)).toBe(true);
        // Per-file warn fired with the constraint-violation message
        // and SQLITE_CONSTRAINT_ errCode metadata.
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

  it('is a no-op when the source file has already been renamed (re-run idempotent)', async () => {
    await runWithTempDir(async (tempDir) => {
      // Simulate a successful prior run: only the renamed file exists.
      const folder = path.join(tempDir, 'groups', 'telegram_main');
      fs.mkdirSync(folder, { recursive: true });
      const renamedPath = path.join(
        folder,
        'nanoclaw-state.json.migrated-2026-04-29',
      );
      fs.writeFileSync(
        renamedPath,
        JSON.stringify({
          last_email_checked: 'already-imported',
          seen_email_ids: ['already-id'],
          resumable_cycles: {
            'tessl__nightly-housekeeping': {
              cycle_id: 'old-cycle',
              slot_key: 'old-slot',
            },
          },
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
            'email_state',
            'email_seen_ids',
            'resumable_cycles',
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

  it('skips a malformed JSON file with a warning and leaves it in place', async () => {
    // Mix one valid file and one malformed file. Malformed must NOT
    // abort the pass — other groups still need to migrate. Mirrors
    // the morning-brief test for the same case.
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeNanoclawStateFile(tempDir, 'a-good', {
        seen_email_ids: ['good-id'],
      });
      const badFolder = path.join(tempDir, 'groups', 'b-bad');
      fs.mkdirSync(badFolder, { recursive: true });
      const badFile = path.join(badFolder, 'nanoclaw-state.json');
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
            .prepare('SELECT email_id FROM email_seen_ids')
            .all() as Array<{ email_id: string }>;
          expect(ids.map((r) => r.email_id)).toEqual(['good-id']);
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

  it('skips a non-object JSON payload (null / array / number) with a warning and leaves it in place', async () => {
    // `JSON.parse` returns null / numbers / strings / arrays for
    // valid-but-wrong-shape payloads. The helper guards before
    // property access so one bad file doesn't abort the pass.
    for (const payload of ['null', '42', '[1, 2, 3]']) {
      await runWithTempDir(async (tempDir) => {
        const goodFile = writeNanoclawStateFile(tempDir, 'a-good', {
          seen_email_ids: [`good-${payload.replace(/\W/g, '')}`],
        });
        const badFolder = path.join(tempDir, 'groups', 'b-bad');
        fs.mkdirSync(badFolder, { recursive: true });
        const badFile = path.join(badFolder, 'nanoclaw-state.json');
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
              .prepare('SELECT email_id FROM email_seen_ids')
              .all() as Array<{ email_id: string }>;
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
