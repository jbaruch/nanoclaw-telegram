import Database, { SqliteError } from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi, afterEach } from 'vitest';

// Direct unit tests for the JSON-state-import helpers extracted in
// epic #293. The migration-level tests in
// `morning-brief-pending-json-migration.test.ts` and
// `orders-json-migration.test.ts` exercise the helpers transitively
// once the existing migrations get refactored to use them; these tests
// exercise each helper in isolation so a failure points at the helper,
// not at one specific migration.
//
// The file follows the same `runWithTempDir` + `process.chdir` pattern
// used by the existing migration tests so the CWD-rooted `GROUPS_DIR`
// constant in `src/config.ts` resolves under the temp dir.

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-jsi-'));
  try {
    process.chdir(tempDir);
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

/**
 * Trigger a real `SqliteError` with the given constraint code by
 * running a failing INSERT against an in-memory schema. Using the
 * native error constructor path (rather than `new SqliteError(...)`
 * directly) keeps the test honest about what better-sqlite3 actually
 * throws at runtime.
 */
type SqliteErrorInstance = InstanceType<typeof SqliteError>;

function captureSqliteError(
  setup: (db: Database.Database) => void,
  failingStatement: (db: Database.Database) => void,
): SqliteErrorInstance {
  const db = new Database(':memory:');
  try {
    setup(db);
    let caught: unknown;
    try {
      failingStatement(db);
      // eslint-disable-next-line no-catch-all/no-catch-all -- intentional capture; the assertion below rethrows non-SqliteError values.
    } catch (err) {
      caught = err;
    }
    if (!(caught instanceof SqliteError)) {
      throw new Error(
        `expected SqliteError from setup, got ${
          caught === undefined
            ? 'no throw'
            : caught instanceof Error
              ? caught.constructor.name
              : typeof caught
        }`,
      );
    }
    return caught as SqliteErrorInstance;
  } finally {
    db.close();
  }
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('isObjectRow', () => {
  it('returns true for plain objects', async () => {
    const { isObjectRow } = await import('./json-state-import.js');
    expect(isObjectRow({})).toBe(true);
    expect(isObjectRow({ id: 'x' })).toBe(true);
    expect(isObjectRow(Object.create(null))).toBe(true);
  });

  it('returns false for null, undefined, arrays, and primitives', async () => {
    const { isObjectRow } = await import('./json-state-import.js');
    expect(isObjectRow(null)).toBe(false);
    expect(isObjectRow(undefined)).toBe(false);
    expect(isObjectRow([])).toBe(false);
    expect(isObjectRow([1, 2, 3])).toBe(false);
    expect(isObjectRow('string')).toBe(false);
    expect(isObjectRow(42)).toBe(false);
    expect(isObjectRow(0)).toBe(false);
    expect(isObjectRow(true)).toBe(false);
    expect(isObjectRow(false)).toBe(false);
  });
});

describe('parseJsonObjectOrWarn', () => {
  it('returns the parsed object on a valid object payload', async () => {
    const { parseJsonObjectOrWarn } = await import('./json-state-import.js');
    const { logger } = await import('./logger.js');
    const warnSpy = vi.spyOn(logger, 'warn');
    const out = parseJsonObjectOrWarn(
      JSON.stringify({ id: 'x', value: 1 }),
      'group_a',
      'fake-state.json',
    );
    expect(out).toEqual({ id: 'x', value: 1 });
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('returns null and warns with parsedType="null" when payload is null', async () => {
    const { parseJsonObjectOrWarn } = await import('./json-state-import.js');
    const { logger } = await import('./logger.js');
    const warnSpy = vi.spyOn(logger, 'warn');
    const out = parseJsonObjectOrWarn('null', 'group_a', 'fake-state.json');
    expect(out).toBeNull();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const [meta, msg] = warnSpy.mock.calls[0] as [
      Record<string, unknown>,
      string,
    ];
    expect(meta).toMatchObject({ folder: 'group_a', parsedType: 'null' });
    expect(msg).toContain('payload is not an object');
  });

  it('returns null and warns with parsedType="number" when payload is a number', async () => {
    const { parseJsonObjectOrWarn } = await import('./json-state-import.js');
    const { logger } = await import('./logger.js');
    const warnSpy = vi.spyOn(logger, 'warn');
    const out = parseJsonObjectOrWarn('42', 'group_a', 'fake-state.json');
    expect(out).toBeNull();
    expect(warnSpy.mock.calls[0]?.[0]).toMatchObject({ parsedType: 'number' });
  });

  it('returns null and warns with parsedType="string" when payload is a string', async () => {
    const { parseJsonObjectOrWarn } = await import('./json-state-import.js');
    const { logger } = await import('./logger.js');
    const warnSpy = vi.spyOn(logger, 'warn');
    const out = parseJsonObjectOrWarn('"hello"', 'group_a', 'fake-state.json');
    expect(out).toBeNull();
    expect(warnSpy.mock.calls[0]?.[0]).toMatchObject({ parsedType: 'string' });
  });

  it('returns null and warns with parsedType="array" (NOT "object") when payload is an array', async () => {
    // The whole point of this distinction: `typeof []` is `'object'`,
    // which used to mask arrays in triage greps. The helper must log
    // `'array'` so an operator searching for `parsedType: array`
    // finds these cases.
    const { parseJsonObjectOrWarn } = await import('./json-state-import.js');
    const { logger } = await import('./logger.js');
    const warnSpy = vi.spyOn(logger, 'warn');
    const out = parseJsonObjectOrWarn(
      '[1, 2, 3]',
      'group_a',
      'fake-state.json',
    );
    expect(out).toBeNull();
    expect(warnSpy.mock.calls[0]?.[0]).toMatchObject({ parsedType: 'array' });
  });

  it('returns null and warns with errName when JSON is malformed', async () => {
    const { parseJsonObjectOrWarn } = await import('./json-state-import.js');
    const { logger } = await import('./logger.js');
    const warnSpy = vi.spyOn(logger, 'warn');
    const out = parseJsonObjectOrWarn(
      '{ not valid json',
      'group_a',
      'fake-state.json',
    );
    expect(out).toBeNull();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const [meta, msg] = warnSpy.mock.calls[0] as [
      Record<string, unknown>,
      string,
    ];
    expect(meta).toMatchObject({ folder: 'group_a', errName: 'SyntaxError' });
    expect(msg).toContain('invalid JSON');
  });

  it('uses the supplied fileLabel verbatim in warn messages', async () => {
    const { parseJsonObjectOrWarn } = await import('./json-state-import.js');
    const { logger } = await import('./logger.js');
    const warnSpy = vi.spyOn(logger, 'warn');
    parseJsonObjectOrWarn('null', 'group_a', 'orders-db.json');
    parseJsonObjectOrWarn('{ bad', 'group_b', 'task-tz-state.json');
    const messages = warnSpy.mock.calls.map((call) => call[1] as string);
    expect(messages[0]).toContain('orders-db.json');
    expect(messages[1]).toContain('task-tz-state.json');
  });
});

describe('handleConstraintViolationOrRethrow', () => {
  it('returns true and warns with errCode for constraint-class SqliteError', async () => {
    const { handleConstraintViolationOrRethrow } =
      await import('./json-state-import.js');
    const { logger } = await import('./logger.js');
    const warnSpy = vi.spyOn(logger, 'warn');
    // Real SQLITE_CONSTRAINT_NOTNULL: schema with NOT NULL `name`,
    // INSERT only the `id`.
    const err = captureSqliteError(
      (db) => {
        db.exec('CREATE TABLE t (id TEXT PRIMARY KEY, name TEXT NOT NULL)');
      },
      (db) => {
        db.prepare('INSERT INTO t (id) VALUES (?)').run('x');
      },
    );
    expect(err.code).toBe('SQLITE_CONSTRAINT_NOTNULL');
    const result = handleConstraintViolationOrRethrow(
      err,
      'group_a',
      'fake-state.json',
    );
    expect(result).toBe(true);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const [meta, msg] = warnSpy.mock.calls[0] as [
      Record<string, unknown>,
      string,
    ];
    expect(meta).toMatchObject({
      folder: 'group_a',
      errCode: 'SQLITE_CONSTRAINT_NOTNULL',
    });
    expect(msg).toContain('violated a DB constraint');
  });

  it('returns true for SQLITE_CONSTRAINT_UNIQUE as well', async () => {
    const { handleConstraintViolationOrRethrow } =
      await import('./json-state-import.js');
    const err = captureSqliteError(
      (db) => {
        db.exec('CREATE TABLE t (id TEXT PRIMARY KEY)');
        db.prepare('INSERT INTO t (id) VALUES (?)').run('dup');
      },
      (db) => {
        db.prepare('INSERT INTO t (id) VALUES (?)').run('dup');
      },
    );
    // Either UNIQUE or PRIMARYKEY depending on the SQLite version —
    // both start with SQLITE_CONSTRAINT_, which is what the helper
    // checks.
    expect(err.code).toMatch(/^SQLITE_CONSTRAINT_/);
    const result = handleConstraintViolationOrRethrow(
      err,
      'group_a',
      'fake-state.json',
    );
    expect(result).toBe(true);
  });

  it('rethrows SqliteError instances whose code does NOT start with SQLITE_CONSTRAINT_', async () => {
    const { handleConstraintViolationOrRethrow } =
      await import('./json-state-import.js');
    // Synthesize a non-constraint SqliteError. We can't easily
    // provoke SQLITE_BUSY in a unit test, but the helper checks the
    // code prefix only — so we instantiate a SqliteError with a
    // non-constraint code and assert the rethrow.
    const fakeBusy = new SqliteError('database is locked', 'SQLITE_BUSY');
    expect(fakeBusy).toBeInstanceOf(SqliteError);
    expect(() =>
      handleConstraintViolationOrRethrow(
        fakeBusy,
        'group_a',
        'fake-state.json',
      ),
    ).toThrow(fakeBusy);
  });

  it('rethrows non-SqliteError values (TypeError, plain Error, string)', async () => {
    const { handleConstraintViolationOrRethrow } =
      await import('./json-state-import.js');
    const typeErr = new TypeError(
      "Cannot read properties of null (reading 'x')",
    );
    expect(() =>
      handleConstraintViolationOrRethrow(typeErr, 'g', 'f.json'),
    ).toThrow(typeErr);

    const plain = new Error('boom');
    expect(() =>
      handleConstraintViolationOrRethrow(plain, 'g', 'f.json'),
    ).toThrow(plain);

    // Non-Error values (a thrown string) — the narrowed catch must
    // still rethrow them, not swallow.
    expect(() =>
      handleConstraintViolationOrRethrow(
        'unexpected string throw',
        'g',
        'f.json',
      ),
    ).toThrow('unexpected string throw');
  });
});

describe('listGroupFoldersForMigration', () => {
  it('returns valid group folders sorted by code-point comparison', async () => {
    await runWithTempDir(async (tempDir) => {
      const groupsDir = path.join(tempDir, 'groups');
      fs.mkdirSync(groupsDir, { recursive: true });
      // Intentionally creating in non-alphabetical order — the helper
      // must sort, not preserve readdir order.
      fs.mkdirSync(path.join(groupsDir, 'zebra'));
      fs.mkdirSync(path.join(groupsDir, 'alpha'));
      fs.mkdirSync(path.join(groupsDir, 'mango'));
      // A non-directory entry that should be filtered out.
      fs.writeFileSync(path.join(groupsDir, 'README.md'), 'not a folder');
      // A directory whose name fails `isValidGroupFolder` (reserved).
      fs.mkdirSync(path.join(groupsDir, 'global'));
      // A directory with an invalid character.
      fs.mkdirSync(path.join(groupsDir, 'has space'));

      vi.resetModules();
      const { listGroupFoldersForMigration } =
        await import('./json-state-import.js');
      const folders = listGroupFoldersForMigration();
      expect(folders).toEqual(['alpha', 'mango', 'zebra']);
    });
  });

  it('returns [] when GROUPS_DIR does not exist (first-boot ENOENT)', async () => {
    await runWithTempDir(async () => {
      // No `groups/` dir under tempDir — fresh first-boot scenario.
      vi.resetModules();
      const { listGroupFoldersForMigration } =
        await import('./json-state-import.js');
      expect(listGroupFoldersForMigration()).toEqual([]);
    });
  });

  it('filters out non-directory entries even when they have valid-folder names', async () => {
    await runWithTempDir(async (tempDir) => {
      const groupsDir = path.join(tempDir, 'groups');
      fs.mkdirSync(groupsDir, { recursive: true });
      // A file (not a directory) with an otherwise-valid folder name.
      fs.writeFileSync(path.join(groupsDir, 'looks_valid'), '');
      fs.mkdirSync(path.join(groupsDir, 'real_folder'));

      vi.resetModules();
      const { listGroupFoldersForMigration } =
        await import('./json-state-import.js');
      expect(listGroupFoldersForMigration()).toEqual(['real_folder']);
    });
  });

  // it.skipIf so a Windows or root run shows up as SKIPPED in the
  // runner output, not as silently passing — that distinction is what
  // protects the contract from regressing unnoticed in CI matrices.
  // (process.getuid is undefined on Windows; the optional chain
  // returns undefined there, which `?? -1` falls through to a
  // non-zero, so the Windows check below never depends on getuid.)
  const isWindows = process.platform === 'win32';
  const isRoot = (process.getuid?.() ?? -1) === 0;
  const cannotChmod = isWindows || isRoot;
  it.skipIf(cannotChmod)(
    'propagates non-ENOENT errors (EACCES, etc.) instead of silently treating them as "first boot"',
    async () => {
      // The contract is "ENOENT → [] (first boot), every other errno
      // → propagate". This test creates a real `groups/` directory
      // with mode 000 so readdirSync hits EACCES at the OS level —
      // exercises the actual catch path, not a mock.
      await runWithTempDir(async (tempDir) => {
        const groupsDir = path.join(tempDir, 'groups');
        fs.mkdirSync(groupsDir);
        fs.chmodSync(groupsDir, 0o000);
        try {
          vi.resetModules();
          const { listGroupFoldersForMigration } =
            await import('./json-state-import.js');
          expect(() => listGroupFoldersForMigration()).toThrow(
            /EACCES|permission/i,
          );
        } finally {
          // Restore permissions so runWithTempDir's rm succeeds.
          fs.chmodSync(groupsDir, 0o755);
        }
      });
    },
  );
});

describe('renameMigratedSource', () => {
  it('renames the source file and logs info with renamed_to + extra context', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = path.join(tempDir, 'state.json');
      fs.writeFileSync(filePath, '{}');
      vi.resetModules();
      const { renameMigratedSource } = await import('./json-state-import.js');
      const { logger } = await import('./logger.js');
      const infoSpy = vi.spyOn(logger, 'info');

      renameMigratedSource(filePath, '2026-04-30', 'group_a', 'state.json', {
        inserted: 5,
        skipped: 0,
      });

      const expectedRenamed = `${filePath}.migrated-2026-04-30`;
      expect(fs.existsSync(filePath)).toBe(false);
      expect(fs.existsSync(expectedRenamed)).toBe(true);
      expect(infoSpy).toHaveBeenCalledTimes(1);
      const [meta, msg] = infoSpy.mock.calls[0] as [
        Record<string, unknown>,
        string,
      ];
      expect(meta).toMatchObject({
        folder: 'group_a',
        inserted: 5,
        skipped: 0,
        renamed_to: expectedRenamed,
      });
      expect(msg).toContain('imported and source renamed');
    });
  });

  it('logs an "already absent" info on ENOENT and does not throw', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = path.join(tempDir, 'state.json');
      // Source does NOT exist — simulate the TOCTOU race where the
      // file was removed between import-commit and rename.
      vi.resetModules();
      const { renameMigratedSource } = await import('./json-state-import.js');
      const { logger } = await import('./logger.js');
      const infoSpy = vi.spyOn(logger, 'info');

      expect(() =>
        renameMigratedSource(filePath, '2026-04-30', 'group_a', 'state.json', {
          inserted: 5,
        }),
      ).not.toThrow();

      expect(infoSpy).toHaveBeenCalledTimes(1);
      const [meta, msg] = infoSpy.mock.calls[0] as [
        Record<string, unknown>,
        string,
      ];
      expect(meta).toMatchObject({ folder: 'group_a', inserted: 5 });
      // The "already absent" branch deliberately omits renamed_to —
      // no rename actually happened.
      expect(meta).not.toHaveProperty('renamed_to');
      expect(msg).toContain('already absent at rename time');
    });
  });

  it('propagates non-ENOENT errnos (e.g. EACCES)', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = path.join(tempDir, 'state.json');
      fs.writeFileSync(filePath, '{}');
      vi.resetModules();
      const { renameMigratedSource } = await import('./json-state-import.js');

      // Mock fs.renameSync to throw EACCES — the helper must rethrow.
      const eacces = Object.assign(new Error('permission denied'), {
        code: 'EACCES',
      }) as NodeJS.ErrnoException;
      const renameSpy = vi.spyOn(fs, 'renameSync').mockImplementation(() => {
        throw eacces;
      });
      try {
        expect(() =>
          renameMigratedSource(filePath, '2026-04-30', 'group_a', 'state.json'),
        ).toThrow(eacces);
      } finally {
        renameSpy.mockRestore();
      }
    });
  });

  it('omits importLogContext gracefully when not supplied', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = path.join(tempDir, 'state.json');
      fs.writeFileSync(filePath, '{}');
      vi.resetModules();
      const { renameMigratedSource } = await import('./json-state-import.js');
      const { logger } = await import('./logger.js');
      const infoSpy = vi.spyOn(logger, 'info');

      renameMigratedSource(filePath, '2026-04-30', 'group_a', 'state.json');

      expect(infoSpy).toHaveBeenCalledTimes(1);
      const [meta] = infoSpy.mock.calls[0] as [Record<string, unknown>];
      expect(meta.folder).toBe('group_a');
      expect(meta.renamed_to).toBe(`${filePath}.migrated-2026-04-30`);
    });
  });
});

describe('migrationDateStamp', () => {
  it('returns a 10-character YYYY-MM-DD string', async () => {
    const { migrationDateStamp } = await import('./json-state-import.js');
    const stamp = migrationDateStamp();
    expect(stamp).toHaveLength(10);
    expect(stamp).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    // Should match today's UTC date (the body of new Date().toISOString()).
    const expected = new Date().toISOString().slice(0, 10);
    expect(stamp).toBe(expected);
  });
});

describe('hasMigratedSibling (#433)', () => {
  it('returns true when a `<file>.migrated-YYYY-MM-DD` sibling exists', async () => {
    const { hasMigratedSibling } = await import('./json-state-import.js');
    await runWithTempDir(async (tempDir) => {
      const dir = path.join(tempDir, 'group');
      fs.mkdirSync(dir);
      fs.writeFileSync(path.join(dir, 'state.json.migrated-2026-04-30'), '{}');
      expect(hasMigratedSibling(path.join(dir, 'state.json'))).toBe(true);
    });
  });

  it('returns false when only the live source file exists', async () => {
    const { hasMigratedSibling } = await import('./json-state-import.js');
    await runWithTempDir(async (tempDir) => {
      const dir = path.join(tempDir, 'group');
      fs.mkdirSync(dir);
      fs.writeFileSync(path.join(dir, 'state.json'), '{}');
      expect(hasMigratedSibling(path.join(dir, 'state.json'))).toBe(false);
    });
  });

  it('returns false when the parent directory does not exist', async () => {
    const { hasMigratedSibling } = await import('./json-state-import.js');
    await runWithTempDir(async (tempDir) => {
      expect(
        hasMigratedSibling(path.join(tempDir, 'no-such-dir', 'state.json')),
      ).toBe(false);
    });
  });

  it('does not match unrelated files that share a prefix', async () => {
    // `state.json` should NOT match `state.json.migrated-...` siblings of
    // a sibling file `state-other.json` — the prefix check is anchored
    // against the exact base name.
    const { hasMigratedSibling } = await import('./json-state-import.js');
    await runWithTempDir(async (tempDir) => {
      const dir = path.join(tempDir, 'group');
      fs.mkdirSync(dir);
      fs.writeFileSync(
        path.join(dir, 'state-other.json.migrated-2026-04-30'),
        '{}',
      );
      expect(hasMigratedSibling(path.join(dir, 'state.json'))).toBe(false);
    });
  });
});

describe('newMigrationSummary + counter threading (#433)', () => {
  it('newMigrationSummary returns zeroed counters with the given name', async () => {
    const { newMigrationSummary } = await import('./json-state-import.js');
    const s = newMigrationSummary('orders-db');
    expect(s.name).toBe('orders-db');
    expect(s.migrated).toBe(0);
    expect(s.skippedAlreadyDone).toBe(0);
    expect(s.leftInPlace).toEqual([]);
  });

  it('parseJsonObjectOrWarn pushes folder to leftInPlace on bad shape', async () => {
    const { parseJsonObjectOrWarn, newMigrationSummary } =
      await import('./json-state-import.js');
    const s = newMigrationSummary('test');
    parseJsonObjectOrWarn('null', 'group_a', 'state.json', s);
    parseJsonObjectOrWarn('not json', 'group_b', 'state.json', s);
    expect(s.leftInPlace).toEqual(['group_a', 'group_b']);
    expect(s.migrated).toBe(0);
  });

  it('parseJsonObjectOrWarn dedups the same folder across calls', async () => {
    // Defensive against future migrations that pump multiple records
    // through the helper for one folder.
    const { parseJsonObjectOrWarn, newMigrationSummary } =
      await import('./json-state-import.js');
    const s = newMigrationSummary('test');
    parseJsonObjectOrWarn('null', 'group_a', 'state.json', s);
    parseJsonObjectOrWarn('null', 'group_a', 'state.json', s);
    expect(s.leftInPlace).toEqual(['group_a']);
  });

  it('parseJsonObjectOrWarn does not bump anything on success', async () => {
    const { parseJsonObjectOrWarn, newMigrationSummary } =
      await import('./json-state-import.js');
    const s = newMigrationSummary('test');
    parseJsonObjectOrWarn(JSON.stringify({ a: 1 }), 'group_a', 'state.json', s);
    expect(s.leftInPlace).toEqual([]);
    expect(s.migrated).toBe(0);
  });

  it('handleConstraintViolationOrRethrow pushes folder to leftInPlace', async () => {
    const { handleConstraintViolationOrRethrow, newMigrationSummary } =
      await import('./json-state-import.js');
    const err = captureSqliteError(
      (db) => db.exec('CREATE TABLE t (id TEXT NOT NULL)'),
      (db) => db.prepare('INSERT INTO t VALUES (NULL)').run(),
    );
    const s = newMigrationSummary('test');
    expect(
      handleConstraintViolationOrRethrow(err, 'group_x', 'state.json', s),
    ).toBe(true);
    expect(s.leftInPlace).toEqual(['group_x']);
  });

  it('renameMigratedSource increments migrated on the rename success path', async () => {
    const { renameMigratedSource, newMigrationSummary } =
      await import('./json-state-import.js');
    await runWithTempDir(async (tempDir) => {
      const dir = path.join(tempDir, 'group_a');
      fs.mkdirSync(dir);
      const src = path.join(dir, 'state.json');
      fs.writeFileSync(src, '{}');
      const s = newMigrationSummary('test');
      renameMigratedSource(src, '2026-05-04', 'group_a', 'state.json', {}, s);
      expect(s.migrated).toBe(1);
      expect(fs.existsSync(`${src}.migrated-2026-05-04`)).toBe(true);
    });
  });

  it('renameMigratedSource increments migrated on ENOENT idempotent-no-op', async () => {
    // The data already landed in SQL; the file just vanished between
    // import and rename — count it as a successful migrated-this-boot.
    const { renameMigratedSource, newMigrationSummary } =
      await import('./json-state-import.js');
    await runWithTempDir(async (tempDir) => {
      const s = newMigrationSummary('test');
      renameMigratedSource(
        path.join(tempDir, 'no-such-file.json'),
        '2026-05-04',
        'group_a',
        'state.json',
        {},
        s,
      );
      expect(s.migrated).toBe(1);
    });
  });

  it('helpers are no-ops on summary when summary is undefined', async () => {
    // Backward compat: callers that haven't been updated to thread a
    // summary through must still work.
    const { parseJsonObjectOrWarn } = await import('./json-state-import.js');
    expect(() =>
      parseJsonObjectOrWarn('null', 'group_a', 'state.json'),
    ).not.toThrow();
  });
});
