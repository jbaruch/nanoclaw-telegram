import type Database from 'better-sqlite3';

// Single shared SQLite connection handle (#751). Ownership stays with
// `initDatabase` / `_initTestDatabase` in `db.ts` — they open the
// database and register the handle here via `setDbHandle`. Domain
// modules read the live `db` binding at call time, so there is exactly
// one open handle per process (WAL + busy_timeout are set by the
// opener, never here).
//
// The initial value is a sentinel that throws an actionable error on
// any property access, so touching the DB before `initDatabase()` /
// `_initTestDatabase()` fails with a named cause instead of a cryptic
// "Cannot read properties of undefined" TypeError.
export let db: Database.Database = new Proxy({} as Database.Database, {
  get(_target, prop) {
    throw new Error(
      `DB handle accessed (property "${String(prop)}") before it was registered — ` +
        `call initDatabase() (or _initTestDatabase() in tests) before touching the database.`,
    );
  },
});

let dbHandleRegistered = false;

export function setDbHandle(handle: Database.Database): void {
  db = handle;
  dbHandleRegistered = true;
}

// True once `setDbHandle` has run. The sentinel Proxy above is always
// truthy, so callers that want a pre-init guard cannot test `db`
// itself — they ask this instead (e.g. `rebuildCadenceRegistryForGroup`
// in `db-tasks.ts`, which throws its own function-specific error).
// `_closeDatabase` closes the handle without unregistering it; a
// closed-but-registered handle fails on use with better-sqlite3's own
// "database connection is not open", which is out of scope here.
export function isDbHandleRegistered(): boolean {
  return dbHandleRegistered;
}
