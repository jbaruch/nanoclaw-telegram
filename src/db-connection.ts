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

export function setDbHandle(handle: Database.Database): void {
  db = handle;
}
