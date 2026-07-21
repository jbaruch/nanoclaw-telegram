import type Database from 'better-sqlite3';

// Single shared SQLite connection handle (#751). Ownership stays with
// `initDatabase` / `_initTestDatabase` in `db.ts` — they open the
// database and register the handle here via `setDbHandle`. Domain
// modules read the live `db` binding at call time, so there is exactly
// one open handle per process (WAL + busy_timeout are set by the
// opener, never here).
export let db: Database.Database;

export function setDbHandle(handle: Database.Database): void {
  db = handle;
}
