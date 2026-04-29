# State-Table Migrations

Versioned schema migrations for the SQLite state tables that replace the
legacy JSON state files (epic #293). Run automatically at orchestrator
startup via `applyStateMigrations` in `src/db.ts`.

Tracked via SQLite's built-in `PRAGMA user_version`. Each migration
bumps `user_version` to its own number on success, inside the same
transaction as its DDL/DML — each individual migration is atomic.
A multi-migration upgrade is *not* one big transaction: if migration
N fails, earlier migrations stay applied and the next startup
re-runs only the pending tail (the version gate skips already-
applied entries). Per-migration atomicity matches the standard
discipline used by Alembic, Flyway, and Rails migrations.

## Adding a new migration

1. Add a new entry to the `STATE_MIGRATIONS` array in `index.ts`.
2. The entry's `version` must equal `index + 1` (contiguous from 1).
   `validateMigrationRegistry` rejects gaps and duplicates at startup.
3. `name` is a short human-readable label (e.g., `"orders table"`).
4. `sql` contains the DDL/DML to run. Use `CREATE TABLE` (not
   `CREATE TABLE IF NOT EXISTS`) so a re-applied migration on a
   corrupted state is loud rather than silent.
5. Add a unit test in `src/state-migrations.test.ts` exercising the
   new entry against a fresh in-memory DB.

## Why versions are contiguous

A migration registry with gaps (1, 2, 4 — skipping 3) is almost always
the result of two parallel branches both claiming version 3 and one
getting renumbered late. Forcing contiguity at the validator catches
that at startup instead of letting a `user_version=3` database silently
not run a different "3" written by a peer branch.

## Why distinct from `createSchema`

`createSchema` predates `PRAGMA user_version` discipline and uses a
blend of `CREATE TABLE IF NOT EXISTS` and PRAGMA-checked `ALTER TABLE`
calls. It's idempotent but not versioned — re-running it can't tell
"already migrated" from "fresh". New state tables (this directory) get
proper version tracking from day one. Existing tables in `createSchema`
are not retroactively migrated to this mechanism — too much risk for
no concrete win.

## Rollback safety

If `user_version` is HIGHER than the highest version in this registry,
`applyStateMigrations` throws at startup. This catches the
"rolled back to an older container" case where the database has a
schema the old code doesn't understand, which would otherwise silently
read/write tables in unexpected shapes.
