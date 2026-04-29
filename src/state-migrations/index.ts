import type { StateMigration } from '../db.js';

/**
 * Registered state-table migrations, applied in order at orchestrator
 * startup. Versions must be contiguous integers starting at 1 — see
 * README.md for the full convention.
 *
 * Invariants enforced by `validateMigrationRegistry` in db.ts:
 *   - Each entry's `version` equals its array index + 1
 *   - `name` is a non-empty human-readable label
 *   - `sql` is a non-empty SQL string (DDL/DML)
 *
 * Empty array is valid (no-op at startup). Real migrations land in
 * follow-ups to epic #293 — orders (#294), email-feedback (#295), etc.
 */
export const STATE_MIGRATIONS: readonly StateMigration[] = [];
