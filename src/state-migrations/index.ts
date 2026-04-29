import type { StateMigration } from '../db.js';

import { STATE_001_ORDERS } from './state-001-orders.js';

/**
 * Registered state-table migrations, applied in order at orchestrator
 * startup. Versions must be contiguous integers starting at 1 — see
 * README.md for the full convention.
 *
 * Invariants enforced by `validateMigrationRegistry` in db.ts:
 *   - Each entry's `version` equals its array index + 1
 *   - `name` is a non-empty human-readable label
 *   - `sql` is a non-empty SQL string (DDL/DML)
 */
export const STATE_MIGRATIONS: readonly StateMigration[] = [STATE_001_ORDERS];
