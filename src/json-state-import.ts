import fs from 'fs';
import path from 'path';
import { SqliteError } from 'better-sqlite3';

import { GROUPS_DIR } from './config.js';
import { isValidGroupFolder } from './group-folder.js';
import { logger } from './logger.js';

/**
 * Per-migration counter object threaded through the shared helpers
 * (#433). Each per-group / per-file migration constructs one of these
 * at entry, passes it to `parseJsonObjectOrWarn` /
 * `handleConstraintViolationOrRethrow` / `renameMigratedSource`, and
 * returns it. `migrateJsonState()` collects the array of summaries and
 * emits one startup-summary log line per migration so the operator
 * has a single grep target for "did the data plane come up clean".
 *
 * - `migrated` is incremented by `renameMigratedSource` on the success
 *   path (including the ENOENT-at-rename idempotent-no-op case where
 *   the data did land but the file was already gone).
 * - `leftInPlace` is a deduped list of group folders where the file
 *   was present but the migration skipped it, leaving the source on
 *   disk un-renamed so the next boot will retry. The skip can be from
 *   `parseJsonObjectOrWarn` returning null (bad JSON shape), from
 *   `handleConstraintViolationOrRethrow` swallowing a DB constraint
 *   violation, OR from any inline guard in a migration that detects a
 *   missing required envelope field / unexpected sub-shape and
 *   bumps the counter directly. **This is the load-bearing signal
 *   #433 surfaces** — the case in #431 sat live for ~3 days because
 *   the only signal was the absence of a `.migrated-*` rename.
 * - `skippedAlreadyDone` is the count of group folders where the
 *   source file is gone but a `.migrated-*` sibling exists (counted
 *   via `hasMigratedSibling`). Lets the operator answer "is this
 *   migration done across the fleet" from one log line.
 */
export interface MigrationSummary {
  name: string;
  migrated: number;
  leftInPlace: string[];
  skippedAlreadyDone: number;
}

export function newMigrationSummary(name: string): MigrationSummary {
  return { name, migrated: 0, leftInPlace: [], skippedAlreadyDone: 0 };
}

function recordLeftInPlace(
  summary: MigrationSummary | undefined,
  folder: string,
): void {
  if (!summary) return;
  // Dedup: parseJsonObjectOrWarn and handleConstraintViolationOrRethrow
  // don't both fire for the same file in current call patterns (parse
  // runs before any DB op), but a future migration that calls both
  // around different rows in one folder shouldn't double-count.
  if (!summary.leftInPlace.includes(folder)) summary.leftInPlace.push(folder);
}

/**
 * True if `<filePath>.migrated-*` exists in the parent directory of
 * `filePath`. Lets the per-group migrations count "already done" runs
 * without re-importing data.
 *
 * Catches only the specific filesystem errnos that legitimately mean
 * "this folder isn't accessible from here" (parent gone, permission
 * denied, not a directory) and treats them as "no sibling" — the
 * skipped-already-done count is informational and shouldn't crash
 * the migration on a permission edge case. Any other errno is a
 * programming bug or environment problem the operator must see, so
 * propagate per `coding-policy: error-handling.Specific Exceptions`.
 */
const MIGRATED_SIBLING_RECOVERABLE_CODES = new Set([
  'ENOENT',
  'ENOTDIR',
  'EACCES',
  'EPERM',
]);

export function hasMigratedSibling(filePath: string): boolean {
  const dir = path.dirname(filePath);
  const base = path.basename(filePath);
  try {
    return fs
      .readdirSync(dir)
      .some((name) => name.startsWith(`${base}.migrated-`));
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code && MIGRATED_SIBLING_RECOVERABLE_CODES.has(code)) return false;
    throw err;
  }
}

// Shared helpers for per-group JSON-state-import migrations (epic #293).
//
// Consolidates the lessons from the four-iteration review history of
// PR #352 (morning-brief-pending data import) so the remaining wave-1
// and wave-2 PRs don't have to re-litigate the same concerns:
//
// 1. `INSERT OR IGNORE` silently swallows NOT NULL / CHECK violations
//    on top of PK conflicts — use `ON CONFLICT(<id>) DO NOTHING` so
//    only the named conflict is suppressed and everything else
//    surfaces. (No helper here: prepared statements are caller-owned;
//    the helpers below assume the caller already followed this rule.)
// 2. `JSON.parse` returns null / numbers / strings / arrays for valid-
//    but-wrong-shape payloads. Property access on those throws
//    "cannot read properties of null"; need an explicit object-shape
//    guard before binding. → `parseJsonObjectOrWarn`.
// 3. `parsedType: typeof parsed` logs `'object'` for arrays. Distinguish
//    the array case so triage greps match the actual shape.
//    → folded into `parseJsonObjectOrWarn`.
// 4. Bare catch-alls around the per-file transaction violate
//    `coding-policy: error-handling`. Narrow to `SqliteError` whose
//    `code` starts with `'SQLITE_CONSTRAINT_'` — those are the
//    recoverable data-quality failures the per-file isolation contract
//    was written for. → `handleConstraintViolationOrRethrow`.
// 5. Per-row object guards inside the transaction prevent stale
//    null/string/number elements from TypeError-ing before any INSERT
//    runs and being propagated by the narrowed catch.
//    → `isObjectRow`.
// 6. Deterministic group-folder sort (plain code-point comparison) so
//    ON CONFLICT resolution is reproducible across filesystems and
//    locales. → `listGroupFoldersForMigration`.
// 7. ENOENT on rename is the only recoverable errno. → `renameMigratedSource`.

/**
 * Type-guard: true only for plain objects (not null, not arrays, not
 * primitives). Use inside per-file transactions before any property
 * access on a row pulled from a JSON array — once the per-file catch
 * is narrowed to `SqliteError` (per #4 above), a TypeError on
 * `null.someField` would propagate as "unexpected" and halt
 * orchestrator startup. Guard the row first; warn-and-skip on miss.
 */
export function isObjectRow(row: unknown): row is Record<string, unknown> {
  return row !== null && typeof row === 'object' && !Array.isArray(row);
}

/**
 * Parse a JSON string and return the parsed value only when it is a
 * plain object. Returns `null` and emits a warn on:
 *
 * - SyntaxError (malformed JSON) — warn message includes `errName`
 *   so triage can distinguish it from the non-object case.
 * - Valid JSON whose top-level value is null / number / string /
 *   boolean / array — warn message mentions "payload is not an
 *   object" and includes `parsedType` distinguishing 'null' / 'array'
 *   from `typeof parsed` (so an array logs `'array'`, not `'object'`).
 *
 * Does NOT catch ENOENT. The TOCTOU race between the caller's
 * `fs.existsSync` and `fs.readFileSync` is handled at the caller —
 * existing migrations (e.g. orders) treat that ENOENT as a
 * recoverable, info-logged no-op (the file vanished between check
 * and read; nothing to migrate). Keeping that handling at the call
 * site lets the caller distinguish ENOENT (info: idempotent skip)
 * from "valid file but bad shape" (warn: triage).
 */
export function parseJsonObjectOrWarn(
  raw: string,
  folder: string,
  fileLabel: string,
  summary?: MigrationSummary,
): Record<string, unknown> | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    if (err instanceof SyntaxError) {
      logger.warn(
        { folder, errName: err.name },
        `${fileLabel} migration: invalid JSON, skipping (file left in place)`,
      );
      recordLeftInPlace(summary, folder);
      return null;
    }
    // JSON.parse on a string only throws SyntaxError; anything else is
    // a programming bug (e.g. raw isn't a string after all). Let it
    // propagate per `coding-policy: error-handling`.
    throw err;
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    logger.warn(
      {
        folder,
        parsedType:
          parsed === null
            ? 'null'
            : Array.isArray(parsed)
              ? 'array'
              : typeof parsed,
      },
      `${fileLabel} migration: payload is not an object, skipping (file left in place)`,
    );
    recordLeftInPlace(summary, folder);
    return null;
  }
  return parsed as Record<string, unknown>;
}

/**
 * Per-file catch helper. If `err` is a `SqliteError` whose `code`
 * starts with `'SQLITE_CONSTRAINT_'` (NOT NULL / UNIQUE / CHECK / PK /
 * FK), log a warn carrying `errCode` and return `true` so the caller
 * pattern reads:
 *
 *   } catch (err) {
 *     if (handleConstraintViolationOrRethrow(err, folder, label)) continue;
 *   }
 *
 * Anything else (a plain `Error`, a `TypeError`, a non-constraint
 * `SqliteError` like `SQLITE_CORRUPT` or `SQLITE_BUSY`) is rethrown so
 * the operator sees the real failure rather than a swept-under-the-rug
 * warn. Per `coding-policy: error-handling`.
 *
 * Returns the literal `true` because the only return path is the
 * warn-and-skip branch — every other branch throws.
 */
export function handleConstraintViolationOrRethrow(
  err: unknown,
  folder: string,
  fileLabel: string,
  summary?: MigrationSummary,
): true {
  if (
    err instanceof SqliteError &&
    typeof err.code === 'string' &&
    err.code.startsWith('SQLITE_CONSTRAINT_')
  ) {
    logger.warn(
      { folder, errCode: err.code, err },
      `${fileLabel} migration: row violated a DB constraint, rolling back and leaving source file in place for triage`,
    );
    recordLeftInPlace(summary, folder);
    return true;
  }
  throw err;
}

/**
 * Scan `GROUPS_DIR` for valid group folders (per
 * `isValidGroupFolder`), filter to directories, and return them sorted
 * by plain code-point comparison so `ON CONFLICT(<id>) DO NOTHING`
 * resolution is deterministic across filesystems and locales.
 *
 * `readdirSync` order is implementation-defined (ext4 hash order,
 * APFS insertion order); `localeCompare` adds a second axis of
 * nondeterminism (Turkish dotted-i, German ß-vs-ss, ICU version
 * skew). Plain `<` / `>` on string operands is locale-free and stable
 * across Node versions.
 *
 * Returns `[]` if `GROUPS_DIR` doesn't exist (first-boot ENOENT). Any
 * other errno propagates so the operator sees real environment
 * problems.
 */
export function listGroupFoldersForMigration(): string[] {
  // Don't pre-check via existsSync — that returns `false` on any
  // stat error (EACCES, EPERM, …), which would silently treat a
  // permission/environment problem as "first boot" and skip the
  // migration. Let readdirSync throw and inspect the errno: only
  // ENOENT means "first boot, no groups dir yet"; everything else
  // is a real environment problem the operator must see.
  try {
    return fs
      .readdirSync(GROUPS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .filter((entry) => isValidGroupFolder(entry.name))
      .map((entry) => entry.name)
      .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw err;
  }
}

/**
 * Rename the migrated source file to
 * `${filePath}.migrated-${dateStamp}` so a re-run of the migration is
 * a no-op (the existsSync gate at the top of the loop won't see the
 * source). On success, log an `info` carrying `renamed_to` plus any
 * `importLogContext` the caller wants merged in (counts, etc.).
 *
 * ENOENT is the only recoverable errno: the source vanished between
 * the import committing and the rename (concurrent migration run,
 * manual file move). The data is already in SQL, so this is an
 * idempotent no-op — log info ("source already absent at rename
 * time") instead of the renamed message. Every other errno
 * (`EACCES`, `EPERM`, `EXDEV`, `ENOSPC`, …) propagates so the
 * operator can fix the underlying problem.
 */
export function renameMigratedSource(
  filePath: string,
  dateStamp: string,
  folder: string,
  fileLabel: string,
  importLogContext: Record<string, unknown> = {},
  summary?: MigrationSummary,
): void {
  const renamedTo = `${filePath}.migrated-${dateStamp}`;
  try {
    fs.renameSync(filePath, renamedTo);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
    logger.info(
      // Spread the caller's context FIRST so it can carry counts /
      // arbitrary metadata, then overlay the load-bearing fields
      // (`folder`) — that way a caller can't accidentally clobber the
      // attribution by passing `{ folder: ... }` in the context bag.
      { ...importLogContext, folder },
      `${fileLabel} migration: imported; source already absent at rename time`,
    );
    // Even though no rename happened on disk, the data did land in
    // SQL — count this as a successful migrated-this-boot from the
    // operator's "did anything land" perspective.
    if (summary) summary.migrated += 1;
    return;
  }
  logger.info(
    { ...importLogContext, folder, renamed_to: renamedTo },
    `${fileLabel} migration: imported and source renamed`,
  );
  if (summary) summary.migrated += 1;
}

/**
 * Construct the `YYYY-MM-DD` date stamp the per-group rename uses.
 * Saves callers from re-deriving `new Date().toISOString().slice(0, 10)`
 * inline (and from accidentally including the `T...Z` suffix).
 */
export function migrationDateStamp(): string {
  return new Date().toISOString().slice(0, 10);
}
