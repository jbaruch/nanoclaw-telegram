// One-shot JSON→SQL state importers, extracted verbatim from src/db.ts
// (#751 seam 1). `initDatabase` dispatches here via `migrateJsonState`
// AFTER `createSchema` + the versioned state-table migrations
// (`state-migrations/*`), so every backfill targets a table that
// already exists. The DATA_DIR migrations (router_state / sessions /
// registered_groups) write through the db.ts accessors; the per-group
// #293-wave importers take the connection handle explicitly — a
// parameter, not a second handle, preserving the single-connection
// ownership in db.ts.
import Database, { SqliteError } from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

import { DATA_DIR, GROUPS_DIR } from './config.js';
import { setRegisteredGroup, setRouterState, setSession } from './db.js';
import { SUPPORTED_TZ_STATE_SCHEMA_VERSION } from './db-tz.js';
import { isFsErrorWithCode } from './fs-errors.js';
import { isValidGroupFolder } from './group-folder.js';
import {
  handleConstraintViolationOrRethrow,
  hasMigratedSibling,
  isObjectRow,
  listGroupFoldersForMigration,
  migrationDateStamp,
  type MigrationSummary,
  newMigrationSummary,
  parseJsonObjectOrWarn,
  renameMigratedSource,
} from './json-state-import.js';
import { logger } from './logger.js';
import { RegisteredGroup } from './types.js';

export function migrateJsonState(db: Database.Database): void {
  const migrateFile = (filename: string) => {
    const filePath = path.join(DATA_DIR, filename);
    if (!fs.existsSync(filePath)) return null;
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      fs.renameSync(filePath, `${filePath}.migrated`);
      return data;
    } catch (err) {
      // best-effort JSON-state migration: malformed JSON (SyntaxError) or an
      // fs errno (read/rename) skips this file; a non-fs, non-parse defect
      // propagates.
      if (
        !(err instanceof SyntaxError) &&
        !isFsErrorWithCode(err, [
          'ENOENT',
          'EACCES',
          'EPERM',
          'EISDIR',
          'ENOTDIR',
          'ELOOP',
          'ENAMETOOLONG',
          'EROFS',
          'EBUSY',
        ])
      ) {
        throw err;
      }
      logger.warn(
        { filePath, err },
        'Skipping malformed/unreadable JSON state file during migration',
      );
      return null;
    }
  };

  // Migrate router_state.json
  const routerState = migrateFile('router_state.json') as {
    last_timestamp?: string;
    last_agent_timestamp?: Record<string, string>;
  } | null;
  if (routerState) {
    if (routerState.last_timestamp) {
      setRouterState('last_timestamp', routerState.last_timestamp);
    }
    if (routerState.last_agent_timestamp) {
      setRouterState(
        'last_agent_timestamp',
        JSON.stringify(routerState.last_agent_timestamp),
      );
    }
  }

  // Migrate sessions.json
  const sessions = migrateFile('sessions.json') as Record<
    string,
    string
  > | null;
  if (sessions) {
    // Legacy JSON state predates parallel-maintenance; all sessions were
    // user-facing, so they migrate to the `default` slot.
    for (const [folder, sessionId] of Object.entries(sessions)) {
      setSession(folder, 'default', sessionId);
    }
  }

  // Migrate registered_groups.json
  const groups = migrateFile('registered_groups.json') as Record<
    string,
    RegisteredGroup
  > | null;
  if (groups) {
    for (const [jid, group] of Object.entries(groups)) {
      // Pre-check the one recoverable case (invalid folder); a DB or
      // programming error from setRegisteredGroup then propagates instead of
      // being silently skipped by a broad catch.
      if (!isValidGroupFolder(group.folder)) {
        logger.warn(
          { jid, folder: group.folder },
          'Skipping migrated registered group with invalid folder',
        );
        continue;
      }
      setRegisteredGroup(jid, group);
    }
  }

  // Per-group migrations (#293 wave). Each returns a `MigrationSummary`
  // (#433) so we can emit one startup-summary log line per migration —
  // the operator's grep target for "did the data plane come up clean".
  // The DATA_DIR migrations above (router_state / sessions /
  // registered_groups) are NOT per-group and do not contribute summaries.
  const summaries: MigrationSummary[] = [];

  // Migrate per-group orders-db.json files (#294). Unlike the helpers
  // above (DATA_DIR-rooted), this scans every `groups/<name>/` folder
  // for an `orders-db.json` because the file historically lived in the
  // admin group's working dir. Idempotent: a successful migration
  // renames the source to `orders-db.json.migrated-YYYY-MM-DD`, so a
  // re-run of `initDatabase` is a no-op once the file is gone.
  summaries.push(migrateOrdersDbJsonFiles(db));

  // Migrate per-group morning-brief-pending.json files (#299). Same
  // per-group-scan pattern as the orders import above: the source
  // file historically lived under each group's working dir, the
  // schema-only migration in state-007 created the three queue tables,
  // and this pass populates them from the JSON-era shape. Idempotent:
  // each successful per-file import renames the source to
  // `morning-brief-pending.json.migrated-YYYY-MM-DD`, so a re-run of
  // `initDatabase` is a no-op once the file is gone.
  summaries.push(migrateMorningBriefPendingJsonFiles(db));

  // Migrate per-group calendar-state.json files (#300). Same
  // per-group-scan pattern: the source file historically lived under
  // each group's working dir as a JSON envelope wrapping a per-day
  // `events` array, the schema-only migration in state-008 created
  // `calendar_snapshots` + `calendar_events` (FK + cascade), and this
  // pass populates them from the JSON-era shape. Idempotent: each
  // successful per-file import renames the source to
  // `calendar-state.json.migrated-YYYY-MM-DD`, so a re-run of
  // `initDatabase` is a no-op once the file is gone.
  summaries.push(migrateCalendarStateJsonFiles(db));

  // Migrate per-group heartbeat-state.json files into the
  // `phase_completions` table (#301). The state-009 schema landed in
  // PR #346; this pass populates the three phase rows (heartbeat,
  // nightly, weekly) from the JSON-era envelope plus the heartbeat-
  // specific `last_composio_check` extra in the heartbeat row's
  // `metadata` JSON blob (writer-decides convention per the state-009
  // doc-header). UPSERT keyed on `phase` so a re-run with newer values
  // wins without resetting defaulted columns. Source renamed to
  // `heartbeat-state.json.migrated-YYYY-MM-DD` on success — re-run is
  // a no-op once the suffix is in place.
  summaries.push(migrateHeartbeatStateJsonFiles(db));

  // Migrate per-group task-tz-state.json files into the singleton
  // `tz_state` row + `follow_me_tasks` per-skill rows (#302). The
  // state-010 schema landed in PR #348; this pass populates both
  // tables from the JSON-era envelope. UPSERT semantics throughout —
  // never `INSERT OR REPLACE`, which would silently reset
  // `schema_version` (a column the writer's UPSERT doesn't name) on
  // every re-run.
  summaries.push(migrateTaskTzStateJsonFiles(db));

  // Migrate per-group session-state.json files (the multi-writer
  // trusted-memory state) into trusted_sessions + trusted_session_singleton
  // (#298). The state-006 schema landed in PR #340; this pass populates
  // both tables from the JSON-era envelope. UPSERT semantics on both
  // tables — never INSERT OR REPLACE.
  summaries.push(migrateTrustedSessionStateJsonFiles(db));

  // Migrate per-group nanoclaw-state.json files (the multi-key junk
  // drawer) into the three state-005 tables: email_state singleton,
  // email_seen_ids set, resumable_cycles per-skill rows (#297).
  // UPSERT throughout — never INSERT OR REPLACE.
  summaries.push(migrateNanoclawStateJsonFiles(db));

  // Migrate per-group scheduled-reminders.json files into the
  // scheduled_reminders table created by state-004 (#296). Append-only
  // INSERT with ON CONFLICT(event_id) DO NOTHING.
  summaries.push(migrateScheduledRemindersJsonFiles(db));

  // Migrate per-group email-feedback.json files into the email_feedback
  // table created by state-002 (#295). Append-only INSERT (id is
  // AUTOINCREMENT; no natural-key dedup). Idempotency gated by source-
  // file rename. Accepts both wrapped {feedback:[...]} and bare-array
  // shapes per the issue body.
  summaries.push(migrateEmailFeedbackJsonFiles(db));

  emitMigrationStartupSummary(summaries);
}

/**
 * Emit one structured log line per per-group migration (#433) so
 * operators have a single grep target — `JSON state migration summary`
 * — for "did the data plane come up clean". INFO when the migration
 * left no files behind for triage; WARN when at least one group folder
 * still holds a source file (bad JSON, DB constraint violation, missing
 * required envelope fields).
 *
 * The literal string `JSON state migration summary` is load-bearing —
 * it's the operator's grep target. Don't rephrase it.
 */
function emitMigrationStartupSummary(summaries: MigrationSummary[]): void {
  for (const s of summaries) {
    const fields = {
      migration: s.name,
      migrated: s.migrated,
      skipped_already_done: s.skippedAlreadyDone,
      left_in_place_count: s.leftInPlace.length,
      left_in_place_groups: s.leftInPlace,
    };
    let message = `JSON state migration summary: ${s.name} — migrated=${s.migrated} skipped-already-done=${s.skippedAlreadyDone} left-in-place=${s.leftInPlace.length}`;
    if (s.leftInPlace.length > 0) {
      message += ` (groups: ${s.leftInPlace.join(', ')})`;
    }
    if (s.leftInPlace.length === 0) {
      logger.info(fields, message);
    } else {
      logger.warn(fields, message);
    }
  }
}

interface OrdersDbJsonRecord {
  id: string;
  source: string;
  status: string;
  amount?: number | null;
  currency?: string | null;
  description: string;
  order_date: string;
  expected_delivery?: string | null;
  email_message_id: string;
  to_address?: string | null;
  flagged?: boolean;
  flag_reason?: string | null;
  // Optional in the source JSON — `check-orders` skill runs don't
  // always populate it. The migration defaults missing values to
  // `order_date` (then current ISO) before the NOT NULL insert (#347).
  last_updated?: string;
}

interface OrdersDbJsonShape {
  orders?: OrdersDbJsonRecord[];
  last_checked?: string;
  last_updated?: string;
}

/**
 * Return the first candidate that is a non-empty string. Treats null,
 * undefined, and `""` all as "missing" — `??` alone would let `""`
 * through (it's defined and non-null), which satisfies SQLite's NOT
 * NULL constraint but leaves a semantically empty value downstream.
 *
 * Falls through to a fresh ISO timestamp when every candidate is
 * missing — guarantees the return value is always a non-empty string,
 * so callers can pass it directly to a NOT NULL `TEXT` column without
 * an extra check.
 *
 * Exported for the dedicated unit test in
 * `orders-json-migration.test.ts`; not part of the public API.
 *
 * @internal exported only for tests (see jbaruch/nanoclaw#347).
 */
export function firstNonEmpty(
  candidates: ReadonlyArray<string | null | undefined>,
): string {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.length > 0) {
      return candidate;
    }
  }
  return new Date().toISOString();
}

function migrateOrdersDbJsonFiles(db: Database.Database): MigrationSummary {
  const summary = newMigrationSummary('orders-db');
  if (!fs.existsSync(GROUPS_DIR)) return summary;
  let groupFolders: string[];
  try {
    groupFolders = fs
      .readdirSync(GROUPS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .filter((entry) => isValidGroupFolder(entry.name))
      .map((entry) => entry.name)
      // Sort so "first writer wins" with ON CONFLICT DO NOTHING below
      // is deterministic across filesystems AND across locales.
      // readdirSync order is implementation-defined (ext4 hash order,
      // APFS insertion order, etc.) and `localeCompare` would add a
      // second axis of nondeterminism (Turkish dotted-i, German
      // ß-vs-ss, ICU version skew). Plain code-point comparison via
      // </> on string operands is locale-free and stable across Node
      // versions.
      .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return summary;
    throw err;
  }

  // Bare `ON CONFLICT DO NOTHING` — handles BOTH the
  // `email_message_id UNIQUE` constraint and the PK `id`. The latter
  // *can* collide in practice: `id` is
  // `{source}-{order_date}-SHA1(description)[:8]`, so two distinct
  // emails with the same source + order_date + description (e.g., a
  // resent confirmation, or two amazon orders for the same item on
  // the same day) produce identical ids despite different
  // email_message_id values. For one-shot data backfill we want
  // idempotency, not strict validation: first row in (sorted by
  // folder above for determinism) wins, every other duplicate is
  // silently skipped. The downstream `check-orders` skill enforces
  // its own merge semantics on subsequent writes.
  const insertOrder = db.prepare(
    `INSERT INTO orders (
       id, source, status, amount, currency, description, order_date,
       expected_delivery, email_message_id, to_address, flagged,
       flag_reason, last_updated
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT DO NOTHING`,
  );
  const upsertMetadata = db.prepare(
    `INSERT INTO orders_metadata (key, value) VALUES (?, ?)
     ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
  );

  const stamp = new Date().toISOString().slice(0, 10);

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'orders-db.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let parsed: OrdersDbJsonShape;
    try {
      parsed = JSON.parse(
        fs.readFileSync(filePath, 'utf-8'),
      ) as OrdersDbJsonShape;
    } catch (err) {
      if (err instanceof SyntaxError) {
        logger.warn(
          { folder, errName: err.name },
          'orders-db.json migration: invalid JSON, skipping (file left in place)',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      // TOCTOU race: the existsSync check above is best-effort, not
      // authoritative — between that check and readFileSync the file
      // can be removed by a concurrent migration run, manual cleanup,
      // or filesystem reorg. Treat ENOENT here the same as ENOENT at
      // rename time: idempotent no-op, log at info, continue. Every
      // other errno propagates per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'orders-db.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }
    if (!Array.isArray(parsed.orders)) {
      logger.warn(
        { folder },
        'orders-db.json migration: missing "orders" array, skipping',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    // Wrap the per-file work in a single transaction so a partial
    // crash mid-import (process kill, IO error on metadata write)
    // can't leave the table in a half-migrated state. Failures here
    // propagate — per `coding-policy: error-handling`, an unexpected
    // exception during a structured INSERT means the data shape
    // doesn't match the schema (a real bug or malformed source file)
    // and the operator must triage before continuing. The schema
    // version gate in applyStateMigrations is independent of this;
    // the schema is already at v1, this is data backfill only.
    const orders = parsed.orders;
    const importFile = db.transaction(() => {
      let insertedRows = 0;
      let skippedRows = 0;
      for (const order of orders) {
        // #347: `orders.last_updated` is `TEXT NOT NULL` but the source
        // JSON (produced incrementally by `check-orders` skill runs) does
        // not always populate it — observed on production 2026-04-30
        // where one entry of 120 had no `last_updated`, taking down the
        // orchestrator on next boot via the SqliteError NOT NULL
        // constraint propagating out of the transaction. Default to
        // `order_date` (the next-best upper bound: we knew about this
        // order at least by then) and finally to a fresh ISO timestamp.
        //
        // Treat null, undefined, AND empty-string as "missing" — `??`
        // alone would let `last_updated: ""` through, satisfying NOT
        // NULL but leaving a semantically empty timestamp downstream
        // (PR #350 review: copilot caught this on the original `??`
        // chain). The helper falls through to the next candidate the
        // same way for any of those three shapes and always returns a
        // non-empty string.
        //
        // Other NOT NULL columns (id, source, status, description,
        // order_date, email_message_id) intentionally have no default —
        // their absence indicates a corrupt source row that should
        // surface as an error rather than be silently masked with a
        // synthesized value, since downstream `check-orders` semantics
        // depend on those fields meaning what the source said.
        const lastUpdated = firstNonEmpty([
          order.last_updated,
          order.order_date,
        ]);
        const result = insertOrder.run(
          order.id,
          order.source,
          order.status,
          order.amount ?? null,
          order.currency ?? null,
          order.description,
          order.order_date,
          order.expected_delivery ?? null,
          order.email_message_id,
          order.to_address ?? null,
          order.flagged ? 1 : 0,
          order.flag_reason ?? null,
          lastUpdated,
        );
        if (result.changes > 0) insertedRows++;
        else skippedRows++;
      }
      if (parsed.last_checked) {
        upsertMetadata.run('last_checked', parsed.last_checked);
      }
      if (parsed.last_updated) {
        upsertMetadata.run('last_updated', parsed.last_updated);
      }
      return { insertedRows, skippedRows };
    });

    const counts = importFile();

    // Rename is metadata cleanup; the data import already committed.
    // Per `coding-policy: error-handling`, only one specific errno is
    // recoverable here: `ENOENT` means the source disappeared between
    // the import and the rename (concurrent migration run, manual
    // file move) — that's an idempotent no-op since the data is
    // already in SQL. Every other errno (`EACCES`, `EPERM`, `EXDEV`,
    // `ENOSPC`, etc.) indicates a real environment problem the
    // operator must fix before startup proceeds; rethrowing those
    // surfaces the issue immediately rather than letting the
    // orchestrator come up with stale JSON files lying around.
    try {
      fs.renameSync(filePath, `${filePath}.migrated-${stamp}`);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
      // Source file already gone — log + continue without the
      // info-level "renamed" message since no rename actually
      // happened.
      logger.info(
        {
          folder,
          inserted: counts.insertedRows,
          skipped: counts.skippedRows,
          total: orders.length,
        },
        'orders-db.json migration: imported; source already absent at rename time',
      );
      // Data did land in SQL; count as migrated-this-boot from the
      // operator's "did anything land" perspective (matches
      // renameMigratedSource's success-counting semantics).
      summary.migrated += 1;
      continue;
    }
    logger.info(
      {
        folder,
        inserted: counts.insertedRows,
        skipped: counts.skippedRows,
        total: orders.length,
      },
      'orders-db.json migration: imported and source renamed',
    );
    summary.migrated += 1;
  }
  return summary;
}

interface MorningBriefCleanupItemJson {
  id: string;
  type: string;
  question?: string | null;
  subject?: string | null;
  sender?: string | null;
  added?: string | null;
}

interface MorningBriefPendingDecisionJson {
  id: string;
  question: string;
  added?: string | null;
}

interface MorningBriefUndatedTaskJson {
  id: string;
  title: string;
  tasklist_id: string;
  added?: string | null;
}

interface MorningBriefPendingJsonShape {
  cleanup_items?: MorningBriefCleanupItemJson[];
  pending_decisions?: MorningBriefPendingDecisionJson[];
  undated_tasks?: MorningBriefUndatedTaskJson[];
}

function migrateMorningBriefPendingJsonFiles(
  db: Database.Database,
): MigrationSummary {
  const summary = newMigrationSummary('morning-brief-pending');
  if (!fs.existsSync(GROUPS_DIR)) return summary;
  let groupFolders: string[];
  try {
    groupFolders = fs
      .readdirSync(GROUPS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .filter((entry) => isValidGroupFolder(entry.name))
      .map((entry) => entry.name)
      // Sort so PK-conflict resolution under `ON CONFLICT(id) DO NOTHING`
      // is deterministic across filesystems and locales — same rationale
      // as the orders migration above (readdirSync order is impl-defined,
      // plain code-point comparison is locale-free).
      .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return summary;
    throw err;
  }

  // ON CONFLICT(id) DO NOTHING (NOT `INSERT OR IGNORE`): re-running
  // with leftover rows (e.g. an operator copied a partial DB back over
  // an already-imported one) is a silent no-op on the PK conflict, but
  // a NOT NULL violation on `type`/`question`/`title`/`tasklist_id`
  // still throws. `INSERT OR IGNORE` would silently swallow those too,
  // and we'd rename the source file thinking the import succeeded —
  // bad data lost without a trace. Per-row, when `added` is missing in
  // the source we omit the column from the INSERT so the schema's
  // `DEFAULT CURRENT_TIMESTAMP` fires — that's the contract documented
  // on state-007.
  const insertCleanupItemWithAdded = db.prepare(
    `INSERT INTO pending_cleanup_items
       (id, type, question, subject, sender, added)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertCleanupItemDefaultAdded = db.prepare(
    `INSERT INTO pending_cleanup_items
       (id, type, question, subject, sender)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertDecisionWithAdded = db.prepare(
    `INSERT INTO pending_decisions (id, question, added)
     VALUES (?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertDecisionDefaultAdded = db.prepare(
    `INSERT INTO pending_decisions (id, question) VALUES (?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertUndatedTaskWithAdded = db.prepare(
    `INSERT INTO pending_undated_tasks
       (id, title, tasklist_id, added)
     VALUES (?, ?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertUndatedTaskDefaultAdded = db.prepare(
    `INSERT INTO pending_undated_tasks (id, title, tasklist_id)
     VALUES (?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );

  const stamp = new Date().toISOString().slice(0, 10);

  for (const folder of groupFolders) {
    const filePath = path.join(
      GROUPS_DIR,
      folder,
      'morning-brief-pending.json',
    );
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let parsed: MorningBriefPendingJsonShape;
    try {
      parsed = JSON.parse(
        fs.readFileSync(filePath, 'utf-8'),
      ) as MorningBriefPendingJsonShape;
    } catch (err) {
      if (err instanceof SyntaxError) {
        logger.warn(
          { folder, errName: err.name },
          'morning-brief-pending.json migration: invalid JSON, skipping (file left in place)',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      // TOCTOU race: existsSync above is best-effort; file may
      // disappear before readFileSync. Mirror orders' handling — log
      // info on ENOENT, propagate every other errno.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'morning-brief-pending.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    // `JSON.parse` happily returns null / numbers / strings / arrays
    // for syntactically valid but non-object payloads. Bind those to
    // a property access and we'd throw "cannot read properties of
    // null" before the array-shape guard below ever ran, halting the
    // whole migration on one bad file. Warn-and-skip per
    // `coding-policy: error-handling` (try alternatives before
    // failing) instead.
    if (
      parsed === null ||
      typeof parsed !== 'object' ||
      Array.isArray(parsed)
    ) {
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
        'morning-brief-pending.json migration: payload is not an object, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const cleanupItems = parsed.cleanup_items;
    const pendingDecisions = parsed.pending_decisions;
    const undatedTasks = parsed.undated_tasks;
    if (
      !Array.isArray(cleanupItems) &&
      !Array.isArray(pendingDecisions) &&
      !Array.isArray(undatedTasks)
    ) {
      logger.warn(
        { folder },
        'morning-brief-pending.json migration: no recognised arrays (cleanup_items / pending_decisions / undated_tasks), skipping',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    // One transaction per file so a mid-import crash can't leave any
    // of the three tables half-populated. Same rationale as the orders
    // migration; see the comment block on `migrateOrdersDbJsonFiles`.
    const cleanupCounts = { inserted: 0, skipped: 0, total: 0 };
    const decisionCounts = { inserted: 0, skipped: 0, total: 0 };
    const undatedTaskCounts = { inserted: 0, skipped: 0, total: 0 };
    try {
      // Per-row object guard: a stale `null`/string/number element in
      // any of the three arrays would otherwise throw a TypeError
      // inside the transaction (e.g. `null.added` blows up before any
      // INSERT runs), and the narrowed catch below would propagate
      // that as an "unexpected" error and halt orchestrator startup.
      // Skip non-object elements with a warn instead — same shape as
      // the file-level non-object guard above.
      const isObjectRow = (row: unknown): row is Record<string, unknown> =>
        row !== null && typeof row === 'object' && !Array.isArray(row);

      const importFile = db.transaction(() => {
        if (Array.isArray(cleanupItems)) {
          cleanupCounts.total = cleanupItems.length;
          for (const item of cleanupItems) {
            if (!isObjectRow(item)) {
              logger.warn(
                { folder, queue: 'cleanup_items' },
                'morning-brief-pending.json migration: skipping non-object row',
              );
              cleanupCounts.skipped++;
              continue;
            }
            const result = item.added
              ? insertCleanupItemWithAdded.run(
                  item.id,
                  item.type,
                  item.question ?? null,
                  item.subject ?? null,
                  item.sender ?? null,
                  item.added,
                )
              : insertCleanupItemDefaultAdded.run(
                  item.id,
                  item.type,
                  item.question ?? null,
                  item.subject ?? null,
                  item.sender ?? null,
                );
            if (result.changes > 0) cleanupCounts.inserted++;
            else cleanupCounts.skipped++;
          }
        }
        if (Array.isArray(pendingDecisions)) {
          decisionCounts.total = pendingDecisions.length;
          for (const decision of pendingDecisions) {
            if (!isObjectRow(decision)) {
              logger.warn(
                { folder, queue: 'pending_decisions' },
                'morning-brief-pending.json migration: skipping non-object row',
              );
              decisionCounts.skipped++;
              continue;
            }
            const result = decision.added
              ? insertDecisionWithAdded.run(
                  decision.id,
                  decision.question,
                  decision.added,
                )
              : insertDecisionDefaultAdded.run(decision.id, decision.question);
            if (result.changes > 0) decisionCounts.inserted++;
            else decisionCounts.skipped++;
          }
        }
        if (Array.isArray(undatedTasks)) {
          undatedTaskCounts.total = undatedTasks.length;
          for (const task of undatedTasks) {
            if (!isObjectRow(task)) {
              logger.warn(
                { folder, queue: 'undated_tasks' },
                'morning-brief-pending.json migration: skipping non-object row',
              );
              undatedTaskCounts.skipped++;
              continue;
            }
            const result = task.added
              ? insertUndatedTaskWithAdded.run(
                  task.id,
                  task.title,
                  task.tasklist_id,
                  task.added,
                )
              : insertUndatedTaskDefaultAdded.run(
                  task.id,
                  task.title,
                  task.tasklist_id,
                );
            if (result.changes > 0) undatedTaskCounts.inserted++;
            else undatedTaskCounts.skipped++;
          }
        }
      });
      importFile();
    } catch (err) {
      // Per `coding-policy: error-handling`: catch only SqliteError
      // with a constraint-class code (NOT NULL / UNIQUE / CHECK /
      // PRIMARY KEY / FOREIGN KEY) — those are the recoverable data-
      // quality failures the per-file isolation contract was written
      // for. Anything else (a TypeError, a ReferenceError, a non-
      // constraint SqliteError like SQLITE_CORRUPT or SQLITE_BUSY) is
      // either a programming bug or a real environment problem that
      // halting startup loudly will surface, instead of being swept
      // under a per-file warn. The transaction rolled back on throw,
      // so no partial rows landed in either case.
      if (
        err instanceof SqliteError &&
        typeof err.code === 'string' &&
        err.code.startsWith('SQLITE_CONSTRAINT_')
      ) {
        logger.warn(
          { folder, errCode: err.code, err },
          'morning-brief-pending.json migration: row violated a DB constraint, rolling back and leaving source file in place for triage',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      throw err;
    }

    // Rename is metadata cleanup; the data import already committed.
    // ENOENT is the only recoverable errno (file vanished between
    // import and rename — idempotent no-op since the data is in SQL);
    // every other errno propagates.
    try {
      fs.renameSync(filePath, `${filePath}.migrated-${stamp}`);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
      logger.info(
        {
          folder,
          cleanup: cleanupCounts,
          decisions: decisionCounts,
          undated_tasks: undatedTaskCounts,
        },
        'morning-brief-pending.json migration: imported; source already absent at rename time',
      );
      summary.migrated += 1;
      continue;
    }
    logger.info(
      {
        folder,
        cleanup: cleanupCounts,
        decisions: decisionCounts,
        undated_tasks: undatedTaskCounts,
        renamed_to: `${filePath}.migrated-${stamp}`,
      },
      'morning-brief-pending.json migration: imported and source renamed',
    );
    summary.migrated += 1;
  }
  return summary;
}

/**
 * Migrate per-group `calendar-state.json` files into the
 * `calendar_snapshots` + `calendar_events` tables created by
 * state-008 (#300). Per-group-scan pattern (same as the orders and
 * morning-brief-pending migrations above) refactored onto the shared
 * helpers in `src/json-state-import.ts`: parse-shape guard, per-row
 * object guard, narrowed `SqliteError` constraint catch, deterministic
 * folder ordering, and ENOENT-tolerant rename. See PR #368 for the
 * helper extraction rationale.
 *
 * JSON-era envelope shape:
 *   {
 *     "date":       "YYYY-MM-DD",
 *     "fetched_at": "ISO8601",
 *     "events":     [
 *       { event_id, title, start, end?, reminder_task_id? }, ...
 *     ]
 *   }
 *
 * Insert order is snapshot-row-first, then per-event rows. Under the
 * current production setting (`PRAGMA foreign_keys` is OFF — see the
 * doc-header on `state-008-calendar-state.ts`) this is purely a
 * style/consistency choice; once the orchestrator flips
 * `foreign_keys = ON` globally, the ordering becomes load-bearing
 * because `calendar_events.date` references `calendar_snapshots.date`
 * via `ON DELETE CASCADE` and an event INSERT for an absent snapshot
 * would fire SQLITE_CONSTRAINT_FOREIGNKEY. Doing the snapshot first
 * inside the per-file transaction keeps the import correct under
 * either FK setting.
 *
 * `ON CONFLICT(date) DO NOTHING` on the snapshot insert: re-running
 * with leftover state (e.g. an operator copied a partial DB back over
 * an already-imported one) is a silent no-op on the PK conflict, but
 * a NOT NULL violation on `fetched_at` still throws and rolls back.
 * Same shape on the event insert keyed on `event_id` — the PK is
 * Google Calendar's own event ID, so re-importing the same source
 * file on top of a partial migration just re-converges on the
 * already-stored row.
 */
function migrateCalendarStateJsonFiles(
  db: Database.Database,
): MigrationSummary {
  const summary = newMigrationSummary('calendar-state');
  const groupFolders = listGroupFoldersForMigration();
  if (groupFolders.length === 0) return summary;

  const insertSnapshot = db.prepare(
    `INSERT INTO calendar_snapshots (date, fetched_at)
     VALUES (?, ?)
     ON CONFLICT(date) DO NOTHING`,
  );
  const insertEvent = db.prepare(
    `INSERT INTO calendar_events
       (event_id, date, title, start, end, reminder_task_id)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(event_id) DO NOTHING`,
  );

  const stamp = migrationDateStamp();

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'calendar-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort; the file may
      // disappear before readFileSync. Mirror the orders / morning-
      // brief-pending migrations — log info on ENOENT, propagate every
      // other errno.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'calendar-state.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(
      raw,
      folder,
      'calendar-state.json',
      summary,
    );
    if (parsed === null) continue;

    const date = parsed.date;
    const fetchedAt = parsed.fetched_at;
    if (typeof date !== 'string' || typeof fetchedAt !== 'string') {
      // The two fields the snapshot row keys on / records. If either
      // is missing or wrong-typed, we can't write a snapshot row at
      // all, and the FK column on the events would dangle. Warn and
      // leave the file in place for triage.
      logger.warn(
        { folder },
        'calendar-state.json migration: missing or non-string date / fetched_at, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }
    // Distinguish "missing" (key absent or undefined) from
    // "wrong-typed" (key present but not an array — e.g. an object
    // or a string). Missing is a valid empty-day shape: still import
    // the snapshot row and rename. Wrong-typed is a corruption signal
    // that the schema can't migrate; warn (with the rejected type) and
    // skip the entire group's file so the events payload survives for
    // human triage instead of being silently discarded by the rename.
    const eventsRaw: unknown = parsed.events;
    let events: unknown[];
    if (eventsRaw === undefined) {
      events = [];
    } else if (Array.isArray(eventsRaw)) {
      events = eventsRaw;
    } else {
      logger.warn(
        {
          folder,
          eventsType: eventsRaw === null ? 'null' : typeof eventsRaw,
        },
        'calendar-state.json migration: events is not an array, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const counts = {
      inserted_snapshots: 0,
      inserted_events: 0,
      skipped_events: 0,
      total_events: events.length,
    };
    try {
      const importFile = db.transaction(() => {
        // Snapshot first — see doc-header. The conflict resolver
        // returns 0 changes if a previous run already inserted today's
        // snapshot; we count `inserted_snapshots` only on the change.
        const snapshotResult = insertSnapshot.run(date, fetchedAt);
        if (snapshotResult.changes > 0) counts.inserted_snapshots++;

        for (const ev of events) {
          if (!isObjectRow(ev)) {
            logger.warn(
              { folder, queue: 'events' },
              'calendar-state.json migration: skipping non-object row',
            );
            counts.skipped_events++;
            continue;
          }
          const result = insertEvent.run(
            ev.event_id,
            date,
            ev.title,
            ev.start,
            ev.end ?? null,
            ev.reminder_task_id ?? null,
          );
          if (result.changes > 0) counts.inserted_events++;
          else counts.skipped_events++;
        }
      });
      importFile();
      // eslint-disable-next-line no-catch-all/no-catch-all -- handleConstraintViolationOrRethrow rethrows non-constraint errors (see its JSDoc); the linter cannot see through the call
    } catch (err) {
      // Narrowed catch: only constraint-class SqliteError is the
      // recoverable per-file failure (NOT NULL on title/start, FK
      // dangling, etc.). Anything else (a TypeError from a bug, a
      // SQLITE_BUSY from a noisy environment) propagates — operator
      // visibility per `coding-policy: error-handling`. The
      // transaction has already rolled back on throw, so no partial
      // rows landed.
      if (
        handleConstraintViolationOrRethrow(
          err,
          folder,
          'calendar-state.json',
          summary,
        )
      )
        continue;
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      'calendar-state.json',
      counts,
      summary,
    );
  }
  return summary;
}

/**
 * Shape of the JSON-era `groups/<name>/heartbeat-state.json` envelope
 * the heartbeat / nightly / weekly skills used to share via
 * `LOCK_EX` (#301). Every key is optional because the file accumulated
 * incrementally — an early-stage group may have only run the heartbeat
 * phase, leaving `nightly_last_completed` / `weekly_last_completed`
 * absent until those phases first ran. The migration imports whichever
 * keys are present and silently skips the absent ones.
 *
 * `last_composio_check` is the heartbeat skill's local extra; per the
 * state-009 doc-header it lands inside the `metadata` JSON blob on the
 * `heartbeat` row when present. `nightly` and `weekly` rows have no
 * phase-specific extras today, so their `metadata` is NULL.
 */
interface HeartbeatStateJsonShape {
  heartbeat_last_completed?: unknown;
  nightly_last_completed?: unknown;
  weekly_last_completed?: unknown;
  last_composio_check?: unknown;
}

function migrateHeartbeatStateJsonFiles(
  db: Database.Database,
): MigrationSummary {
  const summary = newMigrationSummary('heartbeat-state');
  const groupFolders = listGroupFoldersForMigration();
  if (groupFolders.length === 0) return summary;

  // UPSERT keyed on `phase` (PK) — see the state-009 doc-header for
  // the full rationale. Crucially NOT `INSERT OR REPLACE`: the latter
  // is delete+insert in SQLite and would reset defaulted columns the
  // UPSERT doesn't name — load-bearingly `schema_version`, which the
  // owner skill bumps to drive future shape migrations. The UPSERT
  // here touches exactly the three columns the writer cares about
  // and stamps `updated_at` explicitly so a writer-supplied
  // `last_completed` from a clock that drifts can't outrun the row's
  // own mutation log.
  // metadata uses COALESCE(excluded.metadata, metadata) so a NULL
  // payload (e.g. a JSON file with `nightly_last_completed` set but no
  // `last_composio_check`) doesn't wipe metadata that an earlier run
  // already wrote into the row. The non-NULL precedence is "newest
  // wins"; NULL is treated as "no opinion, leave existing".
  const upsertPhase = db.prepare(
    `INSERT INTO phase_completions (phase, last_completed, metadata)
     VALUES (?, ?, ?)
     ON CONFLICT(phase) DO UPDATE SET
       last_completed = excluded.last_completed,
       metadata       = COALESCE(excluded.metadata, phase_completions.metadata),
       updated_at     = CURRENT_TIMESTAMP`,
  );

  const stamp = migrationDateStamp();
  const fileLabel = 'heartbeat-state.json';

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'heartbeat-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU: file may disappear between `existsSync` and
      // `readFileSync` (concurrent migration run, manual cleanup).
      // Treat ENOENT as an idempotent info-level skip; every other
      // errno propagates per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'heartbeat-state.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(raw, folder, fileLabel, summary);
    if (parsed === null) continue;

    // Phase-row plan. The shape is deliberately a small array so the
    // three branches share one loop and one transaction — adding a
    // future phase (e.g. `composio` split out of the heartbeat blob)
    // is a one-line append, not a fourth branch to keep in sync.
    //
    // `metadataObj` is the writer-decides convention from the
    // state-009 doc-header: today only `heartbeat` carries an extra
    // (`last_composio_check`); `nightly` and `weekly` stay NULL. A
    // reader doesn't infer "no extras" from "{}" — absence means
    // "no extras to record".
    const composio = (parsed as HeartbeatStateJsonShape).last_composio_check;
    const phases: ReadonlyArray<{
      phase: string;
      tsKey: keyof HeartbeatStateJsonShape;
      metadataObj: Record<string, unknown> | null;
    }> = [
      {
        phase: 'heartbeat',
        tsKey: 'heartbeat_last_completed',
        metadataObj:
          typeof composio === 'string' && composio.length > 0
            ? { last_composio_check: composio }
            : null,
      },
      { phase: 'nightly', tsKey: 'nightly_last_completed', metadataObj: null },
      { phase: 'weekly', tsKey: 'weekly_last_completed', metadataObj: null },
    ];

    // Wrap the per-file work in a single transaction so a partial
    // failure on one of the three phase UPSERTs rolls all of them
    // back. Without the transaction wrapper, a crash mid-import (e.g.
    // a future CHECK constraint violation on the `weekly` row) would
    // leave the table with `heartbeat` / `nightly` committed and
    // `weekly` missing — the operator would then have to triage a
    // half-migrated state.
    const importFile = db.transaction(() => {
      let importedPhases = 0;
      for (const { phase, tsKey, metadataObj } of phases) {
        const ts = (parsed as HeartbeatStateJsonShape)[tsKey];
        if (typeof ts !== 'string' || ts.length === 0) continue;
        upsertPhase.run(
          phase,
          ts,
          metadataObj === null ? null : JSON.stringify(metadataObj),
        );
        importedPhases++;
      }
      return { importedPhases };
    });

    let counts: { importedPhases: number };
    try {
      counts = importFile();
      // eslint-disable-next-line no-catch-all/no-catch-all -- handleConstraintViolationOrRethrow rethrows non-constraint errors (see its JSDoc); the linter cannot see through the call
    } catch (err) {
      // The helper either returns true (caller continues) or rethrows;
      // there's no third path. Match the calendar-state pattern.
      handleConstraintViolationOrRethrow(err, folder, fileLabel, summary);
      continue;
    }

    renameMigratedSource(filePath, stamp, folder, fileLabel, counts, summary);
  }
  return summary;
}

interface TaskTzStateFollowMeTaskJson {
  name: string;
  // Modern shape (matches the state-010 spec docstring): wall-clock as
  // an `"HH:MM"` string and the cron expression duplicated on the row.
  local_time?: string;
  schedule_value?: string;
  // Legacy shape — the pre-state-010 `task-tz-sync` writer split the
  // wall-clock into integer hour/minute and never duplicated the cron
  // (the cron lives on the sibling `scheduled_tasks` row keyed by
  // `task_id`). When state-010 shipped, in-place JSONs hadn't been
  // rewritten yet (the writer hasn't fired since 2026-04-25) so the
  // migration has to translate this shape into the schema's required
  // `local_time` / `schedule_value` columns. See #431.
  task_id?: string;
  local_hour?: number;
  local_minute?: number;
  last_run_date?: string | null;
  pending_run_at?: string | null;
}

interface TaskTzStateJsonShape {
  current_tz?: string;
  home_tz?: string;
  scheduler_tz?: string | null;
  follow_me_tasks?: TaskTzStateFollowMeTaskJson[];
}

const TASK_TZ_STATE_FILE_LABEL = 'task-tz-state.json';

/**
 * Resolve `local_time` and `schedule_value` for a follow_me row that
 * may carry either the modern state-010 shape or the legacy
 * `local_hour`/`local_minute` + `task_id`-keyed cron shape (#431).
 *
 * Returns null with a warn-log when neither shape supplies enough data
 * to populate the NOT NULL columns; the caller then treats the row as
 * skipped rather than letting the whole transaction roll back via a
 * SQLite constraint violation.
 *
 * `lookupCron` takes a `task_id` and returns the matching
 * `scheduled_tasks.schedule_value` (or undefined). Injected as a
 * callback so the helper stays decoupled from the prepared-statement
 * lifetime owned by `migrateTaskTzStateJsonFiles`.
 */
function resolveFollowMeTaskShape(
  task: TaskTzStateFollowMeTaskJson,
  folder: string,
  lookupCron: (taskId: string) => { schedule_value?: string } | undefined,
): { local_time: string; schedule_value: string } | null {
  let resolvedLocalTime: string | null = null;
  if (typeof task.local_time === 'string' && task.local_time.length > 0) {
    resolvedLocalTime = task.local_time;
  } else if (
    typeof task.local_hour === 'number' &&
    Number.isInteger(task.local_hour) &&
    task.local_hour >= 0 &&
    task.local_hour <= 23 &&
    typeof task.local_minute === 'number' &&
    Number.isInteger(task.local_minute) &&
    task.local_minute >= 0 &&
    task.local_minute <= 59
  ) {
    resolvedLocalTime = `${String(task.local_hour).padStart(2, '0')}:${String(task.local_minute).padStart(2, '0')}`;
  }

  let resolvedScheduleValue: string | null = null;
  if (
    typeof task.schedule_value === 'string' &&
    task.schedule_value.length > 0
  ) {
    resolvedScheduleValue = task.schedule_value;
  } else if (typeof task.task_id === 'string' && task.task_id.length > 0) {
    const row = lookupCron(task.task_id);
    if (
      row &&
      typeof row.schedule_value === 'string' &&
      row.schedule_value.length > 0
    ) {
      resolvedScheduleValue = row.schedule_value;
    }
  }

  if (resolvedLocalTime === null || resolvedScheduleValue === null) {
    // Booleans report *usable* presence (non-empty string / valid integer)
    // rather than just `typeof`-truthy, so a row that carried `local_time:
    // ""` or `schedule_value: ""` doesn't surface as `hasLocalTime: true`
    // when the resolver has effectively rejected it. Operators reading
    // these warns during triage need "could the value actually be used?"
    // not "does the JSON happen to contain that key?".
    logger.warn(
      {
        folder,
        taskName: task.name,
        taskId: task.task_id,
        hasLocalTime:
          typeof task.local_time === 'string' && task.local_time.length > 0,
        hasLocalHour:
          typeof task.local_hour === 'number' &&
          Number.isInteger(task.local_hour) &&
          task.local_hour >= 0 &&
          task.local_hour <= 23,
        hasLocalMinute:
          typeof task.local_minute === 'number' &&
          Number.isInteger(task.local_minute) &&
          task.local_minute >= 0 &&
          task.local_minute <= 59,
        hasScheduleValue:
          typeof task.schedule_value === 'string' &&
          task.schedule_value.length > 0,
        scheduleLookupHit: resolvedScheduleValue !== null,
      },
      `${TASK_TZ_STATE_FILE_LABEL} migration: cannot resolve local_time / schedule_value for follow_me row, skipping row (the per-file rename still happens once the transaction commits — partial imports are normal)`,
    );
    return null;
  }
  return {
    local_time: resolvedLocalTime,
    schedule_value: resolvedScheduleValue,
  };
}

/**
 * Migrate per-group `task-tz-state.json` files (#302, data-import
 * follow-up to schema PR #348). Mirrors the orders / morning-brief /
 * calendar-state pattern: scans every `groups/<name>/task-tz-state.
 * json`, parses the envelope, and writes the three timezone scalars
 * to the singleton `tz_state` row plus N rows to `follow_me_tasks`
 * inside a single transaction.
 *
 * Both writes use `ON CONFLICT(...) DO UPDATE` (UPSERT) — NEVER
 * `INSERT OR REPLACE`, which is delete+insert in SQLite and would
 * silently reset defaulted columns like `schema_version` on every
 * re-import. The multi-group test in
 * `task-tz-state-json-migration.test.ts` pins this down by manually
 * bumping `schema_version` between two group imports and asserting
 * the writer's known shape (`SUPPORTED_TZ_STATE_SCHEMA_VERSION`,
 * currently 4 post-state-015 / #574 Phase 2; was 3 after
 * jbaruch/nanoclaw-admin#229) is what lands on the
 * second import — anything else (1 or the manually-bumped value)
 * would signal either a `INSERT OR REPLACE` regression (1) or that
 * the writer dropped its explicit `schema_version` bind (manual
 * bump survives, gate rejects the row).
 *
 * `tz_state` UPSERT writes `schema_version` explicitly through
 * `SUPPORTED_TZ_STATE_SCHEMA_VERSION` (4 post-state-015; 3 between
 * #229 and state-015; 2 between #542 and #229): a fresh-DB import
 * would otherwise land at the state-010 column default of 1, and
 * the reader gate would reject
 * the imported row as "unfamiliar schema_version" until
 * `applyTripitSegmentsToTzState` rewrote it on the next nightly
 * `sync_tripit` run. The explicit bind keeps the JSON-import's row
 * shape coherent with the writer gate from the moment it lands.
 *
 * `follow_me_tasks` UPSERT re-stamps `updated_at = CURRENT_TIMESTAMP`
 * in the conflict branch so a re-import shows up as a fresh row
 * mutation in the audit log.
 *
 * Skip-and-warn when `current_tz` or `home_tz` is missing/empty —
 * both are `TEXT NOT NULL` on `tz_state`, so attempting the insert
 * would throw a constraint violation. We catch it before the
 * transaction starts so the file stays in place for triage and the
 * `follow_me_tasks` rows below also don't import (the JSON envelope
 * is malformed; partial import would be misleading).
 *
 * Idempotent via the standard `.migrated-YYYY-MM-DD` rename.
 */
function migrateTaskTzStateJsonFiles(db: Database.Database): MigrationSummary {
  const summary = newMigrationSummary('task-tz-state');
  const groupFolders = listGroupFoldersForMigration();
  if (groupFolders.length === 0) return summary;

  // tz_state UPSERT: writes `schema_version` explicitly via
  // `SUPPORTED_TZ_STATE_SCHEMA_VERSION` so the backfilled row matches
  // the reader gate. Without the explicit bind, a fresh-DB import
  // would land at the state-010 column default of 1 — state-012's
  // `UPDATE … WHERE id = 1` runs before this migration but is a
  // no-op when the row doesn't yet exist, and every reader would
  // then reject the imported row as "unfamiliar schema_version".
  // `segments` stays NULL on import — the column gets populated on
  // the next `sync_tripit` run via `applyTripitSegmentsToTzState`.
  // CHECK(id=1) makes tz_state a true singleton — the second
  // group's import updates the same row in place rather than landing
  // id=2 (which would also fail loudly via the CHECK).
  const upsertTzState = db.prepare(
    `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, schema_version)
     VALUES (1, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       current_tz     = excluded.current_tz,
       home_tz        = excluded.home_tz,
       scheduler_tz   = excluded.scheduler_tz,
       schema_version = excluded.schema_version`,
  );

  // follow_me_tasks UPSERT: re-stamp `updated_at` on conflict so the
  // audit log captures the re-import as a fresh row mutation. PK is
  // `name`; sibling rows (other task names) are untouched.
  const upsertFollowMeTask = db.prepare(
    `INSERT INTO follow_me_tasks
       (name, local_time, schedule_value, last_run_date, pending_run_at)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(name) DO UPDATE SET
       local_time     = excluded.local_time,
       schedule_value = excluded.schedule_value,
       last_run_date  = excluded.last_run_date,
       pending_run_at = excluded.pending_run_at,
       updated_at     = CURRENT_TIMESTAMP`,
  );

  // Legacy-shape fallback for `schedule_value`: pre-state-010 follow_me
  // entries don't carry the cron string — it lives on the sibling
  // `scheduled_tasks` row keyed by `task_id`. SELECT-by-PK on a tiny
  // table per legacy row is cheap; the modern-shape path never reaches
  // this query because it short-circuits on `task.schedule_value`.
  const lookupScheduledTaskCron = db.prepare(
    `SELECT schedule_value FROM scheduled_tasks WHERE id = ?`,
  );

  const stamp = migrationDateStamp();

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'task-tz-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort; the file may
      // disappear before readFileSync. Mirror the orders / morning-
      // brief handling — info-log on ENOENT, propagate every other
      // errno per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          `${TASK_TZ_STATE_FILE_LABEL} migration: file disappeared between existsSync and readFileSync, skipping`,
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(
      raw,
      folder,
      TASK_TZ_STATE_FILE_LABEL,
      summary,
    ) as TaskTzStateJsonShape | null;
    if (parsed === null) continue;

    // Both `current_tz` and `home_tz` are NOT NULL on `tz_state`. An
    // empty string would satisfy NOT NULL but break every reader
    // (morning-brief, nightly, weekly, check-calendar, heartbeat-
    // precheck) that uses the value as an IANA zone name. Treat
    // missing / non-string / empty all as "malformed envelope": warn,
    // leave the file in place for triage, and DON'T import the
    // sibling `follow_me_tasks` rows (a partial import would silently
    // ship N task rows without their tz context).
    if (
      typeof parsed.current_tz !== 'string' ||
      parsed.current_tz.length === 0 ||
      typeof parsed.home_tz !== 'string' ||
      parsed.home_tz.length === 0
    ) {
      logger.warn(
        {
          folder,
          hasCurrentTz: typeof parsed.current_tz === 'string',
          hasHomeTz: typeof parsed.home_tz === 'string',
        },
        `${TASK_TZ_STATE_FILE_LABEL} migration: missing or empty current_tz / home_tz, skipping (file left in place)`,
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const currentTz = parsed.current_tz;
    const homeTz = parsed.home_tz;
    // `scheduler_tz` is nullable on the schema (informational only;
    // not load-bearing per the schema-doc). Coerce missing /
    // non-string / empty to NULL.
    const schedulerTz =
      typeof parsed.scheduler_tz === 'string' && parsed.scheduler_tz.length > 0
        ? parsed.scheduler_tz
        : null;

    // `follow_me_tasks`: distinguish "missing" (key absent) from
    // "wrong-typed" (key present but not an array — e.g. an object).
    // Missing is the legitimate "no follow-me jobs configured"
    // envelope; we still land tz_state and rename. Wrong-typed is a
    // corruption signal: warn (with the rejected type) and skip the
    // entire group's file so the original task payload survives for
    // human triage instead of being silently discarded by the rename.
    const followMeRaw: unknown = parsed.follow_me_tasks;
    let followMeTasks: TaskTzStateFollowMeTaskJson[];
    if (followMeRaw === undefined) {
      followMeTasks = [];
    } else if (Array.isArray(followMeRaw)) {
      followMeTasks = followMeRaw as TaskTzStateFollowMeTaskJson[];
    } else {
      logger.warn(
        {
          folder,
          followMeType: followMeRaw === null ? 'null' : typeof followMeRaw,
        },
        `${TASK_TZ_STATE_FILE_LABEL} migration: follow_me_tasks is not an array, skipping (file left in place)`,
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    // The UPSERT counter splits inserts (first import for this
    // group) from refreshes (subsequent groups' singleton refresh)
    // because SQLite returns `changes = 1` for both branches —
    // labelling the combined number `tz_state_inserted` would be
    // misleading on the multi-group / re-run path. Use SELECT-then-
    // UPSERT to distinguish: if the singleton row already exists,
    // count as a refresh; otherwise, an insert.
    const tzStateExistsBefore =
      (
        db.prepare('SELECT COUNT(*) AS n FROM tz_state WHERE id = 1').get() as {
          n: number;
        }
      ).n > 0;
    const counts = {
      tz_state_inserted: 0,
      tz_state_refreshed: 0,
      follow_me_upserted: 0,
      skipped: 0,
    };
    try {
      const importFile = db.transaction(() => {
        const tzResult = upsertTzState.run(
          currentTz,
          homeTz,
          schedulerTz,
          SUPPORTED_TZ_STATE_SCHEMA_VERSION,
        );
        if (tzResult.changes > 0) {
          if (tzStateExistsBefore) counts.tz_state_refreshed++;
          else counts.tz_state_inserted++;
        }
        for (const task of followMeTasks) {
          if (!isObjectRow(task)) {
            logger.warn(
              { folder, queue: 'follow_me_tasks' },
              `${TASK_TZ_STATE_FILE_LABEL} migration: skipping non-object row`,
            );
            counts.skipped++;
            continue;
          }
          // Validate `name` separately from the local_time / schedule_value
          // resolver: PK on `follow_me_tasks` is `name TEXT PRIMARY KEY`, so
          // a missing / non-string / empty `name` would constraint-violate
          // at the upsert step and roll back the entire group's transaction
          // — exactly the failure mode the per-row-skip path was added to
          // avoid (#431). Skip-and-warn here keeps the per-row contract
          // intact for siblings.
          if (typeof task.name !== 'string' || task.name.length === 0) {
            logger.warn(
              {
                folder,
                taskId: (task as TaskTzStateFollowMeTaskJson).task_id,
                nameType:
                  task.name === null
                    ? 'null'
                    : typeof (task as { name?: unknown }).name,
              },
              `${TASK_TZ_STATE_FILE_LABEL} migration: skipping follow_me row with missing or non-string name (PK)`,
            );
            counts.skipped++;
            continue;
          }
          // Resolve `local_time` and `schedule_value` from either the
          // modern shape (state-010 spec) or the legacy shape
          // (`local_hour`/`local_minute` integers + `task_id` keying
          // the cron on the sibling `scheduled_tasks` row). Per-row
          // skip-and-warn rather than a transaction-wide rollback so
          // one malformed row doesn't strand the whole group's import
          // — that was the failure mode #431 hit on the live deployment.
          const resolved = resolveFollowMeTaskShape(
            task as TaskTzStateFollowMeTaskJson,
            folder,
            (taskId) =>
              lookupScheduledTaskCron.get(taskId) as
                | { schedule_value?: string }
                | undefined,
          );
          if (resolved === null) {
            counts.skipped++;
            continue;
          }
          const result = upsertFollowMeTask.run(
            task.name as string,
            resolved.local_time,
            resolved.schedule_value,
            // Both nullable cursor fields use `?? null` so an
            // explicit `null` and a missing key both round-trip as
            // SQL NULL. The reader contract on state-010 explicitly
            // tolerates NULL on both columns.
            (task.last_run_date as string | null | undefined) ?? null,
            (task.pending_run_at as string | null | undefined) ?? null,
          );
          if (result.changes > 0) counts.follow_me_upserted++;
          else counts.skipped++;
        }
      });
      importFile();
      // eslint-disable-next-line no-catch-all/no-catch-all -- handleConstraintViolationOrRethrow rethrows non-constraint errors (see its JSDoc); the linter cannot see through the call
    } catch (err) {
      // Per `coding-policy: error-handling`: only constraint-class
      // SqliteError is recoverable here — those are the per-file
      // data-quality failures the per-file isolation contract was
      // written for. Anything else (TypeError, non-constraint
      // SqliteError like SQLITE_CORRUPT/BUSY) propagates so the
      // operator sees the real failure rather than a swept-under-
      // the-rug warn. Transaction already rolled back on throw, so
      // no partial rows landed in either case.
      if (
        handleConstraintViolationOrRethrow(
          err,
          folder,
          TASK_TZ_STATE_FILE_LABEL,
          summary,
        )
      )
        continue;
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      TASK_TZ_STATE_FILE_LABEL,
      {
        tz_state_inserted: counts.tz_state_inserted,
        tz_state_refreshed: counts.tz_state_refreshed,
        follow_me_upserted: counts.follow_me_upserted,
        skipped: counts.skipped,
        total: followMeTasks.length,
      },
      summary,
    );
  }
  return summary;
}

/**
 * #298 — Migrate per-group `session-state.json` (the multi-writer
 * trusted-memory state file) into `trusted_sessions` +
 * `trusted_session_singleton`. Owner skill: `tessl__trusted-memory`.
 *
 * Source shape (documented on the state-006 doc-header):
 *
 *   {
 *     "schema_version": 1,
 *     "sessions": {"<NANOCLAW_SESSION_NAME>": {
 *        "started", "epoch", "session_id", "last_seen"
 *     }, ...},
 *     "active_session_id": "<top-level back-compat>",
 *     "seen_email_ids": [...],     // NOT migrated here — see below.
 *     "pending_response": {...} | "<string>",
 *     "muted_threads": [...]
 *   }
 *
 * The JSON-era top-level `seen_email_ids` field intentionally does
 * NOT migrate here — that field relocates to the `email_seen_ids`
 * table created by state-005 (#297) where both check-email writers
 * can target it without the old two-file consolidate dance. Don't
 * touch it from this migration; #297's own data-import PR owns that
 * row backfill.
 *
 * Per-named-session entries become `trusted_sessions` rows (UPSERT
 * by `session_name` so a re-run with a moved file in some other
 * group folder won't clobber per-session metadata). The singleton
 * fields become a single `trusted_session_singleton` row at id=1
 * (UPSERT, not INSERT OR REPLACE — REPLACE deletes the existing
 * row and re-inserts, which would reset `schema_version` to its
 * column DEFAULT and mask future migrations).
 *
 * `pending_response` and `muted_threads` are stored as TEXT in the
 * schema; the owner skill treats them as opaque JSON blobs. Stringify
 * here so a structured object/array on disk round-trips through the
 * column without losing shape.
 */
function migrateTrustedSessionStateJsonFiles(
  db: Database.Database,
): MigrationSummary {
  const summary = newMigrationSummary('session-state');
  const groupFolders = listGroupFoldersForMigration();

  // UPSERT, not INSERT OR REPLACE: REPLACE deletes the conflicting
  // row and re-inserts, which would reset `schema_version` to its
  // column DEFAULT(=1). When we later bump trusted_sessions'
  // `schema_version` for a shape change, REPLACE-on-import would
  // silently roll back any post-migration upgrade the owner skill
  // had performed. ON CONFLICT(session_name) DO UPDATE preserves the
  // existing `schema_version` while letting the four data fields
  // refresh.
  const upsertSession = db.prepare(
    `INSERT INTO trusted_sessions
       (session_name, session_id, started, epoch, last_seen)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(session_name) DO UPDATE SET
       session_id = excluded.session_id,
       started    = excluded.started,
       epoch      = excluded.epoch,
       last_seen  = excluded.last_seen`,
  );

  // Same UPSERT-not-REPLACE rationale for the singleton: re-run with
  // a second group's file (multi-host migration order) UPSERTs the
  // existing id=1 row in place, so the row count stays at 1 and the
  // existing `schema_version` is preserved across re-runs.
  const upsertSingleton = db.prepare(
    `INSERT INTO trusted_session_singleton
       (id, active_session_id, pending_response, muted_threads)
     VALUES (1, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       active_session_id = excluded.active_session_id,
       pending_response  = excluded.pending_response,
       muted_threads     = excluded.muted_threads`,
  );

  const stamp = migrationDateStamp();
  const fileLabel = 'session-state.json';

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'session-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort; the file may
      // disappear before readFileSync. Treat ENOENT here the same as
      // ENOENT at rename time — idempotent no-op, log info, continue.
      // Every other errno propagates per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          `${fileLabel} migration: file disappeared between existsSync and readFileSync, skipping`,
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(raw, folder, fileLabel, summary);
    if (!parsed) continue;

    const sessionsField = parsed.sessions;
    const hasSessions =
      sessionsField !== null &&
      typeof sessionsField === 'object' &&
      !Array.isArray(sessionsField);
    const hasSingleton =
      'active_session_id' in parsed ||
      'pending_response' in parsed ||
      'muted_threads' in parsed;

    if (!hasSessions && !hasSingleton) {
      logger.warn(
        { folder },
        `${fileLabel} migration: no recognised fields (sessions / active_session_id / pending_response / muted_threads), skipping`,
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const sessionCounts = { upserted: 0, skipped: 0, total: 0 };
    let singletonUpserted = false;
    try {
      const importFile = db.transaction(() => {
        if (hasSessions) {
          const sessions = sessionsField as Record<string, unknown>;
          const entries = Object.entries(sessions);
          sessionCounts.total = entries.length;
          for (const [sessionName, entry] of entries) {
            if (!isObjectRow(entry)) {
              logger.warn(
                { folder, session_name: sessionName },
                `${fileLabel} migration: skipping non-object session entry`,
              );
              sessionCounts.skipped++;
              continue;
            }
            // `started` / `epoch` / `last_seen` are NOT NULL in the
            // schema. A JSON-era row that pre-dates the field — e.g.
            // an old back-compat shape that only tracked `session_id`
            // — would otherwise throw NOT NULL inside the transaction
            // and abort the whole file's import. Skip-with-warn so the
            // remaining session entries (and the singleton) still
            // import. `session_id` is nullable per the state-006
            // schema (sqlite-error fallback path) so its absence is
            // valid.
            if (
              typeof entry.started !== 'string' ||
              typeof entry.epoch !== 'number' ||
              typeof entry.last_seen !== 'string'
            ) {
              logger.warn(
                {
                  folder,
                  session_name: sessionName,
                  has_started: typeof entry.started === 'string',
                  has_epoch: typeof entry.epoch === 'number',
                  has_last_seen: typeof entry.last_seen === 'string',
                },
                `${fileLabel} migration: session entry missing required fields (started/epoch/last_seen), skipping`,
              );
              sessionCounts.skipped++;
              continue;
            }
            upsertSession.run(
              sessionName,
              typeof entry.session_id === 'string' ? entry.session_id : null,
              entry.started,
              entry.epoch,
              entry.last_seen,
            );
            sessionCounts.upserted++;
          }
        }
        if (hasSingleton) {
          const activeSessionId =
            typeof parsed.active_session_id === 'string'
              ? parsed.active_session_id
              : null;
          // pending_response and muted_threads are TEXT in the schema
          // (opaque JSON blobs per the owner-skill contract). Stringify
          // structured shapes; pass strings through verbatim; treat
          // missing/null as NULL.
          const pendingResponse =
            parsed.pending_response === undefined ||
            parsed.pending_response === null
              ? null
              : typeof parsed.pending_response === 'string'
                ? parsed.pending_response
                : JSON.stringify(parsed.pending_response);
          const mutedThreads =
            parsed.muted_threads === undefined || parsed.muted_threads === null
              ? null
              : typeof parsed.muted_threads === 'string'
                ? parsed.muted_threads
                : JSON.stringify(parsed.muted_threads);
          upsertSingleton.run(activeSessionId, pendingResponse, mutedThreads);
          singletonUpserted = true;
        }
      });
      importFile();
      // eslint-disable-next-line no-catch-all/no-catch-all -- handleConstraintViolationOrRethrow rethrows non-constraint errors (see its JSDoc); the linter cannot see through the call
    } catch (err) {
      if (handleConstraintViolationOrRethrow(err, folder, fileLabel, summary))
        continue;
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      fileLabel,
      {
        sessions: sessionCounts,
        singleton: singletonUpserted,
      },
      summary,
    );
  }
  return summary;
}

interface NanoclawStateResumableCycleJson {
  cycle_id?: unknown;
  slot_key?: unknown;
  continuation_n?: unknown;
  remaining_steps?: unknown;
}

interface NanoclawStateJsonShape {
  last_email_checked?: unknown;
  date?: unknown;
  fetched_at?: unknown;
  seen_email_ids?: unknown;
  resumable_cycles?: unknown;
}

/**
 * Migrate per-group `nanoclaw-state.json` files (#297) into the three
 * SQLite tables created by state-005:
 *
 *   - `email_state`        — singleton row (`id = 1`) holding the email-
 *                            cursor fields (`last_email_checked`, `date`,
 *                            `fetched_at`).
 *   - `email_seen_ids`     — append-mostly dedup set; one row per id from
 *                            the JSON `seen_email_ids` array.
 *   - `resumable_cycles`   — one row per skill_name from the JSON
 *                            `resumable_cycles.<skill_name>` subtree.
 *
 * The JSON-era shape was a multi-writer junk drawer (the exact bug class
 * #293 targets), so per-file isolation matters: a single malformed source
 * must not abort the pass for other groups. Each per-file work is wrapped
 * in `db.transaction` so a row violating a NOT NULL / CHECK / PK
 * constraint inside the writer rolls the whole file's import back; the
 * narrowed `handleConstraintViolationOrRethrow` catch turns the throw
 * into a per-file warn and leaves the source file in place for triage.
 *
 * UPSERT semantics:
 *   - `email_state` is a singleton (`CHECK(id = 1)`) so we use
 *     `INSERT … ON CONFLICT(id) DO UPDATE SET …` — NOT `INSERT OR
 *     REPLACE`, which would silently nuke the existing `schema_version`
 *     column instead of preserving it across multi-group re-import. Each
 *     group's nanoclaw-state.json contributes the same singleton row;
 *     under the deterministic folder sort, the last writer (alphabetical
 *     order) wins on the cursor fields. (In practice each install has at
 *     most one nanoclaw-state.json source, so this only matters for
 *     defensive behaviour during multi-group migration.)
 *   - `email_seen_ids` uses `INSERT … ON CONFLICT(email_id) DO NOTHING` —
 *     re-runs are silent no-ops on duplicate ids (the table is a dedup
 *     set; re-importing the same id should not bump `seen_at`).
 *   - `resumable_cycles` uses `INSERT … ON CONFLICT(skill_name) DO
 *     UPDATE SET …` — the latest source-file shape wins for each skill,
 *     same alphabetical-folder-sort tie-break as `email_state`.
 *
 * Per-column policy on missing fields: if a top-level cursor field
 * (`last_email_checked` / `date` / `fetched_at`) is absent, omit it from
 * the INSERT column list so the schema's column default (NULL on these
 * three) fires rather than binding `null` ourselves. Same approach the
 * morning-brief import uses for its `added` column with
 * `DEFAULT CURRENT_TIMESTAMP`. For `email_state` the three cursor
 * columns share the same nullable semantics, but documenting the
 * pattern keeps the writer aligned with the schema-default contract for
 * the wider epic. See state-005 doc-header for the rationale on UPSERT
 * vs INSERT OR REPLACE and on the `strftime` defaults.
 */
function migrateNanoclawStateJsonFiles(
  db: Database.Database,
): MigrationSummary {
  const summary = newMigrationSummary('nanoclaw-state');
  const groupFolders = listGroupFoldersForMigration();
  const stamp = migrationDateStamp();

  // email_state singleton UPSERT. Build the prepared statements lazily
  // per writer-column-set so we can omit absent cursor fields and let
  // the schema default fire. Building all four shapes up front keeps
  // the per-file path branchless.
  //
  // The DO UPDATE clause references `excluded.<col>` (SQLite's name for
  // the row that would have been inserted). Crucially it does NOT touch
  // `schema_version` — preserving the existing value across re-import,
  // which is the whole reason we pick UPSERT over INSERT OR REPLACE.
  const insertEmailStateAll = db.prepare(
    `INSERT INTO email_state (id, last_email_checked, date, fetched_at)
     VALUES (1, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       last_email_checked = excluded.last_email_checked,
       date               = excluded.date,
       fetched_at         = excluded.fetched_at`,
  );
  const insertEmailStateLastOnly = db.prepare(
    `INSERT INTO email_state (id, last_email_checked) VALUES (1, ?)
     ON CONFLICT(id) DO UPDATE SET last_email_checked = excluded.last_email_checked`,
  );
  const insertEmailStateDateOnly = db.prepare(
    `INSERT INTO email_state (id, date) VALUES (1, ?)
     ON CONFLICT(id) DO UPDATE SET date = excluded.date`,
  );
  const insertEmailStateFetchedOnly = db.prepare(
    `INSERT INTO email_state (id, fetched_at) VALUES (1, ?)
     ON CONFLICT(id) DO UPDATE SET fetched_at = excluded.fetched_at`,
  );
  const insertEmailStateLastDate = db.prepare(
    `INSERT INTO email_state (id, last_email_checked, date) VALUES (1, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       last_email_checked = excluded.last_email_checked,
       date               = excluded.date`,
  );
  const insertEmailStateLastFetched = db.prepare(
    `INSERT INTO email_state (id, last_email_checked, fetched_at) VALUES (1, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       last_email_checked = excluded.last_email_checked,
       fetched_at         = excluded.fetched_at`,
  );
  const insertEmailStateDateFetched = db.prepare(
    `INSERT INTO email_state (id, date, fetched_at) VALUES (1, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       date       = excluded.date,
       fetched_at = excluded.fetched_at`,
  );

  // email_seen_ids: ON CONFLICT(email_id) DO NOTHING — append-mostly
  // dedup set. Re-imports are silent no-ops on duplicates; we don't
  // bump `seen_at` since the JSON-era source carries no per-id
  // timestamp anyway.
  const insertSeenEmailId = db.prepare(
    `INSERT INTO email_seen_ids (email_id) VALUES (?)
     ON CONFLICT(email_id) DO NOTHING`,
  );

  // resumable_cycles: UPSERT by skill_name — the latest source-file
  // shape wins for each skill. `continuation_n` defaults to 0 in the
  // schema; we still bind explicitly (defaulting to 0 here too) so the
  // writer's column list is uniform across rows. `remaining_steps` is
  // a JSON blob (TEXT) — we store whatever the source shape carries
  // verbatim, including null.
  const upsertResumableCycle = db.prepare(
    `INSERT INTO resumable_cycles
       (skill_name, cycle_id, slot_key, continuation_n, remaining_steps)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(skill_name) DO UPDATE SET
       cycle_id        = excluded.cycle_id,
       slot_key        = excluded.slot_key,
       continuation_n  = excluded.continuation_n,
       remaining_steps = excluded.remaining_steps,
       updated_at      = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`,
  );

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'nanoclaw-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort. Mirror the orders
      // / morning-brief handling — log info on ENOENT, propagate every
      // other errno per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'nanoclaw-state.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(
      raw,
      folder,
      'nanoclaw-state.json',
      summary,
    ) as NanoclawStateJsonShape | null;
    if (parsed === null) continue;

    // Pre-flight: warn for each top-level section that's missing or
    // wrong-shape. The contract is "import what we can" — a missing
    // resumable_cycles object doesn't block email_state or
    // email_seen_ids importing.
    const hasEmailCursorField =
      typeof parsed.last_email_checked === 'string' ||
      typeof parsed.date === 'string' ||
      typeof parsed.fetched_at === 'string';
    if (!hasEmailCursorField) {
      logger.warn(
        { folder },
        'nanoclaw-state.json migration: no email cursor fields (last_email_checked / date / fetched_at) at top level, skipping email_state',
      );
    }
    const seenEmailIds = parsed.seen_email_ids;
    if (!Array.isArray(seenEmailIds)) {
      logger.warn(
        {
          folder,
          seenType: Array.isArray(seenEmailIds) ? 'array' : typeof seenEmailIds,
        },
        'nanoclaw-state.json migration: seen_email_ids missing or not an array, skipping email_seen_ids',
      );
    }
    const resumableCyclesObj = parsed.resumable_cycles;
    if (!isObjectRow(resumableCyclesObj)) {
      logger.warn(
        {
          folder,
          cyclesType:
            resumableCyclesObj === null
              ? 'null'
              : Array.isArray(resumableCyclesObj)
                ? 'array'
                : typeof resumableCyclesObj,
        },
        'nanoclaw-state.json migration: resumable_cycles missing or not an object, skipping resumable_cycles',
      );
    }

    if (
      !hasEmailCursorField &&
      !Array.isArray(seenEmailIds) &&
      !isObjectRow(resumableCyclesObj)
    ) {
      // Nothing recognised — leave file in place for triage rather than
      // renaming a source we never actually imported.
      logger.warn(
        { folder },
        'nanoclaw-state.json migration: no recognised sections, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const counts = {
      email_state_upserted: 0,
      seen_ids_inserted: 0,
      seen_ids_skipped: 0,
      cycles_upserted: 0,
      cycles_skipped: 0,
    };

    try {
      const importFile = db.transaction(() => {
        // email_state singleton — only INSERT/UPSERT if at least one
        // cursor field is present. Pick the prepared statement
        // matching the present-fields combination so absent fields
        // are omitted from the column list (and the schema default
        // fires / column stays NULL) rather than binding null.
        if (hasEmailCursorField) {
          const lec =
            typeof parsed.last_email_checked === 'string'
              ? parsed.last_email_checked
              : null;
          const dt = typeof parsed.date === 'string' ? parsed.date : null;
          const fa =
            typeof parsed.fetched_at === 'string' ? parsed.fetched_at : null;
          if (lec !== null && dt !== null && fa !== null) {
            insertEmailStateAll.run(lec, dt, fa);
          } else if (lec !== null && dt !== null) {
            insertEmailStateLastDate.run(lec, dt);
          } else if (lec !== null && fa !== null) {
            insertEmailStateLastFetched.run(lec, fa);
          } else if (dt !== null && fa !== null) {
            insertEmailStateDateFetched.run(dt, fa);
          } else if (lec !== null) {
            insertEmailStateLastOnly.run(lec);
          } else if (dt !== null) {
            insertEmailStateDateOnly.run(dt);
          } else if (fa !== null) {
            insertEmailStateFetchedOnly.run(fa);
          }
          counts.email_state_upserted = 1;
        }

        // email_seen_ids — one row per string in the JSON array; per-row
        // type guard skips non-string entries (the email_id PK is
        // TEXT NOT NULL; binding e.g. a number would coerce silently).
        if (Array.isArray(seenEmailIds)) {
          for (const id of seenEmailIds) {
            if (typeof id !== 'string') {
              logger.warn(
                { folder, idType: id === null ? 'null' : typeof id },
                'nanoclaw-state.json migration: skipping non-string entry in seen_email_ids',
              );
              counts.seen_ids_skipped++;
              continue;
            }
            const result = insertSeenEmailId.run(id);
            if (result.changes > 0) counts.seen_ids_inserted++;
            else counts.seen_ids_skipped++;
          }
        }

        // resumable_cycles — one row per skill_name in the JSON object.
        // Per-row object guard skips stale null/string/number values
        // (a writer bug could plant those; without the guard a property
        // access would TypeError before any INSERT runs and propagate
        // as an unexpected error through the narrowed catch).
        if (isObjectRow(resumableCyclesObj)) {
          for (const [skillName, cycleRecord] of Object.entries(
            resumableCyclesObj,
          )) {
            if (!isObjectRow(cycleRecord)) {
              logger.warn(
                { folder, skillName },
                'nanoclaw-state.json migration: skipping non-object resumable_cycles entry',
              );
              counts.cycles_skipped++;
              continue;
            }
            const cycle = cycleRecord as NanoclawStateResumableCycleJson;
            // cycle_id and slot_key are NOT NULL on the schema; let the
            // constraint catch missing values (handled by the narrowed
            // catch below — file left in place for triage).
            const cycleId =
              typeof cycle.cycle_id === 'string' ? cycle.cycle_id : null;
            const slotKey =
              typeof cycle.slot_key === 'string' ? cycle.slot_key : null;
            const continuationN =
              typeof cycle.continuation_n === 'number'
                ? cycle.continuation_n
                : 0;
            // remaining_steps is a JSON blob — schema column is TEXT so
            // accept either a string (already-stringified) or stringify
            // an object/array on the way in. null stays null.
            let remainingSteps: string | null;
            if (
              cycle.remaining_steps === null ||
              cycle.remaining_steps === undefined
            ) {
              remainingSteps = null;
            } else if (typeof cycle.remaining_steps === 'string') {
              remainingSteps = cycle.remaining_steps;
            } else {
              remainingSteps = JSON.stringify(cycle.remaining_steps);
            }
            upsertResumableCycle.run(
              skillName,
              cycleId,
              slotKey,
              continuationN,
              remainingSteps,
            );
            counts.cycles_upserted++;
          }
        }
      });
      importFile();
      // eslint-disable-next-line no-catch-all/no-catch-all -- handleConstraintViolationOrRethrow rethrows non-constraint errors (see its JSDoc); the linter cannot see through the call
    } catch (err) {
      // Per-file isolation: a constraint violation on any of the three
      // writers rolls the whole transaction back, the file stays put
      // for human triage, and we move on to the next group. Anything
      // else (TypeError, ReferenceError, non-constraint SqliteError)
      // propagates per `coding-policy: error-handling`.
      if (
        handleConstraintViolationOrRethrow(
          err,
          folder,
          'nanoclaw-state.json',
          summary,
        )
      ) {
        continue;
      }
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      'nanoclaw-state.json',
      counts,
      summary,
    );
  }
  return summary;
}

interface ScheduledReminderJson {
  event_id: string;
  title: string;
  utc_time: string;
  reminder_offset_min: number;
  task_id: string;
}

/**
 * A scheduled-reminders row is bindable into `scheduled_reminders` only
 * when every NOT NULL column has a present, correctly-typed value. The
 * JSON-era writer emitted rows with a null/missing `reminder_offset_min`
 * (observed in production); binding that throws a NOT NULL `SqliteError`
 * that rolls the whole-file transaction back (#676). Every source column
 * is NOT NULL (`created_at` / `schema_version` are defaulted, not
 * sourced), so all five fields are required.
 */
function missingRequiredScheduledReminderFields(
  row: Record<string, unknown>,
): string[] {
  const missing: string[] = [];
  if (typeof row.event_id !== 'string') missing.push('event_id');
  if (typeof row.title !== 'string') missing.push('title');
  if (typeof row.utc_time !== 'string') missing.push('utc_time');
  if (
    typeof row.reminder_offset_min !== 'number' ||
    !Number.isFinite(row.reminder_offset_min)
  ) {
    missing.push('reminder_offset_min');
  }
  if (typeof row.task_id !== 'string') missing.push('task_id');
  return missing;
}

/**
 * Per-group import of `scheduled-reminders.json` into the
 * `scheduled_reminders` table created by state-004 (#296). The JSON-era
 * shape is the wrapped form `{ "reminders": [...] }` written by the
 * pre-MCP `append-scheduled-reminders.py` skill; a bare top-level array
 * is also accepted as the legacy fallback the same skill emitted in
 * earlier revisions. Anything else (object without `reminders`, primitive
 * payload, malformed JSON) is warn-and-skipped — the source file stays
 * in place for triage so an operator can fix it without losing data.
 *
 * `event_id` is the PK on the table and the natural dedup key on the
 * source side (each reminder corresponds to exactly one calendar event).
 * The INSERT uses `ON CONFLICT(event_id) DO NOTHING` so a re-run with
 * leftover rows (e.g. operator copied a partial DB back over an already-
 * imported one) is a silent no-op on the PK conflict. A row missing a
 * required NOT NULL field is validated out and skipped per row (#676) —
 * chiefly a null `reminder_offset_min` from the JSON-era writer — rather
 * than letting the bind throw a NOT NULL `SqliteError` that rolls the
 * WHOLE file back: that rollback lost every good reminder in the file
 * AND, because the catch `continue`d before the source was renamed,
 * re-threw on every startup. Per-file work is wrapped in a single
 * transaction so a mid-import crash can't leave the table half-populated;
 * the constraint-class catch helper stays as a defensive net for any
 * violation the per-row guard doesn't anticipate.
 */
function migrateScheduledRemindersJsonFiles(
  db: Database.Database,
): MigrationSummary {
  const summary = newMigrationSummary('scheduled-reminders');
  const groupFolders = listGroupFoldersForMigration();

  const insertReminder = db.prepare(
    `INSERT INTO scheduled_reminders
       (event_id, title, utc_time, reminder_offset_min, task_id)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(event_id) DO NOTHING`,
  );

  const stamp = migrationDateStamp();
  const fileLabel = 'scheduled-reminders.json';

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'scheduled-reminders.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort; the file may
      // disappear between the check and the read (concurrent migration
      // run, manual cleanup). Treat ENOENT as an idempotent no-op and
      // propagate every other errno per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          `${fileLabel} migration: file disappeared between existsSync and readFileSync, skipping`,
        );
        continue;
      }
      throw err;
    }

    // Accept two shapes: the wrapped `{ "reminders": [...] }` form
    // (preferred — the form `append-scheduled-reminders.py` settled on)
    // and a bare top-level array (legacy fallback emitted by earlier
    // revisions of the same skill). For the wrapped shape we delegate
    // to `parseJsonObjectOrWarn` so the malformed-JSON / non-object
    // warn shapes match every other per-group migration. For the bare-
    // array shape we parse inline because the helper (correctly) treats
    // top-level arrays as "not an object" and rejects them — here a
    // bare array is a documented legacy form we still consume.
    //
    // Malformed JSON, primitive payloads, and objects without a
    // `reminders` array are all warn-and-skipped (file left in place
    // for triage) per `coding-policy: error-handling`.
    let reminders: unknown[];
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      if (err instanceof SyntaxError) {
        // Re-route through the helper for the malformed-JSON warn so
        // log shape stays consistent across all per-group migrations.
        // Helper detects SyntaxError, emits the standard warn, returns
        // null; skip the folder.
        parseJsonObjectOrWarn(raw, folder, fileLabel, summary);
        continue;
      }
      // `JSON.parse` only throws SyntaxError on string input;
      // anything else is a programming bug. Propagate per
      // `coding-policy: error-handling`.
      throw err;
    }
    if (Array.isArray(parsed)) {
      reminders = parsed;
    } else if (
      parsed !== null &&
      typeof parsed === 'object' &&
      Array.isArray((parsed as Record<string, unknown>).reminders)
    ) {
      reminders = (parsed as { reminders: unknown[] }).reminders;
    } else if (
      parsed !== null &&
      typeof parsed === 'object' &&
      !Array.isArray(parsed)
    ) {
      logger.warn(
        { folder },
        `${fileLabel} migration: missing "reminders" array, skipping (file left in place)`,
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    } else {
      // Primitive payload (null / number / string / boolean). Use the
      // helper so the warn carries the same `parsedType` field shape as
      // every other per-group migration.
      parseJsonObjectOrWarn(raw, folder, fileLabel, summary);
      continue;
    }

    // Single transaction per file so a mid-import crash can't leave the
    // table half-populated. Per-row object guard skips stale primitives
    // (null/string/number) before the bind throws a TypeError that the
    // narrowed catch would otherwise propagate as "unexpected".
    const counts = { inserted: 0, skipped: 0, total: reminders.length };
    try {
      const importFile = db.transaction(() => {
        for (const reminder of reminders) {
          if (!isObjectRow(reminder)) {
            logger.warn(
              { folder },
              `${fileLabel} migration: skipping non-object row`,
            );
            counts.skipped++;
            continue;
          }
          const missing = missingRequiredScheduledReminderFields(reminder);
          if (missing.length > 0) {
            // #676 — skip a row missing a required NOT NULL field rather
            // than let the bind throw a NOT NULL SqliteError that rolls
            // the whole file back (losing every good reminder in it and
            // re-throwing every startup because the source never renamed).
            logger.warn(
              {
                folder,
                missing,
                ...(typeof reminder.event_id === 'string'
                  ? { event_id: reminder.event_id }
                  : {}),
              },
              `${fileLabel} migration: skipping row missing required NOT NULL field(s)`,
            );
            counts.skipped++;
            continue;
          }
          const row = reminder as unknown as ScheduledReminderJson;
          const result = insertReminder.run(
            row.event_id,
            row.title,
            row.utc_time,
            row.reminder_offset_min,
            row.task_id,
          );
          if (result.changes > 0) counts.inserted++;
          else counts.skipped++;
        }
      });
      importFile();
      // eslint-disable-next-line no-catch-all/no-catch-all -- handleConstraintViolationOrRethrow rethrows non-constraint errors (see its JSDoc); the linter cannot see through the call
    } catch (err) {
      if (handleConstraintViolationOrRethrow(err, folder, fileLabel, summary))
        continue;
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      fileLabel,
      {
        inserted: counts.inserted,
        skipped: counts.skipped,
        total: counts.total,
      },
      summary,
    );
  }
  return summary;
}

// --- email-feedback.json → email_feedback (#295) ---

/**
 * Per-group migration: read each group's `email-feedback.json` and
 * append every well-formed row into the `email_feedback` SQLite table
 * created by state-002 (+ state-003 added the per-record
 * `schema_version` column). The JSON-era shape carried by
 * `nanoclaw-admin/skills/brief-cleanup` evolved across two forms:
 *
 *   - **Wrapped (preferred):** `{"feedback": [ { pattern, label,
 *     source, date }, ... ]}` — what the SKILL Step 6 helper
 *     `append-feedback.py` writes today.
 *   - **Bare array (legacy):** `[ { pattern, label, source, date },
 *     ... ]` — observed in production where earlier writers omitted
 *     the wrapper. Issue #295's body documents this shape verbatim.
 *
 * Both shapes are accepted; bare arrays are treated as if they were
 * `{feedback: <array>}`. Anything else (null, number, string, plain
 * object missing the `feedback` key, malformed JSON) is warned-and-
 * skipped per `coding-policy: error-handling` ("try alternatives
 * before failing"); the source file stays in place for human triage.
 *
 * Append-only contract: the schema's `id INTEGER PRIMARY KEY
 * AUTOINCREMENT` is assigned by SQLite, the writer never supplies it,
 * and there is no natural-key uniqueness to deduplicate on. So no
 * `ON CONFLICT` clause — every well-formed row inserts. Idempotency
 * comes from the rename: once the source becomes
 * `email-feedback.json.migrated-<YYYY-MM-DD>`, the existsSync gate at
 * the top of the loop skips it on subsequent boots. Re-running after
 * a fresh JSON has been dropped over an already-imported DB would
 * double-insert; that's the operator's problem, documented at the
 * dispatch site in `migrateJsonState()`.
 *
 * Per-row missing-required handling: rows lacking `pattern`, `label`,
 * or `date` are skipped with a warn rather than letting the schema's
 * NOT NULL throw — the warn carries the field name so triage can
 * grep for the specific failure class. The schema's
 * `CHECK(label IN ('actionable', 'noise'))` violation still throws as
 * a `SqliteError` with a `SQLITE_CONSTRAINT_CHECK` code; that's
 * caught by `handleConstraintViolationOrRethrow` so the per-file
 * transaction rolls back and the source file stays put.
 *
 * The `source` column has a DDL DEFAULT of `'baruch-response'`. When
 * a JSON-era row omits `source`, we omit that column from the INSERT
 * (rather than passing `null`, which the NOT NULL constraint would
 * reject) so the schema default fires. Matches the morning-brief
 * migration's `added`/CURRENT_TIMESTAMP pattern.
 *
 * The `schema_version` column (added by state-003) has a DDL DEFAULT
 * of `1`. Every JSON-era row was written under contract v1, so we
 * always omit the column from the INSERT and let the default fire —
 * no per-row stamping needed at migration time.
 */
function migrateEmailFeedbackJsonFiles(
  db: Database.Database,
): MigrationSummary {
  const summary = newMigrationSummary('email-feedback');
  const groupFolders = listGroupFoldersForMigration();
  if (groupFolders.length === 0) return summary;

  // Two prepared statements: one with `source`, one without, so the
  // schema default fires when the JSON-era row omitted the field.
  // Same pattern as the morning-brief migration's
  // `withAdded`/`DefaultAdded` split.
  const insertWithSource = db.prepare(
    `INSERT INTO email_feedback (pattern, label, source, date)
     VALUES (?, ?, ?, ?)`,
  );
  const insertDefaultSource = db.prepare(
    `INSERT INTO email_feedback (pattern, label, date)
     VALUES (?, ?, ?)`,
  );

  const stamp = migrationDateStamp();

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'email-feedback.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race between existsSync and readFileSync — file
      // disappeared. Idempotent no-op; every other errno propagates
      // per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'email-feedback.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    // Two-shape parse: bare array (legacy) and `{feedback: [...]}`
    // (preferred). The shared `parseJsonObjectOrWarn` helper would
    // reject bare arrays with a warn, so do the parse + dispatch
    // inline. Same warn vocabulary as the helper so triage greps
    // (`'invalid JSON'`, `'payload is not'`) match either path.
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      if (err instanceof SyntaxError) {
        logger.warn(
          { folder, errName: err.name },
          'email-feedback.json migration: invalid JSON, skipping (file left in place)',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      throw err;
    }

    let feedback: unknown[];
    if (Array.isArray(parsed)) {
      feedback = parsed;
    } else if (parsed !== null && typeof parsed === 'object') {
      const wrapper = parsed as Record<string, unknown>;
      if (!Array.isArray(wrapper.feedback)) {
        logger.warn(
          { folder },
          'email-feedback.json migration: payload is an object but `feedback` is not an array, skipping (file left in place)',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      feedback = wrapper.feedback;
    } else {
      logger.warn(
        {
          folder,
          parsedType: parsed === null ? 'null' : typeof parsed,
        },
        'email-feedback.json migration: payload is not an object or array, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const counts = { inserted: 0, skipped: 0, total: feedback.length };
    try {
      const importFile = db.transaction(() => {
        for (const row of feedback) {
          if (!isObjectRow(row)) {
            logger.warn(
              { folder },
              'email-feedback.json migration: skipping non-object row',
            );
            counts.skipped++;
            continue;
          }
          // Required-field guard: the schema's NOT NULL on pattern /
          // label / date would throw on missing values, rolling back
          // the entire per-file transaction. Skip the row with a
          // warn instead so a single malformed entry doesn't
          // poison-pill the whole file.
          const pattern = row.pattern;
          const label = row.label;
          const date = row.date;
          if (
            typeof pattern !== 'string' ||
            typeof label !== 'string' ||
            typeof date !== 'string'
          ) {
            const missing: string[] = [];
            if (typeof pattern !== 'string') missing.push('pattern');
            if (typeof label !== 'string') missing.push('label');
            if (typeof date !== 'string') missing.push('date');
            logger.warn(
              { folder, missing },
              'email-feedback.json migration: skipping row missing required fields',
            );
            counts.skipped++;
            continue;
          }
          // Omit `source` from the INSERT when the JSON-era row
          // didn't set it, so the schema's
          // `DEFAULT 'baruch-response'` fires. Mirrors the morning-
          // brief migration's `withAdded`/`DefaultAdded` split.
          if (typeof row.source === 'string') {
            insertWithSource.run(pattern, label, row.source, date);
          } else {
            insertDefaultSource.run(pattern, label, date);
          }
          counts.inserted++;
        }
      });
      importFile();
      // eslint-disable-next-line no-catch-all/no-catch-all -- the helper rethrows non-constraint errors via `throw err` (see `handleConstraintViolationOrRethrow` JSDoc); the lint rule can't see through the call.
    } catch (err) {
      // Constraint-class SqliteError (CHECK on label, NOT NULL we
      // didn't pre-guard, etc.) → warn-and-continue, source file
      // stays put for triage. Anything else (programming bug,
      // SQLITE_CORRUPT, SQLITE_BUSY) propagates via the helper.
      if (
        handleConstraintViolationOrRethrow(
          err,
          folder,
          'email-feedback.json',
          summary,
        )
      ) {
        continue;
      }
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      'email-feedback.json',
      counts,
      summary,
    );
  }
  return summary;
}
