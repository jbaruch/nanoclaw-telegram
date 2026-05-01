/**
 * Core SQLite-state-table dump logic for the github_backup pipeline.
 *
 * Originally landed under `scripts/dump-state-tables.ts` (#398) as a
 * standalone CLI, then moved here in #397 so production code (the
 * `github_backup` IPC handler in `src/ipc.ts`) can import the same
 * `runDumpPlan` the CLI uses — `tsconfig.json`'s `rootDir: ./src`
 * forbids importing from `scripts/`. The CLI shim at
 * `scripts/dump-state-tables.ts` is now a thin entry-point wrapper
 * around this module; tests live alongside as `db-dump.test.ts`.
 *
 * After the state-001…state-010 epic moved nine JSON state files into
 * `store/messages.db`, the live writers stopped writing JSON to disk
 * and `groups/<group>/backup-repo/state/` froze. `runDumpPlan`
 * produces text-form SQL dumps (`.dump`) per table so the existing
 * `git add -A && git diff --cached --quiet` pipeline keeps producing
 * meaningful per-day diffs and a destructive-test replay (`cat
 * <table>.sql | sqlite3 fresh.db`) reconstitutes the exact rows.
 *
 * Tables not present in the DB (typical when a group hasn't run a
 * particular state-NNN migration yet) land in `report.skipped` and
 * write nothing — no empty-shell `.sql` files.
 */
import { execFileSync } from 'child_process';
import fs from 'fs';
import path from 'path';

// Tables that comprise the irreplaceable state surface — the queue
// and config tables called out in #398. Adding a new table here
// extends the dump contract: "everything in STATE_TABLES is
// recoverable from <outDir>/<table>.sql".
//
// Intentionally excluded by default: bulk caches that are recoverable
// from their upstream source — `messages`, `chats`, `reactions`
// (Telegram is the source of truth for the message log), `sessions`
// (ephemeral SDK session cache), `smart_home_events` (recoverable
// from Hubitat). They're also large enough that buffering `.dump`
// output through `maxBuffer` would either OOM or produce dumps that
// don't diff usefully in `backup-repo`. Pass `--tables messages`
// (etc.) to dump them explicitly.
export const STATE_TABLES: readonly string[] = [
  // state-001
  'orders',
  'orders_metadata',
  // state-002 / state-003
  'email_feedback',
  // state-004
  'scheduled_reminders',
  // state-005
  'email_state',
  'email_seen_ids',
  'resumable_cycles',
  // state-006
  'trusted_sessions',
  'trusted_session_singleton',
  // state-007
  'pending_cleanup_items',
  'pending_decisions',
  'pending_undated_tasks',
  // state-008
  'calendar_snapshots',
  'calendar_events',
  // state-009
  'phase_completions',
  // state-010
  'tz_state',
  'follow_me_tasks',
  // src/db.ts createSchema — small queue/log tables called out in #398
  'scheduled_tasks',
  'task_run_logs',
];

// SQLite identifiers may contain letters, digits, and underscores,
// and must not start with a digit. Restrictive but covers every
// table this codebase has ever defined and rejects metacharacters
// (`;`, `'`, spaces, ` -- `, etc.) that the `sqlite3` CLI's command
// parser would otherwise interpret. Operator-supplied `--tables`
// values are validated against this regex; failure aborts the run
// rather than silently dropping the offender.
const TABLE_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

function validateTableNames(tables: readonly string[]): void {
  for (const t of tables) {
    if (!TABLE_NAME_RE.test(t)) {
      throw new Error(
        `dump-state-tables: invalid table name ${JSON.stringify(t)} — must match /^[A-Za-z_][A-Za-z0-9_]*$/. Pass only real SQLite table identifiers via --tables.`,
      );
    }
  }
}

export interface DumpReport {
  dumped: string[];
  skipped: string[];
}

function preflightSqlite3Cli(): void {
  try {
    execFileSync('sqlite3', ['--version'], {
      encoding: 'utf8',
      timeout: 5_000,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } catch (e) {
    if (e && typeof e === 'object' && 'code' in e && (e as { code: unknown }).code === 'ENOENT') {
      throw new Error(
        'dump-state-tables: sqlite3 CLI not on PATH — install via your package manager (apt: sqlite3, brew: sqlite, Synology: SynoCommunity sqlite3)',
        { cause: e },
      );
    }
    throw e;
  }
}

function tableExists(dbPath: string, table: string): boolean {
  // Table names are sourced from STATE_TABLES (a literal constant in
  // this file) or an operator-supplied --tables list (validated
  // against TABLE_NAME_RE) — never from untrusted input — so
  // single-quote escaping is sufficient.
  const escaped = table.replace(/'/g, "''");
  const out = execFileSync(
    'sqlite3',
    [
      dbPath,
      `SELECT 1 FROM sqlite_master WHERE type='table' AND name='${escaped}' LIMIT 1;`,
    ],
    { encoding: 'utf8', timeout: 30_000 },
  );
  return out.trim() === '1';
}

function dumpOne(dbPath: string, table: string): string {
  // `.dump` does not accept bind params; the table name is sourced
  // from STATE_TABLES or an operator-supplied --tables list
  // (validated against TABLE_NAME_RE) — never from untrusted input.
  return execFileSync('sqlite3', [dbPath, `.dump ${table}`], {
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024,
    timeout: 120_000,
  });
}

function atomicWrite(outFile: string, content: string): void {
  const tmp = `${outFile}.tmp`;
  fs.writeFileSync(tmp, content);
  fs.renameSync(tmp, outFile);
}

export function runDumpPlan(args: {
  dbPath: string;
  outDir: string;
  tables?: readonly string[];
}): DumpReport {
  const tables = args.tables ?? STATE_TABLES;
  validateTableNames(tables);
  if (!fs.existsSync(args.dbPath)) {
    throw new Error(
      `dump-state-tables: db not found at ${args.dbPath} — pass --db-path pointing at a SQLite file (default project layout: store/messages.db)`,
    );
  }
  preflightSqlite3Cli();
  fs.mkdirSync(args.outDir, { recursive: true });

  const report: DumpReport = { dumped: [], skipped: [] };
  for (const table of tables) {
    if (!tableExists(args.dbPath, table)) {
      report.skipped.push(table);
      continue;
    }
    const sql = dumpOne(args.dbPath, table);
    atomicWrite(path.join(args.outDir, `${table}.sql`), sql);
    report.dumped.push(table);
  }
  return report;
}
