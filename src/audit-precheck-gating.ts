/**
 * Snapshot active recurring tasks + run-log stats for the
 * precheck-gating audit (#375). Pure data fetch — no judgment about
 * drift; the audit-replay agent applies that judgment when diffing
 * the snapshot against `docs/precheck-gating-audit.md`.
 *
 * Lives in `src/` so vitest tests can import the core; the CLI shim
 * at `scripts/audit-precheck-gating.ts` re-exports the entry point.
 *
 * Selects every `scheduled_tasks` row with `status='active'` AND
 * `schedule_type IN ('cron','interval')`, joins each row with
 * `task_run_logs` stats over a window ending now and starting `days`
 * days back (default 90 — quarterly cadence per #375). Window default
 * is 90 days because shorter windows make per-task fire counts too
 * small to detect duration-profile drift on the cron-weekly tasks
 * (only 4 fires per 30 days).
 *
 * `gated_likely` is a heuristic populated from a duration cutoff —
 * NOT ground truth. Real gated runs share the short-`duration_ms`
 * signature with quick-failing runs, and the canonical signal is
 * `result=null`+short-duration combined per the audit doc's "How the
 * gate actually works" section. The replay agent is expected to
 * verify before reporting.
 */
import Database from 'better-sqlite3';
import fs from 'fs';

// Heuristic threshold for `gated_likely`: a run with `duration_ms`
// below this is *probably* precheck-gated (precheck-skip path returns
// within seconds of container spawn). Documented in the audit doc.
export const GATED_DURATION_MS_THRESHOLD = 10_000;

export interface TaskSnapshot {
  task_id: string;
  group_folder: string;
  schedule_type: 'cron' | 'interval';
  schedule_value: string;
  has_precheck: boolean;
  script: string | null;
  fires: number;
  gated_likely: number;
  avg_duration_s: number | null;
  min_duration_s: number | null;
  max_duration_s: number | null;
}

export interface AuditSnapshot {
  snapshot_at: string;
  window_start: string;
  window_end: string;
  window_days: number;
  tasks: TaskSnapshot[];
}

function isoUtc(d: Date): string {
  return d.toISOString().replace(/\.\d{3}Z$/, 'Z');
}

export function runAuditSnapshot(args: {
  dbPath: string;
  windowDays?: number;
  now?: Date;
}): AuditSnapshot {
  const windowDays = args.windowDays ?? 90;
  if (!Number.isInteger(windowDays) || windowDays <= 0) {
    throw new Error(
      `audit-precheck-gating: windowDays must be a positive integer (got ${windowDays})`,
    );
  }
  if (!fs.existsSync(args.dbPath)) {
    throw new Error(
      `audit-precheck-gating: db not found at ${args.dbPath} — pass --db pointing at messages.db`,
    );
  }

  const end = args.now ?? new Date();
  const endIso = isoUtc(end);
  const start = new Date(end.getTime() - windowDays * 86_400_000);
  const startIso = isoUtc(start);

  const db = new Database(args.dbPath, { readonly: true });
  try {
    type TaskRow = {
      id: string;
      group_folder: string;
      schedule_type: string;
      schedule_value: string;
      script: string | null;
    };
    const taskRows = db
      .prepare(
        `SELECT id, group_folder, schedule_type, schedule_value, script
           FROM scheduled_tasks
          WHERE status = 'active'
            AND schedule_type IN ('cron', 'interval')
          ORDER BY group_folder, schedule_type, id`,
      )
      .all() as TaskRow[];

    const statsStmt = db.prepare(
      `SELECT COUNT(*)                                          AS fires,
              SUM(CASE WHEN duration_ms < ? THEN 1 ELSE 0 END)  AS gated_likely,
              AVG(duration_ms)                                  AS avg_ms,
              MIN(duration_ms)                                  AS min_ms,
              MAX(duration_ms)                                  AS max_ms
         FROM task_run_logs
        WHERE task_id = ?
          AND run_at >= ?
          AND run_at <  ?`,
    );

    const tasks: TaskSnapshot[] = taskRows.map((t) => {
      type StatsRow = {
        fires: number | null;
        gated_likely: number | null;
        avg_ms: number | null;
        min_ms: number | null;
        max_ms: number | null;
      };
      const stats = statsStmt.get(
        GATED_DURATION_MS_THRESHOLD,
        t.id,
        startIso,
        endIso,
      ) as StatsRow;
      return {
        task_id: t.id,
        group_folder: t.group_folder,
        schedule_type: t.schedule_type as 'cron' | 'interval',
        schedule_value: t.schedule_value,
        has_precheck: Boolean(t.script),
        script: t.script,
        fires: stats.fires ?? 0,
        gated_likely: stats.gated_likely ?? 0,
        avg_duration_s: stats.avg_ms !== null ? stats.avg_ms / 1000 : null,
        min_duration_s: stats.min_ms !== null ? stats.min_ms / 1000 : null,
        max_duration_s: stats.max_ms !== null ? stats.max_ms / 1000 : null,
      };
    });

    return {
      snapshot_at: endIso,
      window_start: startIso,
      window_end: endIso,
      window_days: windowDays,
      tasks,
    };
  } finally {
    db.close();
  }
}
