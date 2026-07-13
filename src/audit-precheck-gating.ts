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
 * `gated_likely` is a heuristic that ORs three signatures:
 *   1. Canonical (post-#581-followup): rows with
 *      `status='precheck_skipped'` — emitted by the agent-runner's
 *      `runScript` branch when the precheck script returned
 *      `wake_agent: false`. This is ground truth, not a heuristic;
 *      the row was a gate-out by construction.
 *   1b. Host pre-spawn gate (#754): rows with
 *      `status='skipped_out_of_window'` — the host's pre-spawn
 *      eligibility gate declined to spawn at all (e.g. flight-assist
 *      firing outside any trip window). Ground truth like (1), and an
 *      even stronger gate-out: no container spawned. Without counting
 *      it, these fires would inflate the denominator while showing as
 *      ungated — making a windowed cadence look like a hot task.
 *   2. Legacy (pre-#581-followup): rows with `status='success' AND
 *      result IS NULL` paired with a short `duration_ms`. The
 *      agent-runner used to collapse every `wake_agent: false`
 *      decision to that shape, indistinguishable from a wake-and-
 *      empty bug; the duration cutoff was the only post-hoc
 *      discriminator. Historical rows are this shape.
 *
 * The replay agent is expected to verify the legacy branch before
 * reporting (the canonical branch is ground truth).
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

    // `gated_likely` counts both the legacy and the post-#581 gate
    // signatures so historical and current rows both surface in the
    // quarterly audit.
    //
    //   - Legacy (pre-#581-followup): `status='success' AND result IS
    //     NULL` paired with a short duration. The agent-runner used
    //     to emit `writeOutput({status: 'success', result: null})` on
    //     EVERY `wake_agent: false` decision, so the historical rows
    //     are all this shape.
    //
    //   - Canonical (post-#581-followup): `status='precheck_skipped'`
    //     regardless of duration. The agent-runner now emits a
    //     dedicated status + non-null `<internal>precheck-skipped:
    //     ...</internal>` result diagnostic, so the canonical shape
    //     does NOT require the duration cutoff or a NULL result. The
    //     status alone is the gate-out signal.
    //
    //   - Host pre-spawn gate (#754): `status='skipped_out_of_window'`
    //     regardless of duration — the host declined to spawn, so this
    //     is a gate-out by construction like the canonical branch.
    //
    // Counting every short run would conflate fast failures
    // (status='error') with genuine gate-outs and inflate the
    // heuristic, so the legacy branch keeps the duration + NULL gate
    // and the ground-truth branches key off the explicit status.
    const statsStmt = db.prepare(
      `SELECT COUNT(*) AS fires,
              SUM(CASE
                    WHEN status IN ('precheck_skipped', 'skipped_out_of_window')
                    THEN 1
                    WHEN duration_ms < ?
                     AND status = 'success'
                     AND result IS NULL
                    THEN 1
                    ELSE 0
                  END) AS gated_likely,
              AVG(duration_ms) AS avg_ms,
              MIN(duration_ms) AS min_ms,
              MAX(duration_ms) AS max_ms
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
