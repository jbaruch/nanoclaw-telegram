// Scheduled-task and task-run-log accessors (#751 seam 7, extracted
// verbatim from src/db.ts).
//
// Covers the `scheduled_tasks` and `task_run_logs` tables: task CRUD,
// scheduler selectors (due/dormant/zombie rows), per-task session and
// agent-model bookkeeping, and the cadence-registry rebuild glue that
// upserts `source = 'cadence-registry'` rows for a group. The
// `follow_me_tasks` pending-run helpers stay with the timezone/location
// seam — follow-me is tz domain, not scheduler domain.
import {
  rebuildCadenceRegistry,
  type CadenceRegistryDeps,
  type CadenceRegistryRebuildResult,
} from './cadence-registry.js';
import { db, isDbHandleRegistered } from './db-connection.js';
import { ScheduledTask, TaskRunLog } from './types.js';

export function createTask(
  task: Omit<ScheduledTask, 'last_run' | 'last_result'>,
): void {
  db.prepare(
    `
    INSERT INTO scheduled_tasks (id, group_folder, chat_jid, prompt, script, schedule_type, schedule_value, schedule_timezone, context_mode, next_run, status, created_at, created_by_role, continuation_cycle_id, agent_model)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `,
  ).run(
    task.id,
    task.group_folder,
    task.chat_jid,
    task.prompt,
    task.script || null,
    task.schedule_type,
    task.schedule_value,
    task.schedule_timezone || null,
    task.context_mode || 'isolated',
    task.next_run,
    task.status,
    task.created_at,
    task.created_by_role,
    task.continuation_cycle_id || null,
    task.agent_model || null,
  );
}

/**
 * Phase 2 of #305 — idempotent rebuild of cadence-registry rows for one
 * group, called from the container-spawn flow after the per-session
 * skills tree has been published to disk. The cadence-registry module
 * itself is db-handle-agnostic (it accepts the db as a dep, which keeps
 * its unit tests free of singleton state); this wrapper plumbs the
 * module-private `db` handle through so callers don't reach into
 * internals just to invoke the rebuild.
 *
 * Upsert-shaped: a second call with the same SKILL.md frontmatter
 * leaves existing rows alone (preserving `next_run`, `session_id`,
 * `last_run`, `last_result`); only declarations whose cadence string
 * or prompt changed are rewritten, only orphaned rows are deleted.
 * Owner-scheduled rows (`source = 'schedule-task'`) are untouched —
 * the rebuild keys on `source = 'cadence-registry'`.
 *
 * Throws explicitly when called before `initDatabase` so tests that
 * simulate the spawn path without initialising the DB get a
 * function-specific actionable error instead of the generic
 * db-connection sentinel error firing on `db.transaction` deep inside
 * the cadence-registry. Per
 * `coding-policy: error-handling`, an unexpected initialisation
 * failure must propagate, not be papered over with a synthetic
 * success — production paths always init before spawn is reachable
 * so this branch never fires there; a test that hits it should
 * `vi.mock('./db-tasks.js', ...)` the wrapper alongside its other module
 * mocks.
 */
export function rebuildCadenceRegistryForGroup(
  opts: Omit<CadenceRegistryDeps, 'db'>,
): CadenceRegistryRebuildResult {
  // `db` is a live binding whose pre-init value is an always-truthy
  // Proxy sentinel (see db-connection.ts), so `!db` can never detect
  // the uninitialised state — ask the registration flag instead.
  if (!isDbHandleRegistered()) {
    throw new Error(
      `rebuildCadenceRegistryForGroup called before initDatabase (groupFolder=${opts.groupFolder}). ` +
        `Production always invokes initDatabase() in src/index.ts startup before runContainerAgent is reachable; ` +
        `if you're seeing this in a test, mock ./db-tasks.js's rebuildCadenceRegistryForGroup alongside the test's other module mocks.`,
    );
  }
  return rebuildCadenceRegistry({ ...opts, db });
}

export function getTaskById(id: string): ScheduledTask | undefined {
  return db.prepare('SELECT * FROM scheduled_tasks WHERE id = ?').get(id) as
    | ScheduledTask
    | undefined;
}

export function getTasksForGroup(groupFolder: string): ScheduledTask[] {
  return db
    .prepare(
      'SELECT * FROM scheduled_tasks WHERE group_folder = ? ORDER BY created_at DESC',
    )
    .all(groupFolder) as ScheduledTask[];
}

export function getAllTasks(): ScheduledTask[] {
  return db
    .prepare('SELECT * FROM scheduled_tasks ORDER BY created_at DESC')
    .all() as ScheduledTask[];
}

/**
 * #584 — Cron rows scheduled with `schedule_timezone = 'local'`
 * resolve their effective tz at fire time via `tz_state.current_tz`.
 * When `current_tz` flips mid-slot (operator moves zones), already-
 * cached `next_run` values stay anchored to the prior zone until the
 * row fires. The scheduler hooks `tz_state` writers via
 * `recomputeLocalSchedules` (see `task-scheduler.ts`) and uses this
 * selector to find the affected rows. Narrow filter (`'local'` +
 * `'active'`) so a flip doesn't recompute the world.
 */
export function getActiveLocalScheduledTasks(): ScheduledTask[] {
  return db
    .prepare(
      `SELECT * FROM scheduled_tasks
        WHERE schedule_timezone = 'local'
          AND status = 'active'`,
    )
    .all() as ScheduledTask[];
}

/**
 * #584 — Focused next_run writer for `recomputeLocalSchedules`. The
 * caller has just computed a fresh `next_run` for a `'local'`-scheduled
 * row against the new `current_tz`; this writes ONLY `next_run` and
 * leaves every other field alone (status, last_run, last_result,
 * schedule_*). Distinct from `updateTaskAfterRun` (which also bumps
 * `last_run` / `last_result` / status) and `updateTask` (which is the
 * general-purpose multi-field updater used for user-facing IPC writes).
 */
export function setTaskNextRun(id: string, nextRun: string | null): void {
  db.prepare('UPDATE scheduled_tasks SET next_run = ? WHERE id = ?').run(
    nextRun,
    id,
  );
}

export function updateTask(
  id: string,
  updates: Partial<
    Pick<
      ScheduledTask,
      | 'prompt'
      | 'script'
      | 'schedule_type'
      | 'schedule_value'
      | 'schedule_timezone'
      | 'next_run'
      | 'status'
      | 'agent_model'
    >
  >,
): void {
  const fields: string[] = [];
  const values: unknown[] = [];

  if (updates.prompt !== undefined) {
    fields.push('prompt = ?');
    values.push(updates.prompt);
  }
  if (updates.script !== undefined) {
    fields.push('script = ?');
    values.push(updates.script || null);
  }
  if (updates.schedule_type !== undefined) {
    fields.push('schedule_type = ?');
    values.push(updates.schedule_type);
  }
  if (updates.schedule_value !== undefined) {
    fields.push('schedule_value = ?');
    values.push(updates.schedule_value);
  }
  if (updates.schedule_timezone !== undefined) {
    fields.push('schedule_timezone = ?');
    values.push(updates.schedule_timezone || null);
  }
  if (updates.next_run !== undefined) {
    fields.push('next_run = ?');
    values.push(updates.next_run);
  }
  if (updates.status !== undefined) {
    fields.push('status = ?');
    values.push(updates.status);
  }
  if (updates.agent_model !== undefined) {
    // updates.agent_model === null is a legitimate clear of the
    // per-task override (fall back to the Phase 2 ladder). Distinguish
    // it from `undefined` (caller didn't touch this field) so passing
    // `null` writes the column even though falsy.
    fields.push('agent_model = ?');
    values.push(updates.agent_model);
  }

  if (fields.length === 0) return;

  values.push(id);
  db.prepare(
    `UPDATE scheduled_tasks SET ${fields.join(', ')} WHERE id = ?`,
  ).run(...values);
}

/**
 * Set or clear the per-task AGENT_MODEL override for #509 Phase 3.
 * Thin wrapper over `updateTask` exposed as a named function so the
 * IPC handler (`set_task_agent_model` in src/ipc.ts) and the operator-
 * facing helper share one writer. Pass `null` to clear the override
 * (fall back to the Phase 2 ladder); pass a non-empty string to
 * install one. The string is NOT validated here — `resolvePerGroupAgentModel`
 * at spawn time does the prefix check and falls back if it doesn't
 * recognise the shape, so a typo never silently routes to the global
 * default without an audit-log warning. Returns whether the row
 * existed (so the IPC handler can distinguish "no-op" from "task not
 * found" without a second SELECT).
 */
export function setTaskAgentModel(
  id: string,
  agentModel: string | null,
): boolean {
  const row = db
    .prepare('SELECT 1 FROM scheduled_tasks WHERE id = ?')
    .get(id) as { 1: number } | undefined;
  if (!row) return false;
  db.prepare('UPDATE scheduled_tasks SET agent_model = ? WHERE id = ?').run(
    agentModel,
    id,
  );
  return true;
}

export function deleteTask(id: string): void {
  // Delete child records first (FK constraint)
  db.prepare('DELETE FROM task_run_logs WHERE task_id = ?').run(id);
  db.prepare('DELETE FROM scheduled_tasks WHERE id = ?').run(id);
}

/**
 * Persist the per-task SDK session id (#336). Called from `runTask`
 * when the SDK reports `newSessionId` on a recurring task fire so the
 * next fire can pass it as `resume:` and avoid rebuilding the
 * message-history prefix from scratch.
 *
 * Idempotent: re-writing the same id is a no-op at the row level
 * (UPDATE matching the existing value). The caller doesn't have to
 * de-duplicate streamed `newSessionId` events — the SDK can re-issue
 * the id mid-run, in which case "last write wins" is the right
 * semantic (the latest id is the live transcript on disk).
 *
 * No status guard: the caller decides eligibility (recurring vs
 * once-task, paused vs active). This helper only writes.
 *
 * `pluginsHash` (#710) records the plugin-registry content hash the
 * session was created against, written in the same UPDATE so id and
 * hash can never drift apart. NULL means the hash was unknowable at
 * persist time — registry absent, or it vanished mid-walk during a
 * registry swap (see `hashDirectoryTree`). At fire time a NULL stored
 * here mismatches any KNOWN registry hash and rotates (the pre-#710
 * row path); when the CURRENT hash is null the scheduler skips
 * rotation entirely, so registry-less installs keep resuming.
 */
export function setTaskSessionId(
  id: string,
  sessionId: string,
  pluginsHash: string | null = null,
): void {
  db.prepare(
    `UPDATE scheduled_tasks SET session_id = ?, session_plugins_hash = ?
     WHERE id = ?`,
  ).run(sessionId, pluginsHash, id);
}

/**
 * Clear the per-task SDK session id (#336). Used when the SDK-reported
 * id rotated mid-run (the previous id's transcript is now stale and
 * gets wiped from disk separately) or when the caller wants to force
 * the next fire to start fresh without nuking the whole maintenance
 * slot.
 */
export function clearTaskSessionId(id: string): void {
  db.prepare(
    `UPDATE scheduled_tasks
     SET session_id = NULL, session_plugins_hash = NULL WHERE id = ?`,
  ).run(id);
}

/**
 * Clear the per-task SDK session id for every scheduled task in a
 * group (#336). Called from `nukeSession` when the maintenance (or
 * 'all') slot for a group is wiped so the on-disk JSONL transcripts
 * disappear — without this the next fire would try to `resume:` an id
 * whose transcript no longer exists and the SDK would 404 / start
 * fresh anyway, just noisily. Returns the number of rows touched so
 * the caller can log the wipe scope alongside the JSONL count.
 */
export function clearTaskSessionIdsForGroup(groupFolder: string): number {
  const result = db
    .prepare(
      `UPDATE scheduled_tasks
       SET session_id = NULL, session_plugins_hash = NULL
       WHERE group_folder = ? AND session_id IS NOT NULL`,
    )
    .run(groupFolder);
  return result.changes;
}

/**
 * Delete completed once-tasks older than maxAgeMs.
 *
 * Age is measured from `COALESCE(last_run, created_at)` rather than
 * `last_run` alone. The scheduler pre-advances `status='completed'`
 * before dispatch (see `task-scheduler.ts`), and `updateTaskAfterRun`
 * is what actually stamps `last_run`. If a task is marked completed but
 * the dispatch path fails (container crash, maintenance slot wedged,
 * task aborted before the streaming callback fires), `last_run` stays
 * NULL forever — the original `last_run < cutoff` filter would never
 * match, and the orphan row would linger indefinitely. Falling back to
 * `created_at` guarantees these rows are eventually pruned by their
 * own age.
 *
 * Trade-off: a once-task scheduled far in advance and only just now
 * marked completed (with `last_run` NULL because dispatch failed) is
 * pruned earlier than the user-facing "TTL after completion" intent —
 * the row could disappear immediately if `created_at` is already past
 * the cutoff. This is acceptable because (a) such rows were never
 * visible to the user as completed during normal operation, so there's
 * no observable regression vs. the case where the task ran and stamped
 * last_run; (b) the alternative of letting NULL-last_run rows linger
 * indefinitely (the bug we're fixing) is strictly worse. A future
 * `completed_at` column would let us preserve the grace window even for
 * orphans; until then COALESCE is the closest approximation that
 * doesn't require a schema migration.
 *
 * Recurring tasks never reach status='completed' (computeNextRun only
 * returns null for once-tasks), so the schedule_type='once' clause is
 * defensive. Returns row count removed.
 */
export function pruneCompletedTasks(maxAgeMs: number): number {
  const cutoff = new Date(Date.now() - maxAgeMs).toISOString();
  const tx = db.transaction((cutoffIso: string): number => {
    db.prepare(
      `DELETE FROM task_run_logs
       WHERE task_id IN (
         SELECT id FROM scheduled_tasks
         WHERE status = 'completed'
           AND schedule_type = 'once'
           AND COALESCE(last_run, created_at) < ?
       )`,
    ).run(cutoffIso);
    return db
      .prepare(
        `DELETE FROM scheduled_tasks
         WHERE status = 'completed'
           AND schedule_type = 'once'
           AND COALESCE(last_run, created_at) < ?`,
      )
      .run(cutoffIso).changes;
  });
  return tx(cutoff);
}

/**
 * Recover once-tasks whose pre-advance landed but whose dispatch did
 * not. The pre-advance write at `task-scheduler.ts` flips a once-task
 * to `status='completed'` *before* `enqueueTask` is called, so a host
 * crash, a queue shutdown, or a streaming-callback failure between
 * those two lines leaves a row with the orphan signature
 * `status='completed' AND schedule_type='once' AND last_run IS NULL
 * AND next_run IS NOT NULL`. `getDueTasks()` filters on
 * `status='active'`, so the row never re-tries; `pruneCompletedTasks`
 * eventually GCs it via the `COALESCE(last_run, created_at)` fallback,
 * but by then the schedule is silently lost.
 *
 * Called once at scheduler startup (see `task-scheduler.ts`). Flips
 * matching rows back to `active` so the next `getDueTasks()` poll
 * picks them up and `runTask` dispatches them — late but firing.
 *
 * If dispatch fails *again* for the same row (same race), the next
 * restart resurrects it again, and `pruneCompletedTasks` still GCs it
 * on age via the `created_at` fallback — no risk of zombie loop.
 *
 * Returns the list of resurrected task ids (sorted by id for
 * deterministic output) for logging and assertion in tests.
 *
 * Atomicity: SELECT and the per-id UPDATEs run inside a single
 * transaction. The UPDATE re-asserts the full zombie predicate so a
 * row that races (e.g., a concurrent dispatch landing between SELECT
 * and UPDATE) is left alone — the UPDATE is a no-op and the id is
 * dropped from the returned list.
 */
export function resurrectZombieTasks(): string[] {
  const tx = db.transaction((): string[] => {
    const rows = db
      .prepare(
        `SELECT id FROM scheduled_tasks
         WHERE status = 'completed'
           AND schedule_type = 'once'
           AND last_run IS NULL
           AND next_run IS NOT NULL
         ORDER BY id`,
      )
      .all() as Array<{ id: string }>;
    if (rows.length === 0) return [];
    const stmt = db.prepare(
      `UPDATE scheduled_tasks
       SET status = 'active'
       WHERE id = ?
         AND status = 'completed'
         AND schedule_type = 'once'
         AND last_run IS NULL
         AND next_run IS NOT NULL`,
    );
    const resurrectedIds: string[] = [];
    for (const { id } of rows) {
      if (stmt.run(id).changes > 0) resurrectedIds.push(id);
    }
    return resurrectedIds;
  });
  return tx();
}

/**
 * Find recurring (cron / interval) tasks that are still `status='active'`
 * whose age (last_run, falling back to created_at) is older than
 * `maxAgeMs`. These are NOT pruned — only surfaced so the scheduler can
 * emit a warn-level log. A dormant cron is a symptom, not garbage: the
 * row points at a real schedule; what's broken is dispatch (next_run
 * not advancing, container queue stuck, etc.). Visibility first; humans
 * decide whether to delete.
 *
 * `COALESCE(last_run, created_at) < ?` (vs the original
 * `last_run IS NULL OR last_run < ?`) prevents false-positive warnings
 * for freshly-created recurring tasks whose `last_run` is NULL because
 * they simply haven't been due yet — matching the threshold-based
 * semantics for the same NULL-last_run shape that `pruneCompletedTasks`
 * already uses.
 */
export function getDormantRecurringTasks(maxAgeMs: number): ScheduledTask[] {
  const cutoff = new Date(Date.now() - maxAgeMs).toISOString();
  return db
    .prepare(
      `SELECT * FROM scheduled_tasks
       WHERE status = 'active'
         AND schedule_type IN ('cron', 'interval')
         AND COALESCE(last_run, created_at) < ?`,
    )
    .all(cutoff) as ScheduledTask[];
}

export function getDueTasks(): ScheduledTask[] {
  const now = new Date().toISOString();
  return db
    .prepare(
      `
    SELECT * FROM scheduled_tasks
    WHERE status = 'active' AND next_run IS NOT NULL AND next_run <= ?
    ORDER BY next_run
  `,
    )
    .all(now) as ScheduledTask[];
}

export function updateTaskAfterRun(
  id: string,
  nextRun: string | null,
  lastResult: string,
): void {
  const now = new Date().toISOString();
  // Status transitions (in CASE-evaluation order):
  //   - status = 'paused' → stay 'paused'. A runtime parse failure that
  //     paused the task via computeNextRun during this very run must
  //     not be flipped back to 'completed' just because nextRun is null.
  //     See #102 round-4 review.
  //   - nextRun IS NULL (and status is anything other than 'paused')
  //     → 'completed'. Covers the natural once-task end. Note that
  //     'completed' rows that re-enter this code path would also flip
  //     here, which is harmless (they were already terminal).
  //   - otherwise → status unchanged.
  db.prepare(
    `
    UPDATE scheduled_tasks
    SET next_run = ?, last_run = ?, last_result = ?,
        status = CASE
          WHEN status = 'paused' THEN 'paused'
          WHEN ? IS NULL THEN 'completed'
          ELSE status
        END
    WHERE id = ?
  `,
  ).run(nextRun, now, lastResult, nextRun, id);
}

export function logTaskRun(log: TaskRunLog): void {
  db.prepare(
    `
    INSERT INTO task_run_logs (task_id, run_at, duration_ms, status, result, error)
    VALUES (?, ?, ?, ?, ?, ?)
  `,
  ).run(
    log.task_id,
    log.run_at,
    log.duration_ms,
    log.status,
    log.result,
    log.error,
  );
}
