import type { StateMigration } from '../db.js';

/**
 * #302 — Migrate `/workspace/group/task-tz-state.json` to two SQLite
 * tables, replacing the per-group JSON envelope-with-array shape and
 * the `LOCK_EX` + Phase-C/Phase-D re-read ceremony in
 * `nanoclaw-admin/rules/follow-me-two-phase-lock.md`.
 *
 * JSON-era shape that this table set replaces:
 *
 *     {
 *       "current_tz":    "America/Chicago",
 *       "home_tz":       "America/Chicago",
 *       "scheduler_tz":  "...",
 *       "follow_me_tasks": [
 *         {"name": "morning-brief",
 *          "local_time": "08:00",
 *          "schedule_value": "0 13 * * *",
 *          "last_run_date": "2026-04-29",
 *          "pending_run_at": "2026-04-30T13:00:00Z"},
 *         ...
 *       ]
 *     }
 *
 *   - The three top-level timezone scalars become columns on the
 *     singleton `tz_state` row (`id = 1`).
 *   - The `follow_me_tasks` array becomes one row per task in
 *     `follow_me_tasks`, keyed on `name`. Today's writers schedule
 *     `'morning-brief'`, `'nightly-housekeeping'`, and
 *     `'weekly-housekeeping'`, but the schema deliberately does NOT
 *     pin that set with a CHECK — see "Open task-name set" below.
 *
 * Ownership / writer-reader contract:
 *   - Single writer for both tables: `nanoclaw-admin/skills/task-tz-
 *     sync` (issue spec calls this out explicitly). All cursor
 *     mutations on `follow_me_tasks` (`last_run_date`, `pending_run_
 *     at`) flow through that skill via the SQL transaction documented
 *     below.
 *   - 5+ readers (epic #293's classic many-readers / one-writer split
 *     this whole tile-rewrite is built around):
 *       * `nanoclaw-admin/skills/morning-brief/SKILL.md` Steps 2
 *         (Phase C / Phase D re-read) + 20 (confirm)
 *       * `nanoclaw-admin/skills/nightly-housekeeping/SKILL.md`
 *         Steps 2 + 18
 *       * `nanoclaw-admin/skills/weekly-housekeeping/SKILL.md`
 *         Phase A + confirm
 *       * `nanoclaw-admin/skills/check-calendar/SKILL.md`
 *         (`current_tz` reference)
 *       * `nanoclaw-admin/skills/heartbeat/scripts/heartbeat-
 *         precheck.py` Sections 6 + 8
 *       * `nanoclaw-admin/skills/scheduler-timezone/scripts/read-
 *         current-tz.py` (reads `current_tz` for the orchestrator's
 *         scheduler tz; `nanoclaw-admin/skills/scheduler-timezone/
 *         SKILL.md` is the human-facing wrapper)
 *
 *   - The JSON-era `LOCK_EX` ceremony existed because every reader-
 *     that-becomes-a-writer (morning-brief Phase C → Phase D, nightly
 *     Step 2 → Step 18, weekly Phase A → confirm) had to re-read the
 *     whole envelope, mutate one entry in the `follow_me_tasks`
 *     array, and write the whole envelope back. Concurrent reads-
 *     during-write would observe torn state, and concurrent writes
 *     would clobber each other's sibling array entries — exactly the
 *     bug class #293 targets. With `name` as the PK on
 *     `follow_me_tasks`, an `UPDATE follow_me_tasks SET
 *     last_run_date = ? WHERE name = ?` touches exactly one row, the
 *     sibling-array-entry clobber bug class disappears by
 *     construction (writers don't share columns: morning-brief's
 *     `last_run_date` UPDATE on the `'morning-brief'` row never
 *     touches the `'nightly-housekeeping'` row's columns the way an
 *     envelope rewrite used to), and the Phase-C/Phase-D re-read
 *     pattern collapses to a SQL transaction:
 *
 *         BEGIN IMMEDIATE;
 *           SELECT last_run_date, pending_run_at
 *             FROM follow_me_tasks
 *             WHERE name = 'morning-brief';
 *           -- ... reason about whether to fire the task ...
 *           UPDATE follow_me_tasks
 *              SET last_run_date  = ?,
 *                  pending_run_at = ?,
 *                  updated_at     = CURRENT_TIMESTAMP
 *              WHERE name = 'morning-brief';
 *         COMMIT;
 *
 *     SQLite uses transaction isolation, not per-row locks (it has no
 *     row-level locking — writers are serialized at db/page level via
 *     WAL), and `BEGIN IMMEDIATE` takes the database's RESERVED lock
 *     up front so a concurrent writer's `BEGIN IMMEDIATE` blocks
 *     until COMMIT rather than racing the read-then-write window.
 *     The `follow-me-two-phase-lock.md` rule's "re-read in Phase D"
 *     instruction becomes "the SELECT and the UPDATE share the same
 *     BEGIN IMMEDIATE … COMMIT envelope" — `LOCK_EX` retires entirely
 *     (the JSON file it locked is gone), and the file-lock ceremony's
 *     real intent (atomic read-modify-write across the cursor
 *     fields) is now expressed in SQL.
 *
 * Schema rationale:
 *   - `tz_state` is a singleton enforced via `CHECK(id = 1)` — exactly
 *     one timezone configuration per group, ever. The CHECK mirrors
 *     `email_state` from state-005 and makes "accidentally insert
 *     id=2" fail loudly with a CHECK constraint failure rather than
 *     silently shadowing id=1 the way a multi-row table would. Every
 *     reader's lookup is `SELECT ... FROM tz_state WHERE id = 1`,
 *     getting a clear "no state yet" (undefined) on missing rather
 *     than racing on "which row is canonical".
 *   - `tz_state.current_tz` and `home_tz` are `TEXT NOT NULL` — a
 *     timezone configuration without a current or home tz is
 *     unusable to morning-brief / nightly / weekly / check-calendar /
 *     heartbeat-precheck. Let the DB enforce presence rather than
 *     letting a malformed insert land a row that every reader then
 *     has to defend against.
 *   - `tz_state.scheduler_tz` is nullable `TEXT` — the issue spec
 *     calls out that `scheduler_tz` is "informational only; no longer
 *     load-bearing". The orchestrator's scheduler reads
 *     `current_tz` via `read-current-tz.py`; `scheduler_tz` is kept
 *     in the schema for backward-compat with any historical reader
 *     that still touches it, but the column is nullable because no
 *     consumer of the new shape requires it.
 *   - `follow_me_tasks.name` is `TEXT PRIMARY KEY` — the task name is
 *     the natural identity (`'morning-brief'`,
 *     `'nightly-housekeeping'`, `'weekly-housekeeping'`, ...), and
 *     UPSERT via `ON CONFLICT(name) DO UPDATE SET ...` is the
 *     writer's tool for cursor mutations:
 *
 *         INSERT INTO follow_me_tasks
 *           (name, local_time, schedule_value, last_run_date, pending_run_at)
 *           VALUES (?, ?, ?, ?, ?)
 *         ON CONFLICT(name) DO UPDATE SET
 *           local_time     = excluded.local_time,
 *           schedule_value = excluded.schedule_value,
 *           last_run_date  = excluded.last_run_date,
 *           pending_run_at = excluded.pending_run_at,
 *           updated_at     = CURRENT_TIMESTAMP
 *
 *     A surrogate `id INTEGER` PK would force every UPSERT to first
 *     SELECT the existing `id` for `name = ?`; using `name` as the PK
 *     makes the conflict target the natural one and keeps the writer
 *     to one round-trip.
 *   - `follow_me_tasks.local_time` and `schedule_value` are `TEXT NOT
 *     NULL` — a follow-me task without a wall-clock time or a
 *     UTC-cron string is unschedulable. DB-level NOT NULL means a
 *     malformed insert fails loudly rather than landing a row that
 *     silently breaks task-tz-sync's "compute the next run" math.
 *   - `follow_me_tasks.last_run_date` is nullable — a task that has
 *     never run yet legitimately has no last-run date; the cursor
 *     readers (morning-brief Step 2, nightly Step 2, weekly Phase A)
 *     must tolerate NULL and treat it as "this task hasn't run
 *     today/this-week yet".
 *   - `follow_me_tasks.pending_run_at` is nullable — the field is
 *     populated only between Phase C ("we decided to run") and Phase
 *     D ("we ran, clear the pending"); outside that window it's
 *     legitimately NULL. Same tolerance contract as `last_run_date`.
 *   - `follow_me_tasks.updated_at TEXT NOT NULL DEFAULT
 *     CURRENT_TIMESTAMP` — auditing / debugging aid distinct from
 *     `last_run_date`. `last_run_date` is the writer-chosen business
 *     timestamp; `updated_at` is a row-mutation timestamp the UPSERT
 *     above re-stamps explicitly so a writer-supplied
 *     `last_run_date` from a clock that drifts can't outrun the
 *     row's own mutation log. (`tz_state` does not carry an
 *     `updated_at` column — its writer is task-tz-sync editing the
 *     three timezone scalars at most a couple times a year, and the
 *     "when did this last change?" question on a singleton is
 *     answered by the audit log, not a per-row timestamp.)
 *   - `schema_version` on both tables from day one per `coding-policy:
 *     stateful-artifacts` (lessons applied from #295's two-PR
 *     restoration cycle and reinforced in state-005 / state-006 /
 *     state-007 / state-008 / state-009).
 *
 * Open task-name set — no CHECK on `follow_me_tasks.name`:
 *   The schema deliberately does not enforce the `'morning-brief' |
 *   'nightly-housekeeping' | 'weekly-housekeeping'` set with a CHECK
 *   constraint. A future writer (e.g. a new periodic task split out
 *   of one of the existing skills, or a fourth follow-me job
 *   altogether) should not need a schema migration to land. Readers
 *   that decide on a per-name basis must therefore tolerate "no row
 *   for this task yet" the same way they tolerate NULL `last_run_
 *   date` — every read is either a PK lookup
 *   (`SELECT ... WHERE name = ?`) or a small full-table scan.
 *
 * No secondary indexes:
 *   - `tz_state` has at most one row, so the PK on `id` is the only
 *     access path it can ever need.
 *   - `follow_me_tasks` holds a handful of rows in steady state (the
 *     three named writers above, plus any future additions). Every
 *     real-world query is a PK lookup on `name`; a date-based or
 *     pending-based secondary index would cost more in writes than
 *     it could ever save in reads on a table this small.
 *
 * Like prior migrations in this epic, this PR only adds the schema.
 * Data import (extending `migrateJsonState()` to read existing per-
 * group `task-tz-state.json` files), tile-side rewrites
 * (`nanoclaw-admin/skills/task-tz-sync/SKILL.md` writer rewrite,
 * `nanoclaw-admin/skills/scheduler-timezone/scripts/read-current-tz.py`
 * + SKILL.md, `morning-brief/SKILL.md` Steps 2 + 20,
 * `nightly-housekeeping/SKILL.md` Steps 2 + 18,
 * `weekly-housekeeping/SKILL.md` Phase A + confirm,
 * `check-calendar/SKILL.md` `current_tz` reference,
 * `heartbeat/scripts/heartbeat-precheck.py` Sections 6 + 8),
 * the `nanoclaw-admin/rules/follow-me-two-phase-lock.md` rule rewrite
 * to SQL transaction semantics, the `nanoclaw-core/rules/temporal-
 * awareness.md` `current_tz` reference update, and the schema audit
 * in `nanoclaw-admin/skills/weekly-housekeeping/scripts/system-
 * audit.py` (which is part of #303), are follow-up PRs through the
 * staging→promote pipeline.
 */
export const STATE_010_TZ_STATE: StateMigration = {
  version: 10,
  name: 'tz_state + follow_me_tasks tables (#302)',
  sql: `
    CREATE TABLE tz_state (
      id             INTEGER PRIMARY KEY CHECK(id = 1),
      current_tz     TEXT NOT NULL,
      home_tz        TEXT NOT NULL,
      scheduler_tz   TEXT,
      schema_version INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE follow_me_tasks (
      name           TEXT PRIMARY KEY,
      local_time     TEXT NOT NULL,
      schedule_value TEXT NOT NULL,
      last_run_date  TEXT,
      pending_run_at TEXT,
      schema_version INTEGER NOT NULL DEFAULT 1,
      updated_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
  `,
};
