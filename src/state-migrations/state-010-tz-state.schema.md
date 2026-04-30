# `tz_state` + `follow_me_tasks` schema (state-010, #302)

This file is the schema contract for the two SQLite tables introduced
by state-010 (`tz_state`, `follow_me_tasks`). It satisfies the
`coding-policy: stateful-artifacts` requirement that every persistent
state surface ships with a schema document; the authoritative
skill-side mirror lives at
`nanoclaw-admin/skills/task-tz-sync/state-schema.md` and is updated
through the staging→promote pipeline alongside the SKILL.md rewrites
that retire the JSON path.

The TypeScript file `state-010-tz-state.ts` is the source of truth for
the SQL — this Markdown only describes the contract.

## Owner skill

`nanoclaw-admin/skills/task-tz-sync` is the single writer for both
tables. Five-plus reader skills consume the rows:

- `nanoclaw-admin/skills/morning-brief`
- `nanoclaw-admin/skills/nightly-housekeeping`
- `nanoclaw-admin/skills/weekly-housekeeping`
- `nanoclaw-admin/skills/check-calendar`
- `nanoclaw-admin/skills/heartbeat/scripts/heartbeat-precheck.py`
- `nanoclaw-admin/skills/scheduler-timezone/scripts/read-current-tz.py`

## `tz_state` (singleton)

| Column           | Type    | Nullable | Default | Notes                                  |
| ---------------- | ------- | -------- | ------- | -------------------------------------- |
| `id`             | INTEGER | no       | —       | PK with `CHECK(id = 1)` — singleton    |
| `current_tz`     | TEXT    | no       | —       | IANA zone name; load-bearing           |
| `home_tz`        | TEXT    | no       | —       | IANA zone name; reference for jet-lag  |
| `scheduler_tz`   | TEXT    | yes      | NULL    | Informational only; not load-bearing   |
| `schema_version` | INTEGER | no       | `1`     | Bumped on shape change; owner migrates |

Read pattern: `SELECT current_tz, home_tz FROM tz_state WHERE id = 1`.

Write pattern (UPSERT — never `INSERT OR REPLACE`, which is
delete+insert in SQLite and would break any future FK references to
`tz_state(id)` and silently reset defaulted columns like
`schema_version`):

```sql
INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz)
VALUES (1, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  current_tz   = excluded.current_tz,
  home_tz      = excluded.home_tz,
  scheduler_tz = excluded.scheduler_tz;
```

`schema_version` is not in the writer's column list — the column's
default takes care of inserts; shape upgrades bump it via a state-NNN
migration, not via the steady-state writer.

## `follow_me_tasks`

| Column           | Type    | Nullable | Default             | Notes                                    |
| ---------------- | ------- | -------- | ------------------- | ---------------------------------------- |
| `name`           | TEXT    | no       | —                   | PK; natural identity (skill name)        |
| `local_time`     | TEXT    | no       | —                   | `HH:MM` in `current_tz`                  |
| `schedule_value` | TEXT    | no       | —                   | UTC cron string                          |
| `last_run_date`  | TEXT    | yes      | NULL                | ISO date in local tz (per-day idempotent) |
| `pending_run_at` | TEXT    | yes      | NULL                | ISO timestamp; nullable                  |
| `schema_version` | INTEGER | no       | `1`                 | Bumped on shape change; owner migrates   |
| `updated_at`     | TEXT    | no       | `CURRENT_TIMESTAMP` | Row-mutation timestamp                   |

Write pattern (UPSERT via the writer, e.g. `task-tz-sync`):

```sql
INSERT INTO follow_me_tasks
  (name, local_time, schedule_value)
VALUES (?, ?, ?)
ON CONFLICT(name) DO UPDATE SET
  local_time     = excluded.local_time,
  schedule_value = excluded.schedule_value,
  updated_at     = CURRENT_TIMESTAMP;
```

Phase-C/Phase-D read-modify-write (replaces the
`follow-me-two-phase-lock.md` rule's LOCK_EX + re-read):

```sql
BEGIN IMMEDIATE;
  SELECT last_run_date, pending_run_at
    FROM follow_me_tasks
    WHERE name = ?;
  -- ... reason about whether to fire ...
  UPDATE follow_me_tasks
     SET last_run_date  = ?,
         pending_run_at = ?,
         updated_at     = CURRENT_TIMESTAMP
     WHERE name = ?;
COMMIT;
```

`BEGIN IMMEDIATE` takes the database's RESERVED lock up front so a
concurrent writer's `BEGIN IMMEDIATE` blocks until COMMIT rather than
racing the read-then-write window.

## Migration policy

- `schema_version` columns are bumped on every shape change.
- Only `task-tz-sync` migrates: on its own read, detect old
  `schema_version`, upgrade the row, rewrite. Reader skills must
  treat an old `schema_version` as "no usable prior state".
- Schema-level changes (adding/removing columns, renaming) ship as a
  new state-NNN migration in `src/state-migrations/`, with a
  corresponding bump to the relevant `schema_version` default.

## Open `name` set

The schema deliberately does NOT enforce the
`'morning-brief' | 'nightly-housekeeping' | 'weekly-housekeeping'` set
with a CHECK constraint — a future writer (e.g. a fourth follow-me
job, or a phase split out of an existing skill) should not need a
schema migration to land. Readers that decide on a per-name basis
must therefore tolerate "no row for this task yet" the same way they
tolerate NULL `last_run_date`.
