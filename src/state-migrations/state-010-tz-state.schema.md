# `tz_state` + `follow_me_tasks` schema (state-010, #302)

This file is the schema contract for the two SQLite tables introduced
by state-010 (`tz_state`, `follow_me_tasks`). It satisfies the
`coding-policy: stateful-artifacts` requirement that every persistent
state surface ships with a schema document; the authoritative
skill-side mirror lives at
`nanoclaw-admin/skills/task-tz-sync/state-schema.md` and is updated
through the staging→promote pipeline alongside the SKILL.md rewrites
that retire the JSON path.

The TypeScript files `state-010-tz-state.ts` and (post-#542)
`state-012-tz-state-segments.ts` are the source of truth for the SQL
— this Markdown only describes the contract.

## Owner skill

Post-#542 the host orchestrator (`applyTripitSegmentsToTzState` in
`src/db.ts`) is the writer for `tz_state.current_tz` /
`tz_state.segments` after every successful `sync_tripit` run. The
prior owner (`nanoclaw-admin/skills/task-tz-sync`) retires in
jbaruch/nanoclaw-admin#223 (the tile-side cleanup PR). For
`follow_me_tasks`, `task-tz-sync` (until retirement) and the
follow-me skills' Phase C / Phase D updates remain the writers.

Five-plus reader skills consume the rows:

- `nanoclaw-admin/skills/morning-brief`
- `nanoclaw-admin/skills/nightly-housekeeping`
- `nanoclaw-admin/skills/weekly-housekeeping`
- `nanoclaw-admin/skills/check-calendar`
- `nanoclaw-admin/skills/heartbeat/scripts/heartbeat-precheck.py`
- `nanoclaw-admin/skills/scheduler-timezone/scripts/read-current-tz.py`

## `tz_state` (singleton)

| Column           | Type    | Nullable | Default | Notes                                                                                       |
| ---------------- | ------- | -------- | ------- | ------------------------------------------------------------------------------------------- |
| `id`             | INTEGER | no       | —       | PK with `CHECK(id = 1)` — singleton                                                         |
| `current_tz`     | TEXT    | no       | —       | IANA zone name; load-bearing                                                                |
| `home_tz`        | TEXT    | no       | —       | IANA zone name; reference for jet-lag                                                       |
| `scheduler_tz`   | TEXT    | yes      | NULL    | Informational only; not load-bearing                                                        |
| `segments`       | TEXT    | yes      | NULL    | (state-012, #542) JSON-stringified `[{timezone, from, to, label}]` from `sync_tripit`       |
| `schema_version` | INTEGER | no       | `1`     | Column default still `1`; rows are written at the gate's current value (currently `2`)      |

Read pattern: `SELECT current_tz, home_tz FROM tz_state WHERE id = 1`.
Heartbeat advisory walker also reads `segments, schema_version`.

`segments` shape:

```json
[
  {
    "timezone": "Europe/Berlin",
    "from": "2026-05-12",
    "to": "2026-05-19",
    "label": "Devoxx UK 2026 - London"
  }
]
```

`from` / `to` are date-only `YYYY-MM-DD` strings produced upstream by
`reclaim-tripit-timezones-sync/lib/tripit.mjs::formatDate` (UTC-derived
ISO date slice). Lex compare is equivalent to date compare, so the
walker uses plain string `<=`/`<`. The match rule is `from <= today <
to` — strict inequality on the right edge, so a return-flight day flips
back to `home_tz` instead of holding the destination zone.

Write pattern after #542 (host-side `applyTripitSegmentsToTzState`,
called from `sync_tripit`'s success path):

```sql
UPDATE tz_state
   SET current_tz     = ?,
       segments       = ?,
       schema_version = 2
 WHERE id = 1;
```

The row MUST already exist — `home_tz` is NOT NULL on the schema and
isn't derivable from the TripIt payload, so the helper logs and
returns silently if the singleton is absent. First-time setup / the
JSON migration backfill seeds the row with `home_tz`, then the next
`sync_tripit` run takes over the steady-state writes.

`schema_version` is written explicitly as `2` to satisfy the reader
gate (`SUPPORTED_TZ_STATE_SCHEMA_VERSION = 2` in `src/db.ts`). The
state-010 column default of `1` is preserved on the schema for
historical compatibility, but every writer post-#542 writes `2`
directly.

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
- Post-#542 the host orchestrator owns `tz_state` migrations: on its
  own write (`applyTripitSegmentsToTzState`), it writes the current
  `schema_version` directly. Reader skills (the LLM-side morning-
  brief / nightly / weekly / check-calendar / heartbeat-precheck /
  scheduler-timezone) must treat an old `schema_version` as "no
  usable prior state".
- Schema-level changes (adding/removing columns, renaming) ship as a
  new state-NNN migration in `src/state-migrations/`, with a
  corresponding bump to `SUPPORTED_TZ_STATE_SCHEMA_VERSION` in
  `src/db.ts`.

## Open `name` set

The schema deliberately does NOT enforce the
`'morning-brief' | 'nightly-housekeeping' | 'weekly-housekeeping'` set
with a CHECK constraint — a future writer (e.g. a fourth follow-me
job, or a phase split out of an existing skill) should not need a
schema migration to land. Readers that decide on a per-name basis
must therefore tolerate "no row for this task yet" the same way they
tolerate NULL `last_run_date`.
