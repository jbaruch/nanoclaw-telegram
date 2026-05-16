# `tz_state` + `follow_me_tasks` schema (state-010, #302)

This file is the schema contract for the two SQLite tables introduced
by state-010 (`tz_state`, `follow_me_tasks`). It satisfies the
`coding-policy: stateful-artifacts` requirement that every persistent
state surface ships with a schema document; the authoritative
skill-side mirror lives at
`nanoclaw-admin/skills/task-tz-sync/state-schema.md` and is updated
through the staging→promote pipeline alongside the SKILL.md rewrites
that retire the JSON path.

The TypeScript files `state-010-tz-state.ts`, (post-#542)
`state-012-tz-state-segments.ts`, (post-jbaruch/nanoclaw-admin#229)
`state-013-tz-state-segments-datetime.ts`, and (post-#574 Phase 2)
`state-015-tz-state-stale-warning.ts` are the source of truth for the
SQL — this Markdown only describes the contract.

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

| Column                  | Type    | Nullable | Default | Notes                                                                                       |
| ----------------------- | ------- | -------- | ------- | ------------------------------------------------------------------------------------------- |
| `id`                    | INTEGER | no       | —       | PK with `CHECK(id = 1)` — singleton                                                         |
| `current_tz`            | TEXT    | no       | —       | IANA zone name; load-bearing                                                                |
| `home_tz`               | TEXT    | no       | —       | IANA zone name; reference for jet-lag                                                       |
| `scheduler_tz`          | TEXT    | yes      | NULL    | Informational only; not load-bearing                                                        |
| `segments`              | TEXT    | yes      | NULL    | (state-012, #542 / state-013, #229) JSON-stringified `[{timezone, from, to, from_dt?, to_dt?, label}]` from `sync_tripit` |
| `schema_version`        | INTEGER | no       | `1`     | Column default still `1`; rows are written at the gate's current value (currently `4`)      |
| `last_stale_warning_at` | TEXT    | yes      | NULL    | (state-015, #574 Phase 2) ISO-8601 UTC of the last `stale_no_share` chat nag; cooldown stamp |

Read pattern: `SELECT current_tz, home_tz FROM tz_state WHERE id = 1`.
Heartbeat advisory walker also reads `segments, schema_version,
last_stale_warning_at`.

`segments` shape (post-jbaruch/nanoclaw-admin#229 +
reclaim-tripit-timezones-sync#13):

```json
[
  {
    "timezone": "Europe/Berlin",
    "from": "2026-05-12",
    "to": "2026-05-19",
    "from_dt": "2026-05-12T13:30:00.000Z",
    "to_dt": "2026-05-19T15:00:00.000Z",
    "label": "Devoxx UK 2026 - London"
  }
]
```

`from` / `to` are date-only `YYYY-MM-DD` strings (the original pre-#229
shape). `from_dt` / `to_dt` (added in #229) are ISO 8601 UTC strings
preserving the underlying flight arrival / lodging check-in / check-out
wall-clock. Both shapes are produced upstream in
`reclaim-tripit-timezones-sync/lib/tripit.mjs` — the date-only fields
slice `toISOString()` to 10 chars; the datetime fields ARE
`toISOString()` of the same Date object. Lex compare on either shape
is equivalent to chrono compare, so the walker uses plain string
`<=`/`<`.

The walker (`walkTzSegments` in `src/db.ts`) prefers `from_dt` /
`to_dt` per segment when both are present and falls back to date-only
`from` / `to` otherwise. Per-segment fallback handles the mixed-shape
transient between a host deploy (when an older row may still be cached
on disk) and the next `sync_tripit` rewrite. The match rule on either
path keeps strict inequality on the right edge — a return-flight
arrival at exactly `to_dt` (or a return-day `to === todayUtc`) flips
back to `home_tz` rather than holding the destination zone.

Write pattern after #542 (host-side `applyTripitSegmentsToTzState`,
called from `sync_tripit`'s success path):

```sql
UPDATE tz_state
   SET current_tz     = ?,
       segments       = ?,
       schema_version = ?  -- SUPPORTED_TZ_STATE_SCHEMA_VERSION
 WHERE id = 1;
```

The row MUST already exist — `home_tz` is NOT NULL on the schema and
isn't derivable from the TripIt payload, so the helper logs and
returns silently if the singleton is absent. First-time setup / the
JSON migration backfill seeds the row with `home_tz`, then the next
`sync_tripit` run takes over the steady-state writes.

`schema_version` is written through `SUPPORTED_TZ_STATE_SCHEMA_VERSION`
in `src/db.ts` (currently `4` post-#574 Phase 2; was `3` between #229
and #574, `2` between #542 and #229). The state-010 column default of
`1` is preserved on the schema for historical compatibility, but every
writer binds the constant.

`last_stale_warning_at` is owned by `runTzHeartbeatAdvisory` in
`src/db.ts` (post-#574 Phase 2). It is written atomically alongside
any `current_tz` flip on each 30-min advisory tick:

- Set to `now.toISOString()` when the resolver reports
  `warning: 'stale_no_share'` AND the cooldown window
  (`STALE_WARNING_HOURS = 12 h`) has elapsed since the prior fire
  (or the prior fire is NULL). The advisory then emits a
  `warningToFire: 'stale_no_share'` signal to the orchestrator's
  setInterval, which sends a one-shot chat nag to the main group.
- Reset to NULL when the resolver reports `source: 'fresh_location'`
  (owner has genuinely shared again, not merely "warning no longer
  fires"). Reset on `source: 'walker_stale_location'` would erase
  the cooldown stamp during the 4 h ≤ age < 12 h band and re-fire
  the nag on the first ≥ 12 h tick after — undesirable.
- Left unchanged in every other path (no warning due, not freshly
  shared either).

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
