# State Ownership — who owns a table's schema

Companion to [CORE-VS-DOMAIN.md](CORE-VS-DOMAIN.md), for the state plane.
Epic #293 migrated the legacy JSON state files into SQLite via
`src/state-migrations/`, and along the way the orchestrator absorbed the
schema of every skill's private state. The migration *framework* is
platform core; owning every skill's schema in core is not — it couples a
platform deploy to skill evolution and bloats `src/` with shapes only one
skill reads (#854).

## The policy

1. **Single-skill state is skill-owned.** A table or file that exactly one
   skill reads and writes belongs to that skill: the skill (or its tile)
   owns the schema, the `schema_version` stamping, and the migration story,
   per `coding-policy: stateful-artifacts`. The preferred home for new
   skill state is `/workspace/state/<skill>/` JSON with a documented
   schema — not a new core table.
2. **Core owns the substrate and cross-cutting state.** The migration
   machinery (`PRAGMA user_version`, `applyStateMigrations`), DB access,
   and tables the *platform* reads — sessions, scheduling, locations,
   owner-timezone — stay core-owned.
3. **New core migrations for single-skill state need an exception.** A PR
   adding a `state-0NN-*.ts` whose table is single-skill-owned must link
   an epic-level exception explaining why `/workspace/state/<skill>/`
   doesn't fit (e.g. multi-writer race that needs SQLite's atomicity, or
   host-side readers). No link, no merge.
4. **Shipped tables are grandfathered.** The inventory below records
   ownership as-is; nothing forces a rewrite of existing tables (#854 is
   policy, not migration work). If a skill-owned table's schema needs to
   evolve, that evolution is the owning skill's work — core only hosts the
   migration entry, authored with the skill's PR.

## Inventory (as of state-016)

| Migration | Table(s) | Ownership | Owner |
|---|---|---|---|
| 001 | `orders` | skill | orders tile |
| 002, 003 | `email_feedback` | skill | nanoclaw-admin brief-cleanup |
| 004 | `scheduled_reminders` | skill | reminders skill |
| 005 | `email_state`, `email_seen_ids` | skill | tessl__check-email |
| 006 | trusted session tables | skill | tessl__trusted-memory |
| 007 | morning-brief queue tables | skill | morning-brief |
| 008 | calendar tables | skill | check-calendar |
| 009 | `phase_completions` | skill | nanoclaw-admin heartbeat |
| 010, 012, 013, 015 | `tz_state` (+ segments) | platform | host scheduler owner-tz backbone (#748) |
| 011 | session-length-cap tables | platform | orchestrator session caps |
| 014 | `locations` | platform | core location persistence (#849) |
| 016 | `registered_groups` cleanup | platform | orchestrator config |

(`smart_home_events` predates the state-migration framework; its schema is
core-hosted, its reads/writes belong to the Hubitat host plugin — see the
CREATE TABLE comment in `src/db.ts`.)

## For reviewers

When a PR touches `src/state-migrations/`:

- New table read by exactly one skill → ask for the exception link
  (policy point 3) or redirect to `/workspace/state/<skill>/`.
- Schema change to a skill-owned table above → the owning skill's PR
  should motivate it; core-only PRs shouldn't reshape skill state.
- Framework changes (`index.ts` registry, validation, `user_version`
  machinery) → platform work, normal review.
