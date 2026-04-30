import type { StateMigration } from '../db.js';

/**
 * #301 — Migrate `/workspace/group/heartbeat-state.json` to a single
 * SQLite `phase_completions` table, replacing the multi-key sibling-
 * bucket JSON shape and the `LOCK_EX` ceremony in
 * `nanoclaw-admin/skills/heartbeat/scripts/mark-phase-complete.py`.
 *
 * JSON-era shape that this table replaces:
 *
 *     {
 *       "heartbeat_last_completed": "2026-04-30T07:00:00Z",
 *       "nightly_last_completed":   "2026-04-30T02:00:00Z",
 *       "weekly_last_completed":    "2026-04-27T02:00:00Z",
 *       "last_composio_check":      "2026-04-30T07:00:00Z"
 *     }
 *
 *   - The three `*_last_completed` top-level keys become rows keyed by
 *     phase string ('heartbeat', 'nightly', 'weekly'), with the
 *     timestamp landing in the `last_completed` column.
 *   - `last_composio_check` was a phase-adjacent extra (only `heartbeat`
 *     stamps it today). It moves into the `metadata` JSON blob on
 *     whichever phase row writes it. The shared `mark-phase-complete.py`
 *     follow-up is expected to write it on the `heartbeat` row when its
 *     `--composio-overdue` flag flips, e.g.:
 *
 *         metadata = '{"last_composio_check":"2026-04-30T07:00:00Z"}'
 *
 *     The schema doesn't pin which phase owns which metadata key — the
 *     writer decides per phase, the reader (heartbeat-precheck.py and
 *     friends) reads back the same key from the same phase row. Other
 *     phase-adjacent extras (future flags, counters) follow the same
 *     writer-decides convention without touching the schema.
 *
 * Ownership / writer-reader contract:
 *   - Writers (multi-skill via the shared `mark-phase-complete.py`):
 *     `nanoclaw-admin/skills/heartbeat`,
 *     `nanoclaw-admin/skills/nightly-housekeeping`,
 *     `nanoclaw-admin/skills/weekly-housekeeping` — each writes its own
 *     `phase` row only.
 *   - The JSON-era `LOCK_EX` ceremony existed because all three writers
 *     were mutating the same JSON envelope, and a concurrent
 *     read-modify-write would clobber sibling top-level keys. With
 *     `phase` as the PK, the writer's UPSERT targets exactly one row
 *     and the sibling-clobber bug class disappears by construction —
 *     SQLite serializes writers at the database level (page-level in
 *     WAL mode), so a `phase = 'heartbeat'` UPSERT never touches the
 *     `phase = 'nightly'` row's columns the way an envelope rewrite
 *     used to. The win is "writers don't share columns", not row-
 *     level locking (SQLite has no such thing):
 *
 *         INSERT INTO phase_completions
 *           (phase, last_completed, metadata)
 *         VALUES (?, ?, ?)
 *         ON CONFLICT(phase) DO UPDATE SET
 *           last_completed = excluded.last_completed,
 *           metadata       = excluded.metadata,
 *           updated_at     = CURRENT_TIMESTAMP
 *
 *     Multi-skill writers become safe by construction once
 *     `heartbeat-state.json.lock` retires — there's no shared mutable
 *     envelope left to race on.
 *   - Readers: `nanoclaw-admin/skills/heartbeat/scripts/heartbeat-
 *     precheck.py` (reads the three phase rows to decide whether the
 *     next heartbeat needs to run a phase), `nanoclaw-admin/skills/
 *     weekly-housekeeping/scripts/system-audit.py` (reads completion
 *     timestamps for the audit — see #303), and the SKILL.md Step 27 of
 *     each phase skill (decision branches on its own row's
 *     `last_completed`). The `metadata` blob is opaque TEXT to the
 *     schema; readers parse it as JSON in their own code.
 *
 * Schema rationale:
 *   - `phase` is `TEXT PRIMARY KEY` — the phase string is the natural
 *     uniqueness contract (one row per phase, ever) and the UPSERT
 *     conflict target.
 *   - `last_completed` is `TEXT NOT NULL` — a phase row without a
 *     completion timestamp is meaningless; let the DB enforce presence
 *     rather than letting a malformed insert land a row that
 *     heartbeat-precheck would then have to defend against.
 *   - `metadata` is nullable `TEXT` — phases without phase-specific
 *     extras (today: `nightly`, `weekly`) leave it NULL; readers must
 *     tolerate NULL alongside a JSON string.
 *   - `updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP` — auditing /
 *     debugging aid distinct from `last_completed`. `last_completed` is
 *     the phase-completion business timestamp the writer chooses;
 *     `updated_at` is a row-mutation timestamp SQLite stamps on every
 *     INSERT / UPDATE (the UPSERT above sets it explicitly to
 *     `CURRENT_TIMESTAMP` on the UPDATE branch so a writer-supplied
 *     `last_completed` from a clock that drifts can't outrun the row's
 *     own mutation log).
 *   - `schema_version` on the table from day one per `coding-policy:
 *     stateful-artifacts` (lessons applied from #295's two-PR
 *     restoration cycle and reinforced in state-005 / state-006 /
 *     state-007 / state-008).
 *
 * No index beyond the implicit one on `phase` (PK): today's writers
 * are `heartbeat`, `nightly`, `weekly` — three rows in steady state.
 * The schema deliberately does NOT enforce that set with a CHECK
 * constraint: a future writer (e.g. a `composio` phase split out of
 * the current `heartbeat.metadata` blob) should not need a schema
 * migration to land. Readers must therefore tolerate "no row for this
 * phase yet" the same way they tolerate NULL `metadata` — every read
 * is either a PK lookup (`SELECT ... WHERE phase = ?`) or a full
 * table scan over a handful of rows. A secondary index would cost
 * more in writes than it could ever save in reads.
 *
 * Like prior migrations in this epic, this PR only adds the schema.
 * Data import (extending `migrateJsonState()` to read existing per-
 * group `heartbeat-state.json` files), tile-side rewrites
 * (`mark-phase-complete.py` reduction to the UPSERT above,
 * `heartbeat-precheck.py` rewrite, SKILL.md Step 27 updates in
 * heartbeat / nightly / weekly, `system-audit.py` read of completion
 * timestamps which is part of #303), and retiring
 * `heartbeat-state.json.lock` are follow-up PRs through the staging→
 * promote pipeline.
 */
export const STATE_009_PHASE_COMPLETIONS: StateMigration = {
  version: 9,
  name: 'phase_completions table (#301)',
  sql: `
    CREATE TABLE phase_completions (
      phase          TEXT PRIMARY KEY,
      last_completed TEXT NOT NULL,
      metadata       TEXT,
      updated_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      schema_version INTEGER NOT NULL DEFAULT 1
    );
  `,
};
