import type { StateMigration } from '../db.js';

/**
 * #298 — Migrate `session-state.json` (per-group, multi-writer) to
 * SQLite tables. Owner skill: `tessl__trusted-memory`
 * (`register-session.py`).
 *
 * Naming: the issue proposed `sessions` and `session_singleton`,
 * but the orchestrator's `src/db.ts` already declares a `sessions`
 * table for SDK session-id tracking per `(group_folder, session_name)`
 * — completely different semantic from the trusted-memory skill's
 * per-named-session metadata. A bare `CREATE TABLE sessions` here
 * would fail loudly at migration apply time with "table sessions
 * already exists" (the orchestrator's `createSchema` runs first
 * and uses `CREATE TABLE IF NOT EXISTS`, so the row is already in
 * sqlite_master by the time this migration runs). That's a hard
 * startup failure rather than a silent override, but still
 * unwanted — prefixed both new tables with `trusted_` to make the
 * namespace boundary explicit so the failure can never even
 * surface.
 *
 * Table shape:
 *   - `trusted_sessions` — one row per `NANOCLAW_SESSION_NAME`.
 *     `session_name` PK matches the JSON-era top-level key. `started`
 *     and `epoch` are write-once-per-session; `last_seen` is updated
 *     by heartbeat-precheck.py to track whether a maintenance slot
 *     is alive.
 *   - `trusted_session_singleton` — single-row store for fields
 *     that aren't per-session: the back-compat top-level
 *     `active_session_id` and the JSON-blob payload columns
 *     (`pending_response`, `muted_threads` — opaque to the schema,
 *     read/written verbatim by the owner skill). `CHECK(id = 1)`
 *     enforces the singleton contract the same way #297's
 *     `email_state` does.
 *
 * Notes:
 *   - The JSON-era `seen_email_ids` field intentionally does NOT
 *     have a column here — it relocates to `email_seen_ids` from
 *     state-005 (#297) where both check-email writers can target it
 *     without the two-file consolidate dance.
 *   - `session_id` is nullable on `trusted_sessions` because the
 *     SDK's session-id call can fail (sqlite-error fallback path);
 *     the row still exists with `started`/`epoch`/`last_seen` so
 *     readers can distinguish "never registered" from "registered
 *     but session-id pending".
 *   - `schema_version` on every table from day one per
 *     `coding-policy: stateful-artifacts` (lessons applied from
 *     #295's two-PR restoration cycle).
 *   - Like prior migrations in this epic, this only adds the
 *     schema. Data import + tile-side rewrites (register-session.py,
 *     trusted-memory state-schema.md + SKILL.md, heartbeat-precheck.py,
 *     status SKILL, retiring session-state.json.lock) are follow-up
 *     PRs.
 */
export const STATE_006_TRUSTED_SESSION_STATE: StateMigration = {
  version: 6,
  name: 'trusted_sessions + trusted_session_singleton tables (#298)',
  sql: `
    CREATE TABLE trusted_sessions (
      session_name   TEXT PRIMARY KEY,
      session_id     TEXT,
      started        TEXT NOT NULL,
      epoch          INTEGER NOT NULL,
      last_seen      TEXT NOT NULL,
      schema_version INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE trusted_session_singleton (
      id                INTEGER PRIMARY KEY CHECK(id = 1),
      active_session_id TEXT,
      pending_response  TEXT,
      muted_threads     TEXT,
      schema_version    INTEGER NOT NULL DEFAULT 1
    );
  `,
};
