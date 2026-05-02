import type { StateMigration } from '../db.js';

/**
 * #413 — Session-length cap. Tracks per-session running totals so the
 * orchestrator can force a session reset (new `session_id` + brief
 * context handoff) when cumulative input tokens or turn count crosses
 * a configurable threshold. Without this cap a single long-lived
 * session keeps re-reading the entire prior context as `cache_read` on
 * every turn — observed at >18M cache_read in a single query (issue
 * body cites the 2026-04-26 telegram_main run).
 *
 * Distinct from the existing `sessions` table (which is the
 * orchestrator's `(group_folder, session_name) -> session_id` mapping
 * for SDK resumption) and from the kill-auto-compaction per-turn
 * threshold (which fires on a *single* turn's `input_tokens` crossing
 * the context-window percentage). This cap is cumulative — a session
 * can stay below the per-turn threshold for hours while accumulating
 * past the session-cumulative cap.
 *
 * Owner skill: orchestrator (`src/index.ts` runAgent + the new
 * `src/session-length-cap.ts` module). The schema doc lives at
 * `src/session-length-cap.schema.md`. Per `rules/stateful-artifacts.md`
 * the table carries `schema_version` from day one so a future shape
 * change has an auditable migration path.
 *
 * Columns:
 *   - `(group_folder, session_name)` — composite PK matches the
 *     `sessions` table's key shape so the cap state lines up with the
 *     SDK session it tracks.
 *   - `session_id` — the SDK session-id this row's totals are
 *     attributed to. When the orchestrator resets the session, the
 *     row is deleted (the next assistant turn re-creates it under the
 *     new `session_id`).
 *   - `total_input_tokens` — running sum of every assistant turn's
 *     `usage.input_tokens` across the session. Each turn's value is
 *     "tokens the model saw on this turn" — summing approximates
 *     cumulative context size; it is NOT a billing total.
 *   - `turn_count` — number of assistant turns observed. Increments
 *     once per `wrappedOnOutput` invocation that carries a `usage`
 *     payload.
 *   - `last_handoff_summary` — the brief context-handoff prefix the
 *     orchestrator wrote when this session was created after a reset.
 *     NULL on the first session of a chain. Scoped to the current
 *     row only: `consumeSessionReset` deletes the row on reset, so
 *     a chain of N resets writes N independent rows and only the
 *     most recent handoff is retained. Earlier handoffs aren't
 *     preserved in this table — operators wanting full chain history
 *     should grep the orchestrator's structured logs for the
 *     `session_length_cap_reset` event.
 *   - `marked_for_reset` — set to 1 by the threshold checker AFTER
 *     a turn completes. The reset itself happens on the NEXT inbound
 *     message (not mid-turn) so we never yank the current request.
 *   - `reset_reason` — `'token_cap'` or `'turn_cap'`, captured at
 *     mark-time so the consume path can surface a precise user-facing
 *     notification without re-deriving it. NULL while the row is
 *     accumulating normally.
 *   - `reset_cap` — the configured cap value at the moment the row
 *     was marked. Captured because env-driven thresholds can change
 *     between mark and consume; the user-facing notification reflects
 *     what the operator actually configured at trip time.
 *   - `started_at`, `last_updated_at` — wall-clock TEXT (ISO-8601),
 *     for diagnostics. Not load-bearing.
 *   - `schema_version` — `INTEGER NOT NULL DEFAULT 1`.
 *
 * Migration notes:
 *   - Owner skill = orchestrator; only `runAgent` writes this row.
 *     Reader skills (none today) must not migrate per stateful-artifacts;
 *     they treat an unrecognized `schema_version` as "no usable prior
 *     state" and let the next owner-skill turn rewrite the row.
 *   - On rename/removal: bump `schema_version`, update the schema doc,
 *     update CHANGELOG. Reader code stays tolerant of a missing row
 *     (first-turn / fresh-deploy state).
 */
export const STATE_011_SESSION_LENGTH_CAP: StateMigration = {
  version: 11,
  name: 'session_length_state table (#413)',
  sql: `
    CREATE TABLE session_length_state (
      group_folder         TEXT NOT NULL,
      session_name         TEXT NOT NULL,
      session_id           TEXT NOT NULL,
      total_input_tokens   INTEGER NOT NULL DEFAULT 0,
      turn_count           INTEGER NOT NULL DEFAULT 0,
      last_handoff_summary TEXT,
      marked_for_reset     INTEGER NOT NULL DEFAULT 0,
      reset_reason         TEXT,
      reset_cap            INTEGER,
      started_at           TEXT NOT NULL,
      last_updated_at      TEXT NOT NULL,
      schema_version       INTEGER NOT NULL DEFAULT 1,
      PRIMARY KEY (group_folder, session_name)
    );
  `,
};
