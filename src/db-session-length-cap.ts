// Session-length cap state accessors (#413; #751 seam 3, extracted
// verbatim from src/db.ts). Cumulative per-session totals enforced by
// the orchestrator's session-length cap. See
// `src/session-length-cap.schema.md` for the full schema doc; the
// state-011 migration creates the table. Only the orchestrator
// (`runAgent` in `src/message-pipeline.ts`) writes it. The cap
// *logic* (thresholds, handoff prefix) lives in
// `src/session-length-cap.ts`; this module owns only the rows.
import { db } from './db-connection.js';

export interface SessionLengthRow {
  group_folder: string;
  session_name: string;
  session_id: string;
  total_input_tokens: number;
  turn_count: number;
  last_handoff_summary: string | null;
  marked_for_reset: number;
  reset_reason: string | null;
  reset_cap: number | null;
  started_at: string;
  last_updated_at: string;
}

export type SessionResetReason = 'token_cap' | 'turn_cap';

/**
 * Read the cap-state row for a given session slot, or `undefined` if
 * no row exists yet (first turn ever, or post-reset before the next
 * turn writes the new row). Pure read — does not migrate.
 */
export function getSessionLengthState(
  groupFolder: string,
  sessionName: string,
): SessionLengthRow | undefined {
  const row = db
    .prepare(
      `SELECT group_folder, session_name, session_id,
              total_input_tokens, turn_count, last_handoff_summary,
              marked_for_reset, reset_reason, reset_cap,
              started_at, last_updated_at
         FROM session_length_state
         WHERE group_folder = ? AND session_name = ?`,
    )
    .get(groupFolder, sessionName) as SessionLengthRow | undefined;
  return row;
}

/**
 * Record one assistant turn's `usage.input_tokens` against the
 * session's cumulative totals. UPSERT semantics:
 *
 *   - No row: INSERT with `total_input_tokens = inputTokens`,
 *     `turn_count = 1`, `started_at = last_updated_at = now`.
 *   - Row exists, same `session_id`: ADD `inputTokens` to
 *     `total_input_tokens`, increment `turn_count`, bump
 *     `last_updated_at`. `marked_for_reset` is preserved.
 *   - Row exists, DIFFERENT `session_id`: rare drift surface — the
 *     SDK chain rotated under us without going through the
 *     orchestrator's reset path. Treat as a fresh row: REPLACE with
 *     the new session's totals (defensive; we'd rather attribute
 *     correctly than carry stale sums).
 *
 * `lastHandoffSummary`, when non-null, is written verbatim on INSERT
 * (or on the same-session-id UPDATE only when the existing column is
 * NULL — preserves the original handoff across many turns of the
 * post-reset session). Pass `null` for normal turns.
 *
 * Returns the row that the next threshold check should consult.
 */
export function recordSessionTurn(
  groupFolder: string,
  sessionName: string,
  sessionId: string,
  inputTokens: number,
  lastHandoffSummary: string | null,
): SessionLengthRow {
  const now = new Date().toISOString();
  const existing = getSessionLengthState(groupFolder, sessionName);
  if (!existing || existing.session_id !== sessionId) {
    db.prepare(
      `INSERT OR REPLACE INTO session_length_state (
         group_folder, session_name, session_id,
         total_input_tokens, turn_count, last_handoff_summary,
         marked_for_reset, started_at, last_updated_at, schema_version
       ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, 1)`,
    ).run(
      groupFolder,
      sessionName,
      sessionId,
      inputTokens,
      1,
      lastHandoffSummary,
      now,
      now,
    );
  } else {
    db.prepare(
      `UPDATE session_length_state
         SET total_input_tokens = total_input_tokens + ?,
             turn_count = turn_count + 1,
             last_updated_at = ?,
             last_handoff_summary = COALESCE(last_handoff_summary, ?)
         WHERE group_folder = ? AND session_name = ?`,
    ).run(inputTokens, now, lastHandoffSummary, groupFolder, sessionName);
  }
  // Re-read so the caller gets the post-update view in one consistent
  // shape — small extra read in exchange for the threshold-check call
  // site never having to reason about INSERT-vs-UPDATE semantics.
  const row = getSessionLengthState(groupFolder, sessionName);
  if (!row) {
    // Should be unreachable: we just upserted. Throw a specific error
    // so a future schema regression surfaces loudly rather than
    // returning a fictional zero-state.
    throw new Error(
      `recordSessionTurn: row missing immediately after upsert (group=${groupFolder} session=${sessionName})`,
    );
  }
  return row;
}

/**
 * Set `marked_for_reset = 1` and capture the reason + cap value on
 * the cap-state row. Idempotent: a second call is a no-op (the
 * WHERE clause excludes already-marked rows so the UPDATE doesn't
 * churn `last_updated_at` and doesn't overwrite the reason that
 * actually fired first). Returns the number of rows changed (0 =
 * already marked / no row, 1 = newly marked).
 *
 * `reason` and `cap` are stored at mark-time so the consume path on
 * the next inbound spawn can format a precise user-facing
 * notification without re-deriving (env-driven thresholds can
 * change between mark and consume; the notification reflects what
 * the operator actually had configured at trip time).
 */
export function markSessionForReset(
  groupFolder: string,
  sessionName: string,
  reason: SessionResetReason,
  cap: number,
): number {
  const result = db
    .prepare(
      `UPDATE session_length_state
         SET marked_for_reset = 1,
             reset_reason = ?,
             reset_cap = ?
         WHERE group_folder = ?
           AND session_name = ?
           AND marked_for_reset = 0`,
    )
    .run(reason, cap, groupFolder, sessionName);
  return result.changes;
}

/**
 * Atomic "consume the pending reset" — reads the marked-for-reset
 * row's `session_id` + `last_handoff_summary`, then deletes the row,
 * inside a single transaction. Returns `null` when no reset was
 * pending so the caller can distinguish "no-op normal spawn" from
 * "post-reset spawn that needs the handoff prefix".
 *
 * The DELETE is deliberate — the next assistant turn under the new
 * `session_id` will re-INSERT a fresh row via `recordSessionTurn`.
 * Carrying the old row forward and overwriting in place would risk
 * a misattributed turn if the threshold-check fires before the
 * orchestrator has a chance to write the new `session_id`.
 */
export interface PendingReset {
  sessionId: string;
  lastHandoffSummary: string | null;
  reason: SessionResetReason;
  cap: number;
}

export function consumeSessionReset(
  groupFolder: string,
  sessionName: string,
): PendingReset | null {
  const consume = db.transaction(() => {
    const row = db
      .prepare(
        `SELECT session_id, last_handoff_summary, reset_reason, reset_cap
           FROM session_length_state
           WHERE group_folder = ?
             AND session_name = ?
             AND marked_for_reset = 1`,
      )
      .get(groupFolder, sessionName) as
      | {
          session_id: string;
          last_handoff_summary: string | null;
          reset_reason: string | null;
          reset_cap: number | null;
        }
      | undefined;
    if (!row) return null;
    db.prepare(
      `DELETE FROM session_length_state
         WHERE group_folder = ? AND session_name = ?`,
    ).run(groupFolder, sessionName);
    // Validate the persisted reason. A NULL or unknown value
    // indicates the row was marked by a code path that didn't go
    // through `markSessionForReset` (corruption, manual SQL,
    // future-version migration leaving NULL). Fall back to
    // `'token_cap'` so the notification path still has a defined
    // shape; the diagnostic log line at the call site carries the
    // raw row for forensics.
    const reason: SessionResetReason =
      row.reset_reason === 'turn_cap' ? 'turn_cap' : 'token_cap';
    return {
      sessionId: row.session_id,
      lastHandoffSummary: row.last_handoff_summary,
      reason,
      cap: row.reset_cap ?? 0,
    };
  });
  return consume();
}

/**
 * Drop all cap-state rows for a group. Called from nuke handlers so
 * a force-reset wipes the cap accounting alongside the SDK session.
 * Best-effort — never throws on missing row.
 */
export function clearSessionLengthStateForGroup(groupFolder: string): number {
  const result = db
    .prepare('DELETE FROM session_length_state WHERE group_folder = ?')
    .run(groupFolder);
  return result.changes;
}
