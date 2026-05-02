# `session_length_state` schema (#413)

Per-session running totals tracked by the orchestrator's session-length
cap. Owner skill: orchestrator (`src/index.ts` runAgent +
`src/session-length-cap.ts`). Created by state migration
`state-011-session-length-cap`.

## Why

Without an upper bound on a Claude Code session's lifetime, a single
session keeps re-reading the entire prior context as `cache_read` on
every turn — observed at >18M cache_read in a single query. The SDK's
auto-compact summarises but does not delete; summary plus subsequent
turns keep growing on top. This table is the per-session accounting
that lets the orchestrator force a reset on configurable cumulative
thresholds.

Distinct from the existing `sessions` table (orchestrator's
`(group_folder, session_name) -> session_id` mapping for SDK resumption)
and from the per-turn kill-auto-compaction threshold (which fires on a
single turn's `input_tokens` crossing a percentage of the
context-window). This cap is **cumulative across turns** — a session
can stay below the per-turn threshold for hours while accumulating
past the session-cumulative cap.

## Columns

| Column | Type | Notes |
|---|---|---|
| `group_folder` | TEXT NOT NULL | Part of composite PK; matches `sessions.group_folder`. |
| `session_name` | TEXT NOT NULL | Part of composite PK; matches `sessions.session_name`. |
| `session_id` | TEXT NOT NULL | The SDK session-id these totals are attributed to. Row is deleted on reset; the next assistant turn re-creates it under the new `session_id`. |
| `total_input_tokens` | INTEGER NOT NULL DEFAULT 0 | Running sum of every assistant turn's `usage.input_tokens`. Approximates cumulative context size; not a billing total. |
| `turn_count` | INTEGER NOT NULL DEFAULT 0 | Count of assistant turns with a `usage` payload. |
| `last_handoff_summary` | TEXT | Brief context prefix injected when this session was created after a reset. NULL on the first session of a chain. |
| `marked_for_reset` | INTEGER NOT NULL DEFAULT 0 | Set to 1 AFTER a turn completes when threshold is exceeded. The actual reset happens on the NEXT inbound message — never mid-turn. |
| `reset_reason` | TEXT | `'token_cap'` or `'turn_cap'`, captured at mark-time so the consume path can surface the precise cause without re-deriving it. NULL while the row is accumulating normally. |
| `reset_cap` | INTEGER | Configured cap value at the moment the row was marked. Captured because env-driven thresholds can change between mark and consume; the user-facing notification reflects what the operator had configured at trip time. NULL while the row is accumulating normally. |
| `started_at` | TEXT NOT NULL | ISO-8601 wall-clock when the row was first inserted. |
| `last_updated_at` | TEXT NOT NULL | ISO-8601 wall-clock of the most recent token-accumulation write. |
| `schema_version` | INTEGER NOT NULL DEFAULT 1 | Bump on any shape change per `rules/stateful-artifacts.md`. |

PK: `(group_folder, session_name)`.

## Writer / reader contract

**Writers (orchestrator only):**

- `recordTurn(groupFolder, sessionName, sessionId, inputTokens)` —
  upsert. If the row's `session_id` differs from the supplied value,
  reset totals to zero (the SDK chain rotated under us; this is
  defensive, not a normal path). Otherwise add `inputTokens` to
  `total_input_tokens` and increment `turn_count`. Updates
  `last_updated_at`.
- `markForReset(groupFolder, sessionName, reason, cap)` — sets
  `marked_for_reset = 1` and persists `reset_reason` + `reset_cap`
  so the consume path can surface a precise notification without
  re-deriving the threshold. Idempotent on the row but only marks
  rows whose `marked_for_reset` is currently 0 (so a second mark in
  the same turn is a no-op rather than overwriting the original
  cause). Implemented as `markSessionForReset` in `src/db.ts`.
- `consumeReset(groupFolder, sessionName)` — atomically reads-and-deletes
  the marked row, returning
  `{ sessionId, lastHandoffSummary, reason, cap }` if present.
  Used by the next inbound spawn to know it's the fresh-session path
  and to build the user-facing notification from the persisted
  reason + cap rather than re-reading env (which may have changed
  between mark and consume). **Returns null when no reset was
  pending** so callers can distinguish "first turn ever" from
  "post-reset first turn". Implemented as `consumeSessionReset` in
  `src/db.ts`.
- `clearForGroup(groupFolder)` — best-effort cleanup on group nuke.

**Readers:**

- None outside the orchestrator. Treat the table as opaque
  orchestrator state.

## Hints, not authority

This row is a last-seen snapshot: cumulative sums and the marked-for-reset
flag. Before acting on any value, callers verify the live SDK session
chain (`sessions` table) is still intact — a stale row pointing at a
session that no longer exists in `sessions` is dropped on the next
turn (the writer detects the mismatch via `session_id`).

## Migration policy

- Bump `schema_version` for any shape change.
- Owner skill (orchestrator) migrates on its own read; reader skills
  (none today) must not migrate.
- Renaming or removing the table: surface-sync per
  `rules/context-artifacts.md` — update this doc, update every reader,
  add a CHANGELOG entry.

## Lifecycle

```
                    +----------------------+
   first turn       | INSERT row           |
   ─────────────►   | (totals from turn 1) |
                    +----------+-----------+
                               │
                               ▼  every subsequent turn
                    +----------+-----------+
                    | UPDATE: add tokens,  |
                    | increment turn_count |
                    +----------+-----------+
                               │
                turn that crosses cumulative threshold
                               │
                               ▼
                    +----------+-----------+
                    | marked_for_reset = 1 |
                    +----------+-----------+
                               │
                       NEXT inbound message
                               │
                               ▼
                    +----------+-----------+
                    | DELETE row, build    |
                    | handoff prefix,      |
                    | clear sessions row,  |
                    | notify user          |
                    +----------+-----------+
                               │
                  fresh session re-enters at top
```
