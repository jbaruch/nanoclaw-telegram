import type { StateMigration } from '../db.js';

/**
 * #300 — Migrate `/workspace/group/calendar-state.json` to two SQLite
 * tables, replacing the per-group JSON envelope-with-events-array
 * shape. The acceptance criterion the issue calls out — "check-calendar's
 * `reminder_task_id` updates are single-row SQL UPDATEs" — becomes a
 * structural property here: with `event_id` as the PK on
 * `calendar_events`, an `UPDATE calendar_events SET reminder_task_id = ?
 * WHERE event_id = ?` touches exactly one row, regardless of how many
 * other events share the same date. The JSON-era code path had to read
 * the whole snapshot, mutate one entry in the events array, and write
 * the whole snapshot back, racing with concurrent writers on the
 * sibling fields the same way the morning-brief-pending JSON did
 * (lessons applied from state-007 / #299).
 *
 * Ownership / writer-reader contract:
 *   - `calendar_snapshots` — written by `morning-brief` Step 18
 *     (one row per fetched day, keyed on the ISO date). Daily rotation
 *     becomes `DELETE FROM calendar_snapshots WHERE date < ?`, which
 *     cascades to `calendar_events` via the foreign-key (see
 *     "Cascade rationale" below). `check-calendar`'s "do we already
 *     have today?" check becomes
 *     `SELECT 1 FROM calendar_snapshots WHERE date = ?`.
 *   - `calendar_events` — written by `morning-brief` Step 18 alongside
 *     the snapshot row; read by `check-calendar` Steps 1–5; updated by
 *     `check-calendar` (single-row `UPDATE calendar_events SET
 *     reminder_task_id = ? WHERE event_id = ?` when a reminder Task is
 *     created for the event).
 *
 * Schema rationale:
 *   - `calendar_snapshots.date` is `TEXT PRIMARY KEY` — one snapshot
 *     per ISO date is the invariant the issue spec calls out, and it's
 *     also the FK target for `calendar_events.date`.
 *   - `calendar_events.event_id` is `TEXT PRIMARY KEY` — Google
 *     Calendar event IDs are externally meaningful and globally unique
 *     within an account, so PK uniqueness is the natural single-row
 *     UPDATE contract.
 *   - `calendar_events.title` and `start` are `NOT NULL` — a calendar
 *     event without a title or start time is unusable in the brief; let
 *     the DB enforce presence rather than letting a malformed insert
 *     land.
 *   - `calendar_events.end` is nullable — Google all-day events and
 *     some imported entries legitimately omit an end time.
 *   - `calendar_events.reminder_task_id` is nullable — events start
 *     life without a paired reminder Task; `check-calendar` populates
 *     this column as it creates Tasks. The "find unprocessed events"
 *     query is `WHERE date = ? AND reminder_task_id IS NULL`, served
 *     by `idx_calendar_events_date` — calendar tables hold one day's
 *     worth of events (typically <50 rows), so the per-date scan is
 *     trivial without a dedicated `(reminder_task_id)` index.
 *   - `schema_version` on both tables from day one per
 *     `coding-policy: stateful-artifacts` (lessons applied from #295's
 *     two-PR restoration cycle and reinforced in state-005 / state-006
 *     / state-007).
 *
 * Cascade rationale:
 *   - `calendar_events.date REFERENCES calendar_snapshots(date) ON
 *     DELETE CASCADE`. Daily snapshot rotation is the headline
 *     cleanup operation: `DELETE FROM calendar_snapshots WHERE date <
 *     ?` is intended to remove every event belonging to the rotated-
 *     out days in one shot. The cascade encodes that intent so a
 *     future "I'll just delete the snapshot row" refactor can't
 *     accidentally orphan events.
 *
 * IMPORTANT — foreign-key enforcement is OFF in production:
 *   SQLite requires `PRAGMA foreign_keys = ON` per-connection for FK
 *   constraints to actually fire. The orchestrator's `initDatabase()`
 *   in `src/db.ts` does NOT currently set this pragma (it sets
 *   `journal_mode = WAL`, `synchronous = NORMAL`, and `busy_timeout`,
 *   but not `foreign_keys`). The cascade therefore does NOT run in
 *   production today, so until that pragma is flipped globally the
 *   rotation contract is: the daily-rotation writer (morning-brief
 *   Step 18 follow-up PR) MUST explicitly delete the events first,
 *   e.g. inside a single transaction:
 *       BEGIN;
 *         DELETE FROM calendar_events    WHERE date < ?;
 *         DELETE FROM calendar_snapshots WHERE date < ?;
 *       COMMIT;
 *   The migration declares the FK + cascade anyway so that (a) the
 *   schema's intent is captured for future readers, and (b) once the
 *   orchestrator flips `foreign_keys = ON` (out of scope for this PR
 *   — it's a global pragma change with cross-table implications), the
 *   cascade becomes the source of truth and the explicit DELETE on
 *   `calendar_events` becomes redundant rather than load-bearing. The
 *   companion test enables `PRAGMA foreign_keys = ON` explicitly to
 *   lock down the schema's intent under enforced FKs.
 *
 * Like prior migrations in this epic, this PR only adds the schema.
 * Data import (extending `migrateJsonState()` to read existing per-
 * group `calendar-state.json` files) and tile-side rewrites
 * (`nanoclaw-admin/skills/morning-brief/SKILL.md` Step 18 writer,
 * `nanoclaw-admin/skills/check-calendar/SKILL.md` Steps 1–5 reader +
 * `reminder_task_id` updater) are follow-up PRs through the
 * staging→promote pipeline.
 */
export const STATE_008_CALENDAR_STATE: StateMigration = {
  version: 8,
  name: 'calendar_snapshots + calendar_events tables (#300)',
  sql: `
    CREATE TABLE calendar_snapshots (
      date           TEXT PRIMARY KEY,
      fetched_at     TEXT NOT NULL,
      schema_version INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE calendar_events (
      event_id         TEXT PRIMARY KEY,
      date             TEXT NOT NULL REFERENCES calendar_snapshots(date) ON DELETE CASCADE,
      title            TEXT NOT NULL,
      start            TEXT NOT NULL,
      end              TEXT,
      reminder_task_id TEXT,
      schema_version   INTEGER NOT NULL DEFAULT 1
    );

    CREATE INDEX idx_calendar_events_date ON calendar_events(date);
  `,
};
