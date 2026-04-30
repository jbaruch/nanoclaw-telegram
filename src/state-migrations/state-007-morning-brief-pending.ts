import type { StateMigration } from '../db.js';

/**
 * #299 — Migrate `morning-brief-pending.json` to three SQLite queue
 * tables, one per queue. Replaces the multi-key sibling-bucket JSON
 * shape with single-purpose tables so the "sibling-key preservation"
 * bug class disappears entirely (the breaking shape that the recent
 * helper PR `merge-pending-json.py` was patching around).
 *
 * Ownership / writer-reader contract:
 *   - `pending_cleanup_items` — written by `morning-brief`
 *     (cleanup queue produced from the daily brief). Cleared by
 *     `brief-cleanup` Step 9: `DELETE FROM pending_cleanup_items`
 *     replaces the JSON-era "set cleanup_items to []", which used to
 *     also clobber its sibling keys (`pending_decisions`,
 *     `undated_tasks`) when a writer raced. Targeted purges (e.g. a
 *     mute-filter sweep) become `DELETE FROM pending_cleanup_items
 *     WHERE id = ?`.
 *   - `pending_decisions` — written by `morning-brief` (decisions
 *     surfaced for human input).
 *   - `pending_undated_tasks` — appended by `nightly-housekeeping`
 *     Step 18 (Google Tasks rows missing a due date). Consumed by
 *     `morning-brief` to surface in the next brief.
 *
 * Schema rationale:
 *   - `id` is `TEXT PRIMARY KEY` on every table — the IDs are
 *     externally meaningful (Google Task IDs for undated_tasks; stable
 *     per-item IDs minted by morning-brief for the other two), so PK
 *     uniqueness is the natural cleanup contract.
 *   - `added` defaults to `CURRENT_TIMESTAMP` per the issue spec
 *     (these queues are not lex-ordered windowed-DELETE targets like
 *     `email_seen_ids` — `CURRENT_TIMESTAMP` is fine; readers don't
 *     mix it with explicit ISO-8601 writes).
 *   - `pending_decisions.question` and `pending_undated_tasks.title`
 *     + `tasklist_id` are `NOT NULL` — these queues are unusable
 *     without those fields, so the DB enforces presence rather than
 *     letting a malformed insert land.
 *   - `pending_cleanup_items.question / subject / sender` are
 *     nullable — different cleanup-item types populate different
 *     subsets of the trio (an email-cleanup row has subject + sender;
 *     a question-cleanup row has question only); the `type` column
 *     discriminates.
 *   - `schema_version` on every table from day one per
 *     `coding-policy: stateful-artifacts` (lessons applied from
 *     #295's two-PR restoration cycle and reinforced in state-005 /
 *     state-006).
 *
 * Like prior migrations in this epic, this PR only adds the schema.
 * Data import (extending `migrateJsonState()` to read existing
 * `morning-brief-pending.json` files) and tile-side rewrites
 * (`morning-brief-fetch.py`, `morning-brief/SKILL.md` Steps 4 + 20,
 * `brief-cleanup/SKILL.md` Steps 1 + 3 + 9, `nightly-housekeeping`
 * Step 18, retiring `merge-pending-json.py` + its test) are follow-up
 * PRs through the staging→promote pipeline.
 */
export const STATE_007_MORNING_BRIEF_PENDING: StateMigration = {
  version: 7,
  name: 'pending_cleanup_items + pending_decisions + pending_undated_tasks tables (#299)',
  sql: `
    CREATE TABLE pending_cleanup_items (
      id             TEXT PRIMARY KEY,
      type           TEXT NOT NULL,
      question       TEXT,
      subject        TEXT,
      sender         TEXT,
      added          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      schema_version INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE pending_decisions (
      id             TEXT PRIMARY KEY,
      question       TEXT NOT NULL,
      added          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      schema_version INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE pending_undated_tasks (
      id             TEXT PRIMARY KEY,
      title          TEXT NOT NULL,
      tasklist_id    TEXT NOT NULL,
      added          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      schema_version INTEGER NOT NULL DEFAULT 1
    );
  `,
};
