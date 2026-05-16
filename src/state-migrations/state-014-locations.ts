import type { StateMigration } from '../db.js';

/**
 * #574 Phase 3 — Location telemetry table.
 *
 * Static pins, venue shares, and live-location ticks from Telegram
 * (and future channels) all land here. Kept SEPARATE from the
 * `messages` table because:
 *
 *   1. Live-location updates fire ~once per minute for the full
 *      `live_period` (up to 8 hours per Telegram share). Routing
 *      them through `messages` would flood the agent's chat-history
 *      context with position lines that carry no conversational
 *      signal.
 *
 *   2. The host-side TZ resolver (#574 Phase 2) reads "most-recent
 *      owner coords" via a clean SQL query against this table.
 *      Filtering against the messages-with-placeholder pattern would
 *      couple the resolver to the placeholder format string.
 *
 * No FK to `chats` — locations are auxiliary telemetry and are not
 * tied to the registered-groups lifecycle.  A row from a
 * since-unregistered chat still carries valid TZ signal until pruned.
 *
 * `source` discriminates the four shapes Telegram delivers:
 *   - 'static'       — one-time location pin (no live_period)
 *   - 'venue'        — Telegram `message:venue` with nested location
 *   - 'live_initial' — first event of a live share (live_period > 0)
 *   - 'live_update'  — `edited_message:location` tick; same message_id
 *                      as the corresponding `live_initial` row
 *
 * The TZ resolver's hot query is "most recent for this sender":
 * `idx_locations_sender_time` (sender, recorded_at DESC) satisfies it
 * without a table scan. `chat_jid` is unindexed because the resolver
 * doesn't filter on it — Telegram broadcasts the same live-share to
 * every chat the owner enables it in, and we want the freshest signal
 * regardless of which chat surfaced it.
 */
export const STATE_014_LOCATIONS: StateMigration = {
  version: 14,
  name: 'locations table (#574 Phase 3)',
  sql: `
    CREATE TABLE locations (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      chat_jid    TEXT    NOT NULL,
      sender      TEXT    NOT NULL,
      message_id  TEXT    NOT NULL,
      latitude    REAL    NOT NULL,
      longitude   REAL    NOT NULL,
      accuracy_m  REAL,
      -- CHECK constraint keeps the four-value LocationSource enum
      -- honest at the DB layer instead of relying on every caller's
      -- discipline; an INSERT with an out-of-set source value fails
      -- loudly instead of silently breaking the Phase 2 resolver's
      -- source-aware logic. Keep this in lock-step with the
      -- LocationSource union in src/types.ts — any new source value
      -- ships in a follow-up state-NNN migration that recreates the
      -- table with the expanded set (SQLite has no ALTER TABLE
      -- equivalent for CHECK constraints).
      source      TEXT    NOT NULL
        CHECK (source IN ('static', 'venue', 'live_initial', 'live_update')),
      recorded_at TEXT    NOT NULL,
      live_period INTEGER
    );
    CREATE INDEX idx_locations_sender_time
      ON locations(sender, recorded_at DESC);
  `,
};
