import type { StateMigration } from '../db.js';

/**
 * #294 — Migrate orders-db.json (per-group JSON state file) to a
 * shared `orders` SQLite table in messages.db.
 *
 * This migration adds the schema only. The data import (existing
 * JSON file → table rows) and the per-skill consumer rewrites
 * (`nanoclaw-admin/skills/check-orders`, `nanoclaw-admin/skills/morning-brief`)
 * land in follow-up PRs through the tile staging→promote pipeline.
 *
 * Schema rationale:
 *   - `id` is `{source}-{order_date}-{hash}` to match the existing
 *     identifier shape in orders-db.json (preserves cross-skill
 *     references during the transition).
 *   - `email_message_id UNIQUE` lets the merge step (currently in
 *     `merge-orders-db.py`) collapse to
 *     `INSERT ... ON CONFLICT(email_message_id) DO UPDATE` — no more
 *     read-modify-write or LOCK_EX dance.
 *   - `flagged` is INTEGER (0/1) since SQLite doesn't have a native
 *     boolean and TEXT 'true'/'false' would cost a sortable index.
 *   - `idx_orders_source_status` accelerates the most common admin
 *     query ("amazon orders not yet shipped") and
 *     `idx_orders_order_date` accelerates morning-brief's date-window
 *     lookups.
 *   - `orders_metadata` is a kv table for `last_checked` and
 *     `last_updated` markers that the JSON file currently kept as
 *     siblings of the `orders` array.
 */
export const STATE_001_ORDERS: StateMigration = {
  version: 1,
  name: 'orders + orders_metadata tables (#294)',
  sql: `
    CREATE TABLE orders (
      id                TEXT PRIMARY KEY,
      source            TEXT NOT NULL,
      status            TEXT NOT NULL,
      amount            REAL,
      currency          TEXT,
      description       TEXT NOT NULL,
      order_date        TEXT NOT NULL,
      expected_delivery TEXT,
      email_message_id  TEXT NOT NULL UNIQUE,
      to_address        TEXT,
      flagged           INTEGER NOT NULL DEFAULT 0,
      flag_reason       TEXT,
      last_updated      TEXT NOT NULL
    );
    CREATE INDEX idx_orders_source_status ON orders(source, status);
    CREATE INDEX idx_orders_order_date ON orders(order_date);

    CREATE TABLE orders_metadata (
      key   TEXT PRIMARY KEY,
      value TEXT
    );
  `,
};
