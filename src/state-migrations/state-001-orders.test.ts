import Database from 'better-sqlite3';
import { describe, it, expect } from 'vitest';

import { applyStateMigrations } from '../db.js';

import { STATE_001_ORDERS } from './state-001-orders.js';

describe('state-001-orders', () => {
  it('creates the orders + orders_metadata tables and bumps user_version to 1', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [STATE_001_ORDERS]);

      expect(Number(database.pragma('user_version', { simple: true }))).toBe(1);

      const tables = database
        .prepare(
          "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name",
        )
        .all() as Array<{ name: string }>;
      const tableNames = tables.map((t) => t.name);
      expect(tableNames).toContain('orders');
      expect(tableNames).toContain('orders_metadata');
    } finally {
      database.close();
    }
  });

  it('declares every column from the issue spec on the orders table', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [STATE_001_ORDERS]);

      const cols = database
        .prepare('PRAGMA table_info(orders)')
        .all() as Array<{
        name: string;
        type: string;
        notnull: number;
        dflt_value: string | null;
        pk: number;
      }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));

      // PK
      expect(byName['id']).toMatchObject({ type: 'TEXT', pk: 1 });
      // NOT NULL columns from the issue spec
      expect(byName['source']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['status']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['description']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['order_date']).toMatchObject({ type: 'TEXT', notnull: 1 });
      expect(byName['email_message_id']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      expect(byName['last_updated']).toMatchObject({
        type: 'TEXT',
        notnull: 1,
      });
      // Nullable columns
      expect(byName['amount']).toMatchObject({ type: 'REAL', notnull: 0 });
      expect(byName['currency']).toMatchObject({ type: 'TEXT', notnull: 0 });
      expect(byName['expected_delivery']).toMatchObject({
        type: 'TEXT',
        notnull: 0,
      });
      expect(byName['to_address']).toMatchObject({ type: 'TEXT', notnull: 0 });
      expect(byName['flag_reason']).toMatchObject({
        type: 'TEXT',
        notnull: 0,
      });
      // flagged: NOT NULL with default 0 — boolean-equivalent
      expect(byName['flagged']).toMatchObject({
        type: 'INTEGER',
        notnull: 1,
        dflt_value: '0',
      });
    } finally {
      database.close();
    }
  });

  it('enforces the email_message_id UNIQUE constraint', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [STATE_001_ORDERS]);

      const insert = database.prepare(
        `INSERT INTO orders (id, source, status, description, order_date, email_message_id, last_updated)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      );
      insert.run(
        'amazon-2026-04-29-aaa',
        'amazon',
        'shipped',
        'Widget',
        '2026-04-29',
        'msg-123',
        '2026-04-29T00:00:00.000Z',
      );
      // Same email_message_id with a different `id` must conflict —
      // this is the property that lets the merge step collapse to
      // INSERT ... ON CONFLICT(email_message_id) DO UPDATE in the
      // tile-side rewrite.
      expect(() =>
        insert.run(
          'amazon-2026-04-29-bbb',
          'amazon',
          'shipped',
          'Widget',
          '2026-04-29',
          'msg-123',
          '2026-04-29T00:00:00.000Z',
        ),
      ).toThrow(/UNIQUE constraint failed: orders.email_message_id/);
    } finally {
      database.close();
    }
  });

  it('creates the indexes the issue lists', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [STATE_001_ORDERS]);

      const indexes = database
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index'")
        .all() as Array<{ name: string }>;
      const indexNames = indexes.map((i) => i.name);
      expect(indexNames).toContain('idx_orders_source_status');
      expect(indexNames).toContain('idx_orders_order_date');
    } finally {
      database.close();
    }
  });

  it('orders_metadata is a kv table with TEXT key PK', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [STATE_001_ORDERS]);

      const cols = database
        .prepare('PRAGMA table_info(orders_metadata)')
        .all() as Array<{ name: string; type: string; pk: number }>;
      const byName = Object.fromEntries(cols.map((c) => [c.name, c]));
      expect(byName['key']).toMatchObject({ type: 'TEXT', pk: 1 });
      expect(byName['value']).toMatchObject({ type: 'TEXT', pk: 0 });
    } finally {
      database.close();
    }
  });
});
