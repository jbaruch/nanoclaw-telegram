import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';

// Each scenario builds an isolated tempDir + chdir so the
// CWD-rooted `STORE_DIR` and `GROUPS_DIR` constants in `src/config.ts`
// resolve under it. Mirrors the pattern already used in
// `src/db-migration.test.ts`.

interface OrdersJsonShape {
  orders: Array<Record<string, unknown>>;
  last_checked?: string;
  last_updated?: string;
}

function writeOrdersFile(
  tempDir: string,
  groupName: string,
  payload: OrdersJsonShape,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'orders-db.json');
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  return filePath;
}

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-orders-'));
  try {
    process.chdir(tempDir);
    fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

describe('orders-db.json → SQLite migration (#294)', () => {
  it('imports every order, sets metadata kv pairs, and renames the source file', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeOrdersFile(tempDir, 'telegram_swarm', {
        orders: [
          {
            id: 'amazon-2026-04-01-aaa',
            source: 'amazon',
            status: 'shipped',
            amount: 19.99,
            currency: 'USD',
            description: 'Widget',
            order_date: '2026-04-01',
            expected_delivery: '2026-04-05',
            email_message_id: 'msg-aaa',
            to_address: 'user@example.com',
            flagged: false,
            flag_reason: null,
            last_updated: '2026-04-02T00:00:00.000Z',
          },
          {
            id: 'shopify-2026-04-10-bbb',
            source: 'shopify',
            status: 'delivered',
            amount: 49.5,
            currency: 'USD',
            description: 'Doohickey',
            order_date: '2026-04-10',
            expected_delivery: null,
            email_message_id: 'msg-bbb',
            to_address: 'user@example.com',
            flagged: true,
            flag_reason: 'Large purchase: $49.50',
            last_updated: '2026-04-11T00:00:00.000Z',
          },
        ],
        last_checked: '2026-04-12T00:00:00.000Z',
        last_updated: '2026-04-11T00:00:00.000Z',
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        // Assert against the on-disk DB so the test exercises the full
        // file-IO path (not a private in-memory test seam).
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const orders = db
            .prepare('SELECT * FROM orders ORDER BY id')
            .all() as Array<Record<string, unknown>>;
          expect(orders).toHaveLength(2);
          expect(orders[0]).toMatchObject({
            id: 'amazon-2026-04-01-aaa',
            source: 'amazon',
            status: 'shipped',
            email_message_id: 'msg-aaa',
            flagged: 0,
          });
          expect(orders[1]).toMatchObject({
            id: 'shopify-2026-04-10-bbb',
            flagged: 1,
            flag_reason: 'Large purchase: $49.50',
          });

          const metadata = db
            .prepare('SELECT key, value FROM orders_metadata ORDER BY key')
            .all() as Array<{ key: string; value: string }>;
          expect(metadata).toEqual([
            { key: 'last_checked', value: '2026-04-12T00:00:00.000Z' },
            { key: 'last_updated', value: '2026-04-11T00:00:00.000Z' },
          ]);
        } finally {
          db.close();
        }

        // Source file renamed to .migrated-<YYYY-MM-DD> — the version
        // gate alone wouldn't prevent a second migration on next start;
        // the rename is what makes a re-run a no-op.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('orders-db.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('is a no-op when no group folder has orders-db.json', async () => {
    await runWithTempDir(async (tempDir) => {
      // Group folder exists but no orders-db.json — common case for
      // every group except admin.
      fs.mkdirSync(path.join(tempDir, 'groups', 'telegram_jj-dads'), {
        recursive: true,
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const count = (
            db.prepare('SELECT COUNT(*) AS n FROM orders').get() as {
              n: number;
            }
          ).n;
          expect(count).toBe(0);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('is idempotent across re-imports (same email_message_id, ON CONFLICT DO NOTHING)', async () => {
    await runWithTempDir(async (tempDir) => {
      const order = {
        id: 'amazon-2026-04-01-zzz',
        source: 'amazon',
        status: 'shipped',
        amount: 10,
        currency: 'USD',
        description: 'Thing',
        order_date: '2026-04-01',
        expected_delivery: null,
        email_message_id: 'msg-dup',
        to_address: 'user@example.com',
        flagged: false,
        flag_reason: null,
        last_updated: '2026-04-02T00:00:00.000Z',
      };
      writeOrdersFile(tempDir, 'group-a', { orders: [order] });
      writeOrdersFile(tempDir, 'group-b', { orders: [order] });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // ON CONFLICT(email_message_id) DO NOTHING — only the first
          // group's row is inserted, second is silently skipped.
          const count = (
            db.prepare('SELECT COUNT(*) AS n FROM orders').get() as {
              n: number;
            }
          ).n;
          expect(count).toBe(1);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('handles PK id collisions via bare ON CONFLICT DO NOTHING (deterministic by folder sort)', async () => {
    // Two distinct emails with the same source + order_date + description
    // (e.g. a resent confirmation) produce the same `id` — the hash is
    // SHA1(description)[:8] so collisions are possible even with different
    // email_message_id values. For one-shot migration we want the bare
    // ON CONFLICT DO NOTHING to absorb both constraints; the deterministic
    // winner is the alphabetically-first group folder (per the .sort()
    // applied before iteration).
    await runWithTempDir(async (tempDir) => {
      const sharedShape = {
        id: 'amazon-2026-04-01-aaaaaaaa',
        source: 'amazon',
        status: 'shipped',
        description: 'Same product description',
        order_date: '2026-04-01',
        last_updated: '2026-04-02T00:00:00.000Z',
      };
      writeOrdersFile(tempDir, 'b-second-folder', {
        orders: [{ ...sharedShape, email_message_id: 'msg-from-second' }],
      });
      writeOrdersFile(tempDir, 'a-first-folder', {
        orders: [{ ...sharedShape, email_message_id: 'msg-from-first' }],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const rows = db
            .prepare('SELECT id, email_message_id FROM orders')
            .all() as Array<{ id: string; email_message_id: string }>;
          expect(rows).toHaveLength(1);
          // Alphabetical sort puts a-first-folder ahead of b-second-folder,
          // so its email_message_id is the one that survives.
          expect(rows[0].email_message_id).toBe('msg-from-first');
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('skips reserved/non-group directories (e.g. global) via isValidGroupFolder', async () => {
    await runWithTempDir(async (tempDir) => {
      // `global` is a reserved folder per src/group-folder.ts —
      // dropping a stray orders-db.json there must not be migrated
      // (the migration would otherwise treat it as a group's data and
      // silently merge unrelated rows). Mirrors the filter the rest
      // of the orchestrator's group-path handling already uses.
      writeOrdersFile(tempDir, 'global', {
        orders: [
          {
            id: 'amazon-2026-04-01-glb',
            source: 'amazon',
            status: 'shipped',
            description: 'should-be-skipped',
            order_date: '2026-04-01',
            email_message_id: 'msg-glb',
            last_updated: '2026-04-02T00:00:00.000Z',
          },
        ],
      });
      // Plus a real group's file to confirm the filter is selective,
      // not a blanket skip.
      writeOrdersFile(tempDir, 'telegram_main', {
        orders: [
          {
            id: 'amazon-2026-04-01-real',
            source: 'amazon',
            status: 'shipped',
            description: 'real-order',
            order_date: '2026-04-01',
            email_message_id: 'msg-real',
            last_updated: '2026-04-02T00:00:00.000Z',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const ids = db
            .prepare('SELECT id FROM orders ORDER BY id')
            .all() as Array<{ id: string }>;
          expect(ids.map((r) => r.id)).toEqual(['amazon-2026-04-01-real']);
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('skips a malformed JSON file without aborting the pass', async () => {
    await runWithTempDir(async (tempDir) => {
      // Mix one valid and one malformed file. Malformed must NOT
      // abort the pass — other groups still need to migrate.
      const goodFile = writeOrdersFile(tempDir, 'good', {
        orders: [
          {
            id: 'amazon-2026-04-01-good',
            source: 'amazon',
            status: 'shipped',
            description: 'Good',
            order_date: '2026-04-01',
            email_message_id: 'msg-good',
            last_updated: '2026-04-02T00:00:00.000Z',
          },
        ],
      });
      const badFolder = path.join(tempDir, 'groups', 'bad');
      fs.mkdirSync(badFolder, { recursive: true });
      const badFile = path.join(badFolder, 'orders-db.json');
      fs.writeFileSync(badFile, '{ this is not valid json');

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const ids = db.prepare('SELECT id FROM orders').all() as Array<{
            id: string;
          }>;
          expect(ids.map((r) => r.id)).toEqual(['amazon-2026-04-01-good']);
        } finally {
          db.close();
        }
        // Good file renamed; bad file left in place for human triage.
        expect(fs.existsSync(goodFile)).toBe(false);
        expect(fs.existsSync(badFile)).toBe(true);
      } finally {
        _closeDatabase();
      }
    });
  });
});
