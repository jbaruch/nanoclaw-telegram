import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, expect, it, vi } from 'vitest';

describe('database migrations', () => {
  it('defaults Telegram backfill chats to direct messages', async () => {
    const repoRoot = process.cwd();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-db-test-'));

    try {
      process.chdir(tempDir);
      fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });

      const dbPath = path.join(tempDir, 'store', 'messages.db');
      const legacyDb = new Database(dbPath);
      legacyDb.exec(`
        CREATE TABLE chats (
          jid TEXT PRIMARY KEY,
          name TEXT,
          last_message_time TEXT
        );
      `);
      legacyDb
        .prepare(
          `INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)`,
        )
        .run('tg:12345', 'Telegram DM', '2024-01-01T00:00:00.000Z');
      legacyDb
        .prepare(
          `INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)`,
        )
        .run('tg:-10012345', 'Telegram Group', '2024-01-01T00:00:01.000Z');
      legacyDb
        .prepare(
          `INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)`,
        )
        .run('room@g.us', 'WhatsApp Group', '2024-01-01T00:00:02.000Z');
      legacyDb.close();

      vi.resetModules();
      const { initDatabase, getAllChats, _closeDatabase } =
        await import('./db.js');

      initDatabase();

      const chats = getAllChats();
      expect(chats.find((chat) => chat.jid === 'tg:12345')).toMatchObject({
        channel: 'telegram',
        is_group: 0,
      });
      expect(chats.find((chat) => chat.jid === 'tg:-10012345')).toMatchObject({
        channel: 'telegram',
        is_group: 0,
      });
      expect(chats.find((chat) => chat.jid === 'room@g.us')).toMatchObject({
        channel: 'whatsapp',
        is_group: 1,
      });

      _closeDatabase();
    } finally {
      process.chdir(repoRoot);
    }
  });

  it('adds telegram_message_id column + diagnostic index to a pre-existing messages table', async () => {
    const repoRoot = process.cwd();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-db-test-'));

    try {
      process.chdir(tempDir);
      fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });

      const dbPath = path.join(tempDir, 'store', 'messages.db');
      const legacyDb = new Database(dbPath);
      // Legacy shape: messages table WITHOUT telegram_message_id —
      // simulates an existing install that predates the feature. The
      // migration path (PRAGMA-gated ALTER + deferred CREATE INDEX)
      // must upgrade this in place without blocking startup.
      legacyDb.exec(`
        CREATE TABLE chats (
          jid TEXT PRIMARY KEY,
          name TEXT,
          last_message_time TEXT,
          channel TEXT,
          is_group INTEGER
        );
        CREATE TABLE messages (
          id TEXT,
          chat_jid TEXT,
          sender TEXT,
          sender_name TEXT,
          content TEXT,
          timestamp TEXT,
          is_from_me INTEGER,
          is_bot_message INTEGER DEFAULT 0,
          reply_to_message_id TEXT,
          reply_to_message_content TEXT,
          reply_to_sender_name TEXT,
          PRIMARY KEY (id, chat_jid),
          FOREIGN KEY (chat_jid) REFERENCES chats(jid)
        );
      `);
      legacyDb
        .prepare(
          `INSERT INTO chats (jid, name, last_message_time, channel, is_group) VALUES (?, ?, ?, ?, ?)`,
        )
        .run(
          'tg:-1003000000001',
          'Test Group',
          '2026-01-01T00:00:00.000Z',
          'telegram',
          1,
        );
      legacyDb.close();

      vi.resetModules();
      const {
        initDatabase,
        storeMessage,
        getBotMessageByTelegramId,
        _closeDatabase,
      } = await import('./db.js');

      // Must not throw — pre-fix, CREATE INDEX on a column that didn't
      // exist yet would throw "no such column: telegram_message_id"
      // and block all later boot steps.
      initDatabase();

      const upgradedDb = new Database(dbPath);
      const cols = upgradedDb
        .prepare('PRAGMA table_info(messages)')
        .all() as Array<{ name: string }>;
      expect(cols.some((c) => c.name === 'telegram_message_id')).toBe(true);

      const indexes = upgradedDb
        .prepare('PRAGMA index_list(messages)')
        .all() as Array<{ name: string }>;
      expect(
        indexes.some((i) => i.name === 'idx_messages_chat_telegram_id'),
      ).toBe(true);
      upgradedDb.close();

      storeMessage({
        id: 'bot-test-migrate',
        chat_jid: 'tg:-1003000000001',
        sender: 'Agent',
        sender_name: 'Agent',
        content: 'hello',
        timestamp: '2026-01-01T00:00:05.000Z',
        is_from_me: true,
        is_bot_message: true,
        telegram_message_id: '9999',
      });
      const found = getBotMessageByTelegramId('tg:-1003000000001', '9999');
      expect(found).not.toBeNull();
      expect(found!.id).toBe('bot-test-migrate');
      expect(found!.telegram_message_id).toBe('9999');

      _closeDatabase();
    } finally {
      process.chdir(repoRoot);
    }
  });

  // #93/#130 — self-resuming cycles. Pre-existing scheduled_tasks
  // tables (every install before this change) lack the
  // continuation_cycle_id column. The migration must add it without
  // breaking any existing rows; ordinary tasks then read back as
  // continuation_cycle_id = NULL, which the scheduler normalises to
  // `undefined` so the spawned container gets no continuation env vars.
  it('adds continuation_cycle_id to a pre-existing scheduled_tasks table', async () => {
    const repoRoot = process.cwd();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-db-test-'));

    try {
      process.chdir(tempDir);
      fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });

      const dbPath = path.join(tempDir, 'store', 'messages.db');
      const legacyDb = new Database(dbPath);
      // Legacy shape: scheduled_tasks WITHOUT continuation_cycle_id.
      // Mirrors the install before #130 lands. The column is the
      // marker the scheduler reads to decide whether to plumb
      // NANOCLAW_CONTINUATION env vars onto the spawn.
      legacyDb.exec(`
        CREATE TABLE scheduled_tasks (
          id TEXT PRIMARY KEY,
          group_folder TEXT NOT NULL,
          chat_jid TEXT NOT NULL,
          prompt TEXT NOT NULL,
          schedule_type TEXT NOT NULL,
          schedule_value TEXT NOT NULL,
          next_run TEXT,
          last_run TEXT,
          last_result TEXT,
          status TEXT DEFAULT 'active',
          created_at TEXT NOT NULL,
          created_by_role TEXT NOT NULL DEFAULT 'owner'
        );
      `);
      legacyDb
        .prepare(
          `INSERT INTO scheduled_tasks (id, group_folder, chat_jid, prompt, schedule_type, schedule_value, status, created_at, created_by_role) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          'legacy-task',
          'main',
          'main@g.us',
          'pre-existing task',
          'once',
          '2026-04-01T00:00:00.000Z',
          'active',
          '2026-04-01T00:00:00.000Z',
          'owner',
        );
      legacyDb.close();

      vi.resetModules();
      const { initDatabase, getTaskById, _closeDatabase } =
        await import('./db.js');

      // Must not throw — every column-add migration must be PRAGMA-gated
      // so a re-run on an already-upgraded DB is a no-op (idempotent).
      initDatabase();

      const upgradedDb = new Database(dbPath);
      const cols = upgradedDb
        .prepare('PRAGMA table_info(scheduled_tasks)')
        .all() as Array<{ name: string }>;
      expect(cols.some((c) => c.name === 'continuation_cycle_id')).toBe(true);
      upgradedDb.close();

      // Pre-existing row reads back with continuation_cycle_id = NULL.
      // The scheduler's `?? undefined` normalisation depends on this —
      // a non-null backfill default would silently emit continuation
      // env vars on every legacy task on first run after upgrade.
      const legacyTask = getTaskById('legacy-task');
      expect(legacyTask).toBeDefined();
      expect(legacyTask!.continuation_cycle_id).toBeNull();

      _closeDatabase();
    } finally {
      process.chdir(repoRoot);
    }
  });
});
