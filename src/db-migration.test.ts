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
      // Restore CWD before removing the tempDir — `fs.rmSync(tempDir,
      // { recursive: true })` would refuse if the process was still
      // chdir'd inside the tree on some filesystems. Clean-up is in
      // `finally` so the artifact never lingers on CI workers across
      // runs (per `jbaruch/coding-policy: testing-standards` —
      // "Clean up after yourself").
      process.chdir(repoRoot);
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it('drops the dormant tg:1698969 / telegram_main row on initDatabase (#159)', async () => {
    const repoRoot = process.cwd();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-db-test-'));

    try {
      process.chdir(tempDir);
      fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });

      const dbPath = path.join(tempDir, 'store', 'messages.db');
      const legacyDb = new Database(dbPath);
      // Reproduce the dormant pair: real swarm row + dormant
      // telegram_main row keyed by tg:1698969. Spawner reads
      // available_groups.json (built from chats × registered_groups)
      // so the dormant row is invisible at runtime, and there was no
      // inverse of register_group until #159 to clean it up.
      legacyDb.exec(`
        CREATE TABLE registered_groups (
          jid TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          folder TEXT NOT NULL UNIQUE,
          trigger_pattern TEXT NOT NULL,
          added_at TEXT NOT NULL,
          container_config TEXT,
          requires_trigger INTEGER DEFAULT 1,
          is_main INTEGER DEFAULT 0
        );
      `);
      legacyDb
        .prepare(
          `INSERT INTO registered_groups (jid, name, folder, trigger_pattern, added_at, is_main) VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          'tg:1698969',
          'Telegram Main (dormant)',
          'telegram_main',
          '@Andy',
          '2024-01-01T00:00:00.000Z',
          1,
        );
      legacyDb
        .prepare(
          `INSERT INTO registered_groups (jid, name, folder, trigger_pattern, added_at, is_main) VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          'tg:-1009999999',
          'Telegram Swarm (active)',
          'telegram_swarm',
          '@Andy',
          '2024-02-01T00:00:00.000Z',
          1,
        );
      legacyDb.close();

      vi.resetModules();
      const { initDatabase, getRegisteredGroup, _closeDatabase } =
        await import('./db.js');

      initDatabase();

      // Dormant row removed.
      expect(getRegisteredGroup('tg:1698969')).toBeUndefined();
      // Active swarm row preserved — cleanup is anchored by jid AND
      // folder AND is_main, not a wildcard delete.
      expect(getRegisteredGroup('tg:-1009999999')).toBeDefined();

      _closeDatabase();
    } finally {
      // Restore CWD before removing tempDir — see the matching block
      // above. testing-standards `Clean up after yourself` rule.
      process.chdir(repoRoot);
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it('cleanup is idempotent — second initDatabase pass is a no-op (#159)', async () => {
    // Once the dormant row is gone, replaying initDatabase must not
    // throw or mutate any other row.
    const repoRoot = process.cwd();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-db-test-'));

    try {
      process.chdir(tempDir);
      fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });

      vi.resetModules();
      const {
        initDatabase,
        setRegisteredGroup,
        getRegisteredGroup,
        _closeDatabase,
      } = await import('./db.js');

      initDatabase();
      setRegisteredGroup('benign@g.us', {
        name: 'Benign',
        folder: 'benign-group',
        trigger: '@Andy',
        added_at: '2024-01-01T00:00:00.000Z',
      });
      _closeDatabase();

      // Second boot — same DB, no dormant row to remove.
      vi.resetModules();
      const {
        initDatabase: initAgain,
        getRegisteredGroup: getAgain,
        _closeDatabase: closeAgain,
      } = await import('./db.js');
      initAgain();

      expect(getAgain('benign@g.us')).toBeDefined();
      expect(getAgain('tg:1698969')).toBeUndefined();
      closeAgain();

      void getRegisteredGroup; // silence unused-import lint
    } finally {
      // Restore CWD before removing tempDir — see the matching block
      // above. testing-standards `Clean up after yourself` rule.
      process.chdir(repoRoot);
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  // Trigger pattern JSON schema (#81). Pre-existing registered_groups
  // rows store `trigger_pattern` as a literal string ("@Andy"). The
  // backfill migration converts every legacy row to the JSON config
  // shape on first boot so #82's self-improvement loop can record
  // per-pattern metrics. The migration must be idempotent — running
  // it twice on the same DB must leave the converted rows alone.
  it('backfills legacy string trigger_pattern rows to JSON config (#81)', async () => {
    const repoRoot = process.cwd();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-db-test-'));

    try {
      process.chdir(tempDir);
      fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });

      const dbPath = path.join(tempDir, 'store', 'messages.db');
      const legacyDb = new Database(dbPath);
      // Pre-#81 registered_groups schema: trigger_pattern is a literal
      // keyword string. Includes a row already in JSON shape to verify
      // idempotence (the migration must skip it).
      legacyDb.exec(`
        CREATE TABLE registered_groups (
          jid TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          folder TEXT NOT NULL UNIQUE,
          trigger_pattern TEXT NOT NULL,
          added_at TEXT NOT NULL,
          container_config TEXT,
          requires_trigger INTEGER DEFAULT 1,
          is_main INTEGER DEFAULT 0
        );
      `);
      legacyDb
        .prepare(
          `INSERT INTO registered_groups (jid, name, folder, trigger_pattern, added_at, requires_trigger)
           VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          'legacy-a@g.us',
          'Legacy A',
          'whatsapp_legacy_a',
          '@Andy',
          '2024-01-01T00:00:00.000Z',
          1,
        );
      legacyDb
        .prepare(
          `INSERT INTO registered_groups (jid, name, folder, trigger_pattern, added_at, requires_trigger)
           VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          'legacy-b@s.whatsapp.net',
          'Legacy B',
          'whatsapp_legacy_b',
          '@Bot',
          '2024-01-02T00:00:00.000Z',
          0,
        );
      // Already-migrated row: must NOT be touched by the backfill.
      const preMigratedJson = JSON.stringify({
        version: 1,
        patterns: [
          {
            pattern: '@Already',
            kind: 'keyword',
            source: 'learned',
            precision: 0.42,
            sample_count: 99,
            last_matched_at: '2024-06-01T00:00:00.000Z',
            last_updated_at: '2024-06-01T00:00:00.000Z',
          },
        ],
      });
      legacyDb
        .prepare(
          `INSERT INTO registered_groups (jid, name, folder, trigger_pattern, added_at, requires_trigger)
           VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          'already-migrated@g.us',
          'Already Migrated',
          'whatsapp_already',
          preMigratedJson,
          '2024-01-03T00:00:00.000Z',
          1,
        );
      legacyDb.close();

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();

      const upgradedDb = new Database(dbPath);
      const rows = upgradedDb
        .prepare(
          `SELECT jid, trigger_pattern FROM registered_groups ORDER BY jid`,
        )
        .all() as Array<{ jid: string; trigger_pattern: string }>;

      // Legacy A: `@Andy` matches the bare-mention shape
      // `/^@[a-zA-Z0-9_]+$/`, so it backfills as `kind: "mention"`
      // with the leading `@` stripped from the stored pattern. This
      // is the Fix-1 behaviour from #84's followup PR — legacy
      // string `@Andy` is semantically a mention, and pre-classifying
      // it that way means #82's `kind`-aware self-improvement loop
      // doesn't have to retroactively reclassify every existing
      // group's trigger. The mention matcher prepends `@` back at
      // match time, so the runtime regex for `@Andy` is unchanged.
      const a = rows.find((r) => r.jid === 'legacy-a@g.us')!;
      const aParsed = JSON.parse(a.trigger_pattern);
      expect(aParsed).toMatchObject({
        version: 1,
        patterns: [
          expect.objectContaining({
            pattern: 'Andy',
            kind: 'mention',
            source: 'owner-set',
            precision: 0,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          }),
        ],
      });

      // Legacy B: same shape — `@Bot` stripped and stored as
      // `mention`/`Bot`.
      const b = rows.find((r) => r.jid === 'legacy-b@s.whatsapp.net')!;
      const bParsed = JSON.parse(b.trigger_pattern);
      expect(bParsed.patterns[0].pattern).toBe('Bot');
      expect(bParsed.patterns[0].kind).toBe('mention');

      // Already-migrated row: byte-identical to its pre-init value.
      // A second pass must not clobber learned precision/sample_count.
      const already = rows.find((r) => r.jid === 'already-migrated@g.us')!;
      expect(already.trigger_pattern).toBe(preMigratedJson);

      upgradedDb.close();
      _closeDatabase();

      // Idempotence: re-init on the now-converted DB must no-op.
      vi.resetModules();
      const { initDatabase: reinit, _closeDatabase: reclose } =
        await import('./db.js');
      reinit();
      const reopenedDb = new Database(dbPath);
      const rowsAfter = reopenedDb
        .prepare(
          `SELECT jid, trigger_pattern FROM registered_groups ORDER BY jid`,
        )
        .all() as Array<{ jid: string; trigger_pattern: string }>;
      // Every row identical to the previous pass.
      for (const r of rowsAfter) {
        const before = rows.find((x) => x.jid === r.jid)!;
        expect(r.trigger_pattern).toBe(before.trigger_pattern);
      }
      reopenedDb.close();
      reclose();
    } finally {
      process.chdir(repoRoot);
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  // #84 followup PR (review comment-id 4360940433), Fix 1:
  // shape-classify legacy strings during backfill so #82's
  // kind-aware self-improvement loop sees the right category from
  // day one. `/^@[a-zA-Z0-9_]+$/` → `mention` (stored bare); anything
  // else → `keyword` (stored verbatim). Behaviour-equivalent at the
  // matcher layer today, but pinned-down for downstream reclassifiers.
  it('backfill shape-classifies legacy strings into mention vs keyword', async () => {
    const repoRoot = process.cwd();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-db-test-'));
    try {
      process.chdir(tempDir);
      fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });

      const dbPath = path.join(tempDir, 'store', 'messages.db');
      const legacyDb = new Database(dbPath);
      legacyDb.exec(`
        CREATE TABLE registered_groups (
          jid TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          folder TEXT NOT NULL UNIQUE,
          trigger_pattern TEXT NOT NULL,
          added_at TEXT NOT NULL,
          container_config TEXT,
          requires_trigger INTEGER DEFAULT 1,
          is_main INTEGER DEFAULT 0
        );
      `);
      // Cover the three classification branches:
      //   - clean `@<word>` → mention (stored bare)
      //   - bare keyword (no `@`) → keyword
      //   - `@` with a non-identifier char (dash, dot, space) →
      //     keyword (the regex requires \w+, dash falls out)
      const insert = legacyDb.prepare(
        `INSERT INTO registered_groups (jid, name, folder, trigger_pattern, added_at, requires_trigger)
         VALUES (?, ?, ?, ?, ?, ?)`,
      );
      insert.run(
        'mention-row@g.us',
        'Mention Row',
        'whatsapp_mentionrow',
        '@AyeAye',
        '2024-01-01T00:00:00.000Z',
        1,
      );
      insert.run(
        'keyword-row@g.us',
        'Keyword Row',
        'whatsapp_keywordrow',
        'nanoclaw',
        '2024-01-01T00:00:00.000Z',
        1,
      );
      insert.run(
        'dashed-mention@g.us',
        'Dashed Mention',
        'whatsapp_dashedmention',
        '@bot-with-dash',
        '2024-01-01T00:00:00.000Z',
        1,
      );
      legacyDb.close();

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();

      const upgradedDb = new Database(dbPath);
      const rows = upgradedDb
        .prepare(
          `SELECT jid, trigger_pattern FROM registered_groups ORDER BY jid`,
        )
        .all() as Array<{ jid: string; trigger_pattern: string }>;

      const mention = JSON.parse(
        rows.find((r) => r.jid === 'mention-row@g.us')!.trigger_pattern,
      );
      expect(mention.patterns[0]).toMatchObject({
        pattern: 'AyeAye',
        kind: 'mention',
      });

      const keyword = JSON.parse(
        rows.find((r) => r.jid === 'keyword-row@g.us')!.trigger_pattern,
      );
      expect(keyword.patterns[0]).toMatchObject({
        pattern: 'nanoclaw',
        kind: 'keyword',
      });

      // `@bot-with-dash` doesn't match `/^@[a-zA-Z0-9_]+$/` because
      // `-` is not a word char — falls through to `keyword`, stored
      // verbatim with the leading `@`. Without this branch we'd
      // mis-strip the `@` and mention-match `bot-with-dash` against
      // `@bot-with-dash` in inbound text, which would still hit by
      // coincidence today; the point is to keep the migration
      // unambiguous so #82's reclassifier doesn't have to second-
      // guess the original intent.
      const dashed = JSON.parse(
        rows.find((r) => r.jid === 'dashed-mention@g.us')!.trigger_pattern,
      );
      expect(dashed.patterns[0]).toMatchObject({
        pattern: '@bot-with-dash',
        kind: 'keyword',
      });

      upgradedDb.close();
      _closeDatabase();
    } finally {
      process.chdir(repoRoot);
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  // The backfill migration must run on a brand-new DB without any
  // existing rows — i.e. the first-time install code path. No rows
  // means no work, so this is essentially a "doesn't crash and the
  // schema is correct" check.
  it('backfill is a no-op on a fresh DB with no registered_groups rows', async () => {
    const repoRoot = process.cwd();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-db-test-'));
    try {
      process.chdir(tempDir);
      fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      const dbPath = path.join(tempDir, 'store', 'messages.db');
      const inspect = new Database(dbPath);
      const count = inspect
        .prepare(`SELECT COUNT(*) as c FROM registered_groups`)
        .get() as { c: number };
      expect(count.c).toBe(0);
      inspect.close();
      _closeDatabase();
    } finally {
      process.chdir(repoRoot);
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });
});
