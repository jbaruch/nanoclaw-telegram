import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

import { ASSISTANT_NAME, DATA_DIR, GROUPS_DIR, STORE_DIR } from './config.js';
import { isValidGroupFolder } from './group-folder.js';
import { logger } from './logger.js';
import { STATE_MIGRATIONS } from './state-migrations/index.js';
import {
  ContainerConfig,
  NewMessage,
  RegisteredGroup,
  ScheduledTask,
  TaskRunLog,
} from './types.js';

let db: Database.Database;

/**
 * One versioned state-table migration (epic #293). Tracked via SQLite's
 * built-in `PRAGMA user_version`. See `src/state-migrations/README.md`
 * for the convention; the registry lives in
 * `src/state-migrations/index.ts`.
 */
export interface StateMigration {
  version: number;
  name: string;
  sql: string;
}

/**
 * Apply registered state-table migrations to a database, in order.
 *
 * Reads `PRAGMA user_version` (SQLite's app-defined schema-version
 * counter), applies every registered migration whose `version` is
 * greater than the current value, and bumps `user_version` to the
 * applied migration's number inside the same transaction as that
 * migration's DDL/DML. This guarantees each individual migration is
 * atomic: the database cannot report "version N applied" with only
 * part of migration N physically present. If multiple pending
 * migrations exist, earlier migrations remain applied if a later
 * migration fails — the next startup re-runs only the pending tail
 * (the version gate skips already-applied entries).
 *
 * Throws if `user_version` is HIGHER than the highest version this
 * build knows about. That state means the database was migrated by a
 * newer container and the operator has rolled back to an older one;
 * silently running against the future schema would corrupt state.
 *
 * The `migrations` parameter is injectable so tests can drive the
 * loader with a fixture array instead of the production registry.
 */
export function applyStateMigrations(
  database: Database.Database,
  migrations: readonly StateMigration[],
): void {
  validateMigrationRegistry(migrations);

  const currentVersion = Number(
    database.pragma('user_version', { simple: true }),
  );
  const highestKnown =
    migrations.length > 0 ? migrations[migrations.length - 1].version : 0;

  if (currentVersion > highestKnown) {
    throw new Error(
      `Database state schema is at user_version=${currentVersion}, but ` +
        `this build only knows migrations up to version=${highestKnown}. ` +
        `Refusing to start: this state means a newer container migrated ` +
        `the database and the operator has rolled back to an older one. ` +
        `Either run a build that includes migration ${currentVersion}, or ` +
        `restore the database from before the upgrade.`,
    );
  }

  for (const migration of migrations) {
    if (migration.version <= currentVersion) continue;
    // Wrap the DDL/DML and the user_version bump in a single
    // transaction so a SQL error rolls both back together — the
    // database can never end up reporting "version N applied" with
    // only half of N's changes physically present.
    const apply = database.transaction(() => {
      database.exec(migration.sql);
      database.pragma(`user_version = ${migration.version}`);
    });
    apply();
    logger.info(
      { version: migration.version, name: migration.name },
      'state-migration: applied',
    );
  }
}

function validateMigrationRegistry(
  migrations: readonly StateMigration[],
): void {
  for (let i = 0; i < migrations.length; i++) {
    const m = migrations[i];
    if (!Number.isInteger(m.version) || m.version <= 0) {
      throw new Error(
        `state-migrations[${i}]: version must be a positive integer, got ${m.version}`,
      );
    }
    const expectedVersion = i + 1;
    if (m.version !== expectedVersion) {
      throw new Error(
        `state-migrations[${i}]: expected version=${expectedVersion} ` +
          `(contiguous from 1), got version=${m.version}. ` +
          `Migrations must be contiguous and sorted ascending — see ` +
          `src/state-migrations/README.md.`,
      );
    }
    if (typeof m.name !== 'string' || m.name.trim().length === 0) {
      throw new Error(
        `state-migrations[${i}]: name must be a non-empty string`,
      );
    }
    if (typeof m.sql !== 'string' || m.sql.trim().length === 0) {
      throw new Error(`state-migrations[${i}]: sql must be a non-empty string`);
    }
  }
}

function createSchema(database: Database.Database): void {
  database.exec(`
    CREATE TABLE IF NOT EXISTS chats (
      jid TEXT PRIMARY KEY,
      name TEXT,
      last_message_time TEXT,
      channel TEXT,
      is_group INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS messages (
      id TEXT,
      chat_jid TEXT,
      sender TEXT,
      sender_name TEXT,
      content TEXT,
      timestamp TEXT,
      is_from_me INTEGER,
      is_bot_message INTEGER DEFAULT 0,
      telegram_message_id TEXT,
      PRIMARY KEY (id, chat_jid),
      FOREIGN KEY (chat_jid) REFERENCES chats(jid)
    );
    CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp);
    -- Composite index for chat_status latest-is_from_me-1 lookup
    -- (see getLastFromMeMessages). Without it, the predicate
    -- chat_jid = ? AND is_from_me = 1 falls back to a full scan +
    -- sort by timestamp on every chat in the snapshot, scaling
    -- poorly with message history. Trailing timestamp column lets
    -- SQLite satisfy ORDER BY directly from the index.
    CREATE INDEX IF NOT EXISTS idx_messages_fromme_chat
      ON messages(chat_jid, is_from_me, timestamp);

    CREATE TABLE IF NOT EXISTS scheduled_tasks (
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
      -- Provenance of this task's creation. Drives whether the agent-runner
      -- wraps the prompt in <untrusted-input> at fire time:
      --   'owner'           — host code / Baruch's direct tooling (trusted)
      --   'main_agent'      — main group's agent (trusted)
      --   'trusted_agent'   — trusted non-main group's agent (trusted)
      --   'untrusted_agent' — untrusted group's agent (NOT trusted, wrap applies)
      -- Without this, an untrusted agent could self-schedule a prompt that
      -- later fires unwrapped and bypasses the trust boundary.
      created_by_role TEXT NOT NULL DEFAULT 'owner',
      -- Continuation marker for self-resuming cycles (#93/#130). NULL for
      -- ordinary one-shot scheduled tasks. When set by the resumable-cycle
      -- helper skill, the task-scheduler plumbs the value into the spawned
      -- container as NANOCLAW_CONTINUATION=1 +
      -- NANOCLAW_CONTINUATION_CYCLE_ID=<value>. Absence of the env vars is
      -- itself the "fresh invocation" signal the calling skill checks for;
      -- mismatch between the prompt prefix and these env vars fails closed
      -- to fresh, never silently takes the lock-skip branch.
      continuation_cycle_id TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_next_run ON scheduled_tasks(next_run);
    CREATE INDEX IF NOT EXISTS idx_status ON scheduled_tasks(status);

    CREATE TABLE IF NOT EXISTS task_run_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      task_id TEXT NOT NULL,
      run_at TEXT NOT NULL,
      duration_ms INTEGER NOT NULL,
      status TEXT NOT NULL,
      result TEXT,
      error TEXT,
      FOREIGN KEY (task_id) REFERENCES scheduled_tasks(id)
    );
    CREATE INDEX IF NOT EXISTS idx_task_run_logs ON task_run_logs(task_id, run_at);

    CREATE TABLE IF NOT EXISTS reactions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      message_id TEXT NOT NULL,
      message_chat_jid TEXT NOT NULL,
      reactor_jid TEXT NOT NULL,
      reactor_name TEXT NOT NULL,
      emoji TEXT NOT NULL,
      timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_reactions_message ON reactions(message_id, message_chat_jid);

    CREATE TABLE IF NOT EXISTS router_state (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
      group_folder TEXT NOT NULL,
      session_name TEXT NOT NULL DEFAULT 'default',
      session_id TEXT NOT NULL,
      PRIMARY KEY (group_folder, session_name)
    );
    CREATE TABLE IF NOT EXISTS registered_groups (
      jid TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      folder TEXT NOT NULL UNIQUE,
      trigger_pattern TEXT NOT NULL,
      added_at TEXT NOT NULL,
      container_config TEXT,
      requires_trigger INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS smart_home_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      device_id TEXT NOT NULL,
      device_name TEXT NOT NULL,
      attribute_name TEXT NOT NULL,
      value TEXT NOT NULL,
      unit TEXT,
      description TEXT,
      source TEXT DEFAULT 'DEVICE',
      timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_she_timestamp ON smart_home_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_she_device_time ON smart_home_events(device_id, timestamp);
  `);

  // Add context_mode column if it doesn't exist (migration for existing DBs)
  try {
    database.exec(
      `ALTER TABLE scheduled_tasks ADD COLUMN context_mode TEXT DEFAULT 'isolated'`,
    );
  } catch {
    /* column already exists */
  }

  // Add script column if it doesn't exist (migration for existing DBs)
  try {
    database.exec(`ALTER TABLE scheduled_tasks ADD COLUMN script TEXT`);
  } catch {
    /* column already exists */
  }

  // Add schedule_timezone column for #102 — IANA tz used to evaluate
  // cron expressions. NULL means "use TIMEZONE config at fire time"
  // (pre-#102 behavior). Using PRAGMA-check rather than try/catch to
  // match the no-error-suppression rule already applied to
  // created_by_role below.
  const schedTzCols = database
    .prepare('PRAGMA table_info(scheduled_tasks)')
    .all() as Array<{ name: string }>;
  if (!schedTzCols.some((c) => c.name === 'schedule_timezone')) {
    database.exec(
      `ALTER TABLE scheduled_tasks ADD COLUMN schedule_timezone TEXT`,
    );
  }

  // Add created_by_role column (scheduled-task provenance). Existing rows
  // backfill to 'owner' — all pre-migration tasks were either
  // host-auto-registered (src/index.ts heartbeat seeders) or created via
  // Baruch's direct tooling, and both of those should unwrap in the
  // agent-runner. Using PRAGMA check instead of try/catch idiom so the
  // migration failure mode is visible if it ever matters (the existing
  // try/catch pattern on this table predates the no-error-suppression
  // rule and shouldn't spread).
  const scheduledCols = database
    .prepare('PRAGMA table_info(scheduled_tasks)')
    .all() as Array<{ name: string }>;
  if (!scheduledCols.some((c) => c.name === 'created_by_role')) {
    database.exec(
      `ALTER TABLE scheduled_tasks ADD COLUMN created_by_role TEXT NOT NULL DEFAULT 'owner'`,
    );
  }

  // Add continuation_cycle_id column for #93/#130 — self-resuming cycles.
  // NULL for ordinary tasks; set when the resumable-cycle helper skill
  // schedules the next link of a chain. The task-scheduler reads this
  // value at fire time and plumbs it onto the spawned container as
  // NANOCLAW_CONTINUATION=1 + NANOCLAW_CONTINUATION_CYCLE_ID=<value>.
  // PRAGMA-gated rather than try/catch per the no-error-suppression
  // rule (see schedule_timezone migration above).
  const continuationCols = database
    .prepare('PRAGMA table_info(scheduled_tasks)')
    .all() as Array<{ name: string }>;
  if (!continuationCols.some((c) => c.name === 'continuation_cycle_id')) {
    database.exec(
      `ALTER TABLE scheduled_tasks ADD COLUMN continuation_cycle_id TEXT`,
    );
  }

  // Add is_bot_message column if it doesn't exist (migration for existing DBs)
  try {
    database.exec(
      `ALTER TABLE messages ADD COLUMN is_bot_message INTEGER DEFAULT 0`,
    );
    // Backfill: mark existing bot messages that used the content prefix pattern
    database
      .prepare(`UPDATE messages SET is_bot_message = 1 WHERE content LIKE ?`)
      .run(`${ASSISTANT_NAME}:%`);
  } catch {
    /* column already exists */
  }

  // Add is_main column if it doesn't exist (migration for existing DBs)
  try {
    database.exec(
      `ALTER TABLE registered_groups ADD COLUMN is_main INTEGER DEFAULT 0`,
    );
    // Backfill: existing rows with folder = 'main' are the main group
    database.exec(
      `UPDATE registered_groups SET is_main = 1 WHERE folder = 'main'`,
    );
  } catch {
    /* column already exists */
  }

  // Add channel and is_group columns if they don't exist (migration for existing DBs)
  try {
    database.exec(`ALTER TABLE chats ADD COLUMN channel TEXT`);
    database.exec(`ALTER TABLE chats ADD COLUMN is_group INTEGER DEFAULT 0`);
    // Backfill from JID patterns
    database.exec(
      `UPDATE chats SET channel = 'whatsapp', is_group = 1 WHERE jid LIKE '%@g.us'`,
    );
    database.exec(
      `UPDATE chats SET channel = 'whatsapp', is_group = 0 WHERE jid LIKE '%@s.whatsapp.net'`,
    );
    database.exec(
      `UPDATE chats SET channel = 'discord', is_group = 1 WHERE jid LIKE 'dc:%'`,
    );
    database.exec(
      `UPDATE chats SET channel = 'telegram', is_group = 0 WHERE jid LIKE 'tg:%'`,
    );
  } catch {
    /* columns already exist */
  }

  // Add reply context columns if they don't exist (migration for existing DBs)
  try {
    database.exec(`ALTER TABLE messages ADD COLUMN reply_to_message_id TEXT`);
    database.exec(
      `ALTER TABLE messages ADD COLUMN reply_to_message_content TEXT`,
    );
    database.exec(`ALTER TABLE messages ADD COLUMN reply_to_sender_name TEXT`);
  } catch {
    /* columns already exist */
  }

  // `telegram_message_id` migration (PRAGMA-gated, no silent catch).
  // For bot-sent messages the `id` column holds our synthetic
  // `bot-<ts>-<rand>`, so the platform's numeric message ID is nowhere
  // queryable without this column — the symptom that motivated adding
  // it: a Telegram message appeared in a group that nobody could
  // attribute to a specific bot send, because the DB only had the
  // synthetic IDs. An ALTER-in-try-catch was deliberately avoided
  // here (the rest of this file does it, pre-existing) so a real
  // schema-alteration error surfaces instead of being swallowed.
  const messagesCols = database
    .prepare('PRAGMA table_info(messages)')
    .all() as Array<{ name: string }>;
  if (!messagesCols.some((c) => c.name === 'telegram_message_id')) {
    database.exec(`ALTER TABLE messages ADD COLUMN telegram_message_id TEXT`);
  }

  // Diagnostic lookup index: "which DB row produced Telegram message X?".
  // Created AFTER the ALTER above so it works on existing DBs that
  // didn't have the column yet — creating the index in the main CREATE
  // TABLE block would throw "no such column: telegram_message_id" on
  // upgrade and block startup.
  database.exec(
    `CREATE INDEX IF NOT EXISTS idx_messages_chat_telegram_id
       ON messages(chat_jid, telegram_message_id)`,
  );

  // Migrate sessions table to per-session layout (parallel-maintenance).
  // Pre-PR-#55: PK was `(group_folder)` alone — one session per group.
  // Post-PR-#55: PK is `(group_folder, session_name)` so each session
  // (`default`, `maintenance`) maintains its own SDK session chain.
  //
  // `CREATE TABLE IF NOT EXISTS` above already defines the new shape for
  // fresh installs. For existing DBs on the pre-migration shape we detect
  // the missing `session_name` column and recreate the table, tagging all
  // existing rows as `default` (they came from the user-facing container).
  const sessionsCols = database
    .prepare('PRAGMA table_info(sessions)')
    .all() as Array<{ name: string }>;
  const hasSessionName = sessionsCols.some((c) => c.name === 'session_name');
  if (sessionsCols.length > 0 && !hasSessionName) {
    // Wrap in a transaction: the CREATE/INSERT/DROP/RENAME sequence must be
    // atomic. A crash between `DROP TABLE sessions` and
    // `ALTER TABLE sessions_new RENAME TO sessions` would leave the DB
    // without a `sessions` table at all — next startup would find it missing
    // and blow up on any session lookup. `database.transaction()` in
    // better-sqlite3 implicitly rolls back on thrown exceptions.
    database.transaction(() => {
      database.exec(`
        CREATE TABLE sessions_new (
          group_folder TEXT NOT NULL,
          session_name TEXT NOT NULL DEFAULT 'default',
          session_id TEXT NOT NULL,
          PRIMARY KEY (group_folder, session_name)
        );
        INSERT INTO sessions_new (group_folder, session_name, session_id)
          SELECT group_folder, 'default', session_id FROM sessions;
        DROP TABLE sessions;
        ALTER TABLE sessions_new RENAME TO sessions;
      `);
    })();
  }

  // One-shot cleanup (#159): drop the dormant `tg:1698969` /
  // `telegram_main` row. Predates `telegram_swarm` and never appeared in
  // any container's `available_groups.json` — the spawner ignores it
  // because the JSON is authoritative — but it lingered in
  // `registered_groups` because there was no inverse of `register_group`
  // until this issue. Anchored by `(jid, folder, is_main)` so it cannot
  // ever match a current operator-managed row.
  database
    .prepare(
      `DELETE FROM registered_groups
         WHERE jid = 'tg:1698969'
           AND folder = 'telegram_main'
           AND is_main = 1`,
    )
    .run();
}

export function initDatabase(): void {
  const dbPath = path.join(STORE_DIR, 'messages.db');
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });

  db = new Database(dbPath);
  // WAL keeps cross-process readers from seeing partially-written pages
  // (the bot writes from the orchestrator while agent containers and
  // ad-hoc sqlite3 readers query the same file). Without WAL, readers
  // hitting a mid-write rollback journal occasionally surface a false
  // "database disk image is malformed" error. NORMAL is the standard
  // synchronous pairing for WAL; busy_timeout smooths the rare contention.
  // `journal_mode = WAL` can silently fall back to the previous mode when
  // the filesystem can't host WAL (rare network FS, some FUSE mounts) —
  // verify the effective mode so the malformed-image fix can't be a no-op
  // we don't notice.
  const journalMode = String(
    db.pragma('journal_mode = WAL', { simple: true }),
  ).toLowerCase();
  if (journalMode !== 'wal') {
    throw new Error(
      `SQLite WAL mode is required at ${dbPath} but the database reports journal_mode="${journalMode}". ` +
        `Check the underlying filesystem — WAL needs shared-memory mmap support.`,
    );
  }
  db.pragma('synchronous = NORMAL');
  db.pragma('busy_timeout = 5000');
  createSchema(db);

  // Apply versioned state-table migrations (epic #293). Runs AFTER
  // createSchema so the baseline tables exist, and BEFORE
  // migrateJsonState so any JSON-data backfill targets a table that
  // a registered migration has already created.
  applyStateMigrations(db, STATE_MIGRATIONS);

  // Migrate from JSON files if they exist
  migrateJsonState();
}

/** @internal - for tests only. Creates a fresh in-memory database. */
export function _initTestDatabase(): void {
  db = new Database(':memory:');
  createSchema(db);
  applyStateMigrations(db, STATE_MIGRATIONS);
}

/** @internal - for tests only. */
export function _closeDatabase(): void {
  db.close();
}

/**
 * @internal - for tests only.
 *
 * Writes a `registered_groups` row whose `container_config` column is a raw
 * string the caller controls. Lets tests reproduce the malformed-JSON
 * condition that the issue-156 fix guards against, without exporting the
 * module-private `db` handle.
 */
export function _writeRawRegisteredGroup(args: {
  jid: string;
  name: string;
  folder: string;
  trigger: string;
  added_at: string;
  container_config: string | null;
  requires_trigger?: number | null;
  is_main?: number | null;
}): void {
  db.prepare(
    `INSERT OR REPLACE INTO registered_groups (jid, name, folder, trigger_pattern, added_at, container_config, requires_trigger, is_main)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    args.jid,
    args.name,
    args.folder,
    args.trigger,
    args.added_at,
    args.container_config,
    args.requires_trigger ?? null,
    args.is_main ?? 0,
  );
}

/**
 * Store chat metadata only (no message content).
 * Used for all chats to enable group discovery without storing sensitive content.
 */
export function storeChatMetadata(
  chatJid: string,
  timestamp: string,
  name?: string,
  channel?: string,
  isGroup?: boolean,
): void {
  const ch = channel ?? null;
  const group = isGroup === undefined ? null : isGroup ? 1 : 0;

  if (name) {
    // Update with name, preserving existing timestamp if newer
    db.prepare(
      `
      INSERT INTO chats (jid, name, last_message_time, channel, is_group) VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(jid) DO UPDATE SET
        name = excluded.name,
        last_message_time = MAX(last_message_time, excluded.last_message_time),
        channel = COALESCE(excluded.channel, channel),
        is_group = COALESCE(excluded.is_group, is_group)
    `,
    ).run(chatJid, name, timestamp, ch, group);
  } else {
    // Update timestamp only, preserve existing name if any
    db.prepare(
      `
      INSERT INTO chats (jid, name, last_message_time, channel, is_group) VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(jid) DO UPDATE SET
        last_message_time = MAX(last_message_time, excluded.last_message_time),
        channel = COALESCE(excluded.channel, channel),
        is_group = COALESCE(excluded.is_group, is_group)
    `,
    ).run(chatJid, chatJid, timestamp, ch, group);
  }
}

/**
 * Update chat name without changing timestamp for existing chats.
 * New chats get the current time as their initial timestamp.
 * Used during group metadata sync.
 */
export function updateChatName(chatJid: string, name: string): void {
  db.prepare(
    `
    INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)
    ON CONFLICT(jid) DO UPDATE SET name = excluded.name
  `,
  ).run(chatJid, name, new Date().toISOString());
}

export interface ChatInfo {
  jid: string;
  name: string;
  last_message_time: string;
  channel: string;
  is_group: number;
}

/**
 * Get all known chats, ordered by most recent activity.
 */
export function getAllChats(): ChatInfo[] {
  return db
    .prepare(
      `
    SELECT jid, name, last_message_time, channel, is_group
    FROM chats
    ORDER BY last_message_time DESC
  `,
    )
    .all() as ChatInfo[];
}

/**
 * Look up a single chat row by JID. Returns null when no row exists
 * (the channel layer hasn't seen any inbound from that chat yet).
 *
 * Used by the orchestrator's react-first gate (#289) to distinguish
 * 1:1 DMs (`is_group=0`) from group chats (`is_group=1`) without
 * leaning on `requires_trigger` as a proxy — those two flags governed
 * different concerns and the conflation was the original bug.
 */
export function getChatByJid(jid: string): ChatInfo | null {
  const row = db
    .prepare(
      `SELECT jid, name, last_message_time, channel, is_group FROM chats WHERE jid = ?`,
    )
    .get(jid) as ChatInfo | undefined;
  return row ?? null;
}

/**
 * Get timestamp of last group metadata sync.
 */
export function getLastGroupSync(): string | null {
  // Store sync time in a special chat entry
  const row = db
    .prepare(`SELECT last_message_time FROM chats WHERE jid = '__group_sync__'`)
    .get() as { last_message_time: string } | undefined;
  return row?.last_message_time || null;
}

/**
 * Record that group metadata was synced.
 */
export function setLastGroupSync(): void {
  const now = new Date().toISOString();
  db.prepare(
    `INSERT OR REPLACE INTO chats (jid, name, last_message_time) VALUES ('__group_sync__', '__group_sync__', ?)`,
  ).run(now);
}

/**
 * Store a message with full content.
 * Only call this for registered groups where message history is needed.
 */
export function storeMessage(msg: NewMessage): void {
  db.prepare(
    `INSERT OR REPLACE INTO messages (id, chat_jid, sender, sender_name, content, timestamp, is_from_me, is_bot_message, reply_to_message_id, reply_to_message_content, reply_to_sender_name, telegram_message_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    msg.id,
    msg.chat_jid,
    msg.sender,
    msg.sender_name,
    msg.content,
    msg.timestamp,
    msg.is_from_me ? 1 : 0,
    msg.is_bot_message ? 1 : 0,
    msg.reply_to_message_id ?? null,
    msg.reply_to_message_content ?? null,
    msg.reply_to_sender_name ?? null,
    msg.telegram_message_id ?? null,
  );
}

/**
 * Store a message directly.
 */
export function storeMessageDirect(msg: {
  id: string;
  chat_jid: string;
  sender: string;
  sender_name: string;
  content: string;
  timestamp: string;
  is_from_me: boolean;
  is_bot_message?: boolean;
}): void {
  db.prepare(
    `INSERT OR REPLACE INTO messages (id, chat_jid, sender, sender_name, content, timestamp, is_from_me, is_bot_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    msg.id,
    msg.chat_jid,
    msg.sender,
    msg.sender_name,
    msg.content,
    msg.timestamp,
    msg.is_from_me ? 1 : 0,
    msg.is_bot_message ? 1 : 0,
  );
}

/**
 * Look up a single message by its platform message_id and chat_jid.
 * Returns null if not found.
 */
export function getMessageById(
  messageId: string,
  chatJid: string,
): NewMessage | null {
  const row = db
    .prepare(
      `SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me
       FROM messages
       WHERE id = ? AND chat_jid = ?`,
    )
    .get(messageId, chatJid) as
    | {
        id: string;
        chat_jid: string;
        sender: string;
        sender_name: string;
        content: string;
        timestamp: string;
        is_from_me: number;
      }
    | undefined;
  if (!row) return null;
  return {
    id: row.id,
    chat_jid: row.chat_jid,
    sender: row.sender,
    sender_name: row.sender_name,
    content: row.content,
    timestamp: row.timestamp,
    is_from_me: row.is_from_me === 1,
  };
}

/**
 * Cross-chat reply_to safety check (one half of the call-site
 * predicate; see `safeReplyToForChat` in `src/channels/telegram.ts`).
 *
 * Returns true when the message id is stored under at least one
 * chat_jid that is NOT the expected one — i.e. "we have evidence the
 * id belongs to some other chat". This is intentionally NOT a
 * sufficient signal on its own to drop `reply_parameters`: Telegram
 * message IDs are per-chat sequential, so the same numeric id
 * routinely exists in many chats, including the target. The call
 * site MUST first check `getMessageById(id, jid)` for positive
 * evidence the id is local; only if that returns null does this
 * helper's "exists in another chat" answer flip the safety verdict
 * to "drop". Inverting that order is how you accidentally strip
 * reply threading from every legitimate same-chat reply once a
 * deployment has more than one Telegram chat.
 *
 * Why not collapse this into `getMessageById`: that helper takes
 * both id AND chat and returns a single row. We want a single SQL
 * pass with the inverse predicate ("exists in some chat OTHER than
 * X") so the chat_jid != comparison stays inside the query (where
 * the index helps) and the call site reads as two clean predicates.
 */
export function messageExistsInDifferentChat(
  messageId: string,
  expectedChatJid: string,
): boolean {
  const row = db
    .prepare(
      `SELECT 1 AS hit
       FROM messages
       WHERE id = ? AND chat_jid != ?
       LIMIT 1`,
    )
    .get(messageId, expectedChatJid) as { hit: number } | undefined;
  return !!row;
}

/**
 * Look up a bot-sent message by the Telegram-native message ID
 * returned when it was posted. Exists so "what did we post at
 * Telegram ID X in chat Y" stops being a logs-grep exercise — the
 * synthetic `bot-<ts>-<rand>` `id` column gives no way to work
 * back from the Telegram ID otherwise. Narrowly scoped to bot
 * sends on Telegram (other channels either use the platform ID
 * as `id` directly or don't populate this column).
 */
export function getBotMessageByTelegramId(
  chatJid: string,
  telegramMessageId: string,
): NewMessage | null {
  const row = db
    .prepare(
      `SELECT id, chat_jid, sender, sender_name, content, timestamp,
              is_from_me, is_bot_message, reply_to_message_id,
              reply_to_message_content, reply_to_sender_name,
              telegram_message_id
         FROM messages
        WHERE chat_jid = ? AND telegram_message_id = ?
          AND is_bot_message = 1`,
    )
    .get(chatJid, telegramMessageId) as
    | {
        id: string;
        chat_jid: string;
        sender: string;
        sender_name: string;
        content: string;
        timestamp: string;
        is_from_me: number;
        is_bot_message: number;
        reply_to_message_id: string | null;
        reply_to_message_content: string | null;
        reply_to_sender_name: string | null;
        telegram_message_id: string | null;
      }
    | undefined;
  if (!row) return null;
  // Surface NULLs as `null` to match the other message getters
  // (`getMessagesSince`, `getNewMessages`) — existing tests assert
  // `.toBeNull()` on those paths. Using `?? undefined` here would
  // force every caller to handle both shapes.
  return {
    id: row.id,
    chat_jid: row.chat_jid,
    sender: row.sender,
    sender_name: row.sender_name,
    content: row.content,
    timestamp: row.timestamp,
    is_from_me: row.is_from_me === 1,
    is_bot_message: row.is_bot_message === 1,
    reply_to_message_id: row.reply_to_message_id,
    reply_to_message_content: row.reply_to_message_content,
    reply_to_sender_name: row.reply_to_sender_name,
    telegram_message_id: row.telegram_message_id,
  } as NewMessage;
}

export function storeReaction(reaction: {
  message_id: string;
  message_chat_jid: string;
  reactor_jid: string;
  reactor_name: string;
  emoji: string;
  timestamp: string;
}): void {
  db.prepare(
    `INSERT INTO reactions (message_id, message_chat_jid, reactor_jid, reactor_name, emoji, timestamp)
     VALUES (?, ?, ?, ?, ?, ?)`,
  ).run(
    reaction.message_id,
    reaction.message_chat_jid,
    reaction.reactor_jid,
    reaction.reactor_name,
    reaction.emoji,
    reaction.timestamp,
  );
}

export function getReactionsForMessage(
  messageId: string,
  chatJid: string,
): Array<{ reactor_name: string; emoji: string; timestamp: string }> {
  return db
    .prepare(
      `SELECT reactor_name, emoji, timestamp FROM reactions
       WHERE message_id = ? AND message_chat_jid = ?
       ORDER BY timestamp`,
    )
    .all(messageId, chatJid) as Array<{
    reactor_name: string;
    emoji: string;
    timestamp: string;
  }>;
}

export function getLatestMessage(
  chatJid: string,
): { id: string; chat_jid: string } | null {
  const row = db
    .prepare(
      `SELECT id, chat_jid FROM messages WHERE chat_jid = ? ORDER BY timestamp DESC LIMIT 1`,
    )
    .get(chatJid) as { id: string; chat_jid: string } | undefined;
  return row || null;
}

export function getNewMessages(
  jids: string[],
  lastTimestamp: string,
  botPrefix: string,
  limit: number = 200,
): { messages: NewMessage[]; newTimestamp: string } {
  if (jids.length === 0) return { messages: [], newTimestamp: lastTimestamp };

  const placeholders = jids.map(() => '?').join(',');
  // Filter bot messages using both the is_bot_message flag AND the content
  // prefix as a backstop for messages written before the migration ran.
  // Subquery takes the N most recent, outer query re-sorts chronologically.
  const sql = `
    SELECT * FROM (
      SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me,
             reply_to_message_id, reply_to_message_content, reply_to_sender_name
      FROM messages
      WHERE timestamp > ? AND chat_jid IN (${placeholders})
        AND is_bot_message = 0 AND content NOT LIKE ?
        AND content != '' AND content IS NOT NULL
      ORDER BY timestamp DESC
      LIMIT ?
    ) ORDER BY timestamp
  `;

  const rows = db
    .prepare(sql)
    .all(lastTimestamp, ...jids, `${botPrefix}:%`, limit) as NewMessage[];

  let newTimestamp = lastTimestamp;
  for (const row of rows) {
    if (row.timestamp > newTimestamp) newTimestamp = row.timestamp;
  }

  return { messages: rows, newTimestamp };
}

export function getMessagesSince(
  chatJid: string,
  sinceTimestamp: string,
  botPrefix: string,
  limit: number = 200,
): NewMessage[] {
  // Filter bot messages using both the is_bot_message flag AND the content
  // prefix as a backstop for messages written before the migration ran.
  // Subquery takes the N most recent, outer query re-sorts chronologically.
  const sql = `
    SELECT * FROM (
      SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me,
             reply_to_message_id, reply_to_message_content, reply_to_sender_name
      FROM messages
      WHERE chat_jid = ? AND timestamp > ?
        AND is_bot_message = 0 AND content NOT LIKE ?
        AND content != '' AND content IS NOT NULL
      ORDER BY timestamp DESC
      LIMIT ?
    ) ORDER BY timestamp
  `;
  return db
    .prepare(sql)
    .all(chatJid, sinceTimestamp, `${botPrefix}:%`, limit) as NewMessage[];
}

export function getLastBotMessageTimestamp(
  chatJid: string,
  botPrefix: string,
): string | undefined {
  const row = db
    .prepare(
      `SELECT MAX(timestamp) as ts FROM messages
       WHERE chat_jid = ? AND (is_bot_message = 1 OR content LIKE ?)`,
    )
    .get(chatJid, `${botPrefix}:%`) as { ts: string | null } | undefined;
  return row?.ts ?? undefined;
}

/**
 * Latest outbound message in a chat (where the host wrote the row with
 * `is_from_me = 1`, i.e. AyeAye sent it). Returned as `{ timestamp,
 * content }` or `null` if AyeAye never spoke in this chat. Used by the
 * `chat_status` IPC handler so the admin tile can answer "when did
 * AyeAye last respond here, and with what?" for diagnosing silent
 * containers. Single-chat convenience wrapper around the batch helper.
 */
export function getLastFromMeMessage(
  chatJid: string,
): { timestamp: string; content: string } | null {
  return getLastFromMeMessages([chatJid]).get(chatJid) ?? null;
}

/**
 * Batch variant. Resolves "latest is_from_me=1 message per chat" for
 * many JIDs in one SQL round-trip. Backs the all-chats path of the
 * `chat_status` IPC handler — calling the single-chat helper N times
 * was N statement compilations and N scan+sorts; this issues a single
 * grouped query against the `idx_messages_fromme_chat` composite index
 * (created in createSchema). Chats that AyeAye has never spoken in are
 * absent from the returned map, matching the single-chat helper's
 * `null` return.
 */
export function getLastFromMeMessages(
  chatJids: readonly string[],
): Map<string, { timestamp: string; content: string }> {
  const out = new Map<string, { timestamp: string; content: string }>();
  if (chatJids.length === 0) return out;
  const placeholders = chatJids.map(() => '?').join(',');
  // GROUP BY + MAX(timestamp) gives "latest per chat" without a
  // correlated subquery. Pull the matching content via a self-join so
  // the row's `content` corresponds to the same row whose `timestamp`
  // is the MAX — without the join we'd get arbitrary content from any
  // is_from_me=1 row in the chat. Composite index makes both halves
  // (the GROUP BY scan and the join lookup) fast.
  const sql = `
    SELECT m.chat_jid, m.timestamp, m.content
    FROM messages m
    JOIN (
      SELECT chat_jid, MAX(timestamp) AS max_ts
      FROM messages
      WHERE is_from_me = 1 AND chat_jid IN (${placeholders})
      GROUP BY chat_jid
    ) latest
      ON m.chat_jid = latest.chat_jid
     AND m.timestamp = latest.max_ts
     AND m.is_from_me = 1
  `;
  const rows = db.prepare(sql).all(...chatJids) as Array<{
    chat_jid: string;
    timestamp: string;
    content: string;
  }>;
  for (const row of rows) {
    // Multiple is_from_me=1 messages with the same MAX timestamp would
    // produce duplicate rows; the Map dedupes by keeping the last
    // assignment. This is rare enough (millisecond-precision
    // timestamps) that picking arbitrarily is fine — the docstring
    // promises "latest", not a deterministic tiebreak.
    out.set(row.chat_jid, { timestamp: row.timestamp, content: row.content });
  }
  return out;
}

export function createTask(
  task: Omit<ScheduledTask, 'last_run' | 'last_result'>,
): void {
  db.prepare(
    `
    INSERT INTO scheduled_tasks (id, group_folder, chat_jid, prompt, script, schedule_type, schedule_value, schedule_timezone, context_mode, next_run, status, created_at, created_by_role, continuation_cycle_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `,
  ).run(
    task.id,
    task.group_folder,
    task.chat_jid,
    task.prompt,
    task.script || null,
    task.schedule_type,
    task.schedule_value,
    task.schedule_timezone || null,
    task.context_mode || 'isolated',
    task.next_run,
    task.status,
    task.created_at,
    task.created_by_role,
    task.continuation_cycle_id || null,
  );
}

export function getTaskById(id: string): ScheduledTask | undefined {
  return db.prepare('SELECT * FROM scheduled_tasks WHERE id = ?').get(id) as
    | ScheduledTask
    | undefined;
}

export function getTasksForGroup(groupFolder: string): ScheduledTask[] {
  return db
    .prepare(
      'SELECT * FROM scheduled_tasks WHERE group_folder = ? ORDER BY created_at DESC',
    )
    .all(groupFolder) as ScheduledTask[];
}

export function getAllTasks(): ScheduledTask[] {
  return db
    .prepare('SELECT * FROM scheduled_tasks ORDER BY created_at DESC')
    .all() as ScheduledTask[];
}

export function updateTask(
  id: string,
  updates: Partial<
    Pick<
      ScheduledTask,
      | 'prompt'
      | 'script'
      | 'schedule_type'
      | 'schedule_value'
      | 'schedule_timezone'
      | 'next_run'
      | 'status'
    >
  >,
): void {
  const fields: string[] = [];
  const values: unknown[] = [];

  if (updates.prompt !== undefined) {
    fields.push('prompt = ?');
    values.push(updates.prompt);
  }
  if (updates.script !== undefined) {
    fields.push('script = ?');
    values.push(updates.script || null);
  }
  if (updates.schedule_type !== undefined) {
    fields.push('schedule_type = ?');
    values.push(updates.schedule_type);
  }
  if (updates.schedule_value !== undefined) {
    fields.push('schedule_value = ?');
    values.push(updates.schedule_value);
  }
  if (updates.schedule_timezone !== undefined) {
    fields.push('schedule_timezone = ?');
    values.push(updates.schedule_timezone || null);
  }
  if (updates.next_run !== undefined) {
    fields.push('next_run = ?');
    values.push(updates.next_run);
  }
  if (updates.status !== undefined) {
    fields.push('status = ?');
    values.push(updates.status);
  }

  if (fields.length === 0) return;

  values.push(id);
  db.prepare(
    `UPDATE scheduled_tasks SET ${fields.join(', ')} WHERE id = ?`,
  ).run(...values);
}

export function deleteTask(id: string): void {
  // Delete child records first (FK constraint)
  db.prepare('DELETE FROM task_run_logs WHERE task_id = ?').run(id);
  db.prepare('DELETE FROM scheduled_tasks WHERE id = ?').run(id);
}

/**
 * Delete completed once-tasks older than maxAgeMs.
 *
 * Age is measured from `COALESCE(last_run, created_at)` rather than
 * `last_run` alone. The scheduler pre-advances `status='completed'`
 * before dispatch (see `task-scheduler.ts`), and `updateTaskAfterRun`
 * is what actually stamps `last_run`. If a task is marked completed but
 * the dispatch path fails (container crash, maintenance slot wedged,
 * task aborted before the streaming callback fires), `last_run` stays
 * NULL forever — the original `last_run < cutoff` filter would never
 * match, and the orphan row would linger indefinitely. Falling back to
 * `created_at` guarantees these rows are eventually pruned by their
 * own age.
 *
 * Trade-off: a once-task scheduled far in advance and only just now
 * marked completed (with `last_run` NULL because dispatch failed) is
 * pruned earlier than the user-facing "TTL after completion" intent —
 * the row could disappear immediately if `created_at` is already past
 * the cutoff. This is acceptable because (a) such rows were never
 * visible to the user as completed during normal operation, so there's
 * no observable regression vs. the case where the task ran and stamped
 * last_run; (b) the alternative of letting NULL-last_run rows linger
 * indefinitely (the bug we're fixing) is strictly worse. A future
 * `completed_at` column would let us preserve the grace window even for
 * orphans; until then COALESCE is the closest approximation that
 * doesn't require a schema migration.
 *
 * Recurring tasks never reach status='completed' (computeNextRun only
 * returns null for once-tasks), so the schedule_type='once' clause is
 * defensive. Returns row count removed.
 */
export function pruneCompletedTasks(maxAgeMs: number): number {
  const cutoff = new Date(Date.now() - maxAgeMs).toISOString();
  const tx = db.transaction((cutoffIso: string): number => {
    db.prepare(
      `DELETE FROM task_run_logs
       WHERE task_id IN (
         SELECT id FROM scheduled_tasks
         WHERE status = 'completed'
           AND schedule_type = 'once'
           AND COALESCE(last_run, created_at) < ?
       )`,
    ).run(cutoffIso);
    return db
      .prepare(
        `DELETE FROM scheduled_tasks
         WHERE status = 'completed'
           AND schedule_type = 'once'
           AND COALESCE(last_run, created_at) < ?`,
      )
      .run(cutoffIso).changes;
  });
  return tx(cutoff);
}

/**
 * Recover once-tasks whose pre-advance landed but whose dispatch did
 * not. The pre-advance write at `task-scheduler.ts` flips a once-task
 * to `status='completed'` *before* `enqueueTask` is called, so a host
 * crash, a queue shutdown, or a streaming-callback failure between
 * those two lines leaves a row with the orphan signature
 * `status='completed' AND schedule_type='once' AND last_run IS NULL
 * AND next_run IS NOT NULL`. `getDueTasks()` filters on
 * `status='active'`, so the row never re-tries; `pruneCompletedTasks`
 * eventually GCs it via the `COALESCE(last_run, created_at)` fallback,
 * but by then the schedule is silently lost.
 *
 * Called once at scheduler startup (see `task-scheduler.ts`). Flips
 * matching rows back to `active` so the next `getDueTasks()` poll
 * picks them up and `runTask` dispatches them — late but firing.
 *
 * If dispatch fails *again* for the same row (same race), the next
 * restart resurrects it again, and `pruneCompletedTasks` still GCs it
 * on age via the `created_at` fallback — no risk of zombie loop.
 *
 * Returns the list of resurrected task ids (sorted by id for
 * deterministic output) for logging and assertion in tests.
 *
 * Atomicity: SELECT and the per-id UPDATEs run inside a single
 * transaction. The UPDATE re-asserts the full zombie predicate so a
 * row that races (e.g., a concurrent dispatch landing between SELECT
 * and UPDATE) is left alone — the UPDATE is a no-op and the id is
 * dropped from the returned list.
 */
export function resurrectZombieTasks(): string[] {
  const tx = db.transaction((): string[] => {
    const rows = db
      .prepare(
        `SELECT id FROM scheduled_tasks
         WHERE status = 'completed'
           AND schedule_type = 'once'
           AND last_run IS NULL
           AND next_run IS NOT NULL
         ORDER BY id`,
      )
      .all() as Array<{ id: string }>;
    if (rows.length === 0) return [];
    const stmt = db.prepare(
      `UPDATE scheduled_tasks
       SET status = 'active'
       WHERE id = ?
         AND status = 'completed'
         AND schedule_type = 'once'
         AND last_run IS NULL
         AND next_run IS NOT NULL`,
    );
    const resurrectedIds: string[] = [];
    for (const { id } of rows) {
      if (stmt.run(id).changes > 0) resurrectedIds.push(id);
    }
    return resurrectedIds;
  });
  return tx();
}

/**
 * Find recurring (cron / interval) tasks that are still `status='active'`
 * whose age (last_run, falling back to created_at) is older than
 * `maxAgeMs`. These are NOT pruned — only surfaced so the scheduler can
 * emit a warn-level log. A dormant cron is a symptom, not garbage: the
 * row points at a real schedule; what's broken is dispatch (next_run
 * not advancing, container queue stuck, etc.). Visibility first; humans
 * decide whether to delete.
 *
 * `COALESCE(last_run, created_at) < ?` (vs the original
 * `last_run IS NULL OR last_run < ?`) prevents false-positive warnings
 * for freshly-created recurring tasks whose `last_run` is NULL because
 * they simply haven't been due yet — matching the threshold-based
 * semantics for the same NULL-last_run shape that `pruneCompletedTasks`
 * already uses.
 */
export function getDormantRecurringTasks(maxAgeMs: number): ScheduledTask[] {
  const cutoff = new Date(Date.now() - maxAgeMs).toISOString();
  return db
    .prepare(
      `SELECT * FROM scheduled_tasks
       WHERE status = 'active'
         AND schedule_type IN ('cron', 'interval')
         AND COALESCE(last_run, created_at) < ?`,
    )
    .all(cutoff) as ScheduledTask[];
}

export function getDueTasks(): ScheduledTask[] {
  const now = new Date().toISOString();
  return db
    .prepare(
      `
    SELECT * FROM scheduled_tasks
    WHERE status = 'active' AND next_run IS NOT NULL AND next_run <= ?
    ORDER BY next_run
  `,
    )
    .all(now) as ScheduledTask[];
}

export function updateTaskAfterRun(
  id: string,
  nextRun: string | null,
  lastResult: string,
): void {
  const now = new Date().toISOString();
  // Status transitions (in CASE-evaluation order):
  //   - status = 'paused' → stay 'paused'. A runtime parse failure that
  //     paused the task via computeNextRun during this very run must
  //     not be flipped back to 'completed' just because nextRun is null.
  //     See #102 round-4 review.
  //   - nextRun IS NULL (and status is anything other than 'paused')
  //     → 'completed'. Covers the natural once-task end. Note that
  //     'completed' rows that re-enter this code path would also flip
  //     here, which is harmless (they were already terminal).
  //   - otherwise → status unchanged.
  db.prepare(
    `
    UPDATE scheduled_tasks
    SET next_run = ?, last_run = ?, last_result = ?,
        status = CASE
          WHEN status = 'paused' THEN 'paused'
          WHEN ? IS NULL THEN 'completed'
          ELSE status
        END
    WHERE id = ?
  `,
  ).run(nextRun, now, lastResult, nextRun, id);
}

export function logTaskRun(log: TaskRunLog): void {
  db.prepare(
    `
    INSERT INTO task_run_logs (task_id, run_at, duration_ms, status, result, error)
    VALUES (?, ?, ?, ?, ?, ?)
  `,
  ).run(
    log.task_id,
    log.run_at,
    log.duration_ms,
    log.status,
    log.result,
    log.error,
  );
}

// --- Router state accessors ---

export function getRouterState(key: string): string | undefined {
  const row = db
    .prepare('SELECT value FROM router_state WHERE key = ?')
    .get(key) as { value: string } | undefined;
  return row?.value;
}

export function setRouterState(key: string, value: string): void {
  db.prepare(
    'INSERT OR REPLACE INTO router_state (key, value) VALUES (?, ?)',
  ).run(key, value);
}

// --- Session accessors ---
//
// Sessions are keyed by (groupFolder, sessionName). `sessionName` is one of
// the canonical slot names — `default` (user-facing) or `maintenance`
// (scheduled tasks). See `DEFAULT_SESSION_NAME` in `src/container-runner.ts`
// and `MAINTENANCE_SESSION_NAME` in `src/group-queue.ts`.

export function getSession(
  groupFolder: string,
  sessionName: string,
): string | undefined {
  const row = db
    .prepare(
      'SELECT session_id FROM sessions WHERE group_folder = ? AND session_name = ?',
    )
    .get(groupFolder, sessionName) as { session_id: string } | undefined;
  return row?.session_id;
}

export function setSession(
  groupFolder: string,
  sessionName: string,
  sessionId: string,
): void {
  db.prepare(
    'INSERT OR REPLACE INTO sessions (group_folder, session_name, session_id) VALUES (?, ?, ?)',
  ).run(groupFolder, sessionName, sessionId);
}

/**
 * Delete all stored sessions for a group (both default and maintenance).
 * Called on nuke(session='all') so both containers start fresh on their
 * next spawn.
 */
export function deleteSession(groupFolder: string): void {
  db.prepare('DELETE FROM sessions WHERE group_folder = ?').run(groupFolder);
}

/**
 * Delete a single session slot for a group. Called on granular nuke
 * (`nuke_session(session: "default" | "maintenance")`) so the surviving
 * slot keeps its session chain intact.
 */
export function deleteSessionName(
  groupFolder: string,
  sessionName: string,
): void {
  db.prepare(
    'DELETE FROM sessions WHERE group_folder = ? AND session_name = ?',
  ).run(groupFolder, sessionName);
}

export function deleteAllSessions(): number {
  const result = db.prepare('DELETE FROM sessions').run();
  return result.changes;
}

/**
 * Returns sessions keyed first by groupFolder then by sessionName:
 *   { "main": { "default": "abc-123", "maintenance": "def-456" } }
 * Callers looking up a specific session do
 *   `sessions[folder]?.[sessionName]`
 * and handle the missing case (fresh session chain for that slot).
 */
export function getAllSessions(): Record<string, Record<string, string>> {
  const rows = db
    .prepare('SELECT group_folder, session_name, session_id FROM sessions')
    .all() as Array<{
    group_folder: string;
    session_name: string;
    session_id: string;
  }>;
  const result: Record<string, Record<string, string>> = {};
  for (const row of rows) {
    if (!result[row.group_folder]) result[row.group_folder] = {};
    result[row.group_folder][row.session_name] = row.session_id;
  }
  return result;
}

// --- Registered group accessors ---

// Defensive parser shared by getRegisteredGroup and getAllRegisteredGroups.
// A single malformed row (partial write, manual edit, schema-migration glitch)
// must not crash startup — getAllRegisteredGroups runs at orchestrator boot.
//
// Catches SyntaxError specifically (JSON.parse's only throw); other errors
// propagate. Validates the parsed value is a non-null object — JSON.parse
// can legally return primitives, null, or arrays from `"null"`, `"true"`,
// `"[]"`, etc., none of which are valid ContainerConfig shapes.
//
// Logs jid + payload length only — never the payload content. The raw
// container_config string is treated as opaque/possibly-sensitive per
// no-secrets and error-handling rules. Operators inspect the actual row
// via the DB by jid, not via logs.
function parseContainerConfig(
  raw: string | null,
  jid: string,
): ContainerConfig | undefined {
  // Distinguish SQL NULL from empty string: NULL is the documented
  // "no config" state, while an empty string in a TEXT column is itself
  // a corruption indicator (something wrote "" where it should have
  // written NULL). Fall through into the parse path so SyntaxError
  // surfaces it.
  if (raw === null) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    logger.warn(
      { errName: err.name, jid, len: raw.length },
      'registered_groups: invalid container_config JSON, treating as undefined',
    );
    return undefined;
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    logger.warn(
      {
        jid,
        len: raw.length,
        parsedType:
          parsed === null
            ? 'null'
            : Array.isArray(parsed)
              ? 'array'
              : typeof parsed,
      },
      'registered_groups: container_config is not a JSON object, treating as undefined',
    );
    return undefined;
  }
  return parsed as ContainerConfig;
}

export function getRegisteredGroup(
  jid: string,
): (RegisteredGroup & { jid: string }) | undefined {
  const row = db
    .prepare('SELECT * FROM registered_groups WHERE jid = ?')
    .get(jid) as
    | {
        jid: string;
        name: string;
        folder: string;
        trigger_pattern: string;
        added_at: string;
        container_config: string | null;
        requires_trigger: number | null;
        is_main: number | null;
      }
    | undefined;
  if (!row) return undefined;
  if (!isValidGroupFolder(row.folder)) {
    logger.warn(
      { jid: row.jid, folder: row.folder },
      'Skipping registered group with invalid folder',
    );
    return undefined;
  }
  return {
    jid: row.jid,
    name: row.name,
    folder: row.folder,
    trigger: row.trigger_pattern,
    added_at: row.added_at,
    containerConfig: parseContainerConfig(row.container_config, row.jid),
    requiresTrigger:
      row.requires_trigger === null ? undefined : row.requires_trigger === 1,
    isMain: row.is_main === 1 ? true : undefined,
  };
}

export function setRegisteredGroup(jid: string, group: RegisteredGroup): void {
  if (!isValidGroupFolder(group.folder)) {
    throw new Error(`Invalid group folder "${group.folder}" for JID ${jid}`);
  }
  db.prepare(
    `INSERT OR REPLACE INTO registered_groups (jid, name, folder, trigger_pattern, added_at, container_config, requires_trigger, is_main)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    jid,
    group.name,
    group.folder,
    group.trigger,
    group.added_at,
    group.containerConfig ? JSON.stringify(group.containerConfig) : null,
    // Map TS `undefined` to SQL NULL (not 0). NULL and 0 are distinct
    // states elsewhere in the orchestrator: `index.ts` checks
    // `requiresTrigger === false` to decide whether to skip a group's
    // heartbeat sync, and a NULL row should NOT match that branch.
    // Pre-#105, this column wrote 0 for undefined, which silently
    // collapsed the NULL state on every round-trip — biting the new
    // partial-update helpers (`updateGroupTrusted`/`updateGroupTrigger`)
    // because they read existing → reapply. Callers that want explicit
    // false must pass `false` explicitly; callers passing `undefined`
    // get NULL preserved.
    group.requiresTrigger === undefined ? null : group.requiresTrigger ? 1 : 0,
    group.isMain ? 1 : 0,
  );
}

/**
 * Partial update: flip `containerConfig.trusted` only.
 *
 * Returns the updated RegisteredGroup, or `undefined` if the JID isn't
 * registered. The caller is responsible for refreshing in-memory state
 * and snapshots — this function only touches the DB row.
 *
 * Implementation note: we round-trip through `getRegisteredGroup` to
 * preserve every other field (additionalMounts, isMain, etc.) verbatim,
 * then write back via `setRegisteredGroup`. A targeted SQL UPDATE on the
 * JSON column would be marginally faster but would force us to either
 * mutate the JSON string textually (fragile) or duplicate the JSON
 * encoding logic that already lives in `setRegisteredGroup`.
 */
export function updateGroupTrusted(
  jid: string,
  trusted: boolean,
): RegisteredGroup | undefined {
  const existing = getRegisteredGroup(jid);
  if (!existing) return undefined;
  // `getRegisteredGroup` synthesizes its return value with `jid` set as
  // an extra runtime field for caller convenience, but `RegisteredGroup`
  // doesn't declare it. Strip via destructure before spreading so the
  // value we hand back (and the in-memory cache the orchestrator
  // mirrors into) doesn't carry the DB-only key.
  const { jid: _existingJid, ...rest } = existing;
  void _existingJid;
  const updated: RegisteredGroup = {
    ...rest,
    containerConfig: {
      ...(rest.containerConfig ?? {}),
      trusted,
    },
  };
  setRegisteredGroup(jid, updated);
  return updated;
}

/**
 * Partial update: change `trigger_pattern` and optionally `requires_trigger`
 * only. Other fields preserved. Returns updated group or `undefined` if
 * the JID isn't registered or the trigger fails the non-empty invariant.
 *
 * Why reject empty/whitespace triggers: `getTriggerPattern('')` trims
 * and falls back to `DEFAULT_TRIGGER`, so a caller that thinks they're
 * setting a custom trigger would silently get the assistant's default
 * trigger word instead — not what they asked for. Reject at the DB
 * boundary so any future caller (cron migrations, manual fixups,
 * alternate MCP tools) can't bypass the IPC-layer check.
 *
 * The trigger is also `.trim()`ed before persistence so `' @Andy '`
 * doesn't end up stored with surrounding whitespace (which would render
 * that way in `available_groups.json` and elsewhere).
 */
export function updateGroupTrigger(
  jid: string,
  trigger: string,
  requiresTrigger?: boolean,
): RegisteredGroup | undefined {
  if (typeof trigger !== 'string' || trigger.trim().length === 0) {
    logger.warn(
      { jid },
      'updateGroupTrigger: rejecting empty/whitespace trigger',
    );
    return undefined;
  }
  const normalizedTrigger = trigger.trim();
  const existing = getRegisteredGroup(jid);
  if (!existing) return undefined;
  // Strip the DB-only `jid` field so it doesn't leak into the returned
  // RegisteredGroup or into the in-memory cache the orchestrator
  // mirrors into. Same rationale as updateGroupTrusted above.
  const { jid: _existingJid, ...rest } = existing;
  void _existingJid;
  const updated: RegisteredGroup = {
    ...rest,
    trigger: normalizedTrigger,
    ...(requiresTrigger === undefined ? {} : { requiresTrigger }),
  };
  setRegisteredGroup(jid, updated);
  return updated;
}

/**
 * Remove a registered_groups row by JID. Returns true if a row was
 * actually deleted, false if no row matched. Idempotent — repeat calls
 * after deletion are a no-op and report `false`.
 *
 * Caller is responsible for refreshing in-memory state and snapshots —
 * this function only touches the DB row, mirroring the
 * `setRegisteredGroup` / `updateGroupTrusted` contract.
 *
 * Out of scope: the on-disk `groups/<folder>/` directory. Group state
 * (CLAUDE.md, MEMORY.md, scheduled-task workspace) survives unregister
 * — operators delete those manually if/when they want a clean slate.
 * Forces a deliberate destructive action instead of silently nuking
 * agent-curated state when the registration churns.
 */
export function deleteRegisteredGroup(jid: string): boolean {
  const result = db
    .prepare('DELETE FROM registered_groups WHERE jid = ?')
    .run(jid);
  return result.changes > 0;
}

export function getAllRegisteredGroups(): Record<string, RegisteredGroup> {
  const rows = db.prepare('SELECT * FROM registered_groups').all() as Array<{
    jid: string;
    name: string;
    folder: string;
    trigger_pattern: string;
    added_at: string;
    container_config: string | null;
    requires_trigger: number | null;
    is_main: number | null;
  }>;
  const result: Record<string, RegisteredGroup> = {};
  for (const row of rows) {
    if (!isValidGroupFolder(row.folder)) {
      logger.warn(
        { jid: row.jid, folder: row.folder },
        'Skipping registered group with invalid folder',
      );
      continue;
    }
    result[row.jid] = {
      name: row.name,
      folder: row.folder,
      trigger: row.trigger_pattern,
      added_at: row.added_at,
      containerConfig: parseContainerConfig(row.container_config, row.jid),
      requiresTrigger:
        row.requires_trigger === null ? undefined : row.requires_trigger === 1,
      isMain: row.is_main === 1 ? true : undefined,
    };
  }
  return result;
}

// --- Smart Home event accessors ---

export interface SmartHomeEvent {
  id: number;
  device_id: string;
  device_name: string;
  attribute_name: string;
  value: string;
  unit: string | null;
  description: string | null;
  source: string;
  timestamp: string;
}

export function insertSmartHomeEvent(event: Omit<SmartHomeEvent, 'id'>): void {
  db.prepare(
    `INSERT INTO smart_home_events (device_id, device_name, attribute_name, value, unit, description, source, timestamp)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    event.device_id,
    event.device_name,
    event.attribute_name,
    event.value,
    event.unit ?? null,
    event.description ?? null,
    event.source ?? 'DEVICE',
    event.timestamp,
  );
}

export function getSmartHomeEventsSince(
  since: string,
  deviceId?: string,
): SmartHomeEvent[] {
  if (deviceId) {
    return db
      .prepare(
        `SELECT * FROM smart_home_events WHERE timestamp > ? AND device_id = ? ORDER BY timestamp`,
      )
      .all(since, deviceId) as SmartHomeEvent[];
  }
  return db
    .prepare(
      `SELECT * FROM smart_home_events WHERE timestamp > ? ORDER BY timestamp`,
    )
    .all(since) as SmartHomeEvent[];
}

export function getSmartHomeEventsByHour(
  startHour: string,
  endHour: string,
): SmartHomeEvent[] {
  return db
    .prepare(
      `SELECT * FROM smart_home_events WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp`,
    )
    .all(startHour, endHour) as SmartHomeEvent[];
}

export function cleanupOldSmartHomeEvents(retentionDays: number): number {
  const cutoff = new Date(
    Date.now() - retentionDays * 24 * 60 * 60 * 1000,
  ).toISOString();
  const result = db
    .prepare(`DELETE FROM smart_home_events WHERE timestamp < ?`)
    .run(cutoff);
  return result.changes;
}

export function getLatestDeviceStates(): Array<{
  device_id: string;
  device_name: string;
  attribute_name: string;
  value: string;
  unit: string | null;
  timestamp: string;
}> {
  return db
    .prepare(
      `SELECT device_id, device_name, attribute_name, value, unit, timestamp
       FROM smart_home_events e1
       WHERE timestamp = (
         SELECT MAX(timestamp) FROM smart_home_events e2
         WHERE e2.device_id = e1.device_id AND e2.attribute_name = e1.attribute_name
       )
       ORDER BY device_name, attribute_name`,
    )
    .all() as Array<{
    device_id: string;
    device_name: string;
    attribute_name: string;
    value: string;
    unit: string | null;
    timestamp: string;
  }>;
}

// --- JSON migration ---

function migrateJsonState(): void {
  const migrateFile = (filename: string) => {
    const filePath = path.join(DATA_DIR, filename);
    if (!fs.existsSync(filePath)) return null;
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      fs.renameSync(filePath, `${filePath}.migrated`);
      return data;
    } catch {
      return null;
    }
  };

  // Migrate router_state.json
  const routerState = migrateFile('router_state.json') as {
    last_timestamp?: string;
    last_agent_timestamp?: Record<string, string>;
  } | null;
  if (routerState) {
    if (routerState.last_timestamp) {
      setRouterState('last_timestamp', routerState.last_timestamp);
    }
    if (routerState.last_agent_timestamp) {
      setRouterState(
        'last_agent_timestamp',
        JSON.stringify(routerState.last_agent_timestamp),
      );
    }
  }

  // Migrate sessions.json
  const sessions = migrateFile('sessions.json') as Record<
    string,
    string
  > | null;
  if (sessions) {
    // Legacy JSON state predates parallel-maintenance; all sessions were
    // user-facing, so they migrate to the `default` slot.
    for (const [folder, sessionId] of Object.entries(sessions)) {
      setSession(folder, 'default', sessionId);
    }
  }

  // Migrate registered_groups.json
  const groups = migrateFile('registered_groups.json') as Record<
    string,
    RegisteredGroup
  > | null;
  if (groups) {
    for (const [jid, group] of Object.entries(groups)) {
      try {
        setRegisteredGroup(jid, group);
      } catch (err) {
        logger.warn(
          { jid, folder: group.folder, err },
          'Skipping migrated registered group with invalid folder',
        );
      }
    }
  }

  // Migrate per-group orders-db.json files (#294). Unlike the helpers
  // above (DATA_DIR-rooted), this scans every `groups/<name>/` folder
  // for an `orders-db.json` because the file historically lived in the
  // admin group's working dir. Idempotent: a successful migration
  // renames the source to `orders-db.json.migrated-YYYY-MM-DD`, so a
  // re-run of `initDatabase` is a no-op once the file is gone.
  migrateOrdersDbJsonFiles();
}

interface OrdersDbJsonRecord {
  id: string;
  source: string;
  status: string;
  amount?: number | null;
  currency?: string | null;
  description: string;
  order_date: string;
  expected_delivery?: string | null;
  email_message_id: string;
  to_address?: string | null;
  flagged?: boolean;
  flag_reason?: string | null;
  last_updated: string;
}

interface OrdersDbJsonShape {
  orders?: OrdersDbJsonRecord[];
  last_checked?: string;
  last_updated?: string;
}

function migrateOrdersDbJsonFiles(): void {
  if (!fs.existsSync(GROUPS_DIR)) return;
  let groupFolders: string[];
  try {
    groupFolders = fs
      .readdirSync(GROUPS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .filter((entry) => isValidGroupFolder(entry.name))
      .map((entry) => entry.name);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return;
    throw err;
  }

  // ON CONFLICT(email_message_id) DO NOTHING — first writer wins for
  // duplicate email_message_id values. Re-running the migration after
  // a partial success is a no-op for already-imported rows, and the
  // rename-to-`.migrated-<date>` step means a successful run won't be
  // re-attempted at all. A duplicate PK `id` with a *different*
  // email_message_id is not handled by this clause and would still
  // raise — but `id` is `{source}-{order_date}-SHA1(description)[:8]`,
  // so two rows with the same id necessarily came from the same email
  // and thus the same email_message_id; that path is unreachable in
  // practice.
  const insertOrder = db.prepare(
    `INSERT INTO orders (
       id, source, status, amount, currency, description, order_date,
       expected_delivery, email_message_id, to_address, flagged,
       flag_reason, last_updated
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(email_message_id) DO NOTHING`,
  );
  const upsertMetadata = db.prepare(
    `INSERT INTO orders_metadata (key, value) VALUES (?, ?)
     ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
  );

  const stamp = new Date().toISOString().slice(0, 10);

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'orders-db.json');
    if (!fs.existsSync(filePath)) continue;

    let parsed: OrdersDbJsonShape;
    try {
      parsed = JSON.parse(
        fs.readFileSync(filePath, 'utf-8'),
      ) as OrdersDbJsonShape;
    } catch (err) {
      if (!(err instanceof SyntaxError)) throw err;
      logger.warn(
        { folder, errName: err.name },
        'orders-db.json migration: invalid JSON, skipping (file left in place)',
      );
      continue;
    }
    if (!Array.isArray(parsed.orders)) {
      logger.warn(
        { folder },
        'orders-db.json migration: missing "orders" array, skipping',
      );
      continue;
    }

    // Wrap the per-file work in a single transaction so a partial
    // crash mid-import (process kill, IO error on metadata write)
    // can't leave the table in a half-migrated state. Failures here
    // propagate — per `coding-policy: error-handling`, an unexpected
    // exception during a structured INSERT means the data shape
    // doesn't match the schema (a real bug or malformed source file)
    // and the operator must triage before continuing. The schema
    // version gate in applyStateMigrations is independent of this;
    // the schema is already at v1, this is data backfill only.
    const importFile = db.transaction(() => {
      let insertedRows = 0;
      let skippedRows = 0;
      for (const order of parsed.orders!) {
        const result = insertOrder.run(
          order.id,
          order.source,
          order.status,
          order.amount ?? null,
          order.currency ?? null,
          order.description,
          order.order_date,
          order.expected_delivery ?? null,
          order.email_message_id,
          order.to_address ?? null,
          order.flagged ? 1 : 0,
          order.flag_reason ?? null,
          order.last_updated,
        );
        if (result.changes > 0) insertedRows++;
        else skippedRows++;
      }
      if (parsed.last_checked) {
        upsertMetadata.run('last_checked', parsed.last_checked);
      }
      if (parsed.last_updated) {
        upsertMetadata.run('last_updated', parsed.last_updated);
      }
      return { insertedRows, skippedRows };
    });

    const counts = importFile();

    // The rename is metadata cleanup, not data integrity — the import
    // already committed. A rename failure (cross-device EXDEV, EACCES,
    // etc.) must NOT block startup; logging at warn lets the operator
    // see the situation without losing the orchestrator. The next
    // restart will re-import (idempotent via ON CONFLICT) and try the
    // rename again. Filtering on ENOENT mirrors the pattern already
    // used for the readdirSync above — non-existent source means
    // someone else already moved/removed it.
    try {
      fs.renameSync(filePath, `${filePath}.migrated-${stamp}`);
      logger.info(
        {
          folder,
          inserted: counts.insertedRows,
          skipped: counts.skippedRows,
          total: parsed.orders.length,
        },
        'orders-db.json migration: imported and source renamed',
      );
    } catch (err) {
      if (!(err instanceof Error)) throw err;
      const errno = (err as NodeJS.ErrnoException).code;
      if (errno === 'ENOENT') continue;
      logger.warn(
        {
          folder,
          errno,
          errName: err.name,
          inserted: counts.insertedRows,
          skipped: counts.skippedRows,
          total: parsed.orders.length,
        },
        'orders-db.json migration: imported but rename failed; file left in place for retry',
      );
    }
  }
}
