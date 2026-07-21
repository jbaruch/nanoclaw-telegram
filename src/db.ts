import Database, { SqliteError } from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

import { ASSISTANT_NAME, STORE_DIR } from './config.js';
import { db, setDbHandle } from './db-connection.js';
import { migrateJsonState } from './db-json-migrations.js';
import { isTriggerPatternConfig } from './db-registered-groups.js';
import { logger } from './logger.js';
import { STATE_MIGRATIONS } from './state-migrations/index.js';
import { TriggerPatternConfig } from './types.js';

/**
 * True for the SqliteError an idempotent `ALTER TABLE ... ADD COLUMN`
 * migration raises when the column already exists ("duplicate column
 * name"). Narrowed to that message so a different SqliteError (missing
 * table, corruption) still propagates instead of being mistaken for an
 * already-applied migration.
 */
function isDuplicateColumnError(err: unknown): boolean {
  return (
    err instanceof SqliteError && /duplicate column name/i.test(err.message)
  );
}

/**
 * Idempotently add one column. Guarding each `ADD COLUMN` independently
 * (rather than grouping several under one try/catch) avoids the
 * half-migration failure mode where a duplicate-column error on an early
 * ALTER skips the remaining ALTERs. Returns `true` when the column was
 * newly added, `false` when it already existed — callers gate a one-time
 * backfill on that. A non-duplicate SqliteError (or any other defect)
 * propagates.
 */
function addColumnIfMissing(
  database: Database.Database,
  table: string,
  columnDef: string,
): boolean {
  try {
    database.exec(`ALTER TABLE ${table} ADD COLUMN ${columnDef}`);
    return true;
  } catch (err) {
    if (!isDuplicateColumnError(err)) throw err;
    return false;
  }
}

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
      -- Telegram message ID, populated for BOTH directions on tg:%
      -- chats (#691). Inbound rows also keep the Telegram ID in id;
      -- bot sends keep a synthetic bot-<ts>-<rand> in id. This is the
      -- single column reply_to_message_id joins against:
      --   WHERE chat_jid = ? AND telegram_message_id = <reply_to_message_id>
      -- NULL for non-Telegram channels (their platform ID lives in id).
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
      continuation_cycle_id TEXT,
      -- Per-task SDK session id for #336. Recurring tasks (cron / interval)
      -- persist the SDK's newSessionId here on first fire and pass it as
      -- the resume id on every subsequent fire — the API caches the
      -- per-session message-history prefix across the (otherwise expiring)
      -- prompt-cache window, so each fire's cache_create is incremental
      -- rather than full-prefix. NULL for tasks that haven't fired yet, for
      -- once-tasks (out of scope per #336), and for recurring tasks
      -- immediately after a session wipe (nuke_session clears this column
      -- for every scheduled task in the affected group; the next fire
      -- starts fresh). The #193 cross-task bleed concern does NOT apply
      -- here — that bug came from a single id shared via
      -- sessions[group][maintenance] across DIFFERENT tasks. Persistence
      -- here is keyed on task_id, so different tasks have different rows
      -- hence different sessions hence no bleed.
      session_id TEXT,
      -- Content hash of the tessl plugin registry at the moment
      -- session_id was persisted (#710). The scheduler compares it
      -- against the current registry hash at fire time and rotates to
      -- a fresh SDK session on mismatch — a resumed session never
      -- re-reads skill/rule content, so without this gate a pinned
      -- session outlives every plugin update. NULL means "hash unknown
      -- at persist time" (registry absent, or the id predates #710);
      -- paired with session_id, cleared whenever session_id clears.
      session_plugins_hash TEXT,
      -- Row-creation provenance for #305 Phase 2 cadence-registry. See
      -- the ALTER block below for the value set and ownership semantics.
      -- 'schedule-task' is the default so unmigrated callers (the
      -- existing schedule-task IPC path) keep their semantics unchanged.
      source TEXT NOT NULL DEFAULT 'schedule-task',
      -- Per-task AGENT_MODEL override for #509 Phase 3. NULL = no per-task
      -- override; fall through to the Phase 2 ladder. When non-null AND
      -- the resolved value differs from the session-level fallback, the
      -- spawn routes through it and the audit log emits source =
      -- task_override; unknown-prefix or matches-session values still
      -- get stored here but the audit log shows the session-level
      -- source (no effective routing change). See
      -- src/container-runner.ts resolveSessionAgentModel for the
      -- precedence rules.
      agent_model TEXT,
      -- Declared work-evidence contract <relative-file>#<json-field>
      -- for #720. Written by the cadence-registry from evidence:
      -- frontmatter; the scheduler verifies post-run that the named
      -- JSON date field in the group folder was freshened during the
      -- run, else records the run as 'error' and clears the pinned
      -- session. NULL = no evidence contract (all owner-scheduled
      -- tasks).
      evidence TEXT
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
      FOREIGN KEY (task_id) REFERENCES scheduled_tasks(id) ON DELETE CASCADE
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
  addColumnIfMissing(
    database,
    'scheduled_tasks',
    `context_mode TEXT DEFAULT 'isolated'`,
  );

  // Add script column if it doesn't exist (migration for existing DBs)
  addColumnIfMissing(database, 'scheduled_tasks', 'script TEXT');

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

  // Add session_id column for #336 — per-task SDK session reuse across
  // recurring fires. NULL on existing rows; populated on first
  // post-deploy fire for cron/interval tasks (once-tasks stay NULL by
  // design). PRAGMA-gated rather than try/catch per the
  // no-error-suppression rule.
  const sessionIdCols = database
    .prepare('PRAGMA table_info(scheduled_tasks)')
    .all() as Array<{ name: string }>;
  if (!sessionIdCols.some((c) => c.name === 'session_id')) {
    database.exec(`ALTER TABLE scheduled_tasks ADD COLUMN session_id TEXT`);
  }

  // Add session_plugins_hash column for #710 — plugin-content hash
  // paired with session_id so the scheduler can detect that the
  // registry changed since the pinned session was created and rotate
  // to a fresh one. NULL on existing rows: any pre-#710 pinned
  // session_id compares NULL vs current hash on its first post-deploy
  // fire, mismatches, and rotates — which is the fix taking effect
  // for exactly the stale sessions the issue describes.
  const pluginsHashCols = database
    .prepare('PRAGMA table_info(scheduled_tasks)')
    .all() as Array<{ name: string }>;
  if (!pluginsHashCols.some((c) => c.name === 'session_plugins_hash')) {
    database.exec(
      `ALTER TABLE scheduled_tasks ADD COLUMN session_plugins_hash TEXT`,
    );
  }

  // Add source column for #305 Phase 2 — provenance of a scheduled_tasks
  // row's CREATION (distinct from `created_by_role`, which is the trust-
  // boundary provenance for whether the agent-runner wraps the prompt in
  // <untrusted-input> at fire time):
  //   'schedule-task'    — created via the schedule-task IPC tool
  //                        (admin shells, owner-initiated reminders,
  //                        ad-hoc monitors). Default for back-compat.
  //   'cadence-registry' — created by the per-spawn cadence-registry
  //                        rebuild from a SKILL.md `cadence:` frontmatter
  //                        declaration. Idempotently DELETEd + reinserted
  //                        on each container spawn.
  // The cadence-registry's idempotent rebuild ONLY touches rows where
  // `source = 'cadence-registry'` so owner-scheduled tasks survive
  // respawns. PRAGMA-gated rather than try/catch per the no-error-
  // suppression rule.
  const sourceCols = database
    .prepare('PRAGMA table_info(scheduled_tasks)')
    .all() as Array<{ name: string }>;
  if (!sourceCols.some((c) => c.name === 'source')) {
    database.exec(
      `ALTER TABLE scheduled_tasks ADD COLUMN source TEXT NOT NULL DEFAULT 'schedule-task'`,
    );
  }

  // Add agent_model column for #509 Phase 3 — per-task AGENT_MODEL override.
  // NULL on existing rows; populated either declaratively via the
  // cadence-registry's `agentModel:` frontmatter on the next per-spawn
  // rebuild, or imperatively via the `set_task_agent_model` IPC handler.
  // Resolution at spawn time is: per-row agent_model → maintenanceAgentModel
  // (maintenance session) → group agentModel → AGENT_MODEL env →
  // DEFAULT_AGENT_MODEL — see `resolveSessionAgentModel` in
  // src/container-runner.ts. PRAGMA-gated rather than try/catch per the
  // no-error-suppression rule.
  const agentModelCols = database
    .prepare('PRAGMA table_info(scheduled_tasks)')
    .all() as Array<{ name: string }>;
  if (!agentModelCols.some((c) => c.name === 'agent_model')) {
    database.exec(`ALTER TABLE scheduled_tasks ADD COLUMN agent_model TEXT`);
  }

  // Add evidence column for #720 — declared work-evidence contract
  // `<relative-file>#<json-field>`. NULL on existing rows; populated
  // declaratively via the cadence-registry's `evidence:` frontmatter on
  // the next per-spawn rebuild. The scheduler checks it post-run — see
  // `checkTaskEvidence` in src/task-scheduler.ts. PRAGMA-gated rather
  // than try/catch per the no-error-suppression rule.
  const evidenceCols = database
    .prepare('PRAGMA table_info(scheduled_tasks)')
    .all() as Array<{ name: string }>;
  if (!evidenceCols.some((c) => c.name === 'evidence')) {
    database.exec(`ALTER TABLE scheduled_tasks ADD COLUMN evidence TEXT`);
  }

  // Switch task_run_logs.task_id FK to ON DELETE CASCADE. Without this,
  // rebuildCadenceRegistry's orphan DELETE (a scheduled_tasks row whose
  // skill stopped declaring `cadence:`) FK-aborts the entire per-spawn
  // rebuild transaction when run-log children remain — the spawn never
  // happens, the message cursor rolls back, retry loops until the
  // group's circuit breaker trips. Same latent bug applies to
  // unscheduleTask via the schedule-task IPC. SQLite can't ALTER a FK
  // in place, so detect via pragma_foreign_key_list and rebuild the
  // table when on_delete is anything other than CASCADE.
  const trlFk = database
    .prepare(
      `SELECT on_delete FROM pragma_foreign_key_list('task_run_logs') WHERE "table" = 'scheduled_tasks'`,
    )
    .get() as { on_delete?: string } | undefined;
  if (trlFk && trlFk.on_delete !== 'CASCADE') {
    // Standard 12-step table rebuild from the SQLite docs. FKs go OFF
    // for the rewrite so the temporary _new table can be populated and
    // renamed without enforcement; flipped back ON after. Deferred FK
    // checks (foreign_key_check) aren't needed — the only constraint
    // we're modifying is task_run_logs → scheduled_tasks, and INSERT
    // INTO ... SELECT preserves every existing row's task_id literally,
    // so any pre-existing orphans (which couldn't have existed under
    // FK-ON anyway, since their parent DELETE would have FK-aborted)
    // stay exactly as they were.
    database.pragma('foreign_keys = OFF');
    database.exec(`
      BEGIN;
      CREATE TABLE task_run_logs_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        run_at TEXT NOT NULL,
        duration_ms INTEGER NOT NULL,
        status TEXT NOT NULL,
        result TEXT,
        error TEXT,
        FOREIGN KEY (task_id) REFERENCES scheduled_tasks(id) ON DELETE CASCADE
      );
      INSERT INTO task_run_logs_new (id, task_id, run_at, duration_ms, status, result, error)
        SELECT id, task_id, run_at, duration_ms, status, result, error FROM task_run_logs;
      DROP TABLE task_run_logs;
      ALTER TABLE task_run_logs_new RENAME TO task_run_logs;
      CREATE INDEX idx_task_run_logs ON task_run_logs(task_id, run_at);
      COMMIT;
    `);
    database.pragma('foreign_keys = ON');
  }

  // Add is_bot_message column if it doesn't exist (migration for existing DBs)
  if (
    addColumnIfMissing(database, 'messages', 'is_bot_message INTEGER DEFAULT 0')
  ) {
    // Backfill: mark existing bot messages that used the content prefix pattern
    database
      .prepare(`UPDATE messages SET is_bot_message = 1 WHERE content LIKE ?`)
      .run(`${ASSISTANT_NAME}:%`);
  }

  // Add is_main column if it doesn't exist (migration for existing DBs)
  if (
    addColumnIfMissing(
      database,
      'registered_groups',
      'is_main INTEGER DEFAULT 0',
    )
  ) {
    // Backfill: existing rows with folder = 'main' are the main group
    database.exec(
      `UPDATE registered_groups SET is_main = 1 WHERE folder = 'main'`,
    );
  }

  // Add channel and is_group columns if they don't exist (migration for existing DBs)
  const channelAdded = addColumnIfMissing(database, 'chats', 'channel TEXT');
  const isGroupAdded = addColumnIfMissing(
    database,
    'chats',
    'is_group INTEGER DEFAULT 0',
  );
  if (channelAdded || isGroupAdded) {
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
  }

  // Add reply context columns if they don't exist (migration for existing DBs)
  addColumnIfMissing(database, 'messages', 'reply_to_message_id TEXT');
  addColumnIfMissing(database, 'messages', 'reply_to_message_content TEXT');
  addColumnIfMissing(database, 'messages', 'reply_to_sender_name TEXT');

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

  // Normalize `telegram_message_id` to hold the Telegram message ID for
  // BOTH directions (#691). Inbound rows historically stored the
  // Telegram ID only in `id` (leaving `telegram_message_id` NULL), while
  // bot sends store a synthetic `bot-<ts>-<rand>` in `id` and the
  // Telegram ID in `telegram_message_id`. That split meant
  // `reply_to_message_id` (always a bare Telegram ID) had no single
  // column to join against: `WHERE id = ?` silently missed every reply
  // to a bot message, and `WHERE telegram_message_id = ?` missed every
  // reply to an inbound one. Backfilling inbound rows gives
  // `telegram_message_id` as the single join target for all Telegram
  // rows. Idempotent (only touches NULL rows) and index-served by
  // idx_messages_chat_telegram_id, so it stays cheap on every startup.
  //
  // The `id NOT LIKE 'bot-%'` guard is load-bearing: on a DB that
  // predates the telegram_message_id column, the ALTER above leaves
  // EVERY pre-existing row NULL — including legacy bot sends whose `id`
  // is the synthetic `bot-<ts>-<rand>` and whose real Telegram ID was
  // never recorded (pre-#80). Copying that synthetic id into
  // telegram_message_id would plant a non-Telegram value in the column,
  // breaking the "Telegram-native ID only" contract and poisoning the
  // join. Those rows stay NULL (their Telegram ID is unrecoverable);
  // only rows whose `id` IS the Telegram-native ID are normalized.
  database.exec(
    `UPDATE messages
        SET telegram_message_id = id
      WHERE telegram_message_id IS NULL
        AND chat_jid LIKE 'tg:%'
        AND id NOT LIKE 'bot-%'`,
  );

  // Backfill registered_groups.trigger_pattern from legacy string shape
  // to the JSON `TriggerPatternConfig` shape (#81). Idempotent:
  //   - rows that parse as a valid `TriggerPatternConfig` JSON object
  //     are left alone
  //   - everything else (legacy literal strings, malformed JSON,
  //     corrupted rows, even legacy keywords that happen to start
  //     with `{`) gets wrapped in a single-element config
  //     `{version: 1, patterns: [{kind, source: "owner-set", pattern,
  //     precision: 0, sample_count: 0, last_matched_at: null,
  //     last_updated_at: null}]}` — `kind` is shape-classified per the
  //     mention regex below.
  //
  // We can't ALTER the column type (SQLite is dynamic-typed anyway and
  // `trigger_pattern` is already TEXT NOT NULL), so the migration is
  // a row-by-row UPDATE. Wrapped in a transaction so a crash mid-update
  // can't leave the table half-converted (mixed legacy + new shape is
  // read-safe but makes incident triage harder). better-sqlite3
  // implicitly rolls back on thrown exceptions.
  //
  // Two-pass classification: a cheap SQL prefilter excludes the common
  // case (rows whose value starts with `{` after trim — almost certainly
  // already-JSON), then `isTriggerPatternConfig` is the authoritative
  // shape check on the survivors plus any rows whose value looked like
  // JSON but didn't validate. The corner case the `LIKE '{%'` filter
  // alone misses is a legacy keyword that legitimately starts with `{`,
  // OR a corrupted row whose first char is `{` but body is broken
  // — both would otherwise be silently skipped and never converted.
  // Reading the column shape once at boot is cheap (one row per group);
  // the transaction wrapper means a partial pass crashes back to all-or-
  // nothing.
  const allRows = database
    .prepare(`SELECT jid, trigger_pattern FROM registered_groups`)
    .all() as Array<{ jid: string; trigger_pattern: string }>;
  const legacyTriggerRows = allRows.filter((row) => {
    const trimmed = row.trigger_pattern.trim();
    if (!trimmed.startsWith('{')) return true;
    try {
      const parsed = JSON.parse(trimmed);
      return !isTriggerPatternConfig(parsed);
    } catch (err) {
      // Per `coding-policy: error-handling`: catch only the typed
      // exception we expect (`SyntaxError` from JSON.parse on a
      // legacy literal that happens to start with `{`). Any other
      // throw shape — `TypeError`, OOM during parse of an absurd
      // input, an instrumentation error — is a real defect and
      // must propagate so a startup migration doesn't silently
      // mark ALL rows as legacy and rewrite them.
      if (err instanceof SyntaxError) return true;
      throw err;
    }
  });
  if (legacyTriggerRows.length > 0) {
    const updateStmt = database.prepare(
      `UPDATE registered_groups SET trigger_pattern = ? WHERE jid = ?`,
    );
    database.transaction(() => {
      for (const row of legacyTriggerRows) {
        // Shape-classify the legacy string so #82's per-`kind`
        // self-improvement loop sees the right category from day one.
        // `@<word>` (with bare ASCII identifier chars) is a `mention`,
        // stored without the leading `@` to match the
        // mention-matcher's bare-pattern convention. Everything else
        // (loose strings like `nanoclaw`, oddly-formatted handles
        // like `@bot-with-dash`, free-form keywords) stays as
        // `keyword` and is stored trimmed (the `updateGroupTrigger`
        // helper trims on write, so backfill matches that contract;
        // surrounding whitespace would round-trip awkwardly).
        const trimmed = row.trigger_pattern.trim();
        const mentionMatch = /^@([a-zA-Z0-9_]+)$/.exec(trimmed);
        const config: TriggerPatternConfig = mentionMatch
          ? {
              version: 1,
              patterns: [
                {
                  pattern: mentionMatch[1],
                  kind: 'mention',
                  source: 'owner-set',
                  precision: 0,
                  sample_count: 0,
                  last_matched_at: null,
                  last_updated_at: null,
                },
              ],
            }
          : {
              version: 1,
              patterns: [
                {
                  pattern: trimmed,
                  kind: 'keyword',
                  source: 'owner-set',
                  precision: 0,
                  sample_count: 0,
                  last_matched_at: null,
                  last_updated_at: null,
                },
              ],
            };
        updateStmt.run(JSON.stringify(config), row.jid);
      }
    })();
    logger.info(
      { count: legacyTriggerRows.length },
      'registered_groups: backfilled legacy trigger_pattern rows to JSON schema (#81)',
    );
  }

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

  setDbHandle(new Database(dbPath));
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

  // Migrate from JSON files if they exist. The one-shot JSON→SQL
  // importers live in `db-json-migrations.ts` (#751); the connection
  // handle is passed explicitly so this module keeps single-connection
  // ownership.
  migrateJsonState(db);
}

// One-shot JSON→SQL importers moved to `db-json-migrations.ts` (#751
// seam 1). Re-exported so existing importers (tests) keep working.
export { firstNonEmpty } from './db-json-migrations.js';

/** @internal - for tests only. Creates a fresh in-memory database. */
export function _initTestDatabase(): void {
  setDbHandle(new Database(':memory:'));
  createSchema(db);
  applyStateMigrations(db, STATE_MIGRATIONS);
}

/**
 * @internal - for tests only.
 *
 * Re-runs `createSchema` against a caller-supplied database handle so a
 * test can construct a legacy-shape DB by hand (e.g. `task_run_logs`
 * with the pre-#530 NO-ACTION FK) and assert that the migration block
 * upgrades the shape on the next initialisation. Tests use this in
 * preference to round-tripping through `_initTestDatabase` because they
 * need the seam between "legacy schema present" and "createSchema
 * runs" to be observable.
 */
export function _runCreateSchemaForTests(database: Database.Database): void {
  createSchema(database);
}

/** @internal - for tests only. */
export function _closeDatabase(): void {
  db.close();
}

/**
 * @internal - for tests only.
 *
 * Lets tests read arbitrary rows back through the module-internal db
 * handle without exposing a getter for every table. Used by the #496
 * task-scheduler tests to verify `task_run_logs` rows landed with the
 * expected `status` ('success' / 'killed') after a simulated
 * force-close. Production code reads through the typed accessors above
 * — this exists purely so tests can assert outcomes against
 * write-only sinks like `task_run_logs` without proliferating
 * single-purpose readers.
 */
export function _rawQueryForTests<T>(sql: string, params: unknown[] = []): T[] {
  return db.prepare(sql).all(...params) as T[];
}

/**
 * @internal - for tests only.
 *
 * Sibling of `_rawQueryForTests` for non-query SQL (DELETE / INSERT /
 * UPDATE / DDL) so a test can drive the module-internal `db` handle
 * without the caller having to construct a separate connection.
 */
export function _execRawForTests(sql: string, params: unknown[] = []): void {
  db.prepare(sql).run(...params);
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

// Router-state accessors moved to `db-router-state.ts` (#751 seam 2).
// Re-exported so existing importers keep working.
export { getRouterState, setRouterState } from './db-router-state.js';

// Session accessors moved to `db-sessions.ts` (#751 seam 5).
// Re-exported so existing importers keep working.
export {
  getSession,
  setSession,
  deleteSession,
  deleteSessionName,
  deleteAllSessions,
  getAllSessions,
} from './db-sessions.js';

// Session-length cap state accessors moved to
// `db-session-length-cap.ts` (#751 seam 3). Re-exported so existing
// importers keep working.
export {
  type SessionLengthRow,
  type SessionResetReason,
  type PendingReset,
  getSessionLengthState,
  recordSessionTurn,
  markSessionForReset,
  consumeSessionReset,
  clearSessionLengthStateForGroup,
} from './db-session-length-cap.js';

// Registered-group accessors moved to `db-registered-groups.ts` (#751
// seam 4). Re-exported so existing importers keep working.
export {
  type TriggerPattern,
  type TriggerPatternConfig,
  deriveTriggerString,
  getTriggerPatterns,
  setTriggerPatterns,
  getRegisteredGroup,
  setRegisteredGroup,
  updateGroupTrusted,
  updateGroupTrigger,
  deleteRegisteredGroup,
  getAllRegisteredGroups,
} from './db-registered-groups.js';

// Messages, chats, and reactions accessors moved to `db-messages.ts`
// (#751 seam 6). Re-exported so existing importers keep working.
export {
  type ChatInfo,
  storeChatMetadata,
  updateChatName,
  getAllChats,
  getChatByJid,
  getLastGroupSync,
  setLastGroupSync,
  shouldStoreBotMessage,
  storeMessage,
  storeMessageDirect,
  getMessageById,
  messageExistsInDifferentChat,
  getBotMessageByTelegramId,
  storeReaction,
  getReactionsForMessage,
  getLatestMessage,
  getNewMessages,
  getMessagesSince,
  getLastBotMessageTimestamp,
  getLastFromMeMessage,
  getLastFromMeMessages,
} from './db-messages.js';

// Scheduled-task and task-run-log accessors moved to `db-tasks.ts`
// (#751 seam 7). Re-exported so existing importers keep working.
export {
  createTask,
  rebuildCadenceRegistryForGroup,
  getTaskById,
  getTasksForGroup,
  getAllTasks,
  getActiveLocalScheduledTasks,
  setTaskNextRun,
  updateTask,
  setTaskAgentModel,
  deleteTask,
  setTaskSessionId,
  clearTaskSessionId,
  clearTaskSessionIdsForGroup,
  pruneCompletedTasks,
  resurrectZombieTasks,
  getDormantRecurringTasks,
  getDueTasks,
  updateTaskAfterRun,
  logTaskRun,
} from './db-tasks.js';

// Timezone and location accessors moved to `db-tz.ts` (#751 seam 8).
// Re-exported so existing importers keep working.
export {
  type TripitSegment,
  type TzAdvisoryResult,
  type TzStateForContext,
  SUPPORTED_TZ_STATE_SCHEMA_VERSION,
  storeLocation,
  getLatestLocationForSender,
  getCurrentTz,
  walkTzSegments,
  applyTripitSegmentsToTzState,
  readTzStateForContext,
  runTzHeartbeatAdvisory,
  clearStalePendingRunAt,
  getActivePendingRunAtNames,
  _seedFollowMeTaskForTests,
  _seedTzStateForTests,
} from './db-tz.js';

// Smart-home event accessors moved to `db-smart-home.ts` (#751 seam 9).
// Re-exported so existing importers keep working.
export {
  type SmartHomeEvent,
  insertSmartHomeEvent,
  getSmartHomeEventsSince,
  getSmartHomeEventsByHour,
  cleanupOldSmartHomeEvents,
  getLatestDeviceStates,
} from './db-smart-home.js';
