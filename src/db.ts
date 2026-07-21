import Database, { SqliteError } from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

import {
  rebuildCadenceRegistry,
  type CadenceRegistryDeps,
  type CadenceRegistryRebuildResult,
} from './cadence-registry.js';
import { ASSISTANT_NAME, STORE_DIR } from './config.js';
import { db, setDbHandle } from './db-connection.js';
import { migrateJsonState } from './db-json-migrations.js';
import { isTriggerPatternConfig } from './db-registered-groups.js';
import { logger } from './logger.js';
import { STATE_MIGRATIONS } from './state-migrations/index.js';
import { resolveCurrentTz, STALE_WARNING_HOURS } from './tz-resolver.js';
import {
  LocationRecord,
  NewMessage,
  ScheduledTask,
  TaskRunLog,
  TriggerPatternConfig,
} from './types.js';

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
 * #584 — SQLite error codes the `onTzFlipped` callback catches as
 * recoverable contention. SQLITE_BUSY / SQLITE_LOCKED can fire under
 * WAL contention with the orchestrator's other writers; the canonical
 * `tz_state` UPDATE has already landed in the same transaction, and
 * the next scheduler tick will retry the recompute against the
 * stored `current_tz`. Every other error (programming bug, persistent
 * DB failure, malformed schema) propagates. Mirrors the same set
 * used in `src/index.ts` around `runTzHeartbeatAdvisory`.
 */
const TRANSIENT_SQLITE_CODES: ReadonlySet<string> = new Set([
  'SQLITE_BUSY',
  'SQLITE_LOCKED',
]);

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
 * Decide whether to record a `bot-…` row in `messages.db` after a
 * send dispatches. For Telegram we MUST have a Telegram-native message
 * id back from the channel — its absence is the only reliable signal
 * that the send was swallowed (400 from a bad reply_to, network blip,
 * malformed HTML even after the plain-text fallback, blocked-by-user,
 * rate-limit, etc.). A row written without that id is a phantom: the
 * heartbeat / unanswered-cron treats it as evidence of a reply on a
 * chat the user never received anything in, and downstream agents
 * quote-reply to a message id Telegram has no record of.
 *
 * Non-Telegram channels are not gated — their `Channel.sendMessage`
 * contract permits returning `void` on success (see `src/types.ts`),
 * so absence of an id isn't a failure signal there. Until those
 * channels grow their own success-id surface, the gate would punish
 * a passing send.
 *
 * The undefined check is `!== undefined` rather than truthiness on
 * purpose, matching the comment on `sentMsgId` upstream: a future
 * Telegram id of `''` or `'0'` (we don't expect this today, but the
 * contract is `string | undefined`) must still record the row.
 *
 * Shared by every bot-row write site — IPC `send_message` /
 * `send_message_to_chat` handlers (`src/ipc.ts`), the inbound send
 * path (`src/index.ts`), and the scheduled-task forward
 * (`src/task-scheduler.ts`) — so all of them apply the identical gate.
 *
 * `@internal` — not part of the public API surface; `stripInternal:
 * true` keeps it out of the published `.d.ts`. It is exercised directly
 * by unit tests AND called by the production write sites listed above —
 * not a test-only export.
 */
export function shouldStoreBotMessage(
  chatJid: string,
  sentMsgId: string | undefined,
): boolean {
  const isTelegram = chatJid.startsWith('tg:');
  if (!isTelegram) return true;
  return sentMsgId !== undefined;
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
 * Append a location row (#574 Phase 3). Always INSERT, never UPDATE —
 * live-location updates carry the same `message_id` as the initial
 * share, but each tick is a fresh observation worth persisting (the
 * Phase 2 resolver's "most-recent wins" rule operates on `recorded_at`,
 * not on per-message_id state).
 *
 * Callers don't need to deduplicate; Telegram itself only fires
 * `edited_message:location` when the location actually changed, so
 * the natural rate is bounded by movement, not by polling cadence.
 */
export function storeLocation(record: LocationRecord): void {
  db.prepare(
    `INSERT INTO locations (chat_jid, sender, message_id, latitude, longitude, accuracy_m, source, recorded_at, live_period) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    record.chat_jid,
    record.sender,
    record.message_id,
    record.latitude,
    record.longitude,
    record.accuracy_m ?? null,
    record.source,
    record.recorded_at,
    record.live_period ?? null,
  );
}

/**
 * Most-recent location for a given sender across every chat. Used by
 * the #574 Phase 2 TZ resolver as the canonical "where is the owner"
 * query — `sender` is the owner's channel-specific user id (Telegram
 * numeric id; the orchestrator resolves it once via
 * `ASSISTANT_OWNER_TG_USER_ID`).
 *
 * Returns `null` when no rows match (first deploy, owner-id wrong,
 * etc.) — caller falls through to the TripIt walker per the Phase 2
 * cascade.
 */
export function getLatestLocationForSender(
  sender: string,
): LocationRecord | null {
  // Stable tie-breaker on `id DESC` matters because `recorded_at` is
  // second-resolution (Telegram `date` / `edit_date` are both Unix
  // timestamps in seconds). When two ticks land in the same second
  // — common during a fast-moving live share — sorting only by
  // recorded_at gives SQLite implementation-defined ordering and the
  // resolver can flap between two coords for the same instant. `id`
  // is monotonic INTEGER PRIMARY KEY AUTOINCREMENT, so the composite
  // sort is deterministic; the `idx_locations_sender_time` index
  // still satisfies the leading columns and `id DESC` is a small
  // per-group sort after the index scan.
  const row = db
    .prepare(
      `SELECT chat_jid, sender, message_id, latitude, longitude, accuracy_m, source, recorded_at, live_period
       FROM locations WHERE sender = ?
       ORDER BY recorded_at DESC, id DESC LIMIT 1`,
    )
    .get(sender) as
    | {
        chat_jid: string;
        sender: string;
        message_id: string;
        latitude: number;
        longitude: number;
        accuracy_m: number | null;
        source: string;
        recorded_at: string;
        live_period: number | null;
      }
    | undefined;
  if (!row) return null;
  return {
    chat_jid: row.chat_jid,
    sender: row.sender,
    message_id: row.message_id,
    latitude: row.latitude,
    longitude: row.longitude,
    accuracy_m: row.accuracy_m,
    source: row.source as LocationRecord['source'],
    recorded_at: row.recorded_at,
    live_period: row.live_period,
  };
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
 *
 * Two-pass lookup. Inbound messages and non-Telegram bot sends store
 * the platform-native id directly in the `id` column — that's the
 * fast path. Telegram bot sends are different: the `id` column holds
 * a synthetic `bot-<ts>-<rand>` (#80) and the platform-native id
 * lives in `telegram_message_id`. Without the fallback,
 * `reply_to_message_id` lookups against bot-emitted messages miss,
 * which silently breaks `replyTo.isAssistant` in the gate context
 * (the trigger gate's `reply:*` matcher denies the message) and the
 * cross-chat `safeReplyToForChat` guard in `channels/telegram.ts`
 * (drops legitimate same-chat reply threading). Concrete repro:
 * `tg:-1001633120997` msg 378306, a reply to bot row id
 * `bot-1777747764042-6tppb` (telegram_message_id `378305`), denied
 * by the trigger gate with "no trigger pattern matched" because the
 * id-only lookup couldn't see it was a reply to the assistant.
 */
export function getMessageById(
  messageId: string,
  chatJid: string,
): NewMessage | null {
  const row = db
    .prepare(
      `SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me, is_bot_message
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
        is_bot_message: number | null;
      }
    | undefined;
  if (row) {
    return {
      id: row.id,
      chat_jid: row.chat_jid,
      sender: row.sender,
      sender_name: row.sender_name,
      content: row.content,
      timestamp: row.timestamp,
      is_from_me: row.is_from_me === 1,
      is_bot_message: row.is_bot_message === 1,
    };
  }
  return getBotMessageByTelegramId(chatJid, messageId);
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
): Array<{
  reactor_jid: string;
  reactor_name: string;
  emoji: string;
  timestamp: string;
}> {
  return db
    .prepare(
      `SELECT reactor_jid, reactor_name, emoji, timestamp FROM reactions
       WHERE message_id = ? AND message_chat_jid = ?
       ORDER BY timestamp`,
    )
    .all(messageId, chatJid) as Array<{
    reactor_jid: string;
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
    INSERT INTO scheduled_tasks (id, group_folder, chat_jid, prompt, script, schedule_type, schedule_value, schedule_timezone, context_mode, next_run, status, created_at, created_by_role, continuation_cycle_id, agent_model)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    task.agent_model || null,
  );
}

/**
 * Phase 2 of #305 — idempotent rebuild of cadence-registry rows for one
 * group, called from the container-spawn flow after the per-session
 * skills tree has been published to disk. The cadence-registry module
 * itself is db-handle-agnostic (it accepts the db as a dep, which keeps
 * its unit tests free of singleton state); this wrapper plumbs the
 * module-private `db` handle through so callers don't reach into
 * internals just to invoke the rebuild.
 *
 * Upsert-shaped: a second call with the same SKILL.md frontmatter
 * leaves existing rows alone (preserving `next_run`, `session_id`,
 * `last_run`, `last_result`); only declarations whose cadence string
 * or prompt changed are rewritten, only orphaned rows are deleted.
 * Owner-scheduled rows (`source = 'schedule-task'`) are untouched —
 * the rebuild keys on `source = 'cadence-registry'`.
 *
 * Throws explicitly when called before `initDatabase` so tests that
 * simulate the spawn path without initialising the DB get an
 * actionable error instead of `Cannot read properties of undefined
 * (reading 'transaction')` deep inside the cadence-registry. Per
 * `coding-policy: error-handling`, an unexpected initialisation
 * failure must propagate, not be papered over with a synthetic
 * success — production paths always init before spawn is reachable
 * so this branch never fires there; a test that hits it should
 * `vi.mock('./db.js', ...)` the wrapper alongside its other module
 * mocks.
 */
export function rebuildCadenceRegistryForGroup(
  opts: Omit<CadenceRegistryDeps, 'db'>,
): CadenceRegistryRebuildResult {
  if (!db) {
    throw new Error(
      `rebuildCadenceRegistryForGroup called before initDatabase (groupFolder=${opts.groupFolder}). ` +
        `Production always invokes initDatabase() in src/index.ts startup before runContainerAgent is reachable; ` +
        `if you're seeing this in a test, mock ./db.js's rebuildCadenceRegistryForGroup alongside the test's other module mocks.`,
    );
  }
  return rebuildCadenceRegistry({ ...opts, db });
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

/**
 * #584 — Cron rows scheduled with `schedule_timezone = 'local'`
 * resolve their effective tz at fire time via `tz_state.current_tz`.
 * When `current_tz` flips mid-slot (operator moves zones), already-
 * cached `next_run` values stay anchored to the prior zone until the
 * row fires. The scheduler hooks `tz_state` writers via
 * `recomputeLocalSchedules` (see `task-scheduler.ts`) and uses this
 * selector to find the affected rows. Narrow filter (`'local'` +
 * `'active'`) so a flip doesn't recompute the world.
 */
export function getActiveLocalScheduledTasks(): ScheduledTask[] {
  return db
    .prepare(
      `SELECT * FROM scheduled_tasks
        WHERE schedule_timezone = 'local'
          AND status = 'active'`,
    )
    .all() as ScheduledTask[];
}

/**
 * #584 — Focused next_run writer for `recomputeLocalSchedules`. The
 * caller has just computed a fresh `next_run` for a `'local'`-scheduled
 * row against the new `current_tz`; this writes ONLY `next_run` and
 * leaves every other field alone (status, last_run, last_result,
 * schedule_*). Distinct from `updateTaskAfterRun` (which also bumps
 * `last_run` / `last_result` / status) and `updateTask` (which is the
 * general-purpose multi-field updater used for user-facing IPC writes).
 */
export function setTaskNextRun(id: string, nextRun: string | null): void {
  db.prepare('UPDATE scheduled_tasks SET next_run = ? WHERE id = ?').run(
    nextRun,
    id,
  );
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
      | 'agent_model'
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
  if (updates.agent_model !== undefined) {
    // updates.agent_model === null is a legitimate clear of the
    // per-task override (fall back to the Phase 2 ladder). Distinguish
    // it from `undefined` (caller didn't touch this field) so passing
    // `null` writes the column even though falsy.
    fields.push('agent_model = ?');
    values.push(updates.agent_model);
  }

  if (fields.length === 0) return;

  values.push(id);
  db.prepare(
    `UPDATE scheduled_tasks SET ${fields.join(', ')} WHERE id = ?`,
  ).run(...values);
}

/**
 * Set or clear the per-task AGENT_MODEL override for #509 Phase 3.
 * Thin wrapper over `updateTask` exposed as a named function so the
 * IPC handler (`set_task_agent_model` in src/ipc.ts) and the operator-
 * facing helper share one writer. Pass `null` to clear the override
 * (fall back to the Phase 2 ladder); pass a non-empty string to
 * install one. The string is NOT validated here — `resolvePerGroupAgentModel`
 * at spawn time does the prefix check and falls back if it doesn't
 * recognise the shape, so a typo never silently routes to the global
 * default without an audit-log warning. Returns whether the row
 * existed (so the IPC handler can distinguish "no-op" from "task not
 * found" without a second SELECT).
 */
export function setTaskAgentModel(
  id: string,
  agentModel: string | null,
): boolean {
  const row = db
    .prepare('SELECT 1 FROM scheduled_tasks WHERE id = ?')
    .get(id) as { 1: number } | undefined;
  if (!row) return false;
  db.prepare('UPDATE scheduled_tasks SET agent_model = ? WHERE id = ?').run(
    agentModel,
    id,
  );
  return true;
}

export function deleteTask(id: string): void {
  // Delete child records first (FK constraint)
  db.prepare('DELETE FROM task_run_logs WHERE task_id = ?').run(id);
  db.prepare('DELETE FROM scheduled_tasks WHERE id = ?').run(id);
}

/**
 * Persist the per-task SDK session id (#336). Called from `runTask`
 * when the SDK reports `newSessionId` on a recurring task fire so the
 * next fire can pass it as `resume:` and avoid rebuilding the
 * message-history prefix from scratch.
 *
 * Idempotent: re-writing the same id is a no-op at the row level
 * (UPDATE matching the existing value). The caller doesn't have to
 * de-duplicate streamed `newSessionId` events — the SDK can re-issue
 * the id mid-run, in which case "last write wins" is the right
 * semantic (the latest id is the live transcript on disk).
 *
 * No status guard: the caller decides eligibility (recurring vs
 * once-task, paused vs active). This helper only writes.
 *
 * `pluginsHash` (#710) records the plugin-registry content hash the
 * session was created against, written in the same UPDATE so id and
 * hash can never drift apart. NULL means the hash was unknowable at
 * persist time — registry absent, or it vanished mid-walk during a
 * registry swap (see `hashDirectoryTree`). At fire time a NULL stored
 * here mismatches any KNOWN registry hash and rotates (the pre-#710
 * row path); when the CURRENT hash is null the scheduler skips
 * rotation entirely, so registry-less installs keep resuming.
 */
export function setTaskSessionId(
  id: string,
  sessionId: string,
  pluginsHash: string | null = null,
): void {
  db.prepare(
    `UPDATE scheduled_tasks SET session_id = ?, session_plugins_hash = ?
     WHERE id = ?`,
  ).run(sessionId, pluginsHash, id);
}

/**
 * Clear the per-task SDK session id (#336). Used when the SDK-reported
 * id rotated mid-run (the previous id's transcript is now stale and
 * gets wiped from disk separately) or when the caller wants to force
 * the next fire to start fresh without nuking the whole maintenance
 * slot.
 */
export function clearTaskSessionId(id: string): void {
  db.prepare(
    `UPDATE scheduled_tasks
     SET session_id = NULL, session_plugins_hash = NULL WHERE id = ?`,
  ).run(id);
}

/**
 * Clear the per-task SDK session id for every scheduled task in a
 * group (#336). Called from `nukeSession` when the maintenance (or
 * 'all') slot for a group is wiped so the on-disk JSONL transcripts
 * disappear — without this the next fire would try to `resume:` an id
 * whose transcript no longer exists and the SDK would 404 / start
 * fresh anyway, just noisily. Returns the number of rows touched so
 * the caller can log the wipe scope alongside the JSONL count.
 */
export function clearTaskSessionIdsForGroup(groupFolder: string): number {
  const result = db
    .prepare(
      `UPDATE scheduled_tasks
       SET session_id = NULL, session_plugins_hash = NULL
       WHERE group_folder = ? AND session_id IS NOT NULL`,
    )
    .run(groupFolder);
  return result.changes;
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

// Highest tz_state schema_version this reader knows how to interpret.
// Per `coding-policy: stateful-artifacts`, readers that observe a higher
// `schema_version` must treat the row as "no usable prior state" rather
// than guess. Bumped to 2 by state-012 (#542) when the host took over
// as the writer of `tz_state` and added the `segments` column; bumped
// to 3 by state-013 (jbaruch/nanoclaw-admin#229) when `walkTzSegments`
// gained per-segment ISO-datetime resolution (consuming the upstream
// `reclaim-tripit-timezones-sync#13` parser fix); bumped to 4 by
// state-015 (#574 Phase 2) when the row gained
// `last_stale_warning_at` for the stale-location warning cooldown.
// The state-NNN migrations run before this gate is consulted, so any
// row that existed at a prior version has already been bumped by the
// time `getCurrentTz` reads.
// Exported for the tz-state JSON importer in `db-json-migrations.ts` (#751).
export const SUPPORTED_TZ_STATE_SCHEMA_VERSION = 4;

/**
 * Test-only helper: seed a `follow_me_tasks` row directly. The
 * production writer is the agent-side `task-tz-sync` skill (and the
 * other follow-me skills' Phase C / Phase D updates); this shortcut
 * lets the host-side `clearStalePendingRunAt` /
 * `getActivePendingRunAtNames` tests exercise the cleanup helpers
 * without spinning up the full agent stack.
 */
export function _seedFollowMeTaskForTests(args: {
  name: string;
  localTime?: string;
  scheduleValue?: string;
  lastRunDate?: string | null;
  pendingRunAt?: string | null;
}): void {
  db.prepare(
    `INSERT INTO follow_me_tasks
       (name, local_time, schedule_value, last_run_date, pending_run_at)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(name) DO UPDATE SET
       local_time     = excluded.local_time,
       schedule_value = excluded.schedule_value,
       last_run_date  = excluded.last_run_date,
       pending_run_at = excluded.pending_run_at,
       updated_at     = CURRENT_TIMESTAMP`,
  ).run(
    args.name,
    args.localTime ?? '08:00',
    args.scheduleValue ?? '0 13 * * *',
    args.lastRunDate ?? null,
    args.pendingRunAt ?? null,
  );
}

/**
 * Test-only helper: seed the singleton `tz_state` row directly. The
 * production writer is the host-side `applyTripitSegmentsToTzState`
 * (after every `sync_tripit` run, see #542); this shortcut lets
 * `getCurrentTz` and the heartbeat-advisory walker tests exercise
 * read paths without spinning up the full TripIt sync.
 *
 * `segments` defaults to NULL because most read-path tests only care
 * about `current_tz`. Tests that exercise the heartbeat-advisory
 * walker opt in by passing a JSON-stringified payload. The default
 * `schemaVersion` matches `SUPPORTED_TZ_STATE_SCHEMA_VERSION` so
 * readers don't reject the seeded row as unfamiliar; tests that need
 * to verify the gate's "unfamiliar version" branch pass an explicit
 * higher value.
 */
export function _seedTzStateForTests(args: {
  currentTz: string;
  homeTz?: string;
  schedulerTz?: string | null;
  segments?: string | null;
  schemaVersion?: number;
  // ISO-8601 UTC; null clears the stamp (the resolver treats null as
  // "cooldown expired, may fire on next stale check"). Defaults to
  // null so existing tests that don't care about cooldown behave
  // as if no warning has ever fired.
  lastStaleWarningAt?: string | null;
}): void {
  db.prepare(
    `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, segments, schema_version, last_stale_warning_at)
       VALUES (1, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       current_tz            = excluded.current_tz,
       home_tz               = excluded.home_tz,
       scheduler_tz          = excluded.scheduler_tz,
       segments              = excluded.segments,
       schema_version        = excluded.schema_version,
       last_stale_warning_at = excluded.last_stale_warning_at`,
  ).run(
    args.currentTz,
    args.homeTz ?? args.currentTz,
    args.schedulerTz ?? null,
    args.segments ?? null,
    args.schemaVersion ?? SUPPORTED_TZ_STATE_SCHEMA_VERSION,
    args.lastStaleWarningAt ?? null,
  );
}

/**
 * Read `current_tz` from the singleton `tz_state` row. Returns null if
 * the row is absent (pre-state-010 install / migration not yet run) or
 * if `schema_version` is unfamiliar to this reader.
 *
 * Used by `task-scheduler.ts:computeNextRunDetailed` to resolve rows
 * declared with `schedule_timezone = 'local'` (#456): the cron is
 * evaluated against the owner's current zone at fire time without
 * mutating the row's `schedule_value`.
 */
export function getCurrentTz(): string | null {
  const row = db
    .prepare('SELECT current_tz, schema_version FROM tz_state WHERE id = 1')
    .get() as { current_tz: string; schema_version: number } | undefined;
  if (!row) return null;
  if (row.schema_version !== SUPPORTED_TZ_STATE_SCHEMA_VERSION) {
    logger.warn(
      {
        observed: row.schema_version,
        supported: SUPPORTED_TZ_STATE_SCHEMA_VERSION,
      },
      'tz_state schema_version unfamiliar — treating as no usable prior state',
    );
    return null;
  }
  return row.current_tz;
}

/**
 * Single segment shape as emitted by `sync_tripit`'s stdout JSON
 * (`result.segments` in `reclaim-tripit-timezones-sync/sync.mjs`).
 *
 * `from` / `to` are date-only `YYYY-MM-DD` strings — the original
 * shape from before reclaim-tripit-timezones-sync#13. Lexicographic
 * comparison of these strings is equivalent to date comparison, so
 * the walker can use plain string `<=`/`<` when datetime fields are
 * absent.
 *
 * `from_dt` / `to_dt` (jbaruch/nanoclaw-admin#229,
 * reclaim-tripit-timezones-sync#13) are ISO 8601 UTC strings derived
 * from the same underlying `Date` objects in `lib/tripit.mjs` — they
 * preserve flight arrival / lodging check-in wall-clock instead of
 * collapsing to UTC midnight. Optional because (a) a row written by
 * a pre-deploy `applyTripitSegmentsToTzState` carries the old shape
 * until the next `sync_tripit` rewrites it, and (b) the upstream
 * parser may legitimately omit them on edge paths.
 */
export interface TripitSegment {
  timezone: string;
  from: string;
  to: string;
  from_dt?: string;
  to_dt?: string;
  label?: string;
}

/**
 * Pure walker: pick the IANA zone name covering `now` from a segments
 * timeline, falling back to `homeTz` if no segment covers it.
 *
 * Match rule: per-segment, prefer ISO-datetime resolution when the
 * segment carries `from_dt` / `to_dt` (post-#229); fall through to
 * date-only `from` / `to` otherwise. The match keeps strict
 * inequality on the right edge in both shapes — a traveler whose
 * return-flight arrival is `to_dt` should see the next segment (or
 * `homeTz`) take over the instant the flight lands, not hold the
 * destination zone for the rest of that second.
 *
 * Why per-segment, not array-wide: a single segments array can mix
 * shapes across deploys — a row written by a pre-deploy
 * `applyTripitSegmentsToTzState` is date-only until the next
 * `sync_tripit` rewrites it; in the transient between the upstream
 * parser bump (rtts#13) and the next sync, an array can also carry
 * one shape exclusively. Per-segment fallback handles every case
 * with one walk.
 *
 * `from_dt` and `to_dt` are both required for the datetime path —
 * if only one is present (malformed payload), the segment falls
 * back to date-only on its own row rather than the walker emitting
 * a one-sided datetime compare against the missing end.
 *
 * String compare on ISO 8601 UTC strings is equivalent to chrono
 * compare, the same property the date-only path relies on. Both
 * shapes use `now.toISOString()` (`...T...Z`) or its 10-char prefix
 * for `today`, so each path is internally consistent.
 *
 * Edge cases (covered by unit tests):
 *   - Empty / non-array `segments` → `homeTz` (silent fallback).
 *   - Inter-segment gap, transit (the most-recent prev-ended segment
 *     is NOT `homeTz`) → the ARRIVAL segment's tz (#571). Segments
 *     are built from lodging / ground stays; flight legs are gaps by
 *     design. When the heartbeat fires mid-flight between two foreign
 *     stays, the user is travelling TOWARDS the next booked stay, so
 *     returning that segment's tz lets the morning brief / scheduler
 *     think in the arrival zone the user is about to land in.
 *     `home_tz` was wrong for this case: it's often physically
 *     impossible (mid-Atlantic on a Europe-bound leg) and the user
 *     can't act on a "you're home" signal while in the air.
 *   - Inter-segment gap, home-bounded (the most-recent prev-ended
 *     segment IS `homeTz`) → `homeTz` (#573). The #571 arrival-tz
 *     rule over-corrected this case: an inbound home segment IS the
 *     "you're home now" signal, so the gap that follows is the user
 *     sitting at home between trips, not in motion. Without this
 *     branch the walker ships a forward-looking foreign tz for the
 *     entire dwell-at-home window. Detection uses the MOST RECENT
 *     prev-ended segment's tz (chronological-order invariant: the
 *     last segment that classifies as prev-ended in iteration order
 *     wins, mirroring the first-wins rule the arrival branch uses).
 *     The symmetric case (`nextSegTz === homeTz` — foreign trip
 *     followed by a future home segment that hasn't started yet)
 *     needs no special branch: the arrival-tz fallback already
 *     returns `homeTz` because `nextSegTz` IS `homeTz`.
 *   - Before the first segment (all segments are future) → `homeTz`.
 *   - After the last segment (no future segment remaining) →
 *     `homeTz`. This is the post-trip case: the user has returned
 *     and the next `sync_tripit` will eventually clear the row, but
 *     until then home is the right answer.
 *   - `from === to` (date or datetime) → never matches (degenerate
 *     segment); skipped.
 *   - Empty / non-string `timezone` → segment skipped (the upstream
 *     drops these via `result.segments` already, but defend against
 *     a malformed payload anyway).
 *   - Overlapping segments → first match wins (matches the natural
 *     "segments are produced in chronological order, lodging-primary
 *     first" upstream invariant).
 *   - `to_dt === now.toISOString()` (or `to === todayUtc` on the
 *     date-only path) → segment ends now; the next segment (or the
 *     gap / home fallback) takes over.
 *   - Mixed array (some segments with `from_dt`/`to_dt`, some
 *     without) → each segment uses its own shape; the gap-fallback
 *     classification (ended-before-now / starts-after-now) also uses
 *     each segment's own shape against `nowIso` / `todayUtc`.
 */
export function walkTzSegments(
  segments: readonly TripitSegment[] | null | undefined,
  now: Date,
  homeTz: string,
): string {
  if (!Array.isArray(segments) || segments.length === 0) return homeTz;
  const nowIso = now.toISOString();
  const todayUtc = nowIso.slice(0, 10);

  // #571 / #573 — classify each non-covering segment as "ended
  // before now" or "starts after now". When BOTH classes appear in
  // the same walk, the user is in an inter-segment gap; the right
  // answer depends on what the user did last:
  //   - prev-ended segment was a foreign zone → mid-transit toward
  //     the next booked stay → return the arrival (first-future) tz
  //     (#571).
  //   - prev-ended segment was the home zone → user landed at home
  //     and is sitting between trips → return `homeTz` (#573).
  // Both branches rely on the lodging-primary chronological-order
  // invariant: `nextSegTz` keeps the FIRST future segment's tz
  // (first wins), and `prevEndedTz` keeps the LAST prev-ended
  // segment's tz (last wins = most recent under chronological
  // iteration). No cross-shape datetime-vs-date comparison is
  // needed.
  let prevEndedTz: string | null = null;
  let nextSegTz: string | null = null;

  for (const seg of segments) {
    if (
      typeof seg?.timezone !== 'string' ||
      seg.timezone.length === 0 ||
      typeof seg.from !== 'string' ||
      typeof seg.to !== 'string'
    ) {
      continue;
    }
    // Datetime path: both `from_dt` and `to_dt` must be present and
    // strings. Partial-shape segments fall through to the date-only
    // path on their own row.
    if (typeof seg.from_dt === 'string' && typeof seg.to_dt === 'string') {
      // Degenerate (`from_dt === to_dt`) segments are skipped per the
      // function's documented contract — the inside-check already
      // rejects them via `nowIso < seg.to_dt`, but the gap-fallback
      // classifier (#571) would otherwise feed them into
      // `hasPrevEndedSeg` / `nextSegTz` and let a malformed row drive
      // the resolution. Skip BEFORE the gap classifier to keep
      // degenerate segments invisible to every code path here.
      if (seg.from_dt === seg.to_dt) continue;
      if (seg.from_dt <= nowIso && nowIso < seg.to_dt) return seg.timezone;
      // Strict `<` for the prev-ended classification (mirror in the
      // date-only path below): a segment whose `to_dt` exactly
      // equals `now` is at the right-edge boundary and stays out of
      // the gap-fallback's "previous" pool. The mixed-shape case
      // (#229 regression guard) is the load-bearing reason: a
      // legacy date-only segment with `to: 2026-05-12` and a
      // datetime future segment at `from_dt: 2026-05-12T13:30Z`
      // would otherwise let the walker return the future zone at
      // `00:30Z` (departure morning) — same early-flip shape the
      // #229 per-segment-datetime fix existed to prevent.
      if (seg.to_dt < nowIso) {
        prevEndedTz = seg.timezone;
      } else if (nowIso < seg.from_dt) {
        if (nextSegTz === null) nextSegTz = seg.timezone;
      }
      continue;
    }
    // Date-only path — same degenerate-segment guard + strict-`<`
    // right edge for prev-ended classification. A date-only segment
    // whose `to` equals today's UTC date is "ending today" but the
    // user is still in it until UTC midnight rolls; treating it as
    // already-past would flip mid-day for any later segment.
    if (seg.from === seg.to) continue;
    if (seg.from <= todayUtc && todayUtc < seg.to) return seg.timezone;
    if (seg.to < todayUtc) {
      prevEndedTz = seg.timezone;
    } else if (todayUtc < seg.from) {
      if (nextSegTz === null) nextSegTz = seg.timezone;
    }
  }

  // In-gap: previous-ended AND future segment both exist. Distinguish
  // home-bounded gaps (user landed home, sitting between trips) from
  // transit gaps (user is mid-flight toward the next booked stay).
  if (prevEndedTz !== null && nextSegTz !== null) {
    return prevEndedTz === homeTz ? homeTz : nextSegTz;
  }
  // Before-first or after-last: fall back to home_tz.
  return homeTz;
}

/**
 * #542 — Persist `sync_tripit` stdout's `segments[]` onto the
 * singleton `tz_state` row and recompute `current_tz` from the
 * segment timeline (or fall back to `home_tz` for gaps between
 * trips).
 *
 * Writes `schema_version = SUPPORTED_TZ_STATE_SCHEMA_VERSION` (3
 * post-jbaruch/nanoclaw-admin#229; was 2 between #542 and #229).
 * The constant is the source of truth so every writer in this file
 * picks up future state-NNN bumps automatically. The row
 * MUST already exist — `home_tz` is NOT NULL on `tz_state` and isn't
 * derivable from the TripIt payload, so the host can't synthesize a
 * row from scratch. This is by design: `tz_state.home_tz` is set by
 * the historical JSON migration / first-time setup and is never
 * touched by `sync_tripit`. If the row is absent, the helper logs a
 * warning and exits without writing — the caller's run is still
 * counted as a successful TripIt sync (segments were fetched and
 * parsed) so the user gets the upstream success reaction; the
 * follow-up heartbeat advisory will pick up the segments on the next
 * tick once the row exists.
 *
 * Returns the resolved `{ prev, next, changed }` so the caller can
 * audit-log the flip — the `sync_tripit` call site logs at info /
 * warn levels on the orchestrator's structured logger; there is no
 * separate audit-log table per `coding-policy: stateful-artifacts`'s
 * "what counts" section (state files vs. orchestrator logs).
 */
export function applyTripitSegmentsToTzState(
  stdoutJson: { segments?: readonly TripitSegment[] | null } | null,
  now: Date = new Date(),
  onTzFlipped?: (prev: string, next: string) => void,
): { prev: string | null; next: string | null; changed: boolean } {
  const segments =
    stdoutJson && Array.isArray(stdoutJson.segments) ? stdoutJson.segments : [];

  const row = db
    .prepare('SELECT current_tz, home_tz FROM tz_state WHERE id = 1')
    .get() as { current_tz: string; home_tz: string } | undefined;
  if (!row) {
    logger.warn(
      { segmentCount: segments.length },
      'applyTripitSegmentsToTzState: tz_state row missing — skipping (home_tz must be seeded by JSON migration / first-time setup before sync_tripit can persist segments)',
    );
    return { prev: null, next: null, changed: false };
  }

  const next = walkTzSegments(segments, now, row.home_tz);
  const segmentsJson = JSON.stringify(segments);

  // Plain UPDATE rather than UPSERT: the row-existence check above
  // already proved the singleton is present, and an UPSERT's INSERT
  // arm would clobber `scheduler_tz` (which the JSON migration
  // legitimately populates as `informational only`) on a hypothetical
  // concurrent-delete race. Plain UPDATE on a missing row is a
  // no-op which is the safer failure mode.
  //
  // `schema_version` is bound through `SUPPORTED_TZ_STATE_SCHEMA_VERSION`
  // (rather than a SQL literal) so a future state-NNN bump only has to
  // touch the constant — every writer in this file picks up the new
  // value automatically.
  db.prepare(
    `UPDATE tz_state
        SET current_tz     = ?,
            segments       = ?,
            schema_version = ?
      WHERE id = 1`,
  ).run(next, segmentsJson, SUPPORTED_TZ_STATE_SCHEMA_VERSION);

  const changed = row.current_tz !== next;
  if (changed) {
    logger.info(
      { prev: row.current_tz, next, segmentCount: segments.length },
      'tz_state.current_tz flipped via sync_tripit segment walk (#542)',
    );
    // #584 — invoke the recompute hook AFTER the UPDATE landed, so a
    // reader picking up the new current_tz sees the new value. Narrow
    // the catch to transient SQLite contention codes only
    // (SQLITE_BUSY / SQLITE_LOCKED) — those are genuinely recoverable
    // against the orchestrator's other WAL writers and the next
    // scheduler tick will re-anchor `next_run` against the now-canonical
    // `current_tz`. Every other error (programming bug, persistent DB
    // failure, malformed schema) propagates out per
    // `coding-policy: error-handling`. Mirrors the narrowing pattern
    // in `src/index.ts` around `runTzHeartbeatAdvisory`.
    if (onTzFlipped) {
      try {
        onTzFlipped(row.current_tz, next);
      } catch (err) {
        if (
          !(err instanceof SqliteError) ||
          !TRANSIENT_SQLITE_CODES.has(err.code)
        ) {
          throw err;
        }
        logger.warn(
          {
            err: err.message,
            code: err.code,
            prev: row.current_tz,
            next,
          },
          'applyTripitSegmentsToTzState: onTzFlipped transient SQLite contention — tz_state write landed, next scheduler tick will recompute',
        );
      }
    }
  }
  return { prev: row.current_tz, next, changed };
}

/**
 * #542 / #574 Phase 2 — Heartbeat advisory. Re-evaluates `current_tz`
 * against (a) the owner's most-recent location row, falling through to
 * (b) the cached TripIt segments walker. Called from a 30-min
 * `setInterval` in `src/index.ts`; on flip, the caller sends a chat
 * message via the main group's channel. On stale-location warning,
 * the caller sends a "please re-share" notice — bounded by a 12h
 * cooldown stored in `tz_state.last_stale_warning_at`.
 *
 * `ownerSenderId` is the channel-specific owner identifier (Telegram
 * numeric user_id today; the orchestrator resolves it once from
 * `ASSISTANT_OWNER_TG_USER_ID` and threads it in). When null /
 * undefined / unset, the location-first path is skipped and the
 * function falls back to the pre-Phase-2 walker-only behaviour
 * (zero-config installs keep working unchanged).
 *
 * Returns `{ flip, warningToFire }`:
 *   - `flip` carries the `{ prev, next }` zone change when the
 *     computed tz differs from `current_tz`; null otherwise.
 *   - `warningToFire` is `'stale_no_share'` only when the resolver
 *     reports the latest location is ≥12 h old AND the cooldown
 *     window has elapsed since the last warning; null otherwise.
 *     The DB column `last_stale_warning_at` is updated atomically
 *     with the decision so a concurrent advisory can't double-fire.
 *
 * Skip paths (return `{ flip: null, warningToFire: null }`):
 *   - tz_state row missing
 *   - tz_state row at an unfamiliar schema_version (warned + skipped)
 *   - tz_state.segments malformed JSON (warned + skipped; recovers
 *     on next sync_tripit)
 *
 * Lock-step contract with state-015: this function reads / writes
 * `last_stale_warning_at`. A row that hasn't been migrated to v4
 * trips the schema_version gate above and skips, so the column
 * absence is structurally impossible inside the hot path.
 */
export interface TzAdvisoryResult {
  flip: { prev: string; next: string } | null;
  warningToFire: 'stale_no_share' | null;
}

/**
 * Read-only snapshot of `tz_state` for the `<context>`-tag builder
 * in `agent-context.ts`. Returns `home_tz` plus the decoded
 * `segments` array (or `null` when the column is empty / malformed
 * JSON). Does NOT mutate any state, NOT write warnings to the log,
 * and NOT run the resolver — those side effects belong to
 * `runTzHeartbeatAdvisory`, not the per-prompt context builder.
 *
 * Returns `null` when the singleton row is missing entirely (no
 * `tz_state` seeded yet) so the caller can fall through to the
 * container-default context shape per `agent-context.ts`'s "no
 * usable input" early-return.
 */
export interface TzStateForContext {
  home_tz: string;
  segments: readonly TripitSegment[] | null;
}

export function readTzStateForContext(): TzStateForContext | null {
  const row = db
    .prepare(
      'SELECT home_tz, segments, schema_version FROM tz_state WHERE id = 1',
    )
    .get() as
    | {
        home_tz: string;
        segments: string | null;
        schema_version: number;
      }
    | undefined;
  if (!row) return null;
  // Mirror `runTzHeartbeatAdvisory`'s schema-version gate. A row at
  // an unfamiliar version is "no usable prior state" — return null
  // here so the context builder falls back to container_default
  // rather than feeding stale-shape data into the resolver.
  if (row.schema_version !== SUPPORTED_TZ_STATE_SCHEMA_VERSION) return null;

  let segments: readonly TripitSegment[] | null = null;
  if (row.segments) {
    try {
      const decoded = JSON.parse(row.segments) as unknown;
      if (Array.isArray(decoded)) {
        segments = decoded as readonly TripitSegment[];
      }
    } catch (err) {
      if (!(err instanceof SyntaxError)) throw err;
      // Malformed segments JSON: fall back to null. The
      // `runTzHeartbeatAdvisory` walker fires every 30 min and will
      // log this case; no need to re-emit on every agent prompt.
      segments = null;
    }
  }
  return { home_tz: row.home_tz, segments };
}

export function runTzHeartbeatAdvisory(
  now: Date = new Date(),
  ownerSenderId?: string | null,
  onTzFlipped?: (prev: string, next: string) => void,
): TzAdvisoryResult {
  const row = db
    .prepare(
      'SELECT current_tz, home_tz, segments, schema_version, last_stale_warning_at FROM tz_state WHERE id = 1',
    )
    .get() as
    | {
        current_tz: string;
        home_tz: string;
        segments: string | null;
        schema_version: number;
        last_stale_warning_at: string | null;
      }
    | undefined;
  if (!row) return { flip: null, warningToFire: null };
  if (row.schema_version !== SUPPORTED_TZ_STATE_SCHEMA_VERSION) {
    // Same contract as `getCurrentTz` — an unfamiliar version is "no
    // usable prior state". The walker fires every 30 min, so this
    // emits one warn per tick until the operator drops the bad row;
    // there's no log-once dedup. Two tradeoffs in play: (a) a stuck
    // mid-migration row in production is a load-bearing alert that
    // should keep paging until cleared (silencing it would let an
    // operator forget about it for hours), and (b) a 30-min cadence
    // doesn't pollute the structured log meaningfully — twice an
    // hour is far below the heartbeat noise floor on this surface.
    logger.warn(
      {
        observed: row.schema_version,
        supported: SUPPORTED_TZ_STATE_SCHEMA_VERSION,
      },
      'runTzHeartbeatAdvisory: tz_state schema_version unfamiliar — skipping',
    );
    return { flip: null, warningToFire: null };
  }

  let parsed: readonly TripitSegment[] | null = null;
  let segmentsMalformed = false;
  if (row.segments) {
    try {
      const decoded = JSON.parse(row.segments) as unknown;
      if (Array.isArray(decoded)) {
        parsed = decoded as readonly TripitSegment[];
      }
    } catch (err) {
      if (!(err instanceof SyntaxError)) throw err;
      logger.warn(
        { err: err.message },
        'runTzHeartbeatAdvisory: tz_state.segments is malformed JSON — falling through (will recover on next sync_tripit run)',
      );
      segmentsMalformed = true;
    }
  }

  // Read the owner's most-recent location only when the orchestrator
  // told us who the owner is. Empty / null / undefined `ownerSenderId`
  // skips the location-first path entirely (walker-only, pre-Phase-2
  // behaviour).
  const latestLocation = ownerSenderId
    ? getLatestLocationForSender(ownerSenderId)
    : null;

  // Pre-Phase-2 contract preservation: when there's NO usable input
  // for the cascade (no segments OR malformed segments) AND no
  // location row, the advisory was a no-op (`current_tz` untouched,
  // null return). Phase 2's resolver would otherwise call
  // `walkTzSegments(null, ...)` and silently flip `current_tz` to
  // `home_tz`, which is a behavioural change for zero-config installs
  // where `ASSISTANT_OWNER_TG_USER_ID` is unset and the segments
  // cache hasn't been populated yet. Early-return preserves the
  // pre-Phase-2 no-op exactly. NOTE: this guard sits AFTER the
  // location read so an owner with stale-but-existent location data
  // can still drive the cascade through the walker-fallback path
  // (and possibly fire a stale_no_share warning) when segments are
  // broken.
  if (latestLocation === null && (parsed === null || segmentsMalformed)) {
    return { flip: null, warningToFire: null };
  }

  const resolved = resolveCurrentTz({
    now,
    latestLocation,
    segments: parsed,
    home_tz: row.home_tz,
  });

  // Cooldown for the stale-location warning. We fire `stale_no_share`
  // at most once per STALE_WARNING_HOURS window so an owner who travels
  // for a weekend doesn't get the same nag 48 times across 24 h. Reset
  // the cooldown stamp ONLY when the resolver reports
  // `source: 'fresh_location'` — that's the unambiguous signal that
  // the owner has shared again. Resetting on any `warning === null`
  // would erase the cooldown during the 4 h ≤ age < 12 h band (where
  // the location is stale-for-cascade but not yet warning-eligible)
  // and re-fire the nag on the first ≥ 12 h tick after — defeating
  // the cooldown's purpose.
  let warningToFire: 'stale_no_share' | null = null;
  let nextLastWarning: string | null | undefined = undefined;
  if (resolved.warning === 'stale_no_share') {
    const lastWarn = row.last_stale_warning_at
      ? Date.parse(row.last_stale_warning_at)
      : NaN;
    const cooldownExpired =
      !Number.isFinite(lastWarn) ||
      now.getTime() - lastWarn >= STALE_WARNING_HOURS * 60 * 60 * 1000;
    if (cooldownExpired) {
      warningToFire = 'stale_no_share';
      nextLastWarning = now.toISOString();
    }
  } else if (
    resolved.source === 'fresh_location' &&
    row.last_stale_warning_at !== null
  ) {
    // Owner has genuinely shared again — clear the cooldown so the
    // next stale window starts clean.
    nextLastWarning = null;
  }

  // Single UPDATE batches the current_tz flip and the cooldown stamp
  // so a concurrent advisory can't see a half-applied state. SQLite's
  // single-writer model already serialises this, but the batch keeps
  // the read-then-write window tight.
  const flip =
    resolved.tz !== row.current_tz
      ? { prev: row.current_tz, next: resolved.tz }
      : null;

  if (flip !== null || nextLastWarning !== undefined) {
    const setClauses: string[] = [];
    const bindings: (string | null)[] = [];
    if (flip !== null) {
      setClauses.push('current_tz = ?');
      bindings.push(resolved.tz);
    }
    if (nextLastWarning !== undefined) {
      setClauses.push('last_stale_warning_at = ?');
      bindings.push(nextLastWarning);
    }
    db.prepare(`UPDATE tz_state SET ${setClauses.join(', ')} WHERE id = 1`).run(
      ...bindings,
    );
  }

  if (flip) {
    logger.info(
      {
        prev: row.current_tz,
        next: resolved.tz,
        source: resolved.source,
        latest_location_age_seconds: resolved.latest_location_age_seconds,
      },
      'tz_state.current_tz flipped via heartbeat advisory (#574 Phase 2)',
    );
    // #584 — see `applyTripitSegmentsToTzState` for the rationale; same
    // narrowing contract on the heartbeat-advisory writer. Catch only
    // transient SQLite contention (SQLITE_BUSY / SQLITE_LOCKED); every
    // other error propagates so programming bugs / persistent DB
    // failures surface instead of getting swallowed as a warn.
    if (onTzFlipped) {
      try {
        onTzFlipped(flip.prev, flip.next);
      } catch (err) {
        if (
          !(err instanceof SqliteError) ||
          !TRANSIENT_SQLITE_CODES.has(err.code)
        ) {
          throw err;
        }
        logger.warn(
          {
            err: err.message,
            code: err.code,
            prev: flip.prev,
            next: flip.next,
          },
          'runTzHeartbeatAdvisory: onTzFlipped transient SQLite contention — tz_state write landed, next scheduler tick will recompute',
        );
      }
    }
  }

  return { flip, warningToFire };
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

/**
 * Stale-lock recovery for `follow_me_tasks.pending_run_at` (#496).
 *
 * Background: skills running inside agent containers acquire a
 * pending-run lock by setting `pending_run_at` mid-run, then clear it
 * post-run. If the host kills the container mid-run (e.g. the periodic
 * `tessl_update` writes `_close`, the agent-runner watchdog hits its
 * 30s timeout, and the container exits before the post-run clear), the
 * lock is left dangling. Tomorrow's scheduled fire then sees a stale
 * `pending_run_at` and refuses to run on the Phase A gate.
 *
 * This helper is the host-side recovery: any `pending_run_at` older
 * than `maxAgeMs` is treated as orphaned (its owning container is long
 * gone) and cleared. Per `coding-policy: stateful-artifacts`, the host
 * is a NON-OWNER reader of `follow_me_tasks` (the owning skill is
 * `nanoclaw-admin/skills/task-tz-sync`); non-owners must not migrate
 * the schema, but clearing a value field is lock-recovery, not
 * migration — it's the inverse of what the owner skill does on a
 * normal post-run.
 *
 * Returns the names of every task whose `pending_run_at` was cleared
 * so the caller can log the recovery.
 */
export function clearStalePendingRunAt(maxAgeMs: number): string[] {
  if (!Number.isFinite(maxAgeMs) || maxAgeMs <= 0) {
    throw new Error(
      `clearStalePendingRunAt: maxAgeMs must be a positive number (got ${maxAgeMs})`,
    );
  }
  const cutoffIso = new Date(Date.now() - maxAgeMs).toISOString();
  // Two-step (SELECT then UPDATE) so we can return the names of cleared
  // rows for logging. The window between SELECT and UPDATE is racy in
  // principle — a fresh skill could land a NEW `pending_run_at` on the
  // same row between the two — but the UPDATE's WHERE clause re-checks
  // the cutoff, so a row that gained a fresh lock won't be cleared.
  const rows = db
    .prepare(
      `SELECT name FROM follow_me_tasks
        WHERE pending_run_at IS NOT NULL
          AND pending_run_at < ?`,
    )
    .all(cutoffIso) as { name: string }[];
  if (rows.length === 0) return [];
  db.prepare(
    `UPDATE follow_me_tasks
        SET pending_run_at = NULL,
            updated_at     = CURRENT_TIMESTAMP
      WHERE pending_run_at IS NOT NULL
        AND pending_run_at < ?`,
  ).run(cutoffIso);
  return rows.map((r) => r.name);
}

/**
 * Returns the names of every `follow_me_tasks` row with a fresh
 * (within `maxAgeMs`) `pending_run_at` lock. Empty array means no
 * task is currently mid-run from the host's perspective.
 *
 * Used by the periodic `tessl_update` catch-up to skip the
 * session-clear / container-close path while a scheduled task is in
 * flight (#496 mitigation 1). The owning agent containers wouldn't
 * meaningfully observe new tile content until they finish anyway, so
 * deferring is harmless; the next 15-minute tick will retry.
 */
export function getActivePendingRunAtNames(maxAgeMs: number): string[] {
  if (!Number.isFinite(maxAgeMs) || maxAgeMs <= 0) {
    throw new Error(
      `getActivePendingRunAtNames: maxAgeMs must be a positive number (got ${maxAgeMs})`,
    );
  }
  const cutoffIso = new Date(Date.now() - maxAgeMs).toISOString();
  const rows = db
    .prepare(
      `SELECT name FROM follow_me_tasks
        WHERE pending_run_at IS NOT NULL
          AND pending_run_at >= ?`,
    )
    .all(cutoffIso) as { name: string }[];
  return rows.map((r) => r.name);
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
