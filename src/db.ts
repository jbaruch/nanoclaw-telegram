import Database, { SqliteError } from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

import {
  rebuildCadenceRegistry,
  type CadenceRegistryDeps,
  type CadenceRegistryRebuildResult,
} from './cadence-registry.js';
import { ASSISTANT_NAME, DATA_DIR, GROUPS_DIR, STORE_DIR } from './config.js';
import { isValidGroupFolder } from './group-folder.js';
import {
  handleConstraintViolationOrRethrow,
  hasMigratedSibling,
  isObjectRow,
  listGroupFoldersForMigration,
  migrationDateStamp,
  type MigrationSummary,
  newMigrationSummary,
  parseJsonObjectOrWarn,
  renameMigratedSource,
} from './json-state-import.js';
import { logger } from './logger.js';
import { STATE_MIGRATIONS } from './state-migrations/index.js';
import {
  ContainerConfig,
  NewMessage,
  RegisteredGroup,
  ScheduledTask,
  TaskRunLog,
  TriggerPattern,
  TriggerPatternConfig,
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
      -- Row-creation provenance for #305 Phase 2 cadence-registry. See
      -- the ALTER block below for the value set and ownership semantics.
      -- 'schedule-task' is the default so unmigrated callers (the
      -- existing schedule-task IPC path) keep their semantics unchanged.
      source TEXT NOT NULL DEFAULT 'schedule-task'
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

/**
 * Best-effort reverse lookup: given a (sender_jid, chat_jid), return
 * the most recent non-empty `sender_name` we have on file for that
 * sender in that chat. Returns null when nothing is on file (e.g.
 * brand-new sender with no prior message yet).
 *
 * Used by the Stage 2 Haiku classifier (#83) to feed a human-readable
 * display name into the prompt — the GateContext only carries
 * `senderJid`, but Anthropic-grade few-shot prompting works far
 * better with display names than with raw JIDs/numeric ids. Bounded
 * by `chat_jid` so the query stays cheap (no global scan), and
 * ordered by timestamp DESC so display-name renames are picked up.
 */
export function getRecentSenderName(
  senderJid: string,
  chatJid: string,
): string | null {
  const row = db
    .prepare(
      `SELECT sender_name FROM messages
       WHERE sender = ? AND chat_jid = ?
         AND sender_name IS NOT NULL AND LENGTH(sender_name) > 0
       ORDER BY timestamp DESC LIMIT 1`,
    )
    .get(senderJid, chatJid) as { sender_name: string } | undefined;
  return row?.sender_name ?? null;
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
 */
export function setTaskSessionId(id: string, sessionId: string): void {
  db.prepare('UPDATE scheduled_tasks SET session_id = ? WHERE id = ?').run(
    sessionId,
    id,
  );
}

/**
 * Clear the per-task SDK session id (#336). Used when the SDK-reported
 * id rotated mid-run (the previous id's transcript is now stale and
 * gets wiped from disk separately) or when the caller wants to force
 * the next fire to start fresh without nuking the whole maintenance
 * slot.
 */
export function clearTaskSessionId(id: string): void {
  db.prepare('UPDATE scheduled_tasks SET session_id = NULL WHERE id = ?').run(
    id,
  );
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
      `UPDATE scheduled_tasks SET session_id = NULL
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
// `reclaim-tripit-timezones-sync#13` parser fix). The state-NNN
// migrations run before this gate is consulted, so any row that
// existed at a prior version has already been bumped by the time
// `getCurrentTz` reads.
const SUPPORTED_TZ_STATE_SCHEMA_VERSION = 3;

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
}): void {
  db.prepare(
    `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, segments, schema_version)
       VALUES (1, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       current_tz     = excluded.current_tz,
       home_tz        = excluded.home_tz,
       scheduler_tz   = excluded.scheduler_tz,
       segments       = excluded.segments,
       schema_version = excluded.schema_version`,
  ).run(
    args.currentTz,
    args.homeTz ?? args.currentTz,
    args.schedulerTz ?? null,
    args.segments ?? null,
    args.schemaVersion ?? SUPPORTED_TZ_STATE_SCHEMA_VERSION,
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
 *   - Inter-segment gap (transit between two booked stays) → the
 *     ARRIVAL segment's tz (#571). Segments are built from lodging /
 *     ground stays; flight legs are gaps by design. When the heartbeat
 *     fires mid-flight, the user is travelling TOWARDS the next
 *     booked stay, so returning that segment's tz lets the morning
 *     brief / scheduler think in the arrival zone the user is about
 *     to land in. `home_tz` was wrong for this case: it's often
 *     physically impossible (mid-Atlantic on a Europe-bound leg)
 *     and the user can't act on a "you're home" signal while in the
 *     air.
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

  // #571 — classify each non-covering segment as "ended before now"
  // or "starts after now". When BOTH classes appear in the same
  // walk, the user is in an inter-segment gap (mid-flight between
  // two booked stays); the arrival segment (first future one in
  // chronological order — lodging-primary upstream invariant) is the
  // best guess for the heartbeat's tz. No cross-shape datetime-vs-
  // date comparison is needed because we only track presence, plus
  // the FIRST future segment's tz under the chronological-order
  // invariant.
  let hasPrevEndedSeg = false;
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
        hasPrevEndedSeg = true;
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
      hasPrevEndedSeg = true;
    } else if (todayUtc < seg.from) {
      if (nextSegTz === null) nextSegTz = seg.timezone;
    }
  }

  // In-gap: previous-ended AND future segment both exist — user is
  // travelling towards the next booked stay. Return the arrival tz.
  if (hasPrevEndedSeg && nextSegTz !== null) return nextSegTz;
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
  }
  return { prev: row.current_tz, next, changed };
}

/**
 * #542 — Heartbeat advisory walker. Re-walks the cached `segments`
 * timeline against wall-clock `now` without re-fetching iCal. Called
 * from a 30-min `setInterval` in `src/index.ts`; on flip, the caller
 * sends a chat message via the main group's channel.
 *
 * Returns `{ prev, next }` only when the cached timeline produces a
 * different zone than `current_tz`; null on every other path (no row,
 * stale `schema_version`, null/empty `segments`, malformed JSON, or
 * computed value matches `current_tz`). On flip, `current_tz` is
 * updated in place and `segments` / `home_tz` are left untouched.
 */
export function runTzHeartbeatAdvisory(
  now: Date = new Date(),
): { prev: string; next: string } | null {
  const row = db
    .prepare(
      'SELECT current_tz, home_tz, segments, schema_version FROM tz_state WHERE id = 1',
    )
    .get() as
    | {
        current_tz: string;
        home_tz: string;
        segments: string | null;
        schema_version: number;
      }
    | undefined;
  if (!row) return null;
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
    return null;
  }
  if (!row.segments) return null;

  let parsed: readonly TripitSegment[];
  try {
    const decoded = JSON.parse(row.segments) as unknown;
    if (!Array.isArray(decoded)) return null;
    parsed = decoded as readonly TripitSegment[];
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    logger.warn(
      { err: err.message },
      'runTzHeartbeatAdvisory: tz_state.segments is malformed JSON — skipping (will recover on next sync_tripit run)',
    );
    return null;
  }

  const next = walkTzSegments(parsed, now, row.home_tz);
  if (next === row.current_tz) return null;

  db.prepare('UPDATE tz_state SET current_tz = ? WHERE id = 1').run(next);
  logger.info(
    { prev: row.current_tz, next },
    'tz_state.current_tz flipped via heartbeat advisory walker (#542)',
  );
  return { prev: row.current_tz, next };
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

// --- Session-length cap state (#413) ---
//
// Cumulative per-session totals enforced by the orchestrator's
// session-length cap. See `src/session-length-cap.schema.md` for the
// full schema doc. Only the orchestrator (`src/index.ts` runAgent)
// writes this table. The state-011 migration creates it.

export interface SessionLengthRow {
  group_folder: string;
  session_name: string;
  session_id: string;
  total_input_tokens: number;
  turn_count: number;
  last_handoff_summary: string | null;
  marked_for_reset: number;
  reset_reason: string | null;
  reset_cap: number | null;
  started_at: string;
  last_updated_at: string;
}

export type SessionResetReason = 'token_cap' | 'turn_cap';

/**
 * Read the cap-state row for a given session slot, or `undefined` if
 * no row exists yet (first turn ever, or post-reset before the next
 * turn writes the new row). Pure read — does not migrate.
 */
export function getSessionLengthState(
  groupFolder: string,
  sessionName: string,
): SessionLengthRow | undefined {
  const row = db
    .prepare(
      `SELECT group_folder, session_name, session_id,
              total_input_tokens, turn_count, last_handoff_summary,
              marked_for_reset, reset_reason, reset_cap,
              started_at, last_updated_at
         FROM session_length_state
         WHERE group_folder = ? AND session_name = ?`,
    )
    .get(groupFolder, sessionName) as SessionLengthRow | undefined;
  return row;
}

/**
 * Record one assistant turn's `usage.input_tokens` against the
 * session's cumulative totals. UPSERT semantics:
 *
 *   - No row: INSERT with `total_input_tokens = inputTokens`,
 *     `turn_count = 1`, `started_at = last_updated_at = now`.
 *   - Row exists, same `session_id`: ADD `inputTokens` to
 *     `total_input_tokens`, increment `turn_count`, bump
 *     `last_updated_at`. `marked_for_reset` is preserved.
 *   - Row exists, DIFFERENT `session_id`: rare drift surface — the
 *     SDK chain rotated under us without going through the
 *     orchestrator's reset path. Treat as a fresh row: REPLACE with
 *     the new session's totals (defensive; we'd rather attribute
 *     correctly than carry stale sums).
 *
 * `lastHandoffSummary`, when non-null, is written verbatim on INSERT
 * (or on the same-session-id UPDATE only when the existing column is
 * NULL — preserves the original handoff across many turns of the
 * post-reset session). Pass `null` for normal turns.
 *
 * Returns the row that the next threshold check should consult.
 */
export function recordSessionTurn(
  groupFolder: string,
  sessionName: string,
  sessionId: string,
  inputTokens: number,
  lastHandoffSummary: string | null,
): SessionLengthRow {
  const now = new Date().toISOString();
  const existing = getSessionLengthState(groupFolder, sessionName);
  if (!existing || existing.session_id !== sessionId) {
    db.prepare(
      `INSERT OR REPLACE INTO session_length_state (
         group_folder, session_name, session_id,
         total_input_tokens, turn_count, last_handoff_summary,
         marked_for_reset, started_at, last_updated_at, schema_version
       ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, 1)`,
    ).run(
      groupFolder,
      sessionName,
      sessionId,
      inputTokens,
      1,
      lastHandoffSummary,
      now,
      now,
    );
  } else {
    db.prepare(
      `UPDATE session_length_state
         SET total_input_tokens = total_input_tokens + ?,
             turn_count = turn_count + 1,
             last_updated_at = ?,
             last_handoff_summary = COALESCE(last_handoff_summary, ?)
         WHERE group_folder = ? AND session_name = ?`,
    ).run(inputTokens, now, lastHandoffSummary, groupFolder, sessionName);
  }
  // Re-read so the caller gets the post-update view in one consistent
  // shape — small extra read in exchange for the threshold-check call
  // site never having to reason about INSERT-vs-UPDATE semantics.
  const row = getSessionLengthState(groupFolder, sessionName);
  if (!row) {
    // Should be unreachable: we just upserted. Throw a specific error
    // so a future schema regression surfaces loudly rather than
    // returning a fictional zero-state.
    throw new Error(
      `recordSessionTurn: row missing immediately after upsert (group=${groupFolder} session=${sessionName})`,
    );
  }
  return row;
}

/**
 * Set `marked_for_reset = 1` and capture the reason + cap value on
 * the cap-state row. Idempotent: a second call is a no-op (the
 * WHERE clause excludes already-marked rows so the UPDATE doesn't
 * churn `last_updated_at` and doesn't overwrite the reason that
 * actually fired first). Returns the number of rows changed (0 =
 * already marked / no row, 1 = newly marked).
 *
 * `reason` and `cap` are stored at mark-time so the consume path on
 * the next inbound spawn can format a precise user-facing
 * notification without re-deriving (env-driven thresholds can
 * change between mark and consume; the notification reflects what
 * the operator actually had configured at trip time).
 */
export function markSessionForReset(
  groupFolder: string,
  sessionName: string,
  reason: SessionResetReason,
  cap: number,
): number {
  const result = db
    .prepare(
      `UPDATE session_length_state
         SET marked_for_reset = 1,
             reset_reason = ?,
             reset_cap = ?
         WHERE group_folder = ?
           AND session_name = ?
           AND marked_for_reset = 0`,
    )
    .run(reason, cap, groupFolder, sessionName);
  return result.changes;
}

/**
 * Atomic "consume the pending reset" — reads the marked-for-reset
 * row's `session_id` + `last_handoff_summary`, then deletes the row,
 * inside a single transaction. Returns `null` when no reset was
 * pending so the caller can distinguish "no-op normal spawn" from
 * "post-reset spawn that needs the handoff prefix".
 *
 * The DELETE is deliberate — the next assistant turn under the new
 * `session_id` will re-INSERT a fresh row via `recordSessionTurn`.
 * Carrying the old row forward and overwriting in place would risk
 * a misattributed turn if the threshold-check fires before the
 * orchestrator has a chance to write the new `session_id`.
 */
export interface PendingReset {
  sessionId: string;
  lastHandoffSummary: string | null;
  reason: SessionResetReason;
  cap: number;
}

export function consumeSessionReset(
  groupFolder: string,
  sessionName: string,
): PendingReset | null {
  const consume = db.transaction(() => {
    const row = db
      .prepare(
        `SELECT session_id, last_handoff_summary, reset_reason, reset_cap
           FROM session_length_state
           WHERE group_folder = ?
             AND session_name = ?
             AND marked_for_reset = 1`,
      )
      .get(groupFolder, sessionName) as
      | {
          session_id: string;
          last_handoff_summary: string | null;
          reset_reason: string | null;
          reset_cap: number | null;
        }
      | undefined;
    if (!row) return null;
    db.prepare(
      `DELETE FROM session_length_state
         WHERE group_folder = ? AND session_name = ?`,
    ).run(groupFolder, sessionName);
    // Validate the persisted reason. A NULL or unknown value
    // indicates the row was marked by a code path that didn't go
    // through `markSessionForReset` (corruption, manual SQL,
    // future-version migration leaving NULL). Fall back to
    // `'token_cap'` so the notification path still has a defined
    // shape; the diagnostic log line at the call site carries the
    // raw row for forensics.
    const reason: SessionResetReason =
      row.reset_reason === 'turn_cap' ? 'turn_cap' : 'token_cap';
    return {
      sessionId: row.session_id,
      lastHandoffSummary: row.last_handoff_summary,
      reason,
      cap: row.reset_cap ?? 0,
    };
  });
  return consume();
}

/**
 * Drop all cap-state rows for a group. Called from nuke handlers so
 * a force-reset wipes the cap accounting alongside the SDK session.
 * Best-effort — never throws on missing row.
 */
export function clearSessionLengthStateForGroup(groupFolder: string): number {
  const result = db
    .prepare('DELETE FROM session_length_state WHERE group_folder = ?')
    .run(groupFolder);
  return result.changes;
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

// Dual-mode reader for `registered_groups.trigger_pattern` (#81).
// Pre-#81 rows store a single string ("@Andy"). Post-#81 rows store a
// JSON-encoded `TriggerPatternConfig`. The createSchema-time backfill
// converts every legacy row to JSON on first boot, but we keep the
// dual-mode read path for two reasons:
//   1. A user rolling back to a pre-#81 binary then forward again must
//      re-converge cleanly without manual intervention.
//   2. The setup/register CLI and external tooling sometimes write
//      raw strings directly via `_writeRawRegisteredGroup` for tests;
//      we want those to keep working without forcing every test to
//      know the JSON shape.
//
// Returns the canonical config (or null when the row is unreadable)
// plus the derived trigger string used to populate
// `RegisteredGroup.trigger`. Three failure modes are distinguished:
//
//   - Legacy string shape (no leading `{`) → wrap as a synthesised
//     single-element keyword config. Reader-side compat for rollback
//     and direct test writes.
//   - JSON parse fails or shape doesn't validate → loud warn, fall
//     back to a legacy keyword config so a single bad row can't
//     crash boot. The trigger string surfaces the raw column so
//     operators can still see what's there.
//   - JSON parses, shape validates, BUT version > 1 (a future
//     binary's row read by this older binary) → loud warn, return
//     `null` config and `null` trigger. The gate framework treats
//     null as "no opinion" and falls through to the next gate, so
//     the failure is loud-but-non-fatal. See review of #84,
//     comment-id 4360940433: silent fallback-to-keyword on a future
//     row was too quiet.
function parseTriggerPatternColumn(
  raw: string,
  jid: string,
): { config: TriggerPatternConfig | null; trigger: string | null } {
  const trimmed = raw.trim();
  // Cheap shape sniff: JSON config always starts with `{`. Anything
  // else is the legacy string shape (or corruption that we treat as
  // legacy by best-effort).
  if (!trimmed.startsWith('{')) {
    const legacy = legacyTriggerToConfig(raw);
    return {
      config: legacy,
      trigger: deriveTriggerString(legacy),
    };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    logger.warn(
      { errName: err.name, jid, len: raw.length },
      'registered_groups: invalid trigger_pattern JSON, falling back to legacy keyword shape',
    );
    const legacy = legacyTriggerToConfig(raw);
    return {
      config: legacy,
      trigger: raw,
    };
  }
  // Detect a future-version row BEFORE the strict-shape validator
  // would also reject it: a `{version: 2, ...}` row read by a v1
  // binary is a forward-compat scenario (rollback), not corruption.
  // Surface it loudly so the operator notices, and return null so
  // downstream gates treat the row as "no patterns configured"
  // (== fail-open at the gate combinator) rather than silently
  // re-interpreting the raw JSON as a literal keyword pattern.
  if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
    const v = (parsed as Record<string, unknown>).version;
    if (typeof v === 'number' && v > 1) {
      logger.warn(
        { jid, foundVersion: v, expectedVersion: 1, len: raw.length },
        'registered_groups: trigger_pattern JSON has unsupported future version, treating as no-config (gates will fall through to fail-open)',
      );
      return { config: null, trigger: null };
    }
  }
  if (!isTriggerPatternConfig(parsed)) {
    logger.warn(
      { jid, len: raw.length },
      'registered_groups: trigger_pattern JSON is not a valid TriggerPatternConfig, falling back to legacy keyword shape',
    );
    const legacy = legacyTriggerToConfig(raw);
    return {
      config: legacy,
      trigger: raw,
    };
  }
  return {
    config: parsed,
    trigger: deriveTriggerString(parsed),
  };
}

function legacyTriggerToConfig(raw: string): TriggerPatternConfig {
  return {
    version: 1,
    patterns: [
      {
        pattern: raw,
        kind: 'keyword',
        source: 'owner-set',
        precision: 0,
        sample_count: 0,
        last_matched_at: null,
        last_updated_at: null,
      },
    ],
  };
}

/**
 * Derive the legacy `RegisteredGroup.trigger` string from a parsed
 * config. Pinned-down replacement for the unspecified
 * "first-keyword-or-empty" behaviour flagged on PR #84
 * (comment-id 4360940433).
 *
 * Order of preference, locked down so #82 can write mention-only /
 * regex-only configs without surprising any legacy reader:
 *   1. First `keyword`-kind pattern → return its `pattern` verbatim.
 *      This preserves the pre-#81 shape where `@Andy` was stored as
 *      a literal `@Andy` keyword.
 *   2. Else first `mention`-kind pattern → return `'@' + pattern`.
 *      Mentions are stored bare per the migration backfill (Fix 1
 *      above), so re-prepending the `@` keeps the legacy regex
 *      `(?:^|\s)@Andy\b` alive for call sites that haven't migrated
 *      to the new gate.
 *   3. Else `null`. Downstream call sites coerce null → undefined
 *      and let `getTriggerPattern` fall back to the global default
 *      `@<assistant>` regex; this preserves message-loop wakeup for
 *      the global mention even on a config with no per-group
 *      keyword/mention. Returning `null` (vs the previous empty
 *      string) is the type-system signal that "there is no
 *      group-specific trigger string" — ambiguous before, explicit
 *      now.
 */
export function deriveTriggerString(
  config: TriggerPatternConfig | null,
): string | null {
  if (!config) return null;
  const keyword = config.patterns.find((p) => p.kind === 'keyword');
  if (keyword) return keyword.pattern;
  const mention = config.patterns.find((p) => p.kind === 'mention');
  if (mention) {
    return mention.pattern.startsWith('@')
      ? mention.pattern
      : `@${mention.pattern}`;
  }
  return null;
}

// Allow-listed enums for `kind` / `source` per the TriggerPatternKind /
// TriggerPatternSource union types in src/types.ts. Validation here means
// a typo like `kind: "mentoin"` is caught at parse time and the row falls
// through to the warn-log path in `parseTriggerPatternColumn`, instead of
// silently being treated as "valid config" and then ignored downstream
// because `deriveTriggerString` finds no `keyword`/`mention` it recognises.
const TRIGGER_PATTERN_KINDS: ReadonlySet<string> = new Set([
  'keyword',
  'mention',
  'reply',
  'regex',
  'sender_tier',
]);
const TRIGGER_PATTERN_SOURCES: ReadonlySet<string> = new Set([
  'owner-set',
  'learned',
  'universal',
]);

function isTriggerPatternConfig(value: unknown): value is TriggerPatternConfig {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) return false;
  if (!Array.isArray(v.patterns)) return false;
  for (const p of v.patterns) {
    if (p === null || typeof p !== 'object' || Array.isArray(p)) return false;
    const pp = p as Record<string, unknown>;
    if (typeof pp.pattern !== 'string') return false;
    if (typeof pp.kind !== 'string' || !TRIGGER_PATTERN_KINDS.has(pp.kind)) {
      return false;
    }
    if (
      typeof pp.source !== 'string' ||
      !TRIGGER_PATTERN_SOURCES.has(pp.source)
    ) {
      return false;
    }
    if (typeof pp.precision !== 'number') return false;
    if (typeof pp.sample_count !== 'number') return false;
    if (pp.last_matched_at !== null && typeof pp.last_matched_at !== 'string') {
      return false;
    }
    if (pp.last_updated_at !== null && typeof pp.last_updated_at !== 'string') {
      return false;
    }
  }
  return true;
}

/**
 * Serialise a `RegisteredGroup` into the column value for
 * `trigger_pattern`. Always emits JSON (forward-only writes). When the
 * caller already has a `triggerPatterns` config, it wins; otherwise
 * we synthesise a single-element keyword config from `group.trigger`
 * with `source: "owner-set"` so the row is observability-ready
 * immediately. A null `group.trigger` with no `triggerPatterns`
 * yields an empty-pattern config — readable by the gate framework as
 * "no patterns configured" (== `pass`, fail-open).
 */
function serializeTriggerPatternForColumn(group: RegisteredGroup): string {
  if (group.triggerPatterns) return JSON.stringify(group.triggerPatterns);
  if (group.trigger === null || group.trigger === undefined) {
    return JSON.stringify({ version: 1, patterns: [] });
  }
  return JSON.stringify(legacyTriggerToConfig(group.trigger));
}

/**
 * Read the full trigger pattern config for a group.
 *
 * Returns:
 *   - `undefined` — the group does not exist (no row).
 *   - `null` — the row exists but is unreadable for the current
 *     binary (e.g. JSON parse OK but `version > 1`, signalling a
 *     forward-binary write read by an older binary). Loud-warned at
 *     parse time. The trigger gate consumes null as "no opinion"
 *     (decision: `pass`), which falls through the chain to default-
 *     allow rather than silent-misinterpret. See review of #84,
 *     comment-id 4360940433.
 *   - `TriggerPatternConfig` otherwise. Legacy string rows surface as
 *     a synthesised single-element config.
 */
export function getTriggerPatterns(
  jid: string,
): TriggerPatternConfig | null | undefined {
  const row = db
    .prepare('SELECT trigger_pattern FROM registered_groups WHERE jid = ?')
    .get(jid) as { trigger_pattern: string } | undefined;
  if (!row) return undefined;
  return parseTriggerPatternColumn(row.trigger_pattern, jid).config;
}

/**
 * Replace the trigger pattern config for an existing group. Throws
 * if the group is not registered (callers should `setRegisteredGroup`
 * first). Updates `trigger_pattern` only — other columns untouched.
 */
export function setTriggerPatterns(
  jid: string,
  config: TriggerPatternConfig,
): void {
  if (config.version !== 1) {
    throw new Error(
      `Unsupported TriggerPatternConfig version ${config.version} for jid ${jid}`,
    );
  }
  const result = db
    .prepare(`UPDATE registered_groups SET trigger_pattern = ? WHERE jid = ?`)
    .run(JSON.stringify(config), jid);
  if (result.changes === 0) {
    throw new Error(
      `setTriggerPatterns: no registered_groups row for jid ${jid}`,
    );
  }
}

/**
 * Re-export the row-level `TriggerPattern` type so #82's self-improvement
 * code can import its observability shape from a single place.
 */
export type { TriggerPattern, TriggerPatternConfig };

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
  const triggerParsed = parseTriggerPatternColumn(row.trigger_pattern, row.jid);
  return {
    jid: row.jid,
    name: row.name,
    folder: row.folder,
    trigger: triggerParsed.trigger,
    added_at: row.added_at,
    containerConfig: parseContainerConfig(row.container_config, row.jid),
    requiresTrigger:
      row.requires_trigger === null ? undefined : row.requires_trigger === 1,
    isMain: row.is_main === 1 ? true : undefined,
    triggerPatterns: triggerParsed.config ?? undefined,
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
    serializeTriggerPatternForColumn(group),
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
  const {
    jid: _existingJid,
    triggerPatterns: _existingTriggerPatterns,
    ...rest
  } = existing;
  void _existingJid;
  void _existingTriggerPatterns;
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
    const triggerParsed = parseTriggerPatternColumn(
      row.trigger_pattern,
      row.jid,
    );
    result[row.jid] = {
      name: row.name,
      folder: row.folder,
      trigger: triggerParsed.trigger,
      added_at: row.added_at,
      containerConfig: parseContainerConfig(row.container_config, row.jid),
      requiresTrigger:
        row.requires_trigger === null ? undefined : row.requires_trigger === 1,
      isMain: row.is_main === 1 ? true : undefined,
      triggerPatterns: triggerParsed.config ?? undefined,
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

  // Per-group migrations (#293 wave). Each returns a `MigrationSummary`
  // (#433) so we can emit one startup-summary log line per migration —
  // the operator's grep target for "did the data plane come up clean".
  // The DATA_DIR migrations above (router_state / sessions /
  // registered_groups) are NOT per-group and do not contribute summaries.
  const summaries: MigrationSummary[] = [];

  // Migrate per-group orders-db.json files (#294). Unlike the helpers
  // above (DATA_DIR-rooted), this scans every `groups/<name>/` folder
  // for an `orders-db.json` because the file historically lived in the
  // admin group's working dir. Idempotent: a successful migration
  // renames the source to `orders-db.json.migrated-YYYY-MM-DD`, so a
  // re-run of `initDatabase` is a no-op once the file is gone.
  summaries.push(migrateOrdersDbJsonFiles());

  // Migrate per-group morning-brief-pending.json files (#299). Same
  // per-group-scan pattern as the orders import above: the source
  // file historically lived under each group's working dir, the
  // schema-only migration in state-007 created the three queue tables,
  // and this pass populates them from the JSON-era shape. Idempotent:
  // each successful per-file import renames the source to
  // `morning-brief-pending.json.migrated-YYYY-MM-DD`, so a re-run of
  // `initDatabase` is a no-op once the file is gone.
  summaries.push(migrateMorningBriefPendingJsonFiles());

  // Migrate per-group calendar-state.json files (#300). Same
  // per-group-scan pattern: the source file historically lived under
  // each group's working dir as a JSON envelope wrapping a per-day
  // `events` array, the schema-only migration in state-008 created
  // `calendar_snapshots` + `calendar_events` (FK + cascade), and this
  // pass populates them from the JSON-era shape. Idempotent: each
  // successful per-file import renames the source to
  // `calendar-state.json.migrated-YYYY-MM-DD`, so a re-run of
  // `initDatabase` is a no-op once the file is gone.
  summaries.push(migrateCalendarStateJsonFiles());

  // Migrate per-group heartbeat-state.json files into the
  // `phase_completions` table (#301). The state-009 schema landed in
  // PR #346; this pass populates the three phase rows (heartbeat,
  // nightly, weekly) from the JSON-era envelope plus the heartbeat-
  // specific `last_composio_check` extra in the heartbeat row's
  // `metadata` JSON blob (writer-decides convention per the state-009
  // doc-header). UPSERT keyed on `phase` so a re-run with newer values
  // wins without resetting defaulted columns. Source renamed to
  // `heartbeat-state.json.migrated-YYYY-MM-DD` on success — re-run is
  // a no-op once the suffix is in place.
  summaries.push(migrateHeartbeatStateJsonFiles());

  // Migrate per-group task-tz-state.json files into the singleton
  // `tz_state` row + `follow_me_tasks` per-skill rows (#302). The
  // state-010 schema landed in PR #348; this pass populates both
  // tables from the JSON-era envelope. UPSERT semantics throughout —
  // never `INSERT OR REPLACE`, which would silently reset
  // `schema_version` (a column the writer's UPSERT doesn't name) on
  // every re-run.
  summaries.push(migrateTaskTzStateJsonFiles());

  // Migrate per-group session-state.json files (the multi-writer
  // trusted-memory state) into trusted_sessions + trusted_session_singleton
  // (#298). The state-006 schema landed in PR #340; this pass populates
  // both tables from the JSON-era envelope. UPSERT semantics on both
  // tables — never INSERT OR REPLACE.
  summaries.push(migrateTrustedSessionStateJsonFiles());

  // Migrate per-group nanoclaw-state.json files (the multi-key junk
  // drawer) into the three state-005 tables: email_state singleton,
  // email_seen_ids set, resumable_cycles per-skill rows (#297).
  // UPSERT throughout — never INSERT OR REPLACE.
  summaries.push(migrateNanoclawStateJsonFiles());

  // Migrate per-group scheduled-reminders.json files into the
  // scheduled_reminders table created by state-004 (#296). Append-only
  // INSERT with ON CONFLICT(event_id) DO NOTHING.
  summaries.push(migrateScheduledRemindersJsonFiles());

  // Migrate per-group email-feedback.json files into the email_feedback
  // table created by state-002 (#295). Append-only INSERT (id is
  // AUTOINCREMENT; no natural-key dedup). Idempotency gated by source-
  // file rename. Accepts both wrapped {feedback:[...]} and bare-array
  // shapes per the issue body.
  summaries.push(migrateEmailFeedbackJsonFiles());

  emitMigrationStartupSummary(summaries);
}

/**
 * Emit one structured log line per per-group migration (#433) so
 * operators have a single grep target — `JSON state migration summary`
 * — for "did the data plane come up clean". INFO when the migration
 * left no files behind for triage; WARN when at least one group folder
 * still holds a source file (bad JSON, DB constraint violation, missing
 * required envelope fields).
 *
 * The literal string `JSON state migration summary` is load-bearing —
 * it's the operator's grep target. Don't rephrase it.
 */
function emitMigrationStartupSummary(summaries: MigrationSummary[]): void {
  for (const s of summaries) {
    const fields = {
      migration: s.name,
      migrated: s.migrated,
      skipped_already_done: s.skippedAlreadyDone,
      left_in_place_count: s.leftInPlace.length,
      left_in_place_groups: s.leftInPlace,
    };
    let message = `JSON state migration summary: ${s.name} — migrated=${s.migrated} skipped-already-done=${s.skippedAlreadyDone} left-in-place=${s.leftInPlace.length}`;
    if (s.leftInPlace.length > 0) {
      message += ` (groups: ${s.leftInPlace.join(', ')})`;
    }
    if (s.leftInPlace.length === 0) {
      logger.info(fields, message);
    } else {
      logger.warn(fields, message);
    }
  }
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
  // Optional in the source JSON — `check-orders` skill runs don't
  // always populate it. The migration defaults missing values to
  // `order_date` (then current ISO) before the NOT NULL insert (#347).
  last_updated?: string;
}

interface OrdersDbJsonShape {
  orders?: OrdersDbJsonRecord[];
  last_checked?: string;
  last_updated?: string;
}

/**
 * Return the first candidate that is a non-empty string. Treats null,
 * undefined, and `""` all as "missing" — `??` alone would let `""`
 * through (it's defined and non-null), which satisfies SQLite's NOT
 * NULL constraint but leaves a semantically empty value downstream.
 *
 * Falls through to a fresh ISO timestamp when every candidate is
 * missing — guarantees the return value is always a non-empty string,
 * so callers can pass it directly to a NOT NULL `TEXT` column without
 * an extra check.
 *
 * Exported for the dedicated unit test in
 * `orders-json-migration.test.ts`; not part of the public API.
 *
 * @internal exported only for tests (see jbaruch/nanoclaw#347).
 */
export function firstNonEmpty(
  candidates: ReadonlyArray<string | null | undefined>,
): string {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.length > 0) {
      return candidate;
    }
  }
  return new Date().toISOString();
}

function migrateOrdersDbJsonFiles(): MigrationSummary {
  const summary = newMigrationSummary('orders-db');
  if (!fs.existsSync(GROUPS_DIR)) return summary;
  let groupFolders: string[];
  try {
    groupFolders = fs
      .readdirSync(GROUPS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .filter((entry) => isValidGroupFolder(entry.name))
      .map((entry) => entry.name)
      // Sort so "first writer wins" with ON CONFLICT DO NOTHING below
      // is deterministic across filesystems AND across locales.
      // readdirSync order is implementation-defined (ext4 hash order,
      // APFS insertion order, etc.) and `localeCompare` would add a
      // second axis of nondeterminism (Turkish dotted-i, German
      // ß-vs-ss, ICU version skew). Plain code-point comparison via
      // </> on string operands is locale-free and stable across Node
      // versions.
      .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return summary;
    throw err;
  }

  // Bare `ON CONFLICT DO NOTHING` — handles BOTH the
  // `email_message_id UNIQUE` constraint and the PK `id`. The latter
  // *can* collide in practice: `id` is
  // `{source}-{order_date}-SHA1(description)[:8]`, so two distinct
  // emails with the same source + order_date + description (e.g., a
  // resent confirmation, or two amazon orders for the same item on
  // the same day) produce identical ids despite different
  // email_message_id values. For one-shot data backfill we want
  // idempotency, not strict validation: first row in (sorted by
  // folder above for determinism) wins, every other duplicate is
  // silently skipped. The downstream `check-orders` skill enforces
  // its own merge semantics on subsequent writes.
  const insertOrder = db.prepare(
    `INSERT INTO orders (
       id, source, status, amount, currency, description, order_date,
       expected_delivery, email_message_id, to_address, flagged,
       flag_reason, last_updated
     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT DO NOTHING`,
  );
  const upsertMetadata = db.prepare(
    `INSERT INTO orders_metadata (key, value) VALUES (?, ?)
     ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
  );

  const stamp = new Date().toISOString().slice(0, 10);

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'orders-db.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let parsed: OrdersDbJsonShape;
    try {
      parsed = JSON.parse(
        fs.readFileSync(filePath, 'utf-8'),
      ) as OrdersDbJsonShape;
    } catch (err) {
      if (err instanceof SyntaxError) {
        logger.warn(
          { folder, errName: err.name },
          'orders-db.json migration: invalid JSON, skipping (file left in place)',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      // TOCTOU race: the existsSync check above is best-effort, not
      // authoritative — between that check and readFileSync the file
      // can be removed by a concurrent migration run, manual cleanup,
      // or filesystem reorg. Treat ENOENT here the same as ENOENT at
      // rename time: idempotent no-op, log at info, continue. Every
      // other errno propagates per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'orders-db.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }
    if (!Array.isArray(parsed.orders)) {
      logger.warn(
        { folder },
        'orders-db.json migration: missing "orders" array, skipping',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
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
    const orders = parsed.orders;
    const importFile = db.transaction(() => {
      let insertedRows = 0;
      let skippedRows = 0;
      for (const order of orders) {
        // #347: `orders.last_updated` is `TEXT NOT NULL` but the source
        // JSON (produced incrementally by `check-orders` skill runs) does
        // not always populate it — observed on production 2026-04-30
        // where one entry of 120 had no `last_updated`, taking down the
        // orchestrator on next boot via the SqliteError NOT NULL
        // constraint propagating out of the transaction. Default to
        // `order_date` (the next-best upper bound: we knew about this
        // order at least by then) and finally to a fresh ISO timestamp.
        //
        // Treat null, undefined, AND empty-string as "missing" — `??`
        // alone would let `last_updated: ""` through, satisfying NOT
        // NULL but leaving a semantically empty timestamp downstream
        // (PR #350 review: copilot caught this on the original `??`
        // chain). The helper falls through to the next candidate the
        // same way for any of those three shapes and always returns a
        // non-empty string.
        //
        // Other NOT NULL columns (id, source, status, description,
        // order_date, email_message_id) intentionally have no default —
        // their absence indicates a corrupt source row that should
        // surface as an error rather than be silently masked with a
        // synthesized value, since downstream `check-orders` semantics
        // depend on those fields meaning what the source said.
        const lastUpdated = firstNonEmpty([
          order.last_updated,
          order.order_date,
        ]);
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
          lastUpdated,
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

    // Rename is metadata cleanup; the data import already committed.
    // Per `coding-policy: error-handling`, only one specific errno is
    // recoverable here: `ENOENT` means the source disappeared between
    // the import and the rename (concurrent migration run, manual
    // file move) — that's an idempotent no-op since the data is
    // already in SQL. Every other errno (`EACCES`, `EPERM`, `EXDEV`,
    // `ENOSPC`, etc.) indicates a real environment problem the
    // operator must fix before startup proceeds; rethrowing those
    // surfaces the issue immediately rather than letting the
    // orchestrator come up with stale JSON files lying around.
    try {
      fs.renameSync(filePath, `${filePath}.migrated-${stamp}`);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
      // Source file already gone — log + continue without the
      // info-level "renamed" message since no rename actually
      // happened.
      logger.info(
        {
          folder,
          inserted: counts.insertedRows,
          skipped: counts.skippedRows,
          total: orders.length,
        },
        'orders-db.json migration: imported; source already absent at rename time',
      );
      // Data did land in SQL; count as migrated-this-boot from the
      // operator's "did anything land" perspective (matches
      // renameMigratedSource's success-counting semantics).
      summary.migrated += 1;
      continue;
    }
    logger.info(
      {
        folder,
        inserted: counts.insertedRows,
        skipped: counts.skippedRows,
        total: orders.length,
      },
      'orders-db.json migration: imported and source renamed',
    );
    summary.migrated += 1;
  }
  return summary;
}

interface MorningBriefCleanupItemJson {
  id: string;
  type: string;
  question?: string | null;
  subject?: string | null;
  sender?: string | null;
  added?: string | null;
}

interface MorningBriefPendingDecisionJson {
  id: string;
  question: string;
  added?: string | null;
}

interface MorningBriefUndatedTaskJson {
  id: string;
  title: string;
  tasklist_id: string;
  added?: string | null;
}

interface MorningBriefPendingJsonShape {
  cleanup_items?: MorningBriefCleanupItemJson[];
  pending_decisions?: MorningBriefPendingDecisionJson[];
  undated_tasks?: MorningBriefUndatedTaskJson[];
}

function migrateMorningBriefPendingJsonFiles(): MigrationSummary {
  const summary = newMigrationSummary('morning-brief-pending');
  if (!fs.existsSync(GROUPS_DIR)) return summary;
  let groupFolders: string[];
  try {
    groupFolders = fs
      .readdirSync(GROUPS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .filter((entry) => isValidGroupFolder(entry.name))
      .map((entry) => entry.name)
      // Sort so PK-conflict resolution under `ON CONFLICT(id) DO NOTHING`
      // is deterministic across filesystems and locales — same rationale
      // as the orders migration above (readdirSync order is impl-defined,
      // plain code-point comparison is locale-free).
      .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return summary;
    throw err;
  }

  // ON CONFLICT(id) DO NOTHING (NOT `INSERT OR IGNORE`): re-running
  // with leftover rows (e.g. an operator copied a partial DB back over
  // an already-imported one) is a silent no-op on the PK conflict, but
  // a NOT NULL violation on `type`/`question`/`title`/`tasklist_id`
  // still throws. `INSERT OR IGNORE` would silently swallow those too,
  // and we'd rename the source file thinking the import succeeded —
  // bad data lost without a trace. Per-row, when `added` is missing in
  // the source we omit the column from the INSERT so the schema's
  // `DEFAULT CURRENT_TIMESTAMP` fires — that's the contract documented
  // on state-007.
  const insertCleanupItemWithAdded = db.prepare(
    `INSERT INTO pending_cleanup_items
       (id, type, question, subject, sender, added)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertCleanupItemDefaultAdded = db.prepare(
    `INSERT INTO pending_cleanup_items
       (id, type, question, subject, sender)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertDecisionWithAdded = db.prepare(
    `INSERT INTO pending_decisions (id, question, added)
     VALUES (?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertDecisionDefaultAdded = db.prepare(
    `INSERT INTO pending_decisions (id, question) VALUES (?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertUndatedTaskWithAdded = db.prepare(
    `INSERT INTO pending_undated_tasks
       (id, title, tasklist_id, added)
     VALUES (?, ?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );
  const insertUndatedTaskDefaultAdded = db.prepare(
    `INSERT INTO pending_undated_tasks (id, title, tasklist_id)
     VALUES (?, ?, ?)
     ON CONFLICT(id) DO NOTHING`,
  );

  const stamp = new Date().toISOString().slice(0, 10);

  for (const folder of groupFolders) {
    const filePath = path.join(
      GROUPS_DIR,
      folder,
      'morning-brief-pending.json',
    );
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let parsed: MorningBriefPendingJsonShape;
    try {
      parsed = JSON.parse(
        fs.readFileSync(filePath, 'utf-8'),
      ) as MorningBriefPendingJsonShape;
    } catch (err) {
      if (err instanceof SyntaxError) {
        logger.warn(
          { folder, errName: err.name },
          'morning-brief-pending.json migration: invalid JSON, skipping (file left in place)',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      // TOCTOU race: existsSync above is best-effort; file may
      // disappear before readFileSync. Mirror orders' handling — log
      // info on ENOENT, propagate every other errno.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'morning-brief-pending.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    // `JSON.parse` happily returns null / numbers / strings / arrays
    // for syntactically valid but non-object payloads. Bind those to
    // a property access and we'd throw "cannot read properties of
    // null" before the array-shape guard below ever ran, halting the
    // whole migration on one bad file. Warn-and-skip per
    // `coding-policy: error-handling` (try alternatives before
    // failing) instead.
    if (
      parsed === null ||
      typeof parsed !== 'object' ||
      Array.isArray(parsed)
    ) {
      logger.warn(
        {
          folder,
          parsedType:
            parsed === null
              ? 'null'
              : Array.isArray(parsed)
                ? 'array'
                : typeof parsed,
        },
        'morning-brief-pending.json migration: payload is not an object, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const cleanupItems = parsed.cleanup_items;
    const pendingDecisions = parsed.pending_decisions;
    const undatedTasks = parsed.undated_tasks;
    if (
      !Array.isArray(cleanupItems) &&
      !Array.isArray(pendingDecisions) &&
      !Array.isArray(undatedTasks)
    ) {
      logger.warn(
        { folder },
        'morning-brief-pending.json migration: no recognised arrays (cleanup_items / pending_decisions / undated_tasks), skipping',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    // One transaction per file so a mid-import crash can't leave any
    // of the three tables half-populated. Same rationale as the orders
    // migration; see the comment block on `migrateOrdersDbJsonFiles`.
    const cleanupCounts = { inserted: 0, skipped: 0, total: 0 };
    const decisionCounts = { inserted: 0, skipped: 0, total: 0 };
    const undatedTaskCounts = { inserted: 0, skipped: 0, total: 0 };
    try {
      // Per-row object guard: a stale `null`/string/number element in
      // any of the three arrays would otherwise throw a TypeError
      // inside the transaction (e.g. `null.added` blows up before any
      // INSERT runs), and the narrowed catch below would propagate
      // that as an "unexpected" error and halt orchestrator startup.
      // Skip non-object elements with a warn instead — same shape as
      // the file-level non-object guard above.
      const isObjectRow = (row: unknown): row is Record<string, unknown> =>
        row !== null && typeof row === 'object' && !Array.isArray(row);

      const importFile = db.transaction(() => {
        if (Array.isArray(cleanupItems)) {
          cleanupCounts.total = cleanupItems.length;
          for (const item of cleanupItems) {
            if (!isObjectRow(item)) {
              logger.warn(
                { folder, queue: 'cleanup_items' },
                'morning-brief-pending.json migration: skipping non-object row',
              );
              cleanupCounts.skipped++;
              continue;
            }
            const result = item.added
              ? insertCleanupItemWithAdded.run(
                  item.id,
                  item.type,
                  item.question ?? null,
                  item.subject ?? null,
                  item.sender ?? null,
                  item.added,
                )
              : insertCleanupItemDefaultAdded.run(
                  item.id,
                  item.type,
                  item.question ?? null,
                  item.subject ?? null,
                  item.sender ?? null,
                );
            if (result.changes > 0) cleanupCounts.inserted++;
            else cleanupCounts.skipped++;
          }
        }
        if (Array.isArray(pendingDecisions)) {
          decisionCounts.total = pendingDecisions.length;
          for (const decision of pendingDecisions) {
            if (!isObjectRow(decision)) {
              logger.warn(
                { folder, queue: 'pending_decisions' },
                'morning-brief-pending.json migration: skipping non-object row',
              );
              decisionCounts.skipped++;
              continue;
            }
            const result = decision.added
              ? insertDecisionWithAdded.run(
                  decision.id,
                  decision.question,
                  decision.added,
                )
              : insertDecisionDefaultAdded.run(decision.id, decision.question);
            if (result.changes > 0) decisionCounts.inserted++;
            else decisionCounts.skipped++;
          }
        }
        if (Array.isArray(undatedTasks)) {
          undatedTaskCounts.total = undatedTasks.length;
          for (const task of undatedTasks) {
            if (!isObjectRow(task)) {
              logger.warn(
                { folder, queue: 'undated_tasks' },
                'morning-brief-pending.json migration: skipping non-object row',
              );
              undatedTaskCounts.skipped++;
              continue;
            }
            const result = task.added
              ? insertUndatedTaskWithAdded.run(
                  task.id,
                  task.title,
                  task.tasklist_id,
                  task.added,
                )
              : insertUndatedTaskDefaultAdded.run(
                  task.id,
                  task.title,
                  task.tasklist_id,
                );
            if (result.changes > 0) undatedTaskCounts.inserted++;
            else undatedTaskCounts.skipped++;
          }
        }
      });
      importFile();
    } catch (err) {
      // Per `coding-policy: error-handling`: catch only SqliteError
      // with a constraint-class code (NOT NULL / UNIQUE / CHECK /
      // PRIMARY KEY / FOREIGN KEY) — those are the recoverable data-
      // quality failures the per-file isolation contract was written
      // for. Anything else (a TypeError, a ReferenceError, a non-
      // constraint SqliteError like SQLITE_CORRUPT or SQLITE_BUSY) is
      // either a programming bug or a real environment problem that
      // halting startup loudly will surface, instead of being swept
      // under a per-file warn. The transaction rolled back on throw,
      // so no partial rows landed in either case.
      if (
        err instanceof SqliteError &&
        typeof err.code === 'string' &&
        err.code.startsWith('SQLITE_CONSTRAINT_')
      ) {
        logger.warn(
          { folder, errCode: err.code, err },
          'morning-brief-pending.json migration: row violated a DB constraint, rolling back and leaving source file in place for triage',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      throw err;
    }

    // Rename is metadata cleanup; the data import already committed.
    // ENOENT is the only recoverable errno (file vanished between
    // import and rename — idempotent no-op since the data is in SQL);
    // every other errno propagates.
    try {
      fs.renameSync(filePath, `${filePath}.migrated-${stamp}`);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
      logger.info(
        {
          folder,
          cleanup: cleanupCounts,
          decisions: decisionCounts,
          undated_tasks: undatedTaskCounts,
        },
        'morning-brief-pending.json migration: imported; source already absent at rename time',
      );
      summary.migrated += 1;
      continue;
    }
    logger.info(
      {
        folder,
        cleanup: cleanupCounts,
        decisions: decisionCounts,
        undated_tasks: undatedTaskCounts,
        renamed_to: `${filePath}.migrated-${stamp}`,
      },
      'morning-brief-pending.json migration: imported and source renamed',
    );
    summary.migrated += 1;
  }
  return summary;
}

/**
 * Migrate per-group `calendar-state.json` files into the
 * `calendar_snapshots` + `calendar_events` tables created by
 * state-008 (#300). Per-group-scan pattern (same as the orders and
 * morning-brief-pending migrations above) refactored onto the shared
 * helpers in `src/json-state-import.ts`: parse-shape guard, per-row
 * object guard, narrowed `SqliteError` constraint catch, deterministic
 * folder ordering, and ENOENT-tolerant rename. See PR #368 for the
 * helper extraction rationale.
 *
 * JSON-era envelope shape:
 *   {
 *     "date":       "YYYY-MM-DD",
 *     "fetched_at": "ISO8601",
 *     "events":     [
 *       { event_id, title, start, end?, reminder_task_id? }, ...
 *     ]
 *   }
 *
 * Insert order is snapshot-row-first, then per-event rows. Under the
 * current production setting (`PRAGMA foreign_keys` is OFF — see the
 * doc-header on `state-008-calendar-state.ts`) this is purely a
 * style/consistency choice; once the orchestrator flips
 * `foreign_keys = ON` globally, the ordering becomes load-bearing
 * because `calendar_events.date` references `calendar_snapshots.date`
 * via `ON DELETE CASCADE` and an event INSERT for an absent snapshot
 * would fire SQLITE_CONSTRAINT_FOREIGNKEY. Doing the snapshot first
 * inside the per-file transaction keeps the import correct under
 * either FK setting.
 *
 * `ON CONFLICT(date) DO NOTHING` on the snapshot insert: re-running
 * with leftover state (e.g. an operator copied a partial DB back over
 * an already-imported one) is a silent no-op on the PK conflict, but
 * a NOT NULL violation on `fetched_at` still throws and rolls back.
 * Same shape on the event insert keyed on `event_id` — the PK is
 * Google Calendar's own event ID, so re-importing the same source
 * file on top of a partial migration just re-converges on the
 * already-stored row.
 */
function migrateCalendarStateJsonFiles(): MigrationSummary {
  const summary = newMigrationSummary('calendar-state');
  const groupFolders = listGroupFoldersForMigration();
  if (groupFolders.length === 0) return summary;

  const insertSnapshot = db.prepare(
    `INSERT INTO calendar_snapshots (date, fetched_at)
     VALUES (?, ?)
     ON CONFLICT(date) DO NOTHING`,
  );
  const insertEvent = db.prepare(
    `INSERT INTO calendar_events
       (event_id, date, title, start, end, reminder_task_id)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(event_id) DO NOTHING`,
  );

  const stamp = migrationDateStamp();

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'calendar-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort; the file may
      // disappear before readFileSync. Mirror the orders / morning-
      // brief-pending migrations — log info on ENOENT, propagate every
      // other errno.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'calendar-state.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(
      raw,
      folder,
      'calendar-state.json',
      summary,
    );
    if (parsed === null) continue;

    const date = parsed.date;
    const fetchedAt = parsed.fetched_at;
    if (typeof date !== 'string' || typeof fetchedAt !== 'string') {
      // The two fields the snapshot row keys on / records. If either
      // is missing or wrong-typed, we can't write a snapshot row at
      // all, and the FK column on the events would dangle. Warn and
      // leave the file in place for triage.
      logger.warn(
        { folder },
        'calendar-state.json migration: missing or non-string date / fetched_at, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }
    // Distinguish "missing" (key absent or undefined) from
    // "wrong-typed" (key present but not an array — e.g. an object
    // or a string). Missing is a valid empty-day shape: still import
    // the snapshot row and rename. Wrong-typed is a corruption signal
    // that the schema can't migrate; warn (with the rejected type) and
    // skip the entire group's file so the events payload survives for
    // human triage instead of being silently discarded by the rename.
    const eventsRaw: unknown = parsed.events;
    let events: unknown[];
    if (eventsRaw === undefined) {
      events = [];
    } else if (Array.isArray(eventsRaw)) {
      events = eventsRaw;
    } else {
      logger.warn(
        {
          folder,
          eventsType: eventsRaw === null ? 'null' : typeof eventsRaw,
        },
        'calendar-state.json migration: events is not an array, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const counts = {
      inserted_snapshots: 0,
      inserted_events: 0,
      skipped_events: 0,
      total_events: events.length,
    };
    try {
      const importFile = db.transaction(() => {
        // Snapshot first — see doc-header. The conflict resolver
        // returns 0 changes if a previous run already inserted today's
        // snapshot; we count `inserted_snapshots` only on the change.
        const snapshotResult = insertSnapshot.run(date, fetchedAt);
        if (snapshotResult.changes > 0) counts.inserted_snapshots++;

        for (const ev of events) {
          if (!isObjectRow(ev)) {
            logger.warn(
              { folder, queue: 'events' },
              'calendar-state.json migration: skipping non-object row',
            );
            counts.skipped_events++;
            continue;
          }
          const result = insertEvent.run(
            ev.event_id,
            date,
            ev.title,
            ev.start,
            ev.end ?? null,
            ev.reminder_task_id ?? null,
          );
          if (result.changes > 0) counts.inserted_events++;
          else counts.skipped_events++;
        }
      });
      importFile();
    } catch (err) {
      // Narrowed catch: only constraint-class SqliteError is the
      // recoverable per-file failure (NOT NULL on title/start, FK
      // dangling, etc.). Anything else (a TypeError from a bug, a
      // SQLITE_BUSY from a noisy environment) propagates — operator
      // visibility per `coding-policy: error-handling`. The
      // transaction has already rolled back on throw, so no partial
      // rows landed.
      if (
        handleConstraintViolationOrRethrow(
          err,
          folder,
          'calendar-state.json',
          summary,
        )
      )
        continue;
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      'calendar-state.json',
      counts,
      summary,
    );
  }
  return summary;
}

/**
 * Shape of the JSON-era `groups/<name>/heartbeat-state.json` envelope
 * the heartbeat / nightly / weekly skills used to share via
 * `LOCK_EX` (#301). Every key is optional because the file accumulated
 * incrementally — an early-stage group may have only run the heartbeat
 * phase, leaving `nightly_last_completed` / `weekly_last_completed`
 * absent until those phases first ran. The migration imports whichever
 * keys are present and silently skips the absent ones.
 *
 * `last_composio_check` is the heartbeat skill's local extra; per the
 * state-009 doc-header it lands inside the `metadata` JSON blob on the
 * `heartbeat` row when present. `nightly` and `weekly` rows have no
 * phase-specific extras today, so their `metadata` is NULL.
 */
interface HeartbeatStateJsonShape {
  heartbeat_last_completed?: unknown;
  nightly_last_completed?: unknown;
  weekly_last_completed?: unknown;
  last_composio_check?: unknown;
}

function migrateHeartbeatStateJsonFiles(): MigrationSummary {
  const summary = newMigrationSummary('heartbeat-state');
  const groupFolders = listGroupFoldersForMigration();
  if (groupFolders.length === 0) return summary;

  // UPSERT keyed on `phase` (PK) — see the state-009 doc-header for
  // the full rationale. Crucially NOT `INSERT OR REPLACE`: the latter
  // is delete+insert in SQLite and would reset defaulted columns the
  // UPSERT doesn't name — load-bearingly `schema_version`, which the
  // owner skill bumps to drive future shape migrations. The UPSERT
  // here touches exactly the three columns the writer cares about
  // and stamps `updated_at` explicitly so a writer-supplied
  // `last_completed` from a clock that drifts can't outrun the row's
  // own mutation log.
  // metadata uses COALESCE(excluded.metadata, metadata) so a NULL
  // payload (e.g. a JSON file with `nightly_last_completed` set but no
  // `last_composio_check`) doesn't wipe metadata that an earlier run
  // already wrote into the row. The non-NULL precedence is "newest
  // wins"; NULL is treated as "no opinion, leave existing".
  const upsertPhase = db.prepare(
    `INSERT INTO phase_completions (phase, last_completed, metadata)
     VALUES (?, ?, ?)
     ON CONFLICT(phase) DO UPDATE SET
       last_completed = excluded.last_completed,
       metadata       = COALESCE(excluded.metadata, phase_completions.metadata),
       updated_at     = CURRENT_TIMESTAMP`,
  );

  const stamp = migrationDateStamp();
  const fileLabel = 'heartbeat-state.json';

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'heartbeat-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU: file may disappear between `existsSync` and
      // `readFileSync` (concurrent migration run, manual cleanup).
      // Treat ENOENT as an idempotent info-level skip; every other
      // errno propagates per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'heartbeat-state.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(raw, folder, fileLabel, summary);
    if (parsed === null) continue;

    // Phase-row plan. The shape is deliberately a small array so the
    // three branches share one loop and one transaction — adding a
    // future phase (e.g. `composio` split out of the heartbeat blob)
    // is a one-line append, not a fourth branch to keep in sync.
    //
    // `metadataObj` is the writer-decides convention from the
    // state-009 doc-header: today only `heartbeat` carries an extra
    // (`last_composio_check`); `nightly` and `weekly` stay NULL. A
    // reader doesn't infer "no extras" from "{}" — absence means
    // "no extras to record".
    const composio = (parsed as HeartbeatStateJsonShape).last_composio_check;
    const phases: ReadonlyArray<{
      phase: string;
      tsKey: keyof HeartbeatStateJsonShape;
      metadataObj: Record<string, unknown> | null;
    }> = [
      {
        phase: 'heartbeat',
        tsKey: 'heartbeat_last_completed',
        metadataObj:
          typeof composio === 'string' && composio.length > 0
            ? { last_composio_check: composio }
            : null,
      },
      { phase: 'nightly', tsKey: 'nightly_last_completed', metadataObj: null },
      { phase: 'weekly', tsKey: 'weekly_last_completed', metadataObj: null },
    ];

    // Wrap the per-file work in a single transaction so a partial
    // failure on one of the three phase UPSERTs rolls all of them
    // back. Without the transaction wrapper, a crash mid-import (e.g.
    // a future CHECK constraint violation on the `weekly` row) would
    // leave the table with `heartbeat` / `nightly` committed and
    // `weekly` missing — the operator would then have to triage a
    // half-migrated state.
    const importFile = db.transaction(() => {
      let importedPhases = 0;
      for (const { phase, tsKey, metadataObj } of phases) {
        const ts = (parsed as HeartbeatStateJsonShape)[tsKey];
        if (typeof ts !== 'string' || ts.length === 0) continue;
        upsertPhase.run(
          phase,
          ts,
          metadataObj === null ? null : JSON.stringify(metadataObj),
        );
        importedPhases++;
      }
      return { importedPhases };
    });

    let counts: { importedPhases: number };
    try {
      counts = importFile();
    } catch (err) {
      // The helper either returns true (caller continues) or rethrows;
      // there's no third path. Match the calendar-state pattern.
      handleConstraintViolationOrRethrow(err, folder, fileLabel, summary);
      continue;
    }

    renameMigratedSource(filePath, stamp, folder, fileLabel, counts, summary);
  }
  return summary;
}

interface TaskTzStateFollowMeTaskJson {
  name: string;
  // Modern shape (matches the state-010 spec docstring): wall-clock as
  // an `"HH:MM"` string and the cron expression duplicated on the row.
  local_time?: string;
  schedule_value?: string;
  // Legacy shape — the pre-state-010 `task-tz-sync` writer split the
  // wall-clock into integer hour/minute and never duplicated the cron
  // (the cron lives on the sibling `scheduled_tasks` row keyed by
  // `task_id`). When state-010 shipped, in-place JSONs hadn't been
  // rewritten yet (the writer hasn't fired since 2026-04-25) so the
  // migration has to translate this shape into the schema's required
  // `local_time` / `schedule_value` columns. See #431.
  task_id?: string;
  local_hour?: number;
  local_minute?: number;
  last_run_date?: string | null;
  pending_run_at?: string | null;
}

interface TaskTzStateJsonShape {
  current_tz?: string;
  home_tz?: string;
  scheduler_tz?: string | null;
  follow_me_tasks?: TaskTzStateFollowMeTaskJson[];
}

const TASK_TZ_STATE_FILE_LABEL = 'task-tz-state.json';

/**
 * Resolve `local_time` and `schedule_value` for a follow_me row that
 * may carry either the modern state-010 shape or the legacy
 * `local_hour`/`local_minute` + `task_id`-keyed cron shape (#431).
 *
 * Returns null with a warn-log when neither shape supplies enough data
 * to populate the NOT NULL columns; the caller then treats the row as
 * skipped rather than letting the whole transaction roll back via a
 * SQLite constraint violation.
 *
 * `lookupCron` takes a `task_id` and returns the matching
 * `scheduled_tasks.schedule_value` (or undefined). Injected as a
 * callback so the helper stays decoupled from the prepared-statement
 * lifetime owned by `migrateTaskTzStateJsonFiles`.
 */
function resolveFollowMeTaskShape(
  task: TaskTzStateFollowMeTaskJson,
  folder: string,
  lookupCron: (taskId: string) => { schedule_value?: string } | undefined,
): { local_time: string; schedule_value: string } | null {
  let resolvedLocalTime: string | null = null;
  if (typeof task.local_time === 'string' && task.local_time.length > 0) {
    resolvedLocalTime = task.local_time;
  } else if (
    typeof task.local_hour === 'number' &&
    Number.isInteger(task.local_hour) &&
    task.local_hour >= 0 &&
    task.local_hour <= 23 &&
    typeof task.local_minute === 'number' &&
    Number.isInteger(task.local_minute) &&
    task.local_minute >= 0 &&
    task.local_minute <= 59
  ) {
    resolvedLocalTime = `${String(task.local_hour).padStart(2, '0')}:${String(task.local_minute).padStart(2, '0')}`;
  }

  let resolvedScheduleValue: string | null = null;
  if (
    typeof task.schedule_value === 'string' &&
    task.schedule_value.length > 0
  ) {
    resolvedScheduleValue = task.schedule_value;
  } else if (typeof task.task_id === 'string' && task.task_id.length > 0) {
    const row = lookupCron(task.task_id);
    if (
      row &&
      typeof row.schedule_value === 'string' &&
      row.schedule_value.length > 0
    ) {
      resolvedScheduleValue = row.schedule_value;
    }
  }

  if (resolvedLocalTime === null || resolvedScheduleValue === null) {
    // Booleans report *usable* presence (non-empty string / valid integer)
    // rather than just `typeof`-truthy, so a row that carried `local_time:
    // ""` or `schedule_value: ""` doesn't surface as `hasLocalTime: true`
    // when the resolver has effectively rejected it. Operators reading
    // these warns during triage need "could the value actually be used?"
    // not "does the JSON happen to contain that key?".
    logger.warn(
      {
        folder,
        taskName: task.name,
        taskId: task.task_id,
        hasLocalTime:
          typeof task.local_time === 'string' && task.local_time.length > 0,
        hasLocalHour:
          typeof task.local_hour === 'number' &&
          Number.isInteger(task.local_hour) &&
          task.local_hour >= 0 &&
          task.local_hour <= 23,
        hasLocalMinute:
          typeof task.local_minute === 'number' &&
          Number.isInteger(task.local_minute) &&
          task.local_minute >= 0 &&
          task.local_minute <= 59,
        hasScheduleValue:
          typeof task.schedule_value === 'string' &&
          task.schedule_value.length > 0,
        scheduleLookupHit: resolvedScheduleValue !== null,
      },
      `${TASK_TZ_STATE_FILE_LABEL} migration: cannot resolve local_time / schedule_value for follow_me row, skipping row (the per-file rename still happens once the transaction commits — partial imports are normal)`,
    );
    return null;
  }
  return {
    local_time: resolvedLocalTime,
    schedule_value: resolvedScheduleValue,
  };
}

/**
 * Migrate per-group `task-tz-state.json` files (#302, data-import
 * follow-up to schema PR #348). Mirrors the orders / morning-brief /
 * calendar-state pattern: scans every `groups/<name>/task-tz-state.
 * json`, parses the envelope, and writes the three timezone scalars
 * to the singleton `tz_state` row plus N rows to `follow_me_tasks`
 * inside a single transaction.
 *
 * Both writes use `ON CONFLICT(...) DO UPDATE` (UPSERT) — NEVER
 * `INSERT OR REPLACE`, which is delete+insert in SQLite and would
 * silently reset defaulted columns like `schema_version` on every
 * re-import. The multi-group test in
 * `task-tz-state-json-migration.test.ts` pins this down by manually
 * bumping `schema_version` between two group imports and asserting
 * the writer's known shape (`SUPPORTED_TZ_STATE_SCHEMA_VERSION`,
 * currently 3 post-jbaruch/nanoclaw-admin#229) is what lands on the
 * second import — anything else (1 or the manually-bumped value)
 * would signal either a `INSERT OR REPLACE` regression (1) or that
 * the writer dropped its explicit `schema_version` bind (manual
 * bump survives, gate rejects the row).
 *
 * `tz_state` UPSERT writes `schema_version` explicitly through
 * `SUPPORTED_TZ_STATE_SCHEMA_VERSION` (3 post-#229; was 2 between
 * #542 and #229): a fresh-DB import would otherwise land at the
 * state-010 column default of 1, and the reader gate would reject
 * the imported row as "unfamiliar schema_version" until
 * `applyTripitSegmentsToTzState` rewrote it on the next nightly
 * `sync_tripit` run. The explicit bind keeps the JSON-import's row
 * shape coherent with the writer gate from the moment it lands.
 *
 * `follow_me_tasks` UPSERT re-stamps `updated_at = CURRENT_TIMESTAMP`
 * in the conflict branch so a re-import shows up as a fresh row
 * mutation in the audit log.
 *
 * Skip-and-warn when `current_tz` or `home_tz` is missing/empty —
 * both are `TEXT NOT NULL` on `tz_state`, so attempting the insert
 * would throw a constraint violation. We catch it before the
 * transaction starts so the file stays in place for triage and the
 * `follow_me_tasks` rows below also don't import (the JSON envelope
 * is malformed; partial import would be misleading).
 *
 * Idempotent via the standard `.migrated-YYYY-MM-DD` rename.
 */
function migrateTaskTzStateJsonFiles(): MigrationSummary {
  const summary = newMigrationSummary('task-tz-state');
  const groupFolders = listGroupFoldersForMigration();
  if (groupFolders.length === 0) return summary;

  // tz_state UPSERT: writes `schema_version` explicitly via
  // `SUPPORTED_TZ_STATE_SCHEMA_VERSION` so the backfilled row matches
  // the reader gate. Without the explicit bind, a fresh-DB import
  // would land at the state-010 column default of 1 — state-012's
  // `UPDATE … WHERE id = 1` runs before this migration but is a
  // no-op when the row doesn't yet exist, and every reader would
  // then reject the imported row as "unfamiliar schema_version".
  // `segments` stays NULL on import — the column gets populated on
  // the next `sync_tripit` run via `applyTripitSegmentsToTzState`.
  // CHECK(id=1) makes tz_state a true singleton — the second
  // group's import updates the same row in place rather than landing
  // id=2 (which would also fail loudly via the CHECK).
  const upsertTzState = db.prepare(
    `INSERT INTO tz_state (id, current_tz, home_tz, scheduler_tz, schema_version)
     VALUES (1, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       current_tz     = excluded.current_tz,
       home_tz        = excluded.home_tz,
       scheduler_tz   = excluded.scheduler_tz,
       schema_version = excluded.schema_version`,
  );

  // follow_me_tasks UPSERT: re-stamp `updated_at` on conflict so the
  // audit log captures the re-import as a fresh row mutation. PK is
  // `name`; sibling rows (other task names) are untouched.
  const upsertFollowMeTask = db.prepare(
    `INSERT INTO follow_me_tasks
       (name, local_time, schedule_value, last_run_date, pending_run_at)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(name) DO UPDATE SET
       local_time     = excluded.local_time,
       schedule_value = excluded.schedule_value,
       last_run_date  = excluded.last_run_date,
       pending_run_at = excluded.pending_run_at,
       updated_at     = CURRENT_TIMESTAMP`,
  );

  // Legacy-shape fallback for `schedule_value`: pre-state-010 follow_me
  // entries don't carry the cron string — it lives on the sibling
  // `scheduled_tasks` row keyed by `task_id`. SELECT-by-PK on a tiny
  // table per legacy row is cheap; the modern-shape path never reaches
  // this query because it short-circuits on `task.schedule_value`.
  const lookupScheduledTaskCron = db.prepare(
    `SELECT schedule_value FROM scheduled_tasks WHERE id = ?`,
  );

  const stamp = migrationDateStamp();

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'task-tz-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort; the file may
      // disappear before readFileSync. Mirror the orders / morning-
      // brief handling — info-log on ENOENT, propagate every other
      // errno per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          `${TASK_TZ_STATE_FILE_LABEL} migration: file disappeared between existsSync and readFileSync, skipping`,
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(
      raw,
      folder,
      TASK_TZ_STATE_FILE_LABEL,
      summary,
    ) as TaskTzStateJsonShape | null;
    if (parsed === null) continue;

    // Both `current_tz` and `home_tz` are NOT NULL on `tz_state`. An
    // empty string would satisfy NOT NULL but break every reader
    // (morning-brief, nightly, weekly, check-calendar, heartbeat-
    // precheck) that uses the value as an IANA zone name. Treat
    // missing / non-string / empty all as "malformed envelope": warn,
    // leave the file in place for triage, and DON'T import the
    // sibling `follow_me_tasks` rows (a partial import would silently
    // ship N task rows without their tz context).
    if (
      typeof parsed.current_tz !== 'string' ||
      parsed.current_tz.length === 0 ||
      typeof parsed.home_tz !== 'string' ||
      parsed.home_tz.length === 0
    ) {
      logger.warn(
        {
          folder,
          hasCurrentTz: typeof parsed.current_tz === 'string',
          hasHomeTz: typeof parsed.home_tz === 'string',
        },
        `${TASK_TZ_STATE_FILE_LABEL} migration: missing or empty current_tz / home_tz, skipping (file left in place)`,
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const currentTz = parsed.current_tz;
    const homeTz = parsed.home_tz;
    // `scheduler_tz` is nullable on the schema (informational only;
    // not load-bearing per the schema-doc). Coerce missing /
    // non-string / empty to NULL.
    const schedulerTz =
      typeof parsed.scheduler_tz === 'string' && parsed.scheduler_tz.length > 0
        ? parsed.scheduler_tz
        : null;

    // `follow_me_tasks`: distinguish "missing" (key absent) from
    // "wrong-typed" (key present but not an array — e.g. an object).
    // Missing is the legitimate "no follow-me jobs configured"
    // envelope; we still land tz_state and rename. Wrong-typed is a
    // corruption signal: warn (with the rejected type) and skip the
    // entire group's file so the original task payload survives for
    // human triage instead of being silently discarded by the rename.
    const followMeRaw: unknown = parsed.follow_me_tasks;
    let followMeTasks: TaskTzStateFollowMeTaskJson[];
    if (followMeRaw === undefined) {
      followMeTasks = [];
    } else if (Array.isArray(followMeRaw)) {
      followMeTasks = followMeRaw as TaskTzStateFollowMeTaskJson[];
    } else {
      logger.warn(
        {
          folder,
          followMeType: followMeRaw === null ? 'null' : typeof followMeRaw,
        },
        `${TASK_TZ_STATE_FILE_LABEL} migration: follow_me_tasks is not an array, skipping (file left in place)`,
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    // The UPSERT counter splits inserts (first import for this
    // group) from refreshes (subsequent groups' singleton refresh)
    // because SQLite returns `changes = 1` for both branches —
    // labelling the combined number `tz_state_inserted` would be
    // misleading on the multi-group / re-run path. Use SELECT-then-
    // UPSERT to distinguish: if the singleton row already exists,
    // count as a refresh; otherwise, an insert.
    const tzStateExistsBefore =
      (
        db.prepare('SELECT COUNT(*) AS n FROM tz_state WHERE id = 1').get() as {
          n: number;
        }
      ).n > 0;
    const counts = {
      tz_state_inserted: 0,
      tz_state_refreshed: 0,
      follow_me_upserted: 0,
      skipped: 0,
    };
    try {
      const importFile = db.transaction(() => {
        const tzResult = upsertTzState.run(
          currentTz,
          homeTz,
          schedulerTz,
          SUPPORTED_TZ_STATE_SCHEMA_VERSION,
        );
        if (tzResult.changes > 0) {
          if (tzStateExistsBefore) counts.tz_state_refreshed++;
          else counts.tz_state_inserted++;
        }
        for (const task of followMeTasks) {
          if (!isObjectRow(task)) {
            logger.warn(
              { folder, queue: 'follow_me_tasks' },
              `${TASK_TZ_STATE_FILE_LABEL} migration: skipping non-object row`,
            );
            counts.skipped++;
            continue;
          }
          // Validate `name` separately from the local_time / schedule_value
          // resolver: PK on `follow_me_tasks` is `name TEXT PRIMARY KEY`, so
          // a missing / non-string / empty `name` would constraint-violate
          // at the upsert step and roll back the entire group's transaction
          // — exactly the failure mode the per-row-skip path was added to
          // avoid (#431). Skip-and-warn here keeps the per-row contract
          // intact for siblings.
          if (typeof task.name !== 'string' || task.name.length === 0) {
            logger.warn(
              {
                folder,
                taskId: (task as TaskTzStateFollowMeTaskJson).task_id,
                nameType:
                  task.name === null
                    ? 'null'
                    : typeof (task as { name?: unknown }).name,
              },
              `${TASK_TZ_STATE_FILE_LABEL} migration: skipping follow_me row with missing or non-string name (PK)`,
            );
            counts.skipped++;
            continue;
          }
          // Resolve `local_time` and `schedule_value` from either the
          // modern shape (state-010 spec) or the legacy shape
          // (`local_hour`/`local_minute` integers + `task_id` keying
          // the cron on the sibling `scheduled_tasks` row). Per-row
          // skip-and-warn rather than a transaction-wide rollback so
          // one malformed row doesn't strand the whole group's import
          // — that was the failure mode #431 hit on the live deployment.
          const resolved = resolveFollowMeTaskShape(
            task as TaskTzStateFollowMeTaskJson,
            folder,
            (taskId) =>
              lookupScheduledTaskCron.get(taskId) as
                | { schedule_value?: string }
                | undefined,
          );
          if (resolved === null) {
            counts.skipped++;
            continue;
          }
          const result = upsertFollowMeTask.run(
            task.name as string,
            resolved.local_time,
            resolved.schedule_value,
            // Both nullable cursor fields use `?? null` so an
            // explicit `null` and a missing key both round-trip as
            // SQL NULL. The reader contract on state-010 explicitly
            // tolerates NULL on both columns.
            (task.last_run_date as string | null | undefined) ?? null,
            (task.pending_run_at as string | null | undefined) ?? null,
          );
          if (result.changes > 0) counts.follow_me_upserted++;
          else counts.skipped++;
        }
      });
      importFile();
    } catch (err) {
      // Per `coding-policy: error-handling`: only constraint-class
      // SqliteError is recoverable here — those are the per-file
      // data-quality failures the per-file isolation contract was
      // written for. Anything else (TypeError, non-constraint
      // SqliteError like SQLITE_CORRUPT/BUSY) propagates so the
      // operator sees the real failure rather than a swept-under-
      // the-rug warn. Transaction already rolled back on throw, so
      // no partial rows landed in either case.
      if (
        handleConstraintViolationOrRethrow(
          err,
          folder,
          TASK_TZ_STATE_FILE_LABEL,
          summary,
        )
      )
        continue;
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      TASK_TZ_STATE_FILE_LABEL,
      {
        tz_state_inserted: counts.tz_state_inserted,
        tz_state_refreshed: counts.tz_state_refreshed,
        follow_me_upserted: counts.follow_me_upserted,
        skipped: counts.skipped,
        total: followMeTasks.length,
      },
      summary,
    );
  }
  return summary;
}

/**
 * #298 — Migrate per-group `session-state.json` (the multi-writer
 * trusted-memory state file) into `trusted_sessions` +
 * `trusted_session_singleton`. Owner skill: `tessl__trusted-memory`.
 *
 * Source shape (documented on the state-006 doc-header):
 *
 *   {
 *     "schema_version": 1,
 *     "sessions": {"<NANOCLAW_SESSION_NAME>": {
 *        "started", "epoch", "session_id", "last_seen"
 *     }, ...},
 *     "active_session_id": "<top-level back-compat>",
 *     "seen_email_ids": [...],     // NOT migrated here — see below.
 *     "pending_response": {...} | "<string>",
 *     "muted_threads": [...]
 *   }
 *
 * The JSON-era top-level `seen_email_ids` field intentionally does
 * NOT migrate here — that field relocates to the `email_seen_ids`
 * table created by state-005 (#297) where both check-email writers
 * can target it without the old two-file consolidate dance. Don't
 * touch it from this migration; #297's own data-import PR owns that
 * row backfill.
 *
 * Per-named-session entries become `trusted_sessions` rows (UPSERT
 * by `session_name` so a re-run with a moved file in some other
 * group folder won't clobber per-session metadata). The singleton
 * fields become a single `trusted_session_singleton` row at id=1
 * (UPSERT, not INSERT OR REPLACE — REPLACE deletes the existing
 * row and re-inserts, which would reset `schema_version` to its
 * column DEFAULT and mask future migrations).
 *
 * `pending_response` and `muted_threads` are stored as TEXT in the
 * schema; the owner skill treats them as opaque JSON blobs. Stringify
 * here so a structured object/array on disk round-trips through the
 * column without losing shape.
 */
function migrateTrustedSessionStateJsonFiles(): MigrationSummary {
  const summary = newMigrationSummary('session-state');
  const groupFolders = listGroupFoldersForMigration();

  // UPSERT, not INSERT OR REPLACE: REPLACE deletes the conflicting
  // row and re-inserts, which would reset `schema_version` to its
  // column DEFAULT(=1). When we later bump trusted_sessions'
  // `schema_version` for a shape change, REPLACE-on-import would
  // silently roll back any post-migration upgrade the owner skill
  // had performed. ON CONFLICT(session_name) DO UPDATE preserves the
  // existing `schema_version` while letting the four data fields
  // refresh.
  const upsertSession = db.prepare(
    `INSERT INTO trusted_sessions
       (session_name, session_id, started, epoch, last_seen)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(session_name) DO UPDATE SET
       session_id = excluded.session_id,
       started    = excluded.started,
       epoch      = excluded.epoch,
       last_seen  = excluded.last_seen`,
  );

  // Same UPSERT-not-REPLACE rationale for the singleton: re-run with
  // a second group's file (multi-host migration order) UPSERTs the
  // existing id=1 row in place, so the row count stays at 1 and the
  // existing `schema_version` is preserved across re-runs.
  const upsertSingleton = db.prepare(
    `INSERT INTO trusted_session_singleton
       (id, active_session_id, pending_response, muted_threads)
     VALUES (1, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       active_session_id = excluded.active_session_id,
       pending_response  = excluded.pending_response,
       muted_threads     = excluded.muted_threads`,
  );

  const stamp = migrationDateStamp();
  const fileLabel = 'session-state.json';

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'session-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort; the file may
      // disappear before readFileSync. Treat ENOENT here the same as
      // ENOENT at rename time — idempotent no-op, log info, continue.
      // Every other errno propagates per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          `${fileLabel} migration: file disappeared between existsSync and readFileSync, skipping`,
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(raw, folder, fileLabel, summary);
    if (!parsed) continue;

    const sessionsField = parsed.sessions;
    const hasSessions =
      sessionsField !== null &&
      typeof sessionsField === 'object' &&
      !Array.isArray(sessionsField);
    const hasSingleton =
      'active_session_id' in parsed ||
      'pending_response' in parsed ||
      'muted_threads' in parsed;

    if (!hasSessions && !hasSingleton) {
      logger.warn(
        { folder },
        `${fileLabel} migration: no recognised fields (sessions / active_session_id / pending_response / muted_threads), skipping`,
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const sessionCounts = { upserted: 0, skipped: 0, total: 0 };
    let singletonUpserted = false;
    try {
      const importFile = db.transaction(() => {
        if (hasSessions) {
          const sessions = sessionsField as Record<string, unknown>;
          const entries = Object.entries(sessions);
          sessionCounts.total = entries.length;
          for (const [sessionName, entry] of entries) {
            if (!isObjectRow(entry)) {
              logger.warn(
                { folder, session_name: sessionName },
                `${fileLabel} migration: skipping non-object session entry`,
              );
              sessionCounts.skipped++;
              continue;
            }
            // `started` / `epoch` / `last_seen` are NOT NULL in the
            // schema. A JSON-era row that pre-dates the field — e.g.
            // an old back-compat shape that only tracked `session_id`
            // — would otherwise throw NOT NULL inside the transaction
            // and abort the whole file's import. Skip-with-warn so the
            // remaining session entries (and the singleton) still
            // import. `session_id` is nullable per the state-006
            // schema (sqlite-error fallback path) so its absence is
            // valid.
            if (
              typeof entry.started !== 'string' ||
              typeof entry.epoch !== 'number' ||
              typeof entry.last_seen !== 'string'
            ) {
              logger.warn(
                {
                  folder,
                  session_name: sessionName,
                  has_started: typeof entry.started === 'string',
                  has_epoch: typeof entry.epoch === 'number',
                  has_last_seen: typeof entry.last_seen === 'string',
                },
                `${fileLabel} migration: session entry missing required fields (started/epoch/last_seen), skipping`,
              );
              sessionCounts.skipped++;
              continue;
            }
            upsertSession.run(
              sessionName,
              typeof entry.session_id === 'string' ? entry.session_id : null,
              entry.started,
              entry.epoch,
              entry.last_seen,
            );
            sessionCounts.upserted++;
          }
        }
        if (hasSingleton) {
          const activeSessionId =
            typeof parsed.active_session_id === 'string'
              ? parsed.active_session_id
              : null;
          // pending_response and muted_threads are TEXT in the schema
          // (opaque JSON blobs per the owner-skill contract). Stringify
          // structured shapes; pass strings through verbatim; treat
          // missing/null as NULL.
          const pendingResponse =
            parsed.pending_response === undefined ||
            parsed.pending_response === null
              ? null
              : typeof parsed.pending_response === 'string'
                ? parsed.pending_response
                : JSON.stringify(parsed.pending_response);
          const mutedThreads =
            parsed.muted_threads === undefined || parsed.muted_threads === null
              ? null
              : typeof parsed.muted_threads === 'string'
                ? parsed.muted_threads
                : JSON.stringify(parsed.muted_threads);
          upsertSingleton.run(activeSessionId, pendingResponse, mutedThreads);
          singletonUpserted = true;
        }
      });
      importFile();
    } catch (err) {
      if (handleConstraintViolationOrRethrow(err, folder, fileLabel, summary))
        continue;
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      fileLabel,
      {
        sessions: sessionCounts,
        singleton: singletonUpserted,
      },
      summary,
    );
  }
  return summary;
}

interface NanoclawStateResumableCycleJson {
  cycle_id?: unknown;
  slot_key?: unknown;
  continuation_n?: unknown;
  remaining_steps?: unknown;
}

interface NanoclawStateJsonShape {
  last_email_checked?: unknown;
  date?: unknown;
  fetched_at?: unknown;
  seen_email_ids?: unknown;
  resumable_cycles?: unknown;
}

/**
 * Migrate per-group `nanoclaw-state.json` files (#297) into the three
 * SQLite tables created by state-005:
 *
 *   - `email_state`        — singleton row (`id = 1`) holding the email-
 *                            cursor fields (`last_email_checked`, `date`,
 *                            `fetched_at`).
 *   - `email_seen_ids`     — append-mostly dedup set; one row per id from
 *                            the JSON `seen_email_ids` array.
 *   - `resumable_cycles`   — one row per skill_name from the JSON
 *                            `resumable_cycles.<skill_name>` subtree.
 *
 * The JSON-era shape was a multi-writer junk drawer (the exact bug class
 * #293 targets), so per-file isolation matters: a single malformed source
 * must not abort the pass for other groups. Each per-file work is wrapped
 * in `db.transaction` so a row violating a NOT NULL / CHECK / PK
 * constraint inside the writer rolls the whole file's import back; the
 * narrowed `handleConstraintViolationOrRethrow` catch turns the throw
 * into a per-file warn and leaves the source file in place for triage.
 *
 * UPSERT semantics:
 *   - `email_state` is a singleton (`CHECK(id = 1)`) so we use
 *     `INSERT … ON CONFLICT(id) DO UPDATE SET …` — NOT `INSERT OR
 *     REPLACE`, which would silently nuke the existing `schema_version`
 *     column instead of preserving it across multi-group re-import. Each
 *     group's nanoclaw-state.json contributes the same singleton row;
 *     under the deterministic folder sort, the last writer (alphabetical
 *     order) wins on the cursor fields. (In practice each install has at
 *     most one nanoclaw-state.json source, so this only matters for
 *     defensive behaviour during multi-group migration.)
 *   - `email_seen_ids` uses `INSERT … ON CONFLICT(email_id) DO NOTHING` —
 *     re-runs are silent no-ops on duplicate ids (the table is a dedup
 *     set; re-importing the same id should not bump `seen_at`).
 *   - `resumable_cycles` uses `INSERT … ON CONFLICT(skill_name) DO
 *     UPDATE SET …` — the latest source-file shape wins for each skill,
 *     same alphabetical-folder-sort tie-break as `email_state`.
 *
 * Per-column policy on missing fields: if a top-level cursor field
 * (`last_email_checked` / `date` / `fetched_at`) is absent, omit it from
 * the INSERT column list so the schema's column default (NULL on these
 * three) fires rather than binding `null` ourselves. Same approach the
 * morning-brief import uses for its `added` column with
 * `DEFAULT CURRENT_TIMESTAMP`. For `email_state` the three cursor
 * columns share the same nullable semantics, but documenting the
 * pattern keeps the writer aligned with the schema-default contract for
 * the wider epic. See state-005 doc-header for the rationale on UPSERT
 * vs INSERT OR REPLACE and on the `strftime` defaults.
 */
function migrateNanoclawStateJsonFiles(): MigrationSummary {
  const summary = newMigrationSummary('nanoclaw-state');
  const groupFolders = listGroupFoldersForMigration();
  const stamp = migrationDateStamp();

  // email_state singleton UPSERT. Build the prepared statements lazily
  // per writer-column-set so we can omit absent cursor fields and let
  // the schema default fire. Building all four shapes up front keeps
  // the per-file path branchless.
  //
  // The DO UPDATE clause references `excluded.<col>` (SQLite's name for
  // the row that would have been inserted). Crucially it does NOT touch
  // `schema_version` — preserving the existing value across re-import,
  // which is the whole reason we pick UPSERT over INSERT OR REPLACE.
  const insertEmailStateAll = db.prepare(
    `INSERT INTO email_state (id, last_email_checked, date, fetched_at)
     VALUES (1, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       last_email_checked = excluded.last_email_checked,
       date               = excluded.date,
       fetched_at         = excluded.fetched_at`,
  );
  const insertEmailStateLastOnly = db.prepare(
    `INSERT INTO email_state (id, last_email_checked) VALUES (1, ?)
     ON CONFLICT(id) DO UPDATE SET last_email_checked = excluded.last_email_checked`,
  );
  const insertEmailStateDateOnly = db.prepare(
    `INSERT INTO email_state (id, date) VALUES (1, ?)
     ON CONFLICT(id) DO UPDATE SET date = excluded.date`,
  );
  const insertEmailStateFetchedOnly = db.prepare(
    `INSERT INTO email_state (id, fetched_at) VALUES (1, ?)
     ON CONFLICT(id) DO UPDATE SET fetched_at = excluded.fetched_at`,
  );
  const insertEmailStateLastDate = db.prepare(
    `INSERT INTO email_state (id, last_email_checked, date) VALUES (1, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       last_email_checked = excluded.last_email_checked,
       date               = excluded.date`,
  );
  const insertEmailStateLastFetched = db.prepare(
    `INSERT INTO email_state (id, last_email_checked, fetched_at) VALUES (1, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       last_email_checked = excluded.last_email_checked,
       fetched_at         = excluded.fetched_at`,
  );
  const insertEmailStateDateFetched = db.prepare(
    `INSERT INTO email_state (id, date, fetched_at) VALUES (1, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       date       = excluded.date,
       fetched_at = excluded.fetched_at`,
  );

  // email_seen_ids: ON CONFLICT(email_id) DO NOTHING — append-mostly
  // dedup set. Re-imports are silent no-ops on duplicates; we don't
  // bump `seen_at` since the JSON-era source carries no per-id
  // timestamp anyway.
  const insertSeenEmailId = db.prepare(
    `INSERT INTO email_seen_ids (email_id) VALUES (?)
     ON CONFLICT(email_id) DO NOTHING`,
  );

  // resumable_cycles: UPSERT by skill_name — the latest source-file
  // shape wins for each skill. `continuation_n` defaults to 0 in the
  // schema; we still bind explicitly (defaulting to 0 here too) so the
  // writer's column list is uniform across rows. `remaining_steps` is
  // a JSON blob (TEXT) — we store whatever the source shape carries
  // verbatim, including null.
  const upsertResumableCycle = db.prepare(
    `INSERT INTO resumable_cycles
       (skill_name, cycle_id, slot_key, continuation_n, remaining_steps)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(skill_name) DO UPDATE SET
       cycle_id        = excluded.cycle_id,
       slot_key        = excluded.slot_key,
       continuation_n  = excluded.continuation_n,
       remaining_steps = excluded.remaining_steps,
       updated_at      = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`,
  );

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'nanoclaw-state.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort. Mirror the orders
      // / morning-brief handling — log info on ENOENT, propagate every
      // other errno per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'nanoclaw-state.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    const parsed = parseJsonObjectOrWarn(
      raw,
      folder,
      'nanoclaw-state.json',
      summary,
    ) as NanoclawStateJsonShape | null;
    if (parsed === null) continue;

    // Pre-flight: warn for each top-level section that's missing or
    // wrong-shape. The contract is "import what we can" — a missing
    // resumable_cycles object doesn't block email_state or
    // email_seen_ids importing.
    const hasEmailCursorField =
      typeof parsed.last_email_checked === 'string' ||
      typeof parsed.date === 'string' ||
      typeof parsed.fetched_at === 'string';
    if (!hasEmailCursorField) {
      logger.warn(
        { folder },
        'nanoclaw-state.json migration: no email cursor fields (last_email_checked / date / fetched_at) at top level, skipping email_state',
      );
    }
    const seenEmailIds = parsed.seen_email_ids;
    if (!Array.isArray(seenEmailIds)) {
      logger.warn(
        {
          folder,
          seenType: Array.isArray(seenEmailIds) ? 'array' : typeof seenEmailIds,
        },
        'nanoclaw-state.json migration: seen_email_ids missing or not an array, skipping email_seen_ids',
      );
    }
    const resumableCyclesObj = parsed.resumable_cycles;
    if (!isObjectRow(resumableCyclesObj)) {
      logger.warn(
        {
          folder,
          cyclesType:
            resumableCyclesObj === null
              ? 'null'
              : Array.isArray(resumableCyclesObj)
                ? 'array'
                : typeof resumableCyclesObj,
        },
        'nanoclaw-state.json migration: resumable_cycles missing or not an object, skipping resumable_cycles',
      );
    }

    if (
      !hasEmailCursorField &&
      !Array.isArray(seenEmailIds) &&
      !isObjectRow(resumableCyclesObj)
    ) {
      // Nothing recognised — leave file in place for triage rather than
      // renaming a source we never actually imported.
      logger.warn(
        { folder },
        'nanoclaw-state.json migration: no recognised sections, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const counts = {
      email_state_upserted: 0,
      seen_ids_inserted: 0,
      seen_ids_skipped: 0,
      cycles_upserted: 0,
      cycles_skipped: 0,
    };

    try {
      const importFile = db.transaction(() => {
        // email_state singleton — only INSERT/UPSERT if at least one
        // cursor field is present. Pick the prepared statement
        // matching the present-fields combination so absent fields
        // are omitted from the column list (and the schema default
        // fires / column stays NULL) rather than binding null.
        if (hasEmailCursorField) {
          const lec =
            typeof parsed.last_email_checked === 'string'
              ? parsed.last_email_checked
              : null;
          const dt = typeof parsed.date === 'string' ? parsed.date : null;
          const fa =
            typeof parsed.fetched_at === 'string' ? parsed.fetched_at : null;
          if (lec !== null && dt !== null && fa !== null) {
            insertEmailStateAll.run(lec, dt, fa);
          } else if (lec !== null && dt !== null) {
            insertEmailStateLastDate.run(lec, dt);
          } else if (lec !== null && fa !== null) {
            insertEmailStateLastFetched.run(lec, fa);
          } else if (dt !== null && fa !== null) {
            insertEmailStateDateFetched.run(dt, fa);
          } else if (lec !== null) {
            insertEmailStateLastOnly.run(lec);
          } else if (dt !== null) {
            insertEmailStateDateOnly.run(dt);
          } else if (fa !== null) {
            insertEmailStateFetchedOnly.run(fa);
          }
          counts.email_state_upserted = 1;
        }

        // email_seen_ids — one row per string in the JSON array; per-row
        // type guard skips non-string entries (the email_id PK is
        // TEXT NOT NULL; binding e.g. a number would coerce silently).
        if (Array.isArray(seenEmailIds)) {
          for (const id of seenEmailIds) {
            if (typeof id !== 'string') {
              logger.warn(
                { folder, idType: id === null ? 'null' : typeof id },
                'nanoclaw-state.json migration: skipping non-string entry in seen_email_ids',
              );
              counts.seen_ids_skipped++;
              continue;
            }
            const result = insertSeenEmailId.run(id);
            if (result.changes > 0) counts.seen_ids_inserted++;
            else counts.seen_ids_skipped++;
          }
        }

        // resumable_cycles — one row per skill_name in the JSON object.
        // Per-row object guard skips stale null/string/number values
        // (a writer bug could plant those; without the guard a property
        // access would TypeError before any INSERT runs and propagate
        // as an unexpected error through the narrowed catch).
        if (isObjectRow(resumableCyclesObj)) {
          for (const [skillName, cycleRecord] of Object.entries(
            resumableCyclesObj,
          )) {
            if (!isObjectRow(cycleRecord)) {
              logger.warn(
                { folder, skillName },
                'nanoclaw-state.json migration: skipping non-object resumable_cycles entry',
              );
              counts.cycles_skipped++;
              continue;
            }
            const cycle = cycleRecord as NanoclawStateResumableCycleJson;
            // cycle_id and slot_key are NOT NULL on the schema; let the
            // constraint catch missing values (handled by the narrowed
            // catch below — file left in place for triage).
            const cycleId =
              typeof cycle.cycle_id === 'string' ? cycle.cycle_id : null;
            const slotKey =
              typeof cycle.slot_key === 'string' ? cycle.slot_key : null;
            const continuationN =
              typeof cycle.continuation_n === 'number'
                ? cycle.continuation_n
                : 0;
            // remaining_steps is a JSON blob — schema column is TEXT so
            // accept either a string (already-stringified) or stringify
            // an object/array on the way in. null stays null.
            let remainingSteps: string | null;
            if (
              cycle.remaining_steps === null ||
              cycle.remaining_steps === undefined
            ) {
              remainingSteps = null;
            } else if (typeof cycle.remaining_steps === 'string') {
              remainingSteps = cycle.remaining_steps;
            } else {
              remainingSteps = JSON.stringify(cycle.remaining_steps);
            }
            upsertResumableCycle.run(
              skillName,
              cycleId,
              slotKey,
              continuationN,
              remainingSteps,
            );
            counts.cycles_upserted++;
          }
        }
      });
      importFile();
    } catch (err) {
      // Per-file isolation: a constraint violation on any of the three
      // writers rolls the whole transaction back, the file stays put
      // for human triage, and we move on to the next group. Anything
      // else (TypeError, ReferenceError, non-constraint SqliteError)
      // propagates per `coding-policy: error-handling`.
      if (
        handleConstraintViolationOrRethrow(
          err,
          folder,
          'nanoclaw-state.json',
          summary,
        )
      ) {
        continue;
      }
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      'nanoclaw-state.json',
      counts,
      summary,
    );
  }
  return summary;
}

interface ScheduledReminderJson {
  event_id: string;
  title: string;
  utc_time: string;
  reminder_offset_min: number;
  task_id: string;
}

/**
 * Per-group import of `scheduled-reminders.json` into the
 * `scheduled_reminders` table created by state-004 (#296). The JSON-era
 * shape is the wrapped form `{ "reminders": [...] }` written by the
 * pre-MCP `append-scheduled-reminders.py` skill; a bare top-level array
 * is also accepted as the legacy fallback the same skill emitted in
 * earlier revisions. Anything else (object without `reminders`, primitive
 * payload, malformed JSON) is warn-and-skipped — the source file stays
 * in place for triage so an operator can fix it without losing data.
 *
 * `event_id` is the PK on the table and the natural dedup key on the
 * source side (each reminder corresponds to exactly one calendar event).
 * The INSERT uses `ON CONFLICT(event_id) DO NOTHING` so a re-run with
 * leftover rows (e.g. operator copied a partial DB back over an already-
 * imported one) is a silent no-op on the PK conflict, while NOT NULL
 * violations still throw and surface via the constraint-class catch
 * helper. Per-file work is wrapped in a single transaction so a
 * mid-import crash can't leave the table half-populated.
 */
function migrateScheduledRemindersJsonFiles(): MigrationSummary {
  const summary = newMigrationSummary('scheduled-reminders');
  const groupFolders = listGroupFoldersForMigration();

  const insertReminder = db.prepare(
    `INSERT INTO scheduled_reminders
       (event_id, title, utc_time, reminder_offset_min, task_id)
     VALUES (?, ?, ?, ?, ?)
     ON CONFLICT(event_id) DO NOTHING`,
  );

  const stamp = migrationDateStamp();
  const fileLabel = 'scheduled-reminders.json';

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'scheduled-reminders.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race: existsSync above is best-effort; the file may
      // disappear between the check and the read (concurrent migration
      // run, manual cleanup). Treat ENOENT as an idempotent no-op and
      // propagate every other errno per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          `${fileLabel} migration: file disappeared between existsSync and readFileSync, skipping`,
        );
        continue;
      }
      throw err;
    }

    // Accept two shapes: the wrapped `{ "reminders": [...] }` form
    // (preferred — the form `append-scheduled-reminders.py` settled on)
    // and a bare top-level array (legacy fallback emitted by earlier
    // revisions of the same skill). For the wrapped shape we delegate
    // to `parseJsonObjectOrWarn` so the malformed-JSON / non-object
    // warn shapes match every other per-group migration. For the bare-
    // array shape we parse inline because the helper (correctly) treats
    // top-level arrays as "not an object" and rejects them — here a
    // bare array is a documented legacy form we still consume.
    //
    // Malformed JSON, primitive payloads, and objects without a
    // `reminders` array are all warn-and-skipped (file left in place
    // for triage) per `coding-policy: error-handling`.
    let reminders: unknown[];
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      if (err instanceof SyntaxError) {
        // Re-route through the helper for the malformed-JSON warn so
        // log shape stays consistent across all per-group migrations.
        // Helper detects SyntaxError, emits the standard warn, returns
        // null; skip the folder.
        parseJsonObjectOrWarn(raw, folder, fileLabel, summary);
        continue;
      }
      // `JSON.parse` only throws SyntaxError on string input;
      // anything else is a programming bug. Propagate per
      // `coding-policy: error-handling`.
      throw err;
    }
    if (Array.isArray(parsed)) {
      reminders = parsed;
    } else if (
      parsed !== null &&
      typeof parsed === 'object' &&
      Array.isArray((parsed as Record<string, unknown>).reminders)
    ) {
      reminders = (parsed as { reminders: unknown[] }).reminders;
    } else if (
      parsed !== null &&
      typeof parsed === 'object' &&
      !Array.isArray(parsed)
    ) {
      logger.warn(
        { folder },
        `${fileLabel} migration: missing "reminders" array, skipping (file left in place)`,
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    } else {
      // Primitive payload (null / number / string / boolean). Use the
      // helper so the warn carries the same `parsedType` field shape as
      // every other per-group migration.
      parseJsonObjectOrWarn(raw, folder, fileLabel, summary);
      continue;
    }

    // Single transaction per file so a mid-import crash can't leave the
    // table half-populated. Per-row object guard skips stale primitives
    // (null/string/number) before the bind throws a TypeError that the
    // narrowed catch would otherwise propagate as "unexpected".
    const counts = { inserted: 0, skipped: 0, total: reminders.length };
    try {
      const importFile = db.transaction(() => {
        for (const reminder of reminders) {
          if (!isObjectRow(reminder)) {
            logger.warn(
              { folder },
              `${fileLabel} migration: skipping non-object row`,
            );
            counts.skipped++;
            continue;
          }
          const row = reminder as unknown as ScheduledReminderJson;
          const result = insertReminder.run(
            row.event_id,
            row.title,
            row.utc_time,
            row.reminder_offset_min,
            row.task_id,
          );
          if (result.changes > 0) counts.inserted++;
          else counts.skipped++;
        }
      });
      importFile();
    } catch (err) {
      if (handleConstraintViolationOrRethrow(err, folder, fileLabel, summary))
        continue;
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      fileLabel,
      {
        inserted: counts.inserted,
        skipped: counts.skipped,
        total: counts.total,
      },
      summary,
    );
  }
  return summary;
}

// --- email-feedback.json → email_feedback (#295) ---

/**
 * Per-group migration: read each group's `email-feedback.json` and
 * append every well-formed row into the `email_feedback` SQLite table
 * created by state-002 (+ state-003 added the per-record
 * `schema_version` column). The JSON-era shape carried by
 * `nanoclaw-admin/skills/brief-cleanup` evolved across two forms:
 *
 *   - **Wrapped (preferred):** `{"feedback": [ { pattern, label,
 *     source, date }, ... ]}` — what the SKILL Step 6 helper
 *     `append-feedback.py` writes today.
 *   - **Bare array (legacy):** `[ { pattern, label, source, date },
 *     ... ]` — observed in production where earlier writers omitted
 *     the wrapper. Issue #295's body documents this shape verbatim.
 *
 * Both shapes are accepted; bare arrays are treated as if they were
 * `{feedback: <array>}`. Anything else (null, number, string, plain
 * object missing the `feedback` key, malformed JSON) is warned-and-
 * skipped per `coding-policy: error-handling` ("try alternatives
 * before failing"); the source file stays in place for human triage.
 *
 * Append-only contract: the schema's `id INTEGER PRIMARY KEY
 * AUTOINCREMENT` is assigned by SQLite, the writer never supplies it,
 * and there is no natural-key uniqueness to deduplicate on. So no
 * `ON CONFLICT` clause — every well-formed row inserts. Idempotency
 * comes from the rename: once the source becomes
 * `email-feedback.json.migrated-<YYYY-MM-DD>`, the existsSync gate at
 * the top of the loop skips it on subsequent boots. Re-running after
 * a fresh JSON has been dropped over an already-imported DB would
 * double-insert; that's the operator's problem, documented at the
 * dispatch site in `migrateJsonState()`.
 *
 * Per-row missing-required handling: rows lacking `pattern`, `label`,
 * or `date` are skipped with a warn rather than letting the schema's
 * NOT NULL throw — the warn carries the field name so triage can
 * grep for the specific failure class. The schema's
 * `CHECK(label IN ('actionable', 'noise'))` violation still throws as
 * a `SqliteError` with a `SQLITE_CONSTRAINT_CHECK` code; that's
 * caught by `handleConstraintViolationOrRethrow` so the per-file
 * transaction rolls back and the source file stays put.
 *
 * The `source` column has a DDL DEFAULT of `'baruch-response'`. When
 * a JSON-era row omits `source`, we omit that column from the INSERT
 * (rather than passing `null`, which the NOT NULL constraint would
 * reject) so the schema default fires. Matches the morning-brief
 * migration's `added`/CURRENT_TIMESTAMP pattern.
 *
 * The `schema_version` column (added by state-003) has a DDL DEFAULT
 * of `1`. Every JSON-era row was written under contract v1, so we
 * always omit the column from the INSERT and let the default fire —
 * no per-row stamping needed at migration time.
 */
function migrateEmailFeedbackJsonFiles(): MigrationSummary {
  const summary = newMigrationSummary('email-feedback');
  const groupFolders = listGroupFoldersForMigration();
  if (groupFolders.length === 0) return summary;

  // Two prepared statements: one with `source`, one without, so the
  // schema default fires when the JSON-era row omitted the field.
  // Same pattern as the morning-brief migration's
  // `withAdded`/`DefaultAdded` split.
  const insertWithSource = db.prepare(
    `INSERT INTO email_feedback (pattern, label, source, date)
     VALUES (?, ?, ?, ?)`,
  );
  const insertDefaultSource = db.prepare(
    `INSERT INTO email_feedback (pattern, label, date)
     VALUES (?, ?, ?)`,
  );

  const stamp = migrationDateStamp();

  for (const folder of groupFolders) {
    const filePath = path.join(GROUPS_DIR, folder, 'email-feedback.json');
    if (!fs.existsSync(filePath)) {
      if (hasMigratedSibling(filePath)) summary.skippedAlreadyDone += 1;
      continue;
    }

    let raw: string;
    try {
      raw = fs.readFileSync(filePath, 'utf-8');
    } catch (err) {
      // TOCTOU race between existsSync and readFileSync — file
      // disappeared. Idempotent no-op; every other errno propagates
      // per `coding-policy: error-handling`.
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        logger.info(
          { folder, filePath },
          'email-feedback.json migration: file disappeared between existsSync and readFileSync, skipping',
        );
        continue;
      }
      throw err;
    }

    // Two-shape parse: bare array (legacy) and `{feedback: [...]}`
    // (preferred). The shared `parseJsonObjectOrWarn` helper would
    // reject bare arrays with a warn, so do the parse + dispatch
    // inline. Same warn vocabulary as the helper so triage greps
    // (`'invalid JSON'`, `'payload is not'`) match either path.
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      if (err instanceof SyntaxError) {
        logger.warn(
          { folder, errName: err.name },
          'email-feedback.json migration: invalid JSON, skipping (file left in place)',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      throw err;
    }

    let feedback: unknown[];
    if (Array.isArray(parsed)) {
      feedback = parsed;
    } else if (parsed !== null && typeof parsed === 'object') {
      const wrapper = parsed as Record<string, unknown>;
      if (!Array.isArray(wrapper.feedback)) {
        logger.warn(
          { folder },
          'email-feedback.json migration: payload is an object but `feedback` is not an array, skipping (file left in place)',
        );
        if (!summary.leftInPlace.includes(folder))
          summary.leftInPlace.push(folder);
        continue;
      }
      feedback = wrapper.feedback;
    } else {
      logger.warn(
        {
          folder,
          parsedType: parsed === null ? 'null' : typeof parsed,
        },
        'email-feedback.json migration: payload is not an object or array, skipping (file left in place)',
      );
      if (!summary.leftInPlace.includes(folder))
        summary.leftInPlace.push(folder);
      continue;
    }

    const counts = { inserted: 0, skipped: 0, total: feedback.length };
    try {
      const importFile = db.transaction(() => {
        for (const row of feedback) {
          if (!isObjectRow(row)) {
            logger.warn(
              { folder },
              'email-feedback.json migration: skipping non-object row',
            );
            counts.skipped++;
            continue;
          }
          // Required-field guard: the schema's NOT NULL on pattern /
          // label / date would throw on missing values, rolling back
          // the entire per-file transaction. Skip the row with a
          // warn instead so a single malformed entry doesn't
          // poison-pill the whole file.
          const pattern = row.pattern;
          const label = row.label;
          const date = row.date;
          if (
            typeof pattern !== 'string' ||
            typeof label !== 'string' ||
            typeof date !== 'string'
          ) {
            const missing: string[] = [];
            if (typeof pattern !== 'string') missing.push('pattern');
            if (typeof label !== 'string') missing.push('label');
            if (typeof date !== 'string') missing.push('date');
            logger.warn(
              { folder, missing },
              'email-feedback.json migration: skipping row missing required fields',
            );
            counts.skipped++;
            continue;
          }
          // Omit `source` from the INSERT when the JSON-era row
          // didn't set it, so the schema's
          // `DEFAULT 'baruch-response'` fires. Mirrors the morning-
          // brief migration's `withAdded`/`DefaultAdded` split.
          if (typeof row.source === 'string') {
            insertWithSource.run(pattern, label, row.source, date);
          } else {
            insertDefaultSource.run(pattern, label, date);
          }
          counts.inserted++;
        }
      });
      importFile();
      // eslint-disable-next-line no-catch-all/no-catch-all -- the helper rethrows non-constraint errors via `throw err` (see `handleConstraintViolationOrRethrow` JSDoc); the lint rule can't see through the call.
    } catch (err) {
      // Constraint-class SqliteError (CHECK on label, NOT NULL we
      // didn't pre-guard, etc.) → warn-and-continue, source file
      // stays put for triage. Anything else (programming bug,
      // SQLITE_CORRUPT, SQLITE_BUSY) propagates via the helper.
      if (
        handleConstraintViolationOrRethrow(
          err,
          folder,
          'email-feedback.json',
          summary,
        )
      ) {
        continue;
      }
    }

    renameMigratedSource(
      filePath,
      stamp,
      folder,
      'email-feedback.json',
      counts,
      summary,
    );
  }
  return summary;
}
