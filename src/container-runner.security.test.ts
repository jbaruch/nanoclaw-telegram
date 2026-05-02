import { describe, it, expect, beforeEach, afterAll, vi } from 'vitest';
import fs from 'fs';
import path from 'path';
import Database from 'better-sqlite3';

// vi.mock is hoisted. vi.hoisted gives us values that exist at hoist time so
// the config mock factory can reference them without tripping TDZ.
const paths = vi.hoisted(() => {
  const root = `/tmp/nanoclaw-security-test-${process.pid}-${Date.now()}`;
  return {
    TEST_ROOT: root,
    STORE_DIR: `${root}/store`,
    DATA_DIR: `${root}/data`,
    GROUPS_DIR: `${root}/groups`,
    PROJECT_DIR: `${root}/project`,
  };
});

vi.mock('./config.js', () => ({
  STORE_DIR: paths.STORE_DIR,
  DATA_DIR: paths.DATA_DIR,
  GROUPS_DIR: paths.GROUPS_DIR,
  HOST_PROJECT_ROOT: paths.PROJECT_DIR,
  // HOST_UID: 0 short-circuits every `if (uid !== 0) fs.chownSync(...)` branch
  // so tests don't need to match the host's uid.
  HOST_UID: 0,
  HOST_GID: 0,
  AGENT_AUTO_COMPACT_WINDOW: 800000,
  CONTAINER_IMAGE: 'nanoclaw-agent:test',
  CONTAINER_MAX_OUTPUT_SIZE: 1_000_000,
  CONTAINER_TIMEOUT: 60_000,
  CREDENTIAL_PROXY_PORT: 3001,
  ENABLE_THRESHOLD_NUKE: false,
  IDLE_TIMEOUT: 60_000,
  MODEL_CONTEXT_WINDOW: 1000000,
  TILE_OWNER: 'test-owner',
  TIMEZONE: 'UTC',
  CONTAINER_VARS: {},
  MAINTENANCE_RULE_BLOCKLIST: new Set<string>(),
  MAINTENANCE_SKILL_BLOCKLIST: new Set<string>(),
}));

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

vi.mock('./credential-proxy.js', () => ({
  detectAuthMode: vi.fn(() => 'api-key'),
}));

vi.mock('./container-runtime.js', () => ({
  CONTAINER_RUNTIME_BIN: 'docker',
  CONTAINER_HOST_GATEWAY: 'host.docker.internal',
  hostGatewayArgs: () => [],
  readonlyMountArgs: (h: string, c: string) => ['-v', `${h}:${c}:ro`],
  stopContainer: vi.fn(),
}));

vi.mock('./mount-security.js', () => ({
  validateAdditionalMounts: vi.fn(() => []),
}));

import {
  createFilteredDb,
  buildVolumeMounts,
  SECRET_FILES,
  atomicPublishDir,
} from './container-runner.js';
import { validateAdditionalMounts } from './mount-security.js';
import type { RegisteredGroup } from './types.js';

const { TEST_ROOT, STORE_DIR, DATA_DIR, GROUPS_DIR, PROJECT_DIR } = paths;

function seedMessagesDb(opts: { withReactions?: boolean } = {}): string {
  const withReactions = opts.withReactions ?? true;
  fs.mkdirSync(STORE_DIR, { recursive: true });
  const dbPath = path.join(STORE_DIR, 'messages.db');
  if (fs.existsSync(dbPath)) fs.unlinkSync(dbPath);
  const db = new Database(dbPath);
  db.exec(`
    CREATE TABLE chats (
      jid TEXT PRIMARY KEY,
      name TEXT,
      last_message_time INTEGER
    );
    CREATE TABLE messages (
      id TEXT,
      chat_jid TEXT,
      sender TEXT,
      content TEXT,
      timestamp INTEGER,
      PRIMARY KEY (id, chat_jid)
    );
  `);
  // Two chats, each with two messages
  const insertChat = db.prepare(
    'INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)',
  );
  insertChat.run('chatA@g.us', 'Chat A', 1000);
  insertChat.run('chatB@g.us', 'Chat B', 2000);
  const insertMsg = db.prepare(
    'INSERT INTO messages (id, chat_jid, sender, content, timestamp) VALUES (?, ?, ?, ?, ?)',
  );
  insertMsg.run('a1', 'chatA@g.us', 'alice', 'hello from A', 1001);
  insertMsg.run('a2', 'chatA@g.us', 'alice', 'second from A', 1002);
  insertMsg.run('b1', 'chatB@g.us', 'bob', 'hello from B', 2001);
  insertMsg.run('b2', 'chatB@g.us', 'bob', 'second from B', 2002);
  // Reactions are optional so the fallback path (no src.reactions table,
  // pre-migration source DB) can be exercised separately.
  if (withReactions) {
    db.exec(`
      CREATE TABLE reactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT NOT NULL,
        message_chat_jid TEXT NOT NULL,
        reactor_jid TEXT NOT NULL,
        reactor_name TEXT NOT NULL,
        emoji TEXT NOT NULL,
        timestamp TEXT NOT NULL
      )
    `);
    const insertReact = db.prepare(
      'INSERT INTO reactions (message_id, message_chat_jid, reactor_jid, reactor_name, emoji, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
    );
    insertReact.run('a1', 'chatA@g.us', 'alice', 'Alice', '👀', '1001');
    insertReact.run('b1', 'chatB@g.us', 'bob', 'Bob', '👀', '2001');
  }
  db.close();
  return dbPath;
}

beforeEach(() => {
  fs.rmSync(TEST_ROOT, { recursive: true, force: true });
  fs.mkdirSync(TEST_ROOT, { recursive: true });
  fs.mkdirSync(STORE_DIR, { recursive: true });
  fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.mkdirSync(GROUPS_DIR, { recursive: true });
  fs.mkdirSync(PROJECT_DIR, { recursive: true });
});

afterAll(() => {
  fs.rmSync(TEST_ROOT, { recursive: true, force: true });
});

// -----------------------------------------------------------------------------
// Test 1 — createFilteredDb isolates messages by chatJid.
// This is the untrusted-group DB isolation boundary. A regression here leaks
// other groups' messages into an untrusted container.
// -----------------------------------------------------------------------------
describe('createFilteredDb (untrusted DB isolation)', () => {
  it('returns null when source DB does not exist', () => {
    // Fresh tmpdir — no messages.db yet
    expect(createFilteredDb('chatA@g.us', 'folder-a')).toBe(null);
  });

  it('filtered DB contains only target chat rows, zero other-chat rows', () => {
    seedMessagesDb();
    const filtered = createFilteredDb('chatA@g.us', 'folder-a');
    expect(filtered).not.toBe(null);
    const db = new Database(filtered!, { readonly: true });
    try {
      const chats = db.prepare('SELECT jid FROM chats').all() as {
        jid: string;
      }[];
      const messages = db
        .prepare('SELECT id, chat_jid FROM messages')
        .all() as { id: string; chat_jid: string }[];

      expect(chats).toEqual([{ jid: 'chatA@g.us' }]);
      expect(messages.length).toBe(2);
      expect(messages.every((m) => m.chat_jid === 'chatA@g.us')).toBe(true);
      // Explicit negative: chatB must not leak through
      expect(messages.find((m) => m.id === 'b1')).toBeUndefined();
      expect(messages.find((m) => m.id === 'b2')).toBeUndefined();

      // Reactions table must be present (filtered-DB consumers query it)
      // and scoped to the target chat — chatB reactions must not leak.
      const reactionsTable = db
        .prepare(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        )
        .get('reactions');
      expect(reactionsTable).toBeDefined();
      const outOfScopeReaction = db
        .prepare('SELECT 1 FROM reactions WHERE message_chat_jid <> ? LIMIT 1')
        .get('chatA@g.us');
      expect(outOfScopeReaction).toBeUndefined();
      const reactionCount = (
        db.prepare('SELECT COUNT(*) AS c FROM reactions').get() as { c: number }
      ).c;
      expect(reactionCount).toBe(1);
    } finally {
      db.close();
    }
  });

  it('filtered DB has empty reactions table when source DB has no reactions table (pre-migration)', () => {
    // Source DB without reactions — simulates a fresh install before the
    // reactions migration ran. The filtered DB must still expose an empty
    // reactions table so consumers that JOIN on it don't abort with
    // "no such table: reactions". Specific existence check on
    // src.sqlite_master replaced an earlier bare try/catch that would
    // have masked corruption / lock errors as "no reactions".
    seedMessagesDb({ withReactions: false });
    const filtered = createFilteredDb('chatA@g.us', 'folder-a');
    expect(filtered).not.toBe(null);
    const db = new Database(filtered!, { readonly: true });
    try {
      const reactionsTable = db
        .prepare(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        )
        .get('reactions');
      expect(reactionsTable).toBeDefined();
      const reactionCount = (
        db.prepare('SELECT COUNT(*) AS c FROM reactions').get() as { c: number }
      ).c;
      expect(reactionCount).toBe(0);
    } finally {
      db.close();
    }
  });

  it("SQL-injection-shaped chatJid doesn't leak other chats", () => {
    seedMessagesDb();
    // If the code naively interpolated, this would return every row. The
    // escaper doubles single quotes, so the whole string becomes a literal
    // that matches zero jids.
    const filtered = createFilteredDb("' OR '1'='1", 'folder-inject');
    const db = new Database(filtered!, { readonly: true });
    try {
      const chatCount = (
        db.prepare('SELECT COUNT(*) AS c FROM chats').get() as { c: number }
      ).c;
      const msgCount = (
        db.prepare('SELECT COUNT(*) AS c FROM messages').get() as { c: number }
      ).c;
      expect(chatCount).toBe(0);
      expect(msgCount).toBe(0);
    } finally {
      db.close();
    }
  });

  it('second call overwrites the stale filtered DB', () => {
    seedMessagesDb();
    const first = createFilteredDb('chatA@g.us', 'folder-a');
    // Add a new message to the source after the first filter
    const src = new Database(path.join(STORE_DIR, 'messages.db'));
    src
      .prepare(
        'INSERT INTO messages (id, chat_jid, sender, content, timestamp) VALUES (?, ?, ?, ?, ?)',
      )
      .run('a3', 'chatA@g.us', 'alice', 'third from A', 1003);
    src.close();

    const second = createFilteredDb('chatA@g.us', 'folder-a');
    expect(second).toBe(first); // same path

    const db = new Database(second!, { readonly: true });
    try {
      const count = (
        db.prepare('SELECT COUNT(*) AS c FROM messages').get() as { c: number }
      ).c;
      // If the stale copy had been kept, count would still be 2.
      expect(count).toBe(3);
    } finally {
      db.close();
    }
  });

  // Issue #287 follow-up — operators upgrading from a pre-fix version
  // can have stale `-wal`/`-shm` sidecars on disk from when the snapshot
  // ran in WAL mode. The next `createFilteredDb` call must wipe those
  // sidecars too, not just the main DB file. A partial state (main DB
  // gone, sidecars present) is the exact scenario SQLite refuses to
  // open with `unable to open database file`.
  it('createFilteredDb removes leftover -wal/-shm sidecars from a pre-fix snapshot', () => {
    seedMessagesDb();
    // First call to create the filtered dir + main DB.
    const filtered = createFilteredDb('chatA@g.us', 'folder-a');
    expect(filtered).not.toBe(null);
    // Plant fake sidecars as if a pre-fix WAL-mode snapshot had run.
    const walPath = `${filtered}-wal`;
    const shmPath = `${filtered}-shm`;
    fs.writeFileSync(walPath, 'stale-wal');
    fs.writeFileSync(shmPath, 'stale-shm');
    expect(fs.existsSync(walPath)).toBe(true);
    expect(fs.existsSync(shmPath)).toBe(true);

    // Re-run — the stale-copy cleanup must take both sidecars with it.
    const refresh = createFilteredDb('chatA@g.us', 'folder-a');
    expect(refresh).toBe(filtered);
    expect(fs.existsSync(walPath)).toBe(false);
    expect(fs.existsSync(shmPath)).toBe(false);
  });

  // Issue #287 — filtered DB must use a rollback journal, not WAL.
  // Untrusted containers receive this DB on a read-only mount (`fakeowner
  // ro`); a WAL-mode DB cannot be opened even for reads on a RO mount
  // because SQLite needs to write `-wal`/`-shm` sidecars. Forcing
  // `journal_mode = DELETE` makes the file self-contained so every
  // reader's default open succeeds. A regression here surfaces inside
  // untrusted containers as `OperationalError: unable to open database
  // file` from any default-mode reader (Python `sqlite3.connect(path)`,
  // node `new Database(path)`).
  it('filtered DB is created with journal_mode = DELETE (not WAL) — #287', () => {
    seedMessagesDb();
    const filtered = createFilteredDb('chatA@g.us', 'folder-a');
    expect(filtered).not.toBe(null);
    const db = new Database(filtered!, { readonly: true });
    try {
      const mode = db.pragma('journal_mode', { simple: true });
      expect(mode).toBe('delete');
    } finally {
      db.close();
    }
    // No `-wal`/`-shm` sidecars should be present after creation. Their
    // existence is the visible symptom of WAL mode.
    expect(fs.existsSync(`${filtered}-wal`)).toBe(false);
    expect(fs.existsSync(`${filtered}-shm`)).toBe(false);
  });

  // Regression guard for #93 — prior to the atomic temp+rename pattern, any
  // interruption between `new Database(finalPath)` and the schema exec calls
  // would leave a 0-byte file at the canonical path. Subsequent spawns then
  // hit "disk I/O error" on every retry, the circuit breaker tripped, and
  // the group was permanently wedged until an operator manually rm'd it.
  describe('atomic creation (regression #93)', () => {
    it('happy path leaves no .tmp-* leftovers next to the final file', () => {
      seedMessagesDb();
      const filtered = createFilteredDb('chatA@g.us', 'folder-atomic-happy');
      expect(filtered).not.toBe(null);
      expect(fs.existsSync(filtered!)).toBe(true);
      // No half-written temp file should survive a successful run.
      const dir = path.dirname(filtered!);
      const leftovers = fs
        .readdirSync(dir)
        .filter((f) => f.startsWith('messages.db.tmp-'));
      expect(leftovers).toEqual([]);
    });

    it('idempotent re-run produces a valid file with no temp leaks', () => {
      seedMessagesDb();
      const first = createFilteredDb('chatA@g.us', 'folder-atomic-idem');
      const second = createFilteredDb('chatA@g.us', 'folder-atomic-idem');
      expect(second).toBe(first);
      expect(fs.existsSync(second!)).toBe(true);
      const stat = fs.statSync(second!);
      expect(stat.size).toBeGreaterThan(0);
      const dir = path.dirname(second!);
      const leftovers = fs
        .readdirSync(dir)
        .filter((f) => f.startsWith('messages.db.tmp-'));
      expect(leftovers).toEqual([]);
    });

    it('failure during creation does NOT leak a 0-byte file at the canonical path', () => {
      // Seed a valid messages.db, then corrupt it so `ATTACH DATABASE` throws
      // partway through createFilteredDb. This simulates the real-world
      // failure mode (transient SQLite I/O error) that originally wedged
      // the group at 17:12 UTC.
      const dbPath = seedMessagesDb();
      // Overwrite the source with garbage — ATTACH will reject it as
      // "not a database" / disk I/O error, throwing inside the try block.
      fs.writeFileSync(dbPath, Buffer.from('not a sqlite database at all'));

      expect(() =>
        createFilteredDb('chatA@g.us', 'folder-atomic-fail'),
      ).toThrow();

      // The canonical path must NOT exist — that's the whole point of the
      // atomic temp+rename pattern. If the legacy in-place write were still
      // in effect, this would be a 0-byte (or partial-header) file.
      const finalPath = path.join(
        DATA_DIR,
        'filtered-db',
        'folder-atomic-fail',
        'messages.db',
      );
      expect(fs.existsSync(finalPath)).toBe(false);

      // And no temp file should be left lying around either — the catch
      // block cleans it up before rethrowing.
      const dir = path.dirname(finalPath);
      if (fs.existsSync(dir)) {
        const leftovers = fs
          .readdirSync(dir)
          .filter((f) => f.startsWith('messages.db.tmp-'));
        expect(leftovers).toEqual([]);
      }
    });
  });

  // Regression guard for #100 — when the source `messages.db` had a degraded
  // WAL/SHM state (e.g. -shm file evicted by Docker bind-mount on macOS),
  // the ATTACH inside createFilteredDb threw `SqliteError: disk I/O error`
  // even though both the canonical filtered-db and the source database file
  // itself were healthy. The recovery: catch the disk-I/O error, run
  // `PRAGMA wal_checkpoint(TRUNCATE)` on the source via a fresh connection
  // (which re-establishes -shm coordination), and retry the ATTACH+CTAS
  // exactly once.
  describe('WAL recovery on disk I/O error (regression #100)', () => {
    /**
     * Stub `Database.prototype.exec` so that the FIRST `ATTACH DATABASE`
     * call throws a SqliteError-shaped "disk I/O error", and all subsequent
     * calls (including the retry's ATTACH and every CTAS) execute the real
     * implementation. Calls through for non-ATTACH statements so recovery
     * and retry stay realistic.
     *
     * Returns a counter that records how many times the stub matched the
     * ATTACH path so tests can assert exactly-one-throw semantics.
     */
    function failFirstAttach(): {
      attachCalls: { count: number };
      restore: () => void;
    } {
      const real = Database.prototype.exec;
      const attachCalls = { count: 0 };
      let firstAttachThrown = false;
      const spy = vi
        .spyOn(Database.prototype, 'exec')
        .mockImplementation(function (this: Database.Database, sql: string) {
          const isAttach = /^\s*ATTACH\s+DATABASE/i.test(sql);
          if (isAttach) {
            attachCalls.count++;
            if (!firstAttachThrown) {
              firstAttachThrown = true;
              // Match better-sqlite3's actual SqliteError shape so the
              // production code's `instanceof SqliteError` branch fires.
              const SqliteErrorCtor = (
                Database as unknown as { SqliteError: typeof Error }
              ).SqliteError;
              // SqliteError(message, code) — code is the SQLite error
              // identifier; SQLITE_IOERR is what disk-I/O failures carry.
              const err = new (SqliteErrorCtor as unknown as new (
                m: string,
                c: string,
              ) => Error)('disk I/O error', 'SQLITE_IOERR');
              throw err;
            }
          }
          return real.call(this, sql) as unknown as Database.Database;
        });
      return {
        attachCalls,
        restore: () => spy.mockRestore(),
      };
    }

    /**
     * Stub `Database.prototype.exec` so EVERY `ATTACH DATABASE` call throws
     * disk-I/O. Used for the "retry also fails" case.
     */
    function failAllAttach(): {
      attachCalls: { count: number };
      restore: () => void;
    } {
      const real = Database.prototype.exec;
      const attachCalls = { count: 0 };
      const spy = vi
        .spyOn(Database.prototype, 'exec')
        .mockImplementation(function (this: Database.Database, sql: string) {
          if (/^\s*ATTACH\s+DATABASE/i.test(sql)) {
            attachCalls.count++;
            const SqliteErrorCtor = (
              Database as unknown as { SqliteError: typeof Error }
            ).SqliteError;
            throw new (SqliteErrorCtor as unknown as new (
              m: string,
              c: string,
            ) => Error)('disk I/O error', 'SQLITE_IOERR');
          }
          return real.call(this, sql) as unknown as Database.Database;
        });
      return {
        attachCalls,
        restore: () => spy.mockRestore(),
      };
    }

    /**
     * Stub `Database.prototype.exec` so the FIRST `ATTACH DATABASE` throws
     * a SqliteError that is NOT disk-I/O (e.g. "no such table"). The
     * production code must propagate this immediately without invoking
     * recovery — non-disk-I/O errors aren't the SHM-degradation signal.
     */
    function failFirstAttachWithNonIoError(): {
      attachCalls: { count: number };
      restore: () => void;
    } {
      const real = Database.prototype.exec;
      const attachCalls = { count: 0 };
      let firstAttachThrown = false;
      const spy = vi
        .spyOn(Database.prototype, 'exec')
        .mockImplementation(function (this: Database.Database, sql: string) {
          if (/^\s*ATTACH\s+DATABASE/i.test(sql)) {
            attachCalls.count++;
            if (!firstAttachThrown) {
              firstAttachThrown = true;
              const SqliteErrorCtor = (
                Database as unknown as { SqliteError: typeof Error }
              ).SqliteError;
              throw new (SqliteErrorCtor as unknown as new (
                m: string,
                c: string,
              ) => Error)('no such table: src.chats', 'SQLITE_ERROR');
            }
          }
          return real.call(this, sql) as unknown as Database.Database;
        });
      return {
        attachCalls,
        restore: () => spy.mockRestore(),
      };
    }

    /**
     * Spy on `Database.prototype.pragma` so we can detect whether
     * `recoverSourceWalState` actually ran. The recovery path is the only
     * site that calls `pragma('wal_checkpoint(TRUNCATE)')` from the
     * orchestrator process — a hit on that argument is a positive signal.
     */
    function trackRecoveryCalls(): {
      checkpointCalls: { count: number };
      restore: () => void;
    } {
      const checkpointCalls = { count: 0 };
      const real = Database.prototype.pragma;
      const spy = vi
        .spyOn(Database.prototype, 'pragma')
        .mockImplementation(function (
          this: Database.Database,
          source: string,
          options?: Database.PragmaOptions,
        ) {
          if (typeof source === 'string' && /wal_checkpoint/i.test(source)) {
            checkpointCalls.count++;
          }
          return real.call(this, source, options as Database.PragmaOptions);
        });
      return {
        checkpointCalls,
        restore: () => spy.mockRestore(),
      };
    }

    it('happy path: no recovery invoked when ATTACH succeeds', () => {
      seedMessagesDb();
      const recovery = trackRecoveryCalls();
      try {
        const filtered = createFilteredDb('chatA@g.us', 'folder-no-recovery');
        expect(filtered).not.toBe(null);
        // No wal_checkpoint(TRUNCATE) call should have happened — the only
        // checkpoints called by the production code are the recovery path.
        expect(recovery.checkpointCalls.count).toBe(0);
      } finally {
        recovery.restore();
      }
    });

    it('disk I/O on ATTACH triggers recovery and the retry succeeds', () => {
      seedMessagesDb();
      const attachStub = failFirstAttach();
      const recovery = trackRecoveryCalls();
      try {
        const filtered = createFilteredDb('chatA@g.us', 'folder-recover');
        // Retry succeeded — file exists at the canonical path.
        expect(filtered).not.toBe(null);
        expect(fs.existsSync(filtered!)).toBe(true);
        // Recovery ran exactly once.
        expect(recovery.checkpointCalls.count).toBe(1);
        // ATTACH was called twice: once that threw, once that succeeded.
        expect(attachStub.attachCalls.count).toBe(2);
        // Retry actually populated the filtered DB — open and verify.
        const db = new Database(filtered!, { readonly: true });
        try {
          const chats = db.prepare('SELECT jid FROM chats').all() as {
            jid: string;
          }[];
          expect(chats).toEqual([{ jid: 'chatA@g.us' }]);
        } finally {
          db.close();
        }
        // No temp leftovers from either the failed first attempt or the
        // successful retry.
        const dir = path.dirname(filtered!);
        const leftovers = fs
          .readdirSync(dir)
          .filter((f) => f.startsWith('messages.db.tmp-'));
        expect(leftovers).toEqual([]);
      } finally {
        attachStub.restore();
        recovery.restore();
      }
    });

    it('disk I/O on retry too: original error propagates, no further attempts', () => {
      seedMessagesDb();
      const attachStub = failAllAttach();
      const recovery = trackRecoveryCalls();
      try {
        expect(() =>
          createFilteredDb('chatA@g.us', 'folder-retry-fails'),
        ).toThrow(/disk I\/O error/);
        // Recovery ran exactly once between attempts (loop guard).
        expect(recovery.checkpointCalls.count).toBe(1);
        // ATTACH was attempted exactly twice — the original and the retry.
        // A third attempt would mean we're looping, which violates the
        // "retry once" contract.
        expect(attachStub.attachCalls.count).toBe(2);
        // Canonical path must NOT exist — both attempts failed before rename.
        const finalPath = path.join(
          DATA_DIR,
          'filtered-db',
          'folder-retry-fails',
          'messages.db',
        );
        expect(fs.existsSync(finalPath)).toBe(false);
        // No temp leftovers from either failed attempt.
        const dir = path.dirname(finalPath);
        if (fs.existsSync(dir)) {
          const leftovers = fs
            .readdirSync(dir)
            .filter((f) => f.startsWith('messages.db.tmp-'));
          expect(leftovers).toEqual([]);
        }
      } finally {
        attachStub.restore();
        recovery.restore();
      }
    });

    it('non-disk-I/O SqliteError propagates immediately without recovery', () => {
      seedMessagesDb();
      const attachStub = failFirstAttachWithNonIoError();
      const recovery = trackRecoveryCalls();
      try {
        expect(() =>
          createFilteredDb('chatA@g.us', 'folder-no-such-table'),
        ).toThrow(/no such table/);
        // Recovery must NOT have been invoked — only disk-I/O errors
        // should trigger the wal_checkpoint(TRUNCATE) recovery path.
        expect(recovery.checkpointCalls.count).toBe(0);
        // ATTACH was attempted exactly once (no retry for non-disk-I/O).
        expect(attachStub.attachCalls.count).toBe(1);
      } finally {
        attachStub.restore();
        recovery.restore();
      }
    });
  });
});

// -----------------------------------------------------------------------------
// atomicPublishDir — regression guard for #95.
//
// Prior to this fix, the .tessl publish in buildVolumeMounts was a non-atomic
// rmSync(groupTesslDir) + cpSync(dstTessl, groupTesslDir). Two scheduled
// tasks (heartbeat + task-watchdog) firing within the same millisecond on the
// same group both ran that block, with one's rm walk colliding with the
// other's cp into already-walked subdirs — yielding ENOTEMPTY on rmdir and
// wedging both tasks. The fix uses temp+swap-rename so dstDir is always a
// fully-populated directory at any instant.
// -----------------------------------------------------------------------------
describe('atomicPublishDir (regression #95)', () => {
  function seedSrc(name: string): string {
    const dir = path.join(TEST_ROOT, name);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'RULES.md'), 'rules');
    const sub = path.join(
      dir,
      'tiles',
      'jbaruch',
      'nanoclaw-core',
      'skills',
      'status',
      'scripts',
    );
    fs.mkdirSync(sub, { recursive: true });
    fs.writeFileSync(path.join(sub, 'check.sh'), 'echo hi');
    fs.writeFileSync(path.join(sub, 'helper.py'), 'print(1)');
    return dir;
  }

  function listSiblings(dst: string): string[] {
    const parent = path.dirname(dst);
    const base = path.basename(dst);
    if (!fs.existsSync(parent)) return [];
    return fs
      .readdirSync(parent)
      .filter((f) => f !== base && f.startsWith(`${base}.`));
  }

  it('happy path publishes content with no temp/backup leftovers', () => {
    const src = seedSrc('src-happy');
    const dst = path.join(TEST_ROOT, 'dst-happy');
    atomicPublishDir(src, dst);

    expect(fs.existsSync(dst)).toBe(true);
    expect(fs.readFileSync(path.join(dst, 'RULES.md'), 'utf8')).toBe('rules');
    expect(
      fs.existsSync(
        path.join(
          dst,
          'tiles',
          'jbaruch',
          'nanoclaw-core',
          'skills',
          'status',
          'scripts',
          'check.sh',
        ),
      ),
    ).toBe(true);
    expect(listSiblings(dst)).toEqual([]);
  });

  it('idempotent re-run replaces content cleanly with no leftovers', () => {
    const src1 = seedSrc('src-idem-1');
    const dst = path.join(TEST_ROOT, 'dst-idem');
    atomicPublishDir(src1, dst);

    // Mutate src — second publish must overwrite the dst content.
    fs.writeFileSync(path.join(src1, 'RULES.md'), 'rules-v2');
    atomicPublishDir(src1, dst);

    expect(fs.readFileSync(path.join(dst, 'RULES.md'), 'utf8')).toBe(
      'rules-v2',
    );
    expect(listSiblings(dst)).toEqual([]);
  });

  it('concurrent publishes both complete; dst is valid; no leftovers', async () => {
    const src = seedSrc('src-concurrent');
    const dst = path.join(TEST_ROOT, 'dst-concurrent');

    // Pre-seed dst so both calls hit the swap path (the bug's hot path).
    atomicPublishDir(src, dst);

    // Two near-simultaneous publishes, simulating heartbeat + task-watchdog.
    // Wrap each in a Promise that lets failures escape so the test reports
    // the actual ENOTEMPTY if the bug regresses, instead of a generic
    // Promise.all rejection.
    const results = await Promise.allSettled([
      Promise.resolve().then(() => atomicPublishDir(src, dst)),
      Promise.resolve().then(() => atomicPublishDir(src, dst)),
    ]);

    // Every call must succeed — race losers swallow EEXIST/ENOTEMPTY/EPERM.
    for (const r of results) {
      if (r.status === 'rejected') {
        throw new Error(
          `atomicPublishDir threw under concurrency: ${String(r.reason)}`,
        );
      }
    }

    // dst must be valid (fully populated, not half-walked).
    expect(fs.existsSync(dst)).toBe(true);
    expect(fs.readFileSync(path.join(dst, 'RULES.md'), 'utf8')).toBe('rules');
    expect(
      fs.existsSync(
        path.join(
          dst,
          'tiles',
          'jbaruch',
          'nanoclaw-core',
          'skills',
          'status',
          'scripts',
          'check.sh',
        ),
      ),
    ).toBe(true);

    // No temp / swap siblings left behind.
    expect(listSiblings(dst)).toEqual([]);
  });

  it('cpSync failure cleans up tmp and leaves existing dst untouched', () => {
    const src = seedSrc('src-fail');
    const dst = path.join(TEST_ROOT, 'dst-fail');
    // Pre-seed dst so we can verify it's untouched after a mid-publish throw.
    atomicPublishDir(src, dst);
    const originalContent = fs.readFileSync(path.join(dst, 'RULES.md'), 'utf8');

    const cpSpy = vi.spyOn(fs, 'cpSync').mockImplementation(() => {
      throw new Error('synthetic cpSync failure');
    });
    try {
      expect(() => atomicPublishDir(src, dst)).toThrow(
        'synthetic cpSync failure',
      );
    } finally {
      cpSpy.mockRestore();
    }

    // Existing dst preserved (we never got to the rename swap).
    expect(fs.existsSync(dst)).toBe(true);
    expect(fs.readFileSync(path.join(dst, 'RULES.md'), 'utf8')).toBe(
      originalContent,
    );
    // No tmp / swap leftovers.
    expect(listSiblings(dst)).toEqual([]);
  });
});

// -----------------------------------------------------------------------------
// Test 2 — SECRET_FILES list + main-group shadow mounts.
// Defense against agents reading bot tokens directly. Pinning the list
// catches accidental deletions; pinning the mount loop catches shadow regressions.
// -----------------------------------------------------------------------------
describe('SECRET_FILES and main-group shadow mounts', () => {
  // Pinned list — updating this requires updating the orchestrator
  // shadow logic AND documenting why the new file contains secrets.
  it('SECRET_FILES pins the full list of secret files to shadow', () => {
    expect([...SECRET_FILES]).toEqual([
      '.env',
      '.env.bak',
      'data/env/env',
      'scripts/heartbeat-external.conf',
    ]);
  });

  it('main group gets /dev/null shadow mount for every existing secret file', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      // Create each secret file so the exists-check passes
      for (const rel of SECRET_FILES) {
        const abs = path.join(PROJECT_DIR, rel);
        fs.mkdirSync(path.dirname(abs), { recursive: true });
        fs.writeFileSync(abs, 'SECRET=xyz');
      }

      const group: RegisteredGroup = {
        name: 'Main',
        folder: 'main-group',
        trigger: '@Main',
        added_at: new Date().toISOString(),
      };
      // buildVolumeMounts writes AGENTS.md into the group folder — pre-create it
      fs.mkdirSync(path.join(GROUPS_DIR, group.folder), { recursive: true });
      const mounts = buildVolumeMounts(group, true, 'main@g.us');

      const shadowMounts = mounts.filter((m) => m.hostPath === '/dev/null');
      expect(shadowMounts.length).toBe(SECRET_FILES.length);
      expect(shadowMounts.every((m) => m.readonly === true)).toBe(true);

      const shadowedContainerPaths = shadowMounts
        .map((m) => m.containerPath)
        .sort();
      const expected = SECRET_FILES.map(
        (rel) => `/workspace/project/${rel}`,
      ).sort();
      expect(shadowedContainerPaths).toEqual(expected);
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('missing secret files are skipped (no shadow mount for absent files)', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      // Create all but .env.bak
      for (const rel of SECRET_FILES) {
        if (rel === '.env.bak') continue;
        const abs = path.join(PROJECT_DIR, rel);
        fs.mkdirSync(path.dirname(abs), { recursive: true });
        fs.writeFileSync(abs, 'SECRET=xyz');
      }

      const group: RegisteredGroup = {
        name: 'Main',
        folder: 'main-group',
        trigger: '@Main',
        added_at: new Date().toISOString(),
      };
      // buildVolumeMounts writes AGENTS.md into the group folder — pre-create it
      fs.mkdirSync(path.join(GROUPS_DIR, group.folder), { recursive: true });
      const mounts = buildVolumeMounts(group, true, 'main@g.us');

      const shadowMounts = mounts.filter((m) => m.hostPath === '/dev/null');
      // One fewer than the full list
      expect(shadowMounts.length).toBe(SECRET_FILES.length - 1);
      expect(
        shadowMounts.find(
          (m) => m.containerPath === '/workspace/project/.env.bak',
        ),
      ).toBeUndefined();
    } finally {
      process.chdir(originalCwd);
    }
  });
});

// -----------------------------------------------------------------------------
// Test 2b — SECRET_FILES shadow propagates across additionalMounts.
// The main-group `/workspace/project/<relPath>` shadow above only covers the
// canonical project mount. When a group registers an `additionalMount` that
// re-exposes the nanoclaw tree at a different container path (e.g. a group
// config requesting `hostPath: ~/nanoclaw` lands it at
// `/workspace/extra/projects/nanoclaw/`), the secret files under that path
// need their own shadow. Without it, a trusted agent could read the real
// `.env` via the extra mount even though the canonical `.env` is `/dev/null`.
// -----------------------------------------------------------------------------
describe('SECRET_FILES shadow across additionalMounts', () => {
  function makeTrustedGroup(): RegisteredGroup {
    return {
      name: 'Trusted',
      folder: 'trusted-group',
      trigger: '@T',
      added_at: new Date().toISOString(),
      containerConfig: {
        trusted: true,
        additionalMounts: [
          {
            hostPath: '~/nanoclaw',
            readonly: false,
          },
        ],
      },
    };
  }

  beforeEach(() => {
    seedMessagesDb();
    fs.mkdirSync(path.join(GROUPS_DIR, 'trusted-group'), { recursive: true });
  });

  it('shadows every reachable SECRET_FILES entry at the additionalMount container path', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      for (const rel of SECRET_FILES) {
        const abs = path.join(PROJECT_DIR, rel);
        fs.mkdirSync(path.dirname(abs), { recursive: true });
        fs.writeFileSync(abs, 'SECRET=xyz');
      }

      // Mock returns a validated mount whose host path is the project
      // root itself — the exact shape that exposes every SECRET_FILES
      // entry at `/workspace/extra/projects/nanoclaw/<relPath>`.
      vi.mocked(validateAdditionalMounts).mockReturnValueOnce([
        {
          hostPath: PROJECT_DIR,
          containerPath: '/workspace/extra/projects/nanoclaw',
          readonly: false,
        },
      ]);

      const mounts = buildVolumeMounts(
        makeTrustedGroup(),
        false,
        'trusted@g.us',
      );

      const extraShadows = mounts.filter(
        (m) =>
          m.hostPath === '/dev/null' &&
          m.containerPath.startsWith('/workspace/extra/projects/nanoclaw/'),
      );
      expect(extraShadows.length).toBe(SECRET_FILES.length);
      const extraPaths = extraShadows.map((m) => m.containerPath).sort();
      const expected = SECRET_FILES.map(
        (rel) => `/workspace/extra/projects/nanoclaw/${rel}`,
      ).sort();
      expect(extraPaths).toEqual(expected);
      expect(extraShadows.every((m) => m.readonly === true)).toBe(true);
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('shadows only files that exist on the host (missing files skipped)', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      // Only create .env — every other SECRET_FILES entry is missing
      const envAbs = path.join(PROJECT_DIR, '.env');
      fs.writeFileSync(envAbs, 'SECRET=xyz');

      vi.mocked(validateAdditionalMounts).mockReturnValueOnce([
        {
          hostPath: PROJECT_DIR,
          containerPath: '/workspace/extra/projects/nanoclaw',
          readonly: false,
        },
      ]);

      const mounts = buildVolumeMounts(
        makeTrustedGroup(),
        false,
        'trusted@g.us',
      );

      const extraShadows = mounts.filter(
        (m) =>
          m.hostPath === '/dev/null' &&
          m.containerPath.startsWith('/workspace/extra/projects/nanoclaw/'),
      );
      // Only `.env` exists → exactly one extra shadow
      expect(extraShadows.length).toBe(1);
      expect(extraShadows[0].containerPath).toBe(
        '/workspace/extra/projects/nanoclaw/.env',
      );
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('does NOT shadow when the additionalMount host path is unrelated to the project', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      for (const rel of SECRET_FILES) {
        const abs = path.join(PROJECT_DIR, rel);
        fs.mkdirSync(path.dirname(abs), { recursive: true });
        fs.writeFileSync(abs, 'SECRET=xyz');
      }

      // Mount an unrelated host directory that contains no SECRET_FILES.
      // Nothing should be shadowed under this container path (false
      // positives here would be loud — every extra mount the user
      // registers would gain spurious `/dev/null` mounts).
      //
      // Host path lives under `TEST_ROOT` (which is already unique per
      // test process: see the `vi.hoisted` block at the top of this
      // file that derives TEST_ROOT from pid + timestamp). Using a
      // fixed `/tmp/...` path here would collide across concurrent
      // vitest workers.
      const unrelatedDir = path.join(TEST_ROOT, 'unrelated');
      fs.mkdirSync(unrelatedDir, { recursive: true });
      vi.mocked(validateAdditionalMounts).mockReturnValueOnce([
        {
          hostPath: unrelatedDir,
          containerPath: '/workspace/extra/unrelated',
          readonly: false,
        },
      ]);

      const mounts = buildVolumeMounts(
        makeTrustedGroup(),
        false,
        'trusted@g.us',
      );

      const extraShadows = mounts.filter(
        (m) =>
          m.hostPath === '/dev/null' &&
          m.containerPath.startsWith('/workspace/extra/unrelated/'),
      );
      expect(extraShadows.length).toBe(0);
    } finally {
      process.chdir(originalCwd);
      // TEST_ROOT cleanup happens in the file-level `afterAll` — no
      // per-test rmSync needed now that we're inside TEST_ROOT.
    }
  });

  it('shadows the right sub-path when the additionalMount is a parent of the project', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      for (const rel of SECRET_FILES) {
        const abs = path.join(PROJECT_DIR, rel);
        fs.mkdirSync(path.dirname(abs), { recursive: true });
        fs.writeFileSync(abs, 'SECRET=xyz');
      }

      // Mount the PARENT of PROJECT_DIR — the secrets still live inside
      // it, just one level deeper. Relative path under the mount is
      // `<projectBasename>/<relPath>`; container path prefixes `extra/`.
      const parentDir = path.dirname(PROJECT_DIR);
      const projectBasename = path.basename(PROJECT_DIR);
      vi.mocked(validateAdditionalMounts).mockReturnValueOnce([
        {
          hostPath: parentDir,
          containerPath: '/workspace/extra/parent',
          readonly: false,
        },
      ]);

      const mounts = buildVolumeMounts(
        makeTrustedGroup(),
        false,
        'trusted@g.us',
      );

      const extraShadows = mounts.filter(
        (m) =>
          m.hostPath === '/dev/null' &&
          m.containerPath.startsWith('/workspace/extra/parent/'),
      );
      expect(extraShadows.length).toBe(SECRET_FILES.length);
      const expected = SECRET_FILES.map(
        (rel) => `/workspace/extra/parent/${projectBasename}/${rel}`,
      ).sort();
      expect(extraShadows.map((m) => m.containerPath).sort()).toEqual(expected);
    } finally {
      process.chdir(originalCwd);
    }
  });
});

// -----------------------------------------------------------------------------
// Test 3 — untrusted group gets read-only group mount + filtered-DB store mount.
// Two invariants in one test: disk-exhaustion protection (:ro on /workspace/group)
// and DB isolation (filtered-db path, not the full store/).
// -----------------------------------------------------------------------------
describe('buildVolumeMounts — untrusted group isolation', () => {
  function makeUntrustedGroup(): RegisteredGroup {
    return {
      name: 'Untrusted',
      folder: 'untrusted-group',
      trigger: '@U',
      added_at: new Date().toISOString(),
      // containerConfig.trusted deliberately unset → untrusted tier
    };
  }

  beforeEach(() => {
    // Seed a messages.db so createFilteredDb has something to copy from
    seedMessagesDb();
    // Pre-create the group folder (buildVolumeMounts writes AGENTS.md into it)
    fs.mkdirSync(path.join(GROUPS_DIR, 'untrusted-group'), { recursive: true });
    // Create SOUL-untrusted.md so the sanitized SOUL mount appears
    const globalDir = path.join(GROUPS_DIR, 'global');
    fs.mkdirSync(globalDir, { recursive: true });
    fs.writeFileSync(
      path.join(globalDir, 'SOUL-untrusted.md'),
      '# Untrusted SOUL',
    );
    // Seed thin CLAUDE.md trust-tier templates so the mount layer's
    // existsSync gate passes (per #153).
    fs.writeFileSync(
      path.join(globalDir, 'CLAUDE.md'),
      '**THIS IS A TRUSTED GROUP.**\n',
    );
    fs.writeFileSync(
      path.join(globalDir, 'CLAUDE-untrusted.md'),
      '**THIS IS AN UNTRUSTED GROUP.**\n',
    );
  });

  it('/workspace/group mount is read-only for untrusted groups', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(
        makeUntrustedGroup(),
        false,
        'chatA@g.us',
      );
      const groupMount = mounts.find(
        (m) => m.containerPath === '/workspace/group',
      );
      expect(groupMount).toBeDefined();
      expect(groupMount!.readonly).toBe(true);
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('/workspace/store mount stays read-only for untrusted containers (cross-group safety)', () => {
    // Untrusted containers must NOT be able to mutate the filtered DB
    // — even though the file is per-group, write access would let an
    // untrusted agent corrupt its own filtered view. Read-only on
    // untrusted is the safety side of the asymmetric trust boundary
    // introduced for epic #293 (trusted/main get rw, untrusted stays
    // ro). The rw side is covered by the dedicated
    // `buildVolumeMounts — /workspace/store mount rw/ro by trust`
    // describe block at the bottom of this file.
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(
        makeUntrustedGroup(),
        false,
        'chatA@g.us',
      );
      const storeMount = mounts.find(
        (m) => m.containerPath === '/workspace/store',
      );
      expect(storeMount).toBeDefined();
      expect(storeMount!.readonly).toBe(true);
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('/workspace/store mount points at the filtered DB, not the full store', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(
        makeUntrustedGroup(),
        false,
        'chatA@g.us',
      );
      const storeMount = mounts.find(
        (m) => m.containerPath === '/workspace/store',
      );
      expect(storeMount).toBeDefined();
      const expectedFilteredDir = path.join(
        DATA_DIR,
        'filtered-db',
        'untrusted-group',
      );
      expect(storeMount!.hostPath).toBe(expectedFilteredDir);
      // Critically: NOT the real store dir
      expect(storeMount!.hostPath).not.toBe(STORE_DIR);
      expect(storeMount!.hostPath).not.toBe(path.join(PROJECT_DIR, 'store'));
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('untrusted groups get zero secret-shadow mounts (main-only defense)', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      // Even if the secret files exist on disk, untrusted groups shouldn't
      // trigger the shadow logic — they don't mount the project root at all.
      for (const rel of SECRET_FILES) {
        const abs = path.join(PROJECT_DIR, rel);
        fs.mkdirSync(path.dirname(abs), { recursive: true });
        fs.writeFileSync(abs, 'SECRET=xyz');
      }
      const mounts = buildVolumeMounts(
        makeUntrustedGroup(),
        false,
        'chatA@g.us',
      );
      const shadowMounts = mounts.filter((m) => m.hostPath === '/dev/null');
      expect(shadowMounts.length).toBe(0);
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('/workspace/group/CLAUDE.md mounts the untrusted template, readonly (fixes #153 for untrusted)', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(
        makeUntrustedGroup(),
        false,
        'chatA@g.us',
      );
      const claudeMdMount = mounts.find(
        (m) => m.containerPath === '/workspace/group/CLAUDE.md',
      );
      expect(claudeMdMount).toBeDefined();
      expect(claudeMdMount!.hostPath).toBe(
        path.join(GROUPS_DIR, 'global', 'CLAUDE-untrusted.md'),
      );
      expect(claudeMdMount!.readonly).toBe(true);
      // Critically: NOT the per-group CLAUDE.md (which is the bug source
      // — that copy was made once at registration and never reconciled
      // on trust flips).
      expect(claudeMdMount!.hostPath).not.toBe(
        path.join(GROUPS_DIR, 'untrusted-group', 'CLAUDE.md'),
      );
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('creates an empty CLAUDE.md placeholder on the host when the group folder lacks one (fixes #442 readonly mount target)', () => {
    // Symptom of #442: the /workspace/group bind-mount is readonly for
    // untrusted groups, so runc cannot create a missing target file
    // when overlaying the CLAUDE.md bind on top — Docker exits 125
    // with `read-only file system`. The fix is to ensure the host-side
    // target exists before the mount layer hands the spec to Docker.
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const groupDir = path.join(GROUPS_DIR, 'untrusted-group');
      const placeholderTarget = path.join(groupDir, 'CLAUDE.md');
      // Pre-condition: file does NOT exist on the host (mirrors the
      // post-#164 state for groups whose vanilla CLAUDE.md was removed
      // by the migration).
      if (fs.existsSync(placeholderTarget)) {
        fs.unlinkSync(placeholderTarget);
      }
      expect(fs.existsSync(placeholderTarget)).toBe(false);

      buildVolumeMounts(makeUntrustedGroup(), false, 'chatA@g.us');

      // Post-condition: an empty placeholder exists so runc has
      // something to overlay the CLAUDE.md bind onto.
      expect(fs.existsSync(placeholderTarget)).toBe(true);
      expect(fs.readFileSync(placeholderTarget, 'utf8')).toBe('');
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('does not overwrite an existing CLAUDE.md placeholder (idempotent)', () => {
    // Defence against the placeholder being touched repeatedly on every
    // spawn. If a prior spawn left content behind (e.g. from a pre-#164
    // per-group CLAUDE.md the migration didn't match), keep it — the
    // bind-mount shadows it from the container's view anyway.
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const groupDir = path.join(GROUPS_DIR, 'untrusted-group');
      const placeholderTarget = path.join(groupDir, 'CLAUDE.md');
      const sentinel = '# pre-existing per-group CLAUDE.md\n';
      fs.writeFileSync(placeholderTarget, sentinel);

      buildVolumeMounts(makeUntrustedGroup(), false, 'chatA@g.us');

      expect(fs.readFileSync(placeholderTarget, 'utf8')).toBe(sentinel);
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('untrusted groups get SOUL-untrusted.md, not the full global dir', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(
        makeUntrustedGroup(),
        false,
        'chatA@g.us',
      );
      const soulMount = mounts.find(
        (m) => m.containerPath === '/workspace/global/SOUL.md',
      );
      expect(soulMount).toBeDefined();
      expect(soulMount!.hostPath).toBe(
        path.join(GROUPS_DIR, 'global', 'SOUL-untrusted.md'),
      );
      expect(soulMount!.readonly).toBe(true);
      // And there's NO mount of the full global dir
      const globalDirMount = mounts.find(
        (m) => m.containerPath === '/workspace/global',
      );
      expect(globalDirMount).toBeUndefined();
    } finally {
      process.chdir(originalCwd);
    }
  });
});

// -----------------------------------------------------------------------------
// Test 4 — trusted (non-main) group gets the trusted CLAUDE.md template
// mounted readonly over the writable group folder. Same #153 fix as the
// untrusted case, different source file. The mount layering means the
// agent can write anywhere in /workspace/group EXCEPT CLAUDE.md.
// -----------------------------------------------------------------------------
describe('buildVolumeMounts — trusted group CLAUDE.md mount', () => {
  function makeTrustedGroup(): RegisteredGroup {
    return {
      name: 'Trusted',
      folder: 'trusted-group',
      trigger: '@T',
      added_at: new Date().toISOString(),
      containerConfig: { trusted: true },
    };
  }

  beforeEach(() => {
    seedMessagesDb();
    fs.mkdirSync(path.join(GROUPS_DIR, 'trusted-group'), { recursive: true });
    const globalDir = path.join(GROUPS_DIR, 'global');
    fs.mkdirSync(globalDir, { recursive: true });
    fs.writeFileSync(
      path.join(globalDir, 'CLAUDE.md'),
      '**THIS IS A TRUSTED GROUP.**\n',
    );
    fs.writeFileSync(
      path.join(globalDir, 'CLAUDE-untrusted.md'),
      '**THIS IS AN UNTRUSTED GROUP.**\n',
    );
  });

  it('/workspace/group/CLAUDE.md mounts the trusted template, readonly', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(makeTrustedGroup(), false, 'chatT@g.us');
      const claudeMdMount = mounts.find(
        (m) => m.containerPath === '/workspace/group/CLAUDE.md',
      );
      expect(claudeMdMount).toBeDefined();
      expect(claudeMdMount!.hostPath).toBe(
        path.join(GROUPS_DIR, 'global', 'CLAUDE.md'),
      );
      expect(claudeMdMount!.readonly).toBe(true);
      expect(claudeMdMount!.hostPath).not.toBe(
        path.join(GROUPS_DIR, 'global', 'CLAUDE-untrusted.md'),
      );
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('/workspace/group folder mount stays writable for trusted (CLAUDE.md is the only readonly file)', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(makeTrustedGroup(), false, 'chatT@g.us');
      const groupFolderMount = mounts.find(
        (m) => m.containerPath === '/workspace/group',
      );
      expect(groupFolderMount).toBeDefined();
      expect(groupFolderMount!.readonly).toBe(false);
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('CLAUDE.md mount is positioned AFTER the group folder mount so the file shadow takes effect', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(makeTrustedGroup(), false, 'chatT@g.us');
      const folderIdx = mounts.findIndex(
        (m) => m.containerPath === '/workspace/group',
      );
      const fileIdx = mounts.findIndex(
        (m) => m.containerPath === '/workspace/group/CLAUDE.md',
      );
      expect(folderIdx).toBeGreaterThanOrEqual(0);
      expect(fileIdx).toBeGreaterThanOrEqual(0);
      expect(fileIdx).toBeGreaterThan(folderIdx);
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('does NOT write a host-side CLAUDE.md placeholder for trusted groups (#442 review)', () => {
    // Trusted groups have a RW parent mount so runc creates the
    // bind-mount target itself — no host-side placeholder needed.
    // Writing one anyway would pollute `scripts/migrate-thin-claude-md.ts`,
    // which classifies any present non-vanilla `CLAUDE.md` as
    // customized and refuses to migrate it. Gate the placeholder write
    // narrowly to the actual failing condition (untrusted non-main).
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const groupDir = path.join(GROUPS_DIR, 'trusted-group');
      const placeholderTarget = path.join(groupDir, 'CLAUDE.md');
      if (fs.existsSync(placeholderTarget)) {
        fs.unlinkSync(placeholderTarget);
      }
      expect(fs.existsSync(placeholderTarget)).toBe(false);

      buildVolumeMounts(makeTrustedGroup(), false, 'chatT@g.us');

      // Post-condition: no host-side placeholder was created. Docker
      // creates the target inside the RW container overlay at spawn
      // time without polluting the host's group folder.
      expect(fs.existsSync(placeholderTarget)).toBe(false);
    } finally {
      process.chdir(originalCwd);
    }
  });
});

// -----------------------------------------------------------------------------
// Shared auto-memory mount (issue #57): both session containers must see the
// same `/home/node/.claude/projects/-workspace-group/memory/` path. Owner-
// level state (feedback files) doesn't belong split per-session.
// -----------------------------------------------------------------------------
describe('buildVolumeMounts — shared-memory mount', () => {
  function makeMainGroup(): RegisteredGroup {
    return {
      name: 'Main',
      folder: 'shared-memory-test-group',
      trigger: '@Main',
      added_at: new Date().toISOString(),
      isMain: true,
    };
  }

  beforeEach(() => {
    seedMessagesDb();
    fs.mkdirSync(path.join(GROUPS_DIR, 'shared-memory-test-group'), {
      recursive: true,
    });
  });

  // Wrapper: `buildVolumeMounts` resolves several paths via `process.cwd()`
  // (trusted/, store/, tessl-workspace/). Without this chdir the test
  // would touch the real repo working dir and become environment-dependent
  // (e.g. would behave differently if the dev has a top-level `.env`).
  function withProjectCwd<T>(fn: () => T): T {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      return fn();
    } finally {
      process.chdir(originalCwd);
    }
  }

  it('both default and maintenance sessions mount the SAME shared-memory host dir over the project memory/ path', () => {
    withProjectCwd(() => {
      const group = makeMainGroup();

      const defaultMounts = buildVolumeMounts(
        group,
        true,
        'main@g.us',
        'default',
      );
      const maintenanceMounts = buildVolumeMounts(
        group,
        true,
        'main@g.us',
        'maintenance',
      );

      const expectedContainerPath =
        '/home/node/.claude/projects/-workspace-group/memory';
      const defaultMemoryMount = defaultMounts.find(
        (m) => m.containerPath === expectedContainerPath,
      );
      const maintenanceMemoryMount = maintenanceMounts.find(
        (m) => m.containerPath === expectedContainerPath,
      );

      expect(defaultMemoryMount).toBeDefined();
      expect(maintenanceMemoryMount).toBeDefined();
      expect(defaultMemoryMount!.readonly).toBe(false);
      expect(maintenanceMemoryMount!.readonly).toBe(false);

      // Same host dir for both sessions — this is the whole point.
      expect(defaultMemoryMount!.hostPath).toBe(
        maintenanceMemoryMount!.hostPath,
      );
      // And the host dir is session-independent (lives under the group's
      // sessions/ root, not under a per-session subdir).
      expect(defaultMemoryMount!.hostPath).toMatch(
        /sessions\/shared-memory-test-group\/shared-memory$/,
      );
    });
  });

  it('migrates pre-existing per-session memory files into shared-memory on spawn', () => {
    withProjectCwd(() => {
      const group = makeMainGroup();

      // Simulate a pre-#57 deployment: the default session has an accumulated
      // feedback file under its per-session .claude/.
      const perSessionMemoryDir = path.join(
        DATA_DIR,
        'sessions',
        group.folder,
        'default',
        '.claude',
        'projects',
        '-workspace-group',
        'memory',
      );
      fs.mkdirSync(perSessionMemoryDir, { recursive: true });
      fs.writeFileSync(
        path.join(perSessionMemoryDir, 'feedback_no_day_zero_debt.md'),
        '# pre-#57 feedback file',
      );

      // Build mounts (which runs the migration loop).
      buildVolumeMounts(group, true, 'main@g.us', 'default');

      const sharedMemoryDir = path.join(
        DATA_DIR,
        'sessions',
        group.folder,
        'shared-memory',
      );
      const migratedFile = path.join(
        sharedMemoryDir,
        'feedback_no_day_zero_debt.md',
      );
      expect(fs.existsSync(migratedFile)).toBe(true);
      expect(fs.readFileSync(migratedFile, 'utf-8')).toBe(
        '# pre-#57 feedback file',
      );
    });
  });

  it('migration prefers existing shared-memory content over per-session (shared wins on conflict)', () => {
    withProjectCwd(() => {
      const group = makeMainGroup();

      const sharedMemoryDir = path.join(
        DATA_DIR,
        'sessions',
        group.folder,
        'shared-memory',
      );
      fs.mkdirSync(sharedMemoryDir, { recursive: true });
      const sharedFile = path.join(sharedMemoryDir, 'feedback.md');
      fs.writeFileSync(sharedFile, 'shared wins');

      // Per-session has a DIFFERENT copy of the same file.
      const perSessionMemoryDir = path.join(
        DATA_DIR,
        'sessions',
        group.folder,
        'maintenance',
        '.claude',
        'projects',
        '-workspace-group',
        'memory',
      );
      fs.mkdirSync(perSessionMemoryDir, { recursive: true });
      fs.writeFileSync(
        path.join(perSessionMemoryDir, 'feedback.md'),
        'per-session stale content',
      );

      buildVolumeMounts(group, true, 'main@g.us', 'maintenance');

      // Shared copy was NOT overwritten.
      expect(fs.readFileSync(sharedFile, 'utf-8')).toBe('shared wins');
    });
  });

  it('untrusted group gets NO shared-memory mount (auto-memory disabled, shared writable owner state would be a poisoning vector)', () => {
    withProjectCwd(() => {
      // Untrusted tier — settings.json sets CLAUDE_CODE_DISABLE_AUTO_MEMORY=1.
      // The shared-memory mount MUST be skipped so this container has no
      // shared writable owner-state dir to poison.
      const untrustedGroup: RegisteredGroup = {
        name: 'Untrusted',
        folder: 'untrusted-memory-test',
        trigger: '@U',
        added_at: new Date().toISOString(),
        // no isMain, no containerConfig.trusted → untrusted tier
      };
      fs.mkdirSync(path.join(GROUPS_DIR, 'untrusted-memory-test'), {
        recursive: true,
      });
      fs.mkdirSync(path.join(GROUPS_DIR, 'global'), { recursive: true });
      fs.writeFileSync(
        path.join(GROUPS_DIR, 'global', 'SOUL-untrusted.md'),
        '# stub',
      );

      const mounts = buildVolumeMounts(
        untrustedGroup,
        false,
        'untrusted@g.us',
        'default',
      );

      const memoryMount = mounts.find(
        (m) =>
          m.containerPath ===
          '/home/node/.claude/projects/-workspace-group/memory',
      );
      expect(memoryMount).toBeUndefined();

      // And the host dir wasn't created either — untrusted doesn't need
      // any shared-memory state at all.
      const sharedMemoryDir = path.join(
        DATA_DIR,
        'sessions',
        'untrusted-memory-test',
        'shared-memory',
      );
      expect(fs.existsSync(sharedMemoryDir)).toBe(false);
    });
  });
});

// -----------------------------------------------------------------------------
// Issue #287 / #288 review — pre-spawn IPC sweep is handoff-aware.
//
// `buildVolumeMounts` calls `sweepStaleInputs(sessionInputDir, 0)` to wipe
// leftover IPC inputs from previous container lifecycles before a fresh
// spawn. That is correct OUTSIDE a graceful-shutdown handoff window, but
// during one an adopted-but-still-running container from the previous
// orchestrator may share this session's input dir with the fresh spawn —
// and a `graceMs = 0` sweep would unlink files the adopted container
// hasn't drained yet. The pre-spawn sweep must therefore skip while
// `isHandoffActive()` is true.
// -----------------------------------------------------------------------------
import { _resetHandoffWindowForTests, markHandoffActive } from './handoff.js';

describe('buildVolumeMounts — pre-spawn IPC sweep skips during handoff (#288)', () => {
  beforeEach(() => {
    seedMessagesDb();
    _resetHandoffWindowForTests();
  });

  function makeUntrustedGroup(): RegisteredGroup {
    return {
      name: 'Untrusted',
      folder: 'sweep-handoff-test',
      trigger: '@U',
      added_at: new Date().toISOString(),
      containerConfig: { trusted: false },
    };
  }

  function seedSessionInputDir(): {
    inputDir: string;
    plantedFile: string;
  } {
    // buildVolumeMounts calls fs.mkdirSync(sessionInputDir, { recursive: true })
    // so the planted file must be written AFTER buildVolumeMounts ensures
    // the dir exists — but the sweep happens INSIDE buildVolumeMounts.
    // Pre-create the dir + file here so the sweep sees it on first call.
    const inputDir = path.join(
      DATA_DIR,
      'ipc',
      'sweep-handoff-test',
      'input-default',
    );
    fs.mkdirSync(inputDir, { recursive: true });
    fs.mkdirSync(path.join(GROUPS_DIR, 'sweep-handoff-test'), {
      recursive: true,
    });
    const planted = path.join(
      inputDir,
      `${Date.now() - 999_999_999}-aaaa.json`,
    );
    fs.writeFileSync(planted, '{"type":"message","text":"unread"}');
    return { inputDir, plantedFile: planted };
  }

  it('sweeps stale IPC inputs by default (no handoff active)', () => {
    const { plantedFile } = seedSessionInputDir();
    expect(fs.existsSync(plantedFile)).toBe(true);

    buildVolumeMounts(makeUntrustedGroup(), false, 'sweep-default@g.us');

    expect(fs.existsSync(plantedFile)).toBe(false);
  });

  it('does NOT sweep IPC inputs while a handoff window is active', () => {
    const { plantedFile } = seedSessionInputDir();
    markHandoffActive();
    expect(fs.existsSync(plantedFile)).toBe(true);

    buildVolumeMounts(makeUntrustedGroup(), false, 'sweep-handoff@g.us');

    // Adopted container may not have drained this file yet — sweep must
    // leave it in place so the adopted container's next IPC poll can
    // still see it.
    expect(fs.existsSync(plantedFile)).toBe(true);
  });

  it('resumes sweeping once the handoff window expires', () => {
    const { plantedFile } = seedSessionInputDir();
    markHandoffActive();
    // Force the window to be over by resetting in-process state — the
    // public API doesn't expose "set window in the past", but reset is
    // the documented test hook and represents the same logical state
    // (no active handoff).
    _resetHandoffWindowForTests();

    buildVolumeMounts(makeUntrustedGroup(), false, 'sweep-post-handoff@g.us');

    expect(fs.existsSync(plantedFile)).toBe(false);
  });
});

describe('buildVolumeMounts — /workspace/store mount rw/ro by trust (epic #293)', () => {
  function makeMainGroup(): RegisteredGroup {
    return {
      name: 'Main',
      folder: 'main',
      trigger: '@Andy',
      added_at: new Date().toISOString(),
    };
  }

  function makeTrustedGroup(): RegisteredGroup {
    return {
      name: 'Trusted',
      folder: 'trusted-group',
      trigger: '@T',
      added_at: new Date().toISOString(),
      containerConfig: { trusted: true },
    };
  }

  beforeEach(() => {
    // buildVolumeMounts writes a managed AGENTS.md into the group dir
    // on first call, so the dir has to exist. The store mount resolves
    // off `path.join(process.cwd(), 'store')` (process.cwd() is chdir'd
    // to PROJECT_DIR below), so the dir has to exist there too — the
    // mocked STORE_DIR constant is unrelated to how the mount path is
    // built in container-runner.ts.
    seedMessagesDb();
    fs.mkdirSync(path.join(GROUPS_DIR, 'main'), { recursive: true });
    fs.mkdirSync(path.join(GROUPS_DIR, 'trusted-group'), { recursive: true });
    fs.mkdirSync(path.join(PROJECT_DIR, 'store'), { recursive: true });
  });

  // Trusted/main containers need rw access on store/ so skills can write
  // to the per-skill state tables (orders, email_feedback, …) created by
  // the orchestrator's state-NNN-* migrations. Without rw, apply-order.py
  // / write-orders-metadata.py / etc. fail with "attempt to write a
  // readonly database" — production-observed bug after the #294 tile flip.

  it('mounts /workspace/store read-write for the main group', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(makeMainGroup(), true, 'main@g.us');
      const storeMount = mounts.find(
        (m) => m.containerPath === '/workspace/store',
      );
      expect(storeMount).toBeDefined();
      expect(storeMount!.readonly).toBe(false);
    } finally {
      process.chdir(originalCwd);
    }
  });

  it('mounts /workspace/store read-write for trusted (non-main) groups', () => {
    const originalCwd = process.cwd();
    process.chdir(PROJECT_DIR);
    try {
      const mounts = buildVolumeMounts(
        makeTrustedGroup(),
        false,
        'trusted@g.us',
      );
      const storeMount = mounts.find(
        (m) => m.containerPath === '/workspace/store',
      );
      expect(storeMount).toBeDefined();
      expect(storeMount!.readonly).toBe(false);
    } finally {
      process.chdir(originalCwd);
    }
  });
});
