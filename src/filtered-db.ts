// Filtered per-group DB snapshot (#851 slice 4, extracted verbatim
// from src/container-runner.ts).
//
// Security-critical untrusted-group DB isolation: one group's slice
// of messages.db, built ATTACH+CTAS into a self-contained DELETE-
// journal snapshot (#287) with atomic temp+rename publish (#93) and
// a one-shot WAL/SHM recovery retry (#100).
import { randomBytes } from 'crypto';
import fs from 'fs';
import path from 'path';

import Database, { SqliteError } from 'better-sqlite3';

import { DATA_DIR, HOST_GID, HOST_UID, STORE_DIR } from './config.js';
import {
  CR_FS_CODES,
  isErrnoCodedError,
  isFsErrorWithCode,
} from './fs-errors.js';
import { logger } from './logger.js';

/**
 * Try to repair the source `messages.db` WAL/SHM state by running a
 * TRUNCATE checkpoint via a short-lived dedicated connection. Used as
 * a recovery path when `createFilteredDb`'s ATTACH throws "disk I/O error"
 * — most commonly when the source's `-shm` file has been evicted (Docker
 * bind-mount on macOS, OS resource pressure, or partial writer crash) and
 * SQLite can no longer establish WAL read coordination.
 *
 * After this call returns, the next ATTACH attempt should succeed
 * assuming the underlying filesystem is healthy. Idempotent and safe to
 * call when state is already clean.
 *
 * Logged loudly because hitting this path indicates an environment
 * problem (Docker bind-mount eviction, OS resource pressure, partial
 * writer crash); the operator should know it ran. See issue #100.
 */
function recoverSourceWalState(srcDb: string): void {
  logger.warn(
    { srcDb },
    'createFilteredDb: source WAL/SHM state appears degraded, running checkpoint(TRUNCATE) recovery',
  );
  const recovery = new Database(srcDb);
  try {
    recovery.pragma('busy_timeout = 5000');
    // TRUNCATE checkpoint forces all -wal content into the main db
    // and zero-truncates -wal. As a side effect the writer connection
    // re-establishes -shm coordination. PASSIVE/RESTART would also
    // work, but TRUNCATE is the most aggressive and most likely to
    // un-stick a bad state.
    recovery.pragma('wal_checkpoint(TRUNCATE)');
  } finally {
    recovery.close();
  }
}

/**
 * Per `coding-policy: error-handling`: narrow to the specific exception
 * type we expect, rethrow everything else. The retry path only triggers
 * when better-sqlite3 throws a typed `SqliteError` carrying the "disk
 * I/O error" substring (the symptom of degraded source WAL/SHM
 * coordination per `ligolnik#100`). Anything else — a wrapped error, a
 * mocked error in tests, a totally unrelated `Error` whose message
 * happens to mention disk I/O — gates `false` so the call site rethrows
 * instead of running a recovery that doesn't apply.
 */
function isDiskIoError(err: unknown): err is SqliteError {
  return err instanceof SqliteError && err.message.includes('disk I/O error');
}

/**
 * Create a filtered copy of messages.db containing only one group's messages.
 * Returns the path to the filtered DB, or null if the source DB doesn't exist.
 *
 * @internal Exported for tests only — untrusted-group DB isolation is
 *   security-critical and must be pinned by regression tests.
 */
export function createFilteredDb(
  chatJid: string,
  groupFolder: string,
): string | null {
  const srcDb = path.join(STORE_DIR, 'messages.db');
  if (!fs.existsSync(srcDb)) return null;

  const filteredDir = path.join(DATA_DIR, 'filtered-db', groupFolder);
  fs.mkdirSync(filteredDir, { recursive: true });
  const filteredPath = path.join(filteredDir, 'messages.db');

  // Remove stale copy from previous run, including any `-wal`/`-shm`
  // sidecars left behind by a pre-#287 version that ran before the
  // `journal_mode = DELETE` pragma below was in place. Without this,
  // operators upgrading on top of an existing data dir keep the old
  // WAL artefacts indefinitely — both as wasted disk and as the same
  // RO-mount-can't-open failure the pragma is supposed to eliminate.
  // Sidecars are removed unconditionally (independent of whether the
  // main file existed) because a partial wipe — main DB removed but
  // sidecars left — is the exact state SQLite refuses to open.
  // `rmSync({ force: true })` so concurrent refreshes (default +
  // maintenance session for the same untrusted group) tolerate
  // another caller having already removed one or more of these paths.
  for (const suffix of ['', '-wal', '-shm']) {
    fs.rmSync(`${filteredPath}${suffix}`, { force: true });
  }

  /**
   * One full ATTACH + CTAS + atomic-rename attempt. Captured as a
   * closure so we can retry it once after `recoverSourceWalState`
   * (see #100). Each call regenerates its own temp path so a retry
   * never reuses a stale file from the failed first attempt.
   */
  const attemptCreate = (): void => {
    // Atomic temp-file + rename, see #93. Writing the schema directly to the
    // canonical path means any interruption (SIGTERM, OOM, fs hiccup, throw mid
    // ATTACH/CTAS) leaves a 0-byte or partial-header file there. Subsequent
    // spawns then hit "disk I/O error" the moment SQLite tries to read the
    // (missing) header, and the group is permanently wedged behind the circuit
    // breaker. Writing to a sibling temp path and renaming on success keeps the
    // canonical path either healthy or absent — never half-written. The temp
    // file MUST live in the same directory so the rename stays atomic on POSIX.
    // The trailing randomBytes(4) suffix defeats retry-collision: if the first
    // attempt failed mid-flight and the second attempt fires within the same
    // millisecond, pid+Date.now() alone could collide with the prior temp path.
    const tempPath = path.join(
      filteredDir,
      `messages.db.tmp-${process.pid}-${Date.now()}-${randomBytes(4).toString('hex')}`,
    );
    // Stale temp from a previously crashed run — unlink so `new Database` opens
    // a fresh file rather than reattaching to a corrupt one.
    if (fs.existsSync(tempPath)) {
      fs.unlinkSync(tempPath);
    }

    // Use ATTACH to copy schema-agnostically — picks up new columns automatically
    const dst = new Database(tempPath);
    // Source `messages.db` is WAL-mode and actively written by the orchestrator.
    // Without busy_timeout this connection would fail immediately on any lock
    // contention against the source (e.g. during a checkpoint), defeating the
    // whole point of the orchestrator-side WAL setup. Match the orchestrator
    // value (5000ms) so contention smoothing is symmetric across readers.
    dst.pragma('busy_timeout = 5000');
    // Force rollback-journal mode on the snapshot. better-sqlite3 defaults to
    // WAL, which requires the SQLite reader to write `-wal`/`-shm` sidecar
    // files even on opens that are logically read-only. The filtered DB is
    // mounted read-only into untrusted containers (via `fakeowner ro`); a
    // default `sqlite3.connect(path)` from inside the container then fails
    // with `unable to open database file` because the sidecars can't be
    // created. DELETE-journal makes the file self-contained — every reader's
    // default open works without per-script `?mode=ro&immutable=1` plumbing.
    // The filtered DB is a single-writer one-shot snapshot, so WAL gives it
    // nothing anyway. See issue #287.
    dst.pragma('journal_mode = DELETE');
    dst.pragma('synchronous = NORMAL');
    try {
      try {
        dst.exec(`ATTACH DATABASE '${srcDb.replace(/'/g, "''")}' AS src`);
        dst.exec(
          `CREATE TABLE chats AS SELECT * FROM src.chats WHERE jid = '${chatJid.replace(/'/g, "''")}'`,
        );
        dst.exec(
          `CREATE TABLE messages AS SELECT * FROM src.messages WHERE chat_jid = '${chatJid.replace(/'/g, "''")}'`,
        );
        dst.exec(
          'CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)',
        );
        // Reactions scoped to this chat only. Filtered-DB consumers JOIN
        // on this table; without it, those joins hit "no such table:
        // reactions" and abort. Created unconditionally so containers
        // don't depend on whether the host happens to have any reactions
        // yet — even an empty table satisfies the join. CTAS can't run
        // if src.reactions doesn't exist (fresh install before
        // migrations), so check `src.sqlite_master`
        // explicitly and fall back to an empty table with the known schema in
        // that one case. Bare try/catch would also swallow corruption, lock,
        // and permission errors — a missing table is the only fallback case
        // we want to absorb.
        const srcHasReactions = dst
          .prepare(
            "SELECT 1 FROM src.sqlite_master WHERE type = 'table' AND name = 'reactions' LIMIT 1",
          )
          .get();
        if (srcHasReactions) {
          dst.exec(`
            CREATE TABLE reactions AS
              SELECT r.* FROM src.reactions r
              WHERE r.message_chat_jid = '${chatJid.replace(/'/g, "''")}'
          `);
        } else {
          dst.exec(`
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
        }
        dst.exec(
          'CREATE INDEX IF NOT EXISTS idx_reactions_message ON reactions(message_id, message_chat_jid)',
        );
        dst.exec('DETACH src');
      } finally {
        dst.close();
      }
      // Atomic rename — temp file is now a fully-formed SQLite database. On
      // POSIX this is atomic within the same filesystem, so the canonical path
      // flips from "absent or stale" to "complete" with no observable midpoint.
      fs.renameSync(tempPath, filteredPath);
    } catch (err) {
      // Cleanup the partial temp file before rethrowing. Best-effort — if the
      // unlink itself fails for anything other than ENOENT (file already gone),
      // log it so we know about latent disk/permission issues, but don't
      // shadow the original error. The temp path lives next to the canonical
      // path, so leaving it around would also leak disk space across retries.
      try {
        fs.unlinkSync(tempPath);
      } catch (cleanupErr) {
        // Best-effort cleanup: tolerate ANY fs/OS errno so it can't shadow the
        // original rename/build error; only a non-errno defect propagates.
        if (!isErrnoCodedError(cleanupErr)) throw cleanupErr;
        const code = (cleanupErr as NodeJS.ErrnoException).code;
        if (code !== 'ENOENT') {
          logger.warn(
            { err: cleanupErr, tempPath },
            'Failed to clean up partial filtered-db temp file',
          );
        }
      }
      throw err;
    }
  };

  // Outer try with a single retry on "disk I/O error" — the symptom of
  // a degraded source WAL/SHM state (issue #100). Recovery runs a
  // wal_checkpoint(TRUNCATE) on the source via a fresh connection, which
  // re-establishes the -shm region; the retry then attempts the full
  // ATTACH+CTAS again. Retry happens EXACTLY ONCE — if it also fails the
  // error propagates and the circuit breaker can do its job (the FS or
  // data is genuinely broken at that point, not transient SHM eviction).
  //
  // Errors that are NOT disk-I/O (e.g. "no such table", schema bugs,
  // unparseable source) propagate immediately without recovery — there's
  // nothing a checkpoint can fix for those.
  try {
    attemptCreate();
  } catch (err) {
    if (!isDiskIoError(err)) throw err;
    recoverSourceWalState(srcDb);
    attemptCreate(); // retry once; any error here propagates
    logger.info(
      { srcDb, groupFolder },
      'createFilteredDb: succeeded after WAL/SHM recovery + retry',
    );
  }

  // Chown so container user can read
  const uid = HOST_UID ?? 1000;
  const gid = HOST_GID ?? 1000;
  if (uid !== 0) {
    try {
      fs.chownSync(filteredDir, uid, gid);
      fs.chownSync(filteredPath, uid, gid);
    } catch (err: unknown) {
      if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
      logger.warn({ err, filteredPath }, 'Failed to chown filtered DB');
    }
  }

  logger.debug(
    { chatJid, groupFolder, path: filteredPath },
    'Created filtered DB for untrusted container',
  );

  return filteredPath;
}
