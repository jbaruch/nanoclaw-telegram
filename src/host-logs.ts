import fs from 'fs';
import path from 'path';

import { isFsErrorWithCode } from './fs-errors.js';

import { DATA_DIR, HOST_GID, HOST_UID } from './config.js';

// Errno codes a best-effort log-dir walk/prune (readdir, stat, unlink) may
// legitimately hit — including the path-resolution errnos (ENOTDIR/ELOOP/
// ENAMETOOLONG) a component swap can raise. Anything else propagates.
const LOG_FS_PROBE_CODES = [
  'EACCES',
  'EPERM',
  'ENOENT',
  'EISDIR',
  'EBUSY',
  'ENOTDIR',
  'ELOOP',
  'ENAMETOOLONG',
];

/**
 * Host log artifacts the admin tile reads via `/workspace/host-logs/`
 * (mounted read-only from this directory). Two sub-trees:
 *
 *   - `orchestrator.log`     — the orchestrator's own log lines, written
 *                              by `logger.ts` in addition to stdout/stderr.
 *                              Rotated to `.1` when it exceeds the size cap.
 *   - `containers/<folder>/<session>/<iso>.log`
 *                            — per-spawn streaming log of container
 *                              stdout/stderr, opened at spawn and closed
 *                              on exit. Distinct from the existing
 *                              post-exit summary at `groups/<folder>/logs/`.
 *
 * (A `state/snapshot.json` writer was contemplated for the issue's
 * "host-side state" deliverable but the existing `chat_status` MCP
 * tool already exposes the same data on demand. Adding a periodic
 * file writer would duplicate that data; the only marginal value is
 * reachability when the orchestrator is unresponsive, and the
 * recovery path for that scenario is `/scripts/deploy.sh` rather
 * than reading a snapshot file. Folded into the issue's followup if
 * the orchestrator-down case becomes load-bearing in practice.)
 *
 * Only the admin tile gets this directory mounted in. Untrusted / trusted /
 * core / host tiles must NOT receive it — these files are inherently
 * cross-chat (orchestrator log, all groups' container output, all groups'
 * state).
 *
 * Paths are exposed as getters (not module-level `const` strings) so
 * tests that mock `DATA_DIR` after this module is imported still see
 * the mocked value. A `const HOST_LOGS_DIR = path.join(DATA_DIR, ...)`
 * at module top would freeze whatever `DATA_DIR` resolved to at import
 * time — vi.mock factories run async, so the mock can land AFTER an
 * indirect import has already evaluated the const, leaving the test
 * with a path under the real (unmocked) DATA_DIR.
 *
 * The static `import { DATA_DIR } from './config.js'` here creates a
 * potential cycle (config → env → logger → host-logs → config); the
 * cycle is broken at env.ts which lazy-loads logger via require() so
 * env never completes via the import-of-logger path during the cycle.
 * See env.ts for the why.
 */
export function hostLogsDir(): string {
  return path.join(DATA_DIR, 'host-logs');
}
export function hostLogsContainersDir(): string {
  return path.join(hostLogsDir(), 'containers');
}
export function hostLogsStateDir(): string {
  return path.join(hostLogsDir(), 'state');
}
export function hostLogsOrchestratorFile(): string {
  return path.join(hostLogsDir(), 'orchestrator.log');
}

// Retention for per-container streaming logs. Long enough to cover a
// long weekend of debugging history; short enough that the directory
// can't fill the host's disk over months. Seven days is the same
// floor `scripts/logrotate.sh` lands on for the orchestrator log.
export const CONTAINER_LOG_RETENTION_MS = 7 * 24 * 60 * 60 * 1000;

// 10 MB matches `scripts/logrotate.sh`'s rotation threshold for the
// orchestrator log file. Keep them in sync — the orchestrator and the
// rotation script agree on what "too big" means.
export const ORCHESTRATOR_LOG_MAX_BYTES = 10 * 1024 * 1024;

/**
 * Best-effort directory bootstrap. Called from the orchestrator at
 * startup AND lazily by the logger (logger may run before the
 * orchestrator's startup hook on some import-order paths). Failure is
 * non-fatal — callers fall back to stdout-only logging or skip the
 * host-logs mount entirely.
 *
 * Returns `true` if all three directories exist after the call,
 * `false` if any mkdirSync failed (EACCES, EROFS, ENOSPC, etc.).
 * Each call is idempotent; failures don't half-complete state.
 */
export function ensureHostLogDirs(): boolean {
  // Wrap each mkdirSync individually so a failure on the second or
  // third doesn't unwind progress on the first (mkdir is idempotent
  // anyway, but we should still keep the partial-success state if it
  // helps later callers — e.g. the logger sink only needs the root
  // dir, not containers/ or state/).
  //
  // After each successful mkdirSync, chown the directory to
  // HOST_UID/HOST_GID so host-side writers (`scripts/deploy.sh`
  // appending to `data/host-logs/deploy-kills.log`, log rotation,
  // operator inspection) can write into a tree the orchestrator
  // container's root user may have created. The chown runs whether
  // the dir was newly created or already existed (mkdirSync recursive
  // is a no-op on existing dirs) — that's intentional, it repairs
  // pre-existing root:root state from before this fix landed. Without
  // the chown the bind-mount surfaces root:root to the host
  // filesystem, and the host user gets EACCES — see #254.
  let ok = true;
  for (const dir of [
    hostLogsDir(),
    hostLogsContainersDir(),
    hostLogsStateDir(),
  ]) {
    try {
      fs.mkdirSync(dir, { recursive: true });
    } catch (err: unknown) {
      // Discriminate by `err.code`: anything with an errno string
      // (EACCES, EROFS, ENOSPC, EPERM, EEXIST when the path exists as
      // a file, ENOTDIR when a parent component is a file, transient
      // ENOENT, EIO, etc.) is a filesystem-state failure — the
      // fail-open contract holds and the caller proceeds without
      // host-logs visibility. Only errors WITHOUT a `code` (TypeError
      // from a malformed argument, ReferenceError, etc.) indicate a
      // programmer bug and propagate, so unexpected exceptions are
      // never silently swallowed.
      const code = (err as NodeJS.ErrnoException)?.code;
      if (typeof code === 'string') {
        ok = false;
        continue;
      }
      throw err;
    }
    chownToHostUser(dir);
  }
  return ok;
}

/**
 * Best-effort chown to the host operator's uid/gid. Skipped when
 * HOST_UID/HOST_GID aren't set (running directly on the host, not
 * docker-out-of-docker — the orchestrator already owns the file it
 * just created) or when HOST_UID is 0 (matches the chown-skip pattern
 * elsewhere — `container-runner.ts` filtered-DB / state-dir paths —
 * for the case where the in-container user is already root).
 *
 * Uses `lchownSync`, not `chownSync`, to match the symlink-safety
 * posture of `chownRecursive` in `container-runner.ts`. A container
 * with write access to a bind-mounted parent could theoretically
 * replace `data/host-logs/` with a symlink to `/etc/passwd` between
 * mkdir and chown; `lchownSync` operates on the link itself and
 * keeps that escalation path closed.
 *
 * Filesystem failures are tolerated because the orchestrator may run
 * without CAP_CHOWN on some bind targets (user namespaces, restricted
 * mounts), and the dir can race away between mkdir and chown
 * (ENOENT). A chown that didn't take just means the directory keeps
 * orchestrator-container ownership and the host-side writer hits the
 * same "fail open with a warning" path it would have hit before this
 * fix. The catch discriminates by `err.code`: any errno string is a
 * filesystem-state error and is swallowed; anything without a code
 * (TypeError from a malformed argument, etc.) is a programmer bug
 * and propagates.
 *
 * Negative uid/gid values are rejected at the validation gate:
 * `lchownSync(-1, -1)` throws `RangeError` with `code: ERR_OUT_OF_RANGE`,
 * which would technically pass the errno-string check below but
 * indicates a real misconfiguration that we'd rather skip cleanly
 * than swallow as a logged-elsewhere failure. Filtering to
 * non-negative integers up front keeps a misconfigured
 * `HOST_UID=-1` from taking the orchestrator down silently.
 */
function chownToHostUser(dir: string): void {
  if (HOST_UID === undefined || HOST_GID === undefined) return;
  if (!Number.isInteger(HOST_UID) || !Number.isInteger(HOST_GID)) return;
  if (HOST_UID < 0 || HOST_GID < 0) return;
  if (HOST_UID === 0) return;
  try {
    fs.lchownSync(dir, HOST_UID, HOST_GID);
  } catch (err: unknown) {
    const code = (err as NodeJS.ErrnoException)?.code;
    if (typeof code !== 'string') throw err;
  }
}

/**
 * Where the streaming log file for a single container spawn should live.
 * Caller passes the spawn timestamp explicitly so the filename is
 * tied to the spawn moment (millisecond precision), keeping per-spawn
 * files distinct under a high-spawn-rate operator workload.
 *
 * Note: this filename is currently NOT cross-referenced by the existing
 * post-exit summary writer at `groups/<folder>/logs/`. The summary uses
 * its own exit-time `new Date().toISOString()` and the two paths live
 * in different trees serving different audiences (host operator vs.
 * admin tile). If a future change wants symmetry, both writers should
 * accept the spawn timestamp from the caller and stamp the same
 * filename.
 */
export function containerLogPath(
  groupFolder: string,
  sessionName: string,
  startedAt: Date,
): string {
  // Replace `:` and `.` so the filename is portable across filesystems
  // (FAT32 / Windows shares occasionally appear in NAS scenarios).
  const safeTs = startedAt.toISOString().replace(/[:.]/g, '-');
  return path.join(
    hostLogsContainersDir(),
    groupFolder,
    sessionName,
    `${safeTs}.log`,
  );
}

/**
 * Walk the containers directory and delete `.log` files older than the
 * retention cutoff. Idempotent — safe to call repeatedly. Failure on a
 * single file is logged at the call site (this module doesn't import
 * the logger to keep its dep graph minimal — `logger` already imports
 * from this module's sibling `config`).
 *
 * Returns the count of deleted files for the caller to log.
 */
export function pruneOldContainerLogs(now: Date = new Date()): number {
  const containersDir = hostLogsContainersDir();
  if (!fs.existsSync(containersDir)) return 0;
  const cutoffMs = now.getTime() - CONTAINER_LOG_RETENTION_MS;
  let deleted = 0;
  // Two-level walk: containers/<folder>/<session>/<file>.log. We don't
  // want to recurse arbitrarily deep — the structure is fixed, and
  // reading every directory under DATA_DIR would be a footgun if a
  // future rename misplaced something.
  for (const folder of safeReaddir(containersDir)) {
    const folderPath = path.join(containersDir, folder);
    if (!safeIsDir(folderPath)) continue;
    for (const session of safeReaddir(folderPath)) {
      const sessionPath = path.join(folderPath, session);
      if (!safeIsDir(sessionPath)) continue;
      for (const file of safeReaddir(sessionPath)) {
        if (!file.endsWith('.log')) continue;
        const filePath = path.join(sessionPath, file);
        const stat = safeStat(filePath);
        if (!stat) continue;
        if (stat.mtimeMs < cutoffMs) {
          try {
            fs.unlinkSync(filePath);
            deleted++;
          } catch (err) {
            if (!isFsErrorWithCode(err, LOG_FS_PROBE_CODES)) throw err;
            // File raced with another prune or was renamed; skip.
          }
        }
      }
    }
  }
  return deleted;
}

function safeReaddir(p: string): string[] {
  try {
    return fs.readdirSync(p);
  } catch (err) {
    if (!isFsErrorWithCode(err, LOG_FS_PROBE_CODES)) throw err;
    return [];
  }
}

function safeIsDir(p: string): boolean {
  try {
    return fs.statSync(p).isDirectory();
  } catch (err) {
    if (!isFsErrorWithCode(err, LOG_FS_PROBE_CODES)) throw err;
    return false;
  }
}

function safeStat(p: string): fs.Stats | null {
  try {
    return fs.statSync(p);
  } catch (err) {
    if (!isFsErrorWithCode(err, LOG_FS_PROBE_CODES)) throw err;
    return null;
  }
}

/**
 * Strip ANSI escape sequences from a log line. The orchestrator's
 * console output is colorized for human reading, but the file sink and
 * snapshot consumers just want the text — colors render as garbled
 * `\x1b[33m` literals in tools that don't interpret them (most editors,
 * `cat` over SSH, the admin tile's file reads). Exported for the test
 * suite.
 */
// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;]*[A-Za-z]/g;
export function stripAnsi(text: string): string {
  return text.replace(ANSI_RE, '');
}
