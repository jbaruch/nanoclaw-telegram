import { hasErrno } from './operational-errors.js';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

import { DATA_DIR } from './config.js';
import { logger } from './logger.js';

/**
 * Sole runtime reader/writer of remote-control.json. Schema 1 contains
 * schema_version plus pid (positive integer), url, startedBy, startedInChat and
 * startedAt (strings). Accepted legacy records omit only schema_version and are
 * rewritten by this owner after live process verification. This occupied-session
 * record gates a second launch: under the gate-store exception, unsupported
 * versions refuse and are preserved instead of reading as an empty session.
 */
class UnsupportedRemoteControlStateError extends Error {}

interface RemoteControlSession {
  pid: number;
  url: string;
  startedBy: string;
  startedInChat: string;
  startedAt: string;
}

let activeSession: RemoteControlSession | null = null;

const URL_REGEX = /https:\/\/claude\.ai\/code\S+/;
const URL_TIMEOUT_MS = 30_000;
const URL_POLL_MS = 200;
const STATE_FILE = path.join(DATA_DIR, 'remote-control.json');
const STDOUT_FILE = path.join(DATA_DIR, 'remote-control.stdout');
const STDERR_FILE = path.join(DATA_DIR, 'remote-control.stderr');

function saveState(session: RemoteControlSession): void {
  fs.mkdirSync(path.dirname(STATE_FILE), { recursive: true });
  const temporary = fs.mkdtempSync(path.join(DATA_DIR, '.remote-control-'));
  const candidate = path.join(temporary, 'state.json');
  try {
    fs.writeFileSync(
      candidate,
      JSON.stringify({ schema_version: 1, ...session }),
      { mode: 0o600, flag: 'wx' },
    );
    fs.renameSync(candidate, STATE_FILE);
  } catch (error) {
    try {
      fs.rmSync(temporary, { recursive: true });
    } catch (cleanupError) {
      throw new AggregateError(
        [error, cleanupError],
        'Remote Control state write and temporary-file cleanup failed; inspect state storage before retrying.',
        { cause: cleanupError },
      );
    }
    throw error;
  }
  fs.rmSync(temporary, { recursive: true });
}

function clearState(): void {
  try {
    fs.unlinkSync(STATE_FILE);
  } catch (err) {
    if (!hasErrno(err, 'ENOENT')) throw err;
  }
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    if (!hasErrno(err, 'ESRCH')) throw err;
    return false;
  }
}

/**
 * Restore session from disk on startup.
 * If the process is still alive, adopt it. Otherwise, clean up.
 */
export function restoreRemoteControl(): void {
  let data: string;
  try {
    data = fs.readFileSync(STATE_FILE, 'utf-8');
  } catch (err) {
    if (!hasErrno(err, 'ENOENT')) throw err;
    return;
  }

  try {
    const parsed: unknown = JSON.parse(data);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new SyntaxError('Remote Control state must be an object');
    }
    const candidate = parsed as Record<string, unknown>;
    const legacy = !Object.hasOwn(candidate, 'schema_version');
    if (!legacy && candidate.schema_version !== 1)
      throw new UnsupportedRemoteControlStateError(
        'Unsupported Remote Control state version; update NanoClaw before restoring or starting another session.',
      );
    if (
      typeof candidate.pid !== 'number' ||
      !Number.isSafeInteger(candidate.pid) ||
      candidate.pid <= 0 ||
      typeof candidate.url !== 'string' ||
      typeof candidate.startedBy !== 'string' ||
      typeof candidate.startedInChat !== 'string' ||
      typeof candidate.startedAt !== 'string'
    ) {
      throw new SyntaxError('Remote Control state has invalid session fields');
    }
    const session: RemoteControlSession = {
      pid: candidate.pid,
      url: candidate.url,
      startedBy: candidate.startedBy,
      startedInChat: candidate.startedInChat,
      startedAt: candidate.startedAt,
    };
    if (session.pid && isProcessAlive(session.pid)) {
      if (legacy) saveState(session);
      activeSession = session;
      logger.info(
        { pid: session.pid, url: session.url },
        'Restored Remote Control session from previous run',
      );
    } else {
      clearState();
    }
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    throw new Error(
      'Remote Control occupied-session state is malformed; preserve it and inspect the owned process before repairing the record or starting another session.',
      { cause: err },
    );
  }
}

export function getActiveSession(): RemoteControlSession | null {
  return activeSession;
}

/** @internal — exported for testing only */
export function _resetForTesting(): void {
  activeSession = null;
}

/** @internal — exported for testing only */
export function _getStateFilePath(): string {
  return STATE_FILE;
}

export async function startRemoteControl(
  sender: string,
  chatJid: string,
  cwd: string,
): Promise<{ ok: true; url: string } | { ok: false; error: string }> {
  if (!activeSession) restoreRemoteControl();
  if (activeSession) {
    // Verify the process is still alive
    if (isProcessAlive(activeSession.pid)) {
      return { ok: true, url: activeSession.url };
    }
    // Process died — clean up and start a new one
    activeSession = null;
    clearState();
  }

  // Redirect stdout/stderr to files so the process has no pipes to the parent.
  // This prevents SIGPIPE when NanoClaw restarts.
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const stdoutFd = fs.openSync(STDOUT_FILE, 'w');
  const stderrFd = fs.openSync(STDERR_FILE, 'w');

  let proc;
  try {
    proc = spawn('claude', ['remote-control', '--name', 'NanoClaw Remote'], {
      cwd,
      stdio: ['pipe', stdoutFd, stderrFd],
      detached: true,
    });
  } catch (err) {
    fs.closeSync(stdoutFd);
    fs.closeSync(stderrFd);
    if (!hasErrno(err, 'ENOENT') && !hasErrno(err, 'EACCES')) throw err;
    return { ok: false, error: `Failed to start: ${err.message}` };
  }

  // Close parent descriptors before observing the child's asynchronous launch.
  fs.closeSync(stdoutFd);
  fs.closeSync(stderrFd);

  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let settled = false;
    const finish = (
      result: { ok: true; url: string } | { ok: false; error: string },
    ) => {
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };
    const pid = proc.pid;
    let stopAttempted = false;
    const stopStartedProcess = () => {
      stopAttempted = true;
      if (!pid) return;
      try {
        process.kill(-pid, 'SIGTERM');
      } catch (error) {
        if (!hasErrno(error, 'ESRCH')) throw error;
        try {
          process.kill(pid, 'SIGTERM');
        } catch (fallbackError) {
          if (!hasErrno(fallbackError, 'ESRCH')) throw fallbackError;
        }
      }
    };
    const rejectAfterCleanup = (error: unknown) => {
      if (settled) return;
      clearTimeout(timer);
      settled = true;
      if (activeSession?.pid === pid) activeSession = null;
      if (stopAttempted) {
        reject(error);
        return;
      }
      Promise.resolve()
        .then(stopStartedProcess)
        .then(
          () => reject(error),
          (cleanupError: unknown) =>
            reject(
              new AggregateError(
                [error, cleanupError],
                'Remote Control polling and owned-process cleanup failed; inspect the started process before retrying.',
                { cause: error },
              ),
            ),
        );
    };
    proc.on('error', (error: Error) => {
      if (settled) return;
      if (!pid && (hasErrno(error, 'ENOENT') || hasErrno(error, 'EACCES'))) {
        finish({ ok: false, error: `Failed to start: ${error.message}` });
      } else rejectAfterCleanup(error);
    });
    proc.stdin?.on('error', rejectAfterCleanup);
    const poll = async () => {
      if (settled) return;
      if (!pid) return; // A failed launch emits error asynchronously.
      // Check if process died
      if (!isProcessAlive(pid)) {
        finish({ ok: false, error: 'Process exited before producing URL' });
        return;
      }

      // Check for URL in stdout file
      let content = '';
      try {
        content = fs.readFileSync(STDOUT_FILE, 'utf-8');
      } catch (err) {
        if (!hasErrno(err, 'ENOENT')) throw err;
        // File might not have content yet
      }

      const match = content.match(URL_REGEX);
      if (match) {
        const session: RemoteControlSession = {
          pid,
          url: match[0],
          startedBy: sender,
          startedInChat: chatJid,
          startedAt: new Date().toISOString(),
        };
        activeSession = session;
        saveState(session);

        logger.info(
          { url: match[0], pid, sender, chatJid },
          'Remote Control session started',
        );
        finish({ ok: true, url: match[0] });
        return;
      }

      // Timeout check
      if (Date.now() - startTime >= URL_TIMEOUT_MS) {
        stopStartedProcess();
        finish({
          ok: false,
          error: 'Timed out waiting for Remote Control URL',
        });
        return;
      }

      timer = setTimeout(() => {
        void poll().then(undefined, rejectAfterCleanup);
      }, URL_POLL_MS);
    };

    void Promise.resolve()
      .then(() => {
        // Error listeners exist before stdin/unref can emit a launch failure.
        if (!pid) return;
        proc.stdin?.write('y\n');
        proc.stdin?.end();
        proc.unref();
        return poll();
      })
      .then(undefined, rejectAfterCleanup);
  });
}

export function stopRemoteControl():
  | {
      ok: true;
    }
  | { ok: false; error: string } {
  if (!activeSession) {
    return { ok: false, error: 'No active Remote Control session' };
  }

  const { pid } = activeSession;
  try {
    process.kill(pid, 'SIGTERM');
  } catch (err) {
    if (!hasErrno(err, 'ESRCH')) throw err;
    // already dead
  }
  activeSession = null;
  clearState();
  logger.info({ pid }, 'Remote Control session stopped');
  return { ok: true };
}
