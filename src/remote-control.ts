import { hasErrno } from './operational-errors.js';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';

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
const START_FILE = path.join(DATA_DIR, 'remote-control-starting.json');
type StartResult = { ok: true; url: string } | { ok: false; error: string };
let starting: Promise<StartResult> | undefined;

function reserveStart(): () => void {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const directory = fs.lstatSync(DATA_DIR);
  if (
    !directory.isDirectory() ||
    directory.isSymbolicLink() ||
    directory.uid !== process.geteuid!() ||
    (directory.mode & 0o022) !== 0
  )
    throw new Error(
      'Remote Control storage is not owned and private for the runtime writer; preserve it for repair.',
    );
  const syncParent = () => {
    const parent = fs.openSync(
      DATA_DIR,
      fs.constants.O_RDONLY |
        fs.constants.O_NOFOLLOW |
        fs.constants.O_DIRECTORY,
    );
    try {
      fs.fsyncSync(parent);
    } finally {
      fs.closeSync(parent);
    }
  };
  let fd: number;
  try {
    fd = fs.openSync(
      START_FILE,
      fs.constants.O_WRONLY |
        fs.constants.O_CREAT |
        fs.constants.O_EXCL |
        fs.constants.O_NOFOLLOW,
      0o600,
    );
  } catch (error) {
    if (!hasErrno(error, 'EEXIST')) throw error;
    throw new Error(
      'Remote Control starting gate is occupied; preserve it and inspect the owned process before retrying.',
      { cause: error },
    );
  }
  const bytes = JSON.stringify({
    schema_version: 1,
    state: 'starting',
    owner_pid: process.pid,
    attempt_id: randomUUID(),
  });
  let identity: fs.Stats;
  try {
    fs.writeFileSync(fd, bytes);
    fs.fsyncSync(fd);
    identity = fs.fstatSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  syncParent();
  return () => {
    const current = fs.lstatSync(START_FILE);
    if (
      !current.isFile() ||
      current.nlink !== 1 ||
      current.dev !== identity.dev ||
      current.ino !== identity.ino ||
      fs.readFileSync(START_FILE, 'utf8') !== bytes
    )
      throw new Error(
        'Remote Control starting gate changed; preserve it for owner inspection.',
      );
    fs.unlinkSync(START_FILE);
    syncParent();
  };
}

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
    const fd = fs.openSync(
      candidate,
      fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW,
    );
    try {
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
    fs.renameSync(candidate, STATE_FILE);
    syncStateDirectory();
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

function syncStateDirectory(): void {
  const fd = fs.openSync(
    DATA_DIR,
    fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_DIRECTORY,
  );
  try {
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
}

function clearState(): void {
  try {
    fs.unlinkSync(STATE_FILE);
    syncStateDirectory();
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
    if (
      session.pid &&
      (isProcessAlive(session.pid) || isProcessAlive(-session.pid))
    ) {
      if (legacy) saveState(session);
      activeSession = session;
      logger.info(
        { pid: session.pid },
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

export function startRemoteControl(
  sender: string,
  chatJid: string,
  cwd: string,
): Promise<StartResult> {
  if (starting) return starting;
  const work = (async () => {
    // Refuse malformed occupied state before creating another reservation.
    if (!activeSession) restoreRemoteControl();
    if (
      activeSession &&
      (isProcessAlive(activeSession.pid) || isProcessAlive(-activeSession.pid))
    )
      return { ok: true as const, url: activeSession.url };
    const release = reserveStart();
    let disposition: 'no_child' | 'tracked' | 'absent' | 'unknown' = 'no_child';
    const [result] = await Promise.allSettled([
      startRemoteControlOwned(sender, chatJid, cwd, (state) => {
        disposition = state;
      }),
    ]);
    const cleanup = await Promise.allSettled([
      Promise.resolve().then(() => {
        // A mere kill signal is not native absence. Keep the gate on uncertainty.
        if (disposition !== 'unknown') release();
      }),
    ]);
    if (cleanup[0].status === 'rejected') {
      if (result.status === 'rejected')
        throw new AggregateError(
          [result.reason, cleanup[0].reason],
          'Remote Control start and reservation cleanup failed.',
          { cause: result.reason },
        );
      throw cleanup[0].reason;
    }
    if (result.status === 'rejected') throw result.reason;
    return result.value;
  })();
  starting = work;
  void work.then(
    () => {
      starting = undefined;
    },
    () => {
      starting = undefined;
    },
  );
  return work;
}

async function startRemoteControlOwned(
  sender: string,
  chatJid: string,
  cwd: string,
  disposition: (state: 'no_child' | 'tracked' | 'absent' | 'unknown') => void,
): Promise<{ ok: true; url: string } | { ok: false; error: string }> {
  if (!activeSession) restoreRemoteControl();
  if (activeSession) {
    // Verify the process is still alive
    if (
      isProcessAlive(activeSession.pid) ||
      isProcessAlive(-activeSession.pid)
    ) {
      return { ok: true, url: activeSession.url };
    }
    // Process died — clean up and start a new one
    activeSession = null;
    clearState();
  }

  // Redirect stdout/stderr to files so the process has no pipes to the parent.
  // This prevents SIGPIPE when NanoClaw restarts.
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const outputDirectory = fs.mkdtempSync(
    path.join(DATA_DIR, '.remote-control-output-'),
  );
  const stdoutFile = path.join(outputDirectory, 'remote-control.stdout');
  const stderrFile = path.join(outputDirectory, 'remote-control.stderr');
  const [operation] = await Promise.allSettled([
    (async (): Promise<StartResult> => {
      const stdoutFd = fs.openSync(
        stdoutFile,
        fs.constants.O_WRONLY |
          fs.constants.O_CREAT |
          fs.constants.O_EXCL |
          fs.constants.O_NOFOLLOW,
        0o600,
      );
      let stderrFd: number;
      try {
        stderrFd = fs.openSync(
          stderrFile,
          fs.constants.O_WRONLY |
            fs.constants.O_CREAT |
            fs.constants.O_EXCL |
            fs.constants.O_NOFOLLOW,
          0o600,
        );
      } catch (error) {
        fs.closeSync(stdoutFd);
        throw error;
      }

      let proc;
      disposition('unknown');
      try {
        proc = spawn(
          'claude',
          ['remote-control', '--name', 'NanoClaw Remote'],
          {
            cwd,
            stdio: ['pipe', stdoutFd, stderrFd],
            detached: true,
          },
        );
      } catch (err) {
        fs.closeSync(stdoutFd);
        fs.closeSync(stderrFd);
        if (!hasErrno(err, 'ENOENT') && !hasErrno(err, 'EACCES')) throw err;
        disposition('no_child');
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
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          resolve(result);
        };
        const pid = proc.pid;
        let stopPromise: Promise<void> | undefined;
        const retireStartedProcess = async () => {
          if (!pid)
            throw new Error(
              'Remote Control child identity is unavailable; preserve the startup gate.',
            );
          const alive = (identity: number): boolean => {
            try {
              process.kill(identity, 0);
              return true;
            } catch (error) {
              if (hasErrno(error, 'ESRCH')) return false;
              throw error;
            }
          };
          const absent = () => !alive(-pid) && !alive(pid);
          const signal = (value: 'SIGTERM' | 'SIGKILL') => {
            try {
              process.kill(-pid, value);
            } catch (error) {
              if (!hasErrno(error, 'ESRCH')) throw error;
              try {
                process.kill(pid, value);
              } catch (fallbackError) {
                if (!hasErrno(fallbackError, 'ESRCH')) throw fallbackError;
              }
            }
          };
          for (const value of ['SIGTERM', 'SIGKILL'] as const) {
            signal(value);
            for (let check = 0; check < 10; check++) {
              if (absent()) {
                disposition('absent');
                return;
              }
              await new Promise<void>((done) => setTimeout(done, 100));
            }
          }
          if (absent()) {
            disposition('absent');
            return;
          }
          throw new Error(
            'Remote Control process retirement is unverified; preserve the startup gate before retrying.',
          );
        };
        const stopStartedProcess = () =>
          (stopPromise ??= Promise.resolve().then(retireStartedProcess));
        const rejectAfterCleanup = (error: unknown) => {
          if (settled) return;
          clearTimeout(timer);
          settled = true;
          if (activeSession?.pid === pid) activeSession = null;
          Promise.resolve()
            .then(stopStartedProcess)
            .then(
              () => reject(error),
              (cleanupError: unknown) =>
                reject(
                  cleanupError === error
                    ? error
                    : new AggregateError(
                        [error, cleanupError],
                        'Remote Control polling and owned-process cleanup failed; inspect the started process before retrying.',
                        { cause: error },
                      ),
                ),
            );
        };
        proc.on('error', (error: Error) => {
          if (settled) return;
          if (
            !pid &&
            (hasErrno(error, 'ENOENT') || hasErrno(error, 'EACCES'))
          ) {
            disposition('no_child');
            finish({ ok: false, error: `Failed to start: ${error.message}` });
          } else rejectAfterCleanup(error);
        });
        proc.stdin?.on('error', rejectAfterCleanup);
        const poll = async () => {
          if (settled) return;
          if (!pid) return; // A failed launch emits error asynchronously.
          // Check if process died
          if (!isProcessAlive(pid)) {
            await stopStartedProcess();
            finish({ ok: false, error: 'Process exited before producing URL' });
            return;
          }

          // Check for URL in stdout file
          let content = '';
          try {
            content = fs.readFileSync(stdoutFile, 'utf-8');
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
            disposition('tracked');

            logger.info(
              { pid, sender, chatJid },
              'Remote Control session started',
            );
            finish({ ok: true, url: match[0] });
            return;
          }

          // Timeout check
          if (Date.now() - startTime >= URL_TIMEOUT_MS) {
            await stopStartedProcess();
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
    })(),
  ]);
  const [cleanup] = await Promise.allSettled([
    Promise.resolve().then(() =>
      fs.rmSync(outputDirectory, { recursive: true }),
    ),
  ]);
  if (operation.status === 'rejected' && cleanup.status === 'rejected')
    throw new AggregateError(
      [operation.reason, cleanup.reason],
      'Remote Control launch and private output cleanup failed.',
      { cause: operation.reason },
    );
  if (operation.status === 'rejected') throw operation.reason;
  if (cleanup.status === 'rejected') throw cleanup.reason;
  return operation.value;
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
    process.kill(-pid, 'SIGTERM');
  } catch (err) {
    if (!hasErrno(err, 'ESRCH')) throw err;
    try {
      process.kill(pid, 'SIGTERM');
    } catch (fallbackError) {
      if (!hasErrno(fallbackError, 'ESRCH')) throw fallbackError;
    }
  }
  if (isProcessAlive(pid) || isProcessAlive(-pid))
    return {
      ok: false,
      error:
        'Remote Control is stopping; retain its session until native absence is verified.',
    };
  activeSession = null;
  clearState();
  logger.info({ pid }, 'Remote Control session stopped');
  return { ok: true };
}
