import {
  captureRemoteControlIdentity,
  verifyRemoteControlIdentity,
  parseRemoteControlIdentity,
  RemoteControlIdentityError,
  type RemoteControlIdentity,
} from './remote-control-identity.js';
import { hasErrno, isFileSystemError } from './operational-errors.js';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';

import { DATA_DIR } from './config.js';
import { logger } from './logger.js';

/** Sole owner of remote-control.json. Schema2 adds a native instance identity:
 * Linux boot/start/executable/command proof, or a Darwin native command nonce.
 * This occupied-session artifact uses the gate-store exception: unsupported
 * state refuses another launch instead of reading as an empty session. The
 * runtime is its sole reader/writer. No older process record is accepted for
 * migration without its original native instance evidence.
 * Legacy unversioned/schema1 records have no original birth proof and HOLD
 * unchanged; adopting today's PID would invent a migration. Unsupported/newer,
 * malformed or unverifiable records never authorize adoption or signaling.
 * Real-Claude nonce preservation on Darwin is a separate native acceptance gate.
 */
class UnsupportedRemoteControlStateError extends Error {}

interface RemoteControlSession {
  pid: number;
  url: string;
  startedBy: string;
  startedInChat: string;
  startedAt: string;
  identity: RemoteControlIdentity;
}

let activeSession: RemoteControlSession | null = null;

const URL_REGEX = /https:\/\/claude\.ai\/code\S+/;
const URL_TIMEOUT_MS = 30_000;
const URL_POLL_MS = 200;
const STATE_FILE = path.join(DATA_DIR, 'remote-control.json');
const START_FILE = path.join(DATA_DIR, 'remote-control-starting.json');
type StartResult = { ok: true; url: string } | { ok: false; error: string };
let starting: Promise<StartResult> | undefined;
type StopResult = { ok: true } | { ok: false; error: string };
let stopping: Promise<StopResult> | undefined;

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
    const encoded = JSON.stringify({ schema_version: 2, ...session });
    if (Buffer.byteLength(encoded) > 16384)
      throw new UnsupportedRemoteControlStateError(
        'Remote Control state exceeds its bounded owner budget.',
      );
    fs.writeFileSync(candidate, encoded, { mode: 0o600, flag: 'wx' });
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

async function retireOwnedProcess(
  pid: number | undefined,
  assertOwned: () => void,
): Promise<void> {
  if (!pid)
    throw new Error(
      'Remote Control child identity is unavailable; preserve the startup gate.',
    );
  const absent = () => !isProcessAlive(-pid) && !isProcessAlive(pid);
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
    if (absent()) return;
    assertOwned();
    signal(value);
    for (let check = 0; check < 10; check++) {
      if (absent()) return;
      await new Promise<void>((done) => setTimeout(done, 100));
    }
  }
  if (absent()) return;
  throw new Error(
    'Remote Control process retirement is unverified; preserve the startup gate before retrying.',
  );
}

class RemoteControlOutputError extends Error {}
function boundedBytes(
  file: string,
  maximum: number,
): { bytes: Buffer; privateFile: boolean } {
  const fd = fs.openSync(
    file,
    fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK,
  );
  using _lease = { [Symbol.dispose]: () => fs.closeSync(fd) };
  const stat = fs.fstatSync(fd);
  if (
    !stat.isFile() ||
    stat.nlink !== 1 ||
    stat.uid !== process.geteuid!() ||
    stat.size > maximum
  )
    throw new RemoteControlOutputError(
      'Remote Control data is not bounded owned regular data; preserve it for inspection.',
    );
  const bytes = Buffer.alloc(maximum + 1);
  let count = 0;
  while (count < bytes.length) {
    const n = fs.readSync(fd, bytes, count, bytes.length - count, null);
    if (!n) break;
    count += n;
  }
  if (count > maximum)
    throw new RemoteControlOutputError(
      'Remote Control data exceeded its byte bound.',
    );
  return {
    bytes: bytes.subarray(0, count),
    privateFile: (stat.mode & 0o077) === 0,
  };
}
function decoded(bytes: Buffer, stream = false): string {
  try {
    return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(
      bytes,
      { stream },
    );
  } catch (error) {
    if (
      !(error instanceof TypeError) ||
      !hasErrno(error, 'ERR_ENCODING_INVALID_ENCODED_DATA')
    )
      throw error;
    throw new RemoteControlOutputError(
      'Remote Control output is not valid UTF-8.',
      { cause: error },
    );
  }
}
function parsedUrl(value: string): URL | undefined {
  try {
    return new URL(value);
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
    return undefined;
  }
}
function validUrl(value: string): string {
  const result = parsedUrl(value);
  if (!result)
    throw new RemoteControlOutputError(
      'Remote Control output contains an invalid URL.',
    );
  const parsed = result;
  if (
    value.length > 4096 ||
    Array.from(value).some(
      (character) =>
        character.charCodeAt(0) <= 32 || character.charCodeAt(0) === 127,
    ) ||
    parsed.protocol !== 'https:' ||
    parsed.hostname !== 'claude.ai' ||
    parsed.username ||
    parsed.password ||
    parsed.port ||
    !/^\/code(?:\/|$)/.test(parsed.pathname)
  )
    throw new RemoteControlOutputError(
      'Remote Control output contains an invalid URL.',
    );
  return value;
}
function stateJson(raw: string): { ok: true; value: unknown } | { ok: false } {
  try {
    return { ok: true, value: JSON.parse(raw) as unknown };
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
    return { ok: false };
  }
}
function sessionIsLive(session: RemoteControlSession): boolean {
  const status = verifyRemoteControlIdentity(session.pid, session.identity);
  if (status === 'same') return true;
  if (status === 'absent' && !isProcessAlive(-session.pid)) return false;
  throw new UnsupportedRemoteControlStateError(
    'Remote Control native identity is unproved or changed; retain its state without adopting or signaling the PID.',
  );
}
export function restoreRemoteControl(): void {
  let raw: string;
  let privateFile: boolean;
  try {
    const snapshot = boundedBytes(STATE_FILE, 16384);
    raw = decoded(snapshot.bytes);
    privateFile = snapshot.privateFile;
  } catch (error) {
    if (hasErrno(error, 'ENOENT')) return;
    throw error;
  }
  const parsed = stateJson(raw);
  if (
    !parsed.ok ||
    !parsed.value ||
    typeof parsed.value !== 'object' ||
    Array.isArray(parsed.value)
  )
    throw new UnsupportedRemoteControlStateError(
      'Remote Control occupied-session state is malformed; preserve it for owner inspection.',
    );
  const row = parsed.value as Record<string, unknown>;
  if (row.schema_version === undefined || row.schema_version === 1)
    throw new UnsupportedRemoteControlStateError(
      'Legacy Remote Control identity is unproved; preserve the original state for host-owner reconciliation without adopting or signaling its PID.',
    );
  if (row.schema_version !== 2)
    throw new UnsupportedRemoteControlStateError(
      'Unsupported Remote Control state version; update the owner before restoring or starting another session.',
    );
  if (!privateFile)
    throw new UnsupportedRemoteControlStateError(
      'Remote Control instance state is not private; preserve it for owner inspection.',
    );
  if (
    Object.keys(row).sort().join() !==
      [
        'schema_version',
        'pid',
        'url',
        'startedBy',
        'startedInChat',
        'startedAt',
        'identity',
      ]
        .sort()
        .join() ||
    typeof row.pid !== 'number' ||
    !Number.isSafeInteger(row.pid) ||
    row.pid <= 0 ||
    row.pid > 2147483647 ||
    typeof row.url !== 'string' ||
    typeof row.startedBy !== 'string' ||
    typeof row.startedInChat !== 'string' ||
    typeof row.startedAt !== 'string' ||
    !Number.isFinite(new Date(row.startedAt).getTime())
  )
    throw new UnsupportedRemoteControlStateError(
      'Remote Control occupied-session state is malformed; preserve it for owner inspection.',
    );
  const session: RemoteControlSession = {
    pid: row.pid,
    url: validUrl(row.url),
    startedBy: row.startedBy,
    startedInChat: row.startedInChat,
    startedAt: row.startedAt,
    identity: parseRemoteControlIdentity(row.identity),
  };
  if (sessionIsLive(session)) {
    activeSession = session;
    logger.info(
      { pid: session.pid },
      'Restored verified Remote Control instance',
    );
  } else clearState();
}
/** Host startup owns optional Remote Control refusal; the unchanged record still
 * gates every later start. Expected holds must not disable unrelated channels,
 * scheduler or IPC. Unexpected implementation defects still reject startup.
 */
export function restoreRemoteControlAtStartup(): void {
  try {
    restoreRemoteControl();
  } catch (error) {
    if (
      !(error instanceof UnsupportedRemoteControlStateError) &&
      !(error instanceof RemoteControlOutputError) &&
      !(error instanceof RemoteControlIdentityError) &&
      !isFileSystemError(error)
    )
      throw error;
    activeSession = null;
    logger.error(
      'Remote Control remains held; preserve its owner state and reconcile the native instance before retrying. Other host services will continue.',
    );
  }
}

export function getActiveSession(): RemoteControlSession | null {
  return activeSession && sessionIsLive(activeSession) ? activeSession : null;
}

/** @internal — exported for testing only */
export function _resetForTesting(): void {
  activeSession = null;
  starting = undefined;
  stopping = undefined;
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
  if (stopping)
    return Promise.resolve({
      ok: false,
      error: 'Remote Control is stopping; wait for verified native retirement.',
    });
  if (starting) return starting;
  const work = (async () => {
    // Refuse malformed occupied state before creating another reservation.
    if (!activeSession) restoreRemoteControl();
    if (activeSession && sessionIsLive(activeSession))
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
    if (sessionIsLive(activeSession)) {
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

      const instanceNonce = randomUUID();
      let proc;
      disposition('unknown');
      try {
        proc = spawn(
          'claude',
          ['remote-control', '--name', `NanoClaw Remote ${instanceNonce}`],
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
        const stopStartedProcess = () =>
          (stopPromise ??= Promise.resolve()
            .then(() =>
              retireOwnedProcess(pid, () => {
                if (
                  proc.pid !== pid ||
                  proc.exitCode !== null ||
                  proc.signalCode !== null
                )
                  throw new RemoteControlIdentityError(
                    'Original child lifecycle is no longer active; preserve the startup gate.',
                  );
              }),
            )
            .then(() => disposition('absent')));
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
          let content: string;
          try {
            const output = boundedBytes(stdoutFile, 65536);
            if (!output.privateFile)
              throw new RemoteControlOutputError(
                'Remote Control output is not private.',
              );
            const snapshot = decoded(output.bytes, true);
            content = snapshot.slice(0, snapshot.lastIndexOf('\n') + 1);
          } catch (err) {
            if (!(err instanceof RemoteControlOutputError)) throw err;
            await stopStartedProcess();
            finish({ ok: false, error: err.message });
            return;
          }

          const match = content.match(URL_REGEX);
          if (match) {
            let session: RemoteControlSession;
            try {
              session = {
                pid,
                url: validUrl(match[0]),
                startedBy: sender,
                startedInChat: chatJid,
                startedAt: new Date().toISOString(),
                identity: captureRemoteControlIdentity(pid, instanceNonce),
              };
            } catch (error) {
              if (
                !(error instanceof RemoteControlOutputError) &&
                !(error instanceof RemoteControlIdentityError)
              )
                throw error;
              await stopStartedProcess();
              finish({ ok: false, error: error.message });
              return;
            }
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

export function stopRemoteControl(): Promise<StopResult> {
  if (stopping) return stopping;
  if (!activeSession)
    return Promise.resolve({
      ok: false,
      error: 'No active Remote Control session',
    });
  const session = activeSession;
  const work = Promise.resolve()
    .then(() =>
      retireOwnedProcess(session.pid, () => {
        if (
          verifyRemoteControlIdentity(session.pid, session.identity) !== 'same'
        )
          throw new RemoteControlIdentityError(
            'Recalled process identity is unproved; retain state without signaling.',
          );
      }),
    )
    .then(() => {
      clearState();
      activeSession = null;
      logger.info({ pid: session.pid }, 'Remote Control session stopped');
      return { ok: true } as const;
    });
  stopping = work;
  void work.then(
    () => {
      stopping = undefined;
    },
    () => {
      stopping = undefined;
    },
  );
  return work;
}
