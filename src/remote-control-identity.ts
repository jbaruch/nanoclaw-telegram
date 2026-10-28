/** Native identity owner, not a caller-provided process authorization route. */
import fs from 'fs';
import { execFileSync } from 'child_process';
import { createHash } from 'crypto';
import {
  hasErrno,
  isCommandError,
  isFileSystemError,
} from './operational-errors.js';

export class RemoteControlIdentityError extends Error {}
export type RemoteControlIdentity =
  | {
      kind: 'linux-birth';
      boot_id: string;
      start_ticks: string;
      executable: string;
      command_sha256: string;
    }
  | {
      kind: 'darwin-command-nonce';
      nonce: string;
      executable_name: string;
      command_sha256: string;
    };
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const HASH = /^[a-f0-9]{64}$/;
const hasControl = (value: string): boolean =>
  Array.from(value).some(
    (character) =>
      character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127,
  );
const hash = (bytes: Buffer) =>
  createHash('sha256').update(bytes).digest('hex');
const text = (bytes: Buffer) =>
  new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
function bounded(file: string, maximum: number): Buffer {
  const fd = fs.openSync(
    file,
    fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK,
  );
  using _lease = { [Symbol.dispose]: () => fs.closeSync(fd) };
  const stat = fs.fstatSync(fd);
  if (!stat.isFile() || stat.size > maximum)
    throw new RemoteControlIdentityError(
      'Native process metadata is not bounded regular data.',
    );
  const bytes = Buffer.alloc(maximum + 1);
  let count = 0;
  while (count < bytes.length) {
    const n = fs.readSync(fd, bytes, count, bytes.length - count, null);
    if (!n) break;
    count += n;
  }
  if (count > maximum)
    throw new RemoteControlIdentityError(
      'Native process metadata exceeded its bound.',
    );
  return bytes.subarray(0, count);
}
function ticks(pid: number): string {
  const value = text(bounded(`/proc/${pid}/stat`, 8192));
  const end = value.lastIndexOf(') ');
  const fields = value
    .slice(end + 2)
    .trim()
    .split(/\s+/);
  if (
    !value.startsWith(`${pid} (`) ||
    end < 0 ||
    !/^(0|[1-9][0-9]{0,23})$/.test(fields[19] ?? '') ||
    fields[0] === 'Z'
  )
    throw new RemoteControlIdentityError(
      'Native process birth metadata is unavailable.',
    );
  return fields[19];
}
function linuxIdentity(pid: number): RemoteControlIdentity {
  const boot = text(bounded('/proc/sys/kernel/random/boot_id', 64)).trim();
  const start = ticks(pid);
  const executable = text(
    fs.readlinkSync(`/proc/${pid}/exe`, { encoding: 'buffer' }),
  );
  const command = bounded(`/proc/${pid}/cmdline`, 65536);
  if (
    !UUID.test(boot) ||
    !executable.startsWith('/') ||
    hasControl(executable) ||
    executable.length > 4096 ||
    !command.length ||
    command.at(-1) !== 0
  )
    throw new RemoteControlIdentityError(
      'Native process executable metadata is invalid.',
    );
  const commandHash = hash(command);
  if (
    ticks(pid) !== start ||
    text(fs.readlinkSync(`/proc/${pid}/exe`, { encoding: 'buffer' })) !==
      executable ||
    hash(bounded(`/proc/${pid}/cmdline`, 65536)) !== commandHash ||
    text(bounded('/proc/sys/kernel/random/boot_id', 64)).trim() !== boot
  )
    throw new RemoteControlIdentityError(
      'Native process identity changed during observation.',
    );
  return {
    kind: 'linux-birth',
    boot_id: boot,
    start_ticks: start,
    executable,
    command_sha256: commandHash,
  };
}
function ps(pid: number, column: 'comm=' | 'command='): Buffer {
  return execFileSync('/bin/ps', ['-p', String(pid), '-ww', '-o', column], {
    timeout: 1000,
    killSignal: 'SIGKILL',
    maxBuffer: 65536,
    env: { LC_ALL: 'C', LANG: 'C' },
  });
}
function darwinIdentity(pid: number, nonce: string): RemoteControlIdentity {
  if (!UUID.test(nonce))
    throw new RemoteControlIdentityError('Native instance nonce is invalid.');
  const executable = text(ps(pid, 'comm=')).trim();
  const command = ps(pid, 'command=');
  const decoded = text(command).trimEnd();
  const marker = `NanoClaw Remote ${nonce}`;
  const index = decoded.indexOf(marker);
  if (
    !executable ||
    executable.length > 1024 ||
    hasControl(executable) ||
    index < 0 ||
    (index > 0 && !/[ \t]/.test(decoded[index - 1])) ||
    (index + marker.length < decoded.length &&
      !/[ \t]/.test(decoded[index + marker.length]))
  )
    throw new RemoteControlIdentityError(
      'The native CLI does not expose the owned instance marker; preserve state without adopting or signaling a recalled PID.',
    );
  if (
    text(ps(pid, 'comm=')).trim() !== executable ||
    !ps(pid, 'command=').equals(command)
  )
    throw new RemoteControlIdentityError(
      'Native command identity changed during observation.',
    );
  return {
    kind: 'darwin-command-nonce',
    nonce,
    executable_name: executable,
    command_sha256: hash(command),
  };
}
type Observation =
  | { ok: true; identity: RemoteControlIdentity }
  | { ok: false; code: string; status: number | null };
function observe(pid: number, nonce: string): Observation {
  try {
    if (!Number.isSafeInteger(pid) || pid <= 0 || pid > 2147483647)
      throw new RemoteControlIdentityError('Native process ID is invalid.');
    const identity =
      process.platform === 'linux'
        ? linuxIdentity(pid)
        : process.platform === 'darwin'
          ? darwinIdentity(pid, nonce)
          : undefined;
    if (!identity)
      throw new RemoteControlIdentityError(
        'Native process identity is unsupported on this platform.',
      );
    return { ok: true, identity };
  } catch (error) {
    if (error instanceof RemoteControlIdentityError) throw error;
    if (
      !isFileSystemError(error) &&
      !isCommandError(error) &&
      !(
        error instanceof TypeError &&
        hasErrno(error, 'ERR_ENCODING_INVALID_ENCODED_DATA')
      )
    )
      throw error;
    // Expected transport failures retain only fixed native metadata. ps errors
    // can contain another process's private argv in stdout/stderr properties.
    return {
      ok: false,
      code:
        error instanceof Error &&
        'code' in error &&
        typeof error.code === 'string' &&
        /^[A-Z_]{1,32}$/.test(error.code)
          ? error.code
          : 'native_metadata',
      status:
        error instanceof Error &&
        'status' in error &&
        typeof error.status === 'number'
          ? error.status
          : null,
    };
  }
}
export function captureRemoteControlIdentity(
  pid: number,
  nonce: string,
): RemoteControlIdentity {
  const result = observe(pid, nonce);
  if (!result.ok)
    throw new RemoteControlIdentityError(
      'Native process identity could not be established; preserve the occupied state.',
      { cause: result },
    );
  return result.identity;
}
export function verifyRemoteControlIdentity(
  pid: number,
  expected: RemoteControlIdentity,
): 'same' | 'absent' | 'changed' {
  const result = observe(
    pid,
    expected.kind === 'darwin-command-nonce' ? expected.nonce : '',
  );
  if (result.ok)
    return JSON.stringify(result.identity) === JSON.stringify(expected)
      ? 'same'
      : 'changed';
  try {
    process.kill(pid, 0);
  } catch (error) {
    if (hasErrno(error, 'ESRCH')) return 'absent';
    throw error;
  }
  throw new RemoteControlIdentityError(
    'Native process identity is unavailable; retain its state without signaling.',
    { cause: result },
  );
}
export function parseRemoteControlIdentity(
  value: unknown,
): RemoteControlIdentity {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new RemoteControlIdentityError(
      'Remote Control identity is malformed.',
    );
  const row = value as Record<string, unknown>;
  if (typeof row.command_sha256 !== 'string' || !HASH.test(row.command_sha256))
    throw new RemoteControlIdentityError(
      'Remote Control command identity is malformed.',
    );
  if (
    row.kind === 'linux-birth' &&
    Object.keys(row).sort().join() ===
      ['kind', 'boot_id', 'start_ticks', 'executable', 'command_sha256']
        .sort()
        .join() &&
    typeof row.boot_id === 'string' &&
    UUID.test(row.boot_id) &&
    typeof row.start_ticks === 'string' &&
    /^(0|[1-9][0-9]{0,23})$/.test(row.start_ticks) &&
    typeof row.executable === 'string' &&
    row.executable.startsWith('/') &&
    row.executable.length <= 4096 &&
    !hasControl(row.executable)
  )
    return {
      kind: row.kind,
      boot_id: row.boot_id,
      start_ticks: row.start_ticks,
      executable: row.executable,
      command_sha256: row.command_sha256,
    };
  if (
    row.kind === 'darwin-command-nonce' &&
    Object.keys(row).sort().join() ===
      ['kind', 'nonce', 'executable_name', 'command_sha256'].sort().join() &&
    typeof row.nonce === 'string' &&
    UUID.test(row.nonce) &&
    typeof row.executable_name === 'string' &&
    row.executable_name.length > 0 &&
    row.executable_name.length <= 1024 &&
    !hasControl(row.executable_name)
  )
    return {
      kind: row.kind,
      nonce: row.nonce,
      executable_name: row.executable_name,
      command_sha256: row.command_sha256,
    };
  throw new RemoteControlIdentityError(
    'Unsupported or malformed Remote Control identity; retain the record.',
  );
}
