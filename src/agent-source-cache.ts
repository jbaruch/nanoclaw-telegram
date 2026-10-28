/** Sole owner of the host-only agent-runner-source.json schema1 admission
 * marker. See docs/AGENT-SOURCE-CACHE.md. Unknown markers HOLD; they never
 * authorize reuse or erase another version's gate. */
import fs from 'fs';
import path from 'path';
import { createHash } from 'crypto';
import { hasErrno, isFileSystemError } from './operational-errors.js';

export class AgentSourceCacheHold extends Error {}
const NOFOLLOW = fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW;
const MAX_SOURCE_BYTES = 4 * 1024 * 1024;

function bytes(file: string, marker = false): Buffer {
  const fd = fs.openSync(file, NOFOLLOW | fs.constants.O_NONBLOCK);
  using _lease = { [Symbol.dispose]: () => fs.closeSync(fd) };
  const stat = fs.fstatSync(fd),
    limit = marker ? 512 : MAX_SOURCE_BYTES;
  if (
    !stat.isFile() ||
    stat.nlink !== 1 ||
    stat.size > limit ||
    (marker && (stat.uid !== process.geteuid!() || (stat.mode & 0o077) !== 0))
  )
    throw new AgentSourceCacheHold(
      'Agent source metadata is unsafe; preserve it for owner repair.',
    );
  const buffer = Buffer.alloc(limit + 1);
  let count = 0;
  while (count < buffer.length) {
    const size = fs.readSync(fd, buffer, count, buffer.length - count, null);
    if (!size) break;
    count += size;
  }
  if (count > limit)
    throw new AgentSourceCacheHold(
      'Agent source exceeded its bounded owner contract.',
    );
  return buffer.subarray(0, count);
}

function snapshot(source: string): {
  digest: string;
  files: Map<string, Buffer>;
} {
  const stat = fs.lstatSync(source);
  if (!stat.isDirectory() || stat.isSymbolicLink())
    throw new AgentSourceCacheHold(
      'Canonical agent source directory is unsafe.',
    );
  const names: string[] = [];
  const directory = fs.opendirSync(source);
  using _lease = { [Symbol.dispose]: () => directory.closeSync() };
  for (let entry = directory.readSync(); entry; entry = directory.readSync()) {
    if (names.length === 128)
      throw new AgentSourceCacheHold(
        'Canonical agent source count exceeds its owner bound.',
      );
    // The shipped src layout is flat. A new layout must update this owner;
    // directories/aliases are never silently omitted from the fingerprint.
    if (!entry.isFile())
      throw new AgentSourceCacheHold(
        'Unsupported canonical agent source entry; update the source-cache owner.',
      );
    names.push(entry.name);
  }
  names.sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));
  if (!names.includes('index.ts') || !names.includes('ipc-mcp-stdio.ts'))
    throw new AgentSourceCacheHold(
      'Canonical agent entrypoints are incomplete.',
    );
  const files = new Map<string, Buffer>(),
    hash = createHash('sha256');
  let total = 0;
  for (const name of names) {
    const content = bytes(path.join(source, name));
    total += content.length;
    if (total > MAX_SOURCE_BYTES)
      throw new AgentSourceCacheHold(
        'Canonical agent source bytes exceed their owner bound.',
      );
    hash.update(JSON.stringify([name, content.length]) + '\n');
    hash.update(content);
    files.set(name, content);
  }
  return { digest: hash.digest('hex'), files };
}

function ownedDirectory(directory: string): number {
  const fd = fs.openSync(directory, NOFOLLOW | fs.constants.O_DIRECTORY);
  let transferred = false;
  using _lease = {
    [Symbol.dispose]: () => {
      if (!transferred) fs.closeSync(fd);
    },
  };
  const stat = fs.fstatSync(fd),
    bound = fs.lstatSync(directory);
  if (
    !stat.isDirectory() ||
    stat.uid !== process.geteuid!() ||
    bound.isSymbolicLink() ||
    stat.dev !== bound.dev ||
    stat.ino !== bound.ino
  )
    throw new AgentSourceCacheHold(
      'Agent source parent ownership changed; preserve it for owner repair.',
    );
  if (stat.mode & 0o022) {
    fs.fchmodSync(fd, stat.mode & 0o7777 & ~0o022);
    fs.fsyncSync(fd);
  }
  transferred = true;
  return fd;
}

function assertBound(directory: string, fd: number): void {
  const stat = fs.fstatSync(fd),
    bound = fs.lstatSync(directory);
  if (
    !bound.isDirectory() ||
    bound.isSymbolicLink() ||
    stat.dev !== bound.dev ||
    stat.ino !== bound.ino
  )
    throw new AgentSourceCacheHold(
      'Agent source directory binding changed; preserve its state.',
    );
}

function write(file: string, content: Buffer | string, mode: number): void {
  const fd = fs.openSync(
    file,
    fs.constants.O_WRONLY |
      fs.constants.O_CREAT |
      fs.constants.O_EXCL |
      fs.constants.O_NOFOLLOW,
    mode,
  );
  using _lease = { [Symbol.dispose]: () => fs.closeSync(fd) };
  fs.fchmodSync(fd, mode);
  fs.writeFileSync(fd, content);
  fs.fsyncSync(fd);
}

function markerState(
  file: string,
): { digest: string; ready: boolean } | undefined {
  let raw: unknown;
  try {
    raw = JSON.parse(
      new TextDecoder('utf-8', { fatal: true }).decode(bytes(file, true)),
    );
  } catch (error) {
    if (hasErrno(error, 'ENOENT')) return undefined;
    if (
      !(error instanceof SyntaxError) &&
      !isFileSystemError(error) &&
      !hasErrno(error, 'ERR_ENCODING_INVALID_ENCODED_DATA')
    )
      throw error;
    throw new AgentSourceCacheHold(
      'Agent source marker is unreadable; preserve it for owner repair.',
      { cause: error },
    );
  }
  if (
    !raw ||
    typeof raw !== 'object' ||
    Array.isArray(raw) ||
    Object.keys(raw).sort().join() !== 'schema_version,source_sha256,state' ||
    !('schema_version' in raw) ||
    raw.schema_version !== 1 ||
    !('state' in raw) ||
    (raw.state !== 'ready' && raw.state !== 'refreshing') ||
    !('source_sha256' in raw) ||
    typeof raw.source_sha256 !== 'string' ||
    !/^[a-f0-9]{64}$/.test(raw.source_sha256)
  )
    throw new AgentSourceCacheHold(
      'Unsupported agent source marker; update the owner without clearing its gate.',
    );
  return { digest: raw.source_sha256, ready: raw.state === 'ready' };
}

/** Trusted host-only paths from buildVolumeMounts; no agent/CLI selector. */
export function syncAgentRunnerSource(source: string, cache: string): void {
  const canonical = snapshot(source),
    parent = path.dirname(cache);
  const parentFD = ownedDirectory(parent);
  using _parent = { [Symbol.dispose]: () => fs.closeSync(parentFD) };
  const parentPath =
    process.platform === 'linux' ? `/proc/self/fd/${parentFD}` : parent;
  const marker = path.join(parentPath, 'agent-runner-source.json');
  const previous = markerState(marker);
  let present = true;
  try {
    fs.lstatSync(cache);
  } catch (error) {
    if (!hasErrno(error, 'ENOENT')) throw error;
    present = false;
  }
  if (!present) {
    fs.mkdirSync(cache, { mode: 0o755 });
    // Newly created empty code cache; existing private roots are never widened.
    fs.chmodSync(cache, 0o755);
  }
  const cacheFD = ownedDirectory(cache);
  using _cache = { [Symbol.dispose]: () => fs.closeSync(cacheFD) };
  assertBound(parent, parentFD);
  assertBound(cache, cacheFD);
  if (
    present &&
    fs.existsSync(path.join(cache, 'index.ts')) &&
    previous?.ready &&
    previous.digest === canonical.digest
  )
    return;
  const cachePath =
    process.platform === 'linux' ? `/proc/self/fd/${cacheFD}` : cache;
  for (const name of canonical.files.keys()) {
    try {
      if (fs.lstatSync(path.join(cachePath, name)).isDirectory())
        throw new AgentSourceCacheHold(
          'Canonical source conflicts with a customized directory; preserve it for owner repair.',
        );
    } catch (error) {
      if (!hasErrno(error, 'ENOENT')) throw error;
    }
  }
  const staging = fs.mkdtempSync(
    path.join(parentPath, '.agent-runner-refresh-'),
  );
  using _staging = {
    [Symbol.dispose]: () =>
      fs.rmSync(staging, { recursive: true, force: true }),
  };
  for (const [name, content] of canonical.files)
    write(path.join(staging, name), content, 0o644);
  const intent = path.join(staging, 'refreshing.json');
  write(
    intent,
    JSON.stringify({
      schema_version: 1,
      state: 'refreshing',
      source_sha256: canonical.digest,
    }),
    0o600,
  );
  assertBound(parent, parentFD);
  assertBound(cache, cacheFD);
  fs.renameSync(intent, marker);
  fs.fsyncSync(parentFD);
  for (const name of canonical.files.keys()) {
    assertBound(parent, parentFD);
    assertBound(cache, cacheFD);
    // Replace leaf entries atomically, never write through guest-created links.
    fs.renameSync(path.join(staging, name), path.join(cachePath, name));
  }
  fs.fsyncSync(cacheFD);
  for (const [name, content] of canonical.files)
    if (!bytes(path.join(cachePath, name)).equals(content))
      throw new AgentSourceCacheHold(
        'Agent source changed during refresh; preserve the prior marker.',
      );
  if (snapshot(source).digest !== canonical.digest)
    throw new AgentSourceCacheHold(
      'Canonical source changed during refresh; retry from a stable source.',
    );
  const candidate = path.join(staging, 'marker.json');
  write(
    candidate,
    JSON.stringify({
      schema_version: 1,
      state: 'ready',
      source_sha256: canonical.digest,
    }),
    0o600,
  );
  assertBound(parent, parentFD);
  assertBound(cache, cacheFD);
  fs.renameSync(candidate, marker);
  fs.fsyncSync(parentFD);
  assertBound(parent, parentFD);
  assertBound(cache, cacheFD);
}
