/** Fixed host-owned IPC evidence must never enter an agent view or RW alias. */
import fs from 'fs';
import path from 'path';
import { DATA_DIR } from './config.js';
import { createIpcInbox } from './ipc-inbox.js';
import { hasErrno } from './operational-errors.js';
interface Mount {
  hostPath: string;
  containerPath: string;
  readonly: boolean;
}
function contains(parent: string, child: string): boolean {
  const relative = path.relative(parent, child);
  return (
    relative === '' ||
    (!relative.startsWith('..' + path.sep) &&
      relative !== '..' &&
      !path.isAbsolute(relative))
  );
}
export function protectIpcClaimMounts(mounts: readonly Mount[]): Mount[] {
  const base = path.join(DATA_DIR, 'ipc');
  fs.mkdirSync(base, { recursive: true });
  createIpcInbox(base);
  const reserved = path.join(base, '.claimed-v1');
  const canonicalBase = fs.realpathSync(base);
  const canonicalReserved = fs.realpathSync(reserved);
  if (canonicalReserved !== path.join(canonicalBase, '.claimed-v1'))
    throw new Error(
      'IPC claim root is aliased; hold agent launch and inspect owner storage.',
    );
  const empty = path.join(canonicalBase, '.claims-empty');
  try {
    fs.mkdirSync(empty, { mode: 0o700 });
  } catch (error) {
    if (!hasErrno(error, 'EEXIST')) throw error;
  }
  const identity = fs.lstatSync(empty);
  function verifyEmpty() {
    const stat = fs.lstatSync(empty);
    if (
      !stat.isDirectory() ||
      stat.isSymbolicLink() ||
      stat.uid !== process.geteuid!() ||
      (stat.mode & 0o077) !== 0 ||
      stat.dev !== identity.dev ||
      stat.ino !== identity.ino ||
      fs.realpathSync(empty) !== empty
    )
      throw new Error(
        'IPC mask directory has unsafe identity; hold agent launch without changing its bytes.',
      );
    const dir = fs.opendirSync(empty);
    try {
      if (dir.readSync() !== null)
        throw new Error(
          'IPC mask directory is not empty; preserve it and hold agent launch.',
        );
    } finally {
      dir.closeSync();
    }
  }
  verifyEmpty();
  const canonical = mounts.map((mount) => ({
    ...mount,
    hostPath: fs.realpathSync(mount.hostPath),
  }));
  const masks: Mount[] = [];
  for (const mount of canonical) {
    const source = mount.hostPath;
    if (contains(canonicalReserved, source) || contains(empty, source))
      throw new Error(
        'Direct or descendant IPC evidence/mask mounts are forbidden; remove the extra mount.',
      );
    if (!contains(source, canonicalReserved)) continue;
    if (!mount.readonly)
      throw new Error(
        'Read-write parent overlaps reserved IPC evidence; remove the extra mount before launching an agent.',
      );
    const relative = path
      .relative(source, canonicalReserved)
      .split(path.sep)
      .join('/');
    const target = path.posix.join(mount.containerPath, relative);
    if (canonical.some((other) => contains(target, other.containerPath)))
      throw new Error(
        'A nested mount pierces an IPC evidence mask; remove the partial mount layout.',
      );
    if (
      masks.some(
        (other) =>
          contains(other.containerPath, target) ||
          contains(target, other.containerPath),
      )
    )
      throw new Error(
        'IPC evidence masks overlap; remove the ambiguous mount layout.',
      );
    masks.push({ hostPath: empty, containerPath: target, readonly: true });
  }
  verifyEmpty();
  return [...canonical, ...masks];
}
