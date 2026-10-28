/** Private IPC handoff owner. Only the final agent-controlled channel entry is
 * renamed. Batch/slot metadata is schema1; unknown and in-flight work is held,
 * never replayed or evicted. The host IPC base/group parents are not agent mounts.
 */
import fs from 'fs';
import path from 'path';
import { hasErrno, isFileSystemError } from './operational-errors.js';
import { isValidGroupFolder } from './group-folder.js';
import { logger } from './logger.js';

export class IpcInboxHold extends Error {}
type Channel = 'messages' | 'tasks';
const PRIVATE_ROOT = '.claimed-v1';
const MAX_BATCHES = 1024;
const MAX_SOURCE_BATCHES = 64;
const MAX_BYTES = 1024 * 1024;
const nofollow = fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW;
function entries(directory: string, limit = 1024): string[] {
  const dir = fs.opendirSync(directory),
    names: string[] = [];
  try {
    for (let entry = dir.readSync(); entry; entry = dir.readSync()) {
      if (names.length === limit)
        throw new IpcInboxHold(
          'IPC directory bound exceeded; preserve its batch.',
        );
      names.push(entry.name);
    }
    return names.sort();
  } finally {
    dir.closeSync();
  }
}
function syncDirectory(directory: string): void {
  const fd = fs.openSync(directory, nofollow | fs.constants.O_DIRECTORY);
  try {
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
}
function ownedDirectory(directory: string, uid: number, prepare = false): void {
  const stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink() || stat.uid !== uid)
    throw new IpcInboxHold(
      'IPC parent identity is unsafe; preserve requests and repair the runtime-owned directory.',
    );
  if ((stat.mode & 0o022) !== 0 && prepare) {
    const fd = fs.openSync(directory, nofollow | fs.constants.O_DIRECTORY);
    try {
      const held = fs.fstatSync(fd);
      if (
        !held.isDirectory() ||
        held.uid !== uid ||
        held.dev !== stat.dev ||
        held.ino !== stat.ino
      )
        throw new IpcInboxHold(
          'IPC parent changed before structural preparation.',
        );
      fs.fchmodSync(fd, held.mode & 0o7777 & ~0o022);
      fs.fsyncSync(fd);
      const bound = fs.lstatSync(directory);
      if (
        bound.isSymbolicLink() ||
        bound.dev !== held.dev ||
        bound.ino !== held.ino
      )
        throw new IpcInboxHold(
          'IPC parent binding changed during structural preparation.',
        );
    } finally {
      fs.closeSync(fd);
    }
  }
  if ((fs.lstatSync(directory).mode & 0o022) !== 0)
    throw new IpcInboxHold(
      'IPC parent permissions are unsafe; preserve requests and repair the runtime-owned directory.',
    );
}
/** Fixed host-owned quarantine namespace. Existing runtime-owned structural
 * directories only lose write bits; payload evidence is never altered.
 */
export function prepareIpcQuarantineRoot(base: string): string {
  const uid = process.geteuid!();
  ownedDirectory(base, uid, true);
  const root = path.join(base, 'errors');
  try {
    fs.mkdirSync(root, { mode: 0o700 });
  } catch (error) {
    if (!hasErrno(error, 'EEXIST')) throw error;
  }
  ownedDirectory(root, uid, true);
  return root;
}

function channelHasRequests(directory: string): boolean {
  let fd: number;
  try {
    fd = fs.openSync(directory, nofollow | fs.constants.O_DIRECTORY);
  } catch (error) {
    if (!isFileSystemError(error)) throw error;
    return true;
  }
  // Explicit resource disposal preserves a primary defect together with any
  // close failure, rather than losing either error in a finally throw.
  using _fdLease = { [Symbol.dispose]: () => fs.closeSync(fd) };
  let held: fs.Stats;
  try {
    held = fs.fstatSync(fd);
    const before = fs.lstatSync(directory);
    if (
      !before.isDirectory() ||
      before.isSymbolicLink() ||
      before.dev !== held.dev ||
      before.ino !== held.ino
    )
      return true;
  } catch (error) {
    if (!isFileSystemError(error)) throw error;
    return true;
  }
  // Non-authoritative boolean work hint only. Linux inspects our held FD.
  // Other platforms inspect the path: an ABA may expose directory-entry
  // metadata internally, never file contents, names in output, or authority.
  // Every effect still requires atomic capture and private leaf validation.
  let dir: fs.Dir;
  try {
    dir = fs.opendirSync(
      process.platform === 'linux' ? '/proc/self/fd/' + fd : directory,
    );
  } catch (error) {
    if (!isFileSystemError(error)) throw error;
    return true;
  }
  using _dirLease = { [Symbol.dispose]: () => dir.closeSync() };
  try {
    if (dir.readSync()) return true;
    const after = fs.lstatSync(directory);
    return (
      !after.isDirectory() ||
      after.isSymbolicLink() ||
      after.dev !== held.dev ||
      after.ino !== held.ino
    );
  } catch (error) {
    if (!isFileSystemError(error)) throw error;
    return true;
  }
}
function regularBytes(file: string, privateFile = false, uid?: number): Buffer {
  const fd = fs.openSync(file, nofollow | fs.constants.O_NONBLOCK);
  try {
    const stat = fs.fstatSync(fd);
    if (
      !stat.isFile() ||
      stat.nlink !== 1 ||
      stat.size > MAX_BYTES ||
      (privateFile && (stat.uid !== uid || (stat.mode & 0o077) !== 0))
    )
      throw new IpcInboxHold(
        'IPC request is not a bounded single-link regular file; preserve the claimed batch.',
      );
    // Small native owner records do not need a full payload-sized allocation.
    // Grow for concurrent file growth, preserving the same one-MiB admission.
    let buffer = Buffer.alloc(stat.size + 1);
    let count = 0;
    while (count < MAX_BYTES + 1) {
      if (count === buffer.length) {
        const grown = Buffer.alloc(Math.min(MAX_BYTES + 1, buffer.length * 2));
        buffer.copy(grown);
        buffer = grown;
      }
      const n = fs.readSync(fd, buffer, count, buffer.length - count, null);
      if (!n) break;
      count += n;
    }
    const bytes = buffer.subarray(0, count);
    if (bytes.length > MAX_BYTES)
      throw new IpcInboxHold('IPC request exceeded its byte bound.');
    return bytes;
  } finally {
    fs.closeSync(fd);
  }
}
function exclusiveWrite(file: string, bytes: string | Buffer): void {
  const fd = fs.openSync(
    file,
    fs.constants.O_WRONLY |
      fs.constants.O_CREAT |
      fs.constants.O_EXCL |
      fs.constants.O_NOFOLLOW,
    0o600,
  );
  try {
    fs.writeFileSync(fd, bytes);
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  syncDirectory(path.dirname(file));
}
function decodeUtf8(bytes: Buffer): string {
  try {
    return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(
      bytes,
    );
  } catch (error) {
    if (
      !(error instanceof TypeError) ||
      !hasErrno(error, 'ERR_ENCODING_INVALID_ENCODED_DATA')
    )
      throw error;
    throw new SyntaxError(
      'IPC bytes are not valid UTF-8; preserve the original request.',
      { cause: error },
    );
  }
}
function metadata(
  file: string,
  keys: string[],
  uid: number,
): Record<string, unknown> {
  const raw: unknown = JSON.parse(decodeUtf8(regularBytes(file, true, uid)));
  if (!raw || typeof raw !== 'object' || Array.isArray(raw))
    throw new IpcInboxHold('IPC owner metadata is invalid.');
  const record = raw as Record<string, unknown>;
  if (
    record.schema_version !== 1 ||
    Object.keys(record).sort().join() !== keys.sort().join()
  )
    throw new IpcInboxHold(
      'Unsupported IPC owner metadata; preserve the batch.',
    );
  return record;
}
function exists(file: string): boolean {
  try {
    fs.lstatSync(file);
    return true;
  } catch (error) {
    if (hasErrno(error, 'ENOENT')) return false;
    throw error;
  }
}
export interface IpcClaim {
  sourceGroup: string;
  channel: Channel;
  file: string;
  readonly filePath: string;
  read(): string;
  /** Durable dispatch intent MUST precede every external effect or DB mutation. */
  dispatch(): void;
  complete(): void;
}
export interface IpcInbox {
  capture(sourceGroup: string, channel: Channel): void;
  pending(): { claims: IpcClaim[]; held: number };
}
export function createIpcInbox(base: string): IpcInbox {
  // Native process identity, never an environment/caller-provided UID.
  const uid = process.geteuid!();
  ownedDirectory(base, uid, true);
  const root = path.join(base, PRIVATE_ROOT);
  if (!exists(root)) {
    fs.mkdirSync(root, { mode: 0o700 });
    syncDirectory(base);
  }
  ownedDirectory(root, uid);
  if ((fs.lstatSync(root).mode & 0o077) !== 0)
    throw new IpcInboxHold('IPC claim root must be private.');
  function batches(): string[] {
    return entries(root, MAX_BATCHES);
  }
  function batchIdentity(batch: string): {
    sourceGroup: string;
    channel: Channel;
  } {
    ownedDirectory(batch, uid);
    const owner = metadata(
      path.join(batch, 'owner.json'),
      ['schema_version', 'source_group', 'channel'],
      uid,
    );
    if (
      typeof owner.source_group !== 'string' ||
      !isValidGroupFolder(owner.source_group) ||
      (owner.channel !== 'messages' && owner.channel !== 'tasks')
    )
      throw new IpcInboxHold('IPC batch identity is invalid.');
    return { sourceGroup: owner.source_group, channel: owner.channel };
  }
  function assertCaptureCapacity(sourceGroup: string): void {
    const current = batches();
    if (current.length >= MAX_BATCHES)
      throw new IpcInboxHold(
        'IPC claim capacity exhausted; no requests were moved.',
      );
    let attributed = 0,
      unresolved = 0;
    for (const name of current) {
      try {
        if (batchIdentity(path.join(root, name)).sourceGroup === sourceGroup)
          attributed++;
      } catch (error) {
        if (
          !(error instanceof IpcInboxHold) &&
          !(error instanceof SyntaxError) &&
          !isFileSystemError(error)
        )
          throw error;
        // Private unknown owner state still consumes GLOBAL capacity and stays
        // held. Never invent a source identity or clear it to admit a request.
        unresolved++;
      }
    }
    if (unresolved)
      logger.warn(
        { unresolved },
        'IPC batch identity remains held; inspect private owner metadata before reconciliation.',
      );
    if (attributed >= MAX_SOURCE_BATCHES)
      throw new IpcInboxHold(
        'IPC source claim capacity exhausted; preserve this source and reconcile its held batches. No requests were moved.',
      );
  }
  // Verify existing whole-inbox quarantine evidence only. Never create or
  // resume a whole-directory move: an old publisher can still add valid JSON.
  function verifyPriorRemainder(
    batch: string,
    inbox: string,
    sourceGroup: string,
    channel: Channel,
  ): void {
    const batchId = path.basename(batch),
      errors = path.join(base, 'errors');
    if (!exists(errors)) return;
    ownedDirectory(errors, uid, true);
    const evidence = path.join(errors, 'unrecognized-' + batchId);
    if (!exists(evidence)) return;
    ownedDirectory(evidence, uid);
    if ((fs.lstatSync(evidence).mode & 0o077) !== 0)
      throw new IpcInboxHold('Prior IPC quarantine must remain private.');
    const record = metadata(
      path.join(evidence, 'owner.json'),
      ['schema_version', 'source_group', 'channel', 'batch_id'],
      uid,
    );
    if (
      record.source_group !== sourceGroup ||
      record.channel !== channel ||
      record.batch_id !== batchId
    )
      throw new IpcInboxHold(
        'Prior IPC quarantine identity changed; preserve both locations.',
      );
    if (exists(inbox))
      throw new IpcInboxHold(
        'Prior whole-inbox quarantine is incomplete; preserve both locations for owner reconciliation.',
      );
    const retained = fs.lstatSync(path.join(evidence, 'inbox'));
    if (
      !retained.isDirectory() ||
      retained.isSymbolicLink() ||
      entries(evidence, 2).some(
        (name) => name !== 'owner.json' && name !== 'inbox',
      )
    )
      throw new IpcInboxHold(
        'Prior IPC quarantine is incomplete; preserve its evidence.',
      );
    syncDirectory(evidence);
    syncDirectory(errors);
    syncDirectory(batch);
  }
  function quarantineCaptured(
    batch: string,
    slot: string,
    sourceGroup: string,
    channel: Channel,
    file: string,
    refuse = false,
  ): boolean {
    const batchId = path.basename(batch),
      requestId = path.basename(slot);
    const errors = path.join(base, 'errors');
    const evidence = path.join(errors, 'rejected-' + batchId + '-' + requestId);
    let existing = false;
    if (exists(errors)) {
      ownedDirectory(errors, uid, true);
      existing = exists(evidence);
    }
    if (!existing && !refuse) return false;
    // Only pre-dispatch captured input can enter this disposition. A durable
    // pending snapshot or any effect intent alongside it is ambiguous and held.
    if (
      exists(path.join(slot, 'pending')) ||
      exists(path.join(slot, 'in_flight'))
    )
      throw new IpcInboxHold(
        'IPC rejected input conflicts with dispatch state; preserve both.',
      );
    prepareIpcQuarantineRoot(base);
    syncDirectory(base);
    if (!existing) {
      fs.mkdirSync(evidence, { mode: 0o700 });
      exclusiveWrite(
        path.join(evidence, 'owner.json'),
        JSON.stringify({
          schema_version: 1,
          source_group: sourceGroup,
          channel,
          batch_id: batchId,
          request_id: requestId,
          file,
          reason: 'invalid_or_unreadable_input',
        }),
      );
      syncDirectory(errors);
    }
    ownedDirectory(evidence, uid);
    if ((fs.lstatSync(evidence).mode & 0o077) !== 0)
      throw new IpcInboxHold(
        'IPC rejected input quarantine must remain private.',
      );
    const record = metadata(
      path.join(evidence, 'owner.json'),
      [
        'schema_version',
        'source_group',
        'channel',
        'batch_id',
        'request_id',
        'file',
        'reason',
      ],
      uid,
    );
    if (
      record.source_group !== sourceGroup ||
      record.channel !== channel ||
      record.batch_id !== batchId ||
      record.request_id !== requestId ||
      record.file !== file ||
      record.reason !== 'invalid_or_unreadable_input'
    )
      throw new IpcInboxHold(
        'IPC rejected input quarantine identity changed; preserve its evidence.',
      );
    const captured = path.join(slot, 'captured'),
      retained = path.join(evidence, 'captured');
    if (exists(captured)) {
      if (exists(retained))
        throw new IpcInboxHold(
          'IPC rejected input quarantine is ambiguous; preserve both locations.',
        );
      // Rename the entry itself, including aliases/FIFOs/directories. Never open
      // its target or walk/delete unknown child contents during quarantine.
      fs.renameSync(captured, retained);
    }
    if (
      !exists(retained) ||
      entries(evidence, 2).some(
        (name) => name !== 'owner.json' && name !== 'captured',
      )
    )
      throw new IpcInboxHold(
        'IPC rejected input quarantine is incomplete; preserve its evidence.',
      );
    syncDirectory(evidence);
    syncDirectory(errors);
    syncDirectory(slot);
    logger.warn(
      { sourceGroup, channel },
      'IPC captured input refused; inspect private quarantine before any retry.',
    );
    return true;
  }
  return {
    capture(sourceGroup, channel) {
      if (!isValidGroupFolder(sourceGroup))
        throw new IpcInboxHold('Invalid IPC source identity.');
      const group = path.join(base, sourceGroup);
      ownedDirectory(base, uid, true);
      ownedDirectory(group, uid, true);
      const source = path.join(group, channel);
      if (!exists(source) || !channelHasRequests(source)) return;
      assertCaptureCapacity(sourceGroup);
      const batch = fs.mkdtempSync(path.join(root, 'batch-'));
      exclusiveWrite(
        path.join(batch, 'owner.json'),
        JSON.stringify({
          schema_version: 1,
          source_group: sourceGroup,
          channel,
        }),
      );
      fs.mkdirSync(path.join(batch, 'requests'), { mode: 0o700 });
      syncDirectory(batch);
      syncDirectory(root);
      // rename does not follow the final source entry, including an ABA symlink.
      fs.renameSync(source, path.join(batch, 'inbox'));
      syncDirectory(group);
      syncDirectory(batch);
      try {
        fs.mkdirSync(source, { mode: 0o755 });
      } catch (error) {
        if (!hasErrno(error, 'EEXIST')) throw error;
      }
    },
    pending() {
      const claims: IpcClaim[] = [];
      let held = 0;
      for (const name of batches()) {
        const batch = path.join(root, name);
        try {
          if (claims.length >= 1024)
            throw new IpcInboxHold(
              'IPC scan capacity reached; preserve later batches.',
            );
          const { sourceGroup, channel } = batchIdentity(batch);
          const requests = path.join(batch, 'requests');
          ownedDirectory(requests, uid);
          const slots = entries(requests);
          if (slots.length > 1024)
            throw new IpcInboxHold('IPC batch request bound exceeded.');
          for (const slot of slots)
            ownedDirectory(path.join(requests, slot), uid);
          // An unknown post-dispatch result holds the entire batch.
          if (
            slots.some((slot) => exists(path.join(requests, slot, 'in_flight')))
          ) {
            held++;
            continue;
          }
          const inbox = path.join(batch, 'inbox');
          verifyPriorRemainder(batch, inbox, sourceGroup, channel);
          if (exists(inbox)) {
            const stat = fs.lstatSync(inbox);
            if (!stat.isDirectory() || stat.isSymbolicLink())
              throw new IpcInboxHold(
                'Claimed IPC channel is an alias or non-directory.',
              );
            let requestCount = slots.length;
            for (const file of entries(inbox)) {
              if (++requestCount > 1024)
                throw new IpcInboxHold('IPC batch request bound exceeded.');
              const slot = fs.mkdtempSync(path.join(requests, 'request-'));
              exclusiveWrite(
                path.join(slot, 'owner.json'),
                JSON.stringify({ schema_version: 1, file }),
              );
              syncDirectory(requests);
              fs.renameSync(
                path.join(inbox, file),
                path.join(slot, 'captured'),
              );
              syncDirectory(inbox);
              syncDirectory(slot);
            }
          }
          const batchClaims: IpcClaim[] = [];
          for (const slotName of entries(requests)) {
            const slot = path.join(requests, slotName);
            ownedDirectory(slot, uid);
            const request = metadata(
              path.join(slot, 'owner.json'),
              ['schema_version', 'file'],
              uid,
            );
            if (
              typeof request.file !== 'string' ||
              path.basename(request.file) !== request.file ||
              request.file.length === 0
            )
              throw new IpcInboxHold('IPC leaf identity is invalid.');
            const captured = path.join(slot, 'captured'),
              pending = path.join(slot, 'pending'),
              inflight = path.join(slot, 'in_flight');
            const rejected = quarantineCaptured(
              batch,
              slot,
              sourceGroup,
              channel,
              request.file,
              !request.file.endsWith('.json'),
            );
            if (!rejected && exists(captured)) {
              // A new private inode freezes evidence even if a guest retained
              // an open descriptor to the original captured request.
              if (!exists(pending)) {
                const temporary = path.join(slot, 'snapshot.tmp');
                if (exists(temporary)) fs.unlinkSync(temporary);
                let bytes: Buffer | undefined;
                try {
                  const observed = fs.lstatSync(captured);
                  if (
                    !observed.isFile() ||
                    observed.isSymbolicLink() ||
                    observed.nlink !== 1 ||
                    observed.size > MAX_BYTES
                  )
                    throw new IpcInboxHold(
                      'Captured IPC input is not bounded single-link regular data.',
                    );
                  bytes = regularBytes(captured);
                } catch (error) {
                  // Source-controlled type/size/link/access refusal is known
                  // before dispatch. Storage I/O and all other defects HOLD or
                  // propagate through the existing owner; never infer absence.
                  if (
                    !(error instanceof IpcInboxHold) &&
                    !['ELOOP', 'EACCES', 'EPERM'].some((code) =>
                      hasErrno(error, code),
                    )
                  )
                    throw error;
                  quarantineCaptured(
                    batch,
                    slot,
                    sourceGroup,
                    channel,
                    request.file,
                    true,
                  );
                }
                if (bytes !== undefined) {
                  exclusiveWrite(temporary, bytes);
                  fs.renameSync(temporary, pending);
                  syncDirectory(slot);
                }
              }
              if (exists(captured)) {
                fs.unlinkSync(captured);
                syncDirectory(slot);
              }
            }
            if (!exists(pending)) {
              if (exists(inflight))
                throw new IpcInboxHold(
                  'IPC dispatch began during this scan; preserve its owner metadata.',
                );
              fs.unlinkSync(path.join(slot, 'owner.json'));
              fs.rmdirSync(slot);
              continue;
            }
            regularBytes(pending, true, uid);
            let current = pending;
            batchClaims.push({
              sourceGroup,
              channel,
              file: request.file,
              get filePath() {
                return current;
              },
              read: () => decodeUtf8(regularBytes(current, true, uid)),
              dispatch() {
                fs.renameSync(pending, inflight);
                syncDirectory(slot);
                current = inflight;
              },
              complete() {
                fs.unlinkSync(current);
                syncDirectory(slot);
              },
            });
          }
          if (claims.length + batchClaims.length > 1024)
            throw new IpcInboxHold(
              'IPC scan capacity reached; preserve later batches.',
            );
          claims.push(...batchClaims);
          // Remove only known empty containers; unknown children are retained
          // separately in durable quarantine, never deleted to free capacity.
          if (
            entries(requests).length === 0 &&
            (!exists(inbox) || entries(inbox).length === 0)
          ) {
            if (exists(inbox)) fs.rmdirSync(inbox);
            fs.rmdirSync(requests);
            fs.unlinkSync(path.join(batch, 'owner.json'));
            fs.rmdirSync(batch);
            syncDirectory(root);
          }
        } catch (error) {
          if (
            !(error instanceof IpcInboxHold) &&
            !(error instanceof SyntaxError) &&
            !isFileSystemError(error)
          )
            throw error;
          held++;
        }
      }
      // Opaque slot names are storage identities, never delivery order. Retain
      // the old channel's lexical filename ordering across snapshot/reload.
      claims.sort((a, b) =>
        Buffer.compare(
          Buffer.from([a.sourceGroup, a.channel, a.file].join('\0')),
          Buffer.from([b.sourceGroup, b.channel, b.file].join('\0')),
        ),
      );
      return { claims, held };
    },
  };
}
