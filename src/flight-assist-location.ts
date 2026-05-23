import fs from 'fs';
import path from 'path';

import { HOST_GID, HOST_UID } from './config.js';
import { logger } from './logger.js';
import type { LocationRecord, RegisteredGroup } from './types.js';

/**
 * Authoritative schema definition for the `current-location.json`
 * stateful artifact. This module is the artifact's owner per
 * `jbaruch/coding-policy: stateful-artifacts`; flight-assist's
 * `state.read_current_location` is a documented non-owner reader.
 *
 * **Artifact path** (per-group):
 *   <DATA_DIR>/state/<group.folder>/flight-assist/current-location.json
 *   (mounted into the per-group agent container as
 *    /workspace/state/flight-assist/current-location.json)
 *
 * **Schema** (current version 1):
 *
 *   {
 *     "schema_version": 1,         // int, must equal CURRENT_LOCATION_SCHEMA_VERSION
 *     "latitude":       <number>,  // float, degrees in [-90, 90]
 *     "longitude":      <number>,  // float, degrees in [-180, 180]
 *     "captured_at":    "<string>" // ISO-8601 / RFC-3339 UTC instant
 *                                  // (Z or +00:00 suffix; reader rejects
 *                                  // any non-UTC offset)
 *   }
 *
 * **Writer contract** (this module):
 * - Atomic write (tmp + rename) — readers never see a half-written file.
 * - Owner-only — only emits when the location event's sender matches
 *   `ASSISTANT_OWNER_TG_USER_ID`. Group-member pins are ignored.
 * - Every write produces a complete record at the current schema
 *   version. No partial updates, no merge with prior on-disk state.
 * - Self-heals `flight-assist/` ownership on every write: `lchownSync`
 *   to `HOST_UID:HOST_GID` after the idempotent `mkdirSync`, gated by
 *   an `lstatSync` symlink check (the dir lives under a mount the agent
 *   can write, so a non-traversing chown is mandatory — same pattern
 *   as `container-runner.ts: chownRecursive`). Without this, the
 *   orchestrator (root inside its container) leaves the dir root-owned
 *   mode 755 on first creation and the agent's sync_tripit precheck
 *   fails with PermissionError when it tries to write its own state
 *   files (`flight-*.json`, sync-tripit locks). EPERM/EACCES on the
 *   chown means we're not root (unit tests, or an unprivileged
 *   orchestrator) — both surfaced via `logger.warn`.
 *
 * **Reader contract** (flight-assist >= 0.1.9, non-owner):
 * - Returns `None` on missing file, malformed JSON, missing/wrong-type
 *   fields, out-of-range coords, non-UTC `captured_at`, or any
 *   `schema_version` mismatch.
 * - MUST NOT migrate. On `schema_version` mismatch the reader treats
 *   the file as "no usable snapshot" and the precheck's origin-
 *   resolution ladder falls back to `home_address`.
 *
 * **Migration policy** (owner-side):
 * - The host is the sole writer. There is no on-disk state to migrate
 *   in place — each owner write overwrites the previous record with a
 *   complete payload at the current schema version, so a bump simply
 *   produces the new shape on the next location event.
 * - Bumping the schema MUST coordinate with the reader tile in the
 *   same release window: update `CURRENT_LOCATION_SCHEMA_VERSION`
 *   here AND publish a flight-assist release that accepts the new
 *   version. Between the two releases the reader rejects the new
 *   shape and the precheck falls back to `home_address` — the
 *   degradation is graceful, never crashing.
 *
 * **Field freshness** is a reader-side concern, not part of the on-
 * disk shape. flight-assist's `_resolve_time_to_leave_origin` accepts
 * a snapshot only when `now - captured_at <= 30min`.
 */
export const CURRENT_LOCATION_SCHEMA_VERSION = 1;

const STATE_SUBDIR = 'state';
const FLIGHT_ASSIST_SUBDIR = 'flight-assist';
const CURRENT_LOCATION_FILE = 'current-location.json';

export interface FlightAssistLocationOptions {
  groups: Record<string, RegisteredGroup>;
  // Telegram numeric user id of the owner. Captures from other senders
  // (group members, anonymous admins) are ignored — the snapshot
  // models "where the owner is", not "where any message author is".
  // Null/undefined/empty when `ASSISTANT_OWNER_TG_USER_ID` is unset;
  // in that case no file is ever written.
  ownerSenderId: string | null | undefined;
  dataDir: string;
}

export function writeFlightAssistLocation(
  record: LocationRecord,
  opts: FlightAssistLocationOptions,
): void {
  if (!opts.ownerSenderId || record.sender !== opts.ownerSenderId) return;
  const group = opts.groups[record.chat_jid];
  if (!group) return;

  const dir = path.join(
    opts.dataDir,
    STATE_SUBDIR,
    group.folder,
    FLIGHT_ASSIST_SUBDIR,
  );
  const target = path.join(dir, CURRENT_LOCATION_FILE);
  const payload = {
    schema_version: CURRENT_LOCATION_SCHEMA_VERSION,
    latitude: record.latitude,
    longitude: record.longitude,
    captured_at: record.recorded_at,
  };

  try {
    fs.mkdirSync(dir, { recursive: true });
    // Self-heal ownership on every write (mkdir is idempotent under
    // `recursive: true`, so this runs whether the dir is freshly created
    // or already existed from a prior root-owned build).
    //
    // The orchestrator runs as root inside its container; without this
    // chown, `mkdirSync` leaves `flight-assist/` owned by root:root and
    // the agent container (uid HOST_UID) can't write its own state
    // files (flight-*.json, sync-tripit lock files) into the dir
    // (parent mode 755 blocks non-owner writes). `container-runner.ts`
    // chowns the per-group state dir at creation (see HOST_UID block)
    // but never visits host-created per-skill subdirs — so this writer
    // owns its own ownership.
    //
    // Symlink safety: the dir lives under `/workspace/state` which the
    // agent (lower-privilege than root) can write. A non-traversing
    // chown is mandatory — `chownSync` would follow a symlink and let
    // an attacker who planted `flight-assist -> /etc/passwd` (after
    // `rmdir`-ing the empty subdir) redirect the root chown to an
    // arbitrary target. Mirrors `container-runner.ts: chownRecursive`'s
    // `lchownSync` pattern. The `lstatSync` guard refuses the WHOLE
    // write when the path isn't a real directory — chown AND the
    // subsequent `writeFileSync` + `renameSync` would both follow the
    // symlink, so logging without returning would still write
    // `current-location.json` as root through the symlink target.
    //
    // EPERM/EACCES means we're not root (e.g. unit tests, or the
    // orchestrator container running unprivileged). Warn and continue
    // — the file write further down still succeeds when the dir is
    // already owned by the calling user. Other codes propagate to the
    // outer catch.
    if (HOST_UID !== undefined && HOST_GID !== undefined) {
      const stat = fs.lstatSync(dir);
      if (!stat.isDirectory()) {
        logger.warn(
          { dir, chatJid: record.chat_jid, folder: group.folder },
          `flight-assist state path is not a real directory (likely a symlink planted by the agent container) — refusing the entire write. Inspect ${dir} and remove the symlink before the next location event. The precheck will fall back to home_address until then.`,
        );
        return;
      }
      try {
        fs.lchownSync(dir, HOST_UID, HOST_GID);
      } catch (chownErr: unknown) {
        const chownCode = (chownErr as NodeJS.ErrnoException)?.code;
        if (chownCode !== 'EPERM' && chownCode !== 'EACCES') {
          throw chownErr;
        }
        logger.warn(
          { err: chownErr, code: chownCode, dir },
          `flight-assist dir chown skipped (${chownCode}) — orchestrator is not running with CAP_CHOWN. Agent-side sync_tripit precheck will continue to fail with PermissionError until the dir is chowned manually (\`chown -R $HOST_UID:$HOST_GID ${dir}\`) or the orchestrator is restarted as root.`,
        );
      }
    }
    const tmp = `${target}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(payload));
    fs.renameSync(tmp, target);
  } catch (err: unknown) {
    // Narrow to filesystem errors (NodeJS.ErrnoException — anything
    // with a `.code` like ENOENT / EACCES / EPERM / ENOSPC / EROFS /
    // ENOTDIR / EIO). flight-assist's reader treats a missing or
    // unreadable file as "no usable snapshot" and falls back to
    // `home_address`, so swallowing fs failures degrades cleanly to
    // the prior behaviour rather than corrupting either side.
    // Programming bugs (no `.code`) propagate so they surface in dev
    // instead of being hidden behind a warn.
    if (err instanceof Error && 'code' in err) {
      const code = (err as NodeJS.ErrnoException).code;
      logger.warn(
        { err, code, chatJid: record.chat_jid, folder: group.folder, target },
        `flight-assist current-location.json write failed (${code}) — flight-assist precheck will fall back to home_address until the next successful write. Remediation: EACCES/EPERM → \`chown -R $HOST_UID:$HOST_GID ${dir}\` (the orchestrator chowns the state dir at container-runner.ts but a stale-ownership group folder can still trip the per-skill subdir). ENOSPC → free disk on the data volume. EROFS → state volume is mounted read-only; check the mount in docker-compose.yml. ENOTDIR → a path component is a file, not a directory; inspect ${target}.`,
      );
      return;
    }
    throw err;
  }
}
