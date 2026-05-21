import fs from 'fs';
import path from 'path';

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
      logger.warn(
        { err, chatJid: record.chat_jid, folder: group.folder },
        'Failed to write flight-assist current-location.json',
      );
      return;
    }
    throw err;
  }
}
