import fs from 'node:fs';
import path from 'node:path';

import { logger } from './logger.js';
import type { LocationRecord, RegisteredGroup } from './types.js';

// Schema matches `jbaruch/nanoclaw-flight-assist`'s
// `skills/flight-assist/state-schema.md` `current-location.json`
// contract: host orchestrator is sole writer, flight-assist's
// `state.read_current_location` is a non-owner reader per
// `coding-policy: stateful-artifacts`. A bump here MUST land in the
// tile's state-schema.md in the same release window.
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
