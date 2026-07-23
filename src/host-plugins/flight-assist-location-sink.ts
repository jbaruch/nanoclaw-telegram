import { ASSISTANT_OWNER_TG_USER_ID, DATA_DIR } from '../config.js';
import { writeFlightAssistLocation } from './flight-assist-location.js';
import { registerLocationSink } from '../location-sinks.js';
import { registeredGroups } from '../orchestrator-state.js';

let registered = false;

/**
 * Flight-assist location artifact sink (#849), extracted from the core
 * channel `onLocation` path. Writes the travel tile's
 * `current-location.json` for `jbaruch/nanoclaw-travel`'s `precheck.py`
 * origin-resolution ladder (issue `nanoclaw-travel#18`). The DB
 * `locations` row (written by core before the sinks run) drives the
 * host-side TZ resolver; this artifact drives the per-group container's
 * time-to-leave origin. Owner-only filtering stays inside
 * `writeFlightAssistLocation` — it is this writer's policy, not core's.
 */
export function registerFlightAssistLocationSink(): void {
  if (registered) return;
  registered = true;
  registerLocationSink('flight-assist-location', (record) => {
    writeFlightAssistLocation(record, {
      groups: registeredGroups,
      ownerSenderId: ASSISTANT_OWNER_TG_USER_ID,
      dataDir: DATA_DIR,
    });
  });
}
