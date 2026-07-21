import { ASSISTANT_NAME } from './config.js';
import {
  getAllRegisteredGroups,
  getAllSessions,
  getLastBotMessageTimestamp,
  getRouterState,
  setRouterState,
} from './db.js';
import { logger } from './logger.js';
import { RegisteredGroup } from './types.js';

/**
 * Shared orchestrator state and its persistence helpers, extracted from
 * `src/index.ts` (#749). These singletons are read and mutated across the
 * composition root (`index.ts`) and the inbound message pipeline;
 * centralising them here gives both one source of truth without
 * stale-module-binding hazards. Reassignment of the exported bindings
 * happens only inside this module — `loadState`, `_setRegisteredGroups`,
 * and the `setLastTimestamp` setter — so importers get a live read-only
 * view they can still mutate in place.
 */

// Most-recently-seen inbound message timestamp — the poll cursor's
// high-water mark. Reassigned by loadState + setLastTimestamp.
export let lastTimestamp = '';

// Nested by groupFolder → sessionName → sessionId. Tracks the user-facing
// `default` slot's SDK session chain so consecutive inbound messages
// resume the prior turn. `maintenance` entries may still be present here
// (e.g. loaded from persisted session state at startup, or written by a
// pre-#193 build), but scheduled tasks no longer update or resume that
// slot: they always start a fresh SDK turn (#193) to prevent cross-task
// `last_result` bleed, and the scheduler wipes their on-disk session
// artifacts (JSONL transcript + tool-results dir) immediately after
// each run completes.
export let sessions: Record<string, Record<string, string>> = {};
export let registeredGroups: Record<string, RegisteredGroup> = {};
export let lastAgentTimestamp: Record<string, string> = {};

/**
 * Advance the poll cursor high-water mark. A setter because importers of
 * the `lastTimestamp` live binding cannot rebind it themselves.
 */
export function setLastTimestamp(value: string): void {
  lastTimestamp = value;
}

export function loadState(): void {
  lastTimestamp = getRouterState('last_timestamp') || '';
  const agentTs = getRouterState('last_agent_timestamp');
  try {
    lastAgentTimestamp = agentTs ? JSON.parse(agentTs) : {};
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    logger.warn('Corrupted last_agent_timestamp in DB, resetting');
    lastAgentTimestamp = {};
  }
  sessions = getAllSessions();
  registeredGroups = getAllRegisteredGroups();
  logger.info(
    { groupCount: Object.keys(registeredGroups).length },
    'State loaded',
  );
}

export function saveState(): void {
  setRouterState('last_timestamp', lastTimestamp);
  setRouterState('last_agent_timestamp', JSON.stringify(lastAgentTimestamp));
}

/**
 * Return the message cursor for a group, recovering from the last bot reply
 * if lastAgentTimestamp is missing (new group, corrupted state, restart).
 */
export function getOrRecoverCursor(chatJid: string): string {
  const existing = lastAgentTimestamp[chatJid];
  if (existing) return existing;

  const botTs = getLastBotMessageTimestamp(chatJid, ASSISTANT_NAME);
  if (botTs) {
    logger.info(
      { chatJid, recoveredFrom: botTs },
      'Recovered message cursor from last bot reply',
    );
    lastAgentTimestamp[chatJid] = botTs;
    saveState();
    return botTs;
  }
  return '';
}

/** @internal - exported for testing */
export function _setRegisteredGroups(
  groups: Record<string, RegisteredGroup>,
): void {
  registeredGroups = groups;
}
