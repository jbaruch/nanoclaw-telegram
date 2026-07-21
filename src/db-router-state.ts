// Router-state key/value accessors (#751 seam 2, extracted verbatim
// from src/db.ts). Holds the message-loop cursors (`last_timestamp`,
// `last_agent_timestamp`) persisted across restarts; see
// `orchestrator-state.ts` for the in-memory side.
import { db } from './db-connection.js';

export function getRouterState(key: string): string | undefined {
  const row = db
    .prepare('SELECT value FROM router_state WHERE key = ?')
    .get(key) as { value: string } | undefined;
  return row?.value;
}

export function setRouterState(key: string, value: string): void {
  db.prepare(
    'INSERT OR REPLACE INTO router_state (key, value) VALUES (?, ?)',
  ).run(key, value);
}
