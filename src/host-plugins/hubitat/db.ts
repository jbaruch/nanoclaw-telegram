// Smart-home event accessors (#751 seam 9; moved into the Hubitat host
// plugin in #848).
//
// Covers the `smart_home_events` table fed by the Hubitat EventSocket
// listener (`./listener.ts`): event inserts, time-window and
// per-device selectors, retention cleanup, and the latest-state-per-
// device/attribute rollup. The table's schema migration stays in core
// (`src/db.ts`) — see the CREATE TABLE comment there for ownership.
import { db } from '../../db-connection.js';

export interface SmartHomeEvent {
  id: number;
  device_id: string;
  device_name: string;
  attribute_name: string;
  value: string;
  unit: string | null;
  description: string | null;
  source: string;
  timestamp: string;
}

export function insertSmartHomeEvent(event: Omit<SmartHomeEvent, 'id'>): void {
  db.prepare(
    `INSERT INTO smart_home_events (device_id, device_name, attribute_name, value, unit, description, source, timestamp)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    event.device_id,
    event.device_name,
    event.attribute_name,
    event.value,
    event.unit ?? null,
    event.description ?? null,
    event.source ?? 'DEVICE',
    event.timestamp,
  );
}

export function getSmartHomeEventsSince(
  since: string,
  deviceId?: string,
): SmartHomeEvent[] {
  if (deviceId) {
    return db
      .prepare(
        `SELECT * FROM smart_home_events WHERE timestamp > ? AND device_id = ? ORDER BY timestamp`,
      )
      .all(since, deviceId) as SmartHomeEvent[];
  }
  return db
    .prepare(
      `SELECT * FROM smart_home_events WHERE timestamp > ? ORDER BY timestamp`,
    )
    .all(since) as SmartHomeEvent[];
}

export function getSmartHomeEventsByHour(
  startHour: string,
  endHour: string,
): SmartHomeEvent[] {
  return db
    .prepare(
      `SELECT * FROM smart_home_events WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp`,
    )
    .all(startHour, endHour) as SmartHomeEvent[];
}

export function cleanupOldSmartHomeEvents(retentionDays: number): number {
  const cutoff = new Date(
    Date.now() - retentionDays * 24 * 60 * 60 * 1000,
  ).toISOString();
  const result = db
    .prepare(`DELETE FROM smart_home_events WHERE timestamp < ?`)
    .run(cutoff);
  return result.changes;
}

export function getLatestDeviceStates(): Array<{
  device_id: string;
  device_name: string;
  attribute_name: string;
  value: string;
  unit: string | null;
  timestamp: string;
}> {
  return db
    .prepare(
      `SELECT device_id, device_name, attribute_name, value, unit, timestamp
       FROM smart_home_events e1
       WHERE timestamp = (
         SELECT MAX(timestamp) FROM smart_home_events e2
         WHERE e2.device_id = e1.device_id AND e2.attribute_name = e1.attribute_name
       )
       ORDER BY device_name, attribute_name`,
    )
    .all() as Array<{
    device_id: string;
    device_name: string;
    attribute_name: string;
    value: string;
    unit: string | null;
    timestamp: string;
  }>;
}
