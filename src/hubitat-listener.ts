/**
 * Hubitat EventSocket WebSocket listener.
 * Connects to ws://<hub-ip>/eventsocket, deduplicates events,
 * and inserts them into the smart_home_events table.
 */
import Database from 'better-sqlite3';

import { HUBITAT_HUB_IP } from './config.js';
import { insertSmartHomeEvent } from './db.js';
import { logger } from './logger.js';

/** Raw event from Hubitat EventSocket */
interface HubitatRawEvent {
  source: string;
  name: string;
  displayName: string;
  value: string;
  unit?: string;
  deviceId: number;
  hubId?: number;
  locationId?: number;
  installedAppId?: number;
  descriptionText?: string;
}

// --- Dedup ---
// Hubitat sends 2-5 identical events per state change.
// Key: "${deviceId}-${attributeName}", value: "{value}|{timestamp}"
const DEDUP_TTL_MS = 3000;
const dedupMap = new Map<string, { value: string; expiresAt: number }>();

function isDuplicate(deviceId: number, name: string, value: string): boolean {
  const key = `${deviceId}-${name}`;
  const now = Date.now();

  // Prune expired entries periodically (every 100 checks)
  if (dedupMap.size > 500) {
    for (const [k, v] of dedupMap) {
      if (v.expiresAt < now) dedupMap.delete(k);
    }
  }

  const existing = dedupMap.get(key);
  if (existing && existing.value === value && existing.expiresAt > now) {
    return true;
  }

  dedupMap.set(key, { value, expiresAt: now + DEDUP_TTL_MS });
  return false;
}

// --- WebSocket connection ---

let ws: WebSocket | null = null;
let reconnectDelay = 5000;
const MAX_RECONNECT_DELAY = 60000;
let stopping = false;

function connect(): void {
  if (stopping) return;

  const url = `ws://${HUBITAT_HUB_IP}/eventsocket`;
  logger.info({ url }, 'Hubitat EventSocket connecting');

  ws = new WebSocket(url);

  ws.addEventListener('open', () => {
    logger.info('Hubitat EventSocket connected');
    reconnectDelay = 5000; // reset backoff on success
  });

  ws.addEventListener('message', (event) => {
    try {
      const raw: HubitatRawEvent = JSON.parse(String(event.data));

      // Skip non-device events (location events, app events)
      if (raw.source !== 'DEVICE' || raw.deviceId === undefined) return;

      // Dedup
      if (isDuplicate(raw.deviceId, raw.name, raw.value)) return;

      insertSmartHomeEvent({
        device_id: String(raw.deviceId),
        device_name: raw.displayName || `Device ${raw.deviceId}`,
        attribute_name: raw.name,
        value: raw.value,
        unit: raw.unit ?? null,
        description: raw.descriptionText ?? null,
        source: raw.source,
        timestamp: new Date().toISOString(),
      });
    } catch (err) {
      // WS message handler must not throw into the socket loop: a malformed
      // payload (SyntaxError) or a DB insert error (SqliteError) is warned and
      // dropped; anything else is a real defect and propagates.
      if (
        !(err instanceof SyntaxError) &&
        !(err instanceof Database.SqliteError)
      ) {
        throw err;
      }
      logger.warn(
        { err, data: String(event.data).slice(0, 200) },
        'Failed to handle Hubitat event (parse or DB insert)',
      );
    }
  });

  ws.addEventListener('close', (event) => {
    logger.warn(
      { code: event.code, reason: event.reason },
      'Hubitat EventSocket closed',
    );
    scheduleReconnect();
  });

  ws.addEventListener('error', (event) => {
    logger.error({ error: String(event) }, 'Hubitat EventSocket error');
    // close event will fire after this, triggering reconnect
  });
}

function scheduleReconnect(): void {
  if (stopping) return;
  logger.info({ delayMs: reconnectDelay }, 'Hubitat EventSocket reconnecting');
  setTimeout(() => {
    reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
    connect();
  }, reconnectDelay);
}

// --- Public API ---

export function startHubitatListener(): void {
  if (!HUBITAT_HUB_IP) {
    logger.debug('HUBITAT_HUB_IP not set, skipping EventSocket listener');
    return;
  }
  stopping = false;
  connect();
}

export function stopHubitatListener(): void {
  stopping = true;
  if (ws) {
    ws.close();
    ws = null;
  }
}
