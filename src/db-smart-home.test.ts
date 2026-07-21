import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { _closeDatabase, _initTestDatabase } from './db.js';
import {
  cleanupOldSmartHomeEvents,
  getLatestDeviceStates,
  getSmartHomeEventsByHour,
  getSmartHomeEventsSince,
  insertSmartHomeEvent,
  type SmartHomeEvent,
} from './db-smart-home.js';

// Accessor tests for the smart_home_events table (#751 seam 9). Each
// test gets a fresh in-memory SQLite via `_initTestDatabase` so writes
// can't leak between tests (see `rules/testing-standards.md`
// independence). Fixed past timestamps only; the one clock-dependent
// helper (`cleanupOldSmartHomeEvents`) runs under a frozen system time
// so the retention cutoff is deterministic on any run date.

function event(
  overrides: Partial<Omit<SmartHomeEvent, 'id'>>,
): Omit<SmartHomeEvent, 'id'> {
  return {
    device_id: 'dev-1',
    device_name: 'Living Room Sensor',
    attribute_name: 'temperature',
    value: '21.5',
    unit: 'C',
    description: null,
    source: 'DEVICE',
    timestamp: '2026-01-05T10:00:00.000Z',
    ...overrides,
  };
}

beforeEach(() => {
  _initTestDatabase();
});

afterEach(() => {
  _closeDatabase();
});

describe('insertSmartHomeEvent + getSmartHomeEventsSince', () => {
  it('round-trips a full event row', () => {
    insertSmartHomeEvent(event({}));
    const rows = getSmartHomeEventsSince('2026-01-01T00:00:00.000Z');
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      device_id: 'dev-1',
      device_name: 'Living Room Sensor',
      attribute_name: 'temperature',
      value: '21.5',
      unit: 'C',
      description: null,
      source: 'DEVICE',
      timestamp: '2026-01-05T10:00:00.000Z',
    });
    expect(rows[0].id).toEqual(expect.any(Number));
  });

  it('returns only events strictly after `since`, in timestamp order', () => {
    insertSmartHomeEvent(event({ timestamp: '2026-01-05T12:00:00.000Z' }));
    insertSmartHomeEvent(event({ timestamp: '2026-01-05T10:00:00.000Z' }));
    insertSmartHomeEvent(event({ timestamp: '2026-01-05T08:00:00.000Z' }));
    const rows = getSmartHomeEventsSince('2026-01-05T08:00:00.000Z');
    expect(rows.map((r) => r.timestamp)).toEqual([
      '2026-01-05T10:00:00.000Z',
      '2026-01-05T12:00:00.000Z',
    ]);
  });

  it('filters by device_id when one is given', () => {
    insertSmartHomeEvent(event({ device_id: 'dev-1' }));
    insertSmartHomeEvent(
      event({ device_id: 'dev-2', device_name: 'Hallway Motion' }),
    );
    const rows = getSmartHomeEventsSince('2026-01-01T00:00:00.000Z', 'dev-2');
    expect(rows).toHaveLength(1);
    expect(rows[0].device_id).toBe('dev-2');
  });

  it('defaults source to DEVICE when the caller passes a nullish source', () => {
    // The accessor's `event.source ?? 'DEVICE'` fallback; the field is
    // typed non-null but the runtime guard covers listener payloads
    // that omit it.
    insertSmartHomeEvent({
      ...event({}),
      source: undefined as unknown as string,
    });
    const rows = getSmartHomeEventsSince('2026-01-01T00:00:00.000Z');
    expect(rows[0].source).toBe('DEVICE');
  });
});

describe('getSmartHomeEventsByHour', () => {
  it('is inclusive of the start bound and exclusive of the end bound', () => {
    insertSmartHomeEvent(event({ timestamp: '2026-01-05T09:59:59.000Z' }));
    insertSmartHomeEvent(event({ timestamp: '2026-01-05T10:00:00.000Z' }));
    insertSmartHomeEvent(event({ timestamp: '2026-01-05T10:59:59.000Z' }));
    insertSmartHomeEvent(event({ timestamp: '2026-01-05T11:00:00.000Z' }));
    const rows = getSmartHomeEventsByHour(
      '2026-01-05T10:00:00.000Z',
      '2026-01-05T11:00:00.000Z',
    );
    expect(rows.map((r) => r.timestamp)).toEqual([
      '2026-01-05T10:00:00.000Z',
      '2026-01-05T10:59:59.000Z',
    ]);
  });
});

describe('cleanupOldSmartHomeEvents', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-10T00:00:00.000Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('deletes only rows older than the retention window and reports the count', () => {
    // Frozen now = 2026-01-10; retentionDays = 7 puts the cutoff at
    // 2026-01-03.
    insertSmartHomeEvent(event({ timestamp: '2026-01-01T00:00:00.000Z' }));
    insertSmartHomeEvent(event({ timestamp: '2026-01-02T23:59:59.000Z' }));
    insertSmartHomeEvent(event({ timestamp: '2026-01-05T00:00:00.000Z' }));
    const deleted = cleanupOldSmartHomeEvents(7);
    expect(deleted).toBe(2);
    const remaining = getSmartHomeEventsSince('2020-01-01T00:00:00.000Z');
    expect(remaining).toHaveLength(1);
    expect(remaining[0].timestamp).toBe('2026-01-05T00:00:00.000Z');
  });

  it('returns 0 when nothing is old enough', () => {
    insertSmartHomeEvent(event({ timestamp: '2026-01-09T00:00:00.000Z' }));
    expect(cleanupOldSmartHomeEvents(7)).toBe(0);
  });
});

describe('getLatestDeviceStates', () => {
  it('returns the newest row per device/attribute pair, ordered by device then attribute', () => {
    insertSmartHomeEvent(
      event({ value: '20.0', timestamp: '2026-01-05T08:00:00.000Z' }),
    );
    insertSmartHomeEvent(
      event({ value: '22.5', timestamp: '2026-01-05T12:00:00.000Z' }),
    );
    insertSmartHomeEvent(
      event({
        attribute_name: 'humidity',
        value: '40',
        unit: '%',
        timestamp: '2026-01-05T09:00:00.000Z',
      }),
    );
    insertSmartHomeEvent(
      event({
        device_id: 'dev-2',
        device_name: 'Hallway Motion',
        attribute_name: 'motion',
        value: 'active',
        unit: null,
        timestamp: '2026-01-05T11:00:00.000Z',
      }),
    );
    const states = getLatestDeviceStates();
    expect(states).toEqual([
      {
        device_id: 'dev-2',
        device_name: 'Hallway Motion',
        attribute_name: 'motion',
        value: 'active',
        unit: null,
        timestamp: '2026-01-05T11:00:00.000Z',
      },
      {
        device_id: 'dev-1',
        device_name: 'Living Room Sensor',
        attribute_name: 'humidity',
        value: '40',
        unit: '%',
        timestamp: '2026-01-05T09:00:00.000Z',
      },
      {
        device_id: 'dev-1',
        device_name: 'Living Room Sensor',
        attribute_name: 'temperature',
        value: '22.5',
        unit: 'C',
        timestamp: '2026-01-05T12:00:00.000Z',
      },
    ]);
  });

  it('returns an empty array on an empty table', () => {
    expect(getLatestDeviceStates()).toEqual([]);
  });
});
