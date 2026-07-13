import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

import { evaluateSpawnGate } from './spawn-gates.js';

const FLIGHT_ASSIST = 'tessl__flight-assist';

// Fixed fixture trip (no wall-clock dependence): window runs from 24h
// before 2026-06-26 through the end of 2026-07-13.
const TRIP = { start: '2026-06-26', end: '2026-07-13' };

function writeTravelDb(dir: string, body: unknown): void {
  fs.writeFileSync(path.join(dir, 'travel-db.json'), JSON.stringify(body));
}

describe('evaluateSpawnGate', () => {
  let groupDir: string;

  beforeEach(() => {
    groupDir = fs.mkdtempSync(path.join(os.tmpdir(), 'spawn-gate-'));
  });

  afterEach(() => {
    fs.rmSync(groupDir, { recursive: true, force: true });
  });

  it('returns null for a task with no skill (spawns as normal)', () => {
    expect(
      evaluateSpawnGate(undefined, groupDir, new Date('2026-07-01T00:00:00Z')),
    ).toBeNull();
  });

  it('returns null for a skill with no registered gate', () => {
    expect(
      evaluateSpawnGate(
        'tessl__heartbeat',
        groupDir,
        new Date('2026-07-01T00:00:00Z'),
      ),
    ).toBeNull();
  });

  it('is out of window when travel-db is absent (no itinerary to act on)', () => {
    const v = evaluateSpawnGate(
      FLIGHT_ASSIST,
      groupDir,
      new Date('2026-07-01T00:00:00Z'),
    );
    expect(v?.eligible).toBe(false);
    expect(v?.reason).toMatch(/no travel itinerary/i);
  });

  it('is in window when a trip covers now', () => {
    writeTravelDb(groupDir, { trips: { 'trip-a': TRIP } });
    const v = evaluateSpawnGate(
      FLIGHT_ASSIST,
      groupDir,
      new Date('2026-07-01T12:00:00Z'),
    );
    expect(v?.eligible).toBe(true);
    expect(v?.reason).toMatch(/in trip window/i);
  });

  it('opens the window exactly 24h before trip start (inclusive)', () => {
    writeTravelDb(groupDir, { trips: { 'trip-a': TRIP } });
    // start = 2026-06-26T00:00Z → window opens 2026-06-25T00:00Z.
    expect(
      evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-06-25T00:00:00Z'),
      )?.eligible,
    ).toBe(true);
    expect(
      evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-06-24T23:59:59Z'),
      )?.eligible,
    ).toBe(false);
  });

  it('keeps the window open through the whole end day, closes the next day', () => {
    writeTravelDb(groupDir, { trips: { 'trip-a': TRIP } });
    // end = 2026-07-13 → window closes 2026-07-14T00:00Z.
    expect(
      evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-07-13T23:00:00Z'),
      )?.eligible,
    ).toBe(true);
    expect(
      evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-07-14T00:00:00Z'),
      )?.eligible,
    ).toBe(false);
  });

  it('is out of window for a purely future or past trip', () => {
    writeTravelDb(groupDir, { trips: { 'trip-a': TRIP } });
    expect(
      evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-01-01T00:00:00Z'),
      )?.eligible,
    ).toBe(false);
    expect(
      evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-12-01T00:00:00Z'),
      )?.eligible,
    ).toBe(false);
  });

  it('takes the union of overlapping/multiple trips', () => {
    writeTravelDb(groupDir, {
      trips: {
        past: { start: '2026-01-01', end: '2026-01-05' },
        current: TRIP,
      },
    });
    expect(
      evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-07-02T00:00:00Z'),
      )?.eligible,
    ).toBe(true);
  });

  it('is out of window when travel-db has no trips', () => {
    writeTravelDb(groupDir, { schema_version: 1, trips: {} });
    expect(
      evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-07-01T00:00:00Z'),
      )?.eligible,
    ).toBe(false);
  });

  it('fails OPEN when trips is the wrong shape (array)', () => {
    writeTravelDb(groupDir, {
      trips: [{ start: '2026-06-26', end: '2026-07-13' }],
    });
    const v = evaluateSpawnGate(
      FLIGHT_ASSIST,
      groupDir,
      new Date('2026-07-01T00:00:00Z'),
    );
    expect(v?.eligible).toBe(true);
    expect(v?.reason).toMatch(/wrong shape/i);
  });

  it('fails OPEN when the trips key is missing entirely', () => {
    writeTravelDb(groupDir, { schema_version: 1, generated_at: 'x' });
    const v = evaluateSpawnGate(
      FLIGHT_ASSIST,
      groupDir,
      new Date('2026-07-01T00:00:00Z'),
    );
    expect(v?.eligible).toBe(true);
    expect(v?.reason).toMatch(/missing trips key/i);
  });

  it('fails OPEN when travel-db is present but not valid JSON', () => {
    fs.writeFileSync(path.join(groupDir, 'travel-db.json'), 'not json {');
    const v = evaluateSpawnGate(
      FLIGHT_ASSIST,
      groupDir,
      new Date('2026-07-01T00:00:00Z'),
    );
    expect(v?.eligible).toBe(true);
    expect(v?.reason).toMatch(/not valid json/i);
  });

  it('skips trips with unparseable dates and stays out of window', () => {
    writeTravelDb(groupDir, {
      trips: { bad: { start: 'someday', end: 'later' } },
    });
    expect(
      evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-07-01T00:00:00Z'),
      )?.eligible,
    ).toBe(false);
  });

  it('fails OPEN (not closed as absent) when travel-db is a broken symlink', () => {
    // A dangling symlink is present-but-unusable, not absent — realpath
    // would throw ENOENT (same as missing), but lstat distinguishes it.
    fs.symlinkSync(
      path.join(groupDir, 'does-not-exist.json'),
      path.join(groupDir, 'travel-db.json'),
    );
    const v = evaluateSpawnGate(
      FLIGHT_ASSIST,
      groupDir,
      new Date('2026-07-01T00:00:00Z'),
    );
    expect(v?.eligible).toBe(true);
    expect(v?.reason).toMatch(/failing open/i);
  });

  it('fails OPEN when travel-db symlinks outside the group folder', () => {
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'outside-'));
    try {
      const target = path.join(outside, 'evil.json');
      fs.writeFileSync(target, JSON.stringify({ trips: {} }));
      fs.symlinkSync(target, path.join(groupDir, 'travel-db.json'));
      const v = evaluateSpawnGate(
        FLIGHT_ASSIST,
        groupDir,
        new Date('2026-07-01T00:00:00Z'),
      );
      expect(v?.eligible).toBe(true);
      expect(v?.reason).toMatch(/failing open/i);
    } finally {
      fs.rmSync(outside, { recursive: true, force: true });
    }
  });
});
