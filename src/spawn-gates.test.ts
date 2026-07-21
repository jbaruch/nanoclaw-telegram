import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  _resetSpawnGatesForTests,
  evaluateSpawnGate,
  isExpectedFsError,
  registerSpawnGate,
  type SpawnEligibility,
} from './spawn-gates.js';

// Core mechanism only (#846): registration + resolution. The trip-window
// gate's behavior lives with its owner in
// `src/host-plugins/flight-assist-spawn-gate.test.ts`.
describe('spawn-gate registry', () => {
  let groupDir: string;

  beforeEach(() => {
    groupDir = fs.mkdtempSync(path.join(os.tmpdir(), 'spawn-gate-'));
  });

  afterEach(() => {
    // The gate registry is module-global shared state — wipe it so no
    // test's registration leaks into another and order never matters.
    _resetSpawnGatesForTests();
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

  it('routes an evaluation to the gate registered for the skill', () => {
    const verdict: SpawnEligibility = {
      eligible: false,
      reason: 'test gate says no',
    };
    const gate = vi.fn(() => verdict);
    registerSpawnGate('test__windowed-skill', gate);
    const now = new Date('2026-07-01T00:00:00Z');
    expect(evaluateSpawnGate('test__windowed-skill', groupDir, now)).toEqual(
      verdict,
    );
    expect(gate).toHaveBeenCalledWith(groupDir, now);
    // Other skills stay ungated.
    expect(evaluateSpawnGate('test__other-skill', groupDir, now)).toBeNull();
  });

  it('throws on duplicate registration for the same skill', () => {
    registerSpawnGate('test__dup-skill', () => ({
      eligible: true,
      reason: 'x',
    }));
    expect(() =>
      registerSpawnGate('test__dup-skill', () => ({
        eligible: true,
        reason: 'y',
      })),
    ).toThrow(/already registered: test__dup-skill/);
  });
});

describe('isExpectedFsError', () => {
  it('matches an Error carrying an expected errno code', () => {
    const err = Object.assign(new Error('boom'), { code: 'ENOENT' });
    expect(isExpectedFsError(err)).toBe(true);
  });

  it('rejects an Error with an unexpected or missing code', () => {
    expect(
      isExpectedFsError(Object.assign(new Error('x'), { code: 'EIO' })),
    ).toBe(false);
    expect(isExpectedFsError(new Error('x'))).toBe(false);
  });

  it('returns false (does not throw) for non-Error throwables', () => {
    // Exported for plugin catch blocks — must be resilient to arbitrary
    // `unknown` values, not just Error instances.
    expect(isExpectedFsError(null)).toBe(false);
    expect(isExpectedFsError(undefined)).toBe(false);
    expect(isExpectedFsError('ENOENT')).toBe(false);
    expect(isExpectedFsError({ code: 'ENOENT' })).toBe(false);
  });
});
