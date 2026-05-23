import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';

// Module mocks must come before the imports that consume them. Vitest
// hoists `vi.mock` automatically, but keeping them physically above the
// imports makes the read order match the execution order.
vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Pin HOST_UID/HOST_GID to the test runner's own uid/gid so the
// chown-after-mkdir actually mutates filesystem ownership in unit
// tests (chown-to-self always succeeds, no CAP_CHOWN needed) and the
// success-path assertion can observe `lstatSync(dir).uid/gid` per
// `testing-standards: Assert outcomes, not implementation details`.
// In prod HOST_UID/HOST_GID are the agent container's identity and
// the orchestrator runs as root; the codepath is identical, only the
// observable target uid/gid differs.
const TEST_UID = process.getuid?.() ?? 0;
const TEST_GID = process.getgid?.() ?? 0;
vi.mock('./config.js', () => ({
  HOST_UID: process.getuid?.() ?? 0,
  HOST_GID: process.getgid?.() ?? 0,
}));

import {
  CURRENT_LOCATION_SCHEMA_VERSION,
  writeFlightAssistLocation,
} from './flight-assist-location.js';
import type { LocationRecord, RegisteredGroup } from './types.js';

const OWNER_ID = '12345';
const CHAT_JID = 'tg:-1001';
const FOLDER = 'admin';

function ownerRecord(overrides: Partial<LocationRecord> = {}): LocationRecord {
  return {
    chat_jid: CHAT_JID,
    sender: OWNER_ID,
    message_id: '17',
    latitude: 59.6519,
    longitude: 17.9186,
    accuracy_m: 12,
    source: 'static',
    recorded_at: '2026-05-20T11:42:11Z',
    live_period: null,
    ...overrides,
  };
}

function group(overrides: Partial<RegisteredGroup> = {}): RegisteredGroup {
  return {
    name: 'admin',
    folder: FOLDER,
    trigger: null,
    added_at: '2026-05-01T00:00:00Z',
    isMain: true,
    ...overrides,
  };
}

describe('writeFlightAssistLocation', () => {
  let dataDir: string;

  beforeEach(() => {
    dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fa-loc-'));
  });

  afterEach(() => {
    fs.rmSync(dataDir, { recursive: true, force: true });
  });

  function targetPath(folder = FOLDER): string {
    return path.join(
      dataDir,
      'state',
      folder,
      'flight-assist',
      'current-location.json',
    );
  }

  it('writes the documented schema when the owner shares a location', () => {
    writeFlightAssistLocation(ownerRecord(), {
      groups: { [CHAT_JID]: group() },
      ownerSenderId: OWNER_ID,
      dataDir,
    });

    const written = JSON.parse(fs.readFileSync(targetPath(), 'utf8'));
    expect(written).toEqual({
      schema_version: CURRENT_LOCATION_SCHEMA_VERSION,
      latitude: 59.6519,
      longitude: 17.9186,
      captured_at: '2026-05-20T11:42:11Z',
    });
  });

  it('uses recorded_at verbatim — including live-update tick times', () => {
    writeFlightAssistLocation(
      ownerRecord({
        source: 'live_update',
        recorded_at: '2026-05-20T12:00:00Z',
        live_period: 3600,
      }),
      {
        groups: { [CHAT_JID]: group() },
        ownerSenderId: OWNER_ID,
        dataDir,
      },
    );

    const written = JSON.parse(fs.readFileSync(targetPath(), 'utf8'));
    expect(written.captured_at).toBe('2026-05-20T12:00:00Z');
  });

  it('skips writes when the sender is not the owner', () => {
    writeFlightAssistLocation(ownerRecord({ sender: '99999' }), {
      groups: { [CHAT_JID]: group() },
      ownerSenderId: OWNER_ID,
      dataDir,
    });

    expect(fs.existsSync(targetPath())).toBe(false);
  });

  it('skips writes when ownerSenderId is unset', () => {
    writeFlightAssistLocation(ownerRecord(), {
      groups: { [CHAT_JID]: group() },
      ownerSenderId: null,
      dataDir,
    });

    expect(fs.existsSync(targetPath())).toBe(false);
  });

  it('skips writes when the chat_jid is not in the groups map', () => {
    writeFlightAssistLocation(ownerRecord({ chat_jid: 'tg:-9999' }), {
      groups: { [CHAT_JID]: group() },
      ownerSenderId: OWNER_ID,
      dataDir,
    });

    expect(fs.existsSync(targetPath())).toBe(false);
  });

  it('writes into the resolved group folder, not the chat_jid', () => {
    writeFlightAssistLocation(ownerRecord(), {
      groups: { [CHAT_JID]: group({ folder: 'family-trip' }) },
      ownerSenderId: OWNER_ID,
      dataDir,
    });

    expect(fs.existsSync(targetPath('family-trip'))).toBe(true);
    expect(fs.existsSync(targetPath('admin'))).toBe(false);
  });

  it('overwrites a stale snapshot atomically (no .tmp left behind)', () => {
    const opts = {
      groups: { [CHAT_JID]: group() },
      ownerSenderId: OWNER_ID,
      dataDir,
    };
    writeFlightAssistLocation(ownerRecord(), opts);
    writeFlightAssistLocation(
      ownerRecord({
        latitude: 40.7128,
        longitude: -74.006,
        recorded_at: '2026-05-20T13:00:00Z',
      }),
      opts,
    );

    const written = JSON.parse(fs.readFileSync(targetPath(), 'utf8'));
    expect(written.latitude).toBe(40.7128);
    expect(written.longitude).toBe(-74.006);
    expect(written.captured_at).toBe('2026-05-20T13:00:00Z');
    expect(fs.existsSync(`${targetPath()}.tmp`)).toBe(false);
  });

  it('swallows filesystem errors instead of propagating', () => {
    // Point dataDir at a path that exists as a regular file — mkdirSync
    // recursive on a path under it raises ENOTDIR. The function must
    // log+continue rather than throw into the channel handler.
    const file = path.join(dataDir, 'not-a-dir');
    fs.writeFileSync(file, 'sentinel');

    expect(() =>
      writeFlightAssistLocation(ownerRecord(), {
        groups: { [CHAT_JID]: group() },
        ownerSenderId: OWNER_ID,
        dataDir: file,
      }),
    ).not.toThrow();
  });

  it('leaves the flight-assist dir owned by HOST_UID:HOST_GID after creating it', () => {
    // HOST_UID/HOST_GID are mocked to the test runner's own uid/gid so
    // the chown-to-self actually mutates ownership (no CAP_CHOWN needed
    // — chown-to-self succeeds for any user). The observable outcome we
    // care about is "agent container can write into this dir" → the
    // FS-level uid/gid match HOST_UID/HOST_GID. In prod the orchestrator
    // runs as root and HOST_UID is the agent container's uid (999); the
    // codepath is identical, only the observed target uid/gid differ.
    writeFlightAssistLocation(ownerRecord(), {
      groups: { [CHAT_JID]: group() },
      ownerSenderId: OWNER_ID,
      dataDir,
    });

    const expectedDir = path.join(dataDir, 'state', FOLDER, 'flight-assist');
    const stat = fs.lstatSync(expectedDir);
    expect(stat.uid).toBe(TEST_UID);
    expect(stat.gid).toBe(TEST_GID);
  });

  it('swallows EPERM/EACCES from lchown so non-root callers still write', () => {
    const lchownSpy = vi.spyOn(fs, 'lchownSync').mockImplementation(() => {
      const err = new Error(
        'EPERM: operation not permitted',
      ) as NodeJS.ErrnoException;
      err.code = 'EPERM';
      throw err;
    });

    writeFlightAssistLocation(ownerRecord(), {
      groups: { [CHAT_JID]: group() },
      ownerSenderId: OWNER_ID,
      dataDir,
    });

    expect(fs.existsSync(targetPath())).toBe(true);
    lchownSpy.mockRestore();
  });

  it('lets non-EPERM/EACCES lchown errors fall into the outer fs-error catch', () => {
    const lchownSpy = vi.spyOn(fs, 'lchownSync').mockImplementation(() => {
      const err = new Error(
        'EROFS: read-only file system',
      ) as NodeJS.ErrnoException;
      err.code = 'EROFS';
      throw err;
    });

    expect(() =>
      writeFlightAssistLocation(ownerRecord(), {
        groups: { [CHAT_JID]: group() },
        ownerSenderId: OWNER_ID,
        dataDir,
      }),
    ).not.toThrow();
    expect(fs.existsSync(targetPath())).toBe(false);
    lchownSpy.mockRestore();
  });

  it('refuses the entire write when flight-assist is a symlink to a real dir (privilege-escalation guard)', () => {
    // Simulate the attack: the agent (uid HOST_UID, write access on the
    // per-group state dir) rmdirs the empty flight-assist/ and replaces
    // it with a symlink pointing at a dir it controls. With a naive
    // guard (skip chown but continue), `mkdirSync` succeeds (target
    // exists), `writeFileSync` follows the symlink and writes
    // `current-location.json.tmp` into the attacker's dir as root —
    // leaking the owner's coordinates AND giving the attacker a
    // root-owned file under their tree. The contract is to refuse the
    // whole write so nothing flows through the symlink.
    const groupStateDir = path.join(dataDir, 'state', FOLDER);
    fs.mkdirSync(groupStateDir, { recursive: true });
    const attackerTarget = fs.mkdtempSync(path.join(os.tmpdir(), 'attacker-'));
    try {
      fs.symlinkSync(attackerTarget, path.join(groupStateDir, 'flight-assist'));

      writeFlightAssistLocation(ownerRecord(), {
        groups: { [CHAT_JID]: group() },
        ownerSenderId: OWNER_ID,
        dataDir,
      });

      // Security outcomes: nothing flowed through the symlink (no
      // current-location.json, no current-location.json.tmp, no stray
      // file at all in the attacker-controlled target).
      expect(fs.readdirSync(attackerTarget)).toEqual([]);
      expect(
        fs.existsSync(path.join(attackerTarget, 'current-location.json')),
      ).toBe(false);
    } finally {
      fs.rmSync(attackerTarget, { recursive: true, force: true });
    }
  });
});
