import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';

import {
  CURRENT_LOCATION_SCHEMA_VERSION,
  writeFlightAssistLocation,
} from './flight-assist-location.js';
import type { LocationRecord, RegisteredGroup } from './types.js';

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Pin HOST_UID/HOST_GID so the chown-after-mkdir path is exercised in
// unit tests. The values are picked to be different from any real test
// runner uid/gid so chownSync will EPERM — the writer must swallow
// EPERM/EACCES (orchestrator runs as root in prod, but tests don't).
vi.mock('./config.js', () => ({
  HOST_UID: 999,
  HOST_GID: 10,
}));

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

  it('chowns the flight-assist dir to HOST_UID:HOST_GID after creating it', () => {
    const chownSpy = vi.spyOn(fs, 'chownSync');

    writeFlightAssistLocation(ownerRecord(), {
      groups: { [CHAT_JID]: group() },
      ownerSenderId: OWNER_ID,
      dataDir,
    });

    const expectedDir = path.join(dataDir, 'state', FOLDER, 'flight-assist');
    expect(chownSpy).toHaveBeenCalledWith(expectedDir, 999, 10);
    chownSpy.mockRestore();
  });

  it('swallows EPERM/EACCES from chown so non-root callers still write', () => {
    const chownSpy = vi.spyOn(fs, 'chownSync').mockImplementation(() => {
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
    chownSpy.mockRestore();
  });

  it('lets non-EPERM/EACCES chown errors fall into the outer fs-error catch', () => {
    const chownSpy = vi.spyOn(fs, 'chownSync').mockImplementation(() => {
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
    chownSpy.mockRestore();
  });
});
