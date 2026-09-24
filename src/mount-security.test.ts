import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

type MountSecurity = typeof import('./mount-security.js');
type Logger = (typeof import('./logger.js'))['logger'];

const LOAD_FAILED =
  'Failed to load mount allowlist - additional mounts will be BLOCKED';

let tmpHome: string;
let originalHome: string | undefined;
let allowlistPath: string;
let mountSecurity: MountSecurity;
let logger: Logger;

function nodeError(code: string): NodeJS.ErrnoException {
  return Object.assign(new Error(`${code}: simulated failure`), { code });
}

function writeAllowlist(content: string): void {
  fs.mkdirSync(path.dirname(allowlistPath), { recursive: true });
  fs.writeFileSync(allowlistPath, content);
}

function validAllowlist(): string {
  return JSON.stringify({
    allowedRoots: [{ path: tmpHome, allowReadWrite: true }],
    blockedPatterns: ['custom-secret'],
    nonMainReadOnly: true,
  });
}

beforeEach(async () => {
  vi.clearAllMocks();
  tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-mount-test-'));
  originalHome = process.env.HOME;
  // config.js derives MOUNT_ALLOWLIST_PATH from HOME at module load, and
  // mount-security.js caches the allowlist per module instance.
  process.env.HOME = tmpHome;
  allowlistPath = path.join(
    tmpHome,
    '.config',
    'nanoclaw',
    'mount-allowlist.json',
  );
  vi.resetModules();
  mountSecurity = await import('./mount-security.js');
  ({ logger } = await import('./logger.js'));
});

afterEach(() => {
  vi.restoreAllMocks();
  if (originalHome === undefined) {
    delete process.env.HOME;
  } else {
    process.env.HOME = originalHome;
  }
  fs.rmSync(tmpHome, { recursive: true, force: true });
});

describe('loadMountAllowlist', () => {
  it('returns null and warns when the allowlist file is absent', () => {
    expect(mountSecurity.loadMountAllowlist()).toBeNull();
    expect(logger.warn).toHaveBeenCalledWith(
      { path: allowlistPath },
      expect.stringContaining('Mount allowlist not found'),
    );
  });

  it('returns null and logs when the allowlist is not valid JSON', () => {
    writeAllowlist('{ not json');
    expect(mountSecurity.loadMountAllowlist()).toBeNull();
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ path: allowlistPath }),
      LOAD_FAILED,
    );
  });

  it('returns null and logs when the allowlist document is a JSON null', () => {
    writeAllowlist('null');
    expect(mountSecurity.loadMountAllowlist()).toBeNull();
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ path: allowlistPath }),
      LOAD_FAILED,
    );
  });

  it('returns null and logs when the allowlist document is a JSON array', () => {
    writeAllowlist('[]');
    expect(mountSecurity.loadMountAllowlist()).toBeNull();
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ error: 'allowlist must be an object' }),
      LOAD_FAILED,
    );
  });

  it('returns null when an allowed root has the wrong shape', () => {
    writeAllowlist(
      JSON.stringify({
        allowedRoots: [{ path: tmpHome, allowReadWrite: 'yes' }],
        blockedPatterns: [],
        nonMainReadOnly: true,
      }),
    );

    expect(mountSecurity.loadMountAllowlist()).toBeNull();
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ error: 'allowedRoots must be an array' }),
      LOAD_FAILED,
    );
  });

  it('returns null and logs when a required field has the wrong shape', () => {
    writeAllowlist(
      JSON.stringify({
        allowedRoots: [],
        blockedPatterns: [],
        nonMainReadOnly: 'yes',
      }),
    );
    expect(mountSecurity.loadMountAllowlist()).toBeNull();
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ error: 'nonMainReadOnly must be a boolean' }),
      LOAD_FAILED,
    );
  });

  it('returns null and logs when the allowlist file cannot be read', () => {
    writeAllowlist(validAllowlist());
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw nodeError('EACCES');
    }) as typeof fs.readFileSync);

    expect(mountSecurity.loadMountAllowlist()).toBeNull();
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ error: expect.stringContaining('EACCES') }),
      LOAD_FAILED,
    );
  });

  it('propagates failures that are not file, parse, or structure errors', () => {
    writeAllowlist(validAllowlist());
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw new TypeError('unexpected implementation failure');
    }) as typeof fs.readFileSync);

    expect(() => mountSecurity.loadMountAllowlist()).toThrow(
      'unexpected implementation failure',
    );
    expect(logger.error).not.toHaveBeenCalled();
  });

  it('propagates coded programming failures', () => {
    writeAllowlist(validAllowlist());
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw Object.assign(new TypeError('invalid read argument'), {
        code: 'ERR_INVALID_ARG_TYPE',
      });
    }) as typeof fs.readFileSync);

    expect(() => mountSecurity.loadMountAllowlist()).toThrow(
      'invalid read argument',
    );
  });

  it('loads a valid allowlist once, merging the default blocked patterns', () => {
    writeAllowlist(validAllowlist());
    const readSpy = vi.spyOn(fs, 'readFileSync');

    const first = mountSecurity.loadMountAllowlist();
    const second = mountSecurity.loadMountAllowlist();

    expect(first).not.toBeNull();
    expect(second).toBe(first);
    expect(first?.blockedPatterns).toEqual(
      expect.arrayContaining(['.ssh', '.env', 'custom-secret']),
    );
    expect(readSpy).toHaveBeenCalledTimes(1);
  });
});

describe('validateAdditionalMounts', () => {
  it('rejects every additional mount while the allowlist failed to load', () => {
    writeAllowlist('{ not json');
    const requested = path.join(tmpHome, 'project');
    fs.mkdirSync(requested);

    const mounts = mountSecurity.validateAdditionalMounts(
      [{ hostPath: requested, containerPath: 'project' }],
      'test-group',
      true,
    );

    expect(mounts).toEqual([]);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({
        group: 'test-group',
        requestedPath: requested,
        reason: expect.stringContaining('No mount allowlist configured'),
      }),
      'Additional mount REJECTED',
    );
  });

  it('allows a mount under an allowed root once the allowlist loads', () => {
    writeAllowlist(validAllowlist());
    const requested = path.join(tmpHome, 'project');
    fs.mkdirSync(requested);

    const mounts = mountSecurity.validateAdditionalMounts(
      [{ hostPath: requested, containerPath: 'project', readonly: false }],
      'test-group',
      true,
    );

    expect(mounts).toEqual([
      {
        hostPath: fs.realpathSync(requested),
        containerPath: '/workspace/extra/project',
        readonly: false,
      },
    ]);
  });
});
