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

import { logger } from './logger.js';
import { isSenderAllowed, loadSenderAllowlist } from './sender-allowlist.js';

let tmpDir: string;

function cfgPath(): string {
  return path.join(tmpDir, 'sender-allowlist.json');
}

function nodeError(code: string): NodeJS.ErrnoException {
  return Object.assign(new Error(`${code}: simulated failure`), { code });
}

beforeEach(() => {
  vi.clearAllMocks();
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'allowlist-boundaries-'));
});

afterEach(() => {
  vi.restoreAllMocks();
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('loadSenderAllowlist boundaries', () => {
  it('falls back to the defaults when the config file is absent', () => {
    const cfg = loadSenderAllowlist(cfgPath());
    expect(cfg).toEqual({
      default: { allow: '*', mode: 'trigger' },
      chats: {},
      logDenied: true,
    });
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('falls back to the defaults and warns when the config cannot be read', () => {
    fs.writeFileSync(cfgPath(), '{}');
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw nodeError('EACCES');
    }) as typeof fs.readFileSync);

    const cfg = loadSenderAllowlist(cfgPath());

    expect(cfg.default).toEqual({ allow: '*', mode: 'trigger' });
    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ path: cfgPath() }),
      'sender-allowlist: cannot read config',
    );
  });

  it('falls back to the defaults and warns when the config is not JSON', () => {
    fs.writeFileSync(cfgPath(), '{ not json');

    const cfg = loadSenderAllowlist(cfgPath());

    expect(cfg.default).toEqual({ allow: '*', mode: 'trigger' });
    expect(logger.warn).toHaveBeenCalledWith(
      { path: cfgPath() },
      'sender-allowlist: invalid JSON',
    );
  });

  it('preserves the existing throw for a whole-document JSON null', () => {
    fs.writeFileSync(cfgPath(), 'null');

    expect(() => loadSenderAllowlist(cfgPath())).toThrow(TypeError);
  });

  it('propagates failures that are not filesystem or parse errors', () => {
    fs.writeFileSync(cfgPath(), '{}');
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw new TypeError('unexpected implementation failure');
    }) as typeof fs.readFileSync);

    expect(() => loadSenderAllowlist(cfgPath())).toThrow(
      'unexpected implementation failure',
    );
  });

  it('propagates coded programming failures', () => {
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw Object.assign(new TypeError('invalid read argument'), {
        code: 'ERR_INVALID_ARG_TYPE',
      });
    }) as typeof fs.readFileSync);

    expect(() => loadSenderAllowlist(cfgPath())).toThrow(
      'invalid read argument',
    );
  });

  it('still denies unlisted senders from a loaded config', () => {
    fs.writeFileSync(
      cfgPath(),
      JSON.stringify({
        default: { allow: ['1001'], mode: 'trigger' },
        chats: {},
      }),
    );

    const cfg = loadSenderAllowlist(cfgPath());

    expect(isSenderAllowed('tg:1', '1001', cfg)).toBe(true);
    expect(isSenderAllowed('tg:1', '2002', cfg)).toBe(false);
  });
});
