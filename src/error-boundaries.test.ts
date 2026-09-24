import fs from 'fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

const mockExecSync = vi.fn();
vi.mock('child_process', () => ({
  execSync: (...args: unknown[]) => mockExecSync(...args),
}));

import { cleanupOrphans } from './container-runtime.js';
import { createDraftStream, DraftStreamOpts } from './draft-stream.js';
import { readEnvFile } from './env.js';
import {
  assertValidGroupFolder,
  InvalidGroupFolderError,
  resolveGroupFolderPath,
} from './group-folder.js';
import { logger } from './logger.js';
import { isValidTimezone } from './timezone.js';

function nodeError(code: string): NodeJS.ErrnoException {
  return Object.assign(new Error(`${code}: simulated failure`), { code });
}

function signalledExec(message: string): Error {
  return Object.assign(new Error(message), {
    status: null,
    signal: 'SIGTERM',
    stdout: Buffer.alloc(0),
    stderr: Buffer.alloc(0),
  });
}

function missingBinaryExec(): Error {
  return Object.assign(new Error('spawnSync docker ENOENT'), {
    code: 'ENOENT',
    status: null,
    signal: null,
    stdout: Buffer.alloc(0),
    stderr: Buffer.alloc(0),
  });
}

function codedTypeError(): TypeError & { code: string } {
  return Object.assign(new TypeError('invalid argument'), {
    code: 'ERR_INVALID_ARG_TYPE',
  });
}

class TransportError extends Error {}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('readEnvFile', () => {
  it('returns no values when the .env file is absent', () => {
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw nodeError('ENOENT');
    }) as typeof fs.readFileSync);

    expect(readEnvFile(['ASSISTANT_NAME'])).toEqual({});
  });

  it('returns no values when the .env file cannot be read', () => {
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw nodeError('EACCES');
    }) as typeof fs.readFileSync);

    expect(readEnvFile(['ASSISTANT_NAME'])).toEqual({});
    expect(logger.debug).toHaveBeenCalled();
  });

  it('propagates failures that are not filesystem errors', () => {
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw new TypeError('unexpected implementation failure');
    }) as typeof fs.readFileSync);

    expect(() => readEnvFile(['ASSISTANT_NAME'])).toThrow(
      'unexpected implementation failure',
    );
  });

  it('propagates Node programming errors that carry a code', () => {
    const err = codedTypeError();
    vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
      throw err;
    }) as typeof fs.readFileSync);

    expect(() => readEnvFile(['ASSISTANT_NAME'])).toThrow(err);
  });

  it('parses only the requested keys', () => {
    vi.spyOn(fs, 'readFileSync').mockImplementation((() =>
      ['# comment', 'ASSISTANT_NAME="Bob"', 'TZ=UTC', 'IGNORED=1', ''].join(
        '\n',
      )) as unknown as typeof fs.readFileSync);

    expect(readEnvFile(['ASSISTANT_NAME', 'TZ'])).toEqual({
      ASSISTANT_NAME: 'Bob',
      TZ: 'UTC',
    });
  });
});

describe('isValidTimezone', () => {
  it('rejects an unknown zone and accepts a known one', () => {
    expect(isValidTimezone('Not/AZone')).toBe(false);
    expect(isValidTimezone('Europe/Berlin')).toBe(true);
  });

  it('propagates failures that are not invalid-zone errors', () => {
    vi.spyOn(Intl, 'DateTimeFormat').mockImplementation((() => {
      throw new TypeError('unexpected implementation failure');
    }) as unknown as typeof Intl.DateTimeFormat);

    expect(() => isValidTimezone('UTC')).toThrow(
      'unexpected implementation failure',
    );
  });
});

describe('group folder validation errors', () => {
  it('signals invalid folders with InvalidGroupFolderError', () => {
    expect(() => assertValidGroupFolder('../../etc')).toThrow(
      InvalidGroupFolderError,
    );
    expect(() => resolveGroupFolderPath('global')).toThrow(
      InvalidGroupFolderError,
    );
    expect(() => assertValidGroupFolder('family-chat')).not.toThrow();
  });
});

describe('cleanupOrphans runtime failures', () => {
  it('warns and continues when the runtime command is killed by a signal', () => {
    mockExecSync.mockImplementationOnce(() => {
      throw signalledExec('killed');
    });

    expect(() => cleanupOrphans()).not.toThrow();
    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ err: expect.any(Error) }),
      'Failed to clean up orphaned containers',
    );
  });

  it('warns and continues when the runtime binary is missing', () => {
    mockExecSync.mockImplementationOnce(() => {
      throw missingBinaryExec();
    });

    expect(() => cleanupOrphans()).not.toThrow();
    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ err: expect.any(Error) }),
      'Failed to clean up orphaned containers',
    );
  });

  it('keeps stopping the remaining orphans when one stop is killed by a signal', () => {
    mockExecSync.mockReturnValueOnce('nanoclaw-a-1\nnanoclaw-b-2\n');
    mockExecSync.mockImplementationOnce(() => {
      throw signalledExec('killed');
    });
    mockExecSync.mockReturnValueOnce('');

    expect(() => cleanupOrphans()).not.toThrow();
    expect(mockExecSync).toHaveBeenCalledTimes(3);
    expect(logger.info).toHaveBeenCalledWith(
      { count: 2, names: ['nanoclaw-a-1', 'nanoclaw-b-2'] },
      'Stopped orphaned containers',
    );
  });

  it('propagates coded TypeErrors from the runtime client', () => {
    const err = codedTypeError();
    mockExecSync.mockImplementationOnce(() => {
      throw err;
    });

    expect(() => cleanupOrphans()).toThrow(err);
    expect(logger.warn).not.toHaveBeenCalled();
  });
});

describe('draft stream transport failures', () => {
  const longEnough = 'x'.repeat(40);

  function transport(
    overrides: Partial<DraftStreamOpts> = {},
  ): DraftStreamOpts {
    return {
      sendMessage: vi.fn().mockResolvedValue(7),
      editMessage: vi.fn().mockResolvedValue(undefined),
      deleteMessage: vi.fn().mockResolvedValue(undefined),
      isExpectedError: (err): err is TransportError =>
        err instanceof TransportError,
      throttleMs: 0,
      ...overrides,
    };
  }

  it('absorbs classified transport rejections', async () => {
    const stream = createDraftStream(
      transport({
        sendMessage: vi.fn().mockRejectedValue(new TransportError('down')),
      }),
    );

    await expect(stream.finish(longEnough)).resolves.toBe(true);
    expect(logger.debug).toHaveBeenCalledWith(
      expect.objectContaining({ err: expect.any(Error) }),
      'Draft stream send/edit failed',
    );
  });

  it('propagates unclassified Error rejections from the transport', async () => {
    const stream = createDraftStream(
      transport({
        sendMessage: vi
          .fn()
          .mockRejectedValue(new TypeError('implementation failure')),
      }),
    );

    await expect(stream.finish(longEnough)).rejects.toThrow(
      'implementation failure',
    );
  });

  it('propagates non-Error rejections from the transport', async () => {
    const stream = createDraftStream(
      transport({ sendMessage: vi.fn().mockRejectedValue('not an error') }),
    );

    await expect(stream.finish(longEnough)).rejects.toBe('not an error');
  });

  it('absorbs classified rejections while cancelling a preview', async () => {
    const opts = transport({
      deleteMessage: vi
        .fn()
        .mockRejectedValue(new TransportError('already gone')),
    });
    const stream = createDraftStream(opts);
    stream.update(longEnough);

    await expect(stream.cancel()).resolves.toBeUndefined();
    expect(opts.deleteMessage).toHaveBeenCalledWith(7);
  });

  it('propagates non-Error rejections while cancelling a preview', async () => {
    const stream = createDraftStream(
      transport({ deleteMessage: vi.fn().mockRejectedValue('not an error') }),
    );
    stream.update(longEnough);

    await expect(stream.cancel()).rejects.toBe('not an error');
  });
});
