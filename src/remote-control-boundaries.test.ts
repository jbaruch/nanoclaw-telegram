import fs from 'fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Every filesystem call the module makes is mocked below; this directory is
// never touched.
vi.mock('./config.js', () => ({
  DATA_DIR: '/tmp/nanoclaw-rc-boundaries-test',
}));

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

vi.mock('child_process', () => ({
  spawn: vi.fn(),
}));

import {
  _getStateFilePath,
  _resetForTesting,
  getActiveSession,
  restoreRemoteControl,
  stopRemoteControl,
} from './remote-control.js';

const STATE_FILE = _getStateFilePath();

function nodeError(code: string): NodeJS.ErrnoException {
  return Object.assign(new Error(`${code}: simulated failure`), { code });
}

function persistedSession(pid: unknown): string {
  return JSON.stringify({
    pid,
    url: 'https://claude.ai/code?bridge=env_restored',
    startedBy: 'user1',
    startedInChat: 'tg:123',
    startedAt: '2026-01-01T00:00:00.000Z',
  });
}

function stateFileReturns(content: string): void {
  vi.spyOn(fs, 'readFileSync').mockImplementation(((p: string) => {
    if (String(p).endsWith('remote-control.json')) return content;
    throw nodeError('ENOENT');
  }) as typeof fs.readFileSync);
}

function stateFileFails(err: Error): void {
  vi.spyOn(fs, 'readFileSync').mockImplementation((() => {
    throw err;
  }) as typeof fs.readFileSync);
}

let unlinkSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  _resetForTesting();
  unlinkSpy = vi.spyOn(fs, 'unlinkSync').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
  _resetForTesting();
});

describe('restoreRemoteControl', () => {
  it('restores nothing when the state file is absent', () => {
    stateFileFails(nodeError('ENOENT'));

    expect(() => restoreRemoteControl()).not.toThrow();
    expect(getActiveSession()).toBeNull();
    expect(unlinkSpy).not.toHaveBeenCalled();
  });

  it.each(['EACCES', 'EISDIR'])(
    'skips the restore when the state file cannot be read (%s)',
    (code) => {
      stateFileFails(nodeError(code));

      expect(() => restoreRemoteControl()).not.toThrow();
      expect(getActiveSession()).toBeNull();
    },
  );

  it('clears the state when the persisted session is corrupt JSON', () => {
    stateFileReturns('not json{{{');

    expect(() => restoreRemoteControl()).not.toThrow();
    expect(getActiveSession()).toBeNull();
    expect(unlinkSpy).toHaveBeenCalledWith(STATE_FILE);
  });

  it.each(['null', '[]', '{"pid":42}'])(
    'clears the state when the persisted session has the wrong shape (%s)',
    (content) => {
      stateFileReturns(content);

      expect(() => restoreRemoteControl()).not.toThrow();
      expect(getActiveSession()).toBeNull();
      expect(unlinkSpy).toHaveBeenCalledWith(STATE_FILE);
    },
  );

  it.each(['ESRCH', 'EPERM'])(
    'clears the state when the persisted pid cannot be signalled (%s)',
    (code) => {
      stateFileReturns(persistedSession(424242));
      vi.spyOn(process, 'kill').mockImplementation((() => {
        throw nodeError(code);
      }) as typeof process.kill);

      expect(() => restoreRemoteControl()).not.toThrow();
      expect(getActiveSession()).toBeNull();
      expect(unlinkSpy).toHaveBeenCalledWith(STATE_FILE);
    },
  );

  it.each([['not-a-pid'], [1.5], [1e12]])(
    'clears the state when the persisted pid is malformed (%j)',
    (pid) => {
      // Node rejects these before signalling anything, so the real
      // process.kill is safe to reach here.
      stateFileReturns(persistedSession(pid));

      expect(() => restoreRemoteControl()).not.toThrow();
      expect(getActiveSession()).toBeNull();
      expect(unlinkSpy).toHaveBeenCalledWith(STATE_FILE);
    },
  );

  it('adopts a live session', () => {
    stateFileReturns(persistedSession(77777));
    vi.spyOn(process, 'kill').mockImplementation(
      (() => true) as typeof process.kill,
    );

    restoreRemoteControl();

    expect(getActiveSession()).toMatchObject({ pid: 77777 });
    expect(unlinkSpy).not.toHaveBeenCalled();
  });

  it('propagates failures that are not filesystem or parse errors', () => {
    stateFileFails(new TypeError('unexpected implementation failure'));

    expect(() => restoreRemoteControl()).toThrow(
      'unexpected implementation failure',
    );
  });

  it('propagates coded programming errors while reading state', () => {
    stateFileFails(
      Object.assign(new TypeError('invalid read argument'), {
        code: 'ERR_INVALID_ARG_TYPE',
      }),
    );

    expect(() => restoreRemoteControl()).toThrow('invalid read argument');
  });

  it.each(['EACCES', 'EISDIR'])(
    'contains operational failures while clearing stale state (%s)',
    (code) => {
      stateFileReturns('not json{{{');
      unlinkSpy.mockImplementation(() => {
        throw nodeError(code);
      });

      expect(() => restoreRemoteControl()).not.toThrow();
      expect(getActiveSession()).toBeNull();
    },
  );

  it('propagates programming errors while clearing stale state', () => {
    const err = new TypeError('unlink invariant failed');
    stateFileReturns('not json{{{');
    unlinkSpy.mockImplementation(() => {
      throw err;
    });

    expect(() => restoreRemoteControl()).toThrow(err);
  });
});

describe('stopRemoteControl', () => {
  it('reports success and clears state when the process already exited', () => {
    stateFileReturns(persistedSession(77777));
    const killSpy = vi
      .spyOn(process, 'kill')
      .mockImplementation((() => true) as typeof process.kill);
    restoreRemoteControl();

    killSpy.mockImplementation((() => {
      throw nodeError('ESRCH');
    }) as typeof process.kill);

    expect(stopRemoteControl()).toEqual({ ok: true });
    expect(getActiveSession()).toBeNull();
    expect(unlinkSpy).toHaveBeenCalledWith(STATE_FILE);
  });
});
