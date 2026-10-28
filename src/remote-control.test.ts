import fs from 'fs';
import { EventEmitter } from 'events';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock config before importing the module under test
vi.mock('./config.js', () => ({
  DATA_DIR: '/tmp/nanoclaw-rc-test',
}));

// Mock child_process
const spawnMock = vi.fn();
vi.mock('child_process', async (original) => ({
  ...(await original<typeof import('child_process')>()),
  spawn: (...args: unknown[]) => spawnMock(...args),
}));
const identityFixture = vi.hoisted(() => ({
  kind: 'linux-birth' as const,
  boot_id: '00000000-0000-4000-8000-000000000001',
  start_ticks: '1',
  executable: '/synthetic/claude',
  command_sha256: 'a'.repeat(64),
}));
vi.mock('./remote-control-identity.js', async (original) => ({
  ...(await original<typeof import('./remote-control-identity.js')>()),
  captureRemoteControlIdentity: vi.fn(() => ({ ...identityFixture })),
  verifyRemoteControlIdentity: vi.fn((pid: number) => {
    try {
      process.kill(pid, 0);
      return 'same';
    } catch (error) {
      if (
        !(error instanceof Error) ||
        !('code' in error) ||
        error.code !== 'ESRCH'
      )
        throw error;
      return 'absent';
    }
  }),
}));

import { verifyRemoteControlIdentity } from './remote-control-identity.js';
import {
  startRemoteControl,
  stopRemoteControl,
  restoreRemoteControl,
  getActiveSession,
  _resetForTesting,
  _getStateFilePath,
} from './remote-control.js';

// --- Helpers ---

function createMockProcess(pid = 12345) {
  return Object.assign(new EventEmitter(), {
    pid,
    exitCode: null,
    signalCode: null,
    unref: vi.fn(),
    kill: vi.fn(),
    stdin: Object.assign(new EventEmitter(), { write: vi.fn(), end: vi.fn() }),
  });
}

function mockOwnedRetirement() {
  const retired = new Set<number>();
  return vi.spyOn(process, 'kill').mockImplementation((pid, signal) => {
    if (signal === 0 && retired.has(Math.abs(pid)))
      throw Object.assign(new Error('fixed absent owned process'), {
        code: 'ESRCH',
      });
    if (signal === 'SIGTERM' || signal === 'SIGKILL')
      retired.add(Math.abs(pid));
    return true;
  });
}

describe('remote-control', () => {
  const STATE_FILE = _getStateFilePath();
  let readFileSyncSpy: ReturnType<typeof vi.spyOn>;
  let writeFileSyncSpy: ReturnType<typeof vi.spyOn>;
  let unlinkSyncSpy: ReturnType<typeof vi.spyOn>;
  let _mkdirSyncSpy: ReturnType<typeof vi.spyOn>;
  let openSyncSpy: ReturnType<typeof vi.spyOn>;
  let closeSyncSpy: ReturnType<typeof vi.spyOn>;

  // Track what readFileSync should return for the stdout file
  let stdoutFileContent: string | Buffer;
  let readableFile: string;
  let readOffset: number;
  let startingBytes: string;

  beforeEach(() => {
    _resetForTesting();
    spawnMock.mockReset();
    stdoutFileContent = '';
    startingBytes = '';
    readableFile = '';
    readOffset = 0;

    // Default fs mocks
    vi.spyOn(fs, 'mkdtempSync').mockReturnValue(
      '/tmp/nanoclaw-rc-test/.remote-control-fixed',
    );
    vi.spyOn(fs, 'renameSync').mockImplementation(() => {});
    vi.spyOn(fs, 'rmSync').mockImplementation(() => {});
    _mkdirSyncSpy = vi
      .spyOn(fs, 'mkdirSync')
      .mockImplementation(() => undefined);
    writeFileSyncSpy = vi
      .spyOn(fs, 'writeFileSync')
      .mockImplementation((file, bytes) => {
        if (typeof file === 'number') startingBytes = String(bytes);
      });
    unlinkSyncSpy = vi.spyOn(fs, 'unlinkSync').mockImplementation((file) => {
      if (String(file).endsWith('remote-control-starting.json'))
        startingBytes = '';
    });
    openSyncSpy = vi.spyOn(fs, 'openSync').mockImplementation((file, flags) => {
      if (
        String(file).endsWith('remote-control-starting.json') &&
        typeof flags === 'number' &&
        flags & fs.constants.O_EXCL &&
        startingBytes
      )
        throw Object.assign(new Error('fixed occupied gate'), {
          code: 'EEXIST',
        });
      if (
        typeof flags === 'number' &&
        !(flags & (fs.constants.O_WRONLY | fs.constants.O_RDWR))
      ) {
        readableFile = String(file);
        readOffset = 0;
      }
      return 42;
    });
    closeSyncSpy = vi.spyOn(fs, 'closeSync').mockImplementation(() => {});
    vi.spyOn(fs, 'fsyncSync').mockImplementation(() => {});
    vi.spyOn(fs, 'fchmodSync').mockImplementation(() => {});
    const identity = {
      dev: 1,
      ino: 2,
      uid: process.geteuid!(),
      mode: 0o700,
      nlink: 1,
      size: 0,
      isFile: () => true,
      isDirectory: () => true,
      isSymbolicLink: () => false,
    } as fs.Stats;
    vi.spyOn(fs, 'fstatSync').mockReturnValue(identity);
    vi.spyOn(fs, 'lstatSync').mockReturnValue(identity);

    // readFileSync: return stdoutFileContent for the stdout file, state file, etc.
    readFileSyncSpy = vi.spyOn(fs, 'readFileSync').mockImplementation(((
      p: string,
    ) => {
      if (p.endsWith('remote-control.stdout')) return stdoutFileContent;
      if (p.endsWith('remote-control-starting.json')) return startingBytes;
      if (p.endsWith('remote-control.json')) {
        throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
      }
      return '';
    }) as typeof fs.readFileSync);
    const read = (
      _fd: number,
      target: NodeJS.ArrayBufferView,
      ...args: unknown[]
    ): number => {
      if (!Buffer.isBuffer(target))
        throw new TypeError('Unexpected fixture buffer');
      const offset = Number(args[0]),
        length = Number(args[1]),
        position = args[2];
      const value = readFileSyncSpy(readableFile, 'utf8');
      const bytes = Buffer.isBuffer(value)
        ? value
        : Buffer.from(
            String(value) +
              (readableFile.endsWith('remote-control.stdout') ? '\n' : ''),
          );
      const start = typeof position === 'number' ? position : readOffset;
      const n = bytes.copy(
        target,
        offset,
        start,
        Math.min(bytes.length, start + length),
      );
      if (position === null) readOffset += n;
      return n;
    };
    vi.spyOn(fs, 'readSync').mockImplementation(read as typeof fs.readSync);
  });

  function mockState(raw: string) {
    readFileSyncSpy.mockImplementation((file: fs.PathOrFileDescriptor) => {
      if (String(file).endsWith('remote-control.json')) return raw;
      if (String(file).endsWith('remote-control-starting.json'))
        return startingBytes;
      if (String(file).endsWith('remote-control.stdout'))
        return stdoutFileContent;
      return '';
    });
  }

  afterEach(() => {
    _resetForTesting();
    vi.restoreAllMocks();
  });

  // --- startRemoteControl ---

  describe('startRemoteControl', () => {
    it('shares one pending launch across concurrent callers before URL discovery', async () => {
      vi.useFakeTimers();
      spawnMock.mockReturnValue(createMockProcess(12345));
      mockOwnedRetirement();
      const first = startRemoteControl('owner', 'tg:123', '/project');
      const second = startRemoteControl('owner', 'tg:123', '/project');
      expect(second).toBe(first);
      expect(spawnMock).toHaveBeenCalledTimes(1);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_concurrent';
      await vi.advanceTimersByTimeAsync(200);
      await expect(first).resolves.toEqual({
        ok: true,
        url: stdoutFileContent,
      });
      await expect(second).resolves.toEqual({
        ok: true,
        url: stdoutFileContent,
      });
      expect(getActiveSession()?.pid).toBe(12345);
      expect(vi.getTimerCount()).toBe(0);
      vi.useRealTimers();
    });
    it('spawns claude remote-control and returns the URL', async () => {
      const proc = createMockProcess();
      spawnMock.mockReturnValue(proc);

      // Simulate URL appearing in stdout file on first poll
      stdoutFileContent =
        'Session URL: https://claude.ai/code?bridge=env_abc123\n';
      mockOwnedRetirement();

      const result = await startRemoteControl('user1', 'tg:123', '/project');

      expect(result).toEqual({
        ok: true,
        url: 'https://claude.ai/code?bridge=env_abc123',
      });
      expect(spawnMock).toHaveBeenCalledWith(
        'claude',
        [
          'remote-control',
          '--name',
          expect.stringMatching(/^NanoClaw Remote [a-f0-9-]{36}$/),
        ],
        expect.objectContaining({ cwd: '/project', detached: true }),
      );
      expect(proc.unref).toHaveBeenCalled();
    });

    it('uses file descriptors for stdout/stderr (not pipes)', async () => {
      const proc = createMockProcess();
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_test\n';
      mockOwnedRetirement();

      await startRemoteControl('user1', 'tg:123', '/project');

      const spawnCall = spawnMock.mock.calls[0];
      const options = spawnCall[2];
      // stdio[0] is 'pipe' so we can write 'y' to accept the prompt
      expect(options.stdio[0]).toBe('pipe');
      expect(typeof options.stdio[1]).toBe('number');
      expect(typeof options.stdio[2]).toBe('number');
    });

    it('closes file descriptors in parent after spawn', async () => {
      const proc = createMockProcess();
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_test\n';
      mockOwnedRetirement();

      await startRemoteControl('user1', 'tg:123', '/project');

      // Two child log descriptors, the reservation and two parent fsyncs.
      expect(closeSyncSpy.mock.calls.length).toBe(
        openSyncSpy.mock.calls.length,
      );
    });

    it('saves state to disk after capturing URL', async () => {
      const proc = createMockProcess(99999);
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_save\n';
      mockOwnedRetirement();

      await startRemoteControl('user1', 'tg:123', '/project');

      expect(writeFileSyncSpy).toHaveBeenCalledWith(
        '/tmp/nanoclaw-rc-test/.remote-control-fixed/state.json',
        expect.stringContaining('"pid":99999'),
        { mode: 0o600, flag: 'wx' },
      );
    });

    it('returns existing URL if session is already active', async () => {
      const proc = createMockProcess();
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_existing\n';
      mockOwnedRetirement();

      await startRemoteControl('user1', 'tg:123', '/project');

      // Second call should return existing URL without spawning
      const result = await startRemoteControl('user2', 'tg:456', '/project');
      expect(result).toEqual({
        ok: true,
        url: 'https://claude.ai/code?bridge=env_existing',
      });
      expect(spawnMock).toHaveBeenCalledTimes(1);
    });

    it('does not return or signal a recalled PID after native instance identity changes', async () => {
      spawnMock.mockReturnValue(createMockProcess());
      stdoutFileContent = 'https://claude.ai/code?bridge=env_fixed\n';
      const kill = mockOwnedRetirement();
      await startRemoteControl('fixed', 'tg:123', '/project');
      kill.mockClear();
      vi.mocked(verifyRemoteControlIdentity).mockReturnValueOnce('changed');
      await expect(
        startRemoteControl('fixed', 'tg:123', '/project'),
      ).rejects.toThrow('unproved or changed');
      vi.mocked(verifyRemoteControlIdentity).mockReturnValueOnce('changed');
      await expect(stopRemoteControl()).rejects.toThrow(
        'Recalled process identity is unproved',
      );
      expect(kill.mock.calls.filter(([, signal]) => signal !== 0)).toEqual([]);
      expect(spawnMock).toHaveBeenCalledTimes(1);
    });

    it('starts new session if existing process is dead', async () => {
      const proc1 = createMockProcess(11111);
      const proc2 = createMockProcess(22222);
      spawnMock.mockReturnValueOnce(proc1).mockReturnValueOnce(proc2);

      // First start: process alive, URL found
      const killSpy = mockOwnedRetirement();
      stdoutFileContent = 'https://claude.ai/code?bridge=env_first\n';
      await startRemoteControl('user1', 'tg:123', '/project');

      // Old process (11111) is dead, new process (22222) is alive
      killSpy.mockImplementation((pid: number, sig?: string | number) => {
        if (Math.abs(pid) === 11111 && (sig === 0 || sig === undefined)) {
          throw Object.assign(new Error('ESRCH'), { code: 'ESRCH' });
        }
        return true;
      });

      stdoutFileContent = 'https://claude.ai/code?bridge=env_second\n';
      const result = await startRemoteControl('user1', 'tg:123', '/project');

      expect(result).toEqual({
        ok: true,
        url: 'https://claude.ai/code?bridge=env_second',
      });
      expect(spawnMock).toHaveBeenCalledTimes(2);
    });

    it('returns error if process exits before URL', async () => {
      const proc = createMockProcess(33333);
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = '';

      // Process is dead (poll will detect this)
      vi.spyOn(process, 'kill').mockImplementation(() => {
        throw Object.assign(new Error('ESRCH'), { code: 'ESRCH' });
      });

      const result = await startRemoteControl('user1', 'tg:123', '/project');
      expect(result).toEqual({
        ok: false,
        error: 'Process exited before producing URL',
      });
    });

    it('times out if URL never appears', async () => {
      vi.useFakeTimers();
      const proc = createMockProcess(44444);
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = 'no url here';
      mockOwnedRetirement();

      const promise = startRemoteControl('user1', 'tg:123', '/project');

      // Advance past URL_TIMEOUT_MS (30s), with enough steps for polls
      for (let i = 0; i < 160; i++) {
        await vi.advanceTimersByTimeAsync(200);
      }

      const result = await promise;
      expect(result).toEqual({
        ok: false,
        error: 'Timed out waiting for Remote Control URL',
      });

      expect(startingBytes).toBe('');
      stdoutFileContent = 'https://claude.ai/code?bridge=env_retry';
      spawnMock.mockReturnValue(createMockProcess(55555));
      await expect(
        startRemoteControl('user1', 'tg:123', '/project'),
      ).resolves.toEqual({ ok: true, url: stdoutFileContent });
      expect(spawnMock).toHaveBeenCalledTimes(2);
      vi.useRealTimers();
    });

    it('preserves an asynchronous child defect while retirement is already in progress', async () => {
      vi.useFakeTimers();
      const proc = createMockProcess(44444);
      spawnMock.mockReturnValue(proc);
      const defect = new TypeError('fixed child defect during retirement');
      let absent = false;
      vi.spyOn(process, 'kill').mockImplementation((_pid, signal) => {
        if (signal === 'SIGTERM') {
          absent = true;
          proc.stdin.emit('error', defect);
        }
        if (signal === 0 && absent)
          throw Object.assign(new Error('fixed absent'), { code: 'ESRCH' });
        return true;
      });
      const pending = startRemoteControl('owner', 'tg:123', '/project');
      const rejected = expect(pending).rejects.toBe(defect);
      await vi.advanceTimersByTimeAsync(30_200);
      await rejected;
      expect(startingBytes).toBe('');
      expect(vi.getTimerCount()).toBe(0);
      vi.useRealTimers();
    });

    it('retains the exclusive gate when successful signals do not prove native retirement', async () => {
      vi.useFakeTimers();
      spawnMock.mockReturnValue(createMockProcess(44444));
      vi.spyOn(process, 'kill').mockReturnValue(true);
      const pending = startRemoteControl('owner', 'tg:123', '/project');
      const rejected = expect(pending).rejects.toThrow(
        'retirement is unverified',
      );
      await vi.advanceTimersByTimeAsync(32_200);
      await rejected;
      expect(startingBytes).not.toBe('');
      await expect(
        startRemoteControl('owner', 'tg:123', '/project'),
      ).rejects.toThrow();
      expect(spawnMock).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(0);
      vi.useRealTimers();
    });

    it('rejects a delayed poll defect and terminates only its started process', async () => {
      vi.useFakeTimers();
      spawnMock.mockReturnValue(createMockProcess(12345));
      const kill = mockOwnedRetirement();
      const pending = startRemoteControl('user1', 'tg:123', '/project');
      await vi.advanceTimersByTimeAsync(0);
      const defect = new TypeError('fixed delayed reader defect');
      readFileSyncSpy.mockImplementation((file: fs.PathOrFileDescriptor) => {
        if (String(file).endsWith('remote-control-starting.json'))
          return startingBytes;
        throw defect;
      });
      const rejected = expect(pending).rejects.toBe(defect);
      await vi.advanceTimersByTimeAsync(200);
      await rejected;
      expect(kill).toHaveBeenCalledWith(-12345, 'SIGTERM');
      expect(getActiveSession()).toBeNull();
      expect(vi.getTimerCount()).toBe(0);
      vi.useRealTimers();
    });

    it('retains delayed poll and cleanup failures together', async () => {
      vi.useFakeTimers();
      spawnMock.mockReturnValue(createMockProcess(12345));
      const cleanup = Object.assign(new Error('fixed signal denied'), {
        code: 'EPERM',
      });
      vi.spyOn(process, 'kill').mockImplementation((_pid, signal) => {
        if (signal === 'SIGTERM') throw cleanup;
        return true;
      });
      const pending = startRemoteControl('user1', 'tg:123', '/project');
      await vi.advanceTimersByTimeAsync(0);
      const defect = new TypeError('fixed delayed reader defect');
      readFileSyncSpy.mockImplementation((file: fs.PathOrFileDescriptor) => {
        if (String(file).endsWith('remote-control-starting.json'))
          return startingBytes;
        throw defect;
      });
      const rejected = expect(pending).rejects.toMatchObject({
        cause: defect,
        errors: [defect, cleanup],
      });
      await vi.advanceTimersByTimeAsync(200);
      await rejected;
      expect(vi.getTimerCount()).toBe(0);
      vi.useRealTimers();
    });

    it.each(['group', 'fallback'] as const)(
      'rejects a delayed %s termination failure through the caller Promise',
      async (failureAt) => {
        vi.useFakeTimers();
        spawnMock.mockReturnValue(createMockProcess(12345));
        const denied = Object.assign(new Error('fixed signal denied'), {
          code: 'EPERM',
        });
        const missing = Object.assign(new Error('fixed missing group'), {
          code: 'ESRCH',
        });
        const kill = vi
          .spyOn(process, 'kill')
          .mockImplementation((pid, signal) => {
            if (signal === 'SIGTERM') {
              if (pid < 0 && failureAt === 'fallback') throw missing;
              throw denied;
            }
            return true;
          });
        const pending = startRemoteControl('user1', 'tg:123', '/project');
        const rejected = expect(pending).rejects.toBe(denied);
        await vi.advanceTimersByTimeAsync(30_001);
        await rejected;
        expect(
          kill.mock.calls.filter(([, signal]) => signal === 'SIGTERM'),
        ).toHaveLength(failureAt === 'group' ? 1 : 2);
        expect(vi.getTimerCount()).toBe(0);
        vi.useRealTimers();
      },
    );

    it.each(['child', 'stdin'] as const)(
      'rejects a delayed %s error and cleans up only its started process',
      async (emitter) => {
        vi.useFakeTimers();
        const proc = createMockProcess();
        spawnMock.mockReturnValue(proc);
        const kill = mockOwnedRetirement();
        const defect = new TypeError('fixed asynchronous child defect');
        const pending = startRemoteControl('user1', 'tg:123', '/project');
        const rejected = expect(pending).rejects.toBe(defect);
        await vi.advanceTimersByTimeAsync(0);
        (emitter === 'child' ? proc : proc.stdin).emit('error', defect);
        await vi.advanceTimersByTimeAsync(0);
        await rejected;
        expect(kill).toHaveBeenCalledWith(-12345, 'SIGTERM');
        expect(startingBytes).toBe('');
        expect(vi.getTimerCount()).toBe(0);
        vi.useRealTimers();
      },
    );

    it('does not mutate a successful session after a late stdin error', async () => {
      const proc = createMockProcess();
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_fixed';
      const kill = mockOwnedRetirement();
      await expect(
        startRemoteControl('user1', 'tg:123', '/project'),
      ).resolves.toMatchObject({ ok: true });
      const session = getActiveSession();
      kill.mockClear();
      const writes = writeFileSyncSpy.mock.calls.length;
      unlinkSyncSpy.mockClear();
      proc.stdin.emit(
        'error',
        Object.assign(new Error('fixed late closed pipe'), { code: 'EPIPE' }),
      );
      await Promise.resolve();
      expect(getActiveSession()).toBe(session);
      expect(kill.mock.calls.filter(([, signal]) => signal !== 0)).toEqual([]);
      expect(writeFileSyncSpy).toHaveBeenCalledTimes(writes);
      expect(unlinkSyncSpy).not.toHaveBeenCalled();
    });

    it('propagates an unexpected spawn error while closing parent descriptors', async () => {
      const defect = new TypeError('fixed spawn defect');
      spawnMock.mockImplementation(() => {
        throw defect;
      });
      await expect(
        startRemoteControl('user1', 'tg:123', '/project'),
      ).rejects.toBe(defect);
      expect(closeSyncSpy).toHaveBeenCalledTimes(6);
      expect(getActiveSession()).toBeNull();
    });

    it('does not interpret denied process inspection as a dead session', async () => {
      spawnMock.mockReturnValue(createMockProcess());
      stdoutFileContent = 'https://claude.ai/code?bridge=env_fixed\n';
      const kill = mockOwnedRetirement();
      await startRemoteControl('user1', 'tg:123', '/project');
      const denied = Object.assign(new Error('fixed denied'), {
        code: 'EPERM',
      });
      unlinkSyncSpy.mockClear();
      kill.mockImplementation(() => {
        throw denied;
      });
      await expect(stopRemoteControl()).rejects.toBe(denied);
      expect(() => getActiveSession()).toThrow(denied);
      kill.mockImplementation(() => true);
      expect(getActiveSession()?.pid).toBe(12345);
      expect(unlinkSyncSpy).not.toHaveBeenCalled();
    });

    it.each([
      Buffer.concat([
        Buffer.from('https://claude.ai/code?bridge=env_'),
        Buffer.from([0xff]),
        Buffer.from('\n'),
      ]),
      Buffer.alloc(65537, 0x61),
      Buffer.from('https://claude.ai/codeevil?bridge=env_fixed\n'),
    ])(
      'retires its owned child rather than tracking malformed or over-budget stdout',
      async (bytes) => {
        spawnMock.mockReturnValue(createMockProcess());
        const kill = mockOwnedRetirement();
        stdoutFileContent = bytes;
        await expect(
          startRemoteControl('fixed', 'tg:123', '/project'),
        ).resolves.toMatchObject({ ok: false });
        expect(
          kill.mock.calls.filter(([, signal]) => signal === 'SIGTERM'),
        ).toEqual([[-12345, 'SIGTERM']]);
        expect(getActiveSession()).toBeNull();
        expect(
          writeFileSyncSpy.mock.calls.some(
            ([file]: unknown[]) =>
              typeof file === 'string' && file.endsWith('state.json'),
          ),
        ).toBe(false);
      },
    );

    it('waits for a complete native URL line instead of tracking a partial URL snapshot', async () => {
      vi.useFakeTimers();
      try {
        spawnMock.mockReturnValue(createMockProcess());
        mockOwnedRetirement();
        stdoutFileContent = Buffer.from(
          'https://claude.ai/code?bridge=env_part',
        );
        const result = startRemoteControl('fixed', 'tg:123', '/project');
        await vi.advanceTimersByTimeAsync(200);
        expect(getActiveSession()).toBeNull();
        stdoutFileContent = Buffer.from(
          'https://claude.ai/code?bridge=env_partial_complete\n',
        );
        await vi.advanceTimersByTimeAsync(200);
        await expect(result).resolves.toEqual({
          ok: true,
          url: 'https://claude.ai/code?bridge=env_partial_complete',
        });
      } finally {
        vi.useRealTimers();
      }
    });

    it('returns error if spawn throws', async () => {
      spawnMock.mockImplementation(() => {
        throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
      });

      const result = await startRemoteControl('user1', 'tg:123', '/project');
      expect(result).toEqual({
        ok: false,
        error: 'Failed to start: ENOENT',
      });
    });
  });

  // --- stopRemoteControl ---

  describe('stopRemoteControl', () => {
    it('kills the process and clears state', async () => {
      const proc = createMockProcess(55555);
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_stop\n';
      const killSpy = mockOwnedRetirement();

      await startRemoteControl('user1', 'tg:123', '/project');

      const result = await stopRemoteControl();
      expect(result).toEqual({ ok: true });
      expect(killSpy).toHaveBeenCalledWith(-55555, 'SIGTERM');
      expect(unlinkSyncSpy).toHaveBeenCalledWith(STATE_FILE);
      expect(getActiveSession()).toBeNull();
    });

    it('waits for actual absence and serializes stop/start while a native child is retiring', async () => {
      vi.useFakeTimers();
      spawnMock.mockReturnValue(createMockProcess(55555));
      stdoutFileContent = 'https://claude.ai/code?bridge=env_stopping';
      let absent = false;
      vi.spyOn(process, 'kill').mockImplementation((_pid, signal) => {
        if (signal === 0 && absent)
          throw Object.assign(new Error('fixed absent'), { code: 'ESRCH' });
        return true;
      });
      await startRemoteControl('owner', 'tg:123', '/project');
      unlinkSyncSpy.mockClear();
      const first = stopRemoteControl();
      const second = stopRemoteControl();
      expect(second).toBe(first);
      await vi.advanceTimersByTimeAsync(0);
      expect(getActiveSession()?.pid).toBe(55555);
      expect(unlinkSyncSpy).not.toHaveBeenCalledWith(STATE_FILE);
      await expect(
        startRemoteControl('owner', 'tg:123', '/project'),
      ).resolves.toMatchObject({
        ok: false,
        error: expect.stringContaining('stopping'),
      });
      expect(spawnMock).toHaveBeenCalledTimes(1);
      absent = true;
      await vi.advanceTimersByTimeAsync(100);
      await expect(first).resolves.toEqual({ ok: true });
      expect(getActiveSession()).toBeNull();
      expect(unlinkSyncSpy).toHaveBeenCalledWith(STATE_FILE);
      vi.useRealTimers();
    });

    it('returns error when no session is active', async () => {
      const result = await stopRemoteControl();
      expect(result).toEqual({
        ok: false,
        error: 'No active Remote Control session',
      });
    });
  });

  // --- restoreRemoteControl ---

  describe('restoreRemoteControl', () => {
    it('retains a legacy session as unproved without adopting or signaling its PID', () => {
      const session = {
        pid: 77777,
        url: 'https://claude.ai/code?bridge=env_fixed',
        startedBy: 'user1',
        startedInChat: 'tg:123',
        startedAt: '2024-01-01T00:00:00Z',
      };
      mockState(JSON.stringify(session));
      mockOwnedRetirement();
      expect(restoreRemoteControl).toThrow(
        'Legacy Remote Control identity is unproved',
      );
      expect(writeFileSyncSpy).not.toHaveBeenCalled();
      expect(getActiveSession()).toBeNull();
    });

    it('reads a current session without rewriting it', () => {
      const session = {
        pid: 77777,
        url: 'https://claude.ai/code?bridge=env_fixed',
        startedBy: 'user1',
        startedInChat: 'tg:123',
        startedAt: '2024-01-01T00:00:00Z',
      };
      readFileSyncSpy.mockReturnValue(
        JSON.stringify({
          schema_version: 2,
          identity: { ...identityFixture },
          ...session,
        }),
      );
      mockOwnedRetirement();
      restoreRemoteControl();
      expect(writeFileSyncSpy).not.toHaveBeenCalled();
      expect(getActiveSession()).toEqual({
        ...session,
        identity: { ...identityFixture },
      });
    });

    it('preserves and refuses an unsupported state version without process effects', () => {
      const raw = JSON.stringify({
        schema_version: 99,
        pid: 77777,
        url: 'https://claude.ai/code?bridge=env_fixed',
        startedBy: 'user1',
        startedInChat: 'tg:123',
        startedAt: '2024-01-01T00:00:00Z',
      });
      mockState(raw);
      const kill = vi.spyOn(process, 'kill');
      expect(restoreRemoteControl).toThrow(
        'Unsupported Remote Control state version',
      );
      expect(kill).not.toHaveBeenCalled();
      expect(unlinkSyncSpy).not.toHaveBeenCalled();
      expect(writeFileSyncSpy).not.toHaveBeenCalled();
      expect(getActiveSession()).toBeNull();
    });

    it('restores session if state file exists and process is alive', () => {
      const session = {
        schema_version: 2,
        identity: { ...identityFixture },
        pid: 77777,
        url: 'https://claude.ai/code?bridge=env_restored',
        startedBy: 'user1',
        startedInChat: 'tg:123',
        startedAt: '2026-01-01T00:00:00.000Z',
      };
      readFileSyncSpy.mockImplementation((p: string) => {
        if (p.endsWith('remote-control.json')) return JSON.stringify(session);
        return '';
      });
      mockOwnedRetirement();

      restoreRemoteControl();

      const active = getActiveSession();
      expect(active).not.toBeNull();
      expect(active!.pid).toBe(77777);
      expect(active!.url).toBe('https://claude.ai/code?bridge=env_restored');
    });

    it('clears state if process is dead', () => {
      const session = {
        schema_version: 2,
        identity: { ...identityFixture },
        pid: 88888,
        url: 'https://claude.ai/code?bridge=env_dead',
        startedBy: 'user1',
        startedInChat: 'tg:123',
        startedAt: '2026-01-01T00:00:00.000Z',
      };
      readFileSyncSpy.mockImplementation((p: string) => {
        if (p.endsWith('remote-control.json')) return JSON.stringify(session);
        return '';
      });
      vi.spyOn(process, 'kill').mockImplementation(() => {
        throw Object.assign(new Error('ESRCH'), { code: 'ESRCH' });
      });

      restoreRemoteControl();

      expect(getActiveSession()).toBeNull();
      expect(unlinkSyncSpy).toHaveBeenCalled();
    });

    it('does nothing if no state file exists', () => {
      // readFileSyncSpy default throws ENOENT for .json
      restoreRemoteControl();
      expect(getActiveSession()).toBeNull();
    });

    it.each(['null', '[]', '{"pid":"123"}'])(
      'preserves and refuses malformed persisted session %s without process operations',
      (raw) => {
        mockState(raw);
        const kill = vi.spyOn(process, 'kill');
        expect(restoreRemoteControl).toThrow(
          /occupied-session state is malformed|Legacy Remote Control identity is unproved/,
        );
        expect(getActiveSession()).toBeNull();
        expect(kill).not.toHaveBeenCalled();
        expect(unlinkSyncSpy).not.toHaveBeenCalled();
      },
    );

    it('preserves and refuses corrupted occupied-session JSON', () => {
      readFileSyncSpy.mockImplementation((p: string) => {
        if (p.endsWith('remote-control.json')) return 'not json{{{';
        return '';
      });

      expect(restoreRemoteControl).toThrow(
        'occupied-session state is malformed',
      );

      expect(getActiveSession()).toBeNull();
      expect(unlinkSyncSpy).not.toHaveBeenCalled();
    });

    // ** This is the key integration test: restore → stop must work **
    it('stopRemoteControl works after restoreRemoteControl', async () => {
      const session = {
        schema_version: 2,
        identity: { ...identityFixture },
        pid: 77777,
        url: 'https://claude.ai/code?bridge=env_restored',
        startedBy: 'user1',
        startedInChat: 'tg:123',
        startedAt: '2026-01-01T00:00:00.000Z',
      };
      readFileSyncSpy.mockImplementation((p: string) => {
        if (p.endsWith('remote-control.json')) return JSON.stringify(session);
        return '';
      });
      const killSpy = mockOwnedRetirement();

      restoreRemoteControl();
      expect(getActiveSession()).not.toBeNull();

      const result = await stopRemoteControl();
      expect(result).toEqual({ ok: true });
      expect(killSpy).toHaveBeenCalledWith(-77777, 'SIGTERM');
      expect(unlinkSyncSpy).toHaveBeenCalled();
      expect(getActiveSession()).toBeNull();
    });

    it('startRemoteControl returns restored URL without spawning', () => {
      const session = {
        schema_version: 2,
        identity: { ...identityFixture },
        pid: 77777,
        url: 'https://claude.ai/code?bridge=env_restored',
        startedBy: 'user1',
        startedInChat: 'tg:123',
        startedAt: '2026-01-01T00:00:00.000Z',
      };
      readFileSyncSpy.mockImplementation((p: string) => {
        if (p.endsWith('remote-control.json')) return JSON.stringify(session);
        return '';
      });
      mockOwnedRetirement();

      restoreRemoteControl();

      return startRemoteControl('user2', 'tg:456', '/project').then(
        (result) => {
          expect(result).toEqual({
            ok: true,
            url: 'https://claude.ai/code?bridge=env_restored',
          });
          expect(spawnMock).not.toHaveBeenCalled();
        },
      );
    });
  });
});
