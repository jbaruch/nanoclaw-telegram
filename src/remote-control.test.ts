import fs from 'fs';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock config before importing the module under test
vi.mock('./config.js', () => ({
  DATA_DIR: '/tmp/nanoclaw-rc-test',
}));

// Mock child_process
const spawnMock = vi.fn();
vi.mock('child_process', () => ({
  spawn: (...args: unknown[]) => spawnMock(...args),
}));

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
  return {
    pid,
    unref: vi.fn(),
    kill: vi.fn(),
    stdin: { write: vi.fn(), end: vi.fn() },
  };
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
  let stdoutFileContent: string;

  beforeEach(() => {
    _resetForTesting();
    spawnMock.mockReset();
    stdoutFileContent = '';

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
      .mockImplementation(() => {});
    unlinkSyncSpy = vi.spyOn(fs, 'unlinkSync').mockImplementation(() => {});
    openSyncSpy = vi.spyOn(fs, 'openSync').mockReturnValue(42);
    closeSyncSpy = vi.spyOn(fs, 'closeSync').mockImplementation(() => {});

    // readFileSync: return stdoutFileContent for the stdout file, state file, etc.
    readFileSyncSpy = vi.spyOn(fs, 'readFileSync').mockImplementation(((
      p: string,
    ) => {
      if (p.endsWith('remote-control.stdout')) return stdoutFileContent;
      if (p.endsWith('remote-control.json')) {
        throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
      }
      return '';
    }) as typeof fs.readFileSync);
  });

  afterEach(() => {
    _resetForTesting();
    vi.restoreAllMocks();
  });

  // --- startRemoteControl ---

  describe('startRemoteControl', () => {
    it('spawns claude remote-control and returns the URL', async () => {
      const proc = createMockProcess();
      spawnMock.mockReturnValue(proc);

      // Simulate URL appearing in stdout file on first poll
      stdoutFileContent =
        'Session URL: https://claude.ai/code?bridge=env_abc123\n';
      vi.spyOn(process, 'kill').mockImplementation(() => true);

      const result = await startRemoteControl('user1', 'tg:123', '/project');

      expect(result).toEqual({
        ok: true,
        url: 'https://claude.ai/code?bridge=env_abc123',
      });
      expect(spawnMock).toHaveBeenCalledWith(
        'claude',
        ['remote-control', '--name', 'NanoClaw Remote'],
        expect.objectContaining({ cwd: '/project', detached: true }),
      );
      expect(proc.unref).toHaveBeenCalled();
    });

    it('uses file descriptors for stdout/stderr (not pipes)', async () => {
      const proc = createMockProcess();
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_test\n';
      vi.spyOn(process, 'kill').mockImplementation(() => true);

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
      vi.spyOn(process, 'kill').mockImplementation(() => true);

      await startRemoteControl('user1', 'tg:123', '/project');

      // Two openSync calls (stdout + stderr), two closeSync calls
      expect(openSyncSpy).toHaveBeenCalledTimes(2);
      expect(closeSyncSpy).toHaveBeenCalledTimes(2);
    });

    it('saves state to disk after capturing URL', async () => {
      const proc = createMockProcess(99999);
      spawnMock.mockReturnValue(proc);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_save\n';
      vi.spyOn(process, 'kill').mockImplementation(() => true);

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
      vi.spyOn(process, 'kill').mockImplementation(() => true);

      await startRemoteControl('user1', 'tg:123', '/project');

      // Second call should return existing URL without spawning
      const result = await startRemoteControl('user2', 'tg:456', '/project');
      expect(result).toEqual({
        ok: true,
        url: 'https://claude.ai/code?bridge=env_existing',
      });
      expect(spawnMock).toHaveBeenCalledTimes(1);
    });

    it('starts new session if existing process is dead', async () => {
      const proc1 = createMockProcess(11111);
      const proc2 = createMockProcess(22222);
      spawnMock.mockReturnValueOnce(proc1).mockReturnValueOnce(proc2);

      // First start: process alive, URL found
      const killSpy = vi.spyOn(process, 'kill').mockImplementation(() => true);
      stdoutFileContent = 'https://claude.ai/code?bridge=env_first\n';
      await startRemoteControl('user1', 'tg:123', '/project');

      // Old process (11111) is dead, new process (22222) is alive
      killSpy.mockImplementation((pid: number, sig?: string | number) => {
        if (pid === 11111 && (sig === 0 || sig === undefined)) {
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
      vi.spyOn(process, 'kill').mockImplementation(() => true);

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

      vi.useRealTimers();
    });

    it('rejects a delayed poll defect and terminates only its started process', async () => {
      vi.useFakeTimers();
      spawnMock.mockReturnValue(createMockProcess(12345));
      const kill = vi.spyOn(process, 'kill').mockReturnValue(true);
      const pending = startRemoteControl('user1', 'tg:123', '/project');
      await vi.advanceTimersByTimeAsync(0);
      const defect = new TypeError('fixed delayed reader defect');
      readFileSyncSpy.mockImplementation(() => {
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
      readFileSyncSpy.mockImplementation(() => {
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

    it('propagates an unexpected spawn error while closing parent descriptors', async () => {
      const defect = new TypeError('fixed spawn defect');
      spawnMock.mockImplementation(() => {
        throw defect;
      });
      await expect(
        startRemoteControl('user1', 'tg:123', '/project'),
      ).rejects.toBe(defect);
      expect(closeSyncSpy).toHaveBeenCalledTimes(2);
      expect(getActiveSession()).toBeNull();
    });

    it('does not interpret denied process inspection as a dead session', async () => {
      spawnMock.mockReturnValue(createMockProcess());
      stdoutFileContent = 'https://claude.ai/code?bridge=env_fixed\n';
      const kill = vi.spyOn(process, 'kill').mockReturnValue(true);
      await startRemoteControl('user1', 'tg:123', '/project');
      const denied = Object.assign(new Error('fixed denied'), {
        code: 'EPERM',
      });
      kill.mockImplementation(() => {
        throw denied;
      });
      expect(stopRemoteControl).toThrow(denied);
      expect(getActiveSession()?.pid).toBe(12345);
      expect(unlinkSyncSpy).not.toHaveBeenCalled();
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
      const killSpy = vi.spyOn(process, 'kill').mockImplementation(() => true);

      await startRemoteControl('user1', 'tg:123', '/project');

      const result = stopRemoteControl();
      expect(result).toEqual({ ok: true });
      expect(killSpy).toHaveBeenCalledWith(55555, 'SIGTERM');
      expect(unlinkSyncSpy).toHaveBeenCalledWith(STATE_FILE);
      expect(getActiveSession()).toBeNull();
    });

    it('returns error when no session is active', () => {
      const result = stopRemoteControl();
      expect(result).toEqual({
        ok: false,
        error: 'No active Remote Control session',
      });
    });
  });

  // --- restoreRemoteControl ---

  describe('restoreRemoteControl', () => {
    it('rewrites a verified legacy session to schema 1 through its owner', () => {
      const session = {
        pid: 77777,
        url: 'https://claude.ai/code?bridge=env_fixed',
        startedBy: 'user1',
        startedInChat: 'tg:123',
        startedAt: '2024-01-01T00:00:00Z',
      };
      readFileSyncSpy.mockReturnValue(JSON.stringify(session));
      vi.spyOn(process, 'kill').mockReturnValue(true);
      restoreRemoteControl();
      expect(writeFileSyncSpy).toHaveBeenCalledWith(
        '/tmp/nanoclaw-rc-test/.remote-control-fixed/state.json',
        JSON.stringify({ schema_version: 1, ...session }),
        { mode: 0o600, flag: 'wx' },
      );
      expect(getActiveSession()).toEqual(session);
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
        JSON.stringify({ schema_version: 1, ...session }),
      );
      vi.spyOn(process, 'kill').mockReturnValue(true);
      restoreRemoteControl();
      expect(writeFileSyncSpy).not.toHaveBeenCalled();
      expect(getActiveSession()).toEqual(session);
    });

    it('preserves and refuses an unsupported state version without process effects', () => {
      const raw = JSON.stringify({
        schema_version: 2,
        pid: 77777,
        url: 'https://claude.ai/code?bridge=env_fixed',
        startedBy: 'user1',
        startedInChat: 'tg:123',
        startedAt: '2024-01-01T00:00:00Z',
      });
      readFileSyncSpy.mockReturnValue(raw);
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
      vi.spyOn(process, 'kill').mockImplementation(() => true);

      restoreRemoteControl();

      const active = getActiveSession();
      expect(active).not.toBeNull();
      expect(active!.pid).toBe(77777);
      expect(active!.url).toBe('https://claude.ai/code?bridge=env_restored');
    });

    it('clears state if process is dead', () => {
      const session = {
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
      'clears malformed persisted session %s without process operations',
      (raw) => {
        readFileSyncSpy.mockReturnValue(raw);
        const kill = vi.spyOn(process, 'kill');
        restoreRemoteControl();
        expect(getActiveSession()).toBeNull();
        expect(kill).not.toHaveBeenCalled();
        expect(unlinkSyncSpy).toHaveBeenCalledWith(STATE_FILE);
      },
    );

    it('clears state on corrupted JSON', () => {
      readFileSyncSpy.mockImplementation((p: string) => {
        if (p.endsWith('remote-control.json')) return 'not json{{{';
        return '';
      });

      restoreRemoteControl();

      expect(getActiveSession()).toBeNull();
      expect(unlinkSyncSpy).toHaveBeenCalled();
    });

    // ** This is the key integration test: restore → stop must work **
    it('stopRemoteControl works after restoreRemoteControl', () => {
      const session = {
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
      const killSpy = vi.spyOn(process, 'kill').mockImplementation(() => true);

      restoreRemoteControl();
      expect(getActiveSession()).not.toBeNull();

      const result = stopRemoteControl();
      expect(result).toEqual({ ok: true });
      expect(killSpy).toHaveBeenCalledWith(77777, 'SIGTERM');
      expect(unlinkSyncSpy).toHaveBeenCalled();
      expect(getActiveSession()).toBeNull();
    });

    it('startRemoteControl returns restored URL without spawning', () => {
      const session = {
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
      vi.spyOn(process, 'kill').mockImplementation(() => true);

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
