import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock logger
vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Mock child_process
const mockExecSync = vi.fn();
const mockSpawnSync = vi.fn();
vi.mock('child_process', () => ({
  execSync: (...args: unknown[]) => mockExecSync(...args),
  spawnSync: (...args: unknown[]) => mockSpawnSync(...args),
}));

import {
  CONTAINER_RUNTIME_BIN,
  readonlyMountArgs,
  stopContainer,
  ensureContainerRuntimeRunning,
  cleanupOrphans,
} from './container-runtime.js';
import { logger } from './logger.js';

beforeEach(() => {
  vi.clearAllMocks();
  // Default spawnSync to return empty stdout
  mockSpawnSync.mockReturnValue({ stdout: '', stderr: '', status: 0 });
});

// --- Pure functions ---

describe('readonlyMountArgs', () => {
  it('returns -v flag with :ro suffix', () => {
    const args = readonlyMountArgs('/host/path', '/container/path');
    expect(args).toEqual(['-v', '/host/path:/container/path:ro']);
  });
});

describe('stopContainer', () => {
  it('calls docker stop via spawnSync for valid container names', () => {
    stopContainer('nanoclaw-test-123');
    expect(mockSpawnSync).toHaveBeenCalledWith(
      CONTAINER_RUNTIME_BIN,
      ['stop', '-t', '1', 'nanoclaw-test-123'],
      { stdio: 'pipe', timeout: 10_000 },
    );
  });

  it('accepts names with dots and underscores', () => {
    stopContainer('nanoclaw-my_group.test-123');
    expect(mockSpawnSync).toHaveBeenCalledTimes(1);
  });

  it('falls back to docker kill when docker stop fails', () => {
    mockSpawnSync.mockReturnValueOnce({ stdout: '', stderr: '', status: 1 });
    mockSpawnSync.mockReturnValueOnce({ stdout: '', stderr: '', status: 0 });

    stopContainer('nanoclaw-test-123');

    expect(mockSpawnSync).toHaveBeenCalledTimes(2);
    expect(mockSpawnSync).toHaveBeenNthCalledWith(
      1,
      CONTAINER_RUNTIME_BIN,
      ['stop', '-t', '1', 'nanoclaw-test-123'],
      { stdio: 'pipe', timeout: 10_000 },
    );
    expect(mockSpawnSync).toHaveBeenNthCalledWith(
      2,
      CONTAINER_RUNTIME_BIN,
      ['kill', 'nanoclaw-test-123'],
      { stdio: 'pipe', timeout: 5_000 },
    );
  });

  it('falls back to docker kill when docker stop times out (status null)', () => {
    mockSpawnSync.mockReturnValueOnce({ stdout: '', stderr: '', status: null });
    mockSpawnSync.mockReturnValueOnce({ stdout: '', stderr: '', status: 0 });

    stopContainer('nanoclaw-test-123');

    expect(mockSpawnSync).toHaveBeenCalledTimes(2);
    expect(mockSpawnSync).toHaveBeenNthCalledWith(
      1,
      CONTAINER_RUNTIME_BIN,
      ['stop', '-t', '1', 'nanoclaw-test-123'],
      { stdio: 'pipe', timeout: 10_000 },
    );
    expect(mockSpawnSync).toHaveBeenNthCalledWith(
      2,
      CONTAINER_RUNTIME_BIN,
      ['kill', 'nanoclaw-test-123'],
      { stdio: 'pipe', timeout: 5_000 },
    );
  });
});

// --- ensureContainerRuntimeRunning ---

describe('ensureContainerRuntimeRunning', () => {
  it('does nothing when runtime is already running', () => {
    mockExecSync.mockReturnValueOnce('');

    ensureContainerRuntimeRunning();

    expect(mockExecSync).toHaveBeenCalledTimes(1);
    expect(mockExecSync).toHaveBeenCalledWith(`${CONTAINER_RUNTIME_BIN} info`, {
      stdio: 'pipe',
      timeout: 10000,
    });
    expect(logger.debug).toHaveBeenCalledWith(
      'Container runtime already running',
    );
  });

  it('throws when docker info fails', () => {
    mockExecSync.mockImplementation(() => {
      throw new Error('not running');
    });

    expect(() => ensureContainerRuntimeRunning()).toThrow(
      'Container runtime is required but failed to start',
    );
    expect(logger.error).toHaveBeenCalled();
  });
});

// --- cleanupOrphans ---

describe('cleanupOrphans', () => {
  it('stops orphaned nanoclaw containers', () => {
    mockSpawnSync.mockReturnValueOnce({
      stdout: 'nanoclaw-group1-111\nnanoclaw-group2-222\nother-container\n',
      stderr: '',
      status: 0,
    });

    cleanupOrphans();

    // ps + 2 stop calls
    expect(mockSpawnSync).toHaveBeenCalledTimes(3);
    expect(mockSpawnSync).toHaveBeenNthCalledWith(
      2,
      CONTAINER_RUNTIME_BIN,
      ['stop', '-t', '1', 'nanoclaw-group1-111'],
      { stdio: 'pipe', timeout: 10_000 },
    );
    expect(mockSpawnSync).toHaveBeenNthCalledWith(
      3,
      CONTAINER_RUNTIME_BIN,
      ['stop', '-t', '1', 'nanoclaw-group2-222'],
      { stdio: 'pipe', timeout: 10_000 },
    );
    expect(logger.info).toHaveBeenCalledWith(
      { count: 2, names: ['nanoclaw-group1-111', 'nanoclaw-group2-222'] },
      'Stopped orphaned containers',
    );
  });

  it('does nothing when no orphans exist', () => {
    mockSpawnSync.mockReturnValueOnce({ stdout: '\n', stderr: '', status: 0 });

    cleanupOrphans();

    expect(mockSpawnSync).toHaveBeenCalledTimes(1);
    expect(logger.info).not.toHaveBeenCalled();
  });

  it('never kills the litellm gateway despite the nanoclaw- prefix', () => {
    // The UGOS compose project names the gateway container
    // `nanoclaw-litellm-nanoclaw-litellm-<idx>`; it shares the
    // `nanoclaw-` prefix but is persistent infra, not an agent orphan.
    mockSpawnSync.mockReturnValueOnce({
      stdout: 'nanoclaw-group1-111\nnanoclaw-litellm-nanoclaw-litellm-1\n',
      stderr: '',
      status: 0,
    });

    cleanupOrphans();

    // ps + exactly one stop (the agent container only).
    expect(mockSpawnSync).toHaveBeenCalledTimes(2);
    expect(mockSpawnSync).toHaveBeenNthCalledWith(
      2,
      CONTAINER_RUNTIME_BIN,
      ['stop', '-t', '1', 'nanoclaw-group1-111'],
      { stdio: 'pipe', timeout: 10_000 },
    );
    expect(logger.info).toHaveBeenCalledWith(
      { count: 1, names: ['nanoclaw-group1-111'] },
      'Stopped orphaned containers',
    );
  });

  it('excludes the litellm gateway even when it is the only running container', () => {
    mockSpawnSync.mockReturnValueOnce({
      stdout: 'nanoclaw-litellm-nanoclaw-litellm-1\n',
      stderr: '',
      status: 0,
    });

    cleanupOrphans();

    // ps only — no stop call, nothing logged as orphaned.
    expect(mockSpawnSync).toHaveBeenCalledTimes(1);
    expect(logger.info).not.toHaveBeenCalled();
  });

  it('still kills a per-group agent whose slug starts with litellm', () => {
    // The exclusion matches only the gateway's full compose name
    // (`nanoclaw-litellm-nanoclaw-litellm-<idx>`), NOT every `litellm`-ish
    // name. A real agent container for a group slugged `litellm-fans` —
    // `nanoclaw-litellm-fans-<ts>` — must still be reaped as an orphan.
    mockSpawnSync.mockReturnValueOnce({
      stdout:
        'nanoclaw-litellm-fans-1779186257183\nnanoclaw-litellm-nanoclaw-litellm-1\n',
      stderr: '',
      status: 0,
    });

    cleanupOrphans();

    // ps + exactly one stop (the agent, not the gateway).
    expect(mockSpawnSync).toHaveBeenCalledTimes(2);
    expect(mockSpawnSync).toHaveBeenNthCalledWith(
      2,
      CONTAINER_RUNTIME_BIN,
      ['stop', '-t', '1', 'nanoclaw-litellm-fans-1779186257183'],
      { stdio: 'pipe', timeout: 10_000 },
    );
    expect(logger.info).toHaveBeenCalledWith(
      { count: 1, names: ['nanoclaw-litellm-fans-1779186257183'] },
      'Stopped orphaned containers',
    );
  });

  it('warns and continues when ps fails', () => {
    mockSpawnSync.mockImplementationOnce(() => {
      throw new Error('docker not available');
    });

    cleanupOrphans(); // should not throw

    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ err: expect.any(Error) }),
      'Failed to clean up orphaned containers',
    );
  });

  it('continues stopping remaining containers when one stop fails', () => {
    mockSpawnSync.mockReturnValueOnce({
      stdout: 'nanoclaw-a-1\nnanoclaw-b-2\n',
      stderr: '',
      status: 0,
    });
    // First stop fails
    mockSpawnSync.mockImplementationOnce(() => {
      throw new Error('already stopped');
    });
    // Second stop succeeds
    mockSpawnSync.mockReturnValueOnce({ stdout: '', stderr: '', status: 0 });

    cleanupOrphans(); // should not throw

    expect(mockSpawnSync).toHaveBeenCalledTimes(3);
    expect(logger.info).toHaveBeenCalledWith(
      { count: 2, names: ['nanoclaw-a-1', 'nanoclaw-b-2'] },
      'Stopped orphaned containers',
    );
  });

  // --- skipNames (handoff) — #213 ---
  //
  // The new orchestrator passes a Set of names from the prior run's
  // graceful-shutdown marker. Containers in that set are intentional
  // handoffs and must NOT be killed; everything else is a real
  // crash-orphan and gets stopped as before. The pre-#213 default
  // (no skip set) is preserved — the absent / stale / corrupt marker
  // path must fall through to "kill all" so genuine crash recovery
  // still works.

  it('skips killing containers in the handoff skip-set, kills the rest', () => {
    mockSpawnSync.mockReturnValueOnce({
      stdout: 'nanoclaw-adopted-1\nnanoclaw-orphan-2\nnanoclaw-adopted-3\n',
      stderr: '',
      status: 0,
    });
    const skip = new Set(['nanoclaw-adopted-1', 'nanoclaw-adopted-3']);

    cleanupOrphans(skip);

    // ps + ONE stop call (only the orphan).
    expect(mockSpawnSync).toHaveBeenCalledTimes(2);
    expect(mockSpawnSync).toHaveBeenNthCalledWith(
      2,
      CONTAINER_RUNTIME_BIN,
      ['stop', '-t', '1', 'nanoclaw-orphan-2'],
      { stdio: 'pipe', timeout: 10_000 },
    );
    // Both log lines fire — adopted info + orphan info.
    expect(logger.info).toHaveBeenCalledWith(
      { count: 2, names: ['nanoclaw-adopted-1', 'nanoclaw-adopted-3'] },
      'Adopted detached containers from graceful shutdown (not killed)',
    );
    expect(logger.info).toHaveBeenCalledWith(
      { count: 1, names: ['nanoclaw-orphan-2'] },
      'Stopped orphaned containers',
    );
  });

  it('logs adopted-only when every running container is in the skip set', () => {
    mockSpawnSync.mockReturnValueOnce({
      stdout: 'nanoclaw-a\nnanoclaw-b\n',
      stderr: '',
      status: 0,
    });
    const skip = new Set(['nanoclaw-a', 'nanoclaw-b']);

    cleanupOrphans(skip);

    // ps only — nothing to stop.
    expect(mockSpawnSync).toHaveBeenCalledTimes(1);
    expect(logger.info).toHaveBeenCalledWith(
      { count: 2, names: ['nanoclaw-a', 'nanoclaw-b'] },
      'Adopted detached containers from graceful shutdown (not killed)',
    );
    // Critically, the "Stopped orphaned containers" line MUST NOT
    // fire when nothing was stopped — emitting it with count: 0
    // would muddle log analysis tooling that counts kill events.
    expect(logger.info).not.toHaveBeenCalledWith(
      expect.objectContaining({}),
      'Stopped orphaned containers',
    );
  });

  it('falls through to pre-#213 "kill all" when skipNames is undefined', () => {
    // No marker found → orchestrator passes `undefined` → behave
    // exactly as before. This is the crash-recovery safety net: a
    // SIGKILL'd or hung prior orchestrator never wrote a marker, so
    // every nanoclaw-* container it spawned is genuinely abandoned
    // and must be cleaned up.
    mockSpawnSync.mockReturnValueOnce({
      stdout: 'nanoclaw-a\nnanoclaw-b\n',
      stderr: '',
      status: 0,
    });

    cleanupOrphans(undefined);

    expect(mockSpawnSync).toHaveBeenCalledTimes(3);
    expect(logger.info).toHaveBeenCalledWith(
      { count: 2, names: ['nanoclaw-a', 'nanoclaw-b'] },
      'Stopped orphaned containers',
    );
    expect(logger.info).not.toHaveBeenCalledWith(
      expect.objectContaining({}),
      'Adopted detached containers from graceful shutdown (not killed)',
    );
  });
});
