import fs from 'fs';
import {
  ContainerExitError,
  ContainerOutputProtocolError,
} from './operational-errors.js';
import { createDraftStream } from './draft-stream.js';
import { logger } from './logger.js';
import { stopContainer } from './container-runtime.js';
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { EventEmitter } from 'events';
import { PassThrough } from 'stream';

// Sentinel markers must match container-runner.ts
const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

// Mock config
vi.mock('./config.js', () => ({
  CONTAINER_IMAGE: 'nanoclaw-agent:latest',
  CONTAINER_MAX_OUTPUT_SIZE: 10485760,
  CONTAINER_TIMEOUT: 1800000, // 30min
  DATA_DIR: '/tmp/nanoclaw-test-data',
  GROUPS_DIR: '/tmp/nanoclaw-test-groups',
  IDLE_TIMEOUT: 1800000, // 30min
  ONECLI_URL: 'http://localhost:10254',
  TIMEZONE: 'America/Los_Angeles',
}));

// Mock logger
vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Mock fs
vi.mock('fs', async () => {
  const actual = await vi.importActual<typeof import('fs')>('fs');
  return {
    ...actual,
    default: {
      ...actual,
      existsSync: vi.fn(() => false),
      mkdirSync: vi.fn(),
      writeFileSync: vi.fn(),
      readFileSync: vi.fn(() => ''),
      readdirSync: vi.fn(() => []),
      statSync: vi.fn(() => ({ isDirectory: () => false })),
      copyFileSync: vi.fn(),
    },
  };
});

// Mock mount-security
vi.mock('./ipc-mounts.js', () => ({
  protectIpcClaimMounts: vi.fn((mounts) => mounts),
}));
vi.mock('./mount-security.js', () => ({
  validateAdditionalMounts: vi.fn(() => []),
}));

// Mock container-runtime
vi.mock('./container-runtime.js', () => ({
  CONTAINER_RUNTIME_BIN: 'docker',
  hostGatewayArgs: () => [],
  readonlyMountArgs: (h: string, c: string) => ['-v', `${h}:${c}:ro`],
  stopContainer: vi.fn(),
}));

// Mock OneCLI SDK
vi.mock('@onecli-sh/sdk', () => ({
  OneCLI: class {
    applyContainerConfig = vi.fn().mockResolvedValue(true);
    createAgent = vi.fn().mockResolvedValue({ id: 'test' });
    ensureAgent = vi
      .fn()
      .mockResolvedValue({ name: 'test', identifier: 'test', created: true });
  },
}));

// Create a controllable fake ChildProcess
function createFakeProcess() {
  const proc = new EventEmitter() as EventEmitter & {
    stdin: PassThrough;
    stdout: PassThrough;
    stderr: PassThrough;
    kill: ReturnType<typeof vi.fn>;
    pid: number;
  };
  proc.stdin = new PassThrough();
  proc.stdout = new PassThrough();
  proc.stderr = new PassThrough();
  proc.kill = vi.fn();
  proc.pid = 12345;
  return proc;
}

let fakeProc: ReturnType<typeof createFakeProcess>;

// Mock child_process.spawn
vi.mock('child_process', async () => {
  const actual =
    await vi.importActual<typeof import('child_process')>('child_process');
  return {
    ...actual,
    spawn: vi.fn(() => fakeProc),
    exec: vi.fn(
      (_cmd: string, _opts: unknown, cb?: (err: Error | null) => void) => {
        if (cb) cb(null);
        return new EventEmitter();
      },
    ),
  };
});

import { runContainerAgent, ContainerOutput } from './container-runner.js';
import { protectIpcClaimMounts } from './ipc-mounts.js';
import type { RegisteredGroup } from './types.js';

const testGroup: RegisteredGroup = {
  name: 'Test Group',
  folder: 'test-group',
  trigger: '@Andy',
  added_at: new Date().toISOString(),
};

const testInput = {
  prompt: 'Hello',
  groupFolder: 'test-group',
  chatJid: 'test@g.us',
  isMain: false,
};

function emitOutputMarker(
  proc: ReturnType<typeof createFakeProcess>,
  output: ContainerOutput,
) {
  const json = JSON.stringify(output);
  proc.stdout.push(`${OUTPUT_START_MARKER}\n${json}\n${OUTPUT_END_MARKER}\n`);
}

describe('container-runner timeout behavior', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.useRealTimers();
  });

  it('requires the fixed IPC mount closure before configuring or spawning a container', async () => {
    const { spawn } = await import('child_process');
    vi.mocked(spawn).mockClear();
    vi.mocked(protectIpcClaimMounts).mockImplementationOnce(() => {
      throw new Error('fixed unsafe mount');
    });
    await expect(
      runContainerAgent(testGroup, testInput, () => {}),
    ).rejects.toThrow('fixed unsafe mount');
    expect(spawn).not.toHaveBeenCalled();
  });

  it('keeps prompt, script and raw native output out of debug and error logs', async () => {
    vi.stubEnv('LOG_LEVEL', 'debug');
    vi.mocked(fs.writeFileSync).mockClear();
    for (const method of ['debug', 'error', 'warn', 'info'] as const)
      vi.mocked(logger[method]).mockClear();
    const secrets = [
      'fixed-prompt-secret',
      'fixed-script-secret',
      'fixed-stdout-secret',
      'fixed-stderr-secret',
    ];
    const pending = runContainerAgent(
      testGroup,
      { ...testInput, prompt: secrets[0], script: secrets[1] },
      () => {},
      vi.fn(async () => {}),
    );
    await vi.advanceTimersByTimeAsync(10);
    emitOutputMarker(fakeProc, { status: 'success', result: secrets[2] });
    fakeProc.stderr.push(secrets[3]);
    await vi.advanceTimersByTimeAsync(0);
    fakeProc.emit('close', 1);
    const output = await pending;
    const logs = JSON.stringify([
      ...vi
        .mocked(fs.writeFileSync)
        .mock.calls.filter(([file]) => String(file).endsWith('.log')),
      ...['debug', 'error', 'warn', 'info'].flatMap(
        (method) => vi.mocked(logger[method as keyof typeof logger]).mock.calls,
      ),
      output,
    ]);
    for (const secret of secrets) expect(logs).not.toContain(secret);
    expect(logs).toContain('Stdout length:');
    expect(output.status).toBe('error');
  });

  it('retains a fixed native spawn code without logging its raw message or arguments', async () => {
    vi.mocked(logger.error).mockClear();
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      vi.fn(async () => {}),
    );
    await vi.advanceTimersByTimeAsync(0);
    fakeProc.emit(
      'error',
      Object.assign(new Error('fixed-private-native-message'), {
        code: 'ENOENT',
        spawnargs: ['fixed-private-native-argument'],
      }),
    );
    const output = await pending;
    expect(output).toMatchObject({
      status: 'error',
      error: expect.stringContaining('ENOENT'),
    });
    const emitted = JSON.stringify([
      vi.mocked(logger.error).mock.calls,
      output,
    ]);
    expect(emitted).not.toContain('fixed-private-native-message');
    expect(emitted).not.toContain('fixed-private-native-argument');
    expect(vi.getTimerCount()).toBe(0);
  });

  it('decodes split multibyte stdout frames without altering delivered text', async () => {
    const onOutput = vi.fn(async () => {});
    const pending = runContainerAgent(testGroup, testInput, () => {}, onOutput);
    await vi.advanceTimersByTimeAsync(10);
    const text = 'café 👋 你好';
    const frame = Buffer.from(
      `${OUTPUT_START_MARKER}\n${JSON.stringify({ status: 'success', result: text })}\n${OUTPUT_END_MARKER}\n`,
    );
    for (const byte of frame) fakeProc.stdout.push(Buffer.from([byte]));
    await vi.advanceTimersByTimeAsync(0);
    fakeProc.emit('close', 0);
    await expect(pending).resolves.toMatchObject({ status: 'success' });
    expect(onOutput).toHaveBeenCalledExactlyOnceWith({
      status: 'success',
      result: text,
    });
  });

  it('accepts silent completed scheduled prechecks while refusing markerless scripts', async () => {
    const input = {
      ...testInput,
      isScheduledTask: true,
      script: 'fixture only',
    };
    const pending = runContainerAgent(
      testGroup,
      input,
      () => {},
      vi.fn(async () => {}),
    );
    await vi.advanceTimersByTimeAsync(10);
    emitOutputMarker(fakeProc, {
      status: 'success',
      result: null,
      completion: 'script_precheck_completed',
    });
    await vi.advanceTimersByTimeAsync(0);
    fakeProc.emit('close', 0);
    await expect(pending).resolves.toMatchObject({ status: 'success' });
    fakeProc = createFakeProcess();
    const missing = runContainerAgent(
      testGroup,
      input,
      () => {},
      vi.fn(async () => {}),
    );
    await vi.advanceTimersByTimeAsync(10);
    fakeProc.emit('close', 0);
    await expect(missing).resolves.toMatchObject({ status: 'error' });
  });

  it('timeout after output resolves as success', async () => {
    const onOutput = vi.fn(async () => {});
    const resultPromise = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      onOutput,
    );

    // Emit output with a result
    emitOutputMarker(fakeProc, {
      status: 'success',
      result: 'Here is my response',
      newSessionId: 'session-123',
    });

    // Let output processing settle
    await vi.advanceTimersByTimeAsync(10);

    // Fire the hard timeout (IDLE_TIMEOUT + 30s = 1830000ms)
    await vi.advanceTimersByTimeAsync(1830000);

    // Emit close event (as if container was stopped by the timeout)
    fakeProc.emit('close', 137);

    // Let the promise resolve
    await vi.advanceTimersByTimeAsync(10);

    const result = await resultPromise;
    expect(result.status).toBe('success');
    expect(result.newSessionId).toBe('session-123');
    expect(onOutput).toHaveBeenCalledWith(
      expect.objectContaining({ result: 'Here is my response' }),
    );
  });

  it('timeout with no output resolves as error', async () => {
    const onOutput = vi.fn(async () => {});
    const resultPromise = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      onOutput,
    );

    // No output emitted — fire the hard timeout
    await vi.advanceTimersByTimeAsync(1830000);

    // Emit close event
    fakeProc.emit('close', 137);

    await vi.advanceTimersByTimeAsync(10);

    const result = await resultPromise;
    expect(result.status).toBe('error');
    expect(result.error).toContain('timed out');
    expect(onOutput).not.toHaveBeenCalled();
  });

  it.each([null, [], 7, { status: 'success', result: { invalid: true } }])(
    'refuses malformed streaming output %j without throwing from the native data event',
    async (payload) => {
      const onOutput = vi.fn(async () => {});
      const pending = runContainerAgent(
        testGroup,
        testInput,
        () => {},
        onOutput,
      );
      fakeProc.stdout.push(
        `${OUTPUT_START_MARKER}\n${JSON.stringify(payload)}\n${OUTPUT_END_MARKER}\n`,
      );
      await vi.advanceTimersByTimeAsync(10);
      fakeProc.emit('close', 0);
      await expect(pending).resolves.toMatchObject({
        status: 'error',
        error: expect.stringContaining('invalid output'),
      });
      expect(onOutput).not.toHaveBeenCalled();
    },
  );

  it.each([
    null,
    [],
    7,
    { status: 'success', newSessionId: { invalid: true }, result: null },
  ])(
    'refuses malformed legacy output %j through the result contract',
    async (payload) => {
      const pending = runContainerAgent(testGroup, testInput, () => {});
      fakeProc.stdout.push(
        `${OUTPUT_START_MARKER}\n${JSON.stringify(payload)}\n${OUTPUT_END_MARKER}\n`,
      );
      await vi.advanceTimersByTimeAsync(10);
      fakeProc.emit('close', 0);
      await expect(pending).resolves.toMatchObject({
        status: 'error',
        error: expect.stringContaining('Failed to parse container output'),
      });
    },
  );

  it.each(['markerless', 'unfinished'] as const)(
    'bounds %s parser output and retires only its own container',
    async (kind) => {
      vi.mocked(stopContainer).mockClear();
      const pending = runContainerAgent(
        testGroup,
        testInput,
        () => {},
        vi.fn(async () => {}),
      );
      const rejected = expect(pending).rejects.toBeInstanceOf(
        ContainerOutputProtocolError,
      );
      await vi.advanceTimersByTimeAsync(10);
      fakeProc.stdout.push(
        (kind === 'unfinished' ? OUTPUT_START_MARKER : '') +
          'x'.repeat(10485761),
      );
      await vi.advanceTimersByTimeAsync(0);
      await rejected;
      expect(stopContainer).toHaveBeenCalledExactlyOnceWith(
        expect.stringContaining('nanoclaw-test-group-'),
      );
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it('does not treat a session-only marker as terminal timeout success', async () => {
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      vi.fn(async () => {}),
    );
    await vi.advanceTimersByTimeAsync(10);
    emitOutputMarker(fakeProc, {
      status: 'success',
      result: null,
      newSessionId: 'fixed-session',
    });
    await vi.advanceTimersByTimeAsync(10);
    await vi.advanceTimersByTimeAsync(1830000);
    fakeProc.emit('close', 137);
    await expect(pending).resolves.toMatchObject({
      status: 'error',
      error: expect.stringContaining('without a completed terminal response'),
    });
  });

  it.each([
    { status: 'success', result: null, completion: true },
    { status: 'success', result: null, completion: 'unknown' },
    { status: 'error', result: null, completion: 'sdk_turn_completed' },
    {
      status: 'success',
      result: null,
      streamText: 'preview',
      completion: 'sdk_turn_completed',
    },
  ])('refuses an invalid completion discriminator: %j', async (output) => {
    const onOutput = vi.fn(async () => {});
    const pending = runContainerAgent(testGroup, testInput, () => {}, onOutput);
    await vi.advanceTimersByTimeAsync(10);
    fakeProc.stdout.push(
      `${OUTPUT_START_MARKER}\n${JSON.stringify(output)}\n${OUTPUT_END_MARKER}\n`,
    );
    await vi.advanceTimersByTimeAsync(1830000);
    fakeProc.emit('close', 137);
    await expect(pending).resolves.toMatchObject({ status: 'error' });
    expect(onOutput).not.toHaveBeenCalled();
  });

  it('preserves settled visible output across a later session update during idle cleanup', async () => {
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      vi.fn(async () => {}),
    );
    await vi.advanceTimersByTimeAsync(10);
    emitOutputMarker(fakeProc, { status: 'success', result: 'fixed response' });
    emitOutputMarker(fakeProc, {
      status: 'success',
      result: null,
      newSessionId: 'fixed-session',
    });
    await vi.advanceTimersByTimeAsync(10);
    await vi.advanceTimersByTimeAsync(1830000);
    fakeProc.emit('close', 137);
    await expect(pending).resolves.toMatchObject({ status: 'success' });
  });

  it.each(['malformed', 'unfinished'] as const)(
    'does not accept timeout after a completion followed by a %s frame',
    async (kind) => {
      const pending = runContainerAgent(
        testGroup,
        testInput,
        () => {},
        vi.fn(async () => {}),
      );
      await vi.advanceTimersByTimeAsync(10);
      emitOutputMarker(fakeProc, {
        status: 'success',
        result: 'fixed completed result',
      });
      await vi.advanceTimersByTimeAsync(0);
      fakeProc.stdout.push(
        `${OUTPUT_START_MARKER}\n{fixed invalid` +
          (kind === 'malformed' ? `\n${OUTPUT_END_MARKER}\n` : ''),
      );
      await vi.advanceTimersByTimeAsync(1830000);
      fakeProc.emit('close', 137);
      await expect(pending).resolves.toMatchObject({ status: 'error' });
    },
  );

  it('does not treat preview-only activity as terminal timeout success', async () => {
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      vi.fn(async () => {}),
    );
    await vi.advanceTimersByTimeAsync(10);
    emitOutputMarker(fakeProc, {
      status: 'success',
      result: null,
      streamText: 'fixed preview',
    });
    await vi.advanceTimersByTimeAsync(10);
    await vi.advanceTimersByTimeAsync(1830000);
    fakeProc.emit('close', 137);
    await expect(pending).resolves.toMatchObject({
      status: 'error',
      error: expect.stringContaining('without a completed terminal response'),
    });
  });

  it('waits for a pending output callback after nonzero native close and retains both failures', async () => {
    vi.mocked(stopContainer).mockClear();
    const defect = new TypeError('fixed late output defect');
    let rejectOutput!: (error: unknown) => void;
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      () =>
        new Promise<void>((_resolve, reject) => {
          rejectOutput = reject;
        }),
    );
    const rejected = expect(pending).rejects.toMatchObject({
      cause: defect,
      errors: [defect, expect.any(ContainerExitError)],
    });
    let settled = false;
    void pending.then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      },
    );
    await vi.advanceTimersByTimeAsync(10);
    fakeProc.stdout.push(
      `${OUTPUT_START_MARKER}\n${JSON.stringify({ status: 'success', result: 'fixed output' })}\n${OUTPUT_END_MARKER}\n`,
    );
    await vi.advanceTimersByTimeAsync(0);
    fakeProc.emit('close', 1);
    await vi.advanceTimersByTimeAsync(0);
    expect(settled).toBe(false);
    rejectOutput(defect);
    await vi.advanceTimersByTimeAsync(0);
    await rejected;
    expect(stopContainer).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('owns a throttled preview defect without another output marker or finalization', async () => {
    vi.mocked(stopContainer).mockClear();
    const defect = new TypeError('fixed throttled preview defect');
    const stream = createDraftStream({
      sendMessage: vi.fn(async () => 17),
      editMessage: vi.fn(async () => {
        throw defect;
      }),
      deleteMessage: vi.fn(async () => {}),
    });
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      vi.fn(async () => {}),
      stream.failure,
    );
    const rejected = expect(pending).rejects.toBe(defect);
    await vi.advanceTimersByTimeAsync(10);
    await stream.update('fixed initial preview longer than thirty characters');
    await stream.update('fixed later preview longer than thirty characters');
    await vi.advanceTimersByTimeAsync(1000);
    await rejected;
    expect(stopContainer).toHaveBeenCalledExactlyOnceWith(
      expect.stringContaining('nanoclaw-test-group-'),
    );
    expect(vi.getTimerCount()).toBe(0);
  });

  it('rejects a delayed stdin EPIPE through the owning run and retires its container', async () => {
    vi.mocked(stopContainer).mockClear();
    const failure = Object.assign(new Error('fixed closed stdin'), {
      code: 'EPIPE',
    });
    const pending = runContainerAgent(testGroup, testInput, () => {});
    const rejected = expect(pending).rejects.toBe(failure);
    await vi.advanceTimersByTimeAsync(10);
    fakeProc.stdin.emit('error', failure);
    await vi.advanceTimersByTimeAsync(0);
    await rejected;
    expect(stopContainer).toHaveBeenCalledExactlyOnceWith(
      expect.stringContaining('nanoclaw-test-group-'),
    );
    expect(vi.getTimerCount()).toBe(0);
  });

  it('rejects a delayed stdout callback defect and retires only the owning container', async () => {
    vi.mocked(stopContainer).mockClear();
    const defect = new TypeError('fixed stdout callback defect');
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      vi.fn(async () => {}),
    );
    const rejected = expect(pending).rejects.toBe(defect);
    await vi.advanceTimersByTimeAsync(10);
    vi.mocked(logger.warn).mockImplementationOnce(() => {
      throw defect;
    });
    fakeProc.stdout.push(
      `${OUTPUT_START_MARKER}\nnull\n${OUTPUT_END_MARKER}\n`,
    );
    await vi.advanceTimersByTimeAsync(10);
    await rejected;
    expect(stopContainer).toHaveBeenCalledExactlyOnceWith(
      expect.stringContaining('nanoclaw-test-group-'),
    );
    expect(vi.getTimerCount()).toBe(0);
  });

  it('rejects a delayed timeout defect rather than throwing out of the timer', async () => {
    vi.mocked(stopContainer).mockClear();
    const defect = new TypeError('fixed timeout callback defect');
    vi.mocked(stopContainer).mockImplementationOnce(() => {
      throw defect;
    });
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      vi.fn(async () => {}),
    );
    const rejected = expect(pending).rejects.toBe(defect);
    await vi.advanceTimersByTimeAsync(1_830_001);
    await rejected;
    expect(stopContainer).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('rejects a delayed close callback defect without stopping an already closed container', async () => {
    vi.mocked(stopContainer).mockClear();
    const defect = new TypeError('fixed close callback defect');
    const pending = runContainerAgent(testGroup, testInput, () => {});
    const rejected = expect(pending).rejects.toBe(defect);
    await vi.advanceTimersByTimeAsync(10);
    vi.mocked(fs.writeFileSync).mockImplementationOnce(() => {
      throw defect;
    });
    fakeProc.emit('close', 0);
    await rejected;
    expect(stopContainer).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('routes an asynchronous output rejection through cleanup and the caller Promise', async () => {
    vi.mocked(stopContainer).mockClear();
    const defect = new TypeError('fixed output callback defect');
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      vi.fn(async () => {
        throw defect;
      }),
    );
    const rejected = expect(pending).rejects.toBe(defect);
    emitOutputMarker(fakeProc, { status: 'success', result: 'fixed output' });
    await vi.advanceTimersByTimeAsync(10);
    await rejected;
    expect(stopContainer).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
    fakeProc.emit('close', 0);
  });

  it('retains callback and native cleanup failures together', async () => {
    vi.mocked(stopContainer).mockClear();
    const defect = new TypeError('fixed output callback defect');
    const cleanup = new TypeError('fixed native cleanup defect');
    vi.mocked(stopContainer).mockImplementationOnce(() => {
      throw cleanup;
    });
    const pending = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      vi.fn(async () => {
        throw defect;
      }),
    );
    const rejected = expect(pending).rejects.toMatchObject({
      cause: defect,
      errors: [defect, cleanup],
    });
    emitOutputMarker(fakeProc, { status: 'success', result: 'fixed output' });
    await vi.advanceTimersByTimeAsync(10);
    await rejected;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('normal exit after output resolves as success', async () => {
    const onOutput = vi.fn(async () => {});
    const resultPromise = runContainerAgent(
      testGroup,
      testInput,
      () => {},
      onOutput,
    );

    // Emit output
    emitOutputMarker(fakeProc, {
      status: 'success',
      result: 'Done',
      newSessionId: 'session-456',
    });

    await vi.advanceTimersByTimeAsync(10);

    // Normal exit (no timeout)
    fakeProc.emit('close', 0);

    await vi.advanceTimersByTimeAsync(10);

    const result = await resultPromise;
    expect(result.status).toBe('success');
    expect(result.newSessionId).toBe('session-456');
  });
});
