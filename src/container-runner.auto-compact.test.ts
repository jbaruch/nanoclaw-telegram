// Flag-ON path for #252 — separate file because flipping
// `ENABLE_THRESHOLD_NUKE` mid-suite would require resetModules + dynamic
// import, which the rest of `container-runner.test.ts` doesn't do.
//
// Pins the inverse contract from `container-runner.test.ts`:
//   - DISABLE_COMPACT=1 IS injected
//   - CLAUDE_CODE_AUTO_COMPACT_WINDOW is NOT forwarded
//
// Both halves matter. Forwarding the window while DISABLE_COMPACT is set
// drops the SDK's `isAtBlockingLimit` to ~window − 30k; if that lands
// below the orchestrator's 800k nuke threshold the SDK refuses to send
// the request whose response would have triggered the threshold-nuke
// handshake. See #252 for the full trace through the SDK's `Jn()` and
// `UM6()` resolvers in `cli.js`.

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { spawn } from 'child_process';
import { EventEmitter } from 'events';
import { PassThrough } from 'stream';

vi.mock('./config.js', () => ({
  AGENT_AUTO_COMPACT_WINDOW: 800000,
  CONTAINER_IMAGE: 'nanoclaw-agent:latest',
  CONTAINER_MAX_OUTPUT_SIZE: 10485760,
  CONTAINER_TIMEOUT: 1800000,
  CREDENTIAL_PROXY_PORT: 3001,
  DATA_DIR: '/tmp/nanoclaw-test-data',
  ENABLE_THRESHOLD_NUKE: true,
  GROUPS_DIR: '/tmp/nanoclaw-test-groups',
  STORE_DIR: '/tmp/nanoclaw-test-store',
  HOST_PROJECT_ROOT: process.cwd(),
  HOST_UID: undefined,
  HOST_GID: undefined,
  IDLE_TIMEOUT: 1800000,
  MODEL_CONTEXT_WINDOW: 1000000,
  TILE_OWNER: 'test',
  TIMEZONE: 'America/Los_Angeles',
}));

vi.mock('better-sqlite3', () => ({ default: vi.fn() }));

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

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
      renameSync: vi.fn(),
      rmSync: vi.fn(),
      chownSync: vi.fn(),
      symlinkSync: vi.fn(),
      readlinkSync: vi.fn(() => ''),
      lstatSync: vi.fn(() => {
        const err = new Error('ENOENT') as NodeJS.ErrnoException;
        err.code = 'ENOENT';
        throw err;
      }),
    },
  };
});

vi.mock('./mount-security.js', () => ({
  validateAdditionalMounts: vi.fn(() => []),
}));

vi.mock('./container-runtime.js', () => ({
  CONTAINER_RUNTIME_BIN: 'docker',
  CONTAINER_HOST_GATEWAY: 'host.docker.internal',
  hostGatewayArgs: () => [],
  readonlyMountArgs: (h: string, c: string) => ['-v', `${h}:${c}:ro`],
  stopContainer: vi.fn(),
}));

vi.mock('./credential-proxy.js', () => ({
  detectAuthMode: vi.fn(() => 'api-key'),
}));

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

import { runContainerAgent } from './container-runner.js';
import type { RegisteredGroup } from './types.js';

const testGroup: RegisteredGroup = {
  name: 'Test Group',
  folder: 'test-group',
  trigger: '@Andy',
  // Fixed literal — `added_at` is not asserted against, but
  // `jbaruch/coding-policy: testing-standards` forbids self-generated
  // test data even in unused fields so a future assertion can't
  // accidentally introduce flakiness.
  added_at: '2026-01-01T00:00:00.000Z',
};

const testInput = {
  prompt: 'Hello',
  groupFolder: 'test-group',
  chatJid: 'test@g.us',
  isMain: false,
};

describe('CLAUDE_CODE_AUTO_COMPACT_WINDOW forwarding (flag on)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
    vi.mocked(spawn).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('injects DISABLE_COMPACT=1 when ENABLE_THRESHOLD_NUKE is on', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args).toContain('DISABLE_COMPACT=1');
  });

  it('does NOT forward CLAUDE_CODE_AUTO_COMPACT_WINDOW when the master flag is on', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    // Critical to #252: DISABLE_COMPACT only suppresses SDK
    // auto-compaction; the SDK's Jn() resolver still reads
    // CLAUDE_CODE_AUTO_COMPACT_WINDOW unconditionally and feeds it
    // into isAtBlockingLimit. Forwarding both would push the
    // blocking-limit below the 800k orchestrator nuke threshold and
    // wedge the request whose response triggers the handshake.
    expect(
      args.some((a) => a.startsWith('CLAUDE_CODE_AUTO_COMPACT_WINDOW=')),
    ).toBe(false);
  });
});
