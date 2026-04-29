import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { spawn } from 'child_process';
import { EventEmitter } from 'events';
import { PassThrough } from 'stream';

// Sentinel markers must match container-runner.ts
const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

// Mock config
vi.mock('./config.js', () => ({
  AGENT_AUTO_COMPACT_WINDOW: 800000,
  CONTAINER_IMAGE: 'nanoclaw-agent:latest',
  CONTAINER_MAX_OUTPUT_SIZE: 10485760,
  CONTAINER_TIMEOUT: 1800000, // 30min
  CREDENTIAL_PROXY_PORT: 3001,
  DATA_DIR: '/tmp/nanoclaw-test-data',
  ENABLE_THRESHOLD_NUKE: false,
  GROUPS_DIR: '/tmp/nanoclaw-test-groups',
  STORE_DIR: '/tmp/nanoclaw-test-store',
  HOST_PROJECT_ROOT: process.cwd(),
  HOST_UID: undefined,
  HOST_GID: undefined,
  IDLE_TIMEOUT: 1800000, // 30min
  MODEL_CONTEXT_WINDOW: 1000000,
  TILE_OWNER: 'test',
  TIMEZONE: 'America/Los_Angeles',
}));

// Mock better-sqlite3 (used by createFilteredDb)
vi.mock('better-sqlite3', () => ({
  default: vi.fn(),
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

// Mock fs. Several `fs.*` functions are mocked as no-ops because the
// scripts-dir publish path in container-runner.ts calls them on paths
// that were never created (mkdirSync is mocked). The symlink-based
// atomic publish added in the CodeQL-fix commit exercises symlinkSync/
// lstatSync/readlinkSync, so those need no-op mocks too. `lstatSync`
// throws ENOENT by default so the publish takes the first-install path
// (no prior groupScriptsDir) instead of trying to stat a mock-only path.
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
      // chownSync is a no-op so the post-mkdir chown on the
      // /workspace/state mount (and the trusted-dir mount above) doesn't
      // ENOENT against the never-created mock path. Pre-#99-Cat-4 the
      // production code swallowed all chown errors via a broad catch;
      // the narrowed catch (EPERM/EACCES only) lets ENOENT propagate,
      // so the mock must satisfy the call rather than rely on a
      // catch-all.
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

// Mock mount-security
vi.mock('./mount-security.js', () => ({
  validateAdditionalMounts: vi.fn(() => []),
}));

// Mock container-runtime
vi.mock('./container-runtime.js', () => ({
  CONTAINER_RUNTIME_BIN: 'docker',
  CONTAINER_HOST_GATEWAY: 'host.docker.internal',
  hostGatewayArgs: () => [],
  readonlyMountArgs: (h: string, c: string) => ['-v', `${h}:${c}:ro`],
  stopContainer: vi.fn(),
}));

// Mock credential-proxy
vi.mock('./credential-proxy.js', () => ({
  detectAuthMode: vi.fn(() => 'api-key'),
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

import {
  runContainerAgent,
  ContainerOutput,
  selectTiles,
  resolveAgentModel,
  DEFAULT_AGENT_MODEL,
} from './container-runner.js';
import { logger } from './logger.js';
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
    vi.useRealTimers();
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

// --- Tile selection (security-critical) ---

describe('selectTiles', () => {
  it('main group gets core + trusted + admin', () => {
    expect(selectTiles(true, false)).toEqual([
      'nanoclaw-core',
      'nanoclaw-trusted',
      'nanoclaw-admin',
    ]);
  });

  it('main group gets admin even if also marked trusted', () => {
    expect(selectTiles(true, true)).toEqual([
      'nanoclaw-core',
      'nanoclaw-trusted',
      'nanoclaw-admin',
    ]);
  });

  it('trusted group gets core + trusted, NOT admin', () => {
    const tiles = selectTiles(false, true);
    expect(tiles).toEqual(['nanoclaw-core', 'nanoclaw-trusted']);
    expect(tiles).not.toContain('nanoclaw-admin');
  });

  it('untrusted group gets core + untrusted, NOT trusted or admin', () => {
    const tiles = selectTiles(false, false);
    expect(tiles).toEqual(['nanoclaw-core', 'nanoclaw-untrusted']);
    expect(tiles).not.toContain('nanoclaw-trusted');
    expect(tiles).not.toContain('nanoclaw-admin');
  });

  it('all tiers include nanoclaw-core', () => {
    expect(selectTiles(true, false)[0]).toBe('nanoclaw-core');
    expect(selectTiles(false, true)[0]).toBe('nanoclaw-core');
    expect(selectTiles(false, false)[0]).toBe('nanoclaw-core');
  });

  it('admin tile is NEVER in trusted or untrusted selections', () => {
    expect(selectTiles(false, true)).not.toContain('nanoclaw-admin');
    expect(selectTiles(false, false)).not.toContain('nanoclaw-admin');
  });
});

// --- host-logs mount admin-only gating (#103 item 3) ---
//
// The mount is the security boundary for host-side observability:
// orchestrator log + per-container streaming logs are inherently
// cross-chat (every group's output) and must NEVER reach a non-admin
// tile. These tests assert the mount appears in the spawn args when
// the group is admin (`isMain: true`) AND is absent for trusted /
// untrusted groups, by inspecting the args passed to the mocked
// spawn.

describe('host-logs mount admin-only gating', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
    vi.mocked(spawn).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('admin (isMain=true) gets the host-logs mount', async () => {
    const adminGroup: RegisteredGroup = {
      ...testGroup,
      isMain: true,
    };
    const promise = runContainerAgent(
      adminGroup,
      { ...testInput, isMain: true },
      () => {},
    );
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    // The readonlyMountArgs mock formats as `host:container:ro`; the
    // mount appears as a single `-v ...host-logs:/workspace/host-logs:ro`
    // entry in the spawned arg list.
    expect(args.some((a) => a.includes(':/workspace/host-logs:ro'))).toBe(true);
  });

  it('trusted non-main group does NOT get the host-logs mount', async () => {
    const trustedGroup: RegisteredGroup = {
      ...testGroup,
      containerConfig: { trusted: true },
    };
    const promise = runContainerAgent(
      trustedGroup,
      { ...testInput, isMain: false, isTrusted: true },
      () => {},
    );
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    // Cross-chat host artifacts must not reach a trusted-but-not-main
    // container. If this assertion ever fires, every trusted tile
    // would gain visibility into every other group's stdout/stderr —
    // exactly the leak the admin-only gate is designed to prevent.
    expect(args.some((a) => a.includes('/workspace/host-logs'))).toBe(false);
  });

  it('untrusted group does NOT get the host-logs mount', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args.some((a) => a.includes('/workspace/host-logs'))).toBe(false);
  });
});

// --- /workspace/state mount: writable, all tiers (#99 Cat 4) ---
//
// Per-group canonical writable state directory. Must be present for
// every container regardless of trust tier, and must be writable
// (no `:ro` suffix). The whole point of the convention is that skills
// can persist state without caring about the trust tier they're
// running in — the silent-EACCES failure mode that motivated #99 only
// disappears if untrusted ALSO gets the mount.

describe('/workspace/state mount (#99 Cat 4)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
    vi.mocked(spawn).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function expectStateMount(args: string[]) {
    // Writable mounts are emitted as `-v <host>:<container>` (no `:ro`
    // suffix); readonly mounts go through readonlyMountArgs which the
    // mock formats as `<host>:<container>:ro`. Asserting the absence
    // of the `:ro` suffix on the state mount is the contract — a
    // future change that flipped this to readonly would silently
    // reintroduce the trust-tier write-failure mode.
    const stateArg = args.find((a) => a.endsWith(':/workspace/state'));
    expect(stateArg).toBeDefined();
    expect(args.some((a) => a.includes(':/workspace/state:ro'))).toBe(false);
  }

  it('admin (isMain=true) gets /workspace/state writable', async () => {
    const adminGroup: RegisteredGroup = { ...testGroup, isMain: true };
    const promise = runContainerAgent(
      adminGroup,
      { ...testInput, isMain: true },
      () => {},
    );
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    expectStateMount(vi.mocked(spawn).mock.calls[0]![1] as string[]);
  });

  it('trusted non-main group gets /workspace/state writable', async () => {
    const trustedGroup: RegisteredGroup = {
      ...testGroup,
      containerConfig: { trusted: true },
    };
    const promise = runContainerAgent(
      trustedGroup,
      { ...testInput, isMain: false, isTrusted: true },
      () => {},
    );
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    expectStateMount(vi.mocked(spawn).mock.calls[0]![1] as string[]);
  });

  it('untrusted group gets /workspace/state writable', async () => {
    // The whole point of the convention. If this assertion ever fires,
    // the silent-EACCES failure mode #99 Cat 4 was filed against has
    // returned: untrusted skills will appear to write state but the
    // bind-mount layer will reject silently, and the next run will
    // re-do whatever the state was supposed to remember.
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    expectStateMount(vi.mocked(spawn).mock.calls[0]![1] as string[]);
  });

  it('host path is per-group: <DATA_DIR>/state/<folder>', async () => {
    // Per-group scoping is intentional — see the rationale comment
    // above the mount in container-runner.ts. Cross-group leakage is
    // impossible by virtue of the bind being scoped to <folder>.
    // Mock sets DATA_DIR=/tmp/nanoclaw-test-data, so the bind resolves
    // to /tmp/nanoclaw-test-data/state/<folder>:/workspace/state.
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    const stateArg = args.find((a) => a.endsWith(':/workspace/state'));
    expect(stateArg).toBeDefined();
    expect(stateArg).toBe(
      '/tmp/nanoclaw-test-data/state/test-group:/workspace/state',
    );
  });
});

describe('readonly tile-content overlay (#247)', () => {
  // Pins the contract that installed tile content (skills + .tessl)
  // is mounted READONLY inside the agent container, while the parent
  // /home/node/.claude stays writable for SDK transcript / debug /
  // todos / telemetry / session-env / projects-memory writes.
  //
  // Why this matters: pre-fix, an agent could `Write` over its own
  // installed SKILL.md or RULES.md from inside the container. Edits
  // didn't survive container restart (the host-side cpSync at the
  // top of every spawn re-overwrote them) but were live for the
  // current container's lifetime — sometimes minutes-to-hours of
  // monkey-patched behaviour before reset. The kernel-level RO
  // overlay makes those writes fail with EROFS instead.
  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
    vi.mocked(spawn).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('mounts /home/node/.claude/skills as readonly', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    const skillsArg = args.find((a) =>
      a.endsWith(':/home/node/.claude/skills:ro'),
    );
    expect(skillsArg).toBeDefined();
  });

  it('mounts /home/node/.claude/.tessl as readonly', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    const tesslArg = args.find((a) =>
      a.endsWith(':/home/node/.claude/.tessl:ro'),
    );
    expect(tesslArg).toBeDefined();
  });

  it('parent /home/node/.claude mount is declared BEFORE the readonly overlays', async () => {
    // Mount order matters: Docker applies bind mounts in
    // declaration order, so a later parent mount would shadow
    // earlier child overlays and silently restore writability.
    // The argv index of the parent's `:/home/node/.claude` arg
    // must come before both readonly overlay args, otherwise
    // the kernel-level enforcement is structurally broken.
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    const parentIdx = args.findIndex((a) => a.endsWith(':/home/node/.claude'));
    const skillsRoIdx = args.findIndex((a) =>
      a.endsWith(':/home/node/.claude/skills:ro'),
    );
    const tesslRoIdx = args.findIndex((a) =>
      a.endsWith(':/home/node/.claude/.tessl:ro'),
    );
    expect(parentIdx).toBeGreaterThanOrEqual(0);
    expect(skillsRoIdx).toBeGreaterThanOrEqual(0);
    expect(tesslRoIdx).toBeGreaterThanOrEqual(0);
    expect(parentIdx).toBeLessThan(skillsRoIdx);
    expect(parentIdx).toBeLessThan(tesslRoIdx);
  });

  it('keeps /home/node/.claude itself writable (parent mount unchanged)', async () => {
    // The readonly subdir overlays must NOT regress the parent mount,
    // because the SDK writes session JSONL / debug / todos / telemetry
    // into siblings of skills/ and .tessl/. If the parent flipped to
    // readonly, every transcript write would fail and break the agent.
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    // The parent mount is a `<host>:/home/node/.claude` argument
    // WITHOUT `:ro`. The two readonly subdir mounts share the path
    // prefix but end with `/skills:ro` or `/.tessl:ro`, so an exact
    // suffix match on the parent's `:/home/node/.claude` form
    // cleanly distinguishes them.
    const claudeArg = args.find((a) => a.endsWith(':/home/node/.claude'));
    expect(claudeArg).toBeDefined();
    expect(args.some((a) => a.endsWith(':/home/node/.claude:ro'))).toBe(false);
  });

  it('readonly overlay applies uniformly across trust tiers', async () => {
    // The orchestrator's per-spawn cpSync writes installed tile
    // content into <groupSessionsDir> for every tier (main, trusted,
    // untrusted) — see selectTiles. The readonly overlay should
    // also apply to every tier; otherwise a less-trusted container
    // would have weaker enforcement than a more-trusted one, which
    // is precisely backwards.
    for (const profile of [
      { isMain: true, trusted: false },
      { isMain: false, trusted: true },
      { isMain: false, trusted: false },
    ]) {
      vi.mocked(spawn).mockClear();
      fakeProc = createFakeProcess();
      const group: RegisteredGroup = {
        ...testGroup,
        ...(profile.isMain && { isMain: true }),
        ...(profile.trusted && { containerConfig: { trusted: true } }),
      };
      const promise = runContainerAgent(
        group,
        { ...testInput, isMain: profile.isMain, isTrusted: profile.trusted },
        () => {},
      );
      fakeProc.emit('close', 0);
      await vi.advanceTimersByTimeAsync(10);
      await promise;

      const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
      expect(
        args.some((a) => a.endsWith(':/home/node/.claude/skills:ro')),
      ).toBe(true);
      expect(
        args.some((a) => a.endsWith(':/home/node/.claude/.tessl:ro')),
      ).toBe(true);
    }
  });
});

// --- continuation env vars (#93/#130) ---
//
// Self-resuming cycles depend on the container being able to tell
// "this run is a continuation" from "this run is a fresh user
// invocation". The mechanism is two paired env vars set by the
// scheduler when the underlying scheduled_tasks row carried a non-NULL
// `continuation_cycle_id`:
//
//   NANOCLAW_CONTINUATION=1
//   NANOCLAW_CONTINUATION_CYCLE_ID=<value>
//
// Both must be set together, or neither — a partial signal is the bug
// the calling skill's "fail closed to fresh invocation" branch is
// designed to catch, but the orchestrator must never produce a partial
// signal in the first place.

describe('continuation env vars (self-resuming cycles)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
    vi.mocked(spawn).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('sets both NANOCLAW_CONTINUATION env vars when continuationCycleId is provided', async () => {
    const promise = runContainerAgent(
      testGroup,
      { ...testInput, continuationCycleId: '2026-04-21' },
      () => {},
    );
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    // Both env vars must appear together. Asserting on the assembled
    // `-e KEY=value` strings (matching the existing chat-jid / reply-to
    // patterns) rather than parsing the args, since that's how the
    // shell ultimately receives them.
    expect(args).toContain('NANOCLAW_CONTINUATION=1');
    expect(args).toContain('NANOCLAW_CONTINUATION_CYCLE_ID=2026-04-21');
  });

  it('omits both NANOCLAW_CONTINUATION env vars on a fresh invocation', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    // Absence is the "fresh invocation" signal — the calling skill
    // distinguishes a continuation from a user-invoked run by env-var
    // presence. If either var leaks in, a fresh user-triggered run
    // could silently take the lock-skip continuation branch and
    // collide with the original maintenance run's two-phase lock.
    expect(args.some((a) => a.startsWith('NANOCLAW_CONTINUATION='))).toBe(
      false,
    );
    expect(
      args.some((a) => a.startsWith('NANOCLAW_CONTINUATION_CYCLE_ID=')),
    ).toBe(false);
  });

  it('omits both NANOCLAW_CONTINUATION env vars when continuationCycleId is empty string', async () => {
    // `''` is a JS-falsy slot key — should be treated identically to
    // undefined. The DB never persists an empty string (the IPC
    // handler drops `data.continuation_cycle_id` of `''` before
    // calling createTask), but defending in depth here keeps a future
    // accidental empty-string from emitting a half-signal.
    const promise = runContainerAgent(
      testGroup,
      { ...testInput, continuationCycleId: '' },
      () => {},
    );
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args.some((a) => a.startsWith('NANOCLAW_CONTINUATION='))).toBe(
      false,
    );
    expect(
      args.some((a) => a.startsWith('NANOCLAW_CONTINUATION_CYCLE_ID=')),
    ).toBe(false);
  });
});

// ----------------------------------------------------------------------
// CLAUDE_CODE_AUTO_COMPACT_WINDOW forwarding (#252) — flag-OFF path.
//
// File-level mock pins ENABLE_THRESHOLD_NUKE=false, so this describe
// covers the legacy / observe-only regime where the orchestrator
// forwards the configured window to the SDK and DISABLE_COMPACT is NOT
// injected. The flag-ON path lives in container-runner.auto-compact.test.ts
// because flipping the mock mid-file requires resetModules + dynamic
// import, which isn't a pattern this suite uses.
// ----------------------------------------------------------------------

describe('CLAUDE_CODE_AUTO_COMPACT_WINDOW forwarding (flag off)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
    vi.mocked(spawn).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('forwards CLAUDE_CODE_AUTO_COMPACT_WINDOW with the configured default', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    // The mock above sets AGENT_AUTO_COMPACT_WINDOW=800000. If this
    // assertion ever drifts, every container would silently regress to
    // whatever the previous default was — including the 165k upstream
    // hardcode that motivated #252 in the first place.
    expect(args).toContain('CLAUDE_CODE_AUTO_COMPACT_WINDOW=800000');
  });

  it('does not inject DISABLE_COMPACT when the master flag is off', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    // DISABLE_COMPACT and the SDK auto-compact window are mutually
    // exclusive (see container-runner.ts gating). If both ever leak
    // into the same container, the SDK's blocking-limit math drops
    // below the orchestrator's nuke threshold (#252 analysis).
    expect(args.some((a) => a.startsWith('DISABLE_COMPACT='))).toBe(false);
  });
});

// ----------------------------------------------------------------------
// resolveAgentModel — deterministic helper contract for the AGENT_MODEL
// env override, with optional warning on unknown prefixes (logger.warn
// is a side effect, so this isn't strictly pure). Pinned because the
// helper has five distinct branches and the default path is the only
// one the rest of the suite exercises.
// ----------------------------------------------------------------------

describe('resolveAgentModel', () => {
  beforeEach(() => {
    vi.mocked(logger.warn).mockClear();
  });

  it('returns default when env var is undefined', () => {
    expect(resolveAgentModel(undefined)).toBe(DEFAULT_AGENT_MODEL);
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('returns default when env var is the empty string', () => {
    expect(resolveAgentModel('')).toBe(DEFAULT_AGENT_MODEL);
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('returns default when env var is whitespace-only', () => {
    expect(resolveAgentModel('   ')).toBe(DEFAULT_AGENT_MODEL);
    expect(resolveAgentModel('\t\n ')).toBe(DEFAULT_AGENT_MODEL);
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('passes through known-prefix values silently (no warn)', () => {
    expect(resolveAgentModel('claude-opus-4-7[1m]')).toBe(
      'claude-opus-4-7[1m]',
    );
    expect(resolveAgentModel('claude-sonnet-4-6[1m]')).toBe(
      'claude-sonnet-4-6[1m]',
    );
    expect(resolveAgentModel('opus')).toBe('opus');
    expect(resolveAgentModel('sonnet[1m]')).toBe('sonnet[1m]');
    expect(resolveAgentModel('haiku')).toBe('haiku');
    // Mixed case — regex is case-insensitive.
    expect(resolveAgentModel('Claude-opus-4-7')).toBe('Claude-opus-4-7');
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('passes through unknown-prefix values WITH a warn so typos surface at startup', () => {
    // A typo like 'claud-opus' (missing 'e'): doesn't match prefix regex.
    expect(resolveAgentModel('claud-opus-4-7')).toBe('claud-opus-4-7');
    expect(logger.warn).toHaveBeenCalledTimes(1);
    expect(vi.mocked(logger.warn).mock.calls[0][1]).toContain(
      'AGENT_MODEL does not look like a Claude model ID',
    );
  });

  it('trims surrounding whitespace before validation and pass-through', () => {
    // .trim() must run before the prefix check, so `  opus  ` matches
    // 'opus' cleanly and doesn't trigger the warn.
    expect(resolveAgentModel('  claude-opus-4-7[1m]  ')).toBe(
      'claude-opus-4-7[1m]',
    );
    expect(resolveAgentModel('\topus\n')).toBe('opus');
    expect(logger.warn).not.toHaveBeenCalled();
  });
});
