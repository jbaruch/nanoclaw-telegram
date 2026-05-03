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
  ASSISTANT_NAME: 'TestBot',
  ASSISTANT_USERNAME: 'testbot',
  ASSISTANT_USERNAMES: ['testbot'],
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
  MAINTENANCE_RULE_BLOCKLIST: new Set<string>(),
  MAINTENANCE_SKILL_BLOCKLIST: new Set<string>(),
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

// #305 Phase 2a — runContainerAgent now invokes the cadence-registry
// rebuild after `buildVolumeMounts`. The wrapper in `./db.js` throws
// when the module-private `db` handle is uninitialised (this test
// harness simulates the spawn path without `initDatabase()`); stubbing
// the wrapper to a no-op keeps the rest of these assertions
// independent of the registry plumbing.
vi.mock('./db.js', () => ({
  rebuildCadenceRegistryForGroup: vi.fn(() => ({
    deleted: 0,
    inserted: 0,
    updated: 0,
    preserved: 0,
    walked: 0,
    errors: [],
  })),
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
  resolvePerGroupAgentModel,
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

  // --- #305: additionalTiles overlay ---

  it('appends additionalTiles after baseline for trusted group', () => {
    expect(
      selectTiles(false, true, ['nanoclaw-coding', 'nanoclaw-family']),
    ).toEqual([
      'nanoclaw-core',
      'nanoclaw-trusted',
      'nanoclaw-coding',
      'nanoclaw-family',
    ]);
  });

  it('appends additionalTiles after baseline for untrusted group', () => {
    expect(selectTiles(false, false, ['nanoclaw-coding'])).toEqual([
      'nanoclaw-core',
      'nanoclaw-untrusted',
      'nanoclaw-coding',
    ]);
  });

  it('appends additionalTiles after baseline for main group (admin still last among baseline)', () => {
    expect(selectTiles(true, false, ['nanoclaw-coding'])).toEqual([
      'nanoclaw-core',
      'nanoclaw-trusted',
      'nanoclaw-admin',
      'nanoclaw-coding',
    ]);
  });

  it('drops additionalTiles entries that duplicate a baseline tile', () => {
    expect(
      selectTiles(false, true, [
        'nanoclaw-core',
        'nanoclaw-trusted',
        'nanoclaw-coding',
      ]),
    ).toEqual(['nanoclaw-core', 'nanoclaw-trusted', 'nanoclaw-coding']);
  });

  it('de-duplicates repeated entries within additionalTiles', () => {
    expect(
      selectTiles(false, true, ['nanoclaw-coding', 'nanoclaw-coding']),
    ).toEqual(['nanoclaw-core', 'nanoclaw-trusted', 'nanoclaw-coding']);
  });

  it('skips empty / whitespace-only entries in additionalTiles', () => {
    expect(selectTiles(false, true, ['', '   ', 'nanoclaw-coding'])).toEqual([
      'nanoclaw-core',
      'nanoclaw-trusted',
      'nanoclaw-coding',
    ]);
  });

  it('preserves order when additionalTiles are all unique against baseline', () => {
    expect(
      selectTiles(false, true, ['nanoclaw-c', 'nanoclaw-a', 'nanoclaw-b']),
    ).toEqual([
      'nanoclaw-core',
      'nanoclaw-trusted',
      'nanoclaw-c',
      'nanoclaw-a',
      'nanoclaw-b',
    ]);
  });

  it('treats undefined / empty additionalTiles as a no-op', () => {
    expect(selectTiles(false, true)).toEqual([
      'nanoclaw-core',
      'nanoclaw-trusted',
    ]);
    expect(selectTiles(false, true, [])).toEqual([
      'nanoclaw-core',
      'nanoclaw-trusted',
    ]);
    expect(selectTiles(false, true, undefined)).toEqual([
      'nanoclaw-core',
      'nanoclaw-trusted',
    ]);
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

// ----------------------------------------------------------------------
// resolvePerGroupAgentModel — per-group AGENT_MODEL override (#395).
// Stricter than resolveAgentModel: empty AND unknown-prefix both fall
// back to the global default, so a fat-fingered IPC `set_agent_model`
// can't silently route a group's spawns to a non-existent model.
// ----------------------------------------------------------------------

describe('resolvePerGroupAgentModel', () => {
  const FALLBACK = DEFAULT_AGENT_MODEL;

  beforeEach(() => {
    vi.mocked(logger.warn).mockClear();
  });

  it('returns fallback when override is undefined', () => {
    expect(resolvePerGroupAgentModel(undefined, FALLBACK)).toBe(FALLBACK);
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('returns fallback when override is null (clear-the-override path)', () => {
    expect(resolvePerGroupAgentModel(null, FALLBACK)).toBe(FALLBACK);
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('returns fallback when override is empty / whitespace-only', () => {
    expect(resolvePerGroupAgentModel('', FALLBACK)).toBe(FALLBACK);
    expect(resolvePerGroupAgentModel('   ', FALLBACK)).toBe(FALLBACK);
    expect(resolvePerGroupAgentModel('\t\n ', FALLBACK)).toBe(FALLBACK);
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('returns the trimmed override on a known-prefix value', () => {
    expect(resolvePerGroupAgentModel('claude-opus-4-7[1m]', FALLBACK)).toBe(
      'claude-opus-4-7[1m]',
    );
    expect(resolvePerGroupAgentModel('opus', FALLBACK)).toBe('opus');
    expect(resolvePerGroupAgentModel('sonnet[1m]', FALLBACK)).toBe(
      'sonnet[1m]',
    );
    expect(resolvePerGroupAgentModel('haiku', FALLBACK)).toBe('haiku');
    expect(resolvePerGroupAgentModel('  opus  ', FALLBACK)).toBe('opus');
    expect(logger.warn).not.toHaveBeenCalled();
  });

  it('falls back to global default + warns on unknown-prefix override', () => {
    // Stricter than the global resolveAgentModel: per-group overrides
    // are set at runtime via IPC, so a typo can't be caught at startup
    // — failing closed to the global default keeps the group running.
    expect(resolvePerGroupAgentModel('claud-opus-4-7', FALLBACK)).toBe(
      FALLBACK,
    );
    expect(resolvePerGroupAgentModel('foobar', FALLBACK)).toBe(FALLBACK);
    expect(logger.warn).toHaveBeenCalledTimes(2);
    expect(vi.mocked(logger.warn).mock.calls[0][1]).toContain(
      'Per-group AGENT_MODEL override does not look like a Claude model ID',
    );
  });
});

// ----------------------------------------------------------------------
// Per-group AGENT_MODEL override — spawn-arg integration (#395).
// Verifies the override flows through buildContainerArgs to the actual
// `-e AGENT_MODEL=…` arg on the docker command line, and that the global
// default is used otherwise.
// ----------------------------------------------------------------------

describe('per-group AGENT_MODEL override on container spawn', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
    vi.mocked(spawn).mockClear();
    vi.mocked(logger.info).mockClear();
    vi.mocked(logger.warn).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('uses the global default AGENT_MODEL when no per-group override is set', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args).toContain(`AGENT_MODEL=${DEFAULT_AGENT_MODEL}`);
  });

  // #418: spawn-time AGENT_MODEL log fires UNCONDITIONALLY so cost /
  // latency auditing has a per-spawn trail covering both override AND
  // default cases. The `source` field tags which path the value came
  // from so the log is self-explaining.
  it('emits exactly one AGENT_MODEL-resolved log per default spawn (#418)', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const infoCalls = vi
      .mocked(logger.info)
      .mock.calls.filter(
        (c) =>
          typeof c[1] === 'string' &&
          c[1].includes('Container spawn AGENT_MODEL resolved'),
      );
    expect(infoCalls.length).toBe(1);
    expect(infoCalls[0]![0]).toEqual(
      expect.objectContaining({
        agentModel: DEFAULT_AGENT_MODEL,
        globalDefault: DEFAULT_AGENT_MODEL,
        source: 'global_default',
      }),
    );
  });

  it('forwards a valid per-group override and emits one info log', async () => {
    const overrideGroup: RegisteredGroup = {
      ...testGroup,
      containerConfig: { agentModel: 'claude-sonnet-4-6[1m]' },
    };
    const promise = runContainerAgent(overrideGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args).toContain('AGENT_MODEL=claude-sonnet-4-6[1m]');
    expect(args).not.toContain(`AGENT_MODEL=${DEFAULT_AGENT_MODEL}`);

    // #418: same single info log fires for override spawns; the
    // `source` field tags it as `group_override` to distinguish from
    // the default-spawn case.
    const infoCalls = vi
      .mocked(logger.info)
      .mock.calls.filter(
        (c) =>
          typeof c[1] === 'string' &&
          c[1].includes('Container spawn AGENT_MODEL resolved'),
      );
    expect(infoCalls.length).toBe(1);
    expect(infoCalls[0]![0]).toEqual(
      expect.objectContaining({
        agentModel: 'claude-sonnet-4-6[1m]',
        globalDefault: DEFAULT_AGENT_MODEL,
        source: 'group_override',
      }),
    );
  });

  it('falls back to global default + warns when override has unknown prefix', async () => {
    const overrideGroup: RegisteredGroup = {
      ...testGroup,
      containerConfig: { agentModel: 'claud-opus-4-7' },
    };
    const promise = runContainerAgent(overrideGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    // Failed closed: bad prefix → global default, not the typo.
    expect(args).toContain(`AGENT_MODEL=${DEFAULT_AGENT_MODEL}`);
    expect(args).not.toContain('AGENT_MODEL=claud-opus-4-7');
    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ agentModel: 'claud-opus-4-7' }),
      expect.stringContaining(
        'Per-group AGENT_MODEL override does not look like a Claude model ID',
      ),
    );
  });

  it('treats empty-string override as no override (no warn)', async () => {
    const overrideGroup: RegisteredGroup = {
      ...testGroup,
      containerConfig: { agentModel: '' },
    };
    const promise = runContainerAgent(overrideGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args).toContain(`AGENT_MODEL=${DEFAULT_AGENT_MODEL}`);
    // Empty override is the "no override" signal — no warn log.
    const warnCalls = vi
      .mocked(logger.warn)
      .mock.calls.filter(
        (c) => typeof c[1] === 'string' && c[1].includes('AGENT_MODEL'),
      );
    expect(warnCalls.length).toBe(0);
  });
});

// ----------------------------------------------------------------------
// ASSISTANT_NAME / ASSISTANT_USERNAME forwarding (#407, cherry-pick of
// ligolnik/nanoclaw-public#90).
//
// The orchestrator's authoritative identity config must reach the agent
// container so the agent-runner can prepend an identity preamble to
// systemPromptAppend (see container/agent-runner/src/index.ts —
// buildIdentityPreamble). Without this, untrusted-tier containers
// without an explicit identity statement in their persona files have
// been observed templating themselves from fictional bot handles in
// tile rules (a deployment reading the @AyeAye / @AyeAyeSureBot
// canonical example from nanoclaw-core 0.1.94 and claiming those
// handles as its own identity).
//
// These are not secrets, so the test config mock above sets them as
// plain synthetic strings ('TestBot' / 'testbot') and the assertion
// checks they flow straight through to the spawn args via -e.
// ----------------------------------------------------------------------

describe('ASSISTANT_NAME / ASSISTANT_USERNAME forwarding', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
    vi.mocked(spawn).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('forwards ASSISTANT_NAME and the joined ASSISTANT_USERNAMES list on container spawn', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args).toContain('ASSISTANT_NAME=TestBot');
    // Multi-handle forwarding (#464): the orchestrator joins
    // `ASSISTANT_USERNAMES` so the agent-runner sees every alias and
    // its identity preamble can teach the agent that all of them
    // resolve to it. Single-handle case (this fixture) collapses to
    // one bare token — byte-stable with the pre-#464 single-value
    // forwarding. The `len > 1` re-parse is covered end-to-end in
    // `container/agent-runner/src/identity-preamble.test.ts`.
    expect(args).toContain('ASSISTANT_USERNAME=testbot');
  });
});

// ----------------------------------------------------------------------
// USE_CUSTOM_PROMPT forwarding (#465 / ligolnik#122).
//
// Pins the resolution shape `buildContainerArgs` uses to decide whether
// to forward `USE_CUSTOM_PROMPT=1` to the container env:
//
//   1. `containerConfig.useCustomPrompt: true` → forward (any tier).
//   2. `containerConfig.useCustomPrompt: false` → never forward, even
//      if the global env says yes (explicit per-group veto wins).
//   3. `containerConfig.useCustomPrompt` undefined →
//      forward only when `USE_CUSTOM_PROMPT_FOR_MAIN=1` AND `isMain`.
//
// Defense-against-regression: the precedence here is what makes the
// flag flip safe-by-default — silent leak across tiers (forwarding to
// trusted/untrusted via the global env) would surprise operators.
// ----------------------------------------------------------------------

describe('USE_CUSTOM_PROMPT forwarding', () => {
  const ORIGINAL_ENV = process.env.USE_CUSTOM_PROMPT_FOR_MAIN;

  beforeEach(() => {
    vi.useFakeTimers();
    fakeProc = createFakeProcess();
    vi.mocked(spawn).mockClear();
    delete process.env.USE_CUSTOM_PROMPT_FOR_MAIN;
  });

  afterEach(() => {
    vi.useRealTimers();
    if (ORIGINAL_ENV === undefined) {
      delete process.env.USE_CUSTOM_PROMPT_FOR_MAIN;
    } else {
      process.env.USE_CUSTOM_PROMPT_FOR_MAIN = ORIGINAL_ENV;
    }
  });

  it('does not forward when neither global env nor per-group config is set', async () => {
    const promise = runContainerAgent(testGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args).not.toContain('USE_CUSTOM_PROMPT=1');
  });

  it('forwards when per-group containerConfig.useCustomPrompt is true (untrusted tier)', async () => {
    const overrideGroup: RegisteredGroup = {
      ...testGroup,
      containerConfig: { useCustomPrompt: true },
    };
    const promise = runContainerAgent(overrideGroup, testInput, () => {});
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args).toContain('USE_CUSTOM_PROMPT=1');
  });

  it('does NOT forward when per-group config explicitly sets useCustomPrompt: false (even if global env is on)', async () => {
    process.env.USE_CUSTOM_PROMPT_FOR_MAIN = '1';
    const optOutGroup: RegisteredGroup = {
      ...testGroup,
      containerConfig: { useCustomPrompt: false },
    };
    const promise = runContainerAgent(
      optOutGroup,
      { ...testInput, isMain: true },
      () => {},
    );
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args).not.toContain('USE_CUSTOM_PROMPT=1');
  });

  it('global env enables main but does NOT leak to trusted tier', async () => {
    process.env.USE_CUSTOM_PROMPT_FOR_MAIN = '1';
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
    expect(args).not.toContain('USE_CUSTOM_PROMPT=1');
  });

  it('global env enables main and the main-tier container DOES receive USE_CUSTOM_PROMPT=1', async () => {
    process.env.USE_CUSTOM_PROMPT_FOR_MAIN = '1';
    const promise = runContainerAgent(
      testGroup,
      { ...testInput, isMain: true },
      () => {},
    );
    fakeProc.emit('close', 0);
    await vi.advanceTimersByTimeAsync(10);
    await promise;

    const args = vi.mocked(spawn).mock.calls[0]![1] as string[];
    expect(args).toContain('USE_CUSTOM_PROMPT=1');
  });
});
