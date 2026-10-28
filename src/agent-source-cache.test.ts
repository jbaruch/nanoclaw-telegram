import fs from 'fs';
import os from 'os';
import path from 'path';
import { EventEmitter } from 'events';
import { PassThrough } from 'stream';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { syncAgentRunnerSource } from './agent-source-cache.js';

const fixture = vi.hoisted(() => ({ directory: '' }));
vi.mock('./config.js', () => ({
  get DATA_DIR() {
    return path.join(fixture.directory, 'data');
  },
  get GROUPS_DIR() {
    return path.join(fixture.directory, 'groups');
  },
  CONTAINER_IMAGE: 'synthetic',
  CONTAINER_MAX_OUTPUT_SIZE: 102400,
  CONTAINER_TIMEOUT: 10000,
  IDLE_TIMEOUT: 10000,
  ONECLI_URL: 'http://fixture.invalid',
  TIMEZONE: 'UTC',
}));
vi.mock('@onecli-sh/sdk', () => ({
  OneCLI: class {
    applyContainerConfig = async () => false;
  },
}));
vi.mock('./ipc-mounts.js', () => ({
  protectIpcClaimMounts: (mounts: unknown[]) => mounts,
}));
vi.mock('./mount-security.js', () => ({ validateAdditionalMounts: () => [] }));
vi.mock('./container-runtime.js', () => ({
  CONTAINER_RUNTIME_BIN: 'synthetic-no-execution',
  hostGatewayArgs: () => [],
  readonlyMountArgs: (h: string, c: string) => ['-v', `${h}:${c}:ro`],
  stopContainer: vi.fn(),
}));
vi.mock('child_process', async () => ({
  ...(await vi.importActual<typeof import('child_process')>('child_process')),
  spawn: vi.fn(),
}));
import { spawn } from 'child_process';
import { runContainerAgent } from './container-runner.js';

let source: string, cache: string, marker: string;
beforeEach(() => {
  fixture.directory = fs.mkdtempSync(
    path.join(os.tmpdir(), 'telegram-source-cache-'),
  );
  source = path.join(fixture.directory, 'container', 'agent-runner', 'src');
  cache = path.join(
    fixture.directory,
    'data',
    'sessions',
    'main',
    'agent-runner-src',
  );
  marker = path.join(path.dirname(cache), 'agent-runner-source.json');
  fs.mkdirSync(source, { recursive: true });
  fs.mkdirSync(cache, { recursive: true });
  fs.writeFileSync(path.join(source, 'index.ts'), 'canonical index');
  fs.writeFileSync(path.join(source, 'ipc-mcp-stdio.ts'), 'canonical producer');
  vi.spyOn(process, 'cwd').mockReturnValue(fixture.directory);
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  fs.rmSync(fixture.directory, { recursive: true, force: true });
});

async function runSyntheticContainer(): Promise<void> {
  const child = Object.assign(new EventEmitter(), {
    stdin: new PassThrough(),
    stdout: new PassThrough(),
    stderr: new PassThrough(),
    kill: vi.fn(),
    pid: 12345,
  });
  vi.mocked(spawn).mockReturnValueOnce(
    child as unknown as ReturnType<typeof spawn>,
  );
  const result = runContainerAgent(
    {
      name: 'fixture',
      folder: 'main',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00Z',
    },
    {
      prompt: 'fixture',
      groupFolder: 'main',
      chatJid: 'tg:fixture',
      isMain: false,
    },
    () => {},
  );
  await vi.waitFor(() => expect(spawn).toHaveBeenCalled());
  child.stdout.push(
    '---NANOCLAW_OUTPUT_START---\n' +
      JSON.stringify({
        status: 'success',
        result: 'fixture',
        completion: 'sdk_turn_completed',
      }) +
      '\n---NANOCLAW_OUTPUT_END---\n',
  );
  child.emit('close', 0);
  expect((await result).status).toBe('success');
}

it('actual mount construction refreshes a newer legacy cache and a non-index-only source update', async () => {
  fs.writeFileSync(path.join(cache, 'index.ts'), 'legacy customized index');
  fs.writeFileSync(path.join(cache, 'ipc-mcp-stdio.ts'), 'old producer');
  fs.utimesSync(
    path.join(cache, 'index.ts'),
    new Date('2099-01-01'),
    new Date('2099-01-01'),
  );
  await runSyntheticContainer();
  expect(fs.readFileSync(path.join(cache, 'ipc-mcp-stdio.ts'), 'utf8')).toBe(
    'canonical producer',
  );
  expect(JSON.parse(fs.readFileSync(marker, 'utf8'))).toMatchObject({
    schema_version: 1,
    state: 'ready',
  });
  vi.mocked(spawn).mockClear();
  fs.writeFileSync(path.join(source, 'ipc-mcp-stdio.ts'), 'corrected producer');
  fs.utimesSync(
    path.join(cache, 'index.ts'),
    new Date('2099-01-01'),
    new Date('2099-01-01'),
  );
  await runSyntheticContainer();
  expect(fs.readFileSync(path.join(cache, 'ipc-mcp-stdio.ts'), 'utf8')).toBe(
    'corrected producer',
  );
});

it('preserves customized shipped bytes at the same canonical digest and custom-only files across updates', () => {
  syncAgentRunnerSource(source, cache);
  const ready = fs.readFileSync(marker, 'utf8');
  fs.writeFileSync(path.join(cache, 'index.ts'), 'custom index');
  fs.writeFileSync(path.join(cache, 'custom-only.ts'), 'retained custom tool');
  syncAgentRunnerSource(source, cache);
  expect(fs.readFileSync(path.join(cache, 'index.ts'), 'utf8')).toBe(
    'custom index',
  );
  expect(fs.readFileSync(marker, 'utf8')).toBe(ready);
  fs.writeFileSync(path.join(source, 'ipc-mcp-stdio.ts'), 'updated source');
  syncAgentRunnerSource(source, cache);
  expect(fs.readFileSync(path.join(cache, 'index.ts'), 'utf8')).toBe(
    'canonical index',
  );
  expect(fs.readFileSync(path.join(cache, 'custom-only.ts'), 'utf8')).toBe(
    'retained custom tool',
  );
});

it.each(['future', 'malformed', 'symlink', 'hardlink'] as const)(
  'holds an unsafe %s marker unchanged before cache refresh',
  (kind) => {
    const original = 'retained custom bytes';
    fs.writeFileSync(path.join(cache, 'index.ts'), original);
    const foreign = path.join(fixture.directory, 'foreign-marker');
    const raw =
      kind === 'malformed'
        ? '{broken'
        : JSON.stringify({
            schema_version: 2,
            state: 'ready',
            source_sha256: 'a'.repeat(64),
          });
    fs.writeFileSync(foreign, raw, { mode: 0o600 });
    if (kind === 'symlink') fs.symlinkSync(foreign, marker);
    else if (kind === 'hardlink') fs.linkSync(foreign, marker);
    else fs.writeFileSync(marker, raw, { mode: 0o600 });
    expect(() => syncAgentRunnerSource(source, cache)).toThrow();
    expect(fs.readFileSync(path.join(cache, 'index.ts'), 'utf8')).toBe(
      original,
    );
    expect(fs.readFileSync(foreign, 'utf8')).toBe(raw);
    expect(fs.readFileSync(marker, 'utf8')).toBe(raw);
  },
);

it('keeps refresh debt through native rename failure and a canonical rollback', () => {
  syncAgentRunnerSource(source, cache);
  fs.writeFileSync(path.join(source, 'index.ts'), 'new index');
  fs.writeFileSync(path.join(source, 'ipc-mcp-stdio.ts'), 'new producer');
  const rename = fs.renameSync.bind(fs);
  const fault = vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
    if (String(from).endsWith('/ipc-mcp-stdio.ts'))
      throw Object.assign(new Error('fixed native cache rename failure'), {
        code: 'EIO',
      });
    rename(from, to);
  });
  expect(() => syncAgentRunnerSource(source, cache)).toThrow(
    'fixed native cache rename failure',
  );
  expect(JSON.parse(fs.readFileSync(marker, 'utf8')).state).toBe('refreshing');
  expect(fs.readFileSync(path.join(cache, 'index.ts'), 'utf8')).toBe(
    'new index',
  );
  fault.mockRestore();
  fs.writeFileSync(path.join(source, 'index.ts'), 'canonical index');
  fs.writeFileSync(path.join(source, 'ipc-mcp-stdio.ts'), 'canonical producer');
  syncAgentRunnerSource(source, cache);
  expect(fs.readFileSync(path.join(cache, 'index.ts'), 'utf8')).toBe(
    'canonical index',
  );
  expect(JSON.parse(fs.readFileSync(marker, 'utf8')).state).toBe('ready');
});

it.each(['symlink', 'hardlink'] as const)(
  'atomically replaces a cached %s leaf without writing through its foreign target',
  (kind) => {
    const foreign = path.join(fixture.directory, 'foreign-source');
    fs.writeFileSync(foreign, 'private foreign bytes');
    const leaf = path.join(cache, 'index.ts');
    if (kind === 'symlink') fs.symlinkSync(foreign, leaf);
    else fs.linkSync(foreign, leaf);
    syncAgentRunnerSource(source, cache);
    expect(fs.readFileSync(foreign, 'utf8')).toBe('private foreign bytes');
    expect(fs.readFileSync(leaf, 'utf8')).toBe('canonical index');
    expect(fs.lstatSync(leaf).nlink).toBe(1);
  },
);

it('refuses an unrecognized canonical source layout instead of omitting it from the fingerprint', () => {
  fs.mkdirSync(path.join(source, 'new-layout'));
  fs.writeFileSync(path.join(cache, 'index.ts'), 'retained custom');
  expect(() => syncAgentRunnerSource(source, cache)).toThrow(
    'Unsupported canonical agent source entry',
  );
  expect(fs.readFileSync(path.join(cache, 'index.ts'), 'utf8')).toBe(
    'retained custom',
  );
  expect(fs.existsSync(marker)).toBe(false);
});

it('uses explicit readable canonical and private marker modes under restrictive umask', () => {
  const prior = process.umask(0o077);
  try {
    syncAgentRunnerSource(source, cache);
    expect(fs.statSync(path.join(cache, 'index.ts')).mode & 0o777).toBe(0o644);
    expect(fs.statSync(marker).mode & 0o777).toBe(0o600);
  } finally {
    process.umask(prior);
  }
});

it.each([false, true])(
  'preserves a same-base customized entry link independently of host target presence=%s',
  (present) => {
    syncAgentRunnerSource(source, cache);
    const foreign = path.join(
      fixture.directory,
      'container-only-custom-target.ts',
    );
    if (present) fs.writeFileSync(foreign, 'retained target');
    const index = path.join(cache, 'index.ts');
    fs.unlinkSync(index);
    fs.symlinkSync(foreign, index);
    const markerBefore = fs.readFileSync(marker, 'utf8');
    syncAgentRunnerSource(source, cache);
    expect(fs.lstatSync(index).isSymbolicLink()).toBe(true);
    expect(fs.readlinkSync(index)).toBe(foreign);
    expect(fs.readFileSync(marker, 'utf8')).toBe(markerBefore);
    if (present)
      expect(fs.readFileSync(foreign, 'utf8')).toBe('retained target');
  },
);
