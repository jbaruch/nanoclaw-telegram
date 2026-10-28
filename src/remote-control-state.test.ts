import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawn, spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const fixture = vi.hoisted(() => ({ directory: '' }));
vi.mock('./config.js', () => ({
  get DATA_DIR() {
    return fixture.directory;
  },
}));
let owner: typeof import('./remote-control.js');
const nativeKill = process.kill.bind(process);
const legacy = {
  pid: 77777,
  url: 'https://claude.ai/code?bridge=env_fixed',
  startedBy: 'fixed',
  startedInChat: 'tg:123',
  startedAt: '2024-01-01T00:00:00Z',
};
beforeEach(async () => {
  fixture.directory = fs.mkdtempSync(
    path.join(os.tmpdir(), 'nanoclaw-remote-state-'),
  );
  vi.resetModules();
  owner = await import('./remote-control.js');
  vi.spyOn(process, 'kill').mockReturnValue(true);
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  fs.rmSync(fixture.directory, { recursive: true });
});

describe('atomic Remote Control owner migration', () => {
  it('preserves an occupied unknown starting gate without opening child log files', async () => {
    const file = path.join(fixture.directory, 'remote-control-starting.json');
    const bytes = JSON.stringify({ schema_version: 99, state: 'starting' });
    fs.writeFileSync(file, bytes, { mode: 0o600 });
    await expect(
      owner.startRemoteControl('fixed', 'tg:123', fixture.directory),
    ).rejects.toThrow('starting gate is occupied');
    expect(fs.readFileSync(file, 'utf8')).toBe(bytes);
    expect(fs.readdirSync(fixture.directory)).toEqual([
      'remote-control-starting.json',
    ]);
  });

  it('uses the actual cross-process exclusive reservation before a private synthetic CLI launch', async () => {
    const bin = path.join(fixture.directory, 'bin');
    fs.mkdirSync(bin);
    const counter = path.join(fixture.directory, 'synthetic-launches');
    const pidFile = path.join(fixture.directory, 'synthetic-child-pid');
    const release = path.join(fixture.directory, 'release-synthetic-url');
    fs.writeFileSync(
      path.join(bin, 'claude'),
      `#!${process.execPath}\nimport fs from 'node:fs';\nfs.writeFileSync(${JSON.stringify(pidFile)},String(process.pid));\nfs.appendFileSync(${JSON.stringify(counter)},'started\\n');\nconst watcher=fs.watch(${JSON.stringify(fixture.directory)},()=>{if(fs.existsSync(${JSON.stringify(release)})){watcher.close();console.log('https://claude.ai/code?bridge=env_synthetic');}});\nif(fs.existsSync(${JSON.stringify(release)})){watcher.close();console.log('https://claude.ai/code?bridge=env_synthetic');}\nsetInterval(()=>{},60000);\n`,
      { mode: 0o700 },
    );
    const source = fileURLToPath(
      new URL('./remote-control.ts', import.meta.url),
    );
    const loader = fileURLToPath(import.meta.resolve('tsx/esm'));
    const script = `const {startRemoteControl}=await import(${JSON.stringify(source)});try{const result=await startRemoteControl('fixed','tg:123',process.cwd());console.log(JSON.stringify({ok:result.ok}));}catch{console.log(JSON.stringify({held:true}));}`;
    const args = ['--import', loader, '--input-type=module', '-e', script];
    const env = { PATH: bin, HOME: fixture.directory, TZ: 'UTC' };
    const stateDir = path.join(fixture.directory, 'data');
    fs.mkdirSync(stateDir, { mode: 0o755 });
    const gate = path.join(stateDir, 'remote-control-starting.json');
    let gateWatcher: fs.FSWatcher;
    const gateReady = new Promise<void>((resolve) => {
      gateWatcher = fs.watch(stateDir, (_event, name) => {
        if (String(name) === 'remote-control-starting.json') resolve();
      });
    });
    const first = spawn(process.execPath, args, {
      cwd: fixture.directory,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let output = '';
    first.stdout.on('data', (chunk) => {
      output += chunk.toString();
      if (output.length > 65536) first.kill();
    });
    first.stderr.resume();
    const done = new Promise<number | null>((resolve, reject) => {
      const timeout = setTimeout(() => {
        first.kill();
        reject(
          new Error(
            'Synthetic Remote Control fixture exceeded its process deadline.',
          ),
        );
      }, 10000);
      first.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      first.on('close', (code) => {
        clearTimeout(timeout);
        resolve(code);
      });
    });
    const [operation] = await Promise.allSettled([
      (async () => {
        await Promise.race([
          gateReady,
          done.then(() => {
            throw new Error(
              'Synthetic launch ended before reservation notification.',
            );
          }),
        ]);
        expect(fs.existsSync(gate)).toBe(true);
        const second = spawnSync(process.execPath, args, {
          cwd: fixture.directory,
          env,
          encoding: 'utf8',
          timeout: 4000,
          maxBuffer: 65536,
        });
        expect(second.error).toBeUndefined();
        expect(second.status).toBe(0);
        expect(second.stdout).toContain('"held":true');
        fs.writeFileSync(release, 'release');
        expect(await done).toBe(0);
        expect(output).toContain('"ok":true');
        expect(fs.readFileSync(counter, 'utf8')).toBe('started\n');
        expect(fs.existsSync(gate)).toBe(false);
        expect(
          fs
            .readdirSync(stateDir)
            .filter((name) => name.startsWith('.remote-control-output-')),
        ).toEqual([]);
      })(),
    ]);
    const [cleanup] = await Promise.allSettled([
      (async () => {
        gateWatcher!.close();
        first.kill();
        if (fs.existsSync(pidFile)) {
          const pid = Number(fs.readFileSync(pidFile, 'utf8'));
          if (!Number.isSafeInteger(pid) || pid <= 0)
            throw new Error('Synthetic child PID is invalid.');
          try {
            nativeKill(-pid, 'SIGTERM');
          } catch (error) {
            if (
              !(
                error instanceof Error &&
                'code' in error &&
                error.code === 'ESRCH'
              )
            )
              throw error;
          }
        }
      })(),
    ]);
    if (operation.status === 'rejected' && cleanup.status === 'rejected')
      throw new AggregateError(
        [operation.reason, cleanup.reason],
        'Synthetic launch and cleanup failed.',
      );
    if (operation.status === 'rejected') throw operation.reason;
    if (cleanup.status === 'rejected') throw cleanup.reason;
  }, 15000);
  it('handles an actual asynchronous missing-executable launch without an uncaught child error', async () => {
    vi.stubEnv('PATH', fixture.directory);
    const close = vi.spyOn(fs, 'closeSync');
    const result = await owner.startRemoteControl(
      'fixed-owner',
      'tg:123',
      fixture.directory,
    );
    expect(result).toMatchObject({
      ok: false,
      error: expect.stringContaining('ENOENT'),
    });
    expect(close).toHaveBeenCalledTimes(5);
    expect(owner.getActiveSession()).toBeNull();
    expect(process.kill).not.toHaveBeenCalled();
    expect(
      fs.existsSync(path.join(fixture.directory, 'remote-control.json')),
    ).toBe(false);
  });

  it.each([
    '{broken',
    JSON.stringify({ pid: 'invalid' }),
    JSON.stringify({ schema_version: 1, pid: 'invalid' }),
  ])(
    'retains malformed gate bytes and refuses a subsequent launch for %s',
    async (raw) => {
      const state = path.join(fixture.directory, 'remote-control.json');
      fs.writeFileSync(state, raw);
      expect(owner.restoreRemoteControl).toThrow(
        'occupied-session state is malformed',
      );
      await expect(
        owner.startRemoteControl('fixed', 'tg:123', fixture.directory),
      ).rejects.toThrow('occupied-session state is malformed');
      expect(fs.readFileSync(state, 'utf8')).toBe(raw);
      expect(fs.readdirSync(fixture.directory)).toEqual([
        'remote-control.json',
      ]);
      expect(process.kill).not.toHaveBeenCalled();
    },
  );

  it('preserves legacy bytes on failed replacement and can retry the migration', () => {
    const state = path.join(fixture.directory, 'remote-control.json');
    const original = JSON.stringify(legacy);
    fs.writeFileSync(state, original);
    const denied = Object.assign(new Error('fixed replacement denied'), {
      code: 'EACCES',
    });
    const rename = vi.spyOn(fs, 'renameSync').mockImplementationOnce(() => {
      throw denied;
    });
    expect(owner.restoreRemoteControl).toThrow(denied);
    expect(fs.readFileSync(state, 'utf8')).toBe(original);
    expect(fs.readdirSync(fixture.directory)).toEqual(['remote-control.json']);
    expect(owner.getActiveSession()).toBeNull();
    rename.mockRestore();
    owner.restoreRemoteControl();
    expect(JSON.parse(fs.readFileSync(state, 'utf8'))).toEqual({
      schema_version: 1,
      ...legacy,
    });
    expect(fs.statSync(state).mode & 0o777).toBe(0o600);
    expect(owner.getActiveSession()).toEqual(legacy);
  });

  it('holds unknown-version bytes without process or filesystem mutation', () => {
    const state = path.join(fixture.directory, 'remote-control.json');
    const raw = JSON.stringify({ schema_version: 2, ...legacy });
    fs.writeFileSync(state, raw);
    expect(owner.restoreRemoteControl).toThrow(
      'Unsupported Remote Control state version',
    );
    expect(fs.readFileSync(state, 'utf8')).toBe(raw);
    expect(fs.readdirSync(fixture.directory)).toEqual(['remote-control.json']);
    expect(process.kill).not.toHaveBeenCalled();
  });
});

it('fsyncs a migrated state before replacement and its parent before returning tracked authority', () => {
  const state = path.join(fixture.directory, 'remote-control.json');
  fs.writeFileSync(state, JSON.stringify(legacy), { mode: 0o600 });
  const opened = new Map<number, string>();
  const events: string[] = [];
  const open = fs.openSync.bind(fs),
    sync = fs.fsyncSync.bind(fs),
    rename = fs.renameSync.bind(fs);
  vi.spyOn(fs, 'openSync').mockImplementation((file, flags, mode) => {
    const fd = open(file, flags, mode);
    opened.set(fd, String(file));
    return fd;
  });
  vi.spyOn(fs, 'fsyncSync').mockImplementation((fd) => {
    events.push(
      opened.get(fd) === fixture.directory ? 'parent_sync' : 'file_sync',
    );
    sync(fd);
  });
  vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
    events.push('rename');
    rename(from, to);
  });
  owner.restoreRemoteControl();
  expect(events).toEqual(['file_sync', 'rename', 'parent_sync']);
  expect(JSON.parse(fs.readFileSync(state, 'utf8'))).toEqual({
    schema_version: 1,
    ...legacy,
  });
});
