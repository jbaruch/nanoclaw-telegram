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

describe('private Remote Control native identity owner', () => {
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
        const saved = path.join(stateDir, 'remote-control.json');
        expect(JSON.parse(fs.readFileSync(saved, 'utf8'))).toMatchObject({
          schema_version: 2,
          identity: {
            kind:
              process.platform === 'linux'
                ? 'linux-birth'
                : 'darwin-command-nonce',
          },
        });
        expect(fs.statSync(saved).mode & 0o777).toBe(0o600);
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
        /occupied-session state is malformed|Legacy Remote Control identity is unproved/,
      );
      await expect(
        owner.startRemoteControl('fixed', 'tg:123', fixture.directory),
      ).rejects.toThrow(
        /occupied-session state is malformed|Legacy Remote Control identity is unproved/,
      );
      expect(fs.readFileSync(state, 'utf8')).toBe(raw);
      expect(fs.readdirSync(fixture.directory)).toEqual([
        'remote-control.json',
      ]);
      expect(process.kill).not.toHaveBeenCalled();
    },
  );

  it.each([undefined, 1])(
    'holds legacy schema %s unchanged without adopting a current PID or replacing state',
    (version) => {
      const state = path.join(fixture.directory, 'remote-control.json');
      const original = JSON.stringify({
        ...(version === undefined ? {} : { schema_version: version }),
        ...legacy,
      });
      fs.writeFileSync(state, original, { mode: 0o600 });
      const rename = vi.spyOn(fs, 'renameSync');
      const sync = vi.spyOn(fs, 'fsyncSync');
      for (let attempt = 0; attempt < 2; attempt++)
        expect(owner.restoreRemoteControl).toThrow(
          'Legacy Remote Control identity is unproved',
        );
      expect(fs.readFileSync(state, 'utf8')).toBe(original);
      expect(fs.readdirSync(fixture.directory)).toEqual([
        'remote-control.json',
      ]);
      expect(owner.getActiveSession()).toBeNull();
      expect(rename).not.toHaveBeenCalled();
      expect(sync).not.toHaveBeenCalled();
      expect(process.kill).not.toHaveBeenCalled();
    },
  );

  it('holds unknown-version bytes without process or filesystem mutation', () => {
    const state = path.join(fixture.directory, 'remote-control.json');
    const raw = JSON.stringify({ schema_version: 99, ...legacy });
    fs.writeFileSync(state, raw);
    expect(owner.restoreRemoteControl).toThrow(
      'Unsupported Remote Control state version',
    );
    expect(fs.readFileSync(state, 'utf8')).toBe(raw);
    expect(fs.readdirSync(fixture.directory)).toEqual(['remote-control.json']);
    expect(process.kill).not.toHaveBeenCalled();
  });
});

it('retains actual invalid UTF-8 owner bytes without adopting, signaling, or clearing authority', () => {
  const state = path.join(fixture.directory, 'remote-control.json');
  const bytes = Buffer.from([
    0x7b, 0x22, 0x70, 0x69, 0x64, 0x22, 0x3a, 0xff, 0x7d,
  ]);
  fs.writeFileSync(state, bytes, { mode: 0o600 });
  expect(owner.restoreRemoteControl).toThrow('not valid UTF-8');
  expect(fs.readFileSync(state)).toEqual(bytes);
  expect(process.kill).not.toHaveBeenCalled();
});

async function syntheticCli(body: string): Promise<void> {
  vi.mocked(process.kill).mockRestore();
  const bin = path.join(fixture.directory, 'bin');
  fs.mkdirSync(bin);
  fs.writeFileSync(
    path.join(bin, 'claude'),
    `#!${process.execPath}\n${body}\n`,
    { mode: 0o700 },
  );
  vi.stubEnv('PATH', bin);
}

it('fsyncs fresh native-bound state before replacement and parent before returning tracked authority', async () => {
  await syntheticCli(
    "console.log('https://claude.ai/code?bridge=env_synthetic');setInterval(()=>{},60000);",
  );
  const opened = new Map<number, string>(),
    events: string[] = [];
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
      opened.get(fd)?.endsWith('state.json')
        ? 'state_sync'
        : opened.get(fd) === fixture.directory
          ? 'parent_sync'
          : 'other_sync',
    );
    sync(fd);
  });
  vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
    events.push('rename');
    rename(from, to);
  });
  try {
    await expect(
      owner.startRemoteControl('fixed', 'tg:123', fixture.directory),
    ).resolves.toMatchObject({ ok: true });
    const at = events.indexOf('rename');
    expect(events.slice(at - 1, at + 2)).toEqual([
      'state_sync',
      'rename',
      'parent_sync',
    ]);
    expect(owner.getActiveSession()?.identity.kind).toBe(
      process.platform === 'linux' ? 'linux-birth' : 'darwin-command-nonce',
    );
  } finally {
    await owner.stopRemoteControl();
  }
}, 15000);

it('retires the genuine newly owned child if native-bound state replacement fails, without tracking or erasing authority', async () => {
  await syntheticCli(
    "console.log('https://claude.ai/code?bridge=env_synthetic');setInterval(()=>{},60000);",
  );
  const denied = Object.assign(new Error('fixed state replacement denied'), {
    code: 'EACCES',
  });
  vi.spyOn(fs, 'renameSync').mockImplementationOnce(() => {
    throw denied;
  });
  await expect(
    owner.startRemoteControl('fixed', 'tg:123', fixture.directory),
  ).rejects.toBe(denied);
  expect(owner.getActiveSession()).toBeNull();
  expect(
    fs.existsSync(path.join(fixture.directory, 'remote-control.json')),
  ).toBe(false);
  expect(
    fs.existsSync(path.join(fixture.directory, 'remote-control-starting.json')),
  ).toBe(false);
}, 15000);

it.each([
  JSON.stringify(legacy),
  JSON.stringify({ schema_version: 1, ...legacy }),
  JSON.stringify({ schema_version: 99, ...legacy }),
  '{broken',
])(
  'returns from expected startup hold while retaining the exact record and refusing later starts %s',
  async (bytes) => {
    const state = path.join(fixture.directory, 'remote-control.json');
    fs.writeFileSync(state, bytes, { mode: 0o600 });
    expect(() => owner.restoreRemoteControlAtStartup()).not.toThrow();
    expect(fs.readFileSync(state, 'utf8')).toBe(bytes);
    expect(owner.getActiveSession()).toBeNull();
    await expect(
      owner.startRemoteControl('fixed', 'tg:123', fixture.directory),
    ).rejects.toThrow();
    expect(process.kill).not.toHaveBeenCalled();
  },
);

it('does not conceal an unexpected state reader defect during host startup', () => {
  const defect = new TypeError('fixed unexpected reader defect');
  vi.spyOn(fs, 'openSync').mockImplementationOnce(() => {
    throw defect;
  });
  expect(owner.restoreRemoteControlAtStartup).toThrow(defect);
});

it('prepares only a verified legacy same-writer DATA_DIR before native launch and preserves payload bytes', async () => {
  await syntheticCli(
    "console.log('https://claude.ai/code?bridge=env_synthetic');setInterval(()=>{},60000);",
  );
  fs.chmodSync(fixture.directory, 0o775);
  const retained = path.join(fixture.directory, 'retained-owner-data');
  fs.writeFileSync(retained, 'unchanged private data', { mode: 0o600 });
  try {
    await expect(
      owner.startRemoteControl('fixed', 'tg:123', fixture.directory),
    ).resolves.toMatchObject({ ok: true });
    expect(fs.statSync(fixture.directory).mode & 0o777).toBe(0o755);
    expect(fs.readFileSync(retained, 'utf8')).toBe('unchanged private data');
  } finally {
    await owner.stopRemoteControl();
  }
}, 15000);

it('refuses an aliased DATA_DIR without changing foreign metadata or launching a process', async () => {
  const owned = fixture.directory,
    foreign = owned + '-foreign';
  fs.mkdirSync(foreign, { mode: 0o775 });
  fs.chmodSync(foreign, 0o775);
  fs.rmdirSync(owned);
  fs.symlinkSync(foreign, owned);
  try {
    await expect(
      owner.startRemoteControl('fixed', 'tg:123', owned),
    ).rejects.toThrow('storage is not owned');
    expect(fs.statSync(foreign).mode & 0o777).toBe(0o775);
    expect(fs.readdirSync(foreign)).toEqual([]);
    expect(process.kill).not.toHaveBeenCalled();
  } finally {
    fs.unlinkSync(owned);
    fs.mkdirSync(owned);
    fs.rmdirSync(foreign);
  }
});
