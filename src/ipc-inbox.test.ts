import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { createIpcInbox } from './ipc-inbox.js';
let base: string;
beforeEach(() => {
  base = fs.mkdtempSync(path.join(os.tmpdir(), 'telegram-ipc-owner-'));
});
afterEach(() => {
  vi.restoreAllMocks();
  fs.rmSync(base, { recursive: true, force: true });
});
function source(group = 'main') {
  const p = path.join(base, group, 'messages');
  fs.mkdirSync(p, { recursive: true });
  return p;
}
function body(text = 'fixed') {
  return JSON.stringify({ type: 'message', chatJid: 'tg:123', text });
}
it('processes the normal runtime-writer layout and recovers pending snapshots after restart', () => {
  const directory = source();
  fs.writeFileSync(path.join(directory, 'fixed.json'), body());
  const owner = createIpcInbox(base);
  owner.capture('main', 'messages');
  expect(owner.pending().claims).toHaveLength(1);
  const resumed = createIpcInbox(base).pending();
  expect(resumed.held).toBe(0);
  expect(resumed.claims).toHaveLength(1);
  expect(resumed.claims[0].read()).toBe(body());
  resumed.claims[0].dispatch();
  resumed.claims[0].complete();
  expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 0 });
});
it('refuses group aliases and captures final-channel ABA aliases without touching foreign requests', () => {
  const foreign = source('other');
  const file = path.join(foreign, 'private.json');
  fs.writeFileSync(file, body('private'));
  fs.symlinkSync(path.join(base, 'other'), path.join(base, 'alias'));
  const owner = createIpcInbox(base);
  expect(() => owner.capture('alias', 'messages')).toThrow('parent identity');
  const own = source();
  fs.writeFileSync(path.join(own, 'own.json'), body());
  const rename = fs.renameSync.bind(fs);
  vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
    if (from === own) {
      rename(own, path.join(base, 'main', 'old-own'));
      fs.symlinkSync(foreign, own);
    }
    return rename(from, to);
  });
  owner.capture('main', 'messages');
  expect(owner.pending()).toEqual({ claims: [], held: 1 });
  expect(fs.readFileSync(file, 'utf8')).toBe(body('private'));
});
it.each(['symlink', 'hardlink'] as const)(
  'quarantines a %s leaf without reading or deleting its foreign target',
  (kind) => {
    const foreign = source('other'),
      file = path.join(foreign, 'private.json');
    fs.writeFileSync(file, body('private'));
    const own = source(),
      alias = path.join(own, 'alias.json');
    if (kind === 'symlink') fs.symlinkSync(file, alias);
    else fs.linkSync(file, alias);
    const owner = createIpcInbox(base);
    owner.capture('main', 'messages');
    expect(owner.pending()).toEqual({ claims: [], held: 0 });
    expect(fs.readdirSync(path.join(base, '.claimed-v1'))).toEqual([]);
    const evidence = path.join(
      base,
      'errors',
      fs.readdirSync(path.join(base, 'errors'))[0],
      'captured',
    );
    expect(fs.lstatSync(evidence).isSymbolicLink()).toBe(kind === 'symlink');
    if (kind === 'hardlink')
      expect(fs.lstatSync(evidence).ino).toBe(fs.lstatSync(file).ino);
    expect(fs.readFileSync(file, 'utf8')).toBe(body('private'));
  },
);
it('a guest-held old file descriptor cannot mutate the pending or in-flight snapshot', () => {
  const directory = source(),
    file = path.join(directory, 'fixed.json');
  fs.writeFileSync(file, body());
  const fd = fs.openSync(file, 'r+');
  try {
    const owner = createIpcInbox(base);
    owner.capture('main', 'messages');
    const claim = owner.pending().claims[0];
    fs.writeSync(
      fd,
      Buffer.from(body('changed')),
      0,
      body('changed').length,
      0,
    );
    expect(claim.read()).toBe(body());
    claim.dispatch();
    fs.ftruncateSync(fd, 0);
    expect(claim.read()).toBe(body());
    expect(fs.lstatSync(claim.filePath).nlink).toBe(1);
    claim.complete();
  } finally {
    fs.closeSync(fd);
  }
});
it('holds unknown in-flight work across restart while an independent batch progresses', () => {
  for (const group of ['main', 'other']) {
    const directory = source(group);
    fs.writeFileSync(path.join(directory, 'fixed.json'), body(group));
  }
  const owner = createIpcInbox(base);
  owner.capture('main', 'messages');
  owner.capture('other', 'messages');
  const first = owner
    .pending()
    .claims.find((claim) => claim.sourceGroup === 'main')!;
  first.dispatch();
  const resumed = createIpcInbox(base).pending();
  expect(resumed.held).toBe(1);
  expect(resumed.claims).toHaveLength(1);
  expect(resumed.claims[0].sourceGroup).toBe('other');
  resumed.claims[0].dispatch();
  resumed.claims[0].complete();
  expect(fs.readFileSync(first.filePath, 'utf8')).toBe(body('main'));
  expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 1 });
});
it('recovers a durably committed snapshot after a crash before captured-inode cleanup', () => {
  const directory = source();
  fs.writeFileSync(path.join(directory, 'fixed.json'), body());
  const owner = createIpcInbox(base);
  owner.capture('main', 'messages');
  const unlink = fs.unlinkSync.bind(fs);
  const fault = vi.spyOn(fs, 'unlinkSync').mockImplementation((p) => {
    if (String(p).endsWith('/captured'))
      throw Object.assign(new Error('fixed crash seam'), { code: 'EIO' });
    unlink(p);
  });
  expect(owner.pending()).toEqual({ claims: [], held: 1 });
  fault.mockRestore();
  const resumed = createIpcInbox(base).pending();
  expect(resumed.held).toBe(0);
  expect(resumed.claims[0].read()).toBe(body());
});
it('preserves unknown owner metadata instead of treating it as an empty inbox', () => {
  const directory = source();
  fs.writeFileSync(path.join(directory, 'fixed.json'), body());
  const owner = createIpcInbox(base);
  owner.capture('main', 'messages');
  const root = path.join(base, '.claimed-v1');
  const batch = path.join(root, fs.readdirSync(root)[0]);
  const future = JSON.stringify({
    schema_version: 2,
    source_group: 'main',
    channel: 'messages',
  });
  fs.writeFileSync(path.join(batch, 'owner.json'), future);
  expect(owner.pending()).toEqual({ claims: [], held: 1 });
  expect(fs.readFileSync(path.join(batch, 'owner.json'), 'utf8')).toBe(future);
});

it('holds an over-budget channel without parsing, evicting or dispatching its requests', () => {
  const directory = source();
  for (let index = 0; index < 1025; index++)
    fs.writeFileSync(path.join(directory, `fixed-${index}.json`), '{}');
  const owner = createIpcInbox(base);
  owner.capture('main', 'messages');
  expect(owner.pending()).toEqual({ claims: [], held: 1 });
  const root = path.join(base, '.claimed-v1'),
    batch = path.join(root, fs.readdirSync(root)[0]);
  expect(fs.readdirSync(path.join(batch, 'inbox'))).toHaveLength(1025);
});

it('preserves owner evidence if another scan begins dispatch during empty-slot cleanup', () => {
  const directory = source();
  fs.writeFileSync(path.join(directory, 'fixed.json'), body());
  const owner = createIpcInbox(base);
  owner.capture('main', 'messages');
  const claim = owner.pending().claims[0],
    pending = claim.filePath,
    slot = path.dirname(pending);
  const ownerFile = path.join(slot, 'owner.json'),
    before = fs.readFileSync(ownerFile, 'utf8');
  const stat = fs.lstatSync.bind(fs);
  let raced = false;
  vi.spyOn(fs, 'lstatSync').mockImplementation((file, options) => {
    if (String(file) === pending && !raced) {
      raced = true;
      fs.renameSync(pending, path.join(slot, 'in_flight'));
    }
    return stat(file, options);
  });
  expect(owner.pending()).toEqual({ claims: [], held: 1 });
  expect(raced).toBe(true);
  expect(fs.readFileSync(ownerFile, 'utf8')).toBe(before);
  expect(fs.readFileSync(path.join(slot, 'in_flight'), 'utf8')).toBe(body());
});

it('actual host snapshot creation remains admissible under a permissive inherited umask', () => {
  const runner = fileURLToPath(
    new URL('./container-runner.ts', import.meta.url),
  );
  const owner = fileURLToPath(new URL('./ipc-inbox.ts', import.meta.url));
  const script = `import fs from 'node:fs';import path from 'node:path';import {writeTasksSnapshot} from ${JSON.stringify(runner)};import {createIpcInbox} from ${JSON.stringify(owner)};
    process.umask(0o002);writeTasksSnapshot('main',true,[]);
    const base=path.join(process.cwd(),'data','ipc');const group=path.join(base,'main');
    for(const p of [base,group]){if((fs.lstatSync(p).mode&0o777)!==0o755)throw new Error('Unsafe native-created mode');}
    const inbox=createIpcInbox(base);inbox.capture('main','messages');
    if(inbox.pending().held!==0)throw new Error('Native writer refused');process.stdout.write('ADMITTED');`;
  const child = spawnSync(
    process.execPath,
    [
      '--import',
      fileURLToPath(import.meta.resolve('tsx/esm')),
      '--input-type',
      'module',
      '-e',
      script,
    ],
    {
      cwd: base,
      env: { PATH: process.env.PATH ?? '', HOME: base, TZ: 'UTC' },
      encoding: 'utf8',
      timeout: 10000,
      maxBuffer: 262144,
    },
  );
  expect(child.error).toBeUndefined();
  expect(child.status, child.stderr).toBe(0);
  expect(child.stdout).toContain('ADMITTED');
});

it('holds invalid UTF-8 private metadata without widening or removing its gate', () => {
  const directory = source();
  fs.writeFileSync(path.join(directory, 'fixed.json'), body());
  const inbox = createIpcInbox(base);
  inbox.capture('main', 'messages');
  const root = path.join(base, '.claimed-v1');
  const metadata = path.join(root, fs.readdirSync(root)[0], 'owner.json');
  const original = Buffer.from([123, 34, 255, 34, 58, 49, 125]);
  fs.writeFileSync(metadata, original);
  expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 1 });
  expect(fs.readFileSync(metadata)).toEqual(original);
});

it('tightens only verified runtime-owned structural parents from the legacy permissive layout', () => {
  const channel = source();
  fs.chmodSync(base, 0o775);
  fs.chmodSync(path.dirname(channel), 0o775);
  fs.chmodSync(channel, 0o775);
  fs.writeFileSync(path.join(channel, 'fixed.json'), body());
  const inbox = createIpcInbox(base);
  inbox.capture('main', 'messages');
  expect(fs.statSync(base).mode & 0o777).toBe(0o755);
  expect(fs.statSync(path.dirname(channel)).mode & 0o777).toBe(0o755);
  const result = inbox.pending();
  expect(result.held).toBe(0);
  expect(result.claims[0].read()).toBe(body());
});

it('never repairs a foreign-owner or aliased structural directory', () => {
  const channel = source();
  fs.chmodSync(base, 0o775);
  const lstat = fs.lstatSync.bind(fs);
  vi.spyOn(fs, 'lstatSync').mockImplementation((file, options) => {
    const stat = lstat(file, options as never);
    return file === base
      ? Object.assign(stat, { uid: process.geteuid!() + 1 })
      : stat;
  });
  expect(() => createIpcInbox(base)).toThrow('parent identity');
  expect(fs.statSync(base).mode & 0o777).toBe(0o775);
  expect(fs.readdirSync(channel)).toEqual([]);
});

it('avoids idle claim journal writes with the bounded native work hint', () => {
  source();
  const inbox = createIpcInbox(base);
  const sync = vi.spyOn(fs, 'fsyncSync');
  inbox.capture('main', 'messages');
  expect(sync).not.toHaveBeenCalled();
  expect(inbox.pending()).toEqual({ claims: [], held: 0 });
});

it('quarantines a real claimed FIFO without blocking independent regular requests or deleting evidence', () => {
  const owner = fileURLToPath(new URL('./ipc-inbox.ts', import.meta.url));
  const script = `import fs from 'node:fs';import path from 'node:path';import {execFileSync} from 'node:child_process';import {createIpcInbox} from ${JSON.stringify(owner)};
    const base=process.cwd();
    for(const group of ['blocked','healthy'])fs.mkdirSync(path.join(base,group,'messages'),{recursive:true,mode:0o755});
    execFileSync('mkfifo',[path.join(base,'blocked','messages','fixed.json')]);
    fs.writeFileSync(path.join(base,'healthy','messages','fixed.json'),'fixed regular request');
    const inbox=createIpcInbox(base);inbox.capture('blocked','messages');inbox.capture('healthy','messages');
    const result=inbox.pending();
    const root=path.join(base,'errors');let retained=false;
    for(const name of fs.readdirSync(root)){const dir=path.join(root,name);if(JSON.parse(fs.readFileSync(path.join(dir,'owner.json'),'utf8')).source_group!=='blocked')continue;
      retained=fs.lstatSync(path.join(dir,'captured')).isFIFO();}
    process.stdout.write(JSON.stringify({held:result.held,claims:result.claims.map(c=>({group:c.sourceGroup,body:c.read()})),fifo_retained:retained}));`;
  const child = spawnSync(
    process.execPath,
    [
      '--import',
      fileURLToPath(import.meta.resolve('tsx/esm')),
      '--input-type',
      'module',
      '-e',
      script,
    ],
    {
      cwd: base,
      env: { PATH: process.env.PATH ?? '', HOME: base, TZ: 'UTC' },
      encoding: 'utf8',
      timeout: 5000,
      killSignal: 'SIGKILL',
      maxBuffer: 262144,
    },
  );
  expect(child.error).toBeUndefined();
  expect(child.status, child.stderr).toBe(0);
  expect(JSON.parse(child.stdout)).toEqual({
    held: 0,
    claims: [{ group: 'healthy', body: 'fixed regular request' }],
    fifo_retained: true,
  });
});

it('falls back to secure capture when the directory work hint has a native read failure', () => {
  const directory = source();
  fs.writeFileSync(path.join(directory, 'fixed.json'), body());
  const inbox = createIpcInbox(base),
    open = fs.opendirSync.bind(fs);
  vi.spyOn(fs, 'opendirSync').mockImplementation((file, options) => {
    if (file === directory || String(file).startsWith('/proc/self/fd/'))
      throw Object.assign(new Error('fixed hint failure'), { code: 'EIO' });
    return open(file, options);
  });
  inbox.capture('main', 'messages');
  const result = inbox.pending();
  expect(result.held).toBe(0);
  expect(result.claims[0].read()).toBe(body());
});

it('closes work-hint handles while retaining both an unexpected read defect and close failure', () => {
  const directory = source();
  const inbox = createIpcInbox(base);
  const readFailure = new TypeError('fixed hint read defect'),
    closeFailure = new TypeError('fixed hint close defect');
  const open = fs.opendirSync.bind(fs);
  vi.spyOn(fs, 'opendirSync').mockImplementation((file, options) => {
    const dir = open(file, options);
    if (file === directory || String(file).startsWith('/proc/self/fd/')) {
      vi.spyOn(dir, 'readSync').mockImplementation(() => {
        throw readFailure;
      });
      const close = dir.closeSync.bind(dir);
      vi.spyOn(dir, 'closeSync').mockImplementation(() => {
        close();
        throw closeFailure;
      });
    }
    return dir;
  });
  expect(() => inbox.capture('main', 'messages')).toThrowError(
    expect.objectContaining({
      name: 'SuppressedError',
      error: closeFailure,
      suppressed: readFailure,
    }),
  );
});

it('keeps the source channel intact when the new batch parent cannot be durably linked', () => {
  const directory = source(),
    original = body();
  fs.writeFileSync(path.join(directory, 'fixed.json'), original);
  const inbox = createIpcInbox(base),
    root = path.join(base, '.claimed-v1');
  const descriptors = new Map<number, string>(),
    open = fs.openSync.bind(fs),
    sync = fs.fsyncSync.bind(fs);
  vi.spyOn(fs, 'openSync').mockImplementation((file, flags, mode) => {
    const fd = open(file, flags, mode);
    descriptors.set(fd, String(file));
    return fd;
  });
  const failure = Object.assign(
    new Error('fixed batch parent durability failure'),
    { code: 'EIO' },
  );
  const fault = vi.spyOn(fs, 'fsyncSync').mockImplementation((fd) => {
    if (descriptors.get(fd) === root) throw failure;
    sync(fd);
  });
  expect(() => inbox.capture('main', 'messages')).toThrow(failure);
  expect(fs.readFileSync(path.join(directory, 'fixed.json'), 'utf8')).toBe(
    original,
  );
  fault.mockRestore();
  inbox.capture('main', 'messages');
  const resumed = inbox.pending();
  expect(resumed.held).toBe(0);
  expect(resumed.claims).toHaveLength(1);
  expect(resumed.claims[0].read()).toBe(original);
});

it('keeps the captured source intact until its request-slot parent is durable', () => {
  const directory = source(),
    original = body();
  fs.writeFileSync(path.join(directory, 'fixed.json'), original);
  const inbox = createIpcInbox(base);
  inbox.capture('main', 'messages');
  const root = path.join(base, '.claimed-v1'),
    batch = path.join(root, fs.readdirSync(root)[0]),
    requests = path.join(batch, 'requests');
  const descriptors = new Map<number, string>(),
    open = fs.openSync.bind(fs),
    sync = fs.fsyncSync.bind(fs);
  vi.spyOn(fs, 'openSync').mockImplementation((file, flags, mode) => {
    const fd = open(file, flags, mode);
    descriptors.set(fd, String(file));
    return fd;
  });
  const fault = vi.spyOn(fs, 'fsyncSync').mockImplementation((fd) => {
    if (descriptors.get(fd) === requests)
      throw Object.assign(new Error('fixed slot parent durability failure'), {
        code: 'EIO',
      });
    sync(fd);
  });
  expect(inbox.pending()).toEqual({ claims: [], held: 1 });
  expect(fs.readFileSync(path.join(batch, 'inbox', 'fixed.json'), 'utf8')).toBe(
    original,
  );
  fault.mockRestore();
  const resumed = inbox.pending();
  expect(resumed.held).toBe(0);
  expect(resumed.claims).toHaveLength(1);
  expect(resumed.claims[0].read()).toBe(original);
});

it('preserves original channel filename order through opaque snapshots and restart', () => {
  const messages = source();
  const tasks = path.join(base, 'main', 'tasks');
  fs.mkdirSync(tasks);
  for (const channel of [messages, tasks])
    for (const name of ['003-final.json', '001-first.json', '002-middle.json'])
      fs.writeFileSync(path.join(channel, name), body(name));
  const inbox = createIpcInbox(base);
  inbox.capture('main', 'messages');
  inbox.capture('main', 'tasks');
  inbox.pending();
  const resumed = createIpcInbox(base).pending();
  expect(resumed.held).toBe(0);
  expect(
    resumed.claims.map((c) => [c.channel, c.file, JSON.parse(c.read()).text]),
  ).toEqual(
    ['messages', 'tasks'].flatMap((channel) =>
      ['001-first.json', '002-middle.json', '003-final.json'].map((name) => [
        channel,
        name,
        name,
      ]),
    ),
  );
});

it('quarantines non-request trees and aliases without leaking batch capacity across restart', () => {
  const foreign = path.join(base, 'foreign-evidence');
  fs.writeFileSync(foreign, 'untouched foreign bytes');
  for (let cycle = 0; cycle < 3; cycle++) {
    const directory = source();
    fs.writeFileSync(path.join(directory, 'request.json'), body(String(cycle)));
    fs.writeFileSync(path.join(directory, 'stray.bin'), 'retained stray bytes');
    fs.mkdirSync(path.join(directory, 'stray-directory'));
    fs.writeFileSync(
      path.join(directory, 'stray-directory', 'child'),
      'retained child bytes',
    );
    fs.symlinkSync(foreign, path.join(directory, 'stray-alias'));
    const inbox = createIpcInbox(base);
    inbox.capture('main', 'messages');
    const scan = inbox.pending();
    expect(scan.held).toBe(0);
    expect(scan.claims).toHaveLength(1);
    expect(scan.claims[0].read()).toBe(body(String(cycle)));
    scan.claims[0].dispatch();
    scan.claims[0].complete();
    expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 0 });
    expect(fs.readdirSync(path.join(base, '.claimed-v1'))).toEqual([]);
  }
  const quarantines = fs.readdirSync(path.join(base, 'errors'));
  expect(quarantines).toHaveLength(9);
  for (const name of quarantines) {
    const evidence = path.join(base, 'errors', name),
      record = JSON.parse(
        fs.readFileSync(path.join(evidence, 'owner.json'), 'utf8'),
      );
    expect(record).toMatchObject({
      schema_version: 1,
      source_group: 'main',
      channel: 'messages',
      reason: 'invalid_or_unreadable_input',
    });
    if (record.file === 'stray.bin')
      expect(fs.readFileSync(path.join(evidence, 'captured'), 'utf8')).toBe(
        'retained stray bytes',
      );
    else if (record.file === 'stray-directory')
      expect(
        fs.readFileSync(path.join(evidence, 'captured', 'child'), 'utf8'),
      ).toBe('retained child bytes');
    else {
      expect(record.file).toBe('stray-alias');
      expect(
        fs.lstatSync(path.join(evidence, 'captured')).isSymbolicLink(),
      ).toBe(true);
      expect(fs.readlinkSync(path.join(evidence, 'captured'))).toBe(foreign);
    }
  }
  expect(fs.readFileSync(foreign, 'utf8')).toBe('untouched foreign bytes');
  const other = source('other');
  fs.writeFileSync(path.join(other, 'independent.json'), body('independent'));
  const restarted = createIpcInbox(base);
  restarted.capture('other', 'messages');
  expect(restarted.pending().claims[0].read()).toBe(body('independent'));
});

it('holds after uncertain quarantine durability and resumes only after exact evidence is flushed', () => {
  const directory = source();
  fs.writeFileSync(path.join(directory, 'request.json'), body());
  fs.writeFileSync(path.join(directory, 'stray.bin'), 'retained stray bytes');
  const inbox = createIpcInbox(base);
  inbox.capture('main', 'messages');
  const descriptors = new Map<number, string>(),
    open = fs.openSync.bind(fs),
    sync = fs.fsyncSync.bind(fs),
    rename = fs.renameSync.bind(fs);
  let moved = false;
  vi.spyOn(fs, 'openSync').mockImplementation((file, flags, mode) => {
    const fd = open(file, flags, mode);
    descriptors.set(fd, String(file));
    return fd;
  });
  vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
    rename(from, to);
    if (
      String(to).includes('/errors/rejected-') &&
      String(to).endsWith('/captured')
    )
      moved = true;
  });
  const fault = vi.spyOn(fs, 'fsyncSync').mockImplementation((fd) => {
    if (moved && descriptors.get(fd)?.includes('/errors/rejected-'))
      throw Object.assign(new Error('fixed quarantine durability failure'), {
        code: 'EIO',
      });
    sync(fd);
  });
  expect(inbox.pending()).toEqual({ claims: [], held: 1 });
  expect(moved).toBe(true);
  const quarantine = path.join(
    base,
    'errors',
    fs.readdirSync(path.join(base, 'errors'))[0],
  );
  expect(fs.readFileSync(path.join(quarantine, 'captured'), 'utf8')).toBe(
    'retained stray bytes',
  );
  expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 1 });
  fault.mockRestore();
  const resumed = createIpcInbox(base).pending();
  expect(resumed.held).toBe(0);
  expect(resumed.claims).toHaveLength(1);
  expect(resumed.claims[0].read()).toBe(body());
  resumed.claims[0].dispatch();
  resumed.claims[0].complete();
  expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 0 });
  expect(fs.readdirSync(path.join(base, '.claimed-v1'))).toEqual([]);
  expect(fs.readFileSync(path.join(quarantine, 'captured'), 'utf8')).toBe(
    'retained stray bytes',
  );
});

it.each(['schema_version', 'source_group'] as const)(
  'preserves incompatible quarantine %s and does not dispatch its linked batch',
  (field) => {
    const directory = source();
    fs.writeFileSync(path.join(directory, 'request.json'), body());
    fs.writeFileSync(path.join(directory, 'stray.bin'), 'retained stray bytes');
    const inbox = createIpcInbox(base);
    inbox.capture('main', 'messages');
    const nativeRename = fs.renameSync.bind(fs);
    const fault = vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
      if (
        String(to).includes('/errors/rejected-') &&
        String(to).endsWith('/captured')
      )
        throw Object.assign(new Error('fixed pre-move native failure'), {
          code: 'EIO',
        });
      nativeRename(from, to);
    });
    expect(inbox.pending()).toEqual({ claims: [], held: 1 });
    fault.mockRestore();
    const quarantine = path.join(
      base,
      'errors',
      fs.readdirSync(path.join(base, 'errors'))[0],
    );
    const metadata = path.join(quarantine, 'owner.json');
    const original = JSON.parse(fs.readFileSync(metadata, 'utf8'));
    const unsupported = JSON.stringify({
      ...original,
      [field]: field === 'schema_version' ? 2 : 'other',
    });
    fs.writeFileSync(metadata, unsupported);
    expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 1 });
    expect(fs.readFileSync(metadata, 'utf8')).toBe(unsupported);
    const batch = path.join(base, '.claimed-v1', original.batch_id);
    expect(
      fs.readFileSync(
        path.join(batch, 'requests', original.request_id, 'captured'),
        'utf8',
      ),
    ).toBe('retained stray bytes');
    expect(fs.existsSync(path.join(quarantine, 'captured'))).toBe(false);
  },
);

it('quarantines a nonregular JSON leaf while same-batch and other-source requests proceed', () => {
  const own = source(),
    other = source('other');
  fs.mkdirSync(path.join(own, 'a-invalid.json'));
  fs.writeFileSync(path.join(own, 'a-invalid.json', 'child'), 'retained child');
  fs.writeFileSync(path.join(own, 'b-valid.json'), body('same batch'));
  fs.writeFileSync(path.join(other, 'valid.json'), body('other source'));
  const inbox = createIpcInbox(base);
  inbox.capture('main', 'messages');
  inbox.capture('other', 'messages');
  const scan = inbox.pending();
  expect(scan.held).toBe(0);
  expect(scan.claims.map((claim) => claim.read())).toEqual([
    body('same batch'),
    body('other source'),
  ]);
  for (const claim of scan.claims) {
    claim.dispatch();
    claim.complete();
  }
  expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 0 });
  expect(fs.readdirSync(path.join(base, '.claimed-v1'))).toEqual([]);
  const evidence = path.join(
    base,
    'errors',
    fs.readdirSync(path.join(base, 'errors'))[0],
  );
  expect(
    JSON.parse(fs.readFileSync(path.join(evidence, 'owner.json'), 'utf8')),
  ).toMatchObject({
    schema_version: 1,
    source_group: 'main',
    channel: 'messages',
    file: 'a-invalid.json',
  });
  expect(
    fs.readFileSync(path.join(evidence, 'captured', 'child'), 'utf8'),
  ).toBe('retained child');
});

it.each(['EACCES', 'EIO'] as const)(
  'separates captured-leaf access refusal from uncertain storage %s',
  (code) => {
    const own = source();
    fs.writeFileSync(path.join(own, 'a-input.json'), body('retained'));
    fs.writeFileSync(path.join(own, 'b-valid.json'), body('later'));
    const inbox = createIpcInbox(base);
    inbox.capture('main', 'messages');
    const nativeOpen = fs.openSync.bind(fs);
    const fault = vi
      .spyOn(fs, 'openSync')
      .mockImplementation((file, flags, mode) => {
        if (String(file).endsWith('/captured'))
          throw Object.assign(new Error('fixed native read refusal'), { code });
        return nativeOpen(file, flags, mode);
      });
    const scan = inbox.pending();
    if (code === 'EIO') {
      expect(scan).toEqual({ claims: [], held: 1 });
      expect(fs.existsSync(path.join(base, 'errors'))).toBe(false);
    } else {
      expect(scan).toEqual({ claims: [], held: 0 });
      expect(fs.readdirSync(path.join(base, 'errors'))).toHaveLength(2);
    }
    fault.mockRestore();
    const resumed = createIpcInbox(base).pending();
    expect(resumed.held).toBe(0);
    expect(resumed.claims.map((claim) => claim.read())).toEqual(
      code === 'EIO' ? [body('retained'), body('later')] : [],
    );
  },
);

it('retains a rejected leaf and holds same-batch dispatch until post-move quarantine fsync succeeds', () => {
  const own = source();
  fs.mkdirSync(path.join(own, 'a-invalid.json'));
  fs.writeFileSync(path.join(own, 'a-invalid.json', 'child'), 'retained');
  fs.writeFileSync(path.join(own, 'b-valid.json'), body());
  const inbox = createIpcInbox(base);
  inbox.capture('main', 'messages');
  const descriptors = new Map<number, string>(),
    open = fs.openSync.bind(fs),
    sync = fs.fsyncSync.bind(fs),
    rename = fs.renameSync.bind(fs);
  let moved = false;
  vi.spyOn(fs, 'openSync').mockImplementation((file, flags, mode) => {
    const fd = open(file, flags, mode);
    descriptors.set(fd, String(file));
    return fd;
  });
  vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
    rename(from, to);
    if (
      String(to).includes('/errors/rejected-') &&
      String(to).endsWith('/captured')
    )
      moved = true;
  });
  const fault = vi.spyOn(fs, 'fsyncSync').mockImplementation((fd) => {
    if (moved && descriptors.get(fd)?.includes('/errors/rejected-'))
      throw Object.assign(new Error('fixed rejected-leaf durability failure'), {
        code: 'EIO',
      });
    sync(fd);
  });
  expect(inbox.pending()).toEqual({ claims: [], held: 1 });
  expect(moved).toBe(true);
  expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 1 });
  fault.mockRestore();
  const resumed = createIpcInbox(base).pending();
  expect(resumed.held).toBe(0);
  expect(resumed.claims).toHaveLength(1);
  expect(resumed.claims[0].read()).toBe(body());
  const evidence = path.join(
    base,
    'errors',
    fs.readdirSync(path.join(base, 'errors'))[0],
  );
  expect(
    fs.readFileSync(path.join(evidence, 'captured', 'child'), 'utf8'),
  ).toBe('retained');
});

it.each(['schema_version', 'file'] as const)(
  'holds incompatible rejected-leaf %s after interrupted custody without admitting valid siblings',
  (field) => {
    const own = source();
    fs.mkdirSync(path.join(own, 'a-invalid.json'));
    fs.writeFileSync(path.join(own, 'b-valid.json'), body());
    const inbox = createIpcInbox(base);
    inbox.capture('main', 'messages');
    const rename = fs.renameSync.bind(fs);
    const fault = vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
      if (
        String(to).includes('/errors/rejected-') &&
        String(to).endsWith('/captured')
      )
        throw Object.assign(new Error('fixed pre-move refusal'), {
          code: 'EIO',
        });
      rename(from, to);
    });
    expect(inbox.pending()).toEqual({ claims: [], held: 1 });
    fault.mockRestore();
    const evidence = path.join(
        base,
        'errors',
        fs.readdirSync(path.join(base, 'errors'))[0],
      ),
      owner = path.join(evidence, 'owner.json');
    const original = JSON.parse(fs.readFileSync(owner, 'utf8'));
    const unsupported = JSON.stringify({
      ...original,
      [field]: field === 'schema_version' ? 2 : 'other.json',
    });
    fs.writeFileSync(owner, unsupported);
    expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 1 });
    expect(fs.readFileSync(owner, 'utf8')).toBe(unsupported);
    expect(
      fs
        .lstatSync(
          path.join(
            base,
            '.claimed-v1',
            original.batch_id,
            'requests',
            original.request_id,
            'captured',
          ),
        )
        .isDirectory(),
    ).toBe(true);
    expect(fs.existsSync(path.join(evidence, 'captured'))).toBe(false);
  },
);

it.each(['slot-capture', 'leaf-quarantine'] as const)(
  'preserves late valid JSON published through the old directory during %s',
  (phase) => {
    const directory = source();
    fs.writeFileSync(path.join(directory, 'a-first.json'), body('first'));
    fs.writeFileSync(path.join(directory, 'stray.bin'), 'retained');
    const oldFD = fs.openSync(
      directory,
      fs.constants.O_RDONLY | fs.constants.O_DIRECTORY,
    );
    try {
      const inbox = createIpcInbox(base);
      inbox.capture('main', 'messages');
      const batch = path.join(
        base,
        '.claimed-v1',
        fs.readdirSync(path.join(base, '.claimed-v1'))[0],
      );
      const oldInbox = path.join(batch, 'inbox');
      const late = JSON.stringify({
        schema_version: 1,
        type: 'message',
        chatJid: 'tg:123',
        text: 'late valid input',
      });
      const unpublished = path.join(base, 'late-unpublished.tmp');
      fs.writeFileSync(unpublished, late);
      const rename = fs.renameSync.bind(fs);
      let inserted = false;
      vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
        if (
          !inserted &&
          (phase === 'slot-capture'
            ? String(from) === path.join(oldInbox, 'a-first.json')
            : (String(from).endsWith('/captured') &&
                String(to).includes('/errors/rejected-')) ||
              (String(to).includes('/errors/unrecognized-') &&
                String(to).endsWith('/inbox')))
        ) {
          inserted = true;
          // Linux uses the real held directory capability; Darwin explicitly maps
          // that synthetic bind to the same captured inode (verified below).
          expect(fs.lstatSync(oldInbox).ino).toBe(fs.fstatSync(oldFD).ino);
          rename(
            unpublished,
            process.platform === 'linux'
              ? `/proc/self/fd/${oldFD}/late.json`
              : path.join(oldInbox, 'late.json'),
          );
          fs.fsyncSync(oldFD);
        }
        rename(from, to);
      });
      const first = inbox.pending();
      expect(inserted).toBe(true);
      expect(first.held).toBe(0);
      expect(first.claims.map((claim) => claim.read())).toEqual([
        body('first'),
      ]);
      expect(fs.readFileSync(path.join(oldInbox, 'late.json'), 'utf8')).toBe(
        late,
      );
      first.claims[0].dispatch();
      first.claims[0].complete();
      const resumed = createIpcInbox(base).pending();
      expect(resumed.held).toBe(0);
      expect(resumed.claims).toHaveLength(1);
      expect(resumed.claims[0].read()).toBe(late);
      resumed.claims[0].dispatch();
      resumed.claims[0].complete();
      expect(createIpcInbox(base).pending()).toEqual({ claims: [], held: 0 });
      expect(fs.readdirSync(path.join(base, '.claimed-v1'))).toEqual([]);
    } finally {
      fs.closeSync(oldFD);
    }
  },
);

it.each([1, 2])(
  'preserves unresolved prior whole-inbox quarantine schema%s without moving new input',
  (schema) => {
    const directory = source();
    fs.writeFileSync(path.join(directory, 'valid.json'), body());
    const inbox = createIpcInbox(base);
    inbox.capture('main', 'messages');
    const batchId = fs.readdirSync(path.join(base, '.claimed-v1'))[0],
      batch = path.join(base, '.claimed-v1', batchId),
      evidence = path.join(base, 'errors', 'unrecognized-' + batchId);
    fs.mkdirSync(evidence, { recursive: true, mode: 0o700 });
    const record = JSON.stringify({
      schema_version: schema,
      source_group: 'main',
      channel: 'messages',
      batch_id: batchId,
    });
    fs.writeFileSync(path.join(evidence, 'owner.json'), record, {
      mode: 0o600,
    });
    expect(inbox.pending()).toEqual({ claims: [], held: 1 });
    expect(
      fs.readFileSync(path.join(batch, 'inbox', 'valid.json'), 'utf8'),
    ).toBe(body());
    expect(fs.readdirSync(path.join(batch, 'requests'))).toEqual([]);
    expect(fs.readFileSync(path.join(evidence, 'owner.json'), 'utf8')).toBe(
      record,
    );
    expect(fs.existsSync(path.join(evidence, 'inbox'))).toBe(false);
  },
);
