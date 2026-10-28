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
  'holds a %s leaf without reading or deleting its foreign target',
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
    expect(owner.pending()).toEqual({ claims: [], held: 1 });
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

it('refuses a real claimed FIFO without blocking independent regular requests or deleting evidence', () => {
  const owner = fileURLToPath(new URL('./ipc-inbox.ts', import.meta.url));
  const script = `import fs from 'node:fs';import path from 'node:path';import {execFileSync} from 'node:child_process';import {createIpcInbox} from ${JSON.stringify(owner)};
    const base=process.cwd();
    for(const group of ['blocked','healthy'])fs.mkdirSync(path.join(base,group,'messages'),{recursive:true,mode:0o755});
    execFileSync('mkfifo',[path.join(base,'blocked','messages','fixed.json')]);
    fs.writeFileSync(path.join(base,'healthy','messages','fixed.json'),'fixed regular request');
    const inbox=createIpcInbox(base);inbox.capture('blocked','messages');inbox.capture('healthy','messages');
    const result=inbox.pending();
    const root=path.join(base,'.claimed-v1');let retained=false;
    for(const batch of fs.readdirSync(root)){const dir=path.join(root,batch);if(JSON.parse(fs.readFileSync(path.join(dir,'owner.json'),'utf8')).source_group!=='blocked')continue;
      for(const slot of fs.readdirSync(path.join(dir,'requests')))retained=fs.lstatSync(path.join(dir,'requests',slot,'captured')).isFIFO();}
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
    held: 1,
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
