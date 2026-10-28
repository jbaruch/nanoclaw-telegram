import fs from 'fs';
import { HttpError } from 'grammy';
import os from 'os';
import path from 'path';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { afterEach, describe, expect, it, vi } from 'vitest';

const fixture = vi.hoisted(() => ({ data: '' }));
vi.mock('./config.js', () => ({
  get DATA_DIR() {
    return fixture.data;
  },
  IPC_POLL_INTERVAL: 1000,
  TIMEZONE: 'UTC',
}));
import { protectIpcClaimMounts } from './ipc-mounts.js';
import {
  startIpcWatcher,
  IpcRouteUnavailableError,
  type IpcDeps,
} from './ipc.js';

afterEach(() => vi.useRealTimers());

describe('IPC file quarantine', () => {
  it('quarantines malformed shapes and continues processing later valid input', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2024-01-01T00:00:00Z'));
    const temporary = fs.mkdtempSync(
      path.join(os.tmpdir(), 'nanoclaw-ipc-shapes-'),
    );
    fixture.data = temporary;
    const tasks = path.join(temporary, 'ipc', 'main', 'tasks');
    const messages = path.join(temporary, 'ipc', 'main', 'messages');
    fs.mkdirSync(tasks, { recursive: true });
    fs.mkdirSync(messages);
    const incompleteMessage = JSON.stringify({
      type: 'message',
      chatJid: 'tg:123',
    });
    fs.writeFileSync(path.join(messages, 'incomplete.json'), incompleteMessage);
    const invalidUtf8 = Buffer.concat([
      Buffer.from('{"type":"message","chatJid":"tg:123","text":"'),
      Buffer.from([0xff]),
      Buffer.from('"}'),
    ]);
    fs.writeFileSync(path.join(messages, 'invalid-utf8.json'), invalidUtf8);

    const invalid = {
      'incomplete.json': { type: 'unsupported_same_name_task' },
      'folder.json': {
        type: 'register_group',
        jid: 'tg:123',
        name: 'fixed',
        folder: 123,
        trigger: '@Andy',
      },
      'cron.json': {
        type: 'schedule_task',
        targetJid: 'tg:123',
        prompt: 'fixed',
        schedule_type: 'cron',
        schedule_value: { invalid: true },
      },
      'unknown.json': { type: 'unsupported_request' },
      'incomplete-task.json': {
        type: 'schedule_task',
        prompt: 'fixed',
        schedule_type: 'cron',
      },
      'mount.json': {
        type: 'register_group',
        jid: 'tg:123',
        name: 'fixed',
        folder: 'fixed',
        trigger: '@Andy',
        containerConfig: { additionalMounts: [{ hostPath: 123 }] },
      },
    };
    for (const [name, value] of Object.entries(invalid))
      fs.writeFileSync(path.join(tasks, name), JSON.stringify(value));
    const deps: IpcDeps = {
      sendMessage: vi.fn(async () => {}),
      sendReaction: vi.fn(async () => {}),
      registeredGroups: () => ({
        'tg:123': {
          name: 'fixed',
          folder: 'main',
          isMain: true,
          trigger: '@Andy',
          added_at: '2024-01-01T00:00:00Z',
        },
      }),
      registerGroup: vi.fn(),
      syncGroups: vi.fn(async () => {}),
      getAvailableGroups: () => [],
      writeGroupsSnapshot: vi.fn(),
      onTasksChanged: vi.fn(),
    };
    const priorUmask = process.umask(0o002);
    try {
      const scan = vi.spyOn(fs, 'readdirSync').mockImplementationOnce(() => {
        throw Object.assign(new Error('fixed missing directory'), {
          code: 'ENOENT',
        });
      });
      const lifecycle = startIpcWatcher(deps);
      const lifecycleFailed = vi.fn();
      void lifecycle.then(undefined, lifecycleFailed);
      await vi.advanceTimersByTimeAsync(0);
      expect(vi.getTimerCount()).toBe(1);
      scan.mockRestore();
      await vi.advanceTimersByTimeAsync(1000);
      expect(vi.getTimerCount()).toBe(1);
      const quarantine = path.join(temporary, 'ipc', 'errors');
      expect(fs.statSync(quarantine).mode & 0o777).toBe(0o700);
      expect(
        protectIpcClaimMounts([
          {
            hostPath: temporary,
            containerPath: '/workspace/project',
            readonly: true,
          },
        ]).map((mount) => mount.containerPath),
      ).toContain('/workspace/project/ipc/errors');
      expect(deps.registerGroup).not.toHaveBeenCalled();
      expect(deps.sendMessage).not.toHaveBeenCalled();
      expect(
        fs.readFileSync(
          path.join(
            temporary,
            'ipc',
            'errors',
            'main-messages-0-invalid-utf8.json',
          ),
        ),
      ).toEqual(invalidUtf8);

      expect(
        fs.readFileSync(
          path.join(
            temporary,
            'ipc',
            'errors',
            'main-messages-0-incomplete.json',
          ),
          'utf8',
        ),
      ).toBe(incompleteMessage);
      for (const [name, value] of Object.entries(invalid)) {
        expect(fs.existsSync(path.join(tasks, name))).toBe(false);
        expect(
          JSON.parse(
            fs.readFileSync(
              path.join(temporary, 'ipc', 'errors', `main-tasks-0-${name}`),
              'utf8',
            ),
          ),
        ).toEqual(value);
      }
      const laterBody = JSON.stringify({ type: 'unsupported_later_message' });
      fs.writeFileSync(path.join(messages, 'incomplete.json'), laterBody);
      await vi.advanceTimersByTimeAsync(1000);
      expect(
        fs.readFileSync(
          path.join(
            temporary,
            'ipc',
            'errors',
            'main-messages-0-incomplete.json',
          ),
          'utf8',
        ),
      ).toBe(incompleteMessage);
      expect(
        fs.readFileSync(
          path.join(
            temporary,
            'ipc',
            'errors',
            'main-messages-1-incomplete.json',
          ),
          'utf8',
        ),
      ).toBe(laterBody);
      expect(fs.existsSync(path.join(messages, 'incomplete.json'))).toBe(false);
      fs.writeFileSync(
        path.join(messages, 'valid.json'),
        JSON.stringify({
          type: 'message',
          chatJid: 'tg:123',
          text: 'fixed later input',
        }),
      );
      await vi.advanceTimersByTimeAsync(1000);
      expect(deps.sendMessage).toHaveBeenCalledExactlyOnceWith(
        'tg:123',
        'fixed later input',
      );
      expect(fs.existsSync(path.join(messages, 'valid.json'))).toBe(false);
      const reactionFailure = new HttpError(
        'fixed reaction transport refusal',
        new Error('synthetic'),
      );
      vi.mocked(deps.sendReaction!).mockRejectedValueOnce(reactionFailure);
      const reaction = {
        type: 'react_to_message',
        chatJid: 'tg:123',
        messageId: '17',
        emoji: '👍',
      };
      fs.writeFileSync(
        path.join(messages, 'failed-reaction.json'),
        JSON.stringify(reaction),
      );
      await vi.advanceTimersByTimeAsync(1000);
      expect(deps.sendReaction).toHaveBeenCalledExactlyOnceWith(
        'tg:123',
        '17',
        '👍',
      );
      expect(
        JSON.parse(
          fs.readFileSync(
            path.join(
              temporary,
              'ipc',
              'errors',
              'main-messages-0-failed-reaction.json',
            ),
            'utf8',
          ),
        ),
      ).toEqual(reaction);
      expect(fs.existsSync(path.join(messages, 'failed-reaction.json'))).toBe(
        false,
      );
      expect(vi.getTimerCount()).toBe(1);
      const unavailable = new IpcRouteUnavailableError(
        'No active channel owns the IPC destination; verify its route before resubmitting.',
      );
      vi.mocked(deps.sendMessage).mockRejectedValueOnce(unavailable);
      const unknownRoute = {
        type: 'message',
        chatJid: 'unknown:fixed',
        text: 'fixed route payload',
      };
      fs.writeFileSync(
        path.join(messages, 'unavailable-route.json'),
        JSON.stringify(unknownRoute),
      );
      await vi.advanceTimersByTimeAsync(1000);
      expect(
        JSON.parse(
          fs.readFileSync(
            path.join(
              temporary,
              'ipc',
              'errors',
              'main-messages-0-unavailable-route.json',
            ),
            'utf8',
          ),
        ),
      ).toEqual(unknownRoute);
      expect(lifecycleFailed).not.toHaveBeenCalled();
      fs.writeFileSync(
        path.join(messages, 'route-recovered.json'),
        JSON.stringify({
          type: 'message',
          chatJid: 'tg:123',
          text: 'fixed later routed payload',
        }),
      );
      await vi.advanceTimersByTimeAsync(1000);
      expect(deps.sendMessage).toHaveBeenLastCalledWith(
        'tg:123',
        'fixed later routed payload',
      );
      expect(fs.existsSync(path.join(messages, 'route-recovered.json'))).toBe(
        false,
      );
      expect(lifecycleFailed).not.toHaveBeenCalled();
      expect(vi.getTimerCount()).toBe(1);
      const defect = new TypeError('fixed delayed handler defect');
      vi.mocked(deps.sendMessage).mockRejectedValueOnce(defect);
      fs.writeFileSync(
        path.join(messages, 'delayed-defect.json'),
        JSON.stringify({
          type: 'message',
          chatJid: 'tg:123',
          text: 'fixed failing input',
        }),
      );
      const rejected = expect(lifecycle).rejects.toBe(defect);
      await vi.advanceTimersByTimeAsync(1000);
      await rejected;
      expect(vi.getTimerCount()).toBe(1);
      expect(
        fs.existsSync(
          path.join(
            temporary,
            'ipc',
            'errors',
            'main-messages-0-delayed-defect.json',
          ),
        ),
      ).toBe(true);
      fs.writeFileSync(
        path.join(messages, 'after-defect.json'),
        JSON.stringify({
          type: 'message',
          chatJid: 'tg:123',
          text: 'fixed healthy input',
        }),
      );
      await vi.advanceTimersByTimeAsync(1000);
      expect(deps.sendMessage).toHaveBeenLastCalledWith(
        'tg:123',
        'fixed healthy input',
      );
      expect(vi.getTimerCount()).toBe(1);
    } finally {
      process.umask(priorUmask);
      vi.clearAllTimers();
      fs.rmSync(temporary, { recursive: true });
    }
  });

  it('quarantines before an unexpected implementation defect escapes the real child process', () => {
    const temporary = fs.mkdtempSync(
      path.join(os.tmpdir(), 'nanoclaw-ipc-defect-'),
    );
    const messages = path.join(temporary, 'data', 'ipc', 'main', 'messages');
    fs.mkdirSync(messages, { recursive: true });
    const body = JSON.stringify({
      type: 'message',
      chatJid: 'tg:123',
      text: 'fixed input',
    });
    fs.writeFileSync(path.join(messages, 'defect.json'), body);
    fs.writeFileSync(
      path.join(messages, 'z-later.json'),
      JSON.stringify({
        type: 'message',
        chatJid: 'tg:123',
        text: 'fixed later input',
      }),
    );
    const source = fileURLToPath(new URL('./ipc.ts', import.meta.url));
    const loader = fileURLToPath(import.meta.resolve('tsx/esm'));
    const script = `import {startIpcWatcher} from ${JSON.stringify(source)};
      startIpcWatcher({sendMessage:async(_jid,text)=>{if(text==='fixed input')throw new TypeError('fixed callback defect');process.stdout.write('FIXED_LATER_PROCESSED\\n');setImmediate(()=>process.exit(0));},
        registeredGroups:()=>({'tg:123':{name:'fixed',folder:'main',isMain:true,trigger:'@Andy',added_at:'2024-01-01T00:00:00Z'}}),
        registerGroup:()=>{},syncGroups:async()=>{},getAvailableGroups:()=>[],writeGroupsSnapshot:()=>{},onTasksChanged:()=>{}});`;
    try {
      const child = spawnSync(
        process.execPath,
        ['--import', loader, '--input-type', 'module', '-e', script],
        {
          cwd: temporary,
          env: { PATH: process.env.PATH ?? '', HOME: temporary, TZ: 'UTC' },
          encoding: 'utf8',
          timeout: 10000,
        },
      );
      expect(child.error).toBeUndefined();
      // Existing logger reports unhandled rejections without changing exitCode.
      expect(child.status).toBe(0);
      expect(child.stderr).toContain('Unhandled rejection');
      expect(child.stderr).toContain('fixed callback defect');
      expect(child.stdout).toContain('FIXED_LATER_PROCESSED');
      expect(fs.existsSync(path.join(messages, 'z-later.json'))).toBe(false);
      expect(fs.existsSync(path.join(messages, 'defect.json'))).toBe(false);
      expect(
        fs.readFileSync(
          path.join(
            temporary,
            'data',
            'ipc',
            'errors',
            'main-messages-0-defect.json',
          ),
          'utf8',
        ),
      ).toBe(body);
    } finally {
      fs.rmSync(temporary, { recursive: true });
    }
  });
});

it('retains the sole claim when the first quarantine directory parent cannot be made durable', () => {
  const temporary = fs.mkdtempSync(
    path.join(os.tmpdir(), 'telegram-quarantine-parent-'),
  );
  const base = path.join(temporary, 'data', 'ipc'),
    messages = path.join(base, 'main', 'messages');
  fs.mkdirSync(messages, { recursive: true, mode: 0o755 });
  const original = JSON.stringify({ type: 'unsupported_request' });
  fs.writeFileSync(path.join(messages, 'fixed.json'), original);
  const source = fileURLToPath(new URL('./ipc.ts', import.meta.url));
  const script = `import fs from 'node:fs';import path from 'node:path';import {startIpcWatcher} from ${JSON.stringify(source)};
    const base=path.join(process.cwd(),'data','ipc'),errors=path.join(base,'errors');
    const descriptors=new Map(),open=fs.openSync.bind(fs),mkdir=fs.mkdirSync.bind(fs),sync=fs.fsyncSync.bind(fs);let created=false;
    const refusal=Object.assign(new Error('fixed quarantine parent refusal'),{code:'EIO'});
    fs.openSync=(file,flags,mode)=>{const fd=open(file,flags,mode);descriptors.set(fd,String(file));return fd;};
    fs.mkdirSync=(file,options)=>{const result=mkdir(file,options);if(String(file)===errors)created=true;return result;};
    fs.fsyncSync=fd=>{if(created&&descriptors.get(fd)===base)throw refusal;return sync(fd);};
    startIpcWatcher({sendMessage:async()=>{throw new Error('No delivery is authorized');},registeredGroups:()=>({'tg:123':{name:'fixed',folder:'main',isMain:true,trigger:'@Andy',added_at:'2024-01-01T00:00:00Z'}}),registerGroup:()=>{},syncGroups:async()=>{},getAvailableGroups:()=>[],writeGroupsSnapshot:()=>{},onTasksChanged:()=>{}}).then(()=>{throw new Error('Watcher must remain live or refuse');},error=>{
      const root=path.join(base,'.claimed-v1');let retained=[];
      for(const batch of fs.readdirSync(root))for(const slot of fs.readdirSync(path.join(root,batch,'requests'))){const file=path.join(root,batch,'requests',slot,'pending');if(fs.existsSync(file))retained.push(fs.readFileSync(file,'utf8'));}
      const result={refused:error instanceof AggregateError&&error.errors.includes(refusal),retained,quarantine_entries:fs.readdirSync(errors)};
      process.stdout.write('FIXED_RESULT:'+JSON.stringify(result)+'\\n',()=>process.exit(0));
    });`;
  try {
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
        cwd: temporary,
        env: { PATH: process.env.PATH ?? '', HOME: temporary, TZ: 'UTC' },
        encoding: 'utf8',
        timeout: 5000,
        killSignal: 'SIGKILL',
        maxBuffer: 262144,
      },
    );
    expect(child.error).toBeUndefined();
    expect(child.status, child.stderr).toBe(0);
    const result = child.stdout
      .split('\n')
      .find((line) => line.startsWith('FIXED_RESULT:'));
    expect(result).toBeDefined();
    expect(JSON.parse(result!.slice('FIXED_RESULT:'.length))).toEqual({
      refused: true,
      retained: [original],
      quarantine_entries: [],
    });
  } finally {
    fs.rmSync(temporary, { recursive: true });
  }
});
