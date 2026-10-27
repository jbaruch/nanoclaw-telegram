import fs from 'fs';
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
import { startIpcWatcher, type IpcDeps } from './ipc.js';

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
    const invalid = {
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
    try {
      startIpcWatcher(deps);
      await vi.advanceTimersByTimeAsync(0);
      expect(deps.registerGroup).not.toHaveBeenCalled();
      for (const [name, value] of Object.entries(invalid)) {
        expect(fs.existsSync(path.join(tasks, name))).toBe(false);
        expect(
          JSON.parse(
            fs.readFileSync(
              path.join(temporary, 'ipc', 'errors', `main-${name}`),
              'utf8',
            ),
          ),
        ).toEqual(value);
      }
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
    } finally {
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
    const source = fileURLToPath(new URL('./ipc.ts', import.meta.url));
    const loader = fileURLToPath(import.meta.resolve('tsx/esm'));
    const script = `import {startIpcWatcher} from ${JSON.stringify(source)};
      startIpcWatcher({sendMessage:async()=>{throw new TypeError('fixed callback defect')},
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
      expect(fs.existsSync(path.join(messages, 'defect.json'))).toBe(false);
      expect(
        fs.readFileSync(
          path.join(temporary, 'data', 'ipc', 'errors', 'main-defect.json'),
          'utf8',
        ),
      ).toBe(body);
    } finally {
      fs.rmSync(temporary, { recursive: true });
    }
  });
});
