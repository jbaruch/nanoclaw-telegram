import fs from 'fs';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { TEST_DATA_DIR } = vi.hoisted(() => ({
  TEST_DATA_DIR: `/tmp/nanoclaw-ipc-watcher-${process.pid}`,
}));

vi.mock('./config.js', () => ({
  DATA_DIR: TEST_DATA_DIR,
  IPC_POLL_INTERVAL: 100,
  TIMEZONE: 'UTC',
}));

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

import {
  _resetIpcWatcherForTests,
  NoChannelForJidError,
  startIpcWatcher,
} from './ipc.js';
import type { IpcDeps } from './ipc.js';

function makeDeps(sendMessage: IpcDeps['sendMessage']): IpcDeps {
  return {
    sendMessage,
    registeredGroups: () => ({
      'main@g.us': {
        name: 'Main',
        folder: 'main',
        trigger: '@bot',
        added_at: '2026-01-01T00:00:00.000Z',
        isMain: true,
      },
    }),
    registerGroup: vi.fn(),
    syncGroups: vi.fn(async () => {}),
    getAvailableGroups: () => [],
    writeGroupsSnapshot: vi.fn(),
    onTasksChanged: vi.fn(),
  };
}

function writeIpcFile(
  folder: 'messages' | 'tasks',
  name: string,
  contents: string,
): string {
  const dir = path.join(TEST_DATA_DIR, 'ipc', 'main', folder);
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, name);
  fs.writeFileSync(file, contents);
  return file;
}

async function stopWatcher(watcher: Promise<void>): Promise<void> {
  _resetIpcWatcherForTests();
  await vi.advanceTimersByTimeAsync(100);
  await watcher;
}

describe('IPC watcher ownership and quarantine', () => {
  beforeEach(() => {
    fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
    _resetIpcWatcherForTests();
    vi.useFakeTimers();
  });

  afterEach(() => {
    _resetIpcWatcherForTests();
    vi.useRealTimers();
    fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
  });

  it('quarantines null, array, and no-channel messages, then processes a later valid file', async () => {
    writeIpcFile('messages', 'a-null.json', 'null');
    writeIpcFile('messages', 'b-array.json', '[]');
    writeIpcFile(
      'messages',
      'c-no-channel.json',
      JSON.stringify({ type: 'message', chatJid: 'missing@g.us', text: 'bad' }),
    );
    const sendMessage = vi.fn(async (jid: string) => {
      if (jid === 'missing@g.us') throw new NoChannelForJidError(jid);
    });

    const watcher = startIpcWatcher(makeDeps(sendMessage));
    await vi.advanceTimersByTimeAsync(0);

    const errorDir = path.join(TEST_DATA_DIR, 'ipc', 'errors');
    expect(fs.readdirSync(errorDir).sort()).toEqual([
      'main-a-null.json',
      'main-b-array.json',
      'main-c-no-channel.json',
    ]);

    writeIpcFile(
      'messages',
      'd-valid.json',
      JSON.stringify({ type: 'message', chatJid: 'ok@g.us', text: 'hello' }),
    );
    await vi.advanceTimersByTimeAsync(100);

    expect(sendMessage).toHaveBeenCalledWith('ok@g.us', 'hello');
    expect(
      fs.existsSync(
        path.join(TEST_DATA_DIR, 'ipc', 'main', 'messages', 'd-valid.json'),
      ),
    ).toBe(false);
    await stopWatcher(watcher);
  });

  it('quarantines null and array tasks, then processes a later valid task file', async () => {
    writeIpcFile('tasks', 'a-null.json', 'null');
    writeIpcFile('tasks', 'b-array.json', '[]');

    const watcher = startIpcWatcher(makeDeps(vi.fn(async () => {})));
    await vi.advanceTimersByTimeAsync(0);

    writeIpcFile('tasks', 'c-valid.json', JSON.stringify({ type: 'unknown' }));
    await vi.advanceTimersByTimeAsync(100);

    const errorDir = path.join(TEST_DATA_DIR, 'ipc', 'errors');
    expect(fs.readdirSync(errorDir).sort()).toEqual([
      'main-a-null.json',
      'main-b-array.json',
    ]);
    expect(
      fs.existsSync(
        path.join(TEST_DATA_DIR, 'ipc', 'main', 'tasks', 'c-valid.json'),
      ),
    ).toBe(false);
    await stopWatcher(watcher);
  });

  it('rejects its owning promise on a coded programming error', async () => {
    const err = Object.assign(new TypeError('dependency invariant failed'), {
      code: 'ERR_INVALID_ARG_TYPE',
    });
    writeIpcFile(
      'messages',
      'bad.json',
      JSON.stringify({ type: 'message', chatJid: 'ok@g.us', text: 'hello' }),
    );

    const watcher = startIpcWatcher(
      makeDeps(
        vi.fn(async () => {
          throw err;
        }),
      ),
    );

    await expect(watcher).rejects.toBe(err);
  });
});
