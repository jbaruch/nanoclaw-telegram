import { HttpError } from 'grammy';
import { TelegramDeliveryError } from './operational-errors.js';
import { IpcRouteUnavailableError } from './ipc.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Channel } from './types.js';
import type { DraftStream } from './draft-stream.js';

vi.mock('./channels/index.js', () => ({}));
vi.mock('@onecli-sh/sdk', () => ({ OneCLI: class {} }));
vi.mock('./container-runner.js', () => ({
  runContainerAgent: vi.fn(),
  writeGroupsSnapshot: vi.fn(),
  writeTasksSnapshot: vi.fn(),
}));
vi.mock('./router.js', async (original) => ({
  ...(await original<typeof import('./router.js')>()),
  findChannel: vi.fn(),
}));
vi.mock('./group-queue.js', () => ({
  GroupQueue: class {
    notifyIdle = vi.fn();
    closeStdin = vi.fn();
  },
}));
vi.mock('./db.js', () => ({
  getMessagesSince: vi.fn(),
  getLastBotMessageTimestamp: vi.fn(() => ''),
  setRouterState: vi.fn(),
  getAllTasks: () => [],
  getAllChats: () => [],
  setSession: vi.fn(),
}));
import {
  processGroupMessages,
  _setRegisteredGroups,
  sendWithRetry,
  sendIpcMessage,
} from './index.js';
import { getMessagesSince, setRouterState } from './db.js';
import { findChannel } from './router.js';
import { runContainerAgent } from './container-runner.js';

const previous = '2024-01-01T00:00:00Z';
const current = '2024-01-01T00:00:01Z';
let channel: Channel;
let draft: DraftStream;
function input(jid: string, timestamp: string) {
  return {
    id: 'fixed-message',
    chat_jid: jid,
    sender: 'fixed-owner',
    sender_name: 'fixed',
    content: 'fixed input',
    timestamp,
  };
}
function storedCursor(jid: string): string {
  const writes = vi
    .mocked(setRouterState)
    .mock.calls.filter(([key]) => key === 'last_agent_timestamp');
  const state = JSON.parse(writes.at(-1)![1]) as Record<string, string>;
  return state[jid];
}
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(setRouterState).mockReset();
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2024-01-01T00:00:00Z'));
  draft = {
    failure: new Promise(() => {}),
    update: vi.fn(async () => {}),
    finish: vi.fn(async () => true),
    cancel: vi.fn(async () => {}),
  };
  channel = {
    name: 'fixed',
    connect: vi.fn(async () => {}),
    disconnect: vi.fn(async () => {}),
    sendMessage: vi.fn(async () => {}),
    isConnected: () => true,
    ownsJid: () => true,
    setTyping: vi.fn(async () => {}),
    createDraftStream: () => draft,
  };
  vi.mocked(findChannel).mockReturnValue(channel);
});
afterEach(() => vi.useRealTimers());

async function seed(jid: string) {
  _setRegisteredGroups({
    [jid]: {
      name: 'fixed',
      folder: 'fixed',
      isMain: true,
      requiresTrigger: false,
      trigger: '@Andy',
      added_at: previous,
    },
  });
  vi.mocked(getMessagesSince).mockReturnValue([input(jid, previous)]);
  vi.mocked(runContainerAgent).mockResolvedValue({
    status: 'success',
    result: null,
  });
  await expect(processGroupMessages(jid)).resolves.toBe(true);
  vi.mocked(getMessagesSince).mockReturnValue([input(jid, current)]);
  vi.mocked(channel.setTyping!).mockClear();
  vi.mocked(draft.cancel).mockClear();
}

describe('group message error cleanup', () => {
  it('restores undelivered input and clears UI when native execution throws unexpectedly', async () => {
    const jid = 'tg:before-output';
    await seed(jid);
    const defect = new TypeError('fixed execution defect');
    vi.mocked(runContainerAgent).mockRejectedValueOnce(defect);
    await expect(processGroupMessages(jid)).rejects.toBe(defect);
    expect(storedCursor(jid)).toBe(previous);
    expect(channel.setTyping).toHaveBeenLastCalledWith(jid, false);
    expect(draft.cancel).toHaveBeenCalledTimes(1);
  });

  it('restores undelivered input when typing setup itself fails', async () => {
    const jid = 'tg:typing-setup';
    await seed(jid);
    const defect = new TypeError('fixed typing setup defect');
    vi.mocked(channel.setTyping!).mockRejectedValueOnce(defect);
    await expect(processGroupMessages(jid)).rejects.toBe(defect);
    expect(storedCursor(jid)).toBe(previous);
    expect(channel.setTyping).toHaveBeenLastCalledWith(jid, false);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('preserves execution and rollback errors while still clearing UI state', async () => {
    const jid = 'tg:rollback-defect';
    await seed(jid);
    const defect = new TypeError('fixed execution defect');
    const rollback = new TypeError('fixed rollback defect');
    vi.mocked(runContainerAgent).mockRejectedValueOnce(defect);
    vi.mocked(setRouterState).mockImplementation((key, value) => {
      if (
        key === 'last_agent_timestamp' &&
        (JSON.parse(value) as Record<string, string>)[jid] === previous
      )
        throw rollback;
    });
    await expect(processGroupMessages(jid)).rejects.toMatchObject({
      cause: defect,
      errors: [defect, rollback],
    });
    expect(channel.setTyping).toHaveBeenLastCalledWith(jid, false);
    expect(draft.cancel).toHaveBeenCalledTimes(1);
  });

  it('cleans all resources and retains both errors when callback and typing cleanup fail', async () => {
    const jid = 'tg:callback-and-cleanup';
    await seed(jid);
    const defect = new TypeError('fixed callback defect');
    const cleanup = new TypeError('fixed cleanup defect');
    vi.mocked(draft.update).mockImplementationOnce(() => {
      throw defect;
    });
    vi.mocked(channel.setTyping!).mockImplementation(async (_jid, typing) => {
      if (!typing) throw cleanup;
    });
    vi.mocked(runContainerAgent).mockImplementationOnce(
      async (_group, _input, _process, onOutput) => {
        await onOutput!({
          status: 'success',
          result: null,
          streamText: 'fixed preview',
        });
        return { status: 'success', result: null };
      },
    );
    const result = processGroupMessages(jid);
    await expect(result).rejects.toMatchObject({
      cause: defect,
      errors: [defect, cleanup],
    });
    expect(storedCursor(jid)).toBe(previous);
    expect(draft.cancel).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('preserves the delivered cursor and disarms idle close after a later execution defect', async () => {
    const jid = 'tg:after-output';
    await seed(jid);
    const defect = new TypeError('fixed post-delivery defect');
    vi.mocked(runContainerAgent).mockImplementationOnce(
      async (_group, _input, _process, onOutput) => {
        await onOutput!({
          status: 'success',
          result: 'fixed delivered output',
        });
        throw defect;
      },
    );
    await expect(processGroupMessages(jid)).rejects.toBe(defect);
    expect(storedCursor(jid)).toBe(current);
    expect(draft.finish).toHaveBeenCalledWith('fixed delivered output');
    expect(draft.cancel).not.toHaveBeenCalled();
    expect(channel.setTyping).toHaveBeenLastCalledWith(jid, false);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('rolls back rather than marking delivery after a native operational refusal', async () => {
    const jid = 'tg:operational-send';
    await seed(jid);
    channel.createDraftStream = undefined;
    const failure = new HttpError(
      'fixed native delivery failure',
      new Error('synthetic'),
    );
    vi.mocked(channel.sendMessage).mockRejectedValueOnce(failure);
    vi.mocked(runContainerAgent).mockImplementationOnce(
      async (_group, _input, _process, onOutput) => {
        await onOutput!({ status: 'success', result: 'fixed output' });
        return { status: 'success', result: null };
      },
    );
    await expect(processGroupMessages(jid)).resolves.toBe(false);
    expect(storedCursor(jid)).toBe(previous);
    expect(channel.setTyping).toHaveBeenLastCalledWith(jid, false);
  });

  it('restores undelivered input if cleanup alone fails', async () => {
    const jid = 'tg:cleanup-only';
    await seed(jid);
    const cleanup = new TypeError('fixed draft cleanup defect');
    vi.mocked(draft.cancel).mockRejectedValueOnce(cleanup);
    await expect(processGroupMessages(jid)).rejects.toMatchObject({
      cause: cleanup,
      errors: [cleanup],
    });
    expect(storedCursor(jid)).toBe(previous);
    expect(channel.setTyping).toHaveBeenLastCalledWith(jid, false);
  });
});

it('does not replay whole output after acknowledged chunks failed later', async () => {
  const transport = new HttpError(
    'fixed chunk reset',
    Object.assign(new Error('fixed reset'), { code: 'ECONNRESET' }),
  );
  const partial = new TelegramDeliveryError(
    'Inspect acknowledged chunks before retrying.',
    { cause: transport },
  );
  vi.mocked(channel.sendMessage).mockRejectedValueOnce(partial);
  await expect(
    sendWithRetry(channel, 'tg:123', 'fixed full output'),
  ).rejects.toBe(partial);
  expect(channel.sendMessage).toHaveBeenCalledTimes(1);
  expect(vi.getTimerCount()).toBe(0);
});

it('classifies a missing native IPC route without consuming a valid later send', async () => {
  vi.mocked(findChannel).mockReturnValueOnce(undefined);
  const missing = sendIpcMessage('tg:route-recovery', 'fixed route payload');
  await expect(missing).rejects.toBeInstanceOf(IpcRouteUnavailableError);
  await expect(missing).rejects.toMatchObject({
    message: expect.stringContaining('No active channel owns'),
  });
  expect(channel.sendMessage).not.toHaveBeenCalled();
  await expect(
    sendIpcMessage('tg:route-recovery', 'fixed route payload'),
  ).resolves.toBeUndefined();
  expect(channel.sendMessage).toHaveBeenCalledExactlyOnceWith(
    'tg:route-recovery',
    'fixed route payload',
    undefined,
  );
});

it('passes the actual preview failure signal into the native run owner', async () => {
  await seed('tg:preview-owner');
  expect(vi.mocked(runContainerAgent).mock.calls.at(-1)?.[4]).toBe(
    draft.failure,
  );
});
