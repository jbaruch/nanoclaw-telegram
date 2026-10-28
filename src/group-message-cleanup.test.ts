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
const idleFixture = vi.hoisted(() => ({
  pendingTask: false,
  closed: false,
  notifyIdle: vi.fn(),
}));
vi.mock('./group-queue.js', () => ({
  GroupQueue: class {
    notifyIdle = (jid: string) => {
      idleFixture.notifyIdle(jid);
      if (idleFixture.pendingTask) idleFixture.closed = true;
    };
    closeStdin = vi.fn();
    sendMessage = vi.fn(() => true);
  },
}));
vi.mock('./db.js', () => ({
  getMessagesSince: vi.fn(),
  getNewMessages: vi.fn(),
  getLastBotMessageTimestamp: vi.fn(() => ''),
  setRouterState: vi.fn(),
  getAllTasks: () => [],
  getAllChats: () => [],
  setSession: vi.fn(),
}));
import {
  processGroupMessages,
  startMessageLoop,
  _setRegisteredGroups,
  sendWithRetry,
  sendIpcMessage,
  sendIpcReaction,
  sendScheduledMessage,
} from './index.js';
import { getMessagesSince, getNewMessages, setRouterState } from './db.js';
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

it.each(['operational', 'unexpected'] as const)(
  'preserves the cursor and skips queue retry after real channel partial delivery (%s)',
  async (kind) => {
    const jid = 'tg:123456';
    await seed(jid);
    const { TelegramChannel } = await vi.importActual<
      typeof import('./channels/telegram.js')
    >('./channels/telegram.js');
    const { GroupQueue } =
      await vi.importActual<typeof import('./group-queue.js')>(
        './group-queue.js',
      );
    const nativeFailure =
      kind === 'operational'
        ? new HttpError(
            'fixed second chunk failure',
            new Error('fixed transport'),
          )
        : new TypeError('fixed second chunk implementation defect');
    const nativeSend = vi
      .fn()
      .mockResolvedValueOnce({ message_id: 71 })
      .mockRejectedValueOnce(nativeFailure);
    const nativeChannel = new TelegramChannel('123:synthetic', {
      onMessage: vi.fn(),
      onChatMetadata: vi.fn(),
      registeredGroups: () => ({}),
    });
    // Native HTTP is mocked; channel splitting, IDs, partial outcome and queue
    // retry policy below are their real production implementations.
    Object.defineProperty(nativeChannel, 'bot', {
      value: { api: { sendMessage: nativeSend } },
    });
    channel = nativeChannel;
    channel.createDraftStream = undefined;
    channel.setTyping = vi.fn(async () => {});
    vi.mocked(findChannel).mockReturnValue(channel);
    vi.mocked(runContainerAgent).mockImplementationOnce(
      async (_group, _input, _process, onOutput) => {
        await onOutput!({ status: 'success', result: 'a'.repeat(5000) });
        return { status: 'success', result: null };
      },
    );
    const run = vi.fn(processGroupMessages);
    const queue = new GroupQueue();
    const failure =
      kind === 'unexpected'
        ? expect(queue.waitForFailure()).rejects.toMatchObject({
            cause: nativeFailure,
          })
        : undefined;
    queue.setProcessMessagesFn(run);
    queue.enqueueMessageCheck(jid);
    await vi.advanceTimersByTimeAsync(60000);
    if (failure) await failure;
    expect(run).toHaveBeenCalledTimes(1);
    expect(nativeSend).toHaveBeenCalledTimes(2);
    expect(storedCursor(jid)).toBe(current);
    expect(vi.getTimerCount()).toBe(0);
    await queue.shutdown(0);
  },
);

it.each(['cancel', 'oversized'] as const)(
  'retains an acknowledged undeletable preview without queue replay (%s)',
  async (kind) => {
    const jid = `tg:uncertain-preview-${kind}`;
    await seed(jid);
    const { TelegramChannel } = await vi.importActual<
      typeof import('./channels/telegram.js')
    >('./channels/telegram.js');
    const { GroupQueue } =
      await vi.importActual<typeof import('./group-queue.js')>(
        './group-queue.js',
      );
    const nativeSend = vi.fn(async () => ({ message_id: 91 }));
    const nativeDelete = vi
      .fn()
      .mockRejectedValue(
        new HttpError('fixed delete failure', new Error('fixed transport')),
      );
    const nativeChannel = new TelegramChannel('123:synthetic', {
      onMessage: vi.fn(),
      onChatMetadata: vi.fn(),
      registeredGroups: () => ({}),
    });
    Object.defineProperty(nativeChannel, 'bot', {
      value: { api: { sendMessage: nativeSend, deleteMessage: nativeDelete } },
    });
    channel = nativeChannel;
    channel.setTyping = vi.fn(async () => {});
    vi.mocked(findChannel).mockReturnValue(channel);
    vi.mocked(runContainerAgent).mockImplementationOnce(
      async (_group, _input, _process, onOutput) => {
        await onOutput!({
          status: 'success',
          result: null,
          streamText:
            'fixed acknowledged preview longer than thirty characters',
        });
        if (kind === 'oversized')
          await onOutput!({ status: 'success', result: 'x'.repeat(5000) });
        return { status: 'error', result: null, error: 'fixed failed query' };
      },
    );
    const run = vi.fn(processGroupMessages),
      queue = new GroupQueue();
    queue.setProcessMessagesFn(run);
    queue.enqueueMessageCheck(jid);
    await vi.advanceTimersByTimeAsync(60000);
    expect(nativeSend).toHaveBeenCalledTimes(1);
    expect(nativeDelete).toHaveBeenCalledExactlyOnceWith(
      `uncertain-preview-${kind}`,
      91,
    );
    expect(run).toHaveBeenCalledTimes(1);
    expect(storedCursor(jid)).toBe(current);
    expect(vi.getTimerCount()).toBe(0);
    await queue.shutdown(0);
  },
);

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

it('combines the actual preview failure with the owned idle-close signal', async () => {
  const error = new TypeError('fixed preview owner failure');
  draft.failure = Promise.resolve({ error });
  await seed('tg:preview-owner');
  await expect(
    vi.mocked(runContainerAgent).mock.calls.at(-1)?.[4],
  ).resolves.toEqual({ error });
});

it('does not deduplicate an unacknowledged failure, and only suppresses after a successful retry', async () => {
  const failure = new HttpError(
    'fixed failed delivery',
    new Error('synthetic'),
  );
  vi.mocked(channel.sendMessage).mockRejectedValueOnce(failure);
  await expect(
    sendIpcMessage('tg:ack-dedup', 'fixed acknowledgment payload'),
  ).rejects.toBe(failure);
  await expect(
    sendIpcMessage('tg:ack-dedup', 'fixed acknowledgment payload'),
  ).resolves.toBeUndefined();
  await expect(
    sendIpcMessage('tg:ack-dedup', 'fixed acknowledgment payload'),
  ).resolves.toBeUndefined();
  expect(channel.sendMessage).toHaveBeenCalledTimes(2);
});

it('makes concurrent duplicate callers await the same actual acknowledgment', async () => {
  let reject!: (error: unknown) => void;
  const failure = new HttpError(
    'fixed pending delivery failure',
    new Error('synthetic'),
  );
  vi.mocked(channel.sendMessage).mockImplementationOnce(
    () =>
      new Promise<void>((_resolve, fail) => {
        reject = fail;
      }),
  );
  const first = sendIpcMessage('tg:concurrent-ack', 'fixed concurrent payload');
  const second = sendIpcMessage(
    'tg:concurrent-ack',
    'fixed concurrent payload',
  );
  const one = expect(first).rejects.toBe(failure),
    two = expect(second).rejects.toBe(failure);
  expect(channel.sendMessage).toHaveBeenCalledTimes(1);
  reject(failure);
  await one;
  await two;
  await expect(
    sendIpcMessage('tg:concurrent-ack', 'fixed concurrent payload'),
  ).resolves.toBeUndefined();
  expect(channel.sendMessage).toHaveBeenCalledTimes(2);
});

it('refuses unavailable production reaction routes and capabilities', async () => {
  vi.mocked(findChannel).mockReturnValueOnce(undefined);
  await expect(
    sendIpcReaction('missing:fixed', '17', '👍'),
  ).rejects.toBeInstanceOf(IpcRouteUnavailableError);
  await expect(sendIpcReaction('tg:123', '17', '👍')).rejects.toBeInstanceOf(
    IpcRouteUnavailableError,
  );
  await expect(
    sendIpcReaction('tg:123', undefined, '👍'),
  ).rejects.toBeInstanceOf(IpcRouteUnavailableError);
  channel.sendReaction = vi.fn(async () => {});
  await expect(sendIpcReaction('tg:123', '17', '👍')).resolves.toBeUndefined();
  expect(channel.sendReaction).toHaveBeenCalledExactlyOnceWith(
    'tg:123',
    '17',
    '👍',
  );
});

it('refuses a missing production scheduled-result route instead of acknowledging delivery', async () => {
  vi.mocked(findChannel).mockReturnValueOnce(undefined);
  await expect(
    sendScheduledMessage('missing:fixed', 'fixed task output'),
  ).rejects.toBeInstanceOf(IpcRouteUnavailableError);
  expect(channel.sendMessage).not.toHaveBeenCalled();
});

it.each(['EPIPE', 'ECONNRESET', 'aggregate'] as const)(
  'contains an operational native-run %s failure to its group',
  async (kind) => {
    const jid = `tg:operational-run-${kind}`;
    await seed(jid);
    const transport = new HttpError(
      'fixed transport failure',
      new Error('synthetic'),
    );
    const failure =
      kind === 'aggregate'
        ? new AggregateError(
            [
              transport,
              Object.assign(new Error('fixed stop failure'), {
                status: 7,
                signal: null,
              }),
            ],
            'fixed operational aggregate',
            { cause: transport },
          )
        : Object.assign(new Error('fixed closed input'), { code: kind });
    vi.mocked(runContainerAgent).mockRejectedValueOnce(failure);
    await expect(processGroupMessages(jid)).resolves.toBe(false);
    expect(storedCursor(jid)).toBe(previous);
    expect(channel.setTyping).toHaveBeenLastCalledWith(jid, false);
    expect(vi.getTimerCount()).toBe(0);
  },
);

it('does not contain a mixed operational and implementation aggregate as a routine run failure', async () => {
  const jid = 'tg:mixed-run-defect';
  await seed(jid);
  const native = new HttpError('fixed transport', new Error('synthetic'));
  const defect = new TypeError('fixed callback implementation defect');
  const mixed = new AggregateError([native, defect], 'fixed mixed failure', {
    cause: defect,
  });
  vi.mocked(runContainerAgent).mockRejectedValueOnce(mixed);
  await expect(processGroupMessages(jid)).rejects.toBe(mixed);
  expect(storedCursor(jid)).toBe(previous);
  expect(channel.setTyping).toHaveBeenLastCalledWith(jid, false);
});

it('propagates an unexpected typing defect from the actual piped-message consumer', async () => {
  const jid = 'tg:piped-typing-defect';
  _setRegisteredGroups({
    [jid]: {
      name: 'fixed',
      folder: 'fixed',
      isMain: true,
      trigger: '@Andy',
      added_at: previous,
    },
  });
  vi.mocked(getNewMessages).mockReturnValue({
    messages: [input(jid, current)],
    newTimestamp: current,
  });
  vi.mocked(getMessagesSince).mockReturnValue([input(jid, current)]);
  const defect = new TypeError('fixed piped typing defect');
  vi.mocked(channel.setTyping!).mockRejectedValueOnce(defect);
  await expect(startMessageLoop()).rejects.toBe(defect);
  expect(channel.setTyping).toHaveBeenCalledExactlyOnceWith(jid, true);
  const writes = vi
    .mocked(setRouterState)
    .mock.calls.filter(([key]) => key === 'last_agent_timestamp');
  expect(writes.some(([, value]) => JSON.parse(value)[jid] === current)).toBe(
    false,
  );
});

it('does not preempt a producing query for a pending task on preview or session-only updates', async () => {
  const jid = 'tg:pending-task-terminal';
  await seed(jid);
  idleFixture.pendingTask = true;
  idleFixture.closed = false;
  idleFixture.notifyIdle.mockClear();
  vi.mocked(runContainerAgent).mockImplementationOnce(
    async (_group, _input, _process, onOutput) => {
      await onOutput!({
        status: 'success',
        result: null,
        streamText: 'fixed preview',
      });
      await onOutput!({
        status: 'success',
        result: null,
        newSessionId: 'fixed-session',
      });
      expect(idleFixture.closed).toBe(false);
      expect(idleFixture.notifyIdle).not.toHaveBeenCalled();
      await onOutput!({
        status: 'success',
        result: null,
        completion: 'sdk_turn_completed',
      });
      expect(idleFixture.closed).toBe(true);
      expect(idleFixture.notifyIdle).toHaveBeenCalledExactlyOnceWith(jid);
      return { status: 'success', result: null };
    },
  );
  await expect(processGroupMessages(jid)).resolves.toBe(true);
  idleFixture.pendingTask = false;
});

it('retries undelivered final output when its oversized preview was already positively deleted', async () => {
  const jid = 'tg:retired-preview-fallback';
  await seed(jid);
  const { createDraftStream } =
    await vi.importActual<typeof import('./draft-stream.js')>(
      './draft-stream.js',
    );
  const remove = vi
    .fn()
    .mockResolvedValueOnce(undefined)
    .mockRejectedValueOnce(
      new HttpError('must not delete twice', new Error('fixed')),
    );
  draft = createDraftStream({
    sendMessage: async () => 91,
    editMessage: async () => {},
    deleteMessage: remove,
  });
  channel.createDraftStream = () => draft;
  const failure = new HttpError(
    'fixed final send refusal',
    new Error('fixed transport'),
  );
  vi.mocked(channel.sendMessage).mockRejectedValue(failure);
  vi.mocked(runContainerAgent).mockImplementationOnce(
    async (_group, _input, _process, onOutput) => {
      await onOutput!({
        status: 'success',
        result: null,
        streamText: 'fixed acknowledged preview longer than thirty characters',
      });
      await onOutput!({ status: 'success', result: 'x'.repeat(5000) });
      return { status: 'success', result: null };
    },
  );
  const pending = processGroupMessages(jid),
    rejected = expect(pending).resolves.toBe(false);
  await vi.advanceTimersByTimeAsync(60000);
  await rejected;
  expect(remove).toHaveBeenCalledExactlyOnceWith(91);
  expect(storedCursor(jid)).toBe(previous);
});
