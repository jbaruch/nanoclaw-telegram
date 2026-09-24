import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ContainerOutput } from './container-runner.js';

const runContainerAgentMock = vi.hoisted(() => vi.fn());

vi.mock('./container-runner.js', () => ({
  runContainerAgent: runContainerAgentMock,
  writeGroupsSnapshot: vi.fn(),
  writeTasksSnapshot: vi.fn(),
}));

import { _initTestDatabase, storeChatMetadata, storeMessage } from './db.js';
import { GroupQueue } from './group-queue.js';
import {
  _getLastAgentTimestampForTests,
  _processGroupMessagesForTests,
  _resetProcessingStateForTests,
  _setChannelsForTests,
  _setRegisteredGroups,
} from './index.js';
import type { Channel, RegisteredGroup } from './types.js';

const CHAT_JID = 'test@g.us';
const MESSAGE_TIMESTAMP = '2026-01-02T03:04:05.000Z';

function makeChannel(sendMessage: Channel['sendMessage']): Channel {
  return {
    name: 'test',
    connect: async () => {},
    sendMessage,
    isConnected: () => true,
    ownsJid: (jid) => jid === CHAT_JID,
    disconnect: async () => {},
    setTyping: vi.fn(async () => {}),
  };
}

const group: RegisteredGroup = {
  name: 'Test',
  folder: 'test-group',
  trigger: '@bot',
  added_at: '2026-01-01T00:00:00.000Z',
  isMain: true,
};

function seedMessage(): void {
  storeChatMetadata(CHAT_JID, MESSAGE_TIMESTAMP, 'Test', 'test', true);
  storeMessage({
    id: 'message-1',
    chat_jid: CHAT_JID,
    sender: 'user-1',
    sender_name: 'User',
    content: 'hello',
    timestamp: MESSAGE_TIMESTAMP,
  });
}

describe('message delivery failure semantics', () => {
  beforeEach(() => {
    _initTestDatabase();
    _resetProcessingStateForTests();
    _setRegisteredGroups({ [CHAT_JID]: group });
    runContainerAgentMock.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    _resetProcessingStateForTests();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('keeps successful output flow after deterministic 5xx retry exhaustion', async () => {
    const sendMessage = vi
      .fn<Channel['sendMessage']>()
      .mockRejectedValue(
        Object.assign(new Error('service unavailable'), { error_code: 503 }),
      );
    _setChannelsForTests([makeChannel(sendMessage)]);
    seedMessage();

    let releaseContainer!: () => void;
    let outputHandled!: () => void;
    const handled = new Promise<void>((resolve) => {
      outputHandled = resolve;
    });
    runContainerAgentMock.mockImplementation(
      async (
        _group: RegisteredGroup,
        _input: unknown,
        _onProcess: unknown,
        onOutput: (output: ContainerOutput) => Promise<void>,
      ) => {
        await onOutput({ status: 'success', result: 'agent result' });
        outputHandled();
        await new Promise<void>((resolve) => {
          releaseContainer = resolve;
        });
        return { status: 'success', result: 'agent result' };
      },
    );
    const notifyIdleSpy = vi.spyOn(GroupQueue.prototype, 'notifyIdle');

    const processing = _processGroupMessagesForTests(CHAT_JID);
    await vi.advanceTimersByTimeAsync(6000);
    await handled;

    expect(sendMessage).toHaveBeenCalledTimes(3);
    expect(notifyIdleSpy).toHaveBeenCalledWith(CHAT_JID);
    expect(vi.getTimerCount()).toBeGreaterThan(0);
    expect(_getLastAgentTimestampForTests(CHAT_JID)).toBe(MESSAGE_TIMESTAMP);

    releaseContainer();
    await expect(processing).resolves.toBe(true);
    expect(_getLastAgentTimestampForTests(CHAT_JID)).toBe(MESSAGE_TIMESTAMP);
  });

  it('propagates a delivery callback TypeError', async () => {
    const err = Object.assign(new TypeError('send invariant failed'), {
      error_code: 500,
    });
    const sendMessage = vi.fn<Channel['sendMessage']>().mockRejectedValue(err);
    _setChannelsForTests([makeChannel(sendMessage)]);
    seedMessage();
    runContainerAgentMock.mockImplementation(
      async (
        _group: RegisteredGroup,
        _input: unknown,
        _onProcess: unknown,
        onOutput: (output: ContainerOutput) => Promise<void>,
      ) => {
        await onOutput({ status: 'success', result: 'agent result' });
        return { status: 'success', result: 'agent result' };
      },
    );

    await expect(_processGroupMessagesForTests(CHAT_JID)).rejects.toBe(err);
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });
});
