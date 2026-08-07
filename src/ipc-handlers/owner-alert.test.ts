import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  _resetIpcMessageRegistryForTests,
  dispatchIpcMessage,
  type IpcMessagePayload,
} from '../ipc-message-registry.js';
import {
  _resetOwnerAlertIpcHandlersForTests,
  registerOwnerAlertIpcHandlers,
} from './owner-alert.js';
import type { IpcDeps } from '../ipc.js';
import type { RegisteredGroup } from '../types.js';

const SOURCE_GROUP = 'wtf-pod-chat';
const SOURCE_JID = 'tg:-100untrusted';
const MAIN_JID = 'tg:-100main';

const MAIN: RegisteredGroup = {
  name: 'Main Control',
  folder: 'main',
  trigger: null,
  added_at: '2024-01-01T00:00:00.000Z',
  isMain: true,
};
const SOURCE: RegisteredGroup = {
  name: 'WTF Pod Chat',
  folder: SOURCE_GROUP,
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

let sendMessage: ReturnType<typeof vi.fn>;

function deps(): IpcDeps {
  return { sendMessage } as unknown as IpcDeps;
}

async function dispatch(
  data: IpcMessagePayload,
  registeredGroups: Record<string, RegisteredGroup> = {
    [MAIN_JID]: MAIN,
    [SOURCE_JID]: SOURCE,
  },
): Promise<void> {
  await dispatchIpcMessage({
    data,
    sourceGroup: SOURCE_GROUP,
    isMain: false,
    registeredGroups,
    deps: deps(),
    file: 'owner-alert-1.json',
  });
}

beforeEach(() => {
  sendMessage = vi.fn().mockResolvedValue('sent-1');
  _resetIpcMessageRegistryForTests();
  _resetOwnerAlertIpcHandlersForTests();
  registerOwnerAlertIpcHandlers();
});

describe('owner_alert routing', () => {
  it('routes the alert to the main group, never the source chat', async () => {
    await dispatch({
      type: 'owner_alert',
      alertType: 'code-execution',
      action: 'went-silent',
      sender: 'critskiy',
      request: 'docker rm -f $(docker ps -a -q --filter label=nanoclaw)',
    });
    expect(sendMessage).toHaveBeenCalledOnce();
    const [jid, text] = sendMessage.mock.calls[0];
    expect(jid).toBe(MAIN_JID);
    expect(text).toContain('⚠️ Suspicious request — WTF Pod Chat');
    expect(text).toContain('Type: code execution');
    expect(text).toContain(
      '<untrusted-input source="untrusted-container:wtf-pod-chat">',
    );
    expect(text).toContain('Sender: critskiy');
  });

  it('ignores a chat target smuggled in the payload', async () => {
    // The handler must never honor a payload-supplied destination — the
    // alert goes to main regardless.
    await dispatch({
      type: 'owner_alert',
      chatJid: SOURCE_JID,
      sender: 'critskiy',
    });
    expect(sendMessage).toHaveBeenCalledOnce();
    expect(sendMessage.mock.calls[0][0]).toBe(MAIN_JID);
  });

  it('does not send and does not throw when no main group is registered', async () => {
    await dispatch(
      { type: 'owner_alert', sender: 'critskiy' },
      { [SOURCE_JID]: SOURCE },
    );
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('suppresses a second alert from the same group inside the cooldown', async () => {
    await dispatch({ type: 'owner_alert', sender: 'critskiy' });
    await dispatch({ type: 'owner_alert', sender: 'critskiy again' });
    expect(sendMessage).toHaveBeenCalledOnce();
  });

  it('degrades to no-throw when the send fails', async () => {
    sendMessage.mockRejectedValueOnce(new Error('channel down'));
    await expect(
      dispatch({ type: 'owner_alert', sender: 'critskiy' }),
    ).resolves.toBeUndefined();
    expect(sendMessage).toHaveBeenCalledOnce();
  });
});
