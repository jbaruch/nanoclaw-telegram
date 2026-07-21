import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// The tested handlers resolve the target channel via router.findChannel;
// keep the rest of router real so the module loads.
vi.mock('./router.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./router.js')>();
  return { ...actual, findChannel: vi.fn() };
});

import { ipcDeps } from './ipc-deps.js';
import { channels } from './orchestrator-runtime.js';
import { findChannel } from './router.js';

const mockFindChannel = vi.mocked(findChannel);

beforeEach(() => {
  vi.clearAllMocks();
  channels.length = 0;
});

afterEach(() => {
  channels.length = 0;
});

describe('ipcDeps channel-guard handlers', () => {
  it('sendMessage throws when no channel owns the JID', () => {
    mockFindChannel.mockReturnValue(undefined);
    expect(() => ipcDeps.sendMessage('nobody@g.us', 'hi')).toThrow(
      'No channel for JID: nobody@g.us',
    );
  });

  it('sendMessage delegates to the owning channel when one exists', async () => {
    const sendMessage = vi.fn().mockResolvedValue('sent-id');
    mockFindChannel.mockReturnValue({
      name: 'telegram',
      sendMessage,
    } as never);

    await ipcDeps.sendMessage('g@g.us', 'hello', 'reply-1');

    expect(sendMessage).toHaveBeenCalledWith('g@g.us', 'hello', 'reply-1');
  });

  it('sendReaction is a no-op when no channel owns the JID', async () => {
    mockFindChannel.mockReturnValue(undefined);
    // sendReaction is optional on the IpcDeps interface but always wired here.
    await expect(
      ipcDeps.sendReaction!('nobody@g.us', 'm-1', '👍'),
    ).resolves.toBeUndefined();
  });
});
