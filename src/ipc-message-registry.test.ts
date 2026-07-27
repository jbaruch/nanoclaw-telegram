import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  _resetIpcMessageRegistryForTests,
  dispatchIpcMessage,
  hasIpcMessageHandler,
  registerIpcMessageHandler,
  type IpcMessageContext,
} from './ipc-message-registry.js';
import type { IpcDeps } from './ipc.js';

function ctx(overrides: Partial<IpcMessageContext> = {}): IpcMessageContext {
  return {
    data: { type: 'message', chatJid: 'chat@g.us', text: 'hi' },
    sourceGroup: 'grp',
    isMain: false,
    registeredGroups: {},
    deps: {} as IpcDeps,
    file: 'msg-1.json',
    ...overrides,
  };
}

describe('ipc message registry (#878)', () => {
  beforeEach(() => {
    _resetIpcMessageRegistryForTests();
  });

  it('dispatches a payload to the handler registered for its type', async () => {
    const handler = vi.fn();
    registerIpcMessageHandler('message', handler);
    const c = ctx();
    expect(await dispatchIpcMessage(c)).toBe(true);
    expect(handler).toHaveBeenCalledOnce();
    // The handler receives the verified identity, not payload-derived
    // fields — that derivation is the security boundary.
    expect(handler.mock.calls[0][0]).toMatchObject({
      sourceGroup: 'grp',
      isMain: false,
    });
  });

  it('awaits an async handler before reporting the dispatch complete', async () => {
    const order: string[] = [];
    registerIpcMessageHandler('message', async () => {
      await new Promise((r) => setTimeout(r, 5));
      order.push('handler');
    });
    await dispatchIpcMessage(ctx());
    order.push('after-dispatch');
    // The poller unlinks the IPC file right after dispatch resolves, so
    // a handler that hadn't finished would lose its file mid-send.
    expect(order).toEqual(['handler', 'after-dispatch']);
  });

  it('returns false for an unregistered type without throwing', async () => {
    registerIpcMessageHandler('message', vi.fn());
    expect(
      await dispatchIpcMessage(ctx({ data: { type: 'no_such_type' } })),
    ).toBe(false);
  });

  it('returns false for a payload with no type at all', async () => {
    expect(await dispatchIpcMessage(ctx({ data: {} }))).toBe(false);
  });

  it('rejects a duplicate registration as the wiring bug it is', () => {
    registerIpcMessageHandler('message', vi.fn());
    expect(() => registerIpcMessageHandler('message', vi.fn())).toThrow(
      /already registered/,
    );
  });

  it('reports registration state via hasIpcMessageHandler', () => {
    expect(hasIpcMessageHandler('message')).toBe(false);
    registerIpcMessageHandler('message', vi.fn());
    expect(hasIpcMessageHandler('message')).toBe(true);
  });

  it('propagates a handler throw so the poller can quarantine the file', async () => {
    registerIpcMessageHandler('message', () => {
      throw new Error('send failed');
    });
    await expect(dispatchIpcMessage(ctx())).rejects.toThrow('send failed');
  });
});

describe('registerMessageIpcHandlers wiring', () => {
  beforeEach(() => {
    _resetIpcMessageRegistryForTests();
  });

  it('claims the three outbound-message command names', async () => {
    vi.resetModules();
    const registry = await import('./ipc-message-registry.js');
    registry._resetIpcMessageRegistryForTests();
    const { registerMessageIpcHandlers } =
      await import('./ipc-handlers/messages.js');
    registerMessageIpcHandlers();
    for (const name of ['react_to_message', 'send_file', 'message']) {
      expect(registry.hasIpcMessageHandler(name)).toBe(true);
    }
  });
});
