import { describe, it, expect, afterEach, vi } from 'vitest';

import {
  _resetLifecycleHooksForTests,
  registerShutdownHook,
  registerStartupHook,
  runShutdownHooks,
  runStartupHooks,
} from './host-lifecycle.js';
import { logger } from './logger.js';

afterEach(() => {
  // The hook lists are module-global shared state — wipe them so no
  // test's registration leaks into another and order never matters.
  _resetLifecycleHooksForTests();
  vi.restoreAllMocks();
});

describe('host lifecycle hooks', () => {
  it('runs startup hooks in registration order', async () => {
    const order: string[] = [];
    registerStartupHook('first', () => {
      order.push('first');
    });
    registerStartupHook('second', async () => {
      order.push('second');
    });
    registerStartupHook('third', () => {
      order.push('third');
    });
    await runStartupHooks();
    expect(order).toEqual(['first', 'second', 'third']);
  });

  it('runs shutdown hooks in registration order, separately from startup', async () => {
    const calls: string[] = [];
    registerStartupHook('up', () => {
      calls.push('up');
    });
    registerShutdownHook('down-a', () => {
      calls.push('down-a');
    });
    registerShutdownHook('down-b', () => {
      calls.push('down-b');
    });
    await runShutdownHooks();
    expect(calls).toEqual(['down-a', 'down-b']);
  });

  it('throws on duplicate hook names within a phase', () => {
    registerStartupHook('dup', () => {});
    expect(() => registerStartupHook('dup', () => {})).toThrow(
      /Startup hook already registered: dup/,
    );
    // The same name is fine across phases — start/stop pairs share it.
    expect(() => registerShutdownHook('dup', () => {})).not.toThrow();
  });

  it('isolates a throwing hook: logs it and still runs the rest', async () => {
    const errorLog = vi.spyOn(logger, 'error').mockImplementation(() => {});
    const after = vi.fn();
    registerShutdownHook('boom', () => {
      throw new Error('listener exploded');
    });
    registerShutdownHook('after', after);
    await runShutdownHooks();
    expect(after).toHaveBeenCalledOnce();
    expect(errorLog).toHaveBeenCalledWith(
      expect.objectContaining({ hook: 'boom', phase: 'shutdown' }),
      'Lifecycle hook failed',
    );
  });

  it('isolates a rejecting async hook the same way', async () => {
    const errorLog = vi.spyOn(logger, 'error').mockImplementation(() => {});
    const after = vi.fn();
    registerStartupHook('async-boom', async () => {
      throw new Error('async listener exploded');
    });
    registerStartupHook('after', after);
    await runStartupHooks();
    expect(after).toHaveBeenCalledOnce();
    expect(errorLog).toHaveBeenCalledWith(
      expect.objectContaining({ hook: 'async-boom', phase: 'startup' }),
      'Lifecycle hook failed',
    );
  });

  it('is a no-op with nothing registered (platform-only install)', async () => {
    await expect(runStartupHooks()).resolves.toBeUndefined();
    await expect(runShutdownHooks()).resolves.toBeUndefined();
  });

  it('propagates a non-Error throwable (programming defect, not a hook failure)', async () => {
    registerStartupHook('throws-string', () => {
      // A thrown non-Error is the defect shape under test.
      throw 'not an Error instance';
    });
    await expect(runStartupHooks()).rejects.toBe('not an Error instance');
  });

  it('abandons a hung hook at the timeout and still runs the rest', async () => {
    vi.useFakeTimers();
    try {
      const errorLog = vi.spyOn(logger, 'error').mockImplementation(() => {});
      const after = vi.fn();
      registerStartupHook('wedged', () => new Promise<void>(() => {}));
      registerStartupHook('after', after);
      const run = runStartupHooks();
      await vi.advanceTimersByTimeAsync(15_000);
      await run;
      expect(after).toHaveBeenCalledOnce();
      expect(errorLog).toHaveBeenCalledWith(
        expect.objectContaining({
          hook: 'wedged',
          phase: 'startup',
          err: expect.objectContaining({ name: 'HookTimeoutError' }),
        }),
        'Lifecycle hook failed',
      );
    } finally {
      vi.useRealTimers();
    }
  });
});
