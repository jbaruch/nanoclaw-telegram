import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mutable config surface shared with the vi.mock factories below —
// `HUBITAT_HUB_IP` is a module const, so each test picks its value and
// re-imports the modules fresh via vi.resetModules().
const state = vi.hoisted(() => ({ hubIp: '' }));
const listener = vi.hoisted(() => ({
  startHubitatListener: vi.fn(),
  stopHubitatListener: vi.fn(),
}));

vi.mock('../../config.js', async () => {
  const actual =
    await vi.importActual<typeof import('../../config.js')>('../../config.js');
  return {
    ...actual,
    get HUBITAT_HUB_IP() {
      return state.hubIp;
    },
  };
});
vi.mock('./listener.js', () => listener);

/**
 * Re-import the plugin + lifecycle modules with fresh module state (the
 * plugin's once-guard and the lifecycle hook lists are module-global).
 */
async function freshImports() {
  vi.resetModules();
  const plugin = await import('./index.js');
  const lifecycle = await import('../../host-lifecycle.js');
  return { plugin, lifecycle };
}

beforeEach(() => {
  listener.startHubitatListener.mockClear();
  listener.stopHubitatListener.mockClear();
});

describe('registerHubitatPlugin', () => {
  it('registers nothing when HUBITAT_HUB_IP is unset', async () => {
    state.hubIp = '';
    const { plugin, lifecycle } = await freshImports();
    plugin.registerHubitatPlugin();
    await lifecycle.runStartupHooks();
    await lifecycle.runShutdownHooks();
    expect(listener.startHubitatListener).not.toHaveBeenCalled();
    expect(listener.stopHubitatListener).not.toHaveBeenCalled();
  });

  it('registers startup + shutdown hooks when configured', async () => {
    state.hubIp = '192.168.10.99';
    const { plugin, lifecycle } = await freshImports();
    plugin.registerHubitatPlugin();
    await lifecycle.runStartupHooks();
    expect(listener.startHubitatListener).toHaveBeenCalledOnce();
    expect(listener.stopHubitatListener).not.toHaveBeenCalled();
    await lifecycle.runShutdownHooks();
    expect(listener.stopHubitatListener).toHaveBeenCalledOnce();
  });

  it('is idempotent — a second registration adds no duplicate hooks', async () => {
    state.hubIp = '192.168.10.99';
    const { plugin, lifecycle } = await freshImports();
    plugin.registerHubitatPlugin();
    // Must not throw the lifecycle registry's duplicate-name error and
    // must not double-start the listener.
    plugin.registerHubitatPlugin();
    await lifecycle.runStartupHooks();
    expect(listener.startHubitatListener).toHaveBeenCalledOnce();
  });
});
