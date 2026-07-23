import { describe, it, expect, beforeEach, vi } from 'vitest';

import type { LocationRecord } from '../types.js';

const writer = vi.hoisted(() => ({
  writeFlightAssistLocation: vi.fn(),
}));
vi.mock('./flight-assist-location.js', () => writer);

/**
 * Re-import the plugin + sink-registry modules with fresh module state
 * (the plugin's once-guard and the sink list are module-global).
 */
async function freshImports() {
  vi.resetModules();
  const plugin = await import('./flight-assist-location-sink.js');
  const sinks = await import('../location-sinks.js');
  const state = await import('../orchestrator-state.js');
  return { plugin, sinks, state };
}

const RECORD: LocationRecord = {
  chat_jid: 'tg:123',
  sender: 'owner-1',
  message_id: 'msg-1',
  latitude: 41.03,
  longitude: -73.76,
  source: 'static',
  recorded_at: '2026-07-21T12:00:00.000Z',
};

beforeEach(() => {
  writer.writeFlightAssistLocation.mockClear();
});

describe('registerFlightAssistLocationSink', () => {
  it('routes fanned-out locations to writeFlightAssistLocation with host wiring', async () => {
    const { plugin, sinks, state } = await freshImports();
    plugin.registerFlightAssistLocationSink();
    sinks.runLocationSinks(RECORD);
    expect(writer.writeFlightAssistLocation).toHaveBeenCalledOnce();
    const [record, opts] = writer.writeFlightAssistLocation.mock.calls[0] as [
      LocationRecord,
      { groups: unknown; ownerSenderId: string; dataDir: string },
    ];
    expect(record).toBe(RECORD);
    // The groups reference is the live orchestrator registry — the sink
    // must not snapshot it at registration time.
    expect(opts.groups).toBe(state.registeredGroups);
    expect(typeof opts.dataDir).toBe('string');
  });

  it('is idempotent — a second registration adds no duplicate sink', async () => {
    const { plugin, sinks } = await freshImports();
    plugin.registerFlightAssistLocationSink();
    // Must not throw the registry's duplicate-name error and must not
    // double-write per location.
    plugin.registerFlightAssistLocationSink();
    sinks.runLocationSinks(RECORD);
    expect(writer.writeFlightAssistLocation).toHaveBeenCalledOnce();
  });
});
