import { describe, it, expect, vi } from 'vitest';

// Mutable config surface shared with the vi.mock factory below —
// `FLIGHT_ASSIST_ENABLED` is a module const, so each test picks its value
// and re-imports the modules fresh via vi.resetModules(). Same shape as
// `hubitat/index.test.ts`.
const state = vi.hoisted(() => ({ flightAssist: false }));

vi.mock('../config.js', async () => {
  const actual =
    await vi.importActual<typeof import('../config.js')>('../config.js');
  return {
    ...actual,
    // Hubitat stays off in every case here so the assertions below are
    // about the flight-assist gate alone.
    HUBITAT_HUB_IP: '',
    get FLIGHT_ASSIST_ENABLED() {
      return state.flightAssist;
    },
  };
});

/**
 * Re-import the composition root plus the two registries with fresh
 * module state (the once-guard and both registry lists are
 * module-global).
 */
async function freshImports() {
  vi.resetModules();
  const plugins = await import('./index.js');
  const spawnGates = await import('../spawn-gates.js');
  const locationSinks = await import('../location-sinks.js');
  return { plugins, spawnGates, locationSinks };
}

describe('registerHostPlugins — flight-assist config gate (#877)', () => {
  it('registers no spawn gate and no location sink when FLIGHT_ASSIST_ENABLED is unset', async () => {
    state.flightAssist = false;
    const { plugins, spawnGates, locationSinks } = await freshImports();
    await plugins.registerHostPlugins();

    // `evaluateSpawnGate` returns null for a skill with no registered
    // gate — i.e. the scheduler spawns as normal, unaware of trip windows.
    expect(
      spawnGates.evaluateSpawnGate(
        'tessl__flight-assist',
        '/nonexistent-group-dir',
        new Date('2026-07-01T12:00:00.000Z'),
      ),
    ).toBeNull();

    // No sink ran, so nothing tried to write `current-location.json`.
    // `runLocationSinks` over an empty registry is a no-op.
    expect(() =>
      locationSinks.runLocationSinks({
        chat_jid: 'tg:1',
        sender: '1',
        message_id: 'm1',
        latitude: 32.08,
        longitude: 34.78,
        source: 'static',
        recorded_at: '2026-07-01T12:00:00.000Z',
      }),
    ).not.toThrow();
  });

  /**
   * Load probe: a `vi.doMock` factory body runs only when something
   * actually imports the module, so the recorded list is a direct
   * observation of which policy modules were pulled in — not merely of
   * whether their register functions were called.
   */
  async function loadedPolicyModules(): Promise<string[]> {
    vi.resetModules();
    const loaded: string[] = [];
    vi.doMock('./flight-assist-spawn-gate.js', () => {
      loaded.push('spawn-gate');
      return { registerFlightAssistSpawnGate: vi.fn() };
    });
    vi.doMock('./flight-assist-location-sink.js', () => {
      loaded.push('location-sink');
      return { registerFlightAssistLocationSink: vi.fn() };
    });
    try {
      const plugins = await import('./index.js');
      await plugins.registerHostPlugins();
      return loaded;
    } finally {
      vi.doUnmock('./flight-assist-spawn-gate.js');
      vi.doUnmock('./flight-assist-location-sink.js');
    }
  }

  it('does not even load the flight-assist policy modules when unset', async () => {
    state.flightAssist = false;
    expect(await loadedPolicyModules()).toEqual([]);
  });

  it('loads both policy modules when set (positive control for the probe)', async () => {
    // Without this the empty-list assertion above would also pass on a
    // probe that can never observe a load.
    state.flightAssist = true;
    expect(await loadedPolicyModules()).toEqual([
      'spawn-gate',
      'location-sink',
    ]);
  });

  it('registers the trip-window gate when FLIGHT_ASSIST_ENABLED is set', async () => {
    state.flightAssist = true;
    const { plugins, spawnGates } = await freshImports();
    await plugins.registerHostPlugins();

    // A group dir with no travel-db.json → the gate is registered and
    // returns its "absent itinerary" verdict rather than null.
    const verdict = spawnGates.evaluateSpawnGate(
      'tessl__flight-assist',
      '/nonexistent-group-dir',
      new Date('2026-07-01T12:00:00.000Z'),
    );
    expect(verdict).not.toBeNull();
    expect(verdict?.eligible).toBe(false);
  });

  it('is idempotent — a second call registers no duplicates', async () => {
    state.flightAssist = true;
    const { plugins, spawnGates } = await freshImports();
    await plugins.registerHostPlugins();
    // Both registries throw on a duplicate name, so a second call that
    // re-registered would reject here.
    await expect(plugins.registerHostPlugins()).resolves.toBeUndefined();
    expect(
      spawnGates.evaluateSpawnGate(
        'tessl__flight-assist',
        '/nonexistent-group-dir',
        new Date('2026-07-01T12:00:00.000Z'),
      ),
    ).not.toBeNull();
  });
});
