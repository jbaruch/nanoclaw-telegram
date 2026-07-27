import { FLIGHT_ASSIST_ENABLED } from '../config.js';
import { logger } from '../logger.js';
import { registerHubitatPlugin } from './hubitat/index.js';

let registered = false;

/**
 * Register every host-plugin module exactly once (#846/#847/#849, epic
 * #844).
 *
 * Host plugins carry personal/domain policy that the platform core must
 * not hard-code: spawn gates (#846), lifecycle hooks for optional
 * listeners (#847), location sinks (#849), IPC handlers as later #844
 * children land. Startup (`src/index.ts`) awaits this before channels
 * connect and before the scheduler loop starts, so every gate, hook,
 * and sink is registered before the first event that could consult it.
 * Idempotent so tests can call it freely.
 *
 * Every plugin is config-gated, so an unconfigured install registers
 * nothing and never loads the policy module behind it (#877):
 * - Hubitat self-gates on `HUBITAT_HUB_IP` and lazy-imports its listener
 *   inside the lifecycle hooks, so only its registration entrypoint is
 *   evaluated here.
 * - Flight-assist has no equivalent inner seam — `SpawnGate` is a
 *   synchronous callback, so its policy cannot be lazily imported from
 *   inside the gate. The gate therefore sits at the import, which is why
 *   these are dynamic imports and why this function is async: with
 *   `FLIGHT_ASSIST_ENABLED` unset, `flight-assist-spawn-gate.js` and
 *   `flight-assist-location-sink.js` are never loaded at all.
 */
export async function registerHostPlugins(): Promise<void> {
  if (registered) return;
  registered = true;
  registerHubitatPlugin();
  if (!FLIGHT_ASSIST_ENABLED) {
    logger.debug(
      'FLIGHT_ASSIST_ENABLED unset or disabled — skipping flight-assist spawn gate + location sink registration',
    );
    return;
  }
  const { registerFlightAssistSpawnGate } =
    await import('./flight-assist-spawn-gate.js');
  registerFlightAssistSpawnGate();
  const { registerFlightAssistLocationSink } =
    await import('./flight-assist-location-sink.js');
  registerFlightAssistLocationSink();
}
