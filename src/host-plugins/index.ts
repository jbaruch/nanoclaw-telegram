import { registerFlightAssistSpawnGate } from './flight-assist-spawn-gate.js';
import { registerHubitatPlugin } from './hubitat.js';

let registered = false;

/**
 * Register every host-plugin module exactly once (#846/#847, epic #844).
 *
 * Host plugins carry personal/domain policy that the platform core must
 * not hard-code: spawn gates (#846), lifecycle hooks for optional
 * listeners (#847), IPC handlers as later #844 children land. Startup
 * (`src/index.ts`) calls this before the scheduler loop starts so every
 * gate and hook is registered before the first fire is evaluated and
 * before `runStartupHooks()` runs. Idempotent so tests can call it
 * freely.
 */
export function registerHostPlugins(): void {
  if (registered) return;
  registered = true;
  registerFlightAssistSpawnGate();
  registerHubitatPlugin();
}
