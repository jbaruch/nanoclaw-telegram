import { registerFlightAssistSpawnGate } from './flight-assist-spawn-gate.js';

let registered = false;

/**
 * Register every host-plugin module exactly once (#846, epic #844).
 *
 * Host plugins carry personal/domain policy that the platform core must
 * not hard-code (spawn gates today; IPC handlers and lifecycle listeners
 * as later #844 children land). Startup (`src/index.ts`) calls this
 * before the scheduler loop starts so every gate is registered before
 * the first fire is evaluated. Idempotent so tests can call it freely.
 */
export function registerHostPlugins(): void {
  if (registered) return;
  registered = true;
  registerFlightAssistSpawnGate();
}
