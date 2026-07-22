import { HUBITAT_HUB_IP } from '../../config.js';
import {
  registerShutdownHook,
  registerStartupHook,
} from '../../host-lifecycle.js';
import { logger } from '../../logger.js';

let registered = false;

/**
 * Register the Hubitat EventSocket listener's lifecycle hooks (#847,
 * extracted as a self-contained plugin module in #848). Registration
 * is config-gated — with `HUBITAT_HUB_IP` unset the hooks are never
 * registered, and the listener + `smart_home_events` accessor modules
 * are never loaded (they're pulled in via dynamic import inside the
 * hooks), so an unconfigured install runs no listener code and makes
 * no connect attempts — only this registration entrypoint is evaluated
 * at startup. The `smart_home_events` schema migration stays in
 * core `src/db.ts`; product build-out continues under
 * `epic:smart-home`.
 */
export function registerHubitatPlugin(): void {
  if (registered) return;
  registered = true;
  if (!HUBITAT_HUB_IP) {
    logger.debug(
      'HUBITAT_HUB_IP not set — skipping Hubitat lifecycle hook registration',
    );
    return;
  }
  registerStartupHook('hubitat-listener', async () => {
    const { startHubitatListener } = await import('./listener.js');
    startHubitatListener();
  });
  registerShutdownHook('hubitat-listener', async () => {
    const { stopHubitatListener } = await import('./listener.js');
    stopHubitatListener();
  });
}
