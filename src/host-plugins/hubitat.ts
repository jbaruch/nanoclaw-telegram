import { HUBITAT_HUB_IP } from '../config.js';
import {
  registerShutdownHook,
  registerStartupHook,
} from '../host-lifecycle.js';
import {
  startHubitatListener,
  stopHubitatListener,
} from '../hubitat-listener.js';
import { logger } from '../logger.js';

let registered = false;

/**
 * Register the Hubitat EventSocket listener's lifecycle hooks (#847).
 * First consumer of the host lifecycle registry: `main()` no longer
 * hard-codes Hubitat start/stop next to platform teardown. Registration
 * is config-gated — with `HUBITAT_HUB_IP` unset the hooks are never
 * registered, so an unconfigured install runs zero Hubitat code at
 * startup/shutdown. (Full extraction of the listener + its
 * `smart_home_events` storage into an optional plugin is #848; this
 * lands the seam.)
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
  registerStartupHook('hubitat-listener', () => startHubitatListener());
  registerShutdownHook('hubitat-listener', () => stopHubitatListener());
}
