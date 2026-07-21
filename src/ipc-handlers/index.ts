import { registerTaskIpcHandlers } from './tasks.js';

let registered = false;

/**
 * Register every core IPC command module exactly once (#845).
 * Idempotent so both `startIpcWatcher` and direct `processTaskIpc`
 * callers (tests) can invoke it without double-registration throws.
 * Later slices of the #845 migration add their module's register call
 * here; host plugins call `registerIpcHandler` themselves at startup.
 */
export function registerCoreIpcHandlers(): void {
  if (registered) return;
  registered = true;
  registerTaskIpcHandlers();
}
