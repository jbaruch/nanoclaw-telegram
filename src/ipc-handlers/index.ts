import { registerChatAdminIpcHandlers } from './chat-admin.js';
import { registerGroupConfigIpcHandlers } from './group-config.js';
import { registerGroupIpcHandlers } from './groups.js';
import { registerLearnedTriggerIpcHandlers } from './learned-triggers.js';
import { registerOpsIpcHandlers } from './ops.js';
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
  registerGroupIpcHandlers();
  registerGroupConfigIpcHandlers();
  registerLearnedTriggerIpcHandlers();
  registerChatAdminIpcHandlers();
  registerOpsIpcHandlers();
}

/**
 * Reset the once-guard alongside `_resetIpcRegistryForTests` so a test
 * that wiped the registry can re-register the core handlers.
 *
 * @internal — test-only export, stripped from the public `.d.ts`
 * surface (`stripInternal: true`).
 */
export function _resetCoreIpcHandlersForTests(): void {
  registered = false;
}
