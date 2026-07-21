import { logger } from './logger.js';

/**
 * Host lifecycle hook registry (#847, epic #844).
 *
 * Optional host integrations (Hubitat listener, future host plugins)
 * register startup/shutdown work here instead of sitting as
 * unconditional calls inside `main()`. Platform init stays hard-wired
 * in `src/index.ts`; hooks are for the OPTIONAL layer on top.
 *
 * Execution contract:
 *   - Hooks run in registration order.
 *   - Hooks are isolation boundaries: one hook's throw/rejection is
 *     logged (with the hook's name) and the remaining hooks still run.
 *     An optional integration must never take down platform startup,
 *     and a broken listener must never block the rest of shutdown
 *     (queue drain, channel disconnect). That isolation is the reason
 *     the per-hook catch below is intentionally unfiltered: whatever a
 *     plugin hook throws is its own failure, surfaced via the error
 *     log, never a reason to skip its peers.
 */

export type LifecycleHook = () => void | Promise<void>;

interface NamedHook {
  name: string;
  fn: LifecycleHook;
}

const startupHooks: NamedHook[] = [];
const shutdownHooks: NamedHook[] = [];

function register(list: NamedHook[], kind: string, hook: NamedHook): void {
  if (list.some((h) => h.name === hook.name)) {
    throw new Error(`${kind} hook already registered: ${hook.name}`);
  }
  list.push(hook);
}

/** Register optional work to run after platform init. Duplicate names throw. */
export function registerStartupHook(name: string, fn: LifecycleHook): void {
  register(startupHooks, 'Startup', { name, fn });
}

/** Register optional teardown to run at shutdown, before queue/channel teardown. */
export function registerShutdownHook(name: string, fn: LifecycleHook): void {
  register(shutdownHooks, 'Shutdown', { name, fn });
}

async function runHooks(list: NamedHook[], phase: string): Promise<void> {
  for (const { name, fn } of list) {
    try {
      await fn();
      // eslint-disable-next-line no-catch-all/no-catch-all -- hook-isolation contract (module doc): whatever an optional plugin hook throws is its own failure, surfaced via the error log with the hook's name; rethrowing would let one broken integration abort platform startup or skip the remaining shutdown hooks
    } catch (err) {
      logger.error({ err, hook: name, phase }, 'Lifecycle hook failed');
    }
  }
}

/** Run every registered startup hook in registration order. */
export function runStartupHooks(): Promise<void> {
  return runHooks(startupHooks, 'startup');
}

/** Run every registered shutdown hook in registration order. */
export function runShutdownHooks(): Promise<void> {
  return runHooks(shutdownHooks, 'shutdown');
}

/**
 * Wipe both hook lists between tests. The lists are module-global
 * shared state; `testing-standards` requires tests to clean them up so
 * order never matters.
 *
 * @internal — test-only export, stripped from the public `.d.ts`
 * surface (`stripInternal: true`).
 */
export function _resetLifecycleHooksForTests(): void {
  startupHooks.length = 0;
  shutdownHooks.length = 0;
}
