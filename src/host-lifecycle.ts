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
 *   - Hooks are isolation boundaries: one hook's thrown/rejected Error
 *     is logged (with the hook's name) and the remaining hooks still
 *     run. An optional integration must never take down platform
 *     startup, and a broken listener must never block the rest of
 *     shutdown (queue drain, channel disconnect). A non-Error throwable
 *     is a programming defect, not a hook failure, and propagates per
 *     `coding-policy: error-handling`.
 *   - Each hook is bounded by `HOOK_TIMEOUT_MS`: a hook that hangs is
 *     logged and abandoned so a wedged plugin cannot stall startup or
 *     the platform teardown that follows the shutdown hooks.
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

/**
 * Per-hook execution bound. A hook that hasn't settled by this deadline
 * is logged and abandoned (its promise keeps running unobserved — there
 * is no cancellation in JS) so a wedged optional integration cannot
 * stall the remaining hooks or, at shutdown, the queue/channel teardown
 * behind them. Sized well above any legitimate hook (Hubitat's
 * WebSocket start/stop settles in milliseconds) while staying inside
 * deploy.sh's SIGTERM patience.
 */
const HOOK_TIMEOUT_MS = 15_000;

class HookTimeoutError extends Error {
  constructor(name: string) {
    super(
      `Lifecycle hook "${name}" did not settle within ${HOOK_TIMEOUT_MS}ms`,
    );
    this.name = 'HookTimeoutError';
  }
}

function withTimeout(name: string, run: Promise<void>): Promise<void> {
  let timer: NodeJS.Timeout | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () => reject(new HookTimeoutError(name)),
      HOOK_TIMEOUT_MS,
    );
  });
  return Promise.race([run, deadline]).finally(() => clearTimeout(timer));
}

async function runHooks(list: NamedHook[], phase: string): Promise<void> {
  for (const { name, fn } of list) {
    try {
      await withTimeout(name, Promise.resolve().then(fn));
    } catch (err) {
      // outer-boundary-process-contract — this loop is the host's sole
      // execution boundary around third-party plugin hook code, invoked
      // directly from `main()` (startup) and the SIGTERM/SIGINT handler
      // (shutdown); no frame above it can preserve per-hook isolation.
      //   - Caller's silent-failure shape: deploy.sh / launchd read a
      //     hung or non-zero-exiting shutdown as failed teardown — the
      //     SIGTERM handler never reaches queue.shutdown / channel
      //     disconnect / process.exit(0), so the supervisor SIGKILLs
      //     and cascades 137 across every in-flight agent container;
      //     at startup, a propagating hook crash-loops the platform.
      //   - What the catch emits: a structured ERROR log carrying the
      //     hook's name, the phase, and the failure (incl. our own
      //     HookTimeoutError for hung hooks); the remaining hooks and
      //     the platform teardown behind them still run.
      //   - Why propagation breaks the contract: one broken OPTIONAL
      //     integration would abort platform startup or skip the rest
      //     of shutdown — the exact inversion of the plugin/platform
      //     trust relationship (#847).
      // Narrowest everything-except-defects form: an Error is a hook
      // failure and is handled; a non-Error throwable is a programming
      // defect and propagates.
      if (!(err instanceof Error)) throw err;
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
