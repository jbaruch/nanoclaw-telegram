import { logger } from './logger.js';
import type { LocationRecord } from './types.js';

/**
 * Location fan-out sink registry (#849, epic #844).
 *
 * Core owns location PERSISTENCE (the `locations` DB row written by
 * `storeLocation` in the channel `onLocation` callback) and this
 * fan-out list. What else happens to a location — the travel tile's
 * `current-location.json` artifact, future consumers — is capability
 * policy, registered by a host-plugin module at startup instead of
 * being hard-coded in the channel path.
 *
 * Sinks run in registration order after the DB write. Each sink is an
 * isolation boundary: a sink failure is logged with the sink's name and
 * never blocks the other sinks or the message loop that delivered the
 * location.
 */

export type LocationSink = (record: LocationRecord) => void | Promise<void>;

interface NamedSink {
  name: string;
  fn: LocationSink;
}

const sinks: NamedSink[] = [];

/** Register a location sink (#849). Duplicate names are a wiring bug and throw. */
export function registerLocationSink(name: string, fn: LocationSink): void {
  if (sinks.some((s) => s.name === name)) {
    throw new Error(`Location sink already registered: ${name}`);
  }
  sinks.push({ name, fn });
}

/**
 * Fan a stored location out to every registered sink, in order. The
 * caller (the channel `onLocation` callback) is synchronous, so an
 * async sink runs fire-and-forget: its rejection is caught via
 * `.catch` below and logged the same as a synchronous throw — never
 * left as an unhandled rejection.
 */
export function runLocationSinks(record: LocationRecord): void {
  for (const { name, fn } of sinks) {
    try {
      const out = fn(record);
      if (out && typeof out.then === 'function') {
        out.catch((err: unknown) => {
          logger.error({ err, sink: name }, 'Location sink failed');
        });
      }
      // Isolation boundary around plugin sink code, invoked from the
      // channel onLocation callback:
      //   - Caller's silent-failure shape: the channel treats a thrown
      //     onLocation as a failed inbound event — the location's DB
      //     row is already written, but a propagating sink error would
      //     surface as a channel-level receive failure and could drop
      //     or retry the message that carried the location.
      //   - What the catch emits: a structured ERROR log with the
      //     sink's name and the failure (async sinks route rejections
      //     to the same log via the .catch above); the remaining sinks
      //     still run.
      //   - Why propagation breaks the contract: one broken OPTIONAL
      //     capability artifact-writer must not disturb core message
      //     handling or the other sinks (#849).
      // eslint-disable-next-line no-catch-all/no-catch-all -- outer-boundary-process-contract
    } catch (err) {
      logger.error({ err, sink: name }, 'Location sink failed');
    }
  }
}

/**
 * Wipe the sink list between tests. The list is module-global shared
 * state; `testing-standards` requires tests to clean it up so order
 * never matters.
 *
 * @internal — test-only export, stripped from the public `.d.ts`
 * surface (`stripInternal: true`).
 */
export function _resetLocationSinksForTests(): void {
  sinks.length = 0;
}
