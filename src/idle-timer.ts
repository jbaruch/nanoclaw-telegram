/**
 * Per-chat idle-close timer control for the user-facing message loop
 * (#506).
 *
 * Kept outside `processGroupMessages` so the inbound-message piping
 * path can re-anchor the timeout when a follow-up is delivered into
 * an already active container. The map value identity is load-bearing:
 * stale timers from an old container generation must not close a
 * fresh replacement container that took over the same chat.
 *
 * The action on timer expiry (typically `queue.closeStdin`) is
 * injected so this module stays pure and unit-testable without
 * pulling in the queue or channel layers.
 */
import { IDLE_TIMEOUT } from './config.js';
import { logger } from './logger.js';
import type { RegisteredGroup } from './types.js';

export type IdleTimerResetReason = 'agent-output' | 'user-input';

export interface IdleTimerControl {
  reset(reason: IdleTimerResetReason): void;
  clear(): void;
}

const activeIdleTimers = new Map<string, IdleTimerControl>();

/**
 * Install a fresh idle-timer control for the given chat. Any stale
 * control still registered for the same chatJid is cleared first so
 * a previous container generation's timer can never close a fresh
 * replacement container.
 *
 * Tier semantics: main/trusted groups use `IDLE_TIMEOUT` (default 30
 * min). Untrusted groups use a hard-coded 5 min — they receive much
 * less traffic and pay no cold-start tax that warrants a long-lived
 * container.
 */
export function installIdleTimerControl(
  chatJid: string,
  group: RegisteredGroup,
  onTimeout: () => void,
): IdleTimerControl {
  const stale = activeIdleTimers.get(chatJid);
  if (stale) {
    stale.clear();
    activeIdleTimers.delete(chatJid);
    logger.debug(
      { group: group.name, chatJid },
      'Cleared stale idle timer before starting a fresh container',
    );
  }

  const timeoutMs =
    group.isMain || group.containerConfig?.trusted ? IDLE_TIMEOUT : 300_000;
  let idleTimer: ReturnType<typeof setTimeout> | null = null;

  const control: IdleTimerControl = {
    reset(reason) {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        // If another container generation installed a new control for
        // this chat, this timer is stale. Do not close whatever happens
        // to be active now.
        if (activeIdleTimers.get(chatJid) !== control) return;
        activeIdleTimers.delete(chatJid);
        logger.debug(
          { group: group.name, chatJid, reason, timeoutMs },
          'Idle timeout, closing container stdin',
        );
        onTimeout();
      }, timeoutMs);
    },
    clear() {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = null;
    },
  };

  activeIdleTimers.set(chatJid, control);
  return control;
}

/**
 * Look up the currently-installed idle-timer control for a chat.
 * Used by the user-input pipe path to re-anchor the timeout when a
 * follow-up message arrives. Returns `undefined` if no control is
 * currently installed (e.g., no active processGroupMessages cycle).
 */
export function getActiveIdleTimer(
  chatJid: string,
): IdleTimerControl | undefined {
  return activeIdleTimers.get(chatJid);
}

/**
 * End-of-cycle cleanup. Drops the chat's entry from the map only if
 * the supplied control is still the active one — protects against a
 * concurrent processGroupMessages cycle that has already installed
 * a fresh control.
 */
export function releaseIdleTimerControl(
  chatJid: string,
  control: IdleTimerControl,
): void {
  if (activeIdleTimers.get(chatJid) === control) {
    control.clear();
    activeIdleTimers.delete(chatJid);
  }
}

/**
 * Test-only: drop every active timer and clear the map. Production
 * code must not call this — controls are managed exclusively by the
 * install/release lifecycle above.
 */
export function _resetIdleTimerStateForTesting(): void {
  for (const control of activeIdleTimers.values()) control.clear();
  activeIdleTimers.clear();
}
