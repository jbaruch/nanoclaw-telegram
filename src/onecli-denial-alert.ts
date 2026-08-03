/**
 * #893 — detection for an OneCLI gateway credential denial.
 *
 * Since #758 the credential proxy tunnels each container's
 * `/v1/messages` through the OneCLI gateway under that container's own
 * tier agent. When the vault has no usable grant for that agent the
 * gateway answers 401 and every turn in the tier dies. The host logged
 * nothing distinguishable — `OneCLI gateway config applied to
 * container`, `Spawning container agent`, done — and the `untrusted`
 * tier stayed dead for ~19 days until the operator noticed the chats
 * were silent.
 *
 * A log line alone would not have shortened that. The heartbeat reads
 * named host-log artifacts (`deploy-kills.log`,
 * `tessl-update-window.log`), never `orchestrator.log` prose, so an
 * ERROR line there is only found by someone already investigating.
 * This module is the second half: a consecutive-denial streak per tier
 * raises the same kind of immediate operator alert a timeout kill does
 * (#892).
 *
 * Kept separate from `credential-proxy.ts` and free of any channel
 * import so the thresholds and wording are unit-testable without
 * standing up a proxy, an upstream, or a chat channel. The proxy owns
 * *when* a denial happened; this owns *whether it is worth saying* and
 * *what it says*.
 */

import type { TrustTier } from './onecli-client.js';

/**
 * Consecutive denials for one tier before the first alert.
 *
 * Not 1: a single 401 is also what a gateway restart or a token
 * rotation mid-flight looks like, and those self-heal on the next
 * request. Three in a row with no success in between is a standing
 * condition, and at real traffic rates a tier reaches it within
 * seconds of actually breaking.
 */
export const DENIAL_ALERT_STREAK = 3;

/**
 * Minimum gap between two alerts for the SAME tier.
 *
 * A tier whose grant is missing denies every request, so the streak
 * condition holds continuously — without this the operator would get
 * one message per request. One per hour keeps a persistent break
 * visible in chat without flooding it.
 */
export const DENIAL_ALERT_COOLDOWN_MS = 60 * 60 * 1000;

/** Inputs the alert text is built from. */
export interface OneCliDenialAlertInput {
  /** Trust tier whose agent the request was minted against. */
  tier: TrustTier;
  /** Vault agent identifier for that tier. */
  agentIdentifier: string;
  /** Upstream host that answered, e.g. `api.anthropic.com`. */
  upstreamHost: string;
  /** HTTP status the gateway returned (401 or 403). */
  status: number;
  /** Consecutive denials counted for this tier when the alert fired. */
  consecutiveDenials: number;
}

/**
 * Build the operator-facing alert text for a run of gateway denials.
 *
 * Pure so the wording is testable without a gateway. Deliberately plain
 * text: it crosses whatever channel the main group is on, and
 * channel-specific markup renders as literal characters on the others
 * (same reason `buildTimeoutKillAlert` is plain).
 *
 * Carries no token, no proxy URL, and no `.env` value — the tier, the
 * agent identifier, the host, and the status are the whole payload
 * (`coding-policy: no-secrets`).
 */
export function buildOneCliDenialAlert(input: OneCliDenialAlertInput): string {
  const { tier, agentIdentifier, upstreamHost, status, consecutiveDenials } =
    input;
  return [
    `🔒 OneCLI gateway is denying ${tier}-tier requests`,
    `${consecutiveDenials} consecutive ${status}s from ${upstreamHost}. Every turn in this tier is failing.`,
    `The vault agent '${agentIdentifier}' has no usable grant for the Anthropic credential. Check its effective-credentials in the vault and add the policy rule if it is missing.`,
  ].join('\n');
}

/**
 * Per-tier consecutive-denial streaks and alert cooldowns.
 *
 * `now` is passed in rather than read from the clock so the streak and
 * cooldown behaviour is testable against a fixed reference
 * (`coding-policy: testing-standards` Determinism).
 */
export class OneCliDenialTracker {
  private streaks = new Map<TrustTier, number>();
  private lastAlertAt = new Map<TrustTier, number>();

  /**
   * Record a denied request for `tier`.
   *
   * Returns the consecutive-denial count when this denial should raise
   * an operator alert, or `null` when it should not — the streak is
   * still short, or this tier already alerted inside the cooldown.
   */
  recordDenial(tier: TrustTier, nowMs: number): number | null {
    const streak = (this.streaks.get(tier) ?? 0) + 1;
    this.streaks.set(tier, streak);
    if (streak < DENIAL_ALERT_STREAK) return null;

    const last = this.lastAlertAt.get(tier);
    if (last !== undefined && nowMs - last < DENIAL_ALERT_COOLDOWN_MS) {
      return null;
    }
    this.lastAlertAt.set(tier, nowMs);
    return streak;
  }

  /**
   * Record a request for `tier` that the credential let through.
   *
   * Clears the streak so denials separated by working requests never
   * accumulate into a false standing-condition alert. The cooldown
   * stamp is deliberately left alone: it throttles alerts, and a tier
   * that recovers and breaks again within the hour is the same
   * incident, not a new one.
   *
   * Only a 2xx counts. A 429 or a 5xx says nothing about whether the
   * credential is usable, and clearing the streak on one would let an
   * upstream hiccup interleaved with real denials hide a tier that is
   * wholly unable to authenticate.
   */
  recordSuccess(tier: TrustTier): void {
    this.streaks.delete(tier);
  }

  /** Consecutive denials currently counted for `tier`. Test surface. */
  streakFor(tier: TrustTier): number {
    return this.streaks.get(tier) ?? 0;
  }
}

/** The slice of a channel the alert delivery needs. */
export interface AlertChannel {
  isConnected(): boolean;
  sendMessage(jid: string, text: string): Promise<unknown>;
}

/** Everything the sender reaches for, injected so it stays testable. */
export interface TierDenialAlertDeps {
  /** Registered groups keyed by chat JID, read at send time. */
  registeredGroups: () => Record<string, { isMain?: boolean }>;
  /** Channel owning a JID, or null/undefined when none is wired. */
  findChannel: (jid: string) => AlertChannel | null | undefined;
  /** Structured warn sink for the cases that cannot deliver. */
  warn: (fields: Record<string, unknown>, message: string) => void;
}

/**
 * Build the `onTierDenialAlert` callback the credential proxy calls.
 *
 * Lives here rather than inline in `index.ts` so the delivery decisions
 * — which group receives it, what happens when none is main or the
 * channel is down, what a failed send does — are reachable by tests
 * without standing up the orchestrator.
 *
 * Everything it touches arrives through `deps` as a structural type,
 * so this module still imports no channel and no group registry.
 *
 * Routed to the main group rather than the affected tier's own chats,
 * same as the timeout-kill alert: the operator reads main, and a tier
 * this is about is by definition unable to answer.
 *
 * Every failure to deliver degrades to a warn. The ERROR line the proxy
 * already wrote is the durable record; this is the notification, and a
 * send fault must not propagate into the proxy's response handling.
 */
export function createTierDenialAlertSender(
  deps: TierDenialAlertDeps,
): (text: string) => void {
  return (text: string): void => {
    const groups = deps.registeredGroups();
    const mainJid = Object.keys(groups).find((jid) => groups[jid].isMain);
    if (!mainJid) {
      deps.warn(
        { text },
        'OneCLI denial alert not sent — no group is registered as main (#893)',
      );
      return;
    }
    const mainChannel = deps.findChannel(mainJid);
    if (!mainChannel || !mainChannel.isConnected()) {
      deps.warn(
        { mainJid },
        'OneCLI denial alert not sent — main channel is not connected (#893)',
      );
      return;
    }
    // `.catch` rather than await: the proxy calls this synchronously
    // from a response handler, and an unhandled rejection is something
    // newer Node can terminate the orchestrator over.
    void mainChannel.sendMessage(mainJid, text).catch((err: unknown) => {
      deps.warn(
        { err, mainJid },
        'OneCLI denial alert send failed (the denial is still logged) (#893)',
      );
    });
  };
}
