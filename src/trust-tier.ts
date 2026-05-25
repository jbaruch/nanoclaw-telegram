/**
 * Container trust tier — the security boundary applied to each spawn.
 *
 * Centralized so `onecli-client.ts`, `container-runner.ts`, and future
 * consumers stay in sync on the canonical tier set. `usage-log.ts` keeps
 * its own broader union because it also tags sub-agent attribution
 * sources like `classifier` that don't correspond to a spawn tier.
 */
export type TrustTier = 'main' | 'trusted' | 'untrusted';

export const TRUST_TIERS: readonly TrustTier[] = [
  'main',
  'trusted',
  'untrusted',
] as const;
