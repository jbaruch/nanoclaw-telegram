/**
 * Shared Anthropic LiteLLM→direct bypass decision logic (#610, #675).
 *
 * Both egress paths that leave the orchestrator host point their primary
 * attempt at `ANTHROPIC_BASE_URL` (the `nanoclaw-litellm` gateway) and
 * fall back to `ANTHROPIC_BYPASS_URL` (`https://api.anthropic.com`) on a
 * recoverable failure, so a gateway blip can't take Stage 2 down while
 * agent-container spawns sail through:
 *   - the credential proxy (`credential-proxy.ts`) at the raw-socket
 *     layer, for agent containers;
 *   - the Stage 2 Haiku classifier (`gates/haiku-classifier.ts`) at the
 *     SDK layer, for the host-side spawn gate.
 *
 * The transport differs (HTTP stream vs SDK), so the retry MECHANISM
 * can't be shared — but the DECISION logic is identical and lives here
 * so the two paths can't drift: which status codes / network errnos are
 * bypass-eligible, and whether bypass is even enabled for a given config.
 * The SDK-layer retry orchestration (`createMessageWithBypass`) is shared
 * by every SDK caller.
 */
import Anthropic from '@anthropic-ai/sdk';

/**
 * 5xx upstream statuses where a fresh attempt against anthropic-direct is
 * more likely to succeed than passing the error through — the LiteLLM
 * container is up but its own router fallback chain is exhausted.
 */
export const BYPASS_TRIGGER_STATUS_CODES = new Set([500, 502, 503, 504]);

/**
 * Node errno codes that mean the primary upstream was unreachable (the
 * LiteLLM container is down / not resolvable). Matches the set the
 * credential proxy keys its socket-level bypass on.
 */
export const BYPASS_REACHABILITY_CODES = new Set([
  'ECONNREFUSED',
  'ENOTFOUND',
  'EHOSTUNREACH',
  'ECONNRESET',
]);

export function isReachabilityErrorCode(code: unknown): boolean {
  return typeof code === 'string' && BYPASS_REACHABILITY_CODES.has(code);
}

const DEFAULT_ANTHROPIC_URL = 'https://api.anthropic.com';

/**
 * Resolve whether the LiteLLM→direct bypass is active for a given config
 * and the effective URL to bypass TO. The URLs come from operator-set env
 * vars (`ANTHROPIC_BASE_URL` / `ANTHROPIC_BYPASS_URL`); each is resolved
 * via `parseAnthropicUrlOrDefault`, so a malformed value degrades to
 * anthropic-direct (no throw) and is judged on that RESOLVED origin. A
 * malformed `ANTHROPIC_BYPASS_URL` therefore bypasses TO anthropic-direct
 * (enabled) — matching the caller's "falling back to anthropic-direct"
 * handling rather than silently disabling the path. Bypass is a no-op —
 * reported disabled — when the resolved primary and bypass origins match
 * (nothing to fall back to) or there's no API key to authenticate the
 * direct attempt. The returned `bypassUrl` is the effective target: the
 * provided value when valid, else anthropic-direct.
 */
export function resolveBypassTarget(opts: {
  baseUrl?: string;
  bypassUrl?: string;
  hasApiKey: boolean;
}): { enabled: boolean; bypassUrl: string } {
  const base = parseAnthropicUrlOrDefault(opts.baseUrl);
  const bypass = parseAnthropicUrlOrDefault(opts.bypassUrl);
  const enabled = opts.hasApiKey && base.url.origin !== bypass.url.origin;
  const bypassUrl =
    opts.bypassUrl && !bypass.fellBackToDefault
      ? opts.bypassUrl
      : DEFAULT_ANTHROPIC_URL;
  return { enabled, bypassUrl };
}

/**
 * Parse an operator-set Anthropic endpoint env var into a `URL`, falling
 * back to anthropic-direct when the value is malformed (#675). The
 * credential proxy needs concrete `URL` objects for its upstream and
 * bypass targets; a typo'd `ANTHROPIC_BASE_URL` / `ANTHROPIC_BYPASS_URL`
 * must not throw at startup — it degrades to the safe default so the
 * orchestrator stays up, consistent with `resolveBypassTarget`'s no-throw
 * handling rather than a raw `new URL` crash. `fellBackToDefault` is true
 * only when a non-empty value was provided AND unparseable (so the caller
 * can warn on a misconfig but stay silent on the normal unset case).
 */
export function parseAnthropicUrlOrDefault(value: string | undefined): {
  url: URL;
  fellBackToDefault: boolean;
} {
  const candidate = value || DEFAULT_ANTHROPIC_URL;
  if (URL.canParse(candidate)) {
    return { url: new URL(candidate), fellBackToDefault: false };
  }
  return { url: new URL(DEFAULT_ANTHROPIC_URL), fellBackToDefault: true };
}

/**
 * Is a failure from the primary Anthropic attempt eligible for a direct
 * bypass retry? Eligible: a connection failure (unreachable / connect
 * timeout) or a 5xx upstream status. NOT eligible: a deliberate caller
 * abort (`APIUserAbortError` — the classifier's own request-timeout),
 * 4xx, or any non-network error (a programmer defect must surface per
 * `coding-policy: error-handling`, not trigger a second wasted call).
 */
export function isBypassEligibleFailure(err: unknown): boolean {
  if (err instanceof Anthropic.APIUserAbortError) return false;
  if (err instanceof Anthropic.APIConnectionError) return true;
  if (err instanceof Anthropic.APIError) {
    return BYPASS_TRIGGER_STATUS_CODES.has(err.status ?? 0);
  }
  // A raw errno error the SDK didn't wrap.
  if (err instanceof Error) {
    return isReachabilityErrorCode((err as NodeJS.ErrnoException).code);
  }
  return false;
}

export interface AnthropicClientPair {
  /** Client pointed at `ANTHROPIC_BASE_URL` (the LiteLLM gateway). */
  primary: Anthropic;
  /**
   * Client pointed at `ANTHROPIC_BYPASS_URL` (anthropic-direct), or
   * `null` when bypass is disabled for this config (same-origin / no key).
   */
  bypass: Anthropic | null;
}

/**
 * Issue a `messages.create` against the primary client and, on a
 * bypass-eligible failure, retry once against the bypass client with the
 * same params and abort signal. The SDK analogue of the credential
 * proxy's socket-level bypass. Falls through to the primary error (no
 * retry) when bypass is disabled, the signal already aborted (the
 * caller's own timeout fired — retrying would race a dead signal), or the
 * failure isn't bypass-eligible. The bypass attempt's own error
 * propagates unwrapped so the caller's failure handling sees it.
 *
 * `onBypass` is invoked just before the retry so the caller can emit a
 * context-rich log line; the helper itself stays logging-agnostic.
 */
export async function createMessageWithBypass(
  clients: AnthropicClientPair,
  params: Anthropic.MessageCreateParamsNonStreaming,
  options: { signal: AbortSignal },
  onBypass?: (err: unknown) => void,
): Promise<Anthropic.Message> {
  try {
    return await clients.primary.messages.create(params, options);
  } catch (err) {
    if (
      clients.bypass &&
      !options.signal.aborted &&
      isBypassEligibleFailure(err)
    ) {
      onBypass?.(err);
      return await clients.bypass.messages.create(params, options);
    }
    throw err;
  }
}
