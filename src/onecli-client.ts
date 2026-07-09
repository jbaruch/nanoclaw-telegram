/**
 * OneCLI client wrapper — groundwork for #564.
 *
 * Stage 1 (this file): provide tier-scoped `ensureAgent` + `applyContainerConfig`
 * surface. No behavior unless both `ONECLI_URL` and `ONECLI_API_KEY` are set
 * in the orchestrator environment. When unset (the default until the
 * operational sub-issue lands), every export is a graceful no-op.
 *
 * Stage 1's error policy is permissive: SDK errors are logged and treated as
 * "OneCLI not active for this spawn" so we never block a container start.
 * The Anthropic/OpenAI swap sub-issue (#637) tightens this to hard-fail once
 * OneCLI is the only credential path.
 */
import { readFileSync } from 'fs';

import { OneCLI, OneCLIError, OneCLIRequestError } from '@onecli-sh/sdk';
import { readEnvFile } from './env.js';
import { logger } from './logger.js';
import { TRUST_TIERS, type TrustTier } from './trust-tier.js';

export { TRUST_TIERS, type TrustTier };

const AGENT_IDENTIFIER_PREFIX = 'nanoclaw';
/**
 * Tight per-call timeout so a configured-but-unreachable OneCLI gateway
 * degrades fast: 3 tiers × this ms cap at startup, plus this ms cap per
 * container spawn. The SDK falls through to the existing credential-proxy
 * path on timeout (Stage 1 stays additive).
 */
const DEFAULT_TIMEOUT_MS = 1500;

let cachedClient: OneCLI | null = null;

/**
 * Per-tier cache of the minted outbound-proxy config. The credential-proxy
 * calls `getOneCliOutboundConfig` for every Authorization-carrying request it
 * forwards. In this deployment that includes the `/v1/messages` traffic, not
 * just the OAuth exchange: the Claude Code SDK sends a placeholder Bearer that
 * OneCLI swaps (confirmed live — the cutover's self-signed errors were on
 * `/v1/messages`). Minting fresh each time would add a gateway round-trip +
 * CA-file write per message, so cache `{proxyUrl, ca}` for a short TTL: the
 * OneCLI CA is stable and the agent-scoped proxy token is reused across the
 * window; on expiry the next call re-mints.
 */
const OUTBOUND_CACHE_TTL_MS = 60_000;
const outboundConfigCache = new Map<
  TrustTier,
  { proxyUrl: string; ca: string; expiresAt: number }
>();

interface OneCliEnvOptions {
  url: string;
  apiKey: string;
}

/**
 * Read OneCLI config from `.env` via `readEnvFile`, NOT from
 * `process.env`. The orchestrator's `docker-compose.yml` deliberately
 * forwards only an explicit allowlist of env vars to the orchestrator
 * container (per `src/env.ts`: "Does NOT load anything into process.env
 * — this keeps secrets out of the process environment so they don't
 * leak to child processes"). Reading from `.env` directly matches
 * every other credential-read path in the orchestrator.
 */
function readEnvOptions(): OneCliEnvOptions | null {
  const env = readEnvFile(['ONECLI_URL', 'ONECLI_API_KEY']);
  const url = env.ONECLI_URL;
  const apiKey = env.ONECLI_API_KEY;
  if (!url || !apiKey) return null;
  return { url, apiKey };
}

export function isOneCliConfigured(): boolean {
  return readEnvOptions() !== null;
}

/**
 * Gate for the agent-spawn proxy injection (`applyOneCliToSpawn`), SEPARATE
 * from `isOneCliConfigured`. #637 makes the credential-proxy route Anthropic
 * through OneCLI while agents themselves stay proxy-less — putting OneCLI in
 * front of agent traffic is #640 (external-cred swap), which also needs the
 * non-vault-passthrough validation. Without this split, re-enabling
 * `ONECLI_URL` for #637 would re-apply the agent proxy that broke the LLM
 * path (the agent's call to its local credential-proxy got routed through the
 * OneCLI gateway → ECONNRESET). Default OFF; #640 flips it on.
 */
export function oneCliAgentProxyEnabled(): boolean {
  const env = readEnvFile(['ONECLI_AGENT_PROXY']);
  return env.ONECLI_AGENT_PROXY === '1';
}

/**
 * Mint a OneCLI outbound-proxy config for the orchestrator's OWN outbound
 * HTTPS. Used by the credential-proxy to tunnel the Anthropic OAuth-exchange
 * request through the OneCLI gateway so OneCLI injects the vaulted Bearer
 * (#637) — the token then lives only in the vault, not in `.env`.
 *
 * Returns the proxy URL (with the agent-scoped gateway token) and the
 * combined CA-bundle contents to trust the MITM cert, or `null` when OneCLI
 * is unconfigured / unreachable (caller falls back to `.env` injection during
 * the transition; post-cutover a null here is a hard auth failure — correct).
 */
export async function getOneCliOutboundConfig(
  tier: TrustTier,
): Promise<{ proxyUrl: string; ca: string } | null> {
  const client = getClient();
  if (!client) return null;
  const cached = outboundConfigCache.get(tier);
  if (cached && cached.expiresAt > Date.now()) {
    return { proxyUrl: cached.proxyUrl, ca: cached.ca };
  }
  try {
    const probe: string[] = [];
    const active = await client.applyContainerConfig(probe, {
      agent: agentIdentifierForTier(tier),
      combineCaBundle: true,
      addHostMapping: false,
    });
    if (!active) return null;
    let proxyUrl: string | undefined;
    let combinedCaPath: string | undefined;
    let gatewayCaPath: string | undefined;
    for (let i = 0; i < probe.length; i++) {
      const v = probe[i + 1] ?? '';
      if (probe[i] === '-e' && v.startsWith('HTTPS_PROXY=')) {
        proxyUrl = v.slice('HTTPS_PROXY='.length);
      } else if (probe[i] === '-v' && /onecli-combined-ca\.pem:ro$/.test(v)) {
        combinedCaPath = v.split(':')[0];
      } else if (probe[i] === '-v' && /onecli-gateway-ca\.pem:ro$/.test(v)) {
        gatewayCaPath = v.split(':')[0];
      }
    }
    // Prefer the combined bundle (system CAs + OneCLI CA); fall back to the
    // gateway CA alone when combineCaBundle didn't emit a combined mount (its
    // system-CA read can fail). Both contain OneCLI's MITM signing cert.
    const caPath = combinedCaPath ?? gatewayCaPath;
    if (!proxyUrl || !caPath) return null;
    const ca = readFileSync(caPath, 'utf8');
    outboundConfigCache.set(tier, {
      proxyUrl,
      ca,
      expiresAt: Date.now() + OUTBOUND_CACHE_TTL_MS,
    });
    return { proxyUrl, ca };
  } catch (err) {
    if (err instanceof OneCLIRequestError) {
      logger.warn(
        { tier, statusCode: err.statusCode },
        'OneCLI getOneCliOutboundConfig rejected by gateway — Anthropic exchange falls back to .env token injection if present',
      );
      return null;
    }
    if (err instanceof OneCLIError) {
      logger.warn(
        { tier },
        'OneCLI getOneCliOutboundConfig failed before reaching the gateway — Anthropic exchange falls back to .env token injection if present',
      );
      return null;
    }
    throw err;
  }
}

function getClient(): OneCLI | null {
  if (cachedClient) return cachedClient;
  const opts = readEnvOptions();
  if (!opts) return null;
  cachedClient = new OneCLI({
    url: opts.url,
    apiKey: opts.apiKey,
    timeout: DEFAULT_TIMEOUT_MS,
  });
  return cachedClient;
}

// Exposed for tests; not part of the public stable surface.
export function _resetOneCliClient(): void {
  cachedClient = null;
  outboundConfigCache.clear();
}

function agentIdentifierForTier(tier: TrustTier): string {
  return `${AGENT_IDENTIFIER_PREFIX}-${tier}`;
}

function agentNameForTier(tier: TrustTier): string {
  return `NanoClaw ${tier} tier`;
}

/**
 * Idempotently register the tier-scoped OneCLI agent. Safe to call multiple
 * times. A no-op when OneCLI is unconfigured.
 */
export async function ensureAgentForTier(tier: TrustTier): Promise<void> {
  const client = getClient();
  if (!client) return;
  try {
    const result = await client.ensureAgent({
      name: agentNameForTier(tier),
      identifier: agentIdentifierForTier(tier),
    });
    if (result.created) {
      logger.info(
        { tier, identifier: result.identifier },
        'OneCLI agent created',
      );
    } else {
      logger.debug(
        { tier, identifier: result.identifier },
        'OneCLI agent already present',
      );
    }
  } catch (err) {
    if (err instanceof OneCLIRequestError) {
      // Log statusCode only — the SDK's err.url echoes the gateway path
      // and err.message can carry SDK request details, both of which
      // could surface secrets in future SDK revisions (no-secrets rule).
      logger.warn(
        { tier, statusCode: err.statusCode },
        'OneCLI ensureAgent rejected by gateway — confirm ONECLI_URL points at a running gateway, ONECLI_API_KEY matches its project, and `onecli secrets list` resolves on the NAS host; until then, container spawns fall back to the existing credential-proxy path',
      );
      return;
    }
    if (err instanceof OneCLIError) {
      // Drop err.message — SDK precondition errors echo config values
      // (no-secrets rule). The remediation is the same set of operator
      // steps as the request-error branch.
      logger.warn(
        { tier },
        'OneCLI ensureAgent failed before reaching the gateway (SDK precondition error) — confirm both ONECLI_URL and ONECLI_API_KEY are set in `~/nanoclaw/.env` on the NAS and that the gateway process is running; until then, container spawns fall back to the existing credential-proxy path',
      );
      return;
    }
    throw err;
  }
}

/**
 * When OneCLI is configured, mutate the docker-spawn argv to add HTTPS_PROXY
 * env, mount the OneCLI CA bundle, and add the host.docker.internal mapping.
 * Returns whether OneCLI was applied to the spawn.
 *
 * A no-op (returns false) when OneCLI is unconfigured or the gateway is
 * unreachable. Stage 1 stays additive: a spawn that can't reach OneCLI still
 * runs with the existing credential-proxy path.
 */
export async function applyOneCliToSpawn(
  args: string[],
  tier: TrustTier,
): Promise<boolean> {
  const client = getClient();
  if (!client) return false;
  try {
    const active = await client.applyContainerConfig(args, {
      agent: agentIdentifierForTier(tier),
      combineCaBundle: true,
      // #746: the spawn argv already carries `--add-host=host.docker.internal:
      // host-gateway` from hostGatewayArgs() on Linux, so let the SDK skip its
      // duplicate mapping.
      addHostMapping: false,
    });
    if (active) {
      // Info (not debug): a debug-only success line is why the argv-order bug
      // (#746) went unnoticed for weeks while this returned true. Surface
      // whether the proxy env actually landed on the spawn argv.
      logger.info(
        {
          tier,
          httpsProxyApplied: args.some((a) => a.startsWith('HTTPS_PROXY=')),
        },
        'OneCLI gateway config applied to container',
      );
    } else {
      // The SDK catches gateway/config fetch failures and resolves `false`
      // (no throw) — this is the main recoverable failure path that
      // bypasses both catch branches below. Surfacing it as a WARN keeps
      // the silent-fallback case actionable per `jbaruch/coding-policy:
      // error-handling`.
      logger.warn(
        { tier },
        'OneCLI is configured but `applyContainerConfig` returned false (SDK could not reach the gateway, or the gateway returned an empty config) — confirm `curl -sf $ONECLI_URL/health` resolves on the NAS host and the tier-scoped agent exists (`onecli agents list`); the spawn proceeds via the existing credential-proxy path so no message is dropped',
      );
    }
    return active;
  } catch (err) {
    if (err instanceof OneCLIRequestError) {
      // Log statusCode only — same redaction rationale as ensureAgent.
      logger.warn(
        { tier, statusCode: err.statusCode },
        'OneCLI applyContainerConfig rejected by gateway — confirm ONECLI_URL points at a running gateway and the tier-scoped agent exists (`onecli agents list`); the spawn proceeds via the existing credential-proxy path so no message is dropped',
      );
      return false;
    }
    if (err instanceof OneCLIError) {
      logger.warn(
        { tier },
        'OneCLI applyContainerConfig failed before reaching the gateway (SDK precondition error) — confirm both ONECLI_URL and ONECLI_API_KEY are set on the NAS and the gateway process is running; the spawn proceeds via the existing credential-proxy path so no message is dropped',
      );
      return false;
    }
    throw err;
  }
}
