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
import { OneCLI, OneCLIError, OneCLIRequestError } from '@onecli-sh/sdk';
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

interface OneCliEnvOptions {
  url: string;
  apiKey: string;
}

function readEnvOptions(): OneCliEnvOptions | null {
  const url = process.env.ONECLI_URL;
  const apiKey = process.env.ONECLI_API_KEY;
  if (!url || !apiKey) return null;
  return { url, apiKey };
}

export function isOneCliConfigured(): boolean {
  return readEnvOptions() !== null;
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
      addHostMapping: true,
    });
    if (active) {
      logger.debug({ tier }, 'OneCLI gateway config applied to container');
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
