import type { ContainerContext } from './usage-log.js';
import { isMessagesEndpoint } from './wire-tool-filter.js';

export interface PromptCacheTtlStats {
  ttlApplied: number;
}

export interface PromptCacheTtlResult {
  body: Buffer;
  stats: PromptCacheTtlStats;
  applied: boolean;
}

const DEFAULT_1H_GROUPS = ['telegram_swarm'];

function parseGroupList(raw: string | undefined): Set<string> {
  const values = (raw ?? DEFAULT_1H_GROUPS.join(','))
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  return new Set(values);
}

function isDisabled(raw: string | undefined): boolean {
  if (raw === undefined) return false;
  return new Set(['', '0', 'false', 'no', 'off']).has(raw.trim().toLowerCase());
}

function shouldApplyPromptCache1h(
  ctx: ContainerContext | null,
  env: NodeJS.ProcessEnv,
): boolean {
  if (!ctx) return false;
  if (isDisabled(env.NANOCLAW_PROMPT_CACHE_1H)) return false;
  if (ctx.tier !== 'main') return false;
  if (ctx.session !== 'default') return false;
  if (ctx.task_id) return false;

  const groups = parseGroupList(env.NANOCLAW_PROMPT_CACHE_1H_GROUPS);
  return groups.has(ctx.group) || groups.has('*');
}

function mutateCacheControlInPlace(value: unknown): number {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return 0;

  const obj = value as Record<string, unknown>;
  const cc = obj.cache_control;
  if (!cc || typeof cc !== 'object' || Array.isArray(cc)) return 0;

  const cacheControl = cc as Record<string, unknown>;
  if (cacheControl.type !== 'ephemeral') return 0;

  cacheControl.ttl = '1h';
  return 1;
}

function mutateContentBlocksInPlace(value: unknown): number {
  if (!Array.isArray(value)) return 0;
  let count = 0;
  for (const block of value) count += mutateCacheControlInPlace(block);
  return count;
}

function mutateAnthropicCacheTtlsInPlace(value: unknown): number {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return 0;

  const body = value as Record<string, unknown>;
  let count = 0;

  // Keep the rewrite limited to documented Anthropic request locations
  // rather than any nested object that happens to use a `cache_control`
  // key: system content blocks, tool definitions, and message content
  // blocks. The SDK still owns where those breakpoints are emitted.
  count += mutateContentBlocksInPlace(body.system);
  count += mutateContentBlocksInPlace(body.tools);

  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!message || typeof message !== 'object' || Array.isArray(message)) {
        continue;
      }
      const content = (message as Record<string, unknown>).content;
      count += mutateContentBlocksInPlace(content);
    }
  }

  return count;
}

/**
 * Extend existing Anthropic prompt-cache breakpoints to the 1h tier for
 * selected main-DM containers (#537).
 *
 * The Claude Agent SDK owns where `cache_control` breakpoints land. This
 * proxy-side pass deliberately does NOT invent new breakpoints; it only
 * upgrades already-emitted `{type:'ephemeral'}` controls to
 * `{type:'ephemeral', ttl:'1h'}` for the configured group(s). That keeps
 * prefix boundaries byte-for-byte aligned with the SDK while letting the
 * host choose a longer TTL for idle-prone DMs such as `telegram_swarm`.
 */
export function applyPromptCacheTtl(
  url: string,
  method: string | undefined,
  body: Buffer,
  ctx: ContainerContext | null,
  env: NodeJS.ProcessEnv,
  onParseError?: (err: unknown) => void,
): PromptCacheTtlResult {
  const zero: PromptCacheTtlStats = { ttlApplied: 0 };
  if (method !== 'POST') return { body, stats: zero, applied: false };
  // Use the canonical helper from `wire-tool-filter` so the proxy's
  // /v1/messages routing decision stays in sync between the two
  // interceptors. The helper splits off any query string first
  // (`url.split('?')[0]`) then does strict path equality against
  // `/v1/messages` — caller passes the post-proxy `upstreamPath`,
  // where the `/c/<token>/` prefix has already been stripped, so the
  // path-segment match is what we want and `?beta=…` etc. are
  // tolerated.
  if (!isMessagesEndpoint(url)) {
    return { body, stats: zero, applied: false };
  }
  if (!shouldApplyPromptCache1h(ctx, env)) {
    return { body, stats: zero, applied: false };
  }
  if (body.length === 0) return { body, stats: zero, applied: false };

  let parsed: unknown;
  try {
    parsed = JSON.parse(body.toString('utf8'));
  } catch (err) {
    if (err instanceof SyntaxError) {
      onParseError?.(err);
      return { body, stats: zero, applied: false };
    }
    throw err;
  }

  const ttlApplied = mutateAnthropicCacheTtlsInPlace(parsed);
  if (ttlApplied === 0) return { body, stats: zero, applied: false };

  return {
    body: Buffer.from(JSON.stringify(parsed), 'utf8'),
    stats: { ttlApplied },
    applied: true,
  };
}
