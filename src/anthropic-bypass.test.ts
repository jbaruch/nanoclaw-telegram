import { describe, it, expect, vi } from 'vitest';
import Anthropic from '@anthropic-ai/sdk';

import {
  BYPASS_TRIGGER_STATUS_CODES,
  isReachabilityErrorCode,
  resolveBypassTarget,
  parseAnthropicUrlOrDefault,
  isBypassEligibleFailure,
  createMessageWithBypass,
  type AnthropicClientPair,
} from './anthropic-bypass.js';

const LITELLM = 'http://nanoclaw-litellm:4000';
const DIRECT = 'https://api.anthropic.com';

const PARAMS = {
  model: 'claude-haiku-4-5-20251001',
  max_tokens: 16,
  messages: [{ role: 'user', content: 'hi' }],
} as Anthropic.MessageCreateParamsNonStreaming;

const RESPONSE = { id: 'msg_primary' } as unknown as Anthropic.Message;
const BYPASS_RESPONSE = { id: 'msg_bypass' } as unknown as Anthropic.Message;

// Inline fixture (not imported from the module) so the test pins the
// expected reachability set independently of the source of truth.
const BYPASS_REACHABILITY_CODES_FIXTURE = [
  'ECONNREFUSED',
  'ENOTFOUND',
  'EHOSTUNREACH',
  'ECONNRESET',
] as const;

function mockClient(
  create: ReturnType<typeof vi.fn>,
): Anthropic & { messages: { create: ReturnType<typeof vi.fn> } } {
  return { messages: { create } } as unknown as Anthropic & {
    messages: { create: ReturnType<typeof vi.fn> };
  };
}

describe('resolveBypassTarget', () => {
  it('enabled when primary (LiteLLM) and bypass origins differ and a key is present', () => {
    const r = resolveBypassTarget({
      baseUrl: LITELLM,
      bypassUrl: DIRECT,
      hasApiKey: true,
    });
    expect(r.enabled).toBe(true);
    expect(r.bypassUrl).toBe(DIRECT);
  });

  it('disabled when primary and bypass resolve to the same origin (nothing to fall back to)', () => {
    const r = resolveBypassTarget({
      baseUrl: DIRECT,
      bypassUrl: DIRECT,
      hasApiKey: true,
    });
    expect(r.enabled).toBe(false);
  });

  it('disabled without an API key even when origins differ (cannot auth the direct call)', () => {
    const r = resolveBypassTarget({
      baseUrl: LITELLM,
      bypassUrl: DIRECT,
      hasApiKey: false,
    });
    expect(r.enabled).toBe(false);
  });

  it('defaults both URLs to anthropic-direct → disabled (same origin) and reports the default bypass URL', () => {
    const r = resolveBypassTarget({ hasApiKey: true });
    expect(r.enabled).toBe(false);
    expect(r.bypassUrl).toBe(DIRECT);
  });

  it('enabled when only the primary is the gateway and bypass defaults to direct', () => {
    const r = resolveBypassTarget({ baseUrl: LITELLM, hasApiKey: true });
    expect(r.enabled).toBe(true);
    expect(r.bypassUrl).toBe(DIRECT);
  });

  it('disabled (no throw) when the primary URL is unparseable and bypass is direct — both resolve to direct, same origin', () => {
    const r = resolveBypassTarget({
      baseUrl: 'not a url',
      bypassUrl: DIRECT,
      hasApiKey: true,
    });
    expect(r.enabled).toBe(false);
  });

  it('ENABLED to anthropic-direct (no throw) when the bypass URL is unparseable — graceful fallback actually taken', () => {
    // A malformed ANTHROPIC_BYPASS_URL resolves to anthropic-direct, so
    // with a real gateway primary the bypass is enabled and points at
    // direct — it must not warn "falling back to direct" yet silently
    // disable the path (the #678 review finding).
    const r = resolveBypassTarget({
      baseUrl: LITELLM,
      bypassUrl: ':::not-a-url:::',
      hasApiKey: true,
    });
    expect(r.enabled).toBe(true);
    expect(r.bypassUrl).toBe(DIRECT);
  });
});

describe('parseAnthropicUrlOrDefault', () => {
  it('returns a valid provided URL unchanged, no fallback', () => {
    const r = parseAnthropicUrlOrDefault(LITELLM);
    expect(r.url.origin).toBe(new URL(LITELLM).origin);
    expect(r.fellBackToDefault).toBe(false);
  });

  it('returns anthropic-direct for an unset value, NOT flagged as a fallback (normal default)', () => {
    const r = parseAnthropicUrlOrDefault(undefined);
    expect(r.url.origin).toBe(new URL(DIRECT).origin);
    expect(r.fellBackToDefault).toBe(false);
  });

  it('falls back to anthropic-direct AND flags it when a provided value is malformed', () => {
    const r = parseAnthropicUrlOrDefault('not a url');
    expect(r.url.origin).toBe(new URL(DIRECT).origin);
    expect(r.fellBackToDefault).toBe(true);
  });
});

describe('isReachabilityErrorCode', () => {
  it.each([...BYPASS_REACHABILITY_CODES_FIXTURE])(
    'returns true for %s',
    (code) => {
      expect(isReachabilityErrorCode(code)).toBe(true);
    },
  );

  it('returns false for unrelated codes and non-strings', () => {
    expect(isReachabilityErrorCode('EPIPE')).toBe(false);
    expect(isReachabilityErrorCode(undefined)).toBe(false);
    expect(isReachabilityErrorCode(500)).toBe(false);
  });
});

describe('isBypassEligibleFailure', () => {
  it('a deliberate caller abort (APIUserAbortError) is NOT eligible — that is our own timeout', () => {
    expect(isBypassEligibleFailure(new Anthropic.APIUserAbortError())).toBe(
      false,
    );
  });

  it('a connection failure (APIConnectionError) IS eligible — gateway unreachable', () => {
    expect(
      isBypassEligibleFailure(
        new Anthropic.APIConnectionError({ message: 'Connection error.' }),
      ),
    ).toBe(true);
  });

  it.each([...BYPASS_TRIGGER_STATUS_CODES])(
    'a %d upstream status IS eligible',
    (status) => {
      const err = new Anthropic.APIError(
        status,
        undefined,
        'upstream error',
        new Headers(),
      );
      expect(isBypassEligibleFailure(err)).toBe(true);
    },
  );

  it('a 4xx status is NOT eligible (a rate-limit / bad-request would just fail again)', () => {
    const err = new Anthropic.APIError(
      429,
      { type: 'rate_limit_error', message: 'slow down' },
      'rate limited',
      new Headers(),
    );
    expect(isBypassEligibleFailure(err)).toBe(false);
  });

  it('a raw errno reachability error the SDK did not wrap IS eligible', () => {
    const err = Object.assign(new Error('connect ECONNREFUSED'), {
      code: 'ECONNREFUSED',
    });
    expect(isBypassEligibleFailure(err)).toBe(true);
  });

  it('a plain Error / programmer defect is NOT eligible', () => {
    expect(isBypassEligibleFailure(new TypeError('boom'))).toBe(false);
    expect(isBypassEligibleFailure(new Error('???'))).toBe(false);
    expect(isBypassEligibleFailure('not even an error')).toBe(false);
  });
});

describe('createMessageWithBypass', () => {
  const signal = new AbortController().signal;

  it('returns the primary result without touching bypass when the primary succeeds', async () => {
    const primaryCreate = vi.fn().mockResolvedValue(RESPONSE);
    const bypassCreate = vi.fn().mockResolvedValue(BYPASS_RESPONSE);
    const clients: AnthropicClientPair = {
      primary: mockClient(primaryCreate),
      bypass: mockClient(bypassCreate),
    };
    const out = await createMessageWithBypass(clients, PARAMS, { signal });
    expect(out).toBe(RESPONSE);
    expect(bypassCreate).not.toHaveBeenCalled();
  });

  it('retries against bypass on a 5xx and returns the bypass result, invoking onBypass', async () => {
    const fiveOhTwo = new Anthropic.APIError(
      502,
      undefined,
      'bad gateway',
      new Headers(),
    );
    const primaryCreate = vi.fn().mockRejectedValue(fiveOhTwo);
    const bypassCreate = vi.fn().mockResolvedValue(BYPASS_RESPONSE);
    const onBypass = vi.fn();
    const clients: AnthropicClientPair = {
      primary: mockClient(primaryCreate),
      bypass: mockClient(bypassCreate),
    };
    const out = await createMessageWithBypass(
      clients,
      PARAMS,
      { signal },
      onBypass,
    );
    expect(out).toBe(BYPASS_RESPONSE);
    expect(bypassCreate).toHaveBeenCalledTimes(1);
    expect(bypassCreate).toHaveBeenCalledWith(PARAMS, { signal });
    expect(onBypass).toHaveBeenCalledWith(fiveOhTwo);
  });

  it('does NOT retry when bypass is disabled (null) — the primary error propagates', async () => {
    const connErr = new Anthropic.APIConnectionError({
      message: 'Connection error.',
    });
    const primaryCreate = vi.fn().mockRejectedValue(connErr);
    const clients: AnthropicClientPair = {
      primary: mockClient(primaryCreate),
      bypass: null,
    };
    await expect(
      createMessageWithBypass(clients, PARAMS, { signal }),
    ).rejects.toBe(connErr);
  });

  it('does NOT retry a non-eligible failure (4xx) — the primary error propagates', async () => {
    const badRequest = new Anthropic.APIError(
      400,
      undefined,
      'bad request',
      new Headers(),
    );
    const primaryCreate = vi.fn().mockRejectedValue(badRequest);
    const bypassCreate = vi.fn().mockResolvedValue(BYPASS_RESPONSE);
    const clients: AnthropicClientPair = {
      primary: mockClient(primaryCreate),
      bypass: mockClient(bypassCreate),
    };
    await expect(
      createMessageWithBypass(clients, PARAMS, { signal }),
    ).rejects.toBe(badRequest);
    expect(bypassCreate).not.toHaveBeenCalled();
  });

  it('does NOT retry when the signal already aborted (the caller timeout fired)', async () => {
    const aborted = new AbortController();
    aborted.abort();
    const connErr = new Anthropic.APIConnectionError({
      message: 'Connection error.',
    });
    const primaryCreate = vi.fn().mockRejectedValue(connErr);
    const bypassCreate = vi.fn().mockResolvedValue(BYPASS_RESPONSE);
    const clients: AnthropicClientPair = {
      primary: mockClient(primaryCreate),
      bypass: mockClient(bypassCreate),
    };
    await expect(
      createMessageWithBypass(clients, PARAMS, { signal: aborted.signal }),
    ).rejects.toBe(connErr);
    expect(bypassCreate).not.toHaveBeenCalled();
  });

  it('propagates the bypass error when both attempts fail', async () => {
    const primaryErr = new Anthropic.APIConnectionError({
      message: 'primary down',
    });
    const bypassErr = new Anthropic.APIConnectionError({
      message: 'direct down too',
    });
    const primaryCreate = vi.fn().mockRejectedValue(primaryErr);
    const bypassCreate = vi.fn().mockRejectedValue(bypassErr);
    const clients: AnthropicClientPair = {
      primary: mockClient(primaryCreate),
      bypass: mockClient(bypassCreate),
    };
    await expect(
      createMessageWithBypass(clients, PARAMS, { signal }),
    ).rejects.toBe(bypassErr);
    expect(bypassCreate).toHaveBeenCalledTimes(1);
  });
});
