import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

const {
  ensureAgentMock,
  applyContainerConfigMock,
  readFileSyncMock,
  envFileMock,
  FakeOneCLIError,
  FakeOneCLIRequestError,
} = vi.hoisted(() => {
  class FakeOneCLIError extends Error {
    constructor(message: string) {
      super(message);
      this.name = 'OneCLIError';
    }
  }

  class FakeOneCLIRequestError extends Error {
    readonly url: string;
    readonly statusCode: number;
    constructor(
      message: string,
      requestData: { url: string; statusCode: number },
    ) {
      super(message);
      this.name = 'OneCLIRequestError';
      this.url = requestData.url;
      this.statusCode = requestData.statusCode;
    }
  }

  return {
    ensureAgentMock: vi.fn(),
    applyContainerConfigMock: vi.fn(),
    readFileSyncMock: vi.fn(() => 'ca-file-contents'),
    // envFileMock must be hoisted because the `./env.js` mock factory
    // below closes over it. Without `vi.hoisted`, the mock factory
    // (which Vitest hoists above top-level `const` declarations) could
    // capture an uninitialized binding and throw a TDZ ReferenceError
    // on the first `readEnvFile()` call from imported code.
    envFileMock: {} as Record<string, string>,
    FakeOneCLIError,
    FakeOneCLIRequestError,
  };
});

vi.mock('@onecli-sh/sdk', () => {
  class FakeOneCLI {
    ensureAgent = ensureAgentMock;
    applyContainerConfig = applyContainerConfigMock;
  }
  return {
    OneCLI: FakeOneCLI,
    OneCLIError: FakeOneCLIError,
    OneCLIRequestError: FakeOneCLIRequestError,
  };
});

vi.mock('./env.js', () => ({
  readEnvFile: vi.fn((keys: string[]) =>
    Object.fromEntries(
      keys.filter((k) => k in envFileMock).map((k) => [k, envFileMock[k]]),
    ),
  ),
}));

vi.mock('fs', () => ({ readFileSync: readFileSyncMock }));

vi.mock('./logger.js', () => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
  },
}));

import {
  applyOneCliToSpawn,
  ensureAgentForTier,
  getOneCliOutboundConfig,
  isOneCliConfigured,
  oneCliSpawnEnvArgs,
  TRUST_TIERS,
  _resetOneCliClient,
} from './onecli-client.js';

describe('onecli-client', () => {
  beforeEach(() => {
    for (const k of Object.keys(envFileMock)) delete envFileMock[k];
    _resetOneCliClient();
    ensureAgentMock.mockReset();
    applyContainerConfigMock.mockReset();
    readFileSyncMock.mockReset();
    readFileSyncMock.mockReturnValue('ca-file-contents');
  });

  describe('getOneCliOutboundConfig', () => {
    const configure = () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
    };

    // Freeze the clock so the TTL-cache behaviour is deterministic (the cache
    // reads Date.now()); advance it explicitly to exercise expiry.
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(0);
    });
    afterEach(() => {
      vi.useRealTimers();
    });

    it('returns null when OneCLI is unconfigured (no gateway call)', async () => {
      expect(await getOneCliOutboundConfig('main')).toBeNull();
      expect(applyContainerConfigMock).not.toHaveBeenCalled();
    });

    it('extracts the proxy URL + combined CA contents on the active path', async () => {
      configure();
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://x:aoc_tok@gw:10255');
        args.push(
          '-v',
          '/host/onecli-combined-ca.pem:/tmp/onecli-combined-ca.pem:ro',
        );
        return Promise.resolve(true);
      });

      const cfg = await getOneCliOutboundConfig('trusted');
      expect(cfg).toEqual({
        proxyUrl: 'http://x:aoc_tok@gw:10255',
        ca: 'ca-file-contents',
      });
      expect(readFileSyncMock).toHaveBeenCalledWith(
        '/host/onecli-combined-ca.pem',
        'utf8',
      );
    });

    it('falls back to the gateway CA when no combined bundle is mounted', async () => {
      configure();
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://x:aoc_tok@gw:10255');
        args.push(
          '-v',
          '/host/onecli-proxy-ca.pem:/tmp/onecli-gateway-ca.pem:ro',
        );
        return Promise.resolve(true);
      });

      const cfg = await getOneCliOutboundConfig('main');
      expect(cfg?.proxyUrl).toBe('http://x:aoc_tok@gw:10255');
      expect(readFileSyncMock).toHaveBeenCalledWith(
        '/host/onecli-proxy-ca.pem',
        'utf8',
      );
    });

    it('returns null when applyContainerConfig reports inactive', async () => {
      configure();
      applyContainerConfigMock.mockResolvedValue(false);
      const p = getOneCliOutboundConfig('main');
      await vi.runAllTimersAsync(); // flush the #787 retry backoffs
      expect(await p).toBeNull();
      // A persistent false exhausts the bounded retry (#787).
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(3);
    });

    it('returns null when the argv has no CA mount (nothing to read)', async () => {
      configure();
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://x:aoc_tok@gw:10255');
        return Promise.resolve(true);
      });
      expect(await getOneCliOutboundConfig('main')).toBeNull();
      expect(readFileSyncMock).not.toHaveBeenCalled();
    });

    it('returns null (not throw) when the gateway rejects', async () => {
      configure();
      applyContainerConfigMock.mockRejectedValue(
        new FakeOneCLIRequestError('rejected', {
          url: 'http://gw',
          statusCode: 502,
        }),
      );
      const p = getOneCliOutboundConfig('main');
      await vi.runAllTimersAsync(); // flush the #787 retry backoffs
      expect(await p).toBeNull();
      // 502 is transient — retried to exhaustion before returning null (#787).
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(3);
    });

    it('retries a transient blip and recovers within the same call (#787)', async () => {
      configure();
      applyContainerConfigMock
        .mockRejectedValueOnce(new FakeOneCLIError('connreset'))
        .mockImplementationOnce((args: string[]) => {
          args.push('-e', 'HTTPS_PROXY=http://x:aoc_tok@gw:10255');
          args.push(
            '-v',
            '/host/onecli-combined-ca.pem:/tmp/onecli-combined-ca.pem:ro',
          );
          return Promise.resolve(true);
        });

      const p = getOneCliOutboundConfig('main');
      await vi.runAllTimersAsync();
      const cfg = await p;
      expect(cfg).toEqual({
        proxyUrl: 'http://x:aoc_tok@gw:10255',
        ca: 'ca-file-contents',
      });
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(2);
    });

    it('caches the config — a second call within TTL does not re-mint', async () => {
      configure();
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://x:aoc_tok@gw:10255');
        args.push(
          '-v',
          '/host/onecli-combined-ca.pem:/tmp/onecli-combined-ca.pem:ro',
        );
        return Promise.resolve(true);
      });

      const a = await getOneCliOutboundConfig('main');
      const b = await getOneCliOutboundConfig('main');
      expect(a).toEqual(b);
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(1);
    });

    it('caches per-tier (distinct tiers each mint once)', async () => {
      configure();
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://x:aoc_tok@gw:10255');
        args.push(
          '-v',
          '/host/onecli-combined-ca.pem:/tmp/onecli-combined-ca.pem:ro',
        );
        return Promise.resolve(true);
      });

      await getOneCliOutboundConfig('main');
      await getOneCliOutboundConfig('trusted');
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(2);
    });

    it('re-mints after the TTL expires', async () => {
      configure();
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://x:aoc_tok@gw:10255');
        args.push(
          '-v',
          '/host/onecli-combined-ca.pem:/tmp/onecli-combined-ca.pem:ro',
        );
        return Promise.resolve(true);
      });

      await getOneCliOutboundConfig('main');
      vi.advanceTimersByTime(61_000); // past the 60s TTL
      await getOneCliOutboundConfig('main');
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(2);
    });
  });

  describe('isOneCliConfigured', () => {
    it('returns false when neither env is set', () => {
      expect(isOneCliConfigured()).toBe(false);
    });

    it('returns false when only URL is set', () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      expect(isOneCliConfigured()).toBe(false);
    });

    it('returns false when only API key is set', () => {
      envFileMock.ONECLI_API_KEY = 'oc_test';
      expect(isOneCliConfigured()).toBe(false);
    });

    it('returns true when both are set', () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      expect(isOneCliConfigured()).toBe(true);
    });
  });

  describe('ensureAgentForTier', () => {
    it('is a no-op when OneCLI is unconfigured', async () => {
      await ensureAgentForTier('main');
      expect(ensureAgentMock).not.toHaveBeenCalled();
    });

    it('calls SDK ensureAgent with tier-scoped identifier', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      ensureAgentMock.mockResolvedValue({
        name: 'NanoClaw main tier',
        identifier: 'nanoclaw-main',
        created: true,
      });

      await ensureAgentForTier('main');

      expect(ensureAgentMock).toHaveBeenCalledWith({
        name: 'NanoClaw main tier',
        identifier: 'nanoclaw-main',
      });
    });

    it('logs and resolves on OneCLIRequestError', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      ensureAgentMock.mockRejectedValue(
        new FakeOneCLIRequestError('boom', {
          url: 'http://localhost:10254/v1/agents',
          statusCode: 503,
        }),
      );

      await expect(ensureAgentForTier('trusted')).resolves.toBeUndefined();
    });

    it('logs and resolves on OneCLIError', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      ensureAgentMock.mockRejectedValue(new FakeOneCLIError('missing api key'));

      await expect(ensureAgentForTier('untrusted')).resolves.toBeUndefined();
    });

    it('propagates unexpected error classes', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      ensureAgentMock.mockRejectedValue(new TypeError('bad call'));

      await expect(ensureAgentForTier('main')).rejects.toThrow(TypeError);
    });
  });

  describe('applyOneCliToSpawn', () => {
    // The in-spawn retry (#787) sleeps between attempts; fake timers keep the
    // false/throw paths deterministic and fast. Success-on-first-attempt tests
    // schedule no timer and resolve through microtasks unaffected.
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(0);
    });
    afterEach(() => {
      vi.useRealTimers();
    });

    it('returns false and does not mutate args when unconfigured', async () => {
      const args = ['run', '-i', '--rm', 'image'];
      const snapshot = [...args];
      const active = await applyOneCliToSpawn(args, 'main');
      expect(active).toBe(false);
      expect(args).toEqual(snapshot);
      expect(applyContainerConfigMock).not.toHaveBeenCalled();
    });

    it('calls SDK applyContainerConfig with tier agent + bundle/host opts', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://onecli');
        return Promise.resolve(true);
      });

      const args = ['run', '-i', '--rm', 'image'];
      const active = await applyOneCliToSpawn(args, 'untrusted');

      expect(active).toBe(true);
      expect(applyContainerConfigMock).toHaveBeenCalledWith(args, {
        agent: 'nanoclaw-untrusted',
        combineCaBundle: true,
        // #746: spawn argv already carries --add-host from hostGatewayArgs();
        // the SDK must not add its duplicate.
        addHostMapping: false,
      });
      expect(args).toContain('HTTPS_PROXY=http://onecli');
    });

    it('appends NO_PROXY (both cases) excluding the host-gateway when the proxy is applied (#640 — preserves the Anthropic cred-proxy hop)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        // The gateway sets HTTP_PROXY + HTTPS_PROXY but no NO_PROXY.
        args.push('-e', 'HTTP_PROXY=http://onecli');
        args.push('-e', 'HTTPS_PROXY=http://onecli');
        return Promise.resolve(true);
      });

      const args = ['run', '-i', '--rm', 'image'];
      const active = await applyOneCliToSpawn(args, 'main');

      expect(active).toBe(true);
      expect(args).toContain(
        'NO_PROXY=host.docker.internal,localhost,127.0.0.1',
      );
      expect(args).toContain(
        'no_proxy=host.docker.internal,localhost,127.0.0.1',
      );
    });

    it('MERGES the required bypass hosts into a pre-existing NO_PROXY instead of overriding it (#640 — never drops an upstream-set bypass)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://onecli');
        // A future gateway ships its own bypass across MULTIPLE entries (both
        // cases, distinct values) — every one must be preserved, not just the
        // first matched.
        args.push('-e', 'NO_PROXY=internal.corp');
        args.push('-e', 'no_proxy=10.0.0.0/8');
        return Promise.resolve(true);
      });

      const args = ['run', '-i', '--rm', 'image'];
      await applyOneCliToSpawn(args, 'main');

      // The winning (last) NO_PROXY carries the union — every upstream host
      // kept, our bypass hosts added, no duplicates.
      const merged = args
        .filter((a) => a.startsWith('NO_PROXY='))
        .pop()!
        .slice('NO_PROXY='.length)
        .split(',');
      expect(merged).toContain('internal.corp');
      expect(merged).toContain('10.0.0.0/8');
      expect(merged).toContain('host.docker.internal');
      expect(merged).toContain('localhost');
      expect(merged).toContain('127.0.0.1');
      // No duplicate host entries.
      expect(new Set(merged).size).toBe(merged.length);
    });

    it('appends NO_PROXY when proxy env landed on argv even if the SDK reports inactive (#640 — decoupled from `active` so SDK drift cannot resurrect INCIDENT-746)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      // Hypothetical drift: SDK pushes proxy env but still resolves false.
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://onecli');
        return Promise.resolve(false);
      });

      const args = ['run', '-i', '--rm', 'image'];
      const p = applyOneCliToSpawn(args, 'main');
      await vi.runAllTimersAsync(); // flush the #787 retry backoffs
      const active = await p;

      expect(active).toBe(false);
      // The cred-proxy bypass rides on the proxy env being present, not on the
      // return value — so the agent's Anthropic hop stays direct regardless.
      // Truncation between retries means the final argv carries exactly one
      // HTTPS_PROXY / NO_PROXY pair despite three attempts (#787).
      expect(args.filter((a) => a.startsWith('HTTPS_PROXY='))).toHaveLength(1);
      expect(args).toContain(
        'NO_PROXY=host.docker.internal,localhost,127.0.0.1',
      );
      expect(args).toContain(
        'no_proxy=host.docker.internal,localhost,127.0.0.1',
      );
    });

    it('does NOT append NO_PROXY when no proxy env landed on argv (SDK inactive, nothing pushed)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockResolvedValue(false);

      const args = ['run', '-i', '--rm', 'image'];
      const p = applyOneCliToSpawn(args, 'main');
      await vi.runAllTimersAsync(); // flush the #787 retry backoffs
      const active = await p;

      expect(active).toBe(false);
      expect(args.some((a) => a.startsWith('NO_PROXY='))).toBe(false);
      expect(args.some((a) => a.startsWith('no_proxy='))).toBe(false);
    });

    it('strips the gateway-injected ANTHROPIC_API_KEY, keeps OPENAI_API_KEY (#640 — root cause of the cutover 401s)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        // The gateway's container-config injects its MITM placeholders + proxy
        // env. ANTHROPIC_API_KEY=<sentinel> would flip the Claude SDK into
        // api-key mode (x-api-key, no Authorization) — but Anthropic rides the
        // cred-proxy, never the gateway, so the sentinel would reach Anthropic
        // raw → 401. OPENAI_API_KEY is fine: OpenAI DOES traverse the gateway.
        args.push('-e', 'HTTPS_PROXY=http://onecli');
        args.push('-e', 'ANTHROPIC_API_KEY=placeholder');
        args.push('-e', 'OPENAI_API_KEY=placeholder');
        return Promise.resolve(true);
      });

      const args = ['run', '-i', '--rm', 'image'];
      await applyOneCliToSpawn(args, 'main');

      // The gateway ANTHROPIC_API_KEY and its `-e` flag are gone.
      expect(args.some((a) => a.startsWith('ANTHROPIC_API_KEY='))).toBe(false);
      // OpenAI's gateway swap still works, so its placeholder stays.
      expect(args).toContain('OPENAI_API_KEY=placeholder');
      // No dangling `-e`: every remaining flag keeps its KEY=VALUE partner.
      args.forEach((a, i) => {
        if (a === '-e') expect(args[i + 1]).toMatch(/=/);
      });
    });

    it('preserves an ANTHROPIC_API_KEY the caller set BEFORE applyContainerConfig (api-key-mode placeholder untouched)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://onecli');
        args.push('-e', 'ANTHROPIC_API_KEY=gateway-sentinel');
        return Promise.resolve(true);
      });

      // container-runner sets ANTHROPIC_API_KEY=placeholder BEFORE calling
      // applyOneCliToSpawn when the host is in api-key mode; the cred-proxy
      // swaps that one for the real key, so it must survive. Only the tail the
      // SDK appended is scrubbed.
      const args = [
        'run',
        '-i',
        '--rm',
        '-e',
        'ANTHROPIC_API_KEY=placeholder',
        'image',
      ];
      await applyOneCliToSpawn(args, 'main');

      expect(args.filter((a) => a.startsWith('ANTHROPIC_API_KEY='))).toEqual([
        'ANTHROPIC_API_KEY=placeholder',
      ]);
    });

    it('warns when SDK resolves false (configured but unreachable gateway)', async () => {
      // The real SDK catches gateway/config fetch failures and resolves
      // `false` instead of throwing. The `applyOneCliToSpawn` wrapper must
      // surface this as a WARN — otherwise a misconfigured gateway results
      // in silent fallback with zero diagnostic for the operator.
      const { logger } = await import('./logger.js');
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockResolvedValue(false);

      const args = ['run', '-i', '--rm', 'image'];
      const p = applyOneCliToSpawn(args, 'main');
      await vi.runAllTimersAsync(); // flush the #787 retry backoffs
      const active = await p;

      expect(active).toBe(false);
      const warnCalls = vi.mocked(logger.warn).mock.calls;
      const matchingCall = warnCalls.find(
        (call) =>
          typeof call[1] === 'string' &&
          call[1].includes('`applyContainerConfig` returned false'),
      );
      expect(matchingCall).toBeDefined();
      expect(matchingCall![0]).toEqual({ tier: 'main' });
    });

    it('returns false (does not throw) on OneCLIRequestError', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockRejectedValue(
        new FakeOneCLIRequestError('boom', {
          url: 'http://localhost:10254/v1/container-config',
          statusCode: 502,
        }),
      );

      const args = ['run', '-i', '--rm', 'image'];
      const p = applyOneCliToSpawn(args, 'main');
      await vi.runAllTimersAsync(); // flush the #787 retry backoffs
      const active = await p;
      expect(active).toBe(false);
    });

    it('returns false on OneCLIError', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockRejectedValue(
        new FakeOneCLIError('bad config'),
      );

      const args = ['run', '-i', '--rm', 'image'];
      const p = applyOneCliToSpawn(args, 'trusted');
      await vi.runAllTimersAsync(); // flush the #787 retry backoffs
      const active = await p;
      expect(active).toBe(false);
    });

    it('propagates unexpected error classes', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockRejectedValue(new TypeError('bad call'));

      const args = ['run', '-i', '--rm', 'image'];
      // A non-OneCLI error is not retried — it propagates on the first attempt.
      await expect(applyOneCliToSpawn(args, 'main')).rejects.toThrow(TypeError);
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(1);
    });

    it('truncates a partial argv append when the SDK appends then throws (Copilot #788 — no half-applied config leaks to the caller)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      // The SDK pushes proxy env, THEN rejects mid-way — every attempt.
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://onecli');
        return Promise.reject(
          new FakeOneCLIRequestError('mid-append blowup', {
            url: 'http://localhost:10254/v1/container-config',
            statusCode: 503,
          }),
        );
      });

      const args = ['run', '-i', '--rm', 'image'];
      const snapshot = [...args];
      const p = applyOneCliToSpawn(args, 'main');
      await vi.runAllTimersAsync();
      const active = await p;

      // Retryable 503 exhausts to false, and the partial HTTPS_PROXY the failed
      // final attempt appended is gone — argv is exactly what the caller passed.
      expect(active).toBe(false);
      expect(args).toEqual(snapshot);
    });

    it('truncates a partial argv append on a non-retryable rejection (Copilot #788)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockImplementation((args: string[]) => {
        args.push('-e', 'HTTPS_PROXY=http://onecli');
        return Promise.reject(
          new FakeOneCLIRequestError('forbidden', {
            url: 'http://localhost:10254/v1/container-config',
            statusCode: 403,
          }),
        );
      });

      const args = ['run', '-i', '--rm', 'image'];
      const snapshot = [...args];
      const p = applyOneCliToSpawn(args, 'main');
      await vi.runAllTimersAsync();
      const active = await p;

      expect(active).toBe(false);
      // One attempt (403 is deterministic), and no stray proxy env left behind.
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(1);
      expect(args).toEqual(snapshot);
    });

    it('retries a transient blip and recovers within the same spawn (#787)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock
        .mockRejectedValueOnce(
          new FakeOneCLIRequestError('gateway hiccup', {
            url: 'http://localhost:10254/v1/container-config',
            statusCode: 503,
          }),
        )
        .mockImplementationOnce((args: string[]) => {
          args.push('-e', 'HTTPS_PROXY=http://onecli');
          return Promise.resolve(true);
        });

      const args = ['run', '-i', '--rm', 'image'];
      const p = applyOneCliToSpawn(args, 'main');
      await vi.runAllTimersAsync();
      const active = await p;

      expect(active).toBe(true);
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(2);
      // The failed first attempt's would-be append never lands twice — argv
      // carries exactly one HTTPS_PROXY (truncate-between-attempts, #787).
      expect(args.filter((a) => a.startsWith('HTTPS_PROXY='))).toHaveLength(1);
    });

    it('recovers when the first attempt resolves false then the retry succeeds (#787)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock
        .mockResolvedValueOnce(false)
        .mockImplementationOnce((args: string[]) => {
          args.push('-e', 'HTTPS_PROXY=http://onecli');
          return Promise.resolve(true);
        });

      const args = ['run', '-i', '--rm', 'image'];
      const p = applyOneCliToSpawn(args, 'main');
      await vi.runAllTimersAsync();
      const active = await p;

      expect(active).toBe(true);
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(2);
      expect(args.filter((a) => a.startsWith('HTTPS_PROXY='))).toHaveLength(1);
    });

    it('does NOT retry a deterministic 4xx rejection — fails fast (#787)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockRejectedValue(
        new FakeOneCLIRequestError('unknown agent', {
          url: 'http://localhost:10254/v1/container-config',
          statusCode: 404,
        }),
      );

      const args = ['run', '-i', '--rm', 'image'];
      const p = applyOneCliToSpawn(args, 'main');
      await vi.runAllTimersAsync();
      const active = await p;

      expect(active).toBe(false);
      // 404 is deterministic — one attempt, no retry storm against a bad config.
      expect(applyContainerConfigMock).toHaveBeenCalledTimes(1);
    });
  });

  it('TRUST_TIERS lists every tier exactly once', () => {
    expect([...TRUST_TIERS].sort()).toEqual(['main', 'trusted', 'untrusted']);
  });
});

// -----------------------------------------------------------------
// oneCliSpawnEnvArgs — the docker `-e` args that put an in-container
// gateway CLI into OneCLI mode (#748). Pure: gated specifically on
// HTTPS_PROXY landing on the argv (not any proxy), null-safe on env,
// and it must NEVER surface ONECLI_API_KEY onto the spawn.
// -----------------------------------------------------------------
describe('oneCliSpawnEnvArgs', () => {
  const envOptions = { url: 'http://host.docker.internal:8080' };
  const withHttps = [
    'run',
    '-e',
    'HTTPS_PROXY=http://host.docker.internal:9000',
  ];

  it('injects ONECLI_URL + ENABLE_OOO=1 when HTTPS_PROXY landed', () => {
    expect(oneCliSpawnEnvArgs(withHttps, envOptions)).toEqual([
      '-e',
      'ONECLI_URL=http://host.docker.internal:8080',
      '-e',
      'ENABLE_OOO=1',
    ]);
  });

  it('injects nothing when only HTTP_PROXY landed (not HTTPS_PROXY)', () => {
    // The package fail-fasts when ONECLI_URL is set without HTTPS_PROXY, so a
    // partial-config spawn must not receive it.
    const httpOnly = [
      'run',
      '-e',
      'HTTP_PROXY=http://host.docker.internal:9000',
    ];
    expect(oneCliSpawnEnvArgs(httpOnly, envOptions)).toEqual([]);
  });

  it('injects nothing when no proxy env landed', () => {
    expect(
      oneCliSpawnEnvArgs(['run', '-i', '--rm', 'image'], envOptions),
    ).toEqual([]);
  });

  it('injects nothing when OneCLI is unconfigured (null env)', () => {
    expect(oneCliSpawnEnvArgs(withHttps, null)).toEqual([]);
  });

  it('never forwards ONECLI_API_KEY onto the spawn', () => {
    // Even if a key rides on the env-options object, only the URL may leave —
    // the container authenticates with the agent-scoped token in HTTPS_PROXY.
    const withKey = {
      url: 'http://host.docker.internal:8080',
      apiKey: 'sk-secret',
    };
    const out = oneCliSpawnEnvArgs(withHttps, withKey);
    expect(out.join(' ')).not.toContain('ONECLI_API_KEY');
    expect(out.join(' ')).not.toContain('sk-secret');
  });
});
