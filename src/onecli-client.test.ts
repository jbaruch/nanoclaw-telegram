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
  oneCliAgentProxyEnabled,
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
      expect(await getOneCliOutboundConfig('main')).toBeNull();
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
      expect(await getOneCliOutboundConfig('main')).toBeNull();
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

  describe('oneCliAgentProxyEnabled', () => {
    it('is false by default (flag absent)', () => {
      expect(oneCliAgentProxyEnabled()).toBe(false);
    });

    it('is false for any value other than "1"', () => {
      envFileMock.ONECLI_AGENT_PROXY = 'true';
      expect(oneCliAgentProxyEnabled()).toBe(false);
    });

    it('is true only when ONECLI_AGENT_PROXY === "1"', () => {
      envFileMock.ONECLI_AGENT_PROXY = '1';
      expect(oneCliAgentProxyEnabled()).toBe(true);
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

    it('does NOT append NO_PROXY when the SDK reports inactive (no proxy env landed)', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockResolvedValue(false);

      const args = ['run', '-i', '--rm', 'image'];
      const active = await applyOneCliToSpawn(args, 'main');

      expect(active).toBe(false);
      expect(args.some((a) => a.startsWith('NO_PROXY='))).toBe(false);
      expect(args.some((a) => a.startsWith('no_proxy='))).toBe(false);
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
      const active = await applyOneCliToSpawn(args, 'main');

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
      const active = await applyOneCliToSpawn(args, 'main');
      expect(active).toBe(false);
    });

    it('returns false on OneCLIError', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockRejectedValue(
        new FakeOneCLIError('bad config'),
      );

      const args = ['run', '-i', '--rm', 'image'];
      const active = await applyOneCliToSpawn(args, 'trusted');
      expect(active).toBe(false);
    });

    it('propagates unexpected error classes', async () => {
      envFileMock.ONECLI_URL = 'http://localhost:10254';
      envFileMock.ONECLI_API_KEY = 'oc_test';
      applyContainerConfigMock.mockRejectedValue(new TypeError('bad call'));

      const args = ['run', '-i', '--rm', 'image'];
      await expect(applyOneCliToSpawn(args, 'main')).rejects.toThrow(TypeError);
    });
  });

  it('TRUST_TIERS lists every tier exactly once', () => {
    expect([...TRUST_TIERS].sort()).toEqual(['main', 'trusted', 'untrusted']);
  });
});
