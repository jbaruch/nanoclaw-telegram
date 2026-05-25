import { describe, it, expect, beforeEach, vi } from 'vitest';

const {
  ensureAgentMock,
  applyContainerConfigMock,
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
  isOneCliConfigured,
  TRUST_TIERS,
  _resetOneCliClient,
} from './onecli-client.js';

describe('onecli-client', () => {
  beforeEach(() => {
    for (const k of Object.keys(envFileMock)) delete envFileMock[k];
    _resetOneCliClient();
    ensureAgentMock.mockReset();
    applyContainerConfigMock.mockReset();
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
        addHostMapping: true,
      });
      expect(args).toContain('HTTPS_PROXY=http://onecli');
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
