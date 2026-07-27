import fs from 'fs';
import path from 'path';

import { describe, it, expect, beforeEach, afterAll, vi } from 'vitest';

const { TEST_STORE_DIR } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const osMod = require('os') as typeof import('os');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pathMod = require('path') as typeof import('path');
  return {
    TEST_STORE_DIR: pathMod.join(
      osMod.tmpdir(),
      `nanoclaw-tessl-env-test-${process.pid}`,
    ),
  };
});
vi.mock('./config.js', async () => {
  const actual =
    await vi.importActual<typeof import('./config.js')>('./config.js');
  return { ...actual, STORE_DIR: TEST_STORE_DIR };
});

const { mockIsConfigured, mockOutbound } = vi.hoisted(() => ({
  mockIsConfigured: vi.fn(),
  mockOutbound: vi.fn(),
}));
vi.mock('./onecli-client.js', async () => {
  const actual =
    await vi.importActual<typeof import('./onecli-client.js')>(
      './onecli-client.js',
    );
  return {
    ...actual,
    isOneCliConfigured: mockIsConfigured,
    getOneCliOutboundConfig: mockOutbound,
  };
});

import { buildTesslChildEnv } from './tessl-env.js';
import { ONECLI_MANAGED_PLACEHOLDER } from './onecli-client.js';

const PROXY = 'http://x:aoc_testcred@host.docker.internal:10255';
const CA = '-----BEGIN CERTIFICATE-----\nTESTCA\n-----END CERTIFICATE-----\n';

beforeEach(() => {
  fs.rmSync(TEST_STORE_DIR, { recursive: true, force: true });
  mockIsConfigured.mockReset();
  mockOutbound.mockReset();
});

afterAll(() => {
  fs.rmSync(TEST_STORE_DIR, { recursive: true, force: true });
});

describe('buildTesslChildEnv (#887)', () => {
  it('returns nothing when OneCLI is unconfigured, so tessl keeps its direct path', async () => {
    // A dev checkout with no gateway must still be able to run
    // `tessl update` against its ambient ~/.tessl session.
    mockIsConfigured.mockReturnValue(false);
    expect(await buildTesslChildEnv()).toEqual({});
    expect(mockOutbound).not.toHaveBeenCalled();
  });

  it('returns nothing when the gateway is unreachable', async () => {
    mockIsConfigured.mockReturnValue(true);
    mockOutbound.mockResolvedValue(null);
    expect(await buildTesslChildEnv()).toEqual({});
  });

  it('emits the proxy, the CA path, and the placeholder together', async () => {
    mockIsConfigured.mockReturnValue(true);
    mockOutbound.mockResolvedValue({ proxyUrl: PROXY, ca: CA });
    const env = await buildTesslChildEnv();
    expect(env.HTTPS_PROXY).toBe(PROXY);
    expect(env.HTTP_PROXY).toBe(PROXY);
    expect(env.TESSL_TOKEN).toBe(ONECLI_MANAGED_PLACEHOLDER);
    expect(env.NODE_EXTRA_CA_CERTS).toBeTruthy();
  });

  it('never emits the placeholder token without the proxy that makes it valid', async () => {
    // The pairing is the whole contract: a placeholder on the direct
    // path is a dead credential that 401s against api.tessl.io.
    mockIsConfigured.mockReturnValue(true);
    for (const outbound of [null, undefined]) {
      mockOutbound.mockResolvedValue(outbound);
      const env = await buildTesslChildEnv();
      expect(env.TESSL_TOKEN).toBeUndefined();
      expect(env.HTTPS_PROXY).toBeUndefined();
    }
  });

  it('writes the CA to the path it advertises, with the gateway content', async () => {
    // NODE_EXTRA_CA_CERTS takes a PATH but the gateway hands back
    // CONTENT, so the file has to actually exist or the MITM leg fails
    // TLS and every registry call breaks.
    mockIsConfigured.mockReturnValue(true);
    mockOutbound.mockResolvedValue({ proxyUrl: PROXY, ca: CA });
    const env = await buildTesslChildEnv();
    const caPath = env.NODE_EXTRA_CA_CERTS as string;
    expect(fs.existsSync(caPath)).toBe(true);
    expect(fs.readFileSync(caPath, 'utf8')).toBe(CA);
  });

  it('falls back to the direct path when the CA cannot be written', async () => {
    mockIsConfigured.mockReturnValue(true);
    mockOutbound.mockResolvedValue({ proxyUrl: PROXY, ca: CA });
    const spy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {
      throw Object.assign(new Error('EACCES: permission denied'), {
        code: 'EACCES',
      });
    });
    try {
      // Handing tessl a proxy whose CA it cannot validate would break
      // every call; degrade to direct instead.
      expect(await buildTesslChildEnv()).toEqual({});
    } finally {
      spy.mockRestore();
    }
  });

  it('rethrows a non-Error write failure rather than masking a bug', async () => {
    mockIsConfigured.mockReturnValue(true);
    mockOutbound.mockResolvedValue({ proxyUrl: PROXY, ca: CA });
    const spy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {
      // eslint-disable-next-line @typescript-eslint/only-throw-error
      throw 'not-an-error';
    });
    try {
      await expect(buildTesslChildEnv()).rejects.toBe('not-an-error');
    } finally {
      spy.mockRestore();
    }
  });

  it('creates STORE_DIR when it does not exist yet', async () => {
    // First run on a fresh container: writing the CA must not fail
    // merely because the directory has never been created.
    expect(fs.existsSync(TEST_STORE_DIR)).toBe(false);
    mockIsConfigured.mockReturnValue(true);
    mockOutbound.mockResolvedValue({ proxyUrl: PROXY, ca: CA });
    const env = await buildTesslChildEnv();
    expect(fs.existsSync(TEST_STORE_DIR)).toBe(true);
    expect(path.dirname(env.NODE_EXTRA_CA_CERTS as string)).toBe(
      TEST_STORE_DIR,
    );
  });
});
