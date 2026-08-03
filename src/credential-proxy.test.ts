import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import http from 'http';
import { promises as fsp } from 'fs';
import { join } from 'path';
import os from 'os';
import type { AddressInfo } from 'net';

const mockEnv: Record<string, string> = {};
vi.mock('./env.js', () => ({
  readEnvFile: vi.fn(() => ({ ...mockEnv })),
}));

vi.mock('./logger.js', () => ({
  logger: { info: vi.fn(), error: vi.fn(), debug: vi.fn(), warn: vi.fn() },
}));

// #637: default OneCLI OFF (matches every pre-existing test — the proxy
// injects from .env). The OneCLI-exchange test flips isOneCliConfigured on and
// has getOneCliOutboundConfig return a config. HttpsProxyAgent is mocked to a
// plain http.Agent so the request reaches the local upstream mock directly
// (the real CONNECT-tunnel-through-OneCLI path is verified live, not here).
// Spread the real module so unmocked exports stay real — #893's denial
// log names the vault agent via `agentIdentifierForTier`, and a stub
// there would let the test agree with itself while the shipped message
// named something else. Only the two gateway-touching functions are
// replaced.
vi.mock('./onecli-client.js', async () => {
  const actual =
    await vi.importActual<typeof import('./onecli-client.js')>(
      './onecli-client.js',
    );
  return {
    ...actual,
    isOneCliConfigured: vi.fn(() => false),
    getOneCliOutboundConfig: vi.fn(async () => null),
  };
});
// Spy on HttpsProxyAgent construction so a CI test can assert the HTTPS
// OneCLI routing branch builds the agent with the minted proxy URL + CA
// (the real CONNECT/TLS tunnel is platform-bound and verified live at cutover).
// Spy on https.request and delegate to http.request so an `https:` upstream
// URL (isHttps === true) actually reaches the local HTTP upstream mock. Lets a
// CI test assert the OneCLI agent is attached to the HTTPS request options.
const httpsRequestSpy = vi.hoisted(() => vi.fn());
vi.mock('https', async () => {
  const actual = await vi.importActual<typeof import('https')>('https');
  const httpMod = await import('http');
  return {
    ...actual,
    request: (opts: unknown, cb: unknown) => {
      httpsRequestSpy(opts);
      return (httpMod.request as (o: unknown, c: unknown) => unknown)(opts, cb);
    },
  };
});

const httpsProxyAgentCtor = vi.hoisted(() => vi.fn());
vi.mock('https-proxy-agent', async () => {
  const httpMod = await import('http');
  // Real class extending http.Agent so `new HttpsProxyAgent(...)` yields a
  // working agent that connects directly to the local upstream mock.
  return {
    HttpsProxyAgent: class extends httpMod.Agent {
      constructor(...args: unknown[]) {
        super();
        httpsProxyAgentCtor(...args);
      }
    },
  };
});

import { startCredentialProxy } from './credential-proxy.js';
import { isFsErrorWithCode } from './fs-errors.js';
import { logger } from './logger.js';
import { DENIAL_ALERT_STREAK } from './onecli-denial-alert.js';
import {
  isOneCliConfigured,
  getOneCliOutboundConfig,
} from './onecli-client.js';
import { registerContainer, _resetRegistry } from './proxy-registry.js';

function makeRequest(
  port: number,
  options: http.RequestOptions,
  body = '',
): Promise<{
  statusCode: number;
  body: string;
  headers: http.IncomingHttpHeaders;
}> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { ...options, hostname: '127.0.0.1', port },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          resolve({
            statusCode: res.statusCode!,
            body: Buffer.concat(chunks).toString(),
            headers: res.headers,
          });
        });
      },
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

/**
 * Poll `usageLogPath` until it has at least `expectedLines` JSON lines
 * or the deadline (default 5s) elapses. Replaces fixed `setTimeout()`
 * waits so the test isn't timing-dependent on slow CI — completion is
 * tied to observable file state. Returns the lines so the caller can
 * assert on them directly.
 */
async function waitForUsageLines(
  usageLogPath: string,
  expectedLines: number,
  deadlineMs = 5000,
): Promise<string[]> {
  const start = Date.now();
  // Poll every 5ms — short enough to keep the test fast on healthy
  // CI, long enough not to thrash the FS on a slow runner.
  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      const content = await fsp.readFile(usageLogPath, 'utf8');
      const lines = content.trim() ? content.trim().split('\n') : [];
      if (lines.length >= expectedLines) return lines;
    } catch (err) {
      // ENOENT — the proxy hasn't created the usage log yet; keep polling.
      // Any other fs error (or a non-errno defect) surfaces.
      if (!isFsErrorWithCode(err, ['ENOENT'])) throw err;
    }
    if (Date.now() - start > deadlineMs) {
      throw new Error(
        `usage log did not reach ${expectedLines} line(s) within ${deadlineMs}ms`,
      );
    }
    await new Promise((r) => setTimeout(r, 5));
  }
}

describe('credential-proxy', () => {
  let proxyServer: http.Server;
  let upstreamServer: http.Server;
  let proxyPort: number;
  let upstreamPort: number;
  let lastUpstreamHeaders: http.IncomingHttpHeaders;

  beforeEach(async () => {
    lastUpstreamHeaders = {};

    upstreamServer = http.createServer((req, res) => {
      lastUpstreamHeaders = { ...req.headers };
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
    });
    await new Promise<void>((resolve) =>
      upstreamServer.listen(0, '127.0.0.1', resolve),
    );
    upstreamPort = (upstreamServer.address() as AddressInfo).port;
  });

  afterEach(async () => {
    await new Promise<void>((r) => proxyServer?.close(() => r()));
    await new Promise<void>((r) => upstreamServer?.close(() => r()));
    for (const key of Object.keys(mockEnv)) delete mockEnv[key];
    // Restore OneCLI-off default AND clear call history so no later test sees
    // it configured or inherits a prior call count.
    vi.mocked(isOneCliConfigured).mockReset();
    vi.mocked(isOneCliConfigured).mockReturnValue(false);
    vi.mocked(getOneCliOutboundConfig).mockReset();
    vi.mocked(getOneCliOutboundConfig).mockResolvedValue(null);
    httpsProxyAgentCtor.mockClear();
    httpsRequestSpy.mockClear();
  });

  async function startProxy(env: Record<string, string>): Promise<number> {
    Object.assign(mockEnv, env);
    proxyServer = await startCredentialProxy(0, '127.0.0.1', {
      upstreamUrl: new URL(`http://127.0.0.1:${upstreamPort}`),
    });
    return (proxyServer.address() as AddressInfo).port;
  }

  it('API-key mode injects x-api-key and strips placeholder', async () => {
    proxyPort = await startProxy({ ANTHROPIC_API_KEY: 'sk-ant-real-key' });

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: '/v1/messages',
        headers: {
          'content-type': 'application/json',
          'x-api-key': 'placeholder',
        },
      },
      '{}',
    );

    expect(lastUpstreamHeaders['x-api-key']).toBe('sk-ant-real-key');
  });

  it('OAuth mode replaces Authorization when container sends one', async () => {
    proxyPort = await startProxy({
      CLAUDE_CODE_OAUTH_TOKEN: 'real-oauth-token',
    });

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: '/api/oauth/claude_cli/create_api_key',
        headers: {
          'content-type': 'application/json',
          authorization: 'Bearer placeholder',
        },
      },
      '{}',
    );

    expect(lastUpstreamHeaders['authorization']).toBe(
      'Bearer real-oauth-token',
    );
  });

  it('#637: OAuth exchange routes through OneCLI and does NOT inject the .env token when OneCLI is configured', async () => {
    vi.mocked(isOneCliConfigured).mockReturnValue(true);
    vi.mocked(getOneCliOutboundConfig).mockResolvedValue({
      proxyUrl: 'http://x:aoc_tok@gw:10255',
      ca: 'fake-ca',
    });
    proxyPort = await startProxy({
      CLAUDE_CODE_OAUTH_TOKEN: 'real-oauth-token',
    });

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: '/api/oauth/claude_cli/create_api_key',
        headers: {
          'content-type': 'application/json',
          authorization: 'Bearer placeholder',
        },
      },
      '{}',
    );

    // The credential-proxy must NOT replace the placeholder with the .env
    // token — OneCLI injects the real Bearer on the (mocked) tunnel, so the
    // container's placeholder passes through here untouched.
    expect(lastUpstreamHeaders['authorization']).toBe('Bearer placeholder');
    expect(lastUpstreamHeaders['authorization']).not.toBe(
      'Bearer real-oauth-token',
    );
    expect(vi.mocked(getOneCliOutboundConfig)).toHaveBeenCalledWith('main');
    // The HTTPS routing branch built the tunnel agent with the minted proxy
    // URL. The CA is NOT passed to the agent constructor (https-proxy-agent
    // ignores it) — it goes on the request options; see the HTTPS-upstream
    // test below.
    expect(httpsProxyAgentCtor).toHaveBeenCalledWith(
      'http://x:aoc_tok@gw:10255',
    );
  });

  it('#637: attaches the OneCLI agent to the request options on an HTTPS upstream', async () => {
    vi.mocked(isOneCliConfigured).mockReturnValue(true);
    vi.mocked(getOneCliOutboundConfig).mockResolvedValue({
      proxyUrl: 'http://x:aoc_tok@gw:10255',
      ca: 'fake-ca',
    });
    Object.assign(mockEnv, { CLAUDE_CODE_OAUTH_TOKEN: 'real-oauth-token' });
    // https: upstream → isHttps true → the `oneCliAgent && isHttps` branch
    // attaches the agent. Mocked https.request delegates to http so it still
    // reaches the local HTTP upstream mock.
    proxyServer = await startCredentialProxy(0, '127.0.0.1', {
      upstreamUrl: new URL(`https://127.0.0.1:${upstreamPort}`),
    });
    const port = (proxyServer.address() as AddressInfo).port;

    await makeRequest(
      port,
      {
        method: 'POST',
        path: '/api/oauth/claude_cli/create_api_key',
        headers: {
          'content-type': 'application/json',
          authorization: 'Bearer placeholder',
        },
      },
      '{}',
    );

    expect(httpsRequestSpy).toHaveBeenCalledTimes(1);
    const opts = httpsRequestSpy.mock.calls[0]![0] as {
      agent?: unknown;
      ca?: unknown;
    };
    expect(opts.agent).toBeDefined();
    // The CA is applied on the request options (NOT the agent) — this is the
    // fix for the self-signed-certificate failure the live cutover exposed.
    expect(opts.ca).toBe('fake-ca');
  });

  it('#637: /v1/messages WITH an Authorization header routes through OneCLI (agent + CA on options, .env token not injected)', async () => {
    // The load-bearing path this whole PR fixes: in this deployment the SDK
    // sends a placeholder Bearer on /v1/messages, so it routes through OneCLI
    // (the cutover's self-signed errors were on /v1/messages). Assert the CA
    // lands on the request options and the .env token is NOT injected.
    vi.mocked(isOneCliConfigured).mockReturnValue(true);
    vi.mocked(getOneCliOutboundConfig).mockResolvedValue({
      proxyUrl: 'http://x:aoc_tok@gw:10255',
      ca: 'fake-ca',
    });
    Object.assign(mockEnv, { CLAUDE_CODE_OAUTH_TOKEN: 'real-oauth-token' });
    proxyServer = await startCredentialProxy(0, '127.0.0.1', {
      upstreamUrl: new URL(`https://127.0.0.1:${upstreamPort}`),
    });
    const port = (proxyServer.address() as AddressInfo).port;

    await makeRequest(
      port,
      {
        method: 'POST',
        path: '/v1/messages',
        headers: {
          'content-type': 'application/json',
          authorization: 'Bearer placeholder',
        },
      },
      '{}',
    );

    expect(httpsRequestSpy).toHaveBeenCalledTimes(1);
    const opts = httpsRequestSpy.mock.calls[0]![0] as {
      agent?: unknown;
      ca?: unknown;
    };
    expect(opts.agent).toBeDefined();
    expect(opts.ca).toBe('fake-ca');
    expect(vi.mocked(getOneCliOutboundConfig)).toHaveBeenCalled();
    // .env token NOT injected — OneCLI swaps the placeholder on the tunnel.
    expect(lastUpstreamHeaders['authorization']).toBe('Bearer placeholder');
  });

  it('#637: /v1/messages does NOT route through OneCLI (no Authorization header)', async () => {
    vi.mocked(isOneCliConfigured).mockReturnValue(true);
    vi.mocked(getOneCliOutboundConfig).mockResolvedValue({
      proxyUrl: 'http://x:aoc_tok@gw:10255',
      ca: 'fake-ca',
    });
    proxyPort = await startProxy({
      CLAUDE_CODE_OAUTH_TOKEN: 'real-oauth-token',
    });

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: '/v1/messages',
        headers: {
          'content-type': 'application/json',
          'x-api-key': 'temp-key-from-exchange',
        },
      },
      '{}',
    );

    // No Authorization → not the OneCLI path → forwarded direct with the temp
    // key, usage-tap path untouched. OneCLI is never consulted.
    expect(lastUpstreamHeaders['x-api-key']).toBe('temp-key-from-exchange');
    expect(vi.mocked(getOneCliOutboundConfig)).not.toHaveBeenCalled();
  });

  it('OAuth mode does not inject Authorization when container omits it', async () => {
    proxyPort = await startProxy({
      CLAUDE_CODE_OAUTH_TOKEN: 'real-oauth-token',
    });

    // Post-exchange: container uses x-api-key only, no Authorization header
    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: '/v1/messages',
        headers: {
          'content-type': 'application/json',
          'x-api-key': 'temp-key-from-exchange',
        },
      },
      '{}',
    );

    expect(lastUpstreamHeaders['x-api-key']).toBe('temp-key-from-exchange');
    expect(lastUpstreamHeaders['authorization']).toBeUndefined();
  });

  it('strips hop-by-hop headers', async () => {
    proxyPort = await startProxy({ ANTHROPIC_API_KEY: 'sk-ant-real-key' });

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: '/v1/messages',
        headers: {
          'content-type': 'application/json',
          connection: 'keep-alive',
          'keep-alive': 'timeout=5',
          'transfer-encoding': 'chunked',
        },
      },
      '{}',
    );

    // Proxy strips client hop-by-hop headers. Node's HTTP client may re-add
    // its own Connection header (standard HTTP/1.1 behavior), but the client's
    // custom keep-alive and transfer-encoding must not be forwarded.
    expect(lastUpstreamHeaders['keep-alive']).toBeUndefined();
    expect(lastUpstreamHeaders['transfer-encoding']).toBeUndefined();
  });

  it('returns 502 when the upstream is unreachable', async () => {
    // ECONNREFUSED on the sole upstream surfaces as a 502 to the client.
    Object.assign(mockEnv, { ANTHROPIC_API_KEY: 'sk-ant-real-key' });
    proxyServer = await startCredentialProxy(0, '127.0.0.1', {
      upstreamUrl: new URL('http://127.0.0.1:59999'),
    });
    proxyPort = (proxyServer.address() as AddressInfo).port;

    const res = await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: '/v1/messages',
        headers: { 'content-type': 'application/json' },
      },
      '{}',
    );

    expect(res.statusCode).toBe(502);
    expect(res.body).toBe('Bad Gateway');
  });

  it('passes a 4xx upstream error straight through to the client', async () => {
    // A 4xx is a real client error; the proxy forwards it verbatim.
    const upstream401: http.Server = http.createServer((_req, res) => {
      res.writeHead(401, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ type: 'error', error: { type: 'auth' } }));
    });
    await new Promise<void>((r) => upstream401.listen(0, '127.0.0.1', r));
    const upstream401Port = (upstream401.address() as AddressInfo).port;

    try {
      Object.assign(mockEnv, { ANTHROPIC_API_KEY: 'sk-ant-real-key' });
      proxyServer = await startCredentialProxy(0, '127.0.0.1', {
        upstreamUrl: new URL(`http://127.0.0.1:${upstream401Port}`),
      });
      proxyPort = (proxyServer.address() as AddressInfo).port;

      const res = await makeRequest(
        proxyPort,
        {
          method: 'POST',
          path: '/v1/messages',
          headers: { 'content-type': 'application/json' },
        },
        '{}',
      );

      expect(res.statusCode).toBe(401);
    } finally {
      await new Promise<void>((r) => upstream401.close(() => r()));
    }
  });
});

describe('credential-proxy denial visibility (#893)', () => {
  let proxyServer: http.Server;
  let denyingUpstream: http.Server;
  let denyPort: number;
  // Status the mock upstream answers with; each test sets it before
  // issuing a request so 401 and 403 share one server.
  let denyStatus: number;

  beforeEach(async () => {
    denyStatus = 401;
    denyingUpstream = http.createServer((_req, res) => {
      res.writeHead(denyStatus, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ type: 'error', error: { type: 'auth' } }));
    });
    await new Promise<void>((r) => denyingUpstream.listen(0, '127.0.0.1', r));
    denyPort = (denyingUpstream.address() as AddressInfo).port;
  });

  afterEach(async () => {
    await new Promise<void>((r) => proxyServer?.close(() => r()));
    await new Promise<void>((r) => denyingUpstream?.close(() => r()));
    for (const key of Object.keys(mockEnv)) delete mockEnv[key];
    _resetRegistry();
    vi.mocked(isOneCliConfigured).mockReset();
    vi.mocked(isOneCliConfigured).mockReturnValue(false);
    vi.mocked(getOneCliOutboundConfig).mockReset();
    vi.mocked(getOneCliOutboundConfig).mockResolvedValue(null);
    vi.mocked(logger.error).mockClear();
    vi.mocked(logger.warn).mockClear();
    httpsProxyAgentCtor.mockClear();
    httpsRequestSpy.mockClear();
  });

  /**
   * Start a proxy against the denying upstream with OneCLI either on
   * (the tunnelled path) or off (the `.env` fallback path).
   *
   * When `tier` is given, a container is registered and the request
   * carries its `/c/<token>` prefix, which is how the proxy learns the
   * caller's trust tier.
   */
  async function startAgainstDenyingUpstream(opts: {
    oneCli: boolean;
    tier?: 'main' | 'trusted' | 'untrusted';
    onTierDenialAlert?: (text: string) => void;
  }): Promise<{ port: number; pathPrefix: string }> {
    if (opts.oneCli) {
      vi.mocked(isOneCliConfigured).mockReturnValue(true);
      vi.mocked(getOneCliOutboundConfig).mockResolvedValue({
        proxyUrl: 'http://x:aoc_secret_key@gw:10255',
        ca: 'fake-ca',
      });
    }
    Object.assign(mockEnv, { CLAUDE_CODE_OAUTH_TOKEN: 'real-oauth-token' });
    proxyServer = await startCredentialProxy(0, '127.0.0.1', {
      upstreamUrl: new URL(`http://127.0.0.1:${denyPort}`),
      onTierDenialAlert: opts.onTierDenialAlert,
    });
    let pathPrefix = '';
    if (opts.tier) {
      const token = registerContainer({
        group: 'telegram_test',
        tier: opts.tier,
        session: 'session-1',
        task_id: null,
        message_id: null,
      });
      pathPrefix = `/c/${token}`;
    }
    return {
      port: (proxyServer.address() as AddressInfo).port,
      pathPrefix,
    };
  }

  function denyingRequest(
    port: number,
    pathPrefix: string,
  ): Promise<{ statusCode: number }> {
    return makeRequest(
      port,
      {
        method: 'POST',
        path: `${pathPrefix}/v1/messages`,
        headers: {
          'content-type': 'application/json',
          authorization: 'Bearer placeholder',
        },
      },
      '{}',
    );
  }

  it('logs a tunnelled 401 as a vault-grant problem naming the tier and host', async () => {
    const { port, pathPrefix } = await startAgainstDenyingUpstream({
      oneCli: true,
      tier: 'untrusted',
    });

    const res = await denyingRequest(port, pathPrefix);

    // The client still sees the upstream status verbatim — the proxy
    // reports, it does not rewrite or retry.
    expect(res.statusCode).toBe(401);

    const errorCalls = vi.mocked(logger.error).mock.calls;
    const denial = errorCalls.filter(([, msg]) =>
      String(msg).includes('OneCLI gateway denied'),
    );
    // "exactly one actionable orchestrator log line per request".
    expect(denial).toHaveLength(1);
    const [fields, message] = denial[0];
    expect(fields).toMatchObject({ tier: 'untrusted', status: 401 });
    expect((fields as { upstreamHost: string }).upstreamHost).toContain(
      '127.0.0.1',
    );
    expect(String(message)).toContain('untrusted');
    expect(String(message)).toContain('effective-credentials');
    expect(String(message)).toContain('nanoclaw-untrusted');
  });

  it('logs a tunnelled 403 the same way as a 401', async () => {
    denyStatus = 403;
    const { port, pathPrefix } = await startAgainstDenyingUpstream({
      oneCli: true,
      tier: 'trusted',
    });

    await denyingRequest(port, pathPrefix);

    const denial = vi
      .mocked(logger.error)
      .mock.calls.filter(([, msg]) =>
        String(msg).includes('OneCLI gateway denied'),
      );
    expect(denial).toHaveLength(1);
    expect(denial[0][0]).toMatchObject({ tier: 'trusted', status: 403 });
  });

  it('keeps the non-tunnelled 401 on its own .env message', async () => {
    // The `.env` fallback path points at the token in `.env`; sending
    // the operator to the vault here would be a wild goose chase.
    const { port, pathPrefix } = await startAgainstDenyingUpstream({
      oneCli: false,
    });

    await denyingRequest(port, pathPrefix);

    expect(
      vi
        .mocked(logger.error)
        .mock.calls.filter(([, msg]) =>
          String(msg).includes('OneCLI gateway denied'),
        ),
    ).toHaveLength(0);

    const fallback = vi
      .mocked(logger.warn)
      .mock.calls.filter(([, msg]) =>
        String(msg).includes('upstream rejected the request'),
      );
    expect(fallback).toHaveLength(1);
    expect(String(fallback[0][1])).toContain('.env');
    expect(String(fallback[0][1])).not.toContain('effective-credentials');
    expect(fallback[0][0]).not.toHaveProperty('tier');
  });

  it('never puts the gateway URL, its embedded key, or a token in the log', async () => {
    // `getOneCliOutboundConfig` mints a proxy URL carrying the gateway
    // API key in its userinfo. Logging the URL would put a live
    // credential in `orchestrator.log` (`coding-policy: no-secrets`).
    const { port, pathPrefix } = await startAgainstDenyingUpstream({
      oneCli: true,
      tier: 'untrusted',
    });

    await denyingRequest(port, pathPrefix);

    const serialized = JSON.stringify([
      ...vi.mocked(logger.error).mock.calls,
      ...vi.mocked(logger.warn).mock.calls,
    ]);
    expect(serialized).not.toContain('aoc_secret_key');
    expect(serialized).not.toContain('gw:10255');
    expect(serialized).not.toContain('real-oauth-token');
    expect(serialized).not.toContain('fake-ca');
  });

  it('logs the endpoint path without its query string', async () => {
    // `req.url`'s query is caller-controlled and unbounded. Logging it
    // would risk carrying a sensitive parameter into orchestrator.log
    // and let one request write an arbitrarily long line; the endpoint
    // alone is what makes the failure diagnosable.
    const { port, pathPrefix } = await startAgainstDenyingUpstream({
      oneCli: true,
      tier: 'untrusted',
    });

    await makeRequest(
      port,
      {
        method: 'POST',
        path: `${pathPrefix}/v1/messages?beta=true&trace=super-secret-value`,
        headers: {
          'content-type': 'application/json',
          authorization: 'Bearer placeholder',
        },
      },
      '{}',
    );

    const denial = vi
      .mocked(logger.error)
      .mock.calls.filter(([, msg]) =>
        String(msg).includes('OneCLI gateway denied'),
      );
    expect(denial).toHaveLength(1);
    expect(denial[0][0]).toMatchObject({ url: '/v1/messages' });
    expect(JSON.stringify(denial[0])).not.toContain('super-secret-value');
  });

  it('raises one operator alert once a tier reaches the denial streak', async () => {
    const alerts: string[] = [];
    const { port, pathPrefix } = await startAgainstDenyingUpstream({
      oneCli: true,
      tier: 'untrusted',
      onTierDenialAlert: (text) => alerts.push(text),
    });

    for (let i = 0; i < DENIAL_ALERT_STREAK; i++) {
      await denyingRequest(port, pathPrefix);
    }
    expect(alerts).toHaveLength(1);
    expect(alerts[0]).toContain('untrusted');
    expect(alerts[0]).toContain('nanoclaw-untrusted');

    // The cooldown holds for every further denial in this run, so a
    // broken tier can't turn one incident into a message per request.
    for (let i = 0; i < DENIAL_ALERT_STREAK * 2; i++) {
      await denyingRequest(port, pathPrefix);
    }
    expect(alerts).toHaveLength(1);
  });

  it('does not let a 5xx between denials clear the streak', async () => {
    // A 429 or 5xx says nothing about whether the credential is
    // usable. Treating one as a success would let an upstream hiccup
    // interleaved with real denials hide a tier that cannot
    // authenticate at all — exactly the silence #893 is about.
    const alerts: string[] = [];
    const { port, pathPrefix } = await startAgainstDenyingUpstream({
      oneCli: true,
      tier: 'untrusted',
      onTierDenialAlert: (text) => alerts.push(text),
    });

    for (let i = 0; i < DENIAL_ALERT_STREAK - 1; i++) {
      await denyingRequest(port, pathPrefix);
    }
    denyStatus = 503;
    await denyingRequest(port, pathPrefix);
    denyStatus = 401;
    await denyingRequest(port, pathPrefix);

    expect(alerts).toHaveLength(1);
  });

  it('does not alert before the streak is reached', async () => {
    const alerts: string[] = [];
    const { port, pathPrefix } = await startAgainstDenyingUpstream({
      oneCli: true,
      tier: 'untrusted',
      onTierDenialAlert: (text) => alerts.push(text),
    });

    for (let i = 0; i < DENIAL_ALERT_STREAK - 1; i++) {
      await denyingRequest(port, pathPrefix);
    }
    expect(alerts).toHaveLength(0);
  });

  it('logs the denial even with no alert sink wired', async () => {
    // The log line is the durable record and must not depend on a
    // caller having somewhere to deliver a chat message.
    const { port, pathPrefix } = await startAgainstDenyingUpstream({
      oneCli: true,
      tier: 'untrusted',
    });

    for (let i = 0; i < DENIAL_ALERT_STREAK; i++) {
      await denyingRequest(port, pathPrefix);
    }

    expect(
      vi
        .mocked(logger.error)
        .mock.calls.filter(([, msg]) =>
          String(msg).includes('OneCLI gateway denied'),
        ),
    ).toHaveLength(DENIAL_ALERT_STREAK);
  });
});

describe('credential-proxy usage logging', () => {
  let proxyServer: http.Server;
  let upstreamServer: http.Server;
  let upstreamPort: number;
  let usageLogPath: string;
  let upstreamPathSeen = '';

  beforeEach(async () => {
    _resetRegistry();
    upstreamPathSeen = '';
    const dir = await fsp.mkdtemp(join(os.tmpdir(), 'usage-proxy-test-'));
    usageLogPath = join(dir, 'usage.jsonl');

    upstreamServer = http.createServer((req, res) => {
      upstreamPathSeen = req.url || '';
      // Mock Anthropic /v1/messages JSON response (non-streaming).
      // Match with-or-without query string — the SDK adds `?beta=true`.
      if ((req.url || '').split('?')[0] === '/v1/messages') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(
          JSON.stringify({
            id: 'msg_01ES15ssVXAyQm25E2yBGur3',
            model: 'claude-sonnet-4-6',
            usage: {
              input_tokens: 1,
              output_tokens: 52,
              cache_read_input_tokens: 112865,
              cache_creation_input_tokens: 585,
              cache_creation: {
                ephemeral_5m_input_tokens: 585,
                ephemeral_1h_input_tokens: 0,
              },
            },
          }),
        );
      } else {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end('{}');
      }
    });
    await new Promise<void>((r) => upstreamServer.listen(0, '127.0.0.1', r));
    upstreamPort = (upstreamServer.address() as AddressInfo).port;
  });

  afterEach(async () => {
    await new Promise<void>((r) => proxyServer?.close(() => r()));
    await new Promise<void>((r) => upstreamServer?.close(() => r()));
    for (const k of Object.keys(mockEnv)) delete mockEnv[k];
    delete process.env.USAGE_LOG_PATH;
  });

  async function startProxy(): Promise<number> {
    Object.assign(mockEnv, { ANTHROPIC_API_KEY: 'sk-real' });
    process.env.USAGE_LOG_PATH = usageLogPath;
    proxyServer = await startCredentialProxy(0, '127.0.0.1', {
      upstreamUrl: new URL(`http://127.0.0.1:${upstreamPort}`),
    });
    return (proxyServer.address() as AddressInfo).port;
  }

  it('writes a complete JSONL line for /v1/messages with attribution', async () => {
    const proxyPort = await startProxy();
    const token = registerContainer({
      group: 'telegram_main',
      tier: 'main',
      session: 'default',
      task_id: null,
      message_id: '12345',
    });

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: `/c/${token}/v1/messages`,
        headers: { 'content-type': 'application/json' },
      },
      JSON.stringify({ model: 'claude-sonnet-4-6', messages: [] }),
    );

    // The proxy strips the /c/<token>/ prefix before forwarding upstream.
    expect(upstreamPathSeen).toBe('/v1/messages');

    // Poll until the JSONL line lands rather than sleep a fixed window —
    // the write is fire-and-forget after the response ends.
    const lines = await waitForUsageLines(usageLogPath, 1);
    expect(lines).toHaveLength(1);
    const rec = JSON.parse(lines[0]);
    expect(rec).toMatchObject({
      group: 'telegram_main',
      tier: 'main',
      session: 'default',
      task_id: null,
      // #479 sub-#1: trigger message_id flows into the JSONL line so
      // cost reports can break down spend per inbound message.
      message_id: '12345',
      model: 'claude-sonnet-4-6',
      api_id: 'msg_01ES15ssVXAyQm25E2yBGur3',
      in: 1,
      out: 52,
      cache_r: 112865,
      cache_c_5m: 585,
      cache_c_1h: 0,
      cost_micro: 3683625,
    });
    expect(typeof rec.ts).toBe('string');
    expect(typeof rec.dur_ms).toBe('number');
  });

  it('captures usage when the SDK appends ?beta=true to /v1/messages', async () => {
    // Regression: the Claude SDK posts to `/v1/messages?beta=true`, not the
    // bare `/v1/messages`. Strict equality on the path silently disabled
    // capture for every real container call.
    const proxyPort = await startProxy();
    const token = registerContainer({
      group: 'telegram_main',
      tier: 'main',
      session: 'default',
      task_id: null,
      message_id: null,
    });

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: `/c/${token}/v1/messages?beta=true`,
        headers: { 'content-type': 'application/json' },
      },
      JSON.stringify({ model: 'claude-sonnet-4-6', messages: [] }),
    );

    expect(upstreamPathSeen).toBe('/v1/messages?beta=true');

    const lines = await waitForUsageLines(usageLogPath, 1);
    expect(lines).toHaveLength(1);
    const rec = JSON.parse(lines[0]);
    expect(rec.group).toBe('telegram_main');
    expect(rec.in).toBe(1);
    expect(rec.cache_r).toBe(112865);
  });

  it('records group: "unknown" when token is missing', async () => {
    const proxyPort = await startProxy();

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: '/v1/messages',
        headers: { 'content-type': 'application/json' },
      },
      JSON.stringify({ model: 'claude-sonnet-4-6', messages: [] }),
    );

    const lines = await waitForUsageLines(usageLogPath, 1);
    const rec = JSON.parse(lines[0]);
    expect(rec.group).toBe('unknown');
  });

  it('decodes gzip-encoded responses before parsing usage', async () => {
    // Anthropic's edge serves gzip when the SDK sends
    // accept-encoding: gzip. The proxy must decode the captured copy
    // before parsing — otherwise bodyText is binary garbage and the
    // parser silently returns null. This is the actual production
    // bug behind the post-#487 silent-zero-guard alarms: 37 of 37
    // requests had compressed bodies and zero JSONL lines landed.
    const { gzipSync } = await import('zlib');

    // Spin up a one-off upstream that returns gzip-compressed JSON
    // with the expected Content-Encoding header.
    const gzipUpstream = http.createServer((_req, res) => {
      const body = gzipSync(
        Buffer.from(
          JSON.stringify({
            id: 'msg_gz',
            model: 'claude-sonnet-4-6',
            usage: { input_tokens: 11, output_tokens: 22 },
          }),
          'utf8',
        ),
      );
      res.writeHead(200, {
        'content-type': 'application/json',
        'content-encoding': 'gzip',
        'content-length': String(body.length),
      });
      res.end(body);
    });
    await new Promise<void>((r) => gzipUpstream.listen(0, '127.0.0.1', r));
    const gzipPort = (gzipUpstream.address() as AddressInfo).port;

    Object.assign(mockEnv, { ANTHROPIC_API_KEY: 'sk-real' });
    process.env.USAGE_LOG_PATH = usageLogPath;
    proxyServer = await startCredentialProxy(0, '127.0.0.1', {
      upstreamUrl: new URL(`http://127.0.0.1:${gzipPort}`),
    });
    const proxyPort = (proxyServer.address() as AddressInfo).port;

    const token = registerContainer({
      group: 'telegram_main',
      tier: 'main',
      session: 'default',
      task_id: null,
      message_id: null,
    });

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: `/c/${token}/v1/messages`,
        headers: { 'content-type': 'application/json' },
      },
      JSON.stringify({ model: 'claude-sonnet-4-6', messages: [] }),
    );

    const lines = await waitForUsageLines(usageLogPath, 1);
    const rec = JSON.parse(lines[0]);
    expect(rec.api_id).toBe('msg_gz');
    expect(rec.in).toBe(11);
    expect(rec.out).toBe(22);

    await new Promise<void>((r) => gzipUpstream.close(() => r()));
  });

  it('does NOT write usage for non-/v1/messages requests', async () => {
    const proxyPort = await startProxy();
    const token = registerContainer({
      group: 'g',
      tier: 'main',
      session: 's',
      task_id: null,
      message_id: null,
    });

    await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: `/c/${token}/api/oauth/claude_cli/create_api_key`,
        headers: { 'content-type': 'application/json' },
      },
      '{}',
    );

    // Non-/v1/messages requests don't trigger captureUsage, so no
    // async file write is scheduled — checking immediately after the
    // response cycle ends is deterministic. (No sleep needed: if the
    // proxy were to write, the work would be queued by the time
    // makeRequest returned, since `noteCaptureWrite` runs in the
    // upstream `end` handler that fires before the client `end`.)
    const exists = await fsp
      .access(usageLogPath)
      .then(() => true)
      .catch(() => false);
    expect(exists).toBe(false);
  });

  it('still forwards the response when the usage log write fails', async () => {
    // Point USAGE_LOG_PATH at a path under a regular file → mkdir fails.
    const dir = await fsp.mkdtemp(join(os.tmpdir(), 'usage-fail-test-'));
    const blocker = join(dir, 'blocker');
    await fsp.writeFile(blocker, 'x');
    process.env.USAGE_LOG_PATH = join(blocker, 'usage.jsonl');

    Object.assign(mockEnv, { ANTHROPIC_API_KEY: 'sk-real' });
    proxyServer = await startCredentialProxy(0, '127.0.0.1', {
      upstreamUrl: new URL(`http://127.0.0.1:${upstreamPort}`),
    });
    const proxyPort = (proxyServer.address() as AddressInfo).port;

    const res = await makeRequest(
      proxyPort,
      {
        method: 'POST',
        path: '/v1/messages',
        headers: { 'content-type': 'application/json' },
      },
      JSON.stringify({ model: 'claude-sonnet-4-6', messages: [] }),
    );

    // Response stream MUST still complete with the upstream body.
    expect(res.statusCode).toBe(200);
    const parsed = JSON.parse(res.body);
    expect(parsed.id).toBe('msg_01ES15ssVXAyQm25E2yBGur3');
  });
});
