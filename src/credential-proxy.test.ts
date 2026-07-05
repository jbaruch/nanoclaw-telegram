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

import { startCredentialProxy } from './credential-proxy.js';
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
    } catch {
      // File doesn't exist yet — keep polling.
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
    const gzipUpstream = http.createServer((req, res) => {
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
