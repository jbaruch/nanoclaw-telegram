import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import http from 'http';
import type { AddressInfo } from 'net';
import fs from 'fs';
import os from 'os';
import path from 'path';

const mockEnv: Record<string, string> = {};
vi.mock('./env.js', () => ({
  readEnvFile: vi.fn(() => ({ ...mockEnv })),
}));

vi.mock('./logger.js', () => ({
  logger: { info: vi.fn(), error: vi.fn(), debug: vi.fn(), warn: vi.fn() },
}));

import { startCredentialProxy } from './credential-proxy.js';

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
    Object.assign(mockEnv, env, {
      ANTHROPIC_BASE_URL: `http://127.0.0.1:${upstreamPort}`,
    });
    proxyServer = await startCredentialProxy(0);
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

  describe('DUMP_API_REQUESTS', () => {
    let dumpDir: string;
    const originalEnv = process.env.DUMP_API_REQUESTS;

    beforeEach(() => {
      dumpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'dump-api-requests-'));
    });

    afterEach(() => {
      if (originalEnv === undefined) {
        delete process.env.DUMP_API_REQUESTS;
      } else {
        process.env.DUMP_API_REQUESTS = originalEnv;
      }
      // `force: true` already tolerates a missing path — no try/catch
      // needed, and a bare catch-all here would suppress legitimate
      // cleanup bugs per `error-handling.Specific Exceptions`.
      fs.rmSync(dumpDir, { recursive: true, force: true });
    });

    it('default OFF: env unset writes no files', async () => {
      delete process.env.DUMP_API_REQUESTS;
      proxyPort = await startProxy({ ANTHROPIC_API_KEY: 'sk-ant-real-key' });

      await makeRequest(
        proxyPort,
        {
          method: 'POST',
          path: '/v1/messages',
          headers: { 'content-type': 'application/json' },
        },
        '{"hello":"world"}',
      );

      expect(fs.readdirSync(dumpDir)).toHaveLength(0);
    });

    it('enabled: writes the request body to the dump dir', async () => {
      process.env.DUMP_API_REQUESTS = dumpDir;
      proxyPort = await startProxy({ ANTHROPIC_API_KEY: 'sk-ant-real-key' });

      const sentBody = '{"prompt":"capture me"}';
      await makeRequest(
        proxyPort,
        {
          method: 'POST',
          path: '/v1/messages',
          headers: { 'content-type': 'application/json' },
        },
        sentBody,
      );

      const files = fs.readdirSync(dumpDir);
      expect(files).toHaveLength(1);
      expect(files[0]).toMatch(/^.*-POST-messages\.json$/);
      const dumped = fs.readFileSync(path.join(dumpDir, files[0]), 'utf-8');
      expect(dumped).toBe(sentBody);
    });

    it('strips query string from filename (so dumps glob predictably)', async () => {
      process.env.DUMP_API_REQUESTS = dumpDir;
      proxyPort = await startProxy({ ANTHROPIC_API_KEY: 'sk-ant-real-key' });

      await makeRequest(
        proxyPort,
        {
          method: 'POST',
          path: '/v1/messages?stream=true',
          headers: { 'content-type': 'application/json' },
        },
        '{}',
      );

      const files = fs.readdirSync(dumpDir);
      expect(files).toHaveLength(1);
      expect(files[0]).toMatch(/^.*-POST-messages\.json$/);
      // Query-string chars must NOT leak into the filename.
      expect(files[0]).not.toContain('?');
      expect(files[0]).not.toContain('=');
    });

    it('degrades gracefully when the dump dir is unwritable: still forwards the request', async () => {
      // Point DUMP_API_REQUESTS at a path under a regular file — mkdir
      // recursive will fail because the parent isn't a directory. The
      // proxy must log a warn and forward the request anyway.
      const blockingFile = path.join(dumpDir, 'blocker');
      fs.writeFileSync(blockingFile, '');
      process.env.DUMP_API_REQUESTS = path.join(blockingFile, 'subdir');
      proxyPort = await startProxy({ ANTHROPIC_API_KEY: 'sk-ant-real-key' });

      const result = await makeRequest(
        proxyPort,
        {
          method: 'POST',
          path: '/v1/messages',
          headers: { 'content-type': 'application/json' },
        },
        '{}',
      );

      // Forwarding still succeeds — the blocker path can't be written
      // to, but the upstream request continues normally.
      expect(result.statusCode).toBe(200);
      expect(lastUpstreamHeaders['x-api-key']).toBe('sk-ant-real-key');
    });
  });

  it('returns 502 when upstream is unreachable', async () => {
    Object.assign(mockEnv, {
      ANTHROPIC_API_KEY: 'sk-ant-real-key',
      ANTHROPIC_BASE_URL: 'http://127.0.0.1:59999',
    });
    proxyServer = await startCredentialProxy(0);
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
});
