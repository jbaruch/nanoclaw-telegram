/**
 * Credential proxy for container isolation.
 * Containers connect here instead of directly to the Anthropic API.
 * The proxy injects real credentials so containers never see them.
 *
 * Two auth modes:
 *   API key:  Proxy injects x-api-key on every request.
 *   OAuth:    Container CLI exchanges its placeholder token for a temp
 *             API key via /api/oauth/claude_cli/create_api_key.
 *             Proxy injects real OAuth token on that exchange request;
 *             subsequent requests carry the temp key which is valid as-is.
 */
import { createServer, Server } from 'http';
import { request as httpsRequest } from 'https';
import { request as httpRequest, RequestOptions } from 'http';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

import { readEnvFile } from './env.js';
import { logger } from './logger.js';
import { applyWireToolFilter } from './wire-tool-filter.js';

export type AuthMode = 'api-key' | 'oauth';

export interface ProxyConfig {
  authMode: AuthMode;
}

export function startCredentialProxy(
  port: number,
  host = '127.0.0.1',
): Promise<Server> {
  const secrets = readEnvFile([
    'ANTHROPIC_API_KEY',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_BASE_URL',
  ]);

  const authMode: AuthMode = secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
  const oauthToken =
    secrets.CLAUDE_CODE_OAUTH_TOKEN || secrets.ANTHROPIC_AUTH_TOKEN;

  const upstreamUrl = new URL(
    secrets.ANTHROPIC_BASE_URL || 'https://api.anthropic.com',
  );
  const isHttps = upstreamUrl.protocol === 'https:';
  const makeRequest = isHttps ? httpsRequest : httpRequest;

  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', () => {
        const rawBody = Buffer.concat(chunks);

        // Wire-tool catalog interceptor (issue #119): strip SDK-builtin tools
        // we never use and trim the Bash description on outgoing /v1/messages
        // requests. Default ON; disable with STRIP_DEAD_TOOLS=0. Reduces
        // cache_create per cold start by ~12K tokens per tier.
        const originalLength = rawBody.length;
        const filterResult = applyWireToolFilter(
          req.url,
          req.method,
          rawBody,
          process.env,
          (err) => {
            logger.warn(
              { err, url: req.url },
              'Wire-tool filter: body parse failed, forwarding unchanged',
            );
          },
        );
        const body = filterResult.body;
        if (filterResult.applied) {
          // Logged at DEBUG, not INFO. The SDK ships the full catalog
          // on every /v1/messages, so an INFO line per request would
          // dominate the log stream in steady state. The interceptor's
          // effect is observable via DUMP_API_REQUESTS (#467) when an
          // operator wants per-request visibility.
          logger.debug(
            {
              url: req.url,
              toolsStripped: filterResult.stats.toolsStripped,
              descriptionsTrimmed: filterResult.stats.descriptionsTrimmed,
              bodyDelta: body.length - originalLength,
            },
            'Wire-tool interceptor active',
          );
        }

        // Default OFF; never enable in long-running production — the
        // captured request bodies contain user prompts and the SDK's
        // full system prompt + tool catalog, which is sensitive
        // operator material. Single-shot capture pattern: set, fire
        // one request, unset.
        //
        // Placed AFTER the wire-tool interceptor so the dumped body
        // reflects what was actually forwarded upstream (post-filter,
        // post-Bash-trim) — capture before the filter would dump the
        // pre-strip catalog and obscure the interceptor's effect.
        //
        // Failure handling per `error-handling.Specific Exceptions` +
        // `Graceful Fallback`: catch ONLY the known recoverable
        // `NodeJS.ErrnoException` codes that fs operations emit on
        // operator-misconfigured dump targets (EACCES, ENOSPC,
        // ENOENT, ENOTDIR, EPERM, EROFS, EISDIR). On those we log and
        // continue forwarding — the proxy's primary contract is to
        // forward the request; capture is strictly diagnostic. On any
        // OTHER exception (TypeError, ReferenceError, programming
        // bugs introduced by future edits), we rethrow so the bug
        // surfaces loudly instead of being silently swallowed by a
        // catch-all. Sync I/O is intentional: this is operator-toggled
        // debug, used in single-shot mode where dump-path latency is
        // acceptable.
        const dumpDir = process.env.DUMP_API_REQUESTS;
        if (dumpDir) {
          try {
            mkdirSync(dumpDir, { recursive: true });
            const ts = new Date().toISOString().replace(/[:.]/g, '-');
            // Strip query string and sanitize the filename component
            // to a conservative set so dumps glob predictably even when
            // the proxied URL carries `?foo=bar` or other unsafe chars.
            const urlPath = (req.url || '/').split('?')[0];
            const lastRaw = urlPath.split('/').pop() || 'root';
            const last = lastRaw.replace(/[^a-zA-Z0-9._-]/g, '_') || 'root';
            const method =
              (req.method || 'UNKNOWN').replace(/[^A-Z]/g, '') || 'UNKNOWN';
            writeFileSync(join(dumpDir, `${ts}-${method}-${last}.json`), body);
          } catch (err) {
            const RECOVERABLE_FS_CODES = new Set([
              'EACCES',
              'ENOSPC',
              'ENOENT',
              'ENOTDIR',
              'EPERM',
              'EROFS',
              'EISDIR',
            ]);
            const code =
              err instanceof Error && 'code' in err
                ? (err as NodeJS.ErrnoException).code
                : undefined;
            if (code && RECOVERABLE_FS_CODES.has(code)) {
              logger.warn(
                { code, dumpDir, err: (err as Error).message },
                'DUMP_API_REQUESTS write failed — continuing forward without capture',
              );
            } else {
              // Unexpected exception (programming bug, not an fs
              // condition the operator can fix). Surface loudly per
              // `error-handling.Specific Exceptions`.
              throw err;
            }
          }
        }

        const headers: Record<string, string | number | string[] | undefined> =
          {
            ...(req.headers as Record<string, string>),
            host: upstreamUrl.host,
            'content-length': body.length,
          };

        // Strip hop-by-hop headers that must not be forwarded by proxies
        delete headers['connection'];
        delete headers['keep-alive'];
        delete headers['transfer-encoding'];

        if (authMode === 'api-key') {
          // API key mode: inject x-api-key on every request
          delete headers['x-api-key'];
          headers['x-api-key'] = secrets.ANTHROPIC_API_KEY;
        } else {
          // OAuth mode: replace placeholder Bearer token with the real one
          // only when the container actually sends an Authorization header
          // (exchange request + auth probes). Post-exchange requests use
          // x-api-key only, so they pass through without token injection.
          if (headers['authorization']) {
            delete headers['authorization'];
            if (oauthToken) {
              headers['authorization'] = `Bearer ${oauthToken}`;
            }
          }
        }

        const upstream = makeRequest(
          {
            hostname: upstreamUrl.hostname,
            port: upstreamUrl.port || (isHttps ? 443 : 80),
            path: req.url,
            method: req.method,
            headers,
          } as RequestOptions,
          (upRes) => {
            res.writeHead(upRes.statusCode!, upRes.headers);
            upRes.pipe(res);
          },
        );

        upstream.on('error', (err) => {
          logger.error(
            { err, url: req.url },
            'Credential proxy upstream error',
          );
          if (!res.headersSent) {
            res.writeHead(502);
            res.end('Bad Gateway');
          }
        });

        upstream.write(body);
        upstream.end();
      });
    });

    server.listen(port, host, () => {
      logger.info({ port, host, authMode }, 'Credential proxy started');
      resolve(server);
    });

    server.on('error', reject);
  });
}

/** Detect which auth mode the host is configured for. */
export function detectAuthMode(): AuthMode {
  const secrets = readEnvFile(['ANTHROPIC_API_KEY']);
  return secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
}
