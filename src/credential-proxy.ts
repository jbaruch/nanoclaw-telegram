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
import { brotliDecompressSync, gunzipSync, inflateSync } from 'zlib';

import { readEnvFile } from './env.js';
import { logger } from './logger.js';
import { lookupContainer } from './proxy-registry.js';
import {
  appendUsageRecord,
  noteCaptureWrite,
  noteMessagesRequest,
  parseUsageFromBody,
  resolveUsageLogPath,
  type ContainerContext,
} from './usage-log.js';
import { applyPromptCacheTtl } from './prompt-cache-ttl.js';
import { applyWireToolFilter, isMessagesEndpoint } from './wire-tool-filter.js';

export type AuthMode = 'api-key' | 'oauth';

export interface ProxyConfig {
  authMode: AuthMode;
}

/**
 * Path-prefix used to embed a per-container attribution token in the
 * proxy URL. Containers receive `ANTHROPIC_BASE_URL=http://gw:port/c/<token>`;
 * the SDK joins endpoint paths so requests arrive as `/c/<token>/v1/messages`.
 * The proxy strips the prefix, looks up `{group, tier, session, task_id}`
 * via `proxy-registry`, and forwards the un-prefixed path upstream.
 */
const TOKEN_PREFIX_RE = /^\/c\/([A-Za-z0-9_-]+)(\/.*)?$/;

/**
 * Cap the response buffer at 10MB. Typical Anthropic responses are
 * well under 200KB; if we ever see something larger, skip usage
 * capture rather than risk holding huge buffers in memory. The
 * response stream itself is unaffected — we only stop teeing into
 * the buffer.
 */
const USAGE_CAPTURE_BUFFER_CAP = 10 * 1024 * 1024;

/**
 * Orchestrator-side bypass for the LiteLLM router (#610). When
 * `ANTHROPIC_BASE_URL` points at `nanoclaw-litellm` and the primary
 * attempt fails on a recoverable shape — the LiteLLM container is
 * unreachable (ECONNREFUSED / ENOTFOUND), the primary stalls without
 * a response for `BYPASS_IDLE_TIMEOUT_MS`, or LiteLLM itself returns
 * a 5xx — the proxy retries against `ANTHROPIC_BYPASS_URL` (default
 * `https://api.anthropic.com`) using the same `ANTHROPIC_API_KEY`.
 * Complementary to LiteLLM's router-level fallback in
 * `container/litellm/litellm.config.yaml`: that one fires when the
 * litellm.ai gateway responds with 5xx but nanoclaw-litellm itself
 * is up; this one fires when nanoclaw-litellm itself is down. Both
 * layers exist because either can fail independently.
 *
 * Bypass is automatically disabled when the primary and bypass URLs
 * resolve to the same origin — there's nothing to fall back TO if the
 * primary is already Anthropic-direct.
 */
const BYPASS_TRIGGER_STATUS_CODES = new Set([500, 502, 503, 504]);
const BYPASS_IDLE_TIMEOUT_MS = 3000;

export function startCredentialProxy(
  port: number,
  host = '127.0.0.1',
): Promise<Server> {
  const secrets = readEnvFile([
    'ANTHROPIC_API_KEY',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_BASE_URL',
    'ANTHROPIC_BYPASS_URL',
  ]);

  const authMode: AuthMode = secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
  const oauthToken =
    secrets.CLAUDE_CODE_OAUTH_TOKEN || secrets.ANTHROPIC_AUTH_TOKEN;

  const upstreamUrl = new URL(
    secrets.ANTHROPIC_BASE_URL || 'https://api.anthropic.com',
  );
  const bypassUrl = new URL(
    secrets.ANTHROPIC_BYPASS_URL || 'https://api.anthropic.com',
  );
  // Same-origin bypass is a no-op — skip the second attempt entirely
  // so we don't double-bill the same target on every error.
  const bypassEnabled =
    upstreamUrl.origin !== bypassUrl.origin && authMode === 'api-key';

  const usageLogPath = resolveUsageLogPath();

  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      // Detach the per-container attribution token from the URL prefix
      // (`/c/<token>/...`) before doing anything else. The upstream URL
      // is the un-prefixed path; the token feeds the usage-log lookup.
      // Requests without the prefix keep working with `group: "unknown"`
      // — the proxy is backward-compatible during rollout.
      let containerCtx: ContainerContext | null = null;
      let upstreamPath = req.url || '/';
      const m = upstreamPath.match(TOKEN_PREFIX_RE);
      if (m) {
        const token = m[1];
        upstreamPath = m[2] || '/';
        containerCtx = lookupContainer(token);
        if (!containerCtx) {
          logger.warn(
            { token: token.slice(0, 6) + '…', url: req.url },
            'Credential proxy: unknown attribution token, recording as unknown',
          );
        }
      }

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
          upstreamPath,
          req.method,
          rawBody,
          process.env,
          (err) => {
            logger.warn(
              { err, url: upstreamPath },
              'Wire-tool filter: body parse failed, forwarding unchanged',
            );
          },
        );
        let body = filterResult.body;
        if (filterResult.applied) {
          // Logged at DEBUG, not INFO. The SDK ships the full catalog
          // on every /v1/messages, so an INFO line per request would
          // dominate the log stream in steady state. The interceptor's
          // effect is observable via DUMP_API_REQUESTS (#467) when an
          // operator wants per-request visibility.
          logger.debug(
            {
              url: upstreamPath,
              toolsStripped: filterResult.stats.toolsStripped,
              descriptionsTrimmed: filterResult.stats.descriptionsTrimmed,
              bodyDelta: body.length - originalLength,
            },
            'Wire-tool interceptor active',
          );
        }

        // Prompt-cache TTL extension (#537): for configured idle-prone
        // main DM containers, upgrade SDK-emitted ephemeral cache
        // breakpoints to the 1h tier. Runs after tool filtering so the
        // forwarded body, request dumps, and usage accounting all agree.
        const ttlResult = applyPromptCacheTtl(
          upstreamPath,
          req.method,
          body,
          containerCtx,
          process.env,
          (err) => {
            logger.warn(
              { err, url: upstreamPath, group: containerCtx?.group },
              'Prompt-cache TTL rewrite: body parse failed, forwarding unchanged',
            );
          },
        );
        body = ttlResult.body;
        if (ttlResult.applied) {
          // DEBUG, not INFO — this fires on every applicable
          // /v1/messages, which would dominate the log stream in
          // steady state. Matches the wire-tool interceptor's
          // log level above. Per-request visibility is available via
          // DUMP_API_REQUESTS (#467) when needed.
          logger.debug(
            {
              url: upstreamPath,
              group: containerCtx?.group,
              tier: containerCtx?.tier,
              session: containerCtx?.session,
              ttlApplied: ttlResult.stats.ttlApplied,
            },
            'Prompt-cache TTL rewrite active',
          );
        }

        // Optional request capture for prompt/tool-catalog inspection.
        // Set DUMP_API_REQUESTS=<dir> in the orchestrator env to enable.
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
            const urlPath = upstreamPath.split('?')[0];
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

        // Capture response body for usage logging (#479 / ligolnik#125)
        // only on /v1/messages — other endpoints (oauth exchange, health
        // checks) have no `usage` field. Match via `isMessagesEndpoint()`
        // (#479 sub-#4 — share one helper across the proxy so the next
        // `?beta=...` bump can't desync capture from the wire-tool filter
        // the way #126 caught). We tee the upstream stream into both the
        // client response and a buffer, parse usage after the upstream
        // ends, and append a JSONL line. Cap the buffer at 10MB; beyond
        // that we skip capture rather than blow memory. The response
        // stream itself is never gated on this — pipe is wired up first
        // and is never blocked or delayed by capture.
        const captureUsage =
          req.method === 'POST' && isMessagesEndpoint(upstreamPath);
        const requestStartMs = Date.now();
        // Sniff the request body for the model name so we can fall back
        // when the response is malformed / can't be parsed.
        let requestModel: string | null = null;
        if (captureUsage) {
          noteMessagesRequest();
          try {
            const parsed = JSON.parse(body.toString('utf8'));
            if (parsed && typeof parsed.model === 'string')
              requestModel = parsed.model;
          } catch {
            // Body isn't JSON — leave model null; the response usually
            // carries it anyway.
          }
        }

        // Single-attempt sender. Called once with the primary upstream;
        // re-called against `bypassUrl` if the primary trips one of
        // the bypass triggers (ECONNREFUSED / ENOTFOUND / idle timeout
        // / 5xx). `fromBypass=true` prevents infinite recursion — the
        // bypass attempt's own failure surfaces as a normal 502.
        const sendUpstreamRequest = (
          targetUrl: URL,
          fromBypass: boolean,
        ): void => {
          const isHttps = targetUrl.protocol === 'https:';
          const makeRequest = isHttps ? httpsRequest : httpRequest;

          const headers: Record<
            string,
            string | number | string[] | undefined
          > = {
            ...(req.headers as Record<string, string>),
            host: targetUrl.host,
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
            // OAuth mode: replace placeholder Bearer token with the real
            // one only when the container actually sends an Authorization
            // header (exchange request + auth probes). Post-exchange
            // requests use x-api-key only, so they pass through without
            // token injection.
            if (headers['authorization']) {
              delete headers['authorization'];
              if (oauthToken) {
                headers['authorization'] = `Bearer ${oauthToken}`;
              }
            }
          }

          const upstream = makeRequest(
            {
              hostname: targetUrl.hostname,
              port: targetUrl.port || (isHttps ? 443 : 80),
              path: upstreamPath,
              method: req.method,
              headers,
            } as RequestOptions,
            (upRes) => {
              // Bypass on a 5xx from the primary. The LiteLLM container
              // may be up but its own router-level fallback chain has
              // been exhausted; a fresh attempt against api.anthropic.com
              // is more likely to succeed than passing the 5xx through.
              // The bypass attempt's own 5xx is a real upstream failure
              // and gets surfaced normally to the client.
              if (
                !fromBypass &&
                bypassEnabled &&
                BYPASS_TRIGGER_STATUS_CODES.has(upRes.statusCode || 0)
              ) {
                logger.warn(
                  {
                    statusCode: upRes.statusCode,
                    url: upstreamPath,
                    primary: targetUrl.origin,
                    bypass: bypassUrl.origin,
                  },
                  'credential-proxy: primary returned 5xx, bypassing to anthropic-direct',
                );
                // Drain the failed primary response so the socket can
                // return to the pool / close cleanly before retry.
                upRes.resume();
                sendUpstreamRequest(bypassUrl, true);
                return;
              }

              res.writeHead(upRes.statusCode!, upRes.headers);

              if (!captureUsage || upRes.statusCode !== 200) {
                upRes.pipe(res);
                return;
              }

              // Tee: forward chunks to the client AND collect them for
              // usage parsing. We attach the data/end listeners directly
              // on `upRes` (instead of `upRes.pipe(res)` + a separate
              // capture-only data listener) because the pipe-then-attach
              // shape silently dropped capture in production: 37 of 37
              // /v1/messages requests through the post-#487 deploy
              // tripped sub-#3's silent-zero guard. The exact failure
              // mode (listener-attach race vs. pipe consuming chunks
              // before the late listener subscribed) wasn't pinpointed,
              // but reverting to the explicit-tee shape from
              // ligolnik#125 — augmented with explicit pause/resume
              // for backpressure — restores capture and addresses the
              // OpenAI reviewer's original concern about pipe's
              // implicit backpressure being lost. `res.write()` returns
              // false when its buffer is full; we pause `upRes` and
              // resume on the client's `drain` event. This is the
              // mechanism `pipe` uses internally; doing it explicitly
              // keeps the data listener on the same flow.
              const captured: Buffer[] = [];
              let captureSize = 0;
              let capped = false;
              // Backpressure: track whether upRes was paused so resume
              // only fires once per drain. Using a closure flag rather
              // than upRes.isPaused() because the latter is also true
              // briefly after construction.
              let upPaused = false;
              const resumeUpstream = () => {
                if (upPaused) {
                  upPaused = false;
                  upRes.resume();
                }
              };
              // Client-disconnect handler: if the client aborts mid-stream,
              // tear down the upstream so we don't keep consuming data,
              // and drop our listeners so subsequent res.write/end calls
              // can't crash on a destroyed socket.
              const onClientClose = () => {
                if (!upRes.destroyed) upRes.destroy();
                res.removeListener('drain', resumeUpstream);
              };
              res.on('drain', resumeUpstream);
              res.on('close', onClientClose);
              // Helper: guard res.write so an already-ended/destroyed
              // client can't crash the proxy. Returns true on successful
              // write, false otherwise (so the caller can stop capturing).
              // Catches only the specific Node stream errors that
              // res.write() can throw on a torn-down socket — write
              // races between our writableEnded/destroyed check and
              // the write call itself. Any other exception is a
              // programming bug and propagates per
              // error-handling.Specific Exceptions.
              const STREAM_TEARDOWN_CODES = new Set([
                'ERR_STREAM_WRITE_AFTER_END',
                'ERR_STREAM_DESTROYED',
                'ERR_STREAM_ALREADY_FINISHED',
              ]);
              const safeWrite = (chunk: Buffer): boolean => {
                if (res.writableEnded || res.destroyed) return false;
                try {
                  return res.write(chunk);
                } catch (err) {
                  const code =
                    err instanceof Error && 'code' in err
                      ? (err as NodeJS.ErrnoException).code
                      : undefined;
                  if (code && STREAM_TEARDOWN_CODES.has(code)) return false;
                  throw err;
                }
              };
              upRes.on('data', (chunk: Buffer) => {
                const writeOk = safeWrite(chunk);
                if (!writeOk && !upPaused) {
                  upPaused = true;
                  upRes.pause();
                }
                if (capped) return;
                captureSize += chunk.length;
                if (captureSize > USAGE_CAPTURE_BUFFER_CAP) {
                  capped = true;
                  captured.length = 0;
                  logger.warn(
                    { url: upstreamPath, size: captureSize },
                    'usage-log: response too large, skipping capture',
                  );
                } else {
                  captured.push(chunk);
                }
              });
              upRes.on('end', () => {
                if (!res.writableEnded && !res.destroyed) res.end();
                res.removeListener('drain', resumeUpstream);
                res.removeListener('close', onClientClose);
                if (capped) return;
                const rawBuffer = Buffer.concat(captured);
                // The Claude SDK (undici) sends `accept-encoding: gzip,
                // deflate, br` by default, so Anthropic's edge serves
                // compressed responses. The proxy forwards the raw
                // (encoded) bytes to the client so the SDK can
                // decompress them transparently — but for capture we
                // need the decompressed text to parse SSE / JSON. Read
                // the response Content-Encoding and decode accordingly;
                // identity (or unset) keeps the raw bytes. Decompression
                // failures fall through to the raw buffer with a warn
                // log so downstream parsing still gets a chance.
                const encoding = (
                  upRes.headers['content-encoding'] || ''
                ).toLowerCase();
                let bodyBuffer = rawBuffer;
                if (
                  encoding === 'gzip' ||
                  encoding === 'br' ||
                  encoding === 'deflate'
                ) {
                  try {
                    if (encoding === 'gzip') bodyBuffer = gunzipSync(rawBuffer);
                    else if (encoding === 'br')
                      bodyBuffer = brotliDecompressSync(rawBuffer);
                    else bodyBuffer = inflateSync(rawBuffer);
                  } catch (err) {
                    logger.warn(
                      {
                        err,
                        url: upstreamPath,
                        encoding,
                        rawBytes: rawBuffer.length,
                      },
                      'usage-log: response decompression failed, falling back to raw buffer for parse',
                    );
                  }
                }
                const bodyText = bodyBuffer.toString('utf8');
                const ctx: ContainerContext = containerCtx ?? {
                  group: 'unknown',
                  tier: 'untrusted',
                  session: 'unknown',
                  task_id: null,
                  message_id: null,
                };
                try {
                  const record = parseUsageFromBody(
                    bodyText,
                    ctx,
                    Date.now() - requestStartMs,
                    requestModel,
                  );
                  if (record) {
                    noteCaptureWrite();
                    // Fire-and-forget. appendUsageRecord swallows IO
                    // errors internally so this can never reject.
                    void appendUsageRecord(usageLogPath, record);
                  }
                } catch (err) {
                  // Defense in depth: parseUsageFromBody is designed not
                  // to throw, but if it ever does, we MUST NOT propagate.
                  logger.warn(
                    { err, url: upstreamPath },
                    'usage-log: parse failed',
                  );
                }
              });
              upRes.on('error', (err) => {
                logger.warn(
                  { err, url: upstreamPath },
                  'usage-log: upstream stream error during capture',
                );
                res.removeListener('drain', resumeUpstream);
                res.removeListener('close', onClientClose);
                if (!res.writableEnded && !res.destroyed) res.end();
              });
            },
          );

          // Idle-timeout-driven bypass. `setTimeout` fires when no
          // socket activity has happened for `BYPASS_IDLE_TIMEOUT_MS`
          // — covers the LiteLLM-reachable-but-not-responding case
          // (TCP accepted, no HTTP response in 3s). Cleared the moment
          // response headers arrive so a slow-streaming SDK response
          // isn't torn down mid-flight; the `'timeout'` event without
          // a handler is silent, so a `destroy()` is explicit.
          upstream.setTimeout(BYPASS_IDLE_TIMEOUT_MS, () => {
            upstream.destroy(new Error('credential-proxy idle timeout'));
          });
          upstream.once('response', () => upstream.setTimeout(0));

          upstream.on('error', (err) => {
            const code = (err as NodeJS.ErrnoException).code;
            const isReachabilityError =
              code === 'ECONNREFUSED' ||
              code === 'ENOTFOUND' ||
              code === 'EHOSTUNREACH' ||
              code === 'ECONNRESET' ||
              err.message === 'credential-proxy idle timeout';
            if (!fromBypass && bypassEnabled && isReachabilityError) {
              logger.warn(
                {
                  err: err.message,
                  code,
                  url: upstreamPath,
                  primary: targetUrl.origin,
                  bypass: bypassUrl.origin,
                },
                'credential-proxy: primary unreachable, bypassing to anthropic-direct',
              );
              sendUpstreamRequest(bypassUrl, true);
              return;
            }
            logger.error(
              { err, url: upstreamPath, fromBypass },
              'Credential proxy upstream error',
            );
            if (!res.headersSent) {
              res.writeHead(502);
              res.end('Bad Gateway');
            }
          });

          upstream.write(body);
          upstream.end();
        };

        sendUpstreamRequest(upstreamUrl, false);
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
