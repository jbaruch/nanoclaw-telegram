import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';

import { HOST_PROJECT_ROOT, STORE_DIR } from '../config.js';

import {
  buildSnitchmdFlags,
  formatSnitchmdHeader,
  parseFetchMarkdownUrl,
  parseSnitchmdStdout,
  redactUrlsInText,
  scrubFetchedUrl,
} from '../fetch-markdown-args.js';
import { registerIpcHandler, scriptResultPath } from '../ipc-registry.js';
import { logger } from '../logger.js';
import { runSidecar } from '../sidecar-runner.js';

/**
 * External-process delegation (#879, split out of `ops.ts`): the two
 * commands that hand work to a sibling container — `fetch_markdown` via
 * the snitchmd container, and `run_sidecar` via the named-sidecar
 * registry.
 */
export function registerOpsFetchIpcHandlers(): void {
  registerIpcHandler('fetch_markdown', {
    handler: ({ data, sourceGroup }) => {
      if (data.requestId) {
        const resultPath = scriptResultPath(sourceGroup, data);

        const parsed = parseFetchMarkdownUrl(data.url);
        if (!parsed.ok) {
          fs.writeFileSync(resultPath, JSON.stringify({ error: parsed.error }));
          return;
        }
        const parsedUrl = parsed.url;

        const flags = buildSnitchmdFlags(data);

        // Cache directory on the HOST filesystem — the snitchmd sibling
        // container is launched via the orchestrator's docker.sock, so
        // -v mount paths must reference the host's view (HOST_PROJECT_ROOT),
        // not the orchestrator's /app/store. Same convention as the agent
        // container mounts in container-runner.ts.
        const hostCacheDir = path.join(
          HOST_PROJECT_ROOT,
          'store',
          'snitchmd-cache',
        );
        // Also mkdir locally so the path is visible to the orchestrator
        // process (e.g. for size-rollup or cleanup tooling). The actual
        // write into the cache is performed by the sibling container as
        // its own user; we just want the directory to exist.
        fs.mkdirSync(path.join(STORE_DIR, 'snitchmd-cache'), {
          recursive: true,
        });

        // snitchmd is an app-level dependency (a renderer + content
        // extractor), not an API contract — we WANT to ride the latest
        // CloakBrowser fingerprint updates as anti-bot detection
        // evolves, and snitchmd's output shape is stable across point
        // releases. Operators who need a reproducible build can pin to
        // a specific tag or `sha256:…` digest via `SNITCHMD_IMAGE`
        // without a code change.
        //
        // The floating default takes `coding-policy:
        // dependency-management`'s Adversarial-Freshness Dependency
        // Carve-Out; authority-of-record is
        // `nanoclaw-host: snitchmd-image-floating`, enforced at deploy
        // time by `verify_snitchmd_image_floating` in
        // `scripts/deploy.sh` (step 3b-bis), which fails the deploy if
        // this default is ever pinned.
        const snitchmdImage =
          process.env.SNITCHMD_IMAGE || 'syabro/snitchmd:latest';

        logger.info(
          {
            sourceGroup,
            host: parsedUrl.host,
            flags: flags.filter((f) => !f.startsWith('http')),
          },
          'Running fetch_markdown',
        );

        execFile(
          'docker',
          [
            'run',
            '--rm',
            '-v',
            `${hostCacheDir}:/cache`,
            snitchmdImage,
            parsedUrl.toString(),
            ...flags,
          ],
          {
            // First call cold-pulls the image — give it room. Subsequent
            // calls are cached and complete in well under 30s.
            timeout: 240_000,
            // snitchmd's markdown output can run into the megabytes for
            // wiki/long articles; cap at 16 MiB so a runaway page doesn't
            // exhaust orchestrator memory before the agent's --max-chars
            // truncation kicks in.
            maxBuffer: 16 * 1024 * 1024,
          },
          (error, stdout, stderr) => {
            if (error) {
              // execFile attaches `code` (numeric exit code or string like
              // ETIMEDOUT) and `killed` (timeout signal). Surface both so
              // runHostOperation's diagnostic relay can show the agent
              // which failure mode hit (cf. #146 retry context).
              const execErr = error as NodeJS.ErrnoException & {
                code?: number | string;
                killed?: boolean;
              };
              // Per `jbaruch/coding-policy: no-secrets`: Node's `execFile`
              // packs the full command line — including the target URL —
              // into `error.message` as `Command failed: docker run ...
              // <url> --json ...`. A URL with query-string auth (session
              // token, signed-URL signature) would leak into both the
              // structured log AND the result-file the agent reads. Emit
              // a fixed-shape message that names only the host + exit
              // mode; the host attribute itself isn't a secret and the
              // operator can correlate with the structured log.
              const safeError = `fetch_markdown failed for ${parsedUrl.host} (exit_code: ${execErr.code ?? 'unknown'}${execErr.killed ? ', killed' : ''})`;
              // Defense-in-depth: snitchmd's stderr is normally just
              // `snitchmd: title=... quality=... chars=...`, but a
              // Playwright / Chromium fault could echo the input URL.
              // Scrub the exact URL we passed in before writing so a
              // query-string auth secret can't leak via stderr.
              const urlString = parsedUrl.toString();
              const safeStderr = scrubFetchedUrl(
                stderr.slice(-2000),
                urlString,
              );
              logger.warn(
                {
                  sourceGroup,
                  host: parsedUrl.host,
                  exitCode: execErr.code,
                  killed: execErr.killed,
                  stderr: safeStderr.slice(-500),
                },
                'fetch_markdown failed',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: safeError,
                  exit_code: execErr.code,
                  killed: execErr.killed,
                  stderr: safeStderr,
                }),
              );
              return;
            }
            // snitchmd --json writes a single JSON object on stdout.
            // Production smoke-test caught CloakBrowser leaking
            // "Downloading newer chromium" log lines onto stdout BEFORE
            // the JSON on the cold-pull path, so we use a tolerant
            // parser that retries with a `{`-to-`}` slice when the
            // strict parse fails. The successful payload's markdown
            // body is passed up as plain text (the <untrusted-input>
            // envelope is applied client-side via the READ_TOOL_PATTERNS
            // row in untrusted-input-wrap.ts — #321).
            const parseResult = parseSnitchmdStdout(stdout);
            if (!parseResult.ok) {
              // Per `jbaruch/coding-policy: no-secrets`: snitchmd's
              // JSON payload always carries `url` and `final_url`
              // fields, and the raw bytes we couldn't parse may still
              // contain those substrings. Scrub the input URL and the
              // host stderr's same vector before persisting the
              // diagnostic so a query-string auth secret doesn't ride
              // the parse-failure path back to the agent.
              // stdout failed to PARSE, so `final_url` can't be extracted
              // and scrubbed by value — and a redirect can add a credential
              // the caller never sent. Redact every URL-shaped token, not
              // just the one we passed in.
              const urlString = parsedUrl.toString();
              const safeStderr = redactUrlsInText(
                scrubFetchedUrl(stderr.slice(-2000), urlString),
              );
              const safeStdout = redactUrlsInText(
                scrubFetchedUrl(stdout.slice(-2000), urlString),
              );
              logger.warn(
                {
                  sourceGroup,
                  host: parsedUrl.host,
                  stdoutLen: stdout.length,
                },
                'fetch_markdown: stdout did not parse as JSON',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: 'fetch_markdown: snitchmd produced non-JSON output',
                  stderr: safeStderr,
                  stdout: safeStdout,
                }),
              );
              return;
            }
            const payload = parseResult.payload;
            const markdown = payload.markdown ?? '';
            const header = formatSnitchmdHeader(payload, parsedUrl.toString());
            logger.info(
              {
                sourceGroup,
                host: parsedUrl.host,
                chars: payload.chars ?? markdown.length,
                quality: payload.quality,
              },
              'fetch_markdown completed',
            );
            // Scrub the input URL out of stderr on the SUCCESS path too, not
            // just the failure paths (`coding-policy: no-secrets`). snitchmd's
            // stderr is normally a benign `snitchmd: title=... chars=...`
            // line, but a Playwright/Chromium warning can echo the URL it was
            // given — and a fetch whose auth rides in the query string (signed
            // URL, session token) would then persist that secret into the
            // result envelope the agent reads. A successful fetch is exactly
            // the case where nobody thinks to look.
            const safeSuccessStderr = scrubFetchedUrl(
              stderr,
              parsedUrl.toString(),
            ).slice(-500);
            fs.writeFileSync(
              resultPath,
              JSON.stringify({
                stdout: header + markdown,
                stderr: safeSuccessStderr || undefined,
              }),
            );
          },
        );
      }
    },
  });

  registerIpcHandler('run_sidecar', {
    handler: ({ data, sourceGroup, isMain }) => {
      if (data.requestId) {
        // Authorization: run_sidecar spawns a privileged docker sidecar with
        // host bind-mounts defined in the trusted registry — same privilege
        // class as the former audible_backup case, gated on `isMain`. The
        // plugin supplies only the sidecar NAME and registry-allowlisted
        // flags; the image and mount paths never come from the payload, so a
        // compromised non-main container can't request `-v /:/…` (see
        // src/sidecar-runner.ts).
        if (!isMain) {
          logger.warn(
            { sourceGroup, name: data.name },
            'Unauthorized run_sidecar attempt',
          );
          return;
        }

        const sidecarResultPath = scriptResultPath(sourceGroup, data);

        // IPC payloads are raw JSON, so validate the shape at the boundary and
        // write an actionable error envelope rather than let a malformed
        // request (e.g. `flags: "--dry-run"`) crash runSidecar or silently
        // hang the polling MCP caller.
        if (typeof data.name !== 'string' || data.name.length === 0) {
          fs.writeFileSync(
            sidecarResultPath,
            JSON.stringify({
              error: 'run_sidecar: "name" must be a non-empty string.',
            }),
          );
          return;
        }
        if (
          data.flags !== undefined &&
          !(
            Array.isArray(data.flags) &&
            data.flags.every((f) => typeof f === 'string')
          )
        ) {
          fs.writeFileSync(
            sidecarResultPath,
            JSON.stringify({
              error: 'run_sidecar: "flags" must be an array of strings.',
            }),
          );
          return;
        }

        const sidecarName = data.name;
        const sidecarFlags = data.flags as string[] | undefined;

        // Fire-and-forget like github_backup / persist_global_file: a sidecar
        // run can take up to its registry timeout (600s for audible) and must
        // not block the IPC loop. runSidecar resolves an error envelope for
        // operational failures (unknown sidecar, disallowed flag, non-zero
        // docker exit); the #625 partial-success relay lives inside it.
        runSidecar({ name: sidecarName, flags: sidecarFlags })
          .then((payload) => {
            fs.writeFileSync(sidecarResultPath, JSON.stringify(payload));
          })
          // outer-boundary-process-contract: the MCP caller polls for the
          // result file and reads its absence as a silent hang until a 10-min
          // timeout. This catch emits an `{ error }` envelope for a genuine
          // relay bug (a non-SyntaxError stdout parse rejects runSidecar) so
          // the caller gets an actionable failure instead of hanging; letting
          // it propagate would write no file and break that contract.
          .catch((err: unknown) => {
            const message = err instanceof Error ? err.message : String(err);
            logger.error(
              { sourceGroup, name: sidecarName, error: message },
              'run_sidecar failed unexpectedly',
            );
            fs.writeFileSync(
              sidecarResultPath,
              JSON.stringify({
                error: `run_sidecar failed unexpectedly: ${message}`,
              }),
            );
          });
      }
    },
  });
}
