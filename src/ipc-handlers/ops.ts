import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';

import {
  DATA_DIR,
  GROUPS_DIR,
  HOST_PROJECT_ROOT,
  ORCHESTRATOR_REPO_URL,
  STORE_DIR,
} from '../config.js';

import { syncBackupRepo, type SyncResult } from '../backup-sync.js';
import {
  buildSnitchmdFlags,
  formatSnitchmdHeader,
  parseFetchMarkdownUrl,
  parseSnitchmdStdout,
} from '../fetch-markdown-args.js';
import { getInstalledTiles } from '../container-runner.js';
import { deleteAllSessions } from '../db-sessions.js';
import {
  applyTripitSegmentsToTzState,
  getActivePendingRunAtNames,
  getCurrentTz,
  type TripitSegment,
} from '../db-tz.js';
import {
  PERSISTABLE_GLOBAL_FILES,
  backupCommitAndPush,
  runPersonaPersistTask,
  runSerializedPersonaPersist,
  validateGlobalFilesToPersist,
} from '../git-persist.js';
import { registerIpcHandler, scriptResultPath } from '../ipc-registry.js';
import { logger } from '../logger.js';
import { runSidecar } from '../sidecar-runner.js';
import { recomputeLocalSchedules } from '../task-scheduler.js';

// Host-side allowlist for the five tile-repo names the promote flow is
// wired against. The MCP tools' zod enums (ipc-mcp-stdio.ts::TILE_NAMES)
// mirror this list client-side for a clean schema error at tool-call
// time, but the IPC handler is reachable by any payload dropped into
// the tasks dir — a compromised container could skip the MCP path and
// write `{tileName: "../../etc"}` directly, escaping GROUPS_DIR via
// `path.join` or pointing the bash scripts at an attacker-controlled
// git URL. This set is the actual trust boundary; keeping it in sync
// with the client-side enum is a release-hygiene concern.
const KNOWN_TILE_NAMES: ReadonlySet<string> = new Set([
  'nanoclaw-admin',
  'nanoclaw-core',
  'nanoclaw-untrusted',
  'nanoclaw-trusted',
  'nanoclaw-host',
]);

/**
 * Upper bound on the `persist_tz_segments` payload (#748 review). A real
 * itinerary is a handful of timezone segments; this is generous headroom that
 * still refuses a runaway/accidental payload before it bloats the
 * `tz_state.segments` DB row. Kept in lock-step with the `.max()` on the MCP
 * tool's zod schema in `container/agent-runner/src/ipc-mcp-stdio.ts`.
 */
export const MAX_TZ_SEGMENTS = 200;

/**
 * #748 — coerce the raw `segments` payload of a `persist_tz_segments` IPC call
 * to a `TripitSegment[]`. The value arrives as raw JSON off the IPC wire (the
 * in-container TripIt → Reclaim sync's parsed `result.segments`), so anything
 * that is not an array — `undefined`, `null`, an object, a string — returns an
 * empty array with `wasArray: false` rather than throwing. This is pure
 * normalization; it does NOT decide what to persist. The caller
 * (`classifyTzPersist`) uses `wasArray` to REJECT a non-array (a caller bug must
 * not clear owner `tz_state`) — only a genuine array, including an explicit
 * `[]`, is persisted. Element shapes are not deep-validated here:
 * `applyTripitSegmentsToTzState` / `walkTzSegments` already tolerate partial
 * per-field segment shapes (optional-with-fallback, #229).
 */
export function coerceTzSegments(raw: unknown): {
  segments: TripitSegment[];
  wasArray: boolean;
} {
  if (Array.isArray(raw)) {
    return { segments: raw as TripitSegment[], wasArray: true };
  }
  return { segments: [], wasArray: false };
}

/**
 * #748 — decide what the `persist_tz_segments` IPC handler does with a request,
 * as a pure function so the security-sensitive outcomes are unit-testable
 * without staging `processTaskIpc`:
 *
 *  - `deny`    — the caller is not the main container. The MCP tool is
 *                isMain-gated, but the IPC task-file path is reachable by ANY
 *                container, so this re-check (on the directory-verified
 *                `isMain`) is what actually blocks a non-main agent from
 *                poisoning owner `tz_state`.
 *  - `reject`  — a non-array payload, OR over `MAX_TZ_SEGMENTS`. Both refuse to
 *                write rather than mangle owner tz. A non-array is a caller bug,
 *                not a "no trips" signal: persisting an empty set would clear
 *                `tz_state.segments` and could flip the owner's timezone +
 *                recompute local schedules off a malformed call. The legacy
 *                the removed host-op skipped persistence on malformed stdout for
 *                the same reason; an explicit `[]` is the only way to clear.
 *                Over-cap is refused (not truncated — truncation would silently
 *                drop later trips).
 *  - `persist` — a genuine array within bounds (including an explicit `[]`,
 *                which correctly clears to the no-active-trips state).
 */
export type TzPersistDecision =
  | { action: 'deny'; error: string }
  | { action: 'reject'; error: string }
  | { action: 'persist'; segments: TripitSegment[] };

export function classifyTzPersist(
  isMain: boolean,
  raw: unknown,
): TzPersistDecision {
  if (!isMain) {
    return { action: 'deny', error: 'persist_tz_segments is main-group only' };
  }
  const { segments, wasArray } = coerceTzSegments(raw);
  if (!wasArray) {
    return {
      action: 'reject',
      error: 'segments must be an array (send [] to explicitly clear)',
    };
  }
  if (segments.length > MAX_TZ_SEGMENTS) {
    return {
      action: 'reject',
      error: `too many segments (${segments.length} > ${MAX_TZ_SEGMENTS})`,
    };
  }
  return { action: 'persist', segments };
}

/**
 * Named host operations (#845 slice 6): the ops surface — owner-tz
 * persistence, markdown fetching via the snitchmd sibling container,
 * the github backup + persona-persist git pipelines, sidecar runs, the
 * tile promote/fixup flows, and registry maintenance (tessl_update,
 * list_installed_tiles). Bodies are verbatim transplants from the
 * legacy `processTaskIpc` switch; each keeps its original in-handler
 * authorization gate (isMain re-check on the directory-verified
 * identity) and result-envelope conventions.
 */
export function registerOpsIpcHandlers(): void {
  registerIpcHandler('persist_tz_segments', {
    handler: ({ data, sourceGroup, isMain }) => {
      // #748 — credential-free host ingestion for the in-container TripIt →
      // Reclaim sync. The sync itself runs in the agent container now (creds
      // swapped at the OneCLI gateway), so the host no longer runs the CLI or
      // holds TripIt/Reclaim/Google secrets. What stays host-side is the
      // `tz_state` write: the container hands back the parsed `segments[]` and
      // the host persists them exactly as the removed `sync_tripit` host-op's
      // success path did — same `applyTripitSegmentsToTzState` + `#584`
      // `onTzFlipped`
      // next_run invalidation, so the scheduler-timezone skill, the 30-min
      // heartbeat advisory walker, and the #574 location cascade keep their
      // owner-tz backbone. The deny (non-main) / reject (over-cap) / persist
      // decision — the security-sensitive part — lives in the pure, tested
      // `classifyTzPersist`; this case just carries it out (log + result file +
      // DB write). See `classifyTzPersist` for the isMain-re-check and cap
      // rationale.
      const decision = classifyTzPersist(isMain, data.segments);
      if (decision.action === 'deny') {
        logger.warn(
          { sourceGroup },
          'Unauthorized persist_tz_segments attempt blocked (non-main container)',
        );
        if (data.requestId) {
          fs.writeFileSync(
            scriptResultPath(sourceGroup, data),
            JSON.stringify({ error: decision.error }),
          );
        }
        return;
      }
      if (data.requestId) {
        const resultPath = scriptResultPath(sourceGroup, data);
        if (decision.action === 'reject') {
          // A malformed (non-array) or over-cap payload — refuse rather than
          // clear/mangle owner tz_state. See `classifyTzPersist`.
          logger.warn(
            { sourceGroup, reason: decision.error },
            'persist_tz_segments: refused (not persisted)',
          );
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: decision.error }),
          );
          return;
        }
        // Failures from the persistence helper (SqliteError, programming
        // bugs) propagate per `coding-policy: error-handling`; only the
        // #584 recompute's transient SQLITE_BUSY/LOCKED is swallowed-with-warn
        // inside the writer, exactly as the removed host-op did.
        const { segments } = decision;
        applyTripitSegmentsToTzState({ segments }, new Date(), () => {
          recomputeLocalSchedules(getCurrentTz, new Date());
        });
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            stdout: JSON.stringify({ persisted: segments.length }),
          }),
        );
        logger.info(
          { sourceGroup, segmentCount: segments.length },
          'persist_tz_segments completed',
        );
      }
    },
  });

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
              const safeStderr = stderr
                .slice(-2000)
                .split(urlString)
                .join('<URL>');
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
              const urlString = parsedUrl.toString();
              const safeStderr = stderr
                .slice(-2000)
                .split(urlString)
                .join('<URL>');
              const safeStdout = stdout
                .slice(-2000)
                .split(urlString)
                .join('<URL>');
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
            fs.writeFileSync(
              resultPath,
              JSON.stringify({
                stdout: header + markdown,
                stderr: stderr.slice(-500) || undefined,
              }),
            );
          },
        );
      }
    },
  });

  registerIpcHandler('github_backup', {
    handler: async ({ data, sourceGroup, isMain }) => {
      if (data.requestId) {
        // Authorization: github_backup performs a host-side filesystem
        // sync + `git push` using GITHUB_TOKEN — same privilege class
        // as `audible_backup`, `promote_staging`, all
        // of which gate on `isMain`. Untrusted-tier groups have a
        // separate `#324` token gate at the container layer (the
        // confirmation-tokens hook), but the host-side gate here
        // shrinks the blast radius further: a compromised non-main
        // container can't trigger the backup pipeline by writing an
        // IPC task file directly.
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized github_backup attempt');
          return;
        }

        const backupDir = path.join(GROUPS_DIR, sourceGroup, 'backup-repo');
        const dbPath = path.join(STORE_DIR, 'messages.db');
        const resultPath = scriptResultPath(sourceGroup, data);

        // Sync live group state into backup-repo BEFORE git plumbing.
        // The sync mirrors groups/global/ and every non-hidden
        // groups/<name>/ subdir into backup-repo/{global,groups/<name>}
        // with delete-on-missing semantics, and dumps the SQLite
        // state-table surface into backup-repo/state/<table>.sql.
        // Policy: back up everything under groups/ that the runtime
        // mutates and that ISN'T reproducible from deploy or
        // `tessl install` — see `src/backup-sync.ts` for the denylist
        // (.tessl, .claude, node_modules, dist, logs, tmp,
        // conversations, *.bak-*, etc.).
        // syncBackupRepo also validates groupsRoot / backupDir
        // existence and throws an actionable error if either is
        // missing — the catch block below converts that into a
        // structured `{ error, stage: 'sync' }` envelope.
        let syncSummary: SyncResult;
        try {
          syncSummary = syncBackupRepo({
            groupsRoot: GROUPS_DIR,
            backupDir,
            dbPath,
          });
        } catch (e) {
          // Non-Error throws (TypeScript allows `throw 42`) bubble up
          // — those indicate a bug, not an operational sync failure.
          if (!(e instanceof Error)) throw e;
          logger.error(
            {
              sourceGroup,
              groupsRoot: GROUPS_DIR,
              backupDir,
              dbPath,
              error: e.message,
            },
            'github_backup sync failed',
          );
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: e.message, stage: 'sync' }),
          );
          return;
        }

        const commitMsg =
          data.message || `backup: ${new Date().toISOString().split('T')[0]}`;
        logger.info(
          {
            sourceGroup,
            backupDir,
            commitMsg,
            copied: syncSummary.copied.length,
            removed: syncSummary.removed.length,
            dumped: syncSummary.dumped.length,
            skipped: syncSummary.skipped.length,
          },
          'Running github_backup',
        );

        // Read GitHub token for push auth
        const { readEnvFile: readBackupEnv } = await import('../env.js');
        const backupEnvVars = readBackupEnv(['GITHUB_TOKEN']);
        const ghToken = backupEnvVars.GITHUB_TOKEN;

        // Run the commit+push off the IPC loop (like persist_global_file):
        // the promise resolves with the result envelope; the loop never
        // blocks on git. `backupCommitAndPush` is the testable git boundary
        // — argument-array git invocations, so the agent-supplied commit
        // message can't inject host commands (#725).
        backupCommitAndPush({
          backupDir,
          message: commitMsg,
          token: ghToken,
        }).then((backupResult) => {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ ...backupResult, sync_summary: syncSummary }),
          );
          if (backupResult.error) {
            logger.error(
              {
                sourceGroup,
                stage: backupResult.stage,
                error: backupResult.error,
              },
              'github_backup failed',
            );
          } else {
            logger.info(
              { sourceGroup, committed: backupResult.committed },
              'github_backup completed',
            );
          }
        });
      }
    },
  });

  registerIpcHandler('persist_global_file', {
    handler: async ({ data, sourceGroup, isMain }) => {
      if (data.requestId) {
        // Authorization: persist_global_file commits + pushes the
        // orchestrator repo's `main` using GITHUB_TOKEN — same privilege
        // class as `github_backup` / `promote_staging`, gated on `isMain`.
        // A non-main container can't reach the persona source even by
        // writing an IPC task file directly.
        if (!isMain) {
          logger.warn(
            { sourceGroup },
            'Unauthorized persist_global_file attempt',
          );
          return;
        }

        const persistResultPath = scriptResultPath(sourceGroup, data);

        // The container edits `/workspace/global/<file>`, an RW bind onto the
        // host's `groups/global/<file>` (the git-tracked, deploy-seeded
        // source). That edit lands in the working tree but is never
        // committed, so `deploy.sh`'s `git stash; git pull` discards it on
        // the next deploy (jbaruch/nanoclaw-admin#393). This handler commits
        // the working-tree change and pushes `main` so the next `git pull`
        // keeps it. The allowlist gate (`validateGlobalFilesToPersist`)
        // rejects anything but the exact persona basenames, so a compromised
        // container can't commit arbitrary tracked files (path traversal,
        // secrets, workflow YAML).
        const persistValidation = validateGlobalFilesToPersist(data.files);
        if (!persistValidation.ok) {
          logger.warn(
            { sourceGroup, invalidFile: persistValidation.invalid },
            'persist_global_file rejected non-allowlisted file',
          );
          fs.writeFileSync(
            persistResultPath,
            JSON.stringify({
              error: `persist_global_file: ${JSON.stringify(persistValidation.invalid)} is not an allowed global file (allowed: ${PERSISTABLE_GLOBAL_FILES.join(', ')})`,
              stage: 'validate',
            }),
          );
          return;
        }
        const persistRelPaths = persistValidation.relPaths;

        const persistCommitMsg =
          data.message ||
          `soul: persist approved updates ${new Date().toISOString().split('T')[0]}`;

        logger.info(
          {
            sourceGroup,
            relPaths: persistRelPaths,
            commitMsg: persistCommitMsg,
          },
          'Running persist_global_file',
        );

        // Read GitHub token for push auth (same wiring as github_backup).
        const { readEnvFile: readPersistEnv } = await import('../env.js');
        const persistGhToken = readPersistEnv(['GITHUB_TOKEN']).GITHUB_TOKEN;

        // The orchestrator's cwd (`/app`) is the built image dir, not a git
        // worktree (#471), so persist can't commit in place. It operates on a
        // dedicated self-provisioning clone of the deploy-source repo under
        // `data/` (gitignored, mounted, persistent), applies the live runtime
        // edits from `groups/global/` into it, then commits + pushes HEAD:main.
        const personaRepoDir = path.join(DATA_DIR, 'persona-repo');

        // Run clone + copy + commit + push off the IPC loop (like
        // github_backup): the loop never blocks on git. `runPersonaPersistTask`
        // is the testable boundary (ensure-clone → overlay → commit/push,
        // always writing a result envelope). Serialized so two closely-timed
        // persists can't race on the shared clone.
        runSerializedPersonaPersist(() =>
          runPersonaPersistTask({
            personaRepoDir,
            groupsDir: GROUPS_DIR,
            relPaths: persistRelPaths,
            remoteUrl: ORCHESTRATOR_REPO_URL,
            message: persistCommitMsg,
            token: persistGhToken,
            resultPath: persistResultPath,
            sourceGroup,
          }),
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

  registerIpcHandler('promote_staging', {
    handler: ({ data, sourceGroup, isMain }) => {
      if (data.requestId && data.tileName && data.skillName) {
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized promote_staging attempt');
          return;
        }

        const promoteResultPath = scriptResultPath(sourceGroup, data);

        if (!KNOWN_TILE_NAMES.has(data.tileName)) {
          logger.warn(
            { sourceGroup, tileName: data.tileName },
            'promote_staging rejected: tileName not in allowlist',
          );
          fs.writeFileSync(
            promoteResultPath,
            JSON.stringify({
              error: `Invalid tileName "${data.tileName}". Allowed: ${[...KNOWN_TILE_NAMES].join(', ')}.`,
            }),
          );
          return;
        }

        const promoteScript = path.join(
          process.cwd(),
          'scripts',
          'promote-to-tile-repo.sh',
        );

        if (!fs.existsSync(promoteScript)) {
          fs.writeFileSync(
            promoteResultPath,
            JSON.stringify({
              error: 'promote-to-tile-repo.sh not found',
            }),
          );
          return;
        }

        const stagingDir = path.join(
          GROUPS_DIR,
          sourceGroup,
          'staging',
          data.tileName,
        );

        // Read credentials from .env for tile repo push
        const envPath = path.join(process.cwd(), '.env');
        const envContent = fs.existsSync(envPath)
          ? fs.readFileSync(envPath, 'utf-8')
          : '';
        const getEnv = (key: string) =>
          envContent
            .split('\n')
            .find((l) => l.startsWith(`${key}=`))
            ?.split('=')
            .slice(1)
            .join('=') || '';

        logger.info(
          { sourceGroup, tileName: data.tileName, skillName: data.skillName },
          'Running promote_staging',
        );

        execFile(
          'bash',
          [promoteScript, stagingDir, data.tileName, data.skillName],
          {
            // 15 minutes. promote-to-tile-repo.sh runs `tessl skill
            // review --optimize` on each staged skill, and tessl
            // itself tells you that each review "can take up to 1
            // minute." A bulk promote (`skillName=all`) against a tile
            // with 10+ staged skills easily blows past the old
            // 5-minute cap, which observably killed every bulk promote
            // AyeAye tried and returned a mid-run truncated error. 15
            // min fits the typical 10-15 skill worst case with
            // headroom; larger bulk promotes should split the staging
            // directory into smaller batches — every skill is reviewed
            // (no per-skill opt-out).
            timeout: 900_000,
            maxBuffer: 5 * 1024 * 1024,
            env: {
              ...process.env,
              GITHUB_TOKEN: getEnv('GITHUB_TOKEN'),
              TILE_OWNER: getEnv('TILE_OWNER') || 'jbaruch',
              ASSISTANT_NAME: getEnv('ASSISTANT_NAME') || 'AyeAye',
            },
          },
          (error, stdout, stderr) => {
            if (error) {
              logger.error(
                {
                  sourceGroup,
                  error: error.message,
                  stderr: stderr.slice(-500),
                },
                'promote_staging failed',
              );
              fs.writeFileSync(
                promoteResultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
            } else {
              logger.info(
                { sourceGroup },
                'promote_staging opened PR on tile repo',
              );
              fs.writeFileSync(
                promoteResultPath,
                JSON.stringify({ stdout: stdout.trim() }),
              );

              // Post-promote `tessl update` + session clear used to
              // run here on a 5-minute delay, predicated on the old
              // flow that pushed directly to tile main and triggered
              // GHA publish within ~5min. New flow opens a PR and
              // requests Copilot review; publish only happens after
              // the PR is merged (which could be minutes, hours, or
              // never if Copilot/human rejects it). The auto-update
              // would fire against a registry that hasn't changed
              // yet — at best a no-op, at worst tearing down sessions
              // for no reason. The agent now calls the `tessl_update`
              // MCP tool explicitly after the PR merges; a periodic
              // 15-min catch-up in index.ts covers missed invocations.
            }
          },
        );
      }
    },
  });

  registerIpcHandler('tessl_update', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      if (data.requestId) {
        const tesslResultPath = scriptResultPath(sourceGroup, data);
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized tessl_update attempt');
          fs.writeFileSync(
            tesslResultPath,
            JSON.stringify({
              error: 'Only the main group can trigger tessl_update.',
            }),
          );
          return;
        }

        // #496 — gate the on-demand update on the same fresh-lock check
        // the periodic catch-up uses. A scheduled task mid-flight has
        // its skill holding `follow_me_tasks.pending_run_at`; running
        // tessl_update now would force-close the maintenance container
        // and orphan the lock. Defer with a structured response so the
        // calling agent can reason about retrying. Threshold is the same
        // 1h freshness window documented at the constant's definition
        // in `src/index.ts` — duplicated here as a literal because
        // ipc.ts has no shared-config home for it and lifting the
        // constant would force a circular import via index.ts.
        const PENDING_RUN_AT_FRESHNESS_MS = 60 * 60 * 1000;
        let activeLocks: string[];
        try {
          activeLocks = getActivePendingRunAtNames(PENDING_RUN_AT_FRESHNESS_MS);
        } catch (lockErr) {
          if (!(lockErr instanceof Error)) throw lockErr;
          // Don't fail the update on a lock-check error — log and
          // proceed. Failing closed would block legitimate updates
          // forever if the DB or schema regressed.
          logger.warn(
            { sourceGroup, err: lockErr.message },
            'tessl_update: pending_run_at check failed — proceeding without deferral',
          );
          activeLocks = [];
        }
        if (activeLocks.length > 0) {
          logger.info(
            { sourceGroup, activeLocks },
            'tessl_update deferred — scheduled task(s) mid-flight with fresh pending_run_at lock (#496)',
          );
          fs.writeFileSync(
            tesslResultPath,
            JSON.stringify({
              stdout: `Deferred: scheduled task(s) currently mid-flight with fresh pending_run_at lock (${activeLocks.join(', ')}). Retry after the task(s) finish — running tessl_update now would force-close the maintenance container and orphan the lock (#496).`,
              deferred: true,
              activeLocks,
            }),
          );
          return;
        }

        logger.info({ sourceGroup }, 'Running tessl_update');

        execFile(
          'bash',
          [
            '-c',
            'cd /app/tessl-workspace && tessl update --yes --accept-warnings --agent claude-code 2>&1',
          ],
          { timeout: 150_000, maxBuffer: 2 * 1024 * 1024 },
          (error, stdout) => {
            if (error) {
              logger.error(
                {
                  sourceGroup,
                  error: error.message,
                  output: stdout.slice(-500),
                },
                'tessl_update failed',
              );
              fs.writeFileSync(
                tesslResultPath,
                JSON.stringify({
                  error: error.message,
                  stdout: stdout.slice(-2000),
                }),
              );
              return;
            }
            const output = stdout.trim();
            // `tessl update` prints "Updated ..." when a tile actually
            // moved forward. No-op runs don't, and clearing sessions on a
            // no-op would nuke conversation state for nothing — hence
            // the string check instead of an unconditional clear.
            if (/\bUpdated\b/.test(output)) {
              const cleared = deleteAllSessions();
              // Close every currently-running container so it respawns on
              // the next inbound message with the freshly-installed tile
              // content. `deleteAllSessions()` only resets the SDK
              // session-id mapping — without this companion call, a
              // container that's mid-idle-loop right now keeps its old
              // skills/.tessl/ snapshot until its 30-min idle timeout
              // fires (issue #64). The two operations are paired: clear
              // the on-disk session state AND signal the live containers
              // so neither lingers on stale state.
              //
              // Guarded because `closeAllActiveContainers()` rethrows
              // unexpected (non-fs) errors by contract. Without the
              // guard, a programming bug would skip the result-file
              // write below and leave the IPC requester hanging without
              // a structured payload. On failure we still write a
              // partial-success JSON so the caller can distinguish
              // "containers signaled" from "sessions cleared but
              // signaling failed."
              let closed = 0;
              let closeErr: Error | null = null;
              try {
                closed = deps.closeAllActiveContainers();
              } catch (e) {
                // See src/index.ts periodic update for the rationale —
                // outer-boundary guard around an async-callback
                // boundary. Narrow to Error so non-Error throws (a
                // programming bug throwing a non-Error value)
                // propagate per error-handling.md.
                if (!(e instanceof Error)) throw e;
                closeErr = e;
                logger.error(
                  { err: e, sourceGroup, sessionsCleared: cleared },
                  'closeAllActiveContainers threw an unexpected error during tessl_update — sessions still cleared, but live containers will not respawn until idle timeout',
                );
              }
              logger.info(
                {
                  sourceGroup,
                  sessionsCleared: cleared,
                  containersClosed: closed,
                  closeError: closeErr ? String(closeErr) : undefined,
                },
                'tessl_update found new tiles — sessions cleared and running containers signaled to restart',
              );
              fs.writeFileSync(
                tesslResultPath,
                JSON.stringify(
                  closeErr
                    ? {
                        stdout: `${output}\n\nSessions cleared: ${cleared}\nContainers signaled to restart: ${closed}`,
                        warning: `closeAllActiveContainers failed: ${String(closeErr)} — running containers will pick up new tiles on idle timeout (~30 min) instead of immediately`,
                      }
                    : {
                        stdout: `${output}\n\nSessions cleared: ${cleared}\nContainers signaled to restart: ${closed}`,
                      },
                ),
              );
            } else {
              logger.info(
                { sourceGroup },
                'tessl_update completed — no new tiles',
              );
              fs.writeFileSync(
                tesslResultPath,
                JSON.stringify({ stdout: output || '(no output)' }),
              );
            }
          },
        );
      }
    },
  });

  registerIpcHandler('list_installed_tiles', {
    handler: ({ data, sourceGroup, isMain }) => {
      // Admin tile only. Returns the names of every tile currently
      // installed in the local Tessl registry — the same set
      // `set_additional_tiles` validates against. The agent uses this
      // to surface "what overlay tiles can I set on this chat?" to
      // the operator before proposing a config change. Read-only;
      // never mutates the registry.
      //
      // The result includes BOTH overlay tiles and the trust-tier
      // baseline (`nanoclaw-core`, `nanoclaw-trusted`,
      // `nanoclaw-untrusted`, `nanoclaw-admin`). The agent presenting
      // this list is expected to call out the baseline names as
      // reserved — `selectTiles` dedupes them against the baseline so
      // configuring one as an overlay is a no-op, not an error, but
      // it's still a misuse of the surface. We don't filter here
      // because (a) the baseline set is policy that lives in
      // `selectTiles`, not registry truth, and (b) future trust-tier
      // changes shouldn't silently alter what this tool returns.
      const tilesResultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized list_installed_tiles attempt',
        );
        fs.writeFileSync(
          tilesResultPath,
          JSON.stringify({
            error: 'list_installed_tiles is admin-tile only',
          }),
        );
        return;
      }
      const tiles = getInstalledTiles();
      if (tiles === null) {
        // Registry directory doesn't exist (cold start, never ran
        // `tessl install`). Distinct from "registry exists but empty"
        // so the operator knows whether to run `tessl_update` first.
        logger.warn(
          { sourceGroup },
          'list_installed_tiles: registry directory absent — run tessl_update',
        );
        fs.writeFileSync(
          tilesResultPath,
          JSON.stringify({
            stdout: JSON.stringify({
              tiles: [],
              registryAbsent: true,
            }),
          }),
        );
        return;
      }
      logger.info(
        { sourceGroup, count: tiles.length },
        'list_installed_tiles served via IPC',
      );
      fs.writeFileSync(
        tilesResultPath,
        JSON.stringify({
          stdout: JSON.stringify({
            tiles,
            registryAbsent: false,
          }),
        }),
      );
    },
  });

  registerIpcHandler('push_staged_to_branch', {
    handler: ({ data, sourceGroup, isMain }) => {
      if (
        data.requestId &&
        data.tileName &&
        data.branch &&
        data.commitMessage
      ) {
        const pushResultPath = scriptResultPath(sourceGroup, data);
        if (!isMain) {
          logger.warn(
            { sourceGroup },
            'Unauthorized push_staged_to_branch attempt',
          );
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({
              error: 'Only the main group can push to tile branches.',
            }),
          );
          return;
        }

        if (!KNOWN_TILE_NAMES.has(data.tileName)) {
          logger.warn(
            { sourceGroup, tileName: data.tileName },
            'push_staged_to_branch rejected: tileName not in allowlist',
          );
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({
              error: `Invalid tileName "${data.tileName}". Allowed: ${[...KNOWN_TILE_NAMES].join(', ')}.`,
            }),
          );
          return;
        }

        const pushScript = path.join(
          process.cwd(),
          'scripts',
          'push-staged-to-branch.sh',
        );

        if (!fs.existsSync(pushScript)) {
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({ error: 'push-staged-to-branch.sh not found' }),
          );
          return;
        }

        const stagingDir = path.join(
          GROUPS_DIR,
          sourceGroup,
          'staging',
          data.tileName,
        );

        // Same .env reader pattern as promote_staging — the orchestrator
        // holds the tile-repo credentials, containers never see them.
        const envPath = path.join(process.cwd(), '.env');
        const envContent = fs.existsSync(envPath)
          ? fs.readFileSync(envPath, 'utf-8')
          : '';
        const getEnv = (key: string) =>
          envContent
            .split('\n')
            .find((l) => l.startsWith(`${key}=`))
            ?.split('=')
            .slice(1)
            .join('=') || '';

        logger.info(
          {
            sourceGroup,
            tileName: data.tileName,
            branch: data.branch,
            skillName: data.skillName || 'all',
          },
          'Running push_staged_to_branch',
        );

        execFile(
          'bash',
          [
            pushScript,
            stagingDir,
            data.tileName,
            data.branch,
            data.commitMessage,
            data.skillName || 'all',
          ],
          {
            // 5 min is enough for a clone-branch + copy + commit +
            // push. No tessl review loop here — fixups don't re-trigger
            // the local optimize pass.
            timeout: 300_000,
            maxBuffer: 2 * 1024 * 1024,
            env: {
              ...process.env,
              GITHUB_TOKEN: getEnv('GITHUB_TOKEN'),
              TILE_OWNER: getEnv('TILE_OWNER') || 'jbaruch',
              ASSISTANT_NAME: getEnv('ASSISTANT_NAME') || 'AyeAye',
            },
          },
          (error, stdout, stderr) => {
            if (error) {
              logger.error(
                {
                  sourceGroup,
                  error: error.message,
                  stderr: stderr.slice(-500),
                },
                'push_staged_to_branch failed',
              );
              fs.writeFileSync(
                pushResultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
              return;
            }
            logger.info(
              {
                sourceGroup,
                tileName: data.tileName,
                branch: data.branch,
              },
              'push_staged_to_branch pushed fixup',
            );
            fs.writeFileSync(
              pushResultPath,
              JSON.stringify({ stdout: stdout.trim() }),
            );
          },
        );
      }
    },
  });
}
