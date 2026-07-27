import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';

import { GROUPS_DIR } from '../config.js';

import { getInstalledTiles } from '../container-runner.js';
import { deleteAllSessions } from '../db-sessions.js';
import { getActivePendingRunAtNames } from '../db-tz.js';
import { registerIpcHandler, scriptResultPath } from '../ipc-registry.js';
import { logger } from '../logger.js';

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
 * Tile pipeline + registry maintenance (#879, split out of `ops.ts`):
 * the promote/fixup flows against the tile repos (`promote_staging`,
 * `push_staged_to_branch`) and the registry-side maintenance commands
 * (`tessl_update`, `list_installed_tiles`).
 */
export function registerOpsTilesIpcHandlers(): void {
  registerIpcHandler('promote_staging', {
    handler: ({ data, sourceGroup, isMain }) => {
      // Enter on `requestId` alone and validate inside (#885). Gating the
      // whole body on the required fields meant a payload carrying a
      // requestId but missing `tileName`/`skillName` produced NO result
      // file, and the in-container `runHostOperation()` reads an absent
      // file as a hang until its own timeout. The MCP tool's zod schema
      // validates client-side, but the IPC tasks dir is writable by any
      // container, so a payload that skips the MCP path arrives here
      // unvalidated. Same boundary-validation shape as `run_sidecar`.
      if (data.requestId) {
        const promoteResultPath = scriptResultPath(sourceGroup, data);

        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized promote_staging attempt');
          fs.writeFileSync(
            promoteResultPath,
            JSON.stringify({
              error: 'Only the main group can promote staging content.',
            }),
          );
          return;
        }

        if (typeof data.tileName !== 'string' || !data.tileName) {
          fs.writeFileSync(
            promoteResultPath,
            JSON.stringify({
              error: 'promote_staging: "tileName" must be a non-empty string.',
            }),
          );
          return;
        }
        if (typeof data.skillName !== 'string' || !data.skillName) {
          fs.writeFileSync(
            promoteResultPath,
            JSON.stringify({
              error: 'promote_staging: "skillName" must be a non-empty string.',
            }),
          );
          return;
        }

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
      // Same #885 boundary validation as `promote_staging` above: enter on
      // `requestId` and answer every malformed payload with an envelope,
      // never with silence the caller reads as a hang.
      if (data.requestId) {
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

        if (typeof data.tileName !== 'string' || !data.tileName) {
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({
              error:
                'push_staged_to_branch: "tileName" must be a non-empty string.',
            }),
          );
          return;
        }
        if (typeof data.branch !== 'string' || !data.branch) {
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({
              error:
                'push_staged_to_branch: "branch" must be a non-empty string.',
            }),
          );
          return;
        }
        if (typeof data.commitMessage !== 'string' || !data.commitMessage) {
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({
              error:
                'push_staged_to_branch: "commitMessage" must be a non-empty string.',
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
