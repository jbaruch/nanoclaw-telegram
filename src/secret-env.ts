// Secret env-file materialization (#851 slice 2, extracted verbatim
// from src/container-runner.ts).
//
// Which forwarded container env vars must ride an 0600 env-file
// instead of `-e` on the world-readable docker command line, the
// OneCLI managed-placeholder map (#564/#640), and the symlink-race-
// safe env-file writer with its idempotent cleanup.
import { randomBytes } from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { isErrnoCodedError } from './fs-errors.js';
import { logger } from './logger.js';
import { ONECLI_MANAGED_PLACEHOLDER } from './onecli-client.js';

/**
 * Container env vars whose VALUES are sensitive and must never appear
 * on the docker process command line — `ps -ef` and `/proc/<pid>/cmdline`
 * are world-readable on most kernels, so a `-e KEY=<value>` flag leaks
 * the value to any local user (and to monitoring tooling that captures
 * process tables). For these names we materialize an env-file
 * (mode 0600) and pass it via `--env-file <path>`; for everything else
 * (placeholders, non-secret config like AGENT_MODEL, NANOCLAW_CHAT_JID)
 * `-e KEY=value` is fine.
 *
 * What counts as "sensitive" for this set:
 *   - Real credentials (API keys, OAuth tokens, etc.).
 *   - Account-identifying values that aren't strictly credentials but
 *     would let an observer correlate the container to a specific user
 *     account at the upstream provider (an embedded server/account id
 *     in a URL, say — leaking it on `docker ps` would identify the
 *     account even though the value is not a credential on its own).
 * The unifying contract is "nothing the docker command line should
 * reveal", not "only literal credentials".
 *
 * Relationship with the `CONTAINER_VARS` list inside `buildContainerArgs`:
 * `CONTAINER_VARS` decides WHICH variables get forwarded at all (and is
 * gated by the trust tier — untrusted groups forward nothing).
 * `SECRET_CONTAINER_VARS` decides, OF THE FORWARDED ONES, which must
 * route through the env-file rather than `-e`. The two intentionally
 * serve different concerns; do not collapse one into the other.
 *
 * When introducing a new container env var with a sensitive value:
 * (1) add it to the local `CONTAINER_VARS` list so it's forwarded, AND
 * (2) add it here so the value goes through the env-file. Missing
 * either step leaves the value either un-forwarded or back on the
 * command line.
 *
 * Variables with placeholder values (proxied through OneCLI) are NOT
 * sensitive and stay on the command line. Variables with non-secret
 * config (`AGENT_MODEL`, `NANOCLAW_CHAT_JID`, etc.) likewise stay on
 * the command line.
 */
export const SECRET_CONTAINER_VARS: ReadonlySet<string> = new Set([
  // Fine-grained GitHub PAT for the `gh` CLI inside main/trusted-tier
  // containers. Same .env entry the host-side `github_backup` IPC
  // handler uses for `git push`; forwarding it into the container lets
  // skills run `gh issue list/edit/comment` directly. `gh` reads
  // `GITHUB_TOKEN` automatically —
  // no `gh auth login` needed inside the container. The fact that this
  // still lives in the container's environ for the spawn lifetime is
  // the OneCLI-proxy migration target tracked in jbaruch/nanoclaw#564.
  'GITHUB_TOKEN',
  // byAir personal MCP link with API key inline (URL form, but the
  // query string is the credential) — read by
  // `jbaruch/nanoclaw-travel`'s precheck. Treated as secret so
  // the URL with embedded key doesn't appear on `ps`/`docker ps`. See
  // CONTAINER_VARS for the consumer-side context.
  'BYAIR_MCP_URL',
  // Google Maps Distance Matrix API key — read by the same tile.
  // Standard `AIzaSy...` shape, ~$10/1000 requests at our usage.
  'GOOGLE_MAPS_API_KEY',
  // TomTom Routing/Search API key — read by the `jbaruch/nanoclaw-travel`
  // tile's `maps_client.py` (TomTom geocode + calculateRoute backup behind
  // Google Maps) and the `drive-planner` skill. Calls `api.tomtom.com`.
  // Secret so the key stays off `ps`/`docker ps`, same as the Maps key.
  'TOMTOM_API_KEY',
  // YouTube Data API v3 key — read by the admin tile's
  // `youtube-comment-check` skill, which calls the native API directly
  // per jbaruch/nanoclaw-admin#339. Standard `AIzaSy...` key; goes through
  // the env-file rather than `-e` so it stays off `ps`/`docker ps`.
  'YOUTUBE_API_KEY',
  // Sessionize speaker-profile API key — read by the conferences tile's
  // `discover-open-cfps.py` (jbaruch/nanoclaw-conferences#9), which calls
  // GET sessionize.com/api/universal/open-cfps directly from the container
  // for deterministic open-CFP discovery. Generated at Sessionize →
  // Speaker Profile → API / Integrations. Secret so the key stays off
  // `ps`/`docker ps`, same as the other API keys.
  'SESSIONIZE_SPEAKER_KEY',
  // Sessionize event API key — read by the same tile's
  // `verify-sessionize.py` for live per-slug CFP-deadline verification.
  // The container-side driver is now the sole consumer; forwarding the
  // key lets it do the round-trip in-container without IPC. Secret so
  // the key stays off `ps`/`docker ps`, same as the other API keys.
  'SESSIONIZE_EVENT_API_KEY',
]);

// `ONECLI_MANAGED_PLACEHOLDER` (the `onecli-managed` sentinel) now lives in
// `onecli-client.ts` — its canonical OneCLI home, shared with the host-side
// OpenAI transcription swap (#770). Imported above; re-exported here so
// existing consumers of the container-runner surface keep resolving it.
export { ONECLI_MANAGED_PLACEHOLDER };

/**
 * URL-valued managed credential: the sentinel rides INSIDE a syntactically
 * valid URL (query param here) rather than being a bare scalar. The consuming
 * skill reads a real-looking URL so its own URL parse/validation still passes;
 * the gateway overwrites the `onecli-managed` sentinel with the vaulted secret
 * on the outbound request (query-param injection for `api.byairapp.com`). Keep
 * the host in sync with the vault entry's `hostPattern` and the `api_key` param
 * name with its `paramName` (the `/mcp` path is not part of the match).
 */
export const BYAIR_MANAGED_PLACEHOLDER = `https://api.byairapp.com/mcp?api_key=${ONECLI_MANAGED_PLACEHOLDER}`;

/**
 * Credentials that migrate from real-value env-file injection to OneCLI
 * placeholder + gateway swap (umbrella #564, cleanup #640). When OneCLI is
 * configured (`isOneCliConfigured()` — the same gate the spawn uses to apply
 * the gateway proxy), the container receives the var's mapped placeholder
 * instead of the real value, and OneCLI's TLS-MITM injects the real secret for
 * the vault host-pattern (the managed vars below use header and query-param
 * injection; OneCLI also supports URL-path injection, used host-side by the
 * TripIt sync in #748, not by any var here). When OneCLI is unconfigured (local
 * dev), the var falls back to
 * real-value forwarding via the normal `SECRET_CONTAINER_VARS` path. If the
 * gate passes but the gateway proxy cannot actually be applied at spawn time,
 * the caller fails the spawn closed (see `managedPlaceholdersApplied`) rather
 * than shipping a container with a dead placeholder credential.
 *
 * The value is the placeholder to forward. Scalar creds (the secret IS the
 * whole value) use the bare `onecli-managed` sentinel; URL-valued creds (the
 * secret is one field of a URL the skill parses) use a URL-shaped placeholder
 * that embeds the sentinel where the gateway swaps it (see
 * `BYAIR_MANAGED_PLACEHOLDER`).
 *
 * Preconditions to add a var here: (1) a OneCLI vault entry exists for its
 * host with the correct header/param/path injection config, (2) injection is
 * probe-verified for that host (placeholder key/header/path through the gateway
 * returns real data). A host-side reader does NOT disqualify a var: only the
 * CONTAINER is placeholdered here, so a var the host also reads (e.g.
 * `GITHUB_TOKEN` → the `github_backup` IPC handler's `git push`) keeps its real
 * value in `.env` for the host while agents get the placeholder + swap.
 *
 * All entries swap-verified through the gateway (placeholder → real data):
 * `GOOGLE_MAPS_API_KEY`      — `maps.googleapis.com`, param `key` (Distance Matrix).
 * `TOMTOM_API_KEY`           — `api.tomtom.com`, param `key` (routing / geocode).
 * `YOUTUBE_API_KEY`          — `www.googleapis.com` path `/youtube/*`, param `key`.
 * `GITHUB_TOKEN`             — `api.github.com`, header `Authorization: Bearer` (gh);
 *                              host-side `github_backup` still reads the .env value.
 * `SESSIONIZE_SPEAKER_KEY`   — `sessionize.com` path `/api/universal/open-cfps`,
 *                              header `X-API-KEY` (path-scoped so it never
 *                              collides with the event key on the same host).
 * `SESSIONIZE_EVENT_API_KEY` — `sessionize.com` path `/api/universal/event`,
 *                              header `X-API-KEY` (path-scoped).
 * `BYAIR_MCP_URL`            — `api.byairapp.com`, param `api_key`; URL-valued,
 *                              forwarded as `BYAIR_MANAGED_PLACEHOLDER`.
 */
export const ONECLI_MANAGED_VARS: ReadonlyMap<string, string> = new Map([
  ['GOOGLE_MAPS_API_KEY', ONECLI_MANAGED_PLACEHOLDER],
  ['TOMTOM_API_KEY', ONECLI_MANAGED_PLACEHOLDER],
  ['YOUTUBE_API_KEY', ONECLI_MANAGED_PLACEHOLDER],
  ['GITHUB_TOKEN', ONECLI_MANAGED_PLACEHOLDER],
  ['SESSIONIZE_SPEAKER_KEY', ONECLI_MANAGED_PLACEHOLDER],
  ['SESSIONIZE_EVENT_API_KEY', ONECLI_MANAGED_PLACEHOLDER],
  ['BYAIR_MCP_URL', BYAIR_MANAGED_PLACEHOLDER],
]);

/**
 * Result of materializing an env-file for a container spawn.
 * `args` are appended to the `docker run` argv; `cleanup` MUST be
 * invoked after the container exits (close OR error path) to remove
 * the on-disk file. The cleanup is idempotent — safe to call from
 * either handler regardless of whether the other already ran.
 */
export interface SecretEnvFile {
  args: string[];
  cleanup: () => void;
}

/**
 * Materialize a 0600-mode env-file containing the given secrets and
 * return docker `--env-file` args + a cleanup callback. Returns `null`
 * when there are no secrets to forward — the caller skips emitting
 * any extra args.
 *
 * Refuses to write a value containing CR/LF/NUL: docker's env-file
 * parser has no quoting, so an embedded newline would silently truncate
 * the variable or smuggle a second `KEY=...` line. Failing fast at
 * write time keeps the failure visible to the operator instead of
 * surfacing as a confusing "container can't find env var" later.
 *
 * Uses `O_CREAT | O_EXCL` with a random 24-hex-char suffix so a local
 * attacker can't pre-create the path as a symlink (symlink-race) and
 * exfiltrate the secret on write.
 */
export function buildSecretEnvFile(
  env: Record<string, string>,
): SecretEnvFile | null {
  const entries = Object.entries(env).filter(([, v]) => v !== '');
  if (entries.length === 0) return null;

  const lines = entries.map(([k, v]) => {
    if (/[\r\n\0]/.test(v)) {
      throw new Error(
        `Secret env var ${k} contains CR/LF/NUL; refusing to write env-file (docker env-file format has no quoting).`,
      );
    }
    return `${k}=${v}`;
  });

  const tmpPath = path.join(
    os.tmpdir(),
    `nanoclaw-env-${randomBytes(12).toString('hex')}`,
  );
  // O_EXCL fails if path exists; mode 0600 keeps the file unreadable
  // by other local users between open and the docker daemon's read.
  const fd = fs.openSync(
    tmpPath,
    fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL,
    0o600,
  );
  try {
    // Inner try/finally closes the fd on every path FIRST; the outer catch
    // then removes the partial-secret tempfile with the handle already closed
    // (so the unlink can't fail on an open handle) before re-throwing the
    // write error — the real cause. The unlink's narrow-and-rethrow is safe
    // here because it sits in a catch, not a finally (no-unsafe-finally).
    try {
      fs.writeFileSync(fd, lines.join('\n') + '\n');
    } finally {
      fs.closeSync(fd);
    }
  } catch (writeErr) {
    try {
      fs.unlinkSync(tmpPath);
    } catch (unlinkErr) {
      // Best-effort cleanup: tolerate ANY fs/OS errno (EIO, EDQUOT, …) so a
      // cleanup failure never masks writeErr — the real cause, rethrown below.
      if (!isErrnoCodedError(unlinkErr)) throw unlinkErr;
      const code = (unlinkErr as NodeJS.ErrnoException).code;
      if (code !== 'ENOENT') {
        logger.warn(
          { err: unlinkErr, tmpPath },
          'Failed to clean up secret env-file after write error',
        );
      }
    }
    throw writeErr;
  }

  let cleaned = false;
  return {
    args: ['--env-file', tmpPath],
    cleanup: () => {
      if (cleaned) return;
      cleaned = true;
      try {
        fs.unlinkSync(tmpPath);
      } catch (err) {
        // Best-effort teardown: tolerate ANY fs/OS errno; only a non-errno
        // defect propagates. Worst case the 0600 tempfile is cleared on the
        // next reboot via tmpdir.
        if (!isErrnoCodedError(err)) throw err;
        if ((err as NodeJS.ErrnoException).code === 'ENOENT') return;
        logger.warn(
          { err, tmpPath },
          'Failed to clean up secret env-file; will be cleared on next reboot via tmpdir',
        );
      }
    },
  };
}
