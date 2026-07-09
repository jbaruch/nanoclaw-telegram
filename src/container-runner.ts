/**
 * Container Runner for NanoClaw
 * Spawns agent execution in containers and handles IPC
 */
import { ChildProcess, spawn, spawnSync } from 'child_process';
import Database, { SqliteError } from 'better-sqlite3';
import { randomBytes } from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

import {
  AGENT_AUTO_COMPACT_WINDOW,
  ASSISTANT_NAME,
  ASSISTANT_USERNAMES,
  CONTAINER_IMAGE,
  CONTAINER_MAX_OUTPUT_SIZE,
  CONTAINER_TIMEOUT,
  CREDENTIAL_PROXY_PORT,
  DATA_DIR,
  ENABLE_THRESHOLD_NUKE,
  GROUPS_DIR,
  HOST_GID,
  HOST_PROJECT_ROOT,
  HOST_UID,
  IDLE_TIMEOUT,
  MAINTENANCE_CONTAINER_TIMEOUT,
  MAINTENANCE_RULE_BLOCKLIST,
  MAINTENANCE_SKILL_BLOCKLIST,
  STORE_DIR,
  TILE_OWNER,
  TIMEZONE,
} from './config.js';
import { resolveGroupFolderPath, resolveGroupIpcPath } from './group-folder.js';
import {
  containerLogPath,
  ensureHostLogDirs,
  hostLogsDir,
  stripAnsi,
} from './host-logs.js';
import { logger } from './logger.js';
import { computeEffectiveSkillContext } from './skill-dep-closure.js';
import { shouldIncludeRule } from './rule-requires-filter.js';
import { onAgentLine } from './observer.js';
import {
  CONTAINER_HOST_GATEWAY,
  CONTAINER_RUNTIME_BIN,
  hostGatewayArgs,
  readonlyMountArgs,
  stopContainer,
} from './container-runtime.js';
import { defaultComputeNextRun } from './cadence-registry.js';
import { detectAuthMode } from './credential-proxy.js';
import { registerContainer, unregisterContainer } from './proxy-registry.js';
import {
  applyOneCliToSpawn,
  isOneCliConfigured,
  oneCliAgentProxyEnabled,
} from './onecli-client.js';
import type { TrustTier } from './trust-tier.js';
import { rebuildCadenceRegistryForGroup } from './db.js';
import { isHandoffActive } from './handoff.js';
import { sweepStaleInputs } from './ipc-input-sweep.js';
import { validateAdditionalMounts } from './mount-security.js';
import { RegisteredGroup } from './types.js';
import { readEnvFile } from './env.js';

/**
 * Select which tiles to install based on group trust tier, plus any
 * per-chat overlay tiles (#305).
 *
 * Trust-tier baseline:
 *   - Main: core + trusted + admin (admin loads last so it can override
 *     trusted skills).
 *   - Trusted: core + trusted.
 *   - Untrusted: core + untrusted.
 *
 * `additionalTiles` (from `containerConfig.additionalTiles`) is appended
 * after the baseline. Duplicates already present in the baseline are
 * dropped so the install order stays stable. The overlay never replaces
 * the baseline — it only adds capability tiles on top.
 */
export function selectTiles(
  isMain: boolean,
  isTrusted: boolean,
  additionalTiles?: readonly string[],
): string[] {
  const baseline = isMain
    ? ['nanoclaw-core', 'nanoclaw-trusted', 'nanoclaw-admin']
    : isTrusted
      ? ['nanoclaw-core', 'nanoclaw-trusted']
      : ['nanoclaw-core', 'nanoclaw-untrusted'];
  if (!additionalTiles || additionalTiles.length === 0) return baseline;
  const seen = new Set(baseline);
  const overlay: string[] = [];
  for (const tile of additionalTiles) {
    // Skip empty / whitespace-only names defensively — IPC validation
    // already rejects them, but a buggy direct-DB write shouldn't push
    // an empty string into the install loop where it would resolve to
    // `path.join(registryTiles, '')` === `registryTiles` itself.
    const trimmed = typeof tile === 'string' ? tile.trim() : '';
    if (!trimmed) continue;
    if (seen.has(trimmed)) continue;
    seen.add(trimmed);
    overlay.push(trimmed);
  }
  return [...baseline, ...overlay];
}

/**
 * Resolve the directory the local Tessl registry installs tiles into.
 * Single source of truth for both spawn-time tile copy and write-time
 * `set_additional_tiles` validation (#305).
 *
 * tessl >= 0.81 installs to `.tessl/plugins/`; older CLIs used
 * `.tessl/tiles/`. 0.81's `tessl update` migrates an existing tree by
 * writing `plugins/` and DELETING `tiles/`, so once the orchestrator
 * image's floating `npm install -g tessl` crosses 0.81 the workspace
 * flips dirs. Prefer `plugins/` when present; fall back to `tiles/` for
 * pre-0.81 installs and as the cold-start default (so `getInstalledTiles`
 * still ENOENTs to `null` on a never-installed workspace).
 */
export function getRegistryTilesDir(): string {
  const tesslRoot = path.join(process.cwd(), 'tessl-workspace', '.tessl');
  const pluginsDir = path.join(tesslRoot, 'plugins', TILE_OWNER);
  if (fs.existsSync(pluginsDir)) {
    return pluginsDir;
  }
  return path.join(tesslRoot, 'tiles', TILE_OWNER);
}

/**
 * Return the names of tiles installed in the local registry, or `null`
 * if the registry directory doesn't exist (cold start, never ran
 * `tessl install`). Callers distinguish "registry empty" from "registry
 * absent" by checking for `null`.
 *
 * NOTE: presence of a tile directory does NOT guarantee the tile's
 * content is healthy (a partial copy can leave an empty dir). The
 * spawn-time tile-install loop has its own per-tile sanity checks for
 * that; this helper is the cheaper "is the name even known" gate used
 * before persisting a config change so admins see typos at write time.
 */
export function getInstalledTiles(): string[] | null {
  const dir = getRegistryTilesDir();
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch (err: unknown) {
    if (
      err instanceof Error &&
      (err as NodeJS.ErrnoException).code === 'ENOENT'
    ) {
      return null;
    }
    throw err;
  }
  return entries
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort();
}

/**
 * #544b — pre-scan helper for the maintenance skill blocklist.
 *
 * Walks the same three skill source directories the install loop
 * below visits — tile skills under each `tilesToInstall[i]/skills/`,
 * built-in skills under `<cwd>/container/skills/`, and AyeAye-staged
 * skills under `<groupDir>/skills/` — collecting every present
 * `SKILL.md`'s text content. Then runs `computeEffectiveBlocklist`
 * to walk the transitive `Skill(skill: "...")` reference graph from
 * every non-blocklisted skill and exempt anything reachable.
 *
 * Returns a fresh set; the input `originalBlocklist` is not mutated.
 *
 * Failure modes:
 *   - Missing source dir (`ENOENT` on `readdirSync`) — best-effort
 *     skip. Tiles legitimately ship without skills, groups
 *     legitimately ship without staging, the host repo always has
 *     `<cwd>/container/skills/` so an absent dir there would be a
 *     real bug, but we still skip rather than crash.
 *   - Missing `SKILL.md` inside a skill subdir (`ENOENT` on
 *     `readFileSync`) — same. A subdir without a SKILL.md
 *     contributes no references; nothing to walk.
 *   - Other errno (`EACCES`, `EIO`, `ENOTDIR`, fs corruption) —
 *     PROPAGATES per `coding-policy: error-handling`. A perms /
 *     IO failure on the skill source dirs is operator-actionable
 *     drift; failing the spawn loudly here surfaces it instead of
 *     silently shipping a degraded blocklist that could
 *     reintroduce the runtime "Unknown skill" failure the closure
 *     exists to prevent.
 */
interface EffectiveSpawnSkillContext {
  effectiveBlocklist: Set<string>;
  reachableSkills: Set<string>;
}

function computeEffectiveSkillContextForSpawn(
  originalBlocklist: ReadonlySet<string>,
  tilesToInstall: readonly string[],
  registryTiles: string,
  groupDir: string,
): EffectiveSpawnSkillContext {
  const sources = new Map<string, string>();

  const ingestSkillDir = (skillsRoot: string) => {
    // Per `coding-policy: error-handling` the catches below are
    // narrowed to ENOENT only. ENOENT on a skill source dir is
    // expected — tile-without-skills, group without `<groupDir>/
    // skills/` (no staging skills), or a SKILL.md missing on a
    // non-skill subdir all hit ENOENT legitimately. Any other errno
    // (EACCES, EIO, ENOTDIR, fs corruption) is operator-actionable
    // drift the install loop's own per-tile checks below would also
    // surface; let it propagate so the spawn fails loudly rather
    // than silently under-including the closure and reintroducing
    // the `Unknown skill` runtime failure the closure exists to
    // prevent.
    let entries: string[];
    try {
      entries = fs.readdirSync(skillsRoot);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
      return;
    }
    for (const skillDir of entries) {
      const skillPath = path.join(skillsRoot, skillDir);
      let stat: fs.Stats;
      try {
        stat = fs.statSync(skillPath);
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
        continue;
      }
      if (!stat.isDirectory()) continue;
      const skillMdPath = path.join(skillPath, 'SKILL.md');
      try {
        sources.set(skillDir, fs.readFileSync(skillMdPath, 'utf8'));
      } catch (err) {
        // ENOENT on SKILL.md is normal for non-skill subdirs (rare
        // but possible). Other errno propagates per the same
        // rationale as readdir above.
        if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
      }
    }
  };

  for (const tileName of tilesToInstall) {
    ingestSkillDir(path.join(registryTiles, tileName, 'skills'));
  }
  ingestSkillDir(path.join(process.cwd(), 'container', 'skills'));
  ingestSkillDir(path.join(groupDir, 'skills'));

  return computeEffectiveSkillContext(originalBlocklist, sources);
}

// Sentinel markers for robust output parsing (must match agent-runner)
const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

/**
 * Filesystem error codes that indicate a concurrent caller won the race for
 * the same target path. Treat these as benign — the winner produced a valid
 * result, our work is just redundant. Any other errno is a real failure and
 * must propagate.
 *
 * - EEXIST: rename target already exists
 * - ENOTEMPTY: rmdir on a directory that another caller refilled
 * - EPERM / EACCES: rare, but seen on macOS when two processes contend for
 *   a directory rename across the same filesystem under load
 */
// Benign errno codes returned by concurrent atomic-publish callers
// racing on `renameSync(dstDir, backupDir)` and `renameSync(tmpDir,
// dstDir)`. Any of these means another caller already won the swap;
// the loser's copy is equivalent because both built `tmpDir` from the
// same source. NOTE: `ENOENT` is intentionally NOT in this global set —
// `cpSync(srcDir, ...)` throws `ENOENT` when `srcDir` is genuinely
// missing, which is a real publish failure, not a race. The rename
// race-window for ENOENT is handled phase-locally inside
// `atomicPublishDir` (only after `cpSync` has succeeded).
const RACE_CODES = new Set(['EEXIST', 'ENOTEMPTY', 'EPERM', 'EACCES']);
// Cleanup-time errno codes that are SAFE to swallow without logging:
// the artefact has either already been removed by another caller, or
// never existed (e.g. step 2 didn't run because dstDir was absent).
const CLEANUP_BENIGN_CODES = new Set(['ENOENT']);

/**
 * Best-effort recursive remove. Used for cleaning up temp / backup artefacts
 * in atomic-publish flows where leaking a sibling dir is preferable to
 * shadowing the original error (or to throwing during error recovery and
 * hiding the real failure from logs).
 */
function rmBestEffort(target: string): void {
  try {
    fs.rmSync(target, { recursive: true, force: true });
  } catch (err: unknown) {
    // Per `coding-policy: error-handling`: narrow to typed errno
    // shape first, rethrow anything else. Benign codes (e.g.
    // ENOENT — target already gone) swallow silently; real
    // filesystem drift (permissions, disk full, fs corruption)
    // WARN-logs so an operator can see it. A non-errno throw shape
    // (e.g. a synchronous instrumentation error) propagates so a
    // genuine bug in fs.rmSync isn't silently downgraded to
    // "orphaned artefact."
    if (!(err instanceof Error) || !('code' in err)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code && CLEANUP_BENIGN_CODES.has(code)) return;
    logger.warn(
      { err, target },
      'rmBestEffort: unexpected cleanup failure (orphaned artefact)',
    );
  }
}

/**
 * Atomically publish `srcDir` (a fully-built directory) to `dstDir`,
 * replacing any existing content at `dstDir` with no observable
 * partial-write window.
 *
 * Pattern (mirrors createFilteredDb's atomic temp+rename in #93/#94 and
 * the groupScriptsDir symlink-flip below):
 *   1. cpSync(srcDir, tmp) — build a complete sibling
 *   2. rename(dstDir, backup) if it exists
 *   3. rename(tmp, dstDir)
 *   4. rmBestEffort(backup)
 *
 * Concurrent callers race on step 2/3 — the loser hits ENOTEMPTY/EEXIST/
 * EPERM/EACCES, which we swallow as a debug log because the winner's copy
 * is equivalent. Any other error propagates.
 *
 * Why temp+swap-rename instead of rmSync+cpSync (the bug in #95):
 * `fs.rmSync` walks the tree and unlinks children one at a time. While
 * it's mid-walk, a concurrent caller's `fs.cpSync` can re-create files
 * inside subdirs that the walk hasn't reached yet, so the eventual
 * `rmdir` on those subdirs fails ENOTEMPTY. Swap-rename is atomic — at
 * any instant `dstDir` resolves to a fully-populated directory.
 *
 * Both `tmp` and `backup` MUST be on the same filesystem as `dstDir` for
 * rename atomicity. Putting them in the same parent satisfies this.
 */
export function atomicPublishDir(srcDir: string, dstDir: string): void {
  const swapId = `${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const tmpDir = `${dstDir}.tmp-${swapId}`;
  const backupDir = `${dstDir}.swap-${swapId}`;
  let backupCreated = false;
  // Phase tracker: `cpSync` is phase 'build'; the two renames are
  // phase 'rename'. ENOENT during `build` means `srcDir` is missing
  // (a real publish failure). ENOENT during `rename` is a benign
  // race with a concurrent caller. The catch block uses this to
  // gate which errno codes count as benign.
  let phase: 'build' | 'rename' = 'build';
  try {
    fs.cpSync(srcDir, tmpDir, { recursive: true });
    phase = 'rename';
    if (fs.existsSync(dstDir)) {
      fs.renameSync(dstDir, backupDir);
      backupCreated = true;
    }
    fs.renameSync(tmpDir, dstDir);
    if (backupCreated) rmBestEffort(backupDir);
  } catch (err: unknown) {
    // Always drop our publish artefacts before deciding what to do with
    // the error. If the race winner placed correct content at dstDir our
    // tmp is redundant; on a real error we can't trust our partial build.
    rmBestEffort(tmpDir);
    if (backupCreated) {
      // Try to restore the backup so dstDir isn't left absent — best
      // effort, since the failure may be the rename itself. If the
      // restore fails, leave the backup in place and rethrow; an
      // operator can recover from a sibling dir but not from missing
      // content.
      if (!fs.existsSync(dstDir)) {
        try {
          fs.renameSync(backupDir, dstDir);
        } catch (restoreErr: unknown) {
          // Per `coding-policy: error-handling`: narrow to typed
          // errno; let other shapes propagate via the swallowing
          // log here (we don't rethrow because the original publish
          // error is the primary signal).
          if (!(restoreErr instanceof Error) || !('code' in restoreErr)) {
            logger.warn(
              { err: restoreErr, dstDir, backupDir },
              'atomic dir publish: restore from backup failed (non-errno); ' +
                'backup sibling left for operator recovery',
            );
          } else {
            const restoreCode = (restoreErr as NodeJS.ErrnoException).code;
            logger.warn(
              { err: restoreErr, code: restoreCode, dstDir, backupDir },
              'atomic dir publish: restore from backup failed; ' +
                'backup sibling left for operator recovery',
            );
          }
        }
      } else {
        rmBestEffort(backupDir);
      }
    }
    // Per `coding-policy: error-handling`: narrow to typed errno
    // before classifying. A non-errno throw (e.g. a synchronous
    // instrumentation error from inside fs.renameSync) propagates so
    // a real defect surfaces instead of being silently swallowed as
    // a race-loser.
    if (!(err instanceof Error) || !('code' in err)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    // ENOENT is benign ONLY in the `rename` phase: the winner moved
    // the destination directory away after our `existsSync` check
    // saw it, so our own rename hits ENOENT. During `build`, ENOENT
    // means `srcDir` is missing — that's a real failure, not a race.
    const isRaceLoss =
      code &&
      (RACE_CODES.has(code) || (phase === 'rename' && code === 'ENOENT'));
    if (isRaceLoss) {
      logger.debug(
        { err, dstDir, phase },
        'atomic dir publish raced with concurrent caller; keeping winning copy',
      );
      return;
    }
    throw err;
  }
}

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
 *   - Real credentials (API keys, OAuth tokens, etc.). Composio uses
 *     the project-scoped `ak_*` key (`COMPOSIO_API_KEY`, header
 *     `x-api-key`, #107) for BOTH surfaces it exposes to us:
 *       - REST (`backend.composio.dev/api/v3`) — read by the
 *         inline-fetch path in `tessl__composio-fetch`'s precheck
 *         (admin tile).
 *       - A headless custom MCP server
 *         (`backend.composio.dev/v3/mcp/<id>/mcp`, also `x-api-key`)
 *         whose URL lives in `COMPOSIO_MCP_URL` — read by the agent
 *         runner's MCP server registration so `mcp__composio__*` tools
 *         authenticate.
 *     The consumer "Connect" gateway (`connect.composio.dev/mcp`)
 *     migrated to interactive AuthKit-JWT OAuth and can't run in an
 *     unattended container, so the old `ck_*` `COMPOSIO_MCP_KEY` is
 *     gone — both surfaces now share the one `ak_*` key.
 *   - Account-identifying values that aren't strictly credentials but
 *     would let an observer correlate the container to a specific user
 *     account at the upstream provider (`COMPOSIO_USER_ID` per #509 —
 *     binds Composio calls to a specific user's connected accounts;
 *     leaking it on `docker ps` would identify the account even
 *     though the value is not a credential on its own).
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
  'COMPOSIO_API_KEY',
  // Composio headless custom-MCP-server URL (`/v3/mcp/<id>/mcp`). Read
  // by the agent runner's MCP server registration; the embedded server
  // id is account/project-identifying, so it gets the same env-file
  // treatment as COMPOSIO_USER_ID to stay off `ps`/`docker ps`. Auth is
  // COMPOSIO_API_KEY via `x-api-key` — there is no separate MCP key.
  'COMPOSIO_MCP_URL',
  // Composio user_id bound to the connected accounts in the project
  // the COMPOSIO_API_KEY authenticates as. Not a credential per se —
  // identifies WHICH user's connections to act against — but it's
  // account-identifying and gets the same env-file treatment as the
  // API key so it doesn't appear on `ps`/`docker ps` output. Required
  // by `tessl__composio-fetch`'s precheck (admin tile, jbaruch/nanoclaw#509)
  // to do the fetch inline via Composio REST instead of waking the LLM.
  'COMPOSIO_USER_ID',
  // Fine-grained GitHub PAT for the `gh` CLI inside main/trusted-tier
  // containers. Same .env entry the host-side `github_backup` IPC
  // handler uses for `git push`; forwarding it into the container lets
  // skills run `gh issue list/edit/comment` directly without going
  // through Composio's MCP catalog (avoids cache_create on the GitHub
  // tool schemas for high-fire-count skills like the cost-monitor
  // dashboard family — `precheck-gating-monitor`, `session-cap-monitor`,
  // `daily-spend-rollup`). `gh` reads `GITHUB_TOKEN` automatically —
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
  // `youtube-comment-check` skill (Composio's YouTube toolkit has no
  // comment-threads tool, so it calls the native API directly per
  // jbaruch/nanoclaw-admin#339). Standard `AIzaSy...` key; goes through
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
  // `verify-sessionize.py` for live per-slug CFP-deadline verification
  // (the host-side `sessionize_get_events` IPC handler reads the same
  // .env entry; forwarding it lets the deterministic driver do the
  // round-trip in-container without IPC). Same env-file treatment.
  'SESSIONIZE_EVENT_API_KEY',
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
  // success flag drives finally-block cleanup without a catch-all:
  // exceptions from writeFileSync/closeSync propagate naturally, and
  // finally unlinks the on-disk tempfile if the write didn't fully
  // succeed. Without this, a write failure (disk full, EIO, EDQUOT)
  // would leak a partial-secret tempfile because the outer caller
  // never sees a cleanup callback from a throwing buildSecretEnvFile.
  let writeSucceeded = false;
  try {
    fs.writeFileSync(fd, lines.join('\n') + '\n');
    writeSucceeded = true;
  } finally {
    fs.closeSync(fd);
    if (!writeSucceeded) {
      // unlink failures other than ENOENT are logged but don't mask
      // the original error — the outer try is propagating the real
      // cause via finally semantics.
      try {
        fs.unlinkSync(tmpPath);
      } catch (unlinkErr) {
        const code = (unlinkErr as NodeJS.ErrnoException).code;
        if (code !== 'ENOENT') {
          logger.warn(
            { err: unlinkErr, tmpPath },
            'Failed to clean up secret env-file after write error',
          );
        }
      }
    }
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
        if ((err as NodeJS.ErrnoException).code === 'ENOENT') return;
        logger.warn(
          { err, tmpPath },
          'Failed to clean up secret env-file; will be cleared on next reboot via tmpdir',
        );
      }
    },
  };
}

/**
 * Model the agent-runner passes to the SDK's `query()` call. Forwarded as
 * the `AGENT_MODEL` env var on container spawn and read by
 * `container/agent-runner/src/index.ts`. Bumping this constant is the single
 * source of truth for the container agent's model — no rebuild of the
 * agent-runner image needed.
 *
 * Format: SDK model alias (`opus`, `sonnet[1m]`) or full model ID
 * (`claude-opus-4-7[1m]`). See
 * `container/agent-runner/node_modules/@anthropic-ai/claude-agent-sdk/sdk.d.ts`
 * for the `model` field on `Options`.
 *
 * NOTE: changing the model family may require matching changes in
 * agent-runner's `query()` call. The current default (Opus 4.8) and its
 * predecessor 4.7 both take `thinking: { type: 'adaptive', display:
 * 'summarized' }` (manual `type: 'enabled'` is rejected; `display` would
 * otherwise default to `'omitted'` and silently empty out thinking
 * content) and run on the env-driven `xhigh` effort, not `effort: 'max'`.
 * The runner is set up for these expectations — re-verify them before a
 * cross-family bump (Sonnet/Haiku already degrade `xhigh` → `high`
 * gracefully in the SDK).
 */
// Operators can override at deploy time without editing source — handy for
// running a fork on a cheaper model (Sonnet) without forking just to change
// this one constant. If unset, the default below is what the upstream
// runner is tuned for. AGENT_EFFORT (alongside this) is already env-
// overridable in the agent-runner via VALID_AGENT_EFFORTS.
//
// The `[1m]` suffix on the default model selects the 1M-token extended-
// context tier. Long conversations and large per-turn payloads (transcript
// archives, multi-message digests) rely on it; if you override AGENT_MODEL
// to a different model that supports extended context, include `[1m]` to
// match. Models without the suffix run the standard context window and
// will surface as truncation / earlier compaction in long sessions.
//
// Light validation: trim whitespace (so `AGENT_MODEL="  "` falls back to
// the default rather than passing two spaces to the SDK) and warn on
// values that don't look like a Claude model ID. We don't enumerate a
// whitelist because the SDK accepts both aliases (`opus`, `sonnet[1m]`)
// and full IDs (`claude-opus-4-7[1m]`), the set churns with each model
// release, and a missed model would block legit upgrades. The warn
// surfaces typos at startup instead of at first `query()` call deep in
// runtime — a typo like `claud-opus-4-7` is operator-error territory but
// cheap to flag.
/**
 * @internal Exported so tests can assert against the same literal the
 *   helper returns, instead of duplicating the string in two places where
 *   a default-model bump could silently drift.
 */
export const DEFAULT_AGENT_MODEL = 'claude-opus-4-8[1m]';
const KNOWN_MODEL_PREFIX_RE = /^(claude|opus|sonnet|haiku)/i;
export function resolveAgentModel(raw: string | undefined): string {
  const trimmed = raw?.trim();
  if (!trimmed) return DEFAULT_AGENT_MODEL;
  if (!KNOWN_MODEL_PREFIX_RE.test(trimmed)) {
    logger.warn(
      { agentModel: trimmed, fallback: DEFAULT_AGENT_MODEL },
      'AGENT_MODEL does not look like a Claude model ID — will pass to SDK as-is, but check for a typo. Expected forms: full ID like "claude-opus-4-7[1m]" or alias like "opus" / "sonnet[1m]".',
    );
  }
  return trimmed;
}

/**
 * Per-trust-tier base model (#613 Stage 1 — Claude tier-down). The base
 * model for EVERY container spawn — inbound chat AND scheduled/maintenance
 * runs alike — keyed to the spawn's trust tier instead of every tier
 * defaulting to Opus:
 *
 *   - main      → the global default (`AGENT_MODEL`) — quality floor kept
 *   - trusted   → Sonnet 4.6 (near-Opus reasoning, materially cheaper)
 *   - untrusted → Haiku 4.5 (low-stakes; keeps the Claude prompt-cache
 *                 discount on hostile content)
 *
 * Same-family Claude SKUs, so the agent-runner's `query()` config (tuned
 * for Opus, with `xhigh` effort gracefully falling back on Sonnet/Haiku)
 * needs no change. No `[1m]` suffix on the cheaper tiers — the extended-
 * context tier is reserved for main's long sessions.
 *
 * This is the BASE for a spawn: per-group / maintenance / per-task
 * overrides (`resolveSessionAgentModel`) still win on top. Scheduled-task
 * model selection (trivial → Haiku, substantive → Sonnet) is set per-task
 * via each skill's `agentModel:` frontmatter, not here.
 *
 * Exported so tests pin the tier→model mapping independently.
 */
export const TRUSTED_TIER_MODEL = 'claude-sonnet-4-6';
export const UNTRUSTED_TIER_MODEL = 'claude-haiku-4-5-20251001';
export function resolveTierBaseModel(
  isMain: boolean,
  trusted: boolean,
  globalDefault: string,
): string {
  if (isMain) return globalDefault;
  if (trusted) return TRUSTED_TIER_MODEL;
  return UNTRUSTED_TIER_MODEL;
}

/**
 * Resolve a per-group `containerConfig.agentModel` override to the value
 * actually forwarded to the spawned container (#395). Stricter than the
 * global `resolveAgentModel`:
 *
 *   - empty / whitespace-only / undefined / null → fall back to `fallback`
 *     (the global AGENT_MODEL). Same intent as the global helper, so an
 *     operator that clears the per-group field via `set_agent_model`
 *     `null` reverts the group to the global default rather than
 *     surfacing an empty string downstream.
 *
 *   - unknown-prefix value (`'foobar'`, `'claud-opus-4-7'`) → ALSO fall
 *     back to `fallback`, with a warn. The global helper passes
 *     unknown-prefix values through to surface typos at startup; per-
 *     group overrides are set at runtime via IPC by an agent (or
 *     operator) and there's no operator-driven startup audit log to
 *     catch a typo before it kills the next spawn for that group.
 *     Failing closed to the global default keeps the group running
 *     while the warn flags the bad value.
 *
 * Returns the trimmed override on a known prefix, or `fallback` in all
 * other cases. Exported so tests can pin the four branches independently
 * of the global `resolveAgentModel` contract.
 */
export function resolvePerGroupAgentModel(
  raw: string | undefined | null,
  fallback: string,
): string {
  const trimmed = typeof raw === 'string' ? raw.trim() : '';
  if (!trimmed) return fallback;
  if (!KNOWN_MODEL_PREFIX_RE.test(trimmed)) {
    // Phase 3 (#509) callers pass `fallback` = the session-level value
    // (e.g. `maintenanceAgentModel` resolved against the user-facing
    // value), not the global default. The warn message intentionally
    // says "the caller-supplied fallback" rather than "global default"
    // — debugging "I set task X to haik and got Sonnet" is harder when
    // the log claims the value was sent to the global default while
    // the actual fallback was the maintenance override. The `fallback`
    // field in the log payload carries the actual value the caller
    // will route to.
    logger.warn(
      { agentModel: trimmed, fallback },
      'Per-group AGENT_MODEL override does not look like a Claude model ID — falling back to the caller-supplied fallback (see `fallback` field). Expected forms: full ID like "claude-opus-4-7[1m]" or alias like "opus" / "sonnet[1m]".',
    );
    return fallback;
  }
  return trimmed;
}

/**
 * Resolve the effective AGENT_MODEL for a single spawn given the
 * session slot (#509). Per-session-slot model tier with optional
 * per-task override (Phase 3) — currently only the maintenance slot
 * has its own session-level override; user-facing `'default'` (and
 * any future named slot) falls through to `agentModel` → AGENT_MODEL
 * → DEFAULT_AGENT_MODEL.
 *
 * Resolution order (highest precedence first):
 *   1. `taskAgentModel` (Phase 3) — per-row override from the
 *      `scheduled_tasks.agent_model` column. Beats every other knob;
 *      fires for any spawn that carries a task-level value (in
 *      practice always a maintenance-session scheduled-task fire).
 *      Validated through `resolvePerGroupAgentModel` against the
 *      maintenance/user-facing-resolved value as the fallback so a
 *      typo doesn't silently jump past the operator's session-level
 *      override.
 *   2. `containerConfig.maintenanceAgentModel` (Phase 2) — applies
 *      only to maintenance spawns. Validated against the
 *      user-facing-resolved value as the fallback.
 *   3. `containerConfig.agentModel` (#395) — per-group override.
 *      Validated against `globalDefault`.
 *   4. `globalDefault` — the orchestrator's `AGENT_MODEL` env.
 *
 * Non-maintenance spawns skip step 2 but otherwise follow the same
 * ladder.
 *
 * Returns `{ effective, source }` so the per-spawn audit log line
 * (#418) can attribute the value to the right config layer without
 * the caller re-deriving the comparison. `task_override` is the
 * new Phase 3 source-tag; the prior three values are unchanged.
 */
export function resolveSessionAgentModel(
  containerConfig:
    | { agentModel?: string; maintenanceAgentModel?: string }
    | undefined,
  isMaintenance: boolean,
  globalDefault: string,
  taskAgentModel?: string | null,
): {
  effective: string;
  source:
    | 'global_default'
    | 'group_override'
    | 'maintenance_override'
    | 'task_override';
} {
  const userFacingRaw = containerConfig?.agentModel;
  const userFacingResolved = userFacingRaw
    ? resolvePerGroupAgentModel(userFacingRaw, globalDefault)
    : globalDefault;
  const userFacingSource: 'group_override' | 'global_default' =
    userFacingResolved !== globalDefault ? 'group_override' : 'global_default';

  // Compute the session-level value first (step 2 if maintenance, else
  // step 3) so the per-task override has a coherent fallback when its
  // own raw value fails resolvePerGroupAgentModel's prefix check.
  let sessionLevelResolved = userFacingResolved;
  let sessionLevelSource:
    | 'group_override'
    | 'global_default'
    | 'maintenance_override' = userFacingSource;
  if (isMaintenance) {
    const maintenanceRaw = containerConfig?.maintenanceAgentModel;
    if (maintenanceRaw && maintenanceRaw.trim()) {
      const maintenanceResolved = resolvePerGroupAgentModel(
        maintenanceRaw,
        userFacingResolved,
      );
      if (maintenanceResolved !== userFacingResolved) {
        sessionLevelResolved = maintenanceResolved;
        sessionLevelSource = 'maintenance_override';
      }
      // else: unknown-prefix fall-through warned by resolvePerGroupAgentModel,
      // OR maintenance value deliberately matches user-facing — either way
      // there's no effective maintenance-specific routing, so leave the
      // session-level pair at the user-facing values.
    }
  }

  // Step 1 — per-task override beats everything else. A typo / unknown
  // prefix falls back through resolvePerGroupAgentModel to the
  // session-level value, NOT all the way to globalDefault — so an
  // operator who already set a deliberate maintenance override sees a
  // bad per-task value land on maintenance, not on the global default.
  if (taskAgentModel && taskAgentModel.trim()) {
    const taskResolved = resolvePerGroupAgentModel(
      taskAgentModel,
      sessionLevelResolved,
    );
    if (taskResolved !== sessionLevelResolved) {
      return { effective: taskResolved, source: 'task_override' };
    }
    // Unknown-prefix or deliberately-matches-session — no effective
    // task-specific routing happening; emit the session-level pair.
  }

  return { effective: sessionLevelResolved, source: sessionLevelSource };
}

const AGENT_MODEL = resolveAgentModel(process.env.AGENT_MODEL);

/**
 * Effort level the agent-runner passes to the SDK's `query()` call.
 * Forwarded as `AGENT_EFFORT` on container spawn.
 *
 * Resolution order (from highest precedence):
 *   1. `AGENT_EFFORT` environment variable on the orchestrator process
 *   2. The hardcoded default below (`xhigh`)
 *
 * Operators can override at runtime by setting `AGENT_EFFORT` in the
 * orchestrator's env (e.g. docker-compose, systemd unit), no rebuild
 * needed. The agent-runner validates the forwarded value against the
 * allowed set and falls back to `xhigh` on invalid input, so a typo
 * here won't crash containers — it'll log a warning inside the runner.
 *
 * Valid values: `'low' | 'medium' | 'high' | 'xhigh' | 'max'`.
 * - On Opus 4.7: `xhigh` is Anthropic's recommended default for
 *   coding/agentic work; `max` is reserved for frontier problems.
 * - On Opus 4.6 / Sonnet 4.6: `xhigh` silently falls back to `high` in
 *   the SDK.
 * See https://docs.anthropic.com/en/docs/build-with-claude/effort
 */
const AGENT_EFFORT = process.env.AGENT_EFFORT || 'xhigh';

/**
 * Try to repair the source `messages.db` WAL/SHM state by running a
 * TRUNCATE checkpoint via a short-lived dedicated connection. Used as
 * a recovery path when `createFilteredDb`'s ATTACH throws "disk I/O error"
 * — most commonly when the source's `-shm` file has been evicted (Docker
 * bind-mount on macOS, OS resource pressure, or partial writer crash) and
 * SQLite can no longer establish WAL read coordination.
 *
 * After this call returns, the next ATTACH attempt should succeed
 * assuming the underlying filesystem is healthy. Idempotent and safe to
 * call when state is already clean.
 *
 * Logged loudly because hitting this path indicates an environment
 * problem (Docker bind-mount eviction, OS resource pressure, partial
 * writer crash); the operator should know it ran. See issue #100.
 */
function recoverSourceWalState(srcDb: string): void {
  logger.warn(
    { srcDb },
    'createFilteredDb: source WAL/SHM state appears degraded, running checkpoint(TRUNCATE) recovery',
  );
  const recovery = new Database(srcDb);
  try {
    recovery.pragma('busy_timeout = 5000');
    // TRUNCATE checkpoint forces all -wal content into the main db
    // and zero-truncates -wal. As a side effect the writer connection
    // re-establishes -shm coordination. PASSIVE/RESTART would also
    // work, but TRUNCATE is the most aggressive and most likely to
    // un-stick a bad state.
    recovery.pragma('wal_checkpoint(TRUNCATE)');
  } finally {
    recovery.close();
  }
}

/**
 * Per `coding-policy: error-handling`: narrow to the specific exception
 * type we expect, rethrow everything else. The retry path only triggers
 * when better-sqlite3 throws a typed `SqliteError` carrying the "disk
 * I/O error" substring (the symptom of degraded source WAL/SHM
 * coordination per `ligolnik#100`). Anything else — a wrapped error, a
 * mocked error in tests, a totally unrelated `Error` whose message
 * happens to mention disk I/O — gates `false` so the call site rethrows
 * instead of running a recovery that doesn't apply.
 */
function isDiskIoError(err: unknown): err is SqliteError {
  return err instanceof SqliteError && err.message.includes('disk I/O error');
}

/**
 * Create a filtered copy of messages.db containing only one group's messages.
 * Returns the path to the filtered DB, or null if the source DB doesn't exist.
 *
 * @internal Exported for tests only — untrusted-group DB isolation is
 *   security-critical and must be pinned by regression tests.
 */
export function createFilteredDb(
  chatJid: string,
  groupFolder: string,
): string | null {
  const srcDb = path.join(STORE_DIR, 'messages.db');
  if (!fs.existsSync(srcDb)) return null;

  const filteredDir = path.join(DATA_DIR, 'filtered-db', groupFolder);
  fs.mkdirSync(filteredDir, { recursive: true });
  const filteredPath = path.join(filteredDir, 'messages.db');

  // Remove stale copy from previous run, including any `-wal`/`-shm`
  // sidecars left behind by a pre-#287 version that ran before the
  // `journal_mode = DELETE` pragma below was in place. Without this,
  // operators upgrading on top of an existing data dir keep the old
  // WAL artefacts indefinitely — both as wasted disk and as the same
  // RO-mount-can't-open failure the pragma is supposed to eliminate.
  // Sidecars are removed unconditionally (independent of whether the
  // main file existed) because a partial wipe — main DB removed but
  // sidecars left — is the exact state SQLite refuses to open.
  // `rmSync({ force: true })` so concurrent refreshes (default +
  // maintenance session for the same untrusted group) tolerate
  // another caller having already removed one or more of these paths.
  for (const suffix of ['', '-wal', '-shm']) {
    fs.rmSync(`${filteredPath}${suffix}`, { force: true });
  }

  /**
   * One full ATTACH + CTAS + atomic-rename attempt. Captured as a
   * closure so we can retry it once after `recoverSourceWalState`
   * (see #100). Each call regenerates its own temp path so a retry
   * never reuses a stale file from the failed first attempt.
   */
  const attemptCreate = (): void => {
    // Atomic temp-file + rename, see #93. Writing the schema directly to the
    // canonical path means any interruption (SIGTERM, OOM, fs hiccup, throw mid
    // ATTACH/CTAS) leaves a 0-byte or partial-header file there. Subsequent
    // spawns then hit "disk I/O error" the moment SQLite tries to read the
    // (missing) header, and the group is permanently wedged behind the circuit
    // breaker. Writing to a sibling temp path and renaming on success keeps the
    // canonical path either healthy or absent — never half-written. The temp
    // file MUST live in the same directory so the rename stays atomic on POSIX.
    // The trailing randomBytes(4) suffix defeats retry-collision: if the first
    // attempt failed mid-flight and the second attempt fires within the same
    // millisecond, pid+Date.now() alone could collide with the prior temp path.
    const tempPath = path.join(
      filteredDir,
      `messages.db.tmp-${process.pid}-${Date.now()}-${randomBytes(4).toString('hex')}`,
    );
    // Stale temp from a previously crashed run — unlink so `new Database` opens
    // a fresh file rather than reattaching to a corrupt one.
    if (fs.existsSync(tempPath)) {
      fs.unlinkSync(tempPath);
    }

    // Use ATTACH to copy schema-agnostically — picks up new columns automatically
    const dst = new Database(tempPath);
    // Source `messages.db` is WAL-mode and actively written by the orchestrator.
    // Without busy_timeout this connection would fail immediately on any lock
    // contention against the source (e.g. during a checkpoint), defeating the
    // whole point of the orchestrator-side WAL setup. Match the orchestrator
    // value (5000ms) so contention smoothing is symmetric across readers.
    dst.pragma('busy_timeout = 5000');
    // Force rollback-journal mode on the snapshot. better-sqlite3 defaults to
    // WAL, which requires the SQLite reader to write `-wal`/`-shm` sidecar
    // files even on opens that are logically read-only. The filtered DB is
    // mounted read-only into untrusted containers (via `fakeowner ro`); a
    // default `sqlite3.connect(path)` from inside the container then fails
    // with `unable to open database file` because the sidecars can't be
    // created. DELETE-journal makes the file self-contained — every reader's
    // default open works without per-script `?mode=ro&immutable=1` plumbing.
    // The filtered DB is a single-writer one-shot snapshot, so WAL gives it
    // nothing anyway. See issue #287.
    dst.pragma('journal_mode = DELETE');
    dst.pragma('synchronous = NORMAL');
    try {
      try {
        dst.exec(`ATTACH DATABASE '${srcDb.replace(/'/g, "''")}' AS src`);
        dst.exec(
          `CREATE TABLE chats AS SELECT * FROM src.chats WHERE jid = '${chatJid.replace(/'/g, "''")}'`,
        );
        dst.exec(
          `CREATE TABLE messages AS SELECT * FROM src.messages WHERE chat_jid = '${chatJid.replace(/'/g, "''")}'`,
        );
        dst.exec(
          'CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)',
        );
        // Reactions scoped to this chat only. Filtered-DB consumers JOIN
        // on this table; without it, those joins hit "no such table:
        // reactions" and abort. Created unconditionally so containers
        // don't depend on whether the host happens to have any reactions
        // yet — even an empty table satisfies the join. CTAS can't run
        // if src.reactions doesn't exist (fresh install before
        // migrations), so check `src.sqlite_master`
        // explicitly and fall back to an empty table with the known schema in
        // that one case. Bare try/catch would also swallow corruption, lock,
        // and permission errors — a missing table is the only fallback case
        // we want to absorb.
        const srcHasReactions = dst
          .prepare(
            "SELECT 1 FROM src.sqlite_master WHERE type = 'table' AND name = 'reactions' LIMIT 1",
          )
          .get();
        if (srcHasReactions) {
          dst.exec(`
            CREATE TABLE reactions AS
              SELECT r.* FROM src.reactions r
              WHERE r.message_chat_jid = '${chatJid.replace(/'/g, "''")}'
          `);
        } else {
          dst.exec(`
            CREATE TABLE reactions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              message_id TEXT NOT NULL,
              message_chat_jid TEXT NOT NULL,
              reactor_jid TEXT NOT NULL,
              reactor_name TEXT NOT NULL,
              emoji TEXT NOT NULL,
              timestamp TEXT NOT NULL
            )
          `);
        }
        dst.exec(
          'CREATE INDEX IF NOT EXISTS idx_reactions_message ON reactions(message_id, message_chat_jid)',
        );
        dst.exec('DETACH src');
      } finally {
        dst.close();
      }
      // Atomic rename — temp file is now a fully-formed SQLite database. On
      // POSIX this is atomic within the same filesystem, so the canonical path
      // flips from "absent or stale" to "complete" with no observable midpoint.
      fs.renameSync(tempPath, filteredPath);
    } catch (err) {
      // Cleanup the partial temp file before rethrowing. Best-effort — if the
      // unlink itself fails for anything other than ENOENT (file already gone),
      // log it so we know about latent disk/permission issues, but don't
      // shadow the original error. The temp path lives next to the canonical
      // path, so leaving it around would also leak disk space across retries.
      try {
        fs.unlinkSync(tempPath);
      } catch (cleanupErr) {
        const code = (cleanupErr as NodeJS.ErrnoException).code;
        if (code !== 'ENOENT') {
          logger.warn(
            { err: cleanupErr, tempPath },
            'Failed to clean up partial filtered-db temp file',
          );
        }
      }
      throw err;
    }
  };

  // Outer try with a single retry on "disk I/O error" — the symptom of
  // a degraded source WAL/SHM state (issue #100). Recovery runs a
  // wal_checkpoint(TRUNCATE) on the source via a fresh connection, which
  // re-establishes the -shm region; the retry then attempts the full
  // ATTACH+CTAS again. Retry happens EXACTLY ONCE — if it also fails the
  // error propagates and the circuit breaker can do its job (the FS or
  // data is genuinely broken at that point, not transient SHM eviction).
  //
  // Errors that are NOT disk-I/O (e.g. "no such table", schema bugs,
  // unparseable source) propagate immediately without recovery — there's
  // nothing a checkpoint can fix for those.
  try {
    attemptCreate();
  } catch (err) {
    if (!isDiskIoError(err)) throw err;
    recoverSourceWalState(srcDb);
    attemptCreate(); // retry once; any error here propagates
    logger.info(
      { srcDb, groupFolder },
      'createFilteredDb: succeeded after WAL/SHM recovery + retry',
    );
  }

  // Chown so container user can read
  const uid = HOST_UID ?? 1000;
  const gid = HOST_GID ?? 1000;
  if (uid !== 0) {
    try {
      fs.chownSync(filteredDir, uid, gid);
      fs.chownSync(filteredPath, uid, gid);
    } catch (err: unknown) {
      logger.warn({ err, filteredPath }, 'Failed to chown filtered DB');
    }
  }

  logger.debug(
    { chatJid, groupFolder, path: filteredPath },
    'Created filtered DB for untrusted container',
  );

  return filteredPath;
}

export interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isTrusted?: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
  script?: string;
  replyToMessageId?: string;
  /**
   * #479 sub-#1: trigger message id for usage attribution (the
   * gate-allowed message in `evaluateGateChain`'s verdict). Distinct
   * from `replyToMessageId`, which is the reply-target — usually the
   * latest message in a polled batch — for reply-threading. The two
   * differ when the gate allows on an earlier message in a batch and
   * the reply quotes the latest. The credential proxy logs this as
   * `message_id` on each JSONL line; `replyToMessageId` is used as a
   * fallback when no gate verdict was computed (empty gate chain).
   */
  triggerMessageId?: string;
  /**
   * Whether the inbound batch is "addressed to us" — drives the
   * agent-runner's react-first 👀 gate (#289). Resolved by the
   * orchestrator (see `isAddressedToUs` in `src/index.ts`) from
   * isMain / 1:1-DM / trigger-match / reply-to-our-bot, independent
   * of `requires_trigger`. Omitted for non-channel paths (scheduled
   * tasks, scripts) — agent-runner treats `undefined` as "no
   * addressed-ness signal" and falls back to its existing skips.
   */
  addressedToUs?: boolean;
  /**
   * Provenance of a scheduled task (undefined for non-scheduled runs).
   * Drives whether the agent-runner wraps the prompt in `<untrusted-input>`.
   * Only `'untrusted_agent'` triggers the wrap; owner/main/trusted bypass
   * because their content originates from a trusted source.
   *
   * Security boundary: the task-scheduler passes this from the DB row's
   * `created_by_role` column, which was itself set by ipc.ts from the
   * VERIFIED source-group trust tier at schedule_task time. The agent
   * that scheduled the task never got to claim its own role — so an
   * untrusted agent that self-scheduled a prompt sees it come back
   * wrapped on the next fire.
   */
  createdByRole?: 'owner' | 'main_agent' | 'trusted_agent' | 'untrusted_agent';
  /**
   * Which per-group session this container run belongs to. Drives the
   * `.claude/` dir location and the group-queue slot key.
   * - `'default'` (omitted): user-facing AyeAye, serves inbound IPC messages.
   * - `'maintenance'`: scheduled AyeAye, runs scheduled_tasks (heartbeat,
   *   nightly, weekly, reminders). Runs in parallel with `'default'` for the
   *   same group so maintenance never blocks user replies.
   *
   * Invariant: inbound Telegram/etc. messages ALWAYS route to `'default'`.
   * `src/task-scheduler.ts` is the sole writer of `'maintenance'`.
   */
  sessionName?: string;
  /**
   * Continuation marker for self-resuming cycles (#93/#130). Set only on
   * scheduled-task spawns whose `scheduled_tasks.continuation_cycle_id`
   * column is non-NULL. When present the spawned container gets:
   *   - `NANOCLAW_CONTINUATION=1`
   *   - `NANOCLAW_CONTINUATION_CYCLE_ID=<value>`
   *
   * Absence (undefined / empty) means "fresh invocation" — neither env var
   * is set, and that absence is itself the signal the calling skill
   * (resumable-cycle / nightly-housekeeping / etc.) cross-checks against
   * its prompt-prefix marker. Mismatch fails closed to fresh invocation;
   * a scheduler that sets the env but mangles the prompt (or vice versa)
   * therefore never silently bypasses the two-phase lock acquisition.
   */
  continuationCycleId?: string;
  /**
   * Per-task AGENT_MODEL override for #509 Phase 3. Set by the
   * task-scheduler from `scheduled_tasks.agent_model` on this row.
   * When present and non-empty after trim, beats every other knob in
   * the resolveSessionAgentModel ladder (maintenanceAgentModel,
   * group agentModel, AGENT_MODEL env, DEFAULT_AGENT_MODEL); audit
   * log emits `task_override` as the source. Unknown-prefix values
   * fall back to the session-level value via `resolvePerGroupAgentModel`
   * — never silently jump to the global default. Undefined for
   * non-scheduled-task spawns (interactive inbound, IPC scripts).
   */
  taskAgentModel?: string | null;
}

export interface ContainerOutput {
  // `'precheck_skipped'` (#581) — emitted by the agent-runner when the
  // scheduled-task precheck script returned `wake_agent: false`. The
  // agent never woke; the wrapper never ran. Distinct from `'success'`
  // so the silent-success watchdog can tell a precheck-gated no-op
  // (healthy quiet) apart from an agent that woke and produced an
  // empty result (the original #581 bug). A precheck script that
  // crashes / emits non-JSON / omits `wake_agent` is `'error'`, not
  // `'precheck_skipped'`.
  //
  // `'killed'` (#589 reopened / #682) — resolved host-side (never
  // emitted by the agent-runner) when a maintenance container ends
  // without delivering a terminal result. A healthy maintenance
  // one-shot reaches a terminal result and exits naturally
  // (scheduleClose → `_close`) within seconds, so a run that produced
  // only streaming previews and then either (a) was reaped by the
  // MAINTENANCE_CONTAINER_TIMEOUT inactivity timer or (b) exited
  // cleanly (code 0) was incomplete — reaped/exited mid-compose, not
  // the misleading `'success'` that hid the original #589 silent-stop.
  // #682 gates this on the precise `hadTerminalResult` signal (a marker
  // with no `streamText`), so a run that DID deliver a terminal result
  // and then idled out keeps its `'success'` — only a no-terminal-
  // result end is `'killed'`. Incomplete + retriable / redeliverable.
  status: 'success' | 'error' | 'killed' | 'precheck_skipped';
  result: string | null;
  newSessionId?: string;
  error?: string;
  streamText?: string;
  /**
   * #581 — Set to `true` by the agent-runner when (a) the agent
   * successfully used `send_message` / `send_file` during this turn
   * AND (b) the SDK's final result message carried non-empty text.
   * The two-condition gate matches the case the suppression actually
   * targets: the SDK's closing-thought text that would otherwise be
   * echoed as a second user reply on top of the agent's `send_message`.
   * When `chat_displayed` is `true`, the orchestrator (`src/index.ts`)
   * and the task-scheduler (`src/task-scheduler.ts`) skip their
   * chat-echo + `storeMessage` paths but STILL populate
   * `task_run_logs.result` from `result` so observability isn't lost.
   * When `chat_displayed` is absent/false, behavior is unchanged from
   * the pre-#581 contract — including the case where `send_message`
   * succeeded but `textResult` was empty (the orchestrator's outer
   * `if (result.result)` gate already skips chat-echo there).
   */
  chat_displayed?: boolean;
  /**
   * Per-turn token usage from the agent-runner's most recent SDK
   * assistant message. Used by the kill-auto-compaction telemetry +
   * threshold detector (issue #104, design at
   * `docs/proposals/kill-auto-compaction.md`). Optional because
   * non-assistant outputs (errors before any model turn fires)
   * may not carry a usage payload.
   */
  usage?: {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
  };
  /**
   * #689 — stamped `true` by the agent-runner on a TERMINAL `success`
   * marker when a `requires_delivery` skill (morning-brief) ended a
   * runQuery without delivering any user-facing content. The status
   * stays `success` so this container's `scheduleClose` still drains
   * the maintenance slot promptly (no #461 wedge), but the close
   * handler below resolves the run as `killed` (incomplete / retriable)
   * rather than recording the synthesized success as a real delivery —
   * the silent success of the #589/#682 lineage #689 reopened. Absent
   * on every marker from a skill that didn't declare `requires_delivery`
   * and on any run that did deliver. See the agent-runner's
   * `delivery-requirement.ts`.
   */
  noDelivery?: boolean;
}

interface VolumeMount {
  hostPath: string;
  containerPath: string;
  readonly: boolean;
}

/**
 * Recursively chown a host-side directory, NEVER following symlinks.
 *
 * Security: earlier implementation used `fs.chownSync` which follows
 * symlinks. If a container could create a symlink inside one of its
 * writable mounts (e.g. `shared-memory/evil` → `/etc/passwd`), the next
 * spawn's chown would change ownership of the target — a privilege
 * escalation path out of the container into the host. `lchownSync`
 * operates on the link itself; `withFileTypes: true` + `entry.isDirectory()`
 * only recurses into real directories, so symlinks are chowned but not
 * traversed.
 */
function chownRecursive(dir: string, uid: number, gid: number): void {
  fs.lchownSync(dir, uid, gid);
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    fs.lchownSync(fullPath, uid, gid);
    // `isDirectory()` returns false for symlinks (even symlinks to dirs)
    // because we used `withFileTypes: true`, which reads the dirent type
    // without resolving the link. Recursion is therefore symlink-safe.
    if (entry.isDirectory()) {
      chownRecursive(fullPath, uid, gid);
    }
  }
}

/**
 * Translate a local container path to a host path for docker -v arguments.
 * In Docker-out-of-Docker, the orchestrator's filesystem (/app/...) differs
 * from the host's (HOST_PROJECT_ROOT/...). Mount paths must use host paths.
 */
function toHostPath(localPath: string): string {
  const projectRoot = process.cwd();
  if (HOST_PROJECT_ROOT === projectRoot) return localPath; // running directly on host
  const rel = path.relative(projectRoot, localPath);
  if (rel.startsWith('..') || path.isAbsolute(rel)) return localPath; // outside project
  return path.join(HOST_PROJECT_ROOT, rel);
}

/**
 * Files in the project root that contain secrets (bot tokens, API keys).
 * Main-group containers get `/dev/null` mounted over each of these so agents
 * can't read tokens and bypass the credential proxy.
 *
 * Security-critical: adding a new secret file ANYWHERE in the repo requires
 * adding it to this list, or an agent in the main group can read it.
 */
export const SECRET_FILES = [
  '.env',
  '.env.bak',
  'data/env/env',
  'scripts/heartbeat-external.conf',
] as const;

/**
 * Default `sessionName` when callers don't pass one. User-facing paths
 * (inbound IPC messages) resolve here. Scheduled tasks pass `'maintenance'`
 * to get a parallel container slot. See `ContainerInput.sessionName` docs.
 */
export const DEFAULT_SESSION_NAME = 'default';

/**
 * Canonical session name for scheduled work (heartbeat, nightly, weekly,
 * reminders). `src/task-scheduler.ts` is the sole writer of this value;
 * no inbound path ever reaches it. Defined here (not in `group-queue.ts`
 * where it used to live) so the install-loop in `buildVolumeMounts`
 * below can reference it directly without creating a
 * `container-runner ↔ group-queue` import cycle (#337 review). The
 * symbol is re-exported from `group-queue.ts` for callers that already
 * import it from there — no other file needs to change.
 */
export const MAINTENANCE_SESSION_NAME = 'maintenance';

/**
 * Per-session subdir name under `<DATA_DIR>/ipc/<folder>/` for the input
 * side of the IPC channel. Each session gets its own subdir so `_close`
 * sentinels and follow-up JSON messages written for one session never
 * leak into the other session's container.
 *
 * Exported so group-queue writes to the same path the container-runner
 * mounted — both must agree on the location. Kept in sync at compile time.
 *
 * Session name is validated here so every caller (orchestrator-trusted
 * and IPC-untrusted alike) gets the same guard. A malicious container
 * that manages to stamp `sessionName: "../default"` onto its IPC request
 * would, without this check, redirect `scriptResultPath` or mount
 * construction into a directory outside the expected `ipc/<group>/`
 * subtree. The allowlist pattern is deliberately narrow — `default`,
 * `maintenance`, and any hypothetical future slot all fit within
 * `[A-Za-z0-9_-]+`.
 */
const VALID_SESSION_NAME_RE = /^[A-Za-z0-9_-]+$/;
export function sessionInputDirName(sessionName: string): string {
  if (!VALID_SESSION_NAME_RE.test(sessionName)) {
    throw new Error(
      `Invalid session name: ${JSON.stringify(sessionName)} — must match ${VALID_SESSION_NAME_RE}`,
    );
  }
  return `input-${sessionName}`;
}

/**
 * Claude Code's SDK slugifies each project's working directory path into a
 * subdirectory name under `~/.claude/projects/`. For our container the
 * project root is `/workspace/group`, so the slug is `-workspace-group`
 * (slashes replaced with leading dashes). The SDK writes transcripts,
 * feedback, and memory under this path.
 *
 * We bind-mount a shared `memory/` subdir inside it (see `buildVolumeMounts`)
 * so auto-memory is owner-level state, not per-session — otherwise feedback
 * written to one session's `.claude/` is invisible to the other.
 *
 * If Claude Code ever changes its slug convention, update this const.
 * Graceful degradation if it does drift: the mount target mismatches the
 * SDK's path and auto-memory falls back to per-session (pre-PR-#57
 * behaviour) — annoying, not broken.
 */
const CLAUDE_PROJECT_SLUG = '-workspace-group';

/**
 * Publish files from a tile's `<skill>/scripts/` source dir into the
 * group's flat `tmpScriptsDir`. The flat dir is reachable from agents
 * as `/workspace/group/scripts/<name>`, so the publish surface is
 * regular files and symlinks — nothing else. Symlink targets are not
 * inspected; the tile owns whether what its links point to is
 * sensible. Subdirectories (notably Python's `__pycache__/`, written
 * next to a `.py` script after the first import) and any other
 * dirent kind (FIFOs, sockets, devices, which `fs.cpSync` rejects
 * anyway) are skipped via an explicit `isFile() || isSymbolicLink()`
 * allowlist so the spawn never trips on a stray entry the runtime
 * put there.
 *
 * No-op when the source dir doesn't exist (skill ships no scripts).
 *
 * @internal Exported for tests only.
 */
export function copyTileScriptsToFlatDir(srcDir: string, dstDir: string): void {
  if (!fs.existsSync(srcDir)) return;
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    if (!entry.isFile() && !entry.isSymbolicLink()) continue;
    fs.cpSync(path.join(srcDir, entry.name), path.join(dstDir, entry.name));
  }
}

/**
 * @internal Exported for tests only — mount-list construction is
 *   security-critical (trust tiers, secret shadowing, untrusted read-only).
 */
export function buildVolumeMounts(
  group: RegisteredGroup,
  isMain: boolean,
  chatJid: string,
  sessionName: string = DEFAULT_SESSION_NAME,
): VolumeMount[] {
  // Validate `sessionName` at the earliest point it's used as a filesystem
  // path segment. The same allowlist `sessionInputDirName` enforces — kept
  // in sync so no mount can be built with a name that would later be
  // rejected at IPC time, and no caller can smuggle `..` into the sessions
  // dir path (which happens BEFORE `sessionInputDirName` is reached).
  if (!VALID_SESSION_NAME_RE.test(sessionName)) {
    throw new Error(
      `Invalid session name: ${JSON.stringify(sessionName)} — must match ${VALID_SESSION_NAME_RE}`,
    );
  }
  const mounts: VolumeMount[] = [];
  const groupDir = resolveGroupFolderPath(group.folder);

  // Ensure AGENTS.md exists (chains .tessl/RULES.md into Claude Code context).
  // Must be created BEFORE the mount goes read-only for untrusted groups.
  const agentsMdPath = path.join(groupDir, 'AGENTS.md');
  if (!fs.existsSync(agentsMdPath)) {
    fs.writeFileSync(
      agentsMdPath,
      '\n\n# Agent Rules <!-- managed by orchestrator -->\n\n@.tessl/RULES.md follow the [instructions](.tessl/RULES.md)\n',
    );
  }

  if (isMain) {
    // Main gets the project root read-only.
    mounts.push({
      hostPath: toHostPath(process.cwd()),
      containerPath: '/workspace/project',
      readonly: true,
    });
    // Host log artifacts: orchestrator log + per-container streaming
    // logs + state snapshot. Admin-tile only — these files are
    // inherently cross-chat (every group's container output, the
    // orchestrator's own diagnostics) and must not leak to untrusted
    // or trusted-non-main tiles. Read-only by construction so admin
    // can observe but not mutate the host's view of itself.
    //
    // ensureHostLogDirs() is best-effort and returns false on
    // permission / disk errors. Skip the mount if the directory
    // tree didn't materialize — better to lose host-logs visibility
    // than to abort the container spawn entirely. The agent will
    // see no /workspace/host-logs and fall back to chat_status for
    // diagnosis (live data, no historical files).
    if (ensureHostLogDirs()) {
      mounts.push({
        hostPath: toHostPath(hostLogsDir()),
        containerPath: '/workspace/host-logs',
        readonly: true,
      });
    } else {
      logger.warn(
        { group: group.name },
        'host-logs dir bootstrap failed — admin tile will spawn without /workspace/host-logs',
      );
    }
    // The proxy-side `usage.jsonl` lives in the orchestrator's `logs/`
    // dir (not under data/host-logs/), so it isn't covered by the
    // mount above. Surface it RO at /workspace/proxy-logs/usage.jsonl
    // (a SIBLING of /workspace/host-logs/, NOT a child) so admin-tile
    // prechecks (e.g. classifier-emit verification for #493) can read
    // the same authoritative spend log the orchestrator writes.
    //
    // Why not /workspace/host-logs/usage.jsonl: docker can't bind-
    // mount a file inside a parent that's already RO-mounted (the
    // kernel refuses to create the bind target on a read-only
    // filesystem, OCI runtime returns code 125, container fails to
    // spawn). Reference incident: PR #523 deployed the file mount at
    // /workspace/host-logs/usage.jsonl and broke every main-group
    // (telegram_swarm) spawn for ~5h until the circuit breaker
    // tripped — morning-brief / heartbeat / inbound replies all
    // stopped. The sibling-path keeps both mounts as independent
    // top-level binds.
    //
    // Skip silently if the file doesn't exist yet — fresh-deploy
    // orchestrators haven't emitted any records; the precheck
    // tolerates a missing file.
    const usageLogHost = path.join(process.cwd(), 'logs', 'usage.jsonl');
    if (fs.existsSync(usageLogHost)) {
      mounts.push({
        hostPath: toHostPath(usageLogHost),
        containerPath: '/workspace/proxy-logs/usage.jsonl',
        readonly: true,
      });
    }
    // Shadow ALL files containing secrets so agents can't read bot tokens.
    // Without this, subagents curl the Telegram API directly, bypassing MCP.
    // mount --bind inside the container doesn't work (needs CAP_SYS_ADMIN),
    // so we mount /dev/null over every secret file from the orchestrator.
    for (const relPath of SECRET_FILES) {
      const absPath = path.join(process.cwd(), relPath);
      if (fs.existsSync(absPath)) {
        mounts.push({
          hostPath: '/dev/null',
          containerPath: `/workspace/project/${relPath}`,
          readonly: true,
        });
      }
    }
  }

  // Group folder mount. Untrusted groups get read-only (disk exhaustion protection).
  mounts.push({
    hostPath: toHostPath(groupDir),
    containerPath: '/workspace/group',
    readonly: !isMain && !group.containerConfig?.trusted,
  });

  // CLAUDE.md trust-tier mount. The per-group folder is no longer the
  // source of truth for this file — it's a thin pointer (trust marker
  // + @imports of SOUL/MEMORY/RULES/FORMATTING) that depends on the
  // CURRENT trust flag. Mounting it from the global directory at every
  // spawn means a trust flip is reflected on the next message without
  // any reconciliation step (#153). Mounted readonly so the agent can't
  // accidentally diverge it from the template; per-group memory now
  // lives in MEMORY.md (writable for trusted/main, readonly for
  // untrusted by virtue of the group folder mount above).
  //
  // For main the source is the git-managed canonical template at
  // `groups/main/CLAUDE.md` — NOT the per-group folder's CLAUDE.md.
  // The previous code hardcoded `path.join(groupDir, 'CLAUDE.md')`
  // on the assumption that the per-group folder always has a copy
  // because it's "git-managed," but that's only true for the literal
  // `groups/main/` folder. A registered main group whose folder name
  // is something else (e.g. `groups/telegram_swarm/`) is gitignored
  // runtime state with no automatic bootstrap of CLAUDE.md, so the
  // file was missing and the WARN below fired on every spawn until
  // an operator hand-copied the template. Pointing the source at the
  // canonical template makes main symmetric with trusted/untrusted —
  // all three trust tiers now resolve to a git-tracked template that
  // is always on disk regardless of the registered folder name.
  const claudeMdSource = isMain
    ? path.join(GROUPS_DIR, 'main', 'CLAUDE.md')
    : group.containerConfig?.trusted
      ? path.join(GROUPS_DIR, 'global', 'CLAUDE.md')
      : path.join(GROUPS_DIR, 'global', 'CLAUDE-untrusted.md');
  if (fs.existsSync(claudeMdSource)) {
    // The /workspace/group bind-mount above is readonly for untrusted
    // groups, which means runc cannot create a missing target file when
    // overlaying the CLAUDE.md bind on top — the spawn fails with
    // `read-only file system` (see #442). Touch a placeholder on the
    // host (where the group folder is RW from the orchestrator's side)
    // so runc has a target to overlay onto. The placeholder content is
    // irrelevant — the bind-mount shadows it. Gated narrowly to the
    // failing condition: trusted and main both have a RW parent mount
    // so runc creates the target itself. Writing a host-side
    // placeholder for those tiers would also pollute
    // `scripts/migrate-thin-claude-md.ts`, which classifies any
    // non-vanilla `CLAUDE.md` as customized. A future trust flip from
    // trusted → untrusted lands here on the next spawn under the same
    // gate, so no pre-emptive creation needed.
    const groupMountReadonly = !isMain && !group.containerConfig?.trusted;
    if (groupMountReadonly) {
      const placeholderTarget = path.join(groupDir, 'CLAUDE.md');
      if (!fs.existsSync(placeholderTarget)) {
        fs.writeFileSync(placeholderTarget, '');
      }
    }
    mounts.push({
      hostPath: toHostPath(claudeMdSource),
      containerPath: '/workspace/group/CLAUDE.md',
      readonly: true,
    });
  } else {
    // Silent fallback would let the container resolve /workspace/group/CLAUDE.md
    // via the underlying group-folder mount — i.e. whatever stale or customized
    // copy may still be on disk — which is exactly the #153 drift this fix is
    // supposed to eliminate. Log loudly so a mis-deployed install is visible
    // instead of looking healthy until the next trust flip exposes it.
    logger.warn(
      {
        claudeMdSource,
        groupFolder: group.folder,
        isMain,
        trusted: !!group.containerConfig?.trusted,
      },
      'CLAUDE.md trust-tier source missing; /workspace/group/CLAUDE.md will fall back to whatever the group folder contains. Run `git pull` and `scripts/migrate-thin-claude-md.ts --apply`.',
    );
  }

  // Global memory directory (SOUL.md, shared CLAUDE.md).
  // Trusted + main get the full directory. Untrusted get only SOUL-untrusted.md
  // mounted as SOUL.md so core-behavior's "read SOUL.md" still works.
  const globalDir = path.join(GROUPS_DIR, 'global');
  if (isMain || group.containerConfig?.trusted) {
    if (fs.existsSync(globalDir)) {
      mounts.push({
        hostPath: toHostPath(globalDir),
        containerPath: '/workspace/global',
        readonly: !isMain,
      });
    }
  } else {
    // Untrusted: mount only the sanitized SOUL as a single file
    const untrustedSoul = path.join(globalDir, 'SOUL-untrusted.md');
    if (fs.existsSync(untrustedSoul)) {
      mounts.push({
        hostPath: toHostPath(untrustedSoul),
        containerPath: '/workspace/global/SOUL.md',
        readonly: true,
      });
    }
    // Untrusted CLAUDE.md @-imports /workspace/global/FORMATTING.md so the
    // agent picks the right Slack/WA/Telegram/Discord syntax. Mount the
    // file individually instead of the whole global dir — sharing
    // FORMATTING.md across trust tiers is safe (it's universal channel
    // syntax with no owner state in it) but the rest of `global/` stays
    // off-limits per the existing untrusted boundary.
    const untrustedFormatting = path.join(globalDir, 'FORMATTING.md');
    if (fs.existsSync(untrustedFormatting)) {
      mounts.push({
        hostPath: toHostPath(untrustedFormatting),
        containerPath: '/workspace/global/FORMATTING.md',
        readonly: true,
      });
    }
    // Untrusted CLAUDE-untrusted.md @-imports /workspace/global/BASH_SAFETY.md
    // for the same reason main and trusted containers do — the rules apply to
    // every shell/git/gh invocation regardless of tier. Same individual-file
    // mount pattern as FORMATTING.md above (universal content, no owner
    // state, safe to share across tiers).
    const untrustedBashSafety = path.join(globalDir, 'BASH_SAFETY.md');
    if (fs.existsSync(untrustedBashSafety)) {
      mounts.push({
        hostPath: toHostPath(untrustedBashSafety),
        containerPath: '/workspace/global/BASH_SAFETY.md',
        readonly: true,
      });
    }
    // Per-tier custom system prompts (#113 / ligolnik#122). Untrusted
    // doesn't get the full /workspace/global mount, but the
    // agent-runner's resolveSystemPrompt() reads
    // /workspace/global/prompts/<tier>.md when USE_CUSTOM_PROMPT=1 — so
    // the prompts dir must be visible regardless of tier. Unlike the
    // FORMATTING.md / BASH_SAFETY.md mounts above (single-file binds),
    // this is a directory bind covering all three tier templates at
    // once; readonly so the agent can't mutate prompts. Same
    // universal-content rationale (the templates carry no owner state).
    const promptsDir = path.join(globalDir, 'prompts');
    if (fs.existsSync(promptsDir)) {
      mounts.push({
        hostPath: toHostPath(promptsDir),
        containerPath: '/workspace/global/prompts',
        readonly: true,
      });
    }
  }

  // .env shadowing is handled inside the container entrypoint via mount --bind
  // (Apple Container only supports directory mounts, not file mounts like /dev/null)

  // Shared trusted directory — writable space for trusted containers.
  if (isMain || group.containerConfig?.trusted) {
    const trustedDir = path.join(process.cwd(), 'trusted');
    fs.mkdirSync(trustedDir, { recursive: true });
    // Chown so container user can write memory files
    const trustedUid = HOST_UID ?? 1000;
    const trustedGid = HOST_GID ?? 1000;
    if (trustedUid !== 0) {
      try {
        fs.chownSync(trustedDir, trustedUid, trustedGid);
      } catch (err: unknown) {
        logger.warn({ err, trustedDir }, 'Failed to chown trusted dir');
      }
    }
    mounts.push({
      hostPath: toHostPath(trustedDir),
      containerPath: '/workspace/trusted',
      readonly: false,
    });
  }

  // Per-group writable state dir — mounted into EVERY container
  // regardless of trust tier (#99 Cat 4). Solves the silent-EACCES
  // failure mode for skills that need to persist state across runs:
  // `/workspace/group/` is read-only for untrusted, so any skill that
  // wrote there worked for trusted/main but silently broke for
  // untrusted (the audit's "strictly worse than no precheck" case).
  // With this mount, every tier has a single canonical writable
  // location to write to.
  //
  // Per-group (not per-session): matches the established mental model
  // where skills think in terms of "this group's state". A scheduled
  // task and a user-facing turn in the same group can read each
  // other's state; cross-group leakage is impossible by virtue of the
  // bind being scoped to `<folder>`.
  //
  // Always writable. Operators can `rm -rf data/state/<folder>/` to
  // wipe; otherwise grows monotonically with whatever skills choose
  // to persist. Distinct from `/workspace/group/` (group-shared,
  // trust-conditional readonly), `/workspace/trusted/` (trusted-only),
  // `/workspace/store/` (messages.db — rw on trusted/main for the
  // state-NNN-* tables, ro filtered copy on untrusted), and
  // `/workspace/global/` (global config). Skills that previously
  // wrote to `/workspace/group/` for cross-run state should migrate
  // to `/workspace/state/`.
  const stateDir = path.join(DATA_DIR, 'state', group.folder);
  fs.mkdirSync(stateDir, { recursive: true });
  const stateUid = HOST_UID ?? 1000;
  const stateGid = HOST_GID ?? 1000;
  if (stateUid !== 0) {
    try {
      fs.chownSync(stateDir, stateUid, stateGid);
    } catch (err: unknown) {
      // Narrow per `error-handling: Catch specific exception types`.
      // EPERM (we're not the owner and not root) and EACCES (insufficient
      // privileges to chown) are the two expected failure modes when the
      // orchestrator runs without root and the dir is owned by something
      // else — log and continue, the agent can still read/write via its
      // own uid because of the mode bits. Anything else (ENOENT after we
      // just mkdir'd, EROFS, EIO, etc.) is a real bug we want to surface.
      const code = (err as NodeJS.ErrnoException)?.code;
      if (code === 'EPERM' || code === 'EACCES') {
        logger.warn(
          { err, stateDir, code },
          'Failed to chown state dir (insufficient privileges) — continuing',
        );
      } else {
        throw err;
      }
    }
  }
  mounts.push({
    hostPath: toHostPath(stateDir),
    containerPath: '/workspace/state',
    readonly: false,
  });

  // Store directory (messages.db).
  // Trusted/main: full DB (all groups), READ-WRITE so skills can persist
  //   per-skill state in the new `state-NNN-*` tables (epic #293 — orders,
  //   email_feedback, …). Trust model: trusted agents already have write
  //   power on the group folder and on /workspace/state/, so direct DB
  //   writes are inside the same trust boundary. Without rw access here,
  //   apply-order.py / write-orders-metadata.py / etc. fail with
  //   "attempt to write a readonly database" — observed in production on
  //   the first check-orders run after the #294 tile flip.
  // Untrusted: filtered copy (own chat only), READ-ONLY so an untrusted
  //   agent cannot mutate cross-group state via writes against the
  //   filtered DB.
  // The orchestrator opens messages.db in WAL mode (see src/db.ts).
  // WAL coordinates concurrent writers via filesystem locks on
  // messages.db plus its `.wal` and `.shm` sidecars — both processes
  // need rw on all three files in the same directory. Mounting the
  // `store/` dir (rather than the file alone) gives the agent access
  // to the sidecars too. The orchestrator's `busy_timeout = 5000`
  // pragma plus per-script transactions keep contention bounded.
  if (isMain || group.containerConfig?.trusted) {
    const storeDir = path.join(process.cwd(), 'store');
    if (fs.existsSync(storeDir)) {
      mounts.push({
        hostPath: toHostPath(storeDir),
        containerPath: '/workspace/store',
        readonly: false,
      });
    }
  } else {
    // Untrusted: create filtered DB with only this group's messages
    const filteredDb = createFilteredDb(chatJid, group.folder);
    if (filteredDb) {
      mounts.push({
        hostPath: toHostPath(path.dirname(filteredDb)),
        containerPath: '/workspace/store',
        readonly: true,
      });
    }
  }

  // Per-group-per-session Claude sessions directory. The extra `sessionName`
  // segment isolates the user-facing (`default`) and scheduled (`maintenance`)
  // AyeAyes so their SDK transcripts, `settings.json`, skills/ and .tessl/
  // trees never collide when both run concurrently for the same group.
  const groupSessionsDir = path.join(
    DATA_DIR,
    'sessions',
    group.folder,
    sessionName,
    '.claude',
  );
  fs.mkdirSync(groupSessionsDir, { recursive: true });
  // Write settings.json only if content has changed — unnecessary rewrites
  // invalidate the SDK's prompt cache (file mtime changes trigger cache misses).
  // Invariant: this object is built from a pure literal that branches only
  // on `isMain` and trust tier (no per-session inputs), so `default` and
  // `maintenance` sessions of the same group produce byte-identical output
  // and cannot drift. If you add per-session branching here, or merge in
  // any user-editable input, reopen #63 — the file stops being
  // pure-generated and needs to move to the shared-memory mount alongside
  // auto-memory.
  const settingsFile = path.join(groupSessionsDir, 'settings.json');
  const newSettings =
    JSON.stringify(
      {
        env: {
          // NOTE: model selection is NOT controlled here. The agent-runner
          // calls the SDK's query() directly (not via the Claude Code CLI),
          // and the SDK takes `model` as a query() parameter. We pass it as
          // `AGENT_MODEL` on the container env (see below) so the runner can
          // read it at call time. `CLAUDE_CODE_MODEL` is not a real Claude
          // Code env var — previously set here but silently ignored.
          CLAUDE_CODE_MAX_CONTEXT_WINDOW: '1000000',
          CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: '1',
          CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: '1',
          CLAUDE_CODE_EFFORT_LEVEL: 'max',
          // Disable auto-memory for untrusted groups to prevent persistent injection
          CLAUDE_CODE_DISABLE_AUTO_MEMORY:
            isMain || group.containerConfig?.trusted ? '0' : '1',
        },
      },
      null,
      2,
    ) + '\n';
  if (
    !fs.existsSync(settingsFile) ||
    fs.readFileSync(settingsFile, 'utf-8') !== newSettings
  ) {
    fs.writeFileSync(settingsFile, newSettings);
  }

  // Tile delivery — all host-side, no tessl CLI in containers.
  // Build .tessl structure and skills/ from registry-installed tiles (tessl-workspace).
  // Main/trusted get all tiles. Others get nanoclaw-core only.
  const skillsDst = path.join(groupSessionsDir, 'skills');
  if (fs.existsSync(skillsDst)) {
    fs.rmSync(skillsDst, { recursive: true, force: true });
  }
  fs.mkdirSync(skillsDst, { recursive: true });

  const dstTessl = path.join(groupSessionsDir, '.tessl');
  if (fs.existsSync(dstTessl)) {
    fs.rmSync(dstTessl, { recursive: true, force: true });
  }

  const tilesToInstall = selectTiles(
    isMain,
    !!group.containerConfig?.trusted,
    group.containerConfig?.additionalTiles,
  );

  const registryTiles = getRegistryTilesDir();

  // #305 fail-closed guard for `additionalTiles`. The trust-tier
  // baseline (`nanoclaw-core`, `nanoclaw-trusted`/`nanoclaw-untrusted`,
  // and `nanoclaw-admin` for main) is allowed to degrade if the
  // registry is mid-rebuild — the per-tile `existsSync` warning inside
  // the install loop catches that. But every entry in
  // `additionalTiles` was explicitly opted in by config, and silently
  // dropping one would mean the chat loses a capability with no
  // signal. Refuse to spawn so the operator notices.
  //
  // Validation runs against the live registry directory, the same
  // source the install loop reads three blocks down. `getInstalledTiles()`
  // returns `null` when the registry directory itself doesn't exist
  // — treated as "all additionalTiles missing" so the spawn refuses
  // with the same diagnostic.
  //
  // We validate the NORMALIZED overlay (post-`selectTiles` filtering:
  // trimmed, baseline-deduped, within-overlay-deduped, whitespace
  // skipped) rather than the raw config. The IPC handler enforces
  // these invariants at write time, but a config that bypassed IPC
  // (manual DB edit, migration, future IPC variant) might still carry
  // whitespace-padded or duplicate entries; validating raw entries
  // against `installed` would falsely flag "  nanoclaw-coding" as
  // missing while `selectTiles` would happily install the trimmed
  // form. Slicing `tilesToInstall` past the baseline length keeps the
  // guard checking exactly the names that will be installed.
  const baselineLength = selectTiles(
    isMain,
    !!group.containerConfig?.trusted,
  ).length;
  const overlay = tilesToInstall.slice(baselineLength);
  if (overlay.length > 0) {
    const installed = new Set(getInstalledTiles() ?? []);
    const missing = overlay.filter((t) => !installed.has(t));
    if (missing.length > 0) {
      const detail = `additionalTiles missing from registry: ${missing.join(', ')} (registry=${registryTiles}). Run \`tessl update\` in the orchestrator or remove the entries via \`set_additional_tiles\`.`;
      logger.error(
        {
          groupFolder: group.folder,
          missing,
          configuredOverlay: group.containerConfig?.additionalTiles,
          normalizedOverlay: overlay,
          registryTiles,
        },
        `Refusing to spawn container: ${detail}`,
      );
      throw new Error(detail);
    }
  }

  // Build the group's tile-managed scripts/ in a sibling tmp dir, then
  // publish it atomically via a symlink flip (see the swap block below).
  // Writing directly into `groups/<folder>/scripts/` would create two
  // separate problems the moment two sessions run concurrently:
  //   1. Stale scripts from removed-in-new-tile skills lingered (the bug
  //      that drove the "DB size crossed N MB" rogue-heartbeat behaviour).
  //   2. A naive `rmSync(groupScriptsDir)` before the copy loop opens a
  //      race window where the other session reads a half-populated dir.
  // Tmp-then-publish gives both sessions a valid snapshot at all times:
  // whichever session finishes last wins the publish, and both end states
  // are equivalent (same installed tile version).
  const groupScriptsDir = path.join(groupDir, 'scripts');
  const rulesContent: string[] = [];

  // #337 maintenance blocklist. Default-class spawns see no filter; the
  // sets are gated behind sessionName === MAINTENANCE_SESSION_NAME so any
  // misconfiguration on the default-session path is a no-op. Declared at
  // function scope so the tile-install loop, the built-in skills copy,
  // and the staging skills copy all apply the same filter and the
  // single emitted log line aggregates filtered names across all three.
  const isMaintenance = sessionName === MAINTENANCE_SESSION_NAME;
  const ruleBlocklist = isMaintenance ? MAINTENANCE_RULE_BLOCKLIST : null;
  // #544b — compute the EFFECTIVE skill blocklist by walking the
  // transitive `Skill(skill: "...")` reference graph from every
  // non-blocklisted skill. Anything reachable from a loaded skill
  // gets exempted so nested invocations resolve at agent runtime.
  // Reference incident: 2026-05-10 wiki-lint (root) failed with
  // "Unknown skill: wiki" because wiki was blocklisted but invoked
  // from wiki-lint's SKILL.md. The pre-scan walks the same three
  // skill sources the install loop below visits (tile skills,
  // built-in skills, staging skills) so the closure sees the full
  // graph the agent will actually load.
  //
  // #552 — the same pre-scan also produces the positive `reachable`
  // set used by the rule `requires:` filter below. Default-session
  // spawns compute against an empty blocklist (no skills filtered),
  // so `reachableSkills` is simply "every skill present in any
  // installed tile, plus built-in and staging skills" — the input
  // the rule filter needs. The cost is one BFS per spawn over the
  // skill reference graph; previously skipped on default-session
  // spawns, now run unconditionally so the rule filter can decide.
  const skillContext = computeEffectiveSkillContextForSpawn(
    isMaintenance ? MAINTENANCE_SKILL_BLOCKLIST : new Set<string>(),
    tilesToInstall,
    registryTiles,
    groupDir,
  );
  const skillBlocklist = isMaintenance ? skillContext.effectiveBlocklist : null;
  const presentSkills = skillContext.reachableSkills;
  const filteredRules: string[] = [];
  const filteredSkills: string[] = [];

  // Registry-availability guard: if not a single tile in `tilesToInstall`
  // actually exists under `registryTiles`, skip the whole build-and-swap.
  // Otherwise the tmpdir + atomic flip would publish an EMPTY scripts/
  // symlink, wiping the previous version — which is worse than stale.
  // Root causes this defends against: tessl install failed on this spawn,
  // the registry mount glitched, a first-boot race. The per-tile
  // `fs.existsSync(tileSrc)` check inside the loop still handles partial
  // degradation (some tiles present, others missing).
  const anyTileAvailable = tilesToInstall.some((tileName) =>
    fs.existsSync(path.join(registryTiles, tileName)),
  );
  if (!anyTileAvailable) {
    logger.warn(
      { registryTiles, tilesToInstall, groupScriptsDir },
      'No tile sources available — keeping existing groupScriptsDir and .tessl/RULES.md intact. Investigate tessl install state.',
    );
  } else {
    const scriptsTmpSuffix = `${process.pid}.${Date.now()}.${Math.random().toString(36).slice(2, 8)}`;
    const tmpScriptsDir = `${groupScriptsDir}.new.${scriptsTmpSuffix}`;
    fs.mkdirSync(tmpScriptsDir, { recursive: true });

    for (const tileName of tilesToInstall) {
      const tileSrc = path.join(registryTiles, tileName);
      if (!fs.existsSync(tileSrc)) {
        logger.warn(
          { tileName, path: tileSrc },
          'Tile not found — run tessl install in orchestrator',
        );
        continue;
      }

      const dstTileDir = path.join(dstTessl, 'tiles', TILE_OWNER, tileName);

      // Copy rules
      const rulesDir = path.join(tileSrc, 'rules');
      if (fs.existsSync(rulesDir)) {
        for (const ruleFile of fs.readdirSync(rulesDir)) {
          if (!ruleFile.endsWith('.md')) continue;
          if (ruleBlocklist?.has(ruleFile)) {
            filteredRules.push(`${tileName}/${ruleFile}`);
            continue;
          }
          const ruleSrcFile = path.join(rulesDir, ruleFile);
          const ruleSrcContent = fs.readFileSync(ruleSrcFile, 'utf8');
          // #552 — apply the rule `requires:` filter against the
          // spawn's effective skill presence set. A rule with no
          // `requires:` declaration always loads (current default).
          // A rule whose `requires:` lists no skill present in the
          // spawn is filtered out — its content is omitted from
          // both the per-tile mirror copy and the aggregated
          // RULES.md, and the filtered name is logged alongside
          // the existing maintenance-blocklist filtered names.
          const filterResult = shouldIncludeRule(ruleSrcContent, presentSkills);
          if (!filterResult.include) {
            filteredRules.push(
              `${tileName}/${ruleFile} (requires: ${filterResult.requires.join(', ') || '<empty>'})`,
            );
            continue;
          }
          // Write the rule from the in-memory content we already read
          // for the frontmatter parse, rather than a second `cpSync`
          // that re-reads the same bytes off disk. Saves one read per
          // rule per spawn — small per-rule but multiplied by every
          // rule in every installed tile on every spawn.
          const ruleDst = path.join(dstTileDir, 'rules', ruleFile);
          fs.mkdirSync(path.dirname(ruleDst), { recursive: true });
          fs.writeFileSync(ruleDst, ruleSrcContent);
          rulesContent.push(ruleSrcContent);
        }
      }

      // Copy skills and their scripts
      const tileSkillsDir = path.join(tileSrc, 'skills');
      if (fs.existsSync(tileSkillsDir)) {
        for (const skillDir of fs.readdirSync(tileSkillsDir)) {
          const skillSrcDir = path.join(tileSkillsDir, skillDir);
          if (!fs.statSync(skillSrcDir).isDirectory()) continue;
          // #544 — `scripts/` is published to `groups/<folder>/scripts/`
          // and consumed by host MCP handlers (`mcp__nanoclaw__*`)
          // independently of the agent context. Even when a skill's
          // SKILL.md is blocklisted (excluded from the agent's loaded
          // tile context to save tokens), OTHER non-blocklisted
          // skills can still trigger host operations that look up
          // scripts from this dir — e.g. `entertainment-sync`'s Step 1
          // calls `mcp__nanoclaw__fetch_trakt_history()`, whose host
          // handler in `src/ipc.ts` reads
          // `groups/<sourceGroup>/scripts/trakt-watch-history.py`.
          // Scripts don't consume agent context (they're not in the
          // SKILL.md surface the SDK loads), so excluding them along
          // with the prompt is purely accidental. Copy
          // unconditionally; only the prompt-context copies below
          // honour the blocklist.
          copyTileScriptsToFlatDir(
            path.join(skillSrcDir, 'scripts'),
            tmpScriptsDir,
          );
          if (skillBlocklist?.has(skillDir)) {
            filteredSkills.push(`${tileName}/${skillDir}`);
            continue;
          }
          fs.cpSync(skillSrcDir, path.join(dstTileDir, 'skills', skillDir), {
            recursive: true,
          });
          fs.cpSync(skillSrcDir, path.join(skillsDst, `tessl__${skillDir}`), {
            recursive: true,
          });
        }
      }
    }

    // Atomic symlink-based publish for `groups/<folder>/scripts/`. Readers
    // must ALWAYS find `groupScriptsDir` present — the previous "rename old
    // aside, rename new into place" design had a brief ENOENT window between
    // the two renames where a concurrent agent running `/workspace/group/
    // scripts/<file>` would fail (CodeQL correctness finding).
    //
    // New layout:
    //   groups/<folder>/scripts             ──► symlink
    //   groups/<folder>/scripts.version.<id> ──► real directory (one per publish)
    //
    // Publish steps:
    //   1. rename `tmpScriptsDir` → `scripts.version.<id>` (a unique sibling)
    //   2. create a temporary symlink `scripts.link.<id>` → that version
    //   3. atomically rename the symlink over `groupScriptsDir` (POSIX rename
    //      on a symlink replaces an existing symlink atomically)
    //   4. delete the previous version dir (if any)
    //
    // First-install path: no `groupScriptsDir` exists; we just rename the
    // temp symlink into place — still atomic, no window.
    //
    // Legacy path: pre-this-commit installs have a REAL directory at
    // `groupScriptsDir` (not a symlink). We can't atomically replace a
    // non-empty directory with a symlink. For that one-time transition we
    // do `rm -rf <dir>` + `symlink` — has a brief window, but runs exactly
    // once per group, ever, and is bounded.
    // RACE_CODES and rmBestEffort are defined at module scope — see the top
    // of this file. Both atomic-publish flows (this scripts/ symlink-flip
    // and the .tessl/ swap-rename below) share the same race-code semantics.
    const swapId = `${Date.now()}.${process.pid}.${Math.random().toString(36).slice(2, 10)}`;
    const newVersionDir = `${groupScriptsDir}.version.${swapId}`;
    const tmpLink = `${groupScriptsDir}.link.${swapId}`;
    let previousVersionDir: string | null = null;

    try {
      // 1. Publish our tmp build as a versioned sibling.
      fs.renameSync(tmpScriptsDir, newVersionDir);

      // 2. Inspect what's currently at `groupScriptsDir` (symlink, real dir,
      //    or missing).
      let liveStat: fs.Stats | null = null;
      try {
        liveStat = fs.lstatSync(groupScriptsDir);
      } catch (err: unknown) {
        const code = (err as NodeJS.ErrnoException).code;
        if (code !== 'ENOENT') throw err;
      }
      if (liveStat && liveStat.isSymbolicLink()) {
        // Remember the previous version so we can clean it up after the flip.
        try {
          const currentTarget = fs.readlinkSync(groupScriptsDir);
          previousVersionDir = path.isAbsolute(currentTarget)
            ? currentTarget
            : path.resolve(path.dirname(groupScriptsDir), currentTarget);
        } catch {
          /* ignore — if we can't read the link we just won't clean it up */
        }
      }

      // 3. Create the temp symlink. Relative target so moves of the parent
      //    dir don't break the link. `'dir'` hint matters only on Windows.
      fs.symlinkSync(path.basename(newVersionDir), tmpLink, 'dir');

      if (liveStat && !liveStat.isSymbolicLink()) {
        // Legacy layout: real dir at `groupScriptsDir`. Can't atomically
        // replace a non-empty directory with a symlink; remove it first.
        // This is the one-time per-group transition; steady state uses
        // the pure atomic rename below.
        fs.rmSync(groupScriptsDir, { recursive: true, force: true });
      }

      // 4. Atomic flip: POSIX rename on a symlink replaces an existing
      //    symlink atomically. On the first-install path (no prior
      //    symlink) this just creates the symlink. Either way
      //    `groupScriptsDir` resolves to a valid versioned dir from
      //    here on.
      fs.renameSync(tmpLink, groupScriptsDir);

      // 5. Clean up the previous versioned dir (if any).
      if (previousVersionDir) rmBestEffort(previousVersionDir);
    } catch (err: unknown) {
      const code = (err as NodeJS.ErrnoException).code;
      const isRace = code ? RACE_CODES.has(code) : false;

      // Always drop our publish artefacts: if the race winner placed a
      // correct `groupScriptsDir` our versioned dir is redundant; on a
      // real error we can't trust our partial build.
      rmBestEffort(tmpLink);
      rmBestEffort(newVersionDir);
      rmBestEffort(tmpScriptsDir);

      if (isRace) {
        logger.debug(
          { err, groupScriptsDir },
          'scripts/ publish raced with concurrent setup; keeping winning copy',
        );
      } else {
        logger.error(
          { err, code, groupScriptsDir },
          'scripts/ publish failed unexpectedly (not a race)',
        );
        // Rethrow non-race errors so the spawn fails loudly. Unlike the
        // previous design, `groupScriptsDir` (if it existed) is still
        // present — either still a symlink pointing at the previous
        // version, or still the legacy real dir — so even a failed
        // publish doesn't leave the group with missing scripts.
        throw err;
      }
    }
  } // end registry-availability guard

  // Write aggregated RULES.md
  if (rulesContent.length > 0) {
    fs.mkdirSync(dstTessl, { recursive: true });
    fs.writeFileSync(
      path.join(dstTessl, 'RULES.md'),
      rulesContent.join('\n\n---\n\n'),
    );
  }

  // Copy .tessl/ to group folder so AGENTS.md → .tessl/RULES.md resolves.
  // Host-side copy is required because untrusted groups mount /workspace/group
  // read-only — the container cannot write .tessl/ there itself.
  // Skip if RULES.md content is unchanged (avoids unnecessary I/O on every spawn).
  const groupTesslDir = path.join(groupDir, '.tessl');
  if (fs.existsSync(dstTessl)) {
    const srcRules = path.join(dstTessl, 'RULES.md');
    const dstRules = path.join(groupTesslDir, 'RULES.md');
    const needsCopy =
      fs.existsSync(srcRules) &&
      (!fs.existsSync(dstRules) ||
        fs.readFileSync(srcRules, 'utf-8') !==
          fs.readFileSync(dstRules, 'utf-8'));
    if (needsCopy) {
      // Atomic publish (see #95). The previous rm+cp was vulnerable to a
      // race when two scheduled tasks (heartbeat + task-watchdog) fired in
      // the same millisecond — the rmSync walk would hit ENOTEMPTY because
      // the concurrent cpSync had refilled subdirs the walk hadn't reached
      // yet. atomicPublishDir uses temp+swap-rename so dstDir is always a
      // fully-populated directory at any instant; the loser's work is
      // discarded benignly.
      atomicPublishDir(dstTessl, groupTesslDir);
    }
  }

  // Built-in container skills (agent-browser, status, etc.)
  const builtinSkillsDir = path.join(process.cwd(), 'container', 'skills');
  if (fs.existsSync(builtinSkillsDir)) {
    for (const skillDir of fs.readdirSync(builtinSkillsDir)) {
      const srcDir = path.join(builtinSkillsDir, skillDir);
      if (!fs.statSync(srcDir).isDirectory()) continue;
      // #337 maintenance blocklist applies to built-in skills too. Listing
      // by bare name (no `tessl__` prefix) covers both surface forms.
      if (skillBlocklist?.has(skillDir)) {
        filteredSkills.push(`builtin/${skillDir}`);
        continue;
      }
      fs.cpSync(srcDir, path.join(skillsDst, skillDir), { recursive: true });
    }
  }

  // AyeAye-created skills (staging) — override tile skills if names collide
  const groupSkillsDir = path.join(groupDir, 'skills');
  if (fs.existsSync(groupSkillsDir)) {
    const stagingSkills = fs.readdirSync(groupSkillsDir).filter((d) => {
      const p = path.join(groupSkillsDir, d);
      return fs.statSync(p).isDirectory();
    });
    if (stagingSkills.length > 0) {
      logger.warn(
        { folder: group.folder, skills: stagingSkills },
        'Staging skills override tile skills — run verify-tiles to clear',
      );
      for (const skillDir of stagingSkills) {
        // #337 maintenance blocklist applies to staging skills too.
        if (skillBlocklist?.has(skillDir)) {
          filteredSkills.push(`staging/${skillDir}`);
          continue;
        }
        fs.cpSync(
          path.join(groupSkillsDir, skillDir),
          path.join(skillsDst, skillDir),
          { recursive: true },
        );
      }
    }
  }

  // #337 single aggregated emission across all three install sections
  // (tile rules + tile skills + built-in skills + staging skills). One
  // log line per spawn — empty filter list = silence.
  if (filteredRules.length > 0 || filteredSkills.length > 0) {
    logger.info(
      {
        group: group.folder,
        sessionName,
        filteredRules,
        filteredSkills,
      },
      'install_blocklist_filtered',
    );
  }
  // Chown the .claude session dir so the container user (node) can write to it.
  // The SDK creates subdirs like session-env/ at runtime — without this, EACCES.
  const sessionUid = HOST_UID ?? 1000;
  const sessionGid = HOST_GID ?? 1000;
  if (sessionUid !== 0) {
    try {
      chownRecursive(groupSessionsDir, sessionUid, sessionGid);
    } catch (err: unknown) {
      logger.warn(
        { err, groupSessionsDir },
        'Failed to chown .claude session dir',
      );
    }
  }
  mounts.push({
    hostPath: toHostPath(groupSessionsDir),
    containerPath: '/home/node/.claude',
    readonly: false,
  });

  // Tile content read-only overlay (#247).
  //
  // The `/home/node/.claude` mount above MUST stay writable — the SDK
  // writes session JSONL transcripts to `projects/<slug>/`, debug logs
  // to `debug/`, todos to `todos/`, telemetry to `telemetry/`,
  // session-env to `session-env/`, and the auto-memory overlay below
  // also depends on a writable parent. We can't flip the parent
  // readonly without breaking all of that.
  //
  // What we CAN flip readonly is the two specific subdirs that hold
  // installed tile content: `skills/` (per-tile SKILL.md trees, plus
  // bundled scripts and assets) and `.tessl/` (per-tile rules
  // markdown copied under `tiles/<owner>/<tile>/rules/` plus the
  // aggregated RULES.md the orchestrator generates from them). The
  // orchestrator wrote both host-side at the top of this function via
  // cpSync from `tessl-workspace/.tessl/tiles/...`, so by the time
  // the agent container starts the content is already in place. Layer
  // two readonly bind-mounts on top of the writable parent so the
  // kernel rejects any write from inside the container with EROFS.
  // The agent's edit/write tools cannot patch installed content
  // mid-session anymore — modifications must flow through staging →
  // promote → publish → tessl update like every other tile change.
  //
  // Why `tessl update` is unaffected: it runs in the orchestrator
  // container against `/app/tessl-workspace/.tessl/tiles/...`, a
  // completely different filesystem path the agent never sees. The
  // per-spawn cpSync that copies registry tiles into
  // `<groupSessionsDir>/skills/` and `<groupSessionsDir>/.tessl/`
  // runs host-side BEFORE the container starts, so the readonly
  // overlay is not in effect during that copy. The next spawn's
  // `rmSync` calls at the top of this function also run host-side
  // (between the previous container's death and the next one's
  // start) — no overlay in effect at rmSync time either.
  // Pre-create both host directories so Docker doesn't auto-create
  // them as root with surprising permissions when bind-mounting.
  // `skillsDst` is already mkdir'd at the top of this function, but
  // `dstTessl` is only mkdir'd inside the `if (anyTileAvailable)`
  // branch — when no tiles are available (registry mount glitched,
  // first boot, partial install), the .tessl mount source would be
  // missing. Idempotent recursive mkdir handles both cases without
  // disturbing the populated content path.
  fs.mkdirSync(skillsDst, { recursive: true });
  fs.mkdirSync(dstTessl, { recursive: true });
  // Mount-order discipline: these two readonly overlays MUST be
  // pushed AFTER the writable `/home/node/.claude` parent (which
  // happened ~30 lines above this comment). Docker applies bind
  // mounts in declaration order; a later parent mount would shadow
  // earlier child overlays, which would silently restore writability
  // and quietly defeat the whole #247 enforcement. The
  // `readonly tile-content overlay` test in container-runner.test.ts
  // pins this ordering by asserting argv index of the parent mount
  // arg is less than the index of both ro overlay args.
  mounts.push({
    hostPath: toHostPath(skillsDst),
    containerPath: '/home/node/.claude/skills',
    readonly: true,
  });
  mounts.push({
    hostPath: toHostPath(dstTessl),
    containerPath: '/home/node/.claude/.tessl',
    readonly: true,
  });

  // Shared auto-memory mount (issue #57). Claude Code's SDK writes
  // accumulated feedback and owner-profile memory to
  // ~/.claude/projects/<slug>/memory/. PR #55 mounted `.claude/` per-session,
  // which also split memory between `default` and `maintenance` — feedback
  // from one was invisible to the other. Memory describes the owner, not a
  // session, so it belongs shared.
  //
  // Overlay pattern: the per-session `.claude/` mount above gives each
  // container its own `projects/<slug>/*.jsonl` transcripts. This second
  // bind mount overlays only the `memory/` subdirectory with the shared
  // dir. Docker applies nested bind mounts in order; the later mount
  // replaces the contents at its path. Result: transcripts stay
  // per-session, memory is shared.
  //
  // Trust-tier gate: settings.json above sets
  // `CLAUDE_CODE_DISABLE_AUTO_MEMORY: '1'` on untrusted containers to
  // block persistent prompt injection. We must NOT give those containers
  // a shared writable owner-state dir — an untrusted container that
  // bypassed the env var (direct fs write, SDK bug, etc.) would poison
  // the owner-memory that main/trusted containers read. Gate the mount,
  // mkdir, migration, and chown behind the same trust condition.
  const autoMemoryEnabled = isMain || !!group.containerConfig?.trusted;
  if (autoMemoryEnabled) {
    // #658: the main container unifies its auto-memory with the
    // cross-container `/workspace/trusted` corpus. The maintenance skills
    // (memory-hygiene, soul-searching, memory-enrich) run only in the
    // main/maintenance container and read `/workspace/trusted`; sourcing the
    // auto-memory dir from the same host path means one corpus instead of two.
    // Side effect the operator accepts: trusted (non-main) containers can write
    // `/workspace/trusted`, so their content becomes auto-injected into main's
    // context. Acceptable for a single-operator deployment; untrusted stays
    // excluded (above) precisely because that injection path is a poisoning
    // vector. Trusted (non-main) groups keep a per-group shared-memory dir —
    // their auto-memory is per-chat-context state no maintenance skill reads.
    const sharedMemoryDir = isMain
      ? path.join(process.cwd(), 'trusted')
      : path.join(DATA_DIR, 'sessions', group.folder, 'shared-memory');
    fs.mkdirSync(sharedMemoryDir, { recursive: true });

    // Pre-create the overlay mount target inside the `.claude` bind. Docker
    // applies the shared-memory mount at
    // `/home/node/.claude/projects/<slug>/memory` on top of the outer `.claude`
    // mount, which requires the mountpoint path to exist on the lower
    // filesystem. Without this pre-creation Docker auto-mkdirs the missing
    // ancestors as uid 0, leaving `projects/` and `projects/<slug>/` root-owned
    // on the host — which breaks node-user writes to per-session transcripts
    // that the SDK writes alongside `memory/` (e.g. `projects/<slug>/*.jsonl`).
    // The outer `chownRecursive(groupSessionsDir, ...)` above already ran, so
    // we chown the new subtree explicitly here.
    const projectsDir = path.join(groupSessionsDir, 'projects');
    const memoryMountTarget = path.join(
      projectsDir,
      CLAUDE_PROJECT_SLUG,
      'memory',
    );
    fs.mkdirSync(memoryMountTarget, { recursive: true });
    if (sessionUid !== 0) {
      try {
        chownRecursive(projectsDir, sessionUid, sessionGid);
      } catch (err: unknown) {
        logger.warn(
          { err, projectsDir },
          'Failed to chown projects/ overlay mount-target tree',
        );
      }
    }

    // One-shot migration: for installations upgrading from PR #55 (per-session
    // memory) to #57 (shared memory), scan each per-session `memory/` dir for
    // files that haven't made it into shared-memory yet and copy them over.
    // Shared-memory wins on conflict (it's the newer source of truth); the
    // per-session copy is left in place but orphaned — subsequent reads go
    // through the shared mount. Safe to run every spawn: only acts on files
    // that exist per-session but NOT in shared.
    // Hardcoded rather than importing MAINTENANCE_SESSION_NAME from
    // group-queue (that would add a circular dep — group-queue already
    // imports from here). These are the two session names that existed
    // before this migration lands, so the list is fixed by history.
    for (const otherSession of ['default', 'maintenance']) {
      const perSessionMemoryDir = path.join(
        DATA_DIR,
        'sessions',
        group.folder,
        otherSession,
        '.claude',
        'projects',
        CLAUDE_PROJECT_SLUG,
        'memory',
      );
      if (!fs.existsSync(perSessionMemoryDir)) continue;
      let entries: string[];
      try {
        entries = fs.readdirSync(perSessionMemoryDir);
      } catch {
        continue;
      }
      for (const file of entries) {
        const src = path.join(perSessionMemoryDir, file);
        const dst = path.join(sharedMemoryDir, file);
        if (fs.existsSync(dst)) continue;
        try {
          fs.cpSync(src, dst, { recursive: true, force: false });
          logger.info(
            { group: group.folder, file, fromSession: otherSession },
            'Migrated per-session memory file to shared-memory',
          );
        } catch (err: unknown) {
          const code = (err as NodeJS.ErrnoException).code;
          // EEXIST / ERR_FS_CP_EEXIST: another session spawning concurrently
          // won the cpSync — our copy is redundant, their content is valid
          // (same source file, same target). Expected race in steady state;
          // log at debug so parallel startup doesn't spam warn logs.
          if (code === 'EEXIST' || code === 'ERR_FS_CP_EEXIST') {
            logger.debug(
              { src, dst },
              'Concurrent session won the shared-memory migration — keeping winner',
            );
          } else {
            logger.warn(
              { err, src, dst },
              'Failed to migrate per-session memory file',
            );
          }
        }
      }
    }

    // The per-group shared-memory dir is orchestrator-created and may hold
    // docker-auto-mkdir'd root-owned content, so it needs a recursive chown to
    // the session user. For isMain the dir IS /workspace/trusted — already
    // chowned (non-recursively) by the trusted-mount block above to the same
    // HOST_UID, with host-managed files the agent already writes to. Recursively
    // chowning that whole corpus (incl. the wiki/ subtree) on every spawn is
    // wasteful and needless, so skip it for main.
    if (!isMain && sessionUid !== 0) {
      try {
        chownRecursive(sharedMemoryDir, sessionUid, sessionGid);
      } catch (err: unknown) {
        logger.warn(
          { err, sharedMemoryDir },
          'Failed to chown shared-memory dir',
        );
      }
    }
    mounts.push({
      hostPath: toHostPath(sharedMemoryDir),
      containerPath: `/home/node/.claude/projects/${CLAUDE_PROJECT_SLUG}/memory`,
      readonly: false,
    });
  } // end autoMemoryEnabled

  // Claude Code config file — lives at /home/node/.claude.json (outside .claude/).
  // Read-only rootfs can't create it, so we bind-mount it from the sessions dir.
  // Kept alongside the per-session `.claude/` dir so default and maintenance
  // AyeAyes don't share Claude Code's per-session config.
  const claudeJsonPath = path.join(
    DATA_DIR,
    'sessions',
    group.folder,
    sessionName,
    '.claude.json',
  );
  if (!fs.existsSync(claudeJsonPath)) {
    fs.writeFileSync(claudeJsonPath, '{}');
  }
  // Chown so container user can write (Claude Code updates this file at runtime)
  const jsonUid = HOST_UID ?? 1000;
  const jsonGid = HOST_GID ?? 1000;
  if (jsonUid !== 0) {
    try {
      fs.chownSync(claudeJsonPath, jsonUid, jsonGid);
    } catch (err: unknown) {
      logger.warn({ err, claudeJsonPath }, 'Failed to chown .claude.json');
    }
  }
  mounts.push({
    hostPath: toHostPath(claudeJsonPath),
    containerPath: '/home/node/.claude.json',
    readonly: false,
  });

  // Per-group IPC namespace. `input/` is per-session so parallel default
  // and maintenance containers don't step on each other's _close sentinel
  // or follow-up JSON files. `messages/` and `tasks/` stay shared — they're
  // outbound from the container and the host aggregates both sessions' output.
  const groupIpcDir = resolveGroupIpcPath(group.folder);
  const sessionInputSubdir = sessionInputDirName(sessionName);
  const sessionInputDir = path.join(groupIpcDir, sessionInputSubdir);
  const isTrustedIpc = isMain || !!group.containerConfig?.trusted;
  fs.mkdirSync(path.join(groupIpcDir, 'messages'), { recursive: true });
  fs.mkdirSync(sessionInputDir, { recursive: true });

  // Wipe stale sentinel files from previous container lifecycles. A `_close`
  // left over from a graceful shutdown will be consumed by a fresh
  // container's first IPC poll and end its input stream prematurely — the
  // SDK still finishes the in-flight prompt, but the container exits
  // immediately after, so the next user message takes the cost of a fresh
  // spawn. Clean here, where we know we're about to start a new container.
  const staleClose = path.join(sessionInputDir, '_close');
  if (fs.existsSync(staleClose)) {
    try {
      fs.unlinkSync(staleClose);
    } catch (err) {
      // Only ENOENT is expected here — a benign race where another part of
      // the orchestrator (or a concurrent shutdown) already removed the
      // sentinel between the `existsSync` check and the unlink. Anything
      // else (EACCES, EBUSY, EIO, …) is a real filesystem problem we must
      // not silently downgrade — the stale sentinel will still trip the
      // fresh container, so failing loudly here is better than swallowing.
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw err;
      }
      logger.debug(
        { folder: group.folder, sessionName },
        'stale _close sentinel vanished between existsSync and unlink (race)',
      );
    }
  }

  // Wipe leftover IPC inputs from previous container lifecycles. The
  // previous container's drain has already happened (or never will, if it
  // crashed) and the fresh spawn will rebuild its initial-prompt context
  // from the messages.db cursor. Without this, untrusted spawns inherit
  // the entire backlog as their first prompt and cross the auto-compact
  // threshold mid-query (issue #287). `graceMs = 0` is normally safe here
  // — the previous container is gone and the new one isn't drained from
  // yet.
  //
  // EXCEPT during a graceful-shutdown handoff window: an adopted-but-
  // still-running container from the previous orchestrator may share
  // this session's input dir with the fresh spawn we're about to start,
  // and a graceMs=0 sweep would unlink files the adopted container
  // hasn't drained yet (#288 review). Skip the sweep while
  // `isHandoffActive()` is true — the within-handoff backlog is bounded
  // by the HANDOFF_TTL_MS window (5 min), and the next spawn after the
  // window expires will GC normally.
  if (!isHandoffActive()) {
    sweepStaleInputs(sessionInputDir, 0);
  }

  if (isTrustedIpc) {
    fs.mkdirSync(path.join(groupIpcDir, 'tasks'), { recursive: true });
  }
  // Chown IPC dirs so container user can read/write/unlink files
  const ipcUid = HOST_UID ?? 1000;
  const ipcGid = HOST_GID ?? 1000;
  if (ipcUid !== 0) {
    try {
      const subsToChown = isTrustedIpc
        ? ['', 'messages', 'tasks', sessionInputSubdir]
        : ['messages', sessionInputSubdir];
      for (const sub of subsToChown) {
        fs.chownSync(path.join(groupIpcDir, sub), ipcUid, ipcGid);
      }
    } catch (err) {
      logger.warn({ folder: group.folder, err }, 'Failed to chown IPC dirs');
    }
  }

  if (isTrustedIpc) {
    // Trusted/main: mount the whole IPC dir at /workspace/ipc, then overlay
    // the per-session input dir onto /workspace/ipc/input. Docker applies
    // nested bind mounts in order — the second mount replaces the dir entry
    // from the first, giving the container a session-isolated input/ while
    // the shared messages/tasks/ and IPC root files remain aggregated.
    mounts.push({
      hostPath: toHostPath(groupIpcDir),
      containerPath: '/workspace/ipc',
      readonly: false,
    });
    mounts.push({
      hostPath: toHostPath(sessionInputDir),
      containerPath: '/workspace/ipc/input',
      readonly: false,
    });
  } else {
    // Untrusted: split mounts — messages/ writable, input/ read-only, no tasks/.
    // Per-session input dir isolates _close sentinels between sessions.
    mounts.push({
      hostPath: toHostPath(path.join(groupIpcDir, 'messages')),
      containerPath: '/workspace/ipc/messages',
      readonly: false,
    });
    mounts.push({
      hostPath: toHostPath(sessionInputDir),
      containerPath: '/workspace/ipc/input',
      readonly: true,
    });
  }

  // Additional mounts validated against external allowlist
  if (group.containerConfig?.additionalMounts) {
    const validatedMounts = validateAdditionalMounts(
      group.containerConfig.additionalMounts,
      group.name,
      isMain,
    );
    mounts.push(...validatedMounts);

    // Shadow SECRET_FILES that are reachable through an additionalMount.
    //
    // The main-group block above `/dev/null`-mounts each SECRET_FILES
    // entry at `/workspace/project/<relPath>` — but that shadow only
    // covers the canonical project mount. An additionalMount can
    // re-expose the nanoclaw tree at a DIFFERENT container path (e.g.
    // a group registered with `hostPath: ~/nanoclaw` lands it at
    // `/workspace/extra/nanoclaw/`), and the `.env` at
    // `<mount>/.env` has no shadow applied there. A trusted agent
    // could then read the real token out of the extra mount even
    // though `/workspace/project/.env` is `/dev/null`.
    //
    // For every validated additionalMount whose host path CONTAINS any
    // SECRET_FILES entry, add a `/dev/null` bind at the corresponding
    // container path inside the extra mount. `path.relative` returning
    // a non-empty, non-`..`-prefixed, non-absolute string is the
    // "inside" predicate — matches what Docker's path resolution does.
    for (const vm of validatedMounts) {
      for (const relPath of SECRET_FILES) {
        // `toHostPath` is for the PATH COMPARISON only (it translates
        // the orchestrator-local cwd into its host-side equivalent so
        // `path.relative` compares against the host-side `vm.hostPath`
        // that Docker will actually bind). The EXISTENCE CHECK
        // deliberately uses the orchestrator-local path — in DooD mode
        // the orchestrator can't stat arbitrary host paths (it only
        // sees what's mounted into its own container), so a stat on
        // `toHostPath(...)` would wrongly return false and skip the
        // shadow. See `mount-security.ts` for the same "can't stat
        // host paths from inside DooD" note.
        const secretLocalPath = path.join(process.cwd(), relPath);
        const secretHostPath = toHostPath(secretLocalPath);
        const relFromMount = path.relative(vm.hostPath, secretHostPath);
        if (
          !relFromMount ||
          relFromMount.startsWith('..') ||
          path.isAbsolute(relFromMount)
        ) {
          continue;
        }
        if (!fs.existsSync(secretLocalPath)) continue;
        mounts.push({
          hostPath: '/dev/null',
          containerPath: path.posix.join(vm.containerPath, relFromMount),
          readonly: true,
        });
      }
    }
  }

  return mounts;
}

interface BuildContainerArgsResult {
  args: string[];
  // Always set — cleanup() is a no-op when no secret env-file was
  // written, so callers can invoke it unconditionally on container
  // exit without a null-check.
  cleanup: () => void;
}

function buildContainerArgs(
  mounts: VolumeMount[],
  containerName: string,
  group: RegisteredGroup,
  isMain: boolean,
  sessionName: string,
  replyToMessageId?: string,
  chatJid?: string,
  continuationCycleId?: string,
  attributionToken?: string,
  taskAgentModel?: string | null,
): BuildContainerArgsResult {
  const args: string[] = ['run', '-i', '--rm', '--name', containerName];

  // Resource limits and filesystem restrictions for untrusted containers
  if (!isMain && !group.containerConfig?.trusted) {
    args.push(
      '--memory',
      '512m', // 512MB RAM hard limit
      '--memory-swap',
      '512m', // no swap
      '--cpus',
      '1', // 1 CPU core
      '--pids-limit',
      '256', // prevent fork bombs
      '--read-only', // immutable root filesystem
      '--tmpfs',
      '/tmp:size=64m', // writable /tmp via tmpfs (needed for input.json)
    );
    // Group folder is read-only for untrusted (set above).
    // Agent can read CLAUDE.md/skills but can't write 7GB of numbers.
  } else {
    // Trusted/main: cap memory to prevent host OOM when multiple containers
    // run in parallel. The NAS has 7.5GB RAM total; without a cap, multiple
    // Claude SDK processes can exhaust host memory and trigger kernel OOM
    // killer (SIGKILL exit 137). 1.5GB is plenty for Claude Code + skills.
    args.push(
      '--memory',
      '1536m', // 1.5GB RAM hard limit
      '--memory-swap',
      '2048m', // allow 512MB swap as buffer
    );
  }

  // Pass host timezone so container's local time matches the user's
  args.push('-e', `TZ=${TIMEZONE}`);

  // Credential tiers:
  //   Main/Trusted: Composio + GitHub. Composio handles Gmail, Calendar,
  //                 Tasks, and GitHub-via-OAuth; GITHUB_TOKEN handles
  //                 GitHub-via-`gh`-CLI for high-fire-count automation
  //                 (the cost-monitor dashboard skills) that would
  //                 otherwise pay cache_create on the Composio GitHub
  //                 tool schemas every cold maintenance spawn.
  //   Other:        nothing (Anthropic via proxy only).
  //
  // All other host-side credentials (GOOGLE_*, RECLAIM_*, TRIPIT_*,
  // OPENAI_*) stay on the host. Scripts that need them run host-side
  // via IPC.
  const isTrusted = group.containerConfig?.trusted === true;

  const CONTAINER_VARS = [
    'COMPOSIO_API_KEY',
    'COMPOSIO_MCP_URL',
    'COMPOSIO_USER_ID',
    // Forwarded into main/trusted containers so the `gh` CLI authenticates
    // automatically. Same PAT the host-side github_backup handler uses
    // — the operator must expand its scope to cover `Contents: write`
    // (host-side git push) AND `Issues: write` + `Pull requests: write`
    // (container-side `gh issue edit/comment` from the cost-monitor
    // dashboard skills). See .env.example for the scope documentation.
    // OneCLI migration target — see jbaruch/nanoclaw#564 for the path
    // off the "secret-in-container-environ-for-spawn-lifetime" exposure.
    'GITHUB_TOKEN',
    // Forwarded for the `jbaruch/nanoclaw-travel` per-chat overlay
    // tile (added via `containerConfig.additionalTiles`). The precheck
    // reads `BYAIR_MCP_URL` (personal MCP link from
    // https://byairapp.com/mcp/, includes the API key inline) to poll
    // flight status via the byAir streamable-HTTP endpoint. Marked
    // SECRET below so it goes through `--env-file` instead of `-e KEY=...`.
    'BYAIR_MCP_URL',
    // Same tile. The precheck reads `GOOGLE_MAPS_API_KEY` to query the
    // Distance Matrix API for traffic-aware time-to-leave
    // (`departure_time=now`, `traffic_model=best_guess`). Generated at
    // https://console.cloud.google.com/apis/credentials with the
    // Distance Matrix API enabled on a billing-attached project. Marked
    // SECRET below.
    'GOOGLE_MAPS_API_KEY',
    // `jbaruch/nanoclaw-travel` tile: `maps_client.py` uses TomTom
    // (geocode + calculateRoute on `api.tomtom.com`) as the backup behind
    // Google Maps, and the `drive-planner` skill routes through it.
    // Generated at https://developer.tomtom.com. Marked SECRET below.
    'TOMTOM_API_KEY',
    // YouTube Data API v3 key — read by the admin tile's
    // `youtube-comment-check` skill, which calls the native API
    // (commentThreads.list + videos.list) directly because Composio's
    // YouTube toolkit has no comment-threads tool
    // (jbaruch/nanoclaw-admin#339). Marked SECRET below.
    'YOUTUBE_API_KEY',
    // Sessionize keys for the `jbaruch/nanoclaw-conferences` tile's
    // deterministic check-cfps pipeline: `discover-open-cfps.py` reads
    // SESSIONIZE_SPEAKER_KEY (speaker-profile open-CFP discovery) and
    // `verify-sessionize.py` reads SESSIONIZE_EVENT_API_KEY (per-slug
    // deadline verification). Both marked SECRET above so they route
    // through the env-file instead of `-e KEY=...`.
    'SESSIONIZE_SPEAKER_KEY',
    'SESSIONIZE_EVENT_API_KEY',
  ];

  const varsToForward = isMain || isTrusted ? CONTAINER_VARS : [];

  const envFromFile = readEnvFile(CONTAINER_VARS);
  // Partition forwarded vars: secrets get materialized into a 0600
  // env-file (passed via `--env-file`) so they don't appear on the
  // docker process command line; non-secrets stay on `-e KEY=value`.
  // See the SECRET_CONTAINER_VARS docstring for the policy.
  const secretEnv: Record<string, string> = {};
  for (const varName of varsToForward) {
    const value = process.env[varName] || envFromFile[varName];
    if (!value) continue;
    if (SECRET_CONTAINER_VARS.has(varName)) {
      secretEnv[varName] = value;
    } else {
      args.push('-e', `${varName}=${value}`);
    }
  }
  const secretEnvFile = buildSecretEnvFile(secretEnv);
  // Position of `--env-file` in argv is irrelevant for override
  // semantics — docker resolves `-e` over `--env-file` regardless of
  // order. We append here for readability of the assembled command;
  // a future caller adding a non-secret `-e` after this point won't
  // accidentally override a secret because the names don't overlap
  // (SECRET_CONTAINER_VARS membership is the partition rule).
  if (secretEnvFile) args.push(...secretEnvFile.args);

  // Select which model + effort the agent-runner's SDK query() uses.
  // The runner reads `process.env.AGENT_MODEL` and `process.env.AGENT_EFFORT`
  // — see constants at the top of this file. Keeping these on the env
  // (not baked into the agent image) lets model bumps / effort retuning
  // ship with an orchestrator rebuild only.
  //
  // #395: per-group `containerConfig.agentModel` override. When set and
  // valid, replaces the global AGENT_MODEL for this group's spawn only;
  // an unknown-prefix value falls back to the global default (see
  // `resolvePerGroupAgentModel` for branching).
  //
  // #418: emit one info log per spawn UNCONDITIONALLY — every container
  // spawn records the resolved `effectiveAgentModel` so cost / latency
  // attribution has a per-spawn audit trail with no silent-default
  // blind spot (the prior `if (override)`-gated log left default spawns
  // invisible to the audit). The `source` field distinguishes
  // default-vs-override so the log line is self-explaining without
  // joining against group config.
  // #509: per-session-slot model tier. Maintenance spawns
  // (sessionName === MAINTENANCE_SESSION_NAME) consult
  // `maintenanceAgentModel` first, then fall through to the existing
  // `agentModel` → AGENT_MODEL ladder. Non-maintenance spawns are
  // unchanged byte-for-byte from the pre-#509 behavior.
  // Phase 3 (#509): per-task override beats session-level when
  // `input.taskAgentModel` is set AND its resolved value differs from
  // the session-level fallback (sole writer: task-scheduler.ts plumbs
  // `task.agent_model` here). The audit-log `source` field is
  // `task_override` ONLY in that "different from session" case;
  // unknown-prefix values and per-row values that deliberately match
  // the session-level fallback fall through to the session-level
  // source so the log doesn't claim a routing change that didn't
  // happen. Unknown-prefix falls back to the session-level value
  // (NOT the global default), so a typo on a maintenance-pinned
  // group lands on the maintenance value, not Opus.
  const isMaintenanceSpawn = sessionName === MAINTENANCE_SESSION_NAME;
  // #613 Stage 1 — Claude tier-down. The spawn's BASE model is keyed to
  // the group's trust tier (main → global default, trusted → Sonnet,
  // untrusted → Haiku); per-group / maintenance / per-task overrides still
  // win on top via resolveSessionAgentModel. This applies to every spawn,
  // including scheduled/maintenance runs — task-scheduler selects the group
  // by `task.group_folder` and derives `isMain` from it, so a scheduled
  // task bases on its registered group's tier. Per-task `agentModel:`
  // frontmatter (trivial → Haiku, substantive → Sonnet) overrides that base.
  const tierBaseModel = resolveTierBaseModel(isMain, isTrusted, AGENT_MODEL);
  const { effective: effectiveAgentModel, source: agentModelSource } =
    resolveSessionAgentModel(
      group.containerConfig,
      isMaintenanceSpawn,
      tierBaseModel,
      taskAgentModel,
    );
  logger.info(
    {
      groupFolder: group.folder,
      sessionName,
      tier: isMain ? 'main' : isTrusted ? 'trusted' : 'untrusted',
      agentModel: effectiveAgentModel,
      // `tierBaseModel` is the tier-keyed base passed as the resolver's
      // default; `globalDefault` is the true orchestrator-wide AGENT_MODEL.
      // When they differ, a tier-down is in effect and `source` reads
      // `global_default` against the tier base (no per-group/task override).
      tierBaseModel,
      globalDefault: AGENT_MODEL,
      source: agentModelSource,
      // Include the per-task raw value so an unknown-prefix fallback
      // shows up next to the resolved value in the audit log — this
      // is the diagnostic surface for "I set task X to haiku but the
      // spawn still ran on sonnet".
      taskAgentModel: taskAgentModel ?? null,
    },
    'Container spawn AGENT_MODEL resolved',
  );
  args.push('-e', `AGENT_MODEL=${effectiveAgentModel}`);
  args.push('-e', `AGENT_EFFORT=${AGENT_EFFORT}`);

  // USE_CUSTOM_PROMPT (#113) — gates the per-tier custom system prompt
  // path in agent-runner. Default OFF: production behavior is unchanged
  // unless explicitly opted in. Resolution order:
  //   1. Per-group `containerConfig.useCustomPrompt` if set (true or false)
  //      — explicit override always wins.
  //   2. Global env `USE_CUSTOM_PROMPT_FOR_MAIN=1` enables ONLY the main
  //      container; trusted/untrusted stay on the preset.
  //   3. Otherwise OFF.
  // The agent-runner reads `process.env.USE_CUSTOM_PROMPT === '1'` and
  // falls back to the preset path on any other value (or absence).
  const globalUseCustomForMain = process.env.USE_CUSTOM_PROMPT_FOR_MAIN === '1';
  const useCustomPromptResolved =
    typeof group.containerConfig?.useCustomPrompt === 'boolean'
      ? group.containerConfig.useCustomPrompt
      : globalUseCustomForMain && isMain;
  if (useCustomPromptResolved) {
    args.push('-e', 'USE_CUSTOM_PROMPT=1');
    logger.info(
      {
        groupFolder: group.folder,
        source:
          typeof group.containerConfig?.useCustomPrompt === 'boolean'
            ? 'containerConfig'
            : 'USE_CUSTOM_PROMPT_FOR_MAIN',
      },
      'USE_CUSTOM_PROMPT enabled — agent-runner will load per-tier custom prompt',
    );
  }

  // Forward the orchestrator's authoritative assistant identity into the
  // agent container so the agent-runner can prepend an identity preamble
  // to systemPromptAppend (see container/agent-runner/src/index.ts —
  // buildIdentityPreamble). These are not secrets; pass them directly via
  // -e rather than going through the SECRET_FILES env-file plumbing.
  // Without this, untrusted-tier containers lacking explicit identity
  // statements in their persona files have been observed templating
  // themselves from fictional examples in tile rules (e.g. @AyeAye /
  // @AyeAyeSureBot from nanoclaw-core 0.1.94).
  args.push('-e', `ASSISTANT_NAME=${ASSISTANT_NAME}`);
  // Forward the full alias list (not just the primary entry) so the
  // agent-runner's identity preamble can teach the agent every handle
  // that resolves to it (#464). Single-handle installs forward a
  // single token verbatim; multi-handle installs see the joined
  // comma-separated form, which the agent-runner re-parses via
  // `parseUsernames` in `identity-preamble.ts`.
  args.push('-e', `ASSISTANT_USERNAME=${ASSISTANT_USERNAMES.join(',')}`);

  // Tell agent-runner whether the host is running the optional
  // observer pipeline (src/observer.ts). When true, agent-runner
  // emits raw thinking / text / tool_use input / tool_result preview
  // content on stderr — those payloads carry credentials, tokens,
  // and personal data per jbaruch/coding-policy: no-secrets, so they
  // are intentionally disclosed only when the operator explicitly
  // opted into observation by setting OBSERVER_CHAT_JID. With the
  // flag unset (default), agent-runner logs metadata only (block
  // counts, tool name, ok/error status). Same env-source order as
  // observer.ts so a setting in either process.env or .env file
  // toggles both halves consistently.
  const observerHostJid =
    process.env.OBSERVER_CHAT_JID ||
    readEnvFile(['OBSERVER_CHAT_JID']).OBSERVER_CHAT_JID;
  if (observerHostJid) {
    args.push('-e', 'OBSERVER_ENABLED=1');
  }

  // Kill-auto-compaction master flag (issue #104, design at
  // docs/proposals/kill-auto-compaction.md). When the orchestrator's
  // `ENABLE_THRESHOLD_NUKE` is set, the SDK's auto-compaction is
  // disabled — the orchestrator's threshold-nuke handshake replaces
  // it end-to-end. When the flag is off (default), DISABLE_COMPACT is
  // NOT set, auto-compaction stays on, and the orchestrator's
  // threshold detector runs in observe-only mode (telemetry +
  // checkpoint format validation, no destructive nuke). The two
  // halves are gated by the same flag because they must ship
  // together — disabling compaction without the replacement leaves
  // long sessions with no overflow safety net.
  if (ENABLE_THRESHOLD_NUKE) {
    args.push('-e', 'DISABLE_COMPACT=1');
    // Deliberately NOT forwarding CLAUDE_CODE_AUTO_COMPACT_WINDOW here
    // (issue #252). DISABLE_COMPACT=1 only suppresses the SDK's
    // `isAboveAutoCompactThreshold` flag — `Jn()` still reads the env
    // var unconditionally and feeds it into the `isAtBlockingLimit`
    // check (`window − reserved_for_output − e_7`, ~window − 30k).
    // Clamping the working window at or near the orchestrator's 800k
    // nuke threshold drops blocking-limit below the nuke and wedges
    // the request whose response would have triggered the handshake.
    // Letting the SDK fall back to the model context window (1M for
    // opus[1m]) keeps blocking-limit ~970k, well clear of the nuke.
  } else {
    args.push(
      '-e',
      `CLAUDE_CODE_AUTO_COMPACT_WINDOW=${AGENT_AUTO_COMPACT_WINDOW}`,
    );
  }

  // Pass chat JID so container scripts know which group they're in
  if (chatJid) {
    args.push('-e', `NANOCLAW_CHAT_JID=${chatJid}`);
  }

  // Pass reply-to message ID so the first IPC send_message appears as a Telegram reply
  if (replyToMessageId) {
    args.push('-e', `NANOCLAW_REPLY_TO_MESSAGE_ID=${replyToMessageId}`);
  }

  // Continuation marker for self-resuming cycles (#93/#130). Both env
  // vars are emitted together when the scheduled-task row carried a
  // non-NULL `continuation_cycle_id`; neither is emitted on a fresh
  // invocation. The calling skill checks both signals (env vars + the
  // prompt prefix it parses out of the task prompt) and fails closed to
  // "fresh invocation" if they disagree, so the env presence is
  // load-bearing — never paper over a missing value with a default.
  if (continuationCycleId) {
    args.push('-e', 'NANOCLAW_CONTINUATION=1');
    args.push('-e', `NANOCLAW_CONTINUATION_CYCLE_ID=${continuationCycleId}`);
  }

  // Route API traffic through the credential proxy (containers never see real secrets).
  // When an attribution token is supplied (#479 / ligolnik#125), embed it as a
  // path prefix so the proxy can map each request back to {group, tier,
  // session, task_id} for the JSONL usage log. The Claude SDK joins endpoint
  // paths to the base URL, so requests arrive at the proxy as
  // `/c/<token>/v1/messages`. Backward-compatible: a missing token degrades
  // to the un-prefixed URL and the proxy records `group: "unknown"`.
  const baseProxyUrl = `http://${CONTAINER_HOST_GATEWAY}:${CREDENTIAL_PROXY_PORT}`;
  const proxyBaseUrl = attributionToken
    ? `${baseProxyUrl}/c/${attributionToken}`
    : baseProxyUrl;
  args.push('-e', `ANTHROPIC_BASE_URL=${proxyBaseUrl}`);

  // Mirror the host's auth method with a placeholder value.
  // API key mode: SDK sends x-api-key, proxy replaces with real key.
  // OAuth mode:   SDK exchanges placeholder token for temp API key,
  //               proxy injects real OAuth token on that exchange request.
  const authMode = detectAuthMode();
  if (authMode === 'api-key') {
    args.push('-e', 'ANTHROPIC_API_KEY=placeholder');
  } else {
    args.push('-e', 'CLAUDE_CODE_OAUTH_TOKEN=placeholder');
  }

  // Runtime-specific args for host gateway resolution
  args.push(...hostGatewayArgs());

  // Run as host user so bind-mounted files are accessible.
  // In DooD, process.getuid() returns the orchestrator container's uid (1000),
  // not the actual host user. HOST_UID/HOST_GID override this.
  const effectiveUid = HOST_UID ?? process.getuid?.();
  const effectiveGid = HOST_GID ?? process.getgid?.();
  if (effectiveUid != null && effectiveUid !== 0 && effectiveUid !== 1000) {
    args.push('--user', `${effectiveUid}:${effectiveGid}`);
    args.push('-e', 'HOME=/home/node');
  }

  for (const mount of mounts) {
    if (mount.readonly) {
      args.push(...readonlyMountArgs(mount.hostPath, mount.containerPath));
    } else {
      args.push('-v', `${mount.hostPath}:${mount.containerPath}`);
    }
  }

  // #746: CONTAINER_IMAGE is intentionally NOT pushed here. It is appended by
  // the caller AFTER applyOneCliToSpawn runs, so the OneCLI `-e`/`-v`/
  // `--add-host` flags (which the SDK appends) land as `docker run` options
  // before the image token rather than as the container COMMAND after it.
  return {
    args,
    cleanup: secretEnvFile ? secretEnvFile.cleanup : () => {},
  };
}

export async function runContainerAgent(
  group: RegisteredGroup,
  input: ContainerInput,
  onProcess: (proc: ChildProcess, containerName: string) => void,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<ContainerOutput> {
  const startTime = Date.now();

  const groupDir = resolveGroupFolderPath(group.folder);
  fs.mkdirSync(groupDir, { recursive: true });

  const sessionName = input.sessionName ?? DEFAULT_SESSION_NAME;

  // Clean up stale _reply_to file from previous container runs in THIS
  // session. The file must match the path the container reads — with
  // per-session input dirs the container's `/workspace/ipc/input/_reply_to`
  // maps to `<ipc>/<group>/input-<sessionName>/_reply_to`, so the cleanup
  // must target the same session-scoped path. A cleanup against the legacy
  // shared `input/` path would leave the real file in place, and a
  // scheduled task with no replyToMessageId would quote a random old
  // message from a prior run.
  const replyToFile = path.join(
    resolveGroupIpcPath(group.folder),
    sessionInputDirName(sessionName),
    '_reply_to',
  );
  try {
    fs.unlinkSync(replyToFile);
  } catch {
    /* file doesn't exist — fine */
  }

  const mounts = buildVolumeMounts(
    group,
    input.isMain,
    input.chatJid,
    sessionName,
  );

  // #305 Phase 2a — cadence-registry rebuild. Walks the per-session
  // skills tree that `buildVolumeMounts` just published, parses each
  // SKILL.md's `cadence:` / `priority:` frontmatter, and idempotently
  // upserts declared schedules as `scheduled_tasks` rows with
  // `source = 'cadence-registry'`. Owner-scheduled rows (`source =
  // 'schedule-task'` from the existing `schedule-task` IPC path) are
  // untouched.
  //
  // Gated to `DEFAULT_SESSION_NAME` only. The maintenance session has
  // a filtered skill set (`MAINTENANCE_SKILL_BLOCKLIST` applied inside
  // `buildVolumeMounts`), so a maintenance spawn would see fewer
  // skills than the default session; running the rebuild from both
  // would race on the same group_folder rows and the maintenance view
  // would erase cadences declared by skills the default view installs
  // fine. Maintenance is a transient, short-lived spawn for
  // housekeeping — it should not claim authority over the registry.
  if (sessionName === DEFAULT_SESSION_NAME) {
    const skillsDir = path.join(
      DATA_DIR,
      'sessions',
      group.folder,
      sessionName,
      '.claude',
      'skills',
    );
    // Errors propagate by design: the rebuild's per-skill failures are
    // already collected into the `errors` array (cron parse, IANA
    // resolution, etc.) and don't throw; anything that throws past
    // that point is a real fault (db corruption, fs walk fails on
    // EACCES, etc.) and should fail the spawn loudly per the no-error
    // -suppression rule.
    const cadenceResult = rebuildCadenceRegistryForGroup({
      groupFolder: group.folder,
      chatJid: input.chatJid,
      // Cadence-registry rows derive from tile content delivered via
      // the staging→promote→publish→update pipeline (host-vetted,
      // owner-trusted) rather than any in-container agent action — so
      // they unwrap at fire time the same way legacy heartbeat seeders
      // do. The trust-boundary semantics for fire-time wrapping live
      // on `created_by_role`; the registry-vs-IPC provenance lives on
      // the new `source` column. Two columns, two questions.
      createdByRole: 'owner',
      skillsDir,
      computeNextRun: defaultComputeNextRun,
      now: () => new Date(),
    });
    if (
      cadenceResult.inserted > 0 ||
      cadenceResult.updated > 0 ||
      cadenceResult.deleted > 0 ||
      cadenceResult.errors.length > 0
    ) {
      logger.info(
        {
          groupFolder: group.folder,
          chatJid: input.chatJid,
          sessionName,
          ...cadenceResult,
        },
        'cadence-registry rebuilt',
      );
    }
  }

  const safeName = group.folder.replace(/[^a-zA-Z0-9-]/g, '-');
  // Suffix the container name with sessionName (when non-default) so that
  // `docker ps` makes it obvious which slot a running container occupies.
  // Main-group parallelism means two containers can share the group folder;
  // the sessionName tag distinguishes them.
  const sessionSuffix =
    sessionName === DEFAULT_SESSION_NAME ? '' : `-${sessionName}`;
  const containerName = `nanoclaw-${safeName}${sessionSuffix}-${Date.now()}`;

  // #213 Phase A spawn-collision detector. After a graceful-shutdown
  // handoff, the new orchestrator's GroupQueue starts empty — it has
  // no in-memory record of the adopted containers. If a new message
  // for one of those groups arrives during the handoff window, this
  // spawn path will run alongside the still-finishing adopted
  // container, racing on IPC and producing duplicate user-visible
  // replies. Phase B (in-memory adoption + docker-ps polling for
  // natural exit) closes that race; Phase A defers the work behind
  // this detector. When a collision is observed, the WARN message
  // names jbaruch/nanoclaw#213 directly so an operator hitting it
  // weeks/months from now knows exactly where to look — the
  // detector is the trigger to ship Phase B.
  if (isHandoffActive()) {
    const collisionPrefix = `nanoclaw-${safeName}${sessionSuffix}-`;
    // Use CONTAINER_RUNTIME_BIN, not a hard-coded 'docker', so the
    // detector follows the codebase's "swap runtimes by changing one
    // file" contract documented at the top of `container-runtime.ts`.
    const psResult = spawnSync(
      CONTAINER_RUNTIME_BIN,
      ['ps', '--format', '{{.Names}}', '--filter', `name=${collisionPrefix}`],
      { stdio: ['pipe', 'pipe', 'pipe'], encoding: 'utf-8' },
    );
    // spawnSync doesn't throw on non-zero exit; an unreachable docker
    // daemon, permission error, or missing binary surfaces as
    // `psResult.error` (spawn-side) or `psResult.status !== 0`
    // (process-side). Either way, treat the detector as unavailable
    // for this spawn and continue — the spawn itself must NOT be
    // blocked by an observability surface.
    if (psResult.error || psResult.status !== 0) {
      logger.debug(
        {
          issue: 'jbaruch/nanoclaw#213',
          err: psResult.error,
          status: psResult.status,
          stderr: psResult.stderr?.toString().slice(0, 500),
        },
        'Spawn-collision detector unavailable (docker ps failed) — skipping',
      );
    } else {
      const collisions = (psResult.stdout || '')
        .split('\n')
        .map((n) => n.trim())
        .filter((n) => n.startsWith(collisionPrefix));
      if (collisions.length > 0) {
        logger.warn(
          {
            issue: 'jbaruch/nanoclaw#213',
            phase: 'A',
            newContainer: containerName,
            collidingWith: collisions,
            groupFolder: group.folder,
            sessionName,
          },
          'Spawn collision during graceful-shutdown handoff window — ' +
            'an adopted container from the prior orchestrator is still ' +
            'running for this group/session. Both will respond to inbound ' +
            'IPC, producing duplicate user-visible replies. This is the ' +
            'Phase B race the #213 fix deferred; observed firing means it ' +
            'is time to implement in-memory adoption (register adopted ' +
            'containers in GroupQueue, poll docker ps for natural exit). ' +
            'See src/handoff.ts and the comment block above this log.',
        );
      }
    }
  }
  // Per-spawn attribution token for the credential proxy's usage log
  // (#479 / ligolnik#125). Lives only as long as the container; unregistered
  // on close + spawn-error. The token is generated here so the URL embedded
  // in the container env is unique per spawn, even across restarts of the
  // same group/session.
  const trustTier: TrustTier = input.isMain
    ? 'main'
    : group.containerConfig?.trusted === true
      ? 'trusted'
      : 'untrusted';
  const attributionToken = registerContainer({
    group: group.folder,
    tier: trustTier,
    session: sessionName,
    task_id: input.isScheduledTask ? sessionName : null,
    // #479 sub-#1: prefer the gate-verdict's trigger message id
    // (`triggerMessageId`) so attribution lands on the message that
    // actually caused the spawn. Fall back to `replyToMessageId` for
    // callers that don't yet plumb the gate verdict — rare, since
    // `evaluateGateChain` already computes it; and finally null for
    // scheduled tasks / IPC scripts / housekeeping.
    message_id: input.triggerMessageId ?? input.replyToMessageId ?? null,
  });

  // Wrap the spawn-path so a sync throw between registration and the
  // close/error handler attach (mostly fs.mkdirSync calls below)
  // can't leak the registry entry. The promise itself never rejects
  // (close/error handlers always resolve()), so this catch only fires
  // on the early-throw window. The handlers below also call
  // unregisterContainer; unregister is idempotent regardless.
  try {
    const { args: containerArgs, cleanup: cleanupSecretEnvFile } =
      buildContainerArgs(
        mounts,
        containerName,
        group,
        input.isMain,
        sessionName,
        input.replyToMessageId,
        input.chatJid,
        input.continuationCycleId,
        attributionToken,
        input.taskAgentModel,
      );

    // #564 groundwork: when OneCLI is configured, mutate containerArgs to add
    // HTTPS_PROXY env, mount the OneCLI CA bundle, and add
    // host.docker.internal mapping. The synchronous gate keeps the pre-#635
    // spawn path microtask-free when OneCLI is off — relevant for tests that
    // emit 'close' between `runContainerAgent` invocation and the spawn
    // registering its close handler.
    // #637: agent-spawn proxy injection is gated on a SEPARATE flag, not just
    // isOneCliConfigured. #637 enables OneCLI for the credential-proxy's
    // Anthropic hop while agents stay proxy-less; putting OneCLI in front of
    // agent traffic is #640. Without this split, re-enabling ONECLI_URL for
    // #637 would re-apply the agent proxy that broke the LLM path.
    if (isOneCliConfigured() && oneCliAgentProxyEnabled()) {
      await applyOneCliToSpawn(containerArgs, trustTier);
    }

    // #746: append the image LAST — after OneCLI's flags — so `docker run`
    // parses those flags as run options, not as the container COMMAND.
    // buildContainerArgs deliberately returns pre-image args for this reason.
    containerArgs.push(CONTAINER_IMAGE);

    logger.debug(
      {
        group: group.name,
        containerName,
        mounts: mounts.map(
          (m) =>
            `${m.hostPath} -> ${m.containerPath}${m.readonly ? ' (ro)' : ''}`,
        ),
        containerArgs: containerArgs.join(' '),
      },
      'Container mount configuration',
    );

    logger.info(
      {
        group: group.name,
        containerName,
        mountCount: mounts.length,
        isMain: input.isMain,
      },
      'Spawning container agent',
    );

    const logsDir = path.join(groupDir, 'logs');
    fs.mkdirSync(logsDir, { recursive: true });

    // Per-spawn streaming log under host-logs/. Distinct from the
    // post-exit summary written at `logsDir` below — the streaming file
    // captures output line-by-line as the container produces it, so the
    // admin tile can read what a stuck container is actually doing
    // without waiting for it to exit. Failure to open the stream is
    // non-fatal: the container still runs, we just lose the host-logs
    // copy for this spawn (the in-memory buffers + post-exit summary
    // remain unaffected).
    const spawnStartedAt = new Date();
    // `input.sessionName` is optional on the type but is always set by
    // every caller that actually invokes runContainerAgent (default or
    // maintenance). Fall back to the canonical default to keep the file
    // path deterministic if a future caller forgets to stamp the field.
    const streamSessionName = input.sessionName || DEFAULT_SESSION_NAME;
    const streamLogPath = containerLogPath(
      group.folder,
      streamSessionName,
      spawnStartedAt,
    );
    let streamLog: fs.WriteStream | null = null;
    try {
      fs.mkdirSync(path.dirname(streamLogPath), { recursive: true });
      streamLog = fs.createWriteStream(streamLogPath, { flags: 'a' });
      // Attach an error listener BEFORE the first write. Stream errors
      // emit async (e.g. ENOENT if the parent dir is wiped between
      // mkdir and create, EBADF if the fd is reaped) — without a
      // listener, Node's default handler is "throw uncaught exception"
      // which would crash the orchestrator on a logging path. This must
      // never happen: lose the streamed log, keep serving the user.
      streamLog.on('error', (err) => {
        logger.warn(
          { err, group: group.name, streamLogPath },
          'host-logs stream errored mid-spawn; dropping per-spawn stream',
        );
        // Null the local ref so subsequent writes from the data
        // listeners no-op rather than try to push into a broken stream.
        streamLog = null;
      });
      streamLog.write(
        [
          `=== Container Stream Log ===`,
          `Group: ${group.name}`,
          `Folder: ${group.folder}`,
          `Session: ${streamSessionName}`,
          `Container: ${containerName}`,
          `Start: ${spawnStartedAt.toISOString()}`,
          `=== STDOUT/STDERR (line-prefixed) ===`,
          ``,
        ].join('\n'),
      );
    } catch (err) {
      logger.warn(
        { err, group: group.name, streamLogPath },
        'host-logs stream open failed; container will run without per-spawn stream',
      );
      streamLog = null;
    }

    // Buffered line writer per stream. Container output isn't line-
    // aligned (a single `data` event can split a line, or contain many),
    // so we buffer until we see `\n` and emit `[OUT] ` / `[ERR] ` per
    // complete line. Trailing partial line is flushed on container exit.
    //
    // The buffer is bounded: a misbehaving container that emits megabytes
    // without a newline (e.g. binary garbage, a long single-line log
    // dump) would otherwise grow the orchestrator's heap unboundedly.
    // Cap at 64 KB per stream — well above typical line lengths but
    // small enough that even pathological output flushes quickly.
    const LINE_BUFFER_MAX = 64 * 1024;
    const makeLinePrefixer = (prefix: string) => {
      let buffer = '';
      const writeLines = (chunk: string) => {
        if (!streamLog) return;
        buffer += chunk;
        let nl: number;
        while ((nl = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, nl);
          buffer = buffer.slice(nl + 1);
          streamLog.write(`${prefix} ${stripAnsi(line)}\n`);
        }
        // Cap the buffer: if no newline appeared and the buffer crossed
        // the limit, flush the entire current buffer as a synthetic
        // line. Tagged with `[…cap…]` so a reader knows the line was
        // not delimited by a real newline (might cut mid-token).
        if (buffer.length > LINE_BUFFER_MAX) {
          streamLog.write(`${prefix} […cap…] ${stripAnsi(buffer)}\n`);
          buffer = '';
        }
      };
      const flush = () => {
        if (!streamLog || buffer.length === 0) return;
        streamLog.write(`${prefix} ${stripAnsi(buffer)}\n`);
        buffer = '';
      };
      return { writeLines, flush };
    };
    const stdoutPrefixer = makeLinePrefixer('[OUT]');
    const stderrPrefixer = makeLinePrefixer('[ERR]');

    return await new Promise<ContainerOutput>((resolve) => {
      const container = spawn(CONTAINER_RUNTIME_BIN, containerArgs, {
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      onProcess(container, containerName);

      let stdout = '';
      let stderr = '';
      let stdoutTruncated = false;
      let stderrTruncated = false;

      container.stdin.write(JSON.stringify(input));
      container.stdin.end();

      // Streaming output: parse OUTPUT_START/END marker pairs as they arrive
      let parseBuffer = '';
      let newSessionId: string | undefined;
      let outputChain = Promise.resolve();

      container.stdout.on('data', (data) => {
        const chunk = data.toString();

        // Streaming host-logs copy. Best-effort — write errors don't
        // affect the buffer accumulation or marker parsing below.
        stdoutPrefixer.writeLines(chunk);

        // Always accumulate for logging
        if (!stdoutTruncated) {
          const remaining = CONTAINER_MAX_OUTPUT_SIZE - stdout.length;
          if (chunk.length > remaining) {
            stdout += chunk.slice(0, remaining);
            stdoutTruncated = true;
            logger.warn(
              { group: group.name, size: stdout.length },
              'Container stdout truncated due to size limit',
            );
          } else {
            stdout += chunk;
          }
        }

        // Stream-parse for output markers
        if (onOutput) {
          parseBuffer += chunk;
          let startIdx: number;
          while ((startIdx = parseBuffer.indexOf(OUTPUT_START_MARKER)) !== -1) {
            const endIdx = parseBuffer.indexOf(OUTPUT_END_MARKER, startIdx);
            if (endIdx === -1) break; // Incomplete pair, wait for more data

            const jsonStr = parseBuffer
              .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
              .trim();
            parseBuffer = parseBuffer.slice(endIdx + OUTPUT_END_MARKER.length);

            try {
              const parsed: ContainerOutput = JSON.parse(jsonStr);
              if (parsed.newSessionId) {
                newSessionId = parsed.newSessionId;
              }
              hadStreamingOutput = true;
              // #682 — distinguish a TERMINAL result marker from an
              // intermediate streaming PREVIEW. The agent-runner emits a
              // preview marker — the only `writeOutput` carrying
              // `streamText` (container/agent-runner/src/index.ts) — on
              // every throttled assistant-text snapshot; every terminal
              // marker (real result, empty-turn success, #461 silent-stop
              // synthesis, error, precheck-skip) omits `streamText`.
              // `hadStreamingOutput` flips for ANY marker, so it can't
              // tell "delivered a result" from "only previewed"; this
              // flag can. A clean exit that produced only previews never
              // delivered a result — recording it `success` is the
              // silent-success shape #682 closes.
              if (parsed.streamText === undefined) {
                hadTerminalResult = true;
                // #689 — latch: a `requires_delivery` skill stamped this
                // TERMINAL `success` marker `noDelivery` because it ran
                // without delivering any user-facing content. We keep
                // the slot-draining `success` semantics (scheduleClose
                // still fires), but the close handler below downgrades
                // the resolved run to `killed` (retriable). Latched so a
                // later plain session-update success marker can't clear
                // it. Scoped inside the terminal-marker guard (same
                // `streamText === undefined` discriminator as
                // `hadTerminalResult`): the runner only ever stamps
                // `noDelivery` on terminal markers, and a preview marker
                // must never trip the downgrade.
                if (parsed.noDelivery === true) {
                  sawNoDeliveryMarker = true;
                }
              }
              // Activity detected — reset the hard timeout
              resetTimeout();
              // Call onOutput for all markers (including null results)
              // so idle timers start even for "silent" query completions.
              outputChain = outputChain.then(() => onOutput(parsed));
            } catch (err) {
              logger.warn(
                { group: group.name, error: err },
                'Failed to parse streamed output chunk',
              );
            }
          }
        }
      });

      container.stderr.on('data', (data) => {
        const chunk = data.toString();
        stderrPrefixer.writeLines(chunk);
        const lines = chunk.trim().split('\n');
        for (const line of lines) {
          if (line) {
            logger.debug({ container: group.folder }, line);
            onAgentLine(group.folder, line);
          }
        }
        // Don't reset timeout on stderr — SDK writes debug logs continuously.
        // Timeout only resets on actual output (OUTPUT_MARKER in stdout).
        if (stderrTruncated) return;
        const remaining = CONTAINER_MAX_OUTPUT_SIZE - stderr.length;
        if (chunk.length > remaining) {
          stderr += chunk.slice(0, remaining);
          stderrTruncated = true;
          logger.warn(
            { group: group.name, size: stderr.length },
            'Container stderr truncated due to size limit',
          );
        } else {
          stderr += chunk;
        }
      });

      let timedOut = false;
      let hadStreamingOutput = false;
      // #682 — set true once a TERMINAL result marker (one without
      // `streamText`) is observed. The precise "did the run actually
      // deliver a terminal result?" signal the clean-exit and timeout
      // classifications below gate on, distinct from `hadStreamingOutput`
      // (which flips for streaming previews too).
      let hadTerminalResult = false;
      // #689 — set true once a terminal marker stamped `noDelivery`
      // (a `requires_delivery` skill that ended a run without delivering
      // user-facing content) is observed. Latched across markers so a
      // trailing plain session-update success can't clear it. The
      // clean-exit and timeout classifications below downgrade such a
      // maintenance run to `killed` despite `hadTerminalResult` being
      // set by the same (success-status) marker.
      let sawNoDeliveryMarker = false;
      // Untrusted containers get shorter timeout (5 min vs 30 min default)
      const UNTRUSTED_TIMEOUT = 300_000;
      const defaultTimeout =
        input.isMain || group.containerConfig?.trusted
          ? CONTAINER_TIMEOUT
          : UNTRUSTED_TIMEOUT;
      const configTimeout = group.containerConfig?.timeout || defaultTimeout;
      // #461 — maintenance-session inactivity timeout. The kill timer
      // here is reset by `resetTimeout()` on every streamed stdout
      // marker (see the `hadStreamingOutput = true; resetTimeout();`
      // block lower in this function), so this is an *inactivity*
      // timeout, not a wall-clock cap — same shape as the existing
      // default-session timer.
      //
      // Maintenance work is single-turn burst-then-quiet, so it
      // doesn't need the user-facing default's `IDLE_TIMEOUT + 30s`
      // graceful-close floor (that floor exists so a multi-turn
      // conversation can drain through `_close`). With the
      // agent-runner's silent-stop synthesis (#461 layer 1), a healthy
      // maintenance run signals teardown within seconds; this shorter
      // window is the backstop for the "SDK hung past graceful close"
      // pathology. Bypass the IDLE_TIMEOUT floor for maintenance only.
      //
      // Per-group `containerConfig.timeout` still wins when set so
      // operators can extend the window for groups with heavy precheck
      // scripts that run silently for longer than the env default.
      const isMaintenanceSession = sessionName === MAINTENANCE_SESSION_NAME;
      const timeoutMs = isMaintenanceSession
        ? group.containerConfig?.timeout || MAINTENANCE_CONTAINER_TIMEOUT
        : Math.max(configTimeout, IDLE_TIMEOUT + 30_000);

      const killOnTimeout = () => {
        timedOut = true;
        logger.error(
          { group: group.name, containerName },
          'Container timeout, stopping gracefully',
        );
        try {
          stopContainer(containerName);
        } catch (err) {
          logger.warn(
            { group: group.name, containerName, err },
            'Graceful stop failed, force killing',
          );
          container.kill('SIGKILL');
        }
      };

      let timeout = setTimeout(killOnTimeout, timeoutMs);

      // Reset the timeout whenever there's activity (streaming output)
      const resetTimeout = () => {
        clearTimeout(timeout);
        timeout = setTimeout(killOnTimeout, timeoutMs);
      };

      container.on('close', (code) => {
        clearTimeout(timeout);
        // Remove the secret env-file (if any) as soon as docker has
        // exited — the file's only consumer is the docker daemon at
        // spawn time, so the window of exposure ends with the close
        // event. cleanup() is idempotent; the error handler below
        // calls it too in case `close` is skipped (spawn ENOENT etc).
        cleanupSecretEnvFile();
        // Drop the proxy-registry entry for this spawn so dead tokens
        // don't accumulate in memory. Idempotent.
        unregisterContainer(attributionToken);
        const duration = Date.now() - startTime;

        // Flush any partial trailing line and close the streaming log.
        // Failure to close cleanly is non-fatal — Node will GC the fd
        // eventually; the streamed bytes already on disk are intact.
        stdoutPrefixer.flush();
        stderrPrefixer.flush();
        if (streamLog) {
          try {
            streamLog.write(
              `\n=== Container Exited ===\nCode: ${code}\nDuration: ${duration}ms\nEnd: ${new Date().toISOString()}\n`,
            );
            streamLog.end();
          } catch {
            // Stream already errored / closed — nothing useful to do.
          }
        }

        if (timedOut) {
          const ts = new Date().toISOString().replace(/[:.]/g, '-');
          const timeoutLog = path.join(logsDir, `container-${ts}.log`);
          fs.writeFileSync(
            timeoutLog,
            [
              `=== Container Run Log (TIMEOUT) ===`,
              `Timestamp: ${new Date().toISOString()}`,
              `Group: ${group.name}`,
              `Container: ${containerName}`,
              `Duration: ${duration}ms`,
              `Exit Code: ${code}`,
              `Had Streaming Output: ${hadStreamingOutput}`,
            ].join('\n'),
          );

          // #589 (reopened) / #682 — for MAINTENANCE one-shots, streamed
          // preview output is NOT a delivered result. A healthy
          // maintenance run reaches a terminal result and exits
          // naturally (scheduleClose → `_close`) within seconds, so a
          // run still alive at this inactivity timeout having produced
          // only previews was reaped mid-compose (e.g. the morning-brief
          // compose turn stalled on an LLM / proxy blip). Resolving
          // `'success'` here hid that incomplete run (the original #589
          // silent-stop). Classify `'killed'` so `task_run_logs` records
          // non-success and the run is retriable / redeliverable.
          //
          // #682 replaced #589's `hadStreamingOutput` gate with the
          // precise `hadTerminalResult` signal: the rare "delivered a
          // terminal result, then the SDK iterator hung until the host
          // reaped it" shape now keeps its idle-cleanup `success` (the
          // work landed) instead of being over-reported as killed. Only
          // a reap after previews-but-no-terminal-result is `killed`.
          //
          // #689 — `sawNoDeliveryMarker` also forces `killed` here: a
          // `requires_delivery` skill that emitted a terminal marker
          // (so `hadTerminalResult` is set) stamped `noDelivery` because
          // it never delivered to chat. Without this, such a run reaped
          // at the inactivity timeout would fall through to the
          // `hadStreamingOutput` idle-cleanup `success` below.
          if (
            isMaintenanceSession &&
            hadStreamingOutput &&
            (!hadTerminalResult || sawNoDeliveryMarker)
          ) {
            logger.warn(
              {
                group: group.name,
                containerName,
                duration,
                code,
                sawNoDeliveryMarker,
              },
              'Maintenance container reaped by inactivity timeout without delivering user-facing content — classifying killed (incomplete, retriable) (#682/#689)',
            );
            outputChain.then(() => {
              resolve({
                status: 'killed',
                result: null,
                newSessionId,
                error: sawNoDeliveryMarker
                  ? `Maintenance container reaped by inactivity timeout after ${timeoutMs}ms; the requires_delivery skill delivered no user-facing content (noDelivery marker) — incomplete run (reaped mid-compose), retriable`
                  : `Maintenance container reaped by inactivity timeout after ${timeoutMs}ms having streamed only preview output and no terminal result — incomplete run (reaped mid-compose), retriable`,
              });
            });
            return;
          }

          // Interactive / default session, OR a maintenance run that
          // delivered a terminal result before idling out: timeout after
          // output = idle cleanup, not failure. The agent already
          // streamed its reply (and, for multi-turn chats, sent it via
          // `send_message`); this is just the container being reaped
          // after the idle period expired. Unchanged from the pre-#589
          // contract.
          if (hadStreamingOutput) {
            logger.info(
              { group: group.name, containerName, duration, code },
              'Container timed out after output (idle cleanup)',
            );
            outputChain.then(() => {
              resolve({
                status: 'success',
                result: null,
                newSessionId,
              });
            });
            return;
          }

          logger.error(
            { group: group.name, containerName, duration, code },
            'Container timed out with no output',
          );

          resolve({
            status: 'error',
            result: null,
            error: `Container timed out after ${configTimeout}ms`,
          });
          return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const logFile = path.join(logsDir, `container-${timestamp}.log`);
        const isVerbose =
          process.env.LOG_LEVEL === 'debug' ||
          process.env.LOG_LEVEL === 'trace';

        const logLines = [
          `=== Container Run Log ===`,
          `Timestamp: ${new Date().toISOString()}`,
          `Group: ${group.name}`,
          `IsMain: ${input.isMain}`,
          `Duration: ${duration}ms`,
          `Exit Code: ${code}`,
          `Stdout Truncated: ${stdoutTruncated}`,
          `Stderr Truncated: ${stderrTruncated}`,
          ``,
        ];

        const isError = code !== 0;

        if (isVerbose || isError) {
          // On error, log input metadata only — not the full prompt.
          // Full input is only included at verbose level to avoid
          // persisting user conversation content on every non-zero exit.
          if (isVerbose) {
            logLines.push(`=== Input ===`, JSON.stringify(input, null, 2), ``);
          } else {
            logLines.push(
              `=== Input Summary ===`,
              `Prompt length: ${input.prompt.length} chars`,
              `Session ID: ${input.sessionId || 'new'}`,
              ``,
            );
          }
          logLines.push(
            `=== Container Args ===`,
            containerArgs.join(' '),
            ``,
            `=== Mounts ===`,
            mounts
              .map(
                (m) =>
                  `${m.hostPath} -> ${m.containerPath}${m.readonly ? ' (ro)' : ''}`,
              )
              .join('\n'),
            ``,
            `=== Stderr${stderrTruncated ? ' (TRUNCATED)' : ''} ===`,
            stderr,
            ``,
            `=== Stdout${stdoutTruncated ? ' (TRUNCATED)' : ''} ===`,
            stdout,
          );
        } else {
          logLines.push(
            `=== Input Summary ===`,
            `Prompt length: ${input.prompt.length} chars`,
            `Session ID: ${input.sessionId || 'new'}`,
            ``,
            `=== Mounts ===`,
            mounts
              .map((m) => `${m.containerPath}${m.readonly ? ' (ro)' : ''}`)
              .join('\n'),
            ``,
          );
        }

        fs.writeFileSync(logFile, logLines.join('\n'));
        logger.debug({ logFile, verbose: isVerbose }, 'Container log written');

        if (code !== 0) {
          logger.error(
            {
              group: group.name,
              code,
              duration,
              stderr,
              stdout,
              logFile,
            },
            'Container exited with error',
          );

          resolve({
            status: 'error',
            result: null,
            error: `Container exited with code ${code}: ${stderr.slice(-200)}`,
          });
          return;
        }

        // Streaming mode: wait for output chain to settle, return completion marker
        if (onOutput) {
          // #682 — a clean exit (code 0) is recorded `success` only when
          // the run actually delivered a terminal result. A MAINTENANCE
          // container that exits 0 having produced only streaming
          // previews (or nothing) never reached a terminal result — e.g.
          // an in-container `process.exit(0)` fired mid-compose before
          // the SDK loop / #461 silent-stop synthesis could emit one
          // (the 2026-06-11 morning-brief shape: the hard-exit watchdog
          // `process.exit(0)`'d after streaming the compose preview but
          // before `send_message`). `code === 0` + `timedOut === false`
          // lands here, and the old unconditional `success` made that
          // incomplete run indistinguishable from a genuine delivery —
          // zero alerting fired and the brief was never sent. Classify
          // `killed` so `task_run_logs` records a non-success, retriable
          // run. The #461 silent-stop no-op success is NOT mis-flagged:
          // it emits a terminal marker (`{status:'success', result:''}`,
          // no `streamText`), so `hadTerminalResult` is set and it stays
          // `success`. Scoped to maintenance to mirror #683 — the
          // `'killed'` status is consumed by the task-scheduler, and
          // interactive idle-cleanup `success` semantics are unchanged.
          //
          // #689 — `sawNoDeliveryMarker` extends this to the case where
          // a `requires_delivery` skill DID emit a terminal marker
          // (so `hadTerminalResult` is set) but stamped it `noDelivery`
          // because the run delivered nothing to chat — the silent
          // success that defeated the #682 `!hadTerminalResult` gate
          // (morning-brief composed but never sent). Either signal
          // classifies the maintenance run `killed`.
          if (
            isMaintenanceSession &&
            (!hadTerminalResult || sawNoDeliveryMarker)
          ) {
            outputChain.then(() => {
              logger.warn(
                {
                  group: group.name,
                  duration,
                  newSessionId,
                  sawNoDeliveryMarker,
                },
                'Maintenance container exited cleanly (code 0) without delivering user-facing content — classifying killed (incomplete, retriable) (#682/#689)',
              );
              resolve({
                status: 'killed',
                result: null,
                newSessionId,
                error: sawNoDeliveryMarker
                  ? 'Maintenance container exited cleanly (code 0) but the requires_delivery skill delivered no user-facing content (noDelivery marker) — incomplete run (composed but never sent), retriable'
                  : 'Maintenance container exited cleanly (code 0) without delivering a terminal result — incomplete run (exited before producing/sending a result), retriable',
              });
            });
            return;
          }
          outputChain.then(() => {
            logger.info(
              { group: group.name, duration, newSessionId },
              'Container completed (streaming mode)',
            );
            resolve({
              status: 'success',
              result: null,
              newSessionId,
            });
          });
          return;
        }

        // Legacy mode: parse the last output marker pair from accumulated stdout
        try {
          // Extract JSON between sentinel markers for robust parsing
          const startIdx = stdout.indexOf(OUTPUT_START_MARKER);
          const endIdx = stdout.indexOf(OUTPUT_END_MARKER);

          let jsonLine: string;
          if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
            jsonLine = stdout
              .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
              .trim();
          } else {
            // Fallback: last non-empty line (backwards compatibility)
            const lines = stdout.trim().split('\n');
            jsonLine = lines[lines.length - 1];
          }

          const output: ContainerOutput = JSON.parse(jsonLine);

          logger.info(
            {
              group: group.name,
              duration,
              status: output.status,
              hasResult: !!output.result,
            },
            'Container completed',
          );

          resolve(output);
        } catch (err) {
          logger.error(
            {
              group: group.name,
              stdout,
              stderr,
              error: err,
            },
            'Failed to parse container output',
          );

          resolve({
            status: 'error',
            result: null,
            error: `Failed to parse container output: ${err instanceof Error ? err.message : String(err)}`,
          });
        }
      });

      container.on('error', (err) => {
        clearTimeout(timeout);
        // Spawn-error path: docker may never have read the env-file
        // (e.g. ENOENT on the docker binary itself), but the file is
        // still on disk. cleanup() is idempotent — safe to call here
        // and again from `close` if both fire.
        cleanupSecretEnvFile();
        unregisterContainer(attributionToken);
        logger.error(
          { group: group.name, containerName, error: err },
          'Container spawn error',
        );
        // Spawn-error path: the close handler may not fire on some
        // failure modes (e.g. spawn ENOENT — the binary doesn't exist),
        // so flush + close the streaming log here too. Without this the
        // file descriptor leaks until process GC and the file is left
        // open with no exit footer, which makes the on-disk record
        // ambiguous (was the container still running, or did it die
        // before producing any output?).
        stdoutPrefixer.flush();
        stderrPrefixer.flush();
        if (streamLog) {
          try {
            streamLog.write(
              `\n=== Container Spawn Failed ===\nError: ${err.message}\nEnd: ${new Date().toISOString()}\n`,
            );
            streamLog.end();
          } catch {
            // Stream already errored / closed — nothing to recover.
          }
        }
        resolve({
          status: 'error',
          result: null,
          error: `Container spawn error: ${err.message}`,
        });
      });
    });
  } catch (err) {
    // Sync throw before the spawn handlers wired up — handlers can't
    // unregister, so do it here and propagate.
    unregisterContainer(attributionToken);
    throw err;
  }
}

export function writeTasksSnapshot(
  groupFolder: string,
  isMain: boolean,
  tasks: Array<{
    id: string;
    groupFolder: string;
    prompt: string;
    script?: string | null;
    schedule_type: string;
    schedule_value: string;
    status: string;
    next_run: string | null;
  }>,
  isTrusted?: boolean,
): void {
  // Untrusted containers don't get IPC root files — tasks/ not mounted
  if (!isMain && !isTrusted) return;

  // Write filtered tasks to the group's IPC directory
  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

  // Main sees all tasks, others only see their own
  const filteredTasks = isMain
    ? tasks
    : tasks.filter((t) => t.groupFolder === groupFolder);

  const tasksFile = path.join(groupIpcDir, 'current_tasks.json');
  fs.writeFileSync(tasksFile, JSON.stringify(filteredTasks, null, 2));
}

export interface AvailableGroup {
  jid: string;
  name: string;
  lastActivity: string;
  isRegistered: boolean;
  containerConfig?: import('./types.js').RegisteredGroup['containerConfig'];
  requiresTrigger?: boolean;
}

/**
 * Write available groups snapshot for the container to read.
 * Only main group can see all available groups (for activation).
 * Non-main groups only see their own registration status.
 */
export function writeGroupsSnapshot(
  groupFolder: string,
  isMain: boolean,
  groups: AvailableGroup[],
  _registeredJids: Set<string>,
  isTrusted?: boolean,
): void {
  // Untrusted containers don't get IPC root files — available_groups not mounted
  if (!isMain && !isTrusted) return;

  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

  // Main sees all groups; others see nothing (they can't activate groups)
  const visibleGroups = isMain ? groups : [];

  const groupsFile = path.join(groupIpcDir, 'available_groups.json');

  // Preserve JID-keyed entries that agents may have written
  let existing: Record<string, unknown> = {};
  if (fs.existsSync(groupsFile)) {
    try {
      existing = JSON.parse(fs.readFileSync(groupsFile, 'utf-8'));
    } catch {
      existing = {};
    }
  }

  fs.writeFileSync(
    groupsFile,
    JSON.stringify(
      {
        ...existing,
        groups: visibleGroups,
        lastSync: new Date().toISOString(),
      },
      null,
      2,
    ),
  );
}
