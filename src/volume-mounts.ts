// Volume-mount construction (#851 slice 5, extracted verbatim from
// src/container-runner.ts).
//
// Everything that decides what a spawned agent container can see:
// the per-tier mount set, tile-content materialization into the
// group workspace, the OneCLI MITM CA delivery (#640), SECRET_FILES
// /dev/null shadowing, and the symlink-safe chown helpers.
import { randomBytes } from 'crypto';
import fs from 'fs';
import path from 'path';

import {
  DATA_DIR,
  GROUPS_DIR,
  HOST_GID,
  HOST_PROJECT_ROOT,
  HOST_UID,
  MAINTENANCE_RULE_BLOCKLIST,
  MAINTENANCE_SKILL_BLOCKLIST,
  TILE_OWNER,
} from './config.js';
import { createFilteredDb } from './filtered-db.js';
import {
  CR_FS_CODES,
  CR_READLINK_FS_CODES,
  isFsErrorWithCode,
} from './fs-errors.js';
import { isHandoffActive } from './handoff.js';
import { ensureHostLogDirs, hostLogsDir } from './host-logs.js';
import { sweepStaleInputs } from './ipc-input-sweep.js';
import { shouldIncludeRule } from './rule-requires-filter.js';
import { resolveGroupFolderPath, resolveGroupIpcPath } from './group-folder.js';
import { logger } from './logger.js';
import { validateAdditionalMounts } from './mount-security.js';
import { getOneCliOutboundConfig } from './onecli-client.js';
import {
  DEFAULT_SESSION_NAME,
  MAINTENANCE_SESSION_NAME,
  VALID_SESSION_NAME_RE,
  sessionInputDirName,
} from './session-names.js';
import {
  RACE_CODES,
  atomicPublishDir,
  computeEffectiveSkillContextForSpawn,
  getInstalledTiles,
  getRegistryTilesDir,
  rmBestEffort,
  selectTiles,
} from './tile-materialize.js';
import type { TrustTier } from './trust-tier.js';
import { RegisteredGroup } from './types.js';

export interface VolumeMount {
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

/** Container path the OneCLI MITM CA bundle is mounted at for proxied agents. */
const ONECLI_AGENT_CA_CONTAINER_PATH = '/onecli/ca.pem';

/** CA env vars re-pointed at the mounted bundle — covers curl, python-requests,
 * node, git, and openssl-linked tools. */
const ONECLI_AGENT_CA_ENV_VARS = [
  'SSL_CERT_FILE',
  'NODE_EXTRA_CA_CERTS',
  'REQUESTS_CA_BUNDLE',
  'CURL_CA_BUNDLE',
  'GIT_SSL_CAINFO',
] as const;

/**
 * Deliver the OneCLI MITM CA into an agent spawn so its gateway-proxied HTTPS
 * traffic can validate the MITM cert. Returns false when the CA content is
 * unavailable (gateway unreachable) — the caller MUST then fail the spawn
 * closed, because a proxied container without a trusted CA cannot complete ANY
 * external HTTPS (every request now traverses the MITM gateway).
 *
 * Why this exists (jbaruch/nanoclaw#640): the OneCLI SDK's own CA mount is
 * broken under Docker-out-of-Docker. It writes the CA to a temp file INSIDE the
 * orchestrator container and bind-mounts it, but the host docker daemon resolves
 * the `-v` source against the HOST filesystem (not the orchestrator container's),
 * so docker creates an empty directory and every agent gateway-MITM TLS
 * handshake fails. We instead write the CA content to a host-visible path via
 * `toHostPath` (the same DooD translation every other mount uses), mount it
 * read-only, and OVERRIDE every CA env var the SDK leaves broken/empty. The
 * `-v` + `-e` land AFTER the SDK's, so docker's last-wins resolves to ours.
 *
 * Written 0644 (world-readable) because agents run as a non-root `--user`, and
 * onecli's own `ca-bundle.pem` is 0770/uid-999 which a non-root agent cannot
 * read (proven in #640 throwaway-container tests).
 */
export async function mountOneCliAgentCa(
  args: string[],
  tier: TrustTier,
): Promise<boolean> {
  const outbound = await getOneCliOutboundConfig(tier);
  if (!outbound || !outbound.ca) return false;
  // Under DATA_DIR, NOT STORE_DIR: `store/` is bind-mounted RW into main/trusted
  // agents at /workspace/store, which would let an agent overwrite the backing
  // file of its own RO /onecli/ca.pem CA mount (trust-anchor tampering / DoS).
  // DATA_DIR is only mounted into agents via specific per-group subdirs
  // (state/, sessions/, ipc/) — a fresh onecli-ca/ subdir is never mounted.
  const caDir = path.join(DATA_DIR, 'onecli-ca');
  fs.mkdirSync(caDir, { recursive: true });
  const caFile = path.join(caDir, `${tier}.pem`);
  // Atomic write (unique tmp + rename) so concurrent same-tier spawns never
  // read a half-written file. The orchestrator is a SINGLE process, so the tmp
  // name must be unique per CALL — process.pid would collide across concurrent
  // spawns. Content is identical across spawns, so the final rename race is
  // benign. chmod after write because writeFileSync mode is subject to umask.
  const tmpFile = `${caFile}.${randomBytes(8).toString('hex')}.tmp`;
  fs.writeFileSync(tmpFile, outbound.ca, { mode: 0o644 });
  fs.chmodSync(tmpFile, 0o644);
  fs.renameSync(tmpFile, caFile);
  args.push('-v', `${toHostPath(caFile)}:${ONECLI_AGENT_CA_CONTAINER_PATH}:ro`);
  for (const envVar of ONECLI_AGENT_CA_ENV_VARS) {
    args.push('-e', `${envVar}=${ONECLI_AGENT_CA_CONTAINER_PATH}`);
  }
  return true;
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
        if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
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
          // and executed in-container by skills that shell out to a
          // published script by filename (e.g. `scheduler-timezone`'s
          // `compute-schedule-value.py`), independently of the agent
          // context. Even when a skill's SKILL.md is blocklisted
          // (excluded from the agent's loaded tile context to save
          // tokens), OTHER non-blocklisted skills can still invoke a
          // published script from this dir by name. Scripts don't
          // consume agent context (they're not in the SKILL.md surface
          // the SDK loads), so excluding them along with the prompt is
          // purely accidental. Copy unconditionally; only the
          // prompt-context copies below honour the blocklist.
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
        } catch (err) {
          if (!isFsErrorWithCode(err, CR_READLINK_FS_CODES)) throw err;
          /* ignore — if we can't read the link we just won't clean it up
             (EINVAL covers a TOCTOU replace of the symlink by a regular file
             between the lstat probe above and this readlink) */
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
      if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
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
        if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
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
      } catch (err) {
        if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
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
          if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
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
        if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
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
      if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
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
      if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
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
