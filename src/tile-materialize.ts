// Tile selection + skill-context materialization (#851 slice 3,
// extracted verbatim from src/container-runner.ts).
//
// The trust-tier tile baseline + per-chat overlay selection (#305),
// the local Tessl registry directory resolution, the maintenance
// skill-blocklist closure pre-scan (#544b/#441), and the concurrent-
// safe atomic directory publish used to materialize tile content
// into a group workspace.
import fs from 'fs';
import path from 'path';

import { TILE_OWNER } from './config.js';
import { isErrnoCodedError } from './fs-errors.js';
import { logger } from './logger.js';
import { computeEffectiveSkillContext } from './skill-dep-closure.js';

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

/**
 * #441 — read the text of a skill's `scripts/` and `references/` files so
 * `extractMountPathDeps` can see cross-skill `tessl__<name>/` mount-path
 * references that live in a subprocess call (not the SKILL.md prompt). The
 * 2026-07-12 morning-brief Step-9 failure: `resolve-reminder-schedule.py`
 * shells out to `tessl__scheduler-timezone/scripts/compute-schedule-value.py`,
 * but `scheduler-timezone` is on the maintenance blocklist and the closure —
 * scanning only SKILL.md for `Skill()` calls — never rescued it.
 *
 * Best-effort per `coding-policy: error-handling`: a missing subdir or file
 * (ENOENT) contributes no text; any other errno (EACCES, EIO, ENOTDIR)
 * propagates, matching `ingestSkillDir` — a perms/IO fault on the skill tree
 * is operator-actionable, not silently degraded. Bounded to the two subdirs,
 * one level deep (skill scripts/refs are flat); subdirectories (e.g.
 * `__pycache__`) are skipped.
 */
function readAuxSkillText(skillPath: string): string {
  const parts: string[] = [];
  for (const sub of ['scripts', 'references']) {
    const dir = path.join(skillPath, sub);
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
      continue;
    }
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      try {
        parts.push(fs.readFileSync(path.join(dir, entry.name), 'utf8'));
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
      }
    }
  }
  return parts.join('\n');
}

export function computeEffectiveSkillContextForSpawn(
  originalBlocklist: ReadonlySet<string>,
  tilesToInstall: readonly string[],
  registryTiles: string,
  groupDir: string,
): EffectiveSpawnSkillContext {
  const sources = new Map<string, string>();

  // #441 — the aux (scripts/references) scan only affects the outcome when
  // there IS a blocklist to rescue from. On a default-session spawn the
  // blocklist is empty, so every present skill is already a root and
  // `reachableSkills` = every skill regardless of mount-path deps; the scan
  // would be pure wasted spawn I/O (and needless false-positive surface for
  // `reachableSkills`). Gate it to non-empty blocklists (i.e. maintenance
  // spawns), where a subprocess dep on a blocklisted skill actually needs the
  // rescue.
  const scanAux = originalBlocklist.size > 0;

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
      let skillMd: string;
      try {
        skillMd = fs.readFileSync(skillMdPath, 'utf8');
      } catch (err) {
        // ENOENT on SKILL.md is normal for non-skill subdirs (rare
        // but possible). Other errno propagates per the same
        // rationale as readdir above.
        if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
        continue;
      }
      // #441 — a skill's cross-skill dependency can live in its scripts
      // (a subprocess call to another skill's mount path), not just its
      // SKILL.md. Fold the scripts/ + references/ text into the source blob
      // (only when a blocklist exists, per `scanAux`) so
      // `extractMountPathDeps` sees those `tessl__<name>/` references and the
      // closure rescues the depended-on skill from the blocklist.
      sources.set(
        skillDir,
        scanAux ? skillMd + '\n' + readAuxSkillText(skillPath) : skillMd,
      );
    }
  };

  for (const tileName of tilesToInstall) {
    ingestSkillDir(path.join(registryTiles, tileName, 'skills'));
  }
  ingestSkillDir(path.join(process.cwd(), 'container', 'skills'));
  ingestSkillDir(path.join(groupDir, 'skills'));

  return computeEffectiveSkillContext(originalBlocklist, sources);
}

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
export const RACE_CODES = new Set(['EEXIST', 'ENOTEMPTY', 'EPERM', 'EACCES']);
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
export function rmBestEffort(target: string): void {
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
          // Any fs/OS errno while restoring is logged (the original publish
          // error remains the primary signal that propagates via the outer
          // flow); only a non-errno defect (a real bug) propagates and could
          // legitimately displace it.
          if (!isErrnoCodedError(restoreErr)) throw restoreErr;
          const restoreCode = (restoreErr as NodeJS.ErrnoException).code;
          logger.warn(
            { err: restoreErr, code: restoreCode, dstDir, backupDir },
            'atomic dir publish: restore from backup failed; ' +
              'backup sibling left for operator recovery',
          );
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
