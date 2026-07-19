import fs from 'fs';
import path from 'path';

import { DATA_DIR } from './config.js';
import { BEST_EFFORT_FS_CODES, isFsErrorWithCode } from './fs-errors.js';
import { logger } from './logger.js';

/**
 * Delete the on-disk session artifacts (JSONL transcript and per-session
 * tool-results directory) for a given session slot, given the SDK
 * sessionId. Returns the number of filesystem entries actually removed —
 * up to 2 per slug (1 transcript + 1 tool-results dir) summed across
 * every project-slug subdirectory found.
 *
 * Path layout (host side):
 *   ${DATA_DIR}/sessions/<groupFolder>/<sessionName>/.claude/projects/<project-slug>/<sessionId>.jsonl
 *   ${DATA_DIR}/sessions/<groupFolder>/<sessionName>/.claude/projects/<project-slug>/<sessionId>/
 *
 * The project-slug is `-workspace-group` for our containers (see
 * CLAUDE_PROJECT_SLUG in container-runner.ts). We glob the projects/
 * directory rather than hardcoding the slug so a future change to the
 * slug — or any operator who renamed the workspace path — doesn't
 * silently leave stale artifacts behind.
 *
 * Used by `nukeSession` (#100) to actually wipe transcript state, and
 * by the scheduler's per-run finally (#193) to wipe scheduled-task
 * artifacts that aren't tracked in the sessions cache. Without this,
 * the next container spawn re-reads the JSONL and the bad state
 * (poison, stuck plan, corrupt memory) is immediately back, AND
 * orphan tool-results directories accumulate forever under the
 * maintenance slot.
 *
 * **Security**: `sessionId` ultimately originates from container stdout
 * (parsed `newSessionId` from the SDK's stream), which is *untrusted*
 * for untrusted-tier groups. A crafted value containing path separators
 * or `..` segments would otherwise be interpolated into the artifact
 * paths and could escape `projectsDir/<slug>/` to delete arbitrary
 * files or directories anywhere the orchestrator process can write.
 * Defense in depth:
 *   1. Reject anything that isn't a strict UUID-or-token charset.
 *   2. After joining, assert the resolved path stays inside `projectsDir`.
 *   3. The tool-results-dir helper additionally relies on Node's
 *      `fs.rmSync` not following symlinks during recursive removal,
 *      so a malicious container that scattered host-pointing symlinks
 *      inside its own dir cannot redirect the wipe outward.
 */
const SESSION_ID_PATTERN = /^[A-Za-z0-9_-]+$/;

/**
 * Try to unlink `${slugPath}/${sessionId}.jsonl`. Returns 1 if the
 * filesystem entry was unlinked, 0 otherwise.
 *
 * Two paths depending on what `${sessionId}.jsonl` actually is:
 *
 *   - **Regular file**: dereference via `realpath` and verify it
 *     resolves inside `slugPath`'s realpath. This catches the TOCTOU
 *     case where a symlink ancestor of slugPath was swapped between
 *     the outer lstat and here, and would otherwise let an unlink
 *     escape the intended tree.
 *
 *   - **Symlink**: unlink the symlink itself. `fs.unlinkSync` on a
 *     symlink path removes the LINK, not the target — safe regardless
 *     of where the link points (including dangling). This is the
 *     "nuke really nukes" promise: if a compromised container makes
 *     the JSONL a symlink to dodge wipe, the symlink still goes away.
 *     Without this branch, the prior realpath-containment check would
 *     refuse to unlink a symlink-out-of-tree and leave the entry on
 *     disk — defeating the nuke entirely.
 *
 * Companion helper `removeToolResultsDirInSlug` mirrors this for the
 * sibling per-session tool-results directory at `${slugPath}/${sessionId}/`.
 */
function unlinkJsonlInSlug(
  slugPath: string,
  sessionId: string,
  groupFolder: string,
  sessionName: string,
): number {
  const jsonlPath = path.join(slugPath, `${sessionId}.jsonl`);

  // lstat first to learn what the entry actually is, without
  // following any symlink. This is the hinge for the two branches.
  let entryStat: fs.Stats;
  try {
    entryStat = fs.lstatSync(jsonlPath);
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0; // no such jsonl — fine
    logger.warn(
      { err, groupFolder, sessionName, jsonlPath },
      'unlinkJsonlInSlug: lstat failed on jsonl — skipping',
    );
    return 0;
  }

  if (entryStat.isSymbolicLink()) {
    // Unlink the symlink itself. fs.unlinkSync removes the link
    // entry; it never deletes the target file the link points at.
    try {
      fs.unlinkSync(jsonlPath);
      logger.info(
        { groupFolder, sessionName, sessionId, jsonlPath },
        'unlinkJsonlInSlug: unlinked symlinked jsonl (target preserved)',
      );
      return 1;
    } catch (err) {
      if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
      const code = (err as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') return 0;
      logger.warn(
        { err, groupFolder, sessionName, sessionId, jsonlPath },
        'unlinkJsonlInSlug: unlink-of-symlink failed',
      );
      return 0;
    }
  }

  // Regular-file path: realpath containment check before unlink to
  // catch a slugPath ancestor symlink swap between the outer lstat
  // and here. `path.resolve` alone is string-based and wouldn't
  // notice such an escape.
  let realSlug: string;
  let realJsonl: string;
  try {
    realSlug = fs.realpathSync(slugPath);
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, slugPath },
      'unlinkJsonlInSlug: realpath failed on slug — skipping',
    );
    return 0;
  }
  try {
    realJsonl = fs.realpathSync(jsonlPath);
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, jsonlPath },
      'unlinkJsonlInSlug: realpath failed on jsonl — skipping',
    );
    return 0;
  }
  if (!realJsonl.startsWith(realSlug + path.sep)) {
    logger.warn(
      { groupFolder, sessionName, sessionId, jsonlPath, realSlug, realJsonl },
      'unlinkJsonlInSlug: refusing to unlink — realpath escapes slug directory',
    );
    return 0;
  }
  try {
    fs.unlinkSync(jsonlPath);
    return 1;
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, sessionId, jsonlPath },
      'unlinkJsonlInSlug: unlink failed',
    );
    return 0;
  }
}

/**
 * Try to remove `${slugPath}/${sessionId}/` (the per-session tool-results
 * directory the SDK writes alongside `${sessionId}.jsonl`). Returns 1
 * if a filesystem entry was removed, 0 otherwise.
 *
 * Mirrors `unlinkJsonlInSlug` with the same lstat → branch on type →
 * realpath-containment discipline; only the leaf operation differs.
 *
 *   - **Symlink**: unlink the symlink itself. `fs.unlinkSync` removes
 *     the link entry without following it, so a compromised container
 *     can't redirect the wipe to walk into an arbitrary host directory
 *     and `recursive: true` it. Same "nuke really nukes" promise as
 *     the JSONL path.
 *
 *   - **Directory**: realpath the dir and the slug, verify the dir's
 *     real path is inside the slug's real path, then `fs.rmSync` with
 *     `recursive: true`. Node's `rmSync` does NOT traverse symlinks
 *     it encounters inside the tree — they're removed as link entries,
 *     never followed — so a malicious container that drops a symlink
 *     to `/etc` inside its own tool-results dir cannot trick us into
 *     deleting host files. The realpath check guards the parent path
 *     itself against ancestor-symlink swap (TOCTOU between the outer
 *     `wipeSessionJsonl` lstat and this call).
 *
 *   - **Regular file at the dir path**: not something the SDK writes,
 *     but if a compromised container plants one we leave it alone and
 *     log — wiping it would be outside the helper's contract (it's a
 *     directory remover) and could mask whatever produced the file.
 */
function removeToolResultsDirInSlug(
  slugPath: string,
  sessionId: string,
  groupFolder: string,
  sessionName: string,
): number {
  const dirPath = path.join(slugPath, sessionId);

  let entryStat: fs.Stats;
  try {
    entryStat = fs.lstatSync(dirPath);
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, dirPath },
      'removeToolResultsDirInSlug: lstat failed — skipping',
    );
    return 0;
  }

  if (entryStat.isSymbolicLink()) {
    try {
      fs.unlinkSync(dirPath);
      logger.info(
        { groupFolder, sessionName, sessionId, dirPath },
        'removeToolResultsDirInSlug: unlinked symlinked tool-results dir (target preserved)',
      );
      return 1;
    } catch (err) {
      if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
      const code = (err as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') return 0;
      logger.warn(
        { err, groupFolder, sessionName, sessionId, dirPath },
        'removeToolResultsDirInSlug: unlink-of-symlink failed',
      );
      return 0;
    }
  }

  if (!entryStat.isDirectory()) {
    // The SDK only writes directories at this path. A regular file
    // here means something else put it there — leave it alone rather
    // than deleting state we can't account for.
    logger.warn(
      { groupFolder, sessionName, sessionId, dirPath },
      'removeToolResultsDirInSlug: refusing — entry exists but is neither symlink nor directory',
    );
    return 0;
  }

  // Directory path: realpath containment check before rm. Same TOCTOU
  // defense as the JSONL helper — a slugPath ancestor symlink swap
  // between the outer lstat and here would otherwise let `rmSync`
  // recurse into an unintended tree.
  let realSlug: string;
  let realDir: string;
  try {
    realSlug = fs.realpathSync(slugPath);
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, slugPath },
      'removeToolResultsDirInSlug: realpath failed on slug — skipping',
    );
    return 0;
  }
  try {
    realDir = fs.realpathSync(dirPath);
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, dirPath },
      'removeToolResultsDirInSlug: realpath failed on dir — skipping',
    );
    return 0;
  }
  if (!realDir.startsWith(realSlug + path.sep)) {
    logger.warn(
      { groupFolder, sessionName, sessionId, dirPath, realSlug, realDir },
      'removeToolResultsDirInSlug: refusing to remove — realpath escapes slug directory',
    );
    return 0;
  }
  try {
    // `recursive: true` walks the tree. Node never follows symlinks
    // inside — they're removed as entries — so a compromised container
    // that scattered symlinks to host paths in its own tool-results
    // tree cannot redirect the wipe.
    //
    // No `force: true`: we want ENOENT to surface as an error so the
    // returned count reflects actual removals. Without that distinction,
    // a concurrent cleanup that vanished the path between our lstat
    // and rmSync would still count as `1` here, inflating the caller's
    // "entries removed" total.
    fs.rmSync(dirPath, { recursive: true });
    return 1;
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, sessionId, dirPath },
      'removeToolResultsDirInSlug: rmSync failed',
    );
    return 0;
  }
}

/**
 * Wipe the on-disk session artifacts (JSONL transcript and the
 * sibling per-session tool-results directory) for the given sessionId
 * across every project-slug subdirectory under the slot's `projects/`.
 * Returns the total number of filesystem entries removed.
 *
 * The function name retains the historical "Jsonl" suffix from when it
 * only unlinked transcripts; the contract is now a full session-artifact
 * wipe. Both artifact types share one realpath-containment regime, one
 * DoS-cap regime, and one slug-walk traversal — keeping them in a single
 * function avoids walking `projects/` twice for what is conceptually one
 * "wipe everything tied to this sessionId" operation.
 *
 * Production callers:
 *   1. `nukeSession` (#100) — owns the multi-step order-of-operations
 *      wipe (capture sessionIds → kill containers → drop DB rows →
 *      remove session artifacts).
 *   2. `startSchedulerLoop` (#193) — injects this as a dependency so
 *      `runTask`'s post-run finally can wipe the per-run artifacts the
 *      moment a scheduled run completes (its sessionId is never
 *      persisted to the DB, so the time-based `cleanup-sessions.sh`
 *      can't find them later).
 *
 * Tests also import this symbol directly to bypass the full
 * `nukeSession` path.
 *
 * @internal — the orchestrator builds with `tsconfig.stripInternal: true`,
 * so this tag keeps the symbol out of the emitted `.d.ts`. The two
 * production callers above are in-tree and don't need d.ts visibility;
 * tests reach the symbol through the source `.ts` import, not the d.ts.
 */
export function wipeSessionJsonl(
  groupFolder: string,
  sessionName: string,
  sessionId: string,
): number {
  if (!SESSION_ID_PATTERN.test(sessionId)) {
    logger.warn(
      { groupFolder, sessionName, sessionId },
      'wipeSessionJsonl: refusing to wipe — sessionId fails strict-charset check',
    );
    return 0;
  }

  const projectsDir = path.join(
    DATA_DIR,
    'sessions',
    groupFolder,
    sessionName,
    '.claude',
    'projects',
  );

  // Validate `projects/` BEFORE any unlink work — the fast path and
  // the slow walk both depend on it being a real directory inside
  // DATA_DIR, not a symlink swap pointing elsewhere. The per-session
  // `.claude` mount is writable from the container, so a compromised
  // container could replace `.claude/projects` with a symlink. We
  // refuse to traverse a symlinked `projects/` regardless of where
  // it points.
  let projectsLstat: fs.Stats;
  try {
    projectsLstat = fs.lstatSync(projectsDir);
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { err, groupFolder, sessionName, sessionId, projectsDir },
      'wipeSessionJsonl: lstat failed on projects directory',
    );
    return 0;
  }
  if (projectsLstat.isSymbolicLink()) {
    logger.error(
      { groupFolder, sessionName, sessionId, projectsDir },
      'wipeSessionJsonl: refusing to traverse — projects/ itself is a symlink (possible escape attempt)',
    );
    return 0;
  }
  if (!projectsLstat.isDirectory()) {
    return 0;
  }

  // Try the well-known project slug FIRST as a fast path. Two wins:
  //   1. Defeats flood attacks where a compromised container fills
  //      `projects/` with millions of decoy slug-named subdirs to
  //      push the legitimate one past any walk cap. We hit the
  //      legitimate path directly and, on success, return without
  //      walking the tree at all.
  //   2. ~constant-time in the steady state — no opendir+iterate
  //      overhead per nuke.
  // The slow path (full walk) below is the safety net for cases
  // where the slug naming convention drifts (CLAUDE_PROJECT_SLUG
  // changes, operator-renamed workspace, etc.).
  let deleted = 0;
  const fastPathSlug = path.join(projectsDir, '-workspace-group');
  // Same lstat-then-skip discipline as the slow-path loop below
  // (line ~605): without this, a compromised container could replace
  // `projects/-workspace-group` with a symlink to an arbitrary host
  // directory. `unlinkJsonlInSlug`'s realpath-containment check
  // resolves BOTH the slug and the jsonl through the same symlink,
  // so the containment passes and the unlink lands inside the
  // symlink target.
  let fastPathLstat: fs.Stats | undefined;
  try {
    fastPathLstat = fs.lstatSync(fastPathSlug);
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code !== 'ENOENT') {
      logger.warn(
        { err, groupFolder, sessionName, sessionId, fastPathSlug },
        'wipeSessionJsonl: lstat failed on fast-path slug',
      );
    }
  }
  if (fastPathLstat?.isSymbolicLink()) {
    logger.error(
      { groupFolder, sessionName, sessionId, fastPathSlug },
      'wipeSessionJsonl: refusing fast path — slug is a symlink (possible escape attempt)',
    );
  } else if (fastPathLstat?.isDirectory()) {
    deleted += unlinkJsonlInSlug(
      fastPathSlug,
      sessionId,
      groupFolder,
      sessionName,
    );
    deleted += removeToolResultsDirInSlug(
      fastPathSlug,
      sessionId,
      groupFolder,
      sessionName,
    );
  }

  // Walk project-slug subdirectories with `opendirSync` — an
  // iterator-style API that does NOT materialize the full directory
  // listing up front, unlike `readdirSync`.
  //
  // Caps:
  //   - MAX_DIRS_VISITED bounds the slow-path search across many
  //     project slugs. Stray files don't count.
  //   - MAX_TOTAL_ENTRIES bounds total readSync iterations so a
  //     `projects/` filled with millions of stub FILES can't block
  //     the orchestrator event loop synchronously.
  const MAX_DIRS_VISITED = 10000;
  const MAX_TOTAL_ENTRIES = 100000;

  // TOCTOU defense for the parent dir: realpath after opendir.
  // `fs.Dir` doesn't expose its FD, so we can't fstat the open handle
  // — instead we resolve the path through the symlink chain at this
  // moment. If a compromised container swapped `projects/` to a
  // symlink between our lstat above and the opendirSync below, the
  // realpath result will land outside the expected `<DATA_DIR>/...`
  // tree and we abort. Residual race: a container would have to win
  // a sub-millisecond inode swap AND aim it inside DATA_DIR — at
  // which point it has already broken out of its sandbox and the
  // orchestrator has bigger problems. Per-slug realpath checks below
  // catch escape attempts at the leaf level regardless.
  let dir: fs.Dir;
  try {
    dir = fs.opendirSync(projectsDir);
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return deleted;
    logger.warn(
      { err, groupFolder, sessionName, sessionId, projectsDir },
      'wipeSessionJsonl: failed to open projects directory',
    );
    return deleted;
  }
  try {
    const realProjects = fs.realpathSync(projectsDir);
    // Also realpath DATA_DIR to handle macOS where /var → /private/var
    // (or similar OS-level symlinks). Without this both sides could
    // dereference to different absolute prefixes and the prefix check
    // would false-positive even on a perfectly legitimate path.
    const realDataDir = fs.realpathSync(DATA_DIR);
    const expectedPrefix = realDataDir + path.sep;
    if (!realProjects.startsWith(expectedPrefix)) {
      logger.error(
        {
          groupFolder,
          sessionName,
          sessionId,
          projectsDir,
          realProjects,
          expectedPrefix,
        },
        'wipeSessionJsonl: projects/ realpath outside DATA_DIR — aborting (TOCTOU?)',
      );
      dir.closeSync();
      return deleted;
    }
  } catch (err) {
    if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
    logger.warn(
      { err, groupFolder, sessionName, sessionId, projectsDir },
      'wipeSessionJsonl: realpath on projects/ failed — aborting',
    );
    dir.closeSync();
    return deleted;
  }

  let dirsVisited = 0;
  let totalEntries = 0;
  let bailedOnLimit: 'total-entries' | 'dirs-visited' | null = null;
  try {
    let entry: fs.Dirent | null;
    while ((entry = dir.readSync()) !== null) {
      totalEntries++;
      if (totalEntries > MAX_TOTAL_ENTRIES) {
        bailedOnLimit = 'total-entries';
        break;
      }
      // Skip the slug we already tried in the fast path — would
      // double-count `deleted` if the file was already gone.
      if (entry.name === '-workspace-group') continue;

      const slugPath = path.join(projectsDir, entry.name);
      let linkStat: fs.Stats;
      try {
        linkStat = fs.lstatSync(slugPath);
      } catch (err) {
        if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
        const code = (err as NodeJS.ErrnoException).code;
        if (code === 'ENOENT') continue;
        logger.warn(
          { err, groupFolder, sessionName, slugPath },
          'wipeSessionJsonl: lstat failed on slug entry — skipping',
        );
        continue;
      }
      if (linkStat.isSymbolicLink()) {
        logger.warn(
          { groupFolder, sessionName, slugPath },
          'wipeSessionJsonl: refusing to traverse symlink under projects/',
        );
        continue;
      }
      if (!linkStat.isDirectory()) continue;

      dirsVisited++;
      if (dirsVisited > MAX_DIRS_VISITED) {
        bailedOnLimit = 'dirs-visited';
        break;
      }

      deleted += unlinkJsonlInSlug(
        slugPath,
        sessionId,
        groupFolder,
        sessionName,
      );
      deleted += removeToolResultsDirInSlug(
        slugPath,
        sessionId,
        groupFolder,
        sessionName,
      );
    }
  } finally {
    dir.closeSync();
  }

  if (bailedOnLimit === 'total-entries') {
    logger.error(
      {
        groupFolder,
        sessionName,
        sessionId,
        totalEntries,
        limit: MAX_TOTAL_ENTRIES,
        deleted,
      },
      'wipeSessionJsonl: stopped early — total readSync count exceeded MAX_TOTAL_ENTRIES (possible DoS via stub-file flood)',
    );
  } else if (bailedOnLimit === 'dirs-visited') {
    logger.error(
      {
        groupFolder,
        sessionName,
        sessionId,
        dirsVisited,
        limit: MAX_DIRS_VISITED,
        deleted,
      },
      'wipeSessionJsonl: stopped early — directory-traversal count exceeded MAX_DIRS_VISITED (possible DoS via slug-dir flood)',
    );
  }
  return deleted;
}
