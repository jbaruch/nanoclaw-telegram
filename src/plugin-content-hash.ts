/**
 * #710 — Plugin-registry content hash for per-task SDK session
 * invalidation.
 *
 * Recurring scheduled tasks pin an SDK `session_id` across fires
 * (#336) so the prompt-cache prefix survives between cadences. Skill
 * and rule content is injected into the session only at creation, so
 * a pinned session silently outlives any plugin update: the per-spawn
 * `cpSync` snapshot is fresh on disk, but the resumed session never
 * re-reads it and the agent keeps following week-old instructions
 * (plus its own in-context precedent).
 *
 * The scheduler compares the hash stored at session creation
 * (`scheduled_tasks.session_plugins_hash`) against the current
 * registry hash at fire time and rotates to a fresh session on
 * mismatch. The whole registry tree is hashed — not just the skill
 * the task's prompt names — because a session's stale surface spans
 * rules, cross-skill `Skill()` deps, and scripts; under-invalidation
 * is the failure mode #710 describes, and the cost of the broad hash
 * is one lost cache prefix per task per plugin update.
 */
import { createHash } from 'crypto';
import fs from 'fs';
import path from 'path';

import { getRegistryTilesDir } from './container-runner.js';

/**
 * Deterministic sha256 over a directory tree: every regular file's
 * registry-relative path and content, visited in byte-order-sorted
 * name order so the digest is stable across platforms and locales.
 * Entries that are neither files nor directories (sockets, FIFOs,
 * symlinks — the tessl registry install writes none of these) are
 * excluded from the digest.
 *
 * Returns `null` when `dir` does not exist (cold start before the
 * first `tessl install`, dev checkouts without a workspace) or when
 * the tree vanishes mid-walk (`atomicPublishDir`'s temp+swap-rename
 * can race a `tessl update`) — callers treat `null` as "content
 * state unknowable this fire".
 */
export function hashDirectoryTree(dir: string): string | null {
  if (!fs.existsSync(dir)) {
    return null;
  }
  const hash = createHash('sha256');
  try {
    walkInto(hash, dir, '');
  } catch (err) {
    // Narrow to the mid-walk race: a concurrent registry swap deletes
    // the subtree between readdir and read. Anything else (EACCES,
    // EIO) is a real fault and propagates per
    // `jbaruch/coding-policy: error-handling`.
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      return null;
    }
    throw err;
  }
  return hash.digest('hex');
}

function walkInto(
  hash: ReturnType<typeof createHash>,
  root: string,
  rel: string,
): void {
  const entries = fs
    .readdirSync(path.join(root, rel), { withFileTypes: true })
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  for (const entry of entries) {
    const relPath = rel ? `${rel}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      walkInto(hash, root, relPath);
    } else if (entry.isFile()) {
      // NUL separators keep (path, content) pairs unambiguous —
      // without them `a` + `bc` and `ab` + `c` digest identically.
      hash.update(relPath);
      hash.update('\0');
      hash.update(fs.readFileSync(path.join(root, relPath)));
      hash.update('\0');
    }
  }
}

/**
 * Current content hash of the local Tessl plugin registry — the same
 * tree `runContainerAgent` snapshots into each spawned container via
 * `getRegistryTilesDir()`. `null` when the registry is absent.
 */
export function getPluginRegistryHash(): string | null {
  return hashDirectoryTree(getRegistryTilesDir());
}
