/**
 * Sync live group state into `groups/<group>/backup-repo/` for #397 —
 * the github_backup pipeline's source-data sync.
 *
 * Before the state-001…state-010 epic, the JSON state files lived in
 * each group's working tree and `git add -A` in `backup-repo/`
 * naturally picked up daily changes. After the epic, those JSON
 * sources stopped existing on disk (state lives in `store/messages.db`
 * now), and `backup-repo/` froze at its pre-migration snapshot — 18
 * days of nightly housekeeping fired with `git diff --cached --quiet`
 * always returning 0 ("nothing staged"), so no commits.
 *
 * This module restores per-day diff coverage by:
 *
 *   1. Copying single-file group artefacts (MEMORY.md,
 *      daily_discoveries.md) into `backup-repo/` if they exist on
 *      disk — overwriting the stale copy. Missing sources are
 *      skipped silently; this is normal for groups that haven't
 *      written one yet.
 *   2. Mirroring `groups/<group>/memory/` into `backup-repo/memory/`
 *      with delete-on-missing semantics: files in the dest that no
 *      longer exist in the source are removed, so the working tree
 *      reflects the live state. Git history retains the older state.
 *      If the source `memory/` directory itself doesn't exist, the
 *      dest is left untouched (don't blow away historical content
 *      just because the source disappeared).
 *   3. Calling `runDumpPlan` from `scripts/dump-state-tables.ts` to
 *      produce per-table SQL dumps in `backup-repo/state/` (#398) —
 *      restoring coverage of the SQLite-resident state surface.
 *
 * Returns a structured summary so the IPC handler can write a
 * machine-readable result envelope and the calling skill (nightly
 * housekeeping) can detect anomalies (consecutive empty backups,
 * unexpected skip counts, etc.) rather than parsing prose.
 */
import fs from 'fs';
import path from 'path';

import { runDumpPlan } from './db-dump.js';

export interface SyncResult {
  /** Relative paths (POSIX) under backup-repo/ that were written or overwritten. */
  copied: string[];
  /** Relative paths (POSIX) under backup-repo/ removed because the source disappeared. */
  removed: string[];
  /** Tables whose `.dump` SQL was written to backup-repo/state/<table>.sql. */
  dumped: string[];
  /** Tables in STATE_TABLES that didn't exist in the DB and were skipped. */
  skipped: string[];
}

const SINGLE_FILE_NAMES = ['MEMORY.md', 'daily_discoveries.md'] as const;

function copySingleFile(srcPath: string, destPath: string): boolean {
  if (!fs.existsSync(srcPath)) return false;
  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  fs.copyFileSync(srcPath, destPath);
  return true;
}

function listFilesRecursive(root: string, base = ''): string[] {
  const out: string[] = [];
  const entries = fs.readdirSync(root, { withFileTypes: true });
  for (const e of entries) {
    const rel = path.join(base, e.name);
    if (e.isDirectory()) {
      out.push(...listFilesRecursive(path.join(root, e.name), rel));
    } else if (e.isFile()) {
      out.push(rel);
    }
    // Symlinks and other non-file types are ignored — `memory/` is
    // expected to hold plain markdown files written by the agent;
    // anything else would be a bug to surface, not silently sync.
  }
  return out;
}

function toPosixPrefix(prefix: string, rel: string): string {
  return [prefix, ...rel.split(path.sep)].join('/');
}

function mirrorMemoryDir(
  srcDir: string,
  destDir: string,
  result: SyncResult,
): void {
  if (!fs.existsSync(srcDir)) return;
  fs.mkdirSync(destDir, { recursive: true });
  const srcFiles = new Set(listFilesRecursive(srcDir));
  const destFiles = fs.existsSync(destDir)
    ? new Set(listFilesRecursive(destDir))
    : new Set<string>();

  for (const rel of srcFiles) {
    const srcPath = path.join(srcDir, rel);
    const destPath = path.join(destDir, rel);
    fs.mkdirSync(path.dirname(destPath), { recursive: true });
    fs.copyFileSync(srcPath, destPath);
    result.copied.push(toPosixPrefix('memory', rel));
  }
  for (const rel of destFiles) {
    if (!srcFiles.has(rel)) {
      fs.unlinkSync(path.join(destDir, rel));
      result.removed.push(toPosixPrefix('memory', rel));
    }
  }
}

export function syncBackupRepo(args: {
  groupDir: string;
  backupDir: string;
  dbPath: string;
}): SyncResult {
  if (!fs.existsSync(args.groupDir)) {
    throw new Error(
      `backup-sync: group directory not found at ${args.groupDir} — verify the group is registered and its folder exists`,
    );
  }
  if (!fs.existsSync(args.backupDir)) {
    throw new Error(
      `backup-sync: backup-repo not found at ${args.backupDir} — clone the backup branch into this path before running github_backup`,
    );
  }

  const result: SyncResult = {
    copied: [],
    removed: [],
    dumped: [],
    skipped: [],
  };

  for (const name of SINGLE_FILE_NAMES) {
    const wrote = copySingleFile(
      path.join(args.groupDir, name),
      path.join(args.backupDir, name),
    );
    if (wrote) result.copied.push(name);
  }

  mirrorMemoryDir(
    path.join(args.groupDir, 'memory'),
    path.join(args.backupDir, 'memory'),
    result,
  );

  const dumpReport = runDumpPlan({
    dbPath: args.dbPath,
    outDir: path.join(args.backupDir, 'state'),
  });
  result.dumped = dumpReport.dumped;
  result.skipped = dumpReport.skipped;

  return result;
}
