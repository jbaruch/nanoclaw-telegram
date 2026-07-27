import fs from 'fs';
import path from 'path';

import {
  DATA_DIR,
  GROUPS_DIR,
  ORCHESTRATOR_REPO_URL,
  STORE_DIR,
} from '../config.js';

import { syncBackupRepo, type SyncResult } from '../backup-sync.js';
import {
  PERSISTABLE_GLOBAL_FILES,
  backupCommitAndPush,
  runPersonaPersistTask,
  runSerializedPersonaPersist,
  validateGlobalFilesToPersist,
} from '../git-persist.js';
import { registerIpcHandler, scriptResultPath } from '../ipc-registry.js';
import { logger } from '../logger.js';

/**
 * Git-backed persistence (#879, split out of `ops.ts`): the two commands
 * that commit and push from the host — `github_backup` (the whole-store
 * backup repo) and `persist_global_file` (the persona direct-push
 * carve-out, `jbaruch/nanoclaw-host: persona-persist-direct-push`).
 */
export function registerOpsGitIpcHandlers(): void {
  registerIpcHandler('github_backup', {
    // Admin-tile only. The dispatcher enforces this BEFORE the handler
    // runs and writes the refusal envelope itself, so a polling caller
    // gets an actionable result instead of reading absence as a hang
    // (`coding-policy: error-handling` outer-boundary contract).
    requiresMain: true,
    handler: async ({ data, sourceGroup }) => {
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
    // Admin-tile only. The dispatcher enforces this BEFORE the handler
    // runs and writes the refusal envelope itself, so a polling caller
    // gets an actionable result instead of reading absence as a hang
    // (`coding-policy: error-handling` outer-boundary contract).
    requiresMain: true,
    handler: async ({ data, sourceGroup }) => {
      if (data.requestId) {
        // Authorization: persist_global_file commits + pushes the
        // orchestrator repo's `main` using GITHUB_TOKEN — same privilege
        // class as `github_backup` / `promote_staging`, gated on `isMain`.
        // A non-main container can't reach the persona source even by
        // writing an IPC task file directly.
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
}
