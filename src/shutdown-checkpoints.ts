/**
 * Pre-shutdown checkpoint writer (#497).
 *
 * The kill-auto-compaction checkpoint mechanism (#104) writes
 * `<groupDir>/.checkpoints/default.md` only on token-threshold cross.
 * Deploy SIGTERM → graceful-shutdown handoff → deploy.sh force-kill
 * (#249) does NOT trigger a checkpoint, so when a container is
 * mid-turn at deploy time its in-flight reasoning, pending replies,
 * and "do NOT re-execute" state are gone — the next spawn comes up
 * with no `<auto-context section="CHECKPOINT">` injection and re-does
 * already-fired side-effects.
 *
 * This module reuses `writeCheckpoint` so the existing SessionStart
 * hook + `session-reentry` skill pipeline keeps working unchanged on
 * the next spawn — no new reentry surface.
 *
 * Best-effort: per-session writes are independently try/caught so a
 * failure for one group doesn't block the others (mirrors the
 * threshold-cross handler).
 */
import path from 'path';

import { writeCheckpoint } from './checkpoint.js';
import { resolveGroupFolderPath } from './group-folder.js';
import type { Thresholds } from './threshold.js';

/** Subset of `RegisteredGroup` we read for the `groupName` log field. */
export interface RegisteredGroupNameLookup {
  name: string;
}

/** Subset of `getActiveContainersForHandoff()`'s return shape. */
export interface ActiveContainer {
  groupJid: string;
  sessionName: string;
  groupFolder: string | null;
}

export interface ShutdownCheckpointDeps {
  /** Active containers from `queue.getActiveContainersForHandoff()`. */
  active: ActiveContainer[];
  /** The orchestrator's `sessions` map: `groupFolder → sessionName → sessionId`. */
  sessions: Record<string, Record<string, string>>;
  /** Registered groups, keyed by groupJid (for the `groupName` log field). */
  registeredGroups: Record<string, RegisteredGroupNameLookup>;
  /** Thresholds at shutdown time — recorded into the checkpoint Facts section. */
  thresholds: Thresholds;
  /** Default session name (DEFAULT_SESSION_NAME constant). */
  defaultSessionName: string;
  /** DATA_DIR for resolving the JSONL transcript path. */
  dataDir: string;
  /** Logger; injected so the shutdown handler can pass its existing logger. */
  logger: {
    info: (obj: Record<string, unknown>, msg: string) => void;
    error: (obj: Record<string, unknown>, msg: string) => void;
  };
}

/**
 * Write a checkpoint for every active default session.
 *
 * Filtered to the user-facing default slot only — maintenance is bounded
 * and idempotent, so re-checkpointing it would just churn. Sessions
 * with a missing folder, missing sessionId, or a write failure are
 * skipped after logging; the function never throws.
 *
 * Returns the number of checkpoints actually written, for the caller's
 * post-shutdown log line.
 */
export async function writeShutdownCheckpoints(
  deps: ShutdownCheckpointDeps,
): Promise<number> {
  let written = 0;
  for (const c of deps.active) {
    if (c.sessionName !== deps.defaultSessionName) continue;
    if (!c.groupFolder) continue;
    const sessionId = deps.sessions[c.groupFolder]?.[deps.defaultSessionName];
    if (!sessionId) continue;
    try {
      const groupDir = resolveGroupFolderPath(c.groupFolder);
      const jsonlPath = path.join(
        deps.dataDir,
        'sessions',
        c.groupFolder,
        deps.defaultSessionName,
        '.claude',
        'projects',
        '-workspace-group',
        `${sessionId}.jsonl`,
      );
      // Truthy fallback (not nullish): a registered group with an empty
      // string `name` would otherwise render as a blank "Group:" line in
      // the checkpoint Facts section.
      const registeredName = deps.registeredGroups[c.groupJid]?.name;
      await writeCheckpoint({
        groupDir,
        jsonlPath,
        sessionId,
        thresholds: deps.thresholds,
        usedTokens: 0,
        groupName: registeredName || c.groupFolder,
        trigger: 'shutdown',
      });
      written += 1;
      deps.logger.info(
        { group: c.groupFolder, sessionId },
        'Wrote pre-shutdown checkpoint',
      );
    } catch (err) {
      // Resilient per-group loop: any Error from resolving the group folder or
      // writing its checkpoint is logged and the loop moves to the next group;
      // a non-Error throw is a defect and propagates.
      if (!(err instanceof Error)) throw err;
      deps.logger.error(
        { group: c.groupFolder, err },
        'Pre-shutdown checkpoint write failed',
      );
    }
  }
  return written;
}
