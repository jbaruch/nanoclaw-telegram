import fs from 'fs';
import path from 'path';

import { HOST_GID, HOST_UID } from './config.js';
import { getAllChats } from './db-messages.js';
import {
  getTriggerPatterns,
  setRegisteredGroup,
} from './db-registered-groups.js';
import { getTaskById, createTask, deleteTask } from './db-tasks.js';
import { BEST_EFFORT_FS_CODES, isFsErrorWithCode } from './fs-errors.js';
import { isValidGroupFolder, resolveGroupFolderPath } from './group-folder.js';
import type { RegisteredGroup } from './types.js';
import { logger } from './logger.js';
import { registeredGroups } from './orchestrator-state.js';

// Group registry + startup registry-maintenance helpers, extracted from
// src/index.ts (#749 seam 4a). registerGroup is the hub; the others are
// startup passes. Shared state (registeredGroups) comes from
// ./orchestrator-state.js.

/**
 * Cleanup pass for the retired non-main heartbeat task (#453).
 *
 * The non-main heartbeat existed solely to drive `tessl__check-unanswered`
 * — a per-cycle SQL+LLM scan for unreplied messages. The skill was
 * retired in `jbaruch/nanoclaw-core#38` because the steady-state token
 * cost wasn't justified by the rate of genuinely-dropped messages it
 * caught. With no work for the task to do, this orchestrator no longer
 * creates `heartbeat-<folder>` rows for non-main groups, and any rows
 * left behind by older orchestrator versions are deleted on startup so
 * the scheduler stops firing them.
 *
 * `containerConfig.enableHeartbeat` stays in the schema for backwards
 * compatibility (operators upgrading from older orchestrator versions
 * may have it set), but on non-main groups it is now a no-op. The
 * main-group heartbeat (a separate `heartbeat-<folder>` row created
 * inline in `registerGroup` for `group.isMain`) is unaffected — different
 * code path, different prompt, different lifetime.
 *
 * Future feature work that wants non-main heartbeats lands as its own
 * issue with its own task definition.
 */
export function cleanupOrphanNonMainHeartbeats(): void {
  for (const [jid, group] of Object.entries(registeredGroups)) {
    if (group.isMain) continue;
    const heartbeatId = `heartbeat-${group.folder}`;
    if (!getTaskById(heartbeatId)) continue;
    deleteTask(heartbeatId);
    logger.info(
      { jid, folder: group.folder, taskId: heartbeatId },
      'Deleted orphan non-main heartbeat task (retired in nanoclaw-core#38 / #453)',
    );
  }
}

/**
 * Drift detector (#159): log every `registered_groups` row whose JID has
 * no matching `chats` row. The spawner reads `available_groups.json`
 * (which is rebuilt from `getAllChats()`) so a row missing from `chats`
 * is silently ignored at runtime — exactly the dormant-row failure mode
 * #159's one-shot cleanup addressed for `tg:1698969` / `telegram_main`.
 *
 * Read-only by design: future operator-introduced drift gets surfaced
 * for review (operator can resolve it via `unregister_group`) instead
 * of being auto-deleted at startup. Auto-delete would make recovery
 * from a transient `chats` outage (e.g. a partial DB restore that
 * truncated `chats` but kept `registered_groups`) catastrophic — every
 * registered group would vanish on the next restart.
 */
export function logRegisteredGroupOrphans(): void {
  const knownJids = new Set(getAllChats().map((c) => c.jid));
  const orphans: Array<{ jid: string; folder: string }> = [];
  for (const [jid, group] of Object.entries(registeredGroups)) {
    if (knownJids.has(jid)) continue;
    orphans.push({ jid, folder: group.folder });
  }
  if (orphans.length > 0) {
    logger.warn(
      { orphans },
      'registered_groups rows have no matching chats row — invisible to the spawner. Run unregister_group to clean up if intended.',
    );
  }
}

/**
 * Return a copy of `group` whose `triggerPatterns` reflects what is
 * actually persisted in the DB for `jid`.
 *
 * `setRegisteredGroup` synthesizes the `trigger_pattern` column from
 * `group.trigger` when the caller supplied no `triggerPatterns` config —
 * and the IPC `register_group` payload only ever carries the `trigger`
 * string, never a config. The gate path reads `triggerPatterns` straight
 * from the in-memory registry and never re-reads the DB, so caching the
 * raw payload object (with `triggerPatterns: undefined`) makes the
 * trigger gate see "no patterns configured" and fail-open on every
 * message until the next `loadState()` reload. Re-reading the
 * synthesized config keeps the cache consistent with the row (#670).
 *
 * Exported for unit testing; callers should use `registerGroup`.
 */
export function hydrateRegisteredGroupTriggerPatterns(
  group: RegisteredGroup,
  jid: string,
): RegisteredGroup {
  return { ...group, triggerPatterns: getTriggerPatterns(jid) ?? undefined };
}

export function registerGroup(jid: string, group: RegisteredGroup): void {
  // Pre-validate rather than catch resolveGroupFolderPath's throw: it raises
  // a plain Error on an invalid/escaping folder, which a catch-all would not
  // distinguish from a defect. isValidGroupFolder is the same predicate the
  // resolver asserts on, so this rejects exactly the invalid-folder case.
  if (!isValidGroupFolder(group.folder)) {
    logger.warn(
      { jid, folder: group.folder },
      'Rejecting group registration with invalid folder',
    );
    return;
  }
  const groupDir = resolveGroupFolderPath(group.folder);

  setRegisteredGroup(jid, group);
  // Cache the group with its persisted trigger patterns, not the raw
  // payload (which carries `triggerPatterns: undefined`) — otherwise the
  // trigger gate fail-opens until the next reload (#670).
  registeredGroups[jid] = hydrateRegisteredGroupTriggerPatterns(group, jid);

  // Create group folder
  fs.mkdirSync(path.join(groupDir, 'logs'), { recursive: true });

  // CLAUDE.md is no longer copied per-group — it's a thin trust-tier
  // pointer mounted readonly by container-runner.ts at spawn time, so
  // the trust flag at the moment of spawn picks the right template
  // every time (fixes #153 by construction). The agent's mutable
  // per-group memory lives in MEMORY.md; create an empty placeholder
  // here so the @import in CLAUDE.md resolves on the very first
  // message instead of the agent seeing a missing file.
  const memoryMdFile = path.join(groupDir, 'MEMORY.md');
  if (!fs.existsSync(memoryMdFile)) {
    fs.writeFileSync(
      memoryMdFile,
      `# Memory — ${group.name || group.folder}\n\n` +
        '_Persistent notes the agent has accumulated about this group. ' +
        'Append facts the agent should recall in future sessions._\n',
    );
    logger.info({ folder: group.folder }, 'Created empty MEMORY.md for group');
  }

  // Chown group folder to the container user so the agent can write to it.
  // In DooD the orchestrator runs as root — files it creates are root-owned.
  const effectiveUid = HOST_UID ?? process.getuid?.();
  const effectiveGid = HOST_GID ?? process.getgid?.();
  if (effectiveUid != null && effectiveUid !== 0) {
    try {
      chownRecursive(groupDir, effectiveUid, effectiveGid ?? effectiveUid);
    } catch (err) {
      if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
      logger.warn(
        { folder: group.folder, err },
        'Failed to chown group folder',
      );
    }
  }

  // Non-main heartbeat is retired (#453, see `cleanupOrphanNonMainHeartbeats`
  // above). The `containerConfig.enableHeartbeat` flag stays in the schema
  // for backwards compatibility with operators upgrading from older
  // orchestrator versions, but registerGroup no longer creates a
  // `heartbeat-<folder>` row for non-main groups. The startup cleanup
  // pass removes any rows left behind by older orchestrator versions.

  // Auto-create the parallel-maintenance heartbeat for every main group.
  // Mirrors the non-main auto-registration above, but runs in the
  // `maintenance` session slot so it doesn't block user-facing AyeAye.
  // The task-scheduler fires this every 15 minutes via
  // `MAINTENANCE_SESSION_NAME`; the prompt keeps the defensive preamble
  // as belt-and-suspenders against improvisation.
  if (group.isMain) {
    const heartbeatId = `heartbeat-${group.folder}`;
    if (!getTaskById(heartbeatId)) {
      createTask({
        id: heartbeatId,
        group_folder: group.folder,
        chat_jid: jid,
        prompt:
          'MANDATORY FIRST ACTION: Call Skill(skill: "tessl__heartbeat") BEFORE doing anything else. Do NOT improvise checks. Do NOT query databases. Do NOT invent thresholds. Load and execute the skill exactly as written.\n\n' +
          'This is a scheduled heartbeat — no ACK reaction, no reply_to.\n' +
          'Workspace: /workspace/group/\n' +
          'Telegram HTML ONLY: <b>, <i>, <code>, <a href="url">text</a>, • for bullets. NEVER Markdown.\n' +
          'CRITICAL: NEVER set the "sender" parameter on send_message. Always call send_message with only "text" and optionally "pin". The sender parameter routes through pool bots and bypasses the database — messages become ghosts.\n' +
          'If nothing actionable → produce NO output at all. Silence = success.',
        schedule_type: 'interval',
        schedule_value: '900000', // 15 minutes in ms
        // Heartbeats are stateless by design — every input is read from
        // external sources (messages.db, workspace, skills) on each tick,
        // so persisting the SDK session chain across runs is pure
        // liability. After 6 days of 15-min ticks the swarm group's
        // maintenance JSONL hit 187 MB and crossed the AUP-classifier
        // threshold, refusing every subsequent run (#114). A single
        // contaminated tick (poisoned tool_result, oversized image, hung
        // tool output) also got persisted forever and re-read on every
        // later tick. `'isolated'` makes each tick a fresh session —
        // manual recovery becomes unnecessary because there's no
        // accumulated state to wipe.
        context_mode: 'isolated',
        next_run: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
        status: 'active',
        created_at: new Date().toISOString(),
        created_by_role: 'owner',
      });
      logger.info(
        { jid, folder: group.folder },
        'Auto-created maintenance heartbeat for main group',
      );
    }
  }

  logger.info(
    { jid, name: group.name, folder: group.folder },
    'Group registered',
  );
}

function chownRecursive(dir: string, uid: number, gid: number): void {
  fs.chownSync(dir, uid, gid);
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    fs.chownSync(fullPath, uid, gid);
    if (entry.isDirectory()) {
      chownRecursive(fullPath, uid, gid);
    }
  }
}
