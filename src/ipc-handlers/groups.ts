import { deleteTask, getTasksForGroup } from '../db-tasks.js';
import { isValidGroupFolder } from '../group-folder.js';
import { registerIpcHandler } from '../ipc-registry.js';
import { logger } from '../logger.js';

/**
 * Group-registry commands (#845 slice 2): refresh / register /
 * unregister / set_trusted / set_trigger. All five are main-only, so
 * they lean on the dispatcher's `requiresMain` gate instead of a
 * per-handler isMain check — a blocked non-main caller is warned and
 * dropped before the handler runs (these are fire-and-forget commands
 * with no result envelope, matching the pre-registry behavior).
 */
export function registerGroupIpcHandlers(): void {
  registerIpcHandler('refresh_groups', {
    requiresMain: true,
    handler: async ({ sourceGroup, deps }) => {
      const registeredGroups = deps.registeredGroups();
      logger.info({ sourceGroup }, 'Group metadata refresh requested via IPC');
      await deps.syncGroups(true);
      // Write updated snapshot immediately
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        true,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
    },
  });

  registerIpcHandler('register_group', {
    requiresMain: true,
    handler: ({ data, sourceGroup, deps }) => {
      const registeredGroups = deps.registeredGroups();
      if (
        typeof data.jid === 'string' &&
        typeof data.name === 'string' &&
        typeof data.folder === 'string' &&
        typeof data.trigger === 'string' &&
        data.jid.length > 0 &&
        data.name.length > 0 &&
        data.folder.length > 0 &&
        data.trigger.length > 0
      ) {
        // `typeof === 'string'` guards BEFORE calling `.trim()` on
        // any field. IPC payloads are untrusted JSON: a malformed
        // request like `{jid: {}}` or `{name: 42}` would otherwise
        // throw a TypeError and route the task file to ipc/errors,
        // creating a low-effort log-spam / DoS vector. set_trusted /
        // set_trigger already follow this pattern; this reuses it.
        if (!isValidGroupFolder(data.folder)) {
          logger.warn(
            { sourceGroup, folder: data.folder },
            'Invalid register_group request - unsafe folder name',
          );
          return;
        }
        // Trim string fields so this IPC path can't leave a group
        // registered under a whitespace-padded key. set_trusted /
        // set_trigger trim before lookup; an untrimmed register would
        // otherwise produce a "ghost" registration the partial-update
        // tools can never match. Same normalization, same site of
        // truth.
        const trimmedJid = data.jid.trim();
        const trimmedName = data.name.trim();
        const trimmedTrigger = data.trigger.trim();
        if (
          trimmedJid.length === 0 ||
          trimmedName.length === 0 ||
          trimmedTrigger.length === 0
        ) {
          logger.warn(
            { data },
            'Invalid register_group request - empty/whitespace fields',
          );
          return;
        }
        // Defense in depth: agent cannot set isMain via IPC.
        // Preserve isMain from the existing registration so IPC config
        // updates (e.g. adding additionalMounts) don't strip the flag.
        const existingGroup = registeredGroups[trimmedJid];
        deps.registerGroup(trimmedJid, {
          name: trimmedName,
          folder: data.folder,
          trigger: trimmedTrigger,
          added_at: new Date().toISOString(),
          containerConfig: data.containerConfig,
          // Explicitly default to `false` when caller omits — matches
          // the MCP tool's documented default ("respond to all
          // messages"). setRegisteredGroup now preserves undefined as
          // SQL NULL, which is a distinct state from `false`, so we
          // must not pass undefined here or the new row would behave
          // differently than callers expect.
          requiresTrigger: data.requiresTrigger ?? false,
          isMain: existingGroup?.isMain,
        });
        // Refresh snapshot so available_groups.json reflects new trust config immediately
        const availableGroups = deps.getAvailableGroups();
        deps.writeGroupsSnapshot(
          sourceGroup,
          true,
          availableGroups,
          new Set(Object.keys(registeredGroups)),
        );
      } else {
        logger.warn(
          { data },
          'Invalid register_group request - missing required fields',
        );
      }
    },
  });

  registerIpcHandler('unregister_group', {
    requiresMain: true,
    handler: ({ data, sourceGroup, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Inverse of register_group (#159). Same isMain gate — only the
      // main group can change the registry. The dormant-row problem
      // (#159 motivation) is exactly what happens when there is no
      // structured remove path: rows linger forever, the spawner
      // ignores them because the JSON snapshot doesn't list them, and
      // operators can't fix it from inside chat containers because
      // `/workspace/store/messages.db` is mounted read-only there.
      if (typeof data.jid !== 'string' || data.jid.trim().length === 0) {
        logger.warn(
          { data },
          'Invalid unregister_group request - missing/empty jid',
        );
        return;
      }
      const trimmedJid = data.jid.trim();
      const target = registeredGroups[trimmedJid];
      if (!target) {
        logger.warn(
          { jid: trimmedJid },
          'unregister_group: group not registered (no-op)',
        );
        return;
      }
      // Refuse to unregister a main group via IPC. Losing the main
      // registration mid-runtime would leave the orchestrator without
      // any path that can re-create it (the same isMain gate above
      // would reject the corresponding register_group call). The
      // operator can flip `is_main` directly in the DB if they really
      // mean to, which is a deliberate destructive action rather than
      // a one-line MCP call.
      if (target.isMain) {
        logger.warn(
          { jid: trimmedJid, folder: target.folder },
          'unregister_group: refusing to unregister main group',
        );
        return;
      }
      // Cascade-delete scheduled_tasks tied to the unregistered folder
      // BEFORE we drop the registration. Without this, the scheduler
      // keeps firing the auto-created heartbeat (and any other tasks
      // bound to this folder) every cycle, logging "Group not found
      // for task" on each tick — exactly the noisy-orphan behaviour
      // Copilot flagged on PR #198. We do this before unregisterGroup
      // so a crash between the two leaves the registration alive (DB
      // delete is the authoritative atomic step); the inverse ordering
      // would orphan the registration with its tasks already gone,
      // which is the more confusing recovery path.
      const orphanTasks = getTasksForGroup(target.folder);
      for (const task of orphanTasks) {
        deleteTask(task.id);
      }
      if (orphanTasks.length > 0) {
        logger.info(
          {
            jid: trimmedJid,
            folder: target.folder,
            taskIds: orphanTasks.map((t) => t.id),
          },
          'unregister_group: cascade-deleted scheduled tasks for unregistered folder',
        );
        deps.onTasksChanged();
      }

      const removed = deps.unregisterGroup(trimmedJid);
      if (!removed) {
        // In-memory said yes but DB said no — possible if a parallel
        // path raced us. Log and fall through to snapshot refresh
        // anyway: the snapshot is derived state and a refresh is
        // always safe.
        logger.warn(
          { jid: trimmedJid, folder: target.folder },
          'unregister_group: in-memory entry present but DB delete reported no rows',
        );
      } else {
        logger.info(
          { jid: trimmedJid, folder: target.folder },
          'Group unregistered',
        );
      }
      // Refresh snapshot so available_groups.json no longer flags the
      // removed JID as registered. Same site-of-truth pattern as
      // register_group / set_trusted / set_trigger above.
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        true,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
    },
  });

  registerIpcHandler('set_trusted', {
    requiresMain: true,
    handler: ({ data, sourceGroup, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Partial update: flip container_config.trusted only. Same isMain
      // gate as register_group — only the main group can change trust
      // state. See #105.
      if (
        typeof data.jid === 'string' &&
        data.jid.trim().length > 0 &&
        typeof data.trusted === 'boolean'
      ) {
        // Trim JID for the same reason as set_trigger: avoids a
        // misleading "group not registered" warning when a caller
        // passes whitespace-padded JID.
        const trimmedJid = data.jid.trim();
        const ok = deps.setGroupTrusted(trimmedJid, data.trusted);
        if (!ok) {
          logger.warn(
            { jid: trimmedJid },
            'set_trusted: group not registered (use register_group first)',
          );
          return;
        }
        const availableGroups = deps.getAvailableGroups();
        deps.writeGroupsSnapshot(
          sourceGroup,
          true,
          availableGroups,
          new Set(Object.keys(registeredGroups)),
        );
        // setGroupTrusted may have reconciled the heartbeat task's
        // `script` field as a side effect of the trust flip — refresh
        // the per-group task snapshots so containers see the change
        // on their next read instead of waiting for the orchestrator
        // to write a snapshot for some other reason.
        deps.onTasksChanged();
      } else {
        logger.warn(
          { data },
          'Invalid set_trusted request - missing/empty jid or invalid trusted',
        );
      }
    },
  });

  registerIpcHandler('set_trigger', {
    requiresMain: true,
    handler: ({ data, sourceGroup, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Partial update: change trigger_pattern and optionally
      // requires_trigger. Same isMain gate as register_group. See #105.
      if (
        typeof data.jid === 'string' &&
        data.jid.trim().length > 0 &&
        typeof data.trigger === 'string' &&
        data.trigger.trim().length > 0
      ) {
        // Reject empty/whitespace triggers + JIDs, then pass the
        // trimmed values downstream. `getTriggerPattern('')` trims and
        // falls back to `DEFAULT_TRIGGER`, so an empty trigger would
        // silently revert the group to the assistant's default trigger
        // word — not what the caller asked for. Trimming the JID
        // before lookup avoids a misleading "group not registered"
        // warning when a caller passes `' tg:-123 '` (whitespace would
        // never match the registry key).
        const trimmedJid = data.jid.trim();
        const trimmedTrigger = data.trigger.trim();
        const requiresTrigger =
          typeof data.requiresTrigger === 'boolean'
            ? data.requiresTrigger
            : undefined;
        const ok = deps.setGroupTrigger(
          trimmedJid,
          trimmedTrigger,
          requiresTrigger,
        );
        if (!ok) {
          logger.warn(
            { jid: trimmedJid },
            'set_trigger: group not registered (use register_group first)',
          );
          return;
        }
        const availableGroups = deps.getAvailableGroups();
        deps.writeGroupsSnapshot(
          sourceGroup,
          true,
          availableGroups,
          new Set(Object.keys(registeredGroups)),
        );
      } else {
        logger.warn(
          { data },
          'Invalid set_trigger request - missing/empty jid or trigger',
        );
      }
    },
  });
}
