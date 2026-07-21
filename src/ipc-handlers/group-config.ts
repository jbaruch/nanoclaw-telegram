import { getInstalledTiles } from '../container-runner.js';
import { getTaskById, setTaskAgentModel } from '../db-tasks.js';
import { registerIpcHandler } from '../ipc-registry.js';
import { logger } from '../logger.js';
import type { RegisteredGroup } from '../types.js';

/**
 * Per-group / per-task config-override commands (#845 slice 3):
 * agent-model knobs, session caps, and the tile overlay. The model/cap
 * commands use owner-of-the-bill authorization (main can target any
 * group; a non-main caller only its own folder), so they gate
 * internally; `set_additional_tiles` is trust-adjacent and main-only,
 * so it leans on the dispatcher's `requiresMain` gate.
 */
export function registerGroupConfigIpcHandlers(): void {
  registerIpcHandler('set_agent_model', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Partial update: change `containerConfig.agentModel` only (#395).
      // Authorization mirrors schedule_task — main can target any
      // registered group; non-main can target only its own folder so an
      // untrusted agent can't quietly downgrade another group's model
      // (or escalate its own to a more expensive one for someone else's
      // bill). Sibling containerConfig fields (trusted, additionalMounts,
      // enableHeartbeat, timeout) are preserved verbatim — set_trusted /
      // set_trigger semantics, applied to a new column.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      if (!groupFolder) {
        logger.warn(
          { data },
          'Invalid set_agent_model request - missing/empty groupFolder',
        );
        return;
      }
      // `agentModel` accepts string (set/replace) or null (clear). Anything
      // else (number, object, undefined) is rejected — we don't want a
      // malformed payload to silently no-op.
      if (typeof data.agentModel !== 'string' && data.agentModel !== null) {
        logger.warn(
          { data },
          'Invalid set_agent_model request - agentModel must be string or null',
        );
        return;
      }
      // Locate the target group entry by folder. We need the JID to
      // call deps.registerGroup; iterate the in-memory registry rather
      // than touching the DB directly so the source-of-truth stays in
      // the existing pattern.
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'set_agent_model: group not registered (use register_group first)',
        );
        return;
      }
      // Authorization: non-main can only modify its own folder.
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized set_agent_model attempt blocked',
        );
        return;
      }
      const nextContainerConfig: RegisteredGroup['containerConfig'] = {
        ...(targetGroup.containerConfig ?? {}),
      };
      if (data.agentModel === null) {
        // Explicit clear — drop the field so it serialises as absent
        // (not as JSON null) and the runtime falls through to the
        // global AGENT_MODEL.
        delete nextContainerConfig.agentModel;
      } else {
        const trimmed = data.agentModel.trim();
        if (trimmed.length === 0) {
          // Treat empty/whitespace as a clear, same way
          // resolvePerGroupAgentModel folds empty into fallback.
          delete nextContainerConfig.agentModel;
        } else {
          nextContainerConfig.agentModel = trimmed;
        }
      }
      deps.registerGroup(targetJid, {
        ...targetGroup,
        containerConfig: nextContainerConfig,
      });
      logger.info(
        {
          groupFolder,
          agentModel: nextContainerConfig.agentModel ?? null,
          source: sourceGroup,
        },
        'set_agent_model: updated per-group AGENT_MODEL override',
      );
      // Refresh available_groups.json so containers see the new
      // override on their next read. `isMain` (not hardcoded true)
      // because a non-main source legally lands here when modifying
      // its own folder.
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
    },
  });

  registerIpcHandler('set_maintenance_agent_model', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Partial update: change `containerConfig.maintenanceAgentModel`
      // only (#509). Mirrors set_agent_model semantics — owner-of-the-bill
      // can change their own group's model knobs. Sibling containerConfig
      // fields are preserved verbatim.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      if (!groupFolder) {
        logger.warn(
          { data },
          'Invalid set_maintenance_agent_model request - missing/empty groupFolder',
        );
        return;
      }
      // `maintenanceAgentModel` accepts string (set/replace) or null
      // (clear). Anything else (number, object, undefined) is rejected.
      if (
        typeof data.maintenanceAgentModel !== 'string' &&
        data.maintenanceAgentModel !== null
      ) {
        logger.warn(
          { data },
          'Invalid set_maintenance_agent_model request - maintenanceAgentModel must be string or null',
        );
        return;
      }
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'set_maintenance_agent_model: group not registered (use register_group first)',
        );
        return;
      }
      // Authorization mirrors set_agent_model — non-main can only modify
      // its own folder.
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized set_maintenance_agent_model attempt blocked',
        );
        return;
      }
      const nextContainerConfig: RegisteredGroup['containerConfig'] = {
        ...(targetGroup.containerConfig ?? {}),
      };
      if (data.maintenanceAgentModel === null) {
        delete nextContainerConfig.maintenanceAgentModel;
      } else {
        const trimmed = data.maintenanceAgentModel.trim();
        if (trimmed.length === 0) {
          delete nextContainerConfig.maintenanceAgentModel;
        } else {
          nextContainerConfig.maintenanceAgentModel = trimmed;
        }
      }
      deps.registerGroup(targetJid, {
        ...targetGroup,
        containerConfig: nextContainerConfig,
      });
      logger.info(
        {
          groupFolder,
          maintenanceAgentModel:
            nextContainerConfig.maintenanceAgentModel ?? null,
          source: sourceGroup,
        },
        'set_maintenance_agent_model: updated per-group MAINTENANCE_AGENT_MODEL override',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
    },
  });

  registerIpcHandler('set_session_caps', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Partial update: per-group session-length cap overrides (#561).
      // The global SESSION_TURN_CAP / SESSION_TOKEN_CAP is one knob and
      // must cover the busiest group; this lets a quiet group pin a
      // tighter cap so its context (and maintenance spend) frees sooner.
      // Authorization mirrors set_agent_model — owner-of-the-bill: a
      // non-main caller can only touch its own folder. Sibling
      // containerConfig fields are preserved verbatim.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      if (!groupFolder) {
        logger.warn(
          { data },
          'Invalid set_session_caps request - missing/empty groupFolder',
        );
        return;
      }
      // Each cap independently accepts a positive integer (set), null
      // (clear → inherit global), or undefined (leave unchanged). Reject
      // non-integer / non-positive numbers so a stored override is always
      // meaningful — turns and tokens are discrete counts, and disabling
      // a cap stays a global-only op. Integer-tightening here (not only
      // in the MCP `.int()` schema) closes the raw-IPC path that would
      // otherwise persist a fractional cap like 12.5.
      const validCap = (v: unknown): v is number | null | undefined =>
        v === undefined ||
        v === null ||
        (typeof v === 'number' && Number.isInteger(v) && v > 0);
      if (!validCap(data.sessionTurnCap) || !validCap(data.sessionTokenCap)) {
        logger.warn(
          { data },
          'Invalid set_session_caps request - caps must be a positive integer, null, or omitted',
        );
        return;
      }
      // Both omitted is a no-op request — reject so a malformed payload
      // doesn't masquerade as a successful clear.
      if (
        data.sessionTurnCap === undefined &&
        data.sessionTokenCap === undefined
      ) {
        logger.warn(
          { data },
          'Invalid set_session_caps request - at least one of sessionTurnCap / sessionTokenCap required',
        );
        return;
      }
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'set_session_caps: group not registered (use register_group first)',
        );
        return;
      }
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized set_session_caps attempt blocked',
        );
        return;
      }
      const nextContainerConfig: RegisteredGroup['containerConfig'] = {
        ...(targetGroup.containerConfig ?? {}),
      };
      // A provided field updates; null clears (delete → serialises absent
      // so the runtime falls through to the global cap); undefined leaves
      // the existing value untouched.
      if (data.sessionTurnCap === null) {
        delete nextContainerConfig.sessionTurnCap;
      } else if (data.sessionTurnCap !== undefined) {
        nextContainerConfig.sessionTurnCap = data.sessionTurnCap;
      }
      if (data.sessionTokenCap === null) {
        delete nextContainerConfig.sessionTokenCap;
      } else if (data.sessionTokenCap !== undefined) {
        nextContainerConfig.sessionTokenCap = data.sessionTokenCap;
      }
      deps.registerGroup(targetJid, {
        ...targetGroup,
        containerConfig: nextContainerConfig,
      });
      logger.info(
        {
          groupFolder,
          sessionTurnCap: nextContainerConfig.sessionTurnCap ?? null,
          sessionTokenCap: nextContainerConfig.sessionTokenCap ?? null,
          source: sourceGroup,
        },
        'set_session_caps: updated per-group session-length cap override',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
    },
  });

  registerIpcHandler('set_task_agent_model', {
    handler: ({ data, sourceGroup, isMain }) => {
      // Per-task AGENT_MODEL override for #509 Phase 3. Writes
      // scheduled_tasks.agent_model directly; the row is the source of
      // truth, no in-memory registry to refresh. Authorization mirrors
      // set_maintenance_agent_model — owner-of-bill: a non-main caller
      // can only touch tasks belonging to its own group_folder.
      // Re-uses getTaskById's existing row shape rather than a
      // separate ownership-only query so a non-existent taskId surfaces
      // as a clean "not found" error to the caller instead of a 403.
      const taskId = typeof data.taskId === 'string' ? data.taskId.trim() : '';
      if (!taskId) {
        logger.warn(
          { data },
          'Invalid set_task_agent_model request - missing/empty taskId',
        );
        return;
      }
      // `agentModel` accepts string (set/replace) or null (clear). Anything
      // else (number, object, undefined) is rejected.
      if (typeof data.agentModel !== 'string' && data.agentModel !== null) {
        logger.warn(
          { data },
          'Invalid set_task_agent_model request - agentModel must be string or null',
        );
        return;
      }
      const task = getTaskById(taskId);
      if (!task) {
        logger.warn({ taskId }, 'set_task_agent_model: task not found');
        return;
      }
      // Owner-of-bill auth — non-main caller can only touch its own folder.
      if (!isMain && task.group_folder !== sourceGroup) {
        logger.warn(
          { sourceGroup, taskGroupFolder: task.group_folder, taskId },
          'Unauthorized set_task_agent_model attempt blocked',
        );
        return;
      }
      // Cadence-registry rows are declarative state — their shape
      // (including agent_model) is owned by the SKILL.md frontmatter
      // and reasserted on every per-spawn rebuild
      // (rebuildCadenceRegistry in src/cadence-registry.ts). An
      // imperative IPC write to such a row would silently revert on
      // the next tile-touching spawn, exactly the "looks fine, isn't"
      // failure mode this fleet's host-conventions and OpenAI's review
      // on PR #587 called out. Reject loudly and point the operator at
      // the durable surface (SKILL.md agentModel: frontmatter).
      // Non-cadence rows (source = 'schedule-task' — operator-
      // initiated reminders, ad-hoc monitors, the schedule-task IPC
      // surface) accept the imperative write because they have no
      // declarative source to argue with.
      const taskSource = (task as { source?: string }).source;
      if (taskSource === 'cadence-registry') {
        logger.warn(
          { taskId, source: taskSource },
          'set_task_agent_model: refusing to write to cadence-registry-owned row — modify the skill SKILL.md `agentModel:` frontmatter and republish the tile instead',
        );
        return;
      }
      // Normalise: `null` and empty-after-trim both mean "clear the
      // override" (fall back to the Phase 2 ladder). Trim non-empty
      // strings to match the existing knobs' shape.
      let nextValue: string | null;
      if (data.agentModel === null) {
        nextValue = null;
      } else {
        const trimmed = data.agentModel.trim();
        nextValue = trimmed.length === 0 ? null : trimmed;
      }
      const ok = setTaskAgentModel(taskId, nextValue);
      if (!ok) {
        // Race: row existed at getTaskById time but was deleted before
        // the UPDATE. Surfaces as a no-op for the operator.
        logger.warn(
          { taskId },
          'set_task_agent_model: task disappeared between check and update',
        );
        return;
      }
      logger.info(
        {
          taskId,
          groupFolder: task.group_folder,
          agentModel: nextValue,
          source: sourceGroup,
        },
        'set_task_agent_model: updated per-task agent_model override',
      );
    },
  });

  registerIpcHandler('set_additional_tiles', {
    // Trust-adjacent capability (loading extra skill/rule tiles into the
    // chat's container) — a non-main agent must not be able to grant
    // itself capabilities its trust tier wasn't supposed to have.
    // Mirrors `set_trusted` semantics rather than `set_agent_model`'s
    // "owner can change own bill" semantics.
    requiresMain: true,
    handler: ({ data, sourceGroup, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Partial update: change `containerConfig.additionalTiles` only
      // (#305).
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      if (!groupFolder) {
        logger.warn(
          { data },
          'Invalid set_additional_tiles request - missing/empty groupFolder',
        );
        return;
      }
      // Accept array (set/replace) or null (clear). Reject anything
      // else so a malformed payload doesn't silently no-op the way an
      // `undefined` would. Element-level validation (string, non-empty,
      // installed) runs below.
      const raw = data.additionalTiles;
      if (raw !== null && !Array.isArray(raw)) {
        logger.warn(
          { data },
          'Invalid set_additional_tiles request - additionalTiles must be array or null',
        );
        return;
      }
      // Look up target group by folder. Same iteration pattern as
      // set_agent_model — the in-memory registry is the source of
      // truth and we want the JID to call deps.registerGroup.
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'set_additional_tiles: group not registered (use register_group first)',
        );
        return;
      }

      const nextContainerConfig: RegisteredGroup['containerConfig'] = {
        ...(targetGroup.containerConfig ?? {}),
      };

      if (raw === null || raw.length === 0) {
        // Explicit clear — drop the field so it serialises as absent
        // (not as JSON null / empty array) and `selectTiles` falls
        // through to the trust-tier baseline.
        delete nextContainerConfig.additionalTiles;
      } else {
        // Element validation: every entry must be a non-empty trimmed
        // string. Reject the whole write on the first malformed entry
        // — partial acceptance ("dropped 'foo' but kept the rest")
        // would silently lose capabilities the operator asked for.
        const cleaned: string[] = [];
        for (const entry of raw) {
          if (typeof entry !== 'string') {
            logger.warn(
              { groupFolder, entry },
              'set_additional_tiles: every entry must be a string',
            );
            return;
          }
          const trimmed = entry.trim();
          if (!trimmed) {
            logger.warn(
              { groupFolder },
              'set_additional_tiles: empty/whitespace tile name rejected',
            );
            return;
          }
          // De-dup at write time so the persisted value is clean and
          // selectTiles doesn't have to do the work on every spawn.
          if (!cleaned.includes(trimmed)) cleaned.push(trimmed);
        }

        // Registry validation: every entry must resolve to an
        // installed tile under `tessl-workspace/.tessl/tiles/<owner>/`.
        // `getInstalledTiles()` returns null when the registry
        // directory itself doesn't exist (cold start, never ran
        // `tessl install`) — treat as "nothing installed" so the
        // operator sees the failure now instead of on next spawn.
        const installed = new Set(getInstalledTiles() ?? []);
        const missing = cleaned.filter((t) => !installed.has(t));
        if (missing.length > 0) {
          logger.warn(
            { groupFolder, missing, requested: cleaned },
            `set_additional_tiles rejected: tile${missing.length === 1 ? '' : 's'} not in registry: ${missing.map((t) => `'${t}'`).join(', ')}`,
          );
          return;
        }

        nextContainerConfig.additionalTiles = cleaned;
      }

      deps.registerGroup(targetJid, {
        ...targetGroup,
        containerConfig: nextContainerConfig,
      });
      logger.info(
        {
          groupFolder,
          additionalTiles: nextContainerConfig.additionalTiles ?? null,
          source: sourceGroup,
        },
        'set_additional_tiles: updated per-group tile overlay',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        true,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
    },
  });
}
