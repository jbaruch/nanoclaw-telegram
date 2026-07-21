import { clearCheckpoints } from './checkpoint.js';
import { writeGroupsSnapshot, writeTasksSnapshot } from './container-runner.js';
import {
  clearTaskSessionIdsForGroup,
  clearSessionLengthStateForGroup,
  deleteRegisteredGroup,
  deleteSession,
  deleteSessionName,
  getAllTasks,
  updateGroupTrusted,
  updateGroupTrigger,
} from './db.js';
import {
  DEFAULT_SESSION_NAME,
  MAINTENANCE_SESSION_NAME,
} from './group-queue.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { findChannel, formatOutbound } from './router.js';
import { ChannelType } from './text-styles.js';
import { logger } from './logger.js';
import { isBreakerActive } from './circuit-breaker.js';
import { registeredGroups, sessions } from './orchestrator-state.js';
import {
  channels,
  nukeTimestamps,
  pendingReplyTo,
  queue,
} from './orchestrator-runtime.js';
import { getAvailableGroups } from './message-pipeline.js';
import { consumeReplyAnchorOnVisibleSend } from './agent-output-action.js';
import { wipeSessionJsonl } from './session-wipe.js';
import { registerGroup } from './group-registry.js';
import type { IpcDeps } from './ipc.js';

// The orchestrator IPC dependency wiring, extracted from src/index.ts
// (#749 seam 4b). Every handler closes over module-level singletons
// (queue/channels/registeredGroups/... from ./orchestrator-runtime.js and
// ./orchestrator-state.js) and the extracted pipeline/registry functions,
// so main just wires startIpcWatcher(ipcDeps).
export const ipcDeps: IpcDeps = {
  // #722: release the reply anchor when the sending group's OWN chat
  // received a confirmed visible reply via IPC send_message/send_file
  // — the mark-displayed consumption in the output callback arrives
  // only with the SDK result, and a pipe landing in that gap must be
  // able to claim the anchor for the follow-up turn. Cross-chat
  // sends never touch the target chat's anchor.
  onVisibleReply: (chatJid, sourceGroupFolder) => {
    const isOwnChat = registeredGroups[chatJid]?.folder === sourceGroupFolder;
    const consumed = consumeReplyAnchorOnVisibleSend(
      pendingReplyTo,
      chatJid,
      isOwnChat,
    );
    if (consumed) {
      logger.debug(
        { chatJid, sourceGroupFolder },
        'Reply anchor consumed at IPC visible-send boundary',
      );
    }
  },
  sendMessage: (jid, rawText, replyToMessageId) => {
    const channel = findChannel(channels, jid);
    if (!channel) throw new Error(`No channel for JID: ${jid}`);
    const text = formatOutbound(rawText, channel.name as ChannelType);
    if (!text) return Promise.resolve();
    return channel.sendMessage(jid, text, replyToMessageId);
  },
  sendReaction: async (jid, messageId, emoji) => {
    const channel = findChannel(channels, jid);
    if (!channel) return;
    if (messageId) {
      await channel.sendReaction?.(jid, messageId, emoji);
    } else {
      await channel.reactToLatestMessage?.(jid, emoji);
    }
  },
  pinMessage: async (jid, messageId) => {
    const channel = findChannel(channels, jid);
    if (!channel) return;
    await channel.pinMessage?.(jid, messageId);
  },
  sendFile: async (jid, filePath, caption, replyToMessageId) => {
    const channel = findChannel(channels, jid);
    if (!channel) return undefined;
    return channel.sendFile?.(jid, filePath, caption, replyToMessageId);
  },
  registeredGroups: () => registeredGroups,
  registerGroup,
  unregisterGroup: (jid) => {
    // Mirror DB delete into the in-memory registry so subsequent
    // routing decisions stop seeing the JID as registered before any
    // restart. Same site-of-truth pattern as setGroupTrusted /
    // setGroupTrigger above. Returns the DB delete's truthy-changes
    // result so the IPC handler can distinguish "actually removed"
    // from "wasn't there to begin with" — see #159.
    delete registeredGroups[jid];
    return deleteRegisteredGroup(jid);
  },
  setGroupTrusted: (jid, trusted) => {
    const updated = updateGroupTrusted(jid, trusted);
    if (!updated) return false;
    // Mirror DB change into the in-memory registry so subsequent
    // routing decisions see the new trust flag immediately, before any
    // restart. Without this, the agent would have to wait for the
    // orchestrator to reload from DB to see its own update.
    registeredGroups[jid] = updated;

    // Trust flips no longer touch heartbeat task state. The non-main
    // heartbeat was retired in #453 with the check-unanswered skill it
    // depended on; there's no per-trust-tier heartbeat script left to
    // reconcile. Main-group heartbeat is created and managed by
    // `registerGroup`'s `group.isMain` branch and isn't trust-tier
    // sensitive. If a future feature reintroduces a non-main
    // heartbeat with trust-tier-conditional behaviour, restore the
    // reconciliation here.
    return true;
  },
  setGroupTrigger: (jid, trigger, requiresTrigger) => {
    const updated = updateGroupTrigger(jid, trigger, requiresTrigger);
    if (!updated) return false;
    registeredGroups[jid] = updated;
    // Heartbeat lifecycle is intentionally NOT touched here. Pre-#158
    // a flip to `requiresTrigger=true` would auto-create a heartbeat
    // and the inverse flip would log a warning. With heartbeat opt-in
    // via `containerConfig.enableHeartbeat`, trigger config and
    // heartbeat are orthogonal — operators change each independently.
    return true;
  },
  syncGroups: async (force: boolean) => {
    await Promise.all(
      channels.filter((ch) => ch.syncGroups).map((ch) => ch.syncGroups!(force)),
    );
  },
  getAvailableGroups,
  writeGroupsSnapshot: (gf, im, ag, rj) => writeGroupsSnapshot(gf, im, ag, rj),
  nukeSession: (
    groupFolder: string,
    session: 'default' | 'maintenance' | 'all',
    options?: { skipReentry?: boolean },
  ) => {
    // Stamp the nuke's wall-clock timestamp BEFORE doing any of the
    // wipe work — the in-flight spawn handler (runAgent) compares
    // this against its own pre-spawn timestamp to decide whether
    // the SDK result it just got is from a session that's since
    // been nuked. The Date.now() resolution + the
    // setSession-before-nuke sequence is robust against the race
    // observed in #144 bug 1 (concurrent setSession resurrected
    // a row that the same call had just deleted).
    nukeTimestamps[groupFolder] = Date.now();
    // Granular nuke: `session` narrows which slot(s) to kill.
    //   'all'         → kill default + maintenance (pre-parallel default)
    //   'default'     → kill only user-facing container
    //   'maintenance' → kill only scheduled-task container
    // Useful when one session is wedged (e.g. a hung heartbeat in
    // maintenance) and we don't want to drop the user's default
    // conversation state as collateral damage.
    //
    // Per #100, the nuke runs in four steps, in order:
    //   1. Capture the SDK sessionIds we're about to drop (before
    //      clearing them — once they're gone we can't find the
    //      on-disk artifacts).
    //   2. Kill the running container(s) so nothing keeps writing.
    //   3. Delete the session rows from the DB and clear in-memory.
    //   4. Delete the on-disk session artifacts (JSONL transcript
    //      and the per-session tool-results directory beside it).
    //
    // Without step 4, the next container spawn re-reads whatever poison
    // / stuck plan / corrupt state put the session in a bad state and
    // we're right back where we started — see #100 for the Gmail
    // invisible-Unicode incident that motivated this.
    const slotsToWipe: Array<'default' | 'maintenance'> =
      session === 'all'
        ? ['default', 'maintenance']
        : [
            session === 'default'
              ? DEFAULT_SESSION_NAME
              : MAINTENANCE_SESSION_NAME,
          ];
    const sessionIdsToWipe = new Map<string, string>();
    for (const slot of slotsToWipe) {
      const sid = sessions[groupFolder]?.[slot];
      if (sid) sessionIdsToWipe.set(slot, sid);
    }

    const jid =
      Object.entries(registeredGroups).find(
        ([, g]) => g.folder === groupFolder,
      )?.[0] || '';
    if (jid) {
      if (session === 'default' || session === 'all') {
        queue.closeStdin(jid, DEFAULT_SESSION_NAME);
      }
      if (session === 'maintenance' || session === 'all') {
        queue.closeStdin(jid, MAINTENANCE_SESSION_NAME);
      }
    }
    // Clear stored sessionIds for the killed slot(s). `deleteSession`
    // removes every row for the folder — reuse for 'all'. For
    // single-slot nukes we use the new `deleteSessionName` helper so
    // the surviving slot keeps its session chain.
    if (session === 'all') {
      delete sessions[groupFolder];
      deleteSession(groupFolder);
    } else {
      const sessionName =
        session === 'default' ? DEFAULT_SESSION_NAME : MAINTENANCE_SESSION_NAME;
      if (sessions[groupFolder]) delete sessions[groupFolder][sessionName];
      deleteSessionName(groupFolder, sessionName);
    }

    // Step 4: wipe on-disk session artifacts (JSONL transcript +
    // per-session tool-results directory). Delete-while-open is
    // safe on POSIX (the container's open FD keeps writing to a
    // phantom inode that vanishes on close), so we don't have to wait
    // for closeStdin to actually terminate the process. The returned
    // `count` is the total number of filesystem entries removed: up
    // to 2 per slug (1 transcript + 1 tool-results dir), summed
    // across every project-slug subdirectory walked.
    for (const [slot, sessionId] of sessionIdsToWipe) {
      const wiped = wipeSessionJsonl(groupFolder, slot, sessionId);
      if (wiped > 0) {
        logger.info(
          { groupFolder, sessionName: slot, sessionId, count: wiped },
          'Wiped session artifacts (transcript + tool-results dir)',
        );
      }
    }

    // Step 4b (#336): clear per-task `session_id` columns for any
    // scheduled task in this group that referenced the just-wiped
    // maintenance transcripts. Without this, the next fire of a
    // recurring task would pass `resume:` an id whose JSONL is
    // gone — the SDK would 404 and start fresh anyway, just
    // noisily. Only clears when the maintenance slot was actually
    // touched: a 'default'-only nuke leaves scheduled-task sessions
    // untouched (they live in maintenance, with their own ids).
    // Fires before `skipReentry` so checkpoint state and per-task
    // session state both reach "clean slate" together.
    if (session === 'maintenance' || session === 'all') {
      const cleared = clearTaskSessionIdsForGroup(groupFolder);
      if (cleared > 0) {
        logger.info(
          { groupFolder, count: cleared },
          'Cleared per-task session_ids — next fire of each will start a fresh SDK session (#336)',
        );
      }
    }

    // Step 4c (#413): drop session-length-cap accounting for the
    // nuked group. Without this, a stale row could carry a
    // `marked_for_reset = 1` flag against a session_id that's
    // already gone, and the next inbound spawn would consume the
    // marker (harmless but noisy in logs). The cap state is
    // group-level; an 'all' nuke clears every slot, while a
    // single-slot nuke is rare enough that wiping both slots'
    // accounting is fine — the surviving slot's next turn
    // re-INSERTs cleanly.
    const clearedCapRows = clearSessionLengthStateForGroup(groupFolder);
    if (clearedCapRows > 0) {
      logger.info(
        { groupFolder, count: clearedCapRows },
        'Cleared session_length_state rows on nuke (#413)',
      );
    }

    // Step 5 (#127, optional): when `skipReentry` is set, also
    // delete the per-group checkpoint files so the next container
    // spawn has no Facts/Reasoning to load via the reentry skill.
    // Default behaviour (option absent or false) preserves the
    // checkpoint — the standard nuke is "fresh session, but the
    // reentry skill still runs" because checkpoints typically
    // outlive a single nuke (they're written by the threshold-cross
    // path). Skip-reentry exists for the case where the checkpoint
    // itself is the problem (poisoned plan, stale do-not-re-execute
    // list); without this, the operator's only workaround was a
    // manual `rm` from the host.
    //
    // Checkpoint files are per-group, NOT per-slot — there's one
    // pair under `<groupDir>/.checkpoints/` shared by both default
    // and maintenance. So skipReentry deletes the same files
    // regardless of which slot was nuked. That matches the design
    // doc (see `docs/proposals/kill-auto-compaction.md` §1, §2):
    // the Facts section is the orchestrator's view of "what just
    // happened in this group", not slot-specific.
    // Strict boolean check (defense in depth): the IPC layer
    // already filters non-true values, but a future direct caller
    // (test, new IPC handler, refactor) could pass a truthy
    // non-boolean and accidentally erase reentry state. The dispatcher
    // is the last gate before the disk operation, so it owns the
    // strictest check.
    if (options?.skipReentry === true) {
      let groupDir: string;
      try {
        groupDir = resolveGroupFolderPath(groupFolder);
      } catch (err) {
        // Per `jbaruch/coding-policy: error-handling`: only handle
        // the expected case (Error from path validation), let
        // anything else propagate. resolveGroupFolderPath
        // documents Error throws on path-traversal / invalid
        // segment; non-Error throws here would indicate a bug
        // upstream and should bubble up to the IPC dispatch
        // wrapper, which logs and keeps the orchestrator alive.
        if (!(err instanceof Error)) throw err;
        // The expected case: bad groupFolder. Log full error
        // object (logger handles `err` specially — preserves
        // stack, formats nicely) and skip the checkpoint clear
        // without blocking the rest of the nuke. Reentry skill
        // will find the checkpoint still on disk; operator can
        // rerun with a fixed group_folder.
        logger.error(
          { groupFolder, err },
          'skipReentry: cannot resolve group folder — checkpoint files left in place',
        );
        logger.info({ groupFolder, session }, 'Session nuked via IPC');
        return;
      }
      // Best-effort cleanup: if clearCheckpoints throws (e.g.
      // EACCES on unlink — a file was found but couldn't be
      // removed), log at error level and CONTINUE with the rest of
      // the nuke. The main session state (DB rows + JSONL) is
      // already wiped at this point; failing the whole IPC handler
      // would be noisier than helpful and contradicts the
      // best-effort framing the surrounding comments describe.
      // Non-Error throws still propagate as upstream bugs per
      // `jbaruch/coding-policy: error-handling`.
      try {
        const checkpointsDeleted = clearCheckpoints(groupDir);
        logger.info(
          { groupFolder, checkpointsDeleted },
          'Checkpoint files cleared (skipReentry=true)',
        );
      } catch (err) {
        if (!(err instanceof Error)) throw err;
        logger.error(
          { groupFolder, groupDir, session, err },
          'skipReentry: failed to clear checkpoint files — continuing with session nuke',
        );
      }
    }

    logger.info(
      { groupFolder, session, skipReentry: options?.skipReentry === true },
      'Session nuked via IPC',
    );
  },
  getContainerStatus: (chatJid, sessionName) => {
    // Combine the GroupQueue's per-slot signals (active/idleWaiting/
    // retryCount/lastExitStatus) with the long-term per-folder
    // circuit breaker. The breaker lives here, not in GroupQueue,
    // because it's keyed on group.folder and is set by message-loop
    // bookkeeping rather than queue lifecycle. Both signals are
    // cooldown windows from the chat_status caller's perspective.
    const group = registeredGroups[chatJid];
    const breakerActive = group ? isBreakerActive(group.folder) : false;
    return queue.getStatus(chatJid, sessionName, breakerActive);
  },
  onTasksChanged: () => {
    const tasks = getAllTasks();
    const taskRows = tasks.map((t) => ({
      id: t.id,
      groupFolder: t.group_folder,
      prompt: t.prompt,
      script: t.script || undefined,
      schedule_type: t.schedule_type,
      schedule_value: t.schedule_value,
      status: t.status,
      next_run: t.next_run,
    }));
    for (const group of Object.values(registeredGroups)) {
      writeTasksSnapshot(
        group.folder,
        group.isMain === true,
        taskRows,
        !!group.containerConfig?.trusted,
      );
    }
  },
  closeAllActiveContainers: () => queue.closeAllActiveContainers(),
};
