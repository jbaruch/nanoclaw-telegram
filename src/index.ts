import { SqliteError } from 'better-sqlite3';

import {
  ASSISTANT_NAME,
  ASSISTANT_OWNER_TG_USER_ID,
  CREDENTIAL_PROXY_PORT,
  DATA_DIR,
  MAX_MESSAGES_PER_PROMPT,
  MODEL_CONTEXT_WINDOW,
  TELEGRAM_BOT_POOL,
} from './config.js';
import { clearCheckpoints } from './checkpoint.js';
import { writeShutdownCheckpoints } from './shutdown-checkpoints.js';
import { computeThresholds } from './threshold.js';
import { startCredentialProxy } from './credential-proxy.js';
import {
  ensureAgentForTier,
  isOneCliConfigured,
  TRUST_TIERS,
} from './onecli-client.js';
import './channels/index.js';
import {
  getChannelFactory,
  getRegisteredChannelNames,
} from './channels/registry.js';
import { writeGroupsSnapshot, writeTasksSnapshot } from './container-runner.js';
import {
  cleanupOrphans,
  ensureContainerRuntimeRunning,
  PROXY_BIND_HOST,
} from './container-runtime.js';
import {
  markHandoffActive,
  readAndConsumeHandoffMarker,
  writeHandoffMarker,
} from './handoff.js';
import {
  clearStalePendingRunAt,
  clearTaskSessionIdsForGroup,
  clearSessionLengthStateForGroup,
  getActivePendingRunAtNames,
  deleteAllSessions,
  deleteRegisteredGroup,
  deleteSession,
  deleteSessionName,
  getAllTasks,
  getCurrentTz,
  getMessagesSince,
  runTzHeartbeatAdvisory,
  TzAdvisoryResult,
  initDatabase,
  updateGroupTrusted,
  updateGroupTrigger,
  storeChatMetadata,
  storeLocation,
  storeMessage,
} from './db.js';
import {
  DEFAULT_SESSION_NAME,
  MAINTENANCE_SESSION_NAME,
} from './group-queue.js';
import { BEST_EFFORT_FS_CODES, isFsErrorWithCode } from './fs-errors.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { writeFlightAssistLocation } from './flight-assist-location.js';
import { initBotPool } from './channels/telegram.js';
import { startIpcWatcher } from './ipc.js';
import { findChannel, formatOutbound } from './router.js';
import { ChannelType } from './text-styles.js';
import {
  restoreRemoteControl,
  startRemoteControl,
  stopRemoteControl,
} from './remote-control.js';
import { pruneOldContainerLogs } from './host-logs.js';
import { startSessionCleanup } from './session-cleanup.js';
import {
  startHubitatListener,
  stopHubitatListener,
} from './hubitat-listener.js';
import {
  recomputeLocalSchedules,
  startSchedulerLoop,
} from './task-scheduler.js';
import { startTriggerLearner } from './gates/trigger-learner-runtime.js';
import { installTelegramOutboundTap } from './telegram-outbound-tap.js';
import { LocationRecord, NewMessage } from './types.js';
import { logger } from './logger.js';
import { initObserver } from './observer.js';
import { isBreakerActive } from './circuit-breaker.js';
import {
  getOrRecoverCursor,
  loadState,
  registeredGroups,
  sessions,
} from './orchestrator-state.js';
import {
  channels,
  nukeTimestamps,
  pendingReplyTo,
  queue,
} from './orchestrator-runtime.js';
import {
  getAvailableGroups,
  processGroupMessages,
  startMessageLoop,
} from './message-pipeline.js';
import { checkSilentZero } from './usage-log.js';
import { consumeReplyAnchorOnVisibleSend } from './agent-output-action.js';
import { wipeSessionJsonl } from './session-wipe.js';
import {
  cleanupOrphanNonMainHeartbeats,
  logRegisteredGroupOrphans,
  registerGroup,
} from './group-registry.js';

// Re-export for backwards compatibility during refactor
export { escapeXml, formatMessages } from './router.js';
export { wipeSessionJsonl } from './session-wipe.js';
// Orchestrator state moved to ./orchestrator-state.ts (#749); the test
// helper is re-exported here because routing.test.ts / router-integration
// import it from ./index.js.
export { _setRegisteredGroups } from './orchestrator-state.js';
// Message pipeline moved to ./message-pipeline.ts (#749); these are
// re-exported because routing.test.ts / stale-session.test.ts import
// them from ./index.js.
export { getAvailableGroups, isStaleSessionError } from './message-pipeline.js';
// Group registry moved to ./group-registry.ts (#749); the trigger-pattern
// hydrator is re-exported because register-group.test.ts imports it from
// ./index.js.
export { hydrateRegisteredGroupTriggerPatterns } from './group-registry.js';
// Gate orchestration helpers moved to ./gates/orchestrator.ts (#749);
// re-exported here because the gate tests import them from ./index.js.
export {
  resolveGatesForGroup,
  evaluateGateChain,
  gateAllowsSpawn,
} from './gates/orchestrator.js';

// #496 — `follow_me_tasks.pending_run_at` is treated as "fresh" (and
// therefore worth respecting) for this long after stamp-time. Anything
// older is presumed orphaned by a dead/killed container and gets
// reclaimed via `clearStalePendingRunAt`.
//
// Sized for the largest plausible follow-me task duration with margin:
// the morning-brief skill (the longest in the fleet) caps around
// 10–15 min of agent work; 1h gives 4× margin for an unusually slow
// run while still recovering well before the next-day fire. A shorter
// window (e.g. 10 min) would risk reclaiming an in-flight lock from a
// genuinely long but legitimate run; a longer window delays recovery
// past the daily-fire boundary, defeating the point.
const PENDING_RUN_AT_FRESHNESS_MS = 60 * 60 * 1000; // 1 hour

/**
 * Startup recovery: check for unprocessed messages in registered groups.
 * Handles crash between advancing lastTimestamp and processing messages.
 */
function recoverPendingMessages(): void {
  for (const [chatJid, group] of Object.entries(registeredGroups)) {
    const pending = getMessagesSince(
      chatJid,
      getOrRecoverCursor(chatJid),
      ASSISTANT_NAME,
      MAX_MESSAGES_PER_PROMPT,
    );
    if (pending.length > 0) {
      logger.info(
        { group: group.name, pendingCount: pending.length },
        'Recovery: found unprocessed messages',
      );
      queue.enqueueMessageCheck(chatJid);
    }
  }
}

function ensureContainerSystemRunning(): void {
  ensureContainerRuntimeRunning();
  // #213: consult the graceful-shutdown handoff marker before killing
  // orphans. If the prior orchestrator exited cleanly (via the
  // SIGTERM handler below), it left a list of agent containers that
  // were still doing useful work — `cleanupOrphans` skips those and
  // only kills the rest. Without this, every `deploy.sh` cascades
  // 137 across every active conversation/heartbeat container, losing
  // the in-flight turn. A marker absent / stale / corrupt falls
  // through to the pre-#213 "kill all nanoclaw-* containers"
  // behaviour, which is the right default for genuine crash recovery
  // (a SIGKILL'd or hung orchestrator never wrote a marker).
  const handoff = readAndConsumeHandoffMarker();
  const skipNames = handoff
    ? new Set(handoff.containers.map((c) => c.name))
    : undefined;
  if (handoff) {
    logger.info(
      {
        shutdownAt: handoff.shutdown_at,
        containerCount: handoff.containers.length,
      },
      'Found graceful-shutdown handoff marker, adopting containers',
    );
    // Open the spawn-collision detection window (#213 Phase A) ONLY
    // when there's actually something to collide with. An empty
    // handoff (graceful shutdown with no active agents) means no
    // adopted containers exist — opening the window would cost a
    // `docker ps` per spawn for the next HANDOFF_TTL_MS and could
    // produce misleading WARNs against this orchestrator's own
    // freshly-spawned containers (the prefix check has no way to
    // distinguish "leftover from prior run" from "just spawned by
    // this run" once the prior run had nothing). Skip the window
    // when the container list is empty.
    if (handoff.containers.length > 0) {
      markHandoffActive();
    }
  }
  cleanupOrphans(skipNames);
}

async function main(): Promise<void> {
  // Install the outbound-Telegram HTTP tap FIRST — before any grammy Bot,
  // credential proxy, or channel loads. The tap wraps `fetch`, `http/https`
  // `request`, and `child_process.spawn/exec/execFile` to log any outbound
  // call to `api.telegram.org` regardless of which in-process code path
  // originates it. Gated on `LOG_LEVEL=debug` (same as #87's transformer);
  // no overhead at `info` or higher. See `telegram-outbound-tap.ts`.
  installTelegramOutboundTap();
  ensureContainerSystemRunning();
  initDatabase();
  logger.info('Database initialized');
  loadState();
  restoreRemoteControl();

  // Start credential proxy (containers route API calls through this)
  const proxyServer = await startCredentialProxy(
    CREDENTIAL_PROXY_PORT,
    PROXY_BIND_HOST,
  );

  // #564 groundwork: when OneCLI is configured (ONECLI_URL + ONECLI_API_KEY
  // set), register the tier-scoped agents so subsequent container spawns
  // can attach to them via `applyContainerConfig`. No-op otherwise.
  if (isOneCliConfigured()) {
    await Promise.all(TRUST_TIERS.map((tier) => ensureAgentForTier(tier)));
  }

  // Graceful shutdown handlers
  const shutdown = async (signal: string) => {
    logger.info({ signal }, 'Shutdown signal received');
    // #213: snapshot active agent containers BEFORE `queue.shutdown`
    // clears their state, so the next startup can identify them as
    // intentional handoffs and skip the orphan-cleanup that
    // otherwise cascades 137 across every in-flight conversation /
    // heartbeat. Best-effort: a write failure here just means the
    // next startup will fall through to the pre-#213 cleanup
    // behaviour for these names — strictly worse than the new
    // path, but no worse than today.
    let active: ReturnType<typeof queue.getActiveContainersForHandoff> = [];
    try {
      active = queue.getActiveContainersForHandoff();
      writeHandoffMarker(active);
      logger.info(
        { count: active.length, names: active.map((c) => c.name) },
        'Wrote graceful-shutdown handoff marker',
      );
    } catch (err) {
      if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
      logger.warn({ err }, 'Failed to write handoff marker');
    }

    // #497: checkpoint every active default session BEFORE
    // `queue.shutdown` runs, so deploy SIGTERM → force-kill no longer
    // discards in-flight work. The existing SessionStart hook +
    // `session-reentry` skill pipeline picks up the checkpoint on the
    // next fresh spawn — no new reentry surface.
    const checkpointed = await writeShutdownCheckpoints({
      active,
      sessions,
      registeredGroups,
      thresholds: computeThresholds(MODEL_CONTEXT_WINDOW),
      defaultSessionName: DEFAULT_SESSION_NAME,
      dataDir: DATA_DIR,
      logger,
    });
    logger.info(
      { count: checkpointed },
      'Pre-shutdown checkpoint pass complete',
    );

    stopHubitatListener();
    proxyServer.close();
    await queue.shutdown(10000);
    for (const ch of channels) await ch.disconnect();
    process.exit(0);
  };
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  // Handle /remote-control and /remote-control-end commands
  async function handleRemoteControl(
    command: string,
    chatJid: string,
    msg: NewMessage,
  ): Promise<void> {
    const group = registeredGroups[chatJid];
    if (!group?.isMain) {
      logger.warn(
        { chatJid, sender: msg.sender },
        'Remote control rejected: not main group',
      );
      return;
    }

    if (!msg.is_from_me) {
      logger.warn(
        { chatJid, sender: msg.sender },
        'Remote control rejected: sender is not the account owner',
      );
      return;
    }

    const channel = findChannel(channels, chatJid);
    if (!channel) return;

    if (command === '/remote-control') {
      const result = await startRemoteControl(
        msg.sender,
        chatJid,
        process.cwd(),
      );
      if (result.ok) {
        await channel.sendMessage(chatJid, result.url);
      } else {
        await channel.sendMessage(
          chatJid,
          `Remote Control failed: ${result.error}`,
        );
      }
    } else {
      const result = stopRemoteControl();
      if (result.ok) {
        await channel.sendMessage(chatJid, 'Remote Control session ended.');
      } else {
        await channel.sendMessage(chatJid, result.error);
      }
    }
  }

  // Channel callbacks (shared by all channels)
  const channelOpts = {
    onMessage: (chatJid: string, msg: NewMessage) => {
      // Remote control commands — intercept before storage
      const trimmed = msg.content.trim();
      if (trimmed === '/remote-control' || trimmed === '/remote-control-end') {
        handleRemoteControl(trimmed, chatJid, msg).catch((err) =>
          logger.error({ err, chatJid }, 'Remote control command error'),
        );
        return;
      }

      storeMessage(msg);
    },
    onChatMetadata: (
      chatJid: string,
      timestamp: string,
      name?: string,
      channel?: string,
      isGroup?: boolean,
    ) => storeChatMetadata(chatJid, timestamp, name, channel, isGroup),
    onLocation: (record: LocationRecord) => {
      storeLocation(record);
      // Sidecar write for `jbaruch/nanoclaw-travel`'s
      // `precheck.py` origin-resolution ladder (issue
      // `nanoclaw-travel#18`). The DB row drives the host-side
      // TZ resolver; this file drives the per-group container's
      // time-to-leave origin. Filters to owner-only inside.
      writeFlightAssistLocation(record, {
        groups: registeredGroups,
        ownerSenderId: ASSISTANT_OWNER_TG_USER_ID,
        dataDir: DATA_DIR,
      });
    },
    registeredGroups: () => registeredGroups,
  };

  // Create and connect all registered channels.
  // Each channel self-registers via the barrel import above.
  // Factories return null when credentials are missing, so unconfigured channels are skipped.
  for (const channelName of getRegisteredChannelNames()) {
    const factory = getChannelFactory(channelName)!;
    const channel = factory(channelOpts);
    if (!channel) {
      logger.warn(
        { channel: channelName },
        'Channel installed but credentials missing — skipping. Check .env or re-run the channel skill.',
      );
      continue;
    }
    channels.push(channel);
    await channel.connect();
  }
  if (channels.length === 0) {
    logger.fatal('No channels connected');
    process.exit(1);
  }

  // Initialize Telegram bot pool for agent teams (swarm)
  if (TELEGRAM_BOT_POOL.length > 0) {
    await initBotPool(TELEGRAM_BOT_POOL);
  }

  // Cleanup MUST run before the scheduler starts. `startSchedulerLoop`
  // below kicks off its first `loop()` immediately, and if any orphan
  // non-main heartbeat row from an older orchestrator version has
  // `next_run <= now`, it would dispatch the (now-broken) old prompt
  // before cleanup gets a chance to delete the row. Running the cleanup
  // first guarantees the first post-deploy scheduler tick sees only
  // valid task rows.
  cleanupOrphanNonMainHeartbeats();

  // Surface registered-but-invisible-to-spawner rows (#159) at startup
  // so we notice future drift instead of growing dormant rows silently.
  logRegisteredGroupOrphans();

  // Initialize the optional observer channel — opt-in via OBSERVER_CHAT_JID
  // env var (no-op when unset). See src/observer.ts. Awaited so the
  // privacy-gate verification (warn loudly when the configured chat
  // is a multi-participant group, refuse only when no channel owns
  // the JID or chat-type lookup fails) finishes before we start
  // spawning queries that would feed the observer.
  await initObserver(channels, () => registeredGroups);

  // Start subsystems (independently of connection handler).
  // Scheduled tasks run through the shared queue under the parallel
  // `maintenance` slot, but they do NOT resume or persist an SDK session
  // chain across runs (#193). Each run gets a fresh sessionId; the
  // scheduler wipes the per-run on-disk artifacts (JSONL transcript +
  // tool-results dir) via `wipeSessionJsonl` once the run completes so
  // the per-slot `.claude/projects/` tree doesn't accumulate orphans.
  startSchedulerLoop({
    registeredGroups: () => registeredGroups,
    queue,
    onProcess: (groupJid, sessionName, proc, containerName, groupFolder) =>
      queue.registerProcess(
        groupJid,
        sessionName,
        proc,
        containerName,
        groupFolder,
      ),
    sendMessage: async (jid, rawText) => {
      const channel = findChannel(channels, jid);
      if (!channel) {
        logger.warn({ jid }, 'No channel owns JID, cannot send message');
        return;
      }
      const text = formatOutbound(rawText, channel.name as ChannelType);
      if (!text) return;
      // Return the channel-native message id so the scheduled-task
      // forward can record `telegram_message_id` and gate its bot-row
      // write on delivery (#681).
      return channel.sendMessage(jid, text);
    },
    wipeSessionJsonl,
  });
  // Stage 1 trigger-pattern self-improvement loop (#82). Runs at a
  // configurable cadence (default 24h via TRIGGER_LEARNER_INTERVAL_MS),
  // emits PROPOSALS into `registered_groups.trigger_pattern` with
  // `enabled: false` (the trigger gate skips those rows). The owner
  // promotes via the existing admin path — there is no auto-apply.
  // Safety rails: cold-start floor, sender-tier weighting, auto-rollback
  // on FP-rate spike, and pattern versioning. See
  // `src/gates/trigger-learner.ts` for the full design.
  startTriggerLearner();
  startIpcWatcher({
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
        channels
          .filter((ch) => ch.syncGroups)
          .map((ch) => ch.syncGroups!(force)),
      );
    },
    getAvailableGroups,
    writeGroupsSnapshot: (gf, im, ag, rj) =>
      writeGroupsSnapshot(gf, im, ag, rj),
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
          session === 'default'
            ? DEFAULT_SESSION_NAME
            : MAINTENANCE_SESSION_NAME;
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
  });
  startSessionCleanup();
  queue.setProcessMessagesFn(processGroupMessages);
  recoverPendingMessages();

  // Per-container streaming logs grow with every spawn. Without
  // pruning, `data/host-logs/containers/<group>/<session>/*.log`
  // would accumulate indefinitely — a chatty trusted group spawning
  // many times a day fills the disk over a few months. Run prune at
  // startup AND once a day thereafter; both are safe and cheap.
  // Retention window is owned by host-logs.ts (currently 7 days).
  void (async () => {
    try {
      const deleted = pruneOldContainerLogs();
      if (deleted > 0) {
        logger.info(
          { deleted },
          'host-logs prune at startup removed expired per-spawn logs',
        );
      }
    } catch (err) {
      if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
      logger.warn({ err }, 'host-logs prune at startup failed');
    }
  })();
  // 24h interval, unref'd so the timer doesn't keep the orchestrator
  // alive past graceful shutdown. setInterval is fine even though the
  // logical schedule is "once per day" — the orchestrator process
  // typically lives for weeks, and crash recovery brings the timer
  // back on next start.
  const ONE_DAY_MS = 24 * 60 * 60 * 1000;
  setInterval(() => {
    try {
      const deleted = pruneOldContainerLogs();
      if (deleted > 0) {
        logger.info(
          { deleted },
          'host-logs daily prune removed expired per-spawn logs',
        );
      }
    } catch (err) {
      if (!isFsErrorWithCode(err, BEST_EFFORT_FS_CODES)) throw err;
      logger.warn({ err }, 'host-logs daily prune failed');
    }
  }, ONE_DAY_MS).unref();

  // Silent-zero-output guard for the credential-proxy usage log
  // (#479 sub-#3). Polls every minute; emits one ERROR if the proxy
  // has handled at least one /v1/messages POST but produced zero
  // JSONL records past the warmup grace period. Edge-triggered
  // against messagesSeen so a persistent broken state logs once,
  // not every tick — see `checkSilentZero` in src/usage-log.ts.
  const SILENT_ZERO_POLL_MS = 60 * 1000;
  setInterval(() => {
    const diag = checkSilentZero();
    if (diag) {
      logger.error(
        { ...diag },
        'usage-log: /v1/messages traffic seen but zero JSONL records — capture pipeline broken (see #479 sub-#3)',
      );
    }
  }, SILENT_ZERO_POLL_MS).unref();

  // Write available_groups.json for all main/trusted groups on startup.
  // Otherwise the snapshot only updates when a container spawns, which can
  // leave it weeks stale if the group doesn't get traffic.
  const startupGroups = getAvailableGroups();
  const startupRegisteredJids = new Set(Object.keys(registeredGroups));
  for (const [, group] of Object.entries(registeredGroups)) {
    if (group.isMain || group.containerConfig?.trusted) {
      writeGroupsSnapshot(
        group.folder,
        group.isMain === true,
        startupGroups,
        startupRegisteredJids,
        !!group.containerConfig?.trusted,
      );
    }
  }

  // Start Hubitat smart home listener (if configured)
  startHubitatListener();

  // #496 — stale-lock recovery. On startup, clear any
  // `follow_me_tasks.pending_run_at` older than the freshness window.
  // A pre-restart crash (or a `tessl_update`-killed run from before
  // this fix shipped) can leave dangling locks that block the next
  // scheduled fire on its Phase A gate. The host is a non-owner
  // reader of `follow_me_tasks` per
  // `coding-policy: stateful-artifacts`, so we don't migrate the
  // schema; we only NULL out value fields whose owners are demonstrably
  // dead. Threshold matches `PENDING_RUN_AT_FRESHNESS_MS` below — see
  // that constant's doc for the rationale.
  try {
    const cleared = clearStalePendingRunAt(PENDING_RUN_AT_FRESHNESS_MS);
    if (cleared.length > 0) {
      logger.warn(
        { tasks: cleared, ageThresholdMs: PENDING_RUN_AT_FRESHNESS_MS },
        'Cleared stale pending_run_at locks at startup (#496) — likely orphaned by a prior crash or a tessl_update mid-flight kill',
      );
    }
  } catch (err) {
    if (!(err instanceof Error)) throw err;
    logger.error(
      { err },
      'Startup pending_run_at cleanup failed — scheduled tasks with stale locks may refuse to fire on Phase A gate; investigate follow_me_tasks rows manually',
    );
  }

  // Periodic tile update from registry (every 15 min)
  // Heartbeat runs in the container and can't call tessl update.
  // This catches publishes that the post-promote timer missed.
  const { execFile: execTesslUpdate } = await import('child_process');
  setInterval(() => {
    // #496 — defer the periodic tessl_update if a scheduled task is
    // currently mid-flight (its skill has acquired a fresh
    // `follow_me_tasks.pending_run_at` lock). Running tessl_update now
    // would force-close the maintenance container via the
    // `closeAllActiveContainers` path and leave the lock dangling.
    // Skipping this tick is harmless — the next 15-min tick retries,
    // and the periodic catch-up isn't time-critical (the on-demand
    // `tessl_update` MCP tool is the load-bearing path for fresh
    // tile content; this loop is the safety net for missed
    // invocations).
    let activeLocks: string[];
    try {
      activeLocks = getActivePendingRunAtNames(PENDING_RUN_AT_FRESHNESS_MS);
    } catch (err) {
      if (!(err instanceof Error)) throw err;
      logger.warn(
        { err },
        'Periodic tessl update: pending_run_at check failed — proceeding with update (failing closed would skip every tick)',
      );
      activeLocks = [];
    }
    if (activeLocks.length > 0) {
      logger.info(
        { activeLocks },
        'Periodic tessl update deferred — scheduled task(s) mid-flight with fresh pending_run_at lock (#496); will retry on next 15-min tick',
      );
      return;
    }
    execTesslUpdate(
      'bash',
      [
        '-c',
        'cd /app/tessl-workspace && tessl update --yes --accept-warnings --agent claude-code 2>&1',
      ],
      { timeout: 120_000 },
      (err, stdout) => {
        if (err) {
          logger.warn({ error: err.message }, 'Periodic tessl update failed');
        } else if (stdout.includes('Updated')) {
          const cleared = deleteAllSessions();
          // Companion to the on-demand `tessl_update` IPC handler in
          // `ipc.ts`: any path that pulls new tile content into the
          // registry must also signal currently-running containers to
          // restart, otherwise they keep serving requests from the
          // skills/.tessl/ snapshot they copied at spawn time until
          // their 30-min idle timeout (issue #64).
          //
          // Wrapped because `closeAllActiveContainers()` rethrows
          // unexpected (non-fs) errors by contract — without this guard,
          // a programming bug surfacing through that path would propagate
          // out of the `setInterval` callback as an uncaught exception
          // and crash the orchestrator. Sessions stay cleared either way;
          // we degrade to "containers will pick up new tiles on idle
          // timeout" rather than taking the process down.
          let closed = 0;
          try {
            closed = queue.closeAllActiveContainers();
          } catch (closeErr) {
            // Narrow to Error instances (the only thing realistic
            // production code throws). Non-Error throws (a bare string,
            // `null`, etc.) are themselves a programming bug and
            // propagate as uncaught exceptions per error-handling.md
            // ("let unexpected propagate"). This catch is the
            // outer-boundary guard for an async callback — without it,
            // an Error from the close path would terminate the
            // orchestrator process; with it, sessions stay cleared and
            // we degrade to "containers refresh on idle timeout."
            if (!(closeErr instanceof Error)) throw closeErr;
            logger.error(
              { err: closeErr, sessionsCleared: cleared },
              'closeAllActiveContainers threw an unexpected error during periodic tessl update — sessions still cleared, but live containers will not respawn until idle timeout',
            );
          }
          logger.info(
            {
              sessionsCleared: cleared,
              containersClosed: closed,
              output: stdout.trim().slice(-200),
            },
            'Periodic tessl update found new tiles — sessions cleared and running containers signaled to restart',
          );
        }
      },
    );
  }, 900_000);

  // #542 — Heartbeat advisory walker. Re-walks the cached
  // `tz_state.segments` timeline every 30 min so a `from`/`to`
  // segment boundary crossing flips `current_tz` mid-day without
  // waiting for the next nightly `sync_tripit` run. Silent skip on
  // every path where there's nothing to do (no row, no segments,
  // computed tz matches current_tz, malformed cache); on flip,
  // notify the main group via its channel so the user sees the
  // change (matches the circuit-breaker notify pattern).
  const TZ_HEARTBEAT_INTERVAL_MS = 30 * 60 * 1000;
  setInterval(() => {
    let advisory: TzAdvisoryResult;
    try {
      // #584 — `onTzFlipped` invalidates cached `next_run` values on
      // active `schedule_timezone='local'` rows so a zone change at
      // heartbeat time doesn't leave 7am-local tasks anchored to the
      // prior zone. The recompute callback is wrapped inside the
      // writer with a narrowed catch — only transient SQLite
      // contention (`SQLITE_BUSY` / `SQLITE_LOCKED`) is logged and
      // continued so the canonical `tz_state` UPDATE that already
      // landed stays consistent. Programming bugs, persistent DB
      // failures, and unexpected throws propagate through the writer
      // back to this outer catch (which is itself narrowed on the same
      // transient SQLite codes) and finally to the orchestrator's
      // log-and-keep-ticking scheduler loop per
      // `coding-policy: error-handling`.
      advisory = runTzHeartbeatAdvisory(
        new Date(),
        ASSISTANT_OWNER_TG_USER_ID,
        () => {
          recomputeLocalSchedules(getCurrentTz, new Date());
        },
      );
    } catch (err) {
      // Narrowed to transient SQLite contention codes only —
      // SQLITE_BUSY / SQLITE_LOCKED can fire under WAL contention
      // with the orchestrator's own writers and are genuinely
      // recoverable on the next 30-min tick. Every other SqliteError
      // (SQLITE_CORRUPT, SQLITE_SCHEMA, SQLITE_READONLY, missing
      // column from a botched migration, etc.) signals a
      // persistent state-layer problem that hiding behind a periodic
      // warn would silently freeze `current_tz` indefinitely; those
      // propagate per `coding-policy: error-handling`. The
      // malformed-JSON case is handled inside
      // `runTzHeartbeatAdvisory` (narrowed `SyntaxError`) and returns
      // null rather than throwing, so the only thing that reaches
      // this catch is a real DB-side failure.
      const TRANSIENT_SQLITE_CODES = new Set(['SQLITE_BUSY', 'SQLITE_LOCKED']);
      if (
        !(err instanceof SqliteError) ||
        !TRANSIENT_SQLITE_CODES.has(err.code)
      ) {
        throw err;
      }
      logger.warn(
        { err: err.message, code: err.code },
        'tz heartbeat advisory: transient SqliteError on read — will retry on next 30-min tick',
      );
      return;
    }
    if (!advisory.flip && !advisory.warningToFire) return;
    const mainJid = Object.keys(registeredGroups).find(
      (jid) => registeredGroups[jid].isMain,
    );
    if (!mainJid) {
      // No main group registered — the flip + cooldown stamp already
      // landed on the DB row; the next time a main group is
      // registered, the user will see the right `current_tz` without
      // a separate notification. Skipping the chat send here is the
      // right failure mode (chat delivery would have nothing to
      // target).
      return;
    }
    const mainChannel = findChannel(channels, mainJid);
    if (!mainChannel || !mainChannel.isConnected()) return;

    // Chat notify: best-effort. The DB row already reflects the
    // flip / cooldown stamp, so a chat-send failure means "user
    // doesn't see the message this tick" — non-fatal. Each notify
    // gets its own `.catch` to convert rejection into a structured
    // warn instead of bubbling as an unhandled-rejection warning
    // that newer Node versions can terminate the orchestrator over.
    if (advisory.flip) {
      const flipForLog = advisory.flip;
      mainChannel
        .sendMessage(
          mainJid,
          `📍 Timezone changed: ${flipForLog.prev} → ${flipForLog.next}`,
        )
        .catch((err: unknown) => {
          if (!(err instanceof Error)) throw err;
          logger.warn(
            {
              err: err.message,
              mainJid,
              prev: flipForLog.prev,
              next: flipForLog.next,
            },
            'tz heartbeat advisory: chat notify failed (DB row already updated)',
          );
        });
    }
    if (advisory.warningToFire === 'stale_no_share') {
      // Cooldown is already enforced inside runTzHeartbeatAdvisory
      // (state-015 column `last_stale_warning_at`); reaching here
      // means a real fire is due, not a per-tick spam.
      mainChannel
        .sendMessage(
          mainJid,
          `⚠️ I haven't seen your location in 12 h+. If you're traveling, share your live location so TZ tracking stays accurate (#574 Phase 2).`,
        )
        .catch((err: unknown) => {
          if (!(err instanceof Error)) throw err;
          logger.warn(
            { err: err.message, mainJid },
            'tz heartbeat advisory: stale-location nag failed (DB cooldown stamp already updated; next 12 h tick will retry)',
          );
        });
    }
  }, TZ_HEARTBEAT_INTERVAL_MS).unref();

  startMessageLoop().catch((err) => {
    logger.fatal({ err }, 'Message loop crashed unexpectedly');
    process.exit(1);
  });
}

// Guard: only run when executed directly, not when imported by tests
const isDirectRun =
  process.argv[1] &&
  new URL(import.meta.url).pathname ===
    new URL(`file://${process.argv[1]}`).pathname;

if (isDirectRun) {
  main().catch((err) => {
    logger.error({ err }, 'Failed to start NanoClaw');
    process.exit(1);
  });
}
