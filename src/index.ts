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
import { writeGroupsSnapshot } from './container-runner.js';
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
import { initDatabase } from './db.js';
import {
  getMessagesSince,
  storeChatMetadata,
  storeMessage,
} from './db-messages.js';
import { deleteAllSessions } from './db-sessions.js';
import {
  clearStalePendingRunAt,
  getActivePendingRunAtNames,
  getCurrentTz,
  runTzHeartbeatAdvisory,
  TzAdvisoryResult,
  storeLocation,
} from './db-tz.js';
import { DEFAULT_SESSION_NAME } from './group-queue.js';
import { BEST_EFFORT_FS_CODES, isFsErrorWithCode } from './fs-errors.js';
import { runLocationSinks } from './location-sinks.js';
import { initBotPool } from './channels/telegram.js';
import { startIpcWatcher } from './ipc.js';
import { findChannel, formatOutbound } from './router.js';
import { ChannelType } from './text-styles.js';
import {
  restoreRemoteControl,
  startRemoteControl,
  stopRemoteControl,
} from './remote-control.js';
import { runShutdownHooks, runStartupHooks } from './host-lifecycle.js';
import { pruneOldContainerLogs } from './host-logs.js';
import { registerHostPlugins } from './host-plugins/index.js';
import { startSessionCleanup } from './session-cleanup.js';
import {
  recomputeLocalSchedules,
  startSchedulerLoop,
} from './task-scheduler.js';
import { startTriggerLearner } from './gates/trigger-learner-runtime.js';
import { installTelegramOutboundTap } from './telegram-outbound-tap.js';
import { LocationRecord, NewMessage } from './types.js';
import { logger } from './logger.js';
import { initObserver } from './observer.js';
import {
  getOrRecoverCursor,
  loadState,
  registeredGroups,
  sessions,
} from './orchestrator-state.js';
import { channels, queue } from './orchestrator-runtime.js';
import {
  getAvailableGroups,
  processGroupMessages,
  startMessageLoop,
} from './message-pipeline.js';
import { checkSilentZero } from './usage-log.js';
import { wipeSessionJsonl } from './session-wipe.js';
import {
  cleanupOrphanNonMainHeartbeats,
  logRegisteredGroupOrphans,
} from './group-registry.js';
import { ipcDeps } from './ipc-deps.js';

// #496 — `follow_me_tasks.pending_run_at` is treated as "fresh" (and
// therefore worth respecting) for this long after stamp-time. Anything
// older is presumed orphaned by a dead/killed container and gets
// reclaimed via `clearStalePendingRunAt`.
//
// Sized for the largest plausible follow-me task duration with margin:
// the longest scheduled skills in this fleet cap around 10–15 min of
// agent work; 1h gives 4× margin for an unusually slow
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

    // Optional-integration teardown (#847): registered shutdown hooks
    // (Hubitat listener, future host plugins) run first, each isolated
    // so one failure can't skip the platform teardown below.
    await runShutdownHooks();
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

  // Register host-plugin modules (#846/#847/#849) BEFORE channels
  // connect: location sinks must be in the registry when the first
  // inbound location arrives, and spawn gates before the scheduler's
  // first fire below.
  registerHostPlugins();

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
      // Core owns persistence: the DB row (host-side TZ resolver input)
      // is written first, unconditionally. Everything else a location
      // feeds — e.g. the travel tile's current-location.json artifact —
      // is a registered sink (#849), fanned out after the write.
      storeLocation(record);
      runLocationSinks(record);
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
  startIpcWatcher(ipcDeps);
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

  // Optional-integration startup (#847): run the startup hooks host
  // plugins registered in `registerHostPlugins()` above (Hubitat
  // EventSocket listener when configured, future optional listeners).
  // Each hook is isolated — a failing optional integration logs and
  // never takes down platform startup.
  await runStartupHooks();

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
