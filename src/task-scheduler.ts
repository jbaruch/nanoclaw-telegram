import { ChildProcess } from 'child_process';
import { CronExpressionParser } from 'cron-parser';
import fs from 'fs';

import { ASSISTANT_NAME, SCHEDULER_POLL_INTERVAL, TIMEZONE } from './config.js';
import {
  ContainerOutput,
  runContainerAgent,
  writeTasksSnapshot,
} from './container-runner.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';
import {
  getAllTasks,
  getDueTasks,
  getTaskById,
  logTaskRun,
  setSession,
  storeChatMetadata,
  storeMessage,
  updateTask,
  updateTaskAfterRun,
} from './db.js';
import { GroupQueue } from './group-queue.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { logger } from './logger.js';
import { RegisteredGroup, ScheduledTask } from './types.js';

/**
 * Compute the next run time for a recurring task, anchored to the
 * task's scheduled time rather than Date.now() to prevent cumulative
 * drift on interval-based tasks.
 *
 * Co-authored-by: @community-pr-601
 */
/**
 * Result type that lets callers know WHY a recurring task got
 * `nextRun: null` so they can apply remediation against the FRESH DB
 * row (avoiding races against concurrent `update_task` IPC).
 *
 * The legacy `string | null` shape is preserved by `computeNextRun`
 * for backwards compat — call `computeNextRunDetailed` to get the
 * structured result.
 */
export type NextRunRemediation =
  | 'pause-broken-cron' // both per-task tz and TIMEZONE retry failed
  | 'clear-bad-timezone'; // per-task tz failed, TIMEZONE retry succeeded

export interface NextRunResult {
  nextRun: string | null;
  remediation?: NextRunRemediation;
}

export function computeNextRunDetailed(task: ScheduledTask): NextRunResult {
  if (task.schedule_type === 'once') return { nextRun: null };

  const now = Date.now();

  if (task.schedule_type === 'cron') {
    // Per-task `schedule_timezone` (#102) takes precedence over the
    // server-wide TIMEZONE config. NULL/undefined falls back to TIMEZONE
    // — the pre-#102 behavior.
    //
    // Pure function (no DB writes): a previous version called
    // `updateTask` directly here, which raced with concurrent
    // `update_task` IPC — a user fixing a broken tz could have their
    // change clobbered by a still-in-flight scheduler tick that read
    // the old value. Now we just compute and report; the caller is
    // responsible for pausing or clearing the tz against the FRESH
    // DB row.
    try {
      const interval = CronExpressionParser.parse(task.schedule_value, {
        tz: task.schedule_timezone || TIMEZONE,
      });
      return { nextRun: interval.next().toISOString() };
    } catch (err) {
      logger.warn(
        {
          taskId: task.id,
          scheduleValue: task.schedule_value,
          scheduleTimezone: task.schedule_timezone,
          err: err instanceof Error ? err.message : String(err),
        },
        'computeNextRun: cron parse failed — retrying with server TIMEZONE',
      );
      try {
        const interval = CronExpressionParser.parse(task.schedule_value, {
          tz: TIMEZONE,
        });
        return {
          nextRun: interval.next().toISOString(),
          remediation: 'clear-bad-timezone',
        };
      } catch (retryErr) {
        logger.error(
          {
            taskId: task.id,
            scheduleValue: task.schedule_value,
            err:
              retryErr instanceof Error ? retryErr.message : String(retryErr),
          },
          'computeNextRun: cron parse failed even with TIMEZONE fallback',
        );
        return { nextRun: null, remediation: 'pause-broken-cron' };
      }
    }
  }

  if (task.schedule_type === 'interval') {
    const ms = parseInt(task.schedule_value, 10);
    if (!ms || ms <= 0) {
      // Guard against malformed interval that would cause an infinite loop
      logger.warn(
        { taskId: task.id, value: task.schedule_value },
        'Invalid interval value',
      );
      return { nextRun: new Date(now + 60_000).toISOString() };
    }
    // Anchor to the scheduled time, not now, to prevent drift.
    // Skip past any missed intervals so we always land in the future.
    let next = new Date(task.next_run!).getTime() + ms;
    while (next <= now) {
      next += ms;
    }
    return { nextRun: new Date(next).toISOString() };
  }

  return { nextRun: null };
}

/**
 * Backwards-compat shim: `computeNextRun` retains its original
 * `string | null` shape so existing callers that don't care about
 * remediation hints continue to work. Internally delegates to
 * `computeNextRunDetailed` and discards the remediation field —
 * callers that DO need to act on remediation should call the
 * detailed variant directly and apply the remediation against the
 * fresh DB row (re-fetch via `getTaskById`) to avoid clobbering
 * concurrent IPC updates.
 */
export function computeNextRun(task: ScheduledTask): string | null {
  return computeNextRunDetailed(task).nextRun;
}

/**
 * Apply the remediation hint produced by `computeNextRunDetailed`
 * against the FRESH state of the task (re-fetched from DB). If the
 * task changed since the compute step (e.g. a concurrent
 * `update_task` fixed the cron expression or timezone), we skip the
 * remediation — the caller's fix wins.
 */
export function applyComputeNextRunRemediation(
  taskId: string,
  remediation: NextRunRemediation,
  observedScheduleValue: string,
  observedScheduleTimezone: string | null | undefined,
): void {
  const fresh = getTaskById(taskId);
  if (!fresh) return;
  // If the user updated the task between compute and now, the values
  // we'd be remediating against are no longer the source of the
  // failure. Skip — let the next scheduler tick re-evaluate.
  if (
    fresh.schedule_value !== observedScheduleValue ||
    (fresh.schedule_timezone ?? null) !== (observedScheduleTimezone ?? null)
  ) {
    logger.info(
      { taskId, remediation },
      'applyComputeNextRunRemediation: task changed since compute — skipping',
    );
    return;
  }
  if (remediation === 'pause-broken-cron') {
    updateTask(taskId, { status: 'paused' });
    logger.warn(
      { taskId },
      'Paused task — cron expression unparseable with both per-task tz and server TIMEZONE',
    );
  } else if (remediation === 'clear-bad-timezone') {
    updateTask(taskId, { schedule_timezone: null });
    logger.warn(
      { taskId, droppedTimezone: observedScheduleTimezone },
      'Dropped invalid schedule_timezone — falling back to TIMEZONE going forward',
    );
  }
}

export interface SchedulerDependencies {
  registeredGroups: () => Record<string, RegisteredGroup>;
  /**
   * Nested session cache: `folder → sessionName → sessionId`.
   * Scheduled tasks look up the MAINTENANCE slot's sessionId here so
   * consecutive heartbeat/nightly runs resume their own prior session
   * chain, not the user-facing default container's.
   */
  getSessions: () => Record<string, Record<string, string>>;
  queue: GroupQueue;
  onProcess: (
    groupJid: string,
    sessionName: string,
    proc: ChildProcess,
    containerName: string,
    groupFolder: string,
  ) => void;
  sendMessage: (jid: string, text: string) => Promise<void>;
}

async function runTask(
  task: ScheduledTask,
  deps: SchedulerDependencies,
): Promise<void> {
  const startTime = Date.now();
  let groupDir: string;
  try {
    groupDir = resolveGroupFolderPath(task.group_folder);
  } catch (err) {
    // resolveGroupFolderPath throws Error on path-validation failure.
    // Anything else is a bug elsewhere; propagate per
    // `jbaruch/coding-policy: error-handling`.
    if (!(err instanceof Error)) throw err;
    const error = err.message;
    // Stop retry churn for malformed legacy rows.
    updateTask(task.id, { status: 'paused' });
    logger.error(
      { taskId: task.id, groupFolder: task.group_folder, error },
      'Task has invalid group folder',
    );
    logTaskRun({
      task_id: task.id,
      run_at: new Date().toISOString(),
      duration_ms: Date.now() - startTime,
      status: 'error',
      result: null,
      error,
    });
    return;
  }
  fs.mkdirSync(groupDir, { recursive: true });

  logger.info(
    { taskId: task.id, group: task.group_folder },
    'Running scheduled task',
  );

  const groups = deps.registeredGroups();
  const group = Object.values(groups).find(
    (g) => g.folder === task.group_folder,
  );

  if (!group) {
    logger.error(
      { taskId: task.id, groupFolder: task.group_folder },
      'Group not found for task',
    );
    logTaskRun({
      task_id: task.id,
      run_at: new Date().toISOString(),
      duration_ms: Date.now() - startTime,
      status: 'error',
      result: null,
      error: `Group not found: ${task.group_folder}`,
    });
    return;
  }

  // Update tasks snapshot for container to read (filtered by group)
  const isMain = group.isMain === true;
  const tasks = getAllTasks();
  writeTasksSnapshot(
    task.group_folder,
    isMain,
    tasks.map((t) => ({
      id: t.id,
      groupFolder: t.group_folder,
      prompt: t.prompt,
      script: t.script,
      schedule_type: t.schedule_type,
      schedule_value: t.schedule_value,
      status: t.status,
      next_run: t.next_run,
    })),
    !!group.containerConfig?.trusted,
  );

  let result: string | null = null;
  let error: string | null = null;

  // Scheduled tasks resume THEIR OWN session chain from the `maintenance`
  // slot. The sessions map is keyed by `(groupFolder, sessionName)` —
  // maintenance has its own per-session `.claude/` mount, so its
  // sessionIds are stored and resumed separately from the user-facing
  // default container. `context_mode: 'isolated'` starts fresh each run.
  const sessions = deps.getSessions();
  const sessionId =
    task.context_mode === 'group'
      ? sessions[task.group_folder]?.[MAINTENANCE_SESSION_NAME]
      : undefined;

  // After the task produces a result, close the container promptly.
  // Tasks are single-turn — no need to wait IDLE_TIMEOUT (30 min) for the
  // query loop to time out. A short delay handles any final MCP calls.
  // The kill grace after close sentinel is handled by GroupQueue.closeStdin().
  const TASK_CLOSE_DELAY_MS = 10000;
  let closeTimer: ReturnType<typeof setTimeout> | null = null;

  const scheduleClose = () => {
    if (closeTimer) return; // already scheduled
    closeTimer = setTimeout(() => {
      logger.debug({ taskId: task.id }, 'Closing task container after result');
      deps.queue.closeStdin(task.chat_jid, MAINTENANCE_SESSION_NAME);
    }, TASK_CLOSE_DELAY_MS);
  };

  try {
    const output = await runContainerAgent(
      group,
      {
        prompt: task.prompt,
        sessionId,
        groupFolder: task.group_folder,
        chatJid: task.chat_jid,
        isMain,
        isScheduledTask: true,
        assistantName: ASSISTANT_NAME,
        script: task.script || undefined,
        // Provenance: the role that created this task, so the agent-runner
        // can decide whether to wrap the prompt in <untrusted-input>. Only
        // 'untrusted_agent'-created tasks get wrapped; owner/main/trusted
        // bypass. See ContainerInput.createdByRole docs.
        createdByRole: task.created_by_role,
        // Route every scheduled task into the parallel `maintenance` slot so
        // it runs concurrently with user-facing work. Sole writer of this
        // value — inbound paths route to `'default'` instead.
        sessionName: MAINTENANCE_SESSION_NAME,
      },
      (proc, containerName) =>
        deps.onProcess(
          task.chat_jid,
          MAINTENANCE_SESSION_NAME,
          proc,
          containerName,
          task.group_folder,
        ),
      async (streamedOutput: ContainerOutput) => {
        // Persist the maintenance session's own sessionId so the NEXT
        // scheduled task on this group can resume the same chain. Only
        // for `context_mode: 'group'` tasks — an isolated task wants a
        // fresh SDK session and its newSessionId would otherwise overwrite
        // the slot and contaminate the next 'group' task's resume.
        if (streamedOutput.newSessionId && task.context_mode === 'group') {
          const groupSessions =
            sessions[task.group_folder] ?? (sessions[task.group_folder] = {});
          groupSessions[MAINTENANCE_SESSION_NAME] = streamedOutput.newSessionId;
          setSession(
            task.group_folder,
            MAINTENANCE_SESSION_NAME,
            streamedOutput.newSessionId,
          );
        }
        if (streamedOutput.result) {
          result = streamedOutput.result;
          // Strip <internal> tags — suppress entirely if nothing remains
          const cleanResult = streamedOutput.result
            .replace(/<internal>[\s\S]*?<\/internal>/g, '')
            .trim();
          if (cleanResult) {
            await deps.sendMessage(task.chat_jid, cleanResult);
            // Store the bot send so `messages.db` reflects every send
            // out of this session. Without this, scheduled-task sends
            // (heartbeat, housekeeping, morning-brief, etc.) reach
            // Telegram but leave no DB row — the "ghost heartbeat" /
            // "no trace in messages.db" class of jbaruch/nanoclaw#81.
            // The IPC-path `send_message` handler in src/ipc.ts writes
            // the same shape; this mirrors it so heartbeat's answered-
            // check accounting and forensic greps both see the row.
            //
            // Upsert chat metadata first so the `messages.chat_jid →
            // chats.jid` FK doesn't reject the insert on a chat that
            // has no prior metadata (task fires before any user
            // message, or chat was manually registered without the
            // normal group-sync write-through). Idempotent: existing
            // rows keep their `name` because we pass `name` as
            // undefined and `storeChatMetadata` omits `name` from the
            // UPDATE in that branch (not COALESCE); `channel` and
            // `is_group` are preserved via COALESCE when we pass
            // undefined for them. `last_message_time` advances to the
            // outgoing send's timestamp, same as the IPC path would
            // effectively do by chaining a chat-metadata update.
            //
            // Pass inferred `channel` + `isGroup` so a NEW chat row
            // (first-ever metadata write) has the right shape for
            // `getAvailableGroups()`, which filters on `is_group`.
            // Match the channel-name convention the codebase already
            // uses everywhere else (`'telegram'`, `'whatsapp'`) — NOT
            // the JID prefix abbreviation. JID shapes in this repo:
            //   - `tg:<id>` — Telegram. Negative id = group/channel,
            //     positive = private 1:1.
            //   - `<id>@g.us` — WhatsApp group (no `wa:` prefix).
            //   - `<id>@s.whatsapp.net` — WhatsApp DM.
            // Matches the conventions `db.ts`'s legacy-chat backfill
            // uses (`@g.us` → group, `@s.whatsapp.net` → DM).
            // Anything else: leave both undefined so COALESCE in
            // storeChatMetadata preserves existing values rather than
            // writing NULL or an abbreviated channel string.
            const sendTimestamp = new Date().toISOString();
            let inferredChannel: string | undefined;
            let inferredIsGroup: boolean | undefined;
            if (task.chat_jid.startsWith('tg:')) {
              inferredChannel = 'telegram';
              inferredIsGroup = task.chat_jid.startsWith('tg:-');
            } else if (task.chat_jid.endsWith('@g.us')) {
              inferredChannel = 'whatsapp';
              inferredIsGroup = true;
            } else if (task.chat_jid.endsWith('@s.whatsapp.net')) {
              inferredChannel = 'whatsapp';
              inferredIsGroup = false;
            }
            // Wrap the DB writes so a SQLite error (FK constraint,
            // disk full, schema mid-migration) never rejects the
            // `onOutput` promise. The streaming output chain in
            // `container-runner.ts` awaits this via `.then(...)` with
            // no `.catch(...)`, so a throw here can wedge the run
            // from ever resolving and stall the scheduler loop. The
            // send already succeeded; a missing DB row is recoverable
            // (at worst we'd get a duplicate in `unanswered` on the
            // next cycle) — stalling the scheduler is not.
            try {
              storeChatMetadata(
                task.chat_jid,
                sendTimestamp,
                undefined,
                inferredChannel,
                inferredIsGroup,
              );
              storeMessage({
                id: `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                chat_jid: task.chat_jid,
                sender: ASSISTANT_NAME,
                sender_name: ASSISTANT_NAME,
                content: cleanResult,
                timestamp: sendTimestamp,
                is_from_me: true,
                is_bot_message: true,
              });
            } catch (dbErr) {
              logger.error(
                {
                  taskId: task.id,
                  chatJid: task.chat_jid,
                  err: dbErr,
                  preview: cleanResult.slice(0, 200),
                },
                '[task-scheduler] storeChatMetadata/storeMessage failed after send — continuing, send already landed in Telegram',
              );
            }
          }
          // Don't close here — agent may still be polling for host script results.
          // Close only on final 'success' status below.
        }
        if (streamedOutput.status === 'success') {
          // No `notifyIdle` here — `notifyIdle` targets the `default` slot
          // only, so calling it from a maintenance-routed task would flip
          // the wrong container's state and could preempt active user work.
          // `scheduleClose` already winds this container down; when runTask
          // finishes, `drainGroup` chains any pending maintenance task.
          scheduleClose();
        }
        if (streamedOutput.status === 'error') {
          error = streamedOutput.error || 'Unknown error';
        }
      },
    );

    if (closeTimer) clearTimeout(closeTimer);

    // Same write-back path for the terminal `output` (non-streaming case).
    // Same `'group'`-only gate as the streaming path above — don't let an
    // isolated task overwrite the maintenance slot's session chain.
    if (output.newSessionId && task.context_mode === 'group') {
      const groupSessions =
        sessions[task.group_folder] ?? (sessions[task.group_folder] = {});
      groupSessions[MAINTENANCE_SESSION_NAME] = output.newSessionId;
      setSession(
        task.group_folder,
        MAINTENANCE_SESSION_NAME,
        output.newSessionId,
      );
    }

    if (output.status === 'error') {
      error = output.error || 'Unknown error';
    } else if (output.result) {
      // Result was already forwarded to the user via the streaming callback above
      result = output.result;
    }

    logger.info(
      { taskId: task.id, durationMs: Date.now() - startTime },
      'Task completed',
    );
  } catch (err) {
    if (closeTimer) clearTimeout(closeTimer);
    // Per `jbaruch/coding-policy: error-handling`: non-Error throws
    // indicate bugs upstream and should propagate. The scheduler loop
    // (Step 2 of `loop` below) is the last-resort safety net that
    // catches them, logs, and keeps ticking — so re-throwing here
    // doesn't kill the orchestrator.
    if (!(err instanceof Error)) throw err;
    error = err.message;
    logger.error({ taskId: task.id, error }, 'Task failed');
  }

  const durationMs = Date.now() - startTime;

  logTaskRun({
    task_id: task.id,
    run_at: new Date().toISOString(),
    duration_ms: durationMs,
    status: error ? 'error' : 'success',
    result,
    error,
  });

  // Re-fetch the task to compute next_run against the FRESH schedule
  // fields. The captured `task` is from before dispatch — between
  // there and here a user can have called `update_task` to change
  // `schedule_value`, `schedule_timezone`, or `schedule_type`, and
  // their fix shouldn't be clobbered by a write-back computed from
  // the stale capture (the same race `applyComputeNextRunRemediation`
  // already guards against on the remediation path).
  const fresh = getTaskById(task.id) ?? task;
  const computed = computeNextRunDetailed(fresh);
  if (computed.remediation) {
    applyComputeNextRunRemediation(
      fresh.id,
      computed.remediation,
      fresh.schedule_value,
      fresh.schedule_timezone,
    );
  }
  const resultSummary = error
    ? `Error: ${error}`
    : result
      ? result.slice(0, 200)
      : 'Completed';
  updateTaskAfterRun(fresh.id, computed.nextRun, resultSummary);
}

let schedulerRunning = false;

export function startSchedulerLoop(deps: SchedulerDependencies): void {
  if (schedulerRunning) {
    logger.debug('Scheduler loop already running, skipping duplicate start');
    return;
  }
  schedulerRunning = true;
  logger.info('Scheduler loop started');

  const loop = async () => {
    try {
      const dueTasks = getDueTasks();
      if (dueTasks.length > 0) {
        logger.info({ count: dueTasks.length }, 'Found due tasks');
      }

      for (const task of dueTasks) {
        // Re-check task status in case it was paused/cancelled
        const currentTask = getTaskById(task.id);
        if (!currentTask || currentTask.status !== 'active') {
          continue;
        }

        // Pre-advance next_run before dispatch to prevent double-fire on crash.
        const computed = computeNextRunDetailed(currentTask);
        if (computed.remediation) {
          // Apply remediation against the FRESH DB row — if a user
          // raced an `update_task` IPC between the read above and
          // here that fixed the broken cron/tz, the helper detects
          // the mismatch and skips, letting the user's fix stand.
          applyComputeNextRunRemediation(
            currentTask.id,
            computed.remediation,
            currentTask.schedule_value,
            currentTask.schedule_timezone,
          );
        }
        if (computed.nextRun !== null) {
          updateTask(currentTask.id, { next_run: computed.nextRun });
        } else if (currentTask.schedule_type === 'once') {
          // Genuine once-task completion — pre-mark as completed.
          updateTask(currentTask.id, { status: 'completed' });
        }
        // else: cron/interval with nextRun=null means
        // `computeNextRunDetailed` returned a `pause-broken-cron`
        // remediation that the apply step above already handled.
        // Do NOT flip to completed — that would lose the paused
        // state set by the remediation. See #102 round-4 review.

        deps.queue.enqueueTask(
          currentTask.chat_jid,
          currentTask.id,
          MAINTENANCE_SESSION_NAME,
          () => runTask(currentTask, deps),
        );
      }
    } catch (err) {
      // Terminal safety net for the scheduler loop. Inner code paths
      // re-throw non-Error per `jbaruch/coding-policy: error-handling`;
      // this catch is where they finally land. Re-throwing further
      // here would crash the loop and stop every scheduled task — the
      // explicit design choice is "log and keep ticking" so a single
      // bug in one task can't take the orchestrator's whole scheduler
      // down. Distinguishes Error from non-Error in the log so the
      // bug source is identifiable downstream.
      if (err instanceof Error) {
        logger.error({ err }, 'Scheduler loop caught Error');
      } else {
        logger.error(
          { err: String(err) },
          'Scheduler loop caught non-Error throw — fix the upstream call site',
        );
      }
    }

    setTimeout(loop, SCHEDULER_POLL_INTERVAL);
  };

  loop();
}

/** @internal - for tests only. */
export function _resetSchedulerLoopForTests(): void {
  schedulerRunning = false;
}
