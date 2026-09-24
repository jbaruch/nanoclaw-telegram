import { ChildProcess } from 'child_process';
import { CronExpressionParser } from 'cron-parser';
import Database from 'better-sqlite3';
import fs from 'fs';

import { ASSISTANT_NAME, SCHEDULER_POLL_INTERVAL, TIMEZONE } from './config.js';
import {
  ContainerOutput,
  runContainerAgent,
  writeTasksSnapshot,
} from './container-runner.js';
import {
  getAllTasks,
  getDueTasks,
  getTaskById,
  logTaskRun,
  updateTask,
  updateTaskAfterRun,
} from './db.js';
import { GroupQueue } from './group-queue.js';
import {
  InvalidGroupFolderError,
  resolveGroupFolderPath,
} from './group-folder.js';
import { logger } from './logger.js';
import {
  isExecFailure,
  isFileSystemError,
  isSpawnError,
  MessageDeliveryError,
} from './operational-errors.js';
import { RegisteredGroup, ScheduledTask } from './types.js';

/**
 * Compute the next run time for a recurring task, anchored to the
 * task's scheduled time rather than Date.now() to prevent cumulative
 * drift on interval-based tasks.
 *
 * Co-authored-by: @community-pr-601
 */
export function computeNextRun(task: ScheduledTask): string | null {
  if (task.schedule_type === 'once') return null;

  const now = Date.now();

  if (task.schedule_type === 'cron') {
    const interval = CronExpressionParser.parse(task.schedule_value, {
      tz: TIMEZONE,
    });
    return interval.next().toISOString();
  }

  if (task.schedule_type === 'interval') {
    const ms = parseInt(task.schedule_value, 10);
    if (!ms || ms <= 0) {
      // Guard against malformed interval that would cause an infinite loop
      logger.warn(
        { taskId: task.id, value: task.schedule_value },
        'Invalid interval value',
      );
      return new Date(now + 60_000).toISOString();
    }
    // Anchor to the scheduled time, not now, to prevent drift.
    // Skip past any missed intervals so we always land in the future.
    let next = new Date(task.next_run!).getTime() + ms;
    while (next <= now) {
      next += ms;
    }
    return new Date(next).toISOString();
  }

  return null;
}

export interface SchedulerDependencies {
  registeredGroups: () => Record<string, RegisteredGroup>;
  getSessions: () => Record<string, string>;
  queue: GroupQueue;
  onProcess: (
    groupJid: string,
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
  let result: string | null = null;
  let error: string | null = null;
  let keepPausedSchedule = false;
  let finalized = false;

  // After the task produces a result, close the container promptly.
  // Tasks are single-turn — no need to wait IDLE_TIMEOUT (30 min) for the
  // query loop to time out. A short delay handles any final MCP calls.
  const TASK_CLOSE_DELAY_MS = 10000;
  let closeTimer: ReturnType<typeof setTimeout> | null = null;

  const scheduleClose = () => {
    if (closeTimer) return; // already scheduled
    closeTimer = setTimeout(() => {
      logger.debug({ taskId: task.id }, 'Closing task container after result');
      deps.queue.closeStdin(task.chat_jid);
    }, TASK_CLOSE_DELAY_MS);
  };

  const finalizeRun = () => {
    if (finalized) return;
    finalized = true;
    if (closeTimer) clearTimeout(closeTimer);

    logTaskRun({
      task_id: task.id,
      run_at: new Date().toISOString(),
      duration_ms: Date.now() - startTime,
      status: error ? 'error' : 'success',
      result,
      error,
    });

    const nextRun = keepPausedSchedule ? task.next_run : computeNextRun(task);
    const resultSummary = error
      ? `Error: ${error}`
      : result
        ? result.slice(0, 200)
        : 'Completed';
    updateTaskAfterRun(task.id, nextRun, resultSummary);
  };

  try {
    let groupDir: string;
    try {
      groupDir = resolveGroupFolderPath(task.group_folder);
    } catch (err) {
      if (!(err instanceof InvalidGroupFolderError)) throw err;
      error = err.message;
      keepPausedSchedule = true;
      // Stop retry churn for malformed legacy rows.
      updateTask(task.id, { status: 'paused' });
      logger.error(
        { taskId: task.id, groupFolder: task.group_folder, error },
        'Task has invalid group folder',
      );
      return;
    }
    fs.mkdirSync(groupDir, { recursive: true });

    logger.info(
      { taskId: task.id, group: task.group_folder },
      'Running scheduled task',
    );

    const groups = deps.registeredGroups();
    const group = Object.values(groups).find(
      (candidate) => candidate.folder === task.group_folder,
    );

    if (!group) {
      error = `Group not found: ${task.group_folder}`;
      logger.error(
        { taskId: task.id, groupFolder: task.group_folder },
        'Group not found for task',
      );
      return;
    }

    // Update tasks snapshot for container to read (filtered by group)
    const isMain = group.isMain === true;
    const tasks = getAllTasks();
    writeTasksSnapshot(
      task.group_folder,
      isMain,
      tasks.map((scheduledTask) => ({
        id: scheduledTask.id,
        groupFolder: scheduledTask.group_folder,
        prompt: scheduledTask.prompt,
        script: scheduledTask.script,
        schedule_type: scheduledTask.schedule_type,
        schedule_value: scheduledTask.schedule_value,
        status: scheduledTask.status,
        next_run: scheduledTask.next_run,
      })),
    );

    // For group context mode, use the group's current session
    const sessions = deps.getSessions();
    const sessionId =
      task.context_mode === 'group' ? sessions[task.group_folder] : undefined;

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
      },
      (proc, containerName) =>
        deps.onProcess(task.chat_jid, proc, containerName, task.group_folder),
      async (streamedOutput: ContainerOutput) => {
        if (streamedOutput.result) {
          result = streamedOutput.result;
          // Forward result to user (sendMessage handles formatting)
          await deps.sendMessage(task.chat_jid, streamedOutput.result);
          scheduleClose();
        }
        if (streamedOutput.status === 'success') {
          deps.queue.notifyIdle(task.chat_jid);
          scheduleClose(); // Close promptly even when result is null (e.g. IPC-only tasks)
        }
        if (streamedOutput.status === 'error') {
          error = streamedOutput.error || 'Unknown error';
        }
      },
    );

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
    const isExpected =
      err instanceof MessageDeliveryError ||
      isFileSystemError(err) ||
      isSpawnError(err) ||
      isExecFailure(err);
    error = err instanceof Error ? err.message : String(err);
    if (!isExpected) {
      logger.error({ taskId: task.id, error }, 'Task failed');
      finalizeRun();
      throw err;
    }
    logger.error({ taskId: task.id, error }, 'Task failed');
  } finally {
    finalizeRun();
  }
}

let schedulerRunning = false;

export async function startSchedulerLoop(
  deps: SchedulerDependencies,
): Promise<void> {
  if (schedulerRunning) {
    logger.debug('Scheduler loop already running, skipping duplicate start');
    return;
  }
  schedulerRunning = true;
  logger.info('Scheduler loop started');

  while (schedulerRunning) {
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

        deps.queue.enqueueTask(currentTask.chat_jid, currentTask.id, () =>
          runTask(currentTask, deps),
        );
      }
    } catch (err) {
      if (!(err instanceof Database.SqliteError) && !isFileSystemError(err)) {
        throw err;
      }
      logger.error({ err }, 'Error in scheduler loop');
    }

    await new Promise((resolve) =>
      setTimeout(resolve, SCHEDULER_POLL_INTERVAL),
    );
  }
}

/** @internal - for tests only. */
export function _resetSchedulerLoopForTests(): void {
  schedulerRunning = false;
}
