import {
  isFileSystemError,
  isRecoverableSqliteError,
  isOperationalRunError,
} from './operational-errors.js';
import { ChildProcess } from 'child_process';
import { parseTaskSchedule } from './task-schedule.js';
import fs from 'fs';

import { ASSISTANT_NAME, SCHEDULER_POLL_INTERVAL } from './config.js';
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
import { stopContainer } from './container-runtime.js';
import {
  InvalidGroupFolderError,
  resolveGroupFolderPath,
} from './group-folder.js';
import { logger } from './logger.js';
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
  const schedule = parseTaskSchedule(
    task.schedule_type,
    task.schedule_value,
    now,
  );
  if (schedule.type === 'once') return null;
  if (schedule.type === 'cron') return schedule.nextRun;
  const anchor =
    task.next_run === null ? NaN : new Date(task.next_run).getTime();
  if (!Number.isFinite(anchor))
    throw new SyntaxError(
      'Stored task next_run is invalid; pause and repair the schedule.',
    );
  const steps = Math.max(
    1,
    Math.floor((now - anchor) / schedule.intervalMs) + 1,
  );
  const next = new Date(anchor + steps * schedule.intervalMs);
  if (!Number.isFinite(next.getTime()))
    throw new SyntaxError(
      'Stored task interval exceeds the supported date range.',
    );
  return next.toISOString();
}

export interface SchedulerDependencies {
  registeredGroups: () => Record<string, RegisteredGroup>;
  getSessions: () => Record<string, string>;
  queue: Pick<GroupQueue, 'enqueueTask' | 'closeStdin' | 'notifyIdle'>;
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
  let closeTimer: ReturnType<typeof setTimeout> | null = null;
  let ownedContainer: string | undefined;
  let closeFailure: { error: unknown } | undefined;
  let closeCallbacks: Promise<void> = Promise.resolve();
  let rejectClose!: (error: unknown) => void;
  const closeOutcome = new Promise<never>((_resolve, reject) => {
    rejectClose = reject;
  });
  const [execution] = await Promise.allSettled([
    (async () => {
      try {
        if (
          !task.next_run ||
          !Number.isFinite(new Date(task.next_run).getTime())
        )
          throw new SyntaxError('Stored task next_run is invalid.');
        if (task.schedule_type === 'once') {
          if (
            typeof task.schedule_value !== 'string' ||
            !task.schedule_value.trim() ||
            !Number.isFinite(new Date(task.schedule_value).getTime()) ||
            new Date(task.next_run).toISOString() !== task.next_run
          )
            throw new SyntaxError(
              'Stored once task authority is invalid; preserve its captured UTC instant for repair.',
            );
          // Legacy creation already captured an absolute instant (including old
          // minute precision and DST normalization). Use it only in memory;
          // never rewrite historical schedule text or guess the wall time again.
          task = { ...task, schedule_value: task.next_run };
        }
        parseTaskSchedule(task.schedule_type, task.schedule_value);
      } catch (failure) {
        if (!(failure instanceof SyntaxError)) throw failure;
        updateTask(task.id, { status: 'paused' });
        logTaskRun({
          task_id: task.id,
          run_at: new Date().toISOString(),
          duration_ms: Date.now() - startTime,
          status: 'error',
          result: null,
          error: failure.message,
        });
        logger.error(
          { taskId: task.id, error: failure },
          'Invalid stored task schedule; paused until repaired',
        );
        return true;
      }
      let groupDir: string;
      try {
        groupDir = resolveGroupFolderPath(task.group_folder);
      } catch (err) {
        if (!(err instanceof InvalidGroupFolderError)) throw err;
        const error = err instanceof Error ? err.message : String(err);
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
        return true;
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
        error = `Group not found: ${task.group_folder}`;
        return false;
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
      );

      // For group context mode, use the group's current session
      const sessions = deps.getSessions();
      const sessionId =
        task.context_mode === 'group' ? sessions[task.group_folder] : undefined;

      // After the task produces a result, close the container promptly.
      // Tasks are single-turn — no need to wait IDLE_TIMEOUT (30 min) for the
      // query loop to time out. A short delay handles any final MCP calls.
      const TASK_CLOSE_DELAY_MS = 10000;

      const scheduleClose = () => {
        if (closeTimer) return; // already scheduled
        closeTimer = setTimeout(() => {
          closeCallbacks = Promise.resolve()
            .then(() => {
              logger.debug(
                { taskId: task.id },
                'Closing task container after result',
              );
              deps.queue.closeStdin(task.chat_jid);
            })
            .then(undefined, (error: unknown) => {
              closeFailure = { error };
              rejectClose(error);
            });
        }, TASK_CLOSE_DELAY_MS);
      };

      const running = runContainerAgent(
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
        (proc, containerName) => {
          ownedContainer = containerName;
          deps.onProcess(task.chat_jid, proc, containerName, task.group_folder);
        },
        async (streamedOutput: ContainerOutput) => {
          if (streamedOutput.result) {
            result = streamedOutput.result;
            // Forward result to user (sendMessage handles formatting)
            await deps.sendMessage(task.chat_jid, streamedOutput.result);
          }
          if (
            streamedOutput.status === 'success' &&
            (streamedOutput.completion !== undefined ||
              (streamedOutput.result !== null && streamedOutput.result.trim()))
          ) {
            deps.queue.notifyIdle(task.chat_jid);
            scheduleClose(); // Close promptly even when result is null (e.g. IPC-only tasks)
          }
          if (streamedOutput.status === 'error') {
            error = streamedOutput.error || 'Unknown error';
          }
        },
      );

      const output = await Promise.race([running, closeOutcome]);
      // A same-tick native exit cannot discard a close callback already started.
      await closeCallbacks;
      if (closeFailure) throw closeFailure.error;
      if (closeTimer) clearTimeout(closeTimer);

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

      return false;
    })(),
  ]);
  if (closeTimer) clearTimeout(closeTimer);
  if (execution.status === 'fulfilled' && execution.value) return;
  let taskExecution = execution;
  if (execution.status === 'rejected' && closeFailure && ownedContainer) {
    const [cleanup] = await Promise.allSettled([
      Promise.resolve().then(() => stopContainer(ownedContainer!)),
    ]);
    if (cleanup.status === 'rejected')
      taskExecution = {
        status: 'rejected',
        reason: new AggregateError(
          [execution.reason as unknown, cleanup.reason as unknown],
          'Task close and owned-container cleanup failed; inspect before retrying.',
          { cause: execution.reason as unknown },
        ),
      };
  }
  if (taskExecution.status === 'rejected') {
    error =
      taskExecution.reason instanceof Error
        ? taskExecution.reason.message
        : String(taskExecution.reason);
    logger.error({ taskId: task.id, error }, 'Task failed');
  }
  const [bookkeeping] = await Promise.allSettled([
    Promise.resolve().then(() => {
      const durationMs = Date.now() - startTime;

      logTaskRun({
        task_id: task.id,
        run_at: new Date().toISOString(),
        duration_ms: durationMs,
        status: error ? 'error' : 'success',
        result,
        error,
      });

      const nextRun = computeNextRun(task);
      const resultSummary = error
        ? `Error: ${error}`
        : result
          ? result.slice(0, 200)
          : 'Completed';
      updateTaskAfterRun(task.id, nextRun, resultSummary);
    }),
  ]);
  if (bookkeeping.status === 'rejected') {
    if (taskExecution.status === 'rejected')
      throw new AggregateError(
        [taskExecution.reason as unknown, bookkeeping.reason as unknown],
        'Task execution and run bookkeeping failed; repair task storage before retrying.',
        { cause: bookkeeping.reason as unknown },
      );
    throw bookkeeping.reason;
  }
  if (taskExecution.status === 'rejected') {
    const failure: unknown = taskExecution.reason;
    if (!isOperationalRunError(failure)) throw failure;
  }
}

let schedulerRunning = false;
let schedulerLifecycle: Promise<void> | undefined;

export function startSchedulerLoop(deps: SchedulerDependencies): Promise<void> {
  if (schedulerRunning) {
    logger.debug('Scheduler loop already running, skipping duplicate start');
    return schedulerLifecycle!;
  }
  schedulerRunning = true;
  logger.info('Scheduler loop started');

  let rejectLifecycle!: (error: unknown) => void;
  schedulerLifecycle = new Promise<void>((_resolve, reject) => {
    rejectLifecycle = reject;
  });
  const poll = () => {
    void loop().then(undefined, rejectLifecycle);
  };
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

        try {
          deps.queue.enqueueTask(currentTask.chat_jid, currentTask.id, () =>
            runTask(currentTask, deps),
          );
        } catch (error) {
          if (!isFileSystemError(error)) throw error;
          // The queue retains the single pending task and retries preemption on
          // the next poll; unrelated due groups can still make progress.
          logger.error(
            { taskId: currentTask.id, error },
            'Task preemption signal failed; retrying control on the next poll',
          );
        }
      }
    } catch (err) {
      if (!isFileSystemError(err) && !isRecoverableSqliteError(err)) throw err;
      logger.error({ err }, 'Error in scheduler loop');
    } finally {
      setTimeout(poll, SCHEDULER_POLL_INTERVAL);
    }
  };

  poll();
  return schedulerLifecycle;
}

/** @internal - for tests only. */
export function _resetSchedulerLoopForTests(): void {
  schedulerRunning = false;
  schedulerLifecycle = undefined;
}
