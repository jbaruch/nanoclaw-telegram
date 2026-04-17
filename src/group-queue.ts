import { ChildProcess } from 'child_process';
import fs from 'fs';
import path from 'path';

import { DATA_DIR, MAX_CONCURRENT_CONTAINERS } from './config.js';
import {
  DEFAULT_SESSION_NAME,
  sessionInputDirName,
} from './container-runner.js';
import { logger } from './logger.js';

// Re-export so callers that already imported it from group-queue keep working.
// Container-runner is the canonical definer — this file, the task-scheduler,
// and index.ts all need the same string, and the session-aware IPC mount
// lives in container-runner, so that's where the symbol originates.
export { DEFAULT_SESSION_NAME };

/**
 * Canonical session name for scheduled work (heartbeat, nightly, weekly,
 * reminders). `src/task-scheduler.ts` is the sole writer of this value;
 * no inbound path ever reaches it. Enforced by routing at call sites, not
 * by a runtime check — validated in tests.
 */
export const MAINTENANCE_SESSION_NAME = 'maintenance';

interface QueuedTask {
  id: string;
  groupJid: string;
  sessionName: string;
  fn: () => Promise<void>;
}

const MAX_RETRIES = 5;
const BASE_RETRY_MS = 5000;

interface GroupState {
  groupJid: string;
  sessionName: string;
  active: boolean;
  idleWaiting: boolean;
  isTaskContainer: boolean;
  runningTaskId: string | null;
  pendingMessages: boolean;
  pendingTasks: QueuedTask[];
  process: ChildProcess | null;
  containerName: string | null;
  groupFolder: string | null;
  retryCount: number;
}

/**
 * GroupQueue tracks in-flight containers per `(groupJid, sessionName)` pair.
 * Two sessions for the same group (`default` + `maintenance`) can run
 * concurrently — each occupies its own slot. Global concurrency is still
 * capped by `MAX_CONCURRENT_CONTAINERS` across all slots.
 *
 * User-facing paths (message check, IPC send, close stdin, idle notification)
 * target the `default` session implicitly; scheduled paths pass `sessionName`
 * explicitly via `enqueueTask`.
 */
export class GroupQueue {
  private groups = new Map<string, GroupState>();
  private activeCount = 0;
  // Waiting list holds composite keys (`${groupJid}::${sessionName}`) so the
  // same group can have both a default and a maintenance slot waiting.
  private waitingKeys: string[] = [];
  private processMessagesFn: ((groupJid: string) => Promise<boolean>) | null =
    null;
  private shuttingDown = false;

  private keyFor(groupJid: string, sessionName: string): string {
    return `${groupJid}::${sessionName}`;
  }

  private getGroup(groupJid: string, sessionName: string): GroupState {
    const key = this.keyFor(groupJid, sessionName);
    let state = this.groups.get(key);
    if (!state) {
      state = {
        groupJid,
        sessionName,
        active: false,
        idleWaiting: false,
        isTaskContainer: false,
        runningTaskId: null,
        pendingMessages: false,
        pendingTasks: [],
        process: null,
        containerName: null,
        groupFolder: null,
        retryCount: 0,
      };
      this.groups.set(key, state);
    }
    return state;
  }

  setProcessMessagesFn(fn: (groupJid: string) => Promise<boolean>): void {
    this.processMessagesFn = fn;
  }

  /**
   * Inbound user message arrived. Always routes to the `default` session —
   * user-facing AyeAye is the only one that responds to users.
   */
  enqueueMessageCheck(groupJid: string): void {
    if (this.shuttingDown) return;

    const state = this.getGroup(groupJid, DEFAULT_SESSION_NAME);

    if (state.active) {
      state.pendingMessages = true;
      logger.debug({ groupJid }, 'Container active, message queued');
      return;
    }

    if (this.activeCount >= MAX_CONCURRENT_CONTAINERS) {
      state.pendingMessages = true;
      const key = this.keyFor(groupJid, DEFAULT_SESSION_NAME);
      if (!this.waitingKeys.includes(key)) {
        this.waitingKeys.push(key);
      }
      logger.debug(
        { groupJid, activeCount: this.activeCount },
        'At concurrency limit, message queued',
      );
      return;
    }

    this.runForGroup(groupJid, 'messages').catch((err) =>
      logger.error({ groupJid, err }, 'Unhandled error in runForGroup'),
    );
  }

  /**
   * Enqueue a scheduled task. `sessionName` determines which queue slot it
   * goes into:
   * - `'default'` (user-facing): serializes with inbound messages. Rarely
   *   needed — the user-facing session usually handles only IPC messages.
   * - `'maintenance'`: the parallel scheduled-task slot. The scheduler is
   *   the canonical writer of this value.
   */
  enqueueTask(
    groupJid: string,
    taskId: string,
    sessionName: string,
    fn: () => Promise<void>,
  ): void {
    if (this.shuttingDown) return;

    const state = this.getGroup(groupJid, sessionName);

    // Prevent double-queuing: check both pending and currently-running task
    if (state.runningTaskId === taskId) {
      logger.debug(
        { groupJid, sessionName, taskId },
        'Task already running, skipping',
      );
      return;
    }
    if (state.pendingTasks.some((t) => t.id === taskId)) {
      logger.debug(
        { groupJid, sessionName, taskId },
        'Task already queued, skipping',
      );
      return;
    }

    const task: QueuedTask = { id: taskId, groupJid, sessionName, fn };

    if (state.active) {
      state.pendingTasks.push(task);
      if (state.idleWaiting) {
        this.closeStdin(groupJid, sessionName);
      }
      logger.debug(
        { groupJid, sessionName, taskId },
        'Container active, task queued',
      );
      return;
    }

    if (this.activeCount >= MAX_CONCURRENT_CONTAINERS) {
      state.pendingTasks.push(task);
      const key = this.keyFor(groupJid, sessionName);
      if (!this.waitingKeys.includes(key)) {
        this.waitingKeys.push(key);
      }
      logger.debug(
        { groupJid, sessionName, taskId, activeCount: this.activeCount },
        'At concurrency limit, task queued',
      );
      return;
    }

    // Run immediately
    this.runTask(groupJid, sessionName, task).catch((err) =>
      logger.error(
        { groupJid, sessionName, taskId, err },
        'Unhandled error in runTask',
      ),
    );
  }

  registerProcess(
    groupJid: string,
    sessionName: string,
    proc: ChildProcess,
    containerName: string,
    groupFolder?: string,
  ): void {
    const state = this.getGroup(groupJid, sessionName);
    state.process = proc;
    state.containerName = containerName;
    if (groupFolder) state.groupFolder = groupFolder;
  }

  /**
   * Mark the container as idle-waiting (finished work, waiting for IPC input).
   * If tasks are pending, preempt the idle container immediately. Idle-wait
   * only applies to user-facing containers (`default`), so `sessionName` is
   * implicit.
   */
  notifyIdle(groupJid: string): void {
    const state = this.getGroup(groupJid, DEFAULT_SESSION_NAME);
    state.idleWaiting = true;
    if (state.pendingTasks.length > 0) {
      this.closeStdin(groupJid, DEFAULT_SESSION_NAME);
    }
  }

  /**
   * Send a follow-up message to the active user-facing container via IPC
   * file. Only writes when the user-facing (`default`) container is active.
   * Returns true if written, false otherwise.
   */
  sendMessage(
    groupJid: string,
    text: string,
    replyToMessageId?: string,
  ): boolean {
    const state = this.getGroup(groupJid, DEFAULT_SESSION_NAME);
    if (!state.active || !state.groupFolder || state.isTaskContainer)
      return false;
    state.idleWaiting = false; // Agent is about to receive work, no longer idle

    // Follow-ups always target the user-facing session's input dir — the
    // maintenance container's mount points at `input-maintenance/` and
    // therefore can't see messages dropped here. This is the whole point
    // of per-session input dirs.
    const inputDir = path.join(
      DATA_DIR,
      'ipc',
      state.groupFolder,
      sessionInputDirName(DEFAULT_SESSION_NAME),
    );
    try {
      fs.mkdirSync(inputDir, { recursive: true });
      const filename = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}.json`;
      const filepath = path.join(inputDir, filename);
      const tempPath = `${filepath}.tmp`;
      const data: Record<string, string> = { type: 'message', text };
      if (replyToMessageId) data.replyToMessageId = replyToMessageId;
      fs.writeFileSync(tempPath, JSON.stringify(data));
      fs.renameSync(tempPath, filepath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Signal the active container to wind down by writing a close sentinel.
   * `sessionName` selects which container (default for user-message path,
   * maintenance can also be closed when pending tasks arrive idle).
   */
  closeStdin(
    groupJid: string,
    sessionName: string = DEFAULT_SESSION_NAME,
  ): void {
    const state = this.getGroup(groupJid, sessionName);
    if (!state.active || !state.groupFolder) return;

    // Session-scoped sentinel — writing `_close` to `input-default/` affects
    // only the default container; the maintenance container polls its own
    // `input-maintenance/` directory and is unaffected.
    const inputDir = path.join(
      DATA_DIR,
      'ipc',
      state.groupFolder,
      sessionInputDirName(sessionName),
    );
    try {
      fs.mkdirSync(inputDir, { recursive: true });
      fs.writeFileSync(path.join(inputDir, '_close'), '');
    } catch {
      // ignore
    }
  }

  private async runForGroup(
    groupJid: string,
    reason: 'messages' | 'drain',
  ): Promise<void> {
    // Message-check runs always use the default session.
    const state = this.getGroup(groupJid, DEFAULT_SESSION_NAME);
    state.active = true;
    state.idleWaiting = false;
    state.isTaskContainer = false;
    state.pendingMessages = false;
    this.activeCount++;

    logger.debug(
      { groupJid, reason, activeCount: this.activeCount },
      'Starting container for group',
    );

    try {
      if (this.processMessagesFn) {
        const success = await this.processMessagesFn(groupJid);
        if (success) {
          state.retryCount = 0;
        } else {
          this.scheduleRetry(groupJid, state);
        }
      }
    } catch (err) {
      logger.error({ groupJid, err }, 'Error processing messages for group');
      this.scheduleRetry(groupJid, state);
    } finally {
      state.active = false;
      state.process = null;
      state.containerName = null;
      state.groupFolder = null;
      this.activeCount--;
      this.drainGroup(groupJid, DEFAULT_SESSION_NAME);
    }
  }

  private async runTask(
    groupJid: string,
    sessionName: string,
    task: QueuedTask,
  ): Promise<void> {
    const state = this.getGroup(groupJid, sessionName);
    state.active = true;
    state.idleWaiting = false;
    state.isTaskContainer = true;
    state.runningTaskId = task.id;
    this.activeCount++;

    logger.debug(
      {
        groupJid,
        sessionName,
        taskId: task.id,
        activeCount: this.activeCount,
      },
      'Running queued task',
    );

    try {
      await task.fn();
    } catch (err) {
      logger.error(
        { groupJid, sessionName, taskId: task.id, err },
        'Error running task',
      );
    } finally {
      state.active = false;
      state.isTaskContainer = false;
      state.runningTaskId = null;
      state.process = null;
      state.containerName = null;
      state.groupFolder = null;
      this.activeCount--;
      this.drainGroup(groupJid, sessionName);
    }
  }

  private scheduleRetry(groupJid: string, state: GroupState): void {
    state.retryCount++;
    if (state.retryCount > MAX_RETRIES) {
      logger.error(
        { groupJid, retryCount: state.retryCount },
        'Max retries exceeded, dropping messages (will retry on next incoming message)',
      );
      state.retryCount = 0;
      return;
    }

    const delayMs = BASE_RETRY_MS * Math.pow(2, state.retryCount - 1);
    logger.info(
      { groupJid, retryCount: state.retryCount, delayMs },
      'Scheduling retry with backoff',
    );
    setTimeout(() => {
      if (!this.shuttingDown) {
        this.enqueueMessageCheck(groupJid);
      }
    }, delayMs);
  }

  private drainGroup(groupJid: string, sessionName: string): void {
    if (this.shuttingDown) return;

    const state = this.getGroup(groupJid, sessionName);

    // Tasks first (they won't be re-discovered from SQLite like messages)
    if (state.pendingTasks.length > 0) {
      const task = state.pendingTasks.shift()!;
      this.runTask(groupJid, sessionName, task).catch((err) =>
        logger.error(
          { groupJid, sessionName, taskId: task.id, err },
          'Unhandled error in runTask (drain)',
        ),
      );
      return;
    }

    // Then pending messages — only relevant on the default slot.
    if (sessionName === DEFAULT_SESSION_NAME && state.pendingMessages) {
      this.runForGroup(groupJid, 'drain').catch((err) =>
        logger.error(
          { groupJid, err },
          'Unhandled error in runForGroup (drain)',
        ),
      );
      return;
    }

    // Nothing pending for this slot; check if other slots are waiting.
    this.drainWaiting();
  }

  private drainWaiting(): void {
    while (
      this.waitingKeys.length > 0 &&
      this.activeCount < MAX_CONCURRENT_CONTAINERS
    ) {
      const nextKey = this.waitingKeys.shift()!;
      const [nextJid, nextSessionName] = nextKey.split('::');
      if (!nextJid || !nextSessionName) continue;
      const state = this.getGroup(nextJid, nextSessionName);

      // Prioritize tasks over messages
      if (state.pendingTasks.length > 0) {
        const task = state.pendingTasks.shift()!;
        this.runTask(nextJid, nextSessionName, task).catch((err) =>
          logger.error(
            {
              groupJid: nextJid,
              sessionName: nextSessionName,
              taskId: task.id,
              err,
            },
            'Unhandled error in runTask (waiting)',
          ),
        );
      } else if (
        nextSessionName === DEFAULT_SESSION_NAME &&
        state.pendingMessages
      ) {
        this.runForGroup(nextJid, 'drain').catch((err) =>
          logger.error(
            { groupJid: nextJid, err },
            'Unhandled error in runForGroup (waiting)',
          ),
        );
      }
      // If neither pending, skip this slot
    }
  }

  async shutdown(_gracePeriodMs: number): Promise<void> {
    this.shuttingDown = true;

    // Count active containers but don't kill them — they'll finish on their own
    // via idle timeout or container timeout. The --rm flag cleans them up on exit.
    // This prevents WhatsApp reconnection restarts from killing working agents.
    const activeContainers: string[] = [];
    for (const [_key, state] of this.groups) {
      if (state.process && !state.process.killed && state.containerName) {
        activeContainers.push(state.containerName);
      }
    }

    logger.info(
      { activeCount: this.activeCount, detachedContainers: activeContainers },
      'GroupQueue shutting down (containers detached, not killed)',
    );
  }
}
