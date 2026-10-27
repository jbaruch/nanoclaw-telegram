import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SchedulerDependencies } from './task-scheduler.js';

const fixture = vi.hoisted(() => ({ directory: '' }));
vi.mock('./config.js', () => ({
  ASSISTANT_NAME: 'Andy',
  TIMEZONE: 'UTC',
  SCHEDULER_POLL_INTERVAL: 1000,
  get GROUPS_DIR() {
    return fixture.directory;
  },
  get DATA_DIR() {
    return fixture.directory;
  },
  get STORE_DIR() {
    return fixture.directory;
  },
}));
vi.mock('./container-runner.js', () => ({
  runContainerAgent: vi.fn(),
  writeTasksSnapshot: vi.fn(),
}));
import * as database from './db.js';
import { runContainerAgent } from './container-runner.js';
import {
  _resetSchedulerLoopForTests,
  startSchedulerLoop,
} from './task-scheduler.js';

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2024-01-01T00:01:00Z'));
  fixture.directory = fs.mkdtempSync(
    path.join(os.tmpdir(), 'nanoclaw-task-recovery-'),
  );
  database._initTestDatabase();
  _resetSchedulerLoopForTests();
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllTimers();
  vi.useRealTimers();
  database._closeDatabase();
  fs.rmSync(fixture.directory, { recursive: true });
});
function task(kind: 'once' | 'interval') {
  database.createTask({
    id: 'fixed-task',
    group_folder: 'fixed',
    chat_jid: 'tg:123',
    prompt: 'fixed',
    schedule_type: kind,
    schedule_value: kind === 'interval' ? '5000' : '2024-01-01T00:00:59Z',
    context_mode: 'isolated',
    next_run: '2024-01-01T00:00:59.000Z',
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
  });
}
function dependencies(
  enqueueTask: SchedulerDependencies['queue']['enqueueTask'],
): SchedulerDependencies {
  return {
    registeredGroups: () => ({
      'tg:123': {
        name: 'fixed',
        folder: 'fixed',
        isMain: true,
        trigger: '@Andy',
        added_at: '2024-01-01T00:00:00Z',
      },
    }),
    getSessions: () => ({}),
    queue: { enqueueTask, closeStdin: vi.fn(), notifyIdle: vi.fn() },
    onProcess: vi.fn(),
    sendMessage: vi.fn(async () => {}),
  };
}

describe('task lifecycle recovery', () => {
  it.each(['once', 'interval'] as const)(
    'records an unexpected %s failure and advances the task before propagating it',
    async (kind) => {
      task(kind);
      const defect = new TypeError('fixed native implementation defect');
      vi.mocked(runContainerAgent).mockRejectedValueOnce(defect);
      const log = vi.spyOn(database, 'logTaskRun');
      let execute: (() => Promise<void>) | undefined;
      const enqueue = vi.fn(
        (_jid: string, _id: string, fn: () => Promise<void>) => {
          execute = fn;
        },
      );
      startSchedulerLoop(dependencies(enqueue));
      await vi.advanceTimersByTimeAsync(0);
      await expect(execute!()).rejects.toBe(defect);
      expect(log).toHaveBeenCalledWith(
        expect.objectContaining({
          task_id: 'fixed-task',
          status: 'error',
          error: 'fixed native implementation defect',
        }),
      );
      const updated = database.getTaskById('fixed-task');
      expect(updated?.last_result).toBe(
        'Error: fixed native implementation defect',
      );
      expect(updated?.next_run).toBe(
        kind === 'once' ? null : '2024-01-01T00:01:04.000Z',
      );
      expect(updated?.status).toBe(kind === 'once' ? 'completed' : 'active');
      await vi.advanceTimersByTimeAsync(1000);
      expect(enqueue).toHaveBeenCalledTimes(1);
    },
  );

  it('keeps the scheduler alive after an unexpected dispatch rejection with exactly one next poll', async () => {
    task('once');
    const defect = new TypeError('fixed enqueue defect');
    const enqueue = vi.fn().mockImplementationOnce(() => {
      throw defect;
    });
    await expect(startSchedulerLoop(dependencies(enqueue))).rejects.toBe(
      defect,
    );
    expect(vi.getTimerCount()).toBe(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(enqueue).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(1);
  });

  it('exposes an unexpected delayed dispatch failure through the same lifecycle Promise', async () => {
    task('once');
    const enqueue = vi.fn();
    const lifecycle = startSchedulerLoop(dependencies(enqueue));
    await vi.advanceTimersByTimeAsync(0);
    const defect = new TypeError('fixed delayed enqueue defect');
    enqueue.mockImplementationOnce(() => {
      throw defect;
    });
    const rejected = expect(lifecycle).rejects.toBe(defect);
    await vi.advanceTimersByTimeAsync(1000);
    await rejected;
    expect(vi.getTimerCount()).toBe(1);
  });

  it('clears a scheduled close and finishes bookkeeping after a post-output defect', async () => {
    task('once');
    const defect = new TypeError('fixed post-output defect');
    vi.mocked(runContainerAgent).mockImplementationOnce(
      async (_group, _input, _process, onOutput) => {
        await onOutput!({
          status: 'success',
          result: 'fixed synthetic output',
        });
        throw defect;
      },
    );
    let execute: (() => Promise<void>) | undefined;
    const deps = dependencies((_jid, _id, fn) => {
      execute = fn;
    });
    startSchedulerLoop(deps);
    await vi.advanceTimersByTimeAsync(0);
    await expect(execute!()).rejects.toBe(defect);
    expect(deps.sendMessage).toHaveBeenCalledExactlyOnceWith(
      'tg:123',
      'fixed synthetic output',
    );
    expect(database.getTaskById('fixed-task')?.status).toBe('completed');
    expect(vi.getTimerCount()).toBe(1);
    await vi.advanceTimersByTimeAsync(11000);
    expect(deps.queue.closeStdin).not.toHaveBeenCalled();
  });
});
