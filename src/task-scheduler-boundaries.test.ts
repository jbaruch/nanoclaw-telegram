import fs from 'fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const runContainerAgentMock = vi.hoisted(() => vi.fn());
const logTaskRunSpy = vi.hoisted(() => vi.fn());
const updateTaskAfterRunSpy = vi.hoisted(() => vi.fn());

vi.mock('./container-runner.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./container-runner.js')>();
  return {
    ...actual,
    runContainerAgent: runContainerAgentMock,
    writeTasksSnapshot: vi.fn(),
  };
});

vi.mock('./db.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./db.js')>();
  return {
    ...actual,
    logTaskRun: (...args: Parameters<typeof actual.logTaskRun>) => {
      logTaskRunSpy(...args);
      return actual.logTaskRun(...args);
    },
    updateTaskAfterRun: (
      ...args: Parameters<typeof actual.updateTaskAfterRun>
    ) => {
      updateTaskAfterRunSpy(...args);
      return actual.updateTaskAfterRun(...args);
    },
  };
});

import { SCHEDULER_POLL_INTERVAL } from './config.js';
import { _initTestDatabase, createTask, getTaskById } from './db.js';
import { GroupQueue } from './group-queue.js';
import {
  _resetSchedulerLoopForTests,
  startSchedulerLoop,
} from './task-scheduler.js';

describe('scheduled task run boundaries', () => {
  beforeEach(() => {
    _initTestDatabase();
    _resetSchedulerLoopForTests();
    runContainerAgentMock.mockReset();
    logTaskRunSpy.mockReset();
    updateTaskAfterRunSpy.mockReset();
    vi.spyOn(fs, 'mkdirSync').mockImplementation(() => undefined);
    vi.useFakeTimers();
  });

  afterEach(() => {
    _resetSchedulerLoopForTests();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('records and completes a once task before propagating a coded TypeError', async () => {
    const err = Object.assign(new TypeError('container callback failed'), {
      code: 'ERR_INVALID_ARG_TYPE',
    });
    runContainerAgentMock.mockImplementation(
      async (
        _group: unknown,
        _input: unknown,
        _onProcess: unknown,
        onOutput: (output: {
          status: 'success';
          result: string;
        }) => Promise<void>,
      ) => {
        await onOutput({ status: 'success', result: 'delivered result' });
        throw err;
      },
    );
    createTask({
      id: 'task-coded-type-error',
      group_folder: 'test-group',
      chat_jid: 'test@g.us',
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 60_000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
    });

    const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout');
    const queue = new GroupQueue();
    const loop = startSchedulerLoop({
      registeredGroups: () => ({
        'test@g.us': {
          name: 'Test',
          folder: 'test-group',
          trigger: '@bot',
          added_at: '2026-01-01T00:00:00.000Z',
        },
      }),
      getSessions: () => ({}),
      queue,
      onProcess: () => {},
      sendMessage: async () => {},
    });

    await vi.advanceTimersByTimeAsync(0);

    expect(runContainerAgentMock).toHaveBeenCalledTimes(1);
    expect(logTaskRunSpy).toHaveBeenCalledTimes(1);
    expect(logTaskRunSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        task_id: 'task-coded-type-error',
        status: 'error',
        error: 'container callback failed',
      }),
    );
    expect(updateTaskAfterRunSpy).toHaveBeenCalledTimes(1);
    expect(clearTimeoutSpy).toHaveBeenCalledTimes(1);
    expect(getTaskById('task-coded-type-error')).toMatchObject({
      status: 'completed',
      next_run: null,
      last_result: 'Error: container callback failed',
    });

    await vi.advanceTimersByTimeAsync(SCHEDULER_POLL_INTERVAL * 3);
    expect(runContainerAgentMock).toHaveBeenCalledTimes(1);
    expect(logTaskRunSpy).toHaveBeenCalledTimes(1);
    expect(updateTaskAfterRunSpy).toHaveBeenCalledTimes(1);

    _resetSchedulerLoopForTests();
    await vi.advanceTimersByTimeAsync(SCHEDULER_POLL_INTERVAL);
    await loop;
  });
});
