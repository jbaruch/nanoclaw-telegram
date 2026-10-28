import fs from 'fs';
import Database from 'better-sqlite3';
import { ChildProcess } from 'child_process';
import { IpcRouteUnavailableError } from './operational-errors.js';
import { GroupQueue } from './group-queue.js';
import { stopContainer } from './container-runtime.js';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SchedulerDependencies } from './task-scheduler.js';

const fixture = vi.hoisted(() => ({ directory: '' }));
vi.mock('./config.js', () => ({
  ASSISTANT_NAME: 'Andy',
  MAX_CONCURRENT_CONTAINERS: 2,
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
vi.mock('./container-runtime.js', () => ({ stopContainer: vi.fn() }));
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
  vi.mocked(runContainerAgent).mockReset();
  vi.mocked(stopContainer).mockReset();
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
  it('keeps preview and session markers active until genuine completion starts the close timer', async () => {
    task('once');
    let execute!: () => Promise<void>;
    let emit!: NonNullable<Parameters<typeof runContainerAgent>[3]>;
    let settle!: (output: { status: 'success'; result: null }) => void;
    vi.mocked(runContainerAgent).mockImplementationOnce(
      async (_group, _input, _process, onOutput) => {
        emit = onOutput!;
        return new Promise((resolve) => {
          settle = resolve;
        });
      },
    );
    const deps = dependencies((_jid, _id, fn) => {
      execute = fn;
    });
    startSchedulerLoop(deps);
    await vi.advanceTimersByTimeAsync(0);
    const pending = execute();
    await vi.advanceTimersByTimeAsync(0);
    await emit({
      status: 'success',
      result: null,
      streamText: 'fixed preview',
    });
    await vi.advanceTimersByTimeAsync(11000);
    await emit({
      status: 'success',
      result: null,
      newSessionId: 'fixed-session',
    });
    await vi.advanceTimersByTimeAsync(11000);
    expect(deps.queue.notifyIdle).not.toHaveBeenCalled();
    expect(deps.queue.closeStdin).not.toHaveBeenCalled();
    await emit({
      status: 'success',
      result: null,
      completion: 'sdk_turn_completed',
    });
    expect(deps.queue.notifyIdle).toHaveBeenCalledExactlyOnceWith('tg:123');
    await vi.advanceTimersByTimeAsync(9999);
    expect(deps.queue.closeStdin).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(deps.queue.closeStdin).toHaveBeenCalledExactlyOnceWith('tg:123');
    settle({ status: 'success', result: null });
    await pending;
    expect(database.getTaskById('fixed-task')?.status).toBe('completed');
  });
  it('settles an immediate SDK-error native close without waiting for a hard timeout', async () => {
    task('once');
    // The native container transport is mocked; actual agent failure exit is
    // exercised by agent-protocol.test.ts through its unmodified entrypoint.
    vi.mocked(runContainerAgent).mockResolvedValueOnce({
      status: 'error',
      result: null,
      error: 'SDK query did not complete successfully (error_during_execution)',
    });
    let execute!: () => Promise<void>;
    const enqueue = vi.fn(
      (_jid: string, _id: string, fn: () => Promise<void>) => {
        execute = fn;
      },
    );
    const log = vi.spyOn(database, 'logTaskRun');
    startSchedulerLoop(dependencies(enqueue));
    await vi.advanceTimersByTimeAsync(0);
    const now = Date.now();
    await execute();
    expect(Date.now()).toBe(now);
    expect(log).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ status: 'error' }),
    );
    expect(database.getTaskById('fixed-task')).toMatchObject({
      status: 'completed',
      next_run: null,
    });
    expect(vi.getTimerCount()).toBe(1);
  });
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

  it.each(['once', 'interval'] as const)(
    'settles a missing registered group for a %s task instead of replaying each poll',
    async (kind) => {
      task(kind);
      let execute!: () => Promise<void>;
      const enqueue = vi.fn(
        (_jid: string, _id: string, fn: () => Promise<void>) => {
          execute = fn;
        },
      );
      const deps = dependencies(enqueue);
      deps.registeredGroups = () => ({});
      const log = vi.spyOn(database, 'logTaskRun');
      startSchedulerLoop(deps);
      await vi.advanceTimersByTimeAsync(0);
      await expect(execute()).resolves.toBeUndefined();
      expect(log).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({
          task_id: 'fixed-task',
          status: 'error',
          error: 'Group not found: fixed',
        }),
      );
      expect(database.getTaskById('fixed-task')).toMatchObject({
        status: kind === 'once' ? 'completed' : 'active',
        next_run: kind === 'once' ? null : '2024-01-01T00:01:04.000Z',
        last_result: 'Error: Group not found: fixed',
      });
      expect(runContainerAgent).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(1000);
      expect(enqueue).toHaveBeenCalledTimes(1);
      expect(log).toHaveBeenCalledTimes(1);
    },
  );

  it('propagates a SQLite SQL/schema defect through the scheduler lifecycle', async () => {
    const defect = new Database.SqliteError(
      'fixed missing column',
      'SQLITE_ERROR',
    );
    vi.spyOn(database, 'getDueTasks').mockImplementationOnce(() => {
      throw defect;
    });
    await expect(startSchedulerLoop(dependencies(vi.fn()))).rejects.toBe(
      defect,
    );
    expect(vi.getTimerCount()).toBe(1);
  });

  it.each(['0', '1000junk', '1.5'])(
    'pauses a malformed stored interval %s without effects or repeated run logs',
    async (value) => {
      task('interval');
      database.updateTask('fixed-task', { schedule_value: value });
      let execute!: () => Promise<void>;
      const enqueue = vi.fn(
        (_jid: string, _id: string, fn: () => Promise<void>) => {
          execute = fn;
        },
      );
      const log = vi.spyOn(database, 'logTaskRun');
      startSchedulerLoop(dependencies(enqueue));
      await vi.advanceTimersByTimeAsync(0);
      await expect(execute()).resolves.toBeUndefined();
      expect(database.getTaskById('fixed-task')).toMatchObject({
        status: 'paused',
        schedule_value: value,
      });
      expect(log).toHaveBeenCalledTimes(1);
      expect(runContainerAgent).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(1000);
      expect(enqueue).toHaveBeenCalledTimes(1);
    },
  );

  it('records scheduled route-unavailable delivery as a failed run before advancing', async () => {
    task('once');
    let execute!: () => Promise<void>;
    const deps = dependencies((_jid, _id, fn) => {
      execute = fn;
    });
    const unavailable = new IpcRouteUnavailableError(
      'fixed unavailable scheduled route',
    );
    deps.sendMessage = vi.fn(async () => {
      throw unavailable;
    });
    vi.mocked(runContainerAgent).mockImplementationOnce(
      async (_group, _input, _process, onOutput) => {
        await onOutput!({ status: 'success', result: 'fixed scheduled body' });
        return { status: 'success', result: null };
      },
    );
    const log = vi.spyOn(database, 'logTaskRun');
    startSchedulerLoop(deps);
    await vi.advanceTimersByTimeAsync(0);
    await expect(execute()).resolves.toBeUndefined();
    expect(log).toHaveBeenCalledWith(
      expect.objectContaining({ status: 'error', error: unavailable.message }),
    );
    expect(database.getTaskById('fixed-task')).toMatchObject({
      status: 'completed',
      next_run: null,
      last_result: `Error: ${unavailable.message}`,
    });
  });

  it.each([false, true])(
    'surfaces an actual queued task close-timer defect after bookkeeping (cleanup failure: %s)',
    async (cleanupFails) => {
      task('once');
      const defect = new TypeError('fixed delayed task close defect');
      const cleanup = new TypeError('fixed task native cleanup defect');
      const queue = new GroupQueue();
      vi.spyOn(queue, 'closeStdin').mockImplementation(() => {
        throw defect;
      });
      vi.mocked(stopContainer).mockReset();
      if (cleanupFails)
        vi.mocked(stopContainer).mockImplementation(() => {
          throw cleanup;
        });
      let settleNative!: (output: { status: 'success'; result: null }) => void;
      vi.mocked(runContainerAgent).mockImplementationOnce(
        async (_group, _input, onProcess, onOutput) => {
          onProcess(new ChildProcess(), 'owned-scheduled-close');
          await onOutput!({
            status: 'success',
            result: 'fixed synthetic result',
          });
          return new Promise((resolve) => {
            settleNative = resolve;
          });
        },
      );
      const deps = dependencies(queue.enqueueTask.bind(queue));
      deps.queue = queue;
      const lifecycle = queue.waitForFailure();
      const rejected = cleanupFails
        ? expect(lifecycle).rejects.toMatchObject({
            cause: defect,
            errors: [defect, cleanup],
          })
        : expect(lifecycle).rejects.toBe(defect);
      startSchedulerLoop(deps);
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(10000);
      await rejected;
      expect(stopContainer).toHaveBeenCalledExactlyOnceWith(
        'owned-scheduled-close',
      );
      expect(database.getTaskById('fixed-task')).toMatchObject({
        status: 'completed',
        next_run: null,
        last_result: expect.stringContaining('Error:'),
      });
      const snapshot = database.getTaskById('fixed-task');
      await vi.advanceTimersByTimeAsync(2000);
      expect(runContainerAgent).toHaveBeenCalledTimes(1);
      expect(database.getTaskById('fixed-task')).toEqual(snapshot);
      settleNative({ status: 'success', result: null });
      await vi.advanceTimersByTimeAsync(0);
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

it.each([
  ['2024-01-01T00:00', '2024-01-01T00:00:00.000Z'],
  ['2023-11-05T01:30', '2023-11-05T06:30:00.000Z'],
  ['2023-03-12T02:30:00', '2023-03-12T08:30:00.000Z'],
])(
  'executes a legacy once schedule %s at its captured instant without rewriting history',
  async (original, captured) => {
    task('once');
    database.updateTask('fixed-task', {
      schedule_value: original,
      next_run: captured,
    });
    vi.mocked(runContainerAgent).mockResolvedValue({
      status: 'success',
      result: null,
      completion: 'sdk_turn_completed',
    });
    let execute!: () => Promise<void>;
    startSchedulerLoop(
      dependencies((_jid, _id, run) => {
        execute = run;
      }),
    );
    await vi.advanceTimersByTimeAsync(0);
    expect(execute).toBeDefined();
    await execute();
    expect(runContainerAgent).toHaveBeenCalledTimes(1);
    expect(database.getTaskById('fixed-task')).toMatchObject({
      schedule_value: original,
      status: 'completed',
      next_run: null,
    });
    await vi.advanceTimersByTimeAsync(1000);
    expect(runContainerAgent).toHaveBeenCalledTimes(1);
  },
);

it.each([
  ['not-a-date', '2024-01-01T00:00:00.000Z'],
  ['2024-01-01T00:00', '2024-01-01T00:00:00'],
])(
  'holds a stored one-shot without valid captured authority (%s, %s)',
  async (original, captured) => {
    task('once');
    database.updateTask('fixed-task', {
      schedule_value: original,
      next_run: captured,
    });
    let execute!: () => Promise<void>;
    startSchedulerLoop(
      dependencies((_jid, _id, run) => {
        execute = run;
      }),
    );
    await vi.advanceTimersByTimeAsync(0);
    await execute();
    expect(runContainerAgent).not.toHaveBeenCalled();
    expect(database.getTaskById('fixed-task')).toMatchObject({
      schedule_value: original,
      status: 'paused',
      next_run: captured,
    });
  },
);

it('observes a real GroupQueue close-file refusal and retires the owned scheduled container', async () => {
  task('once');
  const queue = new GroupQueue();
  const write = fs.writeFileSync.bind(fs),
    failure = Object.assign(new Error('fixed close permission refusal'), {
      code: 'EACCES',
    });
  vi.spyOn(fs, 'writeFileSync').mockImplementation((file, ...args) => {
    if (String(file).endsWith('/_close')) throw failure;
    return write(file, ...args);
  });
  let settle!: (output: { status: 'success'; result: null }) => void;
  vi.mocked(runContainerAgent).mockImplementationOnce(
    async (_group, _input, onProcess, onOutput) => {
      onProcess(new ChildProcess(), 'owned-native-close-refusal');
      await onOutput!({ status: 'success', result: 'fixed result' });
      return new Promise((resolve) => {
        settle = resolve;
      });
    },
  );
  const deps = dependencies(queue.enqueueTask.bind(queue));
  deps.queue = queue;
  deps.onProcess = (_jid, proc, name, folder) =>
    queue.registerProcess('tg:123', proc, name, folder);
  startSchedulerLoop(deps);
  await vi.advanceTimersByTimeAsync(10000);
  expect(stopContainer).toHaveBeenCalledExactlyOnceWith(
    'owned-native-close-refusal',
  );
  expect(database.getTaskById('fixed-task')).toMatchObject({
    status: 'completed',
    next_run: null,
    last_result: 'Error: fixed close permission refusal',
  });
  const later = vi.fn(async () => {});
  queue.enqueueTask('tg:123', 'later-task', later);
  await vi.advanceTimersByTimeAsync(0);
  expect(later).toHaveBeenCalledTimes(1);
  settle({ status: 'success', result: null });
  await vi.advanceTimersByTimeAsync(0);
});
