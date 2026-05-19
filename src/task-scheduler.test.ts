import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Mock container-runner BEFORE importing task-scheduler so the scheduler
// picks up the mocked `runContainerAgent`. We can't actually spawn a
// container in a unit test, so this fake calls the streaming callback
// with whatever output the test-under-test wants to simulate.
//
// `vi.hoisted` is required because `vi.mock(...)` itself is hoisted to
// the top of the file — a plain top-level `const` would be accessed
// before initialisation inside the factory.
const { mockRunContainerAgent } = vi.hoisted(() => ({
  mockRunContainerAgent: vi.fn(),
}));
vi.mock('./container-runner.js', () => ({
  runContainerAgent: mockRunContainerAgent,
  writeTasksSnapshot: vi.fn(),
  DEFAULT_SESSION_NAME: 'default',
  MAINTENANCE_SESSION_NAME: 'maintenance',
}));

import {
  _execRawForTests,
  _initTestDatabase,
  _rawQueryForTests,
  clearTaskSessionIdsForGroup,
  createTask,
  deleteTask,
  getActiveLocalScheduledTasks,
  getAllChats,
  getLastBotMessageTimestamp,
  getSession,
  getTaskById,
  pruneCompletedTasks,
  resurrectZombieTasks,
  setSession,
  setTaskNextRun,
  setTaskSessionId,
  storeChatMetadata,
  updateTask,
  updateTaskAfterRun,
} from './db.js';
import {
  COMPLETED_TASK_TTL_MS,
  DORMANT_CRON_THRESHOLD_MS,
  DORMANT_WARN_COOLDOWN_MS,
  PRUNE_INTERVAL_MS,
  _resetSchedulerLoopForTests,
  applyComputeNextRunRemediation,
  computeNextRun,
  computeNextRunDetailed,
  getCompletedTaskTtlMs,
  parseTaskSkill,
  recomputeLocalSchedules,
  startOfTodayInTz,
  startSchedulerLoop,
} from './task-scheduler.js';
import type { ScheduledTask } from './types.js';
import { TIMEZONE } from './config.js';
import { CronExpressionParser } from 'cron-parser';
import { logger } from './logger.js';
import type { ContainerOutput } from './container-runner.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';

describe('task scheduler', () => {
  beforeEach(() => {
    _initTestDatabase();
    _resetSchedulerLoopForTests();
    mockRunContainerAgent.mockClear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('pauses due tasks with invalid group folders to prevent retry churn', async () => {
    createTask({
      id: 'task-invalid-folder',
      group_folder: '../../outside',
      chat_jid: 'bad@g.us',
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-02-22T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 60_000).toISOString(),
      status: 'active',
      created_at: '2026-02-22T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({}),
      queue: { enqueueTask } as any,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    const task = getTaskById('task-invalid-folder');
    expect(task?.status).toBe('paused');
  });

  it('computeNextRun anchors interval tasks to scheduled time to prevent drift', () => {
    const scheduledTime = new Date(Date.now() - 2000).toISOString(); // 2s ago
    const task = {
      id: 'drift-test',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'interval' as const,
      schedule_value: '60000', // 1 minute
      context_mode: 'isolated' as const,
      next_run: scheduledTime,
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    const nextRun = computeNextRun(task);
    expect(nextRun).not.toBeNull();

    // Should be anchored to scheduledTime + 60s, NOT Date.now() + 60s
    const expected = new Date(scheduledTime).getTime() + 60000;
    expect(new Date(nextRun!).getTime()).toBe(expected);
  });

  it('computeNextRun returns null for once-tasks', () => {
    const task = {
      id: 'once-test',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'once' as const,
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated' as const,
      next_run: new Date(Date.now() - 1000).toISOString(),
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    expect(computeNextRun(task)).toBeNull();
  });

  it('computeNextRun honors per-task schedule_timezone for cron (#102)', () => {
    const task = {
      id: 'cron-utc',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron' as const,
      schedule_value: '0 12 * * *', // noon
      schedule_timezone: 'UTC',
      context_mode: 'isolated' as const,
      next_run: null,
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    const nextRun = computeNextRun(task);
    expect(nextRun).not.toBeNull();
    const next = new Date(nextRun!);
    expect(next.getUTCHours()).toBe(12);
    expect(next.getUTCMinutes()).toBe(0);
  });

  it("computeNextRun resolves schedule_timezone='local' against the resolver callback (#456)", () => {
    const task = {
      id: 'cron-local',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron' as const,
      schedule_value: '0 7 * * *', // 7am
      schedule_timezone: 'local',
      context_mode: 'isolated' as const,
      next_run: null,
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    // 7am Chicago in summer is 12:00 UTC (CDT, UTC-5).
    const next = computeNextRun(task, () => 'America/Chicago');
    expect(next).not.toBeNull();
    const utc = new Date(next!);
    // Allow either CDT (12:00) or CST (13:00) depending on date; either
    // way 7am Chicago resolves to a fixed UTC hour, NOT 7am UTC.
    expect(utc.getUTCHours()).not.toBe(7);
  });

  it("schedule_timezone='local' produces different next_run on TZ flip without row mutation (#456)", () => {
    createTask({
      id: 'cron-local-flip',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron',
      schedule_value: '0 7 * * *',
      schedule_timezone: 'local',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner',
    });
    const task = getTaskById('cron-local-flip')!;

    const chicagoNext = computeNextRun(task, () => 'America/Chicago');
    const amsterdamNext = computeNextRun(task, () => 'Europe/Amsterdam');

    expect(chicagoNext).not.toBeNull();
    expect(amsterdamNext).not.toBeNull();
    // Same cron, same row, different zone → different UTC instant.
    expect(chicagoNext).not.toBe(amsterdamNext);
    // Pure: no DB writes. The row's schedule_value/_timezone are
    // unchanged — the entire point of #456 vs the legacy mutation pattern.
    const after = getTaskById('cron-local-flip')!;
    expect(after.schedule_value).toBe('0 7 * * *');
    expect(after.schedule_timezone).toBe('local');
  });

  it("schedule_timezone='local' with no resolver falls back to TIMEZONE (#456)", () => {
    const task = {
      id: 'cron-local-no-resolver',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron' as const,
      schedule_value: '0 7 * * *',
      schedule_timezone: 'local',
      context_mode: 'isolated' as const,
      next_run: null,
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    // No resolver passed → falls through to TIMEZONE. Compare against
    // what cron-parser produces with TIMEZONE directly to avoid pinning
    // a specific zone (the test machine's TZ varies between local dev
    // and CI). The contract is "behaves identically to NULL
    // schedule_timezone with the same cron".
    const next = computeNextRun(task);
    const expected = CronExpressionParser.parse('0 7 * * *', { tz: TIMEZONE })
      .next()
      .toDate()
      .toISOString();
    expect(next).toBe(expected);
  });

  it("schedule_timezone='local' with resolver returning null falls back to TIMEZONE (#456)", () => {
    const task = {
      id: 'cron-local-null-resolver',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron' as const,
      schedule_value: '0 7 * * *',
      schedule_timezone: 'local',
      context_mode: 'isolated' as const,
      next_run: null,
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    // Resolver returns null (e.g. tz_state row absent or schema_version
    // unfamiliar) → falls through to TIMEZONE same as no-resolver case.
    const next = computeNextRun(task, () => null);
    const expected = CronExpressionParser.parse('0 7 * * *', { tz: TIMEZONE })
      .next()
      .toDate()
      .toISOString();
    expect(next).toBe(expected);
  });

  it("schedule_timezone='local' + resolver that throws degrades to TIMEZONE without propagating (#456)", () => {
    // Regression guard: a transient DB read failure inside getCurrentTz
    // (or any future resolver) must NOT abort the scheduler tick. Per
    // `coding-policy: error-handling` § Graceful Fallback.
    const task = {
      id: 'cron-local-throwing-resolver',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron' as const,
      schedule_value: '0 7 * * *',
      schedule_timezone: 'local',
      context_mode: 'isolated' as const,
      next_run: null,
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    const result = computeNextRunDetailed(task, () => {
      throw new Error('transient DB read failure');
    });

    // Falls back to TIMEZONE, no remediation, no propagated throw.
    expect(result.nextRun).not.toBeNull();
    expect(result.remediation).toBeUndefined();
    const expected = CronExpressionParser.parse('0 7 * * *', { tz: TIMEZONE })
      .next()
      .toDate()
      .toISOString();
    expect(result.nextRun).toBe(expected);
  });

  it("schedule_timezone='local' + invalid resolver output does NOT trigger clear-bad-timezone remediation (#456)", () => {
    // Regression guard: a malformed `tz_state.current_tz` (e.g. corruption
    // or a future schema mismatch the reader didn't catch) must NOT cause
    // the row's `schedule_timezone` to be cleared — the bad value isn't
    // on the row, it's on the singleton tz_state. Clearing would silently
    // convert a travel-anchored schedule into a server-TZ schedule.
    createTask({
      id: 'cron-local-bad-resolver',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron',
      schedule_value: '0 7 * * *',
      schedule_timezone: 'local',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner',
    });
    const task = getTaskById('cron-local-bad-resolver')!;

    const result = computeNextRunDetailed(task, () => 'Not/A/Real/Zone');

    // Falls back to TIMEZONE for THIS tick, but NO remediation emitted —
    // the row's 'local' token survives so the next tick can retry once
    // tz_state is fixed.
    expect(result.nextRun).not.toBeNull();
    expect(result.remediation).toBeUndefined();
    // Pure: no DB writes from compute.
    expect(getTaskById('cron-local-bad-resolver')?.schedule_timezone).toBe(
      'local',
    );
  });

  it('computeNextRunDetailed flags clear-bad-timezone when per-task tz is invalid but TIMEZONE works (#102)', () => {
    createTask({
      id: 'cron-bad-tz',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron',
      schedule_value: '0 12 * * *',
      schedule_timezone: 'Not/A/Real/Zone',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner',
    });
    const task = getTaskById('cron-bad-tz')!;

    const result = computeNextRunDetailed(task);

    // First parse fails on bad tz, retry with TIMEZONE succeeds.
    expect(result.nextRun).not.toBeNull();
    expect(result.remediation).toBe('clear-bad-timezone');
    // Pure: no DB writes from compute itself.
    expect(getTaskById('cron-bad-tz')?.schedule_timezone).toBe(
      'Not/A/Real/Zone',
    );
  });

  it("applyComputeNextRunRemediation clears bad tz when row hasn't changed (#102)", () => {
    createTask({
      id: 'cron-apply-clear',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron',
      schedule_value: '0 12 * * *',
      schedule_timezone: 'Not/A/Real/Zone',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner',
    });

    applyComputeNextRunRemediation(
      'cron-apply-clear',
      'clear-bad-timezone',
      '0 12 * * *',
      'Not/A/Real/Zone',
    );

    expect(getTaskById('cron-apply-clear')?.schedule_timezone).toBeFalsy();
  });

  it('applyComputeNextRunRemediation skips remediation when row changed since compute (#102)', () => {
    // Simulate: scheduler observed bad tz, but a concurrent update_task
    // fixed it before the apply step ran. The fix should NOT be clobbered.
    createTask({
      id: 'cron-race',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron',
      schedule_value: '0 12 * * *',
      schedule_timezone: 'UTC', // user just fixed it
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner',
    });

    applyComputeNextRunRemediation(
      'cron-race',
      'clear-bad-timezone',
      '0 12 * * *',
      'Not/A/Real/Zone', // observed when compute ran (before fix)
    );

    // User's fix preserved — remediation skipped.
    expect(getTaskById('cron-race')?.schedule_timezone).toBe('UTC');
  });

  it('computeNextRunDetailed flags pause-broken-cron when both parses fail (#102)', () => {
    createTask({
      id: 'cron-broken',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'cron',
      schedule_value: 'not-a-cron-expression',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner',
    });
    const task = getTaskById('cron-broken')!;

    const result = computeNextRunDetailed(task);

    expect(result.nextRun).toBeNull();
    expect(result.remediation).toBe('pause-broken-cron');
    // Compute is pure — status NOT flipped here.
    expect(getTaskById('cron-broken')?.status).toBe('active');

    // Apply step does the actual flip:
    applyComputeNextRunRemediation(
      'cron-broken',
      'pause-broken-cron',
      'not-a-cron-expression',
      null,
    );
    expect(getTaskById('cron-broken')?.status).toBe('paused');
  });

  it('once-task never reads or writes the maintenance slot session cache (#193 + #336)', async () => {
    // #193 regression: the lunch reminder bled heartbeat-loop language
    // from a 6-day-old maintenance turn because every task on a folder
    // shared the same `sessions[folder][maintenance]` resume slot. The
    // structural fix was: scheduled tasks NEVER use the slot cache.
    // #336 adds per-task session reuse via `scheduled_tasks.session_id`,
    // but explicitly out-of-scope for `schedule_type === 'once'` —
    // one-shots stay fresh-per-fire. So even with #336 landed: a once-
    // task must (a) not read the slot cache for `resume`, (b) not write
    // `newSessionId` to the slot cache, AND (c) not persist
    // `newSessionId` to its row's `session_id` column.
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };

    setSession('main', MAINTENANCE_SESSION_NAME, 'prior-maint-session');

    createTask({
      id: 'group-ctx-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: 'new-maint-session',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': MAIN_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    // Container ran in the maintenance slot — but with NO resume target.
    expect(mockRunContainerAgent).toHaveBeenCalled();
    const containerInput = mockRunContainerAgent.mock.calls[0][1];
    expect(containerInput.sessionId).toBeUndefined();
    expect(containerInput.sessionName).toBe(MAINTENANCE_SESSION_NAME);

    // The seeded prior sessionId is left untouched (no overwrite) and
    // the streamed newSessionId was NOT persisted to the slot cache —
    // the next run also starts fresh.
    expect(getSession('main', MAINTENANCE_SESSION_NAME)).toBe(
      'prior-maint-session',
    );
    // #336: the streamed newSessionId also must NOT have written
    // through to `task.session_id` — once-tasks are out of scope.
    expect(getTaskById('group-ctx-task')?.session_id ?? null).toBeNull();
  });

  it('wipes the just-finished JSONL transcript so orphans do not accumulate (#193)', async () => {
    // Companion to the no-resume test above: because the sessionId is
    // never persisted, neither nukeSession nor cleanup-sessions.sh can
    // find this run's transcript later. The scheduler must call
    // wipeSessionJsonl on every newSessionId observed during the run,
    // from the post-run finally block — i.e. after logTaskRun and the
    // updateTaskAfterRun bookkeeping have been attempted.
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };

    createTask({
      id: 'wipe-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: 'fresh-turn-session',
        } as ContainerOutput);
        return {
          status: 'success',
          result: 'ok',
          newSessionId: 'fresh-turn-session',
        };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    const wipeSpy = vi.fn(() => 1);

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': MAIN_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: wipeSpy,
    });

    await vi.advanceTimersByTimeAsync(10);

    expect(wipeSpy).toHaveBeenCalledWith(
      'main',
      MAINTENANCE_SESSION_NAME,
      'fresh-turn-session',
    );
    // Streaming + terminal both reported the same id; the Set
    // de-dups so wipeSpy fires exactly once.
    expect(wipeSpy).toHaveBeenCalledTimes(1);
  });

  // --- continuation_cycle_id flow-through (#93/#130) ---
  //
  // The scheduler is the bridge between the DB row and the spawned
  // container: when a task row's continuation_cycle_id column is
  // non-NULL, the value must reach the ContainerInput so
  // container-runner can emit the matching env vars. Round-tripping
  // through the scheduler is the load-bearing wiring step — without
  // it, a chained continuation row created by the resumable-cycle
  // helper skill would still spawn a container indistinguishable from
  // a fresh user invocation.

  it('passes continuation_cycle_id from task row through to ContainerInput', async () => {
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-04-21T00:00:00.000Z',
      isMain: true,
    };

    createTask({
      id: 'continuation-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt:
        '[CONTINUATION 2026-04-21 #1] Continue tessl__nightly-housekeeping ...',
      schedule_type: 'once',
      schedule_value: '2026-04-21T00:00:30.000Z',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-04-21T00:00:00.000Z',
      created_by_role: 'owner' as const,
      continuation_cycle_id: '2026-04-21',
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': MAIN_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    expect(mockRunContainerAgent).toHaveBeenCalled();
    const containerInput = mockRunContainerAgent.mock.calls[0][1];
    expect(containerInput.continuationCycleId).toBe('2026-04-21');
  });

  it('omits continuationCycleId on ordinary tasks (no continuation env vars)', async () => {
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-04-21T00:00:00.000Z',
      isMain: true,
    };

    createTask({
      id: 'plain-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'plain scheduled task',
      schedule_type: 'once',
      schedule_value: '2026-04-21T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-04-21T00:00:00.000Z',
      created_by_role: 'owner' as const,
      // continuation_cycle_id intentionally omitted — DB stores NULL.
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': MAIN_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    const containerInput = mockRunContainerAgent.mock.calls[0][1];
    // Must be undefined (not null) — the ContainerInput field is
    // typed as optional string and the container-runner uses a
    // truthiness check that treats `null` the same, but downstream
    // consumers (logging, future code) would observe the wrong
    // shape if the scheduler forwarded SQL NULL verbatim.
    expect(containerInput.continuationCycleId).toBeUndefined();
  });

  it('once-task does NOT persist newSessionId on its row (#336 out-of-scope guard)', async () => {
    // The gating field for #336 reuse is `schedule_type` (recurring vs
    // once), NOT `context_mode` (which is inert on the schema per
    // #193's note). This test pins the once-task path: even when the
    // SDK reports a newSessionId, no DB write to `session_id` should
    // happen, and the slot cache stays untouched.
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };

    createTask({
      id: 'isolated-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: 'should-not-be-persisted',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': MAIN_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    // Once-tasks start fresh — no sessionId passed in.
    const containerInput = mockRunContainerAgent.mock.calls[0][1];
    expect(containerInput.sessionId).toBeUndefined();

    // And the streamed newSessionId was NOT persisted — neither to the
    // slot cache (cross-task bleed prevention) nor to the row's
    // `session_id` column (#336 out-of-scope for once-tasks).
    expect(getSession('main', MAINTENANCE_SESSION_NAME)).toBeUndefined();
    expect(getTaskById('isolated-task')?.session_id ?? null).toBeNull();
  });

  it('streamed scheduled-task result writes a bot row to messages.db', async () => {
    // Regression for jbaruch/nanoclaw#81 root cause: pre-fix, the
    // task-scheduler streaming callback sent to Telegram but never
    // called storeMessage — so every heartbeat/housekeeping cycle
    // reached the user but left no DB row ("ghost heartbeat"). The
    // fix mirrors the ipc.ts send_message pattern: storeMessage is
    // called immediately after sendMessage, with the same row shape.
    // This test fails without the fix.
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const chatJid = 'main@g.us';
    // Seed the chats row so the FK from messages.chat_jid to chats.jid
    // doesn't reject the bot insert. Production has this metadata from
    // the first real user message in the chat; in-test we create it
    // explicitly.
    storeChatMetadata(chatJid, '2026-01-01T00:00:00.000Z', 'Main');

    createTask({
      id: 'store-msg-task',
      group_folder: 'main',
      chat_jid: chatJid,
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const streamedText = 'heartbeat: nothing urgent';
    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: streamedText,
          newSessionId: 'new-maint-session',
        } as ContainerOutput);
        return { status: 'success', result: streamedText };
      },
    );

    const sentTexts: string[] = [];
    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [chatJid]: MAIN_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async (_jid: string, text: string) => {
        sentTexts.push(text);
      },
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    // sendMessage was called with the streamed text.
    expect(sentTexts).toEqual([streamedText]);

    // And — the critical assertion — storeMessage was also called, so
    // `messages.db` now has a bot row for this chat. Without the fix
    // the DB would have no row, exposing the "ghost send" bug #81.
    // getLastBotMessageTimestamp returns undefined when no bot row
    // exists for the chat; a string timestamp when one was written.
    const botTs = getLastBotMessageTimestamp(chatJid, 'bot');
    expect(botTs).toBeTruthy();
  });

  it('streamed scheduled-task writes a bot row even when no prior chats row exists', async () => {
    // The FK from `messages.chat_jid` to `chats.jid` means storeMessage
    // throws if no chats row exists for the target chat (scheduled task
    // firing before any inbound message / metadata sync would create
    // one). Verifies the task-scheduler upserts chat metadata before
    // storeMessage so the bot row actually lands in the DB instead of
    // raising a FOREIGN KEY constraint error and recording the run as
    // an error.
    const FRESH_GROUP = {
      name: 'Fresh',
      folder: 'fresh',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const chatJid = 'fresh-no-metadata@g.us';
    // NOTE: NOT calling storeChatMetadata here. The task-scheduler fix
    // must handle the missing-chats-row case on its own AND must write
    // a correctly-shaped chats row (channel='whatsapp', is_group=true
    // for `@g.us` JIDs) so the chat shows up in `getAvailableGroups()`.

    createTask({
      id: 'fresh-chat-task',
      group_folder: 'fresh',
      chat_jid: chatJid,
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const streamedText = 'first send in a fresh chat';
    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: streamedText,
        } as ContainerOutput);
        return { status: 'success', result: streamedText };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [chatJid]: FRESH_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    // Bot row landed despite no pre-existing chats row. If the
    // storeChatMetadata upsert gets dropped, storeMessage will throw
    // FOREIGN KEY constraint failed and this assertion fails.
    expect(getLastBotMessageTimestamp(chatJid, 'bot')).toBeTruthy();

    // And the chats row itself has the right shape — a `@g.us` JID
    // infers `channel: 'whatsapp'`, `is_group: true`, so the chat
    // appears in `getAvailableGroups()` (which filters on
    // `c.is_group`). Missing/NULL here would hide the group from
    // every downstream consumer — the exact behavior Copilot
    // flagged on #83 round 3.
    const chat = getAllChats().find((c) => c.jid === chatJid);
    expect(chat).toBeTruthy();
    expect(chat!.channel).toBe('whatsapp');
    expect(chat!.is_group).toBe(1);
  });

  it('streamed scheduled-task to a Telegram group upserts chats as telegram+group', async () => {
    // Mirror of the `@g.us` test for the Telegram path: negative id
    // after `tg:` indicates a group/channel, positive indicates a
    // private 1:1 — both should infer `channel: 'telegram'`, and
    // only the negative-id case should set `is_group: true`. Guards
    // against channel-prefix abbreviations (`'tg'`) ever landing in
    // the DB.
    const GROUP_REG = {
      name: 'TG Group',
      folder: 'tgg',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const groupJid = 'tg:-1003000000001';
    const dmJid = 'tg:42';

    for (const [jid, taskId] of [
      [groupJid, 'tg-group-task'],
      [dmJid, 'tg-dm-task'],
    ]) {
      createTask({
        id: taskId,
        group_folder: 'tgg',
        chat_jid: jid,
        prompt: 'run',
        schedule_type: 'once',
        schedule_value: '2026-01-01T00:00:00.000Z',
        context_mode: 'group',
        next_run: new Date(Date.now() - 1000).toISOString(),
        status: 'active',
        created_at: '2026-01-01T00:00:00.000Z',
        created_by_role: 'owner' as const,
      });
    }

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [groupJid]: GROUP_REG, [dmJid]: GROUP_REG }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    const group = getAllChats().find((c) => c.jid === groupJid);
    expect(group).toBeTruthy();
    expect(group!.channel).toBe('telegram');
    expect(group!.is_group).toBe(1);

    const dm = getAllChats().find((c) => c.jid === dmJid);
    expect(dm).toBeTruthy();
    expect(dm!.channel).toBe('telegram');
    expect(dm!.is_group).toBe(0);
  });

  it('streamed scheduled-task to a WhatsApp DM (@s.whatsapp.net) upserts chats as whatsapp+dm', async () => {
    // Extension of the TG-group/TG-DM test for WhatsApp's DM JID
    // shape. Matches the db.ts legacy backfill convention
    // (`@s.whatsapp.net` → whatsapp + is_group=0). Without this
    // branch, a first send to a WA DM would land with channel/is_group
    // NULL and the chat would be invisible to getAvailableGroups().
    const GROUP_REG = {
      name: 'WA DM',
      folder: 'wadm',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const chatJid = '15555555555@s.whatsapp.net';

    createTask({
      id: 'wa-dm-task',
      group_folder: 'wadm',
      chat_jid: chatJid,
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [chatJid]: GROUP_REG }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    const chat = getAllChats().find((c) => c.jid === chatJid);
    expect(chat).toBeTruthy();
    expect(chat!.channel).toBe('whatsapp');
    expect(chat!.is_group).toBe(0);
  });

  it('streamed scheduled-task with all-internal result does NOT write a bot row', async () => {
    // Sibling regression: if the streamed text is ENTIRELY wrapped in
    // `<internal>…</internal>` tags, the stripped `cleanResult` is
    // empty and the `if (cleanResult)` guard short-circuits both the
    // send AND the store. Verifies the store gate is aligned with the
    // send gate — we don't accidentally write empty-content rows.
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const chatJid = 'internal-only@g.us';
    storeChatMetadata(chatJid, '2026-01-01T00:00:00.000Z', 'InternalOnly');

    createTask({
      id: 'internal-only-task',
      group_folder: 'main',
      chat_jid: chatJid,
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: '<internal>debug only — not for user</internal>',
        } as ContainerOutput);
        return { status: 'success', result: '' };
      },
    );

    const sentTexts: string[] = [];
    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [chatJid]: MAIN_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async (_jid: string, text: string) => {
        sentTexts.push(text);
      },
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    // No send happened (cleanResult was empty after strip).
    expect(sentTexts).toEqual([]);
    // And no bot row was written for this chat — the store gate must
    // be aligned with the send gate, otherwise we'd get empty-content
    // rows polluting heartbeat's answered-check accounting.
    expect(getLastBotMessageTimestamp(chatJid, 'bot')).toBeFalsy();
  });

  it('streamed scheduled-task with chat_displayed=true skips chat-echo + storeMessage but still records task_run_logs.result (#581)', async () => {
    // Wrapper scheduled-task skills (nightly-external-sync,
    // entertainment-sync, soul-searching-wrapper) always finish by
    // calling send_message themselves — so the agent-runner sets
    // `chat_displayed: true` on the final IPC envelope. Pre-#581 the
    // agent-runner ALSO collapsed `result` to null, which broke
    // `task_run_logs.result` (silent-success: status=success +
    // result=null, no forensic trail).
    //
    // The fix: agent-runner carries the result text + chat_displayed
    // separately. Task-scheduler must (a) NOT re-send via
    // deps.sendMessage (avoids duplicate user reply), (b) NOT write a
    // bot row via storeMessage (the IPC send_message handler already
    // wrote it), and (c) STILL populate `task_run_logs.result` so
    // observability is preserved.
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const chatJid = 'wrapper-skill@g.us';
    storeChatMetadata(chatJid, '2026-01-01T00:00:00.000Z', 'Wrapper');

    createTask({
      id: 'wrapper-result-task',
      group_folder: 'main',
      chat_jid: chatJid,
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const streamedText = 'closing thought from wrapper skill';
    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: streamedText,
          chat_displayed: true,
        } as ContainerOutput);
        return { status: 'success', result: streamedText };
      },
    );

    const sentTexts: string[] = [];
    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [chatJid]: MAIN_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async (_jid: string, text: string) => {
        sentTexts.push(text);
      },
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    // (a) deps.sendMessage was NOT called — the agent already sent.
    expect(sentTexts).toEqual([]);

    // (b) storeMessage for the bot row was NOT called — the IPC
    // send_message handler already wrote it (would double-row in
    // production messages.db otherwise).
    expect(getLastBotMessageTimestamp(chatJid, 'bot')).toBeFalsy();

    // (c) task_run_logs.result IS populated with the streamed text —
    // observability preserved. This is the regression: pre-#581 this
    // would have been null because the agent-runner suppressed the
    // result field along with the chat-echo signal.
    const { _rawQueryForTests } = await import('./db.js');
    const rows = _rawQueryForTests<{
      status: string;
      result: string | null;
    }>(`SELECT status, result FROM task_run_logs WHERE task_id = ?`, [
      'wrapper-result-task',
    ]);
    expect(rows.length).toBe(1);
    expect(rows[0].status).toBe('success');
    expect(rows[0].result).toContain(streamedText);
  });

  it('computeNextRun skips missed intervals without infinite loop', () => {
    // Task was due 10 intervals ago (missed)
    const ms = 60000;
    const missedBy = ms * 10;
    const scheduledTime = new Date(Date.now() - missedBy).toISOString();

    const task = {
      id: 'skip-test',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'interval' as const,
      schedule_value: String(ms),
      context_mode: 'isolated' as const,
      next_run: scheduledTime,
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    const nextRun = computeNextRun(task);
    expect(nextRun).not.toBeNull();
    // Must be in the future
    expect(new Date(nextRun!).getTime()).toBeGreaterThan(Date.now());
    // Must be aligned to the original schedule grid
    const offset =
      (new Date(nextRun!).getTime() - new Date(scheduledTime).getTime()) % ms;
    expect(offset).toBe(0);
  });

  it('pruneCompletedTasks removes once-tasks whose last_run is older than TTL', () => {
    const t0 = new Date('2026-04-01T00:00:00.000Z').getTime();
    vi.setSystemTime(t0);
    createTask({
      id: 'old-completed',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'old',
      schedule_type: 'once',
      schedule_value: new Date(t0).toISOString(),
      context_mode: 'isolated',
      next_run: new Date(t0).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    // Mimic scheduler's terminal write-back: nextRun=null marks it completed
    // and stamps last_run with the current (mocked) time.
    updateTaskAfterRun('old-completed', null, 'ok');

    // Fast-forward past the TTL boundary; prune should now match.
    vi.setSystemTime(t0 + COMPLETED_TASK_TTL_MS + 60_000);

    const removed = pruneCompletedTasks(COMPLETED_TASK_TTL_MS);
    expect(removed).toBe(1);
    expect(getTaskById('old-completed')).toBeUndefined();
  });

  it('pruneCompletedTasks preserves once-tasks completed within the TTL window', () => {
    createTask({
      id: 'recent-completed',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'recent',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: '2026-01-01T00:00:00.000Z',
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    updateTaskAfterRun('recent-completed', null, 'ok');

    const removed = pruneCompletedTasks(COMPLETED_TASK_TTL_MS);
    expect(removed).toBe(0);
    expect(getTaskById('recent-completed')).toBeDefined();
  });

  it('pruneCompletedTasks never touches active tasks regardless of age', () => {
    const old = new Date(Date.now() - COMPLETED_TASK_TTL_MS * 10).toISOString();
    createTask({
      id: 'stale-active',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'still active',
      schedule_type: 'interval',
      schedule_value: '60000',
      context_mode: 'isolated',
      next_run: old,
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const removed = pruneCompletedTasks(COMPLETED_TASK_TTL_MS);
    expect(removed).toBe(0);
    expect(getTaskById('stale-active')).toBeDefined();
  });
  it('pruneCompletedTasks removes completed once-task with NULL last_run when created_at is past TTL', () => {
    // Reproduces task-1777292573285-gvr365: status=completed, schedule_type=once,
    // last_run=NULL. Pre-fix the `last_run IS NOT NULL` guard left this row
    // lingering forever; the COALESCE(last_run, created_at) version uses the
    // creation timestamp as the fallback age signal.
    const t0 = Date.parse('2026-01-01T00:00:00.000Z');
    vi.setSystemTime(t0);

    createTask({
      id: 'orphan-completed',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'never ran',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: '2026-01-01T00:00:00.000Z',
      status: 'active',
      // Backdate creation past the TTL boundary.
      created_at: new Date(t0 - COMPLETED_TASK_TTL_MS - 60_000).toISOString(),
      created_by_role: 'owner' as const,
    });
    // Mark completed WITHOUT going through updateTaskAfterRun — that's the
    // dispatch-failure shape the bug describes. last_run stays NULL.
    updateTask('orphan-completed', { status: 'completed' });

    const before = getTaskById('orphan-completed');
    expect(before?.status).toBe('completed');
    expect(before?.last_run ?? null).toBeNull();

    const removed = pruneCompletedTasks(COMPLETED_TASK_TTL_MS);
    expect(removed).toBe(1);
    expect(getTaskById('orphan-completed')).toBeUndefined();
  });

  it('getCompletedTaskTtlMs honours NANOCLAW_COMPLETED_TASK_TTL_MS env override', () => {
    // Default — no env var.
    vi.stubEnv('NANOCLAW_COMPLETED_TASK_TTL_MS', '');
    expect(getCompletedTaskTtlMs()).toBe(COMPLETED_TASK_TTL_MS);

    // Valid override.
    vi.stubEnv('NANOCLAW_COMPLETED_TASK_TTL_MS', '60000');
    expect(getCompletedTaskTtlMs()).toBe(60_000);

    // Invalid override falls back to the default, doesn't throw.
    vi.stubEnv('NANOCLAW_COMPLETED_TASK_TTL_MS', 'not-a-number');
    expect(getCompletedTaskTtlMs()).toBe(COMPLETED_TASK_TTL_MS);
    vi.stubEnv('NANOCLAW_COMPLETED_TASK_TTL_MS', '-1');
    expect(getCompletedTaskTtlMs()).toBe(COMPLETED_TASK_TTL_MS);
    // 0 is rejected too — "prune everything immediately" is never what
    // the operator meant, and silently honouring it complicates triage.
    vi.stubEnv('NANOCLAW_COMPLETED_TASK_TTL_MS', '0');
    expect(getCompletedTaskTtlMs()).toBe(COMPLETED_TASK_TTL_MS);

    // End-to-end: with the env override active, prune deletes a row that
    // would NOT have matched the 24h default.
    const t0 = Date.parse('2026-02-01T00:00:00.000Z');
    vi.setSystemTime(t0);
    createTask({
      id: 'env-ttl-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'env',
      schedule_type: 'once',
      schedule_value: '2026-02-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: new Date(t0).toISOString(),
      status: 'active',
      created_at: new Date(t0).toISOString(),
      created_by_role: 'owner' as const,
    });
    updateTaskAfterRun('env-ttl-task', null, 'ok');

    // 5 minutes later — well within the 24h default, well past a 60s override.
    vi.setSystemTime(t0 + 5 * 60_000);
    vi.stubEnv('NANOCLAW_COMPLETED_TASK_TTL_MS', '60000');
    expect(pruneCompletedTasks(getCompletedTaskTtlMs())).toBe(1);

    vi.unstubAllEnvs();
  });

  it('scheduler loop runs prune at most once per PRUNE_INTERVAL_MS even on many ticks', async () => {
    // Spy on pruneCompletedTasks via the scheduler's call-site by counting
    // INFO logs of "Pruned completed once-tasks" — the scheduler only logs
    // when count > 0. Seed two expired completed once-tasks; the first
    // gated call removes both in a single transaction (count=2, one log
    // line). To get a SECOND log line we then seed another expired row
    // and cross the PRUNE_INTERVAL_MS boundary.
    const t0 = Date.parse('2026-03-01T00:00:00.000Z');
    vi.setSystemTime(t0);

    for (const id of ['p1', 'p2']) {
      createTask({
        id,
        group_folder: 'main',
        chat_jid: 'main@g.us',
        prompt: id,
        schedule_type: 'once',
        schedule_value: new Date(t0).toISOString(),
        context_mode: 'isolated',
        next_run: new Date(t0).toISOString(),
        status: 'active',
        created_at: new Date(t0 - COMPLETED_TASK_TTL_MS - 60_000).toISOString(),
        created_by_role: 'owner' as const,
      });
      // Stamp `last_run` in the past so resurrectZombieTasks() at
      // startup skips this row (its predicate requires
      // `last_run IS NULL`). The `COALESCE(last_run, created_at)`
      // prune predicate still matches via the past `last_run`,
      // preserving the test's prune-throttle intent. See #37.
      vi.setSystemTime(t0 - COMPLETED_TASK_TTL_MS - 60_000);
      updateTaskAfterRun(id, null, 'ok');
    }
    vi.setSystemTime(t0);

    const infoSpy = vi.spyOn(logger, 'info');

    startSchedulerLoop({
      registeredGroups: () => ({}),
      queue: { enqueueTask: vi.fn(), closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    // SCHEDULER_POLL_INTERVAL is 60s — advance in poll-sized steps so
    // each iteration fires a real loop tick. 30 ticks = 30 minutes,
    // still well below the 1h PRUNE_INTERVAL_MS gate. Only the first
    // tick (lastPruneAt=0) should pass the gate.
    for (let i = 0; i < 30; i += 1) {
      await vi.advanceTimersByTimeAsync(60_000);
    }

    const prunedLogCalls = infoSpy.mock.calls.filter(
      (call) =>
        typeof call[1] === 'string' &&
        call[1] === 'Pruned completed once-tasks',
    );
    // 30 ticks at 60s stride covered ~30 minutes of mocked time, well
    // under PRUNE_INTERVAL_MS (1h). The throttle means only the very
    // first tick (lastPruneAt=0) passes the gate → exactly one
    // "Pruned" log line, even though both seeded rows are eligible.
    expect(prunedLogCalls.length).toBe(1);
    // Both seeded rows were eligible at the first gated tick, so a single
    // prune transaction took both out.
    expect(getTaskById('p1')).toBeUndefined();
    expect(getTaskById('p2')).toBeUndefined();

    // Seed another expired completed once-task so the next gated entry has
    // something to log, then cross the PRUNE_INTERVAL_MS boundary.
    const tNow = Date.now();
    createTask({
      id: 'p3',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'p3',
      schedule_type: 'once',
      schedule_value: new Date(tNow).toISOString(),
      context_mode: 'isolated',
      next_run: new Date(tNow).toISOString(),
      status: 'active',
      created_at: new Date(tNow - COMPLETED_TASK_TTL_MS - 60_000).toISOString(),
      created_by_role: 'owner' as const,
    });
    // See p1/p2 above on the time-shift trick — same reason (#37).
    vi.setSystemTime(tNow - COMPLETED_TASK_TTL_MS - 60_000);
    updateTaskAfterRun('p3', null, 'ok');
    vi.setSystemTime(tNow);

    await vi.advanceTimersByTimeAsync(PRUNE_INTERVAL_MS);

    const prunedAfterBoundary = infoSpy.mock.calls.filter(
      (call) =>
        typeof call[1] === 'string' &&
        call[1] === 'Pruned completed once-tasks',
    );
    expect(prunedAfterBoundary.length).toBeGreaterThanOrEqual(2);
    expect(getTaskById('p3')).toBeUndefined();

    infoSpy.mockRestore();
  });

  it('dormant recurring task (last_run > threshold, status=active) emits a warn log without deletion', async () => {
    // A cron task that hasn't fired in 8 days while still status=active
    // points at a dispatch problem. The scheduler should log a warning so
    // a human notices, but must NOT auto-delete the row — that would
    // silently lose the schedule.
    const t0 = Date.parse('2026-04-01T00:00:00.000Z');
    vi.setSystemTime(t0);

    createTask({
      id: 'dormant-cron',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'morning brief',
      schedule_type: 'cron',
      schedule_value: '0 8 * * *',
      context_mode: 'group',
      next_run: new Date(t0 + 60_000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    // Stamp last_run > DORMANT_CRON_THRESHOLD_MS in the past.
    updateTaskAfterRun(
      'dormant-cron',
      new Date(t0 + 60_000).toISOString(),
      'ok',
    );
    // updateTaskAfterRun stamps last_run to "now". Roll the clock forward
    // past the dormant threshold so the row qualifies on the next tick.
    vi.setSystemTime(t0 + DORMANT_CRON_THRESHOLD_MS + 60 * 60_000);

    const warnSpy = vi.spyOn(logger, 'warn');

    startSchedulerLoop({
      registeredGroups: () => ({}),
      queue: { enqueueTask: vi.fn(), closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    const dormantWarnCall = warnSpy.mock.calls.find(
      (call) =>
        typeof call[1] === 'string' &&
        call[1].startsWith('Dormant recurring task'),
    );
    expect(dormantWarnCall).toBeDefined();
    expect((dormantWarnCall![0] as { taskId: string }).taskId).toBe(
      'dormant-cron',
    );

    // The row is still in the database — visibility-only, no delete.
    expect(getTaskById('dormant-cron')).toBeDefined();
    warnSpy.mockRestore();
  });

  it('dormant warn is rate-limited per task to once per DORMANT_WARN_COOLDOWN_MS', async () => {
    // Without per-task dedup the prune sweep (PRUNE_INTERVAL_MS = 1h)
    // would re-warn the same dormant task 24 times a day. Assert that
    // back-to-back prune cycles only emit one warn for the same id, and
    // that the warn fires again once the cooldown elapses.
    const t0 = Date.parse('2026-04-01T00:00:00.000Z');
    vi.setSystemTime(t0);

    createTask({
      id: 'dormant-dedup',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'morning brief',
      schedule_type: 'cron',
      schedule_value: '0 8 * * *',
      context_mode: 'group',
      next_run: new Date(t0 + 60_000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    updateTaskAfterRun(
      'dormant-dedup',
      new Date(t0 + 60_000).toISOString(),
      'ok',
    );
    // Move past the dormancy threshold so the task qualifies.
    vi.setSystemTime(t0 + DORMANT_CRON_THRESHOLD_MS + 60 * 60_000);

    const warnSpy = vi.spyOn(logger, 'warn');

    startSchedulerLoop({
      registeredGroups: () => ({}),
      queue: { enqueueTask: vi.fn(), closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });
    await vi.advanceTimersByTimeAsync(10);

    const matches = () =>
      warnSpy.mock.calls.filter(
        (call) =>
          typeof call[1] === 'string' &&
          call[1].startsWith('Dormant recurring task') &&
          (call[0] as { taskId: string }).taskId === 'dormant-dedup',
      ).length;

    expect(matches()).toBe(1);

    // A second prune cycle inside the cooldown window must NOT warn again.
    await vi.advanceTimersByTimeAsync(PRUNE_INTERVAL_MS + 10);
    expect(matches()).toBe(1);

    // After the cooldown elapses, the next prune cycle warns once more.
    await vi.advanceTimersByTimeAsync(DORMANT_WARN_COOLDOWN_MS);
    expect(matches()).toBe(2);

    warnSpy.mockRestore();
  });

  it('dormant warn map drops entries for tasks that no longer exist', async () => {
    // The dedup map is keyed by task id; if a task is deleted between
    // prune cycles, its entry must be cleaned up so the map can't grow
    // unbounded over the lifetime of the process. We can't poke at the
    // map directly, so we assert the externally-visible behaviour: a
    // re-created task with the same id (after deletion) gets a fresh
    // warn even inside the cooldown window.
    const t0 = Date.parse('2026-04-01T00:00:00.000Z');
    vi.setSystemTime(t0);

    createTask({
      id: 'dormant-vanish',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'will be deleted',
      schedule_type: 'cron',
      schedule_value: '0 8 * * *',
      context_mode: 'group',
      next_run: new Date(t0 + 60_000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    updateTaskAfterRun(
      'dormant-vanish',
      new Date(t0 + 60_000).toISOString(),
      'ok',
    );
    vi.setSystemTime(t0 + DORMANT_CRON_THRESHOLD_MS + 60 * 60_000);

    const warnSpy = vi.spyOn(logger, 'warn');

    startSchedulerLoop({
      registeredGroups: () => ({}),
      queue: { enqueueTask: vi.fn(), closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });
    await vi.advanceTimersByTimeAsync(10);

    const matches = () =>
      warnSpy.mock.calls.filter(
        (call) =>
          typeof call[1] === 'string' &&
          call[1].startsWith('Dormant recurring task') &&
          (call[0] as { taskId: string }).taskId === 'dormant-vanish',
      ).length;

    expect(matches()).toBe(1);

    // Delete the task and run another prune cycle — this triggers the
    // stale-id cleanup path inside the dormant-warn loop.
    deleteTask('dormant-vanish');
    await vi.advanceTimersByTimeAsync(PRUNE_INTERVAL_MS + 10);
    expect(matches()).toBe(1);

    // Re-create the task with the same id, still inside the original
    // cooldown window. If the map entry was correctly pruned the new
    // dormant task warns; if the map leaked, this would stay at 1.
    //
    // updateTaskAfterRun stamps `last_run = Date.now()` unconditionally,
    // so to seed a dormant `last_run` we briefly roll the system clock
    // back to `t0`, call updateTaskAfterRun (which records that as
    // last_run), then restore the clock to where the prune-cycle test
    // expects it. The 2nd argument is `nextRun`, not last_run.
    const restoreTime = Date.now();
    createTask({
      id: 'dormant-vanish',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'reborn',
      schedule_type: 'cron',
      schedule_value: '0 8 * * *',
      context_mode: 'group',
      next_run: new Date(restoreTime + 60_000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    vi.setSystemTime(t0);
    updateTaskAfterRun(
      'dormant-vanish',
      new Date(restoreTime + 60_000).toISOString(),
      'ok',
    );
    vi.setSystemTime(restoreTime);
    await vi.advanceTimersByTimeAsync(PRUNE_INTERVAL_MS + 10);
    expect(matches()).toBe(2);

    warnSpy.mockRestore();
  });

  it('freshly-created recurring task with NULL last_run is NOT flagged dormant', async () => {
    // A cron created moments ago — last_run is NULL because it simply
    // hasn't been due yet, not because dispatch is broken. The dormant
    // scan should NOT warn until the task's age (created_at) crosses
    // DORMANT_CRON_THRESHOLD_MS. Pre-fix, the SQL used
    // `last_run IS NULL OR last_run < ?` which matched any NULL row
    // regardless of age and produced a false positive on the very first
    // scheduler tick.
    const t0 = Date.parse('2026-04-01T00:00:00.000Z');
    vi.setSystemTime(t0);

    createTask({
      id: 'fresh-cron',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'morning brief',
      schedule_type: 'cron',
      schedule_value: '0 8 * * *',
      context_mode: 'group',
      next_run: new Date(t0 + 60_000).toISOString(),
      status: 'active',
      // created_at = "now" — fresh task, well within DORMANT_CRON_THRESHOLD_MS.
      created_at: new Date(t0).toISOString(),
      created_by_role: 'owner' as const,
    });
    // Deliberately do NOT call updateTaskAfterRun — last_run stays NULL,
    // mirroring a never-run cron.

    const warnSpy = vi.spyOn(logger, 'warn');

    startSchedulerLoop({
      registeredGroups: () => ({}),
      queue: { enqueueTask: vi.fn(), closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });
    // First tick passes the prune gate (lastPruneAt=0). The dormant
    // sweep runs; with the COALESCE fix it must NOT flag this task.
    await vi.advanceTimersByTimeAsync(10);

    const dormantWarns = warnSpy.mock.calls.filter(
      (call) =>
        typeof call[1] === 'string' &&
        call[1].startsWith('Dormant recurring task') &&
        (call[0] as { taskId: string }).taskId === 'fresh-cron',
    );
    expect(dormantWarns.length).toBe(0);

    // Sanity check: once the row's `created_at` is older than the
    // dormancy threshold, it DOES qualify — confirming the fix didn't
    // accidentally exclude all NULL-last_run rows. Roll the clock past
    // the threshold and re-open the prune gate.
    vi.setSystemTime(t0 + DORMANT_CRON_THRESHOLD_MS + 60 * 60_000);
    await vi.advanceTimersByTimeAsync(PRUNE_INTERVAL_MS + 10);
    const dormantWarnsAfter = warnSpy.mock.calls.filter(
      (call) =>
        typeof call[1] === 'string' &&
        call[1].startsWith('Dormant recurring task') &&
        (call[0] as { taskId: string }).taskId === 'fresh-cron',
    );
    expect(dormantWarnsAfter.length).toBe(1);

    warnSpy.mockRestore();
  });

  it('resurrectZombieTasks flips pre-advanced once-tasks back to active so getDueTasks can pick them up', () => {
    // Pre-advance shape: status='completed', last_run=NULL, next_run set,
    // schedule_type='once'. This is what the scheduler writes between
    // the pre-advance UPDATE and the queue.enqueueTask call. If
    // dispatch is dropped (host crash, queue shut down mid-tick), the
    // row sits here forever — pruneCompletedTasks would eventually GC
    // it but the schedule is lost. resurrect puts it back in front of
    // the loop.
    createTask({
      id: 'zombie-once',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'should have run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: '2026-01-01T00:00:00.000Z',
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    // Reproduce the pre-advance write WITHOUT the matching
    // updateTaskAfterRun (which would stamp last_run).
    updateTask('zombie-once', { status: 'completed' });
    expect(getTaskById('zombie-once')?.status).toBe('completed');
    expect(getTaskById('zombie-once')?.last_run ?? null).toBeNull();

    const resurrected = resurrectZombieTasks();
    expect(resurrected).toEqual(['zombie-once']);
    expect(getTaskById('zombie-once')?.status).toBe('active');
    expect(getTaskById('zombie-once')?.next_run).toBe(
      '2026-01-01T00:00:00.000Z',
    );
  });

  it('resurrectZombieTasks leaves genuinely-completed once-tasks alone (last_run is set)', () => {
    // A task that ran successfully has both last_run set and
    // status='completed' (updateTaskAfterRun stamps both). The
    // resurrect query specifically requires `last_run IS NULL`, so this
    // row must NOT be resurrected.
    createTask({
      id: 'ran-properly',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'ran ok',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: '2026-01-01T00:00:00.000Z',
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    updateTaskAfterRun('ran-properly', null, 'ok');
    expect(getTaskById('ran-properly')?.status).toBe('completed');
    expect(getTaskById('ran-properly')?.last_run).toBeTruthy();

    const resurrected = resurrectZombieTasks();
    expect(resurrected).toEqual([]);
    expect(getTaskById('ran-properly')?.status).toBe('completed');
  });

  it('resurrectZombieTasks ignores recurring tasks even if they somehow reach status=completed', () => {
    // computeNextRun never returns null for cron/interval tasks, so
    // they can't legitimately reach status='completed' through the
    // scheduler. The schedule_type='once' clause is defensive — verify
    // it holds even if a recurring row is force-set to completed (e.g.
    // operator action, db migration mishap).
    createTask({
      id: 'cron-frozen',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'daily',
      schedule_type: 'cron',
      schedule_value: '0 9 * * *',
      context_mode: 'group',
      next_run: '2026-01-02T17:00:00.000Z',
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    updateTask('cron-frozen', { status: 'completed' });

    const resurrected = resurrectZombieTasks();
    expect(resurrected).toEqual([]);
    expect(getTaskById('cron-frozen')?.status).toBe('completed');
  });
});

describe('parseTaskSkill', () => {
  // The orchestrator prepends `Skill(skill: "tessl__heartbeat")` (and
  // similar) directly into prompts created by `syncNonMainHeartbeat` /
  // the housekeeping/morning-brief setup paths. The shape is fixed
  // (literal SDK invocation syntax), so the regex extraction is
  // appropriate per `script-delegation.md`'s "fully enumerable"
  // carve-out.

  it('extracts the skill name from a heartbeat-shaped prompt (mid-prompt call)', () => {
    expect(
      parseTaskSkill(
        'MANDATORY FIRST ACTION: Call Skill(skill: "tessl__heartbeat") BEFORE doing anything else.',
      ),
    ).toBe('tessl__heartbeat');
  });

  it('handles single-quoted invocations', () => {
    expect(
      parseTaskSkill("Run Skill(skill: 'tessl__nightly-housekeeping') now."),
    ).toBe('tessl__nightly-housekeeping');
  });

  it('tolerates extra whitespace around the colon', () => {
    expect(parseTaskSkill('Skill(  skill:  "tessl__morning-brief" )')).toBe(
      'tessl__morning-brief',
    );
  });

  it('returns the FIRST match when a prompt mentions multiple skills', () => {
    expect(
      parseTaskSkill(
        'Skill(skill: "tessl__heartbeat") then later Skill(skill: "tessl__morning-brief")',
      ),
    ).toBe('tessl__heartbeat');
  });

  it('returns undefined for raw-text scheduled tasks (one-shot reminders)', () => {
    expect(
      parseTaskSkill('Tell Baruch about lunch in 3 hours'),
    ).toBeUndefined();
  });

  it('returns undefined when the prompt mentions Skill in prose only', () => {
    // Prose mention without the invocation parentheses must not match.
    // Otherwise `the Skill: foo skill` would yield `foo`.
    expect(
      parseTaskSkill("Document the Skill: foo workflow in tomorrow's notes"),
    ).toBeUndefined();
  });
});

describe('per-task session_id reuse (#336)', () => {
  // Each test below drives the scheduler against a recurring task and
  // asserts on three observable surfaces:
  //   1. ContainerInput.sessionId — what gets passed as `resume:` to
  //      runContainerAgent
  //   2. scheduled_tasks.session_id (via getTaskById) — what's
  //      persisted for the next fire
  //   3. wipeSessionJsonl call args — which JSONL transcripts get
  //      cleaned up in the post-run finally
  // Together these pin the contract: recurring tasks reuse the same
  // session id across fires while orphans (rotated mid-run, or stale
  // after a nuke) get cleaned up off disk.

  const RECURRING_GROUP = {
    name: 'Main',
    folder: 'main',
    trigger: 'always',
    added_at: '2026-01-01T00:00:00.000Z',
    isMain: true,
  };

  beforeEach(() => {
    _initTestDatabase();
    _resetSchedulerLoopForTests();
    mockRunContainerAgent.mockClear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  /**
   * Helper: drive the scheduler through one fire of a single recurring
   * task and return the captured ContainerInput + the wipeSpy. Reduces
   * boilerplate across tests below — every case sets up the same four
   * scheduler dependencies (registeredGroups, queue, onProcess,
   * sendMessage) the same way; the per-test variation lives in the
   * mockRunContainerAgent implementation and the task row state.
   */
  async function fireOnce(): Promise<{
    containerInput: { sessionId?: string; sessionName?: string };
    wipeSpy: ReturnType<typeof vi.fn>;
  }> {
    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );
    const wipeSpy = vi.fn(() => 1);
    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': RECURRING_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: wipeSpy,
    });
    await vi.advanceTimersByTimeAsync(10);
    const containerInput = mockRunContainerAgent.mock.calls[0]?.[1];
    return { containerInput, wipeSpy };
  }

  it('first fire of a recurring task persists newSessionId for next fire', async () => {
    createTask({
      id: 'heartbeat-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'Skill(skill: "tessl__heartbeat")',
      schedule_type: 'interval',
      schedule_value: '1800000',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: 'sdk-issued-id-A',
        } as ContainerOutput);
        return {
          status: 'success',
          result: 'ok',
          newSessionId: 'sdk-issued-id-A',
        };
      },
    );

    const { containerInput, wipeSpy } = await fireOnce();

    // No prior id → fresh start (no resume).
    expect(containerInput.sessionId).toBeUndefined();
    // Newly-issued id persisted for next fire.
    expect(getTaskById('heartbeat-task')?.session_id).toBe('sdk-issued-id-A');
    // Live id is NOT wiped — the next fire needs it on disk.
    expect(wipeSpy).not.toHaveBeenCalled();
  });

  it('subsequent fire of a recurring task resumes the persisted session_id', async () => {
    createTask({
      id: 'heartbeat-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'Skill(skill: "tessl__heartbeat")',
      schedule_type: 'interval',
      schedule_value: '1800000',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    setTaskSessionId('heartbeat-task', 'persisted-id-X');

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        // Clean resume: SDK loaded X, kept it, and re-emits X.
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: 'persisted-id-X',
        } as ContainerOutput);
        return {
          status: 'success',
          result: 'ok',
          newSessionId: 'persisted-id-X',
        };
      },
    );

    const { containerInput, wipeSpy } = await fireOnce();

    // Persisted id is passed as `resume:`.
    expect(containerInput.sessionId).toBe('persisted-id-X');
    // Row stays at the same id (no rotation).
    expect(getTaskById('heartbeat-task')?.session_id).toBe('persisted-id-X');
    // The live id is NOT wiped — must survive for the next fire.
    expect(wipeSpy).not.toHaveBeenCalled();
  });

  it('SDK rotation mid-run wipes the orphan and persists the new id', async () => {
    // Edge case: SDK loaded session X, decided to rotate to Y mid-
    // stream (e.g. transcript pruning under the hood). Y's transcript
    // is the new live one; X's is now orphan and must be wiped or
    // it leaks on disk forever.
    createTask({
      id: 'heartbeat-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'Skill(skill: "tessl__heartbeat")',
      schedule_type: 'interval',
      schedule_value: '1800000',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    setTaskSessionId('heartbeat-task', 'rotated-from-X');

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        // Mid-run rotation: SDK reports X first, then switches to Y.
        await onOutput({
          status: 'success',
          result: null,
          newSessionId: 'rotated-from-X',
        } as ContainerOutput);
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: 'rotated-to-Y',
        } as ContainerOutput);
        return {
          status: 'success',
          result: 'ok',
          newSessionId: 'rotated-to-Y',
        };
      },
    );

    const { containerInput, wipeSpy } = await fireOnce();

    // Started with X (passed as resume:).
    expect(containerInput.sessionId).toBe('rotated-from-X');
    // Row ends at Y (last-write-wins).
    expect(getTaskById('heartbeat-task')?.session_id).toBe('rotated-to-Y');
    // X's orphan transcript got wiped.
    expect(wipeSpy).toHaveBeenCalledWith(
      'main',
      MAINTENANCE_SESSION_NAME,
      'rotated-from-X',
    );
    // Y is alive — never wiped.
    const wipeArgs = wipeSpy.mock.calls.map((c) => c[2]);
    expect(wipeArgs).not.toContain('rotated-to-Y');
  });

  it('cron-task gets the same reuse contract as interval-task', async () => {
    // Cron and interval are both "recurring" per #336 — neither is the
    // out-of-scope `once`. This test pins that the gating is on
    // `schedule_type !== 'once'`, not on `interval` specifically.
    createTask({
      id: 'cron-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'Skill(skill: "tessl__nightly-housekeeping")',
      schedule_type: 'cron',
      schedule_value: '0 3 * * *',
      schedule_timezone: 'UTC',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: 'cron-issued-id',
        } as ContainerOutput);
        return {
          status: 'success',
          result: 'ok',
          newSessionId: 'cron-issued-id',
        };
      },
    );

    await fireOnce();

    expect(getTaskById('cron-task')?.session_id).toBe('cron-issued-id');
  });

  it('different recurring tasks in the same group keep independent session_ids (no #193 bleed)', async () => {
    // The #193 cross-task bleed was: a lunch reminder picked up a
    // heartbeat-loop's terminal message because they shared the slot
    // cache. #336's design explicitly avoids reintroducing this — each
    // task's `session_id` is keyed on the row's id, so two distinct
    // recurring tasks in the same group end up with two distinct SDK
    // sessions on disk.
    createTask({
      id: 'task-A',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'Skill(skill: "tessl__heartbeat")',
      schedule_type: 'interval',
      schedule_value: '1800000',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    createTask({
      id: 'task-B',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'Skill(skill: "tessl__morning-brief")',
      schedule_type: 'cron',
      schedule_value: '0 7 * * *',
      schedule_timezone: 'UTC',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    let callCount = 0;
    mockRunContainerAgent.mockImplementation(
      async (_group, input, _onProc, onOutput) => {
        callCount++;
        // Each task's fire emits a distinct id; assert the test mock
        // stays consistent so a regression where one task gets the
        // other's id surfaces here rather than as a quiet bleed.
        const issuedId = input.prompt.includes('heartbeat')
          ? 'id-for-task-A'
          : 'id-for-task-B';
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: issuedId,
        } as ContainerOutput);
        return { status: 'success', result: 'ok', newSessionId: issuedId };
      },
    );

    const enqueueTask = vi.fn(
      (_jid: string, _id: string, _name: string, fn: () => Promise<void>) => {
        void fn();
      },
    );
    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': RECURRING_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });
    await vi.advanceTimersByTimeAsync(10);

    expect(callCount).toBe(2);
    expect(getTaskById('task-A')?.session_id).toBe('id-for-task-A');
    expect(getTaskById('task-B')?.session_id).toBe('id-for-task-B');
  });

  it('clearTaskSessionIdsForGroup wipes all rows under one group, leaves other groups untouched', async () => {
    // The nuke-side helper that `nukeSession('maintenance' | 'all')`
    // calls. Direct DB-level test — doesn't drive the scheduler.
    createTask({
      id: 'main-task-1',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'p',
      schedule_type: 'interval',
      schedule_value: '1800000',
      context_mode: 'isolated',
      next_run: new Date().toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    createTask({
      id: 'main-task-2',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'p',
      schedule_type: 'cron',
      schedule_value: '0 7 * * *',
      schedule_timezone: 'UTC',
      context_mode: 'isolated',
      next_run: new Date().toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    createTask({
      id: 'other-task',
      group_folder: 'other',
      chat_jid: 'other@g.us',
      prompt: 'p',
      schedule_type: 'interval',
      schedule_value: '1800000',
      context_mode: 'isolated',
      next_run: new Date().toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    setTaskSessionId('main-task-1', 'id-1');
    setTaskSessionId('main-task-2', 'id-2');
    setTaskSessionId('other-task', 'id-other');

    const cleared = clearTaskSessionIdsForGroup('main');

    expect(cleared).toBe(2);
    expect(getTaskById('main-task-1')?.session_id ?? null).toBeNull();
    expect(getTaskById('main-task-2')?.session_id ?? null).toBeNull();
    // Other group untouched — a maintenance nuke on `main` doesn't
    // bleed into `other`.
    expect(getTaskById('other-task')?.session_id).toBe('id-other');
  });

  it('clearTaskSessionIdsForGroup is idempotent — no rows touched on second call', async () => {
    createTask({
      id: 'task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'p',
      schedule_type: 'interval',
      schedule_value: '1800000',
      context_mode: 'isolated',
      next_run: new Date().toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    setTaskSessionId('task', 'id-1');

    expect(clearTaskSessionIdsForGroup('main')).toBe(1);
    // Second call: nothing to clear, returns 0 (the WHERE filters
    // already-NULL rows).
    expect(clearTaskSessionIdsForGroup('main')).toBe(0);
  });
});

describe('interval cadence end-to-end (#438)', () => {
  // Pre-#438 the scheduler advanced `next_run` twice per fire: once
  // pre-dispatch in the loop, again post-completion in `runTask`. The
  // post-completion compute step re-fetched the row (already at
  // `N + ms`) and added another `ms`, so every interval task ran at
  // half its configured cadence. Cron tasks were unaffected because
  // their compute step parses the cron expression rather than
  // anchoring on `task.next_run`.
  //
  // These tests drive the loop end-to-end through `startSchedulerLoop`
  // + `fireOnce` and assert the delta between consecutive `next_run`
  // values. Today the assertion is `delta === ms`; before the fix it
  // was `delta === 2 * ms`.

  const RECURRING_GROUP = {
    name: 'Main',
    folder: 'main',
    trigger: 'always',
    added_at: '2026-01-01T00:00:00.000Z',
    isMain: true,
  };

  beforeEach(() => {
    _initTestDatabase();
    _resetSchedulerLoopForTests();
    mockRunContainerAgent.mockClear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  async function fireOnce(): Promise<void> {
    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );
    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': RECURRING_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: vi.fn(() => 1),
    });
    await vi.advanceTimersByTimeAsync(10);
  }

  it('interval task advances next_run by exactly ms per fire (not 2 * ms)', async () => {
    // Anchor `next_run` to a fixed timestamp 1 second in the past so
    // the row is due. With ms = 1_800_000 (30 min), the post-fire
    // `next_run` should land at `t0 + ms` — pre-#438 it landed at
    // `t0 + 2 * ms` (60-min cadence on a 30-min schedule).
    const ms = 1_800_000;
    const t0 = Date.now() - 1000;
    const initialNextRun = new Date(t0).toISOString();
    createTask({
      id: 'interval-cadence',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'noop',
      schedule_type: 'interval',
      schedule_value: String(ms),
      context_mode: 'isolated',
      next_run: initialNextRun,
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    mockRunContainerAgent.mockImplementation(async () => ({
      status: 'success',
      result: 'ok',
      newSessionId: 'sid',
    }));

    await fireOnce();

    const after = getTaskById('interval-cadence');
    expect(after).toBeDefined();
    const delta =
      new Date(after!.next_run!).getTime() - new Date(initialNextRun).getTime();
    expect(delta).toBe(ms);
  });

  it('dispatchedTaskIds prevents the same interval task from being picked up twice mid-fire', async () => {
    // The in-memory dispatched filter replaces the pre-advance write
    // that pre-#438 prevented duplicate dispatches by mutating
    // `next_run`. With the pre-advance gone, the row stays due until
    // `runTask` finishes — without the filter, every scheduler tick
    // during a long-running fire would re-enqueue. We hold the
    // container call open so a second tick is guaranteed to land
    // mid-fire, and assert exactly one dispatch happened.
    const ms = 1_800_000;
    const t0 = Date.now() - 1000;
    createTask({
      id: 'interval-no-dup',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'noop',
      schedule_type: 'interval',
      schedule_value: String(ms),
      context_mode: 'isolated',
      next_run: new Date(t0).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    let releaseContainer!: () => void;
    const containerHeld = new Promise<void>((resolve) => {
      releaseContainer = resolve;
    });
    mockRunContainerAgent.mockImplementation(async () => {
      await containerHeld;
      return { status: 'success', result: 'ok', newSessionId: 'sid' };
    });

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );
    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': RECURRING_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: vi.fn(() => 1),
    });

    // First scheduler tick — picks up the due row, dispatches once.
    await vi.advanceTimersByTimeAsync(10);
    expect(enqueueTask).toHaveBeenCalledTimes(1);

    // Second scheduler tick while the container is still held —
    // without `dispatchedTaskIds` the row would be re-dispatched.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(enqueueTask).toHaveBeenCalledTimes(1);

    // Release the container; finally clears the dispatched set so a
    // future tick (after `next_run` advances) can dispatch again.
    releaseContainer();
    await vi.advanceTimersByTimeAsync(10);
  });

  it('skips dispatch when remediation paused the row mid-tick (cron with broken expression)', async () => {
    // Copilot review on PR #446: a row with an unparseable cron
    // expression returns `remediation: 'pause-broken-cron'` from
    // `computeNextRunDetailed`; `applyComputeNextRunRemediation`
    // flips the DB row to `status='paused'`, but `currentTask` in
    // memory still says 'active' because we read it before the
    // remediation. Without an explicit short-circuit the loop would
    // dispatch the very task we just paused. This test sets up a
    // broken-cron row and asserts `enqueueTask` never fires.
    createTask({
      id: 'broken-cron',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'noop',
      schedule_type: 'cron',
      schedule_value: 'totally not a cron expression',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );
    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': RECURRING_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: vi.fn(() => 1),
    });
    await vi.advanceTimersByTimeAsync(10);

    // No dispatch — the remediation-paused row was correctly
    // skipped. Without the short-circuit the row would have been
    // enqueued exactly once (and then mockRunContainerAgent would
    // have run on a row whose DB status is 'paused').
    expect(enqueueTask).not.toHaveBeenCalled();

    // And the row is in fact paused on disk now.
    expect(getTaskById('broken-cron')?.status).toBe('paused');
  });

  it('clears dispatchedTaskIds when enqueueTask throws synchronously (does not wedge the row)', async () => {
    // OpenAI policy review on PR #446: if `deps.queue.enqueueTask`
    // throws synchronously, the runTask wrapper's `.finally` never
    // runs and the row stays in `dispatchedTaskIds` forever — the
    // scheduler would skip it on every subsequent tick. The fix
    // wraps the enqueue call in try/catch and clears the bookkeeping
    // before re-throwing so the next tick can retry. This test
    // exercises that path: first tick throws on enqueue, second tick
    // (with enqueue restored) successfully dispatches.
    const ms = 1_800_000;
    const t0 = Date.now() - 1000;
    createTask({
      id: 'interval-enqueue-throws',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'noop',
      schedule_type: 'interval',
      schedule_value: String(ms),
      context_mode: 'isolated',
      next_run: new Date(t0).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    mockRunContainerAgent.mockImplementation(async () => ({
      status: 'success',
      result: 'ok',
      newSessionId: 'sid',
    }));

    let firstTickEnqueueCalled = false;
    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        if (!firstTickEnqueueCalled) {
          firstTickEnqueueCalled = true;
          throw new Error('queue saturation simulation');
        }
        void fn();
      },
    );
    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': RECURRING_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: vi.fn(() => 1),
    });

    // First tick — enqueue throws. The terminal scheduler catch
    // swallows the throw and the loop survives; importantly, the
    // dispatched-set cleanup must have run BEFORE re-throw so the
    // row isn't wedged.
    await vi.advanceTimersByTimeAsync(10);
    expect(enqueueTask).toHaveBeenCalledTimes(1);

    // Second tick — enqueue is healthy now; the row must still be
    // due (we never advanced `next_run` because we never completed),
    // and dispatchedTaskIds must NOT contain the id, so the loop
    // picks it up and dispatches it. Without the catch+cleanup the
    // row would stay skipped forever.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(enqueueTask).toHaveBeenCalledTimes(2);

    // Drain the run so the test doesn't leave open promises.
    await vi.advanceTimersByTimeAsync(10);
  });

  // --- killed-status reclassification (#496) ---
  //
  // When the periodic `tessl_update` calls `closeAllActiveContainers`
  // mid-run, the agent-runner's hard-exit watchdog fires
  // `process.exit(0)` 30s after seeing `_close`. The container's
  // exit code is 0, so without the reclassification logic the
  // task_run_logs row would land as `status='success'` even though
  // the task was killed mid-flight and may have left a dangling
  // `pending_run_at` lock on `follow_me_tasks`. The scheduler reads
  // the queue's `forcedCloseAt` stamp after `runContainerAgent`
  // returns and reclassifies the run as `status='killed'` when the
  // stamp falls within the run's window.

  it('reclassifies a force-closed run as status=killed when the queue reports a forcedCloseAt stamp within the run window', async () => {
    const RECURRING_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };

    createTask({
      id: 'killed-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'run',
      schedule_type: 'cron',
      schedule_value: '0 13 * * *',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    // Simulate the watchdog-success shape: container exits 0 with
    // streaming output, `runContainerAgent` returns success.
    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, _onOutput) => {
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    // Queue stub returns a stamp within the run's window — i.e.
    // closeAllActiveContainers fired mid-run.
    const consumeForcedCloseAt = vi.fn(() => Date.now());

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': RECURRING_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt,
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    // The scheduler must consume the stamp from the maintenance slot
    // (where scheduled tasks always run).
    expect(consumeForcedCloseAt).toHaveBeenCalledWith(
      'main@g.us',
      MAINTENANCE_SESSION_NAME,
    );

    const { _rawQueryForTests } = await import('./db.js');
    const rows = _rawQueryForTests<{ status: string; error: string | null }>(
      `SELECT status, error FROM task_run_logs WHERE task_id = ?`,
      ['killed-task'],
    );
    expect(rows.length).toBe(1);
    expect(rows[0].status).toBe('killed');
    expect(rows[0].error).toMatch(/force-closed|tessl_update|#496/i);

    await vi.advanceTimersByTimeAsync(10);
  });

  it('keeps status=success when the queue reports no forcedCloseAt stamp (normal path)', async () => {
    const RECURRING_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };

    createTask({
      id: 'happy-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'run',
      schedule_type: 'cron',
      schedule_value: '0 13 * * *',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(async () => ({
      status: 'success',
      result: 'ok',
    }));

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': RECURRING_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt: vi.fn(() => null),
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    const { _rawQueryForTests } = await import('./db.js');
    const rows = _rawQueryForTests<{ status: string }>(
      `SELECT status FROM task_run_logs WHERE task_id = ?`,
      ['happy-task'],
    );
    expect(rows.length).toBe(1);
    expect(rows[0].status).toBe('success');

    await vi.advanceTimersByTimeAsync(10);
  });

  it('does NOT reclassify when forcedCloseAt is from BEFORE the run started (stale stamp)', async () => {
    const RECURRING_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };

    createTask({
      id: 'stale-stamp-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'run',
      schedule_type: 'cron',
      schedule_value: '0 13 * * *',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(async () => ({
      status: 'success',
      result: 'ok',
    }));

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    // Stamp is from a long time ago — predates this run. The
    // scheduler must reject it as stale and keep status=success.
    const ancientStamp = Date.now() - 10 * 60 * 60 * 1000;
    const consumeForcedCloseAt = vi.fn(() => ancientStamp);

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': RECURRING_GROUP }),
      queue: {
        enqueueTask,
        closeStdin: vi.fn(),
        consumeForcedCloseAt,
      } as never,
      onProcess: () => {},
      sendMessage: async () => {},
      wipeSessionJsonl: () => 0,
    });

    await vi.advanceTimersByTimeAsync(10);

    const { _rawQueryForTests } = await import('./db.js');
    const rows = _rawQueryForTests<{ status: string }>(
      `SELECT status FROM task_run_logs WHERE task_id = ?`,
      ['stale-stamp-task'],
    );
    expect(rows.length).toBe(1);
    expect(rows[0].status).toBe('success');

    await vi.advanceTimersByTimeAsync(10);
  });
});

describe('recomputeLocalSchedules (#584)', () => {
  // Validates the cache-invalidation surface that hooks tz_state's
  // `current_tz` writers. When current_tz flips mid-slot, cached
  // `next_run` values for `schedule_timezone='local'` rows must be
  // recomputed against the NEW zone — without this, a 7am-local task
  // fires at "what was 7am in the prior zone" until the row elapses.
  //
  // Fake timers freeze `Date.now()` so `computeNextRunDetailed`'s
  // internal anchor lines up with the explicit `now` we pass to the
  // helper — without this, cron-parser's `.next()` anchors on the
  // real wall-clock and the assertion against an expected next-fire
  // diverges from the helper's output.
  const FROZEN_NOW = new Date('2026-03-10T00:00:00.000Z');

  beforeEach(() => {
    _initTestDatabase();
    _resetSchedulerLoopForTests();
    vi.useFakeTimers();
    vi.setSystemTime(FROZEN_NOW);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // A cron expression that fires every day at 07:00 wall-clock. Used
  // across cases so the `next_run` advance is observable on any tz.
  const SEVEN_AM_DAILY = '0 7 * * *';

  function seedLocalRow(
    id: string,
    initialNextRun: string,
    overrides: {
      status?: 'active' | 'paused' | 'completed';
      last_run?: string | null;
      schedule_timezone?: string;
    } = {},
  ): void {
    createTask({
      id,
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'noop',
      schedule_type: 'cron',
      schedule_value: SEVEN_AM_DAILY,
      schedule_timezone: overrides.schedule_timezone ?? 'local',
      context_mode: 'isolated',
      next_run: initialNextRun,
      status: overrides.status ?? 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    if (overrides.last_run !== undefined) {
      _execRawForTests(`UPDATE scheduled_tasks SET last_run = ? WHERE id = ?`, [
        overrides.last_run,
        id,
      ]);
    }
  }

  it("updates next_run for active 'local' rows when current_tz changes", () => {
    // Cron-parser anchored to America/Chicago — next 7am Chicago.
    // `.toISOString()` is typed `string | null`; assert non-null
    // because we just produced the date a line ago.
    const initial = CronExpressionParser.parse(SEVEN_AM_DAILY, {
      tz: 'America/Chicago',
      currentDate: FROZEN_NOW,
    })
      .next()
      .toISOString()!;
    seedLocalRow('local-row', initial);

    const result = recomputeLocalSchedules(() => 'Asia/Tokyo', FROZEN_NOW);
    expect(result.recomputed).toBe(1);

    const fresh = getTaskById('local-row');
    expect(fresh).toBeDefined();
    expect(fresh!.next_run).not.toBe(initial);
    // Expected new next_run is the next 7am wall-clock in Tokyo from
    // the freeze-point — verifies the recompute actually consulted
    // the new tz rather than fallback-TIMEZONE.
    const expected = CronExpressionParser.parse(SEVEN_AM_DAILY, {
      tz: 'Asia/Tokyo',
      currentDate: FROZEN_NOW,
    })
      .next()
      .toISOString()!;
    expect(fresh!.next_run).toBe(expected);
  });

  it("skips non-'local' rows", () => {
    const initial = new Date('2099-01-01T07:00:00.000Z').toISOString();
    seedLocalRow('chicago-row', initial, {
      schedule_timezone: 'America/Chicago',
    });

    const result = recomputeLocalSchedules(() => 'Asia/Tokyo', FROZEN_NOW);
    expect(result.recomputed).toBe(0);

    const fresh = getTaskById('chicago-row');
    expect(fresh!.next_run).toBe(initial);
  });

  it('skips non-active rows', () => {
    const initial = new Date('2099-01-01T07:00:00.000Z').toISOString();
    seedLocalRow('paused-row', initial, { status: 'paused' });
    seedLocalRow('completed-row', initial, { status: 'completed' });

    const result = recomputeLocalSchedules(() => 'Asia/Tokyo', FROZEN_NOW);
    expect(result.recomputed).toBe(0);
    expect(getTaskById('paused-row')!.next_run).toBe(initial);
    expect(getTaskById('completed-row')!.next_run).toBe(initial);

    // Sanity: getActiveLocalScheduledTasks should not return them
    // either — the selector itself filters status.
    expect(getActiveLocalScheduledTasks()).toHaveLength(0);
  });

  it('emits catch-up warning when new next_run is in past AND row has not fired today in new zone', () => {
    // The catch-up branch is structurally hard to reach from
    // production compute (`cron-parser.next()` and the interval
    // skip-past-missed loop both produce strictly-future values), so
    // we drive it through the `computeNextRun` dep override — that
    // seam exists for exactly this catch-up scenario, which only
    // fires defensively if a future cron-parser bug or DST
    // discontinuity surfaces a past wall-clock.
    const fakeNow = new Date('2026-03-10T05:00:00.000Z');
    const pastNextRun = new Date(fakeNow.getTime() - 5 * 60_000).toISOString();
    const fakeRow: ScheduledTask = {
      id: 'catchup-overdue',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'noop',
      script: null,
      schedule_type: 'cron',
      schedule_value: '0 7 * * *',
      schedule_timezone: 'local',
      context_mode: 'isolated',
      next_run: pastNextRun,
      last_run: null, // never fired → row is overdue-and-fireable
      last_result: null,
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {});
    const writes = new Map<string, string | null>();

    const result = recomputeLocalSchedules(() => 'Asia/Tokyo', fakeNow, {
      getActiveLocalScheduledTasks: () => [fakeRow],
      setTaskNextRun: (id, nr) => {
        writes.set(id, nr);
      },
      computeNextRun: () => ({ nextRun: pastNextRun }),
    });

    expect(result.recomputed).toBe(1);
    expect(result.caughtUp).toBe(1);
    expect(writes.get('catchup-overdue')).toBe(pastNextRun);
    const catchupCalls = warnSpy.mock.calls.filter(
      (call) =>
        typeof call[1] === 'string' &&
        call[1].startsWith('recomputeLocalSchedules: catch-up'),
    );
    expect(catchupCalls).toHaveLength(1);
    expect(catchupCalls[0][0]).toMatchObject({
      taskId: 'catchup-overdue',
      nextRun: pastNextRun,
      currentTz: 'Asia/Tokyo',
    });

    warnSpy.mockRestore();
  });

  it('does NOT emit catch-up when row already fired today in new zone', () => {
    // `last_run` AFTER today's midnight in the new zone → row already
    // ran today → gate suppresses catch-up warn even if the freshly-
    // computed next_run lands in the past.
    const fakeNow = new Date('2026-03-10T05:00:00.000Z'); // 14:00 Tokyo
    const pastNextRun = new Date(fakeNow.getTime() - 5 * 60_000).toISOString();
    const fakeRow: ScheduledTask = {
      id: 'fired-today',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'noop',
      script: null,
      schedule_type: 'cron',
      schedule_value: '0 7 * * *',
      schedule_timezone: 'local',
      context_mode: 'isolated',
      next_run: pastNextRun,
      // 23:00 UTC on 2026-03-09 = 08:00 Tokyo on 2026-03-10
      // → AFTER Tokyo midnight on 2026-03-10, BEFORE `fakeNow` —
      // row already fired today in Tokyo.
      last_run: new Date('2026-03-09T23:00:00.000Z').toISOString(),
      last_result: 'prior-fire',
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {});

    const result = recomputeLocalSchedules(() => 'Asia/Tokyo', fakeNow, {
      getActiveLocalScheduledTasks: () => [fakeRow],
      setTaskNextRun: () => {},
      computeNextRun: () => ({ nextRun: pastNextRun }),
    });
    expect(result.recomputed).toBe(1);
    expect(result.caughtUp).toBe(0);
    const catchupCalls = warnSpy.mock.calls.filter(
      (call) =>
        typeof call[1] === 'string' &&
        call[1].startsWith('recomputeLocalSchedules: catch-up'),
    );
    expect(catchupCalls).toHaveLength(0);

    warnSpy.mockRestore();
  });

  it('startOfTodayInTz returns expected UTC instant for a known zone', () => {
    // 2026-03-10T05:00:00Z = 2026-03-10 14:00 Tokyo (UTC+9).
    // Tokyo midnight on 2026-03-10 = 2026-03-09T15:00:00Z.
    expect(
      startOfTodayInTz('Asia/Tokyo', new Date('2026-03-10T05:00:00.000Z')),
    ).toBe(Date.UTC(2026, 2, 9, 15, 0, 0));
    // 2026-03-10T08:00:00Z = 2026-03-10 03:00 Chicago (UTC-5 since
    // CDT starts on the second Sunday of March, 2026-03-08).
    // Chicago midnight on 2026-03-10 = 2026-03-10T05:00:00Z.
    expect(
      startOfTodayInTz('America/Chicago', new Date('2026-03-10T08:00:00.000Z')),
    ).toBe(Date.UTC(2026, 2, 10, 5, 0, 0));
  });

  it('returns NaN for unparseable tz string', () => {
    expect(
      startOfTodayInTz('Not/A_Real_Zone', new Date('2026-03-10T05:00:00.000Z')),
    ).toBeNaN();
  });

  it('does NOT emit catch-up when row already fired today in new zone', () => {
    // Seed a cron-`local` row with last_run AFTER today's midnight in
    // the new zone. Even if a future iteration recomputes against the
    // new zone, the gate suppresses the catch-up warn.
    const tz = 'Asia/Tokyo';
    const now = new Date('2026-03-10T05:00:00.000Z'); // 14:00 in Tokyo
    const initial = new Date('2026-03-15T00:00:00.000Z').toISOString();
    // 08:00 Tokyo = 23:00 UTC previous day — AFTER Tokyo midnight,
    // BEFORE `now`.
    seedLocalRow('fired-today-row', initial, {
      last_run: new Date('2026-03-09T23:00:00.000Z').toISOString(),
    });

    const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {});

    // Drive with a fake selector returning the row but with a
    // doctored cron compute path: cron will always pick a future
    // next_run, so we just verify no catch-up warns landed.
    const result = recomputeLocalSchedules(() => tz, now);
    expect(result.recomputed).toBe(1);
    expect(result.caughtUp).toBe(0);
    // The only allowed warns from this call would be the catch-up
    // line; any other warn (per-row compute error etc.) would be a
    // bug surface for this test to flag.
    const catchupCalls = warnSpy.mock.calls.filter(
      (call) =>
        typeof call[1] === 'string' &&
        call[1].includes('recomputeLocalSchedules: catch-up'),
    );
    expect(catchupCalls).toHaveLength(0);

    warnSpy.mockRestore();
  });
});

describe('setTaskNextRun (#584)', () => {
  beforeEach(() => {
    _initTestDatabase();
  });

  it('writes only next_run and leaves other fields intact', () => {
    createTask({
      id: 'narrow-update',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'original-prompt',
      schedule_type: 'cron',
      schedule_value: '0 7 * * *',
      schedule_timezone: 'local',
      context_mode: 'isolated',
      next_run: new Date('2099-01-01T07:00:00.000Z').toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    // Seed last_result so we can verify it survives.
    _execRawForTests(
      `UPDATE scheduled_tasks SET last_result = ?, last_run = ? WHERE id = ?`,
      ['prior-result', '2026-01-15T07:00:00.000Z', 'narrow-update'],
    );

    const newNextRun = new Date('2030-01-01T07:00:00.000Z').toISOString();
    setTaskNextRun('narrow-update', newNextRun);

    const fresh = getTaskById('narrow-update');
    expect(fresh!.next_run).toBe(newNextRun);
    // Everything else preserved.
    expect(fresh!.prompt).toBe('original-prompt');
    expect(fresh!.status).toBe('active');
    expect(fresh!.last_result).toBe('prior-result');
    expect(fresh!.last_run).toBe('2026-01-15T07:00:00.000Z');
    expect(fresh!.schedule_value).toBe('0 7 * * *');
    expect(fresh!.schedule_timezone).toBe('local');
  });

  it('accepts null next_run (paused-broken-cron remediation shape)', () => {
    createTask({
      id: 'null-next-run',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'noop',
      schedule_type: 'cron',
      schedule_value: '0 7 * * *',
      schedule_timezone: 'local',
      context_mode: 'isolated',
      next_run: new Date('2099-01-01T07:00:00.000Z').toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    setTaskNextRun('null-next-run', null);
    const fresh = getTaskById('null-next-run');
    expect(fresh!.next_run).toBeNull();
  });
});
