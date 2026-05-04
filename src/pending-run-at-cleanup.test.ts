/**
 * Tests for the host-side `follow_me_tasks.pending_run_at` recovery
 * helpers added by #496.
 *
 * Background: skills running inside agent containers acquire a per-task
 * lock by setting `pending_run_at` mid-run, then clear it post-run.
 * The periodic `tessl_update` catch-up writes `_close` to every active
 * container; the agent-runner's 30s hard-exit watchdog then makes the
 * container exit 0 — looking like clean success at the host even though
 * the run was killed. The owning skill never reaches its post-run
 * clear step, so the lock is left dangling and tomorrow's fire hits
 * the Phase A gate and refuses to run.
 *
 * `clearStalePendingRunAt(maxAgeMs)` is the recovery path: any
 * `pending_run_at` older than `maxAgeMs` is presumed orphaned by a
 * dead/killed container and gets reclaimed.
 *
 * `getActivePendingRunAtNames(maxAgeMs)` is the prevention path: the
 * orchestrator's periodic `tessl_update` checks for fresh locks
 * before firing, so it doesn't kill an in-flight task.
 *
 * Both are non-owner reader operations on `follow_me_tasks` per
 * `coding-policy: stateful-artifacts` — the host doesn't migrate the
 * schema, only NULLs out value fields whose owners are demonstrably
 * dead.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';

import {
  _closeDatabase,
  _initTestDatabase,
  _seedFollowMeTaskForTests,
  clearStalePendingRunAt,
  getActivePendingRunAtNames,
} from './db.js';

function seedFollowMeTask(args: {
  name: string;
  pendingRunAt: string | null;
}): void {
  _seedFollowMeTaskForTests({
    name: args.name,
    pendingRunAt: args.pendingRunAt,
  });
}

beforeEach(() => {
  _initTestDatabase();
});

afterEach(() => {
  _closeDatabase();
});

describe('clearStalePendingRunAt (#496)', () => {
  it('returns empty array when no follow_me_tasks rows exist', () => {
    const cleared = clearStalePendingRunAt(60 * 60 * 1000);
    expect(cleared).toEqual([]);
  });

  it('returns empty array when all rows have NULL pending_run_at', () => {
    seedFollowMeTask({ name: 'morning-brief', pendingRunAt: null });
    seedFollowMeTask({ name: 'nightly-housekeeping', pendingRunAt: null });
    const cleared = clearStalePendingRunAt(60 * 60 * 1000);
    expect(cleared).toEqual([]);
  });

  it('clears rows whose pending_run_at is older than maxAgeMs', () => {
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
    seedFollowMeTask({ name: 'morning-brief', pendingRunAt: twoHoursAgo });
    seedFollowMeTask({
      name: 'nightly-housekeeping',
      pendingRunAt: twoHoursAgo,
    });

    const cleared = clearStalePendingRunAt(60 * 60 * 1000);

    expect(cleared.sort()).toEqual(['morning-brief', 'nightly-housekeeping']);
    // Confirm the rows were actually cleared by re-running the read
    // path: a second invocation finds nothing.
    expect(clearStalePendingRunAt(60 * 60 * 1000)).toEqual([]);
  });

  it('preserves fresh pending_run_at values within the window', () => {
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    seedFollowMeTask({ name: 'morning-brief', pendingRunAt: fiveMinutesAgo });

    const cleared = clearStalePendingRunAt(60 * 60 * 1000);

    expect(cleared).toEqual([]);
    // The row's lock should still be there for the owning skill to
    // clear on its post-run path.
    expect(getActivePendingRunAtNames(60 * 60 * 1000)).toEqual([
      'morning-brief',
    ]);
  });

  it('clears only the stale rows when mixed with fresh ones', () => {
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    seedFollowMeTask({ name: 'morning-brief', pendingRunAt: twoHoursAgo });
    seedFollowMeTask({
      name: 'weekly-housekeeping',
      pendingRunAt: fiveMinutesAgo,
    });

    const cleared = clearStalePendingRunAt(60 * 60 * 1000);

    expect(cleared).toEqual(['morning-brief']);
    expect(getActivePendingRunAtNames(60 * 60 * 1000)).toEqual([
      'weekly-housekeeping',
    ]);
  });

  it('rejects non-positive maxAgeMs with an actionable error', () => {
    expect(() => clearStalePendingRunAt(0)).toThrow(/positive number/);
    expect(() => clearStalePendingRunAt(-1)).toThrow(/positive number/);
    expect(() => clearStalePendingRunAt(Number.NaN)).toThrow(/positive number/);
  });
});

describe('getActivePendingRunAtNames (#496)', () => {
  it('returns empty array when no rows exist', () => {
    expect(getActivePendingRunAtNames(60 * 60 * 1000)).toEqual([]);
  });

  it('returns empty array when all rows have NULL pending_run_at', () => {
    seedFollowMeTask({ name: 'morning-brief', pendingRunAt: null });
    expect(getActivePendingRunAtNames(60 * 60 * 1000)).toEqual([]);
  });

  it('returns names of rows with fresh pending_run_at locks', () => {
    const justNow = new Date().toISOString();
    seedFollowMeTask({ name: 'morning-brief', pendingRunAt: justNow });
    seedFollowMeTask({
      name: 'nightly-housekeeping',
      pendingRunAt: justNow,
    });

    const names = getActivePendingRunAtNames(60 * 60 * 1000);
    expect(names.sort()).toEqual(['morning-brief', 'nightly-housekeeping']);
  });

  it('excludes rows whose pending_run_at is older than the window', () => {
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
    seedFollowMeTask({ name: 'morning-brief', pendingRunAt: twoHoursAgo });

    expect(getActivePendingRunAtNames(60 * 60 * 1000)).toEqual([]);
  });

  it('rejects non-positive maxAgeMs with an actionable error', () => {
    expect(() => getActivePendingRunAtNames(0)).toThrow(/positive number/);
    expect(() => getActivePendingRunAtNames(-1)).toThrow(/positive number/);
  });
});
