import { describe, it, expect } from 'vitest';

import {
  decideHardExitWatchdog,
  HARD_EXIT_IDLE_BUDGET_MS,
} from './hard-exit-watchdog.js';

describe('decideHardExitWatchdog (#545)', () => {
  // The pure decision function the post-close hard-exit watchdog
  // consults on every fire. Tests cover the two action shapes (exit
  // / rearm) plus the boundary semantics that protect the wall-clock
  // bound from infinite re-arming on a chatty agent.

  it('exits when no SDK activity within the idle budget (stuck SDK iterator)', () => {
    // Pre-#545 shape: 30s wall-clock from close, no activity. With
    // activity-aware budgeting the same outcome holds — idleMs ==
    // budget trips the >= comparison.
    const now = 1_000_000;
    const lastActivity = now - HARD_EXIT_IDLE_BUDGET_MS;
    const decision = decideHardExitWatchdog(now, lastActivity);
    expect(decision).toEqual({
      action: 'exit',
      idleMs: HARD_EXIT_IDLE_BUDGET_MS,
    });
  });

  it('exits when idle window exceeds the budget by any margin', () => {
    // The watchdog timer can fire slightly late (event-loop
    // contention, GC pause). idleMs > budget must still exit, not
    // re-arm at a negative interval.
    const now = 1_000_000;
    const lastActivity = now - (HARD_EXIT_IDLE_BUDGET_MS + 5_000);
    const decision = decideHardExitWatchdog(now, lastActivity);
    expect(decision.action).toBe('exit');
    expect(decision.action === 'exit' && decision.idleMs).toBe(
      HARD_EXIT_IDLE_BUDGET_MS + 5_000,
    );
  });

  it('re-arms with remaining budget when activity is recent (working agent)', () => {
    // Morning-brief shape: agent composed an assistant turn 5s ago,
    // close was detected 25s ago. Pre-#545 this would have killed at
    // close + 30s = 5s in the future. Post-#545 the watchdog re-arms
    // for the remaining budget (25s) after the most recent event.
    const now = 1_000_000;
    const lastActivity = now - 5_000;
    const decision = decideHardExitWatchdog(now, lastActivity);
    expect(decision).toEqual({
      action: 'rearm',
      rearmInMs: HARD_EXIT_IDLE_BUDGET_MS - 5_000,
      idleMs: 5_000,
    });
  });

  it('re-arms with full budget when activity is now (zero idle)', () => {
    // Edge case: timer fires at exactly the same instant as a fresh
    // SDK event lands. idleMs == 0; re-arm for the full budget.
    const now = 1_000_000;
    const decision = decideHardExitWatchdog(now, now);
    expect(decision).toEqual({
      action: 'rearm',
      rearmInMs: HARD_EXIT_IDLE_BUDGET_MS,
      idleMs: 0,
    });
  });

  it('respects an explicit budget override (testability)', () => {
    // Tests of the larger watchdog flow may want a tighter budget
    // (1s) to keep timing deterministic without slowing the suite.
    // The third arg lets them inject one without monkeypatching the
    // module-level constant.
    const now = 1_000_000;
    const decisionExit = decideHardExitWatchdog(now, now - 1_500, 1_000);
    expect(decisionExit.action).toBe('exit');

    const decisionRearm = decideHardExitWatchdog(now, now - 500, 1_000);
    expect(decisionRearm).toEqual({
      action: 'rearm',
      rearmInMs: 500,
      idleMs: 500,
    });
  });

  it('treats the boundary `idleMs == budget` as exit, never rearm-at-zero', () => {
    // Without the >= boundary, the rearm branch could schedule a
    // 0ms setTimeout that fires immediately and re-enters the same
    // decision — a tight CPU-burning loop instead of a clean kill.
    const now = 1_000_000;
    const decision = decideHardExitWatchdog(
      now,
      now - HARD_EXIT_IDLE_BUDGET_MS,
      HARD_EXIT_IDLE_BUDGET_MS,
    );
    expect(decision.action).toBe('exit');
  });
});
