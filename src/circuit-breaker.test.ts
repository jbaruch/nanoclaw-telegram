import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CIRCUIT_BREAKER_COOLDOWN_MINUTES,
  checkCircuitBreaker,
  isBreakerActive,
  recordFailure,
  recordSuccess,
} from './circuit-breaker.js';

// The module keeps process-level state keyed by folder. Each test uses a
// unique folder id so cases stay independent without a test-only reset
// hook (jbaruch/coding-policy: testing-standards — independence).
let _folderCounter = 0;
function folder(): string {
  _folderCounter += 1;
  return `telegram_test_${_folderCounter}`;
}

// Freeze the clock so cooldown math is deterministic (testing-standards —
// no dependence on wall-clock time).
beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'));
});

afterEach(() => {
  vi.useRealTimers();
});

describe('recordFailure / recordSuccess', () => {
  it('does not trip before the fifth consecutive failure', () => {
    const f = folder();
    for (let i = 1; i <= 4; i += 1) {
      const r = recordFailure(f);
      expect(r).toEqual({ tripped: false, failures: i });
    }
    expect(isBreakerActive(f)).toBe(false);
  });

  it('trips on the fifth consecutive failure', () => {
    const f = folder();
    let last;
    for (let i = 1; i <= 5; i += 1) last = recordFailure(f);
    expect(last).toEqual({ tripped: true, failures: 5 });
    expect(isBreakerActive(f)).toBe(true);
  });

  it('recordSuccess clears the failure counter so the breaker never trips', () => {
    const f = folder();
    recordFailure(f);
    recordFailure(f);
    recordSuccess(f);
    // Counter reset — four more failures is only four, not six.
    for (let i = 1; i <= 4; i += 1)
      expect(recordFailure(f).tripped).toBe(false);
    expect(isBreakerActive(f)).toBe(false);
  });
});

describe('checkCircuitBreaker', () => {
  it("returns 'ok' for a group that has never tripped", () => {
    expect(checkCircuitBreaker(folder())).toBe('ok');
  });

  it("returns 'skip' while the cooldown is still active", () => {
    const f = folder();
    for (let i = 1; i <= 5; i += 1) recordFailure(f);
    vi.advanceTimersByTime((CIRCUIT_BREAKER_COOLDOWN_MINUTES - 1) * 60_000);
    expect(checkCircuitBreaker(f)).toBe('skip');
  });

  it("returns 'resumed' and resets state once the cooldown expires", () => {
    const f = folder();
    for (let i = 1; i <= 5; i += 1) recordFailure(f);
    vi.advanceTimersByTime(CIRCUIT_BREAKER_COOLDOWN_MINUTES * 60_000 + 1);
    expect(checkCircuitBreaker(f)).toBe('resumed');
    // State cleared: a subsequent check is 'ok' and the breaker is inactive.
    expect(checkCircuitBreaker(f)).toBe('ok');
    expect(isBreakerActive(f)).toBe(false);
  });
});

describe('isBreakerActive (read-only status)', () => {
  it('never mutates state — repeated reads during cooldown stay skip', () => {
    const f = folder();
    for (let i = 1; i <= 5; i += 1) recordFailure(f);
    expect(isBreakerActive(f)).toBe(true);
    expect(isBreakerActive(f)).toBe(true);
    // A read did not clear the cooldown; the loop check still skips.
    expect(checkCircuitBreaker(f)).toBe('skip');
  });
});
