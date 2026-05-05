import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { IDLE_TIMEOUT } from './config.js';
import {
  _resetIdleTimerStateForTesting,
  getActiveIdleTimer,
  installIdleTimerControl,
  releaseIdleTimerControl,
} from './idle-timer.js';
import type { RegisteredGroup } from './types.js';

const mainGroup: RegisteredGroup = {
  name: 'telegram_main',
  folder: 'telegram_main',
  trigger: null,
  added_at: '2026-01-01T00:00:00Z',
  isMain: true,
};

const untrustedGroup: RegisteredGroup = {
  name: 'telegram_iff-lom-bot-test-low-trust',
  folder: 'telegram_iff-lom-bot-test-low-trust',
  trigger: '@LoMBot',
  added_at: '2026-01-01T00:00:00Z',
};

const chat = 'tg:1234';

describe('idle-timer', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    _resetIdleTimerStateForTesting();
  });

  afterEach(() => {
    _resetIdleTimerStateForTesting();
    vi.useRealTimers();
  });

  it('fires onTimeout exactly once after the configured deadline (main tier)', () => {
    const onTimeout = vi.fn();
    const control = installIdleTimerControl(chat, mainGroup, onTimeout);

    control.reset('agent-output');
    expect(onTimeout).not.toHaveBeenCalled();

    vi.advanceTimersByTime(IDLE_TIMEOUT - 1);
    expect(onTimeout).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1);
    expect(onTimeout).toHaveBeenCalledTimes(1);
    // Map auto-clears when the timer fires.
    expect(getActiveIdleTimer(chat)).toBeUndefined();
  });

  it('uses 5min for untrusted groups instead of IDLE_TIMEOUT', () => {
    const onTimeout = vi.fn();
    const control = installIdleTimerControl(chat, untrustedGroup, onTimeout);
    control.reset('agent-output');

    vi.advanceTimersByTime(300_000 - 1);
    expect(onTimeout).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(onTimeout).toHaveBeenCalledTimes(1);
  });

  it('reset(user-input) extends the deadline mid-window', () => {
    const onTimeout = vi.fn();
    const control = installIdleTimerControl(chat, mainGroup, onTimeout);
    control.reset('agent-output');

    // Advance most of the way to the original deadline.
    vi.advanceTimersByTime(IDLE_TIMEOUT - 1000);
    // User input arrives — re-anchor.
    control.reset('user-input');

    // Past the ORIGINAL deadline but inside the new window: still alive.
    vi.advanceTimersByTime(1001);
    expect(onTimeout).not.toHaveBeenCalled();

    // Past the NEW deadline: fires once.
    vi.advanceTimersByTime(IDLE_TIMEOUT);
    expect(onTimeout).toHaveBeenCalledTimes(1);
  });

  it('stale timer no-ops when a fresh control replaces it', () => {
    const onTimeoutA = vi.fn();
    const controlA = installIdleTimerControl(chat, mainGroup, onTimeoutA);
    controlA.reset('agent-output');

    // Container generation B installs over the top — A must be cleared.
    const onTimeoutB = vi.fn();
    const controlB = installIdleTimerControl(chat, mainGroup, onTimeoutB);
    expect(controlB).not.toBe(controlA);
    expect(getActiveIdleTimer(chat)).toBe(controlB);

    // B has not yet been reset; advancing past A's original deadline must
    // not fire either onTimeout. A's pending setTimeout is cleared by the
    // install path; B has no pending setTimeout because reset() was not
    // called yet.
    vi.advanceTimersByTime(IDLE_TIMEOUT * 2);
    expect(onTimeoutA).not.toHaveBeenCalled();
    expect(onTimeoutB).not.toHaveBeenCalled();

    // B's reset arms its own timer — that one fires normally.
    controlB.reset('agent-output');
    vi.advanceTimersByTime(IDLE_TIMEOUT);
    expect(onTimeoutB).toHaveBeenCalledTimes(1);
    expect(onTimeoutA).not.toHaveBeenCalled();
  });

  it('in-callback identity guard short-circuits if a stale callback fires past install-clear', () => {
    // The previous test verifies install-time clear (the primary defense).
    // This test exercises the SECOND line of defense — the identity check
    // inside the setTimeout body — by capturing A's callback before B's
    // install clears the underlying timer, then invoking it directly.
    // Models a future refactor that bypasses install-clear.
    const onTimeoutA = vi.fn();

    let capturedCallback: (() => void) | null = null;
    const realSetTimeout = globalThis.setTimeout;
    const spy = vi.spyOn(globalThis, 'setTimeout').mockImplementation(((
      cb: () => void,
      ms: number,
    ) => {
      capturedCallback = cb;
      return realSetTimeout(cb, ms);
    }) as typeof setTimeout);

    const controlA = installIdleTimerControl(chat, mainGroup, onTimeoutA);
    controlA.reset('agent-output');
    spy.mockRestore();

    // Install B — clears A's underlying timer, but `capturedCallback`
    // still points at A's closure.
    const onTimeoutB = vi.fn();
    const controlB = installIdleTimerControl(chat, mainGroup, onTimeoutB);

    expect(capturedCallback).not.toBeNull();
    capturedCallback!();
    expect(onTimeoutA).not.toHaveBeenCalled();
    expect(onTimeoutB).not.toHaveBeenCalled();
    // B's slot is preserved — the stale callback didn't delete it.
    expect(getActiveIdleTimer(chat)).toBe(controlB);
  });

  it('releaseIdleTimerControl drops the entry only if identity matches', () => {
    const onTimeoutA = vi.fn();
    const controlA = installIdleTimerControl(chat, mainGroup, onTimeoutA);

    const onTimeoutB = vi.fn();
    const controlB = installIdleTimerControl(chat, mainGroup, onTimeoutB);

    // A's processGroupMessages cycle ends late and tries to release.
    // B is now the active control — A's release must NOT clobber B.
    releaseIdleTimerControl(chat, controlA);
    expect(getActiveIdleTimer(chat)).toBe(controlB);

    // B's own release does drop the entry.
    releaseIdleTimerControl(chat, controlB);
    expect(getActiveIdleTimer(chat)).toBeUndefined();
  });

  it('clear() stops a pending timer', () => {
    const onTimeout = vi.fn();
    const control = installIdleTimerControl(chat, mainGroup, onTimeout);
    control.reset('agent-output');
    control.clear();
    vi.advanceTimersByTime(IDLE_TIMEOUT * 2);
    expect(onTimeout).not.toHaveBeenCalled();
  });

  it('multiple chats are isolated', () => {
    const chatA = 'tg:1';
    const chatB = 'tg:2';
    const onA = vi.fn();
    const onB = vi.fn();

    const a = installIdleTimerControl(chatA, mainGroup, onA);
    const b = installIdleTimerControl(chatB, mainGroup, onB);
    a.reset('agent-output');
    b.reset('agent-output');

    expect(getActiveIdleTimer(chatA)).toBe(a);
    expect(getActiveIdleTimer(chatB)).toBe(b);

    vi.advanceTimersByTime(IDLE_TIMEOUT);
    expect(onA).toHaveBeenCalledTimes(1);
    expect(onB).toHaveBeenCalledTimes(1);
  });
});
