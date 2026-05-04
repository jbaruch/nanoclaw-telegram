import { afterEach, describe, expect, it, vi } from 'vitest';

import { logger } from './logger.js';
import { computeThresholds } from './threshold.js';
import { emitSessionTokens } from './usage-telemetry.js';

const THRESHOLDS = computeThresholds(1_000_000); // warn 700K / nuke 800K

afterEach(() => {
  vi.restoreAllMocks();
});

describe('emitSessionTokens', () => {
  it('returns null and logs nothing when usage is undefined', () => {
    const info = vi.spyOn(logger, 'info').mockImplementation(() => {});
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {});
    const error = vi.spyOn(logger, 'error').mockImplementation(() => {});

    const state = emitSessionTokens(undefined, {
      group: 'g',
      thresholds: THRESHOLDS,
    });

    expect(state).toBeNull();
    expect(info).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
  });

  it('emits info-level `session_tokens` below the warn threshold', () => {
    const info = vi.spyOn(logger, 'info').mockImplementation(() => {});
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {});
    const error = vi.spyOn(logger, 'error').mockImplementation(() => {});

    const state = emitSessionTokens(
      {
        input_tokens: 100_000,
        output_tokens: 500,
        cache_read_input_tokens: 2_000,
        cache_creation_input_tokens: 1_000,
      },
      { group: 'inbound-group', session: 'sess-1', thresholds: THRESHOLDS },
    );

    expect(state).toBe('below_warn');
    expect(info).toHaveBeenCalledTimes(1);
    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
    const [fields, key] = info.mock.calls[0];
    expect(key).toBe('session_tokens');
    expect(fields).toMatchObject({
      group: 'inbound-group',
      session: 'sess-1',
      input_tokens: 100_000,
      output_tokens: 500,
      cache_read: 2_000,
      cache_creation: 1_000,
      turn_context_size: 103_000,
      threshold_state: 'below_warn',
      threshold_warn: 700_000,
      threshold_nuke: 800_000,
    });
    expect((fields as Record<string, unknown>).percent).toBeCloseTo(10.3, 1);
  });

  it('classifies cache-heavy turns by total context, not the input_tokens delta (#498)', () => {
    const info = vi.spyOn(logger, 'info').mockImplementation(() => {});
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {});
    const error = vi.spyOn(logger, 'error').mockImplementation(() => {});

    // Production shape under prompt caching: tiny `input_tokens` delta
    // alongside a huge `cache_read_input_tokens`. The pre-#498 classifier
    // looked only at `input_tokens` and pinned this at `below_warn`.
    const state = emitSessionTokens(
      {
        input_tokens: 2,
        output_tokens: 1,
        cache_read_input_tokens: 750_000,
      },
      { group: 'g', thresholds: THRESHOLDS },
    );

    expect(state).toBe('warn');
    expect(warn).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
    expect(warn.mock.calls[0][1]).toBe('session_tokens_warn');
    expect(warn.mock.calls[0][0]).toMatchObject({
      input_tokens: 2,
      cache_read: 750_000,
      turn_context_size: 750_002,
      threshold_state: 'warn',
    });
  });

  it('counts cache_creation toward the turn context size (#498)', () => {
    const error = vi.spyOn(logger, 'error').mockImplementation(() => {});

    const state = emitSessionTokens(
      {
        input_tokens: 5,
        output_tokens: 1,
        cache_read_input_tokens: 700_000,
        cache_creation_input_tokens: 105_000,
      },
      { group: 'g', thresholds: THRESHOLDS },
    );

    expect(state).toBe('nuke');
    expect(error).toHaveBeenCalledTimes(1);
    expect(error.mock.calls[0][0]).toMatchObject({
      turn_context_size: 805_005,
    });
  });

  it('emits warn-level `session_tokens_warn` at the warn threshold', () => {
    const info = vi.spyOn(logger, 'info').mockImplementation(() => {});
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {});
    const error = vi.spyOn(logger, 'error').mockImplementation(() => {});

    const state = emitSessionTokens(
      { input_tokens: 700_000, output_tokens: 1 },
      { group: 'g', thresholds: THRESHOLDS },
    );

    expect(state).toBe('warn');
    expect(warn).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
    expect(warn.mock.calls[0][1]).toBe('session_tokens_warn');
  });

  it('emits error-level `session_tokens_nuke` at the nuke threshold', () => {
    const info = vi.spyOn(logger, 'info').mockImplementation(() => {});
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {});
    const error = vi.spyOn(logger, 'error').mockImplementation(() => {});

    const state = emitSessionTokens(
      { input_tokens: 850_000, output_tokens: 1 },
      { group: 'g', thresholds: THRESHOLDS },
    );

    expect(state).toBe('nuke');
    expect(error).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    expect(error.mock.calls[0][1]).toBe('session_tokens_nuke');
  });

  it('passes `extra` fields through onto the log payload (scheduled-task variant)', () => {
    const info = vi.spyOn(logger, 'info').mockImplementation(() => {});

    emitSessionTokens(
      {
        input_tokens: 50_000,
        output_tokens: 200,
      },
      {
        group: 'maintenance-group',
        thresholds: THRESHOLDS,
        extra: {
          taskId: 'task-abc-123',
          scheduleType: 'interval',
          taskSkill: 'tessl__heartbeat',
        },
      },
    );

    expect(info).toHaveBeenCalledTimes(1);
    expect(info.mock.calls[0][0]).toMatchObject({
      group: 'maintenance-group',
      taskId: 'task-abc-123',
      scheduleType: 'interval',
      taskSkill: 'tessl__heartbeat',
      threshold_state: 'below_warn',
    });
  });

  it('omits `session` when not provided (scheduled-task contract — #193)', () => {
    const info = vi.spyOn(logger, 'info').mockImplementation(() => {});

    emitSessionTokens(
      { input_tokens: 1, output_tokens: 1 },
      {
        group: 'g',
        thresholds: THRESHOLDS,
        extra: { taskId: 't1' },
      },
    );

    const fields = info.mock.calls[0][0] as Record<string, unknown>;
    expect(fields.session).toBeUndefined();
    expect(fields.taskId).toBe('t1');
  });
});
