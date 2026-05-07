import { describe, it, expect, afterEach, vi } from 'vitest';

import {
  DEFAULT_CONTEXT_STRATEGY,
  _unregisterContextStrategyForTesting,
  getContextStrategy,
  listContextStrategies,
  registerContextStrategy,
  resolveContextStrategy,
  type ContextStrategy,
} from './context-strategy.js';
import type { GateContext } from './index.js';
import { logger } from '../logger.js';

const TEST_NAME = 'unit-test-strategy';

const baseCtx: GateContext = {
  groupJid: 'g@g.us',
  groupFolder: 'telegram_test',
  message: {
    text: 'hi',
    messageId: 'msg-test-1',
    senderJid: 's@s.whatsapp.net',
  },
  triggerPatterns: null,
};

const dummyStrategy: ContextStrategy = {
  name: TEST_NAME,
  async buildContext() {
    return 'unit-test ctx';
  },
};

afterEach(() => {
  if (listContextStrategies().includes(TEST_NAME)) {
    _unregisterContextStrategyForTesting(TEST_NAME);
  }
  vi.restoreAllMocks();
});

describe('ContextStrategy registry', () => {
  it('default strategy is registered at module load', () => {
    expect(getContextStrategy(DEFAULT_CONTEXT_STRATEGY)).toBeDefined();
    expect(listContextStrategies()).toContain(DEFAULT_CONTEXT_STRATEGY);
  });

  it('register + lookup', () => {
    registerContextStrategy(dummyStrategy);
    expect(getContextStrategy(TEST_NAME)).toBe(dummyStrategy);
    expect(listContextStrategies()).toContain(TEST_NAME);
  });

  it('double registration throws', () => {
    registerContextStrategy(dummyStrategy);
    expect(() => registerContextStrategy(dummyStrategy)).toThrow(
      /already registered/,
    );
  });
});

describe('resolveContextStrategy', () => {
  it('returns the requested strategy when registered', () => {
    registerContextStrategy(dummyStrategy);
    expect(resolveContextStrategy(TEST_NAME, baseCtx.groupFolder)).toBe(
      dummyStrategy,
    );
  });

  it('falls back to default for unknown name and logs ERROR', () => {
    const errSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    const resolved = resolveContextStrategy(
      'no-such-strategy',
      baseCtx.groupFolder,
    );
    expect(resolved.name).toBe(DEFAULT_CONTEXT_STRATEGY);
    expect(errSpy).toHaveBeenCalled();
  });

  it('returns default when name is undefined (no error log)', () => {
    const errSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    const resolved = resolveContextStrategy(undefined, baseCtx.groupFolder);
    expect(resolved.name).toBe(DEFAULT_CONTEXT_STRATEGY);
    expect(errSpy).not.toHaveBeenCalled();
  });
});
