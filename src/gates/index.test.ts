import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  GateContext,
  GateDecision,
  GATE_WARN_DURATION_MS,
  _unregisterGateForTesting,
  listRegisteredGates,
  registerGate,
  runGateChain,
} from './index.js';
import { logger } from '../logger.js';

const TEST_GATES = [
  'allow-gate',
  'deny-gate',
  'pass-gate',
  'throwing-gate',
  'slow-gate',
] as const;

const baseCtx: GateContext = {
  groupJid: 'test@g.us',
  groupFolder: 'telegram_test',
  message: {
    text: 'hello world',
    senderJid: 'sender@s.whatsapp.net',
  },
  triggerPatterns: null,
};

beforeEach(() => {
  for (const g of TEST_GATES) {
    if (listRegisteredGates().includes(g)) _unregisterGateForTesting(g);
  }
  registerGate(
    'allow-gate',
    (): GateDecision => ({ decision: 'allow', reason: 'always allow' }),
  );
  registerGate(
    'deny-gate',
    (): GateDecision => ({ decision: 'deny', reason: 'always deny' }),
  );
  registerGate(
    'pass-gate',
    (): GateDecision => ({ decision: 'pass', reason: 'no opinion' }),
  );
  registerGate('throwing-gate', () => {
    throw new Error('boom');
  });
  registerGate('slow-gate', (): GateDecision => {
    const target = Date.now() + GATE_WARN_DURATION_MS + 20;
    while (Date.now() < target) {
      // busy-wait so the chain measures real wall-time over the warn budget
    }
    return { decision: 'allow', reason: 'slow but allowed' };
  });
});

afterEach(() => {
  for (const g of TEST_GATES) {
    if (listRegisteredGates().includes(g)) _unregisterGateForTesting(g);
  }
  vi.restoreAllMocks();
});

describe('runGateChain — combinator', () => {
  it('empty chain → allow (fail-open)', () => {
    const result = runGateChain([], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain).toHaveLength(0);
  });

  it('single allow → allow', () => {
    const result = runGateChain(['allow-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain).toHaveLength(1);
    expect(result.chain[0].decision).toBe('allow');
  });

  it('single deny → deny', () => {
    const result = runGateChain(['deny-gate'], baseCtx);
    expect(result.finalDecision).toBe('deny');
  });

  it('single pass → allow (fail-open when no opinion)', () => {
    const result = runGateChain(['pass-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
  });

  it('allow then deny → deny (deny short-circuits)', () => {
    const result = runGateChain(['allow-gate', 'deny-gate'], baseCtx);
    expect(result.finalDecision).toBe('deny');
    expect(result.chain).toHaveLength(2);
  });

  it('deny stops further evaluation', () => {
    const result = runGateChain(
      ['deny-gate', 'allow-gate', 'pass-gate'],
      baseCtx,
    );
    expect(result.finalDecision).toBe('deny');
    expect(result.chain).toHaveLength(1);
    expect(result.chain[0].gateName).toBe('deny-gate');
  });

  it('pass then allow → allow', () => {
    const result = runGateChain(['pass-gate', 'allow-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain.map((r) => r.decision)).toEqual(['pass', 'allow']);
  });

  it('all pass → allow (fail-open)', () => {
    const result = runGateChain(['pass-gate', 'pass-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain.every((r) => r.decision === 'pass')).toBe(true);
  });
});

describe('runGateChain — failure handling', () => {
  it('throwing gate is treated as pass and chain continues', () => {
    const result = runGateChain(['throwing-gate', 'allow-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain).toHaveLength(2);
    expect(result.chain[0].decision).toBe('pass');
    expect(result.chain[0].error).toBeDefined();
    expect(result.chain[0].error?.message).toBe('boom');
  });

  it('throwing gate alone with no other gates → allow (fail-open)', () => {
    const result = runGateChain(['throwing-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
  });

  it('unknown gate name is treated as pass, chain continues', () => {
    const result = runGateChain(['no-such-gate', 'allow-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain[0].error?.name).toBe('GateNotRegistered');
  });

  it('logs error when gate throws', () => {
    const errSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    runGateChain(['throwing-gate'], baseCtx);
    expect(errSpy).toHaveBeenCalled();
    const call = errSpy.mock.calls.find(
      (c) => typeof c[1] === 'string' && c[1].includes('downgraded to pass'),
    );
    expect(call).toBeDefined();
  });
});

describe('runGateChain — observability', () => {
  it('warns when a gate exceeds the duration budget', () => {
    const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {});
    runGateChain(['slow-gate'], baseCtx);
    const slowWarn = warnSpy.mock.calls.find(
      (c) => typeof c[1] === 'string' && c[1].includes('exceeded warn'),
    );
    expect(slowWarn).toBeDefined();
  });

  it('records duration for each gate run', () => {
    const result = runGateChain(['allow-gate', 'pass-gate'], baseCtx);
    for (const r of result.chain) {
      expect(typeof r.durationMs).toBe('number');
      expect(r.durationMs).toBeGreaterThanOrEqual(0);
    }
    expect(result.totalDurationMs).toBeGreaterThanOrEqual(0);
  });

  it('emits one chain-complete log line', () => {
    // Hot-path observability is at debug level (host-side gates fire
    // on every inbound poll; info-tier per-call would dwarf the
    // orchestrator log).
    const debugSpy = vi.spyOn(logger, 'debug').mockImplementation(() => {});
    runGateChain(['allow-gate'], baseCtx);
    const completeLines = debugSpy.mock.calls.filter(
      (c) => typeof c[1] === 'string' && c[1] === 'gate chain complete',
    );
    expect(completeLines).toHaveLength(1);
  });
});

describe('registerGate', () => {
  it('throws on duplicate registration', () => {
    expect(() =>
      registerGate('allow-gate', () => ({ decision: 'allow', reason: 'x' })),
    ).toThrow(/already registered/);
  });
});
