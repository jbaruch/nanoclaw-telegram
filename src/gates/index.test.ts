import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  GateContext,
  GateDecision,
  GATE_WARN_DURATION_MS,
  RecoverableGateError,
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
  'failing-pass-gate',
  'throwing-gate',
  'slow-gate',
] as const;

const baseCtx: GateContext = {
  groupJid: 'test@g.us',
  groupFolder: 'telegram_test',
  message: {
    text: 'hello world',
    messageId: 'msg-test-1',
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
  // A gate that returns `pass` because it could NOT run — the shape the
  // Haiku classifier emits on api-error/timeout (#671).
  registerGate(
    'failing-pass-gate',
    (): GateDecision => ({
      decision: 'pass',
      reason: 'classifier-failed: api-error',
      failed: true,
    }),
  );
  registerGate('throwing-gate', () => {
    throw new RecoverableGateError('boom');
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

describe('runGateChain — combinator (last-gate-wins, #97)', () => {
  it('empty chain → allow (fail-open)', async () => {
    const result = await runGateChain([], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain).toHaveLength(0);
  });

  // Single-gate chain: preserves pre-#97 behavior.
  it('single allow → allow', async () => {
    const result = await runGateChain(['allow-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain).toHaveLength(1);
    expect(result.chain[0].decision).toBe('allow');
  });

  it('single deny → deny (last-gate deny is terminal)', async () => {
    const result = await runGateChain(['deny-gate'], baseCtx);
    expect(result.finalDecision).toBe('deny');
    expect(result.chain).toHaveLength(1);
  });

  it('single pass → allow (fail-open)', async () => {
    const result = await runGateChain(['pass-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
  });

  // Allow short-circuits — proves the cost-saving behavior. The spied
  // gate must NOT be invoked once a prior allow has fired.
  it('allow then deny → allow; second gate is NOT invoked', async () => {
    const spy = vi.fn(() => ({
      decision: 'deny' as const,
      reason: 'should not run',
    }));
    _unregisterGateForTesting('deny-gate');
    registerGate('deny-gate', spy);

    const result = await runGateChain(['allow-gate', 'deny-gate'], baseCtx);

    expect(result.finalDecision).toBe('allow');
    expect(result.chain).toHaveLength(1);
    expect(result.chain[0].gateName).toBe('allow-gate');
    expect(spy).not.toHaveBeenCalled();
  });

  it('intermediate deny is advisory — chain continues', async () => {
    const result = await runGateChain(
      ['deny-gate', 'allow-gate', 'pass-gate'],
      baseCtx,
    );
    // First allow (gate B) short-circuits; pass-gate not invoked.
    expect(result.finalDecision).toBe('allow');
    expect(result.chain.map((r) => r.gateName)).toEqual([
      'deny-gate',
      'allow-gate',
    ]);
  });

  it('pass then allow → allow', async () => {
    const result = await runGateChain(['pass-gate', 'allow-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain.map((r) => r.decision)).toEqual(['pass', 'allow']);
  });

  it('all pass → allow (fail-open)', async () => {
    const result = await runGateChain(['pass-gate', 'pass-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain.every((r) => r.decision === 'pass')).toBe(true);
  });

  // The wtf-chat shape `[trigger, haiku-classifier]` — full truth table
  // expressed with the synthetic gates. Using gateA/gateB names so the
  // intent is the row, not the gate.
  describe('wtf-chat shape [gateA, gateB]', () => {
    it('[allow, deny] → allow (B never runs)', async () => {
      const spy = vi.fn(() => ({
        decision: 'deny' as const,
        reason: 'never',
      }));
      _unregisterGateForTesting('deny-gate');
      registerGate('deny-gate', spy);
      const r = await runGateChain(['allow-gate', 'deny-gate'], baseCtx);
      expect(r.finalDecision).toBe('allow');
      expect(spy).not.toHaveBeenCalled();
    });

    it('[pass, allow] → allow', async () => {
      const r = await runGateChain(['pass-gate', 'allow-gate'], baseCtx);
      expect(r.finalDecision).toBe('allow');
    });

    it('[pass, deny] → deny (last-gate deny terminal)', async () => {
      const r = await runGateChain(['pass-gate', 'deny-gate'], baseCtx);
      expect(r.finalDecision).toBe('deny');
    });

    it('[deny, allow] → allow (advisory deny ignored, B short-circuits)', async () => {
      const r = await runGateChain(['deny-gate', 'allow-gate'], baseCtx);
      expect(r.finalDecision).toBe('allow');
    });

    it('[deny, deny] → deny (advisory deny ignored, last-gate deny terminal)', async () => {
      const r = await runGateChain(['deny-gate', 'deny-gate'], baseCtx);
      expect(r.finalDecision).toBe('deny');
    });

    it('[deny, pass] → allow (advisory deny ignored, HEALTHY pass + chain-end fail-open)', async () => {
      const r = await runGateChain(['deny-gate', 'pass-gate'], baseCtx);
      expect(r.finalDecision).toBe('allow');
    });
  });

  // #671 — a FAILED downstream gate (classifier api-error/timeout) must
  // not nullify an upstream advisory deny into allow-all. The failed
  // `pass` is distinguished from a healthy `pass` by `failed: true`.
  describe('deny-preservation on downstream gate failure (#671)', () => {
    it('[deny, failing-pass] → deny (advisory deny preserved, not fail-open)', async () => {
      const r = await runGateChain(['deny-gate', 'failing-pass-gate'], baseCtx);
      expect(r.finalDecision).toBe('deny');
      expect(r.reason).toContain('upstream deny preserved');
      expect(r.reason).toContain('always deny');
      // The failing gate is flagged on the chain record — the
      // combinator-internal signal that drove deny-preservation (not
      // part of the `gate decision` log shape).
      expect(r.chain[1].failed).toBe(true);
    });

    it('[deny, throwing] → deny (thrown last gate also collapses the safety net)', async () => {
      const r = await runGateChain(['deny-gate', 'throwing-gate'], baseCtx);
      expect(r.finalDecision).toBe('deny');
      expect(r.reason).toContain('upstream deny preserved');
      expect(r.chain[1].failed).toBe(true);
      expect(r.chain[1].error?.message).toBe('boom');
    });

    it('[deny, unregistered-last] → deny (an unregistered downstream gate also collapses the safety net)', async () => {
      // The unregistered-gate branch must participate in deny-preservation
      // too — otherwise an advisory deny followed by a misconfigured/
      // missing last gate would fall through to fail-open allow.
      const r = await runGateChain(['deny-gate', 'no-such-gate'], baseCtx);
      expect(r.finalDecision).toBe('deny');
      expect(r.reason).toContain('upstream deny preserved');
      expect(r.chain[1].failed).toBe(true);
      expect(r.chain[1].error?.name).toBe('GateNotRegistered');
    });

    it('[pass, failing-pass] → allow (no upstream deny to preserve → fail-open)', async () => {
      const r = await runGateChain(['pass-gate', 'failing-pass-gate'], baseCtx);
      expect(r.finalDecision).toBe('allow');
    });

    it('[deny, failing-pass, allow] → allow (a healthy allow still rescues)', async () => {
      const r = await runGateChain(
        ['deny-gate', 'failing-pass-gate', 'allow-gate'],
        baseCtx,
      );
      expect(r.finalDecision).toBe('allow');
      expect(r.chain).toHaveLength(3);
    });

    it('[failing-pass, deny, pass] → allow (failure BEFORE the advisory deny does not preserve it)', async () => {
      // safetyNetFailed resets at each advisory deny: only a failure
      // from a gate that runs AFTER the deny can preserve it. The
      // healthy last pass means the safety net did not fail.
      const r = await runGateChain(
        ['failing-pass-gate', 'deny-gate', 'pass-gate'],
        baseCtx,
      );
      expect(r.finalDecision).toBe('allow');
    });

    it('[deny, failing-pass, deny] → deny (last-gate deny is decisive regardless)', async () => {
      const r = await runGateChain(
        ['deny-gate', 'failing-pass-gate', 'deny-gate'],
        baseCtx,
      );
      expect(r.finalDecision).toBe('deny');
      expect(r.reason).toBe('always deny');
    });
  });

  describe('three-gate chains', () => {
    it('[deny, deny, allow] → allow (early denies advisory, C short-circuits)', async () => {
      const spy = vi.fn(() => ({
        decision: 'allow' as const,
        reason: 'C wins',
      }));
      _unregisterGateForTesting('allow-gate');
      registerGate('allow-gate', spy);
      const r = await runGateChain(
        ['deny-gate', 'deny-gate', 'allow-gate'],
        baseCtx,
      );
      expect(r.finalDecision).toBe('allow');
      expect(spy).toHaveBeenCalledTimes(1);
      expect(r.chain).toHaveLength(3);
    });

    it('[allow, pass, deny] → allow (A short-circuits; B and C never run)', async () => {
      const passSpy = vi.fn(() => ({
        decision: 'pass' as const,
        reason: 'never',
      }));
      const denySpy = vi.fn(() => ({
        decision: 'deny' as const,
        reason: 'never',
      }));
      _unregisterGateForTesting('pass-gate');
      _unregisterGateForTesting('deny-gate');
      registerGate('pass-gate', passSpy);
      registerGate('deny-gate', denySpy);
      const r = await runGateChain(
        ['allow-gate', 'pass-gate', 'deny-gate'],
        baseCtx,
      );
      expect(r.finalDecision).toBe('allow');
      expect(passSpy).not.toHaveBeenCalled();
      expect(denySpy).not.toHaveBeenCalled();
      expect(r.chain).toHaveLength(1);
    });

    it('[pass, deny, deny] → deny (intermediate deny advisory, last-gate deny terminal)', async () => {
      const r = await runGateChain(
        ['pass-gate', 'deny-gate', 'deny-gate'],
        baseCtx,
      );
      expect(r.finalDecision).toBe('deny');
      expect(r.chain).toHaveLength(3);
    });

    it('[pass, deny, pass] → allow (deny advisory because not last; pass + fail-open)', async () => {
      const r = await runGateChain(
        ['pass-gate', 'deny-gate', 'pass-gate'],
        baseCtx,
      );
      expect(r.finalDecision).toBe('allow');
    });
  });
});

describe('runGateChain — async gates', () => {
  const ASYNC_ALLOW = 'async-allow-gate';
  const ASYNC_THROW = 'async-throw-gate';

  beforeEach(() => {
    if (listRegisteredGates().includes(ASYNC_ALLOW))
      _unregisterGateForTesting(ASYNC_ALLOW);
    if (listRegisteredGates().includes(ASYNC_THROW))
      _unregisterGateForTesting(ASYNC_THROW);
    registerGate(ASYNC_ALLOW, async (): Promise<GateDecision> => {
      await new Promise((r) => setTimeout(r, 1));
      return { decision: 'allow', reason: 'async allow' };
    });
    registerGate(ASYNC_THROW, async (): Promise<GateDecision> => {
      await new Promise((r) => setTimeout(r, 1));
      throw new RecoverableGateError('async boom');
    });
  });

  afterEach(() => {
    if (listRegisteredGates().includes(ASYNC_ALLOW))
      _unregisterGateForTesting(ASYNC_ALLOW);
    if (listRegisteredGates().includes(ASYNC_THROW))
      _unregisterGateForTesting(ASYNC_THROW);
  });

  it('async gate that returns allow → allow', async () => {
    const result = await runGateChain([ASYNC_ALLOW], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain).toHaveLength(1);
    expect(result.chain[0].decision).toBe('allow');
  });

  it('async gate that throws → pass (chain continues)', async () => {
    const result = await runGateChain([ASYNC_THROW, 'allow-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain).toHaveLength(2);
    expect(result.chain[0].decision).toBe('pass');
    expect(result.chain[0].error?.message).toBe('async boom');
  });
});

describe('runGateChain — failure handling', () => {
  it('throwing gate is treated as pass and chain continues', async () => {
    const result = await runGateChain(['throwing-gate', 'allow-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain).toHaveLength(2);
    expect(result.chain[0].decision).toBe('pass');
    expect(result.chain[0].error).toBeDefined();
    expect(result.chain[0].error?.message).toBe('boom');
  });

  it('throwing gate alone with no other gates → allow (fail-open)', async () => {
    const result = await runGateChain(['throwing-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
  });

  it('throwing LAST gate → allow (throw downgraded to pass, fail-open)', async () => {
    // Last-gate-wins requires explicit coverage: a throw on the last
    // gate is downgraded to pass, and chain end with no decisive
    // verdict falls open to allow. (Not "deny because last-gate said
    // deny" — the throw is recorded as pass, not as a synthetic deny.)
    const result = await runGateChain(
      ['allow-gate-2-not-allow', 'throwing-gate'],
      baseCtx,
    );
    // First gate is unknown → recorded as pass with GateNotRegistered.
    // Second gate throws → recorded as pass. Chain end → fail-open.
    expect(result.finalDecision).toBe('allow');
    expect(result.chain[0].decision).toBe('pass');
    expect(result.chain[1].decision).toBe('pass');
    expect(result.chain[1].error?.message).toBe('boom');
  });

  it('unknown gate name is treated as pass, chain continues', async () => {
    const result = await runGateChain(['no-such-gate', 'allow-gate'], baseCtx);
    expect(result.finalDecision).toBe('allow');
    expect(result.chain[0].error?.name).toBe('GateNotRegistered');
  });

  it('logs error when gate throws', async () => {
    const errSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    await runGateChain(['throwing-gate'], baseCtx);
    expect(errSpy).toHaveBeenCalled();
    const call = errSpy.mock.calls.find(
      (c) => typeof c[1] === 'string' && c[1].includes('downgraded to pass'),
    );
    expect(call).toBeDefined();
  });
});

describe('runGateChain — programmer-defect propagation (#674)', () => {
  // A code bug in a gate (TypeError / ReferenceError / RangeError /
  // SyntaxError) must propagate so it surfaces loudly per
  // `coding-policy: error-handling`, NOT be downgraded to a fail-open
  // `pass` that silently degrades the chain. Only an explicit
  // `RecoverableGateError` still downgrades so a known operational blip
  // can't black-hole legitimate traffic.
  const DEFECT_GATE = 'defect-gate';
  const UNEXPECTED_THROW_GATE = 'unexpected-throw-gate';
  const RECOVERABLE_THROW_GATE = 'recoverable-throw-gate';

  afterEach(() => {
    for (const g of [
      DEFECT_GATE,
      UNEXPECTED_THROW_GATE,
      RECOVERABLE_THROW_GATE,
    ]) {
      if (listRegisteredGates().includes(g)) _unregisterGateForTesting(g);
    }
  });

  it.each([
    ['TypeError', TypeError],
    ['ReferenceError', ReferenceError],
    ['RangeError', RangeError],
    ['SyntaxError', SyntaxError],
  ] as const)(
    'propagates a %s thrown by a gate (not downgraded)',
    async (_name, ErrCtor) => {
      registerGate(DEFECT_GATE, (): GateDecision => {
        throw new ErrCtor('synthetic gate code bug');
      });
      await expect(
        runGateChain([DEFECT_GATE, 'allow-gate'], baseCtx),
      ).rejects.toThrow(ErrCtor);
    },
  );

  it('propagates a defect from a non-last gate — later gates are not run', async () => {
    const downstream = vi.fn(
      (): GateDecision => ({ decision: 'allow', reason: 'should not run' }),
    );
    _unregisterGateForTesting('allow-gate');
    registerGate('allow-gate', downstream);
    registerGate(DEFECT_GATE, (): GateDecision => {
      throw new TypeError('boom');
    });
    await expect(
      runGateChain([DEFECT_GATE, 'allow-gate'], baseCtx),
    ).rejects.toBeInstanceOf(TypeError);
    expect(downstream).not.toHaveBeenCalled();
  });

  it('a plain Error now propagates instead of being downgraded', async () => {
    registerGate(UNEXPECTED_THROW_GATE, (): GateDecision => {
      throw new Error('transient DB read failure');
    });
    await expect(
      runGateChain([UNEXPECTED_THROW_GATE, 'allow-gate'], baseCtx),
    ).rejects.toThrow('transient DB read failure');
  });

  it('an explicit RecoverableGateError is downgraded to a failed pass', async () => {
    registerGate(RECOVERABLE_THROW_GATE, (): GateDecision => {
      throw new RecoverableGateError('transient DB read failure');
    });
    const result = await runGateChain(
      [RECOVERABLE_THROW_GATE, 'allow-gate'],
      baseCtx,
    );
    expect(result.finalDecision).toBe('allow');
    expect(result.chain[0].decision).toBe('pass');
    expect(result.chain[0].error?.message).toBe('transient DB read failure');
    expect(result.chain[0].failed).toBe(true);
  });

  it('logs gate context (gateName) before propagating an unexpected error', async () => {
    // The caller's catch (`group-queue.ts` `runForGroup`) logs only
    // `{ groupJid, err }`, so the chain must log gateName/groupFolder
    // before rethrowing or that triage context is lost.
    const errSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    registerGate(UNEXPECTED_THROW_GATE, (): GateDecision => {
      throw new TypeError('boom');
    });
    await expect(
      runGateChain([UNEXPECTED_THROW_GATE], baseCtx),
    ).rejects.toThrow(TypeError);
    const propagateLog = errSpy.mock.calls.find(
      (c) =>
        typeof c[1] === 'string' &&
        c[1].includes('propagating') &&
        typeof c[0] === 'object' &&
        (c[0] as { gateName?: string }).gateName === UNEXPECTED_THROW_GATE,
    );
    expect(propagateLog).toBeDefined();
  });
});

describe('runGateChain — observability', () => {
  it('warns when a gate exceeds the duration budget', async () => {
    const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {});
    await runGateChain(['slow-gate'], baseCtx);
    const slowWarn = warnSpy.mock.calls.find(
      (c) => typeof c[1] === 'string' && c[1].includes('exceeded warn'),
    );
    expect(slowWarn).toBeDefined();
  });

  it('records duration for each gate run', async () => {
    const result = await runGateChain(['allow-gate', 'pass-gate'], baseCtx);
    for (const r of result.chain) {
      expect(typeof r.durationMs).toBe('number');
      expect(r.durationMs).toBeGreaterThanOrEqual(0);
    }
    expect(result.totalDurationMs).toBeGreaterThanOrEqual(0);
  });

  it('emits one chain-complete log line', async () => {
    // Hot-path observability is at debug level (host-side gates fire
    // on every inbound poll; info-tier per-call would dwarf the
    // orchestrator log).
    const debugSpy = vi.spyOn(logger, 'debug').mockImplementation(() => {});
    await runGateChain(['allow-gate'], baseCtx);
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
