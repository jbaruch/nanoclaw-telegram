import { describe, it, expect, beforeEach, afterEach } from 'vitest';

import { _initTestDatabase } from '../db.js';
import {
  _setRegisteredGroups,
  evaluateGateChain,
  gateAllowsSpawn,
  resolveGatesForGroup,
} from '../index.js';
import {
  RecoverableGateError,
  _unregisterGateForTesting,
  listRegisteredGates,
  registerGate,
} from './index.js';
import type { NewMessage, RegisteredGroup } from '../types.js';

const TEST_GATE = 'test-router-allow';
const TEST_DENY = 'test-router-deny';
const TEST_THROW = 'test-router-throw';
const TEST_UNEXPECTED_THROW = 'test-router-unexpected-throw';

beforeEach(() => {
  _initTestDatabase();
  _setRegisteredGroups({});
  for (const g of [TEST_GATE, TEST_DENY, TEST_THROW, TEST_UNEXPECTED_THROW]) {
    if (listRegisteredGates().includes(g)) _unregisterGateForTesting(g);
  }
  registerGate(TEST_GATE, () => ({ decision: 'allow', reason: 'router test' }));
  registerGate(TEST_DENY, () => ({ decision: 'deny', reason: 'router test' }));
  registerGate(TEST_THROW, () => {
    throw new RecoverableGateError('router test boom');
  });
});

afterEach(() => {
  for (const g of [TEST_GATE, TEST_DENY, TEST_THROW, TEST_UNEXPECTED_THROW]) {
    if (listRegisteredGates().includes(g)) _unregisterGateForTesting(g);
  }
});

function group(overrides: Partial<RegisteredGroup> = {}): RegisteredGroup {
  return {
    name: 'g',
    folder: 'telegram_g',
    trigger: '@andy',
    added_at: '2024-01-01T00:00:00Z',
    ...overrides,
  };
}

// Deterministic id generator scoped to this test file. Per
// `jbaruch/coding-policy: testing-standards`: tests must be
// deterministic, no self-generated random data — collision-free
// numeric counter is enough since each test starts a fresh module
// scope (vitest resets module state between files).
let _msgIdCounter = 0;
function msg(content: string, sender = 's@s.whatsapp.net'): NewMessage {
  _msgIdCounter += 1;
  return {
    id: `mid-${_msgIdCounter}`,
    chat_jid: 'g@g.us',
    sender,
    sender_name: 'Sender',
    content,
    timestamp: '2026-01-01T00:00:00.000Z',
  };
}

describe('resolveGatesForGroup — migration path A', () => {
  it('main groups get empty chain', () => {
    const g = group({ isMain: true });
    expect(resolveGatesForGroup(g)).toEqual([]);
  });

  it('non-main with requiresTrigger=undefined → ["trigger"]', () => {
    const g = group({ requiresTrigger: undefined });
    expect(resolveGatesForGroup(g)).toEqual(['trigger']);
  });

  it('non-main with requiresTrigger=true → ["trigger"]', () => {
    const g = group({ requiresTrigger: true });
    expect(resolveGatesForGroup(g)).toEqual(['trigger']);
  });

  it('non-main with requiresTrigger=false → []', () => {
    const g = group({ requiresTrigger: false });
    expect(resolveGatesForGroup(g)).toEqual([]);
  });

  it('explicit containerConfig.gates wins over requiresTrigger', () => {
    const g = group({
      requiresTrigger: true,
      containerConfig: { gates: [TEST_GATE] },
    });
    expect(resolveGatesForGroup(g)).toEqual([TEST_GATE]);
  });

  it('explicit empty containerConfig.gates → no gates', () => {
    const g = group({
      requiresTrigger: true,
      containerConfig: { gates: [] },
    });
    expect(resolveGatesForGroup(g)).toEqual([]);
  });

  // requiresTrigger=false — implicit-trigger runs only when the group
  // has patterns configured; no patterns → empty chain.
  it('requiresTrigger=false + no patterns → []', () => {
    const g = group({ requiresTrigger: false });
    expect(resolveGatesForGroup(g)).toEqual([]);
  });

  it('requiresTrigger=false + has patterns → ["trigger"]', () => {
    const g = group({
      requiresTrigger: false,
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            kind: 'keyword',
            pattern: 'bots',
            source: 'owner-set',
            precision: 0,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          },
        ],
      },
    });
    expect(resolveGatesForGroup(g)).toEqual(['trigger']);
  });

  it('requiresTrigger=undefined + has patterns → ["trigger"] (undefined treated as permissive)', () => {
    const g = group({
      requiresTrigger: undefined,
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            kind: 'keyword',
            pattern: 'bots',
            source: 'owner-set',
            precision: 0,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          },
        ],
      },
    });
    expect(resolveGatesForGroup(g)).toEqual(['trigger']);
  });

  it('main group with no explicit gates → [] (no implicit trigger)', () => {
    const g = group({
      isMain: true,
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            kind: 'keyword',
            pattern: 'whatever',
            source: 'owner-set',
            precision: 0,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          },
        ],
      },
    });
    expect(resolveGatesForGroup(g)).toEqual([]);
  });
});

describe('gateAllowsSpawn', () => {
  it('empty chain → always allow (preserves pre-#80 behaviour)', async () => {
    expect(await gateAllowsSpawn(group(), 'g@g.us', [msg('hi')], [])).toBe(
      true,
    );
  });

  it('allow gate proceeds', async () => {
    expect(
      await gateAllowsSpawn(group(), 'g@g.us', [msg('hi')], [TEST_GATE]),
    ).toBe(true);
  });

  it('deny gate skips spawn', async () => {
    expect(
      await gateAllowsSpawn(group(), 'g@g.us', [msg('hi')], [TEST_DENY]),
    ).toBe(false);
  });

  it('any one allow across messages is enough to spawn', async () => {
    expect(
      await gateAllowsSpawn(
        group(),
        'g@g.us',
        [msg('hi'), msg('@andy hi')],
        [TEST_GATE],
      ),
    ).toBe(true);
  });

  it('throwing gate does not black-hole — falls open to allow', async () => {
    expect(
      await gateAllowsSpawn(group(), 'g@g.us', [msg('hi')], [TEST_THROW]),
    ).toBe(true);
  });

  it('unexpected gate throw propagates to the caller', async () => {
    registerGate(TEST_UNEXPECTED_THROW, () => {
      throw new Error('router test unexpected boom');
    });
    await expect(
      gateAllowsSpawn(group(), 'g@g.us', [msg('hi')], [TEST_UNEXPECTED_THROW]),
    ).rejects.toThrow('router test unexpected boom');
  });

  it('built-in trigger gate denies when no patterns configured does not apply (pass)', async () => {
    // No triggerPatterns → trigger gate returns pass → fail-open allow.
    const g = group();
    expect(await gateAllowsSpawn(g, 'g@g.us', [msg('hi')], ['trigger'])).toBe(
      true,
    );
  });

  it('built-in trigger gate allows on keyword match', async () => {
    const g = group({
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            kind: 'keyword',
            pattern: '@andy',
            source: 'owner-set',
            precision: 0,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          },
        ],
      },
    });
    expect(
      await gateAllowsSpawn(g, 'g@g.us', [msg('@andy hello')], ['trigger']),
    ).toBe(true);
    expect(
      await gateAllowsSpawn(g, 'g@g.us', [msg('hello')], ['trigger']),
    ).toBe(false);
  });
});

describe('evaluateGateChain', () => {
  it('returns the id of the message that produced the allow verdict', async () => {
    const a = msg('hi');
    const b = msg('@andy hi');
    const result = await evaluateGateChain(
      group(),
      'g@g.us',
      [a, b],
      [TEST_GATE],
    );
    expect(result.allowed).toBe(true);
    // First-allow short-circuits on the first message — TEST_GATE
    // allows everything, so it's `a`.
    expect(result.allowedMessageId).toBe(a.id);
  });

  it('returns last-message id when chain is empty (always-allow short-circuit)', async () => {
    const a = msg('hi');
    const b = msg('bye');
    const result = await evaluateGateChain(group(), 'g@g.us', [a, b], []);
    expect(result.allowed).toBe(true);
    expect(result.allowedMessageId).toBe(b.id);
  });

  it('returns no allowedMessageId on deny', async () => {
    const result = await evaluateGateChain(
      group(),
      'g@g.us',
      [msg('hi')],
      [TEST_DENY],
    );
    expect(result.allowed).toBe(false);
    expect(result.allowedMessageId).toBeUndefined();
  });

  it('threads the trigger-matching message id through ["trigger"]', async () => {
    const g = group({
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            kind: 'keyword',
            pattern: '@andy',
            source: 'owner-set',
            precision: 0,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          },
        ],
      },
    });
    const noTrigger = msg('hello');
    const triggerMsg = msg('@andy hello');
    const result = await evaluateGateChain(
      g,
      'g@g.us',
      [noTrigger, triggerMsg],
      ['trigger'],
    );
    expect(result.allowed).toBe(true);
    expect(result.allowedMessageId).toBe(triggerMsg.id);
  });
});
