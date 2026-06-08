import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import { _initTestDatabase } from '../db.js';
import {
  _setRegisteredGroups,
  evaluateGateChain,
  gateAllowsSpawn,
  resolveGatesForGroup,
} from '../index.js';
import {
  _unregisterGateForTesting,
  listRegisteredGates,
  registerGate,
} from './index.js';
import type { NewMessage, RegisteredGroup } from '../types.js';

const TEST_GATE = 'test-router-allow';
const TEST_DENY = 'test-router-deny';
const TEST_THROW = 'test-router-throw';

beforeEach(() => {
  _initTestDatabase();
  _setRegisteredGroups({});
  for (const g of [TEST_GATE, TEST_DENY, TEST_THROW]) {
    if (listRegisteredGates().includes(g)) _unregisterGateForTesting(g);
  }
  registerGate(TEST_GATE, () => ({ decision: 'allow', reason: 'router test' }));
  registerGate(TEST_DENY, () => ({ decision: 'deny', reason: 'router test' }));
  registerGate(TEST_THROW, () => {
    throw new Error('router test boom');
  });
});

afterEach(() => {
  for (const g of [TEST_GATE, TEST_DENY, TEST_THROW]) {
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

  // requiresTrigger=false — implicit-trigger short-circuit before
  // Stage 2 (issue: wtf chat had chain [haiku-classifier] only,
  // paying for grey-zone messages a deterministic match would have
  // caught for free).
  it('requiresTrigger=false + no patterns + stage2=false → []', () => {
    const g = group({ requiresTrigger: false });
    expect(resolveGatesForGroup(g)).toEqual([]);
  });

  it('requiresTrigger=false + has patterns + stage2=false → ["trigger"]', () => {
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

  it('requiresTrigger=false + has patterns + stage2=true → ["trigger","haiku-classifier"] (wtf-chat shape)', () => {
    const g = group({
      requiresTrigger: false,
      containerConfig: { stage2Enabled: true },
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            kind: 'mention',
            pattern: 'TestBot',
            source: 'owner-set',
            precision: 0,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          },
          {
            kind: 'keyword',
            pattern: 'боты',
            source: 'owner-set',
            precision: 0,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          },
        ],
      },
    });
    expect(resolveGatesForGroup(g)).toEqual(['trigger', 'haiku-classifier']);
  });

  it('requiresTrigger=true + has patterns + stage2=true → ["trigger"] (strict-gating: no Haiku)', () => {
    // Strict-trigger groups respond only to deterministic matches.
    // There is no grey zone for Stage 2 to adjudicate, and with the
    // last-gate-wins combinator a Stage 1 deny would fall through to
    // Haiku — silently breaking the strict-gating contract.
    const g = group({
      requiresTrigger: true,
      containerConfig: { stage2Enabled: true },
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
    expect(resolveGatesForGroup(g)).toEqual(['trigger']);
  });

  it('requiresTrigger=undefined + has patterns + stage2=true → ["trigger","haiku-classifier"] (undefined treated as permissive)', () => {
    const g = group({
      requiresTrigger: undefined,
      containerConfig: { stage2Enabled: true },
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
    expect(resolveGatesForGroup(g)).toEqual(['trigger', 'haiku-classifier']);
  });

  it('requiresTrigger=true + explicit gates ["trigger","haiku-classifier"] + stage2=true → explicit wins', () => {
    // Operator-pinned explicit gates take precedence over the
    // strict-trigger-skips-Stage-2 predicate. The operator knows what
    // they are asking for.
    const g = group({
      requiresTrigger: true,
      containerConfig: {
        gates: ['trigger', 'haiku-classifier'],
        stage2Enabled: true,
      },
    });
    expect(resolveGatesForGroup(g)).toEqual(['trigger', 'haiku-classifier']);
  });

  it('explicit containerConfig.gates + stage2=true → custom + haiku-classifier (explicit wins)', () => {
    const g = group({
      requiresTrigger: false,
      containerConfig: { gates: ['custom'], stage2Enabled: true },
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
    expect(resolveGatesForGroup(g)).toEqual(['custom', 'haiku-classifier']);
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

// Stage 2 (#83) chain-resolution + execution. The gate is registered
// at module load; we drive verdicts via the Anthropic client mock seam.
describe('resolveGatesForGroup — stage2 chain ordering', () => {
  it('stage2Enabled appends haiku-classifier last (after trigger) for permissive groups', () => {
    // Permissive (requiresTrigger=false) groups with patterns get the
    // free deterministic trigger first, then Haiku for the grey zone.
    const g = group({
      requiresTrigger: false,
      containerConfig: { stage2Enabled: true },
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
    expect(resolveGatesForGroup(g)).toEqual(['trigger', 'haiku-classifier']);
  });

  it('stage2Enabled appends haiku-classifier even when explicit gates set', () => {
    const g = group({
      containerConfig: {
        gates: ['trigger', 'custom-gate'],
        stage2Enabled: true,
      },
    });
    expect(resolveGatesForGroup(g)).toEqual([
      'trigger',
      'custom-gate',
      'haiku-classifier',
    ]);
  });

  it('stage2Enabled=false leaves chain unchanged', () => {
    const g = group({
      requiresTrigger: true,
      containerConfig: { stage2Enabled: false },
    });
    expect(resolveGatesForGroup(g)).toEqual(['trigger']);
  });

  it('stage2Enabled=true on a main group still appends haiku-classifier', () => {
    const g = group({ isMain: true, containerConfig: { stage2Enabled: true } });
    expect(resolveGatesForGroup(g)).toEqual(['haiku-classifier']);
  });

  it('does not duplicate haiku-classifier when explicit gates already include it', () => {
    const g = group({
      containerConfig: {
        gates: ['haiku-classifier'],
        stage2Enabled: true,
      },
    });
    expect(resolveGatesForGroup(g)).toEqual(['haiku-classifier']);
  });
});

describe('gateAllowsSpawn — stage2Enabled integration', () => {
  // Build a fresh mock client per case via the haiku-classifier seam.
  // We still must seed the registered group in the DB (the gate calls
  // getRegisteredGroup to resolve modelId / strategy).
  const STAGE2_JID = 'stage2@g.us';
  const STAGE2_FOLDER = 'telegram_stage2';

  function buildHaikuMockClient(opts: {
    intent?: 'yes' | 'no';
    throwError?: Error;
    response?: import('@anthropic-ai/sdk').default.Message;
  }): import('@anthropic-ai/sdk').default {
    const create = vi.fn(async () => {
      if (opts.throwError) throw opts.throwError;
      if (opts.response) return opts.response;
      return {
        id: 'msg_int',
        type: 'message',
        role: 'assistant',
        model: 'claude-haiku-4-5-20251001',
        stop_reason: 'tool_use',
        stop_sequence: null,
        content: [
          {
            type: 'tool_use',
            id: 'tu_int',
            name: 'classify_intent',
            input: {
              intent: opts.intent ?? 'yes',
              confidence: 0.9,
              reason: 'integration test',
            },
          },
        ],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      };
    });
    return {
      messages: { create },
    } as unknown as import('@anthropic-ai/sdk').default;
  }

  beforeEach(async () => {
    const { _writeRawRegisteredGroup } = await import('../db.js');
    _writeRawRegisteredGroup({
      jid: STAGE2_JID,
      name: 'Stage2 Test',
      folder: STAGE2_FOLDER,
      trigger: '@andy',
      added_at: '2024-01-01T00:00:00Z',
      container_config: JSON.stringify({ stage2Enabled: true }),
    });
  });

  afterEach(async () => {
    const { _setAnthropicClientForTesting } =
      await import('./haiku-classifier.js');
    _setAnthropicClientForTesting(undefined);
  });

  it('haiku yes → spawn allowed', async () => {
    const { _setAnthropicClientForTesting } =
      await import('./haiku-classifier.js');
    _setAnthropicClientForTesting(buildHaikuMockClient({ intent: 'yes' }));
    const g: RegisteredGroup = {
      name: 'Stage2 Test',
      folder: STAGE2_FOLDER,
      trigger: '@andy',
      added_at: '2024-01-01T00:00:00Z',
      containerConfig: { stage2Enabled: true },
    };
    expect(
      await gateAllowsSpawn(
        g,
        STAGE2_JID,
        [msg('hi there')],
        ['haiku-classifier'],
      ),
    ).toBe(true);
  });

  it('haiku no → spawn denied', async () => {
    const { _setAnthropicClientForTesting } =
      await import('./haiku-classifier.js');
    _setAnthropicClientForTesting(buildHaikuMockClient({ intent: 'no' }));
    const g: RegisteredGroup = {
      name: 'Stage2 Test',
      folder: STAGE2_FOLDER,
      trigger: '@andy',
      added_at: '2024-01-01T00:00:00Z',
      containerConfig: { stage2Enabled: true },
    };
    expect(
      await gateAllowsSpawn(
        g,
        STAGE2_JID,
        [msg('hi there')],
        ['haiku-classifier'],
      ),
    ).toBe(false);
  });

  it('haiku API error → pass → fail-open allow (chain resolution)', async () => {
    const { _setAnthropicClientForTesting } =
      await import('./haiku-classifier.js');
    _setAnthropicClientForTesting(
      buildHaikuMockClient({ throwError: new Error('rate limited') }),
    );
    const g: RegisteredGroup = {
      name: 'Stage2 Test',
      folder: STAGE2_FOLDER,
      trigger: '@andy',
      added_at: '2024-01-01T00:00:00Z',
      containerConfig: { stage2Enabled: true },
    };
    expect(
      await gateAllowsSpawn(
        g,
        STAGE2_JID,
        [msg('hi there')],
        ['haiku-classifier'],
      ),
    ).toBe(true);
  });
});

// Wtf-chat shape end-to-end (#97 last-gate-wins). The chain is
// `[trigger, haiku-classifier]` with real built-in gates and a mocked
// Haiku client. These cases exercise the combinator change directly:
// trigger=allow must short-circuit before Haiku is hit, trigger=deny
// must fall through so Haiku still adjudicates.
describe('gateAllowsSpawn — [trigger, haiku-classifier] last-gate-wins', () => {
  const JID = 'wtf@g.us';
  const FOLDER = 'telegram_wtf';

  const triggerPatterns = {
    version: 1 as const,
    patterns: [
      {
        kind: 'keyword' as const,
        pattern: '@testbot',
        source: 'owner-set' as const,
        precision: 0,
        sample_count: 0,
        last_matched_at: null,
        last_updated_at: null,
      },
    ],
  };

  function makeMockHaikuClient(
    intent: 'yes' | 'no',
  ): import('@anthropic-ai/sdk').default {
    const create = vi.fn(async () => ({
      id: 'msg_wtf',
      type: 'message' as const,
      role: 'assistant' as const,
      model: 'claude-haiku-4-5-20251001',
      stop_reason: 'tool_use' as const,
      stop_sequence: null,
      content: [
        {
          type: 'tool_use' as const,
          id: 'tu',
          name: 'classify_intent',
          input: { intent, confidence: 0.9, reason: 'wtf' },
        },
      ],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
    }));
    return {
      messages: { create },
    } as unknown as import('@anthropic-ai/sdk').default;
  }

  beforeEach(async () => {
    const { _writeRawRegisteredGroup } = await import('../db.js');
    _writeRawRegisteredGroup({
      jid: JID,
      name: 'WTF',
      folder: FOLDER,
      // `trigger` arg is written into the trigger_pattern column. The
      // dual-mode reader parses JSON when present, otherwise falls
      // back to legacy keyword shape.
      trigger: JSON.stringify(triggerPatterns),
      added_at: '2024-01-01T00:00:00Z',
      container_config: JSON.stringify({ stage2Enabled: true }),
    });
  });

  afterEach(async () => {
    const { _setAnthropicClientForTesting } =
      await import('./haiku-classifier.js');
    _setAnthropicClientForTesting(undefined);
  });

  it('trigger=allow → Haiku NEVER called, spawn allowed', async () => {
    const { _setAnthropicClientForTesting } =
      await import('./haiku-classifier.js');
    // Mock would deny — proves Haiku doesn't run.
    const client = makeMockHaikuClient('no');
    _setAnthropicClientForTesting(client);

    const g: RegisteredGroup = {
      name: 'WTF',
      folder: FOLDER,
      trigger: '@testbot',
      added_at: '2024-01-01T00:00:00Z',
      containerConfig: { stage2Enabled: true },
      triggerPatterns,
    };
    const allowed = await gateAllowsSpawn(
      g,
      JID,
      [msg('@testbot please help')],
      ['trigger', 'haiku-classifier'],
    );
    expect(allowed).toBe(true);
    // The cost-saving claim — Haiku was not invoked.
    expect(
      (client.messages.create as ReturnType<typeof vi.fn>).mock.calls,
    ).toHaveLength(0);
  });

  it('trigger=deny + haiku=allow → spawn allowed (Stage 2 catches grey-zone yes)', async () => {
    const { _setAnthropicClientForTesting } =
      await import('./haiku-classifier.js');
    const client = makeMockHaikuClient('yes');
    _setAnthropicClientForTesting(client);

    const g: RegisteredGroup = {
      name: 'WTF',
      folder: FOLDER,
      trigger: '@testbot',
      added_at: '2024-01-01T00:00:00Z',
      containerConfig: { stage2Enabled: true },
      triggerPatterns,
    };
    // No keyword match → trigger denies. Haiku says yes → final allow.
    const allowed = await gateAllowsSpawn(
      g,
      JID,
      [msg('what do you think about this?')],
      ['trigger', 'haiku-classifier'],
    );
    expect(allowed).toBe(true);
    expect(
      (client.messages.create as ReturnType<typeof vi.fn>).mock.calls.length,
    ).toBeGreaterThanOrEqual(1);
  });

  it('trigger=deny + haiku=deny → spawn denied (last-gate decisive)', async () => {
    const { _setAnthropicClientForTesting } =
      await import('./haiku-classifier.js');
    _setAnthropicClientForTesting(makeMockHaikuClient('no'));

    const g: RegisteredGroup = {
      name: 'WTF',
      folder: FOLDER,
      trigger: '@testbot',
      added_at: '2024-01-01T00:00:00Z',
      containerConfig: { stage2Enabled: true },
      triggerPatterns,
    };
    const allowed = await gateAllowsSpawn(
      g,
      JID,
      [msg('random chatter unrelated')],
      ['trigger', 'haiku-classifier'],
    );
    expect(allowed).toBe(false);
  });

  it('trigger=deny + haiku=FAIL → spawn DENIED (advisory deny preserved, #671)', async () => {
    // The money-bleed scenario end-to-end: the classifier can't reach
    // the API ("Connection error"), so it returns a failed pass. The
    // deterministic trigger deny (no keyword match) must NOT fall
    // through to fail-open allow — the chain preserves the deny so the
    // bot does NOT spawn on an untagged message.
    const { _setAnthropicClientForTesting } =
      await import('./haiku-classifier.js');
    const connErr = Object.assign(new Error('Connection error.'), {
      code: 'ECONNREFUSED',
    });
    _setAnthropicClientForTesting({
      messages: {
        create: vi.fn(async () => {
          throw connErr;
        }),
      },
    } as unknown as import('@anthropic-ai/sdk').default);

    const g: RegisteredGroup = {
      name: 'WTF',
      folder: FOLDER,
      trigger: '@testbot',
      added_at: '2024-01-01T00:00:00Z',
      containerConfig: { stage2Enabled: true },
      triggerPatterns,
    };
    const allowed = await gateAllowsSpawn(
      g,
      JID,
      [msg('random chatter unrelated')],
      ['trigger', 'haiku-classifier'],
    );
    expect(allowed).toBe(false);
  });
});
