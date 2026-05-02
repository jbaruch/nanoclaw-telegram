import { describe, it, expect } from 'vitest';

import { triggerGate } from './trigger.js';
import type { GateContext } from './index.js';
import type {
  TriggerPattern,
  TriggerPatternConfig,
  TriggerPatternKind,
} from '../types.js';

function pattern(
  kind: TriggerPatternKind,
  body: string,
  source: TriggerPattern['source'] = 'owner-set',
): TriggerPattern {
  return {
    pattern: body,
    kind,
    source,
    precision: 0,
    sample_count: 0,
    last_matched_at: null,
    last_updated_at: null,
  };
}

function cfg(...patterns: TriggerPattern[]): TriggerPatternConfig {
  return { version: 1, patterns };
}

function ctx(
  text: string,
  overrides: Partial<GateContext['message']> = {},
  triggerPatterns: TriggerPatternConfig | null = null,
): GateContext {
  return {
    groupJid: 'g@g.us',
    groupFolder: 'g',
    message: {
      text,
      senderJid: 's@s.whatsapp.net',
      ...overrides,
    },
    triggerPatterns,
  };
}

describe('triggerGate — empty patterns', () => {
  it('returns pass when triggerPatterns is null', () => {
    const result = triggerGate(ctx('@andy hello', {}, null));
    expect(result.decision).toBe('pass');
  });

  it('returns pass when patterns array is empty', () => {
    const result = triggerGate(
      ctx('@andy hello', {}, { version: 1, patterns: [] }),
    );
    expect(result.decision).toBe('pass');
  });
});

describe('triggerGate — keyword kind', () => {
  it('matches keyword at start of message', () => {
    const result = triggerGate(
      ctx('@andy do thing', {}, cfg(pattern('keyword', '@andy'))),
    );
    expect(result.decision).toBe('allow');
  });

  it('matches keyword after whitespace', () => {
    const result = triggerGate(
      ctx('hey @andy please', {}, cfg(pattern('keyword', '@andy'))),
    );
    expect(result.decision).toBe('allow');
  });

  it('case-insensitive', () => {
    const result = triggerGate(
      ctx('@ANDY hi', {}, cfg(pattern('keyword', '@andy'))),
    );
    expect(result.decision).toBe('allow');
  });

  it('does not match substring without word boundary', () => {
    // Word-boundary semantics from buildTriggerPattern: trailing `\b`
    // means `@andy` should not match `@andybot`.
    const result = triggerGate(
      ctx('@andybot hi', {}, cfg(pattern('keyword', '@andy'))),
    );
    expect(result.decision).toBe('deny');
  });

  it('denies when no keyword matches', () => {
    const result = triggerGate(
      ctx('hello world', {}, cfg(pattern('keyword', '@andy'))),
    );
    expect(result.decision).toBe('deny');
  });

  it('handles regex metacharacters in pattern as literal', () => {
    // escapeRegex must keep `.` literal — no false-positive on
    // `@andy.bot` triggering for arbitrary input.
    const result = triggerGate(
      ctx('@andyxbot', {}, cfg(pattern('keyword', '@andy.bot'))),
    );
    expect(result.decision).toBe('deny');
  });
});

describe('triggerGate — mention kind', () => {
  it('matches @<pattern> in text', () => {
    const result = triggerGate(
      ctx('hey @andy', {}, cfg(pattern('mention', 'andy'))),
    );
    expect(result.decision).toBe('allow');
  });

  it('matches when pattern includes leading @', () => {
    const result = triggerGate(
      ctx('hey @andy', {}, cfg(pattern('mention', '@andy'))),
    );
    expect(result.decision).toBe('allow');
  });

  it('matches via mentions[] array', () => {
    const result = triggerGate(
      ctx(
        'hello',
        { mentions: ['andy', 'bob'] },
        cfg(pattern('mention', 'andy')),
      ),
    );
    expect(result.decision).toBe('allow');
  });

  it('denies when mention not present', () => {
    const result = triggerGate(
      ctx('hello bob', {}, cfg(pattern('mention', 'andy'))),
    );
    expect(result.decision).toBe('deny');
  });
});

describe('triggerGate — reply kind', () => {
  it('allows when replyToMessageId is set', () => {
    const result = triggerGate(
      ctx(
        'thanks',
        { replyToMessageId: 'msg-123' },
        cfg(pattern('reply', 'reply-to-bot')),
      ),
    );
    expect(result.decision).toBe('allow');
  });

  it('denies when replyToMessageId is absent', () => {
    const result = triggerGate(
      ctx('thanks', {}, cfg(pattern('reply', 'reply-to-bot'))),
    );
    expect(result.decision).toBe('deny');
  });
});

describe('triggerGate — sender_tier kind (#82 placeholder)', () => {
  it('returns pass by itself (unevaluatable, defers to next gate)', () => {
    // sender_tier is not implemented in v1. A config containing only
    // forward-looking entries must NOT deny — it must pass so the
    // chain falls through to the next gate / fail-open default.
    const result = triggerGate(
      ctx('hello', {}, cfg(pattern('sender_tier', 'owner'))),
    );
    expect(result.decision).toBe('pass');
  });

  it('does not block other matching patterns in the same set', () => {
    const result = triggerGate(
      ctx(
        '@andy hi',
        {},
        cfg(pattern('sender_tier', 'owner'), pattern('keyword', '@andy')),
      ),
    );
    expect(result.decision).toBe('allow');
  });

  it('mixed sender_tier + keyword (no match) → deny (evaluatable companion did not match)', () => {
    // The evaluatable `keyword` pattern was checked and missed, so
    // the deny accumulator fires. The `sender_tier` entry is ignored
    // for accumulator purposes.
    const result = triggerGate(
      ctx(
        'hello world',
        {},
        cfg(pattern('sender_tier', 'owner'), pattern('keyword', '@andy')),
      ),
    );
    expect(result.decision).toBe('deny');
  });
});

describe('triggerGate — regex kind (deferred to #82)', () => {
  it('returns pass by itself (unevaluatable, no sandboxed runner)', () => {
    const result = triggerGate(
      ctx('hello world', {}, cfg(pattern('regex', '.*'))),
    );
    expect(result.decision).toBe('pass');
  });

  it('multiple unevaluatable kinds → pass', () => {
    const result = triggerGate(
      ctx(
        'hello world',
        {},
        cfg(pattern('regex', '.*'), pattern('sender_tier', 'owner')),
      ),
    );
    expect(result.decision).toBe('pass');
  });
});

describe('triggerGate — unknown future kind', () => {
  it('treats unknown kind as unevaluatable → pass on its own', () => {
    // Forward-compat: a config emitted by a future writer with a kind
    // this gate version doesn't recognise must not black-hole traffic.
    // Cast through unknown to construct an out-of-range kind without
    // poisoning the public TriggerPatternKind union.
    const future = {
      ...pattern('keyword', 'whatever'),
      kind: 'tier_score' as unknown as TriggerPatternKind,
    };
    const result = triggerGate(ctx('hello', {}, cfg(future)));
    expect(result.decision).toBe('pass');
  });
});

describe('triggerGate — legacy compat (requires_trigger=true → ["trigger"])', () => {
  it('matches the legacy single-keyword config produced by parseTriggerPatternColumn', () => {
    // This is the exact shape db.ts builds for a legacy
    // string-trigger row: a single keyword/owner-set entry.
    const legacyConfig: TriggerPatternConfig = {
      version: 1,
      patterns: [pattern('keyword', '@andy', 'owner-set')],
    };
    expect(triggerGate(ctx('@andy hi', {}, legacyConfig)).decision).toBe(
      'allow',
    );
    expect(triggerGate(ctx('hi everyone', {}, legacyConfig)).decision).toBe(
      'deny',
    );
  });
});

describe('triggerGate — first-match wins', () => {
  it('walks patterns in order and returns first allow', () => {
    const result = triggerGate(
      ctx(
        '@andy hello',
        {},
        cfg(
          pattern('keyword', '@bob'),
          pattern('keyword', '@andy'),
          pattern('keyword', '@charlie'),
        ),
      ),
    );
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('@andy');
  });
});
