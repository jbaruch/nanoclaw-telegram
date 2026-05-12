import { describe, it, expect } from 'vitest';

import { triggerGate } from './trigger.js';
import { ASSISTANT_NAME, ASSISTANT_USERNAME } from '../config.js';
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
      messageId: 'msg-test-1',
      senderJid: 's@s.whatsapp.net',
      ...overrides,
    },
    triggerPatterns,
  };
}

describe('triggerGate — empty patterns', () => {
  it('returns pass when triggerPatterns is null and no synthetic match', () => {
    const result = triggerGate(ctx('hello world', {}, null));
    expect(result.decision).toBe('pass');
  });

  it('returns pass when patterns array is empty and no synthetic match', () => {
    const result = triggerGate(
      ctx('hello world', {}, { version: 1, patterns: [] }),
    );
    expect(result.decision).toBe('pass');
  });
});

describe('triggerGate — synthetic identity patterns (auto-injected)', () => {
  // The gate auto-evaluates two synthetic patterns derived from
  // ASSISTANT_NAME / ASSISTANT_USERNAME on every call, before any
  // operator-configured patterns. These are short-circuits for
  // direct identity references at Stage 1 — zero API cost vs. Stage
  // 2 Haiku.
  it('matches @<ASSISTANT_USERNAME> when no operator config', () => {
    const result = triggerGate(
      ctx(`@${ASSISTANT_USERNAME} hello`, {}, { version: 1, patterns: [] }),
    );
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
  });

  it('matches ASSISTANT_NAME vocative when no operator config', () => {
    const result = triggerGate(
      ctx(`${ASSISTANT_NAME}, please help`, {}, { version: 1, patterns: [] }),
    );
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
  });

  it('matches @<ASSISTANT_USERNAME> when triggerPatterns is null', () => {
    const result = triggerGate(ctx(`@${ASSISTANT_USERNAME} ping`, {}, null));
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
  });

  it('passes when neither synthetic pattern matches and no operator config', () => {
    const result = triggerGate(
      ctx('just human chatter', {}, { version: 1, patterns: [] }),
    );
    expect(result.decision).toBe('pass');
  });

  it('synthetic identity match wins over a non-matching operator pattern', () => {
    // Operator pattern `bots` (keyword) does not match "Andy, what
    // about it?" but the synthetic ASSISTANT_NAME keyword does.
    const result = triggerGate(
      ctx(
        `${ASSISTANT_NAME}, what about it?`,
        {},
        cfg(pattern('keyword', 'bots')),
      ),
    );
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
  });

  it('operator-configured pattern still matches when synthetic does not', () => {
    // No identity reference in the message; operator `bots` keyword
    // is the only signal that fires.
    const result = triggerGate(
      ctx('this discussion is about bots', {}, cfg(pattern('keyword', 'bots'))),
    );
    expect(result.decision).toBe('allow');
    // The match comes from the operator pattern, not the synthetic
    // one — reason should NOT carry the `(auto)` tag.
    expect(result.reason).not.toContain('(auto)');
    expect(result.reason).toContain('bots');
  });

  it('synthetic mention is case-insensitive', () => {
    const upper = ASSISTANT_USERNAME.toUpperCase();
    const result = triggerGate(
      ctx(`hey @${upper} ping`, {}, { version: 1, patterns: [] }),
    );
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
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

  // #566: keyword matcher used `\b` (ASCII-only word boundary), so
  // Cyrillic / Hebrew / Arabic / CJK keywords never matched and every
  // miss paid the Stage 2 Haiku tax. Fixed by switching to a
  // Unicode-aware negative lookahead with the `u` flag.
  it('matches Cyrillic keyword at start of message (#566)', () => {
    const result = triggerGate(
      ctx(
        'ботики накидайте @saabeilin как экономтиь на токенах',
        {},
        cfg(pattern('keyword', 'ботики')),
      ),
    );
    expect(result.decision).toBe('allow');
  });

  it('matches Cyrillic keyword after whitespace (#566)', () => {
    const result = triggerGate(
      ctx('эй ботики помогите', {}, cfg(pattern('keyword', 'ботики'))),
    );
    expect(result.decision).toBe('allow');
  });

  it('does not match Cyrillic keyword inside a longer Cyrillic word (#566)', () => {
    // Unicode-aware boundary must reject `ботики` inside `ботикичто`
    // (substring continuation) just as `\b` rejected ASCII `@andybot`
    // for the `@andy` pattern.
    const result = triggerGate(
      ctx('ботикичто-то ещё', {}, cfg(pattern('keyword', 'ботики'))),
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
  it('allows when replyToMessageId is set (legacy v1 wiring)', () => {
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

  // Structured replyTo (#107). The kind:'reply' matcher is now a real
  // signal — it means "reply to THIS assistant" specifically, so
  // replies to peer bots / humans don't fire it.
  it('allows when replyTo.isAssistant=true (structured reply)', () => {
    const result = triggerGate(
      ctx(
        'thanks',
        {
          replyTo: {
            messageId: 'msg-123',
            senderName: 'Andy',
            isBot: true,
            isAssistant: true,
            contentPreview: 'sure I can help',
          },
        },
        cfg(pattern('reply', 'reply-to-bot')),
      ),
    );
    expect(result.decision).toBe('allow');
  });

  it('denies when replyTo.isAssistant=false (reply to peer bot)', () => {
    const result = triggerGate(
      ctx(
        'thanks',
        {
          replyTo: {
            messageId: 'msg-456',
            senderName: 'OtherBot',
            isBot: true,
            isAssistant: false,
            contentPreview: 'hi from peer bot',
          },
        },
        cfg(pattern('reply', 'reply-to-bot')),
      ),
    );
    expect(result.decision).toBe('deny');
  });

  it('denies when replyTo.isAssistant=false (reply to human)', () => {
    const result = triggerGate(
      ctx(
        'got it',
        {
          replyTo: {
            messageId: 'msg-789',
            senderName: 'Bob',
            isBot: false,
            isAssistant: false,
            contentPreview: 'see you tomorrow',
          },
        },
        cfg(pattern('reply', 'reply-to-bot')),
      ),
    );
    expect(result.decision).toBe('deny');
  });
});

describe('triggerGate — quote-prefix false-positive regression (#107)', () => {
  // Original bug: a reply DB row content held the inline
  // `[Replying to ...]` quote prefix containing the assistant's
  // name PLUS the user's actual body "Yes do it". Stage 1's
  // synthetic identity match (auto, kind=keyword) hit the
  // substring inside the quote prefix and short-circuited Stage 2.
  // The fix: GateContext.message.text is now the CLEAN body — the
  // call site (buildGateContext) strips the prefix, exposing reply
  // context structurally via replyTo.
  //
  // This test asserts the gate behaviour after that contract: the
  // gate's input `text` is "Yes do it" (no prefix) and `replyTo`
  // describes the peer-bot reply target. Synthetic identity matchers
  // run on the clean body and find no assistant-name reference — so
  // the gate returns `pass` (or `deny` if operator patterns also
  // miss), letting the chain fall through to Stage 2.
  it('does not Stage-1-allow a "Yes do it" reply to a peer bot whose preview contains the assistant name', () => {
    const result = triggerGate(
      ctx(
        'Yes do it', // clean body — gate input has NO inline prefix
        {
          replyTo: {
            messageId: '4114',
            senderName: 'PeerBot',
            isBot: true,
            isAssistant: false, // peer bot, NOT this assistant
            contentPreview:
              '[Replying to TestBot: "Based on today\'s conversation and the merged PRs, here\'s what changed for me: Stage 2 Haiku..."]',
          },
        },
        // No operator patterns — only the synthetic identity matchers run.
        { version: 1, patterns: [] },
      ),
    );
    // No synthetic match (clean body has no assistant-name / no @-handle).
    // No operator patterns either, so we fall through to "pass" so
    // Stage 2 (Haiku) gets to adjudicate. Critically: NOT `allow`.
    expect(result.decision).toBe('pass');
  });

  it('does not Stage-1-allow when operator keyword would only match the quoted preview', () => {
    // Operator keyword "TestBot" defined; the user body is "Yes do it"
    // and the preview contains TestBot. The matcher sees only the
    // clean body and so returns deny (no operator pattern matched).
    const result = triggerGate(
      ctx(
        'Yes do it',
        {
          replyTo: {
            messageId: '4114',
            senderName: 'PeerBot',
            isBot: true,
            isAssistant: false,
            contentPreview: 'Stage 2 Haiku — see TestBot output above',
          },
        },
        cfg(pattern('keyword', 'TestBot')),
      ),
    );
    expect(result.decision).toBe('deny');
  });

  it('still allows direct @-handle in clean body (existing behaviour preserved)', () => {
    const result = triggerGate(
      ctx(
        `@${ASSISTANT_USERNAME} take a look`,
        {},
        { version: 1, patterns: [] },
      ),
    );
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
  });

  it('still allows vocative ASSISTANT_NAME in clean body (existing behaviour preserved)', () => {
    const result = triggerGate(
      ctx(`${ASSISTANT_NAME}, please help`, {}, { version: 1, patterns: [] }),
    );
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('(auto)');
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
  it('walks operator patterns in order and returns first allow', () => {
    // Use non-identity tokens to keep the synthetic identity gate
    // out of the way; this test pins the operator-pattern ordering
    // semantics specifically.
    const result = triggerGate(
      ctx(
        '@charlie hello',
        {},
        cfg(
          pattern('keyword', '@bob'),
          pattern('keyword', '@charlie'),
          pattern('keyword', '@dave'),
        ),
      ),
    );
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('@charlie');
    // The match comes from the operator pattern, not the auto-injected
    // identity pattern.
    expect(result.reason).not.toContain('(auto)');
  });
});

describe('triggerGate — learner suppression flags (#82)', () => {
  // The self-improvement loop writes proposals as `enabled: false`
  // (pure proposal, owner promotes via admin path) and demotes
  // bad-precision rows by flipping `disabled: true`. The matcher
  // skips both. These tests pin that contract so a future refactor
  // that drops the skip can't silently start matching on un-promoted
  // proposals.
  it('skips a learned pattern with enabled: false (pure proposal)', () => {
    const proposal = pattern('keyword', 'deploy', 'learned');
    proposal.enabled = false;
    const result = triggerGate(ctx('please deploy now', {}, cfg(proposal)));
    // Only-pattern was suppressed → no evaluatable patterns →
    // pass-through (NOT deny — there was nothing to evaluate).
    expect(result.decision).toBe('pass');
  });

  it('skips a learned pattern with disabled: true (auto-rolled-back)', () => {
    const demoted = pattern('keyword', 'deploy', 'learned');
    demoted.disabled = true;
    demoted.enabled = true; // even if owner had promoted it, disabled wins
    const result = triggerGate(ctx('please deploy now', {}, cfg(demoted)));
    expect(result.decision).toBe('pass');
  });

  it('matches a learned pattern with enabled: true and disabled: false', () => {
    const promoted = pattern('keyword', 'deploy', 'learned');
    promoted.enabled = true;
    promoted.disabled = false;
    const result = triggerGate(ctx('please deploy now', {}, cfg(promoted)));
    expect(result.decision).toBe('allow');
  });

  it('owner-set patterns that leave the new flags unset still match', () => {
    // Backward compat: legacy rows have neither `enabled` nor
    // `disabled` set. They must match exactly as before.
    const ownerSet = pattern('keyword', 'deploy'); // default source 'owner-set'
    expect(ownerSet.enabled).toBeUndefined();
    expect(ownerSet.disabled).toBeUndefined();
    const result = triggerGate(ctx('please deploy now', {}, cfg(ownerSet)));
    expect(result.decision).toBe('allow');
  });
});
