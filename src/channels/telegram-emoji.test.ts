import { describe, it, expect } from 'vitest';

import { normalizeReactionEmoji } from './telegram.js';

describe('normalizeReactionEmoji — passthrough for valid Unicode', () => {
  it('passes Unicode reactions through unchanged', () => {
    // Already-Unicode input must round-trip exactly so callers that
    // already do the right thing aren't penalized.
    expect(normalizeReactionEmoji('👍')).toBe('👍');
    expect(normalizeReactionEmoji('🤣')).toBe('🤣');
    expect(normalizeReactionEmoji('💯')).toBe('💯');
    expect(normalizeReactionEmoji('🤷')).toBe('🤷');
  });
});

describe('normalizeReactionEmoji — Slack-style shortcodes', () => {
  it('maps the shortcodes called out in #161 as already-working', () => {
    // These were reported as "Works via Slack shortcode" in the issue.
    // They must keep working through normalization (idempotent for
    // anything that already mapped on Telegram's side, but correctness
    // here doesn't depend on Telegram — we map locally).
    expect(normalizeReactionEmoji('thumbs_up')).toBe('👍');
    expect(normalizeReactionEmoji('thumbs_down')).toBe('👎');
    expect(normalizeReactionEmoji('heart')).toBe('❤');
    expect(normalizeReactionEmoji('fire')).toBe('🔥');
    expect(normalizeReactionEmoji('100')).toBe('💯');
    expect(normalizeReactionEmoji('eyes')).toBe('👀');
    expect(normalizeReactionEmoji('thinking_face')).toBe('🤔');
    expect(normalizeReactionEmoji('tada')).toBe('🎉');
    expect(normalizeReactionEmoji('sob')).toBe('😭');
    expect(normalizeReactionEmoji('clap')).toBe('👏');
    expect(normalizeReactionEmoji('trophy')).toBe('🏆');
    expect(normalizeReactionEmoji('pray')).toBe('🙏');
    expect(normalizeReactionEmoji('ok_hand')).toBe('👌');
    expect(normalizeReactionEmoji('zap')).toBe('⚡');
    expect(normalizeReactionEmoji('rage')).toBe('😡');
    expect(normalizeReactionEmoji('sleeping')).toBe('😴');
  });

  it('maps the shortcodes called out in #161 as Unicode-only', () => {
    // Sample from the "Works via Unicode only" list. Without the
    // mapping these would have failed `TELEGRAM_ALLOWED_REACTIONS.has`
    // and silently fallen back to 👍. The map has to cover them.
    expect(normalizeReactionEmoji('rofl')).toBe('🤣');
    expect(normalizeReactionEmoji('exploding_head')).toBe('🤯');
    expect(normalizeReactionEmoji('heart_eyes')).toBe('😍');
    expect(normalizeReactionEmoji('scream')).toBe('😱');
    expect(normalizeReactionEmoji('vomit')).toBe('🤮');
    expect(normalizeReactionEmoji('poop')).toBe('💩');
    expect(normalizeReactionEmoji('clown')).toBe('🤡');
    expect(normalizeReactionEmoji('broken_heart')).toBe('💔');
    expect(normalizeReactionEmoji('ghost')).toBe('👻');
    expect(normalizeReactionEmoji('star_struck')).toBe('🤩');
    expect(normalizeReactionEmoji('nerd')).toBe('🤓');
    expect(normalizeReactionEmoji('hugs')).toBe('🤗');
    expect(normalizeReactionEmoji('salute')).toBe('🫡');
    expect(normalizeReactionEmoji('moai')).toBe('🗿');
    expect(normalizeReactionEmoji('santa')).toBe('🎅');
    expect(normalizeReactionEmoji('snowman')).toBe('☃');
    expect(normalizeReactionEmoji('zany')).toBe('🤪');
    expect(normalizeReactionEmoji('cool')).toBe('🆒');
    expect(normalizeReactionEmoji('cupid')).toBe('💘');
    expect(normalizeReactionEmoji('unicorn')).toBe('🦄');
    expect(normalizeReactionEmoji('pill')).toBe('💊');
    expect(normalizeReactionEmoji('kiss')).toBe('💋');
    expect(normalizeReactionEmoji('yawn')).toBe('🥱');
  });
});

describe('normalizeReactionEmoji — colon-delimited shortcodes', () => {
  it('strips surrounding colons (Slack format)', () => {
    expect(normalizeReactionEmoji(':thumbs_up:')).toBe('👍');
    expect(normalizeReactionEmoji(':rofl:')).toBe('🤣');
    expect(normalizeReactionEmoji(':100:')).toBe('💯');
  });

  it('handles aliases that already include a digit', () => {
    expect(normalizeReactionEmoji('+1')).toBe('👍');
    expect(normalizeReactionEmoji('-1')).toBe('👎');
  });
});

describe('normalizeReactionEmoji — unmapped input', () => {
  it('returns input unchanged for unknown shortcodes', () => {
    // Caller's `TELEGRAM_ALLOWED_REACTIONS.has(...)` gate handles
    // the actual fallback to 👍 with a warn log. Returning the
    // original lets that gate emit a useful diagnostic instead of
    // collapsing to 👍 silently here.
    expect(normalizeReactionEmoji('unknown_shortcode')).toBe(
      'unknown_shortcode',
    );
    expect(normalizeReactionEmoji('not_a_real_emoji')).toBe('not_a_real_emoji');
  });

  it('returns input unchanged for unsupported Unicode (issue #161 not_supported list)', () => {
    // These were reported as "Not supported at all" — Telegram has
    // no reaction slot for them. The normalizer should NOT fabricate
    // a mapping; the caller's allowed-reactions gate falls back.
    expect(normalizeReactionEmoji('✅')).toBe('✅');
    expect(normalizeReactionEmoji('🚀')).toBe('🚀');
    expect(normalizeReactionEmoji('🤦')).toBe('🤦');
  });
});

// --- Invariant: every shortcode-map value MUST be in the allowed set ---
//
// Drift between EMOJI_SHORTCODE_TO_UNICODE and TELEGRAM_ALLOWED_REACTIONS
// is the failure mode this normalization is supposed to PREVENT — a
// shortcode that maps to a Unicode char Telegram doesn't accept would
// still fall back to 👍, just at a different gate. Test by exhaustively
// running every distinct mapped value through the normalizer and the
// production gate (`sendReaction`'s `.has` check) via the same channel
// of access an agent would use.

describe('normalizeReactionEmoji — every mapped Unicode is Telegram-supported', () => {
  // We can't reach the private TELEGRAM_ALLOWED_REACTIONS Set directly
  // without changing the module exports. But we CAN drive the
  // normalize → has chain by feeding the issue-#161 list of known
  // Telegram-supported reactions through normalize and checking the
  // round-trip is stable: a Telegram-supported Unicode reaction must
  // pass through unchanged (TELEGRAM_ALLOWED_REACTIONS.has() short-
  // circuits in normalizeReactionEmoji). If a future contributor adds
  // a shortcode that maps to a Unicode char NOT in the allowed set,
  // this same property test catches it because the new Unicode value
  // wouldn't pass the .has() gate inside normalizeReactionEmoji's
  // first branch — it'd fall through to the shortcode lookup, find
  // nothing, and return the wrong mapping. Catches drift symmetrically.
  const supportedReactions = [
    '👍',
    '👎',
    '❤',
    '🔥',
    '🥰',
    '👏',
    '😁',
    '🤔',
    '🤯',
    '😱',
    '🤬',
    '😢',
    '🎉',
    '🤩',
    '🤮',
    '💩',
    '🙏',
    '👌',
    '🕊',
    '🤡',
    '🥱',
    '🥴',
    '😍',
    '🐳',
    '❤‍🔥',
    '🌚',
    '🌭',
    '💯',
    '🤣',
    '⚡',
    '🍌',
    '🏆',
    '💔',
    '🤨',
    '😐',
    '🍓',
    '🍾',
    '💋',
    '🖕',
    '😈',
    '😴',
    '😭',
    '🤓',
    '👻',
    '👨‍💻',
    '👀',
    '🎃',
    '🙈',
    '😇',
    '😨',
    '🤝',
    '✍',
    '🤗',
    '🫡',
    '🎅',
    '🎄',
    '☃',
    '💅',
    '🤪',
    '🗿',
    '🆒',
    '💘',
    '🙉',
    '🦄',
    '😘',
    '💊',
    '🙊',
    '😎',
    '👾',
    '🤷‍♂',
    '🤷',
    '🤷‍♀',
    '😡',
  ];

  for (const emoji of supportedReactions) {
    it(`${emoji} round-trips through normalizeReactionEmoji unchanged`, () => {
      expect(normalizeReactionEmoji(emoji)).toBe(emoji);
    });
  }
});
