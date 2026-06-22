import { describe, it, expect } from 'vitest';
import {
  buildIdentityPreamble,
  resolveIdentityPreamble,
} from './identity-preamble.js';

// buildIdentityPreamble — authoritative identity statement injected at
// the top of systemPromptAppend so the agent doesn't template itself
// from fictional bot handles in tile rules (e.g. @AyeAye/@AyeAyeSureBot
// introduced by nanoclaw-core 0.1.94).
describe('buildIdentityPreamble', () => {
  it('substitutes the runtime name and username into the preamble', () => {
    const out = buildIdentityPreamble('TestBot', 'testbot');
    // Display-name form (vocative): both as the bolded "You are" subject
    // and as the example vocative.
    expect(out).toContain('You are **TestBot**');
    expect(out).toContain('"TestBot, please..."');
    // @-handle form (mention): always with the @-prefix so the agent
    // can't conflate it with the display name.
    expect(out).toContain('**@testbot**');
    expect(out).toContain('"@testbot ..."');
  });

  it('explicitly marks tile-example handles as FICTIONAL EXAMPLE so they cannot be templated as identity', () => {
    const out = buildIdentityPreamble('TestBot', 'testbot');
    expect(out).toContain('FICTIONAL EXAMPLE');
    // The example handles from nanoclaw-core 0.1.94 must appear by name
    // so the model can resolve them when matching tile-rule context.
    expect(out).toContain('@AyeAye');
    expect(out).toContain('@AyeAyeSureBot');
  });

  it('emits the authoritative-by-orchestrator framing so the preamble outranks tile rules', () => {
    const out = buildIdentityPreamble('TestBot', 'testbot');
    expect(out).toContain('authoritative');
    expect(out).toContain('orchestrator');
  });

  it('handles different name/username pairs (e.g. when display != handle)', () => {
    const out = buildIdentityPreamble('OtherBot', 'otherbot');
    expect(out).toContain('You are **OtherBot**');
    expect(out).toContain('**@otherbot**');
    // The previous-test name MUST NOT leak.
    expect(out).not.toContain('TestBot');
    expect(out).not.toContain('testbot');
  });

  // Open-note from #407: the preamble enumerates @AyeAye / @AyeAyeSureBot
  // as concrete fictional examples, then generalizes with "or any other
  // bot name." That generalization is the load-bearing bit — tile rules
  // can introduce any number of new fictional handles in the future, and
  // the preamble has to teach the agent to treat them all as
  // substitute-mentally examples regardless of whether they were named
  // here. Probe the abstract case with a never-seen handle to anchor it.
  it('signals that any other bot handle (not just the named examples) is a fictional example', () => {
    const out = buildIdentityPreamble('TestBot', 'testbot');
    expect(out).toMatch(/or any other\s+bot name/i);
  });

  // The agent container is reused across Telegram / WhatsApp / Slack /
  // Discord / Gmail. The preamble wording must NOT bind to any one
  // channel's vocabulary — anchoring "Telegram" in the prompt would
  // be misleading on a Slack-hosted deployment. Channel-neutral framing
  // ("display name" / "@-handle") replaces the original
  // Telegram-specific wording.
  it('uses channel-neutral wording (no "Telegram" binding)', () => {
    const out = buildIdentityPreamble('TestBot', 'testbot');
    expect(out).not.toContain('Telegram');
    expect(out).toContain('display name');
    expect(out).toContain('@-handle');
  });

  // #464 — multi-handle support. When the orchestrator forwards
  // ASSISTANT_USERNAME as a comma-separated list (e.g. an
  // autocomplete-only handle plus an internal/vocative handle), the
  // preamble lists every alias so the agent learns that more than
  // one handle resolves to it.
  describe('multi-handle (#464)', () => {
    it('renders the first handle as canonical and lists aliases', () => {
      const out = buildIdentityPreamble('TestBot', [
        'testbot',
        'testbotsurebot',
      ]);
      expect(out).toContain('Your @-handle is **@testbot**');
      // Aliases appear with @-prefix so the agent can match them
      // against inbound mentions verbatim.
      expect(out).toContain('**@testbotsurebot**');
      // Alias paragraph must teach the equivalence relation, not
      // just list the strings — otherwise the agent could read
      // them as separate identities.
      expect(out).toMatch(/alias|same bot/i);
    });

    it('suppresses the alias paragraph for single-handle bots', () => {
      const out = buildIdentityPreamble('TestBot', ['testbot']);
      // No "also reachable as" / no extra @-handles introduced beyond
      // the canonical one.
      expect(out).not.toMatch(/also reachable as/i);
    });

    it('treats a single-string username the same as a one-element array (back-compat)', () => {
      // Existing call sites pass a plain string; the rendered output
      // must be byte-identical to the array-of-one form so callers
      // can migrate incrementally.
      const fromString = buildIdentityPreamble('TestBot', 'testbot');
      const fromArray = buildIdentityPreamble('TestBot', ['testbot']);
      expect(fromString).toBe(fromArray);
    });
  });
});

// resolveIdentityPreamble — the env-var-aware wrapper used by the agent
// runner. Returns the rendered preamble when both inputs are non-empty,
// or undefined to signal "skip" so the caller can log and omit. Tested
// in isolation because the skip-when-missing decision is the
// safety-critical part — emitting a half-formed preamble (e.g. "You are
// **undefined**") would be worse than no preamble at all.
describe('resolveIdentityPreamble', () => {
  it('returns the rendered preamble when both env vars are set', () => {
    const out = resolveIdentityPreamble('TestBot', 'testbot');
    expect(out).toBeDefined();
    expect(out).toContain('You are **TestBot**');
    expect(out).toContain('**@testbot**');
  });

  it('returns undefined when ASSISTANT_NAME is missing', () => {
    expect(resolveIdentityPreamble(undefined, 'testbot')).toBeUndefined();
  });

  it('returns undefined when ASSISTANT_USERNAME is missing', () => {
    expect(resolveIdentityPreamble('TestBot', undefined)).toBeUndefined();
  });

  it('returns undefined when both are missing', () => {
    expect(resolveIdentityPreamble(undefined, undefined)).toBeUndefined();
  });

  it('returns undefined when ASSISTANT_NAME is the empty string (process.env quirk)', () => {
    expect(resolveIdentityPreamble('', 'testbot')).toBeUndefined();
  });

  it('returns undefined when ASSISTANT_USERNAME is the empty string', () => {
    expect(resolveIdentityPreamble('TestBot', '')).toBeUndefined();
  });

  // Defensive normalization (Copilot review on #408). The orchestrator
  // sets ASSISTANT_NAME / ASSISTANT_USERNAME via `-e` from .env, so the
  // values aren't attacker-controlled in practice — but an operator
  // typo (e.g. a multi-line paste) would otherwise break the markdown
  // structure of the preamble or smuggle stray heading lines into the
  // top-of-context system prompt. Whitespace-only after normalization
  // is treated as missing so a `.env` line like `ASSISTANT_NAME=  `
  // skips rather than rendering "You are **    **".
  it('treats whitespace-only ASSISTANT_NAME as missing after trim', () => {
    expect(resolveIdentityPreamble('   ', 'testbot')).toBeUndefined();
  });

  it('treats whitespace-only ASSISTANT_USERNAME as missing after trim', () => {
    expect(resolveIdentityPreamble('TestBot', '\t\n  ')).toBeUndefined();
  });

  it('strips raw newlines from name/username so preamble structure stays intact', () => {
    const out = resolveIdentityPreamble(
      'TestBot\n# rogue heading',
      'testbot\nfoo',
    );
    expect(out).toBeDefined();
    expect(out).not.toMatch(/\n# rogue heading/);
    // Newlines collapse to spaces, so the name renders as a single line
    // even when the operator's .env value spanned multiple.
    expect(out).toContain('You are **TestBot # rogue heading**');
    expect(out).toContain('**@testbot foo**');
  });

  it('trims surrounding whitespace from both fields', () => {
    const out = resolveIdentityPreamble('  TestBot  ', '  testbot  ');
    expect(out).toContain('You are **TestBot**');
    expect(out).toContain('**@testbot**');
  });

  it('length-caps each field at 256 characters', () => {
    const longName = 'A'.repeat(500);
    const out = resolveIdentityPreamble(longName, 'testbot');
    expect(out).toBeDefined();
    // Bolded form would be `**` + name + `**` = 4 + 256 = 260 chars at
    // most. Probe the boundary directly.
    expect(out).toContain(`**${'A'.repeat(256)}**`);
    expect(out).not.toContain(`**${'A'.repeat(257)}**`);
  });

  // #464 — comma-separated ASSISTANT_USERNAME forwarded by the
  // orchestrator parses into multiple aliases.
  describe('multi-handle (#464)', () => {
    it('parses comma-separated ASSISTANT_USERNAME into aliases', () => {
      const out = resolveIdentityPreamble('TestBot', 'testbot,testbotsurebot');
      expect(out).toBeDefined();
      expect(out).toContain('**@testbot**');
      expect(out).toContain('**@testbotsurebot**');
    });

    it('strips whitespace around comma-separated aliases', () => {
      const out = resolveIdentityPreamble(
        'TestBot',
        '  testbot ,  testbotsurebot  ',
      );
      expect(out).toBeDefined();
      expect(out).toContain('**@testbot**');
      expect(out).toContain('**@testbotsurebot**');
    });

    it('drops empty entries (trailing comma, blank between commas)', () => {
      const out = resolveIdentityPreamble(
        'TestBot',
        'testbot,,testbotsurebot,',
      );
      expect(out).toBeDefined();
      expect(out).toContain('**@testbot**');
      expect(out).toContain('**@testbotsurebot**');
      // No bare "@**" rendered for the dropped empty token.
      expect(out).not.toContain('**@**');
    });

    it('returns undefined when every entry is whitespace-only', () => {
      // ", , ," parses to zero usable entries — same as
      // "missing" semantically; emit nothing rather than half a
      // preamble.
      expect(resolveIdentityPreamble('TestBot', ' , ,')).toBeUndefined();
    });

    it('strips a single leading `@` from each handle (operator-typo tolerance)', () => {
      // Operators reasonably type `@AyeAye,@AyeAyeSureBot` instead of
      // the bare-handle form. Without stripping, the rendered "Your
      // @-handle is" line would become `**@@AyeAye**` and the example
      // mention would be `"@@AyeAye ..."` — both incorrect tokens
      // that defeat the preamble's purpose. Pin the cleaned form.
      const out = resolveIdentityPreamble(
        'TestBot',
        '@testbot,@testbotsurebot',
      );
      expect(out).toBeDefined();
      expect(out).toContain('**@testbot**');
      expect(out).toContain('**@testbotsurebot**');
      expect(out).not.toContain('@@');
    });

    it('de-dupes repeated handles (order-preserving)', () => {
      // Repeats collapse so the canonical primary stays stable; the
      // alias paragraph only mentions distinct extra handles. The
      // canonical handle still renders twice in the preamble itself
      // (the "Your @-handle is" intro line + the closing "trust this
      // preamble" line) — that's the single-handle baseline. The
      // dupe in the alias paragraph is what gets dropped: with
      // `testbot,testbot,testbotsurebot` we expect the same total
      // occurrence count as `testbot,testbotsurebot` (no extra
      // `**@testbot**` in the alias list).
      const dupedOut = resolveIdentityPreamble(
        'TestBot',
        'testbot,testbot,testbotsurebot',
      );
      const cleanOut = resolveIdentityPreamble(
        'TestBot',
        'testbot,testbotsurebot',
      );
      expect(dupedOut).toBeDefined();
      expect(dupedOut).toBe(cleanOut);
      // Sanity-check the alias is preserved (de-dupe should drop
      // only the duplicate, not collapse all of them).
      expect(dupedOut).toContain('**@testbotsurebot**');
    });

    it('treats `@`-only or whitespace-after-`@` entries as empty', () => {
      // `@,@  ,testbot` → after `@` strip only `testbot` is usable.
      const out = resolveIdentityPreamble('TestBot', '@,@  ,testbot');
      expect(out).toBeDefined();
      expect(out).toContain('**@testbot**');
      expect(out).not.toContain('**@**');
    });
  });
});
