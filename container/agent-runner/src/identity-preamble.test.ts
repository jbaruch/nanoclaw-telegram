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
    const out = buildIdentityPreamble('LoMBot', 'limlombot');
    // Display-name form (vocative): both as the bolded "You are" subject
    // and as the example vocative.
    expect(out).toContain('You are **LoMBot**');
    expect(out).toContain('"LoMBot, please..."');
    // @-handle form (mention): always with the @-prefix so the agent
    // can't conflate it with the display name.
    expect(out).toContain('**@limlombot**');
    expect(out).toContain('"@limlombot ..."');
  });

  it('explicitly marks tile-example handles as FICTIONAL EXAMPLE so they cannot be templated as identity', () => {
    const out = buildIdentityPreamble('LoMBot', 'limlombot');
    expect(out).toContain('FICTIONAL EXAMPLE');
    // The example handles from nanoclaw-core 0.1.94 must appear by name
    // so the model can resolve them when matching tile-rule context.
    expect(out).toContain('@AyeAye');
    expect(out).toContain('@AyeAyeSureBot');
  });

  it('emits the authoritative-by-orchestrator framing so the preamble outranks tile rules', () => {
    const out = buildIdentityPreamble('LoMBot', 'limlombot');
    expect(out).toContain('authoritative');
    expect(out).toContain('orchestrator');
  });

  it('handles different name/username pairs (e.g. when display != handle)', () => {
    const out = buildIdentityPreamble('Andy', 'limandy');
    expect(out).toContain('You are **Andy**');
    expect(out).toContain('**@limandy**');
    // The previous-test name MUST NOT leak.
    expect(out).not.toContain('LoMBot');
    expect(out).not.toContain('limlombot');
  });

  // Open-note from #407: the preamble enumerates @AyeAye / @AyeAyeSureBot
  // as concrete fictional examples, then generalizes with "or any other
  // bot name." That generalization is the load-bearing bit — tile rules
  // can introduce any number of new fictional handles in the future, and
  // the preamble has to teach the agent to treat them all as
  // substitute-mentally examples regardless of whether they were named
  // here. Probe the abstract case with a never-seen handle to anchor it.
  it('signals that any other bot handle (not just the named examples) is a fictional example', () => {
    const out = buildIdentityPreamble('LoMBot', 'limlombot');
    expect(out).toMatch(/or any other\s+bot name/i);
  });

  // The agent container is reused across Telegram / WhatsApp / Slack /
  // Discord / Gmail. The preamble wording must NOT bind to any one
  // channel's vocabulary — anchoring "Telegram" in the prompt would
  // be misleading on a Slack-hosted deployment. Channel-neutral framing
  // ("display name" / "@-handle") replaces the original
  // Telegram-specific wording.
  it('uses channel-neutral wording (no "Telegram" binding)', () => {
    const out = buildIdentityPreamble('LoMBot', 'limlombot');
    expect(out).not.toContain('Telegram');
    expect(out).toContain('display name');
    expect(out).toContain('@-handle');
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
    const out = resolveIdentityPreamble('LoMBot', 'limlombot');
    expect(out).toBeDefined();
    expect(out).toContain('You are **LoMBot**');
    expect(out).toContain('**@limlombot**');
  });

  it('returns undefined when ASSISTANT_NAME is missing', () => {
    expect(resolveIdentityPreamble(undefined, 'limlombot')).toBeUndefined();
  });

  it('returns undefined when ASSISTANT_USERNAME is missing', () => {
    expect(resolveIdentityPreamble('LoMBot', undefined)).toBeUndefined();
  });

  it('returns undefined when both are missing', () => {
    expect(resolveIdentityPreamble(undefined, undefined)).toBeUndefined();
  });

  it('returns undefined when ASSISTANT_NAME is the empty string (process.env quirk)', () => {
    expect(resolveIdentityPreamble('', 'limlombot')).toBeUndefined();
  });

  it('returns undefined when ASSISTANT_USERNAME is the empty string', () => {
    expect(resolveIdentityPreamble('LoMBot', '')).toBeUndefined();
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
    expect(resolveIdentityPreamble('   ', 'limlombot')).toBeUndefined();
  });

  it('treats whitespace-only ASSISTANT_USERNAME as missing after trim', () => {
    expect(resolveIdentityPreamble('LoMBot', '\t\n  ')).toBeUndefined();
  });

  it('strips raw newlines from name/username so preamble structure stays intact', () => {
    const out = resolveIdentityPreamble(
      'LoMBot\n# rogue heading',
      'limlombot\nfoo',
    );
    expect(out).toBeDefined();
    expect(out).not.toMatch(/\n# rogue heading/);
    // Newlines collapse to spaces, so the name renders as a single line
    // even when the operator's .env value spanned multiple.
    expect(out).toContain('You are **LoMBot # rogue heading**');
    expect(out).toContain('**@limlombot foo**');
  });

  it('trims surrounding whitespace from both fields', () => {
    const out = resolveIdentityPreamble('  LoMBot  ', '  limlombot  ');
    expect(out).toContain('You are **LoMBot**');
    expect(out).toContain('**@limlombot**');
  });

  it('length-caps each field at 256 characters', () => {
    const longName = 'A'.repeat(500);
    const out = resolveIdentityPreamble(longName, 'limlombot');
    expect(out).toBeDefined();
    // Bolded form would be `**` + name + `**` = 4 + 256 = 260 chars at
    // most. Probe the boundary directly.
    expect(out).toContain(`**${'A'.repeat(256)}**`);
    expect(out).not.toContain(`**${'A'.repeat(257)}**`);
  });
});
