import { describe, it, expect } from 'vitest';
import {
  buildFrozenSystemPromptAppend,
  pickSystemPrompt,
  type FrozenSystemPromptInputs,
} from './system-prompt-assembly.js';

// buildFrozenSystemPromptAppend — pure assembly of the frozen
// (cacheable) portion of the agent-SDK system-prompt append. The
// invariant the cache relies on is byte-identical output across
// successive invocations within the same group/session window — if
// the frozen prefix shifts by a single byte, the Anthropic cache
// (5-min TTL, API-key-scoped) misses and we pay full input-token
// cost on every message. These tests pin that invariant.
describe('buildFrozenSystemPromptAppend', () => {
  it('returns undefined when every input is missing', () => {
    expect(
      buildFrozenSystemPromptAppend({
        identityPreamble: undefined,
        soulMd: undefined,
        formattingMd: undefined,
      }),
    ).toBeUndefined();
  });

  it('emits identity preamble first, then SOUL, then FORMATTING', () => {
    const out = buildFrozenSystemPromptAppend({
      identityPreamble: 'IDENTITY_BLOCK',
      soulMd: 'SOUL_BLOCK',
      formattingMd: 'FORMATTING_BLOCK',
    });
    expect(out).toBe(
      'IDENTITY_BLOCK\n\n---\n\nSOUL_BLOCK\n\n---\n\nFORMATTING_BLOCK',
    );
  });

  it('separates parts with \\n\\n---\\n\\n (preserves pre-refactor format)', () => {
    const out = buildFrozenSystemPromptAppend({
      identityPreamble: 'A',
      soulMd: 'B',
      formattingMd: 'C',
    });
    // Exact separator pattern — any drift here would invalidate
    // every cache prefix written before the change rolled.
    expect(out).toBe('A\n\n---\n\nB\n\n---\n\nC');
  });

  it('skips missing parts without leaving stray separators', () => {
    const out = buildFrozenSystemPromptAppend({
      identityPreamble: 'IDENTITY',
      soulMd: undefined,
      formattingMd: 'FORMATTING',
    });
    expect(out).toBe('IDENTITY\n\n---\n\nFORMATTING');
    expect(out).not.toContain('undefined');
    expect(out).not.toMatch(/---\s*---/);
  });

  it('handles SOUL-only input', () => {
    const out = buildFrozenSystemPromptAppend({
      identityPreamble: undefined,
      soulMd: 'SOUL_ONLY',
      formattingMd: undefined,
    });
    expect(out).toBe('SOUL_ONLY');
  });

  it('handles identity-preamble-only input', () => {
    const out = buildFrozenSystemPromptAppend({
      identityPreamble: 'IDENTITY_ONLY',
      soulMd: undefined,
      formattingMd: undefined,
    });
    expect(out).toBe('IDENTITY_ONLY');
  });

  // Deterministic-classification rule (per the issue's "no judgement
  // at runtime" requirement): the same inputs must produce the same
  // bytes every call. Two successive invocations on the same inputs
  // are the agent's two successive messages within a group/session
  // window — if these diverged by a single byte, the cache prefix
  // would invalidate and the cost lever evaporates.
  it('produces byte-identical output across two successive invocations on the same inputs (cache invariant)', () => {
    const inputs = {
      identityPreamble: '# Your identity\n\nYou are **Test**.',
      soulMd: '# SOUL\nBe helpful.',
      formattingMd: '# FORMATTING\nUse Markdown.',
    };
    const first = buildFrozenSystemPromptAppend(inputs);
    const second = buildFrozenSystemPromptAppend(inputs);
    expect(second).toBe(first);
    // Buffer-level equality so any encoding drift surfaces (UTF-8
    // string equality already implies byte equality, but pinning the
    // length is a cheap second probe against future regressions
    // where an invisible character — e.g. a BOM or a soft hyphen —
    // gets injected by a refactor).
    expect(Buffer.byteLength(second!, 'utf-8')).toBe(
      Buffer.byteLength(first!, 'utf-8'),
    );
  });

  // Volatile content (per the issue: group folder CLAUDE.md /
  // MEMORY.md / current-message mounts, cwd, git status, auto-memory
  // paths, the current message itself, the current timestamp) must
  // NEVER appear in the frozen append because it can shift per
  // message. We enforce that with two independent layers, both of
  // which CI runs:
  //
  //   1. **Runtime read-set pin** (this test). The contract is that
  //      `buildFrozenSystemPromptAppend` reads ONLY the three frozen
  //      keys from its input — no volatile field has any runtime
  //      effect on the output, even when sneaked in via a type cast.
  //      The check builds an output from the frozen subset, then
  //      builds another output passing every volatile field the issue
  //      classifies (currentMessage / gitStatus / cwd / memoryMd /
  //      timestamp / extra random keys), and asserts byte-equality.
  //      If a future refactor wires any volatile field into the
  //      output, the byte-equality fails and this test breaks. That
  //      is the load-bearing pin — it does not depend on the type
  //      system, the type-checker config, or which tsconfig CI uses.
  //
  //   2. **Type-level pin** (the `@ts-expect-error` block below).
  //      Documents the intended signature for human readers and any
  //      editor / IDE that runs language-server type-checking on the
  //      file as it's open. It is NOT validated by any CI step in
  //      this repo: root `tsc --noEmit` includes only `src/**/*`,
  //      `container/agent-runner/tsconfig.json` explicitly excludes
  //      `src/**/*.test.ts`, and Vitest does not type-check tests by
  //      default. Even a per-package `tsc --noEmit` invocation won't
  //      validate this file under the existing config — the only way
  //      to make it land in compile is an explicit per-file
  //      `tsc --noEmit <path>` outside the configured project. The
  //      runtime read-set pin above is the actual CI gate; this block
  //      is documentation + IDE assist, nothing more.
  it('frozen builder ignores volatile fields at runtime (read-set pin: only the three frozen keys influence output)', () => {
    const frozen: FrozenSystemPromptInputs = {
      identityPreamble: 'IDENTITY',
      soulMd: 'SOUL',
      formattingMd: 'FORMATTING',
    };
    const baseline = buildFrozenSystemPromptAppend(frozen);

    // Sneak every volatile field the issue rules out alongside the
    // frozen subset, via an `as` cast that bypasses the type. The
    // function MUST still produce the same bytes — if it doesn't,
    // some volatile field is influencing the output and the cache
    // invariant is broken.
    const volatile = {
      identityPreamble: 'IDENTITY',
      soulMd: 'SOUL',
      formattingMd: 'FORMATTING',
      currentMessage: 'shifts every turn',
      gitStatus: 'On branch main',
      cwd: '/workspace',
      memoryMd: 'group memory',
      timestamp: 1700000000000,
      // A randomly-named extra key so we also catch the case where
      // someone adds a hand-rolled key reader that doesn't match
      // the volatile list above but reads ad-hoc fields.
      surpriseField: 'should be ignored',
    } as unknown as FrozenSystemPromptInputs;
    const withVolatile = buildFrozenSystemPromptAppend(volatile);
    expect(withVolatile).toBe(baseline);
    // Buffer-level equality so any encoding drift surfaces.
    expect(Buffer.byteLength(withVolatile!, 'utf-8')).toBe(
      Buffer.byteLength(baseline!, 'utf-8'),
    );
  });

  // Type-level pin — see the H2 comment block above. Each directive sits
  // on the offending PROPERTY, not on the call: `@ts-expect-error`
  // suppresses only the line immediately after it, and the excess-property
  // error is reported at the property, so a directive parked above the
  // call neither suppresses anything nor pins anything (it just reports
  // as unused). CI enforces these now — `npm run typecheck` in this
  // package reads `tsconfig.test.json`, which includes test files
  // (jbaruch/nanoclaw#795). A signature widening that admits a volatile
  // field fails the build here, not just in an editor.
  it('frozen builder type signature rejects volatile inputs (CI-enforced)', () => {
    const frozen: FrozenSystemPromptInputs = {
      identityPreamble: 'IDENTITY',
      soulMd: 'SOUL',
      formattingMd: 'FORMATTING',
    };
    expect(buildFrozenSystemPromptAppend(frozen)).toBeDefined();

    buildFrozenSystemPromptAppend({
      identityPreamble: 'IDENTITY',
      soulMd: 'SOUL',
      formattingMd: 'FORMATTING',
      // @ts-expect-error — `currentMessage` is volatile (per-message turn text).
      currentMessage: 'shifts every turn',
    });

    buildFrozenSystemPromptAppend({
      identityPreamble: 'IDENTITY',
      soulMd: 'SOUL',
      formattingMd: 'FORMATTING',
      // @ts-expect-error — `gitStatus` is volatile (HEAD ref / dirty flag drift).
      gitStatus: 'On branch main',
    });

    buildFrozenSystemPromptAppend({
      identityPreamble: 'IDENTITY',
      soulMd: 'SOUL',
      formattingMd: 'FORMATTING',
      // @ts-expect-error — `cwd` is volatile (changes when the agent cd's).
      cwd: '/workspace',
    });

    buildFrozenSystemPromptAppend({
      identityPreamble: 'IDENTITY',
      soulMd: 'SOUL',
      formattingMd: 'FORMATTING',
      // @ts-expect-error — `memoryMd` is volatile (group/CLAUDE.md / MEMORY.md edited per turn).
      memoryMd: 'group memory',
    });

    buildFrozenSystemPromptAppend({
      identityPreamble: 'IDENTITY',
      soulMd: 'SOUL',
      formattingMd: 'FORMATTING',
      // @ts-expect-error — `timestamp` is volatile (every call has a different now()).
      timestamp: Date.now(),
    });
  });

  it('placement: identity preamble precedes SOUL precedes FORMATTING in the assembled string', () => {
    const out = buildFrozenSystemPromptAppend({
      identityPreamble: 'ID_MARKER',
      soulMd: 'SOUL_MARKER',
      formattingMd: 'FMT_MARKER',
    });
    expect(out).toBeDefined();
    const idIdx = out!.indexOf('ID_MARKER');
    const soulIdx = out!.indexOf('SOUL_MARKER');
    const fmtIdx = out!.indexOf('FMT_MARKER');
    expect(idIdx).toBeGreaterThanOrEqual(0);
    expect(soulIdx).toBeGreaterThan(idIdx);
    expect(fmtIdx).toBeGreaterThan(soulIdx);
  });
});

// pickSystemPrompt — pure decisional logic for the SDK's `systemPrompt`
// shape under the #465 / ligolnik#122 USE_CUSTOM_PROMPT flag. The
// runtime caller (`index.ts`) does the I/O (env read, file read,
// HTML-comment strip, hash log) and hands inputs here; the picker
// stays a pure function so the precedence invariants are unit-testable
// without spinning up the SDK or the agent-runner process.
//
// Invariants pinned:
//   1. Default OFF — preset shape WITH excludeDynamicSections=true
//      preserves the post-#416 frozen-prefix cache shape.
//   2. Custom path with missing tier file — fall back to preset shape
//      (no half-formed prompt).
//   3. Custom path with present tier file — frozen append goes AFTER
//      custom text so authoritative steering wins over tier prompts.
describe('pickSystemPrompt', () => {
  const FROZEN = 'IDENTITY\n\n---\n\nSOUL\n\n---\n\nFORMATTING';

  it('default path (useCustomPrompt=false) returns preset shape with excludeDynamicSections=true', () => {
    const out = pickSystemPrompt({
      useCustomPrompt: false,
      frozenAppend: FROZEN,
      customText: undefined,
    });
    expect(out).toEqual({
      type: 'preset',
      preset: 'claude_code',
      append: FROZEN,
      excludeDynamicSections: true,
    });
  });

  it('default path with no frozen append still returns preset shape (no append field, excludeDynamicSections kept)', () => {
    const out = pickSystemPrompt({
      useCustomPrompt: false,
      frozenAppend: undefined,
      customText: undefined,
    });
    expect(out).toEqual({
      type: 'preset',
      preset: 'claude_code',
      excludeDynamicSections: true,
    });
  });

  it('custom path with missing tier file falls back to preset shape (does not return a half-formed prompt)', () => {
    const out = pickSystemPrompt({
      useCustomPrompt: true,
      frozenAppend: FROZEN,
      customText: undefined,
    });
    expect(out).toEqual({
      type: 'preset',
      preset: 'claude_code',
      append: FROZEN,
      excludeDynamicSections: true,
    });
  });

  it('custom path with tier file present concatenates customText + frozenAppend with a blank-line separator', () => {
    const out = pickSystemPrompt({
      useCustomPrompt: true,
      frozenAppend: FROZEN,
      customText: 'CUSTOM_TIER_TEXT',
    });
    expect(out).toBe('CUSTOM_TIER_TEXT\n\n' + FROZEN);
  });

  it('custom path: frozen append goes AFTER custom text so identity preamble wins on conflicts', () => {
    const out = pickSystemPrompt({
      useCustomPrompt: true,
      frozenAppend: FROZEN,
      customText: 'CUSTOM',
    });
    expect(typeof out).toBe('string');
    const customIdx = (out as string).indexOf('CUSTOM');
    const frozenIdx = (out as string).indexOf(FROZEN);
    expect(customIdx).toBeGreaterThanOrEqual(0);
    expect(frozenIdx).toBeGreaterThan(customIdx);
  });

  it('custom path with no frozen append returns customText alone (no spurious separator)', () => {
    const out = pickSystemPrompt({
      useCustomPrompt: true,
      frozenAppend: undefined,
      customText: 'CUSTOM_ONLY',
    });
    expect(out).toBe('CUSTOM_ONLY');
  });
});
