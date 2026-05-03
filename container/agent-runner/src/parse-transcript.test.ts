import { describe, it, expect } from 'vitest';
import { parseTranscript } from './parse-transcript.js';

// Build a JSONL transcript line by line. Test inputs are constructed
// programmatically (per `coding-policy: testing-standards` — "Build
// test data programmatically in test setup/fixtures"); no fixture
// files on disk.
function jsonl(...entries: unknown[]): string {
  return entries.map((e) => JSON.stringify(e)).join('\n');
}

// Convenience builders for the SDK transcript shape.
const userTurn = (text: string, extras: Record<string, unknown> = {}) => ({
  type: 'user',
  message: { role: 'user', content: [{ type: 'text', text }] },
  ...extras,
});
const assistantTurn = (text: string, extras: Record<string, unknown> = {}) => ({
  type: 'assistant',
  message: { role: 'assistant', content: [{ type: 'text', text }] },
  ...extras,
});

describe('parseTranscript', () => {
  it('returns user/assistant turns in transcript order', () => {
    const transcript = jsonl(
      userTurn('hi'),
      assistantTurn('hello'),
      userTurn('how are you?'),
      assistantTurn('I am well.'),
    );
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
      { role: 'user', content: 'how are you?' },
      { role: 'assistant', content: 'I am well.' },
    ]);
  });

  // The load-bearing pin for #416. Without this filter, every turn
  // would gain a `**User**: <env-info ...>` boilerplate entry from the
  // synthetic dynamic-section message Claude Code injects when
  // `excludeDynamicSections: true` is set.
  it('skips synthetic dynamic-section user turns (isMeta:true)', () => {
    const dynamicSectionsBoilerplate =
      'Here is useful information about the environment you are running in:\n<env>\nWorking directory: /workspace/group\n</env>';
    const transcript = jsonl(
      userTurn(dynamicSectionsBoilerplate, { isMeta: true }),
      userTurn('actual user message'),
      assistantTurn('actual reply'),
    );
    const parsed = parseTranscript(transcript);
    // The env-info synthetic turn must NOT appear in the archive.
    expect(parsed).toEqual([
      { role: 'user', content: 'actual user message' },
      { role: 'assistant', content: 'actual reply' },
    ]);
    // Belt and braces: the env boilerplate must not have leaked into
    // any rendered content.
    for (const m of parsed) {
      expect(m.content).not.toContain('Working directory:');
      expect(m.content).not.toContain('<env>');
    }
  });

  it('skips local-command-caveat boilerplate (isMeta:true)', () => {
    // The SDK also tags slash-command preludes / local-command shells
    // with isMeta:true. They should not appear in the archive either.
    const transcript = jsonl(
      userTurn('<local-command-caveat>...</local-command-caveat>', {
        isMeta: true,
      }),
      userTurn('real prompt'),
    );
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'real prompt' },
    ]);
  });

  it('skips compact-summary synthetic turns (isCompactSummary:true)', () => {
    const transcript = jsonl(
      userTurn('summary of prior messages...', { isCompactSummary: true }),
      userTurn('hi'),
      assistantTurn('hello'),
    );
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ]);
  });

  it('skips subagent-transcript lines (isSidechain:true)', () => {
    // Subagent (Task-tool) traffic shows up in the parent JSONL with
    // isSidechain:true. The archive is the parent conversation; the
    // subagent's internal exchanges don't belong there.
    const transcript = jsonl(
      userTurn('subagent prompt', { isSidechain: true }),
      assistantTurn('subagent reply', { isSidechain: true }),
      userTurn('parent prompt'),
      assistantTurn('parent reply'),
    );
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'parent prompt' },
      { role: 'assistant', content: 'parent reply' },
    ]);
  });

  it('skips Task-tool / TeamCreate subagent lines (teamName marker)', () => {
    // The SDK's BS filter drops any line with a truthy `teamName`
    // (subagent transcripts carry `teamName: "<agent-name>"`). Match
    // the SDK so the parent archive doesn't pick up TeamCreate
    // chatter from a subagent thread.
    const transcript = jsonl(
      userTurn('subagent prompt', { teamName: 'Explore' }),
      assistantTurn('subagent reply', { teamName: 'Explore' }),
      userTurn('parent prompt'),
      assistantTurn('parent reply'),
    );
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'parent prompt' },
      { role: 'assistant', content: 'parent reply' },
    ]);
  });

  it('keeps lines with empty-string teamName (defensive: only truthy markers suppress)', () => {
    // Defensive: a transcript line with `teamName: ""` is not actually
    // a subagent line — the SDK marker is truthy (the agent name
    // itself). Don't accidentally suppress real conversation turns
    // just because some upstream emitter writes an empty string.
    // Empty string is JS-falsy so `if (e.teamName)` is false, matching
    // the SDK's BS predicate exactly.
    const transcript = jsonl(
      userTurn('real turn', { teamName: '' }),
      assistantTurn('real reply', { teamName: '' }),
    );
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'real turn' },
      { role: 'assistant', content: 'real reply' },
    ]);
  });

  it('drops lines with non-string truthy teamName (SDK BS-predicate parity)', () => {
    // The SDK's BS predicate uses `if($.teamName)` — truthy on ANY
    // truthy value, not just non-empty strings. A non-string truthy
    // teamName (number, object, etc.) is most likely a buggy upstream
    // emitter, but we'd rather drop a malformed subagent line than
    // leak it into the parent archive. Match BS's truthy-only check.
    const transcript = [
      JSON.stringify({
        ...userTurn('subagent prompt'),
        teamName: 1,
      }),
      JSON.stringify({
        ...assistantTurn('subagent reply'),
        teamName: { name: 'Explore' },
      }),
      JSON.stringify(userTurn('parent prompt')),
    ].join('\n');
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'parent prompt' },
    ]);
  });

  it('handles string-form user content (not just text-block arrays)', () => {
    // The SDK occasionally writes user content as a bare string rather
    // than a single-text-block array. Both shapes must extract.
    const transcript = jsonl({
      type: 'user',
      message: { role: 'user', content: 'plain string content' },
    });
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'plain string content' },
    ]);
  });

  it('strips assistant tool_use / thinking blocks from rendered text', () => {
    // The archive only renders text. Tool-use payloads, thinking
    // blocks, and other non-text content must not appear.
    const transcript = jsonl({
      type: 'assistant',
      message: {
        role: 'assistant',
        content: [
          { type: 'thinking', text: 'should not appear' },
          { type: 'text', text: 'visible answer' },
          {
            type: 'tool_use',
            id: 'tu_1',
            name: 'Bash',
            input: { command: 'ls' },
          },
        ],
      },
    });
    expect(parseTranscript(transcript)).toEqual([
      { role: 'assistant', content: 'visible answer' },
    ]);
  });

  it('tolerates malformed JSONL lines (skips and continues)', () => {
    const transcript = [
      JSON.stringify(userTurn('first')),
      '{ this is not json',
      '',
      '   ',
      JSON.stringify(assistantTurn('second')),
    ].join('\n');
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'second' },
    ]);
  });

  it('rethrows non-SyntaxError exceptions from JSON.parse (no bare catch-all)', () => {
    // `JSON.parse` only throws SyntaxError on bad input. Non-SyntaxError
    // exceptions during parse are real bugs (a JSON.parse override gone
    // wrong, an OOM, a TypeError from an instrumentation hook) and must
    // propagate per `coding-policy: error-handling` ("Catch specific
    // exception types, never bare catch-all handlers"). Patch JSON.parse
    // to throw a non-SyntaxError on a specific marker line and assert
    // the parser rethrows.
    const realParse = JSON.parse;
    const TRIGGER = '__non_syntax_error_marker__';
    const sabotaged = (text: string, ...rest: unknown[]) => {
      if (text.includes(TRIGGER)) {
        throw new TypeError('synthetic non-SyntaxError from JSON.parse');
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return realParse(text, ...(rest as [any?]));
    };
    JSON.parse = sabotaged as typeof JSON.parse;
    try {
      const transcript = [
        JSON.stringify(userTurn('first')),
        `{"trigger":"${TRIGGER}"}`,
      ].join('\n');
      expect(() => parseTranscript(transcript)).toThrow(TypeError);
      expect(() => parseTranscript(transcript)).toThrow(
        'synthetic non-SyntaxError from JSON.parse',
      );
    } finally {
      JSON.parse = realParse;
    }
  });

  it('returns an empty array on an empty transcript', () => {
    expect(parseTranscript('')).toEqual([]);
    expect(parseTranscript('\n\n   \n')).toEqual([]);
  });

  it('drops empty-text user turns (no content to render)', () => {
    const transcript = jsonl(
      userTurn(''),
      userTurn('non-empty'),
      assistantTurn(''),
      assistantTurn('reply'),
    );
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'non-empty' },
      { role: 'assistant', content: 'reply' },
    ]);
  });

  // BS-parity: the SDK uses `if($.isMeta)return!1;` — a truthy check,
  // not strict equality. Strict-equality `=== true` would let a
  // malformed upstream emission (e.g. `isMeta: 1`, `isMeta: "yes"`)
  // slip past us even though `getSessionMessages` would hide it. Cost
  // of the false-positive (one missed real turn from a buggy emitter)
  // is lower than the cost of the false-negative (SDK-internal
  // traffic in the archive). The sibling tests pin the same rule for
  // `teamName`; this one pins it for `isMeta` directly.
  it('drops lines with non-boolean truthy isMeta (SDK BS-predicate parity)', () => {
    const transcript = [
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      JSON.stringify(userTurn('truthy-but-not-true', { isMeta: 1 as any })),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      JSON.stringify(userTurn('also-truthy', { isMeta: 'yes' as any })),
      JSON.stringify(userTurn('real prompt')),
    ].join('\n');
    expect(parseTranscript(transcript)).toEqual([
      { role: 'user', content: 'real prompt' },
    ]);
  });
});
