import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import {
  buildPostCompactReminder,
  CompactProvenanceSidecar,
  extractCompactProvenance,
  MAX_SOURCES,
  parseSidecar,
  persistCompactProvenance,
  readAndClearSidecar,
  sidecarPathFor,
  writeCompactProvenanceSidecar,
} from './compact-provenance.js';
import { __ACL_INTERNALS, extractMarkerPrefixes } from './capability-acl.js';

// ---- helpers ----

const wrap = (prefix: string, value: string, body: string) =>
  `<untrusted-input source="${prefix}:${value}">\n${body}\n</untrusted-input>`;
const sentinel = (prefix: string, value: string, id = 'tu_x') =>
  `PROVENANCE_MARKER: source="${prefix}:${value}" tool_use_id="${id}"`;

/**
 * Build a synthetic JSONL transcript: one entry per array element. The
 * extractor only cares about whether marker patterns appear in the
 * line text — `parseSidecar`-style strict shapes don't apply. So we
 * pass plain JSON-stringified objects with arbitrary content fields.
 */
function jsonlOf(blocks: Array<Record<string, unknown>>): string {
  return blocks.map((b) => JSON.stringify(b)).join('\n');
}

let tmpRoot = '';
beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'compact-provenance-'));
});
afterEach(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

// ---- extractCompactProvenance ----

describe('extractCompactProvenance', () => {
  it('returns empty for empty input', () => {
    expect([...extractCompactProvenance('')]).toEqual([]);
  });

  it('returns empty for non-string input', () => {
    // Defensive null-guard — JSON.parse can hand us anything.
    expect([...extractCompactProvenance(null as unknown as string)]).toEqual(
      [],
    );
  });

  it('returns empty when transcript carries no markers', () => {
    const jsonl = jsonlOf([
      { type: 'user', message: { content: 'hello world' } },
      {
        type: 'assistant',
        message: { content: [{ type: 'text', text: 'hi back' }] },
      },
    ]);
    expect([...extractCompactProvenance(jsonl)]).toEqual([]);
  });

  it('extracts an Encoding A wrap from a user message body', () => {
    const jsonl = jsonlOf([
      {
        type: 'user',
        message: {
          content: wrap('web', 'https://example.com/post', 'page text'),
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonl)]).toEqual([
      'web:https://example.com/post',
    ]);
  });

  it('extracts an Encoding B sentinel from a tool_result block', () => {
    const jsonl = jsonlOf([
      {
        type: 'user',
        message: {
          content: [
            {
              type: 'tool_result',
              content: sentinel('gmail', 'msg=abc123'),
            },
          ],
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonl)]).toEqual(['gmail:msg=abc123']);
  });

  it('extracts both encodings across the same transcript', () => {
    const jsonl = jsonlOf([
      { type: 'user', message: { content: wrap('web', 'https://a.io', 'a') } },
      {
        type: 'user',
        message: {
          content: [
            { type: 'tool_result', content: sentinel('calendar', 'evt=xyz') },
          ],
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonl)].sort()).toEqual([
      'calendar:evt=xyz',
      'web:https://a.io',
    ]);
  });

  it('dedupes identical sources across messages', () => {
    const jsonl = jsonlOf([
      { type: 'user', message: { content: wrap('web', 'https://x.io', 'a') } },
      { type: 'user', message: { content: wrap('web', 'https://x.io', 'b') } },
      {
        type: 'user',
        message: {
          content: [
            { type: 'tool_result', content: sentinel('web', 'https://x.io') },
          ],
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonl)]).toEqual(['web:https://x.io']);
  });

  it('preserves distinct sources even when they share a prefix', () => {
    const jsonl = jsonlOf([
      { type: 'user', message: { content: wrap('gmail', 'msg=1', 'a') } },
      { type: 'user', message: { content: wrap('gmail', 'msg=2', 'b') } },
    ]);
    expect([...extractCompactProvenance(jsonl)].sort()).toEqual([
      'gmail:msg=1',
      'gmail:msg=2',
    ]);
  });

  it('skips lines that fail to parse (no JSON validation needed)', () => {
    // Live transcripts can have partial trailing writes during a hot
    // read; the regex sweep is line-agnostic and just keeps going.
    const lines = [
      JSON.stringify({
        type: 'user',
        message: { content: wrap('web', 'a', 'x') },
      }),
      '{"type":"user", malformed garbage',
      JSON.stringify({
        type: 'user',
        message: { content: wrap('gmail', 'b', 'y') },
      }),
    ].join('\n');
    expect([...extractCompactProvenance(lines)].sort()).toEqual([
      'gmail:b',
      'web:a',
    ]);
  });

  it('rejects malformed source attributes (missing colon)', () => {
    // `<untrusted-input source="bare">` has no prefix:value separator;
    // accepting it would let the sidecar carry junk that the ACL would
    // collapse to UNKNOWN_PREFIX for the wrong reason.
    const jsonl = jsonlOf([
      {
        type: 'user',
        message: {
          content: '<untrusted-input source="bare">x</untrusted-input>',
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonl)]).toEqual([]);
  });

  it('rejects sources with empty prefix or empty value', () => {
    const jsonl = jsonlOf([
      {
        type: 'user',
        message: {
          content: '<untrusted-input source=":foo">x</untrusted-input>',
        },
      },
      {
        type: 'user',
        message: {
          content: '<untrusted-input source="bar:">y</untrusted-input>',
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonl)]).toEqual([]);
  });

  it('rejects sources exceeding the byte cap', () => {
    const huge = 'x'.repeat(3000);
    const jsonl = jsonlOf([
      {
        type: 'user',
        message: {
          content: `<untrusted-input source="web:${huge}">y</untrusted-input>`,
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonl)]).toEqual([]);
  });

  it('rejects sources containing newline characters', () => {
    // A `\n` in the source would break the line-anchored Encoding B
    // regex on the synthetic post-compaction marker, AND let an
    // attacker escape the marker line to inject arbitrary
    // system-reminder text. Both Encoding A and Encoding B emitters
    // already reject newlines via attribute escaping, so this is a
    // belt-and-suspenders defence at the sidecar boundary.
    const jsonlA = jsonlOf([
      {
        type: 'user',
        message: {
          content:
            '<untrusted-input source="web:line1\nline2">x</untrusted-input>',
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonlA)]).toEqual([]);
    const jsonlB = jsonlOf([
      {
        type: 'user',
        message: {
          content: [
            {
              type: 'tool_result',
              content:
                'PROVENANCE_MARKER: source="web:a\rb" tool_use_id="tu_x"',
            },
          ],
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonlB)]).toEqual([]);
  });

  it('Encoding A/B regex tokenisation already keeps embedded quotes out of captures', () => {
    // The regexes capture `[^"]*` for source values, so an attempt to
    // sneak a `"` into the source attribute terminates the capture at
    // the first quote — `web:a"b` only ever yields `web:a`, never
    // a multi-attribute payload. Combined with `addIfValid`'s explicit
    // quote rejection, both Encoding A and Encoding B paths are safe;
    // the parseSidecar path covered separately below handles the
    // tampered-at-rest case where the regex isn't involved.
    const jsonl = jsonlOf([
      {
        type: 'user',
        message: {
          content: '<untrusted-input source="web:safe">x</untrusted-input>',
        },
      },
    ]);
    expect([...extractCompactProvenance(jsonl)]).toEqual(['web:safe']);
  });

  it('caps the result at MAX_SOURCES and preserves earliest sources', () => {
    const blocks: Array<Record<string, unknown>> = [];
    for (let i = 0; i < MAX_SOURCES + 5; i++) {
      blocks.push({
        type: 'user',
        message: { content: wrap('web', `https://e${i}.io`, 'x') },
      });
    }
    const result = extractCompactProvenance(jsonlOf(blocks));
    expect(result.size).toBe(MAX_SOURCES);
    // First source seen must be present; sources past the cap dropped.
    expect(result.has('web:https://e0.io')).toBe(true);
    expect(result.has(`web:https://e${MAX_SOURCES + 4}.io`)).toBe(false);
  });
});

// ---- buildPostCompactReminder ----

describe('buildPostCompactReminder', () => {
  it('returns null on empty source set', () => {
    expect(buildPostCompactReminder(new Set())).toBeNull();
  });

  it('emits one PROVENANCE_MARKER line per source', () => {
    const result = buildPostCompactReminder(
      new Set(['web:https://a.io', 'gmail:msg=1']),
    );
    expect(result).not.toBeNull();
    expect(result).toMatch(
      /PROVENANCE_MARKER: source="web:https:\/\/a\.io" tool_use_id="compact-laundering-defence"/,
    );
    expect(result).toMatch(
      /PROVENANCE_MARKER: source="gmail:msg=1" tool_use_id="compact-laundering-defence"/,
    );
  });

  it('reminder text is recognised by capability-acl extractMarkerPrefixes', () => {
    // The whole point: the reminder must round-trip through the
    // walk-back's prefix extractor and produce the same prefixes the
    // pre-compaction transcript carried.
    const sources = new Set(['web:https://x.io', 'calendar:evt=1']);
    const reminder = buildPostCompactReminder(sources)!;
    const prefixes = extractMarkerPrefixes(reminder);
    expect([...prefixes].sort()).toEqual(['calendar', 'web']);
  });

  it('reminder text uses the canonical Encoding B regex form', () => {
    const reminder = buildPostCompactReminder(new Set(['web:https://x.io']))!;
    const re = new RegExp(__ACL_INTERNALS.ENCODING_B_REGEX, 'gm');
    const matches = [...reminder.matchAll(re)];
    expect(matches.length).toBe(1);
    expect(matches[0][1]).toBe('web:https://x.io');
  });

  it('preamble explains the laundering defence rationale', () => {
    const reminder = buildPostCompactReminder(new Set(['web:https://x.io']))!;
    // Loose contract — the preamble exists and references the mechanism.
    // Keeps the test from wedging on minor wording changes while still
    // catching accidental removal of the explanatory text.
    expect(reminder).toMatch(/POST-COMPACTION PROVENANCE/);
    expect(reminder).toMatch(/untrusted/i);
  });
});

// ---- parseSidecar ----

describe('parseSidecar', () => {
  it('accepts a well-formed payload', () => {
    const payload: CompactProvenanceSidecar = {
      schema_version: 1,
      session_id: 'sess-123',
      created_at: 1_700_000_000_000,
      sources: ['web:https://a.io'],
    };
    expect(parseSidecar(payload)).toEqual(payload);
  });

  it('rejects null and non-objects', () => {
    expect(parseSidecar(null)).toBeNull();
    expect(parseSidecar('string')).toBeNull();
    expect(parseSidecar(42)).toBeNull();
  });

  it('rejects missing or wrong schema_version', () => {
    expect(
      parseSidecar({
        schema_version: 0,
        session_id: 'a',
        created_at: 0,
        sources: [],
      }),
    ).toBeNull();
    expect(
      parseSidecar({
        schema_version: 2,
        session_id: 'a',
        created_at: 0,
        sources: [],
      }),
    ).toBeNull();
    expect(
      parseSidecar({ session_id: 'a', created_at: 0, sources: [] }),
    ).toBeNull();
  });

  it('rejects empty session_id', () => {
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: '',
        created_at: 0,
        sources: [],
      }),
    ).toBeNull();
  });

  it('rejects non-array sources', () => {
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: 'web:x',
      }),
    ).toBeNull();
  });

  it('rejects sources containing non-string entries', () => {
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: [42],
      }),
    ).toBeNull();
  });

  it('rejects sources entries exceeding the byte cap', () => {
    const huge = 'x'.repeat(3000);
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: [huge],
      }),
    ).toBeNull();
  });

  it('rejects malformed prefix:value entries (sidecar tampering)', () => {
    // The extractor enforces prefix:value at write time, but a
    // sidecar can be edited at rest. parseSidecar must re-validate
    // every entry — otherwise a tampered sidecar lands synthetic
    // markers whose post-compaction effect isn't what the ACL
    // expects.
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: ['no-colon'],
      }),
    ).toBeNull();
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: [':leading-colon'],
      }),
    ).toBeNull();
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: ['trailing-colon:'],
      }),
    ).toBeNull();
  });

  it('rejects sources containing newlines or quotes (sidecar tampering)', () => {
    // The reminder builder interpolates source values into
    // `source="..."` slots and into single-line markers; an embedded
    // `\n` or `"` from a tampered sidecar would either escape the
    // attribute boundary or break the Encoding B line shape.
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: ['web:line1\nline2'],
      }),
    ).toBeNull();
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: ['web:has"quote'],
      }),
    ).toBeNull();
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: ['web:carriage\rreturn'],
      }),
    ).toBeNull();
  });

  it('rejects sources arrays exceeding MAX_SOURCES', () => {
    // A tampered sidecar with a million entries would otherwise pin a
    // CPU re-validating each one before failing the contents check.
    // The cap matches the writer's so a legitimate file never trips it.
    const oversized = Array.from(
      { length: MAX_SOURCES + 1 },
      (_, i) => `web:https://e${i}.io`,
    );
    expect(
      parseSidecar({
        schema_version: 1,
        session_id: 'a',
        created_at: 0,
        sources: oversized,
      }),
    ).toBeNull();
  });

  it('accepts sources arrays exactly at MAX_SOURCES', () => {
    const atCap = Array.from(
      { length: MAX_SOURCES },
      (_, i) => `web:https://e${i}.io`,
    );
    const result = parseSidecar({
      schema_version: 1,
      session_id: 'a',
      created_at: 0,
      sources: atCap,
    });
    expect(result).not.toBeNull();
    expect(result!.sources.length).toBe(MAX_SOURCES);
  });
});

// ---- writeCompactProvenanceSidecar / readAndClearSidecar ----

describe('sidecar persistence', () => {
  it('writes a sidecar that read-and-clear returns', () => {
    writeCompactProvenanceSidecar(
      tmpRoot,
      'sess-A',
      new Set(['web:https://a.io']),
    );
    const result = readAndClearSidecar(tmpRoot, 'sess-A');
    expect([...result]).toEqual(['web:https://a.io']);
  });

  it('deletes the sidecar after successful read', () => {
    writeCompactProvenanceSidecar(
      tmpRoot,
      'sess-A',
      new Set(['web:https://a.io']),
    );
    const filePath = sidecarPathFor(tmpRoot, 'sess-A');
    expect(fs.existsSync(filePath)).toBe(true);
    readAndClearSidecar(tmpRoot, 'sess-A');
    expect(fs.existsSync(filePath)).toBe(false);
  });

  it('returns empty set + leaves file in place on session id mismatch', () => {
    // Defence-in-depth: the path is keyed on session id, but a stranded
    // file from another session should never be deleted by the wrong
    // owner. Drop a file under sess-B and try to read it as sess-A.
    const bPath = sidecarPathFor(tmpRoot, 'sess-B');
    const aPath = sidecarPathFor(tmpRoot, 'sess-A');
    fs.mkdirSync(tmpRoot, { recursive: true });
    fs.writeFileSync(
      aPath,
      JSON.stringify({
        schema_version: 1,
        session_id: 'sess-B',
        created_at: 1,
        sources: ['web:https://a.io'],
      }),
    );
    const result = readAndClearSidecar(tmpRoot, 'sess-A');
    expect([...result]).toEqual([]);
    expect(fs.existsSync(aPath)).toBe(true);
    // sess-B's own keyed path was never touched.
    expect(fs.existsSync(bPath)).toBe(false);
  });

  it('returns empty set when no sidecar exists', () => {
    expect([...readAndClearSidecar(tmpRoot, 'no-such-session')]).toEqual([]);
  });

  it('drops a corrupt sidecar and returns empty', () => {
    const filePath = sidecarPathFor(tmpRoot, 'sess-A');
    fs.mkdirSync(tmpRoot, { recursive: true });
    fs.writeFileSync(filePath, 'not json {{{');
    const result = readAndClearSidecar(tmpRoot, 'sess-A');
    expect([...result]).toEqual([]);
    expect(fs.existsSync(filePath)).toBe(false);
  });

  it('drops a sidecar with invalid shape and returns empty', () => {
    const filePath = sidecarPathFor(tmpRoot, 'sess-A');
    fs.mkdirSync(tmpRoot, { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify({ schema_version: 99 }));
    const result = readAndClearSidecar(tmpRoot, 'sess-A');
    expect([...result]).toEqual([]);
    expect(fs.existsSync(filePath)).toBe(false);
  });

  it('write returns null on empty source set', () => {
    const filePath = writeCompactProvenanceSidecar(
      tmpRoot,
      'sess-A',
      new Set(),
    );
    expect(filePath).toBeNull();
    expect(fs.existsSync(sidecarPathFor(tmpRoot, 'sess-A'))).toBe(false);
  });

  it('write returns null on empty session id', () => {
    expect(
      writeCompactProvenanceSidecar(tmpRoot, '', new Set(['web:https://a.io'])),
    ).toBeNull();
  });

  it('sidecar path uses basename to defend against traversal', () => {
    // A session id with traversal characters resolves to a flat name —
    // the SDK never produces these, but defence-in-depth means the
    // sidecar can't escape the state dir even if one slips through.
    const safe = sidecarPathFor(tmpRoot, '../escape');
    expect(safe).toBe(path.join(tmpRoot, 'escape.json'));
  });

  it('write creates the state dir if missing', () => {
    const nested = path.join(tmpRoot, 'nested', 'deeper');
    writeCompactProvenanceSidecar(
      nested,
      'sess-A',
      new Set(['web:https://a.io']),
    );
    expect(fs.existsSync(sidecarPathFor(nested, 'sess-A'))).toBe(true);
  });
});

// ---- persistCompactProvenance ----

describe('persistCompactProvenance', () => {
  it('reads transcript, extracts sources, writes sidecar', () => {
    const transcriptPath = path.join(tmpRoot, 'transcript.jsonl');
    fs.writeFileSync(
      transcriptPath,
      jsonlOf([
        {
          type: 'user',
          message: { content: wrap('web', 'https://a.io', 'x') },
        },
        { type: 'user', message: { content: wrap('gmail', 'msg=1', 'y') } },
      ]),
    );
    const stateDir = path.join(tmpRoot, 'state');
    const count = persistCompactProvenance(transcriptPath, stateDir, 'sess-A');
    expect(count).toBe(2);
    const sources = readAndClearSidecar(stateDir, 'sess-A');
    expect([...sources].sort()).toEqual(['gmail:msg=1', 'web:https://a.io']);
  });

  it('returns 0 and skips sidecar when transcript has no markers', () => {
    const transcriptPath = path.join(tmpRoot, 'transcript.jsonl');
    fs.writeFileSync(
      transcriptPath,
      jsonlOf([{ type: 'user', message: { content: 'plain owner prompt' } }]),
    );
    const stateDir = path.join(tmpRoot, 'state');
    expect(persistCompactProvenance(transcriptPath, stateDir, 'sess-A')).toBe(
      0,
    );
    expect(fs.existsSync(sidecarPathFor(stateDir, 'sess-A'))).toBe(false);
  });

  it('returns 0 on missing transcript path (logged, not thrown)', () => {
    const messages: string[] = [];
    const count = persistCompactProvenance(
      path.join(tmpRoot, 'no-such.jsonl'),
      tmpRoot,
      'sess-A',
      (m) => messages.push(m),
    );
    expect(count).toBe(0);
    expect(messages.some((m) => m.includes('transcript read failed'))).toBe(
      true,
    );
  });
});

// ---- end-to-end roundtrip ----

describe('end-to-end roundtrip', () => {
  it('PreCompact-style persist + PostCompact-style read produces the same prefixes the walk-back saw', () => {
    // This is the load-bearing invariant: the synthetic markers we
    // re-inject after compaction must produce the SAME prefix set the
    // capability-ACL would have collected from the live pre-compaction
    // transcript. If this drifts, the post-compaction defence is
    // either over-tightened (wrong allowlist intersection) or under-
    // tightened (gates re-open).
    const transcriptPath = path.join(tmpRoot, 'transcript.jsonl');
    fs.writeFileSync(
      transcriptPath,
      jsonlOf([
        {
          type: 'user',
          message: { content: wrap('web', 'https://a.io', 'x') },
        },
        {
          type: 'user',
          message: {
            content: [
              { type: 'tool_result', content: sentinel('gmail', 'msg=1') },
            ],
          },
        },
        { type: 'user', message: { content: wrap('calendar', 'evt=q', 'y') } },
      ]),
    );

    const stateDir = path.join(tmpRoot, 'state');
    persistCompactProvenance(transcriptPath, stateDir, 'sess-A');
    const sources = readAndClearSidecar(stateDir, 'sess-A');
    const reminder = buildPostCompactReminder(sources)!;

    // Walk-back-style prefix extraction over the synthetic reminder
    // returns the union of prefixes the pre-compaction transcript had.
    const prefixes = extractMarkerPrefixes(reminder);
    expect([...prefixes].sort()).toEqual(['calendar', 'gmail', 'web']);
  });
});
