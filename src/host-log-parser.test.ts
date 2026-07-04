import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect } from 'vitest';

import {
  findGateDecisions,
  parseHostLog,
  readHostLog,
} from './host-log-parser.js';

// #443 — The parser is the contract `inspect_gate_decisions` greps
// against. Tests pin the orchestrator-log block format produced by
// `src/logger.ts:formatData`: `[HH:mm:ss.SSS] LEVEL (pid): msg`
// header line followed by `    key: <JSON.stringify(value)>` indented
// fields. Adjacent records are separated by either a blank line or
// the next header line.

describe('parseHostLog (#443)', () => {
  it('parses a single record with multiple JSON-stringified fields', () => {
    const raw = [
      '[12:53:52.132] INFO (1): multi-field record',
      '    groupFolder: "telegram_old-wtf"',
      '    intent: "no"',
      '    confidence: 0.85',
      '    reason: "Human-to-human reply thread"',
      '    durationMs: 1628',
      '',
    ].join('\n');
    const records = parseHostLog(raw);
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({
      timestamp: '12:53:52.132',
      level: 'INFO',
      pid: 1,
      msg: 'multi-field record',
      fields: {
        groupFolder: 'telegram_old-wtf',
        intent: 'no',
        confidence: 0.85,
        reason: 'Human-to-human reply thread',
        durationMs: 1628,
      },
    });
  });

  it('parses adjacent records separated by a blank line', () => {
    const raw = [
      '[12:00:00.000] INFO (1): first',
      '    a: 1',
      '',
      '[12:00:01.000] WARN (1): second',
      '    b: "two"',
      '',
    ].join('\n');
    const records = parseHostLog(raw);
    expect(records).toHaveLength(2);
    expect(records[0].msg).toBe('first');
    expect(records[0].fields).toEqual({ a: 1 });
    expect(records[1].msg).toBe('second');
    expect(records[1].fields).toEqual({ b: 'two' });
  });

  it('parses adjacent records with no blank-line separator (back-to-back headers)', () => {
    // The logger doesn't always emit a blank between records — high
    // log volume omits the gap so the file stays compact. The parser
    // must close the current record on the next header line, not on
    // a blank-line marker.
    const raw = [
      '[12:00:00.000] INFO (1): first',
      '    a: 1',
      '[12:00:01.000] INFO (1): second',
      '    b: 2',
    ].join('\n');
    const records = parseHostLog(raw);
    expect(records).toHaveLength(2);
    expect(records[0].fields).toEqual({ a: 1 });
    expect(records[1].fields).toEqual({ b: 2 });
  });

  it('inflates JSON-shaped values (objects, arrays) into native types', () => {
    // The `chain` field on `'gate decision'` records is the
    // canonical case: array-of-objects round-trips because the
    // logger JSON-stringifies it on write.
    const raw = [
      '[12:00:00.000] INFO (1): gate decision',
      '    chain: [{"gate":"trigger","decision":"allow","reason":"matched"}]',
      '    finalDecision: "allow"',
      '',
    ].join('\n');
    const records = parseHostLog(raw);
    expect(records).toHaveLength(1);
    expect(records[0].fields.chain).toEqual([
      { gate: 'trigger', decision: 'allow', reason: 'matched' },
    ]);
    expect(records[0].fields.finalDecision).toBe('allow');
  });

  it('keeps non-JSON values as raw strings without throwing', () => {
    // `formatErr` paths in logger.ts can render multi-line stack
    // traces under `err:` that don't JSON-parse on a single line.
    // The parser must not throw on those — keep the raw post-colon
    // text so callers can still see the error class even if the
    // structured-fields contract fails for that record.
    const raw = [
      '[12:00:00.000] ERROR (1): boom',
      '    err: SyntaxError: not actually json',
      '',
    ].join('\n');
    const records = parseHostLog(raw);
    expect(records).toHaveLength(1);
    expect(records[0].fields.err).toBe('SyntaxError: not actually json');
  });

  it('strips ANSI color codes idempotently', () => {
    // The file sink strips ANSI on write (see `logger.ts:stripAnsi`),
    // but rotated logs from older builds may retain them. The
    // parser strips again as a belt-and-braces step so a historical
    // log line still resolves cleanly.
    const raw = [
      '\x1b[36m[12:00:00.000]\x1b[0m \x1b[32mINFO\x1b[0m (1): \x1b[35mhello\x1b[0m',
      '    a: 1',
      '',
    ].join('\n');
    const records = parseHostLog(raw);
    expect(records).toHaveLength(1);
    expect(records[0].msg).toBe('hello');
    expect(records[0].fields.a).toBe(1);
  });

  it('returns an empty array for a missing file via readHostLog (no throw)', () => {
    // `inspect_gate_decisions` treats "no log file" identically to
    // "no matching records" — both yield zero-length results.
    // ENOENT must not propagate.
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nclog-'));
    try {
      const missing = path.join(tempDir, 'never-written.log');
      expect(readHostLog(missing)).toEqual([]);
    } finally {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });
});

describe('findGateDecisions (#443)', () => {
  function makeGateDecisionRaw(args: {
    timestamp: string;
    chatJid: string;
    messageId: string;
    groupFolder: string;
    finalDecision: 'allow' | 'deny';
    reason: string;
    chain: Array<{
      gate: string;
      decision: 'allow' | 'deny' | 'pass';
      reason: string;
    }>;
  }): string {
    return [
      `[${args.timestamp}] INFO (1): gate decision`,
      `    chatJid: ${JSON.stringify(args.chatJid)}`,
      `    messageId: ${JSON.stringify(args.messageId)}`,
      `    groupFolder: ${JSON.stringify(args.groupFolder)}`,
      `    finalDecision: ${JSON.stringify(args.finalDecision)}`,
      `    reason: ${JSON.stringify(args.reason)}`,
      `    chain: ${JSON.stringify(args.chain)}`,
      '',
    ].join('\n');
  }

  it('filters by chatJid and returns most-recent first', () => {
    const raw = [
      makeGateDecisionRaw({
        timestamp: '12:00:00.000',
        chatJid: 'tg:-1',
        messageId: 'msg-a',
        groupFolder: 'g1',
        finalDecision: 'allow',
        reason: 'chain-allow',
        chain: [{ gate: 'trigger', decision: 'allow', reason: 'matched' }],
      }),
      makeGateDecisionRaw({
        timestamp: '12:00:01.000',
        chatJid: 'tg:-2',
        messageId: 'msg-other-chat',
        groupFolder: 'g2',
        finalDecision: 'allow',
        reason: 'unrelated',
        chain: [{ gate: 'trigger', decision: 'allow', reason: 'unrelated' }],
      }),
      makeGateDecisionRaw({
        timestamp: '12:00:02.000',
        chatJid: 'tg:-1',
        messageId: 'msg-b',
        groupFolder: 'g1',
        finalDecision: 'deny',
        reason: 'gate-b-no',
        chain: [
          { gate: 'trigger', decision: 'allow', reason: 'matched' },
          {
            gate: 'gate-b',
            decision: 'deny',
            reason: 'human-to-human',
          },
        ],
      }),
    ].join('');
    const records = parseHostLog(raw);
    const hits = findGateDecisions(records, { chatJid: 'tg:-1' });
    expect(hits).toHaveLength(2);
    // Most-recent first.
    expect(hits[0].messageId).toBe('msg-b');
    expect(hits[0].finalDecision).toBe('deny');
    expect(hits[0].chain).toHaveLength(2);
    expect(hits[1].messageId).toBe('msg-a');
  });

  it('narrows by messageId when provided', () => {
    const raw = [
      makeGateDecisionRaw({
        timestamp: '12:00:00.000',
        chatJid: 'tg:-1',
        messageId: 'msg-a',
        groupFolder: 'g1',
        finalDecision: 'allow',
        reason: 'r1',
        chain: [{ gate: 'trigger', decision: 'allow', reason: 'r1' }],
      }),
      makeGateDecisionRaw({
        timestamp: '12:00:01.000',
        chatJid: 'tg:-1',
        messageId: 'msg-target',
        groupFolder: 'g1',
        finalDecision: 'deny',
        reason: 'r-target',
        chain: [{ gate: 'trigger', decision: 'pass', reason: 'r-target' }],
      }),
      makeGateDecisionRaw({
        timestamp: '12:00:02.000',
        chatJid: 'tg:-1',
        messageId: 'msg-c',
        groupFolder: 'g1',
        finalDecision: 'allow',
        reason: 'r3',
        chain: [{ gate: 'trigger', decision: 'allow', reason: 'r3' }],
      }),
    ].join('');
    const records = parseHostLog(raw);
    const hits = findGateDecisions(records, {
      chatJid: 'tg:-1',
      messageId: 'msg-target',
    });
    expect(hits).toHaveLength(1);
    expect(hits[0].messageId).toBe('msg-target');
    expect(hits[0].reason).toBe('r-target');
  });

  it('honors the limit argument', () => {
    let raw = '';
    for (let i = 0; i < 25; i++) {
      raw += makeGateDecisionRaw({
        timestamp: `12:00:${String(i).padStart(2, '0')}.000`,
        chatJid: 'tg:-1',
        messageId: `msg-${i}`,
        groupFolder: 'g1',
        finalDecision: 'allow',
        reason: `r${i}`,
        chain: [{ gate: 'trigger', decision: 'allow', reason: `r${i}` }],
      });
    }
    const records = parseHostLog(raw);
    const hits = findGateDecisions(records, { chatJid: 'tg:-1', limit: 5 });
    expect(hits).toHaveLength(5);
    // Most-recent first — last 5 records (msg-24 down to msg-20).
    expect(hits.map((h) => h.messageId)).toEqual([
      'msg-24',
      'msg-23',
      'msg-22',
      'msg-21',
      'msg-20',
    ]);
  });

  it('skips records that do not match the gate-decision shape', () => {
    // Other INFO records in the log (e.g. `'multi-field record'`)
    // must not be returned — only the canonical `'gate decision'`
    // line is the contract for this tool.
    const raw = [
      makeGateDecisionRaw({
        timestamp: '12:00:00.000',
        chatJid: 'tg:-1',
        messageId: 'msg-a',
        groupFolder: 'g1',
        finalDecision: 'allow',
        reason: 'r1',
        chain: [{ gate: 'trigger', decision: 'allow', reason: 'r1' }],
      }),
      [
        '[12:00:01.000] INFO (1): multi-field record',
        '    chatJid: "tg:-1"',
        '    intent: "no"',
        '',
      ].join('\n'),
    ].join('');
    const records = parseHostLog(raw);
    const hits = findGateDecisions(records, { chatJid: 'tg:-1' });
    expect(hits).toHaveLength(1);
    expect(hits[0].messageId).toBe('msg-a');
  });

  it('skips malformed gate-decision records (missing chain) without throwing', () => {
    const raw = [
      // Healthy.
      makeGateDecisionRaw({
        timestamp: '12:00:00.000',
        chatJid: 'tg:-1',
        messageId: 'msg-good',
        groupFolder: 'g1',
        finalDecision: 'allow',
        reason: 'r1',
        chain: [{ gate: 'trigger', decision: 'allow', reason: 'r1' }],
      }),
      // Missing `chain` field.
      [
        '[12:00:01.000] INFO (1): gate decision',
        '    chatJid: "tg:-1"',
        '    messageId: "msg-bad"',
        '    groupFolder: "g1"',
        '    finalDecision: "allow"',
        '    reason: "no chain"',
        '',
      ].join('\n'),
    ].join('');
    const records = parseHostLog(raw);
    const hits = findGateDecisions(records, { chatJid: 'tg:-1' });
    expect(hits).toHaveLength(1);
    expect(hits[0].messageId).toBe('msg-good');
  });
});
