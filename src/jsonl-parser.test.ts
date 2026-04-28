import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { parseSessionTranscript } from './jsonl-parser.js';

// `jsonl-parser` walks the SDK's session JSONL transcript and yields
// completed/in-flight tool invocations in execution order. The design
// doc §2 (`docs/proposals/kill-auto-compaction.md`) makes this the
// foundation of the `## Facts` writer's "do NOT re-execute" list, so
// the test plan covers:
//   - basic tool_use → tool_result pairing
//   - in-flight tool_use without a matching result (timestamp = null)
//   - torn write at tail (corrupt JSON line skipped, walk continues)
//   - missing file (returns empty without throwing)
//
// Fixtures are built programmatically in setup per testing-standards
// (no binary fixtures checked in).

let tmpDir: string;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jsonl-parser-test-'));
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

function writeJsonl(filename: string, lines: object[]): string {
  const filePath = path.join(tmpDir, filename);
  fs.writeFileSync(
    filePath,
    lines.map((l) => JSON.stringify(l)).join('\n') + '\n',
  );
  return filePath;
}

describe('parseSessionTranscript — happy path', () => {
  it('pairs tool_use with tool_result in execution order', async () => {
    const filePath = writeJsonl('session.jsonl', [
      {
        type: 'assistant',
        timestamp: '2026-04-27T10:00:00.000Z',
        message: {
          content: [
            {
              type: 'tool_use',
              id: 'toolu_01abc',
              name: 'Write',
              input: { file_path: '/tmp/foo.txt', content: 'hello' },
            },
          ],
        },
      },
      {
        type: 'user',
        timestamp: '2026-04-27T10:00:01.500Z',
        message: {
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_01abc',
              is_error: false,
            },
          ],
        },
      },
      {
        type: 'assistant',
        timestamp: '2026-04-27T10:00:02.000Z',
        message: {
          content: [
            {
              type: 'tool_use',
              id: 'toolu_02def',
              name: 'Bash',
              input: { command: 'ls -la' },
            },
          ],
        },
      },
      {
        type: 'user',
        timestamp: '2026-04-27T10:00:02.300Z',
        message: {
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_02def',
              is_error: false,
            },
          ],
        },
      },
    ]);

    const invocations = await parseSessionTranscript(filePath);
    expect(invocations).toHaveLength(2);
    expect(invocations[0]).toMatchObject({
      name: 'Write',
      id: 'toolu_01abc',
      startedAt: '2026-04-27T10:00:00.000Z',
      completedAt: '2026-04-27T10:00:01.500Z',
      isError: false,
    });
    expect(invocations[0].input).toEqual({
      file_path: '/tmp/foo.txt',
      content: 'hello',
    });
    expect(invocations[1]).toMatchObject({
      name: 'Bash',
      id: 'toolu_02def',
      completedAt: '2026-04-27T10:00:02.300Z',
    });
  });
});

describe('parseSessionTranscript — in-flight tool_use', () => {
  it('returns completedAt=null when no matching tool_result exists', async () => {
    const filePath = writeJsonl('session.jsonl', [
      {
        type: 'assistant',
        timestamp: '2026-04-27T10:00:00.000Z',
        message: {
          content: [
            {
              type: 'tool_use',
              id: 'toolu_inflight',
              name: 'Bash',
              input: { command: 'sleep 30' },
            },
          ],
        },
      },
    ]);

    const invocations = await parseSessionTranscript(filePath);
    expect(invocations).toHaveLength(1);
    expect(invocations[0].completedAt).toBeNull();
    expect(invocations[0].isError).toBeNull();
  });
});

describe('parseSessionTranscript — error result', () => {
  it('captures is_error=true on the matched result', async () => {
    const filePath = writeJsonl('session.jsonl', [
      {
        type: 'assistant',
        timestamp: '2026-04-27T10:00:00.000Z',
        message: {
          content: [
            {
              type: 'tool_use',
              id: 'toolu_err',
              name: 'Bash',
              input: { command: 'false' },
            },
          ],
        },
      },
      {
        type: 'user',
        timestamp: '2026-04-27T10:00:00.500Z',
        message: {
          content: [
            { type: 'tool_result', tool_use_id: 'toolu_err', is_error: true },
          ],
        },
      },
    ]);

    const invocations = await parseSessionTranscript(filePath);
    expect(invocations[0].isError).toBe(true);
  });
});

describe('parseSessionTranscript — torn write tolerance', () => {
  it('skips unparseable lines and continues the walk', async () => {
    const filePath = path.join(tmpDir, 'session.jsonl');
    const goodLine = JSON.stringify({
      type: 'assistant',
      timestamp: '2026-04-27T10:00:00.000Z',
      message: {
        content: [
          {
            type: 'tool_use',
            id: 'toolu_pre',
            name: 'Read',
            input: { file_path: '/etc/hosts' },
          },
        ],
      },
    });
    const tornLine =
      '{"type":"assistant","timestamp":"2026-04-27T10:00:01.000Z","message":{"content":[{"type":"tool_use","id":"toolu_torn","name":"Wri'; // unterminated
    const goodAfter = JSON.stringify({
      type: 'assistant',
      timestamp: '2026-04-27T10:00:02.000Z',
      message: {
        content: [
          {
            type: 'tool_use',
            id: 'toolu_post',
            name: 'Edit',
            input: { file_path: '/tmp/x' },
          },
        ],
      },
    });
    fs.writeFileSync(
      filePath,
      [goodLine, tornLine, goodAfter].join('\n') + '\n',
    );

    const invocations = await parseSessionTranscript(filePath);
    expect(invocations.map((i) => i.name)).toEqual(['Read', 'Edit']);
  });
});

describe('parseSessionTranscript — missing file', () => {
  it('returns empty array without throwing', async () => {
    const result = await parseSessionTranscript(
      path.join(tmpDir, 'does-not-exist.jsonl'),
    );
    expect(result).toEqual([]);
  });
});

describe('parseSessionTranscript — non-content lines', () => {
  it('ignores system messages and result messages with no tool_use blocks', async () => {
    const filePath = writeJsonl('session.jsonl', [
      { type: 'system', subtype: 'init', session_id: 'sess_xyz' },
      { type: 'result', subtype: 'success', total_cost_usd: 0.05 },
      {
        type: 'assistant',
        timestamp: '2026-04-27T10:00:00.000Z',
        message: { content: [{ type: 'text', text: 'plain reply' }] },
      },
    ]);

    const invocations = await parseSessionTranscript(filePath);
    expect(invocations).toEqual([]);
  });
});
