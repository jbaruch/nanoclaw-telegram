import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  checkpointPaths,
  clearCheckpoints,
  renderFacts,
  summariseInput,
  writeCheckpoint,
} from './checkpoint.js';
import { computeThresholds } from './threshold.js';

// Mock logger to keep test output clean.
vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Tests cover the three pieces of the kill-auto-compaction Phase 2
// surface (design doc §1, §2, §6, `docs/proposals/kill-auto-compaction.md`):
//
//   - `renderFacts` produces a stable Markdown shape that reentry can
//     consume verbatim. Tests pin the section headings the reentry
//     hook will key on; if a heading drifts, reentry breaks silently.
//   - `summariseInput` picks the most-load-bearing key per tool family
//     so the Facts list is readable without dumping megabyte-sized
//     Write inputs into the checkpoint.
//   - `writeCheckpoint` rotates default.md → previous.md atomically
//     before the new write. Idempotent across consecutive triggers
//     and tolerant of a missing prior file (first-ever run).

let tmpDir: string;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'checkpoint-test-'));
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('checkpointPaths', () => {
  it('derives dir, live, and previous paths under .checkpoints', () => {
    const paths = checkpointPaths('/groups/test');
    expect(paths.dir).toBe('/groups/test/.checkpoints');
    expect(paths.live).toBe('/groups/test/.checkpoints/default.md');
    expect(paths.previous).toBe('/groups/test/.checkpoints/previous.md');
  });
});

describe('renderFacts', () => {
  const baseInputs = {
    groupDir: '/groups/test',
    jsonlPath: '/dev/null',
    sessionId: 'sess_abc123',
    thresholds: computeThresholds(1_000_000),
    usedTokens: 825_000,
    groupName: 'main',
  };

  it('renders the canonical headings reentry will key on', () => {
    const out = renderFacts(baseInputs, []);
    expect(out).toContain('# Session Checkpoint');
    expect(out).toContain('## Facts');
    expect(out).toContain('### Pending replies');
    expect(out).toContain('### Do NOT re-execute');
  });

  it('renders no-mutating-calls degraded line when invocation list empty', () => {
    const out = renderFacts(baseInputs, []);
    expect(out).toContain('_no mutating calls observed_');
  });

  it('renders pending replies "_none_" placeholder when list missing', () => {
    const out = renderFacts(baseInputs, []);
    expect(out).toContain('- _none_');
  });

  it('lists supplied pending reply ids', () => {
    const out = renderFacts(
      { ...baseInputs, pendingReplyIds: ['msg_1', 'msg_2'] },
      [],
    );
    expect(out).toContain('- `msg_1`');
    expect(out).toContain('- `msg_2`');
  });

  it('formats mutating invocations with tool name, summary, and completion time', () => {
    const out = renderFacts(baseInputs, [
      {
        name: 'Write',
        inputSummary: '`/tmp/foo.txt`',
        completedAt: '2026-04-27T10:00:01.500Z',
        isError: false,
      },
      {
        name: 'mcp__nanoclaw__send_message',
        inputSummary: '42 chars',
        completedAt: null,
        isError: null,
      },
    ]);
    expect(out).toContain('`Write`');
    expect(out).toContain('completed 2026-04-27T10:00:01.500Z');
    expect(out).toContain('`mcp__nanoclaw__send_message`');
    expect(out).toContain('_in-flight at trigger_');
  });

  it('marks errored invocations explicitly', () => {
    const out = renderFacts(baseInputs, [
      {
        name: 'Bash',
        inputSummary: '`false`',
        completedAt: '2026-04-27T10:00:00Z',
        isError: true,
      },
    ]);
    expect(out).toContain('(errored)');
  });

  it('reports tokens used and threshold context for forensic correlation', () => {
    const out = renderFacts(baseInputs, []);
    expect(out).toContain('825,000');
    expect(out).toContain('1,000,000');
    expect(out).toContain('700,000'); // warn
    expect(out).toContain('800,000'); // nuke
  });
});

describe('summariseInput — per-family pick', () => {
  it('Write/Edit/MultiEdit shows file_path', () => {
    expect(summariseInput('Write', { file_path: '/a/b' })).toBe('`/a/b`');
    expect(summariseInput('Edit', { path: '/c/d' })).toBe('`/c/d`');
    expect(summariseInput('MultiEdit', { file_path: '/e' })).toBe('`/e`');
  });

  it('Bash truncates command to 120 chars with marker', () => {
    const long = 'echo ' + 'x'.repeat(200);
    const out = summariseInput('Bash', { command: long });
    expect(out.length).toBeLessThan(140);
    expect(out).toContain('…');
  });

  it('Bash short command renders fully', () => {
    expect(summariseInput('Bash', { command: 'ls -la' })).toBe('`ls -la`');
  });

  it('Skill renders skill name', () => {
    expect(summariseInput('Skill', { skill: 'tessl__release' })).toBe(
      'skill=`tessl__release`',
    );
  });

  it('send_message renders char count + reply_to when present', () => {
    expect(
      summariseInput('mcp__nanoclaw__send_message', {
        message: 'hi',
        reply_to: 'msg_42',
      }),
    ).toBe('2 chars reply_to=`msg_42`');
    expect(
      summariseInput('mcp__nanoclaw__send_message', { message: 'hi' }),
    ).toBe('2 chars');
  });

  it('react_to_message renders emoji + msg id', () => {
    expect(
      summariseInput('mcp__nanoclaw__react_to_message', {
        emoji: '👀',
        message_id: 'msg_99',
      }),
    ).toBe('emoji=`👀` msg=`msg_99`');
  });

  it('schedule_task renders prompt prefix', () => {
    expect(
      summariseInput('mcp__nanoclaw__schedule_task', {
        prompt: 'Run nightly summary at 9pm',
      }),
    ).toContain('Run nightly summary');
  });

  it('unknown tool falls back to key listing', () => {
    expect(summariseInput('SomeTool', { a: 1, b: 2 })).toBe('keys=[a, b]');
  });
});

describe('writeCheckpoint — rotation + write', () => {
  function commonInputs() {
    return {
      groupDir: tmpDir,
      jsonlPath: path.join(tmpDir, 'session.jsonl'),
      sessionId: 'sess_xyz',
      thresholds: computeThresholds(1_000_000),
      usedTokens: 805_000,
      groupName: 'main',
    };
  }

  it('first-ever write creates default.md without erroring on missing previous', async () => {
    fs.writeFileSync(commonInputs().jsonlPath, ''); // empty transcript
    await writeCheckpoint(commonInputs());

    const paths = checkpointPaths(tmpDir);
    expect(fs.existsSync(paths.live)).toBe(true);
    expect(fs.existsSync(paths.previous)).toBe(false);

    const body = fs.readFileSync(paths.live, 'utf-8');
    expect(body).toContain('# Session Checkpoint');
  });

  it('second write rotates default.md → previous.md atomically', async () => {
    fs.writeFileSync(commonInputs().jsonlPath, '');
    await writeCheckpoint(commonInputs());

    const paths = checkpointPaths(tmpDir);
    const firstBody = fs.readFileSync(paths.live, 'utf-8');

    // Second write — change a marker so we can tell them apart.
    await writeCheckpoint({ ...commonInputs(), usedTokens: 900_000 });

    expect(fs.existsSync(paths.previous)).toBe(true);
    const previousBody = fs.readFileSync(paths.previous, 'utf-8');
    expect(previousBody).toBe(firstBody);

    const liveBody = fs.readFileSync(paths.live, 'utf-8');
    expect(liveBody).toContain('900,000');
    expect(liveBody).not.toBe(firstBody);
  });

  it('includes mutating tool invocations from the JSONL transcript', async () => {
    const jsonl = [
      JSON.stringify({
        type: 'assistant',
        timestamp: '2026-04-27T10:00:00Z',
        message: {
          content: [
            {
              type: 'tool_use',
              id: 'tu_1',
              name: 'Write',
              input: { file_path: '/tmp/log.txt', content: 'x' },
            },
          ],
        },
      }),
      JSON.stringify({
        type: 'user',
        timestamp: '2026-04-27T10:00:01Z',
        message: {
          content: [
            { type: 'tool_result', tool_use_id: 'tu_1', is_error: false },
          ],
        },
      }),
      // A read-only call must NOT appear in the Facts list.
      JSON.stringify({
        type: 'assistant',
        timestamp: '2026-04-27T10:00:02Z',
        message: {
          content: [
            {
              type: 'tool_use',
              id: 'tu_2',
              name: 'Read',
              input: { file_path: '/etc/hosts' },
            },
          ],
        },
      }),
    ].join('\n');
    fs.writeFileSync(commonInputs().jsonlPath, jsonl);

    await writeCheckpoint(commonInputs());

    const body = fs.readFileSync(checkpointPaths(tmpDir).live, 'utf-8');
    expect(body).toContain('`Write`');
    expect(body).toContain('`/tmp/log.txt`');
    expect(body).not.toContain('`Read`');
  });
});

describe('clearCheckpoints (#127)', () => {
  // Pins the disk contract for `nuke_session({ skipReentry: true })`:
  // delete the per-group checkpoint pair and report how many were
  // actually unlinked. Idempotent — never-written groups and missing
  // `previous.md` (first-ever-write) are normal.
  it('removes both default.md and previous.md when present', () => {
    const { dir, live, previous } = checkpointPaths(tmpDir);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(live, '# Session Checkpoint (live)');
    fs.writeFileSync(previous, '# Session Checkpoint (previous)');

    const removed = clearCheckpoints(tmpDir);

    expect(removed).toBe(2);
    expect(fs.existsSync(live)).toBe(false);
    expect(fs.existsSync(previous)).toBe(false);
    // The dir itself is left in place — cheap to keep, and
    // `writeCheckpoint` re-creates it via mkdirSync(recursive: true)
    // on the next write anyway.
    expect(fs.existsSync(dir)).toBe(true);
  });

  it('returns 1 when only default.md exists (first-ever-write group)', () => {
    // `previous.md` is created on the SECOND checkpoint write (rotated
    // from the first live). A group that crossed the threshold exactly
    // once has only `default.md` on disk. skipReentry should clear it
    // and report 1.
    const { dir, live } = checkpointPaths(tmpDir);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(live, '# Session Checkpoint');

    const removed = clearCheckpoints(tmpDir);

    expect(removed).toBe(1);
    expect(fs.existsSync(live)).toBe(false);
  });

  it('returns 0 when neither file exists (group never crossed the threshold)', () => {
    // The most common skipReentry-on-a-fresh-group case: the
    // `.checkpoints/` directory may not exist at all. Helper must not
    // throw, and the count reports the truth (zero).
    const removed = clearCheckpoints(tmpDir);

    expect(removed).toBe(0);
  });

  it('returns 0 and does not throw when the .checkpoints dir is absent', () => {
    // Tighter than the previous case: explicitly assert the helper
    // tolerates a missing parent dir, since `unlinkSync` on a
    // non-existent path with a non-existent parent dir also returns
    // ENOENT and we lump that into the "nothing to clear" outcome.
    const fresh = fs.mkdtempSync(path.join(os.tmpdir(), 'no-checkpoints-'));
    try {
      const removed = clearCheckpoints(fresh);
      expect(removed).toBe(0);
    } finally {
      fs.rmSync(fresh, { recursive: true, force: true });
    }
  });

  it('is idempotent across repeated calls', () => {
    const { dir, live, previous } = checkpointPaths(tmpDir);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(live, '# live');
    fs.writeFileSync(previous, '# previous');

    expect(clearCheckpoints(tmpDir)).toBe(2);
    expect(clearCheckpoints(tmpDir)).toBe(0);
    expect(clearCheckpoints(tmpDir)).toBe(0);
  });

  it('refuses to traverse when .checkpoints/ itself is a symlink', () => {
    // Compromised container plants `.checkpoints/` as a symlink to an
    // attacker-chosen host path. realpath would resolve through the
    // symlink and the leaf paths would land outside the group folder.
    // Helper must refuse the whole operation regardless of where the
    // link points (target is irrelevant — the structural check is
    // "is it a symlink").
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'evil-target-'));
    try {
      const sentinel = path.join(outside, 'default.md');
      fs.writeFileSync(sentinel, 'sentinel');

      // Build the symlink: <tmpDir>/.checkpoints → <outside>
      fs.symlinkSync(outside, path.join(tmpDir, '.checkpoints'), 'dir');

      const removed = clearCheckpoints(tmpDir);

      expect(removed).toBe(0);
      expect(fs.existsSync(sentinel)).toBe(true);
      expect(fs.readFileSync(sentinel, 'utf8')).toBe('sentinel');
    } finally {
      fs.rmSync(outside, { recursive: true, force: true });
    }
  });

  it('unlinks a symlinked checkpoint file as a link only (target preserved)', () => {
    // Tighter: `.checkpoints/` is legit, but `default.md` inside is a
    // symlink to a sensitive host path. fs.unlinkSync removes the
    // link entry without following it, so the target stays intact.
    // Without this branch, the realpath check would refuse (the
    // realpath escapes .checkpoints/) and the link would survive on
    // disk — the operator's "give me a fresh checkpoint" intent
    // would be silently ignored.
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'symlink-target-'));
    try {
      const sentinel = path.join(outside, 'must-survive.md');
      fs.writeFileSync(sentinel, 'sentinel');

      const { dir, live } = checkpointPaths(tmpDir);
      fs.mkdirSync(dir, { recursive: true });
      fs.symlinkSync(sentinel, live);

      const removed = clearCheckpoints(tmpDir);

      // Symlink unlinked.
      expect(removed).toBe(1);
      expect(fs.existsSync(live)).toBe(false);
      // Target preserved.
      expect(fs.existsSync(sentinel)).toBe(true);
      expect(fs.readFileSync(sentinel, 'utf8')).toBe('sentinel');
    } finally {
      fs.rmSync(outside, { recursive: true, force: true });
    }
  });
});
