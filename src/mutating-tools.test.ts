import { describe, it, expect } from 'vitest';
import {
  classifyTool,
  ALWAYS_MUTATING_TOOLS,
  ALWAYS_READ_TOOLS,
  BASH_READ_ONLY_COMMANDS,
} from './mutating-tools.js';

// `mutating-tools` is the source of truth for the kill-auto-compaction
// `## Facts` writer's "do NOT re-execute" filter (design doc §3,
// `docs/proposals/kill-auto-compaction.md`). The test plan mirrors the
// failure modes the design calls out: false positives are noise, false
// negatives are the JCON failure mode the epic exists to prevent. The
// asymmetric cost is why every test prefers "classify as mutating" as
// the safe default and why unknown tool names test as mutating.

describe('classifyTool — always-mutating set', () => {
  it.each([
    'Write',
    'Edit',
    'MultiEdit',
    'NotebookEdit',
    'Skill',
    'mcp__nanoclaw__send_message',
    'mcp__nanoclaw__send_file',
    'mcp__nanoclaw__react_to_message',
    'mcp__nanoclaw__pin_message',
    'mcp__nanoclaw__schedule_task',
    'mcp__nanoclaw__update_task',
    'mcp__nanoclaw__register_group',
    'mcp__nanoclaw__set_trusted',
    'mcp__nanoclaw__set_trigger',
    'Task',
    'TeamCreate',
    'TeamDelete',
  ])('%s classifies as mutating', (tool) => {
    expect(classifyTool(tool)).toBe(true);
    expect(ALWAYS_MUTATING_TOOLS.has(tool)).toBe(true);
  });
});

describe('classifyTool — always-read set', () => {
  it.each([
    'Read',
    'Grep',
    'Glob',
    'WebFetch',
    'WebSearch',
    'TaskOutput',
    'TodoWrite',
    'ToolSearch',
  ])('%s classifies as read', (tool) => {
    expect(classifyTool(tool)).toBe(false);
    expect(ALWAYS_READ_TOOLS.has(tool)).toBe(true);
  });
});

describe('classifyTool — Bash with read-only allowlist', () => {
  it.each([
    ['cat /etc/hosts', false],
    ['ls -la /tmp', false],
    ['grep -r foo .', false],
    ['find . -type f', false],
    ['head -5 README.md', false],
    ['tail -f log', false],
    ['wc -l file', false],
    ['sort -u file', false],
    ['uniq -c file', false],
    ['pwd', false],
    ['echo hello', false],
    ['file /bin/sh', false],
    ['stat /tmp', false],
    ['date -u', false],
  ] as Array<[string, boolean]>)(
    'Bash command %s → mutating=%s',
    (cmd, expectedMutating) => {
      expect(classifyTool('Bash', cmd)).toBe(expectedMutating);
    },
  );

  it.each([
    'rm -rf /tmp/foo',
    'mv a b',
    'cp a b',
    'curl https://example.com -o out',
    'npm install',
    'docker run --rm hello-world',
    'tee /etc/passwd',
    'sed -i s/foo/bar/ file',
  ])('Bash mutating command %s → classified mutating', (cmd) => {
    expect(classifyTool('Bash', cmd)).toBe(true);
  });
});

describe('classifyTool — Bash git read vs write split', () => {
  it.each([
    'git status',
    'git log --oneline',
    'git diff HEAD~1',
    'git show HEAD',
    'git rev-parse HEAD',
  ])('read-only git subcommand: %s → not mutating', (cmd) => {
    expect(classifyTool('Bash', cmd)).toBe(false);
  });

  it.each([
    'git push origin main',
    'git commit -m foo',
    'git rebase -i HEAD~3',
    'git reset --hard HEAD',
    'git checkout main',
    'git branch -D oldbranch',
    'git config user.email x@y.z',
  ])('mutating git subcommand: %s → mutating', (cmd) => {
    expect(classifyTool('Bash', cmd)).toBe(true);
  });

  it('bare `git` with no subcommand classifies mutating (safe default)', () => {
    expect(classifyTool('Bash', 'git')).toBe(true);
  });
});

describe('classifyTool — defensive defaults', () => {
  it('Bash with empty/missing command → mutating', () => {
    expect(classifyTool('Bash')).toBe(true);
    expect(classifyTool('Bash', '')).toBe(true);
    expect(classifyTool('Bash', '   ')).toBe(true);
  });

  it('unknown tool name → mutating (safe default)', () => {
    expect(classifyTool('mcp__some_new_tool')).toBe(true);
    expect(classifyTool('TotallyMadeUp')).toBe(true);
  });

  it('Bash command with leading whitespace still parses argv[0]', () => {
    expect(classifyTool('Bash', '   ls -la')).toBe(false);
  });
});

describe('BASH_READ_ONLY_COMMANDS exposed as a frozen-shape set', () => {
  it('contains the v1 allowlist exactly', () => {
    const expected = [
      'cat',
      'ls',
      'grep',
      'find',
      'head',
      'tail',
      'wc',
      'sort',
      'uniq',
      'pwd',
      'echo',
      'file',
      'stat',
      'date',
    ];
    for (const cmd of expected) {
      expect(BASH_READ_ONLY_COMMANDS.has(cmd)).toBe(true);
    }
  });
});
