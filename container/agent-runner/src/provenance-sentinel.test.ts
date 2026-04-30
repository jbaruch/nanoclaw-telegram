import { describe, it, expect } from 'vitest';
import {
  formatSentinel,
  inferSentinelSource,
  isExternalPath,
  SENTINEL_PREFIX,
} from './provenance-sentinel.js';

describe('isExternalPath', () => {
  it('treats workspace mounts as internal', () => {
    expect(isExternalPath('/workspace/group/notes.md')).toBe(false);
    expect(isExternalPath('/workspace/state/run.json')).toBe(false);
    expect(isExternalPath('/workspace/trusted/MEMORY.md')).toBe(false);
    expect(isExternalPath('/workspace/global/SOUL.md')).toBe(false);
    expect(isExternalPath('/workspace/store/messages.db')).toBe(false);
    expect(isExternalPath('/workspace/ipc/output')).toBe(false);
  });

  it('treats other absolute paths as external', () => {
    expect(isExternalPath('/etc/passwd')).toBe(true);
    expect(isExternalPath('/var/log/syslog')).toBe(true);
    expect(isExternalPath('/home/node/.ssh/id_rsa')).toBe(true);
    expect(isExternalPath('/mnt/nas/userdata/foo.txt')).toBe(true);
    expect(isExternalPath('/tmp/scratch.txt')).toBe(true);
  });

  it('treats /workspace paths NOT under a standard mount as external', () => {
    expect(isExternalPath('/workspace/secret/foo')).toBe(true);
    expect(isExternalPath('/workspace')).toBe(true);
    expect(isExternalPath('/workspace/')).toBe(true);
  });

  it('treats relative paths as internal (cwd-relative inside workspace)', () => {
    expect(isExternalPath('notes.md')).toBe(false);
    expect(isExternalPath('./CLAUDE.md')).toBe(false);
    expect(isExternalPath('../sibling/foo')).toBe(false);
  });
});

describe('inferSentinelSource — WebFetch', () => {
  it('emits web: source from the url input', () => {
    expect(
      inferSentinelSource('WebFetch', { url: 'https://example.com/page' }),
    ).toEqual({ prefix: 'web', value: 'https://example.com/page' });
  });

  it('returns null when url is missing', () => {
    expect(inferSentinelSource('WebFetch', {})).toBeNull();
  });

  it('returns null when url is empty string', () => {
    expect(inferSentinelSource('WebFetch', { url: '' })).toBeNull();
  });

  it('returns null when url is non-string', () => {
    expect(inferSentinelSource('WebFetch', { url: 42 })).toBeNull();
  });
});

describe('inferSentinelSource — Read', () => {
  it('emits file: source for external absolute paths', () => {
    expect(
      inferSentinelSource('Read', { file_path: '/etc/hosts' }),
    ).toEqual({ prefix: 'file', value: '/etc/hosts' });
    expect(
      inferSentinelSource('Read', { file_path: '/mnt/nas/data.csv' }),
    ).toEqual({ prefix: 'file', value: '/mnt/nas/data.csv' });
  });

  it('returns null for internal workspace paths', () => {
    expect(
      inferSentinelSource('Read', { file_path: '/workspace/group/notes.md' }),
    ).toBeNull();
    expect(
      inferSentinelSource('Read', { file_path: '/workspace/trusted/x' }),
    ).toBeNull();
  });

  it('returns null for relative paths', () => {
    expect(inferSentinelSource('Read', { file_path: 'notes.md' })).toBeNull();
    expect(
      inferSentinelSource('Read', { file_path: './sub/foo.txt' }),
    ).toBeNull();
  });

  it('returns null when file_path missing or non-string', () => {
    expect(inferSentinelSource('Read', {})).toBeNull();
    expect(inferSentinelSource('Read', { file_path: '' })).toBeNull();
    expect(inferSentinelSource('Read', { file_path: 42 })).toBeNull();
  });
});

describe('inferSentinelSource — Bash agent-browser', () => {
  it('extracts url from agent-browser open <url>', () => {
    expect(
      inferSentinelSource('Bash', {
        command: 'agent-browser open https://example.com',
      }),
    ).toEqual({ prefix: 'agent-browser', value: 'https://example.com' });
  });

  it('extracts url from quoted form', () => {
    expect(
      inferSentinelSource('Bash', {
        command: 'agent-browser open "https://example.com/path?a=1"',
      }),
    ).toEqual({
      prefix: 'agent-browser',
      value: 'https://example.com/path?a=1',
    });
  });

  it('uses :active for non-open agent-browser invocations', () => {
    expect(
      inferSentinelSource('Bash', {
        command: 'agent-browser snapshot -i',
      }),
    ).toEqual({ prefix: 'agent-browser', value: 'active' });
    expect(
      inferSentinelSource('Bash', { command: 'agent-browser click @e1' }),
    ).toEqual({ prefix: 'agent-browser', value: 'active' });
    expect(
      inferSentinelSource('Bash', { command: 'agent-browser back' }),
    ).toEqual({ prefix: 'agent-browser', value: 'active' });
  });

  it('handles leading whitespace', () => {
    expect(
      inferSentinelSource('Bash', {
        command: '   agent-browser open https://x.io',
      }),
    ).toEqual({ prefix: 'agent-browser', value: 'https://x.io' });
  });

  it('returns null for non-agent-browser Bash commands', () => {
    expect(
      inferSentinelSource('Bash', { command: 'cat /workspace/group/foo' }),
    ).toBeNull();
    expect(
      inferSentinelSource('Bash', { command: 'curl https://attacker.com' }),
    ).toBeNull();
    expect(
      inferSentinelSource('Bash', { command: 'npm install' }),
    ).toBeNull();
  });

  it('does not match agent-browser appearing later in a chained command', () => {
    expect(
      inferSentinelSource('Bash', {
        command: 'cd /tmp && agent-browser open https://x.io',
      }),
    ).toBeNull();
    expect(
      inferSentinelSource('Bash', {
        command: 'which agent-browser',
      }),
    ).toBeNull();
  });

  it('returns null when command missing or empty', () => {
    expect(inferSentinelSource('Bash', {})).toBeNull();
    expect(inferSentinelSource('Bash', { command: '' })).toBeNull();
  });
});

describe('inferSentinelSource — other tools', () => {
  it('returns null for tools with no sentinel mapping', () => {
    expect(inferSentinelSource('Write', { file_path: '/etc/passwd' })).toBeNull();
    expect(inferSentinelSource('Edit', {})).toBeNull();
    expect(inferSentinelSource('Glob', { pattern: '**' })).toBeNull();
    expect(inferSentinelSource('Grep', { pattern: 'foo' })).toBeNull();
    expect(inferSentinelSource('mcp__composio__gmail_fetch_emails', {})).toBeNull();
  });

  it('returns null for empty / non-object input', () => {
    expect(inferSentinelSource('WebFetch', null)).toBeNull();
    expect(inferSentinelSource('Read', undefined as unknown as object)).toBeNull();
    expect(inferSentinelSource('Bash', 'string')).toBeNull();
  });

  it('returns null for empty tool name', () => {
    expect(inferSentinelSource('', { url: 'https://x' })).toBeNull();
  });
});

describe('formatSentinel', () => {
  it('produces the canonical PROVENANCE_MARKER line', () => {
    expect(
      formatSentinel(
        { prefix: 'web', value: 'https://example.com' },
        'toolu_abc123',
      ),
    ).toBe(
      'PROVENANCE_MARKER: source="web:https://example.com" tool_use_id="toolu_abc123"',
    );
  });

  it('uses the SENTINEL_PREFIX constant', () => {
    const result = formatSentinel(
      { prefix: 'file', value: '/etc/hosts' },
      'tu_1',
    );
    expect(result.startsWith(SENTINEL_PREFIX)).toBe(true);
  });

  it('escapes & < > " in source value', () => {
    expect(
      formatSentinel(
        { prefix: 'web', value: 'https://x.io/?a=1&b=2' },
        'tu_1',
      ),
    ).toBe(
      'PROVENANCE_MARKER: source="web:https://x.io/?a=1&amp;b=2" tool_use_id="tu_1"',
    );
    expect(
      formatSentinel({ prefix: 'gmail', value: '<msg>' }, 'tu_1'),
    ).toBe(
      'PROVENANCE_MARKER: source="gmail:&lt;msg&gt;" tool_use_id="tu_1"',
    );
  });

  it('collapses newlines in inputs', () => {
    expect(
      formatSentinel(
        { prefix: 'file', value: '/a\n/b' },
        'tu\n_1',
      ),
    ).toBe(
      'PROVENANCE_MARKER: source="file:/a /b" tool_use_id="tu _1"',
    );
  });

  it('escapes & first to avoid double-encoding entity ampersands', () => {
    expect(
      formatSentinel({ prefix: 'web', value: 'a"&b' }, 'tu_1'),
    ).toBe(
      'PROVENANCE_MARKER: source="web:a&quot;&amp;b" tool_use_id="tu_1"',
    );
  });
});
