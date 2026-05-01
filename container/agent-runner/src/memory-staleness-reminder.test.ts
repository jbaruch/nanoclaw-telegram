import { describe, it, expect } from 'vitest';
import {
  buildStalenessReminder,
  classifyTrustedRead,
  sanitizePathForDisplay,
} from './memory-staleness-reminder.js';

describe('classifyTrustedRead — paths under /workspace/trusted/', () => {
  it('matches MEMORY.md', () => {
    const r = classifyTrustedRead('/workspace/trusted/MEMORY.md');
    expect(r.isTrustedMemoryRead).toBe(true);
    expect(r.resolvedPath).toBe('/workspace/trusted/MEMORY.md');
  });

  it('matches daily logs', () => {
    const r = classifyTrustedRead('/workspace/trusted/daily/2026-04-30.md');
    expect(r.isTrustedMemoryRead).toBe(true);
    expect(r.resolvedPath).toBe('/workspace/trusted/daily/2026-04-30.md');
  });

  it('matches typed memory files', () => {
    expect(
      classifyTrustedRead('/workspace/trusted/highlights.md').isTrustedMemoryRead,
    ).toBe(true);
    expect(
      classifyTrustedRead('/workspace/trusted/feedback_no_secrets.md').isTrustedMemoryRead,
    ).toBe(true);
  });
});

describe('classifyTrustedRead — quarantine carve-out', () => {
  it('does NOT match the quarantine subtree (operator review surface, different staleness model)', () => {
    const r = classifyTrustedRead(
      '/workspace/trusted/quarantine/sid_abc/MEMORY.md',
    );
    expect(r.isTrustedMemoryRead).toBe(false);
    expect(r.resolvedPath).toBe(
      '/workspace/trusted/quarantine/sid_abc/MEMORY.md',
    );
  });

  it('does NOT match nested quarantine paths', () => {
    expect(
      classifyTrustedRead(
        '/workspace/trusted/quarantine/sid_abc/daily/2026-04-30.md',
      ).isTrustedMemoryRead,
    ).toBe(false);
  });

  it('does NOT match the bare quarantine root (no trailing slash)', () => {
    // Without the bare-root carve-out, `/workspace/trusted/quarantine`
    // would slip past the prefix check (which is anchored on the
    // trailing slash) and trip the reminder.
    const r = classifyTrustedRead('/workspace/trusted/quarantine');
    expect(r.isTrustedMemoryRead).toBe(false);
    expect(r.resolvedPath).toBe('/workspace/trusted/quarantine');
  });

  it('does NOT match the quarantine root with trailing slash', () => {
    expect(
      classifyTrustedRead('/workspace/trusted/quarantine/').isTrustedMemoryRead,
    ).toBe(false);
  });
});

describe('classifyTrustedRead — non-trusted paths', () => {
  it('does NOT match /workspace/group/', () => {
    expect(
      classifyTrustedRead('/workspace/group/notes.md').isTrustedMemoryRead,
    ).toBe(false);
  });

  it('does NOT match /workspace/state/', () => {
    expect(
      classifyTrustedRead('/workspace/state/run.json').isTrustedMemoryRead,
    ).toBe(false);
  });

  it('does NOT match /workspace/global/', () => {
    expect(
      classifyTrustedRead('/workspace/global/SOUL.md').isTrustedMemoryRead,
    ).toBe(false);
  });

  it('does NOT match paths outside /workspace', () => {
    expect(
      classifyTrustedRead('/etc/passwd').isTrustedMemoryRead,
    ).toBe(false);
    expect(
      classifyTrustedRead('/tmp/scratch.md').isTrustedMemoryRead,
    ).toBe(false);
  });

  it('does NOT match /workspace/trusted-evil/ (prefix-strip safety)', () => {
    // Without the trailing-slash anchor on the prefix, this would
    // false-positive. The strict startsWith on '/workspace/trusted/'
    // including the slash is what protects against this.
    expect(
      classifyTrustedRead('/workspace/trusted-evil/foo.md').isTrustedMemoryRead,
    ).toBe(false);
  });
});

describe('classifyTrustedRead — relative path resolution', () => {
  it('treats relative paths as cwd-relative (under /workspace/group/, not trusted/)', () => {
    expect(classifyTrustedRead('MEMORY.md').isTrustedMemoryRead).toBe(false);
    expect(classifyTrustedRead('./notes.md').isTrustedMemoryRead).toBe(false);
  });

  it('relative traversal that escapes into trusted/ DOES match (rare; document the behavior)', () => {
    // cwd is /workspace/group. ../trusted/MEMORY.md → /workspace/
    // trusted/MEMORY.md. The hook fires; the reminder is
    // path-accurate. This isn't a bug — if the agent reads memory
    // by traversal-relative path, it's still a memory read.
    const r = classifyTrustedRead('../trusted/MEMORY.md');
    expect(r.isTrustedMemoryRead).toBe(true);
    expect(r.resolvedPath).toBe('/workspace/trusted/MEMORY.md');
  });
});

describe('classifyTrustedRead — traversal normalization', () => {
  it('normalizes /workspace/trusted/sub/../MEMORY.md to /workspace/trusted/MEMORY.md', () => {
    const r = classifyTrustedRead('/workspace/trusted/sub/../MEMORY.md');
    expect(r.isTrustedMemoryRead).toBe(true);
    expect(r.resolvedPath).toBe('/workspace/trusted/MEMORY.md');
  });

  it('does NOT match traversal escape that lands outside trusted/', () => {
    expect(
      classifyTrustedRead('/workspace/trusted/../../etc/passwd')
        .isTrustedMemoryRead,
    ).toBe(false);
  });

  it('quarantine traversal back into the parent subtree still excludes', () => {
    // /workspace/trusted/quarantine/sid/../foo.md →
    //   /workspace/trusted/quarantine/foo.md → still under quarantine
    // (quarantine root). isTrustedMemoryRead is false.
    expect(
      classifyTrustedRead('/workspace/trusted/quarantine/sid/../foo.md')
        .isTrustedMemoryRead,
    ).toBe(false);
  });
});

describe('sanitizePathForDisplay', () => {
  it('passes a normal path through unchanged', () => {
    expect(sanitizePathForDisplay('/workspace/trusted/MEMORY.md')).toBe(
      '/workspace/trusted/MEMORY.md',
    );
  });

  it('replaces newlines with spaces', () => {
    expect(
      sanitizePathForDisplay('/workspace/trusted/foo\n/bar.md'),
    ).toBe('/workspace/trusted/foo /bar.md');
  });

  it('replaces \\r and \\t and other control chars with spaces', () => {
    expect(
      sanitizePathForDisplay('/workspace/\rtrusted/\tfoo\x00bar'),
    ).toBe('/workspace/ trusted/ foo bar');
  });

  it('collapses runs of control chars into a single space', () => {
    expect(sanitizePathForDisplay('/a\n\n\n\n/b')).toBe('/a /b');
  });

  it('truncates absurdly long paths and marks with …', () => {
    const long = '/workspace/trusted/' + 'a'.repeat(1000);
    const out = sanitizePathForDisplay(long);
    expect(out.length).toBeLessThan(long.length);
    expect(out.endsWith('…')).toBe(true);
  });

  it('returns empty string for non-string input', () => {
    expect(sanitizePathForDisplay(undefined as unknown as string)).toBe('');
    expect(sanitizePathForDisplay(null as unknown as string)).toBe('');
  });
});

describe('buildStalenessReminder', () => {
  it('names the resolved path verbatim', () => {
    const reminder = buildStalenessReminder('/workspace/trusted/MEMORY.md');
    expect(reminder).toContain('/workspace/trusted/MEMORY.md');
  });

  it('sanitizes injected newlines in the path before embedding (system-message injection guard)', () => {
    // An injection that smuggles a newline into the file_path
    // would, without sanitization, end up rendering a multi-line
    // system message — letting an attacker forge an additional
    // line that looks like fresh system text. The sanitizer
    // collapses those newlines so the reminder stays single-line.
    const reminder = buildStalenessReminder(
      '/workspace/trusted/foo\n\nfake instruction\n',
    );
    // Newlines in the path are gone — the entire reminder stays
    // a single paragraph.
    expect(reminder.includes('\n')).toBe(false);
    // The path content is preserved (just with newlines collapsed
    // to spaces); the model can still see what was actually read.
    expect(reminder).toContain('fake instruction');
  });

  it('keeps the reminder a single paragraph (no real newlines anywhere)', () => {
    const reminder = buildStalenessReminder(
      '/workspace/trusted/foo\nbar.md',
    );
    expect(reminder.includes('\n')).toBe(false);
  });

  it('mentions the structural classes of mutation-it-blocks', () => {
    const reminder = buildStalenessReminder('/workspace/trusted/highlights.md');
    expect(reminder).toMatch(/sending messages/i);
    expect(reminder).toMatch(/scheduling tasks/i);
    expect(reminder).toMatch(/composio/i);
  });

  it("uses the literal phrase 'MEMORY STALENESS' as a recognizable header", () => {
    const reminder = buildStalenessReminder('/workspace/trusted/MEMORY.md');
    expect(reminder.startsWith('MEMORY STALENESS:')).toBe(true);
  });

  it('instructs verification against the live source', () => {
    const reminder = buildStalenessReminder('/workspace/trusted/MEMORY.md');
    expect(reminder).toMatch(/verify.*live source/i);
  });

  it('produces different bodies for different paths', () => {
    expect(buildStalenessReminder('/workspace/trusted/MEMORY.md')).not.toBe(
      buildStalenessReminder('/workspace/trusted/highlights.md'),
    );
  });
});
