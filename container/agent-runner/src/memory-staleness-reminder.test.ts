import { describe, it, expect } from 'vitest';
import {
  buildStalenessReminder,
  classifyTrustedRead,
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

describe('buildStalenessReminder', () => {
  it('names the resolved path verbatim', () => {
    const reminder = buildStalenessReminder('/workspace/trusted/MEMORY.md');
    expect(reminder).toContain('/workspace/trusted/MEMORY.md');
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
