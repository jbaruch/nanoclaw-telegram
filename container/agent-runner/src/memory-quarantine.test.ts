import { describe, it, expect } from 'vitest';
import {
  createQuarantineFlagState,
  decideMemoryWrite,
  promptCarriesUntrustedInput,
  quarantinePathFor,
  resolveTargetPath,
  targetsTrustedMemory,
} from './memory-quarantine.js';

describe('targetsTrustedMemory', () => {
  it('matches absolute paths under /workspace/trusted/', () => {
    expect(targetsTrustedMemory('/workspace/trusted/MEMORY.md')).toBe(true);
    expect(targetsTrustedMemory('/workspace/trusted/daily/2026-04-30.md')).toBe(
      true,
    );
    expect(targetsTrustedMemory('/workspace/trusted/highlights.md')).toBe(true);
  });

  it('does NOT match the quarantine subtree', () => {
    expect(
      targetsTrustedMemory('/workspace/trusted/quarantine/sid1/MEMORY.md'),
    ).toBe(false);
  });

  it('does NOT match other workspace mounts', () => {
    expect(targetsTrustedMemory('/workspace/group/notes.md')).toBe(false);
    expect(targetsTrustedMemory('/workspace/state/run.json')).toBe(false);
    expect(targetsTrustedMemory('/workspace/global/SOUL.md')).toBe(false);
  });

  it('does NOT match paths outside /workspace', () => {
    expect(targetsTrustedMemory('/etc/passwd')).toBe(false);
    expect(targetsTrustedMemory('/tmp/scratch.md')).toBe(false);
  });

  it('treats relative paths as cwd-relative (not under trusted/)', () => {
    // Relative input resolves against /workspace/group, never under
    // /workspace/trusted/.
    expect(targetsTrustedMemory('MEMORY.md')).toBe(false);
    expect(targetsTrustedMemory('./notes.md')).toBe(false);
  });

  it('rejects traversal escapes — /workspace/trusted/../../etc/passwd is NOT under trusted/', () => {
    expect(targetsTrustedMemory('/workspace/trusted/../../etc/passwd')).toBe(
      false,
    );
  });

  it('matches the bare trusted root with trailing slash; rejects without', () => {
    // The strict startsWith match anchors on '/workspace/trusted/'
    // including the trailing slash. The trailing form thus matches
    // (a Write to a directory would fail at OS level anyway, so the
    // gate behaviour is harmless either way); the no-slash form
    // rejects deliberately because '/workspace/trusted-evil/' would
    // otherwise match a prefix scan that omitted the slash.
    expect(targetsTrustedMemory('/workspace/trusted/')).toBe(true);
    expect(targetsTrustedMemory('/workspace/trusted')).toBe(false);
  });
});

describe('resolveTargetPath', () => {
  it('passes absolute paths through after normalization', () => {
    expect(resolveTargetPath('/workspace/trusted/x.md')).toBe(
      '/workspace/trusted/x.md',
    );
    expect(resolveTargetPath('/workspace/trusted/sub/../x.md')).toBe(
      '/workspace/trusted/x.md',
    );
  });

  it('resolves relative paths against /workspace/group', () => {
    expect(resolveTargetPath('notes.md')).toBe('/workspace/group/notes.md');
    expect(resolveTargetPath('./notes.md')).toBe('/workspace/group/notes.md');
    expect(resolveTargetPath('sub/foo.md')).toBe('/workspace/group/sub/foo.md');
  });
});

describe('quarantinePathFor', () => {
  it('maps MEMORY.md to quarantine/<sid>/MEMORY.md', () => {
    expect(quarantinePathFor('/workspace/trusted/MEMORY.md', 'sid_abc')).toBe(
      '/workspace/trusted/quarantine/sid_abc/MEMORY.md',
    );
  });

  it('preserves nested structure (daily/2026-04-30.md)', () => {
    expect(
      quarantinePathFor('/workspace/trusted/daily/2026-04-30.md', 'sid_abc'),
    ).toBe('/workspace/trusted/quarantine/sid_abc/daily/2026-04-30.md');
  });

  it('sanitizes sessionId — path-traversal characters collapse to underscores', () => {
    // `../sneaky/sid` has four non-allowed chars (`.`, `.`, `/`, `/`)
    // → four underscores; alphanumerics in the rest are preserved.
    expect(
      quarantinePathFor('/workspace/trusted/MEMORY.md', '../sneaky/sid'),
    ).toBe('/workspace/trusted/quarantine/___sneaky_sid/MEMORY.md');
    expect(quarantinePathFor('/workspace/trusted/MEMORY.md', 'a/b')).toBe(
      '/workspace/trusted/quarantine/a_b/MEMORY.md',
    );
  });

  it('keeps alphanumerics, hyphens, underscores in sessionId', () => {
    expect(quarantinePathFor('/workspace/trusted/x.md', 'sid-123_abcDEF')).toBe(
      '/workspace/trusted/quarantine/sid-123_abcDEF/x.md',
    );
  });
});

describe('decideMemoryWrite — happy path (flag clear)', () => {
  it('allows Write to trusted/ when flag is clear', () => {
    const flag = createQuarantineFlagState();
    const decision = decideMemoryWrite({
      toolName: 'Write',
      filePath: '/workspace/trusted/MEMORY.md',
      flag,
      sessionId: 'sid1',
    });
    expect(decision.kind).toBe('allow');
  });

  it('allows Edit to trusted/ when flag is clear', () => {
    const flag = createQuarantineFlagState();
    const decision = decideMemoryWrite({
      toolName: 'Edit',
      filePath: '/workspace/trusted/MEMORY.md',
      flag,
      sessionId: 'sid1',
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('decideMemoryWrite — flag set + trusted target', () => {
  it('redirects Write on /workspace/trusted/MEMORY.md to quarantine', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Write',
      filePath: '/workspace/trusted/MEMORY.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('redirect');
    if (decision.kind === 'redirect') {
      expect(decision.originalPath).toBe('/workspace/trusted/MEMORY.md');
      expect(decision.quarantinedTo).toBe(
        '/workspace/trusted/quarantine/sid_abc/MEMORY.md',
      );
    }
  });

  it('redirects Write on /workspace/trusted/daily/2026-04-30.md to nested quarantine', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Write',
      filePath: '/workspace/trusted/daily/2026-04-30.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('redirect');
    if (decision.kind === 'redirect') {
      expect(decision.quarantinedTo).toBe(
        '/workspace/trusted/quarantine/sid_abc/daily/2026-04-30.md',
      );
    }
  });

  it("denies Edit (surgical-edit semantics don't survive redirect)", () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Edit',
      filePath: '/workspace/trusted/MEMORY.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.reason).toContain('memory_quarantine');
      expect(decision.reason).toContain('Edit');
      expect(decision.reason).toContain(
        '/workspace/trusted/quarantine/sid_abc/MEMORY.md',
      );
    }
  });
});

describe('decideMemoryWrite — quarantine subtree carve-out', () => {
  it('allows Write to quarantine/ even with flag set', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Write',
      filePath: '/workspace/trusted/quarantine/sid_x/MEMORY.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('allow');
  });

  it('allows Edit to quarantine/ even with flag set', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Edit',
      filePath: '/workspace/trusted/quarantine/sid_x/MEMORY.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('decideMemoryWrite — non-trusted targets', () => {
  it('allows Write to /workspace/group/ even with flag set', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Write',
      filePath: '/workspace/group/notes.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('allow');
  });

  it('allows Write to /workspace/state/ even with flag set', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Write',
      filePath: '/workspace/state/run.json',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('allow');
  });

  it('allows Write to relative paths (resolve under /workspace/group)', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Write',
      filePath: 'notes.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('decideMemoryWrite — non-Write/Edit tools', () => {
  it('passes Read through unchanged', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Read',
      filePath: '/workspace/trusted/MEMORY.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('allow');
  });

  it('passes Bash through unchanged (the gate is only on Write/Edit)', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Bash',
      filePath: '/workspace/trusted/MEMORY.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('decideMemoryWrite — traversal guards', () => {
  it('redirects Write on a path normalized to trusted/ even if the input contains ..', () => {
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Write',
      filePath: '/workspace/trusted/sub/../MEMORY.md',
      flag,
      sessionId: 'sid_abc',
    });
    expect(decision.kind).toBe('redirect');
    if (decision.kind === 'redirect') {
      expect(decision.originalPath).toBe('/workspace/trusted/MEMORY.md');
    }
  });

  it('does NOT redirect a traversal escape that lands outside trusted/', () => {
    // /workspace/trusted/../../etc/passwd → /etc/passwd (NOT trusted/)
    const flag = { processedExternalContent: true };
    const decision = decideMemoryWrite({
      toolName: 'Write',
      filePath: '/workspace/trusted/../../etc/passwd',
      flag,
      sessionId: 'sid_abc',
    });
    // The hook lets it through here; #322 / bash-safety-net
    // handle the unrelated /etc/passwd write attempt.
    expect(decision.kind).toBe('allow');
  });
});

describe('promptCarriesUntrustedInput', () => {
  it('matches a cross-group wrap', () => {
    expect(
      promptCarriesUntrustedInput(
        '<untrusted-input source="cross-group:tg-123">hi</untrusted-input>',
      ),
    ).toBe(true);
  });

  it('matches an untrusted-container wrap (PR 1 retrofit)', () => {
    expect(
      promptCarriesUntrustedInput(
        'Prefix\n<untrusted-input source="untrusted-container:groupA">body</untrusted-input>\nsuffix',
      ),
    ).toBe(true);
  });

  it('matches multiple wraps (one is enough)', () => {
    expect(
      promptCarriesUntrustedInput(
        '<untrusted-input source="cross-group:a">x</untrusted-input>\n<untrusted-input source="cross-group:b">y</untrusted-input>',
      ),
    ).toBe(true);
  });

  it('returns false for plain prompts', () => {
    expect(promptCarriesUntrustedInput('plain user prompt')).toBe(false);
    expect(promptCarriesUntrustedInput('')).toBe(false);
  });

  it('returns false for tags without a source attr (non-marker mention of the tag name)', () => {
    expect(
      promptCarriesUntrustedInput(
        'Discussion of the <untrusted-input> wrap concept',
      ),
    ).toBe(false);
  });

  it('returns false for non-string input', () => {
    expect(promptCarriesUntrustedInput(undefined as unknown as string)).toBe(
      false,
    );
  });
});

describe('createQuarantineFlagState', () => {
  it('starts with the flag clear', () => {
    expect(createQuarantineFlagState()).toEqual({
      processedExternalContent: false,
    });
  });

  it('returns a fresh object each call (no shared mutation)', () => {
    const a = createQuarantineFlagState();
    const b = createQuarantineFlagState();
    a.processedExternalContent = true;
    expect(b.processedExternalContent).toBe(false);
  });
});
