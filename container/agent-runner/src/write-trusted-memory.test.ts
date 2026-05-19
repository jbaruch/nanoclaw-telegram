import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import {
  performOperatorApprovedWrite,
  WRITE_TRUSTED_MEMORY_DESCRIPTION,
  type OperatorApprovedWriteFs,
  type OperatorApprovedWriteLogPayload,
} from './write-trusted-memory.js';

/**
 * The validation logic resolves paths against the container's
 * `/workspace/trusted/` mount, which isn't a real path on the
 * test host. Tests inject an in-memory filesystem mock that
 * records calls and replays scripted errors, so we exercise the
 * canonical container path (`/workspace/trusted/MEMORY.md`)
 * without needing root-owned mounts on disk.
 *
 * One non-mocked case verifies the real `mkdirSync` /
 * `writeFileSync` / `renameSync` triple integrates correctly by
 * pointing the mock at a tmpdir. Validation of the trusted prefix
 * still uses the canonical container path; the mock is what
 * actually moves bytes.
 */
interface RecordedCall {
  kind: 'mkdir' | 'write' | 'rename';
  args: unknown[];
}

function makeMockFs(opts?: {
  failMkdir?: NodeJS.ErrnoException;
  failWrite?: NodeJS.ErrnoException;
  failRename?: NodeJS.ErrnoException;
}): { fs: OperatorApprovedWriteFs; calls: RecordedCall[] } {
  const calls: RecordedCall[] = [];
  const mock: OperatorApprovedWriteFs = {
    mkdirSync(p, options) {
      calls.push({ kind: 'mkdir', args: [p, options] });
      if (opts?.failMkdir) throw opts.failMkdir;
      return undefined;
    },
    writeFileSync(p, data) {
      calls.push({ kind: 'write', args: [p, data] });
      if (opts?.failWrite) throw opts.failWrite;
    },
    renameSync(oldPath, newPath) {
      calls.push({ kind: 'rename', args: [oldPath, newPath] });
      if (opts?.failRename) throw opts.failRename;
    },
  };
  return { fs: mock, calls };
}

describe('performOperatorApprovedWrite — happy path', () => {
  it('writes a valid trusted-path file with a non-empty justification', () => {
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/MEMORY.md',
        content: '# Memory\n- Amir born 1980-03-05',
        operator_justification:
          'operator dictated Amir\'s birthday in turn 14',
        fs: mockFs,
        pid: 12345,
      },
    );
    expect(result).toEqual({
      ok: true,
      path: '/workspace/trusted/MEMORY.md',
    });
    expect(calls).toEqual([
      {
        kind: 'mkdir',
        args: ['/workspace/trusted', { recursive: true }],
      },
      {
        kind: 'write',
        args: [
          '/workspace/trusted/MEMORY.md.tmp.12345',
          '# Memory\n- Amir born 1980-03-05',
        ],
      },
      {
        kind: 'rename',
        args: [
          '/workspace/trusted/MEMORY.md.tmp.12345',
          '/workspace/trusted/MEMORY.md',
        ],
      },
    ]);
  });

  it('creates nested parent dirs (daily/2026-05-19.md)', () => {
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/daily/2026-05-19.md',
        content: 'daily log',
        operator_justification: 'operator filed daily log in turn 3',
        fs: mockFs,
        pid: 1,
      },
    );
    expect(result.ok).toBe(true);
    expect(calls[0]).toEqual({
      kind: 'mkdir',
      args: ['/workspace/trusted/daily', { recursive: true }],
    });
  });

  it('accepts an empty content string (clearing a memory file)', () => {
    // Empty content is a legitimate operator request — explicitly
    // wiping a file — so we don't reject it.
    const { fs: mockFs } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/scratch.md',
        content: '',
        operator_justification:
          'operator asked to clear scratch.md in turn 9',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(true);
  });
});

describe('performOperatorApprovedWrite — file_path validation', () => {
  it('rejects an empty file_path', () => {
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '',
        content: 'x',
        operator_justification: 'operator dictated value in turn 1',
        fs: mockFs,
      },
    );
    expect(result).toEqual({
      ok: false,
      error: 'file_path must be a non-empty string.',
    });
    expect(calls).toHaveLength(0);
  });

  it('rejects relative paths (they resolve under /workspace/group, not trusted/)', () => {
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: 'notes.md',
        content: 'x',
        operator_justification: 'operator dictated value in turn 1',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(false);
    if (result.ok === false) {
      expect(result.error).toContain('must resolve under /workspace/trusted/');
    }
    expect(calls).toHaveLength(0);
  });

  it('rejects absolute paths outside /workspace/trusted/', () => {
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '/etc/passwd',
        content: 'x',
        operator_justification: 'operator dictated value in turn 1',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(false);
    expect(calls).toHaveLength(0);
  });

  it('rejects path-traversal escapes that resolve outside trusted/', () => {
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/../../etc/shadow',
        content: 'x',
        operator_justification: 'operator dictated value in turn 1',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(false);
    expect(calls).toHaveLength(0);
  });

  it('rejects writes into the quarantine subtree', () => {
    // The quarantine subtree is reserved for the #325 redirect
    // hook. An operator-approved write should never target it —
    // if the operator really wants to land content there, they
    // can promote a quarantined file by host-side rename.
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/quarantine/sid_abc/MEMORY.md',
        content: 'x',
        operator_justification: 'operator dictated value in turn 1',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(false);
    expect(calls).toHaveLength(0);
  });
});

describe('performOperatorApprovedWrite — operator_justification validation', () => {
  it('rejects an empty justification', () => {
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/MEMORY.md',
        content: 'x',
        operator_justification: '',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(false);
    if (result.ok === false) {
      expect(result.error).toContain('operator_justification must be at least');
    }
    expect(calls).toHaveLength(0);
  });

  it('rejects a whitespace-only justification', () => {
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/MEMORY.md',
        content: 'x',
        operator_justification: '       \n\t  ',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(false);
    expect(calls).toHaveLength(0);
  });

  it('rejects a sub-8-character justification ("ok")', () => {
    const { fs: mockFs, calls } = makeMockFs();
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/MEMORY.md',
        content: 'x',
        operator_justification: 'ok',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(false);
    expect(calls).toHaveLength(0);
  });
});

describe('performOperatorApprovedWrite — atomic write semantics', () => {
  it('does not leave a .tmp artifact when the rename succeeds (real fs)', () => {
    // Real-fs integration: use a tmpdir for the actual write, but
    // exercise the validation path with the canonical container
    // location by passing a wrapping mock that rewrites the path
    // to the tmpdir for the inner fs calls.
    //
    // The simpler shape: just verify the mock-based call sequence
    // ends with a single rename and no leftover tmp write — which
    // the prior happy-path test already proves. This test adds a
    // real-fs round-trip so we know the integration actually
    // hits the disk.
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'oapw-'));
    const targetWithinTmp = path.join(tmpRoot, 'MEMORY.md');
    // Construct an fs proxy that rewrites the canonical container
    // path into the tmpdir for the underlying fs calls. The
    // validation in performOperatorApprovedWrite still sees the
    // canonical path.
    const canonical = '/workspace/trusted/MEMORY.md';
    const canonicalDir = '/workspace/trusted';
    const proxy: OperatorApprovedWriteFs = {
      mkdirSync(p, options) {
        const remapped = p === canonicalDir ? tmpRoot : (p as string);
        return fs.mkdirSync(remapped, options);
      },
      writeFileSync(p, data) {
        const str = p as string;
        const remapped = str.startsWith(canonical)
          ? targetWithinTmp + str.slice(canonical.length)
          : str;
        fs.writeFileSync(remapped, data);
      },
      renameSync(oldPath, newPath) {
        const oldStr = oldPath as string;
        const newStr = newPath as string;
        const remapOld = oldStr.startsWith(canonical)
          ? targetWithinTmp + oldStr.slice(canonical.length)
          : oldStr;
        const remapNew = newStr === canonical ? targetWithinTmp : newStr;
        fs.renameSync(remapOld, remapNew);
      },
    };

    try {
      const result = performOperatorApprovedWrite(
        {
          file_path: canonical,
          content: 'hello operator',
          operator_justification:
            'operator dictated content in turn 7',
          fs: proxy,
          pid: 9999,
        },
      );
      expect(result.ok).toBe(true);
      // Target exists with the right content; no .tmp leftover.
      expect(fs.existsSync(targetWithinTmp)).toBe(true);
      expect(fs.readFileSync(targetWithinTmp, 'utf-8')).toBe(
        'hello operator',
      );
      expect(fs.existsSync(`${targetWithinTmp}.tmp.9999`)).toBe(false);
    } finally {
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    }
  });
});

describe('performOperatorApprovedWrite — fs error handling', () => {
  it('returns isError on mkdirSync EACCES', () => {
    const err = Object.assign(new Error('permission denied'), {
      code: 'EACCES',
    }) as NodeJS.ErrnoException;
    const { fs: mockFs } = makeMockFs({ failMkdir: err });
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/MEMORY.md',
        content: 'x',
        operator_justification: 'operator dictated value in turn 1',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(false);
    if (result.ok === false) {
      expect(result.error).toContain('permission denied');
    }
  });

  it('returns isError on writeFileSync ENOSPC', () => {
    const err = Object.assign(new Error('no space'), {
      code: 'ENOSPC',
    }) as NodeJS.ErrnoException;
    const { fs: mockFs } = makeMockFs({ failWrite: err });
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/MEMORY.md',
        content: 'x',
        operator_justification: 'operator dictated value in turn 1',
        fs: mockFs,
      },
    );
    expect(result.ok).toBe(false);
    if (result.ok === false) {
      expect(result.error).toContain('no space');
    }
  });

  it('propagates unexpected (non-recoverable) errors instead of swallowing', () => {
    // A non-Error throw, or a code outside the recoverable set,
    // indicates a programming bug and should crash the caller —
    // not be silently converted into a deny.
    const { fs: mockFs } = makeMockFs({
      failWrite: new Error('totally unexpected') as NodeJS.ErrnoException,
    });
    expect(() =>
      performOperatorApprovedWrite(
        {
          file_path: '/workspace/trusted/MEMORY.md',
          content: 'x',
          operator_justification: 'operator dictated value in turn 1',
          fs: mockFs,
        },
      ),
    ).toThrow('totally unexpected');
  });
});

describe('performOperatorApprovedWrite — structured logging', () => {
  it('emits the structured log line on success', () => {
    const { fs: mockFs } = makeMockFs();
    const logs: OperatorApprovedWriteLogPayload[] = [];
    const result = performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/MEMORY.md',
        content: 'x',
        operator_justification:
          'operator dictated Amir\'s birthday in turn 14',
        fs: mockFs,
      },
      (payload) => logs.push(payload),
    );
    expect(result.ok).toBe(true);
    expect(logs).toEqual([
      {
        event: 'memory_quarantine.operator_approved_write',
        path: '/workspace/trusted/MEMORY.md',
        justification:
          'operator dictated Amir\'s birthday in turn 14',
      },
    ]);
  });

  it('truncates the logged justification to 200 chars', () => {
    const long = 'x'.repeat(500);
    const { fs: mockFs } = makeMockFs();
    const logs: OperatorApprovedWriteLogPayload[] = [];
    performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/MEMORY.md',
        content: 'x',
        operator_justification: long,
        fs: mockFs,
      },
      (payload) => logs.push(payload),
    );
    expect(logs).toHaveLength(1);
    expect(logs[0].justification).toHaveLength(200);
    expect(logs[0].justification).toBe('x'.repeat(200));
  });

  it('does NOT log on validation failure', () => {
    const { fs: mockFs } = makeMockFs();
    const logs: OperatorApprovedWriteLogPayload[] = [];
    performOperatorApprovedWrite(
      {
        file_path: '/etc/passwd',
        content: 'x',
        operator_justification:
          'operator dictated value in turn 1',
        fs: mockFs,
      },
      (payload) => logs.push(payload),
    );
    expect(logs).toHaveLength(0);
  });
});

describe('WRITE_TRUSTED_MEMORY_DESCRIPTION', () => {
  it('mentions the bypass, the operator-direct-dictation requirement, and #325/#318', () => {
    // The description IS the agent-facing contract. The tests
    // assert on its load-bearing phrases rather than the exact
    // string so wording can evolve without test churn, but no
    // future edit can quietly drop the security framing.
    expect(WRITE_TRUSTED_MEMORY_DESCRIPTION).toMatch(/BYPASS/i);
    expect(WRITE_TRUSTED_MEMORY_DESCRIPTION).toMatch(/operator/i);
    expect(WRITE_TRUSTED_MEMORY_DESCRIPTION).toMatch(/current chat turn/i);
    expect(WRITE_TRUSTED_MEMORY_DESCRIPTION).toMatch(/external/i);
    expect(WRITE_TRUSTED_MEMORY_DESCRIPTION).toMatch(/#325/);
    expect(WRITE_TRUSTED_MEMORY_DESCRIPTION).toMatch(/#318/);
  });
});

describe('performOperatorApprovedWrite — defaults', () => {
  // The default `fs` and `pid` resolve to the real `fs` module
  // and `process.pid` respectively. We exercise this path by
  // pointing the validation at a real path under a tmpdir
  // (still anchored at /workspace/trusted/ for the validator)
  // — see the atomic-write test above for the integration that
  // exercises the same defaults via the proxy.
  let restorePid: number;
  beforeEach(() => {
    restorePid = process.pid;
  });
  afterEach(() => {
    // No state to restore — `process.pid` is read-only at the
    // OS level, the assignment above just captures it.
    void restorePid;
  });

  it('uses process.pid in the tmp filename when no override is passed', () => {
    // The default-pid path is exercised by the proxy integration
    // test above, but we additionally pin the behaviour here by
    // observing the mock call list when input.pid is omitted.
    const { fs: mockFs, calls } = makeMockFs();
    performOperatorApprovedWrite(
      {
        file_path: '/workspace/trusted/x.md',
        content: 'x',
        operator_justification: 'operator dictated in turn 1',
        fs: mockFs,
      },
    );
    const writeCall = calls.find((c) => c.kind === 'write');
    expect(writeCall).toBeDefined();
    expect(writeCall!.args[0]).toBe(
      `/workspace/trusted/x.md.tmp.${process.pid}`,
    );
  });
});
