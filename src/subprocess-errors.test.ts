import { describe, it, expect } from 'vitest';

import { isSubprocessError } from './subprocess-errors.js';

describe('isSubprocessError', () => {
  it('accepts a non-zero-exit error (carries status)', () => {
    const err = Object.assign(new Error('exited 1'), {
      status: 1,
      signal: null,
    });
    expect(isSubprocessError(err)).toBe(true);
  });

  it('accepts a killed/timeout error (carries signal)', () => {
    const err = Object.assign(new Error('timed out'), {
      status: null,
      signal: 'SIGTERM',
    });
    expect(isSubprocessError(err)).toBe(true);
  });

  it('accepts a spawn failure with a known errno code', () => {
    for (const code of ['ENOENT', 'EACCES', 'EPERM']) {
      const err = Object.assign(new Error(code), { code });
      expect(isSubprocessError(err)).toBe(true);
    }
  });

  it('rejects a Node programming error with an ERR_* string code', () => {
    // A bad-argument defect carries a string `.code` too — it must NOT be
    // mistaken for a subprocess failure and swallowed.
    const err = Object.assign(new TypeError('bad arg'), {
      code: 'ERR_INVALID_ARG_TYPE',
    });
    expect(isSubprocessError(err)).toBe(false);
  });

  it('rejects a plain programming defect so it propagates', () => {
    expect(isSubprocessError(new TypeError('bad call'))).toBe(false);
    expect(isSubprocessError(new Error('unmarked'))).toBe(false);
  });

  it('rejects non-Error throws', () => {
    expect(isSubprocessError('boom')).toBe(false);
    expect(isSubprocessError(null)).toBe(false);
    expect(isSubprocessError({ status: 1 })).toBe(false);
  });
});
