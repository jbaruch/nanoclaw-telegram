import { describe, it, expect } from 'vitest';

import { isFsErrorWithCode, isExpectedFsError } from './fs-errors.js';

describe('isFsErrorWithCode', () => {
  it('matches an Error whose code is in the accepted set', () => {
    const err = Object.assign(new Error('missing'), { code: 'ENOENT' });
    expect(isFsErrorWithCode(err, ['ENOENT'])).toBe(true);
    expect(isFsErrorWithCode(err, ['ELOOP', 'ENOENT'])).toBe(true);
  });

  it('rejects an Error whose code is not in the accepted set', () => {
    const err = Object.assign(new Error('loop'), { code: 'ELOOP' });
    expect(isFsErrorWithCode(err, ['ENOENT'])).toBe(false);
  });

  it('rejects an Error with no code', () => {
    expect(isFsErrorWithCode(new Error('plain'), ['ENOENT'])).toBe(false);
  });

  it('rejects non-Error values so they propagate', () => {
    expect(isFsErrorWithCode('ENOENT', ['ENOENT'])).toBe(false);
    expect(isFsErrorWithCode(null, ['ENOENT'])).toBe(false);
    expect(isFsErrorWithCode({ code: 'ENOENT' }, ['ENOENT'])).toBe(false);
  });

  it('rejects everything against an empty accepted set', () => {
    const err = Object.assign(new Error('missing'), { code: 'ENOENT' });
    expect(isFsErrorWithCode(err, [])).toBe(false);
  });
});

describe('isExpectedFsError', () => {
  it('accepts the standard best-effort errno codes', () => {
    for (const code of ['EACCES', 'EPERM', 'ENOSPC', 'EROFS', 'ENOENT']) {
      const err = Object.assign(new Error(code), { code });
      expect(isExpectedFsError(err)).toBe(true);
    }
  });

  it('rejects an errno outside the standard set (e.g. ELOOP)', () => {
    const err = Object.assign(new Error('loop'), { code: 'ELOOP' });
    expect(isExpectedFsError(err)).toBe(false);
  });

  it('rejects a non-fs defect so it propagates', () => {
    expect(isExpectedFsError(new TypeError('bug'))).toBe(false);
  });
});
