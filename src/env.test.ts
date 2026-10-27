import fs from 'fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { readEnvFile } from './env.js';

afterEach(() => vi.restoreAllMocks());
describe('environment reader failure boundary', () => {
  it.each(['EACCES', 'EIO', 'EISDIR'])(
    'does not replace a real configuration read failure %s with defaults',
    (code) => {
      const failure = Object.assign(new Error('fixed config read failure'), {
        code,
      });
      vi.spyOn(fs, 'readFileSync').mockImplementationOnce(() => {
        throw failure;
      });
      expect(() => readEnvFile(['FIXTURE'])).toThrow(
        expect.objectContaining({
          cause: failure,
          message: expect.stringContaining('Cannot read .env'),
        }),
      );
    },
  );

  it('retains the missing-file fallback while rejecting unexpected reader errors', () => {
    const read = vi.spyOn(fs, 'readFileSync');
    read.mockImplementationOnce(() => {
      throw Object.assign(new Error('fixed missing'), { code: 'ENOENT' });
    });
    expect(readEnvFile(['FIXTURE'])).toEqual({});
    const defect = new TypeError('fixed reader defect');
    read.mockImplementationOnce(() => {
      throw defect;
    });
    expect(() => readEnvFile(['FIXTURE'])).toThrow(defect);
  });
});
