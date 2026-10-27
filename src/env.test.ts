import fs from 'fs';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { readEnvFile } from './env.js';

afterEach(() => vi.restoreAllMocks());
describe('environment reader failure boundary', () => {
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
