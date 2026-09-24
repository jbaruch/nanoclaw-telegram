import { describe, expect, it } from 'vitest';

import {
  isMessageServiceError,
  isTransientMessageServiceError,
} from './operational-errors.js';

function serviceError(fields: Record<string, unknown>): Error {
  return Object.assign(new Error('service failure'), fields);
}

describe('message service error classification', () => {
  it.each([
    serviceError({ error_code: 429 }),
    serviceError({ error_code: 500 }),
    serviceError({ code: 'ECONNRESET' }),
  ])('recognizes operational service failures', (err) => {
    expect(isMessageServiceError(err)).toBe(true);
  });

  it.each([
    serviceError({ error_code: 200 }),
    serviceError({ error_code: 399 }),
    serviceError({ error_code: 600 }),
    { error_code: 500 },
  ])('rejects non-operational service shapes', (err) => {
    expect(isMessageServiceError(err)).toBe(false);
  });

  it('rejects TypeError even when it carries a service status', () => {
    const err = Object.assign(new TypeError('programming failure'), {
      error_code: 500,
    });

    expect(isMessageServiceError(err)).toBe(false);
    expect(isTransientMessageServiceError(err)).toBe(false);
  });

  it('marks 5xx and network failures transient but not 4xx failures', () => {
    expect(
      isTransientMessageServiceError(serviceError({ error_code: 503 })),
    ).toBe(true);
    expect(
      isTransientMessageServiceError(serviceError({ code: 'ECONNRESET' })),
    ).toBe(true);
    expect(
      isTransientMessageServiceError(serviceError({ error_code: 429 })),
    ).toBe(false);
  });
});
