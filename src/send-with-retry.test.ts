import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { _sendWithRetryForTests } from './index.js';
import { MessageDeliveryError } from './operational-errors.js';
import type { Channel } from './types.js';

function makeChannel(sendMessage: Channel['sendMessage']): Channel {
  return {
    name: 'test',
    connect: async () => {},
    sendMessage,
    isConnected: () => true,
    ownsJid: () => true,
    disconnect: async () => {},
  };
}

function serviceError(status: number): Error {
  return Object.assign(new Error(`service failure ${status}`), {
    error_code: status,
  });
}

describe('sendWithRetry', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('retries transient failures with backoff', async () => {
    const sendMessage = vi
      .fn<Channel['sendMessage']>()
      .mockRejectedValueOnce(serviceError(500))
      .mockRejectedValueOnce(serviceError(503))
      .mockResolvedValue();
    const promise = _sendWithRetryForTests(
      makeChannel(sendMessage),
      'test@g.us',
      'hello',
    );

    await vi.advanceTimersByTimeAsync(2000);
    expect(sendMessage).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(4000);

    await expect(promise).resolves.toBeUndefined();
    expect(sendMessage).toHaveBeenCalledTimes(3);
  });

  it('wraps the final service failure and preserves its cause', async () => {
    const err = serviceError(500);
    const sendMessage = vi.fn<Channel['sendMessage']>().mockRejectedValue(err);
    const promise = _sendWithRetryForTests(
      makeChannel(sendMessage),
      'test@g.us',
      'hello',
    ).catch((caught: unknown) => caught);

    await vi.advanceTimersByTimeAsync(6000);
    const caught = await promise;

    expect(caught).toBeInstanceOf(MessageDeliveryError);
    expect((caught as MessageDeliveryError).cause).toBe(err);
    expect(sendMessage).toHaveBeenCalledTimes(3);
  });

  it('does not retry a non-transient 4xx response', async () => {
    const err = serviceError(400);
    const sendMessage = vi.fn<Channel['sendMessage']>().mockRejectedValue(err);

    await expect(
      _sendWithRetryForTests(makeChannel(sendMessage), 'test@g.us', 'hello'),
    ).rejects.toMatchObject({ cause: err });
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });

  it('propagates TypeError unwrapped on the first attempt', async () => {
    const err = Object.assign(new TypeError('send implementation failed'), {
      error_code: 500,
    });
    const sendMessage = vi.fn<Channel['sendMessage']>().mockRejectedValue(err);

    await expect(
      _sendWithRetryForTests(makeChannel(sendMessage), 'test@g.us', 'hello'),
    ).rejects.toBe(err);
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });
});
