import { HttpError } from 'grammy';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createDraftStream } from './draft-stream.js';

afterEach(() => vi.restoreAllMocks());

describe('draft stream failure boundaries', () => {
  it('rejects an unexpected send implementation error instead of committing the preview', async () => {
    const defect = new TypeError('fixed send defect');
    const send = vi.fn(async () => {
      throw defect;
    });
    const edit = vi.fn(async () => {});
    const stream = createDraftStream({
      sendMessage: send,
      editMessage: edit,
      deleteMessage: vi.fn(async () => {}),
    });
    await expect(stream.finish('fixed preview')).rejects.toBe(defect);
    expect(send).toHaveBeenCalledTimes(1);
    expect(edit).not.toHaveBeenCalled();
  });

  it('does not report success for an operational send failure or retry it blindly', async () => {
    const failure = new HttpError(
      'fixed delivery failure',
      new Error('synthetic'),
    );
    const send = vi.fn(async () => {
      throw failure;
    });
    const stream = createDraftStream({
      sendMessage: send,
      editMessage: vi.fn(async () => {}),
      deleteMessage: vi.fn(async () => {}),
    });
    await expect(stream.finish('fixed output')).rejects.toBe(failure);
    await expect(stream.finish('fixed output')).rejects.toBe(failure);
    expect(send).toHaveBeenCalledTimes(1);
  });

  it('requires a native message ID before committing a successful initial draft', async () => {
    const stream = createDraftStream({
      sendMessage: vi.fn(async () => undefined),
      editMessage: vi.fn(async () => {}),
      deleteMessage: vi.fn(async () => {}),
    });
    await expect(stream.finish('fixed output')).rejects.toThrow(
      'no native message ID',
    );
  });

  it('propagates a failed native edit instead of completing the final draft', async () => {
    const failure = new HttpError('fixed edit failure', new Error('synthetic'));
    const stream = createDraftStream({
      sendMessage: vi.fn(async () => 17),
      editMessage: vi.fn(async () => {
        throw failure;
      }),
      deleteMessage: vi.fn(async () => {}),
    });
    await stream.finish('fixed first output');
    await expect(stream.finish('fixed final output')).rejects.toBe(failure);
  });

  it('keeps expected Telegram cleanup failure best-effort and propagates an unexpected one', async () => {
    const remove = vi.fn(async (_id: number) => {});
    const stream = createDraftStream({
      sendMessage: vi.fn(async () => 17),
      editMessage: vi.fn(async () => {}),
      deleteMessage: remove,
    });
    await expect(stream.finish('fixed preview')).resolves.toBe(true);
    remove.mockRejectedValueOnce(
      new HttpError('fixed transport failure', new Error('synthetic')),
    );
    await expect(stream.cancel()).resolves.toBeUndefined();
    const defect = new TypeError('fixed delete defect');
    remove.mockRejectedValueOnce(defect);
    await expect(stream.cancel()).rejects.toBe(defect);
    expect(remove).toHaveBeenCalledTimes(2);
    expect(remove).toHaveBeenLastCalledWith(17);
  });
});
