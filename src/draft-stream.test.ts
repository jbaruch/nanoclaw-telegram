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
