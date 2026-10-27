import { HttpError } from 'grammy';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createDraftStream } from './draft-stream.js';

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllTimers();
  vi.useRealTimers();
});

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

  it('reports a failed final send and permits a later successful final attempt', async () => {
    const failure = new HttpError(
      'fixed delivery failure',
      new Error('synthetic'),
    );
    const send = vi.fn(async () => 17).mockRejectedValueOnce(failure);
    const stream = createDraftStream({
      sendMessage: send,
      editMessage: vi.fn(async () => {}),
      deleteMessage: vi.fn(async () => {}),
    });
    await expect(stream.finish('fixed output')).rejects.toBe(failure);
    await expect(stream.finish('fixed output')).resolves.toBe(true);
    expect(send).toHaveBeenCalledTimes(2);
  });

  it('attempts the final text after an intermediate initial-send failure', async () => {
    vi.useFakeTimers();
    const failure = new HttpError(
      'fixed preview failure',
      new Error('synthetic'),
    );
    const send = vi.fn(async () => 17).mockRejectedValueOnce(failure);
    const stream = createDraftStream({
      sendMessage: send,
      editMessage: vi.fn(async () => {}),
      deleteMessage: vi.fn(async () => {}),
    });
    stream.update('fixed preliminary preview longer than thirty characters');
    await vi.advanceTimersByTimeAsync(0);
    await expect(stream.finish('fixed final output')).resolves.toBe(true);
    expect(send).toHaveBeenCalledTimes(2);
    expect(send).toHaveBeenLastCalledWith('fixed final output');
    expect(vi.getTimerCount()).toBe(0);
  });

  it.each([
    'fixed final output',
    'fixed preliminary preview longer than thirty characters',
  ])(
    'confirms final text %s after an intermediate edit failure',
    async (finalText) => {
      vi.useFakeTimers();
      const failure = new HttpError(
        'fixed preview edit failure',
        new Error('synthetic'),
      );
      const send = vi.fn(async () => 17);
      const edit = vi.fn(async () => {}).mockRejectedValueOnce(failure);
      const stream = createDraftStream({
        sendMessage: send,
        editMessage: edit,
        deleteMessage: vi.fn(async () => {}),
      });
      stream.update('fixed preliminary preview longer than thirty characters');
      await vi.advanceTimersByTimeAsync(0);
      stream.update(
        'fixed intermediate edited preview longer than thirty characters',
      );
      await vi.advanceTimersByTimeAsync(1000);
      await expect(stream.finish(finalText)).resolves.toBe(true);
      expect(send).toHaveBeenCalledTimes(1);
      expect(edit).toHaveBeenCalledTimes(2);
      expect(edit).toHaveBeenLastCalledWith(17, finalText);
      expect(vi.getTimerCount()).toBe(0);
    },
  );

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
