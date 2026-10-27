import {
  isTelegramError,
  TelegramDeliveryError,
} from './operational-errors.js';
import { logger } from './logger.js';

export interface DraftStream {
  /** Send or edit the preview with accumulated text. Throttled. */
  update(text: string): void;
  /** Finalize the preview. Returns false if text exceeded maxLength (caller should fall back to sendMessage). */
  finish(text: string): Promise<boolean>;
  /** Delete the preview message (e.g., if agent produced no output). */
  cancel(): Promise<void>;
}

export interface DraftStreamOpts {
  /** Send initial message, return platform message ID. */
  sendMessage(text: string): Promise<number | undefined>;
  /** Edit an existing message by ID. */
  editMessage(messageId: number, text: string): Promise<void>;
  /** Delete a message by ID. */
  deleteMessage(messageId: number): Promise<void>;
  /** Minimum ms between edits. Default: 1000 */
  throttleMs?: number;
  /** Max message length before giving up on streaming. Default: 4096 */
  maxLength?: number;
  /** Min chars before sending first message (avoids noisy push notifs). Default: 30 */
  minInitialChars?: number;
}

export function createDraftStream(opts: DraftStreamOpts): DraftStream {
  const throttleMs = opts.throttleMs ?? 1000;
  const maxLength = opts.maxLength ?? 4096;
  const minInitialChars = opts.minInitialChars ?? 30;

  let messageId: number | undefined;
  let deliveryFailure: Error | undefined;
  let unexpectedFailure: { error: unknown } | undefined;
  let lastSentText = '';
  let pendingText = '';
  let stopped = false;
  let inFlight: Promise<void> = Promise.resolve();
  let throttleTimer: ReturnType<typeof setTimeout> | null = null;

  async function doSendOrEdit(text: string): Promise<void> {
    if (text === lastSentText && !deliveryFailure) return;
    try {
      if (!messageId) {
        messageId = await opts.sendMessage(text);
        if (!Number.isSafeInteger(messageId) || !messageId || messageId <= 0)
          throw new TelegramDeliveryError(
            'Draft delivery has no native message ID; inspect Telegram delivery before retrying.',
          );
      } else {
        await opts.editMessage(messageId, text);
      }
      lastSentText = text;
      deliveryFailure = undefined;
    } catch (err) {
      if (!isTelegramError(err)) throw err;
      deliveryFailure = err;
      logger.debug({ err }, 'Draft stream send/edit failed');
    }
  }

  function throwUnexpectedFailure(): void {
    if (unexpectedFailure) throw unexpectedFailure.error;
  }

  function flush(): void {
    if (!pendingText || (pendingText === lastSentText && !deliveryFailure))
      return;
    const text = pendingText;
    inFlight = inFlight
      .then(() => {
        if (!unexpectedFailure) return doSendOrEdit(text);
      })
      .then(undefined, (error: unknown) => {
        unexpectedFailure = { error };
        stopped = true;
        if (throttleTimer) clearTimeout(throttleTimer);
        throttleTimer = null;
      });
  }

  return {
    update(text: string): void {
      if (stopped) return;
      pendingText = text;

      // Wait for minimum content before first message
      if (!messageId && text.length < minInitialChars) return;

      // Throttle edits
      if (throttleTimer) return;
      flush();
      throttleTimer = setTimeout(() => {
        throttleTimer = null;
        if (!stopped) flush();
      }, throttleMs);
    },

    async finish(text: string): Promise<boolean> {
      stopped = true;
      if (throttleTimer) {
        clearTimeout(throttleTimer);
        throttleTimer = null;
      }

      await inFlight;
      throwUnexpectedFailure();

      if (text.length > maxLength) {
        // Text too long for a single message — caller should fall back
        if (messageId) {
          try {
            await opts.deleteMessage(messageId);
          } catch (err) {
            if (!isTelegramError(err)) throw err;
            logger.debug(
              { err },
              'Could not delete draft preview; check Telegram connectivity',
            );
          }
        }
        return false;
      }

      pendingText = text;
      flush();
      await inFlight;
      throwUnexpectedFailure();
      if (deliveryFailure) throw deliveryFailure;
      return true;
    },

    async cancel(): Promise<void> {
      stopped = true;
      if (throttleTimer) {
        clearTimeout(throttleTimer);
        throttleTimer = null;
      }
      await inFlight;
      const [cleanup] = await Promise.allSettled([
        (async () => {
          if (!messageId) return;
          try {
            await opts.deleteMessage(messageId);
          } catch (err) {
            if (!isTelegramError(err)) throw err;
            logger.debug(
              { err },
              'Could not delete draft preview; check Telegram connectivity',
            );
          }
        })(),
      ]);
      if (cleanup.status === 'rejected') {
        if (unexpectedFailure)
          throw new AggregateError(
            [unexpectedFailure.error, cleanup.reason],
            'Draft update and preview cleanup failed; inspect the preview before retrying.',
            { cause: unexpectedFailure.error },
          );
        throw cleanup.reason;
      }
      throwUnexpectedFailure();
    },
  };
}
