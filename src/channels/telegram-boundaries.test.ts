import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./registry.js', () => ({ registerChannel: vi.fn() }));
vi.mock('../env.js', () => ({ readEnvFile: vi.fn(() => ({})) }));
vi.mock('../config.js', () => ({
  ASSISTANT_NAME: 'Andy',
  TRIGGER_PATTERN: /^@Andy\b/i,
}));
vi.mock('../logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

interface MockBotApi {
  sendMessage: ReturnType<typeof vi.fn>;
  sendChatAction: ReturnType<typeof vi.fn>;
  raw: { setMessageReaction: ReturnType<typeof vi.fn> };
}

const botRef = vi.hoisted(() => ({
  current: null as { api: MockBotApi } | null,
}));

// The real GrammyError and HttpError classes stay in place so the guards in
// telegram.ts are exercised against what grammY throws in production.
vi.mock('grammy', async (importOriginal) => {
  const actual = await importOriginal<typeof import('grammy')>();

  class MockBot {
    token: string;
    api: MockBotApi = {
      sendMessage: vi.fn().mockResolvedValue(undefined),
      sendChatAction: vi.fn().mockResolvedValue(undefined),
      raw: { setMessageReaction: vi.fn().mockResolvedValue(true) },
    };

    constructor(token: string) {
      this.token = token;
      botRef.current = this;
    }

    command(): void {}

    on(): void {}

    catch(): void {}

    start(opts: {
      onStart: (botInfo: { username: string; id: number }) => void;
    }): void {
      opts.onStart({ username: 'andy_ai_bot', id: 12345 });
    }

    stop(): void {}
  }

  return { ...actual, Bot: MockBot };
});

import { GrammyError, HttpError } from 'grammy';

import { logger } from '../logger.js';
import { TelegramChannel } from './telegram.js';

function apiError(
  code: number,
  description: string,
  method = 'sendMessage',
): GrammyError {
  return new GrammyError(
    `Call to '${method}' failed! (${code}: ${description})`,
    { ok: false, error_code: code, description },
    method,
    {},
  );
}

function networkError(method: string): HttpError {
  return new HttpError(
    `Network request for '${method}' failed!`,
    new Error('ECONNRESET'),
  );
}

function currentApi(): MockBotApi {
  if (!botRef.current) throw new Error('Telegram mock bot was not created');
  return botRef.current.api;
}

async function connectedChannel(): Promise<TelegramChannel> {
  const channel = new TelegramChannel('test-token', {
    onMessage: vi.fn(),
    onChatMetadata: vi.fn(),
    registeredGroups: () => ({}),
  });
  await channel.connect();
  return channel;
}

beforeEach(() => {
  botRef.current = null;
  vi.clearAllMocks();
});

describe('Telegram send boundaries', () => {
  it('retries as plain text only when Telegram cannot parse the Markdown', async () => {
    const channel = await connectedChannel();
    currentApi().sendMessage.mockRejectedValueOnce(
      apiError(
        400,
        "Bad Request: can't parse entities: Can't find end of the entity starting at byte offset 5",
      ),
    );

    await expect(
      channel.sendMessage('tg:100', '*broken'),
    ).resolves.toBeUndefined();

    const calls = currentApi().sendMessage.mock.calls;
    expect(calls).toHaveLength(2);
    expect(calls[0][2]).toMatchObject({ parse_mode: 'Markdown' });
    expect(calls[1][2]).not.toHaveProperty('parse_mode');
  });

  it('logs and returns without a plain-text retry on other Telegram rejections', async () => {
    const channel = await connectedChannel();
    currentApi().sendMessage.mockRejectedValueOnce(
      apiError(403, 'Forbidden: bot was blocked by the user'),
    );

    await expect(
      channel.sendMessage('tg:100', 'hello'),
    ).resolves.toBeUndefined();

    expect(currentApi().sendMessage).toHaveBeenCalledTimes(1);
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ jid: 'tg:100' }),
      'Failed to send Telegram message',
    );
  });

  it('logs and returns when the Telegram HTTP call fails', async () => {
    const channel = await connectedChannel();
    currentApi().sendMessage.mockRejectedValueOnce(networkError('sendMessage'));

    await expect(
      channel.sendMessage('tg:100', 'hello'),
    ).resolves.toBeUndefined();

    expect(currentApi().sendMessage).toHaveBeenCalledTimes(1);
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ jid: 'tg:100' }),
      'Failed to send Telegram message',
    );
  });

  it('propagates send failures that are not Telegram errors', async () => {
    const channel = await connectedChannel();
    currentApi().sendMessage.mockRejectedValueOnce(
      new TypeError('unexpected implementation failure'),
    );

    await expect(channel.sendMessage('tg:100', 'hello')).rejects.toThrow(
      'unexpected implementation failure',
    );
    expect(logger.error).not.toHaveBeenCalled();
  });

  it('typing indicator absorbs Telegram failures and propagates others', async () => {
    const channel = await connectedChannel();

    currentApi().sendChatAction.mockRejectedValueOnce(
      networkError('sendChatAction'),
    );
    await expect(channel.setTyping('tg:100', true)).resolves.toBeUndefined();

    currentApi().sendChatAction.mockRejectedValueOnce(
      new TypeError('unexpected implementation failure'),
    );
    await expect(channel.setTyping('tg:100', true)).rejects.toThrow(
      'unexpected implementation failure',
    );
  });

  it('reactions absorb Telegram failures and propagate others', async () => {
    const channel = await connectedChannel();

    currentApi().raw.setMessageReaction.mockRejectedValueOnce(
      apiError(400, 'Bad Request: REACTION_INVALID', 'setMessageReaction'),
    );
    await expect(
      channel.sendReaction('tg:100', '5', '👍'),
    ).resolves.toBeUndefined();
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ jid: 'tg:100', messageId: '5' }),
      'Failed to send Telegram reaction',
    );

    currentApi().raw.setMessageReaction.mockRejectedValueOnce(
      new TypeError('unexpected implementation failure'),
    );
    await expect(channel.sendReaction('tg:100', '5', '👍')).rejects.toThrow(
      'unexpected implementation failure',
    );
  });
});
