import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// --- Mocks ---

// Mock registry (registerChannel runs at import time)
vi.mock('./registry.js', () => ({ registerChannel: vi.fn() }));

// Mock env reader (used by the factory, not needed in unit tests)
vi.mock('../env.js', () => ({ readEnvFile: vi.fn(() => ({})) }));

// Mock config
vi.mock('../config.js', () => ({
  ASSISTANT_NAME: 'Andy',
  TRIGGER_PATTERN: /(?:^|\s)@Andy\b/i,
  // Mirror the real builder's word-boundary semantics so trigger-gate
  // tests exercise the same shape the orchestrator does.
  getTriggerPattern: (trigger?: string) => {
    const t = (trigger?.trim() || '@Andy').replace(
      /[.*+?^${}()|[\]\\]/g,
      '\\$&',
    );
    return new RegExp(`(?:^|\\s)${t}\\b`, 'i');
  },
}));

// Mock observer
const noteLatestUserMessageMock = vi.hoisted(() => vi.fn());
vi.mock('../observer.js', () => ({
  noteLatestUserMessage: noteLatestUserMessageMock,
}));

// Mock logger
vi.mock('../logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// --- Grammy mock ---

type Handler = (...args: any[]) => any;

const botRef = vi.hoisted(() => ({ current: null as any }));

// `InputFile` is constructed eagerly in `sendFile` from a path. The real
// grammy implementation touches the filesystem, which our tests don't
// stage — mock it to a passthrough wrapper so the sendDocument call
// site is exercised without hitting disk.
vi.mock('grammy', () => ({
  InputFile: class MockInputFile {
    constructor(public source: string | Buffer) {}
  },
  Bot: class MockBot {
    token: string;
    commandHandlers = new Map<string, Handler>();
    filterHandlers = new Map<string, Handler[]>();
    errorHandler: Handler | null = null;

    api = {
      sendMessage: vi.fn().mockResolvedValue({ message_id: 999 }),
      sendChatAction: vi.fn().mockResolvedValue(undefined),
      sendDocument: vi.fn().mockResolvedValue({ message_id: 1001 }),
      // `config.use` is the hook the grammy API transformer attaches
      // to. Real grammy exposes it on every Bot instance. Tests that
      // want to simulate "hook unavailable" (older or future grammy,
      // renamed surface) can delete `api.config` on the constructed
      // bot before assertions.
      config: { use: vi.fn() },
    };

    constructor(token: string) {
      this.token = token;
      botRef.current = this;
    }

    command(name: string, handler: Handler) {
      this.commandHandlers.set(name, handler);
    }

    on(filter: string, handler: Handler) {
      const existing = this.filterHandlers.get(filter) || [];
      existing.push(handler);
      this.filterHandlers.set(filter, existing);
    }

    catch(handler: Handler) {
      this.errorHandler = handler;
    }

    async start(opts: { onStart: (botInfo: any) => void }) {
      // Real grammy Bot.start() returns a Promise; connect() attaches
      // a `.catch(...)` to it. Returning void would throw TypeError on
      // `.catch` access synchronously — tests pass today only because
      // onStart resolves the outer Promise before the TypeError
      // surfaces. Match the real API shape so stricter runtimes don't
      // trip.
      opts.onStart({ username: 'andy_ai_bot', id: 12345 });
    }

    stop() {}
  },
}));

import {
  TelegramChannel,
  TelegramChannelOpts,
  splitMessage,
} from './telegram.js';
import { logger } from '../logger.js';
import { _initTestDatabase, storeChatMetadata, storeMessage } from '../db.js';

// --- Test helpers ---

function createTestOpts(
  overrides?: Partial<TelegramChannelOpts>,
): TelegramChannelOpts {
  return {
    onMessage: vi.fn(),
    onChatMetadata: vi.fn(),
    registeredGroups: vi.fn(() => ({
      'tg:100200300': {
        name: 'Test Group',
        folder: 'test-group',
        trigger: '@Andy',
        added_at: '2024-01-01T00:00:00.000Z',
      },
    })),
    ...overrides,
  };
}

function createTextCtx(overrides: {
  chatId?: number;
  chatType?: string;
  chatTitle?: string;
  text: string;
  fromId?: number;
  firstName?: string;
  username?: string;
  messageId?: number;
  date?: number;
  entities?: any[];
}) {
  const chatId = overrides.chatId ?? 100200300;
  const chatType = overrides.chatType ?? 'group';
  return {
    chat: {
      id: chatId,
      type: chatType,
      title: overrides.chatTitle ?? 'Test Group',
    },
    from: {
      id: overrides.fromId ?? 99001,
      first_name: overrides.firstName ?? 'Alice',
      username: overrides.username ?? 'alice_user',
    },
    message: {
      text: overrides.text,
      date: overrides.date ?? Math.floor(Date.now() / 1000),
      message_id: overrides.messageId ?? 1,
      entities: overrides.entities ?? [],
    },
    me: { username: 'andy_ai_bot' },
    reply: vi.fn(),
  };
}

function createMediaCtx(overrides: {
  chatId?: number;
  chatType?: string;
  fromId?: number;
  firstName?: string;
  date?: number;
  messageId?: number;
  caption?: string;
  extra?: Record<string, any>;
}) {
  const chatId = overrides.chatId ?? 100200300;
  return {
    chat: {
      id: chatId,
      type: overrides.chatType ?? 'group',
      title: 'Test Group',
    },
    from: {
      id: overrides.fromId ?? 99001,
      first_name: overrides.firstName ?? 'Alice',
      username: 'alice_user',
    },
    message: {
      date: overrides.date ?? Math.floor(Date.now() / 1000),
      message_id: overrides.messageId ?? 1,
      caption: overrides.caption,
      ...(overrides.extra || {}),
    },
    me: { username: 'andy_ai_bot' },
  };
}

function currentBot() {
  return botRef.current;
}

async function triggerTextMessage(ctx: ReturnType<typeof createTextCtx>) {
  const handlers = currentBot().filterHandlers.get('message:text') || [];
  for (const h of handlers) await h(ctx);
}

async function triggerMediaMessage(
  filter: string,
  ctx: ReturnType<typeof createMediaCtx>,
) {
  const handlers = currentBot().filterHandlers.get(filter) || [];
  for (const h of handlers) await h(ctx);
}

// --- Tests ---

describe('TelegramChannel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset to a fresh in-memory DB before every test. The cross-chat
    // reply_to safety check in `sendMessage` / `sendFile` consults the
    // messages table, so the helpers need a real schema to query —
    // without this every send-with-reply test would throw on an
    // uninitialized `db` global and the outer try/catch would swallow
    // the error, masking the failure as a "no API call" assertion miss.
    _initTestDatabase();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // --- Connection lifecycle ---

  describe('connection lifecycle', () => {
    it('resolves connect() when bot starts', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);

      await channel.connect();

      expect(channel.isConnected()).toBe(true);
    });

    it('registers command and message handlers on connect', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);

      await channel.connect();

      expect(currentBot().commandHandlers.has('chatid')).toBe(true);
      expect(currentBot().commandHandlers.has('ping')).toBe(true);
      expect(currentBot().filterHandlers.has('message:text')).toBe(true);
      expect(currentBot().filterHandlers.has('message:photo')).toBe(true);
      expect(currentBot().filterHandlers.has('message:video')).toBe(true);
      expect(currentBot().filterHandlers.has('message:voice')).toBe(true);
      expect(currentBot().filterHandlers.has('message:audio')).toBe(true);
      expect(currentBot().filterHandlers.has('message:document')).toBe(true);
      expect(currentBot().filterHandlers.has('message:sticker')).toBe(true);
      expect(currentBot().filterHandlers.has('message:location')).toBe(true);
      expect(currentBot().filterHandlers.has('message:contact')).toBe(true);
    });

    it('registers error handler on connect', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);

      await channel.connect();

      expect(currentBot().errorHandler).not.toBeNull();
    });

    it('disconnects cleanly', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);

      await channel.connect();
      expect(channel.isConnected()).toBe(true);

      await channel.disconnect();
      expect(channel.isConnected()).toBe(false);
    });

    it('isConnected() returns false before connect', () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);

      expect(channel.isConnected()).toBe(false);
    });

    // --- grammy API transformer gate (#81 diagnostic, PR #87) ---

    it('does NOT attach grammy transformer when LOG_LEVEL is not debug', async () => {
      const prev = process.env.LOG_LEVEL;
      delete process.env.LOG_LEVEL;
      try {
        const opts = createTestOpts();
        const channel = new TelegramChannel('test-token', opts);
        await channel.connect();
        expect(currentBot().api.config.use).not.toHaveBeenCalled();
      } finally {
        if (prev === undefined) delete process.env.LOG_LEVEL;
        else process.env.LOG_LEVEL = prev;
      }
    });

    it('attaches grammy transformer when LOG_LEVEL=debug', async () => {
      const prev = process.env.LOG_LEVEL;
      process.env.LOG_LEVEL = 'debug';
      try {
        const opts = createTestOpts();
        const channel = new TelegramChannel('test-token', opts);
        await channel.connect();
        expect(currentBot().api.config.use).toHaveBeenCalledTimes(1);
        // First arg is the transformer function — sanity check.
        expect(typeof currentBot().api.config.use.mock.calls[0][0]).toBe(
          'function',
        );
        // Info log announces the attachment so operators can see it
        // in docker logs when LOG_LEVEL=debug is set post-restart.
        expect(logger.info).toHaveBeenCalledWith(
          expect.stringContaining('Grammy API transformer attached'),
        );
      } finally {
        if (prev === undefined) delete process.env.LOG_LEVEL;
        else process.env.LOG_LEVEL = prev;
      }
    });

    it('warns and skips attach when LOG_LEVEL=debug but api.config.use is unavailable', async () => {
      const prev = process.env.LOG_LEVEL;
      process.env.LOG_LEVEL = 'debug';
      try {
        const opts = createTestOpts();
        const channel = new TelegramChannel('test-token', opts);
        // Simulate the "grammy API surface shifted or the Bot is a
        // minimal mock" case — the guard should keep connect() from
        // crashing and surface the skip as a warn.
        // We have to splice the api object AFTER the Bot is constructed
        // inside connect(), so grab the mock before connect runs.
        const origBot = (await import('grammy')) as unknown as {
          Bot: new (token: string) => {
            api: { config?: { use: unknown } };
          };
        };
        const OrigBot = origBot.Bot;
        // Wrap Bot so that right after construction we drop api.config.
        origBot.Bot = class extends OrigBot {
          constructor(token: string) {
            super(token);
            // Type assertion: the test mock's api is a concrete object.
            (this.api as { config?: unknown }).config = undefined;
          }
        } as typeof OrigBot;
        try {
          await channel.connect();
          expect(logger.warn).toHaveBeenCalledWith(
            expect.stringContaining('bot.api.config.use unavailable'),
          );
        } finally {
          origBot.Bot = OrigBot;
        }
      } finally {
        if (prev === undefined) delete process.env.LOG_LEVEL;
        else process.env.LOG_LEVEL = prev;
      }
    });
  });

  // --- Text message handling ---

  describe('text message handling', () => {
    it('delivers message for registered group', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({ text: 'Hello everyone' });
      await triggerTextMessage(ctx);

      expect(opts.onChatMetadata).toHaveBeenCalledWith(
        'tg:100200300',
        expect.any(String),
        'Test Group',
        'telegram',
        true,
      );
      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          id: '1',
          chat_jid: 'tg:100200300',
          sender: '99001',
          sender_name: 'Alice (@alice_user)',
          content: 'Hello everyone',
          is_from_me: false,
        }),
      );
    });

    it('only emits metadata for unregistered chats', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({ chatId: 999999, text: 'Unknown chat' });
      await triggerTextMessage(ctx);

      expect(opts.onChatMetadata).toHaveBeenCalledWith(
        'tg:999999',
        expect.any(String),
        'Test Group',
        'telegram',
        true,
      );
      expect(opts.onMessage).not.toHaveBeenCalled();
    });

    it('skips bot commands (/chatid, /ping) but passes other / messages through', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      // Bot commands should be skipped
      const ctx1 = createTextCtx({ text: '/chatid' });
      await triggerTextMessage(ctx1);
      expect(opts.onMessage).not.toHaveBeenCalled();
      expect(opts.onChatMetadata).not.toHaveBeenCalled();

      const ctx2 = createTextCtx({ text: '/ping' });
      await triggerTextMessage(ctx2);
      expect(opts.onMessage).not.toHaveBeenCalled();

      // Non-bot /commands should flow through
      const ctx3 = createTextCtx({ text: '/remote-control' });
      await triggerTextMessage(ctx3);
      expect(opts.onMessage).toHaveBeenCalledTimes(1);
      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ content: '/remote-control' }),
      );
    });

    it('extracts sender name from first_name', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({ text: 'Hi', firstName: 'Bob' });
      await triggerTextMessage(ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ sender_name: 'Bob (@alice_user)' }),
      );
    });

    it('falls back to username when first_name missing', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({ text: 'Hi' });
      ctx.from.first_name = undefined as any;
      await triggerTextMessage(ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ sender_name: 'alice_user (@alice_user)' }),
      );
    });

    it('falls back to user ID when name and username missing', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({ text: 'Hi', fromId: 42 });
      ctx.from.first_name = undefined as any;
      ctx.from.username = undefined as any;
      await triggerTextMessage(ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ sender_name: '42' }),
      );
    });

    it('uses sender name as chat name for private chats', async () => {
      const opts = createTestOpts({
        registeredGroups: vi.fn(() => ({
          'tg:100200300': {
            name: 'Private',
            folder: 'private',
            trigger: '@Andy',
            added_at: '2024-01-01T00:00:00.000Z',
          },
        })),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({
        text: 'Hello',
        chatType: 'private',
        firstName: 'Alice',
      });
      await triggerTextMessage(ctx);

      expect(opts.onChatMetadata).toHaveBeenCalledWith(
        'tg:100200300',
        expect.any(String),
        'Alice (@alice_user)', // Private chats use sender name with username
        'telegram',
        false,
      );
    });

    it('uses chat title as name for group chats', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({
        text: 'Hello',
        chatType: 'supergroup',
        chatTitle: 'Project Team',
      });
      await triggerTextMessage(ctx);

      expect(opts.onChatMetadata).toHaveBeenCalledWith(
        'tg:100200300',
        expect.any(String),
        'Project Team',
        'telegram',
        true,
      );
    });

    it('converts message.date to ISO timestamp', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const unixTime = 1704067200; // 2024-01-01T00:00:00.000Z
      const ctx = createTextCtx({ text: 'Hello', date: unixTime });
      await triggerTextMessage(ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          timestamp: '2024-01-01T00:00:00.000Z',
        }),
      );
    });
  });

  // --- @mention translation ---

  describe('@mention translation', () => {
    it('translates @bot_username mention to trigger format', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({
        text: '@andy_ai_bot what time is it?',
        entities: [{ type: 'mention', offset: 0, length: 12 }],
      });
      await triggerTextMessage(ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          content: '@Andy @andy_ai_bot what time is it?',
        }),
      );
    });

    it('does not translate if message already matches trigger', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({
        text: '@Andy @andy_ai_bot hello',
        entities: [{ type: 'mention', offset: 6, length: 12 }],
      });
      await triggerTextMessage(ctx);

      // Should NOT double-prepend — already starts with @Andy
      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          content: '@Andy @andy_ai_bot hello',
        }),
      );
    });

    it('does not translate mentions of other bots', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({
        text: '@some_other_bot hi',
        entities: [{ type: 'mention', offset: 0, length: 15 }],
      });
      await triggerTextMessage(ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          content: '@some_other_bot hi', // No translation
        }),
      );
    });

    it('handles mention in middle of message', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({
        text: 'hey @andy_ai_bot check this',
        entities: [{ type: 'mention', offset: 4, length: 12 }],
      });
      await triggerTextMessage(ctx);

      // Bot is mentioned, message doesn't match trigger → prepend trigger
      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          content: '@Andy hey @andy_ai_bot check this',
        }),
      );
    });

    it('handles message with no entities', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({ text: 'plain message' });
      await triggerTextMessage(ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          content: 'plain message',
        }),
      );
    });

    it('ignores non-mention entities', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createTextCtx({
        text: 'check https://example.com',
        entities: [{ type: 'url', offset: 6, length: 19 }],
      });
      await triggerTextMessage(ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          content: 'check https://example.com',
        }),
      );
    });
  });

  // --- Non-text messages ---

  describe('non-text messages', () => {
    it('stores photo with placeholder', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({
        extra: { photo: [{ file_id: 'f1', width: 100, height: 100 }] },
      });
      await triggerMediaMessage('message:photo', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ content: '[Image]' }),
      );
    });

    it('stores photo with caption', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({
        caption: 'Look at this',
        extra: { photo: [{ file_id: 'f1', width: 100, height: 100 }] },
      });
      await triggerMediaMessage('message:photo', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          content: '[Image] Look at this',
        }),
      );
    });

    it('stores video with placeholder', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({});
      await triggerMediaMessage('message:video', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ content: '[Video]' }),
      );
    });

    it('stores voice message with placeholder', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({
        extra: { voice: { file_id: 'v1' } },
      });
      await triggerMediaMessage('message:voice', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          content: '[Voice message - transcription failed]',
        }),
      );
    });

    it('stores audio with placeholder', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({});
      await triggerMediaMessage('message:audio', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ content: '[Audio]' }),
      );
    });

    it('stores document with filename', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({
        extra: { document: { file_name: 'report.pdf' } },
      });
      await triggerMediaMessage('message:document', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({
          content: '[Document: report.pdf - no file_id]',
        }),
      );
    });

    it('stores document with fallback name when filename missing', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({ extra: { document: {} } });
      await triggerMediaMessage('message:document', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ content: '[Document: file - no file_id]' }),
      );
    });

    it('stores sticker with emoji', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({
        extra: { sticker: { emoji: '😂' } },
      });
      await triggerMediaMessage('message:sticker', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ content: '[Sticker 😂]' }),
      );
    });

    it('stores location with placeholder', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({});
      await triggerMediaMessage('message:location', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ content: '[Location]' }),
      );
    });

    it('stores contact with placeholder', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({});
      await triggerMediaMessage('message:contact', ctx);

      expect(opts.onMessage).toHaveBeenCalledWith(
        'tg:100200300',
        expect.objectContaining({ content: '[Contact]' }),
      );
    });

    it('ignores non-text messages from unregistered chats', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const ctx = createMediaCtx({ chatId: 999999 });
      await triggerMediaMessage('message:photo', ctx);

      expect(opts.onMessage).not.toHaveBeenCalled();
    });
  });

  // --- sendMessage ---

  describe('sendMessage', () => {
    it('sends message via bot API', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendMessage('tg:100200300', 'Hello');

      expect(currentBot().api.sendMessage).toHaveBeenCalledWith(
        '100200300',
        'Hello',
        { parse_mode: 'HTML' },
      );
    });

    it('strips tg: prefix from JID', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendMessage('tg:-1001234567890', 'Group message');

      expect(currentBot().api.sendMessage).toHaveBeenCalledWith(
        '-1001234567890',
        'Group message',
        { parse_mode: 'HTML' },
      );
    });

    it('splits messages exceeding 4096 characters', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const longText = 'x'.repeat(5000);
      await channel.sendMessage('tg:100200300', longText);

      expect(currentBot().api.sendMessage).toHaveBeenCalledTimes(2);
      expect(currentBot().api.sendMessage).toHaveBeenNthCalledWith(
        1,
        '100200300',
        'x'.repeat(4096),
        { parse_mode: 'HTML' },
      );
      expect(currentBot().api.sendMessage).toHaveBeenNthCalledWith(
        2,
        '100200300',
        'x'.repeat(904),
        { parse_mode: 'HTML' },
      );
    });

    it('sends exactly one message at 4096 characters', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const exactText = 'y'.repeat(4096);
      await channel.sendMessage('tg:100200300', exactText);

      expect(currentBot().api.sendMessage).toHaveBeenCalledTimes(1);
    });

    it('handles send failure gracefully', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      // Both Markdown and plain text fail
      currentBot().api.sendMessage.mockRejectedValue(
        new Error('Network error'),
      );

      // Should not throw — error is caught internally
      await expect(
        channel.sendMessage('tg:100200300', 'Will fail'),
      ).resolves.toBeUndefined();
    });

    it('preserves link URLs in the degraded fallback so users can still reach them (PR #308 review)', async () => {
      // After #282 the fallback's `text` is sanitized HTML; a naive
      // tag-strip would drop `<a href="…">` entirely and leave only
      // the link label. `htmlToPlainText` converts `<a href="url">
      // label</a>` to `label (url)` so the URL survives the
      // formatting failure — exactly the moment the user most needs
      // it. Copilot caught the gap on PR #308.
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      currentBot()
        .api.sendMessage.mockRejectedValueOnce(
          new Error("can't parse entities"),
        )
        .mockResolvedValueOnce({ message_id: 9300 });

      await channel.sendMessage(
        'tg:100200300',
        'see [Docs](https://example.com/p) for details',
      );

      const fallbackArgs = currentBot().api.sendMessage.mock.calls[1];
      expect(fallbackArgs[1]).toBe(
        '⚠️ formatting failed; raw text below\n\nsee Docs (https://example.com/p) for details',
      );
    });

    it('marks the plain-text fallback with a visible degraded prefix (#278, #282)', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      // First call (HTML attempt) rejects; second call (plain fallback) resolves.
      // The fallback caption MUST carry a user-visible warning so the
      // user knows they're seeing a degraded rendering, not the agent's
      // intended formatting.
      //
      // After #282 (sanitize-then-split), the channel sanitizes the
      // whole message ONCE before chunking, so by the time the
      // fallback fires the input to `sendTelegramMessage` is already
      // HTML (`feeling <i>great</i> today`). The fallback's
      // `htmlToPlainText` strips tags + decodes entities so the user
      // sees readable plain text rather than literal-tag rendering
      // (`feeling great today` instead of `feeling <i>great</i>
      // today`). Italic markup is lost in the fallback rendering;
      // the visible warning prefix tells the user formatting failed.
      currentBot()
        .api.sendMessage.mockRejectedValueOnce(
          new Error("can't parse entities"),
        )
        .mockResolvedValueOnce({ message_id: 9001 });

      await channel.sendMessage('tg:100200300', 'feeling _great_ today');

      expect(currentBot().api.sendMessage).toHaveBeenCalledTimes(2);
      const fallbackArgs = currentBot().api.sendMessage.mock.calls[1];
      expect(fallbackArgs[1]).toBe(
        '⚠️ formatting failed; raw text below\n\nfeeling great today',
      );
      // Fallback options must NOT include parse_mode — that's what
      // turns raw markdown into "literal `<b>…</b>` tags rendered" if
      // smuggled through.
      expect(fallbackArgs[2]?.parse_mode).toBeUndefined();
      // Fallback options must NOT include the internal preSanitized
      // marker either — it's a host-side flag, not a Telegram API
      // field, and the fallback is plain text anyway.
      expect(fallbackArgs[2]?.preSanitized).toBeUndefined();
    });

    it('does not attempt plain-text fallback when DEV_NO_HTML_FALLBACK=1 (#278)', async () => {
      const prev = process.env.DEV_NO_HTML_FALLBACK;
      process.env.DEV_NO_HTML_FALLBACK = '1';
      try {
        const opts = createTestOpts();
        const channel = new TelegramChannel('test-token', opts);
        await channel.connect();

        currentBot().api.sendMessage.mockRejectedValueOnce(
          new Error("can't parse entities"),
        );

        // Outer sendMessage swallows the throw, but the inner fallback
        // must NOT issue a second sendMessage when the dev flag is set
        // — the operator wants the actual 400 to surface in CI logs,
        // not a successful degraded fallback masking the bug.
        await channel.sendMessage('tg:100200300', 'feeling _great_ today');

        expect(currentBot().api.sendMessage).toHaveBeenCalledTimes(1);
      } finally {
        if (prev === undefined) delete process.env.DEV_NO_HTML_FALLBACK;
        else process.env.DEV_NO_HTML_FALLBACK = prev;
      }
    });

    it('truncates the degraded-fallback body so the prefix never overflows MAX_LENGTH (#278)', async () => {
      // Pre-fix, prepending the warning prefix to a near-MAX_LENGTH
      // chunk could push it past Telegram's 4096-char message limit
      // and turn a recoverable HTML-parse error into a "fallback
      // also failed" lost message — strictly worse than the original
      // symptom. The fallback must always come in at or below
      // MAX_LENGTH.
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      currentBot()
        .api.sendMessage.mockRejectedValueOnce(
          new Error("can't parse entities"),
        )
        .mockResolvedValueOnce({ message_id: 9100 });

      // Single chunk exactly at MAX_LENGTH (4096). No paragraph or
      // newline boundaries so splitMessage doesn't pre-chunk it
      // smaller — the whole 4096 chars reach sendTelegramMessage as
      // one call.
      const huge = 'z'.repeat(4096);
      await channel.sendMessage('tg:100200300', huge);

      const fallbackText = currentBot().api.sendMessage.mock
        .calls[1][1] as string;
      expect(fallbackText.length).toBeLessThanOrEqual(4096);
      expect(
        fallbackText.startsWith('⚠️ formatting failed; raw text below\n\n'),
      ).toBe(true);
    });

    it('keeps the fallback WARN log content-free — no user text in any field (#278)', async () => {
      // `jbaruch/coding-policy: no-secrets` is explicit: "Never log
      // secrets — not at any log level" and "Sanitize or redact
      // sensitive values before they reach any logging or monitoring
      // system." A preview slice doesn't sanitize — a token can fit
      // in 200 chars. The fallback WARN/ERROR contexts ship metadata
      // only (err, chatId, lengths). Operators correlate by chatId +
      // timestamp and pull the actual text from chat history / DB
      // for repro; the 400's err object already carries Telegram's
      // byte-offset diagnostic. Full-body in-the-moment repro goes
      // through DEV_NO_HTML_FALLBACK=1 instead.
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      currentBot()
        .api.sendMessage.mockRejectedValueOnce(
          new Error("can't parse entities"),
        )
        .mockResolvedValueOnce({ message_id: 9101 });

      const sensitive = 'token=AKIA' + 'X'.repeat(300) + ' please format this';
      await channel.sendMessage('tg:100200300', sensitive);

      const warnCalls = vi.mocked(logger.warn).mock.calls;
      const fallbackWarn = warnCalls.find(
        ([, msg]) =>
          typeof msg === 'string' && msg.startsWith('[send] HTML send failed'),
      );
      expect(fallbackWarn).toBeDefined();
      const ctx = fallbackWarn![0] as Record<string, unknown>;
      // No user-content fields whatsoever — neither full bodies nor
      // truncated previews. Only metadata.
      expect(ctx.rawText).toBeUndefined();
      expect(ctx.sanitizedHtml).toBeUndefined();
      expect(ctx.rawPreview).toBeUndefined();
      expect(ctx.sanitizedPreview).toBeUndefined();
      expect(ctx.rawLen).toBe(sensitive.length);
      // Defensive: serialize the entire log context and confirm the
      // sentinel substring from the input does NOT appear anywhere
      // — catches future drift where someone re-introduces a
      // content field with a name we didn't think to negate above.
      expect(JSON.stringify(ctx)).not.toContain('AKIA');
    });

    it('does nothing when bot is not initialized', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);

      // Don't connect — bot is null
      await channel.sendMessage('tg:100200300', 'No bot');

      // No error, no API call
    });

    // --- cross-chat reply_to safety (PR #232) ---
    //
    // Telegram message IDs are per-chat sequential, so a `replyToMessageId`
    // captured in chat A may both (a) coincidentally match an unrelated
    // message in chat B and (b) routinely match a legitimate same-chat
    // reply target whose id ALSO happens to exist in some other chat.
    // The channel-level guard must:
    //   - keep `reply_parameters` when the id is present in the target chat
    //     (positive evidence of a local target — pathological "shared id in
    //     BOTH chats" case collapses to "keep");
    //   - keep `reply_parameters` when the id is absent from the DB entirely
    //     (let Telegram be authoritative; covers ids from before the
    //     orchestrator was running);
    //   - drop `reply_parameters` only when the id is present ONLY in some
    //     other chat (positive evidence the id is foreign).

    it('keeps reply_parameters when reply_to id exists in the target chat', async () => {
      storeChatMetadata('tg:100200300', '2026-04-28T00:00:00.000Z');
      storeMessage({
        id: '4242',
        chat_jid: 'tg:100200300',
        sender: 'user',
        sender_name: 'Alice',
        content: 'message in target chat',
        timestamp: '2026-04-28T00:00:00.000Z',
        is_from_me: false,
      });
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendMessage('tg:100200300', 'Hello back', '4242');

      const options = currentBot().api.sendMessage.mock.calls[0][2];
      expect(options.reply_parameters).toEqual({ message_id: 4242 });
    });

    it('drops reply_parameters when reply_to id only exists in a different chat', async () => {
      storeChatMetadata('tg:100200300', '2026-04-28T00:00:00.000Z');
      storeChatMetadata('tg:999888777', '2026-04-28T00:00:00.000Z');
      storeMessage({
        id: '4242',
        chat_jid: 'tg:999888777',
        sender: 'user',
        sender_name: 'ForeignAlice',
        content: 'message in foreign chat',
        timestamp: '2026-04-28T00:00:00.000Z',
        is_from_me: false,
      });
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendMessage('tg:100200300', 'Cross-chat broadcast', '4242');

      const options = currentBot().api.sendMessage.mock.calls[0][2];
      expect(options.reply_parameters).toBeUndefined();
      // The drop is logged at warn so operators can spot orchestrator bugs
      // that produce cross-chat reply_to ids in the first place — silence
      // would let those bugs continue to push misrouted reply ids forever.
      expect(logger.warn).toHaveBeenCalledWith(
        expect.objectContaining({
          jid: 'tg:100200300',
          replyToMessageId: '4242',
        }),
        expect.stringContaining('Dropping cross-chat reply_to'),
      );
    });

    it('keeps reply_parameters when the same id exists in BOTH the target and another chat', async () => {
      // Per-chat-sequential ids guarantee this case in any deployment with
      // more than one Telegram chat. Positive evidence (id-in-target) wins
      // over "id exists elsewhere", or threading would be stripped from
      // most legitimate same-chat replies.
      storeChatMetadata('tg:100200300', '2026-04-28T00:00:00.000Z');
      storeChatMetadata('tg:999888777', '2026-04-28T00:00:00.000Z');
      storeMessage({
        id: '4242',
        chat_jid: 'tg:100200300',
        sender: 'user',
        sender_name: 'LocalAlice',
        content: 'in target',
        timestamp: '2026-04-28T00:00:00.000Z',
        is_from_me: false,
      });
      storeMessage({
        id: '4242',
        chat_jid: 'tg:999888777',
        sender: 'user',
        sender_name: 'ForeignBob',
        content: 'in foreign',
        timestamp: '2026-04-28T00:00:00.000Z',
        is_from_me: false,
      });
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendMessage('tg:100200300', 'Same-chat reply', '4242');

      const options = currentBot().api.sendMessage.mock.calls[0][2];
      expect(options.reply_parameters).toEqual({ message_id: 4242 });
      // Importantly, no warn — there's nothing to flag here.
      expect(logger.warn).not.toHaveBeenCalledWith(
        expect.anything(),
        expect.stringContaining('Dropping cross-chat reply_to'),
      );
    });

    it('keeps reply_parameters when reply_to id is absent from the DB entirely', async () => {
      // Covers ids from before the orchestrator was running, or from a
      // freshly-deployed bot. We refuse to drop without positive evidence
      // the id is foreign — Telegram remains authoritative on existence.
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendMessage('tg:100200300', 'Reply to legacy', '9999');

      const options = currentBot().api.sendMessage.mock.calls[0][2];
      expect(options.reply_parameters).toEqual({ message_id: 9999 });
    });
  });

  // --- sendFile ---

  describe('sendFile', () => {
    it('sends document with HTML-sanitized caption and parse_mode HTML', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendFile(
        'tg:100200300',
        '/tmp/nanoclaw-test.png',
        'feeling _great_ today',
      );

      expect(currentBot().api.sendDocument).toHaveBeenCalledTimes(1);
      const call = currentBot().api.sendDocument.mock.calls[0];
      expect(call[0]).toBe('100200300'); // tg: prefix stripped
      expect(call[2].parse_mode).toBe('HTML');
      // sanitizeTelegramHtml rewrites `_great_` → `<i>great</i>` (see
      // telegram-sanitize.test.ts) — proves the HTML path is active and
      // the caption went through the sanitizer, not a bypass.
      expect(call[2].caption).toBe('feeling <i>great</i> today');
    });

    it('sends without parse_mode when no caption is provided', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendFile('tg:100200300', '/tmp/nanoclaw-test.png');

      expect(currentBot().api.sendDocument).toHaveBeenCalledTimes(1);
      const options = currentBot().api.sendDocument.mock.calls[0][2];
      expect(options.caption).toBeUndefined();
      expect(options.parse_mode).toBeUndefined();
    });

    it('falls back to plain caption without parse_mode when HTML send fails', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      // First call (HTML attempt) rejects; second call (plain fallback) resolves.
      currentBot()
        .api.sendDocument.mockRejectedValueOnce(
          new Error("can't parse entities"),
        )
        .mockResolvedValueOnce({ message_id: 2002 });

      await channel.sendFile(
        'tg:100200300',
        '/tmp/nanoclaw-test.png',
        'feeling _great_ today',
      );

      expect(currentBot().api.sendDocument).toHaveBeenCalledTimes(2);
      // Second call sends the ORIGINAL (pre-sanitize) caption with a
      // user-visible degraded-fallback prefix (#278) and no
      // parse_mode — the user knows this is a fallback rendering, not
      // the agent's intended formatting, and the operator's WARN log
      // line correlates with a visible breadcrumb in the chat.
      const plainOptions = currentBot().api.sendDocument.mock.calls[1][2];
      expect(plainOptions.parse_mode).toBeUndefined();
      expect(plainOptions.caption).toBe(
        '⚠️ formatting failed; raw caption below\n\nfeeling _great_ today',
      );
    });

    it('does not attempt plain-caption fallback when DEV_NO_HTML_FALLBACK=1 (#278)', async () => {
      const prev = process.env.DEV_NO_HTML_FALLBACK;
      process.env.DEV_NO_HTML_FALLBACK = '1';
      try {
        const opts = createTestOpts();
        const channel = new TelegramChannel('test-token', opts);
        await channel.connect();

        currentBot().api.sendDocument.mockRejectedValueOnce(
          new Error("can't parse entities"),
        );

        // The outer try/catch in sendFile still swallows so no throw at
        // the channel boundary, but the inner fallback must NOT issue a
        // second sendDocument when the dev flag is set.
        await channel.sendFile(
          'tg:100200300',
          '/tmp/nanoclaw-test.png',
          'feeling _great_ today',
        );

        expect(currentBot().api.sendDocument).toHaveBeenCalledTimes(1);
      } finally {
        if (prev === undefined) delete process.env.DEV_NO_HTML_FALLBACK;
        else process.env.DEV_NO_HTML_FALLBACK = prev;
      }
    });

    it('truncates the degraded caption so the prefix never overflows MAX_CAPTION_LENGTH (#278)', async () => {
      // Telegram caps `sendDocument` captions at 1024 chars. Pre-fix,
      // prepending the warning prefix to a near-1024 caption could
      // push it over the cap and turn a recoverable HTML-parse error
      // into a "fallback also failed" lost attachment.
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      currentBot()
        .api.sendDocument.mockRejectedValueOnce(
          new Error("can't parse entities"),
        )
        .mockResolvedValueOnce({ message_id: 9200 });

      const huge = 'q'.repeat(1024);
      await channel.sendFile('tg:100200300', '/tmp/nanoclaw-test.png', huge);

      const fallbackOpts = currentBot().api.sendDocument.mock.calls[1][2];
      expect(fallbackOpts.caption.length).toBeLessThanOrEqual(1024);
      expect(
        fallbackOpts.caption.startsWith(
          '⚠️ formatting failed; raw caption below\n\n',
        ),
      ).toBe(true);
    });

    it('does not retry sendDocument when no caption was provided', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      // Caption-less sends can't benefit from the plain-caption
      // fallback (payload would be identical). A retry would just
      // double API traffic on transient network errors.
      currentBot().api.sendDocument.mockRejectedValue(
        new Error('network blip'),
      );

      // Should not throw — outer catch swallows.
      await expect(
        channel.sendFile('tg:100200300', '/tmp/nanoclaw-test.png'),
      ).resolves.toBeUndefined();

      expect(currentBot().api.sendDocument).toHaveBeenCalledTimes(1);
    });

    it('includes reply_parameters when replyToMessageId is provided', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendFile(
        'tg:100200300',
        '/tmp/nanoclaw-test.png',
        'hello',
        '4242',
      );

      const options = currentBot().api.sendDocument.mock.calls[0][2];
      expect(options.reply_parameters).toEqual({ message_id: 4242 });
    });

    it('does nothing when bot is not initialized', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);

      // Don't connect — bot is null; should exit cleanly, not throw.
      await expect(
        channel.sendFile('tg:100200300', '/tmp/nanoclaw-test.png', 'hi'),
      ).resolves.toBeUndefined();
    });

    it('drops reply_parameters when reply_to id only exists in a different chat', async () => {
      // sendFile shares the same `safeReplyToForChat` predicate as
      // sendMessage — verify the wire-up so a regression on either path
      // wouldn't silently bypass the cross-chat guard.
      storeChatMetadata('tg:100200300', '2026-04-28T00:00:00.000Z');
      storeChatMetadata('tg:999888777', '2026-04-28T00:00:00.000Z');
      storeMessage({
        id: '4242',
        chat_jid: 'tg:999888777',
        sender: 'user',
        sender_name: 'ForeignAlice',
        content: 'message in foreign chat',
        timestamp: '2026-04-28T00:00:00.000Z',
        is_from_me: false,
      });
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.sendFile(
        'tg:100200300',
        '/tmp/nanoclaw-test.png',
        'hello',
        '4242',
      );

      const options = currentBot().api.sendDocument.mock.calls[0][2];
      expect(options.reply_parameters).toBeUndefined();
    });
  });

  // --- ownsJid ---

  describe('ownsJid', () => {
    it('owns tg: JIDs', () => {
      const channel = new TelegramChannel('test-token', createTestOpts());
      expect(channel.ownsJid('tg:123456')).toBe(true);
    });

    it('owns tg: JIDs with negative IDs (groups)', () => {
      const channel = new TelegramChannel('test-token', createTestOpts());
      expect(channel.ownsJid('tg:-1001234567890')).toBe(true);
    });

    it('does not own WhatsApp group JIDs', () => {
      const channel = new TelegramChannel('test-token', createTestOpts());
      expect(channel.ownsJid('12345@g.us')).toBe(false);
    });

    it('does not own WhatsApp DM JIDs', () => {
      const channel = new TelegramChannel('test-token', createTestOpts());
      expect(channel.ownsJid('12345@s.whatsapp.net')).toBe(false);
    });

    it('does not own unknown JID formats', () => {
      const channel = new TelegramChannel('test-token', createTestOpts());
      expect(channel.ownsJid('random-string')).toBe(false);
    });
  });

  // --- setTyping ---

  describe('setTyping', () => {
    it('sends typing action when isTyping is true', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.setTyping('tg:100200300', true);

      expect(currentBot().api.sendChatAction).toHaveBeenCalledWith(
        '100200300',
        'typing',
      );
    });

    it('does nothing when isTyping is false', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      await channel.setTyping('tg:100200300', false);

      expect(currentBot().api.sendChatAction).not.toHaveBeenCalled();
    });

    it('does nothing when bot is not initialized', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);

      // Don't connect
      await channel.setTyping('tg:100200300', true);

      // No error, no API call
    });

    it('handles typing indicator failure gracefully', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      currentBot().api.sendChatAction.mockRejectedValueOnce(
        new Error('Rate limited'),
      );

      await expect(
        channel.setTyping('tg:100200300', true),
      ).resolves.toBeUndefined();
    });
  });

  // --- Bot commands ---

  describe('bot commands', () => {
    it('/chatid replies with chat ID and metadata', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const handler = currentBot().commandHandlers.get('chatid')!;
      const ctx = {
        chat: { id: 100200300, type: 'group' as const },
        from: { first_name: 'Alice' },
        reply: vi.fn(),
      };

      await handler(ctx);

      expect(ctx.reply).toHaveBeenCalledWith(
        expect.stringContaining('tg:100200300'),
        expect.objectContaining({ parse_mode: 'HTML' }),
      );
    });

    it('/chatid shows chat type', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const handler = currentBot().commandHandlers.get('chatid')!;
      const ctx = {
        chat: { id: 555, type: 'private' as const },
        from: { first_name: 'Bob' },
        reply: vi.fn(),
      };

      await handler(ctx);

      expect(ctx.reply).toHaveBeenCalledWith(
        expect.stringContaining('private'),
        expect.any(Object),
      );
    });

    it('/ping replies with bot status', async () => {
      const opts = createTestOpts();
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();

      const handler = currentBot().commandHandlers.get('ping')!;
      const ctx = { reply: vi.fn() };

      await handler(ctx);

      expect(ctx.reply).toHaveBeenCalledWith('Andy is online.');
    });
  });

  // --- Channel properties ---

  describe('channel properties', () => {
    it('has name "telegram"', () => {
      const channel = new TelegramChannel('test-token', createTestOpts());
      expect(channel.name).toBe('telegram');
    });
  });

  // --- Trigger gate (#289) ---

  describe('trigger gate', () => {
    function makeRegisteredGroups(group: Record<string, unknown>) {
      return vi.fn(() => ({
        'tg:100200300': {
          name: 'Test Group',
          folder: 'test-group',
          trigger: '@Andy',
          added_at: '2024-01-01T00:00:00.000Z',
          ...group,
        },
      }));
    }

    it('main group: 👀 fires regardless of trigger match', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({ isMain: true }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      await triggerTextMessage(createTextCtx({ text: 'random chatter' }));

      expect(reactSpy).toHaveBeenCalledWith('tg:100200300', '1', '👀');
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });

    it('trusted + requires_trigger: no 👀 and no observer note when trigger absent', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: true,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      await triggerTextMessage(
        createTextCtx({ text: 'just chatting, no mention' }),
      );

      expect(reactSpy).not.toHaveBeenCalled();
      expect(noteLatestUserMessageMock).not.toHaveBeenCalled();
    });

    it('trusted + requires_trigger: 👀 fires when trigger word matches', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: true,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      await triggerTextMessage(
        createTextCtx({ text: '@Andy what time is it?' }),
      );

      expect(reactSpy).toHaveBeenCalledWith('tg:100200300', '1', '👀');
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });

    it('trusted + requires_trigger: 👀 fires when reply targets OUR bot', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: true,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      const ctx = createTextCtx({ text: 'sounds good' });
      (ctx.message as Record<string, unknown>).reply_to_message = {
        message_id: 42,
        date: 1735689600,
        chat: { id: 100200300, type: 'group' },
        from: {
          id: 12345,
          is_bot: true,
          username: 'andy_ai_bot',
          first_name: 'Andy',
        },
        text: 'previous bot reply',
      };
      await triggerTextMessage(ctx);

      expect(reactSpy).toHaveBeenCalledWith('tg:100200300', '1', '👀');
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });

    it('trusted + requires_trigger: ignores replies to OTHER bots', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: true,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      const ctx = createTextCtx({ text: 'thanks' });
      (ctx.message as Record<string, unknown>).reply_to_message = {
        message_id: 99,
        date: 1735689600,
        chat: { id: 100200300, type: 'group' },
        from: {
          id: 67890,
          is_bot: true,
          username: 'someone_else_bot',
          first_name: 'OtherBot',
        },
        text: 'unrelated bot output',
      };
      await triggerTextMessage(ctx);

      // Orchestrator's isReplyToBot only routes for our bot — the host
      // gate must mirror that or we leak 👀 onto cross-bot threads
      // the orchestrator drops.
      expect(reactSpy).not.toHaveBeenCalled();
      expect(noteLatestUserMessageMock).not.toHaveBeenCalled();
    });

    it('untrusted + requires_trigger: no 👀 from host even when trigger matches', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          requiresTrigger: true,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      await triggerTextMessage(createTextCtx({ text: '@Andy hello' }));

      // Host stays out of untrusted reactions; the agent's
      // bad-actor-disengage rule decides whether to react later.
      expect(reactSpy).not.toHaveBeenCalled();
      // ...but the observer still needs to know about an addressed
      // message so progress emojis can attach if the agent engages.
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });

    it('trusted + requires_trigger=false (group): no 👀 on bystander chatter', async () => {
      // Decouple regression. `requires_trigger=false` lets the agent
      // reason about every message but the host must NOT visibly
      // react to bystander traffic — that's the exact leak from the
      // multi-bot test chat in the original PR.
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: false,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      await triggerTextMessage(createTextCtx({ text: 'small talk' }));

      expect(reactSpy).not.toHaveBeenCalled();
      expect(noteLatestUserMessageMock).not.toHaveBeenCalled();
    });

    it('trusted + requires_trigger=false (group): 👀 fires when trigger matches', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: false,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      await triggerTextMessage(createTextCtx({ text: '@Andy do the thing' }));

      expect(reactSpy).toHaveBeenCalledWith('tg:100200300', '1', '👀');
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });

    it('trusted + requires_trigger=false (multi-bot): bystander reply to other bot stays silent', async () => {
      // Direct reproduction of the post-deploy report from the
      // `Old.wtf` test chat — `requires_trigger=false`, two bots in
      // the room, user replies to the OTHER bot.
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: false,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      const ctx = createTextCtx({ text: 'thanks' });
      (ctx.message as Record<string, unknown>).reply_to_message = {
        message_id: 99,
        date: 1735689600,
        chat: { id: 100200300, type: 'group' },
        from: {
          id: 67890,
          is_bot: true,
          username: 'someone_else_bot',
          first_name: 'OtherBot',
        },
        text: 'unrelated bot output',
      };
      await triggerTextMessage(ctx);

      expect(reactSpy).not.toHaveBeenCalled();
      expect(noteLatestUserMessageMock).not.toHaveBeenCalled();
    });

    it('private chat (1:1 DM, trusted): 👀 fires without trigger — every message is for us', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: false,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      await triggerTextMessage(
        createTextCtx({ text: 'hey', chatType: 'private' }),
      );

      expect(reactSpy).toHaveBeenCalledWith('tg:100200300', '1', '👀');
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });

    it('private chat (1:1 DM, untrusted): observer notes but no host 👀', async () => {
      // Untrusted DM is unusual but supported. The chat-type
      // short-circuit makes triggerHit true (every solo inbound is
      // for us), so the observer can attach progress emojis if the
      // agent engages. The 👀 ack still requires main/trusted, so
      // host stays silent and the agent decides.
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          requiresTrigger: false,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      await triggerTextMessage(
        createTextCtx({ text: 'hello?', chatType: 'private' }),
      );

      expect(reactSpy).not.toHaveBeenCalled();
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });

    // Voice messages exercise the same gate after transcription. With
    // no transcription mock the `transcribeVoice` path throws and
    // `content` resolves to '[Voice message - transcription failed]',
    // which deliberately fails the trigger pattern — that's the
    // "transcript didn't address us" signal we want to gate on.
    it('voice — main group: 👀 fires regardless of transcript', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({ isMain: true }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      const ctx = createMediaCtx({
        extra: { voice: { file_id: 'v1' } },
      });
      await triggerMediaMessage('message:voice', ctx);

      expect(reactSpy).toHaveBeenCalledWith('tg:100200300', '1', '👀');
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });

    it('voice — trusted + requires_trigger: no 👀 when transcript lacks trigger', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: true,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      const ctx = createMediaCtx({
        extra: { voice: { file_id: 'v1' } },
      });
      await triggerMediaMessage('message:voice', ctx);

      expect(reactSpy).not.toHaveBeenCalled();
      expect(noteLatestUserMessageMock).not.toHaveBeenCalled();
    });

    it('voice — trusted + requires_trigger=false (group): no 👀 when transcript is silent on us', async () => {
      // Same decouple as the text path — `requires_trigger=false`
      // does NOT short-circuit the host gate. Transcription fails in
      // tests, so the resolved content holds no trigger and no reply
      // signal; gate stays silent.
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: false,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      const ctx = createMediaCtx({
        extra: { voice: { file_id: 'v1' } },
      });
      await triggerMediaMessage('message:voice', ctx);

      expect(reactSpy).not.toHaveBeenCalled();
      expect(noteLatestUserMessageMock).not.toHaveBeenCalled();
    });

    it('voice — private chat: 👀 fires without trigger', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          containerConfig: { trusted: true },
          requiresTrigger: false,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      const ctx = createMediaCtx({
        chatType: 'private',
        extra: { voice: { file_id: 'v1' } },
      });
      await triggerMediaMessage('message:voice', ctx);

      expect(reactSpy).toHaveBeenCalledWith('tg:100200300', '1', '👀');
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });

    it('voice — untrusted + requires_trigger: no 👀 from host even when trigger would match', async () => {
      const opts = createTestOpts({
        registeredGroups: makeRegisteredGroups({
          requiresTrigger: true,
        }),
      });
      const channel = new TelegramChannel('test-token', opts);
      await channel.connect();
      const reactSpy = vi
        .spyOn(channel, 'sendReaction')
        .mockResolvedValue(undefined);

      // Reply-to-our-bot satisfies triggerHit even on a failed
      // transcript, so we exercise the untrusted leak surface
      // without depending on a transcription mock.
      const ctx = createMediaCtx({
        extra: {
          voice: { file_id: 'v1' },
          reply_to_message: {
            message_id: 42,
            date: Math.floor(Date.now() / 1000),
            chat: { id: 100200300, type: 'group' },
            from: {
              id: 12345,
              is_bot: true,
              username: 'andy_ai_bot',
              first_name: 'Andy',
            },
            text: 'previous bot reply',
          },
        },
      });
      (ctx as Record<string, unknown>).me = { username: 'andy_ai_bot' };
      await triggerMediaMessage('message:voice', ctx);

      expect(reactSpy).not.toHaveBeenCalled();
      // ...but the observer still needs to know about an addressed
      // message so progress emojis can attach if the agent engages.
      expect(noteLatestUserMessageMock).toHaveBeenCalledWith(
        'tg:100200300',
        '1',
      );
    });
  });
});

// --- splitMessage HTML-aware boundaries (#286) + sanitize-then-split (#282) ---
//
// `splitMessage` runs on already-sanitized HTML now (callers
// `TelegramChannel.sendMessage` + `sendPoolMessage` sanitize once
// before chunking, see #282). The function must produce chunks that
// each parse as valid Telegram HTML — never cut inside `<...>`,
// never orphan an opening tag from its closing.

describe('splitMessage — HTML-aware boundaries (#286)', () => {
  const MAX_LENGTH = 4096;

  it('returns input unchanged when below MAX_LENGTH', () => {
    const text = 'short message';
    expect(splitMessage(text)).toEqual([text]);
  });

  it('never cuts inside an HTML tag — bold span at boundary kept intact', () => {
    // Bold span fits well within one chunk (body = 2000 chars).
    // Trailing prose pushes total past MAX_LENGTH to force a split.
    // The split MUST land at a depth-0 boundary, never inside the
    // `<b>...</b>` body and never between `<b>` and `</b>`.
    const text = `aa <b>${'x'.repeat(2000)}</b> bb ${'.'.repeat(3000)}`;
    const chunks = splitMessage(text);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      // Tag balance: opens == closes in each chunk.
      const opens = (c.match(/<b>/g) || []).length;
      const closes = (c.match(/<\/b>/g) || []).length;
      expect(opens).toBe(closes);
      // No chunk ends inside a tag (last `<` after last `>` would
      // mean an unterminated tag).
      const lastLt = c.lastIndexOf('<');
      const lastGt = c.lastIndexOf('>');
      expect(lastLt).toBeLessThanOrEqual(lastGt);
    }
  });

  it('never cuts inside a link tag — long href near boundary', () => {
    // The `<a href="...">` opening tag itself is long. A naive
    // boundary check that only forbade splits between `<` and `>`
    // would still allow a split between the opening `<a ...>` and
    // its closing `</a>`. HTML-aware must reject both.
    const head = 'a'.repeat(4000);
    const longUrl = 'https://example.com/' + 'p'.repeat(80);
    const text = `${head} <a href="${longUrl}">label</a> tail`;
    const chunks = splitMessage(text);
    // No chunk may contain an unbalanced `<a` / `</a>`.
    for (const c of chunks) {
      const opens = (c.match(/<a\s/g) || []).length;
      const closes = (c.match(/<\/a>/g) || []).length;
      expect(opens).toBe(closes);
      // No chunk ends inside a tag (i.e. last `<` after last `>`
      // would mean unterminated tag).
      const lastLt = c.lastIndexOf('<');
      const lastGt = c.lastIndexOf('>');
      expect(lastLt).toBeLessThanOrEqual(lastGt);
    }
  });

  it('prefers `</pre>` boundary for long fenced-code', () => {
    // After sanitize, fenced-code becomes `<pre>...</pre>`. The
    // `</pre>` boundary is a Phase-1a-equivalent safe split point.
    const codeBody = 'c'.repeat(2000);
    const after = ' after the code\n' + 't'.repeat(2200);
    const text = `<pre>${codeBody}</pre>${after}`;
    const chunks = splitMessage(text);
    // First chunk should end at `</pre>` (or `</pre>\n`), not
    // mid-content of the trailing prose.
    expect(chunks[0].endsWith('</pre>')).toBe(true);
  });

  it('lowers paragraph threshold so a clean `\\n\\n` at byte 1100 is preferred over hard cut (#286)', () => {
    // Pre-fix the 30%-of-MAX_LENGTH minimum (1228) rejected a
    // paragraph break at byte 1100 and fell through to a space or
    // hard cut. New 5% threshold (204) accepts it.
    const head = 'p'.repeat(1099) + '\n\n';
    const tail = 'q'.repeat(MAX_LENGTH);
    const text = head + tail;
    const chunks = splitMessage(text);
    // Chunk 1 ends right after the `\n\n` (byte 1101 — `\n\n` at
    // 1099-1100, end-position 1101).
    expect(chunks[0].length).toBe(1101);
    expect(chunks[0].endsWith('\n\n')).toBe(true);
  });

  it('lastSafeIndexOf never returns end > MAX_LENGTH (PR #308 review)', () => {
    // Regression: pre-fix `lastIndexOf('\n', MAX_LENGTH)` could find
    // a `\n` starting at byte MAX_LENGTH and return end-position
    // MAX_LENGTH + 1, producing a chunk of MAX_LENGTH + 1 chars —
    // over Telegram's limit. The fix searches from
    // `at - needle.length`, so end is bounded by `at`.
    const head = 'a'.repeat(MAX_LENGTH); // bytes 0..MAX_LENGTH-1
    const text = head + '\n' + 'b'.repeat(2000); // `\n` at byte MAX_LENGTH
    const chunks = splitMessage(text);
    // No chunk may exceed MAX_LENGTH.
    for (const c of chunks) {
      expect(c.length).toBeLessThanOrEqual(MAX_LENGTH);
    }
  });

  it('hard-cuts as last resort when the entire input is one unsplittable HTML span', () => {
    // A single `<pre>...</pre>` block longer than MAX_LENGTH has
    // no depth-0 split position. The function falls back to a hard
    // cut at MAX_LENGTH; the resulting chunk's HTML is technically
    // broken, and the sanitizer's plain-text fallback handles it.
    const text = `<pre>${'z'.repeat(MAX_LENGTH * 2)}</pre>`;
    const chunks = splitMessage(text);
    expect(chunks.length).toBeGreaterThan(1);
    // First chunk size respects MAX_LENGTH even when no safe
    // position exists.
    expect(chunks[0].length).toBeLessThanOrEqual(MAX_LENGTH);
  });
});

describe('TelegramChannel.sendMessage — sanitize-then-split contract (#282)', () => {
  it('sanitizes ONCE before splitting — chunks are HTML, not raw markdown', async () => {
    const opts = createTestOpts();
    const channel = new TelegramChannel('test-token', opts);
    await channel.connect();

    // Below MAX_LENGTH so it produces one chunk; the assertion is
    // that the chunk delivered to the API is sanitized HTML.
    await channel.sendMessage(
      'tg:100200300',
      'see [Docs](https://example.com) with **bold** here',
    );

    const sentArgs = currentBot().api.sendMessage.mock.calls[0];
    expect(sentArgs[1]).toBe(
      'see <a href="https://example.com">Docs</a> with <b>bold</b> here',
    );
    // parse_mode is HTML on the first attempt (no fallback fired).
    expect(sentArgs[2]?.parse_mode).toBe('HTML');
    // The internal preSanitized marker MUST NOT be smuggled into
    // the API options object — Telegram would reject unknown
    // fields strictly, and this is host-side bookkeeping.
    expect(sentArgs[2]?.preSanitized).toBeUndefined();
  });

  it('long markdown link split across chunks renders as a single link in chunk 2 (no half-construct)', async () => {
    // Pre-fix (split-then-sanitize), splitting between `]` and `(`
    // of a markdown link left chunk 1 ending with `]` and chunk 2
    // starting with `(https://…)` — neither half matched the
    // sanitizer's link regex. Post-fix, sanitize runs on the whole
    // input first, the link becomes `<a href="…">…</a>`, and
    // splitMessage's HTML-awareness keeps the tag intact in one
    // chunk.
    const opts = createTestOpts();
    const channel = new TelegramChannel('test-token', opts);
    await channel.connect();

    const head = 'h'.repeat(4080);
    const text = `${head} [click](https://example.com/p) tail`;
    await channel.sendMessage('tg:100200300', text);

    const calls = currentBot().api.sendMessage.mock.calls;
    expect(calls.length).toBeGreaterThanOrEqual(2);
    // Whichever chunk contains the link must contain the WHOLE
    // `<a>` tag, not a fragment.
    const allChunks = calls.map((c: any[]) => c[1] as string).join(' ');
    expect(
      allChunks.includes('<a href="https://example.com/p">click</a>'),
    ).toBe(true);
    // Defensive: no chunk has an unterminated `<a` opening.
    for (const c of calls) {
      const chunk = c[1] as string;
      const opens = (chunk.match(/<a\s/g) || []).length;
      const closes = (chunk.match(/<\/a>/g) || []).length;
      expect(opens).toBe(closes);
    }
  });
});
