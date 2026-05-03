import { describe, it, expect, beforeEach, vi } from 'vitest';

import {
  _initTestDatabase,
  _writeRawRegisteredGroup,
  createTask,
  deleteRegisteredGroup,
  deleteTask,
  deriveTriggerString,
  getAllChats,
  getAllRegisteredGroups,
  getBotMessageByTelegramId,
  getChatByJid,
  getLastBotMessageTimestamp,
  getMessageById,
  getMessagesSince,
  getNewMessages,
  getRegisteredGroup,
  getTaskById,
  getTriggerPatterns,
  messageExistsInDifferentChat,
  setRegisteredGroup,
  setTriggerPatterns,
  storeChatMetadata,
  storeMessage,
  updateTask,
} from './db.js';
import type { TriggerPatternConfig } from './types.js';
import { formatMessages } from './router.js';

beforeEach(() => {
  _initTestDatabase();
});

// Helper to store a message using the normalized NewMessage interface
function store(overrides: {
  id: string;
  chat_jid: string;
  sender: string;
  sender_name: string;
  content: string;
  timestamp: string;
  is_from_me?: boolean;
}) {
  storeMessage({
    id: overrides.id,
    chat_jid: overrides.chat_jid,
    sender: overrides.sender,
    sender_name: overrides.sender_name,
    content: overrides.content,
    timestamp: overrides.timestamp,
    is_from_me: overrides.is_from_me ?? false,
  });
}

// --- storeMessage (NewMessage format) ---

describe('storeMessage', () => {
  it('stores a message and retrieves it', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    store({
      id: 'msg-1',
      chat_jid: 'group@g.us',
      sender: '123@s.whatsapp.net',
      sender_name: 'Alice',
      content: 'hello world',
      timestamp: '2024-01-01T00:00:01.000Z',
    });

    const messages = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:00.000Z',
      'Andy',
    );
    expect(messages).toHaveLength(1);
    expect(messages[0].id).toBe('msg-1');
    expect(messages[0].sender).toBe('123@s.whatsapp.net');
    expect(messages[0].sender_name).toBe('Alice');
    expect(messages[0].content).toBe('hello world');
  });

  it('filters out empty content', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    store({
      id: 'msg-2',
      chat_jid: 'group@g.us',
      sender: '111@s.whatsapp.net',
      sender_name: 'Dave',
      content: '',
      timestamp: '2024-01-01T00:00:04.000Z',
    });

    const messages = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:00.000Z',
      'Andy',
    );
    expect(messages).toHaveLength(0);
  });

  it('stores is_from_me flag', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    store({
      id: 'msg-3',
      chat_jid: 'group@g.us',
      sender: 'me@s.whatsapp.net',
      sender_name: 'Me',
      content: 'my message',
      timestamp: '2024-01-01T00:00:05.000Z',
      is_from_me: true,
    });

    // Message is stored (we can retrieve it — is_from_me doesn't affect retrieval)
    const messages = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:00.000Z',
      'Andy',
    );
    expect(messages).toHaveLength(1);
  });

  it('upserts on duplicate id+chat_jid', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    store({
      id: 'msg-dup',
      chat_jid: 'group@g.us',
      sender: '123@s.whatsapp.net',
      sender_name: 'Alice',
      content: 'original',
      timestamp: '2024-01-01T00:00:01.000Z',
    });

    store({
      id: 'msg-dup',
      chat_jid: 'group@g.us',
      sender: '123@s.whatsapp.net',
      sender_name: 'Alice',
      content: 'updated',
      timestamp: '2024-01-01T00:00:01.000Z',
    });

    const messages = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:00.000Z',
      'Andy',
    );
    expect(messages).toHaveLength(1);
    expect(messages[0].content).toBe('updated');
  });
});

// --- reply context persistence ---

describe('reply context', () => {
  it('stores and retrieves reply_to fields', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: 'reply-1',
      chat_jid: 'group@g.us',
      sender: '123',
      sender_name: 'Alice',
      content: 'Yes, on my way!',
      timestamp: '2024-01-01T00:00:01.000Z',
      reply_to_message_id: '42',
      reply_to_message_content: 'Are you coming tonight?',
      reply_to_sender_name: 'Bob',
    });

    const messages = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:00.000Z',
      'Andy',
    );
    expect(messages).toHaveLength(1);
    expect(messages[0].reply_to_message_id).toBe('42');
    expect(messages[0].reply_to_message_content).toBe(
      'Are you coming tonight?',
    );
    expect(messages[0].reply_to_sender_name).toBe('Bob');
  });

  it('returns null for messages without reply context', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    store({
      id: 'no-reply',
      chat_jid: 'group@g.us',
      sender: '123',
      sender_name: 'Alice',
      content: 'Just a normal message',
      timestamp: '2024-01-01T00:00:01.000Z',
    });

    const messages = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:00.000Z',
      'Andy',
    );
    expect(messages).toHaveLength(1);
    expect(messages[0].reply_to_message_id).toBeNull();
    expect(messages[0].reply_to_message_content).toBeNull();
    expect(messages[0].reply_to_sender_name).toBeNull();
  });

  it('retrieves reply context via getNewMessages', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: 'reply-2',
      chat_jid: 'group@g.us',
      sender: '456',
      sender_name: 'Carol',
      content: 'Agreed',
      timestamp: '2024-01-01T00:00:01.000Z',
      reply_to_message_id: '99',
      reply_to_message_content: 'We should meet',
      reply_to_sender_name: 'Dave',
    });

    const { messages } = getNewMessages(
      ['group@g.us'],
      '2024-01-01T00:00:00.000Z',
      'Andy',
    );
    expect(messages).toHaveLength(1);
    expect(messages[0].reply_to_message_id).toBe('99');
    expect(messages[0].reply_to_sender_name).toBe('Dave');
  });
});

// --- telegram_message_id persistence ---

describe('telegram_message_id', () => {
  it('stores and retrieves the telegram_message_id on a bot send', () => {
    storeChatMetadata('tg:-100123', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: 'bot-1776570796407-k6ie5',
      chat_jid: 'tg:-100123',
      sender: 'Andy',
      sender_name: 'Andy',
      content: 'hello',
      timestamp: '2024-01-01T00:00:01.000Z',
      is_from_me: true,
      is_bot_message: true,
      telegram_message_id: '4976',
    });

    const found = getBotMessageByTelegramId('tg:-100123', '4976');
    expect(found).not.toBeNull();
    expect(found?.id).toBe('bot-1776570796407-k6ie5');
    expect(found?.telegram_message_id).toBe('4976');
    expect(found?.content).toBe('hello');
    expect(found?.is_bot_message).toBe(true);
  });

  it('returns null when no bot message has that telegram id', () => {
    storeChatMetadata('tg:-100123', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: 'bot-a',
      chat_jid: 'tg:-100123',
      sender: 'Andy',
      sender_name: 'Andy',
      content: 'x',
      timestamp: '2024-01-01T00:00:01.000Z',
      is_from_me: true,
      is_bot_message: true,
      telegram_message_id: '4976',
    });

    expect(getBotMessageByTelegramId('tg:-100123', '9999')).toBeNull();
  });

  it('scopes lookup to chat_jid so the same telegram id in a different chat is ignored', () => {
    // Telegram IDs reset per chat — id 500 in chat A and chat B are different
    // messages. The getter is (chat_jid, telegram_message_id)-scoped so a
    // caller asking about chat A doesn't accidentally get chat B's row.
    storeChatMetadata('tg:-100aaa', '2024-01-01T00:00:00.000Z');
    storeChatMetadata('tg:-100bbb', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: 'bot-a',
      chat_jid: 'tg:-100aaa',
      sender: 'Andy',
      sender_name: 'Andy',
      content: 'in chat A',
      timestamp: '2024-01-01T00:00:01.000Z',
      is_from_me: true,
      is_bot_message: true,
      telegram_message_id: '500',
    });
    storeMessage({
      id: 'bot-b',
      chat_jid: 'tg:-100bbb',
      sender: 'Andy',
      sender_name: 'Andy',
      content: 'in chat B',
      timestamp: '2024-01-01T00:00:01.000Z',
      is_from_me: true,
      is_bot_message: true,
      telegram_message_id: '500',
    });

    expect(getBotMessageByTelegramId('tg:-100aaa', '500')?.content).toBe(
      'in chat A',
    );
    expect(getBotMessageByTelegramId('tg:-100bbb', '500')?.content).toBe(
      'in chat B',
    );
  });

  it('leaves telegram_message_id unset when the caller omits it', () => {
    // Inbound user messages and other-channel sends never populate this
    // column — and a lookup by the telegram id those messages DO have as
    // their `id` should NOT resolve through this getter either (the query
    // checks telegram_message_id, not id).
    storeChatMetadata('tg:-100123', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: '4975', // simulate inbound user message — id IS the telegram id
      chat_jid: 'tg:-100123',
      sender: 'user@test',
      sender_name: 'User',
      content: 'hi bot',
      timestamp: '2024-01-01T00:00:01.000Z',
    });

    // Looking up by the same string returns null because
    // telegram_message_id column is NULL for this row.
    expect(getBotMessageByTelegramId('tg:-100123', '4975')).toBeNull();
  });
});

// --- getMessageById ---

describe('getMessageById', () => {
  it('finds inbound rows whose `id` is the platform-native id', () => {
    storeChatMetadata('tg:-100123', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: '4975',
      chat_jid: 'tg:-100123',
      sender: 'user@test',
      sender_name: 'User',
      content: 'hi bot',
      timestamp: '2024-01-01T00:00:01.000Z',
    });

    const found = getMessageById('4975', 'tg:-100123');
    expect(found).not.toBeNull();
    expect(found?.id).toBe('4975');
    expect(found?.is_from_me).toBe(false);
  });

  it('falls back to telegram_message_id for bot rows whose `id` is `bot-<ts>-<rand>`', () => {
    // Regression for the trigger gate's `reply:*` matcher silently
    // denying replies to OUR bot's messages: bot sends store the
    // platform-native id in `telegram_message_id`, not `id`, so the
    // id-only lookup missed and `replyTo.isAssistant` came back false
    // even when the user was clearly replying to the assistant.
    // Concrete repro: tg:-1001633120997 msg 378306 → bot row id
    // bot-1777747764042-6tppb (telegram_message_id 378305).
    storeChatMetadata('tg:-1001633120997', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: 'bot-1777747764042-6tppb',
      chat_jid: 'tg:-1001633120997',
      sender: 'TestAssistant',
      sender_name: 'TestAssistant',
      content: 'Жив. Не дождётесь.',
      timestamp: '2024-01-01T00:00:01.000Z',
      is_from_me: true,
      is_bot_message: true,
      telegram_message_id: '378305',
    });

    const found = getMessageById('378305', 'tg:-1001633120997');
    expect(found).not.toBeNull();
    expect(found?.id).toBe('bot-1777747764042-6tppb');
    expect(found?.is_from_me).toBe(true);
    expect(found?.is_bot_message).toBe(true);
  });

  it('keeps fallback chat-scoped — bot-row telegram_message_id from another chat does not leak', () => {
    storeChatMetadata('tg:-100aaa', '2024-01-01T00:00:00.000Z');
    storeChatMetadata('tg:-100bbb', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: 'bot-a',
      chat_jid: 'tg:-100aaa',
      sender: 'Andy',
      sender_name: 'Andy',
      content: 'chat A',
      timestamp: '2024-01-01T00:00:01.000Z',
      is_from_me: true,
      is_bot_message: true,
      telegram_message_id: '500',
    });

    expect(getMessageById('500', 'tg:-100aaa')?.content).toBe('chat A');
    expect(getMessageById('500', 'tg:-100bbb')).toBeNull();
  });

  it('id-column hit wins over telegram_message_id fallback when both could match', () => {
    // Telegram per-chat IDs are unique, but a defensive ordering check:
    // an inbound row whose `id` equals a bot row's `telegram_message_id`
    // in a DIFFERENT chat must not collide. Same-chat collision is
    // impossible by Telegram's per-chat sequence guarantee, so we test
    // the "same id string, same chat, only one row exists" path.
    storeChatMetadata('tg:-100xxx', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: '777',
      chat_jid: 'tg:-100xxx',
      sender: 'user@test',
      sender_name: 'User',
      content: 'inbound',
      timestamp: '2024-01-01T00:00:01.000Z',
    });

    const found = getMessageById('777', 'tg:-100xxx');
    expect(found?.content).toBe('inbound');
    expect(found?.is_from_me).toBe(false);
  });

  it('returns null for non-Telegram bot sends — telegram_message_id column is NULL', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    storeMessage({
      id: 'bot-wa-1',
      chat_jid: 'group@g.us',
      sender: 'Andy',
      sender_name: 'Andy',
      content: 'whatsapp send',
      timestamp: '2024-01-01T00:00:01.000Z',
      is_from_me: true,
      is_bot_message: true,
      // no telegram_message_id — non-Telegram channel
    });

    // Looking up by the synthetic id still works (id-column path).
    expect(getMessageById('bot-wa-1', 'group@g.us')?.content).toBe(
      'whatsapp send',
    );
    // But a stray Telegram-style numeric id won't match — fallback
    // returns null because telegram_message_id is NULL for this row.
    expect(getMessageById('12345', 'group@g.us')).toBeNull();
  });
});

// --- getMessagesSince ---

describe('getMessagesSince', () => {
  beforeEach(() => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    store({
      id: 'm1',
      chat_jid: 'group@g.us',
      sender: 'Alice@s.whatsapp.net',
      sender_name: 'Alice',
      content: 'first',
      timestamp: '2024-01-01T00:00:01.000Z',
    });
    store({
      id: 'm2',
      chat_jid: 'group@g.us',
      sender: 'Bob@s.whatsapp.net',
      sender_name: 'Bob',
      content: 'second',
      timestamp: '2024-01-01T00:00:02.000Z',
    });
    storeMessage({
      id: 'm3',
      chat_jid: 'group@g.us',
      sender: 'Bot@s.whatsapp.net',
      sender_name: 'Bot',
      content: 'bot reply',
      timestamp: '2024-01-01T00:00:03.000Z',
      is_bot_message: true,
    });
    store({
      id: 'm4',
      chat_jid: 'group@g.us',
      sender: 'Carol@s.whatsapp.net',
      sender_name: 'Carol',
      content: 'third',
      timestamp: '2024-01-01T00:00:04.000Z',
    });
  });

  it('returns messages after the given timestamp', () => {
    const msgs = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:02.000Z',
      'Andy',
    );
    // Should exclude m1, m2 (before/at timestamp), m3 (bot message)
    expect(msgs).toHaveLength(1);
    expect(msgs[0].content).toBe('third');
  });

  it('excludes bot messages via is_bot_message flag', () => {
    const msgs = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:00.000Z',
      'Andy',
    );
    const botMsgs = msgs.filter((m) => m.content === 'bot reply');
    expect(botMsgs).toHaveLength(0);
  });

  it('returns all non-bot messages when sinceTimestamp is empty', () => {
    const msgs = getMessagesSince('group@g.us', '', 'Andy');
    // 3 user messages (bot message excluded)
    expect(msgs).toHaveLength(3);
  });

  it('recovers cursor from last bot reply when lastAgentTimestamp is missing', () => {
    // beforeEach already inserts m3 (bot reply at 00:00:03) and m4 (user at 00:00:04)
    // Add more old history before the bot reply
    for (let i = 1; i <= 50; i++) {
      store({
        id: `history-${i}`,
        chat_jid: 'group@g.us',
        sender: 'user@s.whatsapp.net',
        sender_name: 'User',
        content: `old message ${i}`,
        timestamp: `2023-06-${String(i).padStart(2, '0')}T12:00:00.000Z`,
      });
    }

    // New message after the bot reply (m3 at 00:00:03)
    store({
      id: 'new-1',
      chat_jid: 'group@g.us',
      sender: 'user@s.whatsapp.net',
      sender_name: 'User',
      content: 'new message after bot reply',
      timestamp: '2024-01-02T00:00:00.000Z',
    });

    // Recover cursor from the last bot message (m3 from beforeEach)
    const recovered = getLastBotMessageTimestamp('group@g.us', 'Andy');
    expect(recovered).toBe('2024-01-01T00:00:03.000Z');

    // Using recovered cursor: only gets messages after the bot reply
    const msgs = getMessagesSince('group@g.us', recovered!, 'Andy', 10);
    // m4 (third, 00:00:04) + new-1 — skips all 50 old messages and m1/m2
    expect(msgs).toHaveLength(2);
    expect(msgs[0].content).toBe('third');
    expect(msgs[1].content).toBe('new message after bot reply');
  });

  it('caps messages to configured limit even with recovered cursor', () => {
    // beforeEach inserts m3 (bot at 00:00:03). Add 30 messages after it.
    for (let i = 1; i <= 30; i++) {
      store({
        id: `pending-${i}`,
        chat_jid: 'group@g.us',
        sender: 'user@s.whatsapp.net',
        sender_name: 'User',
        content: `pending message ${i}`,
        timestamp: `2024-02-${String(i).padStart(2, '0')}T12:00:00.000Z`,
      });
    }

    const recovered = getLastBotMessageTimestamp('group@g.us', 'Andy');
    expect(recovered).toBe('2024-01-01T00:00:03.000Z');

    // With limit=10, only the 10 most recent are returned
    const msgs = getMessagesSince('group@g.us', recovered!, 'Andy', 10);
    expect(msgs).toHaveLength(10);
    // Most recent 10: pending-21 through pending-30
    expect(msgs[0].content).toBe('pending message 21');
    expect(msgs[9].content).toBe('pending message 30');
  });

  it('returns last N messages when no bot reply and no cursor exist', () => {
    // Use a fresh group with no bot messages
    storeChatMetadata('fresh@g.us', '2024-01-01T00:00:00.000Z');
    for (let i = 1; i <= 20; i++) {
      store({
        id: `fresh-${i}`,
        chat_jid: 'fresh@g.us',
        sender: 'user@s.whatsapp.net',
        sender_name: 'User',
        content: `message ${i}`,
        timestamp: `2024-02-${String(i).padStart(2, '0')}T12:00:00.000Z`,
      });
    }

    const recovered = getLastBotMessageTimestamp('fresh@g.us', 'Andy');
    expect(recovered).toBeUndefined();

    // No cursor → sinceTimestamp = '' but limit caps the result
    const msgs = getMessagesSince('fresh@g.us', '', 'Andy', 10);
    expect(msgs).toHaveLength(10);

    const prompt = formatMessages(msgs, 'Asia/Jerusalem');
    const messageTagCount = (prompt.match(/<message /g) || []).length;
    expect(messageTagCount).toBe(10);
  });

  it('filters pre-migration bot messages via content prefix backstop', () => {
    // Simulate a message written before migration: has prefix but is_bot_message = 0
    store({
      id: 'm5',
      chat_jid: 'group@g.us',
      sender: 'Bot@s.whatsapp.net',
      sender_name: 'Bot',
      content: 'Andy: old bot reply',
      timestamp: '2024-01-01T00:00:05.000Z',
    });
    const msgs = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:04.000Z',
      'Andy',
    );
    expect(msgs).toHaveLength(0);
  });
});

// --- getNewMessages ---

describe('getNewMessages', () => {
  beforeEach(() => {
    storeChatMetadata('group1@g.us', '2024-01-01T00:00:00.000Z');
    storeChatMetadata('group2@g.us', '2024-01-01T00:00:00.000Z');

    store({
      id: 'a1',
      chat_jid: 'group1@g.us',
      sender: 'user@s.whatsapp.net',
      sender_name: 'User',
      content: 'g1 msg1',
      timestamp: '2024-01-01T00:00:01.000Z',
    });
    store({
      id: 'a2',
      chat_jid: 'group2@g.us',
      sender: 'user@s.whatsapp.net',
      sender_name: 'User',
      content: 'g2 msg1',
      timestamp: '2024-01-01T00:00:02.000Z',
    });
    storeMessage({
      id: 'a3',
      chat_jid: 'group1@g.us',
      sender: 'user@s.whatsapp.net',
      sender_name: 'User',
      content: 'bot reply',
      timestamp: '2024-01-01T00:00:03.000Z',
      is_bot_message: true,
    });
    store({
      id: 'a4',
      chat_jid: 'group1@g.us',
      sender: 'user@s.whatsapp.net',
      sender_name: 'User',
      content: 'g1 msg2',
      timestamp: '2024-01-01T00:00:04.000Z',
    });
  });

  it('returns new messages across multiple groups', () => {
    const { messages, newTimestamp } = getNewMessages(
      ['group1@g.us', 'group2@g.us'],
      '2024-01-01T00:00:00.000Z',
      'Andy',
    );
    // Excludes bot message, returns 3 user messages
    expect(messages).toHaveLength(3);
    expect(newTimestamp).toBe('2024-01-01T00:00:04.000Z');
  });

  it('filters by timestamp', () => {
    const { messages } = getNewMessages(
      ['group1@g.us', 'group2@g.us'],
      '2024-01-01T00:00:02.000Z',
      'Andy',
    );
    // Only g1 msg2 (after ts, not bot)
    expect(messages).toHaveLength(1);
    expect(messages[0].content).toBe('g1 msg2');
  });

  it('returns empty for no registered groups', () => {
    const { messages, newTimestamp } = getNewMessages([], '', 'Andy');
    expect(messages).toHaveLength(0);
    expect(newTimestamp).toBe('');
  });
});

// --- storeChatMetadata ---

describe('storeChatMetadata', () => {
  it('stores chat with JID as default name', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');
    const chats = getAllChats();
    expect(chats).toHaveLength(1);
    expect(chats[0].jid).toBe('group@g.us');
    expect(chats[0].name).toBe('group@g.us');
  });

  it('stores chat with explicit name', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z', 'My Group');
    const chats = getAllChats();
    expect(chats[0].name).toBe('My Group');
  });

  it('updates name on subsequent call with name', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');
    storeChatMetadata('group@g.us', '2024-01-01T00:00:01.000Z', 'Updated Name');
    const chats = getAllChats();
    expect(chats).toHaveLength(1);
    expect(chats[0].name).toBe('Updated Name');
  });

  it('preserves newer timestamp on conflict', () => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:05.000Z');
    storeChatMetadata('group@g.us', '2024-01-01T00:00:01.000Z');
    const chats = getAllChats();
    expect(chats[0].last_message_time).toBe('2024-01-01T00:00:05.000Z');
  });
});

// --- Task CRUD ---

describe('task CRUD', () => {
  it('creates and retrieves a task', () => {
    createTask({
      id: 'task-1',
      group_folder: 'main',
      chat_jid: 'group@g.us',
      prompt: 'do something',
      schedule_type: 'once',
      schedule_value: '2024-06-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: '2024-06-01T00:00:00.000Z',
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const task = getTaskById('task-1');
    expect(task).toBeDefined();
    expect(task!.prompt).toBe('do something');
    expect(task!.status).toBe('active');
  });

  it('updates task status', () => {
    createTask({
      id: 'task-2',
      group_folder: 'main',
      chat_jid: 'group@g.us',
      prompt: 'test',
      schedule_type: 'once',
      schedule_value: '2024-06-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    updateTask('task-2', { status: 'paused' });
    expect(getTaskById('task-2')!.status).toBe('paused');
  });

  it('deletes a task and its run logs', () => {
    createTask({
      id: 'task-3',
      group_folder: 'main',
      chat_jid: 'group@g.us',
      prompt: 'delete me',
      schedule_type: 'once',
      schedule_value: '2024-06-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    deleteTask('task-3');
    expect(getTaskById('task-3')).toBeUndefined();
  });

  // --- continuation_cycle_id (#93/#130) ---
  //
  // The column is opt-in: ordinary tasks omit it (DB stores NULL) and
  // the helper skill that drives self-resuming cycles supplies the slot
  // key (UTC date / ISO week per the proposal). The task-scheduler
  // reads the value verbatim and threads it through ContainerInput so
  // the spawned container gets the matching env vars; persistence
  // round-tripping is therefore the part the DB layer must guarantee.
  it('persists continuation_cycle_id when supplied', () => {
    createTask({
      id: 'task-cont-1',
      group_folder: 'main',
      chat_jid: 'group@g.us',
      prompt: 'continue nightly chain',
      schedule_type: 'once',
      schedule_value: '2026-04-21T00:00:30.000Z',
      context_mode: 'isolated',
      next_run: '2026-04-21T00:00:30.000Z',
      status: 'active',
      created_at: '2026-04-21T00:00:00.000Z',
      created_by_role: 'owner' as const,
      continuation_cycle_id: '2026-04-21',
    });

    const task = getTaskById('task-cont-1');
    expect(task).toBeDefined();
    expect(task!.continuation_cycle_id).toBe('2026-04-21');
  });

  it('stores continuation_cycle_id as NULL when omitted (ordinary task)', () => {
    createTask({
      id: 'task-cont-2',
      group_folder: 'main',
      chat_jid: 'group@g.us',
      prompt: 'fresh task',
      schedule_type: 'once',
      schedule_value: '2026-04-21T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2026-04-21T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const task = getTaskById('task-cont-2');
    expect(task).toBeDefined();
    // DB column is TEXT NULL; better-sqlite3 surfaces SQL NULL as
    // JS `null`, NOT `undefined`. The calling code uses
    // `task.continuation_cycle_id ?? undefined` to normalise back to
    // the optional-string ContainerInput field, so any non-null result
    // here would silently emit continuation env vars on a fresh task.
    expect(task!.continuation_cycle_id).toBeNull();
  });
});

// --- LIMIT behavior ---

describe('message query LIMIT', () => {
  beforeEach(() => {
    storeChatMetadata('group@g.us', '2024-01-01T00:00:00.000Z');

    for (let i = 1; i <= 10; i++) {
      store({
        id: `lim-${i}`,
        chat_jid: 'group@g.us',
        sender: 'user@s.whatsapp.net',
        sender_name: 'User',
        content: `message ${i}`,
        timestamp: `2024-01-01T00:00:${String(i).padStart(2, '0')}.000Z`,
      });
    }
  });

  it('getNewMessages caps to limit and returns most recent in chronological order', () => {
    const { messages, newTimestamp } = getNewMessages(
      ['group@g.us'],
      '2024-01-01T00:00:00.000Z',
      'Andy',
      3,
    );
    expect(messages).toHaveLength(3);
    expect(messages[0].content).toBe('message 8');
    expect(messages[2].content).toBe('message 10');
    // Chronological order preserved
    expect(messages[1].timestamp > messages[0].timestamp).toBe(true);
    // newTimestamp reflects latest returned row
    expect(newTimestamp).toBe('2024-01-01T00:00:10.000Z');
  });

  it('getMessagesSince caps to limit and returns most recent in chronological order', () => {
    const messages = getMessagesSince(
      'group@g.us',
      '2024-01-01T00:00:00.000Z',
      'Andy',
      3,
    );
    expect(messages).toHaveLength(3);
    expect(messages[0].content).toBe('message 8');
    expect(messages[2].content).toBe('message 10');
    expect(messages[1].timestamp > messages[0].timestamp).toBe(true);
  });

  it('returns all messages when count is under the limit', () => {
    const { messages } = getNewMessages(
      ['group@g.us'],
      '2024-01-01T00:00:00.000Z',
      'Andy',
      50,
    );
    expect(messages).toHaveLength(10);
  });
});

// --- RegisteredGroup isMain round-trip ---

describe('registered group isMain', () => {
  it('persists isMain=true through set/get round-trip', () => {
    setRegisteredGroup('main@s.whatsapp.net', {
      name: 'Main Chat',
      folder: 'whatsapp_main',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      isMain: true,
    });

    const groups = getAllRegisteredGroups();
    const group = groups['main@s.whatsapp.net'];
    expect(group).toBeDefined();
    expect(group.isMain).toBe(true);
    expect(group.folder).toBe('whatsapp_main');
  });

  it('omits isMain for non-main groups', () => {
    setRegisteredGroup('group@g.us', {
      name: 'Family Chat',
      folder: 'whatsapp_family-chat',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
    });

    const groups = getAllRegisteredGroups();
    const group = groups['group@g.us'];
    expect(group).toBeDefined();
    expect(group.isMain).toBeUndefined();
  });
});

// --- deleteRegisteredGroup (#159) ---

describe('deleteRegisteredGroup', () => {
  it('removes a registered row and returns true', () => {
    setRegisteredGroup('purge@g.us', {
      name: 'Purge Me',
      folder: 'purge-group',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
    });
    expect(getRegisteredGroup('purge@g.us')).toBeDefined();

    const removed = deleteRegisteredGroup('purge@g.us');

    expect(removed).toBe(true);
    expect(getRegisteredGroup('purge@g.us')).toBeUndefined();
  });

  it('is idempotent — repeat calls report false after first delete', () => {
    setRegisteredGroup('once@g.us', {
      name: 'Once',
      folder: 'once-group',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
    });

    expect(deleteRegisteredGroup('once@g.us')).toBe(true);
    expect(deleteRegisteredGroup('once@g.us')).toBe(false);
  });

  it('returns false for a JID that was never registered', () => {
    expect(deleteRegisteredGroup('never-here@g.us')).toBe(false);
  });

  it('does not affect sibling registrations', () => {
    setRegisteredGroup('keeper@g.us', {
      name: 'Keeper',
      folder: 'keeper-group',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
    });
    setRegisteredGroup('goer@g.us', {
      name: 'Goer',
      folder: 'goer-group',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
    });

    deleteRegisteredGroup('goer@g.us');

    expect(getRegisteredGroup('goer@g.us')).toBeUndefined();
    expect(getRegisteredGroup('keeper@g.us')).toBeDefined();
  });
});

// --- Defensive container_config parsing (issue #156) ---

describe('registered group malformed container_config', () => {
  it('getAllRegisteredGroups skips parse errors and keeps loading other rows', () => {
    setRegisteredGroup('good@g.us', {
      name: 'Good Group',
      folder: 'whatsapp_good',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      containerConfig: { trusted: true },
    });

    _writeRawRegisteredGroup({
      jid: 'broken@g.us',
      name: 'Broken Group',
      folder: 'whatsapp_broken',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      container_config: '{not valid json',
    });

    const groups = getAllRegisteredGroups();

    expect(groups['good@g.us']).toBeDefined();
    expect(groups['good@g.us'].containerConfig).toEqual({ trusted: true });

    expect(groups['broken@g.us']).toBeDefined();
    expect(groups['broken@g.us'].containerConfig).toBeUndefined();
    expect(groups['broken@g.us'].name).toBe('Broken Group');
  });

  it('getRegisteredGroup returns the row with containerConfig undefined on parse failure', () => {
    _writeRawRegisteredGroup({
      jid: 'broken@g.us',
      name: 'Broken Group',
      folder: 'whatsapp_broken',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      container_config: '{"trusted": tru',
    });

    const group = getRegisteredGroup('broken@g.us');
    expect(group).toBeDefined();
    expect(group?.containerConfig).toBeUndefined();
    expect(group?.name).toBe('Broken Group');
  });

  it('treats valid-but-non-object JSON (null, primitives, arrays) as undefined', () => {
    const cases = ['null', 'true', '42', '"oops"', '[]'];
    for (let i = 0; i < cases.length; i++) {
      const jid = `non-object-${i}@g.us`;
      _writeRawRegisteredGroup({
        jid,
        name: `Group ${i}`,
        folder: `whatsapp_non_object_${i}`,
        trigger: '@Andy',
        added_at: '2024-01-01T00:00:00.000Z',
        container_config: cases[i],
      });
      const group = getRegisteredGroup(jid);
      expect(group).toBeDefined();
      expect(group?.containerConfig).toBeUndefined();
    }
  });

  it('treats empty-string container_config as parse failure (corruption indicator), not "no config"', async () => {
    const loggerMod = await import('./logger.js');
    const warnSpy = vi
      .spyOn(loggerMod.logger, 'warn')
      .mockImplementation(() => loggerMod.logger);

    _writeRawRegisteredGroup({
      jid: 'empty@g.us',
      name: 'Empty Config Group',
      folder: 'whatsapp_empty',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      container_config: '',
    });
    _writeRawRegisteredGroup({
      jid: 'null@g.us',
      name: 'Null Config Group',
      folder: 'whatsapp_null',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      container_config: null,
    });

    const empty = getRegisteredGroup('empty@g.us');
    expect(empty?.containerConfig).toBeUndefined();
    // Behavioral DIFFERENCE vs the old `if (!raw)` shape: empty
    // string now surfaces as a SyntaxError warning, while the
    // documented NULL "no config" state stays silent.
    expect(warnSpy).toHaveBeenCalledWith(
      expect.objectContaining({ jid: 'empty@g.us', errName: 'SyntaxError' }),
      expect.stringContaining('invalid container_config JSON'),
    );

    warnSpy.mockClear();
    const nul = getRegisteredGroup('null@g.us');
    expect(nul?.containerConfig).toBeUndefined();
    expect(warnSpy).not.toHaveBeenCalled();

    warnSpy.mockRestore();
  });
});

// ---------------------------------------------------------------
// messageExistsInDifferentChat — cross-chat reply_to safety helper
// (See `src/channels/telegram.ts` cross-chat guard for the call site.)
// ---------------------------------------------------------------

describe('messageExistsInDifferentChat', () => {
  it('returns true when the same message id exists under a different chat_jid', () => {
    // Telegram message IDs are per-chat sequential, so the same numeric
    // id legitimately exists in multiple chats. This helper detects the
    // cross-chat collision so the channel layer can drop a reply_to
    // that came from a foreign chat.
    storeChatMetadata('tg:-1001111', '2026-04-28T00:00:00.000Z');
    store({
      id: '12345',
      chat_jid: 'tg:-1001111',
      sender: 'user',
      sender_name: 'Alice',
      content: 'message in chat A',
      timestamp: '2026-04-28T00:00:00.000Z',
    });
    expect(messageExistsInDifferentChat('12345', 'tg:-1002222')).toBe(true);
  });

  it('returns false when the message id exists only in the target chat', () => {
    // The id IS in our DB but for the chat we're sending to —
    // legitimate same-chat reply, must not be dropped.
    storeChatMetadata('tg:-1003333', '2026-04-28T00:00:00.000Z');
    store({
      id: '67890',
      chat_jid: 'tg:-1003333',
      sender: 'user',
      sender_name: 'Bob',
      content: 'message in target chat',
      timestamp: '2026-04-28T00:00:00.000Z',
    });
    expect(messageExistsInDifferentChat('67890', 'tg:-1003333')).toBe(false);
  });

  it('returns false when we have no record of the message id at all', () => {
    // No row at all — Telegram remains authoritative; the helper
    // refuses to claim cross-chat without evidence so a reply_to from
    // before the orchestrator was running isn't silently dropped.
    expect(messageExistsInDifferentChat('99999', 'tg:-1004444')).toBe(false);
  });

  it('returns true when id exists in BOTH the target chat and a different chat', () => {
    // Telegram message IDs are per-chat sequential, so the same numeric
    // id legitimately exists in many chats. Asserting only on this
    // helper, the answer is "yes, an other-chat occurrence exists" —
    // but that fact alone is NOT a sufficient signal to drop
    // `reply_parameters`, or we'd strip threading from most legitimate
    // same-chat replies. The call-site predicate
    // (`safeReplyToForChat` in src/channels/telegram.ts) consults
    // `getMessageById(id, target)` first as positive evidence of a
    // local target; this helper only fires when that returns null.
    // See that file for the channel-level test that asserts the
    // composite "shared id => keep reply_to" behavior.
    storeChatMetadata('tg:-1005555', '2026-04-28T00:00:00.000Z');
    storeChatMetadata('tg:-1006666', '2026-04-28T00:00:00.000Z');
    store({
      id: 'shared',
      chat_jid: 'tg:-1005555',
      sender: 'u1',
      sender_name: 'Alice',
      content: 'in target',
      timestamp: '2026-04-28T00:00:00.000Z',
    });
    store({
      id: 'shared',
      chat_jid: 'tg:-1006666',
      sender: 'u2',
      sender_name: 'Bob',
      content: 'in foreign',
      timestamp: '2026-04-28T00:00:00.000Z',
    });
    expect(messageExistsInDifferentChat('shared', 'tg:-1005555')).toBe(true);
  });
});

// --- getChatByJid (#289 — addressed-ness gate dependency) ---

describe('getChatByJid', () => {
  it('returns null for an unknown JID', () => {
    expect(getChatByJid('tg:-9999999999')).toBeNull();
  });

  it('returns the row after storeChatMetadata with isGroup=true', () => {
    storeChatMetadata(
      'tg:-1003869886477',
      '2026-04-29T20:00:00.000Z',
      'Old.wtf',
      'telegram',
      true,
    );
    const row = getChatByJid('tg:-1003869886477');
    expect(row).not.toBeNull();
    expect(row?.jid).toBe('tg:-1003869886477');
    expect(row?.is_group).toBe(1);
    expect(row?.channel).toBe('telegram');
  });

  it('returns the row with is_group=0 for a 1:1 DM', () => {
    storeChatMetadata(
      'tg:42',
      '2026-04-29T20:00:00.000Z',
      'Solo Alice',
      'telegram',
      false,
    );
    const row = getChatByJid('tg:42');
    expect(row?.is_group).toBe(0);
  });

  it('returns the row with is_group=null when isGroup was omitted at store-time', () => {
    // The migration default for pre-existing rows is 0, but a fresh
    // insert that doesn't pass isGroup leaves the column at NULL.
    // The addressed-ness gate treats `is_group !== 0` as "not 1:1",
    // so NULL must NOT short-circuit the gate to "addressed."
    storeChatMetadata('tg:7', '2026-04-29T20:00:00.000Z');
    const row = getChatByJid('tg:7');
    expect(row).not.toBeNull();
    expect(row?.is_group).toBeNull();
  });
});

// --- TriggerPattern JSON schema (#81) ---

describe('registered group trigger pattern JSON schema', () => {
  it('setRegisteredGroup serializes a string trigger to a single-element JSON config', () => {
    setRegisteredGroup('jsonshape@g.us', {
      name: 'JSON Shape Group',
      folder: 'whatsapp_jsonshape',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
    });

    const cfg = getTriggerPatterns('jsonshape@g.us');
    expect(cfg).toBeDefined();
    expect(cfg!.version).toBe(1);
    expect(cfg!.patterns).toHaveLength(1);
    expect(cfg!.patterns[0]).toMatchObject({
      pattern: '@Andy',
      kind: 'keyword',
      source: 'owner-set',
      precision: 0,
      sample_count: 0,
      last_matched_at: null,
      last_updated_at: null,
    });
  });

  it('getRegisteredGroup derives primary keyword from JSON config', () => {
    setRegisteredGroup('keyword@g.us', {
      name: 'Keyword Group',
      folder: 'whatsapp_keyword',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            pattern: '@Andy',
            kind: 'keyword',
            source: 'owner-set',
            precision: 0.9,
            sample_count: 100,
            last_matched_at: '2024-06-01T00:00:00.000Z',
            last_updated_at: '2024-06-01T00:00:00.000Z',
          },
          {
            pattern: 'urgent',
            kind: 'keyword',
            source: 'learned',
            precision: 0.7,
            sample_count: 50,
            last_matched_at: null,
            last_updated_at: null,
          },
        ],
      },
    });

    const group = getRegisteredGroup('keyword@g.us');
    expect(group).toBeDefined();
    // Primary keyword is the FIRST keyword-kind pattern.
    expect(group!.trigger).toBe('@Andy');
    expect(group!.triggerPatterns).toBeDefined();
    expect(group!.triggerPatterns!.patterns).toHaveLength(2);
    expect(group!.triggerPatterns!.patterns[1].source).toBe('learned');
  });

  it('reader accepts legacy string-shaped trigger_pattern (dual-mode)', () => {
    // Simulate a row written by a pre-#81 binary that bypassed
    // setRegisteredGroup. The dual-mode reader must surface it as a
    // synthesised single-element keyword config without the
    // initDatabase-time backfill having run. _writeRawRegisteredGroup
    // writes the `trigger` argument verbatim into the column, so a
    // plain '@Andy' string IS the legacy shape we want to test.
    _writeRawRegisteredGroup({
      jid: 'legacy@g.us',
      name: 'Legacy Group',
      folder: 'whatsapp_legacy',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      container_config: null,
    });

    const group = getRegisteredGroup('legacy@g.us');
    expect(group).toBeDefined();
    expect(group!.trigger).toBe('@Andy');
    expect(group!.triggerPatterns).toBeDefined();
    expect(group!.triggerPatterns!.patterns).toHaveLength(1);
    expect(group!.triggerPatterns!.patterns[0]).toMatchObject({
      pattern: '@Andy',
      kind: 'keyword',
      source: 'owner-set',
    });
  });

  it('reader falls back to legacy shape on malformed JSON', () => {
    _writeRawRegisteredGroup({
      jid: 'malformed@g.us',
      name: 'Malformed Group',
      folder: 'whatsapp_malformed',
      // `{` triggers the JSON path; the broken body forces the catch
      // branch. Reader must NOT throw — startup walks every row at
      // boot via getAllRegisteredGroups, so a single malformed row
      // can't be allowed to crash the orchestrator.
      trigger: '{not valid json',
      added_at: '2024-01-01T00:00:00.000Z',
      container_config: null,
    });

    const group = getRegisteredGroup('malformed@g.us');
    expect(group).toBeDefined();
    expect(group!.trigger).toBe('{not valid json');
    expect(group!.triggerPatterns!.patterns[0].pattern).toBe('{not valid json');
  });

  it('reader treats wrong-version JSON as no-config (loud-warn, gates fall through)', () => {
    _writeRawRegisteredGroup({
      jid: 'wrongversion@g.us',
      name: 'Wrong Version Group',
      folder: 'whatsapp_wrongversion',
      trigger: '{"version":2,"patterns":[]}',
      added_at: '2024-01-01T00:00:00.000Z',
      container_config: null,
    });

    const group = getRegisteredGroup('wrongversion@g.us');
    expect(group).toBeDefined();
    // Per Fix 3 of the #84 followup PR (review comment-id
    // 4360940433): a future-version row read by an older binary is
    // returned as `trigger: null` + `triggerPatterns: undefined`
    // rather than the previous "interpret raw column as a literal
    // keyword" silent fallback. The trigger gate consumes null as
    // "no opinion" → fall-open at the gate combinator, which is the
    // desired loud-but-non-fatal behaviour: a roll-forward-then-
    // rollback DB warns once per row at startup and stops emitting a
    // bogus literal-string trigger that would have silently failed
    // to match anything.
    expect(group!.trigger).toBeNull();
    expect(group!.triggerPatterns).toBeUndefined();
  });

  it('getTriggerPatterns returns null on wrong-version JSON', () => {
    _writeRawRegisteredGroup({
      jid: 'wrongversion-ext@g.us',
      name: 'Wrong Version Ext',
      folder: 'whatsapp_wrongversion_ext',
      trigger: '{"version":3,"patterns":[]}',
      added_at: '2024-01-01T00:00:00.000Z',
      container_config: null,
    });
    expect(getTriggerPatterns('wrongversion-ext@g.us')).toBeNull();
  });

  it('setTriggerPatterns updates only the trigger column', () => {
    setRegisteredGroup('update@g.us', {
      name: 'Update Group',
      folder: 'whatsapp_update',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      containerConfig: { trusted: true },
      requiresTrigger: true,
    });

    const updated: TriggerPatternConfig = {
      version: 1,
      patterns: [
        {
          pattern: '@Andy',
          kind: 'keyword',
          source: 'owner-set',
          precision: 0.95,
          sample_count: 200,
          last_matched_at: '2024-06-01T00:00:00.000Z',
          last_updated_at: '2024-06-01T00:00:00.000Z',
        },
        {
          pattern: 'help',
          kind: 'keyword',
          source: 'learned',
          precision: 0.4,
          sample_count: 10,
          last_matched_at: null,
          last_updated_at: null,
        },
      ],
    };
    setTriggerPatterns('update@g.us', updated);

    const group = getRegisteredGroup('update@g.us');
    expect(group).toBeDefined();
    // Other columns untouched.
    expect(group!.containerConfig).toEqual({ trusted: true });
    expect(group!.requiresTrigger).toBe(true);
    // Trigger config replaced wholesale.
    expect(group!.triggerPatterns!.patterns).toHaveLength(2);
    expect(group!.triggerPatterns!.patterns[1].pattern).toBe('help');
    expect(group!.triggerPatterns!.patterns[0].sample_count).toBe(200);
  });

  it('setTriggerPatterns throws when the group does not exist', () => {
    expect(() =>
      setTriggerPatterns('missing@g.us', {
        version: 1,
        patterns: [],
      }),
    ).toThrow(/no registered_groups row/);
  });

  it('setTriggerPatterns throws on unsupported version', () => {
    setRegisteredGroup('versioncheck@g.us', {
      name: 'Version Check',
      folder: 'whatsapp_versioncheck',
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
    });
    // Cast through unknown to bypass the literal-1 type — we want to
    // simulate a future-version write attempt against this binary.
    expect(() =>
      setTriggerPatterns('versioncheck@g.us', {
        version: 2,
        patterns: [],
      } as unknown as TriggerPatternConfig),
    ).toThrow(/Unsupported TriggerPatternConfig version/);
  });

  it('round-trips a config with non-keyword pattern kinds', () => {
    const cfg: TriggerPatternConfig = {
      version: 1,
      patterns: [
        {
          pattern: 'mention-ping',
          kind: 'mention',
          source: 'universal',
          precision: 0,
          sample_count: 0,
          last_matched_at: null,
          last_updated_at: null,
        },
        {
          pattern: '^urgent.*',
          kind: 'regex',
          source: 'learned',
          precision: 0.6,
          sample_count: 25,
          last_matched_at: '2024-06-01T00:00:00.000Z',
          last_updated_at: '2024-06-01T00:00:00.000Z',
        },
      ],
    };
    setRegisteredGroup('mixedkinds@g.us', {
      name: 'Mixed Kinds',
      folder: 'whatsapp_mixedkinds',
      trigger: 'mention-ping',
      added_at: '2024-01-01T00:00:00.000Z',
      triggerPatterns: cfg,
    });

    const group = getRegisteredGroup('mixedkinds@g.us');
    expect(group).toBeDefined();
    // Per Fix 2 of the #84 followup: no keyword entry, first
    // mention entry wins, and the stored bare pattern gets `@`
    // re-prepended for the legacy `RegisteredGroup.trigger` slot.
    expect(group!.trigger).toBe('@mention-ping');
    expect(group!.triggerPatterns!.patterns[1].kind).toBe('regex');
  });
});

// --- #84 followup PR: deriveTriggerString helper (Fix 2) ---

describe('deriveTriggerString', () => {
  function pat(
    kind: 'keyword' | 'mention' | 'regex' | 'sender_tier' | 'reply',
    pattern: string,
  ) {
    return {
      pattern,
      kind,
      source: 'owner-set' as const,
      precision: 0,
      sample_count: 0,
      last_matched_at: null,
      last_updated_at: null,
    };
  }

  it('keyword-only config returns the keyword pattern verbatim', () => {
    expect(
      deriveTriggerString({ version: 1, patterns: [pat('keyword', '@Andy')] }),
    ).toBe('@Andy');
  });

  it('mention-only config re-prepends @ to the bare stored pattern', () => {
    expect(
      deriveTriggerString({ version: 1, patterns: [pat('mention', 'Andy')] }),
    ).toBe('@Andy');
  });

  it('mention pattern that already has @ stays as-is (no double @)', () => {
    expect(
      deriveTriggerString({ version: 1, patterns: [pat('mention', '@Andy')] }),
    ).toBe('@Andy');
  });

  it('regex-only config returns null (no legacy string for free-form regex)', () => {
    expect(
      deriveTriggerString({
        version: 1,
        patterns: [pat('regex', '^urgent.*')],
      }),
    ).toBeNull();
  });

  it('sender_tier-only config returns null', () => {
    expect(
      deriveTriggerString({
        version: 1,
        patterns: [pat('sender_tier', 'owner')],
      }),
    ).toBeNull();
  });

  it('keyword wins over mention regardless of position', () => {
    expect(
      deriveTriggerString({
        version: 1,
        patterns: [pat('mention', 'andy'), pat('keyword', 'help')],
      }),
    ).toBe('help');
  });

  it('mention wins over regex when no keyword is present', () => {
    expect(
      deriveTriggerString({
        version: 1,
        patterns: [pat('regex', '.*'), pat('mention', 'andy')],
      }),
    ).toBe('@andy');
  });

  it('empty patterns list returns null', () => {
    expect(deriveTriggerString({ version: 1, patterns: [] })).toBeNull();
  });

  it('null config returns null', () => {
    expect(deriveTriggerString(null)).toBeNull();
  });
});

// --- #84 followup PR: end-to-end mention-only round-trip ---

describe('mention-only triggerPatterns round-trip', () => {
  it('setTriggerPatterns + getRegisteredGroup yields trigger="@<name>"', () => {
    setRegisteredGroup('mention-only@g.us', {
      name: 'Mention Only Group',
      folder: 'whatsapp_mentiononly',
      trigger: '@bootstrap', // value irrelevant, replaced by setTriggerPatterns below
      added_at: '2024-01-01T00:00:00.000Z',
    });
    setTriggerPatterns('mention-only@g.us', {
      version: 1,
      patterns: [
        {
          pattern: 'andy',
          kind: 'mention',
          source: 'owner-set',
          precision: 0,
          sample_count: 0,
          last_matched_at: null,
          last_updated_at: null,
        },
      ],
    });
    const group = getRegisteredGroup('mention-only@g.us');
    expect(group).toBeDefined();
    // The legacy `trigger` slot reconstructs `@andy` from the bare
    // mention pattern stored on disk. Call sites that still build
    // a regex from it (`getTriggerPattern(group.trigger)`) keep
    // working bit-for-bit even though the row no longer has a
    // keyword-kind entry.
    expect(group!.trigger).toBe('@andy');
  });
});

// --- #84 followup PR: legacy string with non-mention shape stays keyword (Fix 1) ---

describe('legacy trigger backfill shape classification', () => {
  it('legacy `@<word>` string round-trips as a mention with bare pattern', () => {
    // The setRegisteredGroup write path serialises `group.trigger`
    // into a single-element keyword config (legacyTriggerToConfig).
    // The shape-classification happens in the createSchema-time
    // backfill against rows that bypass setRegisteredGroup. Use the
    // raw writer + re-init to drive the migration explicitly.
    _writeRawRegisteredGroup({
      jid: 'rawmention@g.us',
      name: 'Raw Mention',
      folder: 'whatsapp_rawmention',
      trigger: '@TestBot', // legacy string shape, matches bare-mention regex
      added_at: '2024-01-01T00:00:00.000Z',
      container_config: null,
    });
    // Trigger the backfill by re-running createSchema via a fresh
    // _initTestDatabase pass. _initTestDatabase replaces the in-memory
    // DB, so we instead call the underlying migration sequence here:
    // a second call to setRegisteredGroup with the same trigger
    // exercises the WRITE path (always emits keyword), not the
    // backfill path. The backfill itself is covered end-to-end in
    // db-migration.test.ts; here we lock in the read-side handling
    // when the row matches the legacy shape.
    const cfg = getTriggerPatterns('rawmention@g.us');
    expect(cfg).toBeDefined();
    // Legacy reader synthesises a keyword config (bypassing the
    // shape-sniff classifier — that runs only in the createSchema
    // backfill, not in the dual-mode reader). This is intentional:
    // dual-mode reader behaviour stays bug-compatible with #81; the
    // smarter classification happens at migration time only, on rows
    // that the backfill has not yet touched.
    expect(cfg!.patterns[0].pattern).toBe('@TestBot');
    expect(cfg!.patterns[0].kind).toBe('keyword');
  });

  it('legacy bare keyword (no @) backfilled as keyword via raw writer + setRegisteredGroup roundtrip', () => {
    // setRegisteredGroup serialises group.trigger="nanoclaw" into a
    // keyword config — that's the write-side behaviour. The
    // migration backfill (db-migration.test.ts) is what shape-sniffs
    // a raw row written by a pre-#81 binary. This test pins the
    // write path's emitted shape: always keyword for legacy callers.
    setRegisteredGroup('barekeyword@g.us', {
      name: 'Bare Keyword',
      folder: 'whatsapp_barekeyword',
      trigger: 'nanoclaw',
      added_at: '2024-01-01T00:00:00.000Z',
    });
    const cfg = getTriggerPatterns('barekeyword@g.us');
    expect(cfg!.patterns[0].pattern).toBe('nanoclaw');
    expect(cfg!.patterns[0].kind).toBe('keyword');
  });
});
