import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ASSISTANT_NAME } from './config.js';
import type { NewMessage, RegisteredGroup } from './types.js';

vi.mock('./db.js', () => ({
  getMessageById: vi.fn(),
  getChatByJid: vi.fn(),
}));

import { getChatByJid, getMessageById } from './db.js';
import { isAddressedToUs, isReplyToBot } from './message-classify.js';

const mockGetMessageById = vi.mocked(getMessageById);
const mockGetChatByJid = vi.mocked(getChatByJid);

let _msgIdCounter = 0;
function msg(content: string, overrides: Partial<NewMessage> = {}): NewMessage {
  _msgIdCounter += 1;
  return {
    id: `mid-${_msgIdCounter}`,
    chat_jid: 'g@g.us',
    sender: 's@s.whatsapp.net',
    sender_name: 'Sender',
    content,
    timestamp: '2026-01-01T00:00:00.000Z',
    ...overrides,
  };
}

function group(overrides: Partial<RegisteredGroup> = {}): RegisteredGroup {
  return {
    name: 'g',
    folder: 'telegram_g',
    trigger: '@andy',
    added_at: '2024-01-01T00:00:00Z',
    ...overrides,
  };
}

beforeEach(() => {
  mockGetMessageById.mockReset();
  mockGetChatByJid.mockReset();
  // Default: no chat row and no reply target unless a test says so.
  mockGetChatByJid.mockReturnValue(null);
  mockGetMessageById.mockReturnValue(null);
});

describe('isReplyToBot', () => {
  it('is true when content carries the assistant reply-quote prefix', () => {
    const m = msg(`[Replying to ${ASSISTANT_NAME}: "hi"]\nyo`);
    expect(isReplyToBot(m)).toBe(true);
    // Prefix match short-circuits before any DB lookup.
    expect(mockGetMessageById).not.toHaveBeenCalled();
  });

  it('is true when the replied-to message is from the bot', () => {
    mockGetMessageById.mockReturnValue({ is_from_me: 1 } as never);
    const m = msg('plain reply', { reply_to_message_id: 'orig-1' });
    expect(isReplyToBot(m)).toBe(true);
    expect(mockGetMessageById).toHaveBeenCalledWith('orig-1', 'g@g.us');
  });

  it('is false when the replied-to message is not from the bot', () => {
    mockGetMessageById.mockReturnValue({ is_from_me: 0 } as never);
    const m = msg('plain reply', { reply_to_message_id: 'orig-2' });
    expect(isReplyToBot(m)).toBe(false);
  });

  it('is false for an ordinary message with no reply target', () => {
    expect(isReplyToBot(msg('just chatting'))).toBe(false);
  });
});

describe('isAddressedToUs', () => {
  it('is true for the main control group regardless of content', () => {
    expect(isAddressedToUs(group({ isMain: true }), 'g@g.us', [msg('x')])).toBe(
      true,
    );
  });

  it('is true for a 1:1 DM (chats.is_group=0)', () => {
    mockGetChatByJid.mockReturnValue({ is_group: 0 } as never);
    expect(isAddressedToUs(group(), 'dm@s.whatsapp.net', [msg('hi')])).toBe(
      true,
    );
  });

  it('is true when a message in the batch matches the trigger pattern', () => {
    mockGetChatByJid.mockReturnValue({ is_group: 1 } as never);
    expect(
      isAddressedToUs(group(), 'g@g.us', [msg('nope'), msg('hey @andy')]),
    ).toBe(true);
  });

  it('is false when no message triggers, replies, or is a DM/main', () => {
    mockGetChatByJid.mockReturnValue({ is_group: 1 } as never);
    expect(isAddressedToUs(group(), 'g@g.us', [msg('just talking')])).toBe(
      false,
    );
  });

  it('is true when a message replies to the bot even without a trigger', () => {
    mockGetChatByJid.mockReturnValue({ is_group: 1 } as never);
    mockGetMessageById.mockReturnValue({ is_from_me: 1 } as never);
    expect(
      isAddressedToUs(group(), 'g@g.us', [
        msg('untriggered', { reply_to_message_id: 'orig-3' }),
      ]),
    ).toBe(true);
  });
});
