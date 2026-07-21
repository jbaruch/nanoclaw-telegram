import { ASSISTANT_NAME, getTriggerPattern } from './config.js';
import { getChatByJid, getMessageById } from './db.js';
import { NewMessage, RegisteredGroup } from './types.js';

export function isReplyToBot(msg: NewMessage): boolean {
  // Check content prefix — resolveReply adds [Replying to SenderName: "..."]
  if (msg.content.startsWith(`[Replying to ${ASSISTANT_NAME}:`)) return true;
  // Check reply_to_message_id in DB — covers cases where prefix format differs
  if (msg.reply_to_message_id) {
    const original = getMessageById(msg.reply_to_message_id, msg.chat_jid);
    if (original?.is_from_me) return true;
  }
  return false;
}

/**
 * Decide whether an inbound batch is "addressed to us" — drives the
 * agent-runner's react-first 👀 gate (#289). Independent of
 * `requires_trigger`, which governs whether the agent ANSWERS
 * deterministically vs. reasons about every inbound. The 👀 ack is
 * about whether the message was directed at us at all.
 *
 * Resolves true iff:
 *  - the chat is the main control group, OR
 *  - the chat is a 1:1 DM (`chats.is_group=0`) — every solo inbound is
 *    implicitly for us, OR
 *  - at least one message in the batch matches the trigger pattern, OR
 *  - at least one message replies to OUR bot (per `isReplyToBot`).
 *
 * Note: `requires_trigger=false` on a multi-bot group (e.g. `Old.wtf`)
 * does NOT short-circuit — that was the original bug.
 */
export function isAddressedToUs(
  group: RegisteredGroup,
  chatJid: string,
  messages: NewMessage[],
): boolean {
  if (group.isMain === true) return true;
  const chat = getChatByJid(chatJid);
  if (chat && chat.is_group === 0) return true;
  const triggerPattern = getTriggerPattern(group.trigger ?? undefined);
  return messages.some(
    (m) => triggerPattern.test(m.content.trim()) || isReplyToBot(m),
  );
}
