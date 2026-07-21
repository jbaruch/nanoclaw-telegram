// Messages, chats, and reactions accessors (#751 seam 6, extracted
// verbatim from src/db.ts).
//
// Covers the `messages`, `chats`, and `reactions` tables: inbound/outbound
// message rows (including the Telegram synthetic bot-id two-pass lookup),
// chat metadata used for group discovery, and per-message reactions.
// Location rows (`locations` table) stay with the timezone/location seam.
import { db } from './db-connection.js';
import { NewMessage } from './types.js';

/**
 * Store chat metadata only (no message content).
 * Used for all chats to enable group discovery without storing sensitive content.
 */
export function storeChatMetadata(
  chatJid: string,
  timestamp: string,
  name?: string,
  channel?: string,
  isGroup?: boolean,
): void {
  const ch = channel ?? null;
  const group = isGroup === undefined ? null : isGroup ? 1 : 0;

  if (name) {
    // Update with name, preserving existing timestamp if newer
    db.prepare(
      `
      INSERT INTO chats (jid, name, last_message_time, channel, is_group) VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(jid) DO UPDATE SET
        name = excluded.name,
        last_message_time = MAX(last_message_time, excluded.last_message_time),
        channel = COALESCE(excluded.channel, channel),
        is_group = COALESCE(excluded.is_group, is_group)
    `,
    ).run(chatJid, name, timestamp, ch, group);
  } else {
    // Update timestamp only, preserve existing name if any
    db.prepare(
      `
      INSERT INTO chats (jid, name, last_message_time, channel, is_group) VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(jid) DO UPDATE SET
        last_message_time = MAX(last_message_time, excluded.last_message_time),
        channel = COALESCE(excluded.channel, channel),
        is_group = COALESCE(excluded.is_group, is_group)
    `,
    ).run(chatJid, chatJid, timestamp, ch, group);
  }
}

/**
 * Update chat name without changing timestamp for existing chats.
 * New chats get the current time as their initial timestamp.
 * Used during group metadata sync.
 */
export function updateChatName(chatJid: string, name: string): void {
  db.prepare(
    `
    INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)
    ON CONFLICT(jid) DO UPDATE SET name = excluded.name
  `,
  ).run(chatJid, name, new Date().toISOString());
}

export interface ChatInfo {
  jid: string;
  name: string;
  last_message_time: string;
  channel: string;
  is_group: number;
}

/**
 * Get all known chats, ordered by most recent activity.
 */
export function getAllChats(): ChatInfo[] {
  return db
    .prepare(
      `
    SELECT jid, name, last_message_time, channel, is_group
    FROM chats
    ORDER BY last_message_time DESC
  `,
    )
    .all() as ChatInfo[];
}

/**
 * Look up a single chat row by JID. Returns null when no row exists
 * (the channel layer hasn't seen any inbound from that chat yet).
 *
 * Used by the orchestrator's react-first gate (#289) to distinguish
 * 1:1 DMs (`is_group=0`) from group chats (`is_group=1`) without
 * leaning on `requires_trigger` as a proxy — those two flags governed
 * different concerns and the conflation was the original bug.
 */
export function getChatByJid(jid: string): ChatInfo | null {
  const row = db
    .prepare(
      `SELECT jid, name, last_message_time, channel, is_group FROM chats WHERE jid = ?`,
    )
    .get(jid) as ChatInfo | undefined;
  return row ?? null;
}

/**
 * Get timestamp of last group metadata sync.
 */
export function getLastGroupSync(): string | null {
  // Store sync time in a special chat entry
  const row = db
    .prepare(`SELECT last_message_time FROM chats WHERE jid = '__group_sync__'`)
    .get() as { last_message_time: string } | undefined;
  return row?.last_message_time || null;
}

/**
 * Record that group metadata was synced.
 */
export function setLastGroupSync(): void {
  const now = new Date().toISOString();
  db.prepare(
    `INSERT OR REPLACE INTO chats (jid, name, last_message_time) VALUES ('__group_sync__', '__group_sync__', ?)`,
  ).run(now);
}

/**
 * Decide whether to record a `bot-…` row in `messages.db` after a
 * send dispatches. For Telegram we MUST have a Telegram-native message
 * id back from the channel — its absence is the only reliable signal
 * that the send was swallowed (400 from a bad reply_to, network blip,
 * malformed HTML even after the plain-text fallback, blocked-by-user,
 * rate-limit, etc.). A row written without that id is a phantom: the
 * heartbeat / unanswered-cron treats it as evidence of a reply on a
 * chat the user never received anything in, and downstream agents
 * quote-reply to a message id Telegram has no record of.
 *
 * Non-Telegram channels are not gated — their `Channel.sendMessage`
 * contract permits returning `void` on success (see `src/types.ts`),
 * so absence of an id isn't a failure signal there. Until those
 * channels grow their own success-id surface, the gate would punish
 * a passing send.
 *
 * The undefined check is `!== undefined` rather than truthiness on
 * purpose, matching the comment on `sentMsgId` upstream: a future
 * Telegram id of `''` or `'0'` (we don't expect this today, but the
 * contract is `string | undefined`) must still record the row.
 *
 * Shared by every bot-row write site — IPC `send_message` /
 * `send_message_to_chat` handlers (`src/ipc.ts`), the inbound send
 * path (`src/index.ts`), and the scheduled-task forward
 * (`src/task-scheduler.ts`) — so all of them apply the identical gate.
 *
 * `@internal` — not part of the public API surface; `stripInternal:
 * true` keeps it out of the published `.d.ts`. It is exercised directly
 * by unit tests AND called by the production write sites listed above —
 * not a test-only export.
 */
export function shouldStoreBotMessage(
  chatJid: string,
  sentMsgId: string | undefined,
): boolean {
  const isTelegram = chatJid.startsWith('tg:');
  if (!isTelegram) return true;
  return sentMsgId !== undefined;
}

/**
 * Store a message with full content.
 * Only call this for registered groups where message history is needed.
 */
export function storeMessage(msg: NewMessage): void {
  db.prepare(
    `INSERT OR REPLACE INTO messages (id, chat_jid, sender, sender_name, content, timestamp, is_from_me, is_bot_message, reply_to_message_id, reply_to_message_content, reply_to_sender_name, telegram_message_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    msg.id,
    msg.chat_jid,
    msg.sender,
    msg.sender_name,
    msg.content,
    msg.timestamp,
    msg.is_from_me ? 1 : 0,
    msg.is_bot_message ? 1 : 0,
    msg.reply_to_message_id ?? null,
    msg.reply_to_message_content ?? null,
    msg.reply_to_sender_name ?? null,
    msg.telegram_message_id ?? null,
  );
}

/**
 * Store a message directly.
 */
export function storeMessageDirect(msg: {
  id: string;
  chat_jid: string;
  sender: string;
  sender_name: string;
  content: string;
  timestamp: string;
  is_from_me: boolean;
  is_bot_message?: boolean;
}): void {
  db.prepare(
    `INSERT OR REPLACE INTO messages (id, chat_jid, sender, sender_name, content, timestamp, is_from_me, is_bot_message) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    msg.id,
    msg.chat_jid,
    msg.sender,
    msg.sender_name,
    msg.content,
    msg.timestamp,
    msg.is_from_me ? 1 : 0,
    msg.is_bot_message ? 1 : 0,
  );
}

/**
 * Look up a single message by its platform message_id and chat_jid.
 * Returns null if not found.
 *
 * Two-pass lookup. Inbound messages and non-Telegram bot sends store
 * the platform-native id directly in the `id` column — that's the
 * fast path. Telegram bot sends are different: the `id` column holds
 * a synthetic `bot-<ts>-<rand>` (#80) and the platform-native id
 * lives in `telegram_message_id`. Without the fallback,
 * `reply_to_message_id` lookups against bot-emitted messages miss,
 * which silently breaks `replyTo.isAssistant` in the gate context
 * (the trigger gate's `reply:*` matcher denies the message) and the
 * cross-chat `safeReplyToForChat` guard in `channels/telegram.ts`
 * (drops legitimate same-chat reply threading). Concrete repro:
 * `tg:-1001633120997` msg 378306, a reply to bot row id
 * `bot-1777747764042-6tppb` (telegram_message_id `378305`), denied
 * by the trigger gate with "no trigger pattern matched" because the
 * id-only lookup couldn't see it was a reply to the assistant.
 */
export function getMessageById(
  messageId: string,
  chatJid: string,
): NewMessage | null {
  const row = db
    .prepare(
      `SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me, is_bot_message
       FROM messages
       WHERE id = ? AND chat_jid = ?`,
    )
    .get(messageId, chatJid) as
    | {
        id: string;
        chat_jid: string;
        sender: string;
        sender_name: string;
        content: string;
        timestamp: string;
        is_from_me: number;
        is_bot_message: number | null;
      }
    | undefined;
  if (row) {
    return {
      id: row.id,
      chat_jid: row.chat_jid,
      sender: row.sender,
      sender_name: row.sender_name,
      content: row.content,
      timestamp: row.timestamp,
      is_from_me: row.is_from_me === 1,
      is_bot_message: row.is_bot_message === 1,
    };
  }
  return getBotMessageByTelegramId(chatJid, messageId);
}

/**
 * Cross-chat reply_to safety check (one half of the call-site
 * predicate; see `safeReplyToForChat` in `src/channels/telegram.ts`).
 *
 * Returns true when the message id is stored under at least one
 * chat_jid that is NOT the expected one — i.e. "we have evidence the
 * id belongs to some other chat". This is intentionally NOT a
 * sufficient signal on its own to drop `reply_parameters`: Telegram
 * message IDs are per-chat sequential, so the same numeric id
 * routinely exists in many chats, including the target. The call
 * site MUST first check `getMessageById(id, jid)` for positive
 * evidence the id is local; only if that returns null does this
 * helper's "exists in another chat" answer flip the safety verdict
 * to "drop". Inverting that order is how you accidentally strip
 * reply threading from every legitimate same-chat reply once a
 * deployment has more than one Telegram chat.
 *
 * Why not collapse this into `getMessageById`: that helper takes
 * both id AND chat and returns a single row. We want a single SQL
 * pass with the inverse predicate ("exists in some chat OTHER than
 * X") so the chat_jid != comparison stays inside the query (where
 * the index helps) and the call site reads as two clean predicates.
 */
export function messageExistsInDifferentChat(
  messageId: string,
  expectedChatJid: string,
): boolean {
  const row = db
    .prepare(
      `SELECT 1 AS hit
       FROM messages
       WHERE id = ? AND chat_jid != ?
       LIMIT 1`,
    )
    .get(messageId, expectedChatJid) as { hit: number } | undefined;
  return !!row;
}

/**
 * Look up a bot-sent message by the Telegram-native message ID
 * returned when it was posted. Exists so "what did we post at
 * Telegram ID X in chat Y" stops being a logs-grep exercise — the
 * synthetic `bot-<ts>-<rand>` `id` column gives no way to work
 * back from the Telegram ID otherwise. Narrowly scoped to bot
 * sends on Telegram (other channels either use the platform ID
 * as `id` directly or don't populate this column).
 */
export function getBotMessageByTelegramId(
  chatJid: string,
  telegramMessageId: string,
): NewMessage | null {
  const row = db
    .prepare(
      `SELECT id, chat_jid, sender, sender_name, content, timestamp,
              is_from_me, is_bot_message, reply_to_message_id,
              reply_to_message_content, reply_to_sender_name,
              telegram_message_id
         FROM messages
        WHERE chat_jid = ? AND telegram_message_id = ?
          AND is_bot_message = 1`,
    )
    .get(chatJid, telegramMessageId) as
    | {
        id: string;
        chat_jid: string;
        sender: string;
        sender_name: string;
        content: string;
        timestamp: string;
        is_from_me: number;
        is_bot_message: number;
        reply_to_message_id: string | null;
        reply_to_message_content: string | null;
        reply_to_sender_name: string | null;
        telegram_message_id: string | null;
      }
    | undefined;
  if (!row) return null;
  // Surface NULLs as `null` to match the other message getters
  // (`getMessagesSince`, `getNewMessages`) — existing tests assert
  // `.toBeNull()` on those paths. Using `?? undefined` here would
  // force every caller to handle both shapes.
  return {
    id: row.id,
    chat_jid: row.chat_jid,
    sender: row.sender,
    sender_name: row.sender_name,
    content: row.content,
    timestamp: row.timestamp,
    is_from_me: row.is_from_me === 1,
    is_bot_message: row.is_bot_message === 1,
    reply_to_message_id: row.reply_to_message_id,
    reply_to_message_content: row.reply_to_message_content,
    reply_to_sender_name: row.reply_to_sender_name,
    telegram_message_id: row.telegram_message_id,
  } as NewMessage;
}

export function storeReaction(reaction: {
  message_id: string;
  message_chat_jid: string;
  reactor_jid: string;
  reactor_name: string;
  emoji: string;
  timestamp: string;
}): void {
  db.prepare(
    `INSERT INTO reactions (message_id, message_chat_jid, reactor_jid, reactor_name, emoji, timestamp)
     VALUES (?, ?, ?, ?, ?, ?)`,
  ).run(
    reaction.message_id,
    reaction.message_chat_jid,
    reaction.reactor_jid,
    reaction.reactor_name,
    reaction.emoji,
    reaction.timestamp,
  );
}

export function getReactionsForMessage(
  messageId: string,
  chatJid: string,
): Array<{
  reactor_jid: string;
  reactor_name: string;
  emoji: string;
  timestamp: string;
}> {
  return db
    .prepare(
      `SELECT reactor_jid, reactor_name, emoji, timestamp FROM reactions
       WHERE message_id = ? AND message_chat_jid = ?
       ORDER BY timestamp`,
    )
    .all(messageId, chatJid) as Array<{
    reactor_jid: string;
    reactor_name: string;
    emoji: string;
    timestamp: string;
  }>;
}

export function getLatestMessage(
  chatJid: string,
): { id: string; chat_jid: string } | null {
  const row = db
    .prepare(
      `SELECT id, chat_jid FROM messages WHERE chat_jid = ? ORDER BY timestamp DESC LIMIT 1`,
    )
    .get(chatJid) as { id: string; chat_jid: string } | undefined;
  return row || null;
}

export function getNewMessages(
  jids: string[],
  lastTimestamp: string,
  botPrefix: string,
  limit: number = 200,
): { messages: NewMessage[]; newTimestamp: string } {
  if (jids.length === 0) return { messages: [], newTimestamp: lastTimestamp };

  const placeholders = jids.map(() => '?').join(',');
  // Filter bot messages using both the is_bot_message flag AND the content
  // prefix as a backstop for messages written before the migration ran.
  // Subquery takes the N most recent, outer query re-sorts chronologically.
  const sql = `
    SELECT * FROM (
      SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me,
             reply_to_message_id, reply_to_message_content, reply_to_sender_name
      FROM messages
      WHERE timestamp > ? AND chat_jid IN (${placeholders})
        AND is_bot_message = 0 AND content NOT LIKE ?
        AND content != '' AND content IS NOT NULL
      ORDER BY timestamp DESC
      LIMIT ?
    ) ORDER BY timestamp
  `;

  const rows = db
    .prepare(sql)
    .all(lastTimestamp, ...jids, `${botPrefix}:%`, limit) as NewMessage[];

  let newTimestamp = lastTimestamp;
  for (const row of rows) {
    if (row.timestamp > newTimestamp) newTimestamp = row.timestamp;
  }

  return { messages: rows, newTimestamp };
}

export function getMessagesSince(
  chatJid: string,
  sinceTimestamp: string,
  botPrefix: string,
  limit: number = 200,
): NewMessage[] {
  // Filter bot messages using both the is_bot_message flag AND the content
  // prefix as a backstop for messages written before the migration ran.
  // Subquery takes the N most recent, outer query re-sorts chronologically.
  const sql = `
    SELECT * FROM (
      SELECT id, chat_jid, sender, sender_name, content, timestamp, is_from_me,
             reply_to_message_id, reply_to_message_content, reply_to_sender_name
      FROM messages
      WHERE chat_jid = ? AND timestamp > ?
        AND is_bot_message = 0 AND content NOT LIKE ?
        AND content != '' AND content IS NOT NULL
      ORDER BY timestamp DESC
      LIMIT ?
    ) ORDER BY timestamp
  `;
  return db
    .prepare(sql)
    .all(chatJid, sinceTimestamp, `${botPrefix}:%`, limit) as NewMessage[];
}

export function getLastBotMessageTimestamp(
  chatJid: string,
  botPrefix: string,
): string | undefined {
  const row = db
    .prepare(
      `SELECT MAX(timestamp) as ts FROM messages
       WHERE chat_jid = ? AND (is_bot_message = 1 OR content LIKE ?)`,
    )
    .get(chatJid, `${botPrefix}:%`) as { ts: string | null } | undefined;
  return row?.ts ?? undefined;
}

/**
 * Latest outbound message in a chat (where the host wrote the row with
 * `is_from_me = 1`, i.e. AyeAye sent it). Returned as `{ timestamp,
 * content }` or `null` if AyeAye never spoke in this chat. Used by the
 * `chat_status` IPC handler so the admin tile can answer "when did
 * AyeAye last respond here, and with what?" for diagnosing silent
 * containers. Single-chat convenience wrapper around the batch helper.
 */
export function getLastFromMeMessage(
  chatJid: string,
): { timestamp: string; content: string } | null {
  return getLastFromMeMessages([chatJid]).get(chatJid) ?? null;
}

/**
 * Batch variant. Resolves "latest is_from_me=1 message per chat" for
 * many JIDs in one SQL round-trip. Backs the all-chats path of the
 * `chat_status` IPC handler — calling the single-chat helper N times
 * was N statement compilations and N scan+sorts; this issues a single
 * grouped query against the `idx_messages_fromme_chat` composite index
 * (created in createSchema). Chats that AyeAye has never spoken in are
 * absent from the returned map, matching the single-chat helper's
 * `null` return.
 */
export function getLastFromMeMessages(
  chatJids: readonly string[],
): Map<string, { timestamp: string; content: string }> {
  const out = new Map<string, { timestamp: string; content: string }>();
  if (chatJids.length === 0) return out;
  const placeholders = chatJids.map(() => '?').join(',');
  // GROUP BY + MAX(timestamp) gives "latest per chat" without a
  // correlated subquery. Pull the matching content via a self-join so
  // the row's `content` corresponds to the same row whose `timestamp`
  // is the MAX — without the join we'd get arbitrary content from any
  // is_from_me=1 row in the chat. Composite index makes both halves
  // (the GROUP BY scan and the join lookup) fast.
  const sql = `
    SELECT m.chat_jid, m.timestamp, m.content
    FROM messages m
    JOIN (
      SELECT chat_jid, MAX(timestamp) AS max_ts
      FROM messages
      WHERE is_from_me = 1 AND chat_jid IN (${placeholders})
      GROUP BY chat_jid
    ) latest
      ON m.chat_jid = latest.chat_jid
     AND m.timestamp = latest.max_ts
     AND m.is_from_me = 1
  `;
  const rows = db.prepare(sql).all(...chatJids) as Array<{
    chat_jid: string;
    timestamp: string;
    content: string;
  }>;
  for (const row of rows) {
    // Multiple is_from_me=1 messages with the same MAX timestamp would
    // produce duplicate rows; the Map dedupes by keeping the last
    // assignment. This is rare enough (millisecond-precision
    // timestamps) that picking arbitrarily is fine — the docstring
    // promises "latest", not a deterministic tiebreak.
    out.set(row.chat_jid, { timestamp: row.timestamp, content: row.content });
  }
  return out;
}
