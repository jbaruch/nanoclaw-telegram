import fs from 'fs';
import https from 'https';
import path from 'path';
import { Api, Bot } from 'grammy';
import OpenAI from 'openai';

import { ASSISTANT_NAME, GROUPS_DIR, TRIGGER_PATTERN } from '../config.js';
import { getMessageById } from '../db.js';
import { readEnvFile } from '../env.js';
import { logger } from '../logger.js';
import { registerChannel, ChannelOpts } from './registry.js';
import {
  Channel,
  OnChatMetadata,
  OnInboundMessage,
  RegisteredGroup,
} from '../types.js';

export interface TelegramChannelOpts {
  onMessage: OnInboundMessage;
  onChatMetadata: OnChatMetadata;
  registeredGroups: () => Record<string, RegisteredGroup>;
}

/**
 * Send a message with Telegram Markdown parse mode, falling back to plain text.
 * Claude's output naturally matches Telegram's Markdown v1 format:
 *   *bold*, _italic_, `code`, ```code blocks```, [links](url)
 */
async function sendTelegramMessage(
  api: { sendMessage: Api['sendMessage'] },
  chatId: string | number,
  text: string,
  options: {
    message_thread_id?: number;
    reply_parameters?: { message_id: number };
  } = {},
): Promise<void> {
  try {
    await api.sendMessage(chatId, text, {
      ...options,
      parse_mode: 'Markdown',
    });
  } catch (err) {
    // Fallback: send as plain text if Markdown parsing fails
    logger.debug({ err }, 'Markdown send failed, falling back to plain text');
    await api.sendMessage(chatId, text, options);
  }
}

/**
 * Truncate a string to a maximum length, appending "..." if truncated.
 */
function truncate(s: string, max = 120): string {
  return s.length > max ? s.slice(0, max) + '...' : s;
}

/**
 * Resolve a Telegram reply context: look up the replied-to message in the DB
 * and return a prefix string with the quoted content.
 * Falls back to the reply message text if DB lookup fails (common for bot messages
 * whose DB id doesn't match Telegram message_id).
 */
function resolveReply(
  replyMsg: {
    message_id: number;
    text?: string;
    caption?: string;
    from?: { first_name?: string };
  },
  chatJid: string,
): string {
  // Try DB lookup first
  const original = getMessageById(replyMsg.message_id.toString(), chatJid);
  if (original) {
    return `[Replying to ${original.sender_name}: "${truncate(original.content, 200)}"]\n`;
  }
  // Fall back to the reply message text directly from Telegram
  const text = replyMsg.text || replyMsg.caption;
  if (text) {
    const sender = replyMsg.from?.first_name || 'Unknown';
    return `[Replying to ${sender}: "${truncate(text, 200)}"]\n`;
  }
  return '';
}

/**
 * Resolve t.me/c/<chat_id>/<message_id> links in content.
 * Replaces each link with `[Message: "<content>"]` if found in DB.
 */
function resolveMessageLinks(content: string): string {
  return content.replace(
    /https?:\/\/t\.me\/c\/(\d+)\/(\d+)/g,
    (_match, rawChatId, msgId) => {
      // Telegram supergroup JID: URL chat_id is bare id without -100 prefix
      const candidateJids = [`tg:-100${rawChatId}`, `tg:${rawChatId}`];
      for (const jid of candidateJids) {
        const msg = getMessageById(msgId, jid);
        if (msg) return `[Message: "${truncate(msg.content)}"]`;
      }
      return `[Message: not found]`;
    },
  );
}

/**
 * Download a file from Telegram's file API.
 * Returns a Buffer with the file contents.
 */
async function downloadTelegramFile(bot: Bot, fileId: string): Promise<Buffer> {
  const file = await bot.api.getFile(fileId);
  const filePath = file.file_path!;
  const token = bot.token;
  const url = `https://api.telegram.org/file/bot${token}/${filePath}`;

  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (chunk: Buffer) => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    });
  });
}

/**
 * Save a Telegram document to the group's workspace and return the container path.
 */
async function saveDocument(
  bot: Bot,
  fileId: string,
  fileName: string,
  groupFolder: string,
): Promise<string | null> {
  try {
    const buffer = await downloadTelegramFile(bot, fileId);
    const docsDir = path.join(GROUPS_DIR, groupFolder, 'documents');
    fs.mkdirSync(docsDir, { recursive: true });
    // Prefix with timestamp to avoid collisions
    const safeName = `${Date.now()}-${fileName.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
    const filePath = path.join(docsDir, safeName);
    fs.writeFileSync(filePath, buffer);
    logger.info(
      { groupFolder, fileName: safeName, size: buffer.length },
      'Saved Telegram document',
    );
    return `/workspace/group/documents/${safeName}`;
  } catch (err) {
    logger.error({ err, fileName }, 'Failed to save Telegram document');
    return null;
  }
}

/**
 * Save a Telegram photo to the group's workspace and return the file path.
 * Downloads the highest-resolution version of the photo.
 */
async function savePhoto(
  bot: Bot,
  photoSizes: Array<{ file_id: string; width: number; height: number }>,
  groupFolder: string,
): Promise<string | null> {
  try {
    // Pick the largest photo
    const largest = photoSizes.reduce((a, b) =>
      a.width * a.height > b.width * b.height ? a : b,
    );
    const buffer = await downloadTelegramFile(bot, largest.file_id);
    const imagesDir = path.join(GROUPS_DIR, groupFolder, 'images');
    fs.mkdirSync(imagesDir, { recursive: true });
    const filename = `${Date.now()}.jpg`;
    const filePath = path.join(imagesDir, filename);
    fs.writeFileSync(filePath, buffer);
    logger.info(
      { groupFolder, filename, size: buffer.length },
      'Saved Telegram photo',
    );
    return `/workspace/group/images/${filename}`;
  } catch (err) {
    logger.error({ err }, 'Failed to save Telegram photo');
    return null;
  }
}

/**
 * Transcribe a voice message using OpenAI Whisper API.
 * Returns the transcript text, or null on failure.
 */
async function transcribeVoice(audioBuffer: Buffer): Promise<string | null> {
  const envVars = readEnvFile(['OPENAI_API_KEY']);
  const apiKey = process.env.OPENAI_API_KEY || envVars.OPENAI_API_KEY;
  if (!apiKey) {
    logger.warn('OPENAI_API_KEY not set, cannot transcribe voice');
    return null;
  }

  try {
    const openai = new OpenAI({ apiKey });
    const file = new File([audioBuffer], 'voice.ogg', { type: 'audio/ogg' });
    const transcription = await openai.audio.transcriptions.create({
      model: 'whisper-1',
      file,
    });
    return transcription.text;
  } catch (err) {
    logger.error({ err }, 'OpenAI transcription failed');
    return null;
  }
}

export class TelegramChannel implements Channel {
  name = 'telegram';

  private bot: Bot | null = null;
  private opts: TelegramChannelOpts;
  private botToken: string;

  constructor(botToken: string, opts: TelegramChannelOpts) {
    this.botToken = botToken;
    this.opts = opts;
  }

  async connect(): Promise<void> {
    this.bot = new Bot(this.botToken, {
      client: {
        baseFetchConfig: { agent: https.globalAgent, compress: true },
      },
    });

    // Command to get chat ID (useful for registration)
    this.bot.command('chatid', (ctx) => {
      const chatId = ctx.chat.id;
      const chatType = ctx.chat.type;
      const chatName =
        chatType === 'private'
          ? ctx.from?.first_name || 'Private'
          : (ctx.chat as any).title || 'Unknown';

      ctx.reply(
        `Chat ID: \`tg:${chatId}\`\nName: ${chatName}\nType: ${chatType}`,
        { parse_mode: 'Markdown' },
      );
    });

    // Command to check bot status
    this.bot.command('ping', (ctx) => {
      ctx.reply(`${ASSISTANT_NAME} is online.`);
    });

    // Telegram bot commands handled above — skip them in the general handler
    // so they don't also get stored as messages. All other /commands flow through.
    const TELEGRAM_BOT_COMMANDS = new Set(['chatid', 'ping']);

    this.bot.on('message:text', async (ctx) => {
      if (ctx.message.text.startsWith('/')) {
        const cmd = ctx.message.text.slice(1).split(/[\s@]/)[0].toLowerCase();
        if (TELEGRAM_BOT_COMMANDS.has(cmd)) return;
      }

      const chatJid = `tg:${ctx.chat.id}`;
      let content = ctx.message.text;
      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id.toString() ||
        'Unknown';
      const sender = ctx.from?.id.toString() || '';
      const msgId = ctx.message.message_id.toString();
      const threadId = ctx.message.message_thread_id;

      // Determine chat name
      const chatName =
        ctx.chat.type === 'private'
          ? senderName
          : (ctx.chat as any).title || chatJid;

      // Translate Telegram @bot_username mentions into TRIGGER_PATTERN format.
      // Telegram @mentions (e.g., @andy_ai_bot) won't match TRIGGER_PATTERN
      // (e.g., ^@Andy\b), so we prepend the trigger when the bot is @mentioned.
      const botUsername = ctx.me?.username?.toLowerCase();
      if (botUsername) {
        const entities = ctx.message.entities || [];
        const isBotMentioned = entities.some((entity) => {
          if (entity.type === 'mention') {
            const mentionText = content
              .substring(entity.offset, entity.offset + entity.length)
              .toLowerCase();
            return mentionText === `@${botUsername}`;
          }
          return false;
        });
        if (isBotMentioned && !TRIGGER_PATTERN.test(content)) {
          content = `@${ASSISTANT_NAME} ${content}`;
        }
      }

      // Store chat metadata for discovery
      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';
      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        chatName,
        'telegram',
        isGroup,
      );

      // Only deliver full message for registered groups
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) {
        logger.debug(
          { chatJid, chatName },
          'Message from unregistered Telegram chat',
        );
        return;
      }

      // Resolve reply context — include quoted message content for the agent
      const replyTo = ctx.message.reply_to_message;
      if (replyTo) {
        const prefix = resolveReply(replyTo, chatJid);
        if (prefix) content = prefix + content;
      }

      // Handle Telegram's quote feature (selected text excerpt)
      if (ctx.message.quote?.text) {
        content = `[Quoted: "${truncate(ctx.message.quote.text, 300)}"]\n${content}`;
      }

      // Resolve t.me/c message links
      content = resolveMessageLinks(content);

      // Deliver message — startMessageLoop() will pick it up
      this.opts.onMessage(chatJid, {
        id: msgId,
        chat_jid: chatJid,
        sender,
        sender_name: senderName,
        content,
        timestamp,
        is_from_me: false,
        thread_id: threadId ? threadId.toString() : undefined,
      });

      logger.info(
        { chatJid, chatName, sender: senderName },
        'Telegram message stored',
      );
    });

    // Handle non-text messages with placeholders so the agent knows something was sent
    const storeNonText = (ctx: any, placeholder: string) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id?.toString() ||
        'Unknown';
      const caption = ctx.message.caption ? ` ${ctx.message.caption}` : '';

      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';
      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        undefined,
        'telegram',
        isGroup,
      );
      this.opts.onMessage(chatJid, {
        id: ctx.message.message_id.toString(),
        chat_jid: chatJid,
        sender: ctx.from?.id?.toString() || '',
        sender_name: senderName,
        content: `${placeholder}${caption}`,
        timestamp,
        is_from_me: false,
      });
    };

    this.bot.on('message:photo', async (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id?.toString() ||
        'Unknown';
      const caption = ctx.message.caption ? ` ${ctx.message.caption}` : '';
      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';
      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        undefined,
        'telegram',
        isGroup,
      );

      const containerPath = await savePhoto(
        this.bot!,
        ctx.message.photo,
        group.folder,
      );
      const placeholder = containerPath
        ? `[Image: ${containerPath}]`
        : '[Image - download failed]';

      this.opts.onMessage(chatJid, {
        id: ctx.message.message_id.toString(),
        chat_jid: chatJid,
        sender: ctx.from?.id?.toString() || '',
        sender_name: senderName,
        content: `${placeholder}${caption}`,
        timestamp,
        is_from_me: false,
      });
      logger.info(
        { chatJid, senderName, containerPath },
        'Telegram photo stored',
      );
    });

    this.bot.on('message:video', (ctx) => storeNonText(ctx, '[Video]'));
    this.bot.on('message:voice', async (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id?.toString() ||
        'Unknown';
      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';
      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        undefined,
        'telegram',
        isGroup,
      );

      let content: string;
      try {
        const buffer = await downloadTelegramFile(
          this.bot!,
          ctx.message.voice.file_id,
        );
        const transcript = await transcribeVoice(buffer);
        content = transcript
          ? `[Voice: ${transcript}]`
          : '[Voice message - transcription unavailable]';
        if (transcript) {
          logger.info(
            { chatJid, senderName, chars: transcript.length },
            'Transcribed voice message',
          );
        }
      } catch (err) {
        logger.error({ err }, 'Failed to process voice message');
        content = '[Voice message - transcription failed]';
      }

      this.opts.onMessage(chatJid, {
        id: ctx.message.message_id.toString(),
        chat_jid: chatJid,
        sender: ctx.from?.id?.toString() || '',
        sender_name: senderName,
        content,
        timestamp,
        is_from_me: false,
      });
    });
    this.bot.on('message:audio', (ctx) => storeNonText(ctx, '[Audio]'));
    this.bot.on('message:document', async (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id?.toString() ||
        'Unknown';
      const caption = ctx.message.caption ? ` ${ctx.message.caption}` : '';
      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';
      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        undefined,
        'telegram',
        isGroup,
      );

      const fileName = ctx.message.document?.file_name || 'file';
      const fileId = ctx.message.document?.file_id;
      let content: string;

      if (fileId) {
        const containerPath = await saveDocument(
          this.bot!,
          fileId,
          fileName,
          group.folder,
        );
        content = containerPath
          ? `[Document: ${containerPath}]${caption}`
          : `[Document: ${fileName} - download failed]${caption}`;
        if (containerPath) {
          logger.info(
            { chatJid, senderName, containerPath },
            'Telegram document stored',
          );
        }
      } else {
        content = `[Document: ${fileName} - no file_id]${caption}`;
      }

      this.opts.onMessage(chatJid, {
        id: ctx.message.message_id.toString(),
        chat_jid: chatJid,
        sender: ctx.from?.id?.toString() || '',
        sender_name: senderName,
        content,
        timestamp,
        is_from_me: false,
      });
    });
    this.bot.on('message:sticker', (ctx) => {
      const emoji = ctx.message.sticker?.emoji || '';
      storeNonText(ctx, `[Sticker ${emoji}]`);
    });
    this.bot.on('message:location', (ctx) => storeNonText(ctx, '[Location]'));
    this.bot.on('message:contact', (ctx) => storeNonText(ctx, '[Contact]'));

    // Handle errors gracefully
    this.bot.catch((err) => {
      logger.error({ err: err.message }, 'Telegram bot error');
    });

    // Start polling — returns a Promise that resolves when started
    return new Promise<void>((resolve) => {
      this.bot!.start({
        onStart: (botInfo) => {
          logger.info(
            { username: botInfo.username, id: botInfo.id },
            'Telegram bot connected',
          );
          console.log(`\n  Telegram bot: @${botInfo.username}`);
          console.log(
            `  Send /chatid to the bot to get a chat's registration ID\n`,
          );
          resolve();
        },
      });
    });
  }

  async sendMessage(
    jid: string,
    text: string,
    threadId?: string,
  ): Promise<void> {
    if (!this.bot) {
      logger.warn('Telegram bot not initialized');
      return;
    }

    try {
      const numericId = jid.replace(/^tg:/, '');
      const options = threadId
        ? { message_thread_id: parseInt(threadId, 10) }
        : {};

      // Telegram has a 4096 character limit per message — split if needed
      const MAX_LENGTH = 4096;
      if (text.length <= MAX_LENGTH) {
        await sendTelegramMessage(this.bot.api, numericId, text, options);
      } else {
        for (let i = 0; i < text.length; i += MAX_LENGTH) {
          await sendTelegramMessage(
            this.bot.api,
            numericId,
            text.slice(i, i + MAX_LENGTH),
            options,
          );
        }
      }
      logger.info(
        { jid, length: text.length, threadId },
        'Telegram message sent',
      );
    } catch (err) {
      logger.error({ jid, err }, 'Failed to send Telegram message');
    }
  }

  isConnected(): boolean {
    return this.bot !== null;
  }

  ownsJid(jid: string): boolean {
    return jid.startsWith('tg:');
  }

  async disconnect(): Promise<void> {
    if (this.bot) {
      this.bot.stop();
      this.bot = null;
      logger.info('Telegram bot stopped');
    }
  }

  async setTyping(jid: string, isTyping: boolean): Promise<void> {
    if (!this.bot || !isTyping) return;
    try {
      const numericId = jid.replace(/^tg:/, '');
      await this.bot.api.sendChatAction(numericId, 'typing');
    } catch (err) {
      logger.debug({ jid, err }, 'Failed to send Telegram typing indicator');
    }
  }
}

registerChannel('telegram', (opts: ChannelOpts) => {
  const envVars = readEnvFile(['TELEGRAM_BOT_TOKEN']);
  const token =
    process.env.TELEGRAM_BOT_TOKEN || envVars.TELEGRAM_BOT_TOKEN || '';
  if (!token) {
    logger.warn('Telegram: TELEGRAM_BOT_TOKEN not set');
    return null;
  }
  return new TelegramChannel(token, opts);
});
