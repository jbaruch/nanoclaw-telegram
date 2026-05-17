import { Channel, NewMessage } from './types.js';
import { formatLocalTime } from './timezone.js';
import { logger } from './logger.js';
import { parseTextStyles, ChannelType } from './text-styles.js';

const MAX_OUTBOUND_LENGTH = 50_000;

export function escapeXml(s: string): string {
  if (!s) return '';
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Per-invocation context options for the formatted agent prompt.
 *
 * - `timezone` is the wall-clock zone used for the per-message
 *   `<message time="...">` display (pre-existing behaviour).
 * - `contextTag` is the pre-built `<context ... />` header line.
 *   Optional for backward compatibility — when omitted, the legacy
 *   single-attribute `<context timezone="..." />` is emitted (the
 *   pre-#576 shape). The orchestrator's main message-handling paths
 *   build the enriched tag via `agent-context.ts` and pass it
 *   through; legacy / test paths that just need a plain timezone
 *   keep calling with a string.
 */
export interface MessageFormatContext {
  timezone: string;
  contextTag?: string;
}

export function formatMessages(
  messages: NewMessage[],
  ctxOrTimezone: string | MessageFormatContext,
): string {
  const ctx: MessageFormatContext =
    typeof ctxOrTimezone === 'string'
      ? { timezone: ctxOrTimezone }
      : ctxOrTimezone;

  const lines = messages.map((m) => {
    const displayTime = formatLocalTime(m.timestamp, ctx.timezone);
    const idAttr = m.id ? ` id="${escapeXml(m.id)}"` : '';
    const replyAttr = m.reply_to_message_id
      ? ` reply_to="${escapeXml(m.reply_to_message_id)}"`
      : '';
    const replySnippet =
      m.reply_to_message_content && m.reply_to_sender_name
        ? `\n  <quoted_message from="${escapeXml(m.reply_to_sender_name)}">${escapeXml(m.reply_to_message_content)}</quoted_message>`
        : '';
    return `<message${idAttr} sender="${escapeXml(m.sender_name)}" time="${escapeXml(displayTime)}"${replyAttr}>${replySnippet}${escapeXml(m.content)}</message>`;
  });

  const contextTag =
    ctx.contextTag ?? `<context timezone="${escapeXml(ctx.timezone)}" />`;

  return `${contextTag}\n<messages>\n${lines.join('\n')}\n</messages>`;
}

export function stripInternalTags(text: string): string {
  return text.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
}

export function formatOutbound(rawText: string, channel?: ChannelType): string {
  let text = stripInternalTags(rawText);
  if (!text) return '';
  if (text.length > MAX_OUTBOUND_LENGTH) {
    logger.warn(
      { originalLength: text.length },
      'Truncating oversized outbound message',
    );
    text = text.slice(0, MAX_OUTBOUND_LENGTH) + '\n\n[Message truncated]';
  }
  return channel ? parseTextStyles(text, channel) : text;
}

export function routeOutbound(
  channels: Channel[],
  jid: string,
  text: string,
): Promise<string | void> {
  const channel = channels.find((c) => c.ownsJid(jid) && c.isConnected());
  if (!channel) throw new Error(`No channel for JID: ${jid}`);
  return channel.sendMessage(jid, text);
}

export function findChannel(
  channels: Channel[],
  jid: string,
): Channel | undefined {
  return channels.find((c) => c.ownsJid(jid));
}
