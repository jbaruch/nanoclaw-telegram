import fs from 'fs';
import https from 'https';
import path from 'path';
import { Api, Bot, GrammyError, InputFile } from 'grammy';
import OpenAI from 'openai';

import { ASSISTANT_NAME, GROUPS_DIR, TRIGGER_PATTERN } from '../config.js';
import {
  getLatestMessage,
  getMessageById,
  messageExistsInDifferentChat,
  storeReaction,
} from '../db.js';
import { readEnvFile } from '../env.js';
import { logger } from '../logger.js';
import { registerChannel, ChannelOpts } from './registry.js';
import { sanitizeTelegramHtml } from './telegram-sanitize.js';
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
 * Send a message with Telegram HTML parse mode, falling back to plain text.
 * Supports: <b>bold</b>, <i>italic</i>, <s>strikethrough</s>, <u>underline</u>,
 * <code>inline code</code>, <pre>code blocks</pre>, <blockquote>quotes</blockquote>,
 * <a href="url">links</a>, <tg-spoiler>spoilers</tg-spoiler>
 */
async function sendTelegramMessage(
  api: { sendMessage: Api['sendMessage'] },
  chatId: string | number,
  text: string,
  options: {
    message_thread_id?: number;
    reply_parameters?: { message_id: number };
    // When `true`, `text` is already sanitized HTML — skip the
    // sanitize pass. Set by callers (`TelegramChannel.sendMessage`,
    // `sendPoolMessage`) that sanitize ONCE before splitting into
    // chunks (#282): pre-fix, splitMessage ran on raw markdown and
    // could cut mid-construct (`[label](https://...)` split between
    // `]` and `(`), leaving the sanitizer with a half-construct in
    // chunk 1 and a dangling fragment in chunk 2. Sanitizing first
    // produces explicit HTML tag boundaries that splitMessage can
    // detect cleanly, but only if we don't double-sanitize each
    // chunk here — `sanitize(sanitize(x))` is idempotent for valid
    // HTML but a second pass on a chunk that ends mid-tag (e.g.
    // hard-cut last-resort fallback) would re-process the orphan
    // and amplify the corruption.
    preSanitized?: boolean;
  } = {},
): Promise<number | undefined> {
  // Idempotent Markdown→HTML pass — agents sometimes produce `**bold**` or
  // `[text](url)` despite being told to use HTML. Well-formed HTML passes
  // through unchanged; URLs/emails/existing tags are protected.
  const sanitized = options.preSanitized ? text : sanitizeTelegramHtml(text);
  const rawChanged = sanitized !== text;
  logger.debug(
    {
      chatId,
      rawLen: text.length,
      rawPreview: text.slice(0, 80),
      sanitizedLen: sanitized.length,
      sanitizedPreview: sanitized.slice(0, 80),
      rawChanged,
      hasReplyTo: Boolean(options.reply_parameters?.message_id),
    },
    '[send] sendTelegramMessage entered',
  );
  // Strip the local `preSanitized` flag from the API payload — it's
  // an internal sanitize-once marker, not a Telegram Bot API field.
  const { preSanitized: _preSanitized, ...apiOptions } = options;
  try {
    const msg = await api.sendMessage(chatId, sanitized, {
      ...apiOptions,
      parse_mode: 'HTML',
    });
    logger.debug(
      { chatId, messageId: msg.message_id, sanitizedLen: sanitized.length },
      '[send] HTML send OK',
    );
    return msg.message_id;
  } catch (err) {
    // Narrow the fallback to the specific Telegram-side HTML parse
    // rejection (#414). Pre-#414, the catch swallowed every error —
    // including `HttpError` (network: DNS hiccup, reset, timeout),
    // 5xx `GrammyError`s, 429 rate-limits, and anything else from
    // grammy's transport layer — and re-sent as plain text. For
    // network failures the first send never reached Telegram, so a
    // second send isn't a "fallback" but a fresh attempt that may
    // duplicate, may also fail, and degrades the user-visible
    // formatting unnecessarily; for rate-limits a retry from inside
    // the catch hits the same 429 and burns the limit faster. Re-
    // throw everything except the specific 400 + "can't parse
    // entities" case the plain-text fallback was actually built for,
    // and let the caller's outer error handling decide.
    if (
      !(
        err instanceof GrammyError &&
        err.error_code === 400 &&
        /can't parse entities/i.test(err.description)
      )
    ) {
      throw err;
    }
    // Fallback: HTML parsing failed. The user-facing send ships a
    // marked-degraded version (raw text prefixed with a visible warning)
    // so the user knows the formatting they're seeing is a fallback,
    // not the agent's intent. Pre-fix, the fallback shipped the
    // ORIGINAL `text` silently — a sanitizer bug that produced invalid
    // HTML, 400'd, and fell back was invisible to the operator because
    // the user just saw raw Markdown that LOOKED LIKE a hook didn't
    // fire. See jbaruch/nanoclaw#278 (msg 6240, the audit-report
    // message itself).
    //
    // The WARN log carries failure metadata only — `err`, `chatId`,
    // and lengths. Zero user content reaches the log sink:
    // `jbaruch/coding-policy: no-secrets` says "Never log secrets —
    // not at any log level" and "Sanitize or redact sensitive values
    // before they reach any logging or monitoring system." User text
    // can carry pasted tokens, third-party API responses surfaced via
    // tool results, or other credentials, and a 200-char slice
    // doesn't sanitize them. Operators correlate the WARN line with
    // a specific message via `chatId` plus the timestamp and pull
    // the actual text from the source (chat history, DB) for repro.
    // The 400's `err` object already carries the byte offset and
    // expected/found tag from Telegram (e.g. "Unmatched end tag at
    // byte offset 1049, expected </b>, found </code>") so phase
    // diagnosis often doesn't need the input at all.
    //
    // For full-body in-the-moment repro, dev/CI flips
    // `DEV_NO_HTML_FALLBACK=1` and the original 400 surfaces
    // directly — the operator sees the actual sanitizer output via
    // the throwing path, without touching the log sink. Production
    // keeps the fallback for user-friendliness.
    logger.warn(
      {
        err,
        chatId,
        rawLen: text.length,
        sanitizedLen: sanitized.length,
      },
      '[send] HTML send failed, falling back to plain text (user will see tag-stripped plain text with warning prefix)',
    );
    if (process.env.DEV_NO_HTML_FALLBACK === '1') throw err;
    // Strip HTML tags + decode entities before shipping as plain
    // text. After the sanitize-then-split reorder (#282), `text` is
    // already-sanitized HTML when `preSanitized` is true; shipping
    // it verbatim in the fallback would render `<i>great</i>` as
    // literal-tag text to the user. `htmlToPlainText` recovers a
    // readable rendering. For pre-sanitized callers this loses
    // formatting (the user sees "great" instead of "_great_"), but
    // the alternative — readable raw markdown — is no longer
    // reachable from a per-chunk caller after the reorder. The
    // visible `⚠️ formatting failed` prefix tells the user they're
    // seeing a degraded view either way.
    //
    // Truncate the body so the prefix never pushes the fallback
    // over Telegram's 4096-char message limit — a recoverable
    // HTML-parse error becoming a "fallback also failed" lost
    // message would be strictly worse than the original symptom.
    const prefix = '⚠️ formatting failed; raw text below\n\n';
    const plain = htmlToPlainText(text);
    const degraded = `${prefix}${plain.slice(0, MAX_LENGTH - prefix.length)}`;
    try {
      const msg = await api.sendMessage(chatId, degraded, apiOptions);
      logger.warn(
        { chatId, messageId: msg.message_id },
        '[send] Plain-text fallback OK — DB row will still be written by the caller',
      );
      return msg.message_id;
    } catch (fallbackErr) {
      // BOTH sends failed. The message may or may not have reached
      // Telegram (depending on where the second failure occurred). Log
      // loudly and re-throw so the caller's try/catch can decide.
      logger.error(
        {
          err: fallbackErr,
          chatId,
          rawLen: text.length,
          originalHtmlErr: err,
        },
        '[send] Both HTML and plain-text sends failed — message may be lost',
      );
      throw fallbackErr;
    }
  }
}

/**
 * Strip Telegram HTML tags and decode the entities Telegram uses to
 * encode special characters in content. Used by `sendTelegramMessage`'s
 * fallback path to recover a readable plain-text rendering when the
 * HTML send 400s and `text` is already sanitized HTML (post-#282
 * sanitize-then-split). `sendFile`'s caption fallback does NOT use
 * this — its `caption` parameter is the original raw markdown the
 * caller passed in, which is already readable as plain text.
 *
 * Pre-sanitized HTML may contain only the Telegram-allowed tags
 * (`<b>`, `<i>`, `<u>`, `<s>`, `<code>`, `<pre>`, `<blockquote>`,
 * `<a>`, `<tg-spoiler>`); stray tags are HTML-escaped at sanitize
 * time so they're not real tags here.
 *
 * Link URLs are preserved in the fallback as `label (url)` so the
 * user can still reach them — Copilot review on PR #308 caught that
 * a naive tag-strip drops `<a href="…">` entirely and leaves only
 * the label, removing important information at exactly the moment
 * the user most needs it (formatting failed, but at least the URL
 * should survive). The `<a>`-specific replace runs BEFORE the
 * generic tag strip so the href is captured before tags are
 * removed; remaining replacements are order-independent on this
 * domain.
 */
function htmlToPlainText(s: string): string {
  return s
    .replace(/<a\s+href="([^"]*)"[^>]*>([^<]*)<\/a>/g, '$2 ($1)')
    .replace(/<\/?[a-zA-Z][^>]*>/g, '')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&');
}

const MAX_LENGTH = 4096;
// Telegram Bot API caps `sendDocument` (and other media) caption length
// at 1024 chars. Used by `sendFile`'s degraded-fallback path to size
// the truncation so the prefix doesn't push an otherwise-valid caption
// over the limit and turn a recoverable HTML-parse error into a
// "fallback also failed" lost attachment.
const MAX_CAPTION_LENGTH = 1024;

// Slack-style shortcode → Unicode mapping for the 73 Telegram-supported
// reactions (Bot API 7.x). Covers every entry in
// TELEGRAM_ALLOWED_REACTIONS below. Agents and skills emit shortcodes
// like `thumbs_up` or `:thumbs_up:` because that's the lingua franca
// across Slack/GitHub/GitLab/Mattermost; without normalization those
// shortcodes failed `TELEGRAM_ALLOWED_REACTIONS.has(...)` and fell
// back silently to 👍 — see #161 for the production sighting where
// only ~16 emoji worked via shortcode while ~57 silently failed.
//
// Multiple shortcodes can map to the same Unicode char where common
// aliases exist (e.g. both `+1` and `thumbs_up` → 👍).
//
// IMPORTANT: every value in this map MUST also exist in
// TELEGRAM_ALLOWED_REACTIONS. Drift between the two means a
// successfully-mapped shortcode would still fall back to 👍 at the
// allowed-reactions gate; the test suite enforces this invariant.
const EMOJI_SHORTCODE_TO_UNICODE: Record<string, string> = {
  // — affirmation & engagement
  thumbs_up: '👍',
  '+1': '👍',
  thumbsup: '👍',
  thumbs_down: '👎',
  '-1': '👎',
  thumbsdown: '👎',
  heart: '❤',
  red_heart: '❤',
  fire: '🔥',
  smiling_face_with_hearts: '🥰',
  smiling_face_with_3_hearts: '🥰',
  clap: '👏',
  beaming_face_with_smiling_eyes: '😁',
  grin: '😁',
  thinking_face: '🤔',
  thinking: '🤔',
  exploding_head: '🤯',
  shocked_face: '😱',
  scream: '😱',
  face_with_symbols_on_mouth: '🤬',
  cursing_face: '🤬',
  crying_face: '😢',
  cry: '😢',
  party_popper: '🎉',
  tada: '🎉',
  smiling_face_with_starry_eyes: '🤩',
  star_struck: '🤩',
  face_vomiting: '🤮',
  vomit: '🤮',
  pile_of_poo: '💩',
  poop: '💩',
  hankey: '💩',
  shit: '💩',
  folded_hands: '🙏',
  pray: '🙏',
  ok_hand: '👌',
  dove: '🕊',
  dove_of_peace: '🕊',
  clown_face: '🤡',
  clown: '🤡',
  yawning_face: '🥱',
  yawn: '🥱',
  woozy_face: '🥴',
  smiling_face_with_heart_eyes: '😍',
  heart_eyes: '😍',
  whale: '🐳',
  heart_on_fire: '❤‍🔥',
  new_moon_face: '🌚',
  new_moon_with_face: '🌚',
  hot_dog: '🌭',
  hotdog: '🌭',
  hundred_points: '💯',
  hundred: '💯',
  '100': '💯',
  rolling_on_the_floor_laughing: '🤣',
  rofl: '🤣',
  high_voltage: '⚡',
  zap: '⚡',
  banana: '🍌',
  trophy: '🏆',
  broken_heart: '💔',
  face_with_raised_eyebrow: '🤨',
  raised_eyebrow: '🤨',
  neutral_face: '😐',
  strawberry: '🍓',
  bottle_with_popping_cork: '🍾',
  champagne: '🍾',
  kiss_mark: '💋',
  kiss: '💋',
  middle_finger: '🖕',
  fu: '🖕',
  smiling_face_with_horns: '😈',
  smiling_imp: '😈',
  sleeping_face: '😴',
  sleeping: '😴',
  loudly_crying_face: '😭',
  sob: '😭',
  nerd_face: '🤓',
  nerd: '🤓',
  ghost: '👻',
  technologist: '👨‍💻',
  man_technologist: '👨‍💻',
  eyes: '👀',
  jack_o_lantern: '🎃',
  see_no_evil: '🙈',
  see_no_evil_monkey: '🙈',
  smiling_face_with_halo: '😇',
  innocent: '😇',
  fearful_face: '😨',
  fearful: '😨',
  handshake: '🤝',
  writing_hand: '✍',
  pencil: '✍',
  smiling_face_with_open_hands: '🤗',
  hugs: '🤗',
  saluting_face: '🫡',
  salute: '🫡',
  santa_claus: '🎅',
  santa: '🎅',
  christmas_tree: '🎄',
  snowman: '☃',
  nail_polish: '💅',
  zany_face: '🤪',
  zany: '🤪',
  moai: '🗿',
  cool: '🆒',
  heart_with_arrow: '💘',
  cupid: '💘',
  hear_no_evil: '🙉',
  hear_no_evil_monkey: '🙉',
  unicorn: '🦄',
  unicorn_face: '🦄',
  face_blowing_a_kiss: '😘',
  kissing_heart: '😘',
  pill: '💊',
  speak_no_evil: '🙊',
  speak_no_evil_monkey: '🙊',
  smiling_face_with_sunglasses: '😎',
  sunglasses: '😎',
  alien_monster: '👾',
  space_invader: '👾',
  man_shrugging: '🤷‍♂',
  shrug: '🤷',
  person_shrugging: '🤷',
  woman_shrugging: '🤷‍♀',
  pouting_face: '😡',
  enraged: '😡',
  rage: '😡',
};

/**
 * Normalize a reaction emoji input to a Telegram-supported Unicode
 * character, accepting both shortcodes (`thumbs_up`, `:thumbs_up:`)
 * and raw Unicode (`👍`, `❤️`). Returns the input unchanged if it
 * can't be mapped; the caller (`sendReaction`) then runs the
 * TELEGRAM_ALLOWED_REACTIONS gate, which falls back to 👍 with a
 * warn log so unmapped inputs are visible rather than silently
 * accepted.
 *
 * Strips:
 *   - U+FE0F (variation selector 16) so emoji-presentation forms
 *     like `❤️`, `☃️`, `✍️`, `🤷‍♂️` match the no-VS16 entries in
 *     TELEGRAM_ALLOWED_REACTIONS. Telegram's reaction set uses the
 *     bare codepoints; agents and clients commonly emit the
 *     VS16-suffixed form.
 *   - Surrounding `:` so both `thumbs_up` and `:thumbs_up:` map —
 *     Slack-style colon delimiters are common in agent output.
 */
export function normalizeReactionEmoji(input: string): string {
  // Strip U+FE0F (VARIATION SELECTOR-16). Written as the explicit
  // `️` escape — an invisible literal in the regex source is
  // hard to audit and trivially altered by editor reformatting.
  const noVS16 = input.replace(/\uFE0F/g, '');
  if (TELEGRAM_ALLOWED_REACTIONS.has(noVS16)) return noVS16;
  const stripped = noVS16.replace(/^:|:$/g, '');
  // Guard against prototype pollution: indexing a plain object with
  // a string like `toString` / `__proto__` returns an inherited
  // value (function / object) from Object.prototype, not undefined.
  // Object.hasOwn restricts the lookup to own properties so the
  // function's string-or-input contract holds for any input.
  const fromShortcode = Object.hasOwn(EMOJI_SHORTCODE_TO_UNICODE, stripped)
    ? EMOJI_SHORTCODE_TO_UNICODE[stripped]
    : undefined;
  return fromShortcode ?? input;
}

/**
 * Predicate exposed for tests so the forward drift invariant
 * (every value in `EMOJI_SHORTCODE_TO_UNICODE` is in
 * `TELEGRAM_ALLOWED_REACTIONS`) can be asserted per-shortcode with
 * an `it()` per row, producing readable test names. The Set itself
 * is also exposed (`_TELEGRAM_ALLOWED_REACTIONS`) for the inverse
 * direction (#285) where the test iterates the allowed set; use
 * this predicate for the forward direction and the Set for the
 * inverse.
 *
 * @internal
 */
export function _isAllowedReaction(emoji: string): boolean {
  return TELEGRAM_ALLOWED_REACTIONS.has(emoji);
}

/**
 * The shortcode → Unicode map exposed for tests so the drift
 * invariant (every value is in TELEGRAM_ALLOWED_REACTIONS) can be
 * asserted exhaustively rather than against a hardcoded list.
 *
 * @internal
 */
export const _EMOJI_SHORTCODE_TO_UNICODE = EMOJI_SHORTCODE_TO_UNICODE;

// Telegram's allowed reaction emoji (as of Bot API 7.x)
const TELEGRAM_ALLOWED_REACTIONS = new Set([
  '👍',
  '👎',
  '❤',
  '🔥',
  '🥰',
  '👏',
  '😁',
  '🤔',
  '🤯',
  '😱',
  '🤬',
  '😢',
  '🎉',
  '🤩',
  '🤮',
  '💩',
  '🙏',
  '👌',
  '🕊',
  '🤡',
  '🥱',
  '🥴',
  '😍',
  '🐳',
  '❤‍🔥',
  '🌚',
  '🌭',
  '💯',
  '🤣',
  '⚡',
  '🍌',
  '🏆',
  '💔',
  '🤨',
  '😐',
  '🍓',
  '🍾',
  '💋',
  '🖕',
  '😈',
  '😴',
  '😭',
  '🤓',
  '👻',
  '👨‍💻',
  '👀',
  '🎃',
  '🙈',
  '😇',
  '😨',
  '🤝',
  '✍',
  '🤗',
  '🫡',
  '🎅',
  '🎄',
  '☃',
  '💅',
  '🤪',
  '🗿',
  '🆒',
  '💘',
  '🙉',
  '🦄',
  '😘',
  '💊',
  '🙊',
  '😎',
  '👾',
  '🤷‍♂',
  '🤷',
  '🤷‍♀',
  '😡',
]);

/**
 * The allowed-reactions set exposed for tests so the inverse drift
 * invariant (every Telegram-allowed reaction has at least one
 * shortcode pointing at it) can be asserted exhaustively. Catches
 * the case where Telegram's allowed set grows but the shortcode map
 * doesn't — those new emoji become unreachable from agents that
 * emit shortcode form (every skill that calls `react_to_message`
 * passes through `normalizeReactionEmoji`'s shortcode lookup).
 *
 * @internal
 */
export const _TELEGRAM_ALLOWED_REACTIONS: ReadonlySet<string> =
  TELEGRAM_ALLOWED_REACTIONS;

/**
 * Build a "safe to split before this index" map for HTML-aware
 * chunking. Position `i` is safe iff splitting `text` at `i`
 * produces two halves that are each well-formed Telegram HTML —
 * specifically, neither half is inside a `<...>` tag and the
 * paired-tag depth at `i` is zero (no opening `<b>` orphaned in
 * the left half without its closing `</b>`, and vice versa).
 *
 * Self-closing tags (`<br/>`) and Phase 1c-protected URLs (which
 * appear as bare strings, not tags) don't change depth. Stray
 * tags from the agent are HTML-escaped by the sanitizer before
 * they reach here, so any real `<` in the input is the start of
 * a paired or self-closing Telegram-allowed tag.
 *
 * Position 0 is safe (start of string); position `text.length` is
 * safe iff depth ended at zero (well-formed input).
 */
function buildSafeSplitMap(text: string): boolean[] {
  const safe = new Array(text.length + 1).fill(false);
  safe[0] = true;
  let depth = 0;
  let i = 0;
  while (i < text.length) {
    if (text[i] === '<') {
      const closeIdx = text.indexOf('>', i + 1);
      if (closeIdx === -1) {
        // Unterminated `<` — treat the rest of the string as
        // unsafe to split inside. Should never happen on
        // sanitized input but we degrade gracefully.
        return safe;
      }
      const tag = text.slice(i, closeIdx + 1);
      const isClosing = tag.startsWith('</');
      const isSelfClosing = tag.endsWith('/>');
      if (isClosing) depth = Math.max(0, depth - 1);
      else if (!isSelfClosing) depth++;
      i = closeIdx + 1;
      if (depth === 0) safe[i] = true;
    } else {
      i++;
      if (depth === 0) safe[i] = true;
    }
  }
  return safe;
}

/**
 * Walk back through `text` to find the latest occurrence of `needle`
 * whose end-position (`idx + needle.length`) is `<= at` AND marked
 * safe in `safe`. Returns the safe end-position, or -1 if no safe
 * match exists.
 *
 * Searches from `at - needle.length` so the returned end-position
 * never exceeds `at`. Pre-fix the function searched from `at`
 * directly, which let `lastIndexOf('\n', MAX_LENGTH)` return a
 * needle starting at index MAX_LENGTH — the resulting end-position
 * MAX_LENGTH + 1 was beyond the caller's intended budget and
 * produced an oversized chunk (Copilot review on PR #308).
 */
function lastSafeIndexOf(
  text: string,
  needle: string,
  at: number,
  safe: boolean[],
): number {
  let from = at - needle.length;
  while (from >= 0) {
    const idx = text.lastIndexOf(needle, from);
    if (idx === -1) return -1;
    const end = idx + needle.length;
    if (safe[end]) return end;
    from = idx - 1;
  }
  return -1;
}

/**
 * Split text into chunks that respect both content boundaries and
 * HTML structure. Input is expected to be sanitized HTML (callers
 * `TelegramChannel.sendMessage` and `sendPoolMessage` run
 * `sanitizeTelegramHtml` first, #282), but the function tolerates
 * raw text — `buildSafeSplitMap` returns "safe everywhere" for
 * input without `<...>` tokens.
 *
 * Priority within safe positions (#286):
 *   1. Code block boundary (`\n```\n` in raw markdown — relic of
 *      pre-sanitize input — OR `</pre>` followed by `\n` in
 *      sanitized HTML).
 *   2. Paragraph boundary (`\n\n`).
 *   3. Single newline.
 *   4. Space.
 *   5. Latest safe position ≤ MAX_LENGTH.
 *   6. Hard cut at MAX_LENGTH (last resort — only fires when even
 *      no safe position exists, e.g. a single `<pre>...</pre>`
 *      block longer than MAX_LENGTH; produces a chunk with an
 *      orphan tag and the fallback path will land it).
 *
 * Pre-fix the priority logic enforced a 30%-of-MAX_LENGTH minimum
 * on paragraph/newline/space boundaries, which meant a clean
 * `\n\n` at byte 1100 of a 4096-char chunk fell through to a
 * mid-sentence space cut or hard cut at 4096. Threshold lowered to
 * 5% (paragraphs/newlines) and 30% (spaces — unchanged because a
 * mid-paragraph space cut at byte 50 still looks worse than at
 * 1500 for prose continuity).
 */
export function splitMessage(text: string): string[] {
  if (text.length <= MAX_LENGTH) return [text];

  const chunks: string[] = [];
  let remaining = text;

  while (remaining.length > MAX_LENGTH) {
    const safe = buildSafeSplitMap(remaining);
    let splitAt = -1;

    // 1. Code-block boundary (raw markdown fence on its own line).
    const codeBlockPattern = /\n```\n/g;
    let match;
    while ((match = codeBlockPattern.exec(remaining)) !== null) {
      const pos = match.index + match[0].length;
      if (pos <= MAX_LENGTH && pos > splitAt && safe[pos]) splitAt = pos;
    }
    // 1b. Sanitized-HTML fenced-code boundary: `</pre>` followed
    // by a newline (the sanitizer replaces ```...``` with
    // `<pre>escaped</pre>`).
    if (splitAt === -1) {
      const preEndPattern = /<\/pre>\n?/g;
      while ((match = preEndPattern.exec(remaining)) !== null) {
        const pos = match.index + match[0].length;
        if (pos <= MAX_LENGTH && pos > splitAt && safe[pos]) splitAt = pos;
      }
    }

    // 2. Paragraph boundary (double newline) — 5% minimum.
    if (splitAt === -1) {
      const pos = lastSafeIndexOf(remaining, '\n\n', MAX_LENGTH, safe);
      if (pos > MAX_LENGTH * 0.05) splitAt = pos;
    }

    // 3. Single newline — 5% minimum.
    if (splitAt === -1) {
      const pos = lastSafeIndexOf(remaining, '\n', MAX_LENGTH, safe);
      if (pos > MAX_LENGTH * 0.05) splitAt = pos;
    }

    // 4. Space — 30% minimum (mid-paragraph cut at byte 50 looks
    // worse than at 1500 for prose continuity).
    if (splitAt === -1) {
      const pos = lastSafeIndexOf(remaining, ' ', MAX_LENGTH, safe);
      if (pos > MAX_LENGTH * 0.3) splitAt = pos;
    }

    // 5. Latest safe position ≤ MAX_LENGTH (no content-boundary
    // marker matched, but the structural-safety map still has a
    // depth-0, not-inside-tag position we can use).
    if (splitAt === -1) {
      for (let i = MAX_LENGTH; i > 0; i--) {
        if (safe[i]) {
          splitAt = i;
          break;
        }
      }
    }

    // 6. Hard cut at MAX_LENGTH — last resort. Only fires when no
    // safe position exists ≤ MAX_LENGTH (e.g. a single
    // `<pre>...</pre>` block larger than the limit). The chunk
    // will have an orphan tag and trigger the sanitizer fallback,
    // which is strictly less bad than a "fallback also failed"
    // lost message.
    if (splitAt === -1 || splitAt === 0) splitAt = MAX_LENGTH;

    chunks.push(remaining.slice(0, splitAt));
    remaining = remaining.slice(splitAt);
  }

  if (remaining) chunks.push(remaining);
  return chunks;
}

/**
 * Truncate a string to a maximum length, appending "..." if truncated.
 */
function truncate(s: string, max = 120): string {
  return s.length > max ? s.slice(0, max) + '...' : s;
}

/**
 * Cross-chat reply_to safety check for outbound Telegram sends.
 *
 * Telegram message IDs are per-chat sequential, so a `replyToMessageId`
 * captured in chat A can both (a) coincidentally match an unrelated
 * message in chat B and (b) routinely match a legitimate same-chat
 * reply target whose id ALSO happens to exist in some other chat.
 * Either case alone — "exists somewhere else" — therefore can't be the
 * trigger to drop the reply, or we'd strip threading from most
 * legitimate same-chat replies in any deployment with more than one
 * Telegram chat.
 *
 * Predicate (true == it's safe to attach `reply_parameters`):
 *  - the id is present in the target chat → safe (positive evidence
 *    of a local target wins over any cross-chat occurrence).
 *  - the id is absent from our DB entirely → safe; let Telegram be
 *    authoritative on existence so a missing-from-DB reply target
 *    (e.g. a message from before the orchestrator started) isn't
 *    silently dropped.
 *  - the id is present ONLY in some OTHER chat → unsafe; drop with a
 *    warn log so the send doesn't 400 or attach the reply arrow to a
 *    coincidentally-matching unrelated message.
 *
 * `getMessageById` is the positive check; `messageExistsInDifferentChat`
 * is only consulted when the positive check failed, so the
 * pathological "shared id in BOTH chats" case (per-chat-sequential ids)
 * collapses to "safe" — we'd rather keep a legitimate reply arrow than
 * paranoia-drop it.
 */
function safeReplyToForChat(replyToMessageId: string, jid: string): boolean {
  if (getMessageById(replyToMessageId, jid)) return true;
  if (messageExistsInDifferentChat(replyToMessageId, jid)) {
    logger.warn(
      { jid, replyToMessageId },
      'Dropping cross-chat reply_to (id belongs to a different chat)',
    );
    return false;
  }
  return true;
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

// Bot pool for agent teams: send-only Api instances (no polling)
const poolApis: Api[] = [];
// Maps "{groupFolder}:{senderName}" → pool Api index for stable assignment
const senderBotMap = new Map<string, number>();
let nextPoolIndex = 0;

/**
 * Initialize send-only Api instances for the bot pool.
 * Each pool bot can send messages but doesn't poll for updates.
 */
export async function initBotPool(tokens: string[]): Promise<void> {
  for (const token of tokens) {
    try {
      const api = new Api(token);
      const me = await api.getMe();
      poolApis.push(api);
      logger.info(
        { username: me.username, id: me.id, poolSize: poolApis.length },
        'Pool bot initialized',
      );
    } catch (err) {
      logger.error({ err }, 'Failed to initialize pool bot');
    }
  }
  if (poolApis.length > 0) {
    logger.info({ count: poolApis.length }, 'Telegram bot pool ready');
  }
}

/**
 * Send a message via a pool bot assigned to the given sender name.
 * Assigns bots round-robin on first use; subsequent messages from the
 * same sender in the same group always use the same bot.
 * On first assignment, renames the bot to match the sender's role.
 */
export async function sendPoolMessage(
  chatId: string,
  text: string,
  sender: string,
  groupFolder: string,
): Promise<string | undefined> {
  logger.debug(
    {
      chatId,
      sender,
      groupFolder,
      textLen: text.length,
      preview: text.slice(0, 80),
      poolSize: poolApis.length,
    },
    '[send] sendPoolMessage entered',
  );
  if (poolApis.length === 0) {
    // No pool bots configured — return undefined without sending.
    // Earlier comment claimed "fall back to main bot sendMessage via
    // channel" but no such fallback is implemented here; callers that
    // observe undefined must treat it as a hard send failure for the
    // pool path (the IPC handler in `src/ipc.ts` logs the returned id
    // and stores it on the bot row, so `undefined` correctly surfaces
    // as "no Telegram id recorded" rather than a silent drop).
    logger.warn(
      { chatId, sender, groupFolder },
      '[send] sendPoolMessage called with empty pool — returning undefined (message NOT sent; pool-identity sends require TELEGRAM_BOT_POOL to be configured)',
    );
    return undefined;
  }

  const key = `${groupFolder}:${sender}`;
  let idx = senderBotMap.get(key);
  if (idx === undefined) {
    idx = nextPoolIndex % poolApis.length;
    nextPoolIndex++;
    senderBotMap.set(key, idx);
    // Rename the bot to match the sender's role, then wait for Telegram to propagate
    try {
      await poolApis[idx].setMyName(sender);
      await new Promise((r) => setTimeout(r, 2000));
      logger.info(
        { sender, groupFolder, poolIndex: idx },
        'Assigned and renamed pool bot',
      );
    } catch (err) {
      logger.warn(
        { sender, err },
        'Failed to rename pool bot (sending anyway)',
      );
    }
  }

  const api = poolApis[idx];
  try {
    const numericId = chatId.replace(/^tg:/, '');
    // Sanitize-once before split (#282) — see TelegramChannel.sendMessage
    // for the rationale. Pool sends use the same shape so a long pool
    // message with markdown markers crossing the chunk boundary doesn't
    // half-render either side.
    const sanitized = sanitizeTelegramHtml(text);
    const chunks = splitMessage(sanitized);
    logger.debug(
      { chatId, sender, poolIndex: idx, chunkCount: chunks.length },
      '[send] sendPoolMessage: sending chunks',
    );
    // Return the LAST chunk's Telegram ID — matches `channel.sendMessage`
    // above and is the one reply_to threads point at. Callers that want
    // per-chunk IDs would need to change the signature; no current caller
    // cares (the stored `messages.db` row represents the full text, so
    // one ID is enough to trace the send).
    let lastMsgId: number | undefined;
    for (let i = 0; i < chunks.length; i++) {
      lastMsgId = await sendTelegramMessage(api, numericId, chunks[i], {
        preSanitized: true,
      });
      logger.debug(
        { chatId, sender, poolIndex: idx, chunkIndex: i },
        '[send] sendPoolMessage: chunk sent',
      );
    }
    logger.info(
      {
        chatId,
        sender,
        poolIndex: idx,
        length: text.length,
        chunks: chunks.length,
      },
      'Pool message sent',
    );
    return lastMsgId?.toString();
  } catch (err) {
    // Swallowed — caller won't know. Log at ERROR so at least the
    // operator sees it. The message MAY have reached Telegram before
    // the failure (e.g. fallback succeeded but Grammy threw post-send);
    // if no DB row lands, correlate this error with what appears in the
    // chat.
    logger.error(
      {
        chatId,
        sender,
        poolIndex: idx,
        err,
        preview: text.slice(0, 200),
      },
      '[send] Failed to send pool message — caller will still call storeMessage, but the send may have partially landed in Telegram',
    );
    return undefined;
  }
}

/**
 * Build sender display name with @username if available.
 * e.g. "JBáruch (@jbaruch)" or "Unknown" if no info.
 */
function buildSenderName(from?: {
  first_name?: string;
  username?: string;
  id?: number;
}): string {
  const displayName =
    from?.first_name || from?.username || from?.id?.toString() || 'Unknown';
  return from?.username ? `${displayName} (@${from.username})` : displayName;
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

    // Grammy API transformer — catches every outbound call on THIS Bot
    // instance regardless of which internal code path invoked it. Logs
    // method + payload preview + a stack trace of the caller. Existing
    // [send] tracepoints cover every path we currently know about
    // (sendTelegramMessage wrapper, sendFile, sendPoolMessage), but
    // issue #81's ghost messages keep showing up with no matching
    // trace — meaning some path we haven't discovered is invoking
    // `this.bot.api.*`. A transformer is the ONLY place that sees
    // every grammy-originated call without relying on callers to
    // opt-in to logging.
    //
    // Enabled only when LOG_LEVEL=debug. The custom logger in
    // `src/logger.ts` only defines debug/info/warn/error/fatal — unknown
    // levels fall back to info, so gating on "trace" would attach the
    // transformer and pay the stack/preview cost while logger.debug
    // output was suppressed. Keep the gate strictly to the level that
    // actually prints.
    const traceGrammy = process.env.LOG_LEVEL === 'debug';
    // Guard the grammy internal surface. `bot.api.config.use` exists on
    // real grammy Bot instances, but unit tests mock `this.bot` without
    // the `api.config` tree, and nothing in grammy's API stability
    // policy promises this hook. If it's missing, log and continue —
    // the diagnostic is a nice-to-have; crashing `connect()` because a
    // future grammy release renamed `config` would be much worse.
    const grammyConfig = this.bot.api?.config as
      | { use?: (transformer: Parameters<Api['config']['use']>[0]) => void }
      | undefined;
    if (traceGrammy && typeof grammyConfig?.use === 'function') {
      grammyConfig.use(async (prev, method, payload, signal) => {
        // Stack trace — Error().stack captures the synchronous call
        // chain up to this transformer. Slice the top frames so the
        // grammy internals don't drown out the interesting caller.
        const stack = new Error().stack?.split('\n').slice(2, 10).join('\n');
        // Payload preview — trim strings to avoid dumping 4KB of
        // message text into every log line. Only text / caption / chat
        // routing fields matter for forensics.
        const preview: Record<string, unknown> = {};
        if (payload && typeof payload === 'object') {
          const p = payload as Record<string, unknown>;
          if ('chat_id' in p) preview.chat_id = p.chat_id;
          if ('message_id' in p) preview.message_id = p.message_id;
          if ('text' in p && typeof p.text === 'string') {
            preview.textLen = p.text.length;
            preview.textPreview = p.text.slice(0, 120);
          }
          if ('caption' in p && typeof p.caption === 'string') {
            preview.captionLen = p.caption.length;
            preview.captionPreview = p.caption.slice(0, 120);
          }
          if ('parse_mode' in p) preview.parse_mode = p.parse_mode;
          // Log only the `message_id` from reply_parameters. The full
          // object can carry nested `quote` text / entities that would
          // defeat the "trimmed preview" goal and potentially echo user
          // content into debug logs.
          if ('reply_parameters' in p) {
            const rp = p.reply_parameters as { message_id?: unknown } | null;
            if (rp && typeof rp === 'object' && 'message_id' in rp) {
              preview.reply_to_message_id = rp.message_id;
            }
          }
        }
        logger.debug({ method, preview, stack }, '[grammy-api] outbound call');
        return prev(method, payload, signal);
      });
      logger.info(
        'Grammy API transformer attached — every bot.api.* call will be traced',
      );
    } else if (traceGrammy) {
      logger.warn(
        '[grammy-api] LOG_LEVEL=debug set but bot.api.config.use unavailable — transformer skipped (likely mocked Bot in tests, or a grammy API change)',
      );
    }

    // Command to get chat ID (useful for registration)
    this.bot.command('chatid', (ctx) => {
      const chatId = ctx.chat.id;
      const chatType = ctx.chat.type;
      const chatName =
        chatType === 'private'
          ? ctx.from?.first_name || 'Private'
          : (ctx.chat as any).title || 'Unknown';

      ctx.reply(
        `Chat ID: <code>tg:${chatId}</code>\nName: ${chatName}\nType: ${chatType}`,
        { parse_mode: 'HTML' },
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
      const senderName = buildSenderName(ctx.from);
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

      // No host-side reaction or observer seeding here — both moved
      // out in #289. 👀 fires from the agent-runner's react-first
      // hook (only when the container is genuinely alive); the
      // observer's progress emojis pin to `target_message_id` from
      // the agent-runner's Query input log line and gate themselves
      // on that line's `addressed=true|false` field.
    });

    // Handle non-text messages with placeholders so the agent knows something was sent
    const storeNonText = (ctx: any, placeholder: string) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName = buildSenderName(ctx.from);
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
      const senderName = buildSenderName(ctx.from);
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

      const isTrustedPhoto = group.isMain || !!group.containerConfig?.trusted;
      let placeholder: string;
      if (isTrustedPhoto) {
        const containerPath = await savePhoto(
          this.bot!,
          ctx.message.photo,
          group.folder,
        );
        placeholder = containerPath
          ? `[Image: ${containerPath}]`
          : '[Image - download failed]';
      } else {
        placeholder = '[Image]';
      }

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
        { chatJid, senderName, placeholder },
        'Telegram photo stored',
      );
    });

    this.bot.on('message:video', (ctx) => storeNonText(ctx, '[Video]'));

    this.bot.on('message:voice', async (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName = buildSenderName(ctx.from);
      const msgId = ctx.message.message_id.toString();
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
        id: msgId,
        chat_jid: chatJid,
        sender: ctx.from?.id?.toString() || '',
        sender_name: senderName,
        content,
        timestamp,
        is_from_me: false,
      });

      // No host-side reaction here either — see the matching block
      // in the message:text handler for the rationale.
    });
    this.bot.on('message:audio', (ctx) => storeNonText(ctx, '[Audio]'));
    this.bot.on('message:document', async (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName = buildSenderName(ctx.from);
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
      const isTrusted = group.isMain || !!group.containerConfig?.trusted;
      let content: string;

      if (fileId && isTrusted) {
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
      } else if (fileId && !isTrusted) {
        content = `[Document: ${fileName}]${caption}`;
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

    // Handle emoji reactions
    this.bot.on('message_reaction', (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const update = ctx.messageReaction;
      const reactorId = update.user?.id?.toString() || '';
      const reactorName =
        update.user?.first_name || update.user?.username || 'Unknown';
      const timestamp = new Date(update.date * 1000).toISOString();

      for (const reaction of update.new_reaction || []) {
        if (reaction.type === 'emoji') {
          storeReaction({
            message_id: update.message_id.toString(),
            message_chat_jid: chatJid,
            reactor_jid: `${reactorId}@telegram`,
            reactor_name: reactorName,
            emoji: reaction.emoji,
            timestamp,
          });
          logger.info(
            { chatJid, reactorName, emoji: reaction.emoji },
            'Telegram reaction stored',
          );
        }
      }
    });

    // Handle errors gracefully
    this.bot.catch((err) => {
      logger.error({ err: err.message }, 'Telegram bot error');
    });

    // Start polling with auto-restart on transient failures (e.g. 409 Conflict).
    // Grammy's polling loop dies silently on getUpdates errors — we catch that
    // and restart after a backoff so the bot doesn't go deaf.
    const MAX_POLLING_RETRIES = 5;
    let pollingRetries = 0;

    const startPolling = (): Promise<void> => {
      return new Promise<void>((resolve, reject) => {
        let resolved = false;
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
            pollingRetries = 0; // reset on successful start
            if (!resolved) {
              resolved = true;
              resolve();
            }
          },
        }).catch((err: Error) => {
          pollingRetries++;
          if (pollingRetries > MAX_POLLING_RETRIES) {
            logger.fatal(
              { err: err.message, retries: pollingRetries },
              'Telegram polling failed too many times, giving up',
            );
            if (!resolved) {
              resolved = true;
              reject(err);
            }
            return;
          }
          const backoffMs = Math.min(10_000 * pollingRetries, 60_000);
          logger.error(
            { err: err.message, retry: pollingRetries, backoffMs },
            'Telegram polling loop crashed, restarting',
          );
          setTimeout(() => {
            logger.info(
              { retry: pollingRetries },
              'Restarting Telegram polling loop',
            );
            startPolling().catch((retryErr: Error) => {
              logger.error(
                { err: retryErr.message },
                'Telegram polling restart failed',
              );
            });
          }, backoffMs);
          // Only reject if we haven't resolved the initial start yet
          if (!resolved) {
            resolved = true;
            reject(err);
          }
        });
      });
    };

    return startPolling();
  }

  async sendMessage(
    jid: string,
    text: string,
    replyToMessageId?: string,
  ): Promise<string | void> {
    logger.debug(
      {
        jid,
        textLen: text.length,
        preview: text.slice(0, 80),
        replyToMessageId,
      },
      '[send] TelegramChannel.sendMessage entered',
    );
    if (!this.bot) {
      logger.warn('Telegram bot not initialized');
      return;
    }

    try {
      const numericId = jid.replace(/^tg:/, '');
      const options: {
        message_thread_id?: number;
        reply_parameters?: { message_id: number };
      } = {};

      if (replyToMessageId && safeReplyToForChat(replyToMessageId, jid)) {
        options.reply_parameters = {
          message_id: parseInt(replyToMessageId, 10),
        };
      }

      // Sanitize ONCE on the whole message, then split (#282).
      // Pre-fix order was split-then-sanitize, which let splitMessage
      // cut mid-construct (e.g. between `]` and `(` of a markdown
      // link, mid-word inside `**bold**`, mid-fence inside ``` ```
      // ```) and handed each chunk to the sanitizer as half a
      // construct — neither half matched the sanitizer's regex, so
      // the user saw raw markdown markers across chunks. Sanitizing
      // first produces explicit `<a>...</a>`, `<b>...</b>`,
      // `<pre>...</pre>` tag boundaries that the HTML-aware
      // splitMessage refuses to cut inside (#286). Each chunk goes
      // to `sendTelegramMessage` with `preSanitized: true` so the
      // sanitizer doesn't run again on already-sanitized content.
      const sanitized = sanitizeTelegramHtml(text);
      const chunks = splitMessage(sanitized);
      logger.debug(
        { jid, chunkCount: chunks.length },
        '[send] TelegramChannel.sendMessage: sending chunks',
      );
      let lastMsgId: number | undefined;
      for (let i = 0; i < chunks.length; i++) {
        const chunkOptions =
          i === 0
            ? { ...options, preSanitized: true as const }
            : { preSanitized: true as const };
        lastMsgId = await sendTelegramMessage(
          this.bot.api,
          numericId,
          chunks[i],
          chunkOptions,
        );
      }
      logger.info(
        {
          jid,
          length: text.length,
          replyToMessageId,
          chunks: chunks.length,
          lastMsgId,
        },
        'Telegram message sent',
      );
      return lastMsgId?.toString();
    } catch (err) {
      // Err here only if sendTelegramMessage's fallback catch re-threw
      // (i.e. both HTML and plain-text sends failed). Return undefined
      // so the caller's `if (sentMsgId)` guards skip the post-send
      // work. The message did NOT reach Telegram in this path.
      logger.error(
        {
          jid,
          err,
          preview: text.slice(0, 200),
        },
        '[send] Failed to send Telegram message — returning undefined (message NOT delivered)',
      );
    }
  }

  async sendFile(
    jid: string,
    filePath: string,
    caption?: string,
    replyToMessageId?: string,
  ): Promise<void> {
    if (!this.bot) return;
    try {
      const numericId = jid.replace(/^tg:/, '');
      // Sanitize the caption same as `sendTelegramMessage` does for text:
      // Markdown → HTML, then parse_mode: 'HTML'. Without this, an agent
      // that invokes `mcp__nanoclaw__send_file` with a Markdown caption
      // gets the Markdown rendered literally on Telegram — and bypasses
      // our sanitizer entirely. Captions previously shipped as plain text
      // with no parse_mode, so `_heartbeat_` rendered as `_heartbeat_`.
      const sanitizedCaption = caption
        ? sanitizeTelegramHtml(caption)
        : undefined;
      const options: {
        caption?: string;
        parse_mode?: 'HTML';
        reply_parameters?: { message_id: number };
      } = {};
      if (sanitizedCaption) {
        options.caption = sanitizedCaption;
        options.parse_mode = 'HTML';
      }
      if (replyToMessageId && safeReplyToForChat(replyToMessageId, jid)) {
        options.reply_parameters = {
          message_id: parseInt(replyToMessageId, 10),
        };
      }
      try {
        await this.bot.api.sendDocument(
          numericId,
          new InputFile(filePath),
          options,
        );
      } catch (err) {
        // Fallback only makes sense when the first attempt used
        // `parse_mode: 'HTML'` — i.e. a caption was provided and
        // sanitized. Without that, the retry payload would be
        // identical to the first attempt, so retrying just doubles
        // API traffic on transient/network failures. Let the error
        // bubble to the outer catch/logger instead.
        if (options.parse_mode !== 'HTML') throw err;
        // Narrow the fallback to the specific Telegram-side HTML
        // parse rejection (#414) — same reasoning as
        // `sendTelegramMessage`'s catch above. `HttpError` /
        // 5xx / 429 / anything not a 400 "can't parse entities"
        // re-throws to the outer handler instead of producing a
        // duplicate caption send on transport-layer failures.
        if (
          !(
            err instanceof GrammyError &&
            err.error_code === 400 &&
            /can't parse entities/i.test(err.description)
          )
        ) {
          throw err;
        }
        // Mirror sendTelegramMessage's fallback hardening (#278).
        // Metadata only — no caption content reaches the log sink:
        // captions are user input, `jbaruch/coding-policy:
        // no-secrets` says "Never log secrets — not at any log
        // level" and requires sanitize-or-redact, and a preview
        // slice doesn't sanitize. Marked-degraded caption
        // (`⚠️ formatting failed; raw caption below…`) so the user
        // knows they're seeing a fallback rather than the agent's
        // intended formatting; `DEV_NO_HTML_FALLBACK=1` short-
        // circuits the fallback in dev/CI so the actual 400 surfaces
        // directly.
        logger.warn(
          {
            err,
            jid,
            filePath,
            rawCaptionLen: caption?.length,
            sanitizedCaptionLen: sanitizedCaption?.length,
          },
          'HTML caption parse failed, falling back to plain caption (user will see raw Markdown with warning prefix)',
        );
        if (process.env.DEV_NO_HTML_FALLBACK === '1') throw err;
        const plainOptions: {
          caption?: string;
          reply_parameters?: { message_id: number };
        } = {};
        if (caption) {
          // Truncate so the prefix never pushes the caption past
          // Telegram's 1024-char `sendDocument` caption limit. A
          // recoverable HTML-parse error becoming a "fallback also
          // failed" lost message would be strictly worse than the
          // original symptom.
          const captionPrefix = '⚠️ formatting failed; raw caption below\n\n';
          plainOptions.caption = `${captionPrefix}${caption.slice(0, MAX_CAPTION_LENGTH - captionPrefix.length)}`;
        }
        if (replyToMessageId && safeReplyToForChat(replyToMessageId, jid)) {
          plainOptions.reply_parameters = {
            message_id: parseInt(replyToMessageId, 10),
          };
        }
        await this.bot.api.sendDocument(
          numericId,
          new InputFile(filePath),
          plainOptions,
        );
      }
      logger.info({ jid, filePath, caption }, 'Telegram file sent');
    } catch (err) {
      logger.error({ jid, filePath, err }, 'Failed to send Telegram file');
    }
  }

  async pinMessage(jid: string, messageId: string): Promise<void> {
    if (!this.bot) return;
    try {
      const numericId = jid.replace(/^tg:/, '');
      await this.bot.api.pinChatMessage(numericId, parseInt(messageId, 10));
      logger.info({ jid, messageId }, 'Telegram message pinned');
    } catch (err) {
      logger.error({ jid, messageId, err }, 'Failed to pin Telegram message');
    }
  }

  isConnected(): boolean {
    return this.bot !== null;
  }

  ownsJid(jid: string): boolean {
    return jid.startsWith('tg:');
  }

  async isPrivateChat(jid: string): Promise<boolean> {
    if (!this.bot) return false;
    const numericId = jid.replace(/^tg:/, '');
    // Telegram getChat returns type ∈ {private, group, supergroup, channel}.
    // Only "private" (a 1:1 DM with the bot) is safe for the observer
    // chat — the other three types have multiple readers, including
    // potentially-untrusted external participants.
    const chat = await this.bot.api.getChat(numericId);
    return chat.type === 'private';
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

  async sendReaction(
    jid: string,
    messageId: string,
    emoji: string,
  ): Promise<void> {
    if (!this.bot) return;
    const numericId = jid.replace(/^tg:/, '');
    const msgId = parseInt(messageId, 10);
    // Telegram only allows specific emoji as reactions. Normalize
    // shortcodes (`thumbs_up`, `:thumbs_up:`) to Unicode first so
    // agents that emit Slack-style names don't silently fall back
    // to 👍 — see #161.
    const normalized = normalizeReactionEmoji(emoji);
    const reactionAllowed = TELEGRAM_ALLOWED_REACTIONS.has(normalized);
    const validEmoji = reactionAllowed ? normalized : '👍';
    if (!reactionAllowed) {
      // Real recoverable issue — caller asked for an emoji Telegram
      // doesn't support, we're substituting 👍 silently from the
      // user's perspective. Operators want to see this.
      logger.warn(
        {
          jid,
          messageId,
          requested: emoji,
          normalized,
          using: validEmoji,
        },
        'Invalid Telegram reaction emoji, falling back to 👍',
      );
    } else if (normalized !== emoji) {
      // Successful normalization (shortcode → Unicode, or VS16
      // strip). Not a problem — log at debug so production logs
      // aren't flooded when agents commonly emit shortcodes.
      logger.debug(
        { jid, messageId, requested: emoji, normalized },
        'Telegram reaction emoji normalized to Unicode',
      );
    }
    try {
      await this.bot.api.raw.setMessageReaction({
        chat_id: numericId,
        message_id: msgId,
        reaction: [{ type: 'emoji', emoji: validEmoji as any }],
      });
      // Store outbound reaction so unanswered-message checks see it
      storeReaction({
        message_id: messageId,
        message_chat_jid: jid,
        reactor_jid: 'bot@telegram',
        reactor_name: ASSISTANT_NAME,
        emoji: validEmoji,
        timestamp: new Date().toISOString(),
      });
      logger.info(
        { jid, messageId, emoji: validEmoji },
        'Telegram reaction sent',
      );
    } catch (err) {
      logger.error(
        { jid, messageId, emoji: validEmoji, err },
        'Failed to send Telegram reaction',
      );
    }
  }

  async reactToLatestMessage(jid: string, emoji: string): Promise<void> {
    const latest = getLatestMessage(jid);
    if (!latest) {
      logger.warn({ jid }, 'No messages found to react to');
      return;
    }
    await this.sendReaction(jid, latest.id, emoji);
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
