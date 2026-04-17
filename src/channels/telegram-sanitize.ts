/**
 * Markdown → Telegram HTML sanitizer, ported from
 * tessl-workspace/.tessl/tiles/jbaruch/nanoclaw-admin/skills/heartbeat/scripts/sanitize-html.py.
 *
 * Run at Telegram send time so agent discipline stops mattering: if a skill
 * (or a subagent, or a forgetful prompt) produces Markdown, we convert it to
 * Telegram-flavored HTML here. Idempotent — already-valid HTML passes through.
 *
 * Protected regions (never rewritten):
 *   - Existing HTML tags
 *   - http/https/ftp URLs
 *   - Email addresses
 *
 * Converted patterns:
 *   [text](url)      → <a href="url">text</a>
 *   `code`           → <code>code</code>
 *   **bold** / __b__ → <b>bold</b>
 *   *italic* / _i_   → <i>italic</i>  (only when delimiters look like formatting)
 *   # heading        → <b>heading</b> (line-start, 1-6 hashes)
 *   - item / * item  → • item (line-start bullet)
 */

const PH_PREFIX = '\u0000PH';
const PH_SUFFIX = '\u0000';

export function sanitizeTelegramHtml(text: string): string {
  if (!text) return text;

  const placeholders: string[] = [];
  const protect = (match: string): string => {
    const idx = placeholders.length;
    placeholders.push(match);
    return `${PH_PREFIX}${idx}${PH_SUFFIX}`;
  };

  // Phase 1: protect regions that must not be touched.
  // Order matters — HTML tags first so protected-URL regex doesn't eat href values.
  let out = text.replace(
    /<\/?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^>]*)?\s*\/?>/g,
    protect,
  );
  out = out.replace(/https?:\/\/[^\s<>")\]]+/g, protect);
  out = out.replace(/ftp:\/\/[^\s<>")\]]+/g, protect);
  out = out.replace(/[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+/g, protect);

  // Phase 2: Markdown → HTML.
  // 2a. Markdown links [text](url) — url may be a placeholder from Phase 1.
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

  // 2b. Inline code — before bold/italic so backticked content isn't mangled.
  out = out.replace(/`([^`\n]+)`/g, '<code>$1</code>');

  // 2c. Bold: **x** or __x__
  out = out.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  out = out.replace(/__(.+?)__/g, '<b>$1</b>');

  // 2d. Italic: *x* or _x_ — must look like formatting, not identifier parts.
  //     Non-word char (or string edge) required around delimiters; no
  //     whitespace directly inside.
  out = out.replace(/(^|[^\w])\*(\S(?:.*?\S)?)\*(?!\w)/g, '$1<i>$2</i>');
  out = out.replace(/(^|[^\w])_(\S(?:.*?\S)?)_(?!\w)/g, '$1<i>$2</i>');

  // 2e. Headings: # to ###### at line start → <b>...</b>
  out = out.replace(/^#{1,6}\s+(.+)$/gm, '<b>$1</b>');

  // 2f. Bullets: - item / * item at line start → • item
  out = out.replace(/^[-*]\s+/gm, '\u2022 ');

  // Phase 3: restore placeholders.
  out = out.replace(
    new RegExp(`${PH_PREFIX}(\\d+)${PH_SUFFIX}`, 'g'),
    (_, idx: string) => placeholders[Number(idx)],
  );

  return out;
}
