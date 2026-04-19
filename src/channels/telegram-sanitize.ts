/**
 * Markdown → Telegram HTML sanitizer, ported from and hardened beyond
 * tessl-workspace/.tessl/tiles/jbaruch/nanoclaw-admin/skills/heartbeat/scripts/sanitize-html.py.
 *
 * Run at Telegram send time so agent discipline stops mattering: if a skill
 * (or a subagent, or a forgetful prompt) produces Markdown, we convert it to
 * Telegram-flavored HTML here. Idempotent — already-valid HTML passes through.
 *
 * Protected regions (never rewritten):
 *   - Fenced code blocks (```...```) — preserved as <pre> with contents escaped.
 *   - Whole inline HTML element spans (`<code>…</code>`, `<pre>…</pre>`,
 *     `<b>…</b>`, `<i>…</i>`, `<u>…</u>`, `<s>…</s>`, `<a>…</a>`,
 *     `<blockquote>…</blockquote>`, `<tg-spoiler>…</tg-spoiler>`) — the
 *     element AND its contents are protected, so Markdown markers inside
 *     pre-formatted HTML (e.g. `<code>*literal*</code>`) aren't rewritten.
 *   - Stray HTML tag tokens (self-closing, mismatched).
 *   - http / https / ftp URLs, email addresses.
 *
 * Converted patterns (captured text is HTML-escaped before insertion so
 * characters like `&`, `<`, `>`, `"` in content don't produce invalid entities):
 *   [text](url)      → <a href="url">text</a>
 *   `code`           → <code>code</code>
 *   **bold** / __b__ → <b>bold</b>
 *   *italic* / _i_   → <i>italic</i>  (only when delimiters look like formatting)
 *   # heading        → <b>heading</b> (line-start, 1-6 hashes)
 *   - item / * item  → • item (line-start bullet)
 */

const PH_PREFIX = '\u0000PH';
const PH_SUFFIX = '\u0000';

/** Escape `&`, `<`, `>`, `"` so captured text is safe inside HTML content / attributes. */
function htmlEscape(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Inline HTML tags whose full span (opening + contents + closing) must be
 * treated as opaque. Order matters only inside this list for readability.
 */
const PROTECTED_SPAN_TAGS = [
  'pre',
  'code',
  'blockquote',
  'a',
  'b',
  'i',
  'u',
  's',
  'tg-spoiler',
];

export function sanitizeTelegramHtml(text: string): string {
  if (!text) return text;

  const placeholders: string[] = [];
  const protect = (match: string): string => {
    const idx = placeholders.length;
    placeholders.push(match);
    return `${PH_PREFIX}${idx}${PH_SUFFIX}`;
  };

  // Regex for the NUL-delimited placeholder tokens we inject during
  // Phases 0/1a/1b/1c. Reused in `escapeCaptured` below AND in Phase 3.
  const PH_RE = new RegExp(`${PH_PREFIX}(\\d+)${PH_SUFFIX}`, 'g');

  // Phase 2 conversions (`[…](…)`, `` `…` ``, `**…**`, etc.) capture
  // content and wrap it in Telegram HTML tags. That captured content may
  // contain placeholders injected earlier (Phase 1b protected stray tag
  // tokens like `<N>`). A plain `htmlEscape` on the captured text sees
  // only the opaque `\u0000PH<idx>\u0000` markers — no `<`, `>`, `&`, or
  // `"` to escape — so it's a no-op, and Phase 3 restores the raw `<N>`
  // INSIDE our freshly-created `<code>`/`<b>`/etc. tag. Telegram then
  // chokes on the unknown `<N>` tag, rejects the whole message with 400,
  // and `sendTelegramMessage`'s catch fires the plain-text fallback —
  // which sends the original Markdown literally and is exactly the
  // "`**foo**` leaked through the preprocessor" symptom.
  //
  // Fix: inside Phase 2 captures, resolve the placeholder to its
  // underlying text BEFORE HTML-escaping, so the stray tag becomes
  // `&lt;N&gt;` inside the final `<code>…</code>` span. Placeholders
  // outside any Phase 2 capture (stray tags in plain prose) still
  // survive through Phase 3 unchanged — that's pre-existing behavior
  // and out of scope for this fix.
  const escapeCaptured = (s: string): string =>
    htmlEscape(
      s.replace(PH_RE, (_m, idx: string) => placeholders[Number(idx)]),
    );

  let out = text;

  // Phase 0: fenced code blocks — stash entire ```…``` regions, rewritten
  // into <pre> with contents HTML-escaped so code samples containing `**`
  // or `_` or `<` aren't mangled downstream.
  out = out.replace(
    /```(?:[\w-]+)?\r?\n([\s\S]*?)\r?\n```/g,
    (_m, code: string) => protect(`<pre>${htmlEscape(code)}</pre>`),
  );
  // Single-line / unterminated fenced blocks (defensive — less common).
  out = out.replace(/```([\s\S]*?)```/g, (_m, code: string) =>
    protect(`<pre>${htmlEscape(code)}</pre>`),
  );

  // Phase 1a: protect full HTML element spans (tag + contents + closing tag)
  // so Markdown markers inside already-formatted HTML remain literal.
  // Non-greedy; does not attempt to handle nesting of the same tag.
  for (const tag of PROTECTED_SPAN_TAGS) {
    const re = new RegExp(`<${tag}(?:\\s[^>]*)?>[\\s\\S]*?<\\/${tag}>`, 'gi');
    out = out.replace(re, protect);
  }

  // Phase 1b: protect stray tag tokens (self-closing, mismatched, or tags
  // we don't recognise as span-ful). This keeps any remaining raw HTML from
  // being touched, even if we can't pair it.
  out = out.replace(/<\/?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^>]*)?\s*\/?>/g, protect);

  // Phase 1c: protect URLs and email addresses so their underscores/dots
  // don't get mistaken for Markdown formatting.
  out = out.replace(/https?:\/\/[^\s<>")\]]+/g, protect);
  out = out.replace(/ftp:\/\/[^\s<>")\]]+/g, protect);
  out = out.replace(/[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+/g, protect);

  // Phase 2: Markdown → HTML. Captured groups run through `escapeCaptured`
  // (resolve placeholders, then HTML-escape) so `**a & b**` → `<b>a &amp;
  // b</b>` and `` `<N>` `` → `<code>&lt;N&gt;</code>` — no raw `&` or
  // stray tag tokens end up inside our freshly-created Telegram tags.
  //
  // 2a. Links — url may be a placeholder from Phase 1c (protected URL),
  // or a mix of protected URL prefix + stray-tag suffix when the href
  // contains `<`/`>` that broke Phase 1c's URL regex.
  out = out.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    (_m, txt: string, url: string) =>
      `<a href="${escapeCaptured(url)}">${escapeCaptured(txt)}</a>`,
  );

  // 2b. Inline code — before bold/italic so backticked content isn't mangled.
  out = out.replace(
    /`([^`\n]+)`/g,
    (_m, code: string) => `<code>${escapeCaptured(code)}</code>`,
  );

  // 2c. Bold: **x** or __x__
  out = out.replace(
    /\*\*(.+?)\*\*/g,
    (_m, t: string) => `<b>${escapeCaptured(t)}</b>`,
  );
  out = out.replace(
    /__(.+?)__/g,
    (_m, t: string) => `<b>${escapeCaptured(t)}</b>`,
  );

  // 2d. Italic: *x* or _x_ — must look like formatting, not identifier parts.
  out = out.replace(
    /(^|[^\w])\*(\S(?:.*?\S)?)\*(?!\w)/g,
    (_m, pre: string, t: string) => `${pre}<i>${escapeCaptured(t)}</i>`,
  );
  out = out.replace(
    /(^|[^\w])_(\S(?:.*?\S)?)_(?!\w)/g,
    (_m, pre: string, t: string) => `${pre}<i>${escapeCaptured(t)}</i>`,
  );

  // 2e. Headings: # to ###### at line start → <b>…</b>
  out = out.replace(
    /^#{1,6}\s+(.+)$/gm,
    (_m, t: string) => `<b>${escapeCaptured(t)}</b>`,
  );

  // 2f. Bullets: - item / * item at line start → • item
  out = out.replace(/^[-*]\s+/gm, '\u2022 ');

  // Phase 3: restore any placeholders that were NOT inside a Phase 2
  // capture (stray tags in plain prose, full HTML spans from Phase 1a,
  // fenced blocks from Phase 0, raw URLs/emails from Phase 1c).
  out = out.replace(PH_RE, (_m, idx: string) => placeholders[Number(idx)]);

  return out;
}
