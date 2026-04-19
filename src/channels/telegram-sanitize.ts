/**
 * Markdown → Telegram HTML sanitizer, ported from and hardened beyond
 * tessl-workspace/.tessl/tiles/jbaruch/nanoclaw-admin/skills/heartbeat/scripts/sanitize-html.py.
 *
 * Run at Telegram send time so agent discipline stops mattering: if a skill
 * (or a subagent, or a forgetful prompt) produces Markdown, we convert it to
 * Telegram-flavored HTML here. Idempotent — already-valid HTML passes through.
 *
 * Always-protected regions (never rewritten, whether inside a Markdown
 * capture or not):
 *   - Fenced code blocks (```...```) — preserved as <pre> with contents escaped.
 *   - Whole inline HTML element spans (`<code>…</code>`, `<pre>…</pre>`,
 *     `<b>…</b>`, `<i>…</i>`, `<u>…</u>`, `<s>…</s>`, `<a>…</a>`,
 *     `<blockquote>…</blockquote>`, `<tg-spoiler>…</tg-spoiler>`) — the
 *     element AND its contents are protected, so Markdown markers inside
 *     pre-formatted HTML (e.g. `<code>*literal*</code>`) aren't rewritten.
 *   - http / https / ftp URLs, email addresses.
 *
 * Conditionally-protected regions (rewritten ONLY inside a Markdown
 * capture, passed through unchanged in plain prose):
 *   - Stray tag tokens (self-closing, mismatched, or tags we can't
 *     pair, e.g. `<N>` or `<bar>`). In plain prose they pass through
 *     so the agent can emit literal HTML if it needs to. Inside a
 *     Phase 2 capture (`` `<N>` `` or `**<foo>**`) the token gets
 *     HTML-escaped so the output is `<code>&lt;N&gt;</code>` rather
 *     than `<code><N></code>`. Telegram's HTML parser only accepts a
 *     fixed tag whitelist; leaving the raw `<N>` inside our freshly-
 *     created `<code>`/`<b>`/etc. span causes a 400 rejection and
 *     dumps the whole message into the plain-text fallback in
 *     src/channels/telegram.ts, shipping literal Markdown to the user.
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

// Two separate placeholder namespaces.
//
// `PH_PREFIX` ("protect and preserve") is used for regions whose content
// must be passed to Telegram verbatim inside the final output — fenced
// code blocks (Phase 0), already-valid HTML element spans (Phase 1a),
// and URLs / email addresses (Phase 1c). These are RESTORED by Phase 3
// and MUST survive through Phase 2 Markdown captures intact, so a
// `**<code>x</code>**` input still ships `<b><code>x</code></b>` with
// the inner span untouched.
//
// `PH_STRAY_PREFIX` ("protect until Phase 2 decides") is used only for
// stray tag tokens (Phase 1b) like `<N>` or `<bar>`. Inside a Phase 2
// capture (`` ` … ` ``, `**…**`, etc.) the stray token must be
// RESOLVED and HTML-escaped so the captured content ends up like
// `<code>&lt;N&gt;</code>` — otherwise Telegram sees a bare `<N>`,
// rejects the message, and the send falls back to raw Markdown.
// Outside of a capture, stray placeholders are restored in Phase 3
// just like protect placeholders (pre-existing behavior).
const PH_PREFIX = '\u0000PH';
const PH_STRAY_PREFIX = '\u0000ST';
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
  const strayPlaceholders: string[] = [];

  const protect = (match: string): string => {
    const idx = placeholders.length;
    placeholders.push(match);
    return `${PH_PREFIX}${idx}${PH_SUFFIX}`;
  };
  const protectStray = (match: string): string => {
    const idx = strayPlaceholders.length;
    strayPlaceholders.push(match);
    return `${PH_STRAY_PREFIX}${idx}${PH_SUFFIX}`;
  };

  const PH_RE = new RegExp(`${PH_PREFIX}(\\d+)${PH_SUFFIX}`, 'g');
  const PH_STRAY_RE = new RegExp(`${PH_STRAY_PREFIX}(\\d+)${PH_SUFFIX}`, 'g');

  // Phase 2 conversions (`[…](…)`, `` `…` ``, `**…**`, etc.) capture
  // content and wrap it in Telegram HTML tags. That captured content may
  // contain placeholders injected earlier. A plain `htmlEscape` on the
  // captured text sees only the opaque `\u0000…\u0000` markers — no `<`,
  // `>`, `&`, or `"` to escape — so it's a no-op, and Phase 3 restores
  // the raw text INSIDE our freshly-created `<code>`/`<b>`/etc. tag.
  //
  // For stray-tag placeholders this is a bug: Telegram would see a raw
  // `<N>` inside `<code>`, reject the whole message with 400, and
  // `sendTelegramMessage`'s catch would fire the plain-text fallback —
  // which ships the original Markdown verbatim. That's the exact
  // "`**foo**` leaked through the preprocessor" symptom the fix
  // addresses: resolve stray-tag placeholders inside the capture and
  // HTML-escape them, so the output is `<code>&lt;N&gt;</code>`.
  //
  // For protect placeholders (Phase 0/1a/1c — fenced code, already-
  // valid HTML spans, URLs, emails) the old no-op was INTENTIONAL: the
  // "Protected regions (never rewritten)" contract says inputs like
  // `**<code>x</code>**` must preserve the inner `<code>x</code>` span
  // verbatim inside `<b>…</b>`. So only stray placeholders get
  // resolved here; protect ones keep surviving through to Phase 3.
  // Bounds-checked array lookup. When we emit placeholders ourselves
  // via `protect`/`protectStray`, the index is always `arr.length` at
  // emit time, so the round-trip is safe by construction. The range
  // check is defense against INPUTS that happen to contain a NUL-
  // delimited PH/ST-shaped sequence we didn't produce — e.g.
  // adversarial text, or a previous sanitizer output round-tripped
  // through this one. In that case the extracted index can point
  // well past `arr.length` (or be a multi-digit number we never
  // pushed). Returning `match` preserves the literal bytes rather
  // than emitting "undefined".
  const resolveFrom =
    (arr: string[]) =>
    (match: string, idx: string): string => {
      const n = Number(idx);
      return n >= 0 && n < arr.length ? arr[n] : match;
    };

  const escapeCaptured = (s: string): string =>
    htmlEscape(s.replace(PH_STRAY_RE, resolveFrom(strayPlaceholders)));

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
  // we don't recognise as span-ful). Uses `protectStray` so a token that
  // ends up inside a Phase 2 capture gets resolved+escaped there
  // (`escapeCaptured`) instead of restored raw in Phase 3 — otherwise
  // Telegram rejects the unknown tag and the send falls back to plain
  // text.
  out = out.replace(
    /<\/?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^>]*)?\s*\/?>/g,
    protectStray,
  );

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

  // Phase 3: restore any placeholders still in the text — these are
  // the ones that were NOT inside a Phase 2 capture. Bounds-checked so
  // a crafted `\u0000PH<big>\u0000` token in the input doesn't emit
  // "undefined".
  out = out.replace(PH_RE, resolveFrom(placeholders));
  out = out.replace(PH_STRAY_RE, resolveFrom(strayPlaceholders));

  return out;
}
