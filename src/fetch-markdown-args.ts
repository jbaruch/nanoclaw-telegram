/**
 * Pure helpers for the `fetch_markdown` IPC handler — extracted from
 * `ipc.ts` so URL validation, flag construction, and the JSON-payload
 * header builder can be unit-tested without staging the full IPC
 * watcher (file events, deps mock, sibling-docker spawn, etc.).
 *
 * The handler in `ipc.ts` `case 'fetch_markdown'` calls into these
 * helpers; the runtime contract is the same — pre-extraction the
 * shapes were inline in the switch case.
 */

export type ParseUrlResult =
  | { ok: true; url: URL }
  | { ok: false; error: string };

/**
 * Validate the raw IPC payload's `url` field. Accepts only absolute
 * http(s) URLs; rejects unparseable strings and non-http(s) protocols
 * with an actionable error message the agent can act on.
 *
 * Returning a discriminated-union rather than throwing keeps the
 * handler's error-path simple (write the error file and break) and
 * matches the rest of the file's flat-control-flow style.
 */
export function parseFetchMarkdownUrl(raw: unknown): ParseUrlResult {
  if (typeof raw !== 'string' || raw === '') {
    return {
      ok: false,
      error:
        'fetch_markdown: invalid URL. Pass an absolute http(s) URL, e.g. https://example.com/path.',
    };
  }
  let url: URL;
  try {
    url = new URL(raw);
  } catch (parseErr) {
    // `new URL(input)` throws `TypeError` for any unparseable string —
    // see WHATWG URL spec. Catch that specifically per
    // `jbaruch/coding-policy: error-handling`; let any other error
    // propagate so a programming bug here (e.g. unexpected throw from
    // a Node platform change) doesn't get masked as "invalid URL".
    if (!(parseErr instanceof TypeError)) throw parseErr;
    return {
      ok: false,
      error:
        'fetch_markdown: invalid URL. Pass an absolute http(s) URL, e.g. https://example.com/path.',
    };
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return {
      ok: false,
      error: `fetch_markdown: only http(s) URLs are supported (got ${url.protocol}).`,
    };
  }
  return { ok: true, url };
}

export interface FetchMarkdownFlagInput {
  wait?: unknown;
  waitUntil?: unknown;
  waitForSelector?: unknown;
  favorPrecision?: unknown;
  favorRecall?: unknown;
  includeLinks?: unknown;
  includeImages?: unknown;
  maxChars?: unknown;
  noCache?: unknown;
  timeout?: unknown;
}

const VALID_WAIT_UNTIL = new Set([
  'commit',
  'domcontentloaded',
  'load',
  'networkidle',
]);

/**
 * Build the snitchmd CLI flag array from the IPC payload fields.
 * Always emits `--json` so the handler can parse the structured
 * output. Malformed / wrong-typed values are silently dropped — the
 * MCP tool's zod schema enforces shape at the agent boundary, but
 * the host handler treats IPC input as untrusted and ignores anything
 * that doesn't pass a positive-integer / known-enum / boolean check
 * rather than passing it through to snitchmd where it would surface
 * as a `bail()` exit-code-2.
 */
export function buildSnitchmdFlags(input: FetchMarkdownFlagInput): string[] {
  const flags: string[] = ['--json'];
  if (
    typeof input.wait === 'number' &&
    Number.isInteger(input.wait) &&
    input.wait > 0
  ) {
    flags.push('--wait', String(input.wait));
  }
  if (
    typeof input.waitUntil === 'string' &&
    VALID_WAIT_UNTIL.has(input.waitUntil)
  ) {
    flags.push('--wait-until', input.waitUntil);
  }
  if (
    typeof input.waitForSelector === 'string' &&
    input.waitForSelector !== ''
  ) {
    flags.push('--wait-for-selector', input.waitForSelector);
  }
  if (input.favorPrecision === true) flags.push('--favor-precision');
  if (input.favorRecall === true) flags.push('--favor-recall');
  if (input.includeLinks === true) flags.push('--include-links');
  if (input.includeImages === true) flags.push('--include-images');
  if (
    typeof input.maxChars === 'number' &&
    Number.isInteger(input.maxChars) &&
    input.maxChars > 0
  ) {
    flags.push('--max-chars', String(input.maxChars));
  }
  if (input.noCache === true) flags.push('--no-cache');
  if (
    typeof input.timeout === 'number' &&
    Number.isInteger(input.timeout) &&
    input.timeout > 0
  ) {
    flags.push('--timeout', String(input.timeout));
  }
  return flags;
}

export interface SnitchmdPayload {
  markdown?: string;
  title?: string;
  final_url?: string;
  quality?: number | null;
  chars?: number;
}

export type ParsePayloadResult =
  | { ok: true; payload: SnitchmdPayload }
  | { ok: false };

/**
 * Parse snitchmd's `--json` stdout, tolerating pre-JSON noise.
 *
 * Production smoke-test on 2026-05-21 caught CloakBrowser's internal
 * `[cloakbrowser] Newer Chromium available: ... Downloading in
 * background... Download progress: 9% ...` chatter leaking onto stdout
 * BEFORE the JSON payload on the cold-pull path. The strict
 * `JSON.parse(stdout)` choked even though the actual fetch succeeded
 * (the JSON payload was present, just preceded by garbage). Cached
 * calls don't have the chatter, so the cached path was clean.
 *
 * Strategy: try the strict parse first (fast path, always succeeds on
 * cached calls). If that fails, slice from the FIRST `{` to the LAST
 * `}` and retry. The payload uses double-quoted JSON strings, so a `{`
 * inside an extracted page title can't confuse the slice — JSON strings
 * never carry a bare `}` followed by EOF and the LAST `}` always pairs
 * with snitchmd's outermost object. Return `{ ok: false }` if both
 * attempts fail so the caller can route to the "non-JSON output"
 * diagnostic branch.
 */
export function parseSnitchmdStdout(stdout: string): ParsePayloadResult {
  try {
    return { ok: true, payload: JSON.parse(stdout) as SnitchmdPayload };
  } catch (parseErr) {
    if (!(parseErr instanceof SyntaxError)) throw parseErr;
  }
  const firstBrace = stdout.indexOf('{');
  const lastBrace = stdout.lastIndexOf('}');
  if (firstBrace < 0 || lastBrace <= firstBrace) {
    return { ok: false };
  }
  const sliced = stdout.slice(firstBrace, lastBrace + 1);
  try {
    return { ok: true, payload: JSON.parse(sliced) as SnitchmdPayload };
  } catch (parseErr) {
    if (!(parseErr instanceof SyntaxError)) throw parseErr;
    return { ok: false };
  }
}

/**
 * Replace every occurrence of the fetched URL in `text` with `<URL>`.
 *
 * snitchmd's stderr is normally a benign `snitchmd: title=... chars=...`
 * line, but a Playwright/Chromium fault can echo the URL it was handed —
 * and a fetch whose auth rides in the query string (a signed URL, a
 * session token) would then persist that secret into the result envelope
 * the agent reads, or into the host log (`coding-policy: no-secrets`).
 *
 * Every path that surfaces snitchmd output — failure, parse-failure, AND
 * success — routes through this. The success path is the one that matters
 * most in practice: it is the case nobody thinks to inspect.
 *
 * Plain `split`/`join` rather than a regex: the URL is arbitrary
 * user-supplied text and would need escaping to be a safe pattern.
 */
export function scrubFetchedUrl(text: string, url: string): string {
  if (!url) return text;
  return text.split(url).join('<URL>');
}

/**
 * Strip the secret-bearing parts of a URL before it is persisted into the
 * `fetch_markdown` result header (`coding-policy: no-secrets`).
 *
 * The header's `# source:` line is useful attribution — it names what was
 * actually fetched, which matters when a redirect moved the target. But it
 * ends up in the envelope the agent reads and in the conversation, and a
 * fetch can legitimately authenticate through the URL itself: a signed S3
 * link, a session token in the query, `user:pass@host` userinfo. `final_url`
 * makes this worse than the requested URL — a redirect chain can ADD a token
 * the caller never saw.
 *
 * So: keep scheme, host and path; drop userinfo and fragment outright; and
 * replace a non-empty query with a fixed `?<redacted>` marker so the reader
 * can still tell parameters were involved without learning them.
 *
 * A value that doesn't parse as a URL is returned as the fixed string
 * `<unparseable-url>` rather than passed through — an unparseable value is
 * exactly the case where we cannot reason about what it contains.
 */
export function redactUrlForHeader(raw: string): string {
  let u: URL;
  try {
    u = new URL(raw);
  } catch (err) {
    // `new URL()` throws TypeError on an unparseable value. Anything else
    // is a programming bug and propagates rather than being masked as a
    // bad URL (`coding-policy: error-handling`).
    if (!(err instanceof TypeError)) throw err;
    return '<unparseable-url>';
  }
  u.username = '';
  u.password = '';
  u.hash = '';
  const hadQuery = u.search.length > 0;
  u.search = '';
  return hadQuery ? `${u.toString()}?<redacted>` : u.toString();
}

/**
 * Redact every URL-looking token in free text through
 * `redactUrlForHeader`.
 *
 * Used where the text's STRUCTURE is unknown — the parse-failure
 * diagnostic, where snitchmd's stdout could not be parsed as JSON, so
 * `final_url` cannot be extracted and scrubbed by value. A redirect can
 * add a credential the caller never sent, so scrubbing only the URL we
 * passed in leaves that case exposed (`coding-policy: no-secrets`).
 *
 * Deliberately greedy about what counts as a URL: over-redacting a
 * diagnostic costs readability, under-redacting costs a leaked token.
 */
export function redactUrlsInText(text: string): string {
  return text.replace(/https?:\/\/[^\s"'`<>)\]}]+/g, (m) =>
    redactUrlForHeader(m),
  );
}

/**
 * Compose the 3-line header the handler prepends to the markdown body
 * before writing the IPC result file. Pulled out so the format is
 * pinned in tests against accidental whitespace / ordering drift —
 * downstream skills (wiki, check-cfps) consume the header for
 * provenance, so a silent reformat would break their parsing.
 */
export function formatSnitchmdHeader(
  payload: SnitchmdPayload,
  fallbackUrl: string,
): string {
  const markdown = payload.markdown ?? '';
  const chars = payload.chars ?? markdown.length;
  const qualityLine =
    payload.quality !== null && payload.quality !== undefined
      ? ` quality: ${payload.quality}`
      : '';
  return (
    `# ${payload.title || '(untitled)'}\n` +
    `# source: ${redactUrlForHeader(payload.final_url || fallbackUrl)}\n` +
    `# chars: ${chars}${qualityLine}\n\n`
  );
}
