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
    `# source: ${payload.final_url || fallbackUrl}\n` +
    `# chars: ${chars}${qualityLine}\n\n`
  );
}
