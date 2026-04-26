/**
 * Poison defense — pure helpers used by hooks in `index.ts`.
 *
 * Two independent concerns share this file because both halves of #101
 * (split into #116 and #117) wire into the same `query()` hooks block:
 *
 *  - `shouldDenyTaskOutputBlock`  — gate for the `TaskOutput` PreToolUse
 *    hook. Blocking polls (`block !== false`) leak a raw chunk of the
 *    sub-agent's JSONL transcript on timeout, which is how the
 *    2026-04-24 maintenance session was poisoned with invisible-Unicode
 *    padding from a Composio Gmail body.
 *
 *  - `sanitizeToolResponse`       — sanitizer for the MCP `PostToolUse`
 *    hook. Strips Cf-category (invisible) characters and caps each
 *    text block at a configurable byte budget, so a noisy upstream
 *    can't smuggle padding into the JSONL or push it past
 *    AUP-classifier thresholds.
 *
 * Kept SDK-free so the root vitest can exercise them without spinning
 * up `@anthropic-ai/claude-agent-sdk`.
 */

/**
 * Default cap for any single tool_result text block, in bytes.
 * Override at runtime via `TOOL_RESULT_MAX_BYTES`.
 *
 * 64 KiB sits well above legitimate tool output (file reads, search
 * results, API JSON) but well below the JSONL bloat profile that
 * triggered the original incident (single blocks > 200 KiB of padded
 * Gmail body).
 */
export const DEFAULT_TOOL_RESULT_MAX_BYTES = 65536;

const TRUNCATION_MARKER =
  '\n\n[truncated by tool_result sanitizer — original exceeded byte cap]';

/**
 * Strips Unicode "Cf" (Format) category — zero-width joiners, BOM,
 * directional marks, language tags, etc. These render as nothing but
 * tokenize, so they're the standard padding/smuggling vector.
 *
 * The Cf class is the superset the issue calls out as the target
 * (U+200B–U+200D, U+FEFF, etc.). Using `\p{Cf}` directly is more
 * future-proof than enumerating ranges by hand.
 */
const INVISIBLE_UNICODE_RE = /\p{Cf}/gu;

export interface SanitizeStats {
  strippedBytes: number;
  truncatedBytes: number;
}

export interface SanitizeResult {
  sanitized: unknown;
  stats: SanitizeStats;
}

function utf8ByteLength(s: string): number {
  return Buffer.byteLength(s, 'utf-8');
}

/**
 * Strip + cap a single text block. Pure.
 */
export function sanitizeText(
  text: string,
  byteCap: number,
): { out: string; stats: SanitizeStats } {
  const beforeBytes = utf8ByteLength(text);
  const stripped = text.replace(INVISIBLE_UNICODE_RE, '');
  const strippedBytes = beforeBytes - utf8ByteLength(stripped);

  if (utf8ByteLength(stripped) <= byteCap) {
    return { out: stripped, stats: { strippedBytes, truncatedBytes: 0 } };
  }

  // Truncate to byteCap. Buffer slicing can split a multi-byte codepoint;
  // decoding with `toString('utf-8')` substitutes U+FFFD for the partial
  // tail, which is acceptable noise for a truncation marker.
  const buf = Buffer.from(stripped, 'utf-8');
  const head = buf.subarray(0, byteCap).toString('utf-8');
  const truncatedBytes = utf8ByteLength(stripped) - utf8ByteLength(head);
  return {
    out: head + TRUNCATION_MARKER,
    stats: { strippedBytes, truncatedBytes },
  };
}

/**
 * Walks an MCP tool response of the standard
 * `{ content: [{ type: 'text', text: string }, ...] }` shape and
 * sanitizes each `text` field in place.
 *
 * Non-text blocks (image, resource, etc.) and unrecognized shapes pass
 * through untouched — sanitizing image bytes would break legitimate
 * payloads, and the invisible-Unicode vector is text-only.
 */
export function sanitizeToolResponse(
  response: unknown,
  byteCap: number = DEFAULT_TOOL_RESULT_MAX_BYTES,
): SanitizeResult {
  const stats: SanitizeStats = { strippedBytes: 0, truncatedBytes: 0 };

  if (!response || typeof response !== 'object') {
    return { sanitized: response, stats };
  }

  const content = (response as { content?: unknown }).content;
  if (!Array.isArray(content)) {
    return { sanitized: response, stats };
  }

  const newContent = content.map((block) => {
    if (
      !block ||
      typeof block !== 'object' ||
      (block as { type?: unknown }).type !== 'text' ||
      typeof (block as { text?: unknown }).text !== 'string'
    ) {
      return block;
    }
    const { out, stats: blockStats } = sanitizeText(
      (block as { text: string }).text,
      byteCap,
    );
    stats.strippedBytes += blockStats.strippedBytes;
    stats.truncatedBytes += blockStats.truncatedBytes;
    return { ...(block as object), text: out };
  });

  return {
    sanitized: { ...(response as object), content: newContent },
    stats,
  };
}

/**
 * `TaskOutput` accepts `{ task_id, block?, timeout? }`. The SDK
 * defaults `block` to `true` (see `cli.js`: `block: z.block ?? !0`),
 * so an absent flag still triggers the leak path.
 *
 * Treat any `block` value other than the literal boolean `false` as
 * blocking — including `undefined`, `true`, `'true'`, `1`, etc. The
 * deny path is the only safe default; legitimate non-blocking polls
 * are a one-character change to add `block: false`.
 */
export function shouldDenyTaskOutputBlock(toolInput: unknown): {
  deny: boolean;
  reason?: string;
} {
  if (!toolInput || typeof toolInput !== 'object') {
    // Malformed input — let the SDK surface its own validation error
    // rather than masking it with a deny.
    return { deny: false };
  }
  const block = (toolInput as { block?: unknown }).block;
  if (block === false) {
    return { deny: false };
  }
  return {
    deny: true,
    reason:
      'TaskOutput(block=true) leaks the raw sub-agent JSONL transcript ' +
      'into this session on timeout — the exact mechanism that poisoned ' +
      'the maintenance session on 2026-04-24 with invisible-Unicode ' +
      'padding from a Composio Gmail body. Call TaskOutput with ' +
      '`block: false` and poll status, or have the sub-agent write its ' +
      'final JSON to a known path under /workspace/group and read it ' +
      'with the Read tool.',
  };
}
