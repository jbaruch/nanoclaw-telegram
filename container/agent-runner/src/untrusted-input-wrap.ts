/**
 * #321 PR 2 — Encoding A wrap for MCP read-tool results.
 *
 * Pattern-matches MCP tool names against a read-action allowlist and
 * wraps each text content block in `<untrusted-input source="...">`.
 * The resulting envelope is the same one used by the orchestrator-side
 * prompt wrap (PR 1) and the agent-browser script wrap (PR 3), so #322's
 * walk-back has a single in-band signal to grep for across all three
 * Encoding-A entry points.
 *
 * Built-in tools (WebFetch, Bash, Read) cannot use this path — the SDK's
 * `updatedMCPToolOutput` hook surface is MCP-only. Encoding B (sidecar
 * `additionalContext` sentinel) covers them in PR 4.
 *
 * The hook is wired in `index.ts` AFTER the #117 sanitizer and #140
 * fidelity inspector so neither sees synthetic wrap bytes; the wrap
 * runs last and only mutates the text the model ultimately reads.
 */

import { SourcePrefix, wrapUntrustedInput } from './untrusted-input-sources.js';

export interface ReadSource {
  prefix: SourcePrefix;
  value: string;
}

/**
 * Allowlist of MCP read-action patterns. A tool name that matches a
 * pattern emits Encoding A wrap with the associated `prefix`. Patterns
 * are deliberately conservative — only verbs that return external
 * content (`fetch`, `get`, `list`, `search`, `find`, `read`, `history`)
 * are included; mutating verbs (`send`, `post`, `create`, `update`,
 * `delete`, `modify`, `archive`, `reply`) are excluded by omission and
 * stay un-wrapped.
 *
 * Grow this list per real usage — adding a row is a one-line change.
 */
const READ_TOOL_PATTERNS: ReadonlyArray<{
  pattern: RegExp;
  prefix: SourcePrefix;
}> = [
  {
    // Tessl registry MCP — `search` (registry tile search), `outdated`
    // (remote version check), `query_library_docs` (external library
    // docs). Mutating verbs (`install`, `login`, `update`, `uninstall`,
    // `new_tile`) and the local-only `status` are excluded by omission.
    pattern: /^mcp__tessl__(search|outdated|query_library_docs)\w*$/i,
    prefix: 'tessl',
  },
  {
    // snitchmd fetch — host-side docker render of an arbitrary URL
    // returning extracted markdown. Body content is attacker-controlled
    // (any page on the web) so it wraps as `web:fetch_markdown` (the
    // `inferReadRow` value strips the `mcp__<server>__` prefix and
    // emits the bare tool name as the source-value); #322's walk-back
    // applies cross-source-cell rate limits.
    pattern: /^mcp__nanoclaw__fetch_markdown$/i,
    prefix: 'web',
  },
];

interface InternalReadRow {
  prefix: SourcePrefix;
  value: string;
}

function inferReadRow(toolName: string): InternalReadRow | null {
  if (typeof toolName !== 'string' || toolName.length === 0) return null;
  for (const row of READ_TOOL_PATTERNS) {
    if (row.pattern.test(toolName)) {
      const value = toolName.replace(/^mcp__[^_]+__/, '');
      return { prefix: row.prefix, value };
    }
  }
  return null;
}

export function inferReadSource(toolName: string): ReadSource | null {
  const row = inferReadRow(toolName);
  if (!row) return null;
  return { prefix: row.prefix, value: row.value };
}

export interface WrapResult {
  wrapped: unknown;
  mutated: boolean;
}

/**
 * Wraps each `{ type: 'text', text }` block in the response with the
 * Encoding A envelope. Accepts both shapes the MCP wire format produces:
 *
 *   - wrapped:  `{ content: [{ type: 'text', text }, ...] }`
 *   - bare:     `[{ type: 'text', text }, ...]`
 *
 * Non-text blocks (image, resource) and unrecognized shapes pass through
 * untouched. Empty-text blocks are skipped — wrapping an empty string in
 * the envelope adds noise without provenance value.
 *
 * Returns `{ mutated: false }` for any tool name not in the read-action
 * allowlist; the caller should treat that as no-op (no
 * `updatedMCPToolOutput` emitted).
 */
export function wrapMcpToolResult(
  toolName: string,
  response: unknown,
): WrapResult {
  const row = inferReadRow(toolName);
  if (!row) {
    return { wrapped: response, mutated: false };
  }
  if (!response || typeof response !== 'object') {
    return { wrapped: response, mutated: false };
  }

  const isBareArray = Array.isArray(response);
  const content = isBareArray
    ? (response as unknown[])
    : (response as { content?: unknown }).content;

  if (!Array.isArray(content)) {
    return { wrapped: response, mutated: false };
  }

  let mutated = false;
  const newContent: unknown[] = [];
  for (const block of content) {
    if (
      !block ||
      typeof block !== 'object' ||
      (block as { type?: unknown }).type !== 'text' ||
      typeof (block as { text?: unknown }).text !== 'string'
    ) {
      newContent.push(block);
      continue;
    }
    const original = (block as { text: string }).text;
    if (original.length === 0) {
      newContent.push(block);
      continue;
    }

    mutated = true;
    newContent.push({
      ...(block as object),
      text: wrapUntrustedInput(
        neutralizeWrapTokens(original),
        row.prefix,
        row.value,
      ),
    });
  }

  if (!mutated) {
    return { wrapped: response, mutated: false };
  }

  return {
    wrapped: isBareArray
      ? newContent
      : { ...(response as object), content: newContent },
    mutated: true,
  };
}

/**
 * Neutralize literal `<untrusted-input ...>` and `</untrusted-input>`
 * sequences inside the text we are about to wrap. A read tool's output
 * (email body, Slack message, GitHub issue) can contain those tokens
 * verbatim — adversarial or otherwise — and a naive wrap would let them
 * spoof a nested envelope or close the outer one early, breaking #322's
 * walk-back parser.
 *
 * We only escape the leading `<` of each opening / closing token. The
 * model still sees recognizable text ("&lt;untrusted-input>"), but the
 * walk-back regex (which keys off `<untrusted-input` literal) skips it.
 */
function neutralizeWrapTokens(text: string): string {
  return text.replace(/<(\/?untrusted-input)\b/gi, '&lt;$1');
}
