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
    pattern: /^mcp__composio__gmail_(fetch|get|list|search|find|read)\w*$/i,
    prefix: 'gmail',
  },
  {
    pattern: /^mcp__composio__googlecalendar_(list|get|find|search|read|fetch)\w*$/i,
    prefix: 'calendar',
  },
  {
    pattern: /^mcp__composio__slack_(list|fetch|get|search|read|history)\w*$/i,
    prefix: 'slack',
  },
  {
    pattern: /^mcp__composio__github_(get|list|search|read|find|fetch)\w*$/i,
    prefix: 'github',
  },
];

export function inferReadSource(toolName: string): ReadSource | null {
  if (typeof toolName !== 'string' || toolName.length === 0) return null;
  for (const { pattern, prefix } of READ_TOOL_PATTERNS) {
    if (pattern.test(toolName)) {
      const value = toolName.replace(/^mcp__composio__/, '');
      return { prefix, value };
    }
  }
  return null;
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
  const source = inferReadSource(toolName);
  if (!source) return { wrapped: response, mutated: false };
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
  const newContent = content.map((block) => {
    if (
      !block ||
      typeof block !== 'object' ||
      (block as { type?: unknown }).type !== 'text' ||
      typeof (block as { text?: unknown }).text !== 'string'
    ) {
      return block;
    }
    const original = (block as { text: string }).text;
    if (original.length === 0) return block;
    mutated = true;
    return {
      ...(block as object),
      text: wrapUntrustedInput(
        neutralizeWrapTokens(original),
        source.prefix,
        source.value,
      ),
    };
  });

  if (!mutated) return { wrapped: response, mutated: false };

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
