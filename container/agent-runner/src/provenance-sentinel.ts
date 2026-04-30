/**
 * #321 PR 4 — Encoding B sidecar provenance sentinel for built-in tools.
 *
 * The Claude Agent SDK exposes `updatedMCPToolOutput` for MCP tools but
 * NO equivalent mutation surface for built-in tools (`WebFetch`, `Read`,
 * `Bash`) — see `node_modules/@anthropic-ai/claude-agent-sdk/sdk.d.ts:1663`.
 * Encoding A (literal `<untrusted-input>` wrap) is therefore impossible
 * for those tools.
 *
 * This module emits a machine-parseable sentinel via `additionalContext`
 * on PostToolUse — a sidecar marker the model sees as a system reminder,
 * and that #322's walk-back greps for alongside the in-band Encoding A
 * envelope. Both encodings carry the same typed `prefix:value` source
 * taxonomy from `untrusted-input-sources.ts` so the walk-back has a
 * single dispatch table.
 *
 * Coverage:
 *   - `WebFetch`               → `web:<url>`
 *   - `Read` on external paths → `file:<path>` (any path NOT under
 *                                 /workspace/{group,state,trusted,
 *                                 global,store,ipc}/)
 *   - `Bash` invocations whose
 *     command starts with
 *     `agent-browser`          → `agent-browser:<url-from-open-or-active>`
 *
 * Out of scope for v1: generic Bash egress (`curl`, `wget`, etc.). The
 * exfil-via-shell path is a separate concern; #322 can still gate it via
 * the existing trust-tier check, just without per-host source granularity.
 */

import { SourcePrefix, formatSource } from './untrusted-input-sources.js';

export const SENTINEL_PREFIX = 'PROVENANCE_MARKER:';

const WORKSPACE_INTERNAL_PREFIXES: ReadonlyArray<string> = [
  '/workspace/group/',
  '/workspace/state/',
  '/workspace/trusted/',
  '/workspace/global/',
  '/workspace/store/',
  '/workspace/ipc/',
];

export interface SentinelSource {
  prefix: SourcePrefix;
  value: string;
}

/**
 * Decide whether a built-in tool call needs a provenance sentinel and
 * what source to attach. Returns null for calls that don't need one
 * (internal-path Read, non-agent-browser Bash, malformed inputs).
 */
export function inferSentinelSource(
  toolName: string,
  toolInput: unknown,
): SentinelSource | null {
  if (typeof toolName !== 'string' || toolName.length === 0) return null;
  if (!toolInput || typeof toolInput !== 'object') return null;

  if (toolName === 'WebFetch') {
    const url = (toolInput as { url?: unknown }).url;
    if (typeof url !== 'string' || url.length === 0) return null;
    return { prefix: 'web', value: url };
  }

  if (toolName === 'Read') {
    const filePath = (toolInput as { file_path?: unknown }).file_path;
    if (typeof filePath !== 'string' || filePath.length === 0) return null;
    if (!isExternalPath(filePath)) return null;
    return { prefix: 'file', value: filePath };
  }

  if (toolName === 'Bash') {
    const command = (toolInput as { command?: unknown }).command;
    if (typeof command !== 'string' || command.length === 0) return null;
    const trimmed = command.trimStart();
    if (!isAgentBrowserCommand(trimmed)) return null;
    const url = extractAgentBrowserUrl(trimmed);
    return { prefix: 'agent-browser', value: url ?? 'active' };
  }

  return null;
}

/**
 * `Read` on absolute paths under any of the standard workspace mounts
 * is internal-trusted. Anything else absolute (`/etc/...`, `/mnt/...`,
 * `/home/...`, `/var/...`, `/tmp/...`) is external. Relative paths are
 * cwd-relative and the container's cwd is `/workspace/group` per
 * `index.ts`, so they resolve internal — return false.
 */
export function isExternalPath(filePath: string): boolean {
  if (!filePath.startsWith('/')) return false;
  return !WORKSPACE_INTERNAL_PREFIXES.some((p) => filePath.startsWith(p));
}

function isAgentBrowserCommand(command: string): boolean {
  // Match `agent-browser` as the first token (followed by space, end, or
  // a subcommand). Doesn't match `cat agent-browser-output.txt` or
  // `which agent-browser`.
  return /^agent-browser(\s|$)/.test(command);
}

function extractAgentBrowserUrl(command: string): string | null {
  // Match `agent-browser open <url>` (with optional flags between
  // `agent-browser` and `open`). The URL is the first non-flag arg
  // after `open`.
  const m = command.match(
    /^agent-browser(?:\s+--?\S+)*\s+open\s+(?:--?\S+\s+)*(\S+)/,
  );
  if (!m) return null;
  // Strip surrounding quotes if any (`agent-browser open "https://..."`).
  return m[1].replace(/^["']|["']$/g, '');
}

/**
 * Format the sidecar marker. Single-line, key=value, easy to grep —
 * #322's walk-back uses a literal-prefix scan to find these.
 *
 * The `tool_use_id` ties the marker back to the specific tool result it
 * annotates — without it, two interleaved tool calls of the same kind
 * would produce indistinguishable markers.
 */
export function formatSentinel(source: SentinelSource, toolUseId: string): string {
  const sourceStr = escapeAttr(formatSource(source.prefix, source.value));
  const idStr = escapeAttr(toolUseId);
  return `${SENTINEL_PREFIX} source="${sourceStr}" tool_use_id="${idStr}"`;
}

function escapeAttr(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/[\r\n]+/g, ' ');
}
