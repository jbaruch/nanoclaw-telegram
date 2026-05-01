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
 *   - `WebSearch`              → `web:<query>` (search results are external
 *                                 web bytes; same prefix as WebFetch so the
 *                                 ACL applies the same sink restrictions)
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

import path from 'path';
import {
  SourcePrefix,
  escapeAttr,
  formatSource,
} from './untrusted-input-sources.js';

/**
 * Container cwd at runtime, used to resolve relative `Read` paths
 * before classifying them. Set in `index.ts` via the SDK `cwd` option;
 * mirrored here so `isExternalPath` can do its check without an SDK
 * dependency.
 */
const CONTAINER_CWD = '/workspace/group';

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

  if (toolName === 'WebSearch') {
    const query = (toolInput as { query?: unknown }).query;
    if (typeof query !== 'string' || query.length === 0) return null;
    return { prefix: 'web', value: query };
  }

  if (toolName === 'Read') {
    const filePath = (toolInput as { file_path?: unknown }).file_path;
    if (typeof filePath !== 'string' || filePath.length === 0) return null;
    const classified = classifyReadPath(filePath);
    if (!classified.isExternal) return null;
    // Emit the resolved/normalized absolute path so the marker is
    // unambiguous — `../../etc/passwd` becomes `file:/etc/passwd` and
    // `/workspace/group/../secret/x` becomes `file:/workspace/secret/x`.
    // The walk-back never has to re-resolve to know what was actually
    // read.
    return { prefix: 'file', value: classified.resolved };
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
 * Decide whether a `Read` target is external (i.e. outside every
 * standard workspace mount).
 *
 * Resolves and normalizes the path BEFORE prefix-checking so a poisoned
 * model can't slip past via traversal:
 *   - Relative input → resolved against the container cwd
 *     (`/workspace/group`), so `notes.md` → `/workspace/group/notes.md`
 *     (internal) but `../../etc/passwd` → `/etc/passwd` (external).
 *   - Absolute input → normalized so `/workspace/group/../secret/x`
 *     collapses to `/workspace/secret/x` (external) instead of being
 *     mis-classified by a naive `startsWith('/workspace/group/')` check.
 *
 * Internal mount roots are matched WITHOUT the trailing `/` so that the
 * mount root itself (e.g. `/workspace/group`) is also classified as
 * internal — `startsWith('/workspace/group/')` would have rejected the
 * bare root. Sibling paths like `/workspace/secret` are still external
 * because the prefix list does not contain them.
 */
export function isExternalPath(filePath: string): boolean {
  return classifyReadPath(filePath).isExternal;
}

/**
 * Resolve + classify a `Read` target. Returns both the normalized
 * absolute path (used as the sentinel `value`) and the internal/external
 * verdict in one pass — callers shouldn't double-resolve.
 */
export function classifyReadPath(filePath: string): {
  resolved: string;
  isExternal: boolean;
} {
  const resolved = path.posix.isAbsolute(filePath)
    ? path.posix.normalize(filePath)
    : path.posix.resolve(CONTAINER_CWD, filePath);
  const isExternal = !WORKSPACE_INTERNAL_PREFIXES.some(
    (p) => resolved === p.replace(/\/$/, '') || resolved.startsWith(p),
  );
  return { resolved, isExternal };
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

