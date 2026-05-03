/**
 * Wire-tool catalog interceptor (issue #119).
 *
 * The Claude Agent SDK ships a fixed catalog of built-in tools and emits
 * the entire array on every /v1/messages request. `allowedTools` is only
 * a prompt hint — it does NOT trim the wire payload. To actually shrink
 * cache_create cost we have to mutate the outgoing `tools[]` array at the
 * credential-proxy boundary.
 *
 * This module:
 *  1. Strips a fixed REMOVE list of SDK-builtin tools we never use.
 *  2. Replaces the `Bash` tool description (~3,350 tokens) with a trimmed
 *     ~250-token version. Safety rules live in groups/global/BASH_SAFETY.md
 *     (backfilled in #468) and are @-imported by every tier's CLAUDE.md, so
 *     they don't need to repeat here.
 *
 * Default ON; gated by STRIP_DEAD_TOOLS env var. Set STRIP_DEAD_TOOLS=0 to
 * disable (full-catalog passthrough).
 *
 * Private uses Composio for Gmail/Calendar (not Superhuman MCP), so the
 * Superhuman filter shim that ships in `ligolnik#120` is intentionally
 * NOT cherry-picked here — there is no Superhuman tool catalog to filter.
 */

/**
 * SDK-builtin tools we never use. Names verified against the live captured
 * tool catalog (references/sdk-preset-v0.2.112-tool-catalog.json).
 *
 * NOTE: `TodoWrite`, `TeamCreate`, `TeamDelete`, `SendMessage`, `TaskOutput`,
 * `TaskStop` were on the original audit REMOVE list but are explicitly KEPT
 * (defense-in-depth — the `Agent` tool IS used and these may fire inside
 * its flow).
 */
export const DEAD_TOOL_NAMES: ReadonlySet<string> = new Set([
  'NotebookEdit',
  'PushNotification',
  'EnterPlanMode',
  'ExitPlanMode',
  'EnterWorktree',
  'ExitWorktree',
  'RemoteTrigger',
  'ListMcpResourcesTool',
  'ReadMcpResourceTool',
  'CronCreate',
  'CronDelete',
  'CronList',
  'ScheduleWakeup',
  'Monitor',
  'AskUserQuestion',
]);

/**
 * Trimmed Bash tool description (~250 tokens). Replaces the full ~3,350
 * token version emitted by the SDK. Operational essentials only — safety
 * rules live in groups/global/CLAUDE.md.
 */
export const TRIMMED_BASH_DESCRIPTION = `Executes a given bash command and returns its output.

Working directory persists between commands; shell state does not. Quote paths with spaces. Prefer absolute paths over \`cd\`. Set \`run_in_background: true\` for fire-and-forget commands; results readable later via Read. Default timeout 120000ms, max 600000ms. Don't sleep between commands. Don't \`find /\` (use \`.\` instead).

For one-off file reads/writes/edits, prefer Read/Write/Edit. For searches, prefer Grep/Glob. Bash is for shell-only operations: \`gh\`, \`git\`, \`curl\`, build/test scripts, system commands.`;

export interface FilterStats {
  toolsStripped: number;
  descriptionsTrimmed: number;
}

interface ToolEntry {
  name?: unknown;
  description?: unknown;
  [k: string]: unknown;
}

interface MessagesRequestBody {
  tools?: ToolEntry[];
  [k: string]: unknown;
}

/**
 * Mutate a parsed /v1/messages request body in place: strip dead tools and
 * trim the Bash description. Returns stats describing what changed.
 *
 * If `tools` is absent or not an array, returns zero stats (no-op).
 */
export function filterToolsInBody(parsed: MessagesRequestBody): FilterStats {
  const stats: FilterStats = { toolsStripped: 0, descriptionsTrimmed: 0 };

  if (!parsed || typeof parsed !== 'object') return stats;
  if (!Array.isArray(parsed.tools)) return stats;

  const before = parsed.tools.length;
  const kept: ToolEntry[] = [];
  for (const tool of parsed.tools) {
    if (
      tool &&
      typeof tool === 'object' &&
      typeof tool.name === 'string' &&
      DEAD_TOOL_NAMES.has(tool.name)
    ) {
      continue;
    }
    if (
      tool &&
      typeof tool === 'object' &&
      tool.name === 'Bash' &&
      typeof tool.description === 'string' &&
      tool.description !== TRIMMED_BASH_DESCRIPTION
    ) {
      tool.description = TRIMMED_BASH_DESCRIPTION;
      stats.descriptionsTrimmed += 1;
    }
    kept.push(tool);
  }
  stats.toolsStripped = before - kept.length;
  parsed.tools = kept;

  return stats;
}

/**
 * True iff this request URL targets the messages endpoint (with or without
 * query string).
 */
export function isMessagesEndpoint(url: string | undefined): boolean {
  if (!url) return false;
  const path = url.split('?')[0];
  return path === '/v1/messages';
}

/**
 * True iff the wire-tool interceptor is enabled. Default ON; disabled only
 * when STRIP_DEAD_TOOLS is explicitly set to '0'.
 */
export function isInterceptorEnabled(env: NodeJS.ProcessEnv): boolean {
  return env.STRIP_DEAD_TOOLS !== '0';
}

/**
 * Apply the wire-tool filter to a raw request body buffer for the
 * /v1/messages endpoint. Returns the (possibly rewritten) body buffer and
 * stats. On any non-fatal condition (not a messages request, interceptor
 * disabled, body not JSON, no tools field) returns the original buffer
 * unchanged with zero stats.
 *
 * The optional `onParseError` hook lets callers log parse failures without
 * silently swallowing them — required by host-conventions ("no error
 * suppression").
 */
export function applyWireToolFilter(
  url: string | undefined,
  method: string | undefined,
  body: Buffer,
  env: NodeJS.ProcessEnv,
  onParseError?: (err: unknown) => void,
): { body: Buffer; stats: FilterStats; applied: boolean } {
  const zero: FilterStats = { toolsStripped: 0, descriptionsTrimmed: 0 };

  if (!isInterceptorEnabled(env)) {
    return { body, stats: zero, applied: false };
  }
  if (method !== 'POST') {
    return { body, stats: zero, applied: false };
  }
  if (!isMessagesEndpoint(url)) {
    return { body, stats: zero, applied: false };
  }
  if (body.length === 0) {
    return { body, stats: zero, applied: false };
  }

  let parsed: MessagesRequestBody;
  try {
    parsed = JSON.parse(body.toString('utf8'));
  } catch (err) {
    // Narrow per `error-handling.Specific Exceptions`. A malformed
    // /v1/messages body is recoverable: the proxy's contract is to
    // forward the request as-is when it can't parse, and the upstream
    // will reject the bad JSON itself with a clearer error than we'd
    // synthesize here. Anything else (TypeError from a future
    // refactor, etc.) propagates so the bug surfaces.
    if (err instanceof SyntaxError) {
      if (onParseError) onParseError(err);
      return { body, stats: zero, applied: false };
    }
    throw err;
  }

  const stats = filterToolsInBody(parsed);
  if (stats.toolsStripped === 0 && stats.descriptionsTrimmed === 0) {
    return { body, stats, applied: false };
  }

  const next = Buffer.from(JSON.stringify(parsed), 'utf8');
  return { body: next, stats, applied: true };
}
