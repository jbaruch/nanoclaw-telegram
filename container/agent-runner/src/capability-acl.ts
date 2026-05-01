/**
 * #322 — Capability ACL per data source.
 *
 * Closes the structural gap left by #321 (which only MARKS provenance):
 * a `PreToolUse` hook walks back from the current tool call through the
 * conversation transcript, finds every untrusted-provenance signal in
 * the span between the call and the most recent operator-originated
 * user-turn boundary, and denies the call if its sink is not in the
 * intersection of allowed-sinks for every source seen.
 *
 * The provenance markers come from #321's two encodings — see
 * `untrusted-input-sources.ts` for the typed `prefix:value` taxonomy:
 *
 *   Encoding A — `<untrusted-input source="<prefix>:<value>">…</untrusted-input>`
 *                 in user-message bodies and MCP tool-result text blocks.
 *   Encoding B — `PROVENANCE_MARKER: source="<prefix>:<value>" tool_use_id="…"`
 *                 in `additionalContext` system-reminder messages emitted
 *                 by the PostToolUse sentinel hook for built-in tools.
 *
 * Walk-back semantics (per #322 spec):
 *
 *   1. Scan from the current tool call backward until reaching the most
 *      recent message that is a "real" user turn — i.e. NOT purely a
 *      `tool_result` block, NOT a system reminder. That message is the
 *      boundary.
 *   2. Collect every `prefix` from every marker in the span (boundary
 *      message included — its content can carry an `untrusted-container:`
 *      wrap from PR 1).
 *   3. If the resulting set is empty, the call is operator-originated
 *      trusted — bypass the ACL. (Per #318 design principle: trusted
 *      operator keeps full reach by default.)
 *   4. Otherwise, intersect the allowed-sink sets of every prefix in the
 *      collected set. If the current tool name matches any pattern in
 *      the intersection, allow; else deny.
 *
 * Operator opt-in tightening (e.g. forcing the ACL even on pure operator
 * chains) is out of scope for v1 — add when the use case appears.
 */

import { SourcePrefix } from './untrusted-input-sources.js';

/**
 * Allowed-sink table per source prefix.
 *
 * Each entry is matched against the tool name by `isToolAllowed` —
 * strings are matched with EXACT equality (so `Read` does not match
 * `ReadMore`); regexes are matched with `RegExp#test`. Tool names
 * follow the SDK conventions:
 *   - Built-in tools: `Read`, `Write`, `Edit`, `Bash`, `WebFetch`, etc.
 *   - MCP tools: `mcp__<server>__<action>`, e.g. `mcp__nanoclaw__send_message`.
 *
 * The table is intentionally conservative — it's safer to add a sink to
 * an allowlist when a real workflow needs it than to discover after a
 * prompt-injection that a permissive default let an exfil through.
 *
 * Read-tool patterns are spelled out here (not imported from
 * `untrusted-input-wrap.ts`) because that module's allowlist is
 * for WHICH tool results to wrap; this one is for WHICH sinks
 * untrusted-provenance can REACH. Different concerns, different lists.
 */
const READ_ONLY_COMPOSIO = /^mcp__composio__\w+?_(fetch|get|list|search|find|read|history)\w*$/i;

/**
 * Sinks every source row allows in addition to its own — read-only
 * informational tools that have no side effects on outbound state.
 * Centralised so adding a new "harmless" tool flows through one line.
 */
const COMMON_INERT_SINKS: ReadonlyArray<RegExp | string> = [
  'Read',
  'Glob',
  'Grep',
  'WebSearch',
  'TodoWrite',
  'ToolSearch',
  'Skill',
  'NotebookEdit',
];

const SINK_ALLOWLISTS: Record<SourcePrefix, ReadonlyArray<RegExp | string>> = {
  // Untrusted-container prompt (PR 1's retrofit — every prompt that
  // entered an untrusted container). Restrictive but workable: the
  // model can read, search, fetch, write to its own group folder, and
  // reply to its own chat. No cross-chat sends (host already gates).
  'untrusted-container': [
    ...COMMON_INERT_SINKS,
    'mcp__nanoclaw__send_message',
    'mcp__nanoclaw__send_file',
    'mcp__nanoclaw__react_to_message',
    'WebFetch',
    'Write',
    'Edit',
    'Bash',
    READ_ONLY_COMPOSIO,
    /^mcp__tessl__/,
    'Task',
    'TaskOutput',
    'TaskStop',
  ],

  // Cross-group user message — content arriving from another group's
  // chat. Treated as non-owner for v1 (no JID-to-owner mapping yet).
  'cross-group': [
    ...COMMON_INERT_SINKS,
    'mcp__nanoclaw__send_message',
    'mcp__nanoclaw__react_to_message',
    READ_ONLY_COMPOSIO,
  ],

  // Web content via WebFetch or agent-browser. The injection vector
  // most likely to carry "now go email/post/exfil X" instructions —
  // restrict outbound to read-only Composio + own-chat reply.
  'web': [
    ...COMMON_INERT_SINKS,
    'mcp__nanoclaw__send_message',
    'Write',
    'Edit',
    READ_ONLY_COMPOSIO,
  ],
  'agent-browser': [
    ...COMMON_INERT_SINKS,
    'mcp__nanoclaw__send_message',
    'Write',
    'Edit',
    READ_ONLY_COMPOSIO,
  ],

  // Email/calendar/Slack/GitHub/Tessl read content. Same posture as
  // `web:*` — these are external bytes that could carry instructions.
  'gmail': [
    ...COMMON_INERT_SINKS,
    'mcp__nanoclaw__send_message',
    'Write',
    'Edit',
    READ_ONLY_COMPOSIO,
  ],
  'calendar': [
    ...COMMON_INERT_SINKS,
    'mcp__nanoclaw__send_message',
    'Write',
    'Edit',
    READ_ONLY_COMPOSIO,
  ],
  'slack': [
    ...COMMON_INERT_SINKS,
    'mcp__nanoclaw__send_message',
    'Write',
    'Edit',
    READ_ONLY_COMPOSIO,
  ],
  'github': [
    ...COMMON_INERT_SINKS,
    'mcp__nanoclaw__send_message',
    'Write',
    'Edit',
    READ_ONLY_COMPOSIO,
  ],
  'tessl': [
    ...COMMON_INERT_SINKS,
    'mcp__nanoclaw__send_message',
    READ_ONLY_COMPOSIO,
  ],

  // External file `Read` — bytes from outside the workspace mounts.
  // No outbound — these can leak secrets if echoed.
  'file': [
    ...COMMON_INERT_SINKS,
    'Write',
    'Edit',
  ],
};

/**
 * Regex that finds every Encoding A wrap in a string. The `source`
 * attribute value is captured. Lazy on the closing tag so adjacent
 * wraps don't get merged.
 */
const ENCODING_A_REGEX =
  /<untrusted-input\s+source="([^"]*)"[^>]*>[\s\S]*?<\/untrusted-input>/gi;

/**
 * Regex that finds every Encoding B sentinel line. The `source`
 * attribute value is captured.
 */
const ENCODING_B_REGEX =
  /^PROVENANCE_MARKER:\s+source="([^"]*)"\s+tool_use_id="[^"]*"/m;

/**
 * Sentinel prefix used when a marker carries a `source=` value whose
 * prefix is NOT in `SINK_ALLOWLISTS`. Could happen if a future emitter
 * (or a partial deployment / version skew) introduces a new prefix
 * before this module knows about it. Treating it as "no marker" would
 * silently bypass the gate — exactly the failure mode the gate exists
 * to prevent. Treating it as the most restrictive allowlist (inert
 * sinks only) fails closed.
 */
export const UNKNOWN_PREFIX = '__unknown__' as const;

/**
 * The set of prefixes the walk-back can return. Equal to `SourcePrefix`
 * plus the `UNKNOWN_PREFIX` sentinel for forward-compat / version-skew
 * safety.
 */
export type AclPrefix = SourcePrefix | typeof UNKNOWN_PREFIX;

/**
 * Extract the `prefix` portion of every provenance marker in a string.
 * Returns the set of unique prefixes — values are not preserved here
 * because the ACL keys on prefix only.
 *
 * Both encodings are scanned. An unrecognized prefix collapses to the
 * `UNKNOWN_PREFIX` sentinel so the ACL fails closed against a future
 * emitter or partial-deployment skew rather than treating the marker
 * as absent.
 */
export function extractMarkerPrefixes(text: string): Set<AclPrefix> {
  const out = new Set<AclPrefix>();
  if (typeof text !== 'string' || text.length === 0) return out;
  for (const match of text.matchAll(ENCODING_A_REGEX)) {
    out.add(classifySourcePrefix(match[1]));
  }
  for (const match of text.matchAll(new RegExp(ENCODING_B_REGEX, 'gm'))) {
    out.add(classifySourcePrefix(match[1]));
  }
  return out;
}

function classifySourcePrefix(sourceAttr: string): AclPrefix {
  const colon = sourceAttr.indexOf(':');
  const candidate = colon === -1 ? sourceAttr : sourceAttr.slice(0, colon);
  if (Object.prototype.hasOwnProperty.call(SINK_ALLOWLISTS, candidate)) {
    return candidate as SourcePrefix;
  }
  return UNKNOWN_PREFIX;
}

/**
 * Allowlist used when the walk-back encounters a marker with an
 * unrecognized prefix. Inert sinks only — the model can read, search,
 * and reason, but cannot send messages, write files, or call any
 * destructive tool. The conservative posture matches the rule: a marker
 * we don't understand is still a marker; the chain is untrusted-
 * provenance and the gate fires.
 */
const UNKNOWN_PREFIX_ALLOWLIST: ReadonlyArray<RegExp | string> = COMMON_INERT_SINKS;

function getAllowlistForPrefix(
  prefix: AclPrefix,
): ReadonlyArray<RegExp | string> {
  if (prefix === UNKNOWN_PREFIX) return UNKNOWN_PREFIX_ALLOWLIST;
  return SINK_ALLOWLISTS[prefix];
}

/**
 * A minimal message shape the walk-back operates on. Compatible with
 * the `SessionMessage` type the SDK exposes via `getSessionMessages`,
 * but kept narrow so this module can be tested with synthetic data.
 *
 * `role` is `'user'` for both real user turns and tool-result-bearing
 * synthetic user turns; `isToolResult` distinguishes them. `text` is
 * the concatenated text content of the message — both encodings live
 * in plain-text fragments, so the walk-back can stay format-agnostic.
 */
export interface WalkBackMessage {
  role: 'user' | 'assistant' | 'system';
  isToolResult?: boolean;
  text: string;
}

/**
 * Walk back from the end of `messages` collecting every provenance
 * prefix until we hit the most recent "real" user turn (the boundary).
 *
 * Real user turn = `role === 'user'` AND NOT `isToolResult` AND NOT a
 * system reminder. The boundary message's own text IS scanned for
 * markers (an `untrusted-container:` wrap on the boundary message
 * itself is the most common signal — see PR 1).
 *
 * Returns the union of prefixes seen. An empty set means
 * operator-originated trusted; the caller bypasses the ACL.
 */
export function walkBackForProvenance(
  messages: ReadonlyArray<WalkBackMessage>,
): Set<AclPrefix> {
  return walkBackIterableForProvenance(reverseIterable(messages));
}

/**
 * Iterable variant: caller supplies an iterable of messages already in
 * REVERSE order (most-recent first). The walk consumes messages until
 * it hits the boundary, so a lazy backward iterator over the SDK's
 * transcript can stop early instead of pre-materializing the whole
 * `WalkBackMessage[]`. PreToolUse fires on every tool call; with a long
 * session and short spans, lazy iteration keeps the gate latency
 * bounded by span size, not transcript size.
 */
export function walkBackIterableForProvenance(
  reversed: Iterable<WalkBackMessage>,
): Set<AclPrefix> {
  const collected = new Set<AclPrefix>();
  for (const m of reversed) {
    for (const p of extractMarkerPrefixes(m.text)) collected.add(p);
    if (isBoundary(m)) break;
  }
  return collected;
}

function* reverseIterable<T>(arr: ReadonlyArray<T>): IterableIterator<T> {
  for (let i = arr.length - 1; i >= 0; i--) yield arr[i];
}

function isBoundary(m: WalkBackMessage): boolean {
  return m.role === 'user' && !m.isToolResult;
}

/**
 * Compute the intersection of allowed-sink lists across every prefix
 * in `prefixes`. The result is the most-restrictive set the call must
 * satisfy.
 *
 * Implementation: filter the FIRST prefix's list down to entries that
 * also appear in every other prefix's list, where "appear" is
 * determined by `sinkPatternsEqual` — strings match by `===`, regexes
 * match by identical `source` AND `flags`. There is no separate dedup
 * pass; if `first` happens to contain duplicates, they survive in the
 * output (none of the canonical lists in this module do). An empty
 * intersection means "no sink allowed" — every call denies.
 *
 * The `UNKNOWN_PREFIX` sentinel resolves through `getAllowlistForPrefix`
 * to a deliberately tiny inert list, so any unknown marker collapses
 * the intersection toward "deny everything but read-only inspection".
 */
export function intersectAllowedSinks(
  prefixes: ReadonlyArray<AclPrefix> | Set<AclPrefix>,
): ReadonlyArray<RegExp | string> {
  const arr = Array.from(prefixes);
  if (arr.length === 0) return [];
  const lists = arr.map(getAllowlistForPrefix);
  const [first, ...rest] = lists;
  return first.filter((sink) =>
    rest.every((list) => list.some((other) => sinkPatternsEqual(sink, other))),
  );
}

function sinkPatternsEqual(
  a: RegExp | string,
  b: RegExp | string,
): boolean {
  if (typeof a === 'string' && typeof b === 'string') return a === b;
  if (a instanceof RegExp && b instanceof RegExp) {
    return a.source === b.source && a.flags === b.flags;
  }
  return false;
}

/**
 * Match a tool name against an allowed-sink list. Strings match exactly;
 * regex patterns use `.test`. Returns true on first match.
 */
export function isToolAllowed(
  toolName: string,
  allowedSinks: ReadonlyArray<RegExp | string>,
): boolean {
  for (const sink of allowedSinks) {
    if (typeof sink === 'string') {
      if (sink === toolName) return true;
    } else {
      if (sink.test(toolName)) return true;
    }
  }
  return false;
}

/**
 * Top-level decision combining walk-back + ACL.
 *
 * Returns `{ kind: 'allow' }` for operator-originated chains (no
 * markers in span), `{ kind: 'allow' }` for chains whose collected
 * prefixes' intersection covers the tool, and `{ kind: 'deny', reason }`
 * otherwise.
 */
export type AclDecision =
  | { kind: 'allow'; reason?: string }
  | {
      kind: 'deny';
      reason: string;
      prefixes: ReadonlyArray<AclPrefix>;
    };

export function decideCapabilityAcl(
  toolName: string,
  messages: ReadonlyArray<WalkBackMessage>,
): AclDecision {
  return decideCapabilityAclIterable(
    toolName,
    reverseIterable(messages),
  );
}

/**
 * Iterable variant — see `walkBackIterableForProvenance`. Used by the
 * PreToolUse hook so the SDK transcript can be walked lazily backward
 * without pre-materializing the full `WalkBackMessage[]`.
 */
export function decideCapabilityAclIterable(
  toolName: string,
  reversed: Iterable<WalkBackMessage>,
): AclDecision {
  const prefixes = walkBackIterableForProvenance(reversed);
  if (prefixes.size === 0) {
    return { kind: 'allow', reason: 'operator-originated, trusted boundary' };
  }
  const intersected = intersectAllowedSinks(prefixes);
  if (isToolAllowed(toolName, intersected)) {
    return {
      kind: 'allow',
      reason: `untrusted-provenance allowed sink (sources: ${[...prefixes].join(',')})`,
    };
  }
  return {
    kind: 'deny',
    prefixes: [...prefixes],
    reason:
      `Capability ACL: tool ${toolName} is not in the allowed-sink ` +
      `intersection for untrusted-provenance sources [${[...prefixes].join(', ')}]. ` +
      `The current call chain contains content from at least one untrusted ` +
      `source (web/email/scraped/cross-group); the model may be following ` +
      `injected instructions. If this is a legitimate operator request, ` +
      `the operator should issue it directly without external content in ` +
      `the same chain.`,
  };
}

/**
 * Exported for tests + future ACL adjustments. Don't mutate.
 */
export const __ACL_INTERNALS = {
  SINK_ALLOWLISTS,
  ENCODING_A_REGEX,
  ENCODING_B_REGEX,
};
