/**
 * authoritative-source — pure detection logic for the
 * `authoritative-source-nudge` PreToolUse hook (#226, tracks #214).
 *
 * The agent has a recurring meta-bug: when an authoritative pointer
 * exists for a known fact (memory file, env var, canonical SQLite
 * table), it skips the pointer and runs a fresh search/list, then
 * grabs the first plausible result. The bug surfaces as many distinct
 * symptoms (wrong repo on a PR comment, wrong chat JID on a SQL
 * lookup, wrong group registration via JSON snapshot) sharing one root
 * cause: lazy entity lookup.
 *
 * This hook fires AT the decision point — before the search/list/SQL
 * tool call lands — and injects a `systemMessage` pointing the agent
 * at the canonical source. The tool call is NOT denied; standing rules
 * already cover that ground (and denying entity lookups outright would
 * be too coarse — sometimes the search is legitimate). The nudge
 * exists to interrupt the pattern-match loop.
 *
 * The catalogue below mirrors the incident table in #214: each entry
 * comes from an observed misroute. Adding a new entry should require
 * an observed production incident, not speculation.
 *
 * Kept SDK-free so the root vitest can exercise it without spinning up
 * `@anthropic-ai/claude-agent-sdk`.
 */

export interface AuthoritativeEntity {
  /** Stable identifier surfaced to logs and tests. */
  id: string;
  /** Human label used inside the reminder text. */
  label: string;
  /** Where the agent should look — surfaced inside the reminder. */
  pointer: string;
  /** Tool-name regexes; at least one must match the calling tool. */
  toolNames: RegExp[];
  /** Regex tested against the JSON-stringified `tool_input`. */
  inputPattern: RegExp;
}

export interface AuthoritativeNudgeDecision {
  /** True iff the hook should inject a reminder. The tool call still fires. */
  nudge: boolean;
  /** Reminder text. Empty when `nudge === false`. */
  systemMessage: string;
  /** Diagnostic info for logging. */
  matched?: { id: string; label: string; pointer: string };
}

/**
 * Curated catalogue of known lazy-lookup → authoritative-pointer pairs.
 *
 * Each entry's `inputPattern` is tested against the tool input
 * serialised as JSON (or used as-is if it's already a string). The
 * regexes are deliberately narrow: a fresh `SELECT jid FROM chats
 * LIMIT` query is the documented bug, while `SELECT jid FROM chats
 * WHERE name=...` is a legitimate scoped lookup and must not trip the
 * nudge. False positives here turn the nudge into noise the agent
 * learns to ignore.
 */
const ENTITIES: AuthoritativeEntity[] = [
  {
    id: 'nanoclaw-repo',
    label: 'the canonical NanoClaw repo target',
    pointer: '/workspace/trusted/memory/reference_nanoclaw_repo.md',
    // Composio search/list tools and WebSearch are the documented
    // misroute surface — the agent searches GitHub or the web for
    // "nanoclaw" and grabs the upstream/qwibitai fork instead of the
    // canonical jbaruch fork.
    toolNames: [/^mcp__composio__.*(search|list).*/i, /^WebSearch$/],
    inputPattern: /\bnanoclaw\b/i,
  },
  {
    id: 'chat-jid',
    label: 'the active chat JID',
    pointer: 'the NANOCLAW_CHAT_JID environment variable',
    // The lazy form is `SELECT ... FROM chats ... LIMIT N` — pulling
    // the first row instead of reading the env var the orchestrator
    // already wired in. A scoped `SELECT ... FROM chats WHERE jid=...`
    // is legitimate and won't match.
    toolNames: [/^Bash$/],
    inputPattern: /\bSELECT\b[^;]*\bFROM\s+chats\b[^;]*\bLIMIT\b/i,
  },
  {
    id: 'registered-groups',
    label: 'the registered group list',
    pointer: 'the SQLite `registered_groups` table on the host messages.db',
    // The host writes an `available_groups.json` snapshot for the
    // spawner; the agent then re-reads that JSON instead of querying
    // the table that owns the data. Read/Grep/Glob/Bash all surface
    // the same bug class.
    toolNames: [/^(Read|Grep|Glob|Bash)$/],
    inputPattern: /\bavailable_groups\.json\b/,
  },
];

/**
 * Inspect a tool call and decide whether to inject an authoritative-
 * source reminder. The tool call is never denied — this is a NUDGE.
 *
 * `toolName` and `toolInput` come from the SDK's
 * `PreToolUseHookInput`. Both are typed `unknown` upstream; we
 * narrow here so the helper stays SDK-free.
 *
 * The `catalogue` parameter exists for tests — production callers use
 * the default `ENTITIES`.
 */
export function detectAuthoritativeLookup(
  toolName: unknown,
  toolInput: unknown,
  catalogue: AuthoritativeEntity[] = ENTITIES,
): AuthoritativeNudgeDecision {
  if (typeof toolName !== 'string' || toolName.length === 0) {
    return { nudge: false, systemMessage: '' };
  }
  // Filter catalogue by tool name first so the JSON.stringify cost is
  // only paid when at least one entity could plausibly match.
  const candidates = catalogue.filter((entity) =>
    entity.toolNames.some((re) => re.test(toolName)),
  );
  if (candidates.length === 0) {
    return { nudge: false, systemMessage: '' };
  }
  const inputStr = serialiseToolInput(toolInput);
  if (inputStr === null) {
    return { nudge: false, systemMessage: '' };
  }
  for (const entity of candidates) {
    if (!entity.inputPattern.test(inputStr)) {
      continue;
    }
    return {
      nudge: true,
      matched: {
        id: entity.id,
        label: entity.label,
        pointer: entity.pointer,
      },
      systemMessage: buildReminder(entity),
    };
  }
  return { nudge: false, systemMessage: '' };
}

function serialiseToolInput(input: unknown): string | null {
  if (typeof input === 'string') {
    return input;
  }
  if (input === null || input === undefined) {
    return '';
  }
  try {
    return JSON.stringify(input);
  } catch (err: unknown) {
    // Only the documented circular-structure case (TypeError on
    // self-referential objects) is expected here. Anything else is
    // a real bug — propagate so it isn't silently masked.
    if (err instanceof TypeError) {
      return null;
    }
    throw err;
  }
}

function buildReminder(entity: AuthoritativeEntity): string {
  return (
    `Authoritative source for ${entity.label}: ${entity.pointer}. ` +
    'Read it before searching/listing — a fresh search result has no ' +
    'priority over the canonical pointer, and grabbing the first plausible ' +
    'match is the recurring lazy-lookup bug (#214). If the pointer is stale ' +
    'or missing, say so explicitly before falling back to search.'
  );
}

/** Exposed for tests so the catalogue id set is stable. */
export const AUTHORITATIVE_ENTITY_IDS = ENTITIES.map((e) => e.id);
