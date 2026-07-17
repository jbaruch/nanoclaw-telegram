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
  /**
   * Regex tested against the tool input. By default that is the
   * JSON-stringified `tool_input`; when `inputField` is set, it is the raw
   * string value of that field instead (see `inputField`).
   */
  inputPattern: RegExp;
  /**
   * When set, `inputPattern` is tested against the RAW string value of this
   * `tool_input` field (e.g. `command` for Bash, `url` for WebFetch) rather
   * than the JSON-stringified whole. Matching the raw field avoids the
   * JSON-escaping pitfalls of the serialized form — a real newline is an
   * actual `\n` (not the two chars `\` + `n`), a quoted argument is a plain
   * `"`, and a URL can be host-anchored with `^`. If the field is absent or
   * not a string, the entity does not match. Entities without `inputField`
   * keep testing against the serialized whole.
   */
  inputField?: string;
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
    // WebSearch is the documented misroute surface — the agent
    // searches the web for "nanoclaw" and grabs the upstream/qwibitai
    // fork instead of the canonical jbaruch fork.
    toolNames: [/^WebSearch$/],
    inputPattern: /\bnanoclaw\b/i,
  },
  {
    id: 'nanoclaw-repo-gh',
    label: 'the canonical NanoClaw repo target',
    pointer: '/workspace/trusted/memory/reference_nanoclaw_repo.md',
    // Same misroute as `nanoclaw-repo`, different transport: GitHub repo
    // lookups moved off Composio MCP tools onto the `gh` CLI over Bash
    // (#797, #639). A bare `nanoclaw` match on Bash is unusable — the repo
    // path itself is `~/nanoclaw`, so nearly every command mentions it — and
    // decoupled lookaheads are just as bad: they'd fire when a lookup of one
    // repo shares a command with an unrelated `~/nanoclaw` path. So the
    // `nanoclaw` token must belong to the lookup itself, in the same command
    // segment (no crossing `|`, `&&`, `;`, or a newline). Three shapes, each
    // where nanoclaw is the ambiguous target:
    //   - `gh search repos … nanoclaw` — a repo search (not a bare
    //     `gh search`, which also covers issues/code/prs);
    //   - `gh api …search/repositories…nanoclaw` — the REST equivalent;
    //   - `gh repo view …nanoclaw` where the owner is NOT the canonical
    //     `jbaruch/` — `gh repo view nanoclaw` / `gh repo view qwibitai/
    //     nanoclaw` nudge; `gh repo view jbaruch/nanoclaw` already names the
    //     fork and stays silent (the `(?<!jbaruch\/)` lookbehind).
    // A scoped `gh issue list --repo jbaruch/nanoclaw` and ordinary
    // path-bearing commands never match. `inputField: 'command'` runs this
    // against the RAW command string (not `JSON.stringify(tool_input)`), so
    // the segment class stops at an actual newline (`\n` / `\r`) and a
    // double-quoted argument (`gh search repos "nanoclaw"`) reads as a plain
    // `"` — no JSON-escape handling needed.
    // The repo target is matched as the EXACT token `nanoclaw` —
    // `(?<![\w-])nanoclaw(?![\w-])` — so a hyphenated sibling like
    // `nanoclaw-tools` (a different repo) does not match (`\b` alone treats
    // the `-` as a boundary). The canonical-owner exclusion uses a
    // word-boundaried `(?<!\bjbaruch\/)`, so only the exact owner `jbaruch`
    // is treated as canonical — an impersonator like `notjbaruch/nanoclaw`
    // still nudges.
    toolNames: [/^Bash$/],
    inputField: 'command',
    inputPattern:
      /\bgh\s+(?:search\s+repos\b[^|&;\n\r`]*(?<![\w-])nanoclaw(?![\w-])|api\b[^|&;\n\r`]*search\/repositories[^|&;\n\r`]*(?<![\w-])nanoclaw(?![\w-])|repo\s+view\b[^|&;\n\r`]*?(?<!\bjbaruch\/)(?<![\w-])nanoclaw(?![\w-]))/i,
  },
  {
    id: 'nanoclaw-repo-webfetch',
    label: 'the canonical NanoClaw repo target',
    pointer: '/workspace/trusted/memory/reference_nanoclaw_repo.md',
    // The WebFetch arm of the same misroute (#797): reading a GitHub repo
    // page for "nanoclaw" and landing on the upstream qwibitai fork. Scoped
    // to a NON-canonical repo-page URL shape — `github.com/<owner>/nanoclaw`
    // where the owner is not `jbaruch` — so reading the canonical
    // `github.com/jbaruch/nanoclaw` page (already the right fork) stays
    // silent, and a github.com *search* URL (`github.com/search?q=nanoclaw`)
    // or an unrelated page never matches. The repo segment is the EXACT
    // `nanoclaw` (`(?![\w-])` terminator), so a hyphenated sibling repo like
    // `github.com/qwibitai/nanoclaw-tools` does not match; the canonical
    // exclusion is anchored right after the host, so `notjbaruch/nanoclaw`
    // still nudges. `inputField: 'url'` runs this against the RAW URL and the
    // `^` anchors github.com to the FETCHED host — a non-github fetch that
    // merely embeds a github.com URL in its query (`https://example.com/?u=
    // https://github.com/qwibitai/nanoclaw`) does not match. A separate
    // entity from `nanoclaw-repo-gh` so the github.com shape never fires on a
    // Bash command that merely mentions a URL.
    toolNames: [/^WebFetch$/],
    inputField: 'url',
    inputPattern:
      /^https?:\/\/(?:www\.)?github\.com\/(?!jbaruch\/nanoclaw(?![\w-]))[^/\s"]+\/nanoclaw(?![\w-])/i,
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
    // Entities with `inputField` test against the raw field value (no JSON
    // escaping); an absent/non-string field means the entity can't match.
    const target = entity.inputField
      ? rawInputField(toolInput, entity.inputField)
      : inputStr;
    if (target === null || !entity.inputPattern.test(target)) {
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

/**
 * Return the RAW string value of `field` on a tool-input object, or null when
 * the input is not an object or the field is absent / not a string. Used by
 * entities with `inputField` so `inputPattern` runs against the unescaped
 * field (e.g. Bash `command`, WebFetch `url`) instead of the serialized whole.
 */
function rawInputField(input: unknown, field: string): string | null {
  if (input === null || typeof input !== 'object') {
    return null;
  }
  const value = (input as Record<string, unknown>)[field];
  return typeof value === 'string' ? value : null;
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
