/**
 * #324 — Destructive-op confirmation gates with owner-issued tokens.
 *
 * Provenance-conditional gate complementing #320 / #322. Tools that
 * mutate session lifecycle, registration, or run host-level destructive
 * commands (nuke_session, github_backup, set_trusted, etc.) are gated:
 *
 *   - Operator-originated trusted chain → allow. The operator's
 *     in-chat confirmation IS the confirmation; minting an OOB token
 *     for something Baruch literally just typed is paperwork theatre
 *     that trains him to mint reflexively.
 *   - Untrusted-provenance chain → require a matching unspent token
 *     issued via the host-side `aye-confirm` CLI. Token is single-use,
 *     scoped, TTL'd.
 *
 * Token store lives at `/workspace/trusted/aye_confirm_tokens.json`.
 * Host writes (via the CLI); the agent only reads + marks-used. A
 * separate Write-hook gate blocks the agent from mutating the file
 * directly so a poisoned chain can't forge tokens.
 *
 * `aye-confirm` CLI contract:
 *
 *   aye-confirm --scope <scope> --reason <text> --ttl <duration>
 *
 *   --scope   one of the values from `DestructiveScope`
 *   --reason  short prose for the audit trail (visible in token file)
 *   --ttl     duration like `5m`, `1h`, `30s` (default 5m)
 *
 *   Prints the minted token to stdout. The token is a host-minted
 *   LATCH — once written to the file, it stays there until the next
 *   matching destructive call consumes it (or until it expires /
 *   another `aye-confirm` overwrites). The agent does NOT need to be
 *   handed the token value; the hook simply checks for ANY unspent,
 *   unexpired, scope-matching token at PreToolUse time. This keeps
 *   the operator flow short — mint and walk away — without an
 *   extra round-trip through the chat.
 */

export type DestructiveScope =
  | 'nuke_session'
  | 'nuke_chat'
  | 'github_backup'
  | 'persist_global_file'
  | 'set_trusted'
  | 'set_trigger'
  | 'set_agent_model'
  | 'set_maintenance_agent_model'
  | 'set_task_agent_model'
  | 'set_session_caps'
  | 'register_group'
  | 'unregister_group'
  | 'promote_staging'
  | 'push_staged_to_branch'
  | 'tessl_update'
  | 'schedule_task_harness'
  | 'egress_allowlist_mutation'
  | 'file_delete_outside_group';

export interface ConfirmationToken {
  /** Random hex string the operator pastes into the agent. */
  token: string;
  /** Which destructive op this token authorizes. */
  scope: DestructiveScope;
  /** Operator-supplied prose for the audit log. */
  reason: string;
  /** ISO-8601 timestamp the token was issued. */
  issued_at: string;
  /** ISO-8601 timestamp after which the token is invalid. */
  expires_at: string;
  /** True after the hook consumes the token. Single-use. */
  used: boolean;
  /**
   * Optional chat scope — when set, the token is only valid for calls
   * originating in this chat. Future-extensible; v1 leaves it
   * unenforced (the host-side CLI may set it for telemetry).
   */
  chat_jid?: string;
}

export interface ConfirmationTokenFile {
  schema_version: 1;
  tokens: ConfirmationToken[];
}

export interface ConfirmationTokenFs {
  existsSync(p: string): boolean;
  readFileSync(p: string, enc: BufferEncoding): string | Buffer;
  writeFileSync(p: string, data: string, enc?: BufferEncoding): void;
  /**
   * Optional. If supplied, `saveConfirmationTokens` uses temp-file +
   * rename for an atomic write. If absent, falls back to a direct
   * writeFileSync (non-atomic). The host orchestrator and the
   * production hook always pass real `fs` which has `renameSync`;
   * tests may pass a minimal mock that omits it.
   */
  renameSync?(oldPath: string, newPath: string): void;
}

/**
 * Read the confirmation-token file. Returns an empty array if the file
 * doesn't exist — the absence of any tokens means "no destructive op
 * has been pre-authorized," and untrusted-provenance calls deny.
 *
 * Throws on JSON parse error or unexpected schema. Per `error-handling`
 * policy: configuration bugs surface, not silently-skipped reads.
 */
export function loadConfirmationTokens(
  fs: ConfirmationTokenFs,
  path: string,
): ConfirmationToken[] {
  if (!fs.existsSync(path)) return [];
  const rawValue = fs.readFileSync(path, 'utf-8');
  const raw =
    typeof rawValue === 'string' ? rawValue : rawValue.toString('utf-8');
  if (raw.trim().length === 0) return [];
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== 'object') {
    throw new TokenFileValidationError(
      `token file at ${path} did not parse to an object`,
    );
  }
  let rawTokens: unknown[];
  if (Array.isArray(parsed)) {
    // Tolerate legacy/manual shape — bare array. The CLI writes the
    // wrapped shape; reading both keeps mixed-version migrations
    // tolerable.
    rawTokens = parsed;
  } else {
    const tokens = (parsed as { tokens?: unknown }).tokens;
    if (!Array.isArray(tokens)) {
      throw new TokenFileValidationError(
        `token file at ${path} missing 'tokens' array`,
      );
    }
    rawTokens = tokens;
  }
  return rawTokens.map((t, idx) => validateTokenRecord(t, path, idx));
}

/**
 * Thrown by `loadConfirmationTokens` when the file's structural
 * top-level shape is wrong (not an object, missing `tokens` array,
 * etc.) or when an individual record fails validation. Distinct
 * subclass so the hook's catch can narrow to "expected: corrupt /
 * manually-edited file" without also catching unexpected SDK or
 * filesystem errors.
 */
export class TokenFileValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TokenFileValidationError';
  }
}

/**
 * Validate a single token record. Throws TokenFileValidationError on
 * any shape error so the gate fails CLOSED on corruption / manual-
 * edit typos rather than silently treating bogus data as valid (e.g.
 * `expires_at: 'zzzz'` lexicographically beats every real ISO
 * timestamp and would authorize a destructive op).
 */
function validateTokenRecord(
  raw: unknown,
  path: string,
  idx: number,
): ConfirmationToken {
  const where = `${path} tokens[${idx}]`;
  if (!raw || typeof raw !== 'object') {
    throw new TokenFileValidationError(`${where} is not an object`);
  }
  const r = raw as Record<string, unknown>;
  requireString(r, 'token', where);
  requireString(r, 'scope', where);
  requireString(r, 'reason', where);
  requireIsoTimestamp(r, 'issued_at', where);
  requireIsoTimestamp(r, 'expires_at', where);
  if (typeof r.used !== 'boolean') {
    throw new TokenFileValidationError(`${where}.used is not a boolean`);
  }
  if (r.chat_jid !== undefined && typeof r.chat_jid !== 'string') {
    throw new TokenFileValidationError(`${where}.chat_jid is not a string`);
  }
  return r as unknown as ConfirmationToken;
}

function requireString(
  r: Record<string, unknown>,
  key: string,
  where: string,
): void {
  if (typeof r[key] !== 'string' || (r[key] as string).length === 0) {
    throw new TokenFileValidationError(
      `${where}.${key} is not a non-empty string`,
    );
  }
}

function requireIsoTimestamp(
  r: Record<string, unknown>,
  key: string,
  where: string,
): void {
  const v = r[key];
  if (typeof v !== 'string') {
    throw new TokenFileValidationError(`${where}.${key} is not a string`);
  }
  // Date.parse returns NaN for invalid; an unparseable timestamp would
  // sort lexicographically and could appear to be in the future. Reject
  // hard.
  if (Number.isNaN(Date.parse(v))) {
    throw new TokenFileValidationError(
      `${where}.${key} is not a parseable ISO timestamp: ${v}`,
    );
  }
}

/**
 * Write the token file back atomically when the supplied `fs`
 * implements `renameSync` (production: real Node `fs`). Falls back to
 * a direct write when `renameSync` is absent (test mocks). Atomic
 * mode: write to `<path>.tmp.<pid>.<ms>`, then rename onto target —
 * a concurrent reader sees either the old file or the new file, never
 * a partial JSON.
 */
export function saveConfirmationTokens(
  fs: ConfirmationTokenFs,
  path: string,
  tokens: ConfirmationToken[],
): void {
  const file: ConfirmationTokenFile = {
    schema_version: 1,
    tokens,
  };
  const body = JSON.stringify(file, null, 2) + '\n';
  if (typeof fs.renameSync === 'function') {
    const tmp = `${path}.tmp.${process.pid}.${Date.now()}`;
    fs.writeFileSync(tmp, body, 'utf-8');
    fs.renameSync(tmp, path);
    return;
  }
  fs.writeFileSync(path, body, 'utf-8');
}

export interface TokenLookupResult {
  /** The matching token, or null if none. */
  token: ConfirmationToken | null;
  /** Reason the lookup failed (only populated when `token` is null). */
  reason?: 'no_tokens_for_scope' | 'all_tokens_expired' | 'all_tokens_used';
}

/**
 * Find the FIRST valid (unspent, unexpired, scope-matching) token in
 * the array. Returns a structured result so the caller can produce a
 * helpful deny reason.
 */
export function findValidToken(
  tokens: ReadonlyArray<ConfirmationToken>,
  scope: DestructiveScope,
  nowIso: string,
): TokenLookupResult {
  const scoped = tokens.filter((t) => t.scope === scope);
  if (scoped.length === 0)
    return { token: null, reason: 'no_tokens_for_scope' };
  const unspent = scoped.filter((t) => !t.used);
  if (unspent.length === 0) return { token: null, reason: 'all_tokens_used' };
  const valid = unspent.filter((t) => t.expires_at > nowIso);
  if (valid.length === 0) return { token: null, reason: 'all_tokens_expired' };
  return { token: valid[0] };
}

/**
 * Return a NEW array with the matching token marked `used: true`. The
 * input array is not mutated — the hook writes the new array back via
 * `saveConfirmationTokens`.
 */
export function markTokenUsed(
  tokens: ReadonlyArray<ConfirmationToken>,
  token: ConfirmationToken,
): ConfirmationToken[] {
  return tokens.map((t) =>
    t.token === token.token ? { ...t, used: true } : t,
  );
}

/**
 * Identify if a tool call is destructive and what scope it falls under.
 * Returns null for non-destructive calls (the hook returns no opinion).
 *
 * The mapping is intentionally explicit — adding a new destructive
 * tool flows through one entry, and the test suite can grep for the
 * full set.
 */
export function classifyDestructiveOp(
  toolName: string,
  _toolInput: unknown,
): { scope: DestructiveScope; label: string } | null {
  if (typeof toolName !== 'string') return null;

  // NanoClaw MCP tools — explicit name → scope mapping.
  const directMap: Record<string, { scope: DestructiveScope; label: string }> =
    {
      mcp__nanoclaw__nuke_session: {
        scope: 'nuke_session',
        label: 'kill the running container session',
      },
      mcp__nanoclaw__nuke_chat: {
        scope: 'nuke_chat',
        label: 'drop a chat registration + history',
      },
      mcp__nanoclaw__github_backup: {
        scope: 'github_backup',
        label: 'push host content to a GitHub backup repo',
      },
      mcp__nanoclaw__persist_global_file: {
        scope: 'persist_global_file',
        label: 'commit + push a global persona file to the deploy source',
      },
      mcp__nanoclaw__set_trusted: {
        scope: 'set_trusted',
        label: "change a group's trust tier",
      },
      mcp__nanoclaw__set_trigger: {
        scope: 'set_trigger',
        label: 'change how the agent activates in a chat',
      },
      mcp__nanoclaw__set_agent_model: {
        scope: 'set_agent_model',
        label: "pin a group's Claude model (cost impact)",
      },
      mcp__nanoclaw__set_maintenance_agent_model: {
        scope: 'set_maintenance_agent_model',
        label: 'pin the maintenance-session Claude model (cost impact)',
      },
      mcp__nanoclaw__set_task_agent_model: {
        scope: 'set_task_agent_model',
        label: "pin a scheduled task's Claude model (cost impact)",
      },
      mcp__nanoclaw__set_session_caps: {
        scope: 'set_session_caps',
        label: "change a group's session-length reset caps",
      },
      mcp__nanoclaw__register_group: {
        scope: 'register_group',
        label: 'register a new chat as a group',
      },
      mcp__nanoclaw__unregister_group: {
        scope: 'unregister_group',
        label: 'remove a group registration',
      },
      mcp__nanoclaw__promote_staging: {
        scope: 'promote_staging',
        label: 'promote staged content to a tile repo',
      },
      mcp__nanoclaw__push_staged_to_branch: {
        scope: 'push_staged_to_branch',
        label: 'push staged content to a tile-repo branch',
      },
      mcp__nanoclaw__tessl_update: {
        scope: 'tessl_update',
        label: 'update tessl tile registry',
      },
    };
  if (Object.prototype.hasOwnProperty.call(directMap, toolName)) {
    return directMap[toolName];
  }

  return null;
}

export type ConfirmationDecision =
  | { kind: 'pass'; reason: string }
  | { kind: 'allow'; reason: string }
  | {
      kind: 'allow_with_token';
      token: ConfirmationToken;
      updatedTokens: ConfirmationToken[];
    }
  | { kind: 'deny'; reason: string; scope: DestructiveScope; label: string };

/**
 * Top-level decision combining classification + provenance + token
 * lookup. Consumers (the hook) handle the side effects (writing the
 * updated token list back, denying the call, etc.).
 */
export function decideConfirmation(args: {
  toolName: string;
  toolInput: unknown;
  hasUntrustedProvenance: boolean;
  tokens: ReadonlyArray<ConfirmationToken>;
  nowIso: string;
}): ConfirmationDecision {
  const { toolName, toolInput, hasUntrustedProvenance, tokens, nowIso } = args;
  const classification = classifyDestructiveOp(toolName, toolInput);
  if (!classification) return { kind: 'pass', reason: 'tool is not gated' };

  if (!hasUntrustedProvenance) {
    return {
      kind: 'allow',
      reason:
        `operator-originated trusted chain — in-chat confirmation ` +
        `sufficient for ${classification.scope}`,
    };
  }

  const lookup = findValidToken(tokens, classification.scope, nowIso);
  if (!lookup.token) {
    return {
      kind: 'deny',
      scope: classification.scope,
      label: classification.label,
      reason: denyReason(classification, lookup.reason!),
    };
  }
  return {
    kind: 'allow_with_token',
    token: lookup.token,
    updatedTokens: markTokenUsed(tokens, lookup.token),
  };
}

function denyReason(
  c: { scope: DestructiveScope; label: string },
  reason: 'no_tokens_for_scope' | 'all_tokens_expired' | 'all_tokens_used',
): string {
  const baseDescription =
    `Destructive op '${c.scope}' (${c.label}) denied: the call ` +
    `chain has untrusted-provenance content and the model may be ` +
    `following injected instructions.`;
  const hint =
    `To allow this call, the operator runs:\n\n` +
    `    aye-confirm --scope ${c.scope} --reason '<short justification>' --ttl 5m\n\n` +
    `from a host-side terminal, then pastes the printed token into the ` +
    `chat (or the agent retries automatically once the token file is ` +
    `updated).`;
  switch (reason) {
    case 'no_tokens_for_scope':
      return `${baseDescription} No matching unspent token found.\n\n${hint}`;
    case 'all_tokens_used':
      return `${baseDescription} All matching tokens have already been consumed.\n\n${hint}`;
    case 'all_tokens_expired':
      return `${baseDescription} All matching tokens have expired (default TTL is 5m).\n\n${hint}`;
  }
}

/**
 * Parse a duration string like `5m`, `1h`, `30s`, `7d`. Returns the
 * delta in milliseconds. Used by the host-side CLI; exported here so
 * tests + future config readers don't reinvent the parsing.
 */
export function parseDuration(input: string): number {
  const m = /^(\d+)\s*(s|m|h|d)$/i.exec(input.trim());
  if (!m) {
    throw new Error(
      `invalid duration '${input}' — expected like '5m', '1h', '30s', '7d'`,
    );
  }
  const n = parseInt(m[1], 10);
  const unit = m[2].toLowerCase();
  const factor =
    unit === 's'
      ? 1000
      : unit === 'm'
        ? 60_000
        : unit === 'h'
          ? 3_600_000
          : unit === 'd'
            ? 86_400_000
            : 0;
  return n * factor;
}
