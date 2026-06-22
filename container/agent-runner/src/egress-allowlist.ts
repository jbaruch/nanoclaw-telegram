/**
 * #320 — Egress allowlist for outbound communication.
 *
 * Provenance-conditional gate that complements #322's capability ACL.
 * Where #322 says "what tools can the chain even reach," this module
 * says "for the outbound tools that ARE reached, who's an allowed
 * destination."
 *
 * Gating shape (per #318 design principle, mirrored from #322):
 *
 *   - Operator-originated trusted chain (no untrusted-provenance
 *     markers in walk-back span): bypass the allowlist by default.
 *     Operator may opt in via `enforce_for_operator: true` in the
 *     allowlist file.
 *   - Untrusted-provenance chain (any link is web/email/scraped/
 *     cross-group): allowlist enforced. Destination must match an
 *     entry; unmatched calls deny.
 *
 * Allowlist file lives at `/workspace/trusted/egress_allowlist.json`.
 * Owner-managed: filesystem perms restrict writes to host-side; the
 * destructive-op confirmation token from #324 is required to mutate
 * the file via the agent.
 *
 * Sinks gated by this module:
 *   - `mcp__composio__gmail_send_email` (and `gmail_send_*`, `gmail_reply_*`)
 *     → destination = `recipient` / `recipients` / `to`; checked against
 *       `gmail_send.allowed_recipients` (exact) AND `allowed_domains`
 *       (suffix match on `@<domain>`).
 *   - `mcp__composio__slack_post_message` / `slack_send_*` → destination =
 *     `channel`; checked against `slack_post.allowed_channels`.
 *   - `mcp__nanoclaw__send_message_to_chat` → destination = `chat_id`
 *     / `chat_jid`; checked against `send_message_to_chat.allowed_chat_jids`.
 *
 * Tools NOT in this map (e.g. `mcp__composio__github_create_issue`)
 * pass through this hook unchanged — #322's capability ACL is the
 * other line of defense.
 */

import * as path from 'path';

export interface EgressAllowlist {
  /**
   * Gmail send config. Recipients matched exactly; domains matched as
   * `@<domain>` suffix (case-insensitive).
   */
  gmail_send?: {
    allowed_recipients?: string[];
    allowed_domains?: string[];
  };
  /**
   * Slack post config. Channels matched exactly (case-insensitive on
   * the leading `#`-prefix variants — Slack itself is case-sensitive
   * on channel IDs but not on names; we accept both).
   */
  slack_post?: {
    allowed_channels?: string[];
  };
  /**
   * Cross-chat NanoClaw send. JIDs matched exactly.
   */
  send_message_to_chat?: {
    allowed_chat_jids?: string[];
  };
  /**
   * Operator opt-in tightening. When true, the allowlist applies even
   * to operator-originated trusted chains. Default false — trusted
   * operator keeps full reach.
   */
  enforce_for_operator?: boolean;
}

/**
 * Minimal `fs` shape this module needs. Compatible with both the real
 * Node `fs` module (whose `readFileSync` has overloaded signatures
 * returning `string | Buffer`) and synthetic mocks the tests pass in.
 */
export interface EgressAllowlistFs {
  existsSync(p: string): boolean;
  readFileSync(p: string, enc: BufferEncoding): string | Buffer;
}

/**
 * Synchronously read + parse + validate the egress allowlist. Returns
 * `null` if the file does not exist (treat as empty allowlist, deny
 * everything under untrusted-provenance).
 *
 * Validation: each per-sink field is checked for the expected shape
 * (string-array). Invalid fields are dropped with a `console.warn`
 * rather than thrown — an operator typo on `allowed_channels` (e.g.
 * a bare string instead of an array) shouldn't crash every outbound
 * call. The remaining valid entries still gate.
 *
 * Throws ONLY on JSON parse errors and the top-level "didn't parse to
 * an object" check. Configuration bugs at the structural level surface
 * loudly (per `error-handling` policy); per-field typos degrade
 * gracefully.
 */
export function loadEgressAllowlist(
  fs: EgressAllowlistFs,
  path: string,
): EgressAllowlist | null {
  if (!fs.existsSync(path)) return null;
  const rawValue = fs.readFileSync(path, 'utf-8');
  const raw =
    typeof rawValue === 'string' ? rawValue : rawValue.toString('utf-8');
  if (raw.trim().length === 0) return null;
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== 'object') {
    throw new Error(`egress allowlist at ${path} did not parse to an object`);
  }
  return validateAllowlist(parsed as Record<string, unknown>, path);
}

function validateAllowlist(
  raw: Record<string, unknown>,
  path: string,
): EgressAllowlist {
  const out: EgressAllowlist = {};

  const gmail = raw.gmail_send;
  if (gmail && typeof gmail === 'object') {
    const cfg: NonNullable<EgressAllowlist['gmail_send']> = {};
    const recipients = pickStringArray(
      gmail,
      'allowed_recipients',
      `${path} gmail_send`,
    );
    if (recipients) cfg.allowed_recipients = recipients;
    const domains = pickStringArray(
      gmail,
      'allowed_domains',
      `${path} gmail_send`,
    );
    if (domains) cfg.allowed_domains = domains;
    if (Object.keys(cfg).length > 0) out.gmail_send = cfg;
  }

  const slack = raw.slack_post;
  if (slack && typeof slack === 'object') {
    const channels = pickStringArray(
      slack,
      'allowed_channels',
      `${path} slack_post`,
    );
    if (channels) out.slack_post = { allowed_channels: channels };
  }

  const sm = raw.send_message_to_chat;
  if (sm && typeof sm === 'object') {
    const jids = pickStringArray(
      sm,
      'allowed_chat_jids',
      `${path} send_message_to_chat`,
    );
    if (jids) out.send_message_to_chat = { allowed_chat_jids: jids };
  }

  if (typeof raw.enforce_for_operator === 'boolean') {
    out.enforce_for_operator = raw.enforce_for_operator;
  } else if (raw.enforce_for_operator !== undefined) {
    console.warn(
      `egress allowlist: ${path}.enforce_for_operator is not boolean — ignoring`,
    );
  }

  return out;
}

function pickStringArray(
  obj: object,
  key: string,
  context: string,
): string[] | undefined {
  const v = (obj as Record<string, unknown>)[key];
  if (v === undefined) return undefined;
  if (!Array.isArray(v)) {
    console.warn(
      `egress allowlist: ${context}.${key} is not an array — ignoring`,
    );
    return undefined;
  }
  const valid = v.filter(
    (x): x is string => typeof x === 'string' && x.length > 0,
  );
  if (valid.length !== v.length) {
    console.warn(
      `egress allowlist: ${context}.${key} contained ${v.length - valid.length} non-string entries — using ${valid.length} valid`,
    );
  }
  return valid;
}

export type EgressDecision =
  | { kind: 'allow'; reason: string }
  | {
      kind: 'deny';
      reason: string;
      sink: string;
      destination: string;
    }
  | { kind: 'pass'; reason: string };

/**
 * Top-level decision. The hook calls this after walking the transcript
 * for provenance.
 *
 *   `kind: 'pass'` — the tool isn't an outbound sink we gate; the hook
 *   should return `{}` (no opinion).
 *   `kind: 'allow'` — explicit allow (operator-originated chain
 *   bypassing, or destination matched the allowlist).
 *   `kind: 'deny'` — blocked, with a reason naming sink + destination.
 */
export function decideEgress(args: {
  toolName: string;
  toolInput: unknown;
  hasUntrustedProvenance: boolean;
  allowlist: EgressAllowlist | null;
}): EgressDecision {
  const { toolName, toolInput, hasUntrustedProvenance, allowlist } = args;
  const sink = classifySink(toolName);
  if (!sink) return { kind: 'pass', reason: 'tool is not a gated sink' };

  const enforceForOperator = !!allowlist?.enforce_for_operator;
  if (!hasUntrustedProvenance && !enforceForOperator) {
    return {
      kind: 'allow',
      reason:
        'operator-originated trusted chain — allowlist bypassed by default',
    };
  }

  const destinations = extractDestinations(sink, toolInput);
  if (destinations.length === 0) {
    return {
      kind: 'deny',
      sink,
      destination: '<missing>',
      reason:
        `egress_allowlist: ${sink} call with no extractable destination — ` +
        `cannot verify against allowlist. The injection-driven path requires ` +
        `the operator to add the destination via 'aye-confirm' (#324).`,
    };
  }

  for (const dest of destinations) {
    if (!isDestinationAllowed(sink, dest, allowlist)) {
      return {
        kind: 'deny',
        sink,
        destination: dest,
        reason:
          `egress_allowlist: ${sink} to '${dest}' is not in ` +
          `/workspace/trusted/egress_allowlist.json. The current call ` +
          `chain contains untrusted-provenance content; the model may be ` +
          `following injected instructions. To allow this destination, ` +
          `the operator runs 'aye-confirm --reason ... --ttl 5m' (#324) ` +
          `and adds the entry to the allowlist via the main chat.`,
      };
    }
  }

  return {
    kind: 'allow',
    reason: `all destinations matched ${sink} allowlist`,
  };
}

export type GatedSink = 'gmail_send' | 'slack_post' | 'send_message_to_chat';

export function classifySink(toolName: string): GatedSink | null {
  if (typeof toolName !== 'string') return null;
  if (/^mcp__composio__gmail_(send|reply)\w*$/i.test(toolName)) {
    return 'gmail_send';
  }
  if (/^mcp__composio__slack_(send|post)\w*$/i.test(toolName)) {
    return 'slack_post';
  }
  if (toolName === 'mcp__nanoclaw__send_message_to_chat') {
    return 'send_message_to_chat';
  }
  return null;
}

/**
 * Pull the destination(s) out of the tool input. Returns a list because
 * gmail send commonly takes multiple recipients in one call — every
 * one must pass.
 */
export function extractDestinations(
  sink: GatedSink,
  toolInput: unknown,
): string[] {
  if (!toolInput || typeof toolInput !== 'object') return [];
  const input = toolInput as Record<string, unknown>;

  if (sink === 'gmail_send') {
    const fields = ['recipient', 'recipients', 'to', 'recipient_email'];
    const out: string[] = [];
    for (const f of fields) {
      const v = input[f];
      if (typeof v === 'string' && v.length > 0) {
        // Composio (and most mail APIs) accept comma- or semicolon-
        // separated lists in a single string field — `"a@x.io,b@y.io"`.
        // Without this split, `isDestinationAllowed` would evaluate
        // the combined string once and could incorrectly pass via the
        // `allowed_domains` suffix match on the LAST address, leaking
        // the earlier addresses through the gate. Split, trim, drop
        // empties; every parsed entry must pass independently.
        for (const part of splitRecipientList(v)) out.push(part);
      } else if (Array.isArray(v)) {
        for (const item of v) {
          if (typeof item === 'string' && item.length > 0) {
            // Same split applies to per-element strings inside an
            // array — some callers send `["a@x.io, b@y.io", "c@z.io"]`.
            for (const part of splitRecipientList(item)) out.push(part);
          }
        }
      }
    }
    return out;
  }

  if (sink === 'slack_post') {
    const v = input.channel;
    if (typeof v === 'string' && v.length > 0) return [v];
    return [];
  }

  if (sink === 'send_message_to_chat') {
    const v = input.chat_id ?? input.chat_jid;
    if (typeof v === 'string' && v.length > 0) return [v];
    return [];
  }

  return [];
}

/**
 * Decide whether a `Write` / `Edit` `file_path` argument targets the
 * allowlist file. Robust against bypass attempts that vary by:
 *   - basename equality (e.g. cwd is `/workspace/trusted` and the
 *     model passes a relative `egress_allowlist.json`),
 *   - path normalization (`/workspace/trusted/./egress_allowlist.json`,
 *     `/workspace/trusted/sub/../egress_allowlist.json`),
 *   - resolved-realpath equality (catches symlink redirection — the
 *     candidate path resolves through a symlink that ultimately points
 *     at the allowlist file).
 *
 * Tests cover each layer; CI catches if any one is removed.
 */
export function pathTargetsAllowlist(
  candidate: string,
  allowlistPath: string,
  fsRealpath: (p: string) => string | null = () => null,
  pathLib: {
    basename: (p: string) => string;
    isAbsolute: (p: string) => boolean;
    normalize: (p: string) => string;
    resolve: (...parts: string[]) => string;
  } = defaultPathLib(),
): boolean {
  if (typeof candidate !== 'string' || candidate.length === 0) return false;
  if (candidate === allowlistPath) return true;
  if (pathLib.basename(candidate) === pathLib.basename(allowlistPath)) {
    return true;
  }
  const resolved = pathLib.isAbsolute(candidate)
    ? pathLib.normalize(candidate)
    : pathLib.resolve(candidate);
  if (resolved === allowlistPath) return true;
  const candidateRealpath = fsRealpath(candidate);
  const allowlistRealpath = fsRealpath(allowlistPath);
  return (
    candidateRealpath !== null &&
    allowlistRealpath !== null &&
    candidateRealpath === allowlistRealpath
  );
}

/**
 * Decide whether a `Bash` command shell-string mentions the allowlist
 * file path or basename. Bash mention alone is enough — a poisoned
 * chain that wants to mutate the file via shell indirection can use
 * `cd /workspace/trusted && echo > egress_allowlist.json`, here-docs,
 * heredocs, redirections, etc. The basename match catches the cases
 * where the absolute path doesn't appear literally.
 *
 * Note: this is intentionally conservative (false-positive on harmless
 * `grep egress_allowlist.json …` reads). The file is operator-managed
 * and read-only-by-policy from the agent's perspective; reading via
 * `cat` is allowed via `Read`, no Bash detour needed.
 */
export function bashTargetsAllowlist(
  command: string,
  allowlistPath: string,
  pathLib: { basename: (p: string) => string } = defaultPathLib(),
): boolean {
  if (typeof command !== 'string' || command.length === 0) return false;
  if (command.includes(allowlistPath)) return true;
  return command.includes(pathLib.basename(allowlistPath));
}

function defaultPathLib() {
  return {
    basename: path.basename,
    isAbsolute: path.isAbsolute,
    normalize: path.normalize,
    resolve: (p: string) => path.resolve(p),
  };
}

/**
 * Split a string-valued recipient field into individual addresses.
 * Splits on `,` and `;` (the two separators every mail API accepts);
 * trims each entry and drops empties. Returns `[trimmed]` when no
 * separator is present (most calls).
 */
export function splitRecipientList(raw: string): string[] {
  if (typeof raw !== 'string' || raw.length === 0) return [];
  return raw
    .split(/[,;]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function isDestinationAllowed(
  sink: GatedSink,
  destination: string,
  allowlist: EgressAllowlist | null,
): boolean {
  if (!allowlist) return false;

  if (sink === 'gmail_send') {
    const cfg = allowlist.gmail_send;
    if (!cfg) return false;
    const lower = destination.toLowerCase();
    if (cfg.allowed_recipients) {
      for (const r of cfg.allowed_recipients) {
        if (r.toLowerCase() === lower) return true;
      }
    }
    if (cfg.allowed_domains) {
      for (const d of cfg.allowed_domains) {
        if (lower.endsWith('@' + d.toLowerCase())) return true;
      }
    }
    return false;
  }

  if (sink === 'slack_post') {
    const cfg = allowlist.slack_post;
    if (!cfg?.allowed_channels) return false;
    const lower = destination.toLowerCase();
    return cfg.allowed_channels.some((c) => c.toLowerCase() === lower);
  }

  if (sink === 'send_message_to_chat') {
    const cfg = allowlist.send_message_to_chat;
    if (!cfg?.allowed_chat_jids) return false;
    return cfg.allowed_chat_jids.includes(destination);
  }

  return false;
}
