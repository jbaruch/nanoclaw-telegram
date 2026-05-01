/**
 * #325 — Quarantine memory writes from sessions that touched
 * external content.
 *
 * `/workspace/trusted/` is the agent's long-term context — `MEMORY.md`,
 * daily logs, typed memory files. Anything written there becomes
 * future-session authority. Today the only gate on a write is the
 * model's own judgment, which means a session that processed external
 * content (a webpage, an email, a calendar invite) can launder an
 * injection straight into the trust boundary: poisoned text → daily
 * log → archived weekly summary → indistinguishable from a fact the
 * operator typed.
 *
 * This module's structural enforcement:
 *
 *   1. A per-`runQuery` flag tracks whether the session has processed
 *      external content. Any PostToolUse hook that emits a #321
 *      provenance marker (Encoding A wrap or Encoding B sentinel)
 *      flips it.
 *
 *   2. A PreToolUse hook on `Write` / `Edit` checks the flag when the
 *      target path is under `/workspace/trusted/`. If the flag is set,
 *      `Write` is REDIRECTED — the hook denies the original call AND
 *      writes the model's content to `/workspace/trusted/quarantine/
 *      <session-id>/<rel-path>` itself, so the structural redirect
 *      doesn't depend on agent cooperation.
 *
 *   3. `Edit` under quarantine condition denies hard — Edit's surgical
 *      semantics don't survive a redirect (the original file isn't
 *      modified; the quarantined snapshot doesn't exist yet). The
 *      deny reason instructs the model to issue a fresh `Write` to
 *      the quarantine path explicitly.
 *
 * The quarantine subtree itself (`/workspace/trusted/quarantine/`)
 * is exempt from the gate — once the operator has reviewed and
 * promoted a quarantined record, an explicit `Write` to that subtree
 * by the agent is fine (this case typically arrives via a host-side
 * promote flow rather than the agent, but the carve-out keeps
 * recovery paths from being blocked by their own gate).
 *
 * Out of scope for v1 (deferred to follow-ups):
 *   - Host-side notification when a quarantined record lands.
 *   - `nightly-housekeeping` exclusion of quarantined entries from
 *     weekly summaries.
 *   - Compaction summary exclusion (rolls up under #327).
 */

import * as path from 'path';

const TRUSTED_PREFIX = '/workspace/trusted/';
const QUARANTINE_PREFIX = '/workspace/trusted/quarantine/';
const CONTAINER_CWD = '/workspace/group';

/**
 * Per-runQuery flag. A single field today; the shape stays an object
 * so future fields (e.g. counters of provenance kinds, last-marker
 * timestamp) can be added without rewriting every call site.
 */
export interface QuarantineFlagState {
  processedExternalContent: boolean;
}

export function createQuarantineFlagState(): QuarantineFlagState {
  return { processedExternalContent: false };
}

/**
 * Resolve a path string the same way #321's classifyReadPath does so
 * that `../../etc/foo` doesn't slip past `/workspace/trusted/` matching:
 *   - relative inputs resolve against the container cwd
 *     (`/workspace/group`) — `notes.md` → `/workspace/group/notes.md`,
 *     not under trusted/.
 *   - absolute inputs normalize so `/workspace/trusted/../../etc/x`
 *     collapses to `/etc/x` and the prefix check fails.
 */
export function resolveTargetPath(filePath: string): string {
  return path.posix.isAbsolute(filePath)
    ? path.posix.normalize(filePath)
    : path.posix.resolve(CONTAINER_CWD, filePath);
}

/**
 * True when `filePath` resolves to a target inside `/workspace/
 * trusted/` AND outside the carved-out quarantine subtree. The
 * quarantine subtree is a deliberate exemption — the operator can
 * promote an entry by host-side rename, and an explicit Write into
 * that subtree (e.g. by a host-side agent) shouldn't be blocked by
 * its own gate.
 */
export function targetsTrustedMemory(filePath: string): boolean {
  const resolved = resolveTargetPath(filePath);
  if (!resolved.startsWith(TRUSTED_PREFIX)) return false;
  if (resolved.startsWith(QUARANTINE_PREFIX)) return false;
  return true;
}

/**
 * Map an in-trusted target to its per-session quarantine equivalent.
 * Preserves the relative path under trusted/ — `MEMORY.md` →
 * `quarantine/<sid>/MEMORY.md`, `daily/2026-04-30.md` →
 * `quarantine/<sid>/daily/2026-04-30.md`.
 *
 * Caller is expected to have validated `targetsTrustedMemory` first;
 * passing an out-of-tree path would silently produce a nonsense
 * quarantine target.
 */
export function quarantinePathFor(filePath: string, sessionId: string): string {
  const resolved = resolveTargetPath(filePath);
  const relUnderTrusted = resolved.slice(TRUSTED_PREFIX.length);
  // Sanitize sessionId: strip path-traversal characters so a
  // poisoned id (unlikely but defense-in-depth) can't escape the
  // quarantine subtree. Allow alphanumerics, hyphen, underscore.
  const safeSid = sessionId.replace(/[^a-zA-Z0-9_-]/g, '_');
  return path.posix.join(QUARANTINE_PREFIX, safeSid, relUnderTrusted);
}

export type QuarantineDecision =
  | { kind: 'allow' }
  | {
      kind: 'redirect';
      originalPath: string;
      quarantinedTo: string;
      reason: string;
    }
  | {
      kind: 'deny';
      originalPath: string;
      reason: string;
    };

export interface DecideMemoryWriteInput {
  toolName: string;
  filePath: string;
  flag: QuarantineFlagState;
  sessionId: string;
}

/**
 * Decide what to do with a Write/Edit call given the current flag
 * state and target path. Returns:
 *   - `allow` — call passes through unchanged. Path is outside
 *     trusted/, OR the session hasn't touched external content.
 *   - `redirect` — Write should be intercepted: the hook itself
 *     persists the content to `quarantinedTo` and denies the
 *     original call.
 *   - `deny` — Edit cannot be redirected coherently; reject with
 *     a message instructing the model to issue a fresh Write to
 *     the quarantine path.
 */
export function decideMemoryWrite(
  input: DecideMemoryWriteInput,
): QuarantineDecision {
  if (input.toolName !== 'Write' && input.toolName !== 'Edit') {
    return { kind: 'allow' };
  }
  if (!targetsTrustedMemory(input.filePath)) {
    return { kind: 'allow' };
  }
  if (!input.flag.processedExternalContent) {
    return { kind: 'allow' };
  }

  const resolvedOriginal = resolveTargetPath(input.filePath);
  const quarantinedTo = quarantinePathFor(input.filePath, input.sessionId);

  if (input.toolName === 'Edit') {
    return {
      kind: 'deny',
      originalPath: resolvedOriginal,
      reason:
        `memory_quarantine: this session has processed external content ` +
        `(web/email/calendar/file from outside the workspace). Edit on ` +
        `${resolvedOriginal} is denied because the surgical edit ` +
        `semantics don't survive a redirect to the quarantine subtree. ` +
        `If the change is intentional, issue a fresh \`Write\` to ` +
        `${quarantinedTo} containing the full file body — the operator ` +
        `will review and promote.`,
    };
  }

  return {
    kind: 'redirect',
    originalPath: resolvedOriginal,
    quarantinedTo,
    reason:
      `memory_quarantine: redirected Write on ${resolvedOriginal} to ` +
      `${quarantinedTo} because this session has processed external ` +
      `content. The original file is unchanged; the operator will review ` +
      `the quarantined snapshot and promote if intentional.`,
  };
}

/**
 * Detect whether a string contains any #321 / #29 provenance marker
 * — used by the UserPromptSubmit flag-flip path so a prompt that
 * carries `<untrusted-input source="cross-group:...">` (or any
 * Encoding A wrap) flips the per-runQuery flag at prompt-time, not
 * at tool-time. The sentinel (Encoding B) doesn't appear in user
 * prompts; the wrap is the only encoding the orchestrator places
 * in prompt bodies.
 *
 * The match is intentionally lax — any wrap counts, regardless of
 * source prefix, because the flag's purpose is "did this session
 * touch external content?" and any wrap is a yes.
 */
export function promptCarriesUntrustedInput(text: string): boolean {
  if (typeof text !== 'string' || text.length === 0) return false;
  return /<untrusted-input\s+source="[^"]+">/i.test(text);
}
