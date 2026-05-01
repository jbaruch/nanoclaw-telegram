/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF, like before)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json — polled and consumed
 *          Sentinel: /workspace/ipc/input/_close — signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent teams result).
 *   Final marker after loop ends signals completion.
 */

import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import {
  query,
  getSessionMessages,
  HookCallback,
  PostToolUseHookInput,
  PreCompactHookInput,
  PreToolUseHookInput,
  SessionMessage,
  SessionStartHookInput,
  StopHookInput,
  UserPromptSubmitHookInput,
} from '@anthropic-ai/claude-agent-sdk';
import {
  DEFAULT_TOOL_RESULT_MAX_BYTES,
  sanitizeToolResponse,
  shouldDenyTaskOutputBlock,
} from './poison-defense.js';
import { detectAuthoritativeLookup } from './authoritative-source.js';
import { evaluateBashCommand } from './bash-safety-net.js';
import { validateComposioArgs } from './composio-arg-validator.js';
import { detectComposioFidelity } from './composio-fidelity.js';
import { decideGroundTruthReminder } from './ground-truth-reminder.js';
import { detectLazyVerification } from './lazy-verification.js';
import { createReadonlyWarner } from './ipc-readonly-warn.js';
import { rewriteMarkdownToHtml } from './markdown-to-html.js';
import { isStaleSessionError } from './stale-session.js';
import {
  DEFAULT_HYGIENE_WINDOW_MS,
  decideHygieneCadence,
  extractHygieneSignatures,
} from './path-hygiene-cadence.js';
import { ReactToMessageIpcPayload, runReactFirstHook } from './react-first.js';
import {
  applyReplyThreadingDecision,
  createReplyThreadingState,
  decideReplyThreading,
  extractLatestInboundId,
} from './reply-threading.js';
import { composeAutoContext } from './session-start-context.js';
import {
  SilentTurnState,
  createSilentTurnState,
  decideSilentTurnAudit,
} from './silent-turn-audit.js';
import { buildSubagentRuleFilePaths } from './subagent-prompt.js';
import { wrapUntrustedInput } from './untrusted-input-sources.js';
import { wrapMcpToolResult } from './untrusted-input-wrap.js';
import {
  formatSentinel,
  inferSentinelSource,
} from './provenance-sentinel.js';
import {
  decideCapabilityAclIterable,
  walkBackIterableForProvenance,
  WalkBackMessage,
} from './capability-acl.js';
import {
  bashTargetsAllowlist,
  classifySink,
  decideEgress,
  loadEgressAllowlist,
  pathTargetsAllowlist,
} from './egress-allowlist.js';
import {
  classifyDestructiveOp,
  decideConfirmation,
  loadConfirmationTokens,
  saveConfirmationTokens,
  TokenFileValidationError,
} from './confirmation-tokens.js';
import { fileURLToPath } from 'url';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isTrusted?: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
  script?: string;
  replyToMessageId?: string;
  /**
   * Provenance of a scheduled task (undefined on non-scheduled runs).
   * Only `'untrusted_agent'` triggers the `<untrusted-input>` wrap below;
   * owner / main_agent / trusted_agent bypass. Mirrors the orchestrator-
   * side `ContainerInput.createdByRole` in `src/container-runner.ts` —
   * the security boundary is enforced there (orchestrator derives this
   * from the verified source-group trust tier at schedule_task time);
   * this side just honors the decision.
   */
  createdByRole?: 'owner' | 'main_agent' | 'trusted_agent' | 'untrusted_agent';
  /**
   * Which per-group session this container run belongs to. Mirrors the
   * orchestrator-side `ContainerInput.sessionName` in `src/container-runner.ts`.
   *
   * Consumed here to set the `NANOCLAW_SESSION_NAME` env var on the MCP
   * stdio server (see the `mcpServersConfig.nanoclaw.env` block below),
   * which stamps `sessionName` onto every TASKS_DIR IPC request so the
   * host responder routes `_script_result_*` replies back to THIS
   * session's `input-<session>/` dir. Mount-based session isolation
   * (`groupSessionsDir`, `input/` overlay) is set up by the orchestrator
   * before spawn; this value flows through to the MCP env at runtime.
   */
  sessionName?: string;
  /**
   * Whether the inbound batch is "addressed to us" — drives the
   * `react-first` 👀 gate (#289). Resolved orchestrator-side from
   * isMain / 1:1-DM / trigger-match / reply-to-our-bot, independent
   * of `requires_trigger`. `undefined` means "no signal" — for
   * non-channel paths (scheduled tasks, scripts) this is expected
   * and the existing skips (subagent / scheduled-task / etc.)
   * still gate the hook.
   */
  addressedToUs?: boolean;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
  streamText?: string;
  /**
   * Per-turn token usage from the most recent assistant message of
   * this query. Captured from the SDK message stream's
   * `message.usage` field (the Anthropic Messages API echoes
   * input_tokens / output_tokens / cache fields on every response).
   *
   * `input_tokens` is the size of the conversation context the model
   * saw on this turn, which the orchestrator's threshold-cross
   * detector reads as "current context size." Other fields are
   * surfaced for telemetry / forensic curves; the orchestrator does
   * not gate on them.
   *
   * Optional because non-assistant turns (errors, init-only messages)
   * may not carry a usage payload. Absent → telemetry skips this
   * turn; threshold detector reads it as below_warn (no trigger).
   */
  usage?: {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
  };
}

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;

/**
 * Effort levels the SDK's `query()` accepts (as of
 * `@anthropic-ai/claude-agent-sdk` 0.2.112). Kept here as a runtime
 * whitelist so a typo in `AGENT_EFFORT` doesn't propagate to the API
 * as a 400 — we fall back to the default and log.
 */
const VALID_AGENT_EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'] as const;
type AgentEffort = (typeof VALID_AGENT_EFFORTS)[number];
const DEFAULT_AGENT_EFFORT: AgentEffort = 'xhigh';

function resolveAgentEffort(raw: string | undefined): AgentEffort {
  if (!raw) return DEFAULT_AGENT_EFFORT;
  if ((VALID_AGENT_EFFORTS as readonly string[]).includes(raw)) {
    return raw as AgentEffort;
  }
  console.error(
    `[agent-runner] Invalid AGENT_EFFORT="${raw}" — falling back to ` +
      `"${DEFAULT_AGENT_EFFORT}". Valid values: ${VALID_AGENT_EFFORTS.join(', ')}.`,
  );
  return DEFAULT_AGENT_EFFORT;
}

/**
 * Push-based async iterable for streaming user messages to the SDK.
 * Keeps the iterable alive until end() is called, preventing isSingleUserTurn.
 */
class MessageStream {
  private queue: SDKUserMessage[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push({
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    });
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<SDKUserMessage> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>((r) => {
        this.waiting = r;
      });
      this.waiting = null;
    }
  }
}

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

/**
 * Observer-content gate. When `OBSERVER_ENABLED=1` is set on the
 * container env (the orchestrator sets it iff the host has
 * `OBSERVER_CHAT_JID` configured), the agent-runner emits the full
 * raw payloads the observer pipeline needs: thinking text, assistant
 * text, tool_use input JSON, tool_result preview. Those payloads can
 * carry credentials, tokens, and personal data copied by users into
 * messages — they are intentionally disclosed to the operator who
 * opted into observing. With the flag unset (default), the agent
 * logs metadata only (block counts, tool name, ok/error status, ids)
 * so `docker logs` post-mortems still work without leaking content.
 *
 * The gate is read once at module load: containers are short-lived
 * and the orchestrator's container-runner sets the env var per spawn
 * based on whether the host observer is enabled, so there's no need
 * to re-check at every log call.
 */
const OBSERVER_ENABLED = process.env.OBSERVER_ENABLED === '1';

function getSessionSummary(
  sessionId: string,
  transcriptPath: string,
): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');

  if (!fs.existsSync(indexPath)) {
    log(`Sessions index not found at ${indexPath}`);
    return null;
  }

  try {
    const index: SessionsIndex = JSON.parse(
      fs.readFileSync(indexPath, 'utf-8'),
    );
    const entry = index.entries.find((e) => e.sessionId === sessionId);
    if (entry?.summary) {
      return entry.summary;
    }
  } catch (err) {
    log(
      `Failed to read sessions index: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  return null;
}

/**
 * Archive the full transcript to conversations/ before compaction.
 */
function createPreCompactHook(assistantName?: string): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;

    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
      log('No transcript found for archiving');
      return {};
    }

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);

      if (messages.length === 0) {
        log('No messages to archive');
        return {};
      }

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();

      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      const filename = `${date}-${name}.md`;
      const filePath = path.join(conversationsDir, filename);

      const markdown = formatTranscriptMarkdown(
        messages,
        summary,
        assistantName,
      );
      fs.writeFileSync(filePath, markdown);

      log(`Archived conversation to ${filePath}`);
    } catch (err) {
      log(
        `Failed to archive transcript: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    return {};
  };
}

/**
 * #116 — Deny `TaskOutput` calls that would block on a sub-agent.
 * The SDK's `TaskOutput(block=true)` returns a raw chunk of the
 * sub-agent JSONL on timeout, leaking any high-entropy / invisible-
 * Unicode payload the sub-agent received from a noisy upstream
 * straight into this session's context. The deny path forces the main
 * session onto either non-blocking polling or the file-result pattern.
 */
function createTaskOutputBlockGateHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    if (pre.tool_name !== 'TaskOutput') {
      return {};
    }
    const decision = shouldDenyTaskOutputBlock(pre.tool_input);
    if (!decision.deny) {
      return {};
    }
    log(`PreToolUse: denied TaskOutput(block!=false) — ${decision.reason}`);
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        permissionDecision: 'deny' as const,
        permissionDecisionReason: decision.reason,
      },
    };
  };
}

/**
 * #117 — Sanitize MCP tool results before they reach the model. Strips
 * Cf-class invisible-Unicode characters and caps each text block at
 * `TOOL_RESULT_MAX_BYTES` (default 64 KiB).
 *
 * Scope: MCP tools only. The SDK's `updatedMCPToolOutput` field is the
 * only documented mutation surface for tool results post-fact, and it
 * is MCP-tool-scoped. Built-in tools (WebFetch, Bash) keep their raw
 * output — accepted gap; the triggering incident was Composio (MCP).
 */
function createMcpToolResultSanitizerHook(): HookCallback {
  const byteCap = (() => {
    const raw = process.env.TOOL_RESULT_MAX_BYTES;
    if (!raw) return DEFAULT_TOOL_RESULT_MAX_BYTES;
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      log(
        `Invalid TOOL_RESULT_MAX_BYTES=${raw}; falling back to ${DEFAULT_TOOL_RESULT_MAX_BYTES}`,
      );
      return DEFAULT_TOOL_RESULT_MAX_BYTES;
    }
    return parsed;
  })();

  return async (input, _toolUseId, _context) => {
    const post = input as PostToolUseHookInput;
    // Matcher already filters to mcp__*, but double-check defensively
    // — a misconfigured matcher would otherwise let `updatedMCPToolOutput`
    // silently no-op on non-MCP tools.
    if (!post.tool_name?.startsWith('mcp__')) {
      return {};
    }
    const { sanitized, stats } = sanitizeToolResponse(
      post.tool_response,
      byteCap,
    );
    if (stats.strippedBytes === 0 && stats.truncatedBytes === 0) {
      return {};
    }
    log(
      `tool_result_sanitizer tool=${post.tool_name} stripped_bytes=${stats.strippedBytes} truncated_bytes=${stats.truncatedBytes}`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PostToolUse' as const,
        updatedMCPToolOutput: sanitized,
      },
    };
  };
}

/**
 * #321 PR 2 — Wrap MCP read-tool results in `<untrusted-input
 * source="...">` so #322's walk-back can dispatch on source kind.
 *
 * Scope: same MCP-tools-only constraint as the sanitizer (#117) — the
 * SDK's `updatedMCPToolOutput` is the only documented mutation surface
 * for tool results, and it is MCP-scoped. Built-in tools get the sidecar
 * `additionalContext` sentinel from PR 4 instead.
 *
 * Order in the PostToolUse chain: sanitizer (#117) → fidelity (#140) →
 * this wrap. Sanitizer normalises bytes first so the wrap is applied to
 * the model-visible final form; fidelity inspects raw text before the
 * envelope is added so it doesn't mistake wrap tags for fabricated IDs.
 */
function createUntrustedInputWrapHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const post = input as PostToolUseHookInput;
    if (!post.tool_name?.startsWith('mcp__')) return {};
    const { wrapped, mutated } = wrapMcpToolResult(
      post.tool_name,
      post.tool_response,
    );
    if (!mutated) return {};
    log(`untrusted_input_wrap tool=${post.tool_name}`);
    return {
      hookSpecificOutput: {
        hookEventName: 'PostToolUse' as const,
        updatedMCPToolOutput: wrapped,
      },
    };
  };
}

/**
 * #321 PR 4 — Encoding B sidecar provenance sentinel for built-in
 * tools (`WebFetch`, `Read` on external paths, `Bash` invocations of
 * `agent-browser`). The SDK exposes no `tool_response` mutation
 * surface for built-in tools, so the literal `<untrusted-input>` wrap
 * (#321 PR 2 / Encoding A) is impossible here. Instead we emit a
 * machine-parseable marker line via `additionalContext` — #322's
 * walk-back recognises both encodings as equally authoritative.
 *
 * The hook delegates the decide-and-format logic to
 * `provenance-sentinel.ts`; this wrapper only adapts the SDK callback
 * shape and emits the marker when the source-inference yields a non-
 * null result.
 */
function createProvenanceSentinelHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const post = input as PostToolUseHookInput;
    const source = inferSentinelSource(post.tool_name, post.tool_input);
    if (!source) return {};
    const marker = formatSentinel(source, post.tool_use_id);
    // Log the prefix only — `source.value` carries the raw URL / file
    // path / agent-browser target, which can include signed-URL tokens,
    // sensitive mount paths, or other material that the no-secrets rule
    // forbids in any log level. The marker itself (delivered via
    // additionalContext to the model) carries the full value; logs stay
    // metadata-only.
    log(
      `provenance_sentinel tool=${post.tool_name} source_prefix=${source.prefix}`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PostToolUse' as const,
        additionalContext: marker,
      },
    };
  };
}

/**
 * Convert an SDK `SessionMessage` to the `WalkBackMessage` shape the
 * `capability-acl` walk-back operates on.
 *
 * Two transforms collapse here:
 *   1. The SDK message's content array (text, tool_use, tool_result,
 *      thinking, etc.) is flattened to a single concatenated string —
 *      both Encoding A wraps and Encoding B sentinels live in plain
 *      text fragments, so the walk-back is format-agnostic.
 *   2. A user-role message is classified as `isToolResult` if EVERY
 *      content block is a `tool_result` block. Mixed content (rare —
 *      models occasionally emit text alongside a tool_result) counts
 *      as a real user turn / boundary; better to over-include a
 *      boundary than to silently consume a real user message.
 */
function sessionMessageToWalkBack(m: SessionMessage): WalkBackMessage | null {
  const role = m.type;
  if (role !== 'user' && role !== 'assistant' && role !== 'system') {
    return null;
  }
  const inner = (m as { message?: unknown }).message;
  if (!inner || typeof inner !== 'object') {
    return { role, text: '' };
  }
  const content = (inner as { content?: unknown }).content;
  if (typeof content === 'string') {
    return { role, text: content };
  }
  if (!Array.isArray(content)) {
    return { role, text: '' };
  }
  let text = '';
  let allToolResult = content.length > 0;
  for (const block of content) {
    if (!block || typeof block !== 'object') {
      allToolResult = false;
      continue;
    }
    const t = (block as { type?: unknown }).type;
    if (t !== 'tool_result') allToolResult = false;
    // Pull text fields out of any block shape we've seen (text /
    // tool_use input / tool_result content / thinking). The walk-back
    // greps for marker patterns; concatenating with '\n' keeps line-
    // anchored regexes correct.
    const blockText = extractBlockText(block);
    if (blockText) text += (text ? '\n' : '') + blockText;
  }
  return {
    role,
    isToolResult: role === 'user' && allToolResult,
    text,
  };
}

function extractBlockText(block: unknown): string {
  if (!block || typeof block !== 'object') return '';
  const t = (block as { type?: unknown }).type;
  if (t === 'text' || t === 'thinking') {
    const s = (block as { text?: unknown }).text;
    return typeof s === 'string' ? s : '';
  }
  if (t === 'tool_result') {
    const c = (block as { content?: unknown }).content;
    if (typeof c === 'string') return c;
    if (Array.isArray(c)) {
      return c
        .map((b) => extractBlockText(b))
        .filter(Boolean)
        .join('\n');
    }
    return '';
  }
  if (t === 'tool_use') {
    // Tool_use inputs CAN carry strings the model is about to send
    // (e.g. send_message text). Including them in the walk-back text
    // would conflate planned-output with received-input — skip.
    return '';
  }
  return '';
}

/**
 * #322 — Capability ACL per data source. Walks back from the current
 * tool call through the session transcript, finds every untrusted-
 * provenance marker (Encoding A or B from #321) in the span between
 * the call and the most recent operator user-turn boundary, and denies
 * the call if its sink is not in the intersection of allowed-sinks for
 * every source seen.
 *
 * Fail-open ONLY for narrowly-expected transcript-unavailable failures
 * (ENOENT — file not written yet; SyntaxError — caught mid-flush during
 * a partial JSON write). Anything else propagates per
 * `jbaruch/coding-policy: error-handling` ("Catch specific exception
 * types, never bare catch-all handlers" + "Let unexpected exceptions
 * propagate — they indicate bugs that need fixing, not hiding").
 *
 * Why fail-open at all: pre-init and first-message sessions can fire
 * tool calls before the SDK has flushed the transcript file. Denying
 * every tool until the file is readable would brick fresh sessions.
 * Real injection attempts ride on stable markers that survive any
 * transient miss — the operationally-recoverable race doesn't unlock a
 * persistent attack vector. Unexpected errors (permissions misconfig,
 * SDK regression, OOM) ARE bugs and should surface, not silently allow.
 */
function createCapabilityAclHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    let session: SessionMessage[];
    try {
      session = await getSessionMessages(pre.session_id);
    } catch (err) {
      if (!isExpectedTranscriptUnavailable(err)) throw err;
      log(
        `capability_acl: transcript not yet readable, allowing tool=${pre.tool_name} reason=${
          err instanceof Error ? err.message : String(err)
        }`,
      );
      return {};
    }
    // Walk backward lazily — the iterator emits one converted
    // WalkBackMessage at a time, so the walk-back stops at the boundary
    // without paying the cost of converting older messages. On a long
    // session with a short span, this keeps the gate latency bounded by
    // span size rather than transcript size.
    const decision = decideCapabilityAclIterable(
      pre.tool_name,
      lazyReverseWalkBack(session),
    );
    if (decision.kind === 'allow') return {};
    log(
      `capability_acl: DENY tool=${pre.tool_name} prefixes=[${decision.prefixes.join(',')}]`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        permissionDecision: 'deny' as const,
        permissionDecisionReason: decision.reason,
      },
    };
  };
}

/**
 * Predicate for the narrow fail-open path in `createCapabilityAclHook`.
 * Only two failure shapes are operationally recoverable:
 *   - `ENOENT` — the SDK hasn't created the transcript file yet
 *     (fresh / pre-init session). Denying every call until the file
 *     exists would brick new sessions.
 *   - `SyntaxError` — the SDK is mid-flush and `getSessionMessages`
 *     read a partial JSONL line. Resolves on the next call.
 * Everything else (permission errors, SDK regressions, OOM, unknown)
 * is a bug or misconfiguration and propagates.
 */
function isExpectedTranscriptUnavailable(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  if (err instanceof SyntaxError) return true;
  const code = (err as NodeJS.ErrnoException).code;
  return code === 'ENOENT';
}

/**
 * Predicate for the narrow fail-handling path in
 * `createConfirmationTokenHook` when loading the token file.
 * Only two failure shapes are operationally expected:
 *   - `SyntaxError` — malformed JSON (operator typo, mid-flush partial)
 *   - `TokenFileValidationError` — token-record shape / type /
 *     timestamp validation rejected a record (intentional fail-closed
 *     branch from `validateTokenRecord`)
 * Everything else (EACCES, EIO, unexpected runtime errors) propagates
 * so a real bug or misconfiguration surfaces, per `error-handling`
 * policy.
 */
function isExpectedTokenLoadError(err: unknown): boolean {
  if (err instanceof SyntaxError) return true;
  if (err instanceof TokenFileValidationError) return true;
  return false;
}

/**
 * Predicate for the narrow fail-open path in
 * `createEgressAllowlistHook`. Same shape as
 * `isExpectedTranscriptUnavailable`: only `SyntaxError` (parse mid-flush
 * or operator typo) and `ENOENT` (TOCTOU between `existsSync` and
 * `readFileSync`) qualify. Permission errors, unexpected runtime
 * errors, etc. propagate so they surface as real bugs.
 */
function isExpectedAllowlistLoadError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  if (err instanceof SyntaxError) return true;
  const code = (err as NodeJS.ErrnoException).code;
  return code === 'ENOENT';
}

/**
 * Path of the egress allowlist file. Mounted RW into trusted/main
 * containers only (untrusted containers don't have this mount), so a
 * compromised untrusted-container session can't tamper with the file
 * via Write — the only path that can edit it is the operator from main
 * (per #324, mutations require the OOB confirmation token).
 */
const EGRESS_ALLOWLIST_PATH = '/workspace/trusted/egress_allowlist.json';

/**
 * #320 — Egress allowlist for outbound communication. Provenance-
 * conditional gate complementing #322's structural ACL: where #322
 * decides whether the chain can REACH an outbound tool at all, this
 * hook decides — for tools that ARE reached — whether the destination
 * is on the operator-managed allowlist.
 */
function createEgressAllowlistHook(fs: typeof import('fs')): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    if (classifySink(pre.tool_name) === null) return {};

    let session: SessionMessage[];
    try {
      session = await getSessionMessages(pre.session_id);
    } catch (err) {
      if (!isExpectedTranscriptUnavailable(err)) throw err;
      log(
        `egress_allowlist: transcript not yet readable, allowing tool=${pre.tool_name} reason=${
          err instanceof Error ? err.message : String(err)
        }`,
      );
      return {};
    }

    const prefixes = walkBackIterableForProvenance(
      lazyReverseWalkBack(session),
    );
    const hasUntrustedProvenance = prefixes.size > 0;

    let allowlist;
    try {
      allowlist = loadEgressAllowlist(fs, EGRESS_ALLOWLIST_PATH);
    } catch (err) {
      if (!isExpectedAllowlistLoadError(err)) throw err;
      log(
        `egress_allowlist: load failed at ${EGRESS_ALLOWLIST_PATH} — ${
          err instanceof Error ? err.message : String(err)
        }`,
      );
      if (!hasUntrustedProvenance) return {};
      return {
        hookSpecificOutput: {
          hookEventName: 'PreToolUse' as const,
          permissionDecision: 'deny' as const,
          permissionDecisionReason:
            `egress_allowlist: ${EGRESS_ALLOWLIST_PATH} could not be loaded ` +
            `and the call chain has untrusted-provenance content. The gate ` +
            `is failing closed. Operator: fix the allowlist JSON, then retry.`,
        },
      };
    }

    const decision = decideEgress({
      toolName: pre.tool_name,
      toolInput: pre.tool_input,
      hasUntrustedProvenance,
      allowlist,
    });

    if (decision.kind !== 'deny') return {};
    log(
      `egress_allowlist: DENY tool=${pre.tool_name} sink=${decision.sink} provenance_prefixes=[${[...prefixes].join(',')}]`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        permissionDecision: 'deny' as const,
        permissionDecisionReason: decision.reason,
      },
    };
  };
}

/**
 * Block agent-side mutations to the egress allowlist file. Same
 * structural protection as the confirmation-token write gate.
 */
function createEgressAllowlistWriteGate(): HookCallback {
  const safeRealpath = (p: string): string | null => {
    try {
      return fs.realpathSync(p);
    } catch (_err) {
      return null;
    }
  };
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    const targetsAllowlistFile = (() => {
      if (pre.tool_name === 'Write' || pre.tool_name === 'Edit') {
        const fp = (pre.tool_input as { file_path?: unknown })?.file_path;
        return (
          typeof fp === 'string' &&
          pathTargetsAllowlist(fp, EGRESS_ALLOWLIST_PATH, safeRealpath)
        );
      }
      if (pre.tool_name === 'Bash') {
        const cmd = (pre.tool_input as { command?: unknown })?.command;
        return (
          typeof cmd === 'string' &&
          bashTargetsAllowlist(cmd, EGRESS_ALLOWLIST_PATH)
        );
      }
      return false;
    })();
    if (!targetsAllowlistFile) return {};
    log(
      `egress_allowlist_write_gate: DENY tool=${pre.tool_name} — agent cannot mutate ${EGRESS_ALLOWLIST_PATH}`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        permissionDecision: 'deny' as const,
        permissionDecisionReason:
          `${EGRESS_ALLOWLIST_PATH} is operator-managed — the agent ` +
          `cannot mutate it directly. To add a destination, the operator ` +
          `edits the file from a host-side terminal.`,
      },
    };
  };
}

function* lazyReverseWalkBack(
  session: ReadonlyArray<SessionMessage>,
): IterableIterator<WalkBackMessage> {
  for (let i = session.length - 1; i >= 0; i--) {
    const wb = sessionMessageToWalkBack(session[i]);
    if (wb !== null) yield wb;
  }
}

/**
 * Confirmation-token file path. Mounted RW into trusted/main containers
 * (untrusted containers don't have `/workspace/trusted/`). Tampering by
 * the agent is structurally blocked by `createConfirmationTokenWriteGate`
 * below — Write/Edit on this path always denies, regardless of tier.
 */
const CONFIRMATION_TOKENS_PATH = '/workspace/trusted/aye_confirm_tokens.json';

/**
 * #324 — Destructive-op confirmation gate. Provenance-conditional:
 *
 *   - Operator-originated trusted chain → allow. The in-chat
 *     confirmation IS the confirmation; minting an OOB token for
 *     something the operator literally just typed is paperwork
 *     theatre that trains them to mint reflexively.
 *   - Untrusted-provenance chain → require an unspent, scope-matching,
 *     unexpired token issued via the host-side `aye-confirm` CLI.
 *     Token consumed on use; the file is rewritten with the new
 *     `used: true` flag.
 *
 * Fail-CLOSED on token-file load errors under untrusted provenance —
 * a corrupt token file under attack conditions must not silently widen
 * the gate. Operator-trusted chains pass through with a log when the
 * file is malformed (the operator will see the error).
 */
function createConfirmationTokenHook(fs: typeof import('fs')): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    const classification = classifyDestructiveOp(pre.tool_name, pre.tool_input);
    if (!classification) return {};

    let session: SessionMessage[];
    try {
      session = await getSessionMessages(pre.session_id);
    } catch (err) {
      if (!isExpectedTranscriptUnavailable(err)) throw err;
      // FAIL CLOSED for destructive ops when provenance can't be
      // determined. Allowing every destructive call during pre-init
      // / mid-flush windows would be a bypass for any chain that
      // could engineer a transcript-read race. The cost is one
      // failed retry on legitimate fresh sessions; the call succeeds
      // on the next attempt once the transcript is readable.
      const reason = err instanceof Error ? err.message : String(err);
      log(
        `confirmation_token: transcript not yet readable, denying tool=${pre.tool_name} reason=${reason}`,
      );
      return {
        hookSpecificOutput: {
          hookEventName: 'PreToolUse' as const,
          permissionDecision: 'deny' as const,
          permissionDecisionReason:
            `confirmation_token: cannot verify call-chain provenance for ` +
            `destructive tool '${pre.tool_name}' because the session ` +
            `transcript is not yet readable (${reason}). Failing closed; ` +
            `retry once the transcript is readable.`,
        },
      };
    }

    const prefixes = walkBackIterableForProvenance(
      lazyReverseWalkBack(session),
    );
    const hasUntrustedProvenance = prefixes.size > 0;

    let tokens;
    try {
      tokens = loadConfirmationTokens(fs, CONFIRMATION_TOKENS_PATH);
    } catch (err) {
      // Narrow catch to expected file-content failures only:
      //   - SyntaxError: malformed JSON (manual-edit typo or
      //     mid-flush partial)
      //   - TokenFileValidationError: shape / type / timestamp
      //     validation rejected a record
      // Anything else (EACCES, EIO, unexpected runtime errors, etc.)
      // propagates per `jbaruch/coding-policy: error-handling`.
      if (!isExpectedTokenLoadError(err)) throw err;
      log(
        `confirmation_token: token file load failed at ${CONFIRMATION_TOKENS_PATH} — ${
          err instanceof Error ? err.message : String(err)
        }`,
      );
      if (!hasUntrustedProvenance) return {};
      return {
        hookSpecificOutput: {
          hookEventName: 'PreToolUse' as const,
          permissionDecision: 'deny' as const,
          permissionDecisionReason:
            `confirmation_token: ${CONFIRMATION_TOKENS_PATH} could not be ` +
            `loaded and the call chain has untrusted-provenance content. ` +
            `The gate is failing closed. Operator: fix the token file JSON, ` +
            `or remint via 'aye-confirm', then retry.`,
        },
      };
    }

    const decision = decideConfirmation({
      toolName: pre.tool_name,
      toolInput: pre.tool_input,
      hasUntrustedProvenance,
      tokens,
      nowIso: new Date().toISOString(),
    });

    if (decision.kind === 'pass' || decision.kind === 'allow') return {};

    if (decision.kind === 'allow_with_token') {
      // Persist the consumed-token state IMMEDIATELY so an interleaved
      // call from another tool-use can't reuse the same token. The
      // write itself bypasses the write-gate by setting an env var
      // sentinel the gate checks (see createConfirmationTokenWriteGate).
      try {
        process.env.NANOCLAW_INTERNAL_TOKEN_WRITE = '1';
        saveConfirmationTokens(fs, CONFIRMATION_TOKENS_PATH, decision.updatedTokens);
      } finally {
        delete process.env.NANOCLAW_INTERNAL_TOKEN_WRITE;
      }
      // Log scope + outcome only — `decision.token.token` is secret
      // material. Even a 6-char prefix is token material per
      // `jbaruch/coding-policy: no-secrets` (no logging of secrets at
      // any level). The operator can correlate via the file's
      // `issued_at` if they need traceability.
      log(
        `confirmation_token: consumed scope=${decision.token.scope} expires_at=${decision.token.expires_at}`,
      );
      return {};
    }

    log(
      `confirmation_token: DENY tool=${pre.tool_name} scope=${decision.scope} provenance_prefixes=[${[...prefixes].join(',')}]`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        permissionDecision: 'deny' as const,
        permissionDecisionReason: decision.reason,
      },
    };
  };
}

/**
 * Block agent-side mutations to the confirmation-token file. The host
 * CLI is the only legitimate writer; the agent forging tokens via
 * Write/Edit/Bash would defeat the gate entirely.
 *
 * Detection layers — all three must miss for the call to pass:
 *   1. Absolute-path equality (Write/Edit `file_path` === target).
 *   2. Resolved-path equality after `path.resolve` from cwd
 *      (catches `Write file_path: 'aye_confirm_tokens.json'` when
 *      cwd is `/workspace/trusted`).
 *   3. Realpath equality (catches symlinks pointing at the target).
 *   4. Bash command containing the absolute path OR the basename
 *      `aye_confirm_tokens.json` (catches `cd /workspace/trusted &&
 *      echo > aye_confirm_tokens.json` and similar shell indirection).
 *
 * Bypass for legitimate consumption writes: the hook sets
 * `NANOCLAW_INTERNAL_TOKEN_WRITE=1` for the duration of its own
 * `saveConfirmationTokens` call. Tool-driven Write/Edit/Bash never
 * sets this var.
 */
function createConfirmationTokenWriteGate(): HookCallback {
  const tokenBasename = path.basename(CONFIRMATION_TOKENS_PATH);
  const safeRealpath = (p: string): string | null => {
    try {
      return fs.realpathSync(p);
    } catch (err) {
      // Narrow to the path-resolution errno cases that legitimately
      // mean "this path doesn't resolve" — the candidate doesn't
      // exist, isn't a directory mid-path, or has a symlink loop.
      // Anything else (EACCES on a real path, unexpected runtime
      // errors) propagates per `jbaruch/coding-policy: error-handling`.
      const code = (err as NodeJS.ErrnoException).code;
      if (code === 'ENOENT' || code === 'ENOTDIR' || code === 'ELOOP') {
        return null;
      }
      throw err;
    }
  };
  const tokenFileRealpath = safeRealpath(CONFIRMATION_TOKENS_PATH);

  function pathTargetsTokenFile(candidate: string): boolean {
    if (candidate === CONFIRMATION_TOKENS_PATH) return true;
    if (path.basename(candidate) === tokenBasename) return true;
    const resolved = path.isAbsolute(candidate)
      ? path.normalize(candidate)
      : path.resolve(candidate);
    if (resolved === CONFIRMATION_TOKENS_PATH) return true;
    const rp = safeRealpath(candidate);
    return (
      rp !== null && tokenFileRealpath !== null && rp === tokenFileRealpath
    );
  }

  return async (input, _toolUseId, _context) => {
    if (process.env.NANOCLAW_INTERNAL_TOKEN_WRITE === '1') return {};
    const pre = input as PreToolUseHookInput;
    const targetsTokenFile = (() => {
      if (pre.tool_name === 'Write' || pre.tool_name === 'Edit') {
        const fp = (pre.tool_input as { file_path?: unknown })?.file_path;
        return typeof fp === 'string' && pathTargetsTokenFile(fp);
      }
      if (pre.tool_name === 'Bash') {
        const cmd = (pre.tool_input as { command?: unknown })?.command;
        if (typeof cmd !== 'string') return false;
        return (
          cmd.includes(CONFIRMATION_TOKENS_PATH) ||
          cmd.includes(tokenBasename)
        );
      }
      return false;
    })();
    if (!targetsTokenFile) return {};
    log(
      `confirmation_token_write_gate: DENY tool=${pre.tool_name} — agent cannot mutate ${CONFIRMATION_TOKENS_PATH}`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        permissionDecision: 'deny' as const,
        permissionDecisionReason:
          `${CONFIRMATION_TOKENS_PATH} is host-only — the agent cannot ` +
          `mint, modify, or delete confirmation tokens. To authorize a ` +
          `destructive op, the operator runs 'aye-confirm' from a ` +
          `host-side terminal; the file then updates and the agent's ` +
          `next attempt at the gated tool consumes the matching token.`,
      },
    };
  };
}

/**
 * Path the silent-turn audit log is appended to. Mounted via the
 * orchestrator alongside other host-logs.
 */
const SILENT_TURN_LOG = '/workspace/host-logs/silent-turns.log';

/**
 * Extract the LAST `<message id="...">` from a UserPromptSubmit
 * prompt. Inlined here (rather than importing from reply-threading)
 * to keep #142 reviewable independently — duplicating two lines is
 * cheaper than coupling unrelated PRs.
 */
function extractTriggeringInboundIdForAudit(prompt: unknown): string | null {
  if (typeof prompt !== 'string' || prompt.length === 0) {
    return null;
  }
  const re = /<message\b[^>]*\bid="([^"]+)"/g;
  let last: string | null = null;
  let m: RegExpExecArray | null;
  while ((m = re.exec(prompt)) !== null) {
    last = m[1];
  }
  return last;
}

/**
 * #142 — stop-hook-end-of-turn-audit. The most insidious failure
 * mode: user sends a message, the agent runs a turn, exits — but
 * never reacts to or replies to the user's actual message. From the
 * user's view it looks like dropped silence.
 *
 * Three callbacks share an in-process state object:
 *  - `createSilentTurnPromptHook` — UserPromptSubmit seeds
 *    `triggeringInboundId` and resets the per-turn flags.
 *  - `createSilentTurnTrackingHook` — PreToolUse on
 *    `react_to_message` and `send_message` flips
 *    `reactedToInbound` / `repliedToInbound`.
 *  - `createSilentTurnStopHook` — Stop consults
 *    `decideSilentTurnAudit` and appends a JSONL entry to
 *    `/workspace/host-logs/silent-turns.log` on a silent turn.
 *
 * Observability only — never blocks the turn.
 */
function createSilentTurnPromptHook(state: SilentTurnState): HookCallback {
  return async (input, _toolUseId, _context) => {
    const submit = input as UserPromptSubmitHookInput;
    state.triggeringInboundId = extractTriggeringInboundIdForAudit(
      submit.prompt,
    );
    state.turnStartedAtMs = Date.now();
    state.reactedToInbound = false;
    state.repliedToInbound = false;
    state.anySendMessage = false;
    return {};
  };
}

function createSilentTurnTrackingHook(state: SilentTurnState): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    if (pre.tool_name === 'mcp__nanoclaw__react_to_message') {
      const args =
        (pre.tool_input as { messageId?: unknown } | undefined) ?? {};
      const explicitId =
        typeof args.messageId === 'string' ? args.messageId : null;
      // A reaction with no explicit id defaults to the most-recent
      // message in the chat — which is the triggering inbound. Treat
      // both shapes as "addressed".
      if (
        state.triggeringInboundId &&
        (explicitId === null || explicitId === state.triggeringInboundId)
      ) {
        state.reactedToInbound = true;
      }
    } else if (pre.tool_name === 'mcp__nanoclaw__send_message') {
      const args = (pre.tool_input as { reply_to?: unknown } | undefined) ?? {};
      state.anySendMessage = true;
      const replyTo = typeof args.reply_to === 'string' ? args.reply_to : null;
      if (state.triggeringInboundId && replyTo === state.triggeringInboundId) {
        state.repliedToInbound = true;
      }
    }
    return {};
  };
}

/**
 * #141 — session-start-auto-context. Inject MEMORY.md, RUNBOOK.md,
 * and the most-recent daily log into the session at startup so the
 * agent doesn't need to invoke the `tessl__trusted-memory` skill on
 * its own. Skill-based reads are advisory; under load the model
 * skips them. The hook makes the read deterministic.
 *
 * Skips when:
 *  - The session source is not `startup` — `resume` and `clear`
 *    sessions already have context, and `compact` sessions are
 *    handled by `PreCompact`.
 *  - The container has no named user-facing assistant (subagents,
 *    maintenance probes).
 *
 * Composition + truncation logic lives in `session-start-context.ts`
 * (SDK-free, vitest-covered).
 */
function createSessionStartAutoContextHook(
  containerInput: ContainerInput,
): HookCallback {
  return async (input, _toolUseId, _context) => {
    const start = input as SessionStartHookInput;
    if (start.source !== 'startup') {
      return {};
    }
    if (
      !containerInput.assistantName ||
      containerInput.assistantName.length === 0
    ) {
      log('SessionStart: auto-context skipped (no assistantName)');
      return {};
    }
    // Memory file path mirrors the orchestrator's mount layout. The
    // dash-prefixed dir name encodes the original `/workspace/group`
    // path the way Claude Code projects the dir under `~/.claude`.
    const memoryFile =
      '/home/node/.claude/projects/-workspace-group/memory/MEMORY.md';
    const runbookFile = '/workspace/group/RUNBOOK.md';
    const dailyLogDir = '/workspace/group/daily';
    // Kill-auto-compaction reentry (#104, design at
    // docs/proposals/kill-auto-compaction.md). The orchestrator
    // writes this file at threshold-cross before nuking the session;
    // on the next spawn the hook reads it and injects it as
    // additional context. When the file is missing (first-ever
    // spawn, no recent threshold-cross, or operator-cleared), the
    // composer silently skips this section.
    const checkpointFile = '/workspace/group/.checkpoints/default.md';
    const result = composeAutoContext({
      memoryFile,
      runbookFile,
      dailyLogDir,
      checkpointFile,
    });
    if (result.composed.length === 0) {
      log('SessionStart: auto-context found no source files');
      return {};
    }
    const foundLabels = result.sections
      .filter((s) => s.found)
      .map((s) => s.label)
      .join(',');
    log(`SessionStart: auto-context injected sections=${foundLabels}`);
    return {
      hookSpecificOutput: {
        hookEventName: 'SessionStart' as const,
        additionalContext: result.composed,
      },
    };
  };
}

/**
 * IPC outbox the MCP `send_message` / `react_to_message` tools write
 * into. The host watches this dir and dispatches to the channel.
 *
 * Mirrored verbatim from `ipc-mcp-stdio.ts`. Kept un-shared because the
 * MCP module is a separate stdio process that runs independently of the
 * agent-runner; touching its constants from here would couple two
 * processes that today exchange data only through filesystem paths.
 */
const IPC_MESSAGES_DIR = '/workspace/ipc/messages';

/**
 * Real fs-backed IPC writer used by `createReactFirstHook`. Same
 * shape the MCP `react_to_message` tool emits — see
 * `ipc-mcp-stdio.ts:writeIpcFile` — so the host's outbound
 * dispatcher consumes it transparently. Atomic via tempfile + rename.
 */
function writeReactToMessageIpc(payload: ReactToMessageIpcPayload): void {
  fs.mkdirSync(IPC_MESSAGES_DIR, { recursive: true });
  const filename = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`;
  const filepath = path.join(IPC_MESSAGES_DIR, filename);
  const tempPath = `${filepath}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(payload, null, 2));
  fs.renameSync(tempPath, filepath);
}

/**
 * #136 — react-first. Synthesise a `react_to_message` IPC call before
 * the model spends any tokens on the inbound. The agent's behaviour
 * rule is "react with an emoji to acknowledge — silence means
 * success"; without an enforced reaction, a healthy run that just
 * forgot to react reads to the user as "container down".
 *
 * Hook execution lives in `react-first.ts` (`runReactFirstHook`) so
 * the unit tests can exercise the full path — gate, payload shaping,
 * graceful IPC-failure handling — without spinning up the SDK or
 * touching the filesystem. This wrapper just plumbs container input
 * + the real fs-backed writer into the pure executor.
 */
function createReactFirstHook(containerInput: ContainerInput): HookCallback {
  return async (input, _toolUseId, _context) => {
    const submit = input as UserPromptSubmitHookInput;
    // Per-pipe values win over spawn-time — a freshly-piped
    // `@AyeAye` reply on a container originally spawned for
    // bystander traffic should flip the gate AND retarget the
    // reaction at the new piped message. Each falls back to its
    // spawn-time value when no pipe has updated it yet (first
    // prompt of the spawn).
    const effectiveAddressedToUs =
      latestPipedAddressedToUs ?? containerInput.addressedToUs;
    const effectiveMessageId =
      latestPipedReplyTo ?? containerInput.replyToMessageId;
    const result = runReactFirstHook(
      {
        isScheduledTask: containerInput.isScheduledTask === true,
        isSubagent:
          typeof submit.agent_id === 'string' && submit.agent_id.length > 0,
        prompt: typeof submit.prompt === 'string' ? submit.prompt : '',
        assistantName: containerInput.assistantName,
        addressedToUs: effectiveAddressedToUs,
        chatJid: containerInput.chatJid,
        groupFolder: containerInput.groupFolder,
        // Match the falsy-empty-string fallback used elsewhere
        // (`NANOCLAW_SESSION_NAME: containerInput.sessionName || 'default'`)
        // so an accidentally-empty `sessionName` doesn't get stamped
        // onto the IPC payload.
        sessionName: containerInput.sessionName || 'default',
        // The triggering inbound message ID — host reacts to THIS
        // specific message rather than the chat's latest, which could
        // have moved on between routing and hook fire.
        messageId: effectiveMessageId,
      },
      writeReactToMessageIpc,
    );
    switch (result.kind) {
      case 'skipped':
        log(`UserPromptSubmit: react-first skipped (${result.skipReason})`);
        break;
      case 'emitted':
        log(`UserPromptSubmit: react-first emitted ${result.emoji}`);
        break;
      case 'ipc-failed':
        // Don't block the prompt on an IPC-write failure — a missing
        // acknowledgement is bad, but a dropped prompt is worse.
        // `runReactFirstHook` already narrowed to expected
        // NodeJS.ErrnoException codes (other errors propagated).
        log(
          `UserPromptSubmit: react-first IPC write failed (${result.code}): ${result.message}`,
        );
        break;
    }
    return {};
  };
}

/**
 * #227 (tracks #214) — ground-truth-reminder. Inject a tight
 * verification reminder into `additionalContext` on every interactive
 * user turn. SOUL is re-injected via the system prompt every turn,
 * but the verification rules load once via the tile and fade in
 * salience over a long session — the asymmetry biases the model
 * toward shape over truth. This hook restores parity by re-injecting
 * ground-truth at the same per-turn cadence SOUL enjoys.
 *
 * Decision logic lives in `ground-truth-reminder.ts` so the unit
 * tests can exercise skip cases (sub-agent, scheduled task,
 * `[SCHEDULED TASK]` prompt wrap, no assistantName) without spinning
 * up the SDK.
 */
function createGroundTruthReminderHook(
  containerInput: ContainerInput,
): HookCallback {
  return async (input, _toolUseId, _context) => {
    const submit = input as UserPromptSubmitHookInput;
    const result = decideGroundTruthReminder({
      isSubagent:
        typeof submit.agent_id === 'string' && submit.agent_id.length > 0,
      isScheduledTask: containerInput.isScheduledTask === true,
      prompt: typeof submit.prompt === 'string' ? submit.prompt : '',
      assistantName: containerInput.assistantName,
    });
    if (!result.inject) {
      log(
        `UserPromptSubmit: ground-truth-reminder skipped (${result.skippedBy})`,
      );
      return {};
    }
    log('UserPromptSubmit: ground-truth-reminder injected');
    return {
      hookSpecificOutput: {
        hookEventName: 'UserPromptSubmit' as const,
        additionalContext: result.additionalContext,
      },
    };
  };
}

/**
 * #138 — no-markdown-in-send-message. Telegram (and the rest of the
 * channels routed through `send_message`) renders HTML only. The model
 * leaks Markdown — `**bold**`, `[label](url)`, `` `code` ``, `- `
 * bullets — especially under load or after compaction. This hook
 * auto-rewrites the four common patterns to HTML before delivery.
 *
 * Rewriting (rather than denying) is deliberate: a deny just makes
 * the model re-emit the same tokens and waste a turn. Code-block
 * regions (``` fences, &lt;pre&gt;, &lt;code&gt;) are passed through
 * bytewise so the agent can quote raw Markdown samples back at the
 * user without the hook mangling them.
 *
 * Matches `mcp__nanoclaw__send_message` (`text` field) and
 * `mcp__nanoclaw__send_file` (`caption` field). Both flow to the same
 * Telegram render path with the same constraints.
 */
function createNoMarkdownInSendMessageHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    const isSendMessage = pre.tool_name === 'mcp__nanoclaw__send_message';
    const isSendFile = pre.tool_name === 'mcp__nanoclaw__send_file';
    if (!isSendMessage && !isSendFile) {
      return {};
    }
    const toolInput = (pre.tool_input as Record<string, unknown>) ?? {};
    const fieldName = isSendMessage ? 'text' : 'caption';
    const original = toolInput[fieldName];
    const result = rewriteMarkdownToHtml(original);
    if (!result.changed) {
      return {};
    }
    log(
      `PreToolUse: markdown→html ${pre.tool_name} field=${fieldName} ` +
        `bold=${result.stats.bold} links=${result.stats.links} ` +
        `code=${result.stats.codeSpans} bullets=${result.stats.bulletLines}`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        updatedInput: { ...toolInput, [fieldName]: result.out },
      },
    };
  };
}

/**
 * #143 — bash-safety-net. Block known-destructive Bash commands at the
 * PreToolUse stage so a model under load can't talk itself into running
 * `rm -rf /`, force-push to main, mkfs, raw-disk dd, etc. The catalogue
 * + matching live in `bash-safety-net.ts` so they're unit-testable
 * without spinning up the SDK.
 */
function createBashSafetyNetHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    if (pre.tool_name !== 'Bash') {
      return {};
    }
    const command = (pre.tool_input as { command?: unknown } | undefined)
      ?.command;
    const decision = evaluateBashCommand(command);
    if (!decision.deny) {
      return {};
    }
    log(
      `PreToolUse: bash-safety-net denied Bash — rule=${decision.matched} reason=${decision.reason}`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        permissionDecision: 'deny' as const,
        permissionDecisionReason:
          decision.reason ?? 'denied by bash-safety-net',
      },
    };
  };
}

/**
 * #326 — Composio outbound argument validator. Checks per-field
 * constraints on the way out (header-injection guard on `subject`/
 * `to`/`cc`/`bcc`/`channel`/`user`, byte cap + control-char guard on
 * `body`/`text`) before the bytes leave the container. Tool-only —
 * does not look at provenance, because header injection and oversize
 * bodies are wrong regardless of who issued the call. Symmetric
 * counterpart to #117's incoming-result sanitizer.
 *
 * Logged-deny field is the field name (not the value) per `no-secrets`
 * — a body or recipient string can carry tokens or PII.
 */
function createComposioArgValidatorHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    const decision = validateComposioArgs(pre.tool_name, pre.tool_input);
    if (decision.kind === 'allow') return {};
    log(
      `PreToolUse: composio_arg_validator DENY tool=${pre.tool_name} field=${decision.field} violation=${decision.violation}`,
    );
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        permissionDecision: 'deny' as const,
        permissionDecisionReason: decision.reason,
      },
    };
  };
}

/**
 * #226 (tracks #214) — authoritative-source-nudge. Intercept entity-
 * lookup tool calls (Composio search/list, raw `SELECT FROM chats
 * LIMIT`, reads of the `available_groups.json` snapshot) and inject a
 * `systemMessage` pointing the agent at the canonical source. The
 * tool call is NOT denied — this is a nudge, because some entity
 * lookups legitimately need search when no pointer exists. Catalogue
 * + matching live in `authoritative-source.ts` so they're unit-
 * testable without spinning up the SDK.
 */
function createAuthoritativeSourceNudgeHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    const decision = detectAuthoritativeLookup(pre.tool_name, pre.tool_input);
    if (!decision.nudge) {
      return {};
    }
    log(
      `PreToolUse: authoritative-source nudge fired — tool=${pre.tool_name} entity=${decision.matched?.id}`,
    );
    return {
      systemMessage: decision.systemMessage,
    };
  };
}

/**
 * #137 — reply-threading-enforcement. Mid-turn `send_message` calls
 * that omit `reply_to` land at the bottom of an active chat instead
 * of as a quoted reply. In multi-user threads this looks like AyeAye
 * answering a different question or talking to itself.
 *
 * The pair of hooks below shares an in-process state object scoped to
 * one `runQuery()` call (one user-facing turn / chained MessageStream
 * sequence). UserPromptSubmit re-seeds the inner flags on every new
 * inbound, so the single state is safe across the chained turns the
 * MessageStream may carry within a single `runQuery()`.
 *  - `createReplyThreadingPromptHook(state)` runs on UserPromptSubmit
 *    and seeds `state.latestInboundId` from the prompt's last
 *    `<message id="...">` tag.
 *  - `createReplyThreadingPreToolHook(state, isMaintenanceSession)`
 *    runs on PreToolUse(send_message) and denies a standalone call
 *    while the latest inbound is unanswered.
 *
 * Single-turn enforcement only — cross-turn de-dup (the
 * "bot-already-spoke-since" multi-user carve-out) needs a SQL query
 * against the mounted `messages.db` and is intentionally deferred to
 * a follow-up. The single-turn check catches the recurring "standalone
 * mid-turn" bug class on its own.
 */
function createReplyThreadingPromptHook(
  state: ReturnType<typeof createReplyThreadingState>,
): HookCallback {
  return async (input, _toolUseId, _context) => {
    const submit = input as UserPromptSubmitHookInput;
    const inboundId = extractLatestInboundId(submit.prompt);
    state.latestInboundId = inboundId;
    state.repliedToInbound = false;
    if (inboundId) {
      log(
        `UserPromptSubmit: reply-threading seeded latestInboundId=${inboundId}`,
      );
    }
    return {};
  };
}

function createReplyThreadingPreToolHook(
  state: ReturnType<typeof createReplyThreadingState>,
  isMaintenanceSession: boolean,
): HookCallback {
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    const decision = decideReplyThreading({
      toolName: pre.tool_name,
      toolInput: pre.tool_input,
      isMaintenanceSession,
      state,
    });
    applyReplyThreadingDecision(state, decision);
    if (decision.kind !== 'deny') {
      return {};
    }
    log(`PreToolUse: reply-threading denied send_message — ${decision.reason}`);
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse' as const,
        permissionDecision: 'deny' as const,
        permissionDecisionReason: decision.reason,
      },
    };
  };
}

/**
 * #135 — lazy-verification-detector. The agent has a banned-excuse
 * catalogue ("site is JS-rendered", "page is thin", "can't access
 * this", etc.) — each one collapses the moment the agent launches a
 * real browser tool, hits a domain API, or runs code. Rules are
 * advisory; the model under load surfaces these excuses anyway.
 *
 * The hook fires on `Stop` (the agent is about to ship its turn-end
 * message). On detection, it returns `decision: 'block'` so the SDK
 * runs another turn with the injected reminder in scope. The
 * `last_assistant_message` field on `StopHookInput` carries the text
 * — no transcript-parse needed.
 *
 * Genuine-failure carve-out: when the agent enumerates real attempts
 * via the "Tried X — got Y; tried A — got B" shape, the message is
 * allowed through even if a banned phrase appears in it. That shape
 * is the rule-sanctioned way to report unverifiable.
 */
function createLazyVerificationDetectorHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const stop = input as StopHookInput;
    // Avoid re-blocking ourselves: if the SDK already triggered a
    // stop-hook re-run, `stop_hook_active` is true and re-blocking
    // would loop indefinitely.
    if (stop.stop_hook_active === true) {
      return {};
    }
    const decision = detectLazyVerification(stop.last_assistant_message);
    if (!decision.block) {
      return {};
    }
    log(
      `Stop: lazy-verification-detector blocked — phrases=${decision.matches.map((m) => m.phrase).join(',')}`,
    );
    return {
      decision: 'block' as const,
      reason: decision.reinjection,
      systemMessage: decision.reinjection,
    };
  };
}

/**
 * Path the fidelity audit log is appended to. Mounted via the
 * `host-logs` bind from the orchestrator (alongside the existing
 * agent-runner stderr destination). On a misconfigured container
 * where the dir doesn't exist, the hook silently best-efforts the
 * write — fabrication detection still fires; observability degrades.
 */
const FIDELITY_AUDIT_LOG = '/workspace/host-logs/fidelity-alerts.log';

/**
 * #140 — composio-fidelity. Background sub-agents that wrap Composio
 * tool calls sometimes return synthetic data with fabricated IDs
 * (sequential `email_01..email_18`, `pr1_notif`, `promo_001`) when
 * the upstream API hiccups. Pure text rules can't catch it — the
 * model has finished generating by the time the data lands.
 *
 * The hook fires on PostToolUse, scans the result text for known
 * fabrication signatures, appends a structured audit entry to
 * `/workspace/host-logs/fidelity-alerts.log`, and injects a
 * systemMessage flagging the result as untrusted so the agent
 * re-runs or treats it skeptically. We do NOT silently rewrite the
 * tool result — that would mask the failure mode from the agent and
 * downstream observability.
 */
function createComposioFidelityHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const post = input as PostToolUseHookInput;
    // Limit to MCP tools — Composio is the documented driver of the
    // bug. Built-in tools (Bash, Read, etc.) generate their own kinds
    // of output that would false-positive on the regexes (sequential
    // file numbering in `ls`, etc.).
    if (!post.tool_name?.startsWith('mcp__')) {
      return {};
    }
    const decision = detectComposioFidelity(post.tool_response);
    if (!decision.fabricated) {
      return {};
    }
    const auditEntry =
      JSON.stringify({
        ts: new Date().toISOString(),
        tool: post.tool_name,
        toolUseId: post.tool_use_id,
        sessionId: post.session_id,
        findings: decision.findings,
      }) + '\n';
    try {
      fs.mkdirSync(path.dirname(FIDELITY_AUDIT_LOG), { recursive: true });
      fs.appendFileSync(FIDELITY_AUDIT_LOG, auditEntry);
    } catch (err) {
      // Narrow to filesystem error class — the audit-log mount may be
      // missing on a misconfigured container or hit ENOSPC under load.
      // Both should degrade observability without breaking the
      // fidelity check itself. Other error types (TypeError on a
      // malformed `auditEntry`, etc.) propagate so they aren't
      // silently absorbed.
      const errno = err as NodeJS.ErrnoException;
      if (typeof errno.code !== 'string') {
        throw err;
      }
      log(
        `composio-fidelity: failed to append to audit log (${errno.code}): ${errno.message}`,
      );
    }
    log(
      `PostToolUse: composio-fidelity flagged ${post.tool_name} — rules=${decision.findings.map((f) => f.rule).join(',')}`,
    );
    return {
      systemMessage: decision.reinjection,
    };
  };
}

/**
 * #139 — path-hygiene-cadence. Heartbeat scans surface persistent
 * path-hygiene issues every tick. Without throttling the same
 * complaint reaches Baruch every ~30 min, which got memory'd as
 * "don't re-report same issue within ~4h"; the memory is advisory
 * and the model under load ignores it.
 *
 * The hook uses an in-process `Map<signature, lastReportedAtMs>`
 * scoped to a runQuery() lifetime, seeded at construction time from
 * the per-group daily logs (`/workspace/group/daily/<YYYY-MM-DD>.md`
 * for today and yesterday — covers the 4h window across midnight).
 * On every PreToolUse(send_message) we extract signatures, compare
 * against the map, and either deny (some signature seen <4h ago) or
 * pass and record the new reporting time.
 *
 * Carve-outs (mirroring #137):
 *  - `pin: true` bypasses (the user explicitly requested an update).
 *  - `reply_to` to a recent user message bypasses (responding to an
 *    explicit ask about hygiene).
 */
const HYGIENE_DAILY_LOG_DIR = '/workspace/group/daily';

function loadHygieneSignaturesFromDailyLogs(
  nowMs: number,
): Map<string, number> {
  const out = new Map<string, number>();
  if (!fs.existsSync(HYGIENE_DAILY_LOG_DIR)) {
    return out;
  }
  // Load today + yesterday — together they always cover the full 4h
  // window even when "now" sits just past midnight.
  const today = new Date(nowMs);
  const yesterday = new Date(nowMs - 24 * 60 * 60 * 1000);
  const filenames = [today, yesterday].map((d) => {
    const yyyy = d.getUTCFullYear();
    const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(d.getUTCDate()).padStart(2, '0');
    return path.join(HYGIENE_DAILY_LOG_DIR, `${yyyy}-${mm}-${dd}.md`);
  });
  for (const file of filenames) {
    if (!fs.existsSync(file)) continue;
    let body: string;
    let mtimeMs: number;
    try {
      body = fs.readFileSync(file, 'utf-8');
      mtimeMs = fs.statSync(file).mtimeMs;
    } catch (err) {
      const errno = err as NodeJS.ErrnoException;
      if (typeof errno.code === 'string') {
        log(
          `path-hygiene: failed to read/stat ${file} (${errno.code}): ${errno.message}`,
        );
        continue;
      }
      throw err;
    }
    for (const sig of extractHygieneSignatures(body)) {
      const prior = out.get(sig.signature);
      if (prior === undefined || prior < mtimeMs) {
        out.set(sig.signature, mtimeMs);
      }
    }
  }
  return out;
}

function createPathHygieneCadenceHook(): HookCallback {
  // State is constructed once per runQuery. Seeded from on-disk daily
  // logs so cross-container cadence works across container restarts —
  // the in-process Map then keeps the bookkeeping cheap for repeated
  // hits within the same turn.
  const seenAt = loadHygieneSignaturesFromDailyLogs(Date.now());
  return async (input, _toolUseId, _context) => {
    const pre = input as PreToolUseHookInput;
    if (pre.tool_name !== 'mcp__nanoclaw__send_message') {
      return {};
    }
    const args = (pre.tool_input as Record<string, unknown>) ?? {};
    if (args.pin === true) {
      return {};
    }
    if (typeof args.reply_to === 'string' && args.reply_to.length > 0) {
      // Responding to an explicit ask — let it through even on a
      // freshly reported signature.
      return {};
    }
    const text = typeof args.text === 'string' ? args.text : '';
    const nowMs = Date.now();
    const decision = decideHygieneCadence({
      text,
      lookupLastReportedAtMs: (sig) => seenAt.get(sig),
      nowMs,
      windowMs: DEFAULT_HYGIENE_WINDOW_MS,
    });
    if (decision.kind === 'deny') {
      log(
        `PreToolUse: path-hygiene-cadence denied — ${decision.suppressed.map((s) => s.signature).join(',')}`,
      );
      return {
        hookSpecificOutput: {
          hookEventName: 'PreToolUse' as const,
          permissionDecision: 'deny' as const,
          permissionDecisionReason: decision.reason,
        },
      };
    }
    // Pass: record each fresh signature so subsequent calls in the
    // same turn / container lifetime see it as "just reported".
    for (const sig of extractHygieneSignatures(text)) {
      seenAt.set(sig.signature, nowMs);
    }
    return {};
  };
}

/**
 * #142 — silent-turn-audit Stop hook. Observability only — never
 * blocks the turn. If neither a `react_to_message` to the triggering
 * inbound nor a `send_message` with `reply_to === triggeringInboundId`
 * landed, append a JSONL entry to /workspace/host-logs/silent-turns.log
 * for later investigation. Tracking state populated by the prompt
 * + pretool callbacks above.
 */
function createSilentTurnStopHook(
  state: SilentTurnState,
  containerInput: ContainerInput,
): HookCallback {
  return async (input, _toolUseId, _context) => {
    const stop = input as StopHookInput;
    const decision = decideSilentTurnAudit({
      isSubagent: typeof stop.agent_id === 'string' && stop.agent_id.length > 0,
      isMaintenanceSession: containerInput.sessionName === 'maintenance',
      isScheduledTask: containerInput.isScheduledTask === true,
      state,
    });
    if (decision.kind !== 'log') {
      return {};
    }
    const auditEntry =
      JSON.stringify({
        ts: new Date().toISOString(),
        chatJid: containerInput.chatJid,
        groupFolder: containerInput.groupFolder,
        sessionName: containerInput.sessionName,
        sessionId: stop.session_id,
        ...decision.record,
      }) + '\n';
    let appendOk = true;
    try {
      fs.mkdirSync(path.dirname(SILENT_TURN_LOG), { recursive: true });
      fs.appendFileSync(SILENT_TURN_LOG, auditEntry);
    } catch (err) {
      // Narrow to NodeJS.ErrnoException — host-log mount may be
      // missing or full. Anything else (TypeError on a malformed
      // entry, etc.) is a programming bug and propagates.
      const errno = err as NodeJS.ErrnoException;
      if (typeof errno.code !== 'string') {
        throw err;
      }
      appendOk = false;
      log(
        `silent-turn-audit: failed to append to log (${errno.code}): ${errno.message}`,
      );
    }
    // Only log success when the append actually landed — without
    // this gate the log line printed even after an EACCES, which
    // misled investigations into thinking the entry was on disk.
    if (appendOk) {
      log(
        `Stop: silent-turn-audit logged silent turn — inbound=${decision.record.triggeringInboundId}`,
      );
    }
    return {};
  };
}

function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

function generateFallbackName(): string {
  const time = new Date();
  return `conversation-${time.getHours().toString().padStart(2, '0')}${time.getMinutes().toString().padStart(2, '0')}`;
}

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text =
          typeof entry.message.content === 'string'
            ? entry.message.content
            : entry.message.content
                .map((c: { text?: string }) => c.text || '')
                .join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        const text = textParts.join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {}
  }

  return messages;
}

function formatTranscriptMarkdown(
  messages: ParsedMessage[],
  title?: string | null,
  assistantName?: string,
): string {
  const now = new Date();
  const formatDateTime = (d: Date) =>
    d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : assistantName || 'Assistant';
    const content =
      msg.content.length > 2000
        ? msg.content.slice(0, 2000) + '...'
        : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Check for _close sentinel.
 */
function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try {
      fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL);
    } catch {
      /* ignore */
    }
    return true;
  }
  return false;
}

/**
 * Drain all pending IPC input messages.
 * Returns messages found, or empty array.
 * Tracks consumed files in memory so read-only mounts don't cause infinite loops.
 */
const REPLY_TO_FILE = path.join(IPC_INPUT_DIR, '_reply_to');
const consumedInputFiles = new Set<string>();

/**
 * Latest per-pipe `addressedToUs` flag, updated by `drainIpcInput`
 * each time the host writes a new IPC message into the input dir.
 * The react-first hook reads this at fire time so a freshly-piped
 * inbound that addresses us flips a previously-spawned-for-bystander
 * container into "👀 fires." `undefined` means "no signal seen yet"
 * — the hook then falls back to `containerInput.addressedToUs` (the
 * spawn-time value).
 */
let latestPipedAddressedToUs: boolean | undefined;

/**
 * Latest per-pipe `replyToMessageId` — the ID of the most recent
 * inbound piped to this container. The react-first hook stamps this
 * on the IPC payload so the host reacts to the PIPED trigger, not
 * the spawn-time one. Without this, a piped `@AyeAye` reply would
 * still fire 👀 (the addressed-ness gate flips correctly per the
 * fix above), but the reaction would land on the spawn-time message
 * rather than the new addressed pipe.
 */
let latestPipedReplyTo: string | undefined;
// EROFS/EACCES on the input dir means the IPC mount is read-only — by
// design, for untrusted containers (issue #287). The agent itself can
// never recover from this; the host-side sweep is responsible. But
// silently swallowing the error for two days is what hid the original
// backlog bug — surface the FIRST occurrence loudly so the next
// regression of this class is visible at a glance in container logs.
const readonlyWarner = createReadonlyWarner(log);

function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs
      .readdirSync(IPC_INPUT_DIR)
      .filter(
        (f) =>
          f.endsWith('.json') &&
          !f.startsWith('_script_result_') &&
          !consumedInputFiles.has(f),
      )
      .sort();

    const messages: string[] = [];
    let latestReplyTo: string | undefined;
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        consumedInputFiles.add(file);
        try {
          fs.unlinkSync(filePath);
        } catch (e: any) {
          // ENOENT — file vanished between readFileSync and unlink
          // (host-side sweep ran, or another race). Benign; matches the
          // error-path catch below.
          if (e.code !== 'EROFS' && e.code !== 'EACCES' && e.code !== 'ENOENT')
            throw e;
          if (e.code === 'EROFS' || e.code === 'EACCES') {
            readonlyWarner.warn(e.code, file);
          }
        }
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
          if (data.replyToMessageId) {
            latestReplyTo = data.replyToMessageId;
            latestPipedReplyTo = data.replyToMessageId;
          }
          // Per-pipe addressed-ness for the react-first hook.
          // Latest-wins is fine for the failure mode this fixes
          // (an `@AyeAye` reply piped into a bystander-spawned
          // container): the addressed pipe sets the flag true and
          // the next UserPromptSubmit reads true. A subsequent
          // non-addressed pipe would flip back to false; that's
          // also correct — the agent can still reason about the
          // message via observer's commit-gated emojis.
          if (typeof data.addressedToUs === 'boolean') {
            latestPipedAddressedToUs = data.addressedToUs;
          }
        }
      } catch (err) {
        log(
          `Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`,
        );
        consumedInputFiles.add(file);
        try {
          fs.unlinkSync(filePath);
        } catch (e: any) {
          if (e.code !== 'EROFS' && e.code !== 'EACCES' && e.code !== 'ENOENT')
            throw e;
          if (e.code === 'EROFS' || e.code === 'EACCES') {
            readonlyWarner.warn(e.code, file);
          }
        }
      }
    }
    // Write the latest replyToMessageId so the MCP server can pick it up
    if (latestReplyTo) {
      try {
        fs.writeFileSync(REPLY_TO_FILE, latestReplyTo);
      } catch {
        /* ignore */
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

/**
 * Wait for a new IPC message or _close sentinel.
 * Returns the messages as a single string, or null if _close.
 */
function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

/**
 * Run a single query and stream results via writeOutput.
 * Uses MessageStream (AsyncIterable) to keep isSingleUserTurn=false,
 * allowing agent teams subagents to run to completion.
 * Also pipes IPC messages into the stream during the query.
 */
async function runQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
): Promise<{
  newSessionId?: string;
  lastAssistantUuid?: string;
  closedDuringQuery: boolean;
  errorResult: boolean;
}> {
  const stream = new MessageStream();
  stream.push(prompt);
  let sawErrorResult = false;

  // #137 — single-turn reply-threading state shared by the
  // UserPromptSubmit hook (which seeds the latest inbound id) and the
  // PreToolUse hook (which gates standalone send_message calls).
  // Lifetime: one runQuery() call. The UserPromptSubmit hook resets
  // the inner `repliedToInbound` flag on every new inbound, so the
  // state is safe to re-use across the chained turns a single
  // MessageStream may carry.
  const replyThreadingState = createReplyThreadingState();
  // Seed from the initial prompt up-front so the very first
  // send_message of the turn is gated even before UserPromptSubmit
  // fires (the SDK fires the prompt-submit hook AFTER the prompt is
  // accepted, but a model can in principle emit a tool call before
  // that — the seed makes the gate strict from t=0).
  replyThreadingState.latestInboundId = extractLatestInboundId(prompt);
  const isMaintenanceSession = containerInput.sessionName === 'maintenance';

  // #142 — silent-turn-audit state shared by the UserPromptSubmit /
  // PreToolUse / Stop hooks. Lifetime: one runQuery() call.
  // UserPromptSubmit resets the per-turn flags, so this single state
  // safely handles the chained turns a MessageStream may carry.
  const silentTurnState = createSilentTurnState(Date.now());
  silentTurnState.triggeringInboundId =
    extractTriggeringInboundIdForAudit(prompt);

  // Observability: emit a structured "query started" stderr line so the
  // optional observer channel (src/observer.ts) can detect query
  // boundaries and arm its watchdog. Emits explicit metadata fields
  // (length, the resolved target message id, scheduled-task flag) —
  // never the raw prompt body, which can contain user-pasted
  // credentials, API keys, auth links, and personal data per
  // jbaruch/coding-policy: no-secrets. The explicit
  // `target_message_id` field also fixes the previous "preview slice
  // didn't include the trailing <message id="…">" race that pinned
  // observer reactions to an earlier message.
  const queryStartedAt = Date.now();
  const targetMessageId = replyThreadingState.latestInboundId ?? '-';
  const isScheduledTask = containerInput.isScheduledTask === true;
  // #289 — emit the per-query addressed-ness signal so observer.ts
  // can suppress its progress-emoji ladder (🤔/⚡/✍/watchdog blinks)
  // for non-addressed inbounds in `requires_trigger=false` rooms.
  // Without this, the agent reasons about every message and the
  // observer's commit-gate fires a full progress sequence on
  // bystander chatter — the exact noise the gate was meant to kill.
  // Per-pipe `latestPipedAddressedToUs` wins over spawn-time, falls
  // back to `containerInput.addressedToUs`. `-` is the "no signal"
  // sentinel (legacy paths, scheduled tasks, callers that don't
  // pass the flag) — observer parses it as `addressed=undefined`
  // and falls through to the engagement-gate alone, preserving
  // pre-#289 behavior so legacy / non-channel paths keep their
  // reactions when they commit.
  const queryAddressed =
    latestPipedAddressedToUs ?? containerInput.addressedToUs;
  const addressedField =
    typeof queryAddressed === 'boolean' ? String(queryAddressed) : '-';
  log(
    `Query input: ${prompt.length} chars, target_message_id=${targetMessageId}, scheduled_task=${isScheduledTask}, addressed=${addressedField}`,
  );

  // Poll IPC for the _close sentinel during the query. We deliberately do
  // NOT drain JSON message files here — there's a race where pollIpc fires
  // after the SDK has emitted Result and agent-runner has broken out of
  // the for-await over responses, but BEFORE runQuery returns and
  // ipcPolling flips to false. In that window, draining a file consumes
  // it from disk (delete + consumedInputFiles entry) and pushes it into a
  // stream the SDK has stopped reading from — message is silently lost.
  // Mid-query draining is also unnecessary: the SDK conversation is
  // turn-based, and adding a second user message mid-turn doesn't get
  // processed until after the current turn ends anyway. Letting
  // waitForIpcMessage drain between queries gives the same throughput
  // with deterministic delivery — files only disappear when their content
  // has been read into a string that becomes the next runQuery's prompt.
  let ipcPolling = true;
  let closedDuringQuery = false;
  const pollIpcDuringQuery = () => {
    if (!ipcPolling) return;
    if (shouldClose()) {
      log('Close sentinel detected during query, ending stream');
      closedDuringQuery = true;
      stream.end();
      ipcPolling = false;
      return;
    }
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
  };
  setTimeout(pollIpcDuringQuery, IPC_POLL_MS);

  let newSessionId: string | undefined;
  let lastAssistantUuid: string | undefined;
  let messageCount = 0;
  let resultCount = 0;
  // Track whether the agent invoked an explicit user-facing send tool
  // AND the tool actually succeeded during this query. If so, the SDK's
  // final `result.text` is a closing-thought / summary aimed at the
  // harness, not a second answer to the user — forwarding it produces
  // visible duplicates ("Awake, bud" + "Confirmed."). We require a
  // successful tool_result (not is_error) so a hook-denied or errored
  // send_message doesn't suppress the final text and leave the user
  // staring at silence.
  const pendingUserFacingToolUseIds = new Set<string>();
  let userFacingSendSucceeded = false;

  // Streaming preview: accumulate assistant text and emit throttled
  let streamingTextAccum = '';
  let lastStreamEmit = 0;
  const STREAM_THROTTLE_MS = 300;

  // Most-recent per-turn token usage from the SDK message stream.
  // Pinned onto every writeOutput payload (stream + final result) so
  // the orchestrator can drive the kill-auto-compaction telemetry +
  // threshold detector. See ContainerOutput.usage docstring for why
  // it's optional. Carries across turns within a single runQuery
  // invocation; a fresh runQuery starts undefined.
  let latestUsage: ContainerOutput['usage'] | undefined;

  // Load SOUL.md and FORMATTING.md into systemPrompt.append so they survive
  // compaction. The SDK re-injects system prompt content every turn — behavioral
  // instructions placed here won't drift after long conversations or compaction.
  // NOTE: /workspace/global/SOUL.md resolves to the correct file per trust tier —
  // trusted containers mount the full SOUL.md, untrusted mount SOUL-untrusted.md
  // at the same path. No trust check needed here; the mount layer handles it.
  // Per-group CLAUDE.md is no longer loaded here (it's a thin trust-marker
  // + @import pointer post-#153) — the imported targets (SOUL/FORMATTING)
  // are loaded directly so they make it into the persistent system prompt.
  const soulMdPath = '/workspace/global/SOUL.md';
  const formattingMdPath = '/workspace/global/FORMATTING.md';
  const appendParts: string[] = [];
  if (fs.existsSync(soulMdPath)) {
    appendParts.push(fs.readFileSync(soulMdPath, 'utf-8'));
  }
  if (fs.existsSync(formattingMdPath)) {
    appendParts.push(fs.readFileSync(formattingMdPath, 'utf-8'));
  }
  const systemPromptAppend =
    appendParts.length > 0 ? appendParts.join('\n\n---\n\n') : undefined;

  // Rules are loaded by the SDK via the tessl chain: CLAUDE.md → AGENTS.md → .tessl/RULES.md
  // For untrusted groups, the orchestrator copies .tessl from a main group's session.

  // Discover additional directories mounted at /workspace/extra/*
  // These are passed to the SDK so their CLAUDE.md files are loaded automatically
  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) {
        extraDirs.push(fullPath);
      }
    }
  }
  if (extraDirs.length > 0) {
    log(`Additional directories: ${extraDirs.join(', ')}`);
  }

  // Discover installed skill names for subagent definitions.
  // Subagents spawned via TeamCreate don't inherit the parent's skills
  // or settingSources — they only get what's explicitly defined here.
  const skillsDir = '/home/node/.claude/skills';
  const installedSkills: string[] = [];
  if (fs.existsSync(skillsDir)) {
    for (const entry of fs.readdirSync(skillsDir)) {
      if (fs.statSync(path.join(skillsDir, entry)).isDirectory()) {
        installedSkills.push(entry);
      }
    }
  }
  if (installedSkills.length > 0) {
    log(`Discovered ${installedSkills.length} skills for subagent definitions`);
  }

  // MCP servers config — shared between main agent and subagents.
  //
  // `ToolSearch` is in `allowedTools` (below) which enables deferred MCP
  // tool loading: schemas for unmarked servers stay out of the prompt and
  // the agent discovers them on demand via ToolSearch. `alwaysLoad: true`
  // pins servers whose tools fire on most turns, so they bypass the
  // discovery hop. See https://github.com/jbaruch/nanoclaw/issues/30.
  const mcpServersConfig = {
    nanoclaw: {
      command: 'node',
      args: [mcpServerPath],
      env: {
        NANOCLAW_CHAT_JID: containerInput.chatJid,
        NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
        NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
        // Session identity. The MCP stdio server stamps this onto every
        // IPC request so the host responder knows which session's
        // `input-<session>/` dir should receive the `_script_result_*`
        // reply. Without it, responses to a maintenance container's
        // requests would land in `input-default/` and never be seen.
        NANOCLAW_SESSION_NAME: containerInput.sessionName || 'default',
        ...(containerInput.replyToMessageId
          ? { NANOCLAW_REPLY_TO_MESSAGE_ID: containerInput.replyToMessageId }
          : {}),
      },
      // IPC tools (send_message, react_to_message, schedule_task, …)
      // fire on most turns; deferring them would force a ToolSearch hop
      // before every reply.
      alwaysLoad: true,
    },
    ...(process.env.COMPOSIO_API_KEY
      ? {
          composio: {
            type: 'http' as const,
            url: 'https://connect.composio.dev/mcp',
            headers: {
              'x-consumer-api-key': process.env.COMPOSIO_API_KEY,
            },
          },
        }
      : {}),
    ...(fs.existsSync('/home/node/.tessl/api-credentials.json')
      ? {
          tessl: {
            command: 'tessl',
            args: ['mcp', 'start'],
          },
        }
      : {}),
  };

  // Subagent tools — same as parent minus TeamCreate/TeamDelete (no nesting)
  const subagentTools = [
    'Bash',
    'Read',
    'Write',
    'Edit',
    'Glob',
    'Grep',
    'WebSearch',
    'WebFetch',
    'TodoWrite',
    'ToolSearch',
    'Skill',
    'NotebookEdit',
    'mcp__nanoclaw__*',
  ];

  // Define a general-purpose subagent that inherits all skills and MCP
  // servers. When the main agent uses TeamCreate, it can reference this
  // agent type and the subagent will have full access to skills/rules.
  // Build subagent prompt with all rules and behavioral instructions.
  // Subagents don't inherit settingSources, CLAUDE.md, or .tessl/RULES.md
  // from the parent — they only get what's in their prompt + skills array.
  // Read all rule/context files and inject them into the subagent prompt.
  const subagentPromptParts: string[] = [
    'You are a background agent with the same capabilities as the main agent.',
    'Follow ALL rules below. Use skills via the Skill tool.',
    'Report results via mcp__nanoclaw__send_message.',
  ];

  // Build the rule/behavior chain. Group CLAUDE.md is now a thin
  // pointer (post-#153), so loading it would only inject @import lines
  // as raw text — this loader doesn't resolve @imports. The helper
  // enumerates the imported targets directly and adds main-only files
  // (project-root RULES.md + ADMIN.md) when the container is main.
  // See `subagent-prompt.ts` for unit tests covering the branching.
  const ruleFiles = buildSubagentRuleFilePaths({
    isMain: !!containerInput.isMain,
    soulMdPath,
    formattingMdPath,
  });
  for (const rulePath of ruleFiles) {
    if (fs.existsSync(rulePath)) {
      const content = fs.readFileSync(rulePath, 'utf-8').trim();
      if (content) {
        subagentPromptParts.push(
          `\n---\n# ${path.basename(rulePath)}\n${content}`,
        );
      }
    }
  }

  // Also load individual rule files referenced in RULES.md
  const tesslTilesDir = '/home/node/.claude/.tessl/tiles';
  if (fs.existsSync(tesslTilesDir)) {
    const walkRules = (dir: string) => {
      for (const entry of fs.readdirSync(dir)) {
        const fullPath = path.join(dir, entry);
        const stat = fs.statSync(fullPath);
        if (stat.isDirectory()) {
          walkRules(fullPath);
        } else if (entry.endsWith('.md') && fullPath.includes('/rules/')) {
          const content = fs.readFileSync(fullPath, 'utf-8').trim();
          if (content) {
            subagentPromptParts.push(`\n---\n# Rule: ${entry}\n${content}`);
          }
        }
      }
    };
    walkRules(tesslTilesDir);
  }

  const subagentPrompt = subagentPromptParts.join('\n');
  log(
    `Subagent prompt built: ${subagentPrompt.length} chars, ${installedSkills.length} skills`,
  );

  const agentDefinitions = {
    'general-purpose': {
      description:
        'General-purpose agent with full access to all skills, MCP tools, ' +
        'and rules. Use for any background task that needs the same ' +
        'capabilities as the main agent (heartbeat, research, analysis, etc.).',
      prompt: subagentPrompt,
      tools: subagentTools,
      skills: installedSkills,
      mcpServers: Object.keys(mcpServersConfig),
    },
  };

  for await (const message of query({
    prompt: stream,
    options: {
      cwd: '/workspace/group',
      additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
      // `resume: sessionId` continues the existing session at its
      // natural cursor (the JSONL tail). We deliberately do NOT pass
      // `resumeSessionAt: <lastAssistantUuid>` here — the Anthropic
      // Agent SDK treats that argument as fork-at-turn semantics, not
      // continue. Combining `resume` + `resumeSessionAt` for the same
      // session produces a `result.subtype = 'error_during_execution'`
      // on the very next query and wedges every subsequent IPC in the
      // container's lifetime (no `system/init`, no `assistant`, just
      // a single error result whose `lastAssistantUuid` is `none`,
      // which prevents any further progress). See #148 for the full
      // repro and root-cause analysis. The previous defensive recovery
      // (drop-resumeAt-on-error) only papered over the symptom; the
      // fix is to never set up the poison in the first place.
      resume: sessionId,
      systemPrompt: systemPromptAppend
        ? {
            type: 'preset' as const,
            preset: 'claude_code' as const,
            append: systemPromptAppend,
          }
        : undefined,
      // AGENT_MODEL is set by the orchestrator (`src/container-runner.ts`)
      // so the model can be bumped without rebuilding the agent-runner image.
      // Fallback matches the historical hardcoded value.
      model: process.env.AGENT_MODEL || 'opus[1m]',
      // Opus 4.7 rejects the old `thinking.type=enabled` shape entirely
      // and runs with thinking OFF unless adaptive is explicitly requested.
      // Adaptive also auto-enables interleaved thinking, which matters for
      // our multi-tool-call agentic workflow. Safe on 4.6/Sonnet 4.6 (both
      // support adaptive and will use it over the deprecated manual mode).
      //
      // `display: 'summarized'` is pinned because Opus 4.7 silently flipped
      // its default to `'omitted'` — without the pin, thinking blocks come
      // back as empty content with an opaque encrypted signature, breaking
      // any downstream consumer that reads thinking text.
      // See https://docs.anthropic.com/en/docs/build-with-claude/adaptive-thinking
      thinking: { type: 'adaptive' as const, display: 'summarized' as const },
      // AGENT_EFFORT is set by the orchestrator alongside AGENT_MODEL so
      // cost/latency can be tuned per deploy without rebuilding this image.
      // xhigh is Opus 4.7's recommended default for coding/agentic work
      // (Anthropic docs: "recommended starting point for coding and agentic
      // work"). On 4.6 and Sonnet 4.6 the SDK silently falls back to `high`.
      // Dropped from `max` — Anthropic recommends against max on 4.7 unless
      // evals show measurable headroom; xhigh is the sweet spot.
      //
      // NOTE: `thinking` is deliberately NOT env-configurable — its valid
      // shape is coupled to the model family (4.7 rejects `type: 'enabled'`,
      // older models require it), so independent config would let the two
      // drift and silently reproduce the 400-error outage. Model-family
      // changes are a code review, not a redeploy knob.
      effort: resolveAgentEffort(process.env.AGENT_EFFORT),
      allowedTools: [
        'Bash',
        'Read',
        'Write',
        'Edit',
        'Glob',
        'Grep',
        'WebSearch',
        'WebFetch',
        'Task',
        'TaskOutput',
        'TaskStop',
        'TeamCreate',
        'TeamDelete',
        'SendMessage',
        'TodoWrite',
        'ToolSearch',
        'Skill',
        'NotebookEdit',
        'mcp__nanoclaw__*',
      ],
      agents: agentDefinitions,
      env: sdkEnv,
      permissionMode: 'bypassPermissions',
      allowDangerouslySkipPermissions: true,
      settingSources: ['project', 'user'],
      mcpServers: mcpServersConfig,
      hooks: {
        PreCompact: [
          { hooks: [createPreCompactHook(containerInput.assistantName)] },
        ],
        // #141 — auto-inject MEMORY.md / RUNBOOK.md / latest daily log
        // before the first turn fires. Source filter (`startup` only)
        // and assistantName gate live inside the callback.
        SessionStart: [
          { hooks: [createSessionStartAutoContextHook(containerInput)] },
        ],
        // #136 — react-first guarantees the user gets an acknowledgement
        // emoji before any LLM tokens are spent. The gate inside
        // `react-first.ts` handles sub-agent / scheduled-task / no-name
        // containers; this entry just wires the callback in.
        // #137 — seed reply-threading state with the latest inbound id
        // on every new submitted prompt. Pairs with the PreToolUse
        // entry below.
        // #142 — silent-turn-audit seeds + resets per-turn state on
        // every new submitted prompt; the Stop hook below evaluates
        // it.
        UserPromptSubmit: [
          { hooks: [createReactFirstHook(containerInput)] },
          // #227 (tracks #214) — re-inject ground-truth reminder every
          // turn so verification discipline rides at the same salience
          // tier as SOUL. Skip cases (sub-agent, scheduled task, no
          // assistantName) match react-first.
          { hooks: [createGroundTruthReminderHook(containerInput)] },
          {
            hooks: [createReplyThreadingPromptHook(replyThreadingState)],
          },
          { hooks: [createSilentTurnPromptHook(silentTurnState)] },
        ],
        // #135 — block end-of-turn messages that surface banned
        // verification excuses without enumerating real attempts.
        // The hook re-runs the turn with a reminder injected.
        // #142 — Stop hook appends a JSONL entry to silent-turns.log
        // when the turn ended without acknowledging the triggering
        // inbound. Observability only — never blocks.
        Stop: [
          { hooks: [createLazyVerificationDetectorHook()] },
          {
            hooks: [createSilentTurnStopHook(silentTurnState, containerInput)],
          },
        ],
        // #116 — gate TaskOutput before it can leak the sub-agent
        // transcript. Matcher must be `TaskOutput` (not `mcp__*`) —
        // it's an SDK built-in, not an MCP tool.
        // #143 — bash-safety-net denies known-destructive Bash
        // commands. Separate matcher entry because the SDK runs
        // matchers independently per tool name.
        // #138 — auto-rewrite Markdown to HTML on send_message /
        // send_file before delivery. Matcher restricts the regex sweep
        // to those two MCP tools so unrelated MCP traffic isn't paid
        // for on every call.
        // #137 — deny standalone send_message while the latest inbound
        // is unanswered. Matcher restricts to send_message so unrelated
        // MCP traffic isn't paid for on every call.
        // #139 — suppress duplicate path-hygiene reports within 4h.
        // State is per-runQuery + seeded from the daily-log mtime so
        // cadence persists across container restarts.
        // #142 — track react / reply on the two MCP tools that
        // address the inbound.
        PreToolUse: [
          {
            matcher: 'TaskOutput',
            hooks: [createTaskOutputBlockGateHook()],
          },
          {
            matcher: 'Bash',
            hooks: [createBashSafetyNetHook()],
          },
          // #226 (tracks #214) — nudge the agent toward known
          // authoritative pointers when it's about to run a fresh
          // entity-lookup. Matcher restricts the regex sweep to the
          // tool families documented in the incident table, and
          // includes only the Composio tools whose names indicate
          // lookup-style search/list operations; the catalogue further
          // narrows by tool input shape so unrelated calls inside this
          // family don't pay the cost.
          {
            matcher:
              '^(Bash|Read|Grep|Glob|WebSearch|mcp__composio__.*(?:search|list).*)$',
            hooks: [createAuthoritativeSourceNudgeHook()],
          },
          {
            matcher: 'mcp__nanoclaw__send_(message|file)',
            hooks: [createNoMarkdownInSendMessageHook()],
          },
          {
            matcher: 'mcp__nanoclaw__send_message',
            hooks: [
              createReplyThreadingPreToolHook(
                replyThreadingState,
                isMaintenanceSession,
              ),
            ],
          },
          {
            matcher: 'mcp__nanoclaw__send_message',
            hooks: [createPathHygieneCadenceHook()],
          },
          {
            matcher: 'mcp__nanoclaw__(react_to_message|send_message)',
            hooks: [createSilentTurnTrackingHook(silentTurnState)],
          },
          // #326 — Composio outbound arg validator. Per-field shape
          // checks (header-injection guard on subject/to/cc/bcc/
          // channel/user, byte cap + control-char guard on body/text)
          // before the bytes leave the container. Runs BEFORE the
          // egress allowlist so a malformed arg is rejected without
          // paying the transcript-walk cost. Tool-only — does not
          // look at provenance. Matcher restricts the sweep to the
          // Composio outbound family; the validator's internal rule
          // table is the source of truth (matcher is a perf hint).
          {
            matcher:
              '^mcp__composio__(gmail_(send|reply)|slack_(post|send))\\w*$',
            hooks: [createComposioArgValidatorHook()],
          },
          // #320 — egress allowlist. Destination-level filter for
          // outbound tools (Composio gmail.send, slack.post,
          // send_message_to_chat). Reads /workspace/trusted/
          // egress_allowlist.json. Operator-originated chains bypass
          // by default; under untrusted-provenance, destination must
          // match an entry. Anchored alternation matcher to fire only
          // on the gated sinks — the sink classification inside the
          // hook is the source of truth, the matcher is a perf hint.
          {
            matcher:
              '^(mcp__composio__(gmail_(send|reply)|slack_(post|send))\\w*|mcp__nanoclaw__send_message_to_chat)$',
            hooks: [createEgressAllowlistHook(fs)],
          },
          // #320 — block agent-side mutations to the allowlist file.
          // Without this, an injection-driven chain in a trusted/main
          // container could call Write/Edit/Bash to add an attacker-
          // controlled destination before sending, undermining the
          // egress gate.
          {
            matcher: '^(Write|Edit|Bash)$',
            hooks: [createEgressAllowlistWriteGate()],
          },
          // #324 — destructive-op confirmation gate. Provenance-
          // conditional: operator-originated trusted chains pass,
          // untrusted-provenance chains require an unspent OOB token
          // minted by the host-side `aye-confirm` CLI. Matches every
          // nanoclaw MCP tool; classifyDestructiveOp() inside the
          // hook is the source of truth (so adding a new destructive
          // tool is a one-line edit in confirmation-tokens.ts, not a
          // matcher edit here).
          {
            matcher: '^mcp__nanoclaw__.*$',
            hooks: [createConfirmationTokenHook(fs)],
          },
          // #324 — block agent-side mutations to the token file. Matches
          // Write/Edit/Bash because all three could touch the file;
          // the gate inside the hook is the source of truth (matcher
          // is a perf hint).
          {
            matcher: '^(Write|Edit|Bash)$',
            hooks: [createConfirmationTokenWriteGate()],
          },
          // #322 — capability ACL. Walks the transcript back to the
          // most recent operator user-turn boundary, collects every
          // untrusted-provenance marker (Encoding A wrap or Encoding B
          // sentinel from #321) in the span, and denies the call if
          // its sink isn't in the intersection of allowed-sinks for
          // every source seen. Operator-originated chains (no markers
          // in span) bypass — preserves the trusted-tier capability
          // principle from #318. Matches every tool because the gate
          // is sink-agnostic; the ACL itself decides what's allowed.
          {
            matcher: '.*',
            hooks: [createCapabilityAclHook()],
          },
        ],
        // #117 — strip invisible-Unicode + cap byte size on every MCP
        // tool result. Matcher restricts to MCP because that's the
        // only tool family `updatedMCPToolOutput` can mutate.
        // #140 — flag fabricated-ID signatures (sequential email_01..,
        // pr1_notif, promo_001) in MCP tool returns.
        // #321 PR 2 — wrap MCP read-tool results (Composio gmail/
        // calendar/slack/github reads) in `<untrusted-input source>` so
        // #322's walk-back has a single in-band signal to grep for.
        // SDK invokes hooks in registration order: sanitizer normalises
        // bytes first, fidelity inspects raw text before the envelope
        // is added, wrap runs last and only mutates the final form.
        // #321 PR 4 — sidecar provenance sentinel for built-in tools
        // (WebFetch / WebSearch / Read external / Bash agent-browser).
        // Separate matcher because `WebFetch | WebSearch | Read | Bash`
        // is the built-in family and shares no other PostToolUse hooks.
        PostToolUse: [
          {
            matcher: 'mcp__.*',
            hooks: [
              createMcpToolResultSanitizerHook(),
              createComposioFidelityHook(),
              createUntrustedInputWrapHook(),
            ],
          },
          {
            // Anchored alternation — `WebFetch|Read|Bash` (unanchored)
            // would also match e.g. a third-party tool literally named
            // `MyBash` or `ReadMore`. Match only the exact built-in
            // tool names.
            matcher: '^(WebFetch|WebSearch|Read|Bash)$',
            hooks: [createProvenanceSentinelHook()],
          },
        ],
      },
    },
  })) {
    messageCount++;
    const msgType =
      message.type === 'system'
        ? `system/${(message as { subtype?: string }).subtype}`
        : message.type;
    log(`[msg #${messageCount}] type=${msgType}`);

    if (message.type === 'assistant' && 'uuid' in message) {
      lastAssistantUuid = (message as { uuid: string }).uuid;
      // Capture per-turn token usage off the assistant message's
      // wrapped Anthropic API response. Stored as `latestUsage` so the
      // final result emit (below) can pin it onto the ContainerOutput
      // payload — kill-auto-compaction Phase 1 (#125, tracks #104)
      // requires the orchestrator to see input_tokens to drive the
      // threshold detector. Stream-throttled emits below also include
      // it so the orchestrator can fire the warn-level note before
      // the result lands.
      const assistantMsg = message as {
        message?: {
          content?: Array<{
            type: string;
            text?: string;
            thinking?: string;
            name?: string;
            id?: string;
            input?: unknown;
          }>;
          usage?: {
            input_tokens?: number;
            output_tokens?: number;
            cache_read_input_tokens?: number;
            cache_creation_input_tokens?: number;
          };
        };
      };
      const u = assistantMsg.message?.usage;
      if (
        u &&
        typeof u.input_tokens === 'number' &&
        typeof u.output_tokens === 'number'
      ) {
        latestUsage = {
          input_tokens: u.input_tokens,
          output_tokens: u.output_tokens,
          cache_read_input_tokens: u.cache_read_input_tokens,
          cache_creation_input_tokens: u.cache_creation_input_tokens,
        };
      }
      // Extract text content for streaming preview, and emit per-block
      // observability log lines so the optional observer channel
      // (src/observer.ts on the host) can mirror live agent reasoning
      // to a status chat. These lines have no behavioral effect — they
      // go to stderr (docker logs) only. The observer parses them via
      // regex when OBSERVER_CHAT_JID is set; when unset, they're just
      // useful post-mortem context.
      const content = assistantMsg.message?.content;
      if (content) {
        const text = content
          .filter((c) => c.type === 'text' && c.text)
          .map((c) => c.text!)
          .join('');
        if (text) {
          streamingTextAccum = text;
          const now = Date.now();
          if (now - lastStreamEmit >= STREAM_THROTTLE_MS) {
            writeOutput({
              status: 'success',
              result: null,
              streamText: streamingTextAccum,
              newSessionId,
              usage: latestUsage,
            });
            lastStreamEmit = now;
          }
        }
        // Per-block observability lines (whitespace collapsed so each
        // block is one stderr line; downstream parsers split on \n).
        // Also stash any user-facing send tool_use ids so the
        // success-gated suppression branch below can match them up
        // with their tool_result.
        //
        // Note on send_voice (preserved from main): `send_voice` is
        // intentionally excluded from `pendingUserFacingToolUseIds`
        // because the host IPC processor (src/ipc.ts) doesn't yet
        // handle `type: 'send_voice'`, so the file is dropped silently.
        // Suppressing the final text on a dropped voice send would
        // leave the user with nothing. Until host support lands, the
        // tool itself returns isError so this code path is unreachable
        // for send_voice anyway, but pinning the allowlist keeps the
        // two layers in sync if either is touched independently.
        for (const block of content) {
          if (block.type === 'thinking' && block.thinking) {
            // Content-bearing logs are gated on OBSERVER_ENABLED — see
            // the gate's docblock above. Default emission is metadata
            // only (block kind + length) so post-mortem `docker logs`
            // greps still work without leaking reasoning content.
            if (OBSERVER_ENABLED) {
              log(
                `[msg #${messageCount}] thinking="${block.thinking.replace(/\s+/g, ' ')}"`,
              );
            } else {
              log(
                `[msg #${messageCount}] thinking len=${block.thinking.length}`,
              );
            }
          } else if (block.type === 'redacted_thinking') {
            log(`[msg #${messageCount}] redacted_thinking (encrypted)`);
          } else if (block.type === 'text' && block.text) {
            if (OBSERVER_ENABLED) {
              log(
                `[msg #${messageCount}] text="${block.text.replace(/\s+/g, ' ').slice(0, 400)}"`,
              );
            } else {
              log(`[msg #${messageCount}] text len=${block.text.length}`);
            }
          } else if (block.type === 'tool_use') {
            // Tool inputs frequently carry tokens, email content, IDs,
            // and other sensitive user data. Default emission is just
            // the tool name + id; the full input JSON is gated on
            // OBSERVER_ENABLED.
            if (OBSERVER_ENABLED) {
              const inputStr = JSON.stringify(block.input ?? {}).slice(0, 400);
              log(
                `[msg #${messageCount}] tool_use=${block.name} id=${block.id} input=${inputStr}`,
              );
            } else {
              log(
                `[msg #${messageCount}] tool_use=${block.name} id=${block.id}`,
              );
            }
            if (
              block.id &&
              (block.name === 'mcp__nanoclaw__send_message' ||
                block.name === 'mcp__nanoclaw__send_file')
            ) {
              pendingUserFacingToolUseIds.add(block.id);
            }
          }
        }
      }
    } else if (message.type === 'user' && 'message' in message) {
      // tool_result lives on user-typed messages following a tool_use.
      // Log success/error + a short preview so observers can spot tool
      // failures live and so post-mortem `docker logs` greps work.
      // Also flip the user-facing-send-succeeded flag here (only on
      // non-error results) so the final-text suppression at result-
      // time doesn't fire for hook-denied or errored sends.
      const content = (
        message as {
          message?: {
            content?: Array<{
              type: string;
              tool_use_id?: string;
              is_error?: boolean;
              content?: unknown;
            }>;
          };
        }
      ).message?.content;
      if (content) {
        for (const block of content) {
          if (block.type === 'tool_result') {
            const status = block.is_error ? 'error' : 'ok';
            // Tool results echo back arbitrary external data — emails,
            // API responses, file contents, search results. Default
            // emission is status only (id + ok|error); the content
            // preview is gated on OBSERVER_ENABLED so the operator
            // explicitly opts in to logging it.
            if (OBSERVER_ENABLED) {
              const preview = JSON.stringify(block.content ?? '').slice(0, 200);
              log(
                `[msg #${messageCount}] tool_result id=${block.tool_use_id} ${status} preview=${preview}`,
              );
            } else {
              log(
                `[msg #${messageCount}] tool_result id=${block.tool_use_id} ${status}`,
              );
            }
            if (
              block.tool_use_id &&
              pendingUserFacingToolUseIds.has(block.tool_use_id) &&
              block.is_error !== true
            ) {
              userFacingSendSucceeded = true;
            }
          }
        }
      }
    }

    if (message.type === 'system' && message.subtype === 'init') {
      newSessionId = message.session_id;
      log(`Session initialized: ${newSessionId}`);
    }

    if (
      message.type === 'system' &&
      (message as { subtype?: string }).subtype === 'task_notification'
    ) {
      const tn = message as {
        task_id: string;
        status: string;
        summary: string;
      };
      log(
        `Task notification: task=${tn.task_id} status=${tn.status} summary=${tn.summary}`,
      );
    }

    if (message.type === 'result') {
      resultCount++;
      const textResult =
        'result' in message ? (message as { result?: string }).result : null;
      // SDK result messages have a required `subtype` per the SDK
      // types, but other call sites in this file treat it as optional
      // and a missing/undefined value here would silently classify as
      // an error and emit `"undefined: undefined"` to the orchestrator
      // (per #149 review). Default to `'unknown'` so the diagnostic
      // string stays readable, and prefer the SDK's explicit
      // `is_error` flag when available — relying on the `subtype !==
      // 'success'` heuristic was the indirect path.
      const errMsg = message as {
        subtype?: string;
        errors?: string[];
        terminal_reason?: string;
        permission_denials?: unknown[];
        is_error?: boolean;
        stop_reason?: string | null;
        num_turns?: number;
        total_cost_usd?: number;
      };
      const subtype = errMsg.subtype || 'unknown';
      const isError =
        errMsg.is_error === true ||
        (subtype !== 'success' && subtype !== 'unknown');
      if (isError) {
        // SDKResultError carries the actual diagnostic context that the
        // generic 'error_during_execution' subtype name buries. Pull every
        // field the SDK exposes so the orchestrator log can pin down WHICH
        // failure mode tripped (prompt_too_long, model_error, blocking_limit,
        // rapid_refill_breaker, etc.). Without this, every failure looks
        // identical from outside.
        log(
          `Result #${resultCount}: subtype=${subtype} ERROR ` +
            `terminal_reason=${errMsg.terminal_reason || 'none'} ` +
            `errors=${JSON.stringify(errMsg.errors || [])} ` +
            `stop_reason=${errMsg.stop_reason || 'none'} ` +
            `permission_denials=${(errMsg.permission_denials || []).length} ` +
            `num_turns=${errMsg.num_turns ?? 'n/a'} ` +
            `cost=$${errMsg.total_cost_usd ?? 'n/a'}`,
        );
        // Pick the most-informative source for the human-readable
        // summary, falling through to `textResult` (the SDK sometimes
        // puts the only readable error there) before settling for the
        // bare subtype. Cap to 500 chars and collapse newlines so a
        // verbose error string can't blow up the IPC marker JSON we
        // write to stdout — that JSON gets parsed by the orchestrator
        // and excessively large strings have caused buffer issues
        // before.
        const rawSummary =
          errMsg.terminal_reason ||
          (errMsg.errors && errMsg.errors[0]) ||
          textResult ||
          subtype;
        const summary = String(rawSummary).replace(/\s+/g, ' ').slice(0, 500);
        writeOutput({
          status: 'error',
          result: null,
          newSessionId,
          error: `${subtype}: ${summary}`,
          usage: latestUsage,
        });
        sawErrorResult = true;
      } else {
        // #47: if the agent already used send_message / send_file
        // successfully (tracked above), suppress the SDK's final
        // closing-thought text so it doesn't get echoed as a second
        // user reply. The orchestrator's `if (result.result)` gate in
        // src/index.ts skips when result is null.
        const suppressFinalText = userFacingSendSucceeded && !!textResult;
        if (suppressFinalText) {
          log(
            `Suppressing result.text echo (send tool already succeeded): ${textResult!.slice(0, 80)}`,
          );
        }
        log(
          `Result #${resultCount}: subtype=${subtype}${textResult ? ` text=${textResult.slice(0, 200)}` : ''}`,
        );
        writeOutput({
          status: 'success',
          result: suppressFinalText ? null : textResult || null,
          newSessionId,
          usage: latestUsage,
        });
      }
      // Break out of the for-await loop after receiving the result.
      // Without this, the iterator hangs waiting for more SDK messages
      // that will never come, and follow-up IPC messages are lost.
      // The outer while(true) loop handles follow-ups via waitForIpcMessage().
      // See: https://github.com/qwibitai/nanoclaw/issues/233
      break;
    }
  }

  ipcPolling = false;
  // Build the metrics tail for the observer summary. Wall is
  // measured from the runQuery entry point. Tokens come from the
  // most recent assistant message's `usage` payload (already
  // captured for #125's threshold detector). Cache hit rate follows
  // Anthropic's documented definition:
  //   cache_read_input_tokens / (cache_read_input_tokens + input_tokens)
  // i.e. the share of context that came from cache vs. fresh
  // tokenization. Output tokens are excluded by definition. When the
  // denominator is 0 (no assistant turn / dry run), report n/a so
  // the observer renders it as "n/a" instead of NaN%.
  const wallMs = Date.now() - queryStartedAt;
  const tokensIn = latestUsage?.input_tokens ?? 0;
  const tokensOut = latestUsage?.output_tokens ?? 0;
  const cacheRead = latestUsage?.cache_read_input_tokens ?? 0;
  const cacheDenom = cacheRead + tokensIn;
  const cacheHitRate =
    cacheDenom > 0 ? `${((cacheRead / cacheDenom) * 100).toFixed(1)}` : 'n/a';
  log(
    `Query done. Messages: ${messageCount}, results: ${resultCount}, lastAssistantUuid: ${lastAssistantUuid || 'none'}, closedDuringQuery: ${closedDuringQuery}, wall_ms=${wallMs}, tokens_in=${tokensIn}, tokens_out=${tokensOut}, cache_hit_rate=${cacheHitRate}`,
  );
  return {
    newSessionId,
    lastAssistantUuid,
    closedDuringQuery,
    errorResult: sawErrorResult,
  };
}

interface ScriptResult {
  wakeAgent: boolean;
  data?: unknown;
}

const SCRIPT_TIMEOUT_MS = 30_000;

async function runScript(script: string): Promise<ScriptResult | null> {
  const scriptPath = '/tmp/task-script.sh';
  fs.writeFileSync(scriptPath, script, { mode: 0o755 });

  return new Promise((resolve) => {
    execFile(
      'bash',
      [scriptPath],
      {
        timeout: SCRIPT_TIMEOUT_MS,
        maxBuffer: 1024 * 1024,
        env: process.env,
      },
      (error, stdout, stderr) => {
        if (stderr) {
          log(`Script stderr: ${stderr.slice(0, 500)}`);
        }

        if (error) {
          log(`Script error: ${error.message}`);
          return resolve(null);
        }

        // Parse last non-empty line of stdout as JSON
        const lines = stdout.trim().split('\n');
        const lastLine = lines[lines.length - 1];
        if (!lastLine) {
          log('Script produced no output');
          return resolve(null);
        }

        try {
          const result = JSON.parse(lastLine);
          if (typeof result.wakeAgent !== 'boolean') {
            log(
              `Script output missing wakeAgent boolean: ${lastLine.slice(0, 200)}`,
            );
            return resolve(null);
          }
          resolve(result as ScriptResult);
        } catch {
          log(`Script output is not valid JSON: ${lastLine.slice(0, 200)}`);
          resolve(null);
        }
      },
    );
  });
}

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    try {
      fs.unlinkSync('/tmp/input.json');
    } catch {
      /* may not exist */
    }
    log(`Received input for group: ${containerInput.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`,
    });
    process.exit(1);
  }

  // Credentials are injected by the host's credential proxy via ANTHROPIC_BASE_URL.
  // No real secrets exist in the container environment.
  //
  // CLAUDE_CODE_AUTO_COMPACT_WINDOW is forwarded by the orchestrator
  // (`src/container-runner.ts`) — and only when ENABLE_THRESHOLD_NUKE is
  // off. We deliberately do NOT default it here: a hardcoded fallback
  // would silently mask a missing forward and reintroduce #252 (SDK
  // blocking-limit dropping below the orchestrator's 800k nuke threshold
  // when the kill-auto-compaction master flag is on). Whatever the
  // orchestrator placed in process.env passes through `...process.env`.
  const sdkEnv: Record<string, string | undefined> = { ...process.env };

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  let sessionId = containerInput.sessionId;
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs
  try {
    fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL);
  } catch {
    /* ignore */
  }

  // Build initial prompt (drain any pending IPC messages too)
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // --- Slash command handling ---
  // Only known session slash commands are handled here. This prevents
  // accidental interception of user prompts that happen to start with '/'.
  const KNOWN_SESSION_COMMANDS = new Set(['/compact']);
  const trimmedPrompt = prompt.trim();
  const isSessionSlashCommand = KNOWN_SESSION_COMMANDS.has(trimmedPrompt);

  if (isSessionSlashCommand) {
    log(`Handling session command: ${trimmedPrompt}`);
    let slashSessionId: string | undefined;
    let compactBoundarySeen = false;
    let hadError = false;
    let resultEmitted = false;

    try {
      for await (const message of query({
        prompt: trimmedPrompt,
        options: {
          cwd: '/workspace/group',
          resume: sessionId,
          systemPrompt: undefined,
          allowedTools: [],
          env: sdkEnv,
          permissionMode: 'bypassPermissions' as const,
          allowDangerouslySkipPermissions: true,
          settingSources: ['project', 'user'] as const,
          hooks: {
            PreCompact: [
              { hooks: [createPreCompactHook(containerInput.assistantName)] },
            ],
            // Mirror the main query()'s poison-defense hooks so the two
            // configurations don't drift if `allowedTools` is ever broadened
            // on this branch. Today the slash-command path has no tools, so
            // these are inert — but defining them here keeps the contract
            // single-sourced.
            PreToolUse: [
              {
                matcher: 'TaskOutput',
                hooks: [createTaskOutputBlockGateHook()],
              },
              { matcher: 'Bash', hooks: [createBashSafetyNetHook()] },
              {
                matcher: 'mcp__nanoclaw__send_(message|file)',
                hooks: [createNoMarkdownInSendMessageHook()],
              },
            ],
            PostToolUse: [
              {
                matcher: 'mcp__.*',
                hooks: [
                  createMcpToolResultSanitizerHook(),
                  createComposioFidelityHook(),
                ],
              },
            ],
          },
        },
      })) {
        const msgType =
          message.type === 'system'
            ? `system/${(message as { subtype?: string }).subtype}`
            : message.type;
        log(`[slash-cmd] type=${msgType}`);

        if (message.type === 'system' && message.subtype === 'init') {
          slashSessionId = message.session_id;
          log(`Session after slash command: ${slashSessionId}`);
        }

        // Observe compact_boundary to confirm compaction completed
        if (
          message.type === 'system' &&
          (message as { subtype?: string }).subtype === 'compact_boundary'
        ) {
          compactBoundarySeen = true;
          log('Compact boundary observed — compaction completed');
        }

        if (message.type === 'result') {
          const resultSubtype = (message as { subtype?: string }).subtype;
          const textResult =
            'result' in message
              ? (message as { result?: string }).result
              : null;

          if (resultSubtype?.startsWith('error')) {
            hadError = true;
            writeOutput({
              status: 'error',
              result: null,
              error: textResult || 'Session command failed.',
              newSessionId: slashSessionId,
            });
          } else {
            writeOutput({
              status: 'success',
              result: textResult || 'Conversation compacted.',
              newSessionId: slashSessionId,
            });
          }
          resultEmitted = true;
        }
      }
    } catch (err) {
      hadError = true;
      const errorMsg = err instanceof Error ? err.message : String(err);
      log(`Slash command error: ${errorMsg}`);
      writeOutput({ status: 'error', result: null, error: errorMsg });
    }

    log(
      `Slash command done. compactBoundarySeen=${compactBoundarySeen}, hadError=${hadError}`,
    );

    // Warn if compact_boundary was never observed — compaction may not have occurred
    if (!hadError && !compactBoundarySeen) {
      log(
        'WARNING: compact_boundary was not observed. Compaction may not have completed.',
      );
    }

    // Only emit final session marker if no result was emitted yet and no error occurred
    if (!resultEmitted && !hadError) {
      writeOutput({
        status: 'success',
        result: compactBoundarySeen
          ? 'Conversation compacted.'
          : 'Compaction requested but compact_boundary was not observed.',
        newSessionId: slashSessionId,
      });
    } else if (!hadError) {
      // Emit session-only marker so host updates session tracking
      writeOutput({
        status: 'success',
        result: null,
        newSessionId: slashSessionId,
      });
    }
    return;
  }
  // --- End slash command handling ---

  // Script phase: run script before waking agent
  if (containerInput.script && containerInput.isScheduledTask) {
    log('Running task script...');
    const scriptResult = await runScript(containerInput.script);

    if (!scriptResult || !scriptResult.wakeAgent) {
      const reason = scriptResult
        ? 'wakeAgent=false'
        : 'script error/no output';
      log(`Script decided not to wake agent: ${reason}`);
      writeOutput({
        status: 'success',
        result: null,
      });
      return;
    }

    // Script says wake agent — enrich prompt with script data
    log(`Script wakeAgent=true, enriching prompt with data`);
    prompt = `[SCHEDULED TASK]\n\nScript output:\n${JSON.stringify(scriptResult.data, null, 2)}\n\nInstructions:\n${containerInput.prompt}`;
  }

  // Tag untrusted group prompts with origin markers so the model (and compaction)
  // can distinguish user instructions from untrusted input. Trusted and main group
  // prompts are left untagged — they carry the same authority as system instructions.
  //
  // Scheduled-task provenance override: for untrusted groups, the default is
  // still "wrap everything", but orchestrator-trusted scheduled tasks
  // (`createdByRole` ∈ owner / main_agent / trusted_agent) are unwrapped
  // even on untrusted containers. Without this override, an untrusted
  // group's auto-registered heartbeat (which the host seeded) would get
  // its own instructions flagged as untrusted and the agent would refuse
  // to act — the task fires, costs tokens, and accomplishes nothing.
  // The `'untrusted_agent'` case stays wrapped: if an untrusted agent
  // self-scheduled the task, its prompt came FROM the untrusted group
  // and the defensive wrap is still appropriate.
  const isUntrustedContainer =
    !containerInput.isMain && !containerInput.isTrusted;
  const isOrchestratorTrustedTask =
    containerInput.isScheduledTask &&
    containerInput.createdByRole !== undefined &&
    containerInput.createdByRole !== 'untrusted_agent';
  if (isUntrustedContainer && !isOrchestratorTrustedTask) {
    prompt = wrapUntrustedInput(
      prompt,
      'untrusted-container',
      containerInput.groupFolder,
    );
  }

  // Query loop: run query → wait for IPC message → run new query → repeat.
  //
  // No `resumeAt` plumbing — see the comment on `resume:` inside
  // `runQuery` for why passing `resumeSessionAt` poisons the session.
  // The natural-cursor `resume: sessionId` is sufficient for both the
  // first query and every follow-up; the SDK appends to the JSONL tail.
  let consecutiveErrors = 0;
  try {
    while (true) {
      log(`Starting query (session: ${sessionId || 'new'})...`);

      let queryResult;
      try {
        queryResult = await runQuery(
          prompt,
          sessionId,
          mcpServerPath,
          containerInput,
          sdkEnv,
        );
      } catch (resumeErr) {
        const msg =
          resumeErr instanceof Error ? resumeErr.message : String(resumeErr);
        // Use the centralised predicate (#152): the previous narrow
        // regex `/session|conversation not found|resume/i` missed the
        // `error_during_execution` and `ENOENT.*\.jsonl` shapes that
        // #144 broadened the orchestrator-side check to catch. Without
        // this fix, a stale-session error THROWN out of `runQuery`
        // (rather than reported via the result-message path) would
        // bubble up as a generic failure instead of triggering the
        // fresh-session retry that the throw branch is meant to enact.
        if (sessionId && isStaleSessionError(msg)) {
          log(`Session resume failed (${msg}), retrying with fresh session`);
          sessionId = undefined;
          queryResult = await runQuery(
            prompt,
            undefined,
            mcpServerPath,
            containerInput,
            sdkEnv,
          );
        } else {
          // Drift surface for #155 lives on the orchestrator side
          // (`src/index.ts` debug-log when sessionId+error don't match
          // the predicate). The rethrown exception here propagates up
          // to the orchestrator as `output.error`, where that log
          // fires — adding a duplicate here would double-log every
          // miss and use the agent-runner's unconditional
          // console.error path (no level gating), making it
          // effectively error-level instead of debug-level.
          throw resumeErr;
        }
      }
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }

      // Recover from SDK result-message errors (`error_during_execution`,
      // `error_max_turns`, etc.) that aren't thrown exceptions. The
      // `resumeSessionAt` poison that USED to fire here is gone (#148),
      // so consecutive errors now indicate a genuine SDK / model issue,
      // not a self-inflicted resume mismatch. After two in a row, drop
      // the sessionId so the next IPC message starts a fresh
      // conversation — the previous conversation's lost but the
      // container stops burning tokens on a chain that won't recover.
      if (queryResult.errorResult) {
        consecutiveErrors++;
        log(
          `Error result detected (consecutive=${consecutiveErrors}).` +
            (consecutiveErrors >= 2
              ? ` Dropping sessionId=${sessionId || 'none'}`
              : ''),
        );
        if (consecutiveErrors >= 2) {
          sessionId = undefined;
        }
      } else {
        consecutiveErrors = 0;
      }

      // If _close was consumed during the query, exit immediately.
      // Don't emit a session-update marker (it would reset the host's
      // idle timer and cause a 30-min delay before the next _close).
      if (queryResult.closedDuringQuery) {
        log('Close sentinel consumed during query, exiting');
        break;
      }

      // Emit session update so host can track it
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      log('Query ended, waiting for next IPC message...');

      // Wait for the next message or _close sentinel
      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      error: errorMessage,
    });
    process.exit(1);
  }
}

main().then(() => process.exit(0));
