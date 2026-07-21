import fs from 'fs';
import path from 'path';

import { DATA_DIR } from './config.js';
import {
  DEFAULT_SESSION_NAME,
  sessionInputDirName,
} from './container-runner.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';
import { logger } from './logger.js';
import type { IpcDeps } from './ipc.js';
import type { RegisteredGroup } from './types.js';

/**
 * Task-file payload shape shared by every IPC command handler (#845).
 * Deserialized from raw JSON dropped into a group's `tasks/` IPC dir —
 * every field is untrusted container input; handlers validate before use.
 */
export interface IpcTaskPayload {
  type: string;
  taskId?: string;
  // #512: `prompt` and `script` cross the IPC boundary as raw JSON, so a
  // writer can (and does — the JSON-Buffer `{type:'Buffer',data:[...]}`
  // shape) send non-string values. Typed `unknown` so every handler is
  // forced through `coerceTaskTextField` instead of trusting the wire
  // shape and reintroducing BLOB / "[object Object]" persistence.
  prompt?: unknown;
  schedule_type?: string;
  schedule_value?: string;
  /**
   * IANA timezone for cron expressions; #102.
   *
   * `null` is also accepted on the update path (where it means
   * "clear back to TIMEZONE default"). Must be `string | null` — not
   * just `string` — because IPC payloads arrive as raw JSON and the
   * caller can legitimately send `null` to unset; TS strict mode
   * would otherwise reject the `data.timezone === null` check.
   */
  timezone?: string | null;
  context_mode?: string;
  // Raw wire value — see `prompt` above (#512 applies to both fields).
  script?: unknown;
  groupFolder?: string;
  chatJid?: string;
  targetJid?: string;
  // For register_group
  jid?: string;
  name?: string;
  folder?: string;
  trigger?: string;
  requiresTrigger?: boolean;
  containerConfig?: RegisteredGroup['containerConfig'];
  // For set_trusted
  trusted?: boolean;
  // For set_agent_model (#395). `string` = per-group override, `null` =
  // clear the override (fall back to global AGENT_MODEL).
  // `undefined` is rejected at the handler.
  agentModel?: string | null;
  // For set_maintenance_agent_model (#509). `string` = per-session-slot
  // override that applies only to the maintenance container slot for
  // this group; `null` = clear the override (maintenance falls back to
  // the per-group `agentModel` → global `AGENT_MODEL` ladder, which is
  // the pre-#509 behavior).
  maintenanceAgentModel?: string | null;
  // For set_session_caps (#561). Positive number = per-group cap
  // override; `null` = clear the override (fall back to the global
  // SESSION_TURN_CAP / SESSION_TOKEN_CAP). A field left `undefined`
  // leaves that cap unchanged; both undefined is rejected at the
  // handler. Non-positive / non-finite numbers are rejected.
  sessionTurnCap?: number | null;
  sessionTokenCap?: number | null;
  // For set_additional_tiles (#305). Array of tile names from the
  // local registry to overlay on top of the trust-tier baseline.
  // `null` or `[]` clears the override. Anything else (string,
  // object, undefined) is rejected at the handler.
  additionalTiles?: string[] | null;
  // For host operations / github_backup / promote_staging
  requestId?: string;
  message?: string;
  // persist_tz_segments (#748): the `segments[]` array the in-container
  // TripIt → Reclaim sync parsed from `reclaim-tripit-timezones-sync`'s
  // `--output=json` stdout. Raw JSON off the IPC wire — validated to an
  // array at the handler before it reaches `applyTripitSegmentsToTzState`.
  segments?: unknown;
  // persist_global_file (#393): allowlisted global persona filenames to
  // commit + push. Validated by `validateGlobalFilesToPersist` at the
  // handler against `PERSISTABLE_GLOBAL_FILES`.
  files?: unknown;
  tileName?: string;
  skillName?: string;
  // push_staged_to_branch
  branch?: string;
  commitMessage?: string;
  // For run_sidecar (#750): allowlisted flags the plugin appends to the
  // named sidecar's command line. Image + mounts come from the trusted
  // registry, never the payload. Typed `unknown` because IPC payloads
  // arrive as raw JSON — the handler validates it is a string[] before use.
  flags?: unknown;
  command?: string;
  payload?: string | Record<string, unknown>;
  confirm?: boolean;
  // chat_status / nuke_chat / send_message_to_chat / inspect_gate_decisions
  chat_id?: string;
  chat_name?: string;
  session?: 'default' | 'maintenance' | 'all';
  // inspect_gate_decisions (#443)
  message_id?: string;
  limit?: number;
  // send_message_to_chat
  text?: string;
  pin?: boolean;
  sender?: string;
  /**
   * Continuation marker for self-resuming cycles (#93/#130). Set by the
   * resumable-cycle helper skill when scheduling the next link of a
   * chain via `schedule_task`. Persisted onto the scheduled_tasks row
   * verbatim; surfaced to the spawned container at fire time as
   * `NANOCLAW_CONTINUATION=1` + `NANOCLAW_CONTINUATION_CYCLE_ID=<value>`.
   * Free-form opaque slot key (UTC date / ISO week per the proposal),
   * but type-narrowed to string for safety; non-string values are
   * dropped at the handler.
   */
  continuation_cycle_id?: string;
  // For promote_learned_trigger (#451 item 1). Identifies a learned
  // proposal in registered_groups.trigger_pattern by its {kind,
  // pattern} tuple — the same identity the trigger-learner schema
  // uses to supersede prior versions.
  kind?: string;
  pattern?: string;
  // For fetch_markdown (#169). All optional except `url`. Mirrors the
  // snitchmd CLI flag set; see docs/fetch-tools.md for the decision
  // matrix and snitchmd's own README for flag semantics.
  url?: string;
  wait?: number;
  waitUntil?: string;
  waitForSelector?: string;
  favorPrecision?: boolean;
  favorRecall?: boolean;
  includeLinks?: boolean;
  includeImages?: boolean;
  maxChars?: number;
  noCache?: boolean;
  timeout?: number;
}

/**
 * Everything a command handler gets to work with. `sourceGroup` and
 * `isMain` are VERIFIED identity — derived from the IPC directory the
 * task file arrived in, never from payload fields. That derivation is
 * the security boundary; handlers must base authorization on these,
 * not on anything inside `data`.
 */
export interface IpcHandlerContext {
  data: IpcTaskPayload;
  /** Verified identity from the IPC directory path. */
  sourceGroup: string;
  /** Verified from the directory path, not the payload. */
  isMain: boolean;
  deps: IpcDeps;
}

export type IpcHandler = (ctx: IpcHandlerContext) => Promise<void> | void;

export interface IpcHandlerSpec {
  /**
   * When true, the dispatcher enforces the admin gate BEFORE the
   * handler runs: a non-main caller is logged, gets an error envelope
   * (only when the payload carries a valid `requestId` — mirrors the
   * pre-registry per-case behavior where fire-and-forget commands
   * blocked silently), and the handler is never invoked. Handlers with
   * finer-grained authorization (e.g. "main OR own-group") leave this
   * unset and gate internally on `ctx.isMain` / `ctx.sourceGroup`.
   */
  requiresMain?: boolean;
  handler: IpcHandler;
}

const handlers = new Map<string, IpcHandlerSpec>();

/**
 * Register a host IPC command (#845). New capabilities plug in here
 * instead of editing the dispatcher in `ipc.ts`. Duplicate names are a
 * wiring bug (two modules claiming one command) and fail loudly.
 */
export function registerIpcHandler(name: string, spec: IpcHandlerSpec): void {
  if (handlers.has(name)) {
    throw new Error(`IPC handler already registered: ${name}`);
  }
  handlers.set(name, spec);
}

export function hasIpcHandler(name: string): boolean {
  return handlers.has(name);
}

/**
 * Dispatch one task payload to its registered handler. Returns false
 * when no handler is registered for `data.type` — the caller falls
 * through to the legacy switch until every command has migrated.
 */
export async function dispatchIpcTask(
  ctx: IpcHandlerContext,
): Promise<boolean> {
  const spec = handlers.get(ctx.data.type);
  if (!spec) return false;
  if (spec.requiresMain && !ctx.isMain) {
    logger.warn(
      { sourceGroup: ctx.sourceGroup, type: ctx.data.type },
      'Unauthorized IPC command blocked (main-only)',
    );
    // Only a VALID requestId gets an error envelope — a missing, empty,
    // or malformed id means fire-and-forget (or a payload we refuse to
    // route a reply for), matching the legacy per-case convention where
    // those blocked silently with just the warn above.
    if (
      typeof ctx.data.requestId === 'string' &&
      VALID_REQUEST_ID_RE.test(ctx.data.requestId)
    ) {
      fs.writeFileSync(
        scriptResultPath(ctx.sourceGroup, ctx.data),
        JSON.stringify({
          error: `${ctx.data.type} is admin-tile only`,
        }),
      );
    }
    return true;
  }
  await spec.handler(ctx);
  return true;
}

// Session names accepted on IPC requests: ONLY the two the orchestrator
// ever creates. A broader regex (e.g. `[A-Za-z0-9_-]+`) would let a
// container send distinct valid-looking names and force the host into
// unbounded `input-<session>/` dir creation below — an empty-dir DoS.
// Canonical enum is the right level of trust for payload-supplied values.
const KNOWN_SESSION_NAMES: ReadonlySet<string> = new Set([
  DEFAULT_SESSION_NAME,
  MAINTENANCE_SESSION_NAME,
]);
const VALID_REQUEST_ID_RE = /^[A-Za-z0-9_-]+$/;

/**
 * Compute the host path where an IPC response file should land.
 *
 * Both `data.sessionName` and `data.requestId` arrive from the container's
 * IPC payload — treat as untrusted. Without validation, crafted values
 * like `../default` or `../../etc/passwd` would make `path.join` escape
 * the expected `<DATA_DIR>/ipc/<sourceGroup>/input-<session>/` subtree.
 *
 * Fail-safe strategy, two independent fallbacks:
 * - Invalid `requestId` → fixed filename `_script_result_invalid.json`.
 *   Keeps path traversal out of the filename AND prevents a noisy/
 *   malicious container from filling disk by spamming unique ids —
 *   at most one orphan file per session's input dir, overwritten in
 *   place each time. The SESSION dir is still whatever was validated
 *   from the payload (the `sessionName` check is separate).
 * - Invalid `sessionName` → fall back to `DEFAULT_SESSION_NAME`. Blocks
 *   `..`-style path-segment escape into a different group's subtree.
 *
 * Both fallbacks log at warn level for auditing. The malformed request
 * effectively times out (its response lands where no container polls),
 * which is the correct outcome for a bad payload. This keeps every
 * caller's `fs.writeFileSync(resultPath, ...)` pattern intact (no null-
 * checking at 10+ call sites) while still blocking path traversal.
 */
export function scriptResultPath(
  sourceGroup: string,
  data: { sessionName?: string; requestId?: string },
): string {
  let requestId: string;
  if (
    typeof data.requestId === 'string' &&
    VALID_REQUEST_ID_RE.test(data.requestId)
  ) {
    requestId = data.requestId;
  } else {
    logger.warn(
      { sourceGroup, requestId: data.requestId },
      'IPC request has missing or invalid requestId — routing response to orphan path',
    );
    // Fixed filename for all invalid requests so a noisy/malicious container
    // can't spam unique requestIds and fill disk with orphan replies. At
    // most one `_script_result_invalid.json` file exists per input dir, and
    // it gets overwritten on every subsequent malformed request.
    requestId = 'invalid';
  }
  let session = DEFAULT_SESSION_NAME;
  if (typeof data.sessionName === 'string' && data.sessionName) {
    if (KNOWN_SESSION_NAMES.has(data.sessionName)) {
      session = data.sessionName;
    } else {
      logger.warn(
        { sourceGroup, sessionName: data.sessionName },
        'IPC request has unknown sessionName — falling back to default',
      );
    }
  }
  const inputDir = path.join(
    DATA_DIR,
    'ipc',
    sourceGroup,
    sessionInputDirName(session),
  );
  // Ensure the session's input dir exists before the caller writes into it.
  // In the common path both sessions have already spawned at least once and
  // the dir exists — but a maintenance-only group (or a container that has
  // never gone through default) won't have `input-default/`, and our
  // fallback routes here for malformed payloads. Creating the dir
  // defensively keeps `fs.writeFileSync(resultPath, ...)` from throwing
  // ENOENT at every caller.
  fs.mkdirSync(inputDir, { recursive: true });
  return path.join(inputDir, `_script_result_${requestId}.json`);
}
