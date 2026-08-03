/**
 * Container Runner for NanoClaw
 * Spawns agent execution in containers and handles IPC
 */
import { ChildProcess, spawn, spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';

import {
  AGENT_AUTO_COMPACT_WINDOW,
  ASSISTANT_NAME,
  ASSISTANT_USERNAMES,
  CONTAINER_IMAGE,
  CONTAINER_MAX_OUTPUT_SIZE,
  CONTAINER_TIMEOUT,
  CREDENTIAL_PROXY_PORT,
  DATA_DIR,
  ENABLE_THRESHOLD_NUKE,
  HOST_GID,
  HOST_UID,
  IDLE_TIMEOUT,
  MAINTENANCE_CONTAINER_TIMEOUT,
  TIMEZONE,
} from './config.js';
import { resolveGroupFolderPath, resolveGroupIpcPath } from './group-folder.js';
import { containerLogPath, stripAnsi } from './host-logs.js';
import { CR_FS_CODES, isFsErrorWithCode } from './fs-errors.js';
import { logger } from './logger.js';
import { onAgentLine } from './observer.js';
import {
  CONTAINER_HOST_GATEWAY,
  CONTAINER_RUNTIME_BIN,
  hostGatewayArgs,
  readonlyMountArgs,
  stopContainer,
} from './container-runtime.js';
import { defaultComputeNextRun } from './cadence-registry.js';
import { detectAuthMode } from './credential-proxy.js';
import { registerContainer, unregisterContainer } from './proxy-registry.js';
import {
  applyOneCliToSpawn,
  isOneCliConfigured,
  ONECLI_MANAGED_PLACEHOLDER,
} from './onecli-client.js';
import type { TrustTier } from './trust-tier.js';
import { rebuildCadenceRegistryForGroup } from './db-tasks.js';
import { isHandoffActive } from './handoff.js';
import { RegisteredGroup } from './types.js';
import { readEnvFile } from './env.js';

/**
 * Typed error for an infrastructure failure raised by `runContainerAgent`'s
 * spawn setup — container spawn, docker, filesystem, OneCLI fail-closed.
 * `runAgent` (`src/index.ts`) narrows on this to convert an operational
 * agent-run failure into a typed `'error'` result (the caller branches on it
 * for #428 cursor-rollback) while letting programmer defects propagate to the
 * queue's per-group boundary. Introduced by #784 so that boundary is a closed
 * `instanceof` set rather than a defect-blacklist.
 *
 * The clearly-operational NON-fs failures (the OneCLI fail-closed CA/proxy
 * throws) construct this type directly at their throw sites; the fs failures
 * are converted by `toContainerAgentError`. Everything else — validation,
 * refuse-to-spawn, and init-guard `new Error`s (e.g. `resolveGroupFolderPath`'s
 * escaping-folder check, `rebuildCadenceRegistryForGroup`'s
 * `called before initDatabase` guard) — is a defect that must surface, so it is
 * NOT converted.
 */
export class ContainerAgentError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'ContainerAgentError';
  }
}

/**
 * Re-raise a failure from `runContainerAgent`'s spawn setup: convert an
 * infrastructure fs failure to a `ContainerAgentError`, or propagate
 * everything else untouched. Always throws (`never`).
 *
 * Converts ONLY an fs `ErrnoException` whose code the spawn path treats as
 * operational (`CR_FS_CODES` — EACCES/ENOSPC/EROFS/… on a mkdir/write). That is
 * the sole "infrastructure failure by design" a caller can't otherwise
 * distinguish. Everything else propagates so the bug surfaces instead of being
 * retried as a routine agent failure:
 *   - an `Error` subclass (`TypeError`, `ReferenceError`, …) — a defect;
 *   - a non-`Error` throw;
 *   - an fs `ErrnoException` with an UNEXPECTED code — the same signal the
 *     module's `isFsErrorWithCode(err, CR_FS_CODES)` guards rethrow on;
 *   - a code-less `new Error` — a validation / refuse-to-spawn / init-guard
 *     defect (e.g. `rebuildCadenceRegistryForGroup`'s
 *     `called before initDatabase`). The operational non-fs failures are typed
 *     at their sites (OneCLI fail-closed), so they never rely on this path.
 * An already-typed `ContainerAgentError` is re-raised as-is by the final
 * `throw` (its code check is false — it carries no errno code).
 */
export function toContainerAgentError(err: unknown): never {
  if (isFsErrorWithCode(err, CR_FS_CODES)) {
    throw new ContainerAgentError((err as Error).message, { cause: err });
  }
  throw err;
}

/**
 * Run one setup step of `runContainerAgent`'s pre-spawn path (before the
 * container is registered / the spawn handlers are attached). Any failure is
 * re-raised through `toContainerAgentError` so an infrastructure fault becomes
 * a typed `ContainerAgentError` and a defect propagates. #784.
 */
function spawnSetup<T>(step: () => T): T {
  try {
    return step();
  } catch (err) {
    // toContainerAgentError always throws (returns `never`); the `throw`
    // keyword is what re-raises here (and satisfies no-catch-all).
    throw toContainerAgentError(err);
  }
}

// Tile selection + atomic publish moved to ./tile-materialize.ts
// (#851 slice 3). Import for local use + facade re-export so the
// existing import surface is unchanged (facade sanctioned by #851).

export {
  atomicPublishDir,
  getInstalledTiles,
  getRegistryTilesDir,
  selectTiles,
} from './tile-materialize.js';

// Sentinel markers for robust output parsing (must match agent-runner)
const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

// Secret env-file machinery moved to ./secret-env.ts (#851 slice 2).
// Import for local use + facade re-export so the existing import
// surface is unchanged (facade sanctioned by #851).
import {
  ONECLI_MANAGED_VARS,
  SECRET_CONTAINER_VARS,
  buildSecretEnvFile,
} from './secret-env.js';

export {
  BYAIR_MANAGED_PLACEHOLDER,
  ONECLI_MANAGED_VARS,
  SECRET_CONTAINER_VARS,
  buildSecretEnvFile,
  type SecretEnvFile,
} from './secret-env.js';

export { ONECLI_MANAGED_PLACEHOLDER };

// Model resolution moved to ./model-resolve.ts (#851 slice 1). Import
// for local use + facade re-export so ipc/scheduler/tests keep their
// container-runner import surface (facade sanctioned by #851).
import {
  resolveAgentModel,
  resolveSessionAgentModel,
  resolveTierBaseModel,
} from './model-resolve.js';

export {
  DEFAULT_AGENT_MODEL,
  TRUSTED_TIER_MODEL,
  UNTRUSTED_TIER_MODEL,
  resolveAgentModel,
  resolvePerGroupAgentModel,
  resolveSessionAgentModel,
  resolveTierBaseModel,
} from './model-resolve.js';

const AGENT_MODEL = resolveAgentModel(process.env.AGENT_MODEL);

/**
 * Effort level the agent-runner passes to the SDK's `query()` call.
 * Forwarded as `AGENT_EFFORT` on container spawn.
 *
 * Resolution order (from highest precedence):
 *   1. `AGENT_EFFORT` environment variable on the orchestrator process
 *   2. The hardcoded default below (`xhigh`)
 *
 * Operators can override at runtime by setting `AGENT_EFFORT` in the
 * orchestrator's env (e.g. docker-compose, systemd unit), no rebuild
 * needed. The agent-runner validates the forwarded value against the
 * allowed set and falls back to `xhigh` on invalid input, so a typo
 * here won't crash containers — it'll log a warning inside the runner.
 *
 * Valid values: `'low' | 'medium' | 'high' | 'xhigh' | 'max'`.
 * - On Opus 4.7: `xhigh` is Anthropic's recommended default for
 *   coding/agentic work; `max` is reserved for frontier problems.
 * - On Opus 4.6 / Sonnet 4.6: `xhigh` silently falls back to `high` in
 *   the SDK.
 * See https://docs.anthropic.com/en/docs/build-with-claude/effort
 */
const AGENT_EFFORT = process.env.AGENT_EFFORT || 'xhigh';

// Filtered-DB snapshot machinery moved to ./filtered-db.ts (#851
// slice 4). Import for local use + facade re-export so the existing
// import surface is unchanged (facade sanctioned by #851).

export { createFilteredDb } from './filtered-db.js';

export interface ContainerInput {
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
   * #479 sub-#1: trigger message id for usage attribution (the
   * gate-allowed message in `evaluateGateChain`'s verdict). Distinct
   * from `replyToMessageId`, which is the reply-target — usually the
   * latest message in a polled batch — for reply-threading. The two
   * differ when the gate allows on an earlier message in a batch and
   * the reply quotes the latest. The credential proxy logs this as
   * `message_id` on each JSONL line; `replyToMessageId` is used as a
   * fallback when no gate verdict was computed (empty gate chain).
   */
  triggerMessageId?: string;
  /**
   * Whether the inbound batch is "addressed to us" — drives the
   * agent-runner's react-first 👀 gate (#289). Resolved by the
   * orchestrator (see `isAddressedToUs` in `src/index.ts`) from
   * isMain / 1:1-DM / trigger-match / reply-to-our-bot, independent
   * of `requires_trigger`. Omitted for non-channel paths (scheduled
   * tasks, scripts) — agent-runner treats `undefined` as "no
   * addressed-ness signal" and falls back to its existing skips.
   */
  addressedToUs?: boolean;
  /**
   * Provenance of a scheduled task (undefined for non-scheduled runs).
   * Drives whether the agent-runner wraps the prompt in `<untrusted-input>`.
   * Only `'untrusted_agent'` triggers the wrap; owner/main/trusted bypass
   * because their content originates from a trusted source.
   *
   * Security boundary: the task-scheduler passes this from the DB row's
   * `created_by_role` column, which was itself set by ipc.ts from the
   * VERIFIED source-group trust tier at schedule_task time. The agent
   * that scheduled the task never got to claim its own role — so an
   * untrusted agent that self-scheduled a prompt sees it come back
   * wrapped on the next fire.
   */
  createdByRole?: 'owner' | 'main_agent' | 'trusted_agent' | 'untrusted_agent';
  /**
   * Which per-group session this container run belongs to. Drives the
   * `.claude/` dir location and the group-queue slot key.
   * - `'default'` (omitted): user-facing AyeAye, serves inbound IPC messages.
   * - `'maintenance'`: scheduled AyeAye, runs scheduled_tasks (heartbeat,
   *   nightly, weekly, reminders). Runs in parallel with `'default'` for the
   *   same group so maintenance never blocks user replies.
   *
   * Invariant: inbound Telegram/etc. messages ALWAYS route to `'default'`.
   * `src/task-scheduler.ts` is the sole writer of `'maintenance'`.
   */
  sessionName?: string;
  /**
   * Continuation marker for self-resuming cycles (#93/#130). Set only on
   * scheduled-task spawns whose `scheduled_tasks.continuation_cycle_id`
   * column is non-NULL. When present the spawned container gets:
   *   - `NANOCLAW_CONTINUATION=1`
   *   - `NANOCLAW_CONTINUATION_CYCLE_ID=<value>`
   *
   * Absence (undefined / empty) means "fresh invocation" — neither env var
   * is set, and that absence is itself the signal the calling skill
   * (resumable-cycle / nightly-housekeeping / etc.) cross-checks against
   * its prompt-prefix marker. Mismatch fails closed to fresh invocation;
   * a scheduler that sets the env but mangles the prompt (or vice versa)
   * therefore never silently bypasses the two-phase lock acquisition.
   */
  continuationCycleId?: string;
  /**
   * Per-task AGENT_MODEL override for #509 Phase 3. Set by the
   * task-scheduler from `scheduled_tasks.agent_model` on this row.
   * When present and non-empty after trim, beats every other knob in
   * the resolveSessionAgentModel ladder (maintenanceAgentModel,
   * group agentModel, AGENT_MODEL env, DEFAULT_AGENT_MODEL); audit
   * log emits `task_override` as the source. Unknown-prefix values
   * fall back to the session-level value via `resolvePerGroupAgentModel`
   * — never silently jump to the global default. Undefined for
   * non-scheduled-task spawns (interactive inbound, IPC scripts).
   */
  taskAgentModel?: string | null;
}

export interface ContainerOutput {
  // `'precheck_skipped'` (#581) — emitted by the agent-runner when the
  // scheduled-task precheck script returned `wake_agent: false`. The
  // agent never woke; the wrapper never ran. Distinct from `'success'`
  // so the silent-success watchdog can tell a precheck-gated no-op
  // (healthy quiet) apart from an agent that woke and produced an
  // empty result (the original #581 bug). A precheck script that
  // crashes / emits non-JSON / omits `wake_agent` is `'error'`, not
  // `'precheck_skipped'`.
  //
  // `'killed'` (#589 reopened / #682) — resolved host-side (never
  // emitted by the agent-runner) when a maintenance container ends
  // without delivering a terminal result. A healthy maintenance
  // one-shot reaches a terminal result and exits naturally
  // (scheduleClose → `_close`) within seconds, so a run that produced
  // only streaming previews and then either (a) was reaped by the
  // MAINTENANCE_CONTAINER_TIMEOUT inactivity timer or (b) exited
  // cleanly (code 0) was incomplete — reaped/exited mid-compose, not
  // the misleading `'success'` that hid the original #589 silent-stop.
  // #682 gates this on the precise `hadTerminalResult` signal (a marker
  // with no `streamText`), so a run that DID deliver a terminal result
  // and then idled out keeps its `'success'` — only a no-terminal-
  // result end is `'killed'`. Incomplete + retriable / redeliverable.
  status: 'success' | 'error' | 'killed' | 'precheck_skipped';
  result: string | null;
  newSessionId?: string;
  error?: string;
  streamText?: string;
  /**
   * #890 follow-up — `true` when this run ended on a TIMEOUT rather
   * than any other failure. Two timeouts can end a run and both stamp
   * it: the precheck's own declared `precheck_timeout_ms` budget
   * (stamped in-container by `buildPrecheckErrorOutput`, arriving
   * through the marker JSON) and this host's container kill.
   *
   * Structured rather than inferred from `error` prose, so the operator
   * alert in `task-scheduler.ts` keys off a field both sides own. The
   * three emitters word their messages differently and are free to be
   * reworded; a rewording must not silently stop the alerting.
   *
   * Absent on every non-timeout outcome — a plain crash or non-zero
   * exit stays covered by the heartbeat's task-failure report.
   */
  timedOut?: boolean;
  /**
   * #581 — Set to `true` by the agent-runner when (a) the agent
   * successfully used `send_message` / `send_file` during this turn
   * AND (b) the SDK's final result message carried non-empty text.
   * The two-condition gate matches the case the suppression actually
   * targets: the SDK's closing-thought text that would otherwise be
   * echoed as a second user reply on top of the agent's `send_message`.
   * When `chat_displayed` is `true`, the orchestrator (`src/index.ts`)
   * and the task-scheduler (`src/task-scheduler.ts`) skip their
   * chat-echo + `storeMessage` paths but STILL populate
   * `task_run_logs.result` from `result` so observability isn't lost.
   * When `chat_displayed` is absent/false, behavior is unchanged from
   * the pre-#581 contract — including the case where `send_message`
   * succeeded but `textResult` was empty (the orchestrator's outer
   * `if (result.result)` gate already skips chat-echo there).
   */
  chat_displayed?: boolean;
  /**
   * Per-turn token usage from the agent-runner's most recent SDK
   * assistant message. Used by the kill-auto-compaction telemetry +
   * threshold detector (issue #104, design at
   * `docs/proposals/kill-auto-compaction.md`). Optional because
   * non-assistant outputs (errors before any model turn fires)
   * may not carry a usage payload.
   */
  usage?: {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
  };
  /**
   * #689 — stamped `true` by the agent-runner on a TERMINAL `success`
   * marker when a `requires_delivery` skill (morning-brief) ended a
   * runQuery without delivering any user-facing content. The status
   * stays `success` so this container's `scheduleClose` still drains
   * the maintenance slot promptly (no #461 wedge), but the close
   * handler below resolves the run as `killed` (incomplete / retriable)
   * rather than recording the synthesized success as a real delivery —
   * the silent success of the #589/#682 lineage #689 reopened. Absent
   * on every marker from a skill that didn't declare `requires_delivery`
   * and on any run that did deliver. See the agent-runner's
   * `delivery-requirement.ts`.
   */
  noDelivery?: boolean;
  /**
   * #901 — stamped `true` by the agent-runner on a TERMINAL `success`
   * marker when no assistant message carried a `usage` payload during
   * the run, i.e. the SDK returned without the model producing a single
   * turn. The subscription-cap abort takes this shape: the CLI ends the
   * turn before issuing any API request and hands back a clean result
   * whose text is the cap notice, so nothing in the result marks it a
   * failure and the run recorded `success` while doing no work.
   *
   * Handled exactly like `noDelivery`: the marker's `success` status is
   * preserved so `scheduleClose` still drains the maintenance slot
   * promptly (no #461 wedge), and the close handler below resolves the
   * run as `killed` (incomplete / retriable). Unlike `noDelivery` this
   * needs no per-skill declaration — a run in which the model never ran
   * is a failed run for every skill. See the agent-runner's
   * `model-work.ts`.
   */
  noModelWork?: boolean;
}

// Session naming moved to ./session-names.ts and the mount builder to
// ./volume-mounts.ts (#851 slice 5). Import for local use + facade
// re-export so the existing import surface is unchanged.
import {
  DEFAULT_SESSION_NAME,
  MAINTENANCE_SESSION_NAME,
  sessionInputDirName,
} from './session-names.js';
import {
  buildVolumeMounts,
  mountOneCliAgentCa,
  type VolumeMount,
} from './volume-mounts.js';

export {
  DEFAULT_SESSION_NAME,
  MAINTENANCE_SESSION_NAME,
  sessionInputDirName,
} from './session-names.js';
export {
  SECRET_FILES,
  buildVolumeMounts,
  copyTileScriptsToFlatDir,
} from './volume-mounts.js';

interface BuildContainerArgsResult {
  args: string[];
  // Always set — cleanup() is a no-op when no secret env-file was
  // written, so callers can invoke it unconditionally on container
  // exit without a null-check.
  cleanup: () => void;
  // True when at least one `ONECLI_MANAGED_VARS` credential was forwarded as
  // an `onecli-managed` placeholder (real value withheld from the container).
  // The caller MUST fail the spawn closed if the gateway proxy is then not
  // actually applied (`applyOneCliToSpawn` returns false), because a withheld
  // credential would otherwise go out as a dead placeholder on a direct
  // request (REQUEST_DENIED). Distinct from "OneCLI is configured": untrusted
  // spawns forward no vars, so nothing is placeholdered and no fail-close is
  // owed even when OneCLI is configured.
  managedPlaceholdersApplied: boolean;
}

function buildContainerArgs(
  mounts: VolumeMount[],
  containerName: string,
  group: RegisteredGroup,
  isMain: boolean,
  sessionName: string,
  replyToMessageId?: string,
  chatJid?: string,
  continuationCycleId?: string,
  attributionToken?: string,
  taskAgentModel?: string | null,
): BuildContainerArgsResult {
  const args: string[] = ['run', '-i', '--rm', '--name', containerName];

  // Resource limits and filesystem restrictions for untrusted containers
  if (!isMain && !group.containerConfig?.trusted) {
    args.push(
      '--memory',
      '512m', // 512MB RAM hard limit
      '--memory-swap',
      '512m', // no swap
      '--cpus',
      '1', // 1 CPU core
      '--pids-limit',
      '256', // prevent fork bombs
      '--read-only', // immutable root filesystem
      '--tmpfs',
      '/tmp:size=64m', // writable /tmp via tmpfs (needed for input.json)
    );
    // Group folder is read-only for untrusted (set above).
    // Agent can read CLAUDE.md/skills but can't write 7GB of numbers.
  } else {
    // Trusted/main: cap memory to prevent host OOM when multiple containers
    // run in parallel. The NAS has 7.5GB RAM total; without a cap, multiple
    // Claude SDK processes can exhaust host memory and trigger kernel OOM
    // killer (SIGKILL exit 137). 1.5GB is plenty for Claude Code + skills.
    args.push(
      '--memory',
      '1536m', // 1.5GB RAM hard limit
      '--memory-swap',
      '2048m', // allow 512MB swap as buffer
    );
  }

  // Pass host timezone so container's local time matches the user's
  args.push('-e', `TZ=${TIMEZONE}`);

  // Credential tiers:
  //   Main/Trusted: GITHUB_TOKEN for GitHub-via-`gh`-CLI, plus the
  //                 per-tile API keys below. Google (Gmail, Calendar,
  //                 Tasks, Drive) needs no container credential — the
  //                 OneCLI gateway injects and refreshes the Bearer on
  //                 the wire (jbaruch/nanoclaw#638).
  //   Other:        nothing (Anthropic via proxy only).
  //
  // All other host-side credentials (GOOGLE_*, RECLAIM_*, TRIPIT_*,
  // OPENAI_*) stay on the host. Scripts that need them run host-side
  // via IPC.
  const isTrusted = group.containerConfig?.trusted === true;

  const CONTAINER_VARS = [
    // Forwarded into main/trusted containers so the `gh` CLI authenticates
    // automatically. Same PAT the host-side github_backup handler uses
    // — the operator must expand its scope to cover `Contents: write`
    // (host-side git push) AND `Issues: write` + `Pull requests: write`
    // (container-side `gh issue edit/comment` from the cost-monitor
    // dashboard skills). See .env.example for the scope documentation.
    // OneCLI migration target — see jbaruch/nanoclaw#564 for the path
    // off the "secret-in-container-environ-for-spawn-lifetime" exposure.
    'GITHUB_TOKEN',
    // Forwarded for the `jbaruch/nanoclaw-travel` per-chat overlay
    // tile (added via `containerConfig.additionalTiles`). The precheck
    // reads `BYAIR_MCP_URL` (personal MCP link from
    // https://byairapp.com/mcp/, includes the API key inline) to poll
    // flight status via the byAir streamable-HTTP endpoint. Marked
    // SECRET below so it goes through `--env-file` instead of `-e KEY=...`.
    'BYAIR_MCP_URL',
    // Same tile. The precheck reads `GOOGLE_MAPS_API_KEY` to query the
    // Distance Matrix API for traffic-aware time-to-leave
    // (`departure_time=now`, `traffic_model=best_guess`). Generated at
    // https://console.cloud.google.com/apis/credentials with the
    // Distance Matrix API enabled on a billing-attached project. Marked
    // SECRET below.
    'GOOGLE_MAPS_API_KEY',
    // `jbaruch/nanoclaw-travel` tile: `maps_client.py` uses TomTom
    // (geocode + calculateRoute on `api.tomtom.com`) as the backup behind
    // Google Maps, and the `drive-planner` skill routes through it.
    // Generated at https://developer.tomtom.com. Marked SECRET below.
    'TOMTOM_API_KEY',
    // YouTube Data API v3 key — read by the admin tile's
    // `youtube-comment-check` skill, which calls the native API
    // (commentThreads.list + videos.list) directly
    // (jbaruch/nanoclaw-admin#339). Marked SECRET below.
    'YOUTUBE_API_KEY',
    // Sessionize keys for the `jbaruch/nanoclaw-conferences` tile's
    // deterministic check-cfps pipeline: `discover-open-cfps.py` reads
    // SESSIONIZE_SPEAKER_KEY (speaker-profile open-CFP discovery) and
    // `verify-sessionize.py` reads SESSIONIZE_EVENT_API_KEY (per-slug
    // deadline verification). Both marked SECRET above so they route
    // through the env-file instead of `-e KEY=...`.
    'SESSIONIZE_SPEAKER_KEY',
    'SESSIONIZE_EVENT_API_KEY',
  ];

  const varsToForward = isMain || isTrusted ? CONTAINER_VARS : [];

  const envFromFile = readEnvFile(CONTAINER_VARS);
  // Partition forwarded vars: secrets get materialized into a 0600
  // env-file (passed via `--env-file`) so they don't appear on the
  // docker process command line; non-secrets stay on `-e KEY=value`.
  // See the SECRET_CONTAINER_VARS docstring for the policy.
  // #640: when OneCLI is configured the gateway swaps real secrets in on the
  // outbound request, so OneCLI-managed vars enter the container as a non-empty
  // placeholder and their real value never touches the container environ.
  // Falls back to real-value forwarding when OneCLI is unconfigured.
  //
  // The gate mirrors EXACTLY the condition that decides whether the gateway
  // proxy is applied to the spawn (`isOneCliConfigured()` at the call site):
  // placeholdering must happen iff the proxy-application branch runs, else a
  // placeholder would go out on a DIRECT request → REQUEST_DENIED. When OneCLI
  // is unconfigured (dev, or `ONECLI_URL`/`ONECLI_API_KEY` unset) the proxy is
  // never applied and vars fall back to real-value forwarding. The remaining
  // gap — `applyOneCliToSpawn` itself returning false (gateway unreachable) —
  // is closed at the call site by failing the spawn when
  // `managedPlaceholdersApplied` is true (see below).
  const oneCliManagesSecrets = isOneCliConfigured();
  let managedPlaceholdersApplied = false;
  const secretEnv: Record<string, string> = {};
  for (const varName of varsToForward) {
    const managedPlaceholder = oneCliManagesSecrets
      ? ONECLI_MANAGED_VARS.get(varName)
      : undefined;
    if (managedPlaceholder !== undefined) {
      args.push('-e', `${varName}=${managedPlaceholder}`);
      managedPlaceholdersApplied = true;
      continue;
    }
    const value = process.env[varName] || envFromFile[varName];
    if (!value) continue;
    if (SECRET_CONTAINER_VARS.has(varName)) {
      secretEnv[varName] = value;
    } else {
      args.push('-e', `${varName}=${value}`);
    }
  }
  const secretEnvFile = buildSecretEnvFile(secretEnv);
  // Position of `--env-file` in argv is irrelevant for override
  // semantics — docker resolves `-e` over `--env-file` regardless of
  // order. We append here for readability of the assembled command;
  // a future caller adding a non-secret `-e` after this point won't
  // accidentally override a secret because the names don't overlap
  // (SECRET_CONTAINER_VARS membership is the partition rule).
  if (secretEnvFile) args.push(...secretEnvFile.args);

  // Select which model + effort the agent-runner's SDK query() uses.
  // The runner reads `process.env.AGENT_MODEL` and `process.env.AGENT_EFFORT`
  // — see constants at the top of this file. Keeping these on the env
  // (not baked into the agent image) lets model bumps / effort retuning
  // ship with an orchestrator rebuild only.
  //
  // #395: per-group `containerConfig.agentModel` override. When set and
  // valid, replaces the global AGENT_MODEL for this group's spawn only;
  // an unknown-prefix value falls back to the global default (see
  // `resolvePerGroupAgentModel` for branching).
  //
  // #418: emit one info log per spawn UNCONDITIONALLY — every container
  // spawn records the resolved `effectiveAgentModel` so cost / latency
  // attribution has a per-spawn audit trail with no silent-default
  // blind spot (the prior `if (override)`-gated log left default spawns
  // invisible to the audit). The `source` field distinguishes
  // default-vs-override so the log line is self-explaining without
  // joining against group config.
  // #509: per-session-slot model tier. Maintenance spawns
  // (sessionName === MAINTENANCE_SESSION_NAME) consult
  // `maintenanceAgentModel` first, then fall through to the existing
  // `agentModel` → AGENT_MODEL ladder. Non-maintenance spawns are
  // unchanged byte-for-byte from the pre-#509 behavior.
  // Phase 3 (#509): per-task override beats session-level when
  // `input.taskAgentModel` is set AND its resolved value differs from
  // the session-level fallback (sole writer: task-scheduler.ts plumbs
  // `task.agent_model` here). The audit-log `source` field is
  // `task_override` ONLY in that "different from session" case;
  // unknown-prefix values and per-row values that deliberately match
  // the session-level fallback fall through to the session-level
  // source so the log doesn't claim a routing change that didn't
  // happen. Unknown-prefix falls back to the session-level value
  // (NOT the global default), so a typo on a maintenance-pinned
  // group lands on the maintenance value, not Opus.
  const isMaintenanceSpawn = sessionName === MAINTENANCE_SESSION_NAME;
  // #613 Stage 1 — Claude tier-down. The spawn's BASE model is keyed to
  // the group's trust tier (main → global default, trusted → Sonnet,
  // untrusted → Haiku); per-group / maintenance / per-task overrides still
  // win on top via resolveSessionAgentModel. This applies to every spawn,
  // including scheduled/maintenance runs — task-scheduler selects the group
  // by `task.group_folder` and derives `isMain` from it, so a scheduled
  // task bases on its registered group's tier. Per-task `agentModel:`
  // frontmatter (trivial → Haiku, substantive → Sonnet) overrides that base.
  const tierBaseModel = resolveTierBaseModel(isMain, isTrusted, AGENT_MODEL);
  const { effective: effectiveAgentModel, source: agentModelSource } =
    resolveSessionAgentModel(
      group.containerConfig,
      isMaintenanceSpawn,
      tierBaseModel,
      taskAgentModel,
    );
  logger.info(
    {
      groupFolder: group.folder,
      sessionName,
      tier: isMain ? 'main' : isTrusted ? 'trusted' : 'untrusted',
      agentModel: effectiveAgentModel,
      // `tierBaseModel` is the tier-keyed base passed as the resolver's
      // default; `globalDefault` is the true orchestrator-wide AGENT_MODEL.
      // When they differ, a tier-down is in effect and `source` reads
      // `global_default` against the tier base (no per-group/task override).
      tierBaseModel,
      globalDefault: AGENT_MODEL,
      source: agentModelSource,
      // Include the per-task raw value so an unknown-prefix fallback
      // shows up next to the resolved value in the audit log — this
      // is the diagnostic surface for "I set task X to haiku but the
      // spawn still ran on sonnet".
      taskAgentModel: taskAgentModel ?? null,
    },
    'Container spawn AGENT_MODEL resolved',
  );
  args.push('-e', `AGENT_MODEL=${effectiveAgentModel}`);
  args.push('-e', `AGENT_EFFORT=${AGENT_EFFORT}`);

  // USE_CUSTOM_PROMPT (#113) — gates the per-tier custom system prompt
  // path in agent-runner. Default OFF: production behavior is unchanged
  // unless explicitly opted in. Resolution order:
  //   1. Per-group `containerConfig.useCustomPrompt` if set (true or false)
  //      — explicit override always wins.
  //   2. Global env `USE_CUSTOM_PROMPT_FOR_MAIN=1` enables ONLY the main
  //      container; trusted/untrusted stay on the preset.
  //   3. Otherwise OFF.
  // The agent-runner reads `process.env.USE_CUSTOM_PROMPT === '1'` and
  // falls back to the preset path on any other value (or absence).
  const globalUseCustomForMain = process.env.USE_CUSTOM_PROMPT_FOR_MAIN === '1';
  const useCustomPromptResolved =
    typeof group.containerConfig?.useCustomPrompt === 'boolean'
      ? group.containerConfig.useCustomPrompt
      : globalUseCustomForMain && isMain;
  if (useCustomPromptResolved) {
    args.push('-e', 'USE_CUSTOM_PROMPT=1');
    logger.info(
      {
        groupFolder: group.folder,
        source:
          typeof group.containerConfig?.useCustomPrompt === 'boolean'
            ? 'containerConfig'
            : 'USE_CUSTOM_PROMPT_FOR_MAIN',
      },
      'USE_CUSTOM_PROMPT enabled — agent-runner will load per-tier custom prompt',
    );
  }

  // Forward the orchestrator's authoritative assistant identity into the
  // agent container so the agent-runner can prepend an identity preamble
  // to systemPromptAppend (see container/agent-runner/src/index.ts —
  // buildIdentityPreamble). These are not secrets; pass them directly via
  // -e rather than going through the SECRET_FILES env-file plumbing.
  // Without this, untrusted-tier containers lacking explicit identity
  // statements in their persona files have been observed templating
  // themselves from fictional examples in tile rules (e.g. @AyeAye /
  // @AyeAyeSureBot from nanoclaw-core 0.1.94).
  args.push('-e', `ASSISTANT_NAME=${ASSISTANT_NAME}`);
  // Forward the full alias list (not just the primary entry) so the
  // agent-runner's identity preamble can teach the agent every handle
  // that resolves to it (#464). Single-handle installs forward a
  // single token verbatim; multi-handle installs see the joined
  // comma-separated form, which the agent-runner re-parses via
  // `parseUsernames` in `identity-preamble.ts`.
  args.push('-e', `ASSISTANT_USERNAME=${ASSISTANT_USERNAMES.join(',')}`);

  // Tell agent-runner whether the host is running the optional
  // observer pipeline (src/observer.ts). When true, agent-runner
  // emits raw thinking / text / tool_use input / tool_result preview
  // content on stderr — those payloads carry credentials, tokens,
  // and personal data per jbaruch/coding-policy: no-secrets, so they
  // are intentionally disclosed only when the operator explicitly
  // opted into observation by setting OBSERVER_CHAT_JID. With the
  // flag unset (default), agent-runner logs metadata only (block
  // counts, tool name, ok/error status). Same env-source order as
  // observer.ts so a setting in either process.env or .env file
  // toggles both halves consistently.
  const observerHostJid =
    process.env.OBSERVER_CHAT_JID ||
    readEnvFile(['OBSERVER_CHAT_JID']).OBSERVER_CHAT_JID;
  if (observerHostJid) {
    args.push('-e', 'OBSERVER_ENABLED=1');
  }

  // Kill-auto-compaction master flag (issue #104, design at
  // docs/proposals/kill-auto-compaction.md). When the orchestrator's
  // `ENABLE_THRESHOLD_NUKE` is set, the SDK's auto-compaction is
  // disabled — the orchestrator's threshold-nuke handshake replaces
  // it end-to-end. When the flag is off (default), DISABLE_COMPACT is
  // NOT set, auto-compaction stays on, and the orchestrator's
  // threshold detector runs in observe-only mode (telemetry +
  // checkpoint format validation, no destructive nuke). The two
  // halves are gated by the same flag because they must ship
  // together — disabling compaction without the replacement leaves
  // long sessions with no overflow safety net.
  if (ENABLE_THRESHOLD_NUKE) {
    args.push('-e', 'DISABLE_COMPACT=1');
    // Deliberately NOT forwarding CLAUDE_CODE_AUTO_COMPACT_WINDOW here
    // (issue #252). DISABLE_COMPACT=1 only suppresses the SDK's
    // `isAboveAutoCompactThreshold` flag — `Jn()` still reads the env
    // var unconditionally and feeds it into the `isAtBlockingLimit`
    // check (`window − reserved_for_output − e_7`, ~window − 30k).
    // Clamping the working window at or near the orchestrator's 800k
    // nuke threshold drops blocking-limit below the nuke and wedges
    // the request whose response would have triggered the handshake.
    // Letting the SDK fall back to the model context window (1M for
    // opus[1m]) keeps blocking-limit ~970k, well clear of the nuke.
  } else {
    args.push(
      '-e',
      `CLAUDE_CODE_AUTO_COMPACT_WINDOW=${AGENT_AUTO_COMPACT_WINDOW}`,
    );
  }

  // Pass chat JID so container scripts know which group they're in
  if (chatJid) {
    args.push('-e', `NANOCLAW_CHAT_JID=${chatJid}`);
  }

  // Pass reply-to message ID so the first IPC send_message appears as a Telegram reply
  if (replyToMessageId) {
    args.push('-e', `NANOCLAW_REPLY_TO_MESSAGE_ID=${replyToMessageId}`);
  }

  // Continuation marker for self-resuming cycles (#93/#130). Both env
  // vars are emitted together when the scheduled-task row carried a
  // non-NULL `continuation_cycle_id`; neither is emitted on a fresh
  // invocation. The calling skill checks both signals (env vars + the
  // prompt prefix it parses out of the task prompt) and fails closed to
  // "fresh invocation" if they disagree, so the env presence is
  // load-bearing — never paper over a missing value with a default.
  if (continuationCycleId) {
    args.push('-e', 'NANOCLAW_CONTINUATION=1');
    args.push('-e', `NANOCLAW_CONTINUATION_CYCLE_ID=${continuationCycleId}`);
  }

  // Route API traffic through the credential proxy (containers never see real secrets).
  // When an attribution token is supplied (#479 / ligolnik#125), embed it as a
  // path prefix so the proxy can map each request back to {group, tier,
  // session, task_id} for the JSONL usage log. The Claude SDK joins endpoint
  // paths to the base URL, so requests arrive at the proxy as
  // `/c/<token>/v1/messages`. Backward-compatible: a missing token degrades
  // to the un-prefixed URL and the proxy records `group: "unknown"`.
  const baseProxyUrl = `http://${CONTAINER_HOST_GATEWAY}:${CREDENTIAL_PROXY_PORT}`;
  const proxyBaseUrl = attributionToken
    ? `${baseProxyUrl}/c/${attributionToken}`
    : baseProxyUrl;
  args.push('-e', `ANTHROPIC_BASE_URL=${proxyBaseUrl}`);

  // Mirror the host's auth method with a placeholder value.
  // API key mode: SDK sends x-api-key, proxy replaces with real key.
  // OAuth mode:   SDK exchanges placeholder token for temp API key,
  //               proxy injects real OAuth token on that exchange request.
  const authMode = detectAuthMode();
  if (authMode === 'api-key') {
    args.push('-e', 'ANTHROPIC_API_KEY=placeholder');
  } else {
    args.push('-e', 'CLAUDE_CODE_OAUTH_TOKEN=placeholder');
  }

  // Runtime-specific args for host gateway resolution
  args.push(...hostGatewayArgs());

  // Run as host user so bind-mounted files are accessible.
  // In DooD, process.getuid() returns the orchestrator container's uid (1000),
  // not the actual host user. HOST_UID/HOST_GID override this.
  const effectiveUid = HOST_UID ?? process.getuid?.();
  const effectiveGid = HOST_GID ?? process.getgid?.();
  if (effectiveUid != null && effectiveUid !== 0 && effectiveUid !== 1000) {
    args.push('--user', `${effectiveUid}:${effectiveGid}`);
    args.push('-e', 'HOME=/home/node');
  }

  for (const mount of mounts) {
    if (mount.readonly) {
      args.push(...readonlyMountArgs(mount.hostPath, mount.containerPath));
    } else {
      args.push('-v', `${mount.hostPath}:${mount.containerPath}`);
    }
  }

  // #746: CONTAINER_IMAGE is intentionally NOT pushed here. It is appended by
  // the caller AFTER applyOneCliToSpawn runs, so the OneCLI `-e`/`-v`/
  // `--add-host` flags (which the SDK appends) land as `docker run` options
  // before the image token rather than as the container COMMAND after it.
  return {
    args,
    cleanup: secretEnvFile ? secretEnvFile.cleanup : () => {},
    managedPlaceholdersApplied,
  };
}

export async function runContainerAgent(
  group: RegisteredGroup,
  input: ContainerInput,
  onProcess: (proc: ChildProcess, containerName: string) => void,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<ContainerOutput> {
  const startTime = Date.now();

  const groupDir = spawnSetup(() => {
    const dir = resolveGroupFolderPath(group.folder);
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  });

  const sessionName = input.sessionName ?? DEFAULT_SESSION_NAME;

  // Clean up stale _reply_to file from previous container runs in THIS
  // session. The file must match the path the container reads — with
  // per-session input dirs the container's `/workspace/ipc/input/_reply_to`
  // maps to `<ipc>/<group>/input-<sessionName>/_reply_to`, so the cleanup
  // must target the same session-scoped path. A cleanup against the legacy
  // shared `input/` path would leave the real file in place, and a
  // scheduled task with no replyToMessageId would quote a random old
  // message from a prior run.
  const replyToFile = spawnSetup(() =>
    path.join(
      resolveGroupIpcPath(group.folder),
      sessionInputDirName(sessionName),
      '_reply_to',
    ),
  );
  try {
    fs.unlinkSync(replyToFile);
  } catch (err) {
    if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
    /* file doesn't exist — fine */
  }

  const mounts = spawnSetup(() =>
    buildVolumeMounts(group, input.isMain, input.chatJid, sessionName),
  );

  // #305 Phase 2a — cadence-registry rebuild. Walks the per-session
  // skills tree that `buildVolumeMounts` just published, parses each
  // SKILL.md's `cadence:` / `priority:` frontmatter, and idempotently
  // upserts declared schedules as `scheduled_tasks` rows with
  // `source = 'cadence-registry'`. Owner-scheduled rows (`source =
  // 'schedule-task'` from the existing `schedule-task` IPC path) are
  // untouched.
  //
  // Gated to `DEFAULT_SESSION_NAME` only. The maintenance session has
  // a filtered skill set (`MAINTENANCE_SKILL_BLOCKLIST` applied inside
  // `buildVolumeMounts`), so a maintenance spawn would see fewer
  // skills than the default session; running the rebuild from both
  // would race on the same group_folder rows and the maintenance view
  // would erase cadences declared by skills the default view installs
  // fine. Maintenance is a transient, short-lived spawn for
  // housekeeping — it should not claim authority over the registry.
  if (sessionName === DEFAULT_SESSION_NAME) {
    const skillsDir = path.join(
      DATA_DIR,
      'sessions',
      group.folder,
      sessionName,
      '.claude',
      'skills',
    );
    // Errors propagate by design: the rebuild's per-skill failures are
    // already collected into the `errors` array (cron parse, IANA
    // resolution, etc.) and don't throw; anything that throws past
    // that point is a real fault (db corruption, fs walk fails on
    // EACCES, etc.) and should fail the spawn loudly per the no-error
    // -suppression rule.
    const cadenceResult = spawnSetup(() =>
      rebuildCadenceRegistryForGroup({
        groupFolder: group.folder,
        chatJid: input.chatJid,
        // Cadence-registry rows derive from tile content delivered via
        // the staging→promote→publish→update pipeline (host-vetted,
        // owner-trusted) rather than any in-container agent action — so
        // they unwrap at fire time the same way legacy heartbeat seeders
        // do. The trust-boundary semantics for fire-time wrapping live
        // on `created_by_role`; the registry-vs-IPC provenance lives on
        // the new `source` column. Two columns, two questions.
        createdByRole: 'owner',
        skillsDir,
        computeNextRun: defaultComputeNextRun,
        now: () => new Date(),
      }),
    );
    if (
      cadenceResult.inserted > 0 ||
      cadenceResult.updated > 0 ||
      cadenceResult.deleted > 0 ||
      cadenceResult.errors.length > 0
    ) {
      logger.info(
        {
          groupFolder: group.folder,
          chatJid: input.chatJid,
          sessionName,
          ...cadenceResult,
        },
        'cadence-registry rebuilt',
      );
    }
  }

  const safeName = group.folder.replace(/[^a-zA-Z0-9-]/g, '-');
  // Suffix the container name with sessionName (when non-default) so that
  // `docker ps` makes it obvious which slot a running container occupies.
  // Main-group parallelism means two containers can share the group folder;
  // the sessionName tag distinguishes them.
  const sessionSuffix =
    sessionName === DEFAULT_SESSION_NAME ? '' : `-${sessionName}`;
  const containerName = `nanoclaw-${safeName}${sessionSuffix}-${Date.now()}`;

  // #213 Phase A spawn-collision detector. After a graceful-shutdown
  // handoff, the new orchestrator's GroupQueue starts empty — it has
  // no in-memory record of the adopted containers. If a new message
  // for one of those groups arrives during the handoff window, this
  // spawn path will run alongside the still-finishing adopted
  // container, racing on IPC and producing duplicate user-visible
  // replies. Phase B (in-memory adoption + docker-ps polling for
  // natural exit) closes that race; Phase A defers the work behind
  // this detector. When a collision is observed, the WARN message
  // names jbaruch/nanoclaw#213 directly so an operator hitting it
  // weeks/months from now knows exactly where to look — the
  // detector is the trigger to ship Phase B.
  if (isHandoffActive()) {
    const collisionPrefix = `nanoclaw-${safeName}${sessionSuffix}-`;
    // Use CONTAINER_RUNTIME_BIN, not a hard-coded 'docker', so the
    // detector follows the codebase's "swap runtimes by changing one
    // file" contract documented at the top of `container-runtime.ts`.
    const psResult = spawnSync(
      CONTAINER_RUNTIME_BIN,
      ['ps', '--format', '{{.Names}}', '--filter', `name=${collisionPrefix}`],
      { stdio: ['pipe', 'pipe', 'pipe'], encoding: 'utf-8' },
    );
    // spawnSync doesn't throw on non-zero exit; an unreachable docker
    // daemon, permission error, or missing binary surfaces as
    // `psResult.error` (spawn-side) or `psResult.status !== 0`
    // (process-side). Either way, treat the detector as unavailable
    // for this spawn and continue — the spawn itself must NOT be
    // blocked by an observability surface.
    if (psResult.error || psResult.status !== 0) {
      logger.debug(
        {
          issue: 'jbaruch/nanoclaw#213',
          err: psResult.error,
          status: psResult.status,
          stderr: psResult.stderr?.toString().slice(0, 500),
        },
        'Spawn-collision detector unavailable (docker ps failed) — skipping',
      );
    } else {
      const collisions = (psResult.stdout || '')
        .split('\n')
        .map((n) => n.trim())
        .filter((n) => n.startsWith(collisionPrefix));
      if (collisions.length > 0) {
        logger.warn(
          {
            issue: 'jbaruch/nanoclaw#213',
            phase: 'A',
            newContainer: containerName,
            collidingWith: collisions,
            groupFolder: group.folder,
            sessionName,
          },
          'Spawn collision during graceful-shutdown handoff window — ' +
            'an adopted container from the prior orchestrator is still ' +
            'running for this group/session. Both will respond to inbound ' +
            'IPC, producing duplicate user-visible replies. This is the ' +
            'Phase B race the #213 fix deferred; observed firing means it ' +
            'is time to implement in-memory adoption (register adopted ' +
            'containers in GroupQueue, poll docker ps for natural exit). ' +
            'See src/handoff.ts and the comment block above this log.',
        );
      }
    }
  }
  // Per-spawn attribution token for the credential proxy's usage log
  // (#479 / ligolnik#125). Lives only as long as the container; unregistered
  // on close + spawn-error. The token is generated here so the URL embedded
  // in the container env is unique per spawn, even across restarts of the
  // same group/session.
  const trustTier: TrustTier = input.isMain
    ? 'main'
    : group.containerConfig?.trusted === true
      ? 'trusted'
      : 'untrusted';
  const attributionToken = registerContainer({
    group: group.folder,
    tier: trustTier,
    session: sessionName,
    task_id: input.isScheduledTask ? sessionName : null,
    // #479 sub-#1: prefer the gate-verdict's trigger message id
    // (`triggerMessageId`) so attribution lands on the message that
    // actually caused the spawn. Fall back to `replyToMessageId` for
    // callers that don't yet plumb the gate verdict — rare, since
    // `evaluateGateChain` already computes it; and finally null for
    // scheduled tasks / IPC scripts / housekeeping.
    message_id: input.triggerMessageId ?? input.replyToMessageId ?? null,
  });

  // Wrap the spawn-path so a sync throw between registration and the
  // close/error handler attach (mostly fs.mkdirSync calls below)
  // can't leak the registry entry. The promise itself never rejects
  // (close/error handlers always resolve()), so this catch only fires
  // on the early-throw window. The handlers below also call
  // unregisterContainer; unregister is idempotent regardless.
  try {
    const {
      args: containerArgs,
      cleanup: cleanupSecretEnvFile,
      managedPlaceholdersApplied,
    } = buildContainerArgs(
      mounts,
      containerName,
      group,
      input.isMain,
      sessionName,
      input.replyToMessageId,
      input.chatJid,
      input.continuationCycleId,
      attributionToken,
      input.taskAgentModel,
    );

    // #564 groundwork: when OneCLI is configured, mutate containerArgs to add
    // HTTPS_PROXY env, mount the OneCLI CA bundle, and add
    // host.docker.internal mapping. The synchronous gate keeps the pre-#635
    // spawn path microtask-free when OneCLI is off — relevant for tests that
    // emit 'close' between `runContainerAgent` invocation and the spawn
    // registering its close handler.
    // #640: whenever OneCLI is configured, the agent's outbound traffic
    // traverses the OneCLI gateway so external creds swap in at the MITM and
    // never touch the agent environ. (The earlier INCIDENT-746 break — the
    // agent's call to its local credential-proxy getting routed through the
    // gateway → ECONNRESET — is prevented by the NO_PROXY host-gateway bypass
    // in applyOneCliToSpawn; the LLM 401 break was the gateway-injected
    // ANTHROPIC_API_KEY sentinel, stripped there too.) The synchronous
    // isOneCliConfigured() gate keeps the spawn path microtask-free in dev
    // where OneCLI is unconfigured.
    if (isOneCliConfigured()) {
      // Any failure applying OneCLI must clean up the 0600 secret env-file that
      // buildContainerArgs already materialized — the close/error handlers that
      // otherwise clean it up aren't installed yet, so a bare throw here would
      // leave forwarded secrets on disk (no-secrets). This covers the
      // fail-closed throws below AND a throw from applyOneCliToSpawn or
      // mountOneCliAgentCa's fs writes / getOneCliOutboundConfig.
      try {
        const proxyApplied = await applyOneCliToSpawn(containerArgs, trustTier);
        if (proxyApplied) {
          // The agent's outbound HTTPS now traverses the OneCLI MITM gateway;
          // deliver the CA so it can validate the MITM cert. The SDK's own CA
          // mount is broken under DooD (see mountOneCliAgentCa) — without a
          // trusted CA EVERY external HTTPS call fails TLS, so a CA-delivery
          // failure is fail-closed, same as a withheld managed credential.
          const caMounted = await mountOneCliAgentCa(containerArgs, trustTier);
          if (!caMounted) {
            // #784: an operational fail-closed failure (gateway down) — typed
            // directly so runAgent returns 'error' and the queue retries.
            throw new ContainerAgentError(
              'OneCLI agent proxy applied but its MITM CA could not be ' +
                'delivered to the container — every external HTTPS call would ' +
                'fail TLS. Confirm the OneCLI gateway is reachable (`curl -sf ' +
                '$ONECLI_URL/health` on the NAS); the queue retries with backoff.',
            );
          }
        } else if (managedPlaceholdersApplied) {
          // #640 fail-closed: buildContainerArgs replaced managed credentials
          // with placeholders under this same gate, withholding the real values.
          // The gateway proxy could NOT be applied, so those placeholders would
          // go out on DIRECT requests and fail (REQUEST_DENIED). Refuse to spawn
          // a dead-credentialed container. The queue retries with backoff
          // (index.ts catch → 'error'), so a transient gateway blip self-heals;
          // untrusted spawns forward no vars, so this path is skipped for them.
          // #784: operational fail-closed (gateway down) — typed directly.
          throw new ContainerAgentError(
            'OneCLI agent proxy could not be applied to this spawn — managed ' +
              'credentials were withheld as placeholders and would fail on a ' +
              'direct request. Confirm the OneCLI gateway is reachable (`curl ' +
              '-sf $ONECLI_URL/health` on the NAS) and the tier agent exists ' +
              '(`onecli agents list`), then retry.',
          );
        }
      } catch (err) {
        cleanupSecretEnvFile();
        throw err;
      }
    }

    // #746: append the image LAST — after OneCLI's flags — so `docker run`
    // parses those flags as run options, not as the container COMMAND.
    // buildContainerArgs deliberately returns pre-image args for this reason.
    containerArgs.push(CONTAINER_IMAGE);

    logger.debug(
      {
        group: group.name,
        containerName,
        mounts: mounts.map(
          (m) =>
            `${m.hostPath} -> ${m.containerPath}${m.readonly ? ' (ro)' : ''}`,
        ),
        containerArgs: containerArgs.join(' '),
      },
      'Container mount configuration',
    );

    logger.info(
      {
        group: group.name,
        containerName,
        mountCount: mounts.length,
        isMain: input.isMain,
      },
      'Spawning container agent',
    );

    const logsDir = path.join(groupDir, 'logs');
    fs.mkdirSync(logsDir, { recursive: true });

    // Per-spawn streaming log under host-logs/. Distinct from the
    // post-exit summary written at `logsDir` below — the streaming file
    // captures output line-by-line as the container produces it, so the
    // admin tile can read what a stuck container is actually doing
    // without waiting for it to exit. Failure to open the stream is
    // non-fatal: the container still runs, we just lose the host-logs
    // copy for this spawn (the in-memory buffers + post-exit summary
    // remain unaffected).
    const spawnStartedAt = new Date();
    // `input.sessionName` is optional on the type but is always set by
    // every caller that actually invokes runContainerAgent (default or
    // maintenance). Fall back to the canonical default to keep the file
    // path deterministic if a future caller forgets to stamp the field.
    const streamSessionName = input.sessionName || DEFAULT_SESSION_NAME;
    const streamLogPath = containerLogPath(
      group.folder,
      streamSessionName,
      spawnStartedAt,
    );
    let streamLog: fs.WriteStream | null = null;
    try {
      fs.mkdirSync(path.dirname(streamLogPath), { recursive: true });
      streamLog = fs.createWriteStream(streamLogPath, { flags: 'a' });
      // Attach an error listener BEFORE the first write. Stream errors
      // emit async (e.g. ENOENT if the parent dir is wiped between
      // mkdir and create, EBADF if the fd is reaped) — without a
      // listener, Node's default handler is "throw uncaught exception"
      // which would crash the orchestrator on a logging path. This must
      // never happen: lose the streamed log, keep serving the user.
      streamLog.on('error', (err) => {
        logger.warn(
          { err, group: group.name, streamLogPath },
          'host-logs stream errored mid-spawn; dropping per-spawn stream',
        );
        // Null the local ref so subsequent writes from the data
        // listeners no-op rather than try to push into a broken stream.
        streamLog = null;
      });
      streamLog.write(
        [
          `=== Container Stream Log ===`,
          `Group: ${group.name}`,
          `Folder: ${group.folder}`,
          `Session: ${streamSessionName}`,
          `Container: ${containerName}`,
          `Start: ${spawnStartedAt.toISOString()}`,
          `=== STDOUT/STDERR (line-prefixed) ===`,
          ``,
        ].join('\n'),
      );
    } catch (err) {
      if (!isFsErrorWithCode(err, CR_FS_CODES)) throw err;
      logger.warn(
        { err, group: group.name, streamLogPath },
        'host-logs stream open failed; container will run without per-spawn stream',
      );
      streamLog = null;
    }

    // Buffered line writer per stream. Container output isn't line-
    // aligned (a single `data` event can split a line, or contain many),
    // so we buffer until we see `\n` and emit `[OUT] ` / `[ERR] ` per
    // complete line. Trailing partial line is flushed on container exit.
    //
    // The buffer is bounded: a misbehaving container that emits megabytes
    // without a newline (e.g. binary garbage, a long single-line log
    // dump) would otherwise grow the orchestrator's heap unboundedly.
    // Cap at 64 KB per stream — well above typical line lengths but
    // small enough that even pathological output flushes quickly.
    const LINE_BUFFER_MAX = 64 * 1024;
    const makeLinePrefixer = (prefix: string) => {
      let buffer = '';
      const writeLines = (chunk: string) => {
        if (!streamLog) return;
        buffer += chunk;
        let nl: number;
        while ((nl = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, nl);
          buffer = buffer.slice(nl + 1);
          streamLog.write(`${prefix} ${stripAnsi(line)}\n`);
        }
        // Cap the buffer: if no newline appeared and the buffer crossed
        // the limit, flush the entire current buffer as a synthetic
        // line. Tagged with `[…cap…]` so a reader knows the line was
        // not delimited by a real newline (might cut mid-token).
        if (buffer.length > LINE_BUFFER_MAX) {
          streamLog.write(`${prefix} […cap…] ${stripAnsi(buffer)}\n`);
          buffer = '';
        }
      };
      const flush = () => {
        if (!streamLog || buffer.length === 0) return;
        streamLog.write(`${prefix} ${stripAnsi(buffer)}\n`);
        buffer = '';
      };
      return { writeLines, flush };
    };
    const stdoutPrefixer = makeLinePrefixer('[OUT]');
    const stderrPrefixer = makeLinePrefixer('[ERR]');

    return await new Promise<ContainerOutput>((resolve) => {
      const container = spawn(CONTAINER_RUNTIME_BIN, containerArgs, {
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      onProcess(container, containerName);

      let stdout = '';
      let stderr = '';
      let stdoutTruncated = false;
      let stderrTruncated = false;

      container.stdin.write(JSON.stringify(input));
      container.stdin.end();

      // Streaming output: parse OUTPUT_START/END marker pairs as they arrive
      let parseBuffer = '';
      let newSessionId: string | undefined;
      let outputChain = Promise.resolve();

      container.stdout.on('data', (data) => {
        const chunk = data.toString();

        // Streaming host-logs copy. Best-effort — write errors don't
        // affect the buffer accumulation or marker parsing below.
        stdoutPrefixer.writeLines(chunk);

        // Always accumulate for logging
        if (!stdoutTruncated) {
          const remaining = CONTAINER_MAX_OUTPUT_SIZE - stdout.length;
          if (chunk.length > remaining) {
            stdout += chunk.slice(0, remaining);
            stdoutTruncated = true;
            logger.warn(
              { group: group.name, size: stdout.length },
              'Container stdout truncated due to size limit',
            );
          } else {
            stdout += chunk;
          }
        }

        // Stream-parse for output markers
        if (onOutput) {
          parseBuffer += chunk;
          let startIdx: number;
          while ((startIdx = parseBuffer.indexOf(OUTPUT_START_MARKER)) !== -1) {
            const endIdx = parseBuffer.indexOf(OUTPUT_END_MARKER, startIdx);
            if (endIdx === -1) break; // Incomplete pair, wait for more data

            const jsonStr = parseBuffer
              .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
              .trim();
            parseBuffer = parseBuffer.slice(endIdx + OUTPUT_END_MARKER.length);

            try {
              const parsed: ContainerOutput = JSON.parse(jsonStr);
              if (parsed.newSessionId) {
                newSessionId = parsed.newSessionId;
              }
              hadStreamingOutput = true;
              // #682 — distinguish a TERMINAL result marker from an
              // intermediate streaming PREVIEW. The agent-runner emits a
              // preview marker — the only `writeOutput` carrying
              // `streamText` (container/agent-runner/src/index.ts) — on
              // every throttled assistant-text snapshot; every terminal
              // marker (real result, empty-turn success, #461 silent-stop
              // synthesis, error, precheck-skip) omits `streamText`.
              // `hadStreamingOutput` flips for ANY marker, so it can't
              // tell "delivered a result" from "only previewed"; this
              // flag can. A clean exit that produced only previews never
              // delivered a result — recording it `success` is the
              // silent-success shape #682 closes.
              if (parsed.streamText === undefined) {
                hadTerminalResult = true;
                // #689 — latch: a `requires_delivery` skill stamped this
                // TERMINAL `success` marker `noDelivery` because it ran
                // without delivering any user-facing content. We keep
                // the slot-draining `success` semantics (scheduleClose
                // still fires), but the close handler below downgrades
                // the resolved run to `killed` (retriable). Latched so a
                // later plain session-update success marker can't clear
                // it. Scoped inside the terminal-marker guard (same
                // `streamText === undefined` discriminator as
                // `hadTerminalResult`): the runner only ever stamps
                // `noDelivery` on terminal markers, and a preview marker
                // must never trip the downgrade.
                if (parsed.noDelivery === true) {
                  sawNoDeliveryMarker = true;
                }
                // #901 — same latch for a terminal marker the runner
                // stamped `noModelWork`: the model never ran a turn, so
                // whatever text the result carried (the subscription-cap
                // notice) is not the product of a completed run.
                if (parsed.noModelWork === true) {
                  sawNoModelWorkMarker = true;
                }
              }
              // Activity detected — reset the hard timeout
              resetTimeout();
              // Call onOutput for all markers (including null results)
              // so idle timers start even for "silent" query completions.
              outputChain = outputChain.then(() => onOutput(parsed));
            } catch (err) {
              if (!(err instanceof SyntaxError)) throw err;
              logger.warn(
                { group: group.name, error: err },
                'Failed to parse streamed output chunk',
              );
            }
          }
        }
      });

      container.stderr.on('data', (data) => {
        const chunk = data.toString();
        stderrPrefixer.writeLines(chunk);
        const lines = chunk.trim().split('\n');
        for (const line of lines) {
          if (line) {
            logger.debug({ container: group.folder }, line);
            onAgentLine(group.folder, line);
          }
        }
        // Don't reset timeout on stderr — SDK writes debug logs continuously.
        // Timeout only resets on actual output (OUTPUT_MARKER in stdout).
        if (stderrTruncated) return;
        const remaining = CONTAINER_MAX_OUTPUT_SIZE - stderr.length;
        if (chunk.length > remaining) {
          stderr += chunk.slice(0, remaining);
          stderrTruncated = true;
          logger.warn(
            { group: group.name, size: stderr.length },
            'Container stderr truncated due to size limit',
          );
        } else {
          stderr += chunk;
        }
      });

      let timedOut = false;
      let hadStreamingOutput = false;
      // #682 — set true once a TERMINAL result marker (one without
      // `streamText`) is observed. The precise "did the run actually
      // deliver a terminal result?" signal the clean-exit and timeout
      // classifications below gate on, distinct from `hadStreamingOutput`
      // (which flips for streaming previews too).
      let hadTerminalResult = false;
      // #689 — set true once a terminal marker stamped `noDelivery`
      // (a `requires_delivery` skill that ended a run without delivering
      // user-facing content) is observed. Latched across markers so a
      // trailing plain session-update success can't clear it. The
      // clean-exit and timeout classifications below downgrade such a
      // maintenance run to `killed` despite `hadTerminalResult` being
      // set by the same (success-status) marker.
      let sawNoDeliveryMarker = false;
      // #901 — set true once a terminal marker stamped `noModelWork` (a
      // run in which no assistant message ever carried usage, i.e. the
      // model never ran a turn) is observed. Latched on the same terms
      // as `sawNoDeliveryMarker` and consumed by the same two
      // classifications below, so a did-nothing run resolves `killed`
      // instead of recording the subscription-cap notice as a success.
      let sawNoModelWorkMarker = false;
      // Untrusted containers get shorter timeout (5 min vs 30 min default)
      const UNTRUSTED_TIMEOUT = 300_000;
      const defaultTimeout =
        input.isMain || group.containerConfig?.trusted
          ? CONTAINER_TIMEOUT
          : UNTRUSTED_TIMEOUT;
      const configTimeout = group.containerConfig?.timeout || defaultTimeout;
      // #461 — maintenance-session inactivity timeout. The kill timer
      // here is reset by `resetTimeout()` on every streamed stdout
      // marker (see the `hadStreamingOutput = true; resetTimeout();`
      // block lower in this function), so this is an *inactivity*
      // timeout, not a wall-clock cap — same shape as the existing
      // default-session timer.
      //
      // Maintenance work is single-turn burst-then-quiet, so it
      // doesn't need the user-facing default's `IDLE_TIMEOUT + 30s`
      // graceful-close floor (that floor exists so a multi-turn
      // conversation can drain through `_close`). With the
      // agent-runner's silent-stop synthesis (#461 layer 1), a healthy
      // maintenance run signals teardown within seconds; this shorter
      // window is the backstop for the "SDK hung past graceful close"
      // pathology. Bypass the IDLE_TIMEOUT floor for maintenance only.
      //
      // Per-group `containerConfig.timeout` still wins when set so
      // operators can extend the window for groups with heavy precheck
      // scripts that run silently for longer than the env default.
      const isMaintenanceSession = sessionName === MAINTENANCE_SESSION_NAME;
      const timeoutMs = isMaintenanceSession
        ? group.containerConfig?.timeout || MAINTENANCE_CONTAINER_TIMEOUT
        : Math.max(configTimeout, IDLE_TIMEOUT + 30_000);

      const killOnTimeout = () => {
        timedOut = true;
        logger.error(
          { group: group.name, containerName },
          'Container timeout, stopping gracefully',
        );
        try {
          stopContainer(containerName);
        } catch (err) {
          if (!(err instanceof Error)) throw err;
          logger.warn(
            { group: group.name, containerName, err },
            'Graceful stop failed, force killing',
          );
          container.kill('SIGKILL');
        }
      };

      let timeout = setTimeout(killOnTimeout, timeoutMs);

      // Reset the timeout whenever there's activity (streaming output)
      const resetTimeout = () => {
        clearTimeout(timeout);
        timeout = setTimeout(killOnTimeout, timeoutMs);
      };

      container.on('close', (code) => {
        clearTimeout(timeout);
        // Remove the secret env-file (if any) as soon as docker has
        // exited — the file's only consumer is the docker daemon at
        // spawn time, so the window of exposure ends with the close
        // event. cleanup() is idempotent; the error handler below
        // calls it too in case `close` is skipped (spawn ENOENT etc).
        cleanupSecretEnvFile();
        // Drop the proxy-registry entry for this spawn so dead tokens
        // don't accumulate in memory. Idempotent.
        unregisterContainer(attributionToken);
        const duration = Date.now() - startTime;

        // Flush any partial trailing line and close the streaming log.
        // Failure to close cleanly is non-fatal — Node will GC the fd
        // eventually; the streamed bytes already on disk are intact.
        stdoutPrefixer.flush();
        stderrPrefixer.flush();
        if (streamLog) {
          try {
            streamLog.write(
              `\n=== Container Exited ===\nCode: ${code}\nDuration: ${duration}ms\nEnd: ${new Date().toISOString()}\n`,
            );
            streamLog.end();
          } catch (err) {
            if (!(err instanceof Error)) throw err;
            // Stream already errored / closed — nothing useful to do.
          }
        }

        if (timedOut) {
          const ts = new Date().toISOString().replace(/[:.]/g, '-');
          const timeoutLog = path.join(logsDir, `container-${ts}.log`);
          fs.writeFileSync(
            timeoutLog,
            [
              `=== Container Run Log (TIMEOUT) ===`,
              `Timestamp: ${new Date().toISOString()}`,
              `Group: ${group.name}`,
              `Container: ${containerName}`,
              `Duration: ${duration}ms`,
              `Exit Code: ${code}`,
              `Had Streaming Output: ${hadStreamingOutput}`,
            ].join('\n'),
          );

          // #589 (reopened) / #682 — for MAINTENANCE one-shots, streamed
          // preview output is NOT a delivered result. A healthy
          // maintenance run reaches a terminal result and exits
          // naturally (scheduleClose → `_close`) within seconds, so a
          // run still alive at this inactivity timeout having produced
          // only previews was reaped mid-compose (e.g. the morning-brief
          // compose turn stalled on an LLM / proxy blip). Resolving
          // `'success'` here hid that incomplete run (the original #589
          // silent-stop). Classify `'killed'` so `task_run_logs` records
          // non-success and the run is retriable / redeliverable.
          //
          // #682 replaced #589's `hadStreamingOutput` gate with the
          // precise `hadTerminalResult` signal: the rare "delivered a
          // terminal result, then the SDK iterator hung until the host
          // reaped it" shape now keeps its idle-cleanup `success` (the
          // work landed) instead of being over-reported as killed. Only
          // a reap after previews-but-no-terminal-result is `killed`.
          //
          // #689 — `sawNoDeliveryMarker` also forces `killed` here: a
          // `requires_delivery` skill that emitted a terminal marker
          // (so `hadTerminalResult` is set) stamped `noDelivery` because
          // it never delivered to chat. Without this, such a run reaped
          // at the inactivity timeout would fall through to the
          // `hadStreamingOutput` idle-cleanup `success` below.
          if (
            isMaintenanceSession &&
            hadStreamingOutput &&
            (!hadTerminalResult || sawNoDeliveryMarker || sawNoModelWorkMarker)
          ) {
            logger.warn(
              {
                group: group.name,
                containerName,
                duration,
                code,
                sawNoDeliveryMarker,
                sawNoModelWorkMarker,
              },
              'Maintenance container reaped by inactivity timeout without delivering user-facing content — classifying killed (incomplete, retriable) (#682/#689/#901)',
            );
            outputChain.then(() => {
              resolve({
                status: 'killed',
                result: null,
                newSessionId,
                // #890 follow-up — a real timeout kill of work in
                // flight: reaped mid-compose with no terminal result.
                // Alertable.
                timedOut: true,
                // #901 is checked before #689: when the model never ran,
                // "delivered nothing" is a consequence of that, and the
                // no-model-work reason is the actionable one.
                error: sawNoModelWorkMarker
                  ? `Maintenance container reaped by inactivity timeout after ${timeoutMs}ms; no assistant turn ever ran (noModelWork marker) — the model produced no output at all, which is the shape a subscription-cap abort takes. Incomplete run, retriable once the cap resets`
                  : sawNoDeliveryMarker
                    ? `Maintenance container reaped by inactivity timeout after ${timeoutMs}ms; the requires_delivery skill delivered no user-facing content (noDelivery marker) — incomplete run (reaped mid-compose), retriable`
                    : `Maintenance container reaped by inactivity timeout after ${timeoutMs}ms having streamed only preview output and no terminal result — incomplete run (reaped mid-compose), retriable`,
              });
            });
            return;
          }

          // Interactive / default session, OR a maintenance run that
          // delivered a terminal result before idling out: timeout after
          // output = idle cleanup, not failure. The agent already
          // streamed its reply (and, for multi-turn chats, sent it via
          // `send_message`); this is just the container being reaped
          // after the idle period expired. Unchanged from the pre-#589
          // contract.
          if (hadStreamingOutput) {
            logger.info(
              { group: group.name, containerName, duration, code },
              'Container timed out after output (idle cleanup)',
            );
            outputChain.then(() => {
              resolve({
                status: 'success',
                result: null,
                newSessionId,
                // Deliberately NOT stamped `timedOut` (#890 follow-up):
                // the clock ran out, but nothing was killed mid-work —
                // the agent already delivered and this is the idle
                // reaper collecting a finished container. Alerting here
                // would page the operator on every healthy maintenance
                // run that idles out, which is most of them.
              });
            });
            return;
          }

          logger.error(
            { group: group.name, containerName, duration, code },
            'Container timed out with no output',
          );

          resolve({
            status: 'error',
            result: null,
            // #890 follow-up — killed by the container timeout having
            // produced nothing at all. Alertable.
            timedOut: true,
            // `timeoutMs`, not `configTimeout`: the two differ for both
            // session kinds — maintenance arms
            // `MAINTENANCE_CONTAINER_TIMEOUT` and non-maintenance arms
            // `Math.max(configTimeout, IDLE_TIMEOUT + 30_000)`. Reporting
            // `configTimeout` named a duration the timer was never set
            // to, and #890's alert now puts this string in front of the
            // operator, so the misreport would be read as the budget to
            // tune.
            error: `Container timed out after ${timeoutMs}ms`,
          });
          return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const logFile = path.join(logsDir, `container-${timestamp}.log`);
        const isVerbose =
          process.env.LOG_LEVEL === 'debug' ||
          process.env.LOG_LEVEL === 'trace';

        const logLines = [
          `=== Container Run Log ===`,
          `Timestamp: ${new Date().toISOString()}`,
          `Group: ${group.name}`,
          `IsMain: ${input.isMain}`,
          `Duration: ${duration}ms`,
          `Exit Code: ${code}`,
          `Stdout Truncated: ${stdoutTruncated}`,
          `Stderr Truncated: ${stderrTruncated}`,
          ``,
        ];

        const isError = code !== 0;

        if (isVerbose || isError) {
          // On error, log input metadata only — not the full prompt.
          // Full input is only included at verbose level to avoid
          // persisting user conversation content on every non-zero exit.
          if (isVerbose) {
            logLines.push(`=== Input ===`, JSON.stringify(input, null, 2), ``);
          } else {
            logLines.push(
              `=== Input Summary ===`,
              `Prompt length: ${input.prompt.length} chars`,
              `Session ID: ${input.sessionId || 'new'}`,
              ``,
            );
          }
          logLines.push(
            `=== Container Args ===`,
            containerArgs.join(' '),
            ``,
            `=== Mounts ===`,
            mounts
              .map(
                (m) =>
                  `${m.hostPath} -> ${m.containerPath}${m.readonly ? ' (ro)' : ''}`,
              )
              .join('\n'),
            ``,
            `=== Stderr${stderrTruncated ? ' (TRUNCATED)' : ''} ===`,
            stderr,
            ``,
            `=== Stdout${stdoutTruncated ? ' (TRUNCATED)' : ''} ===`,
            stdout,
          );
        } else {
          logLines.push(
            `=== Input Summary ===`,
            `Prompt length: ${input.prompt.length} chars`,
            `Session ID: ${input.sessionId || 'new'}`,
            ``,
            `=== Mounts ===`,
            mounts
              .map((m) => `${m.containerPath}${m.readonly ? ' (ro)' : ''}`)
              .join('\n'),
            ``,
          );
        }

        fs.writeFileSync(logFile, logLines.join('\n'));
        logger.debug({ logFile, verbose: isVerbose }, 'Container log written');

        if (code !== 0) {
          logger.error(
            {
              group: group.name,
              code,
              duration,
              stderr,
              stdout,
              logFile,
            },
            'Container exited with error',
          );

          resolve({
            status: 'error',
            result: null,
            error: `Container exited with code ${code}: ${stderr.slice(-200)}`,
          });
          return;
        }

        // Streaming mode: wait for output chain to settle, return completion marker
        if (onOutput) {
          // #682 — a clean exit (code 0) is recorded `success` only when
          // the run actually delivered a terminal result. A MAINTENANCE
          // container that exits 0 having produced only streaming
          // previews (or nothing) never reached a terminal result — e.g.
          // an in-container `process.exit(0)` fired mid-compose before
          // the SDK loop / #461 silent-stop synthesis could emit one
          // (the 2026-06-11 morning-brief shape: the hard-exit watchdog
          // `process.exit(0)`'d after streaming the compose preview but
          // before `send_message`). `code === 0` + `timedOut === false`
          // lands here, and the old unconditional `success` made that
          // incomplete run indistinguishable from a genuine delivery —
          // zero alerting fired and the brief was never sent. Classify
          // `killed` so `task_run_logs` records a non-success, retriable
          // run. The #461 silent-stop no-op success is NOT mis-flagged:
          // it emits a terminal marker (`{status:'success', result:''}`,
          // no `streamText`), so `hadTerminalResult` is set and it stays
          // `success`. Scoped to maintenance to mirror #683 — the
          // `'killed'` status is consumed by the task-scheduler, and
          // interactive idle-cleanup `success` semantics are unchanged.
          //
          // #689 — `sawNoDeliveryMarker` extends this to the case where
          // a `requires_delivery` skill DID emit a terminal marker
          // (so `hadTerminalResult` is set) but stamped it `noDelivery`
          // because the run delivered nothing to chat — the silent
          // success that defeated the #682 `!hadTerminalResult` gate
          // (morning-brief composed but never sent). Either signal
          // classifies the maintenance run `killed`.
          if (
            isMaintenanceSession &&
            (!hadTerminalResult || sawNoDeliveryMarker || sawNoModelWorkMarker)
          ) {
            outputChain.then(() => {
              logger.warn(
                {
                  group: group.name,
                  duration,
                  newSessionId,
                  sawNoDeliveryMarker,
                  sawNoModelWorkMarker,
                },
                'Maintenance container exited cleanly (code 0) without delivering user-facing content — classifying killed (incomplete, retriable) (#682/#689/#901)',
              );
              resolve({
                status: 'killed',
                result: null,
                newSessionId,
                // #901 before #689: a run whose model never ran also
                // delivered nothing, and the no-model-work reason is the
                // one that tells the operator why.
                error: sawNoModelWorkMarker
                  ? 'Maintenance container exited cleanly (code 0) but no assistant turn ever ran (noModelWork marker) — the model produced no output at all, which is the shape a subscription-cap abort takes. Incomplete run, retriable once the cap resets'
                  : sawNoDeliveryMarker
                    ? 'Maintenance container exited cleanly (code 0) but the requires_delivery skill delivered no user-facing content (noDelivery marker) — incomplete run (composed but never sent), retriable'
                    : 'Maintenance container exited cleanly (code 0) without delivering a terminal result — incomplete run (exited before producing/sending a result), retriable',
              });
            });
            return;
          }
          outputChain.then(() => {
            logger.info(
              { group: group.name, duration, newSessionId },
              'Container completed (streaming mode)',
            );
            resolve({
              status: 'success',
              result: null,
              newSessionId,
            });
          });
          return;
        }

        // Legacy mode: parse the last output marker pair from accumulated stdout
        try {
          // Extract JSON between sentinel markers for robust parsing
          const startIdx = stdout.indexOf(OUTPUT_START_MARKER);
          const endIdx = stdout.indexOf(OUTPUT_END_MARKER);

          let jsonLine: string;
          if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
            jsonLine = stdout
              .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
              .trim();
          } else {
            // Fallback: last non-empty line (backwards compatibility)
            const lines = stdout.trim().split('\n');
            jsonLine = lines[lines.length - 1];
          }

          const output: ContainerOutput = JSON.parse(jsonLine);

          logger.info(
            {
              group: group.name,
              duration,
              status: output.status,
              hasResult: !!output.result,
            },
            'Container completed',
          );

          resolve(output);
        } catch (err) {
          if (!(err instanceof SyntaxError)) throw err;
          logger.error(
            {
              group: group.name,
              stdout,
              stderr,
              error: err,
            },
            'Failed to parse container output',
          );

          resolve({
            status: 'error',
            result: null,
            error: `Failed to parse container output: ${err instanceof Error ? err.message : String(err)}`,
          });
        }
      });

      container.on('error', (err) => {
        clearTimeout(timeout);
        // Spawn-error path: docker may never have read the env-file
        // (e.g. ENOENT on the docker binary itself), but the file is
        // still on disk. cleanup() is idempotent — safe to call here
        // and again from `close` if both fire.
        cleanupSecretEnvFile();
        unregisterContainer(attributionToken);
        logger.error(
          { group: group.name, containerName, error: err },
          'Container spawn error',
        );
        // Spawn-error path: the close handler may not fire on some
        // failure modes (e.g. spawn ENOENT — the binary doesn't exist),
        // so flush + close the streaming log here too. Without this the
        // file descriptor leaks until process GC and the file is left
        // open with no exit footer, which makes the on-disk record
        // ambiguous (was the container still running, or did it die
        // before producing any output?).
        stdoutPrefixer.flush();
        stderrPrefixer.flush();
        if (streamLog) {
          try {
            streamLog.write(
              `\n=== Container Spawn Failed ===\nError: ${err.message}\nEnd: ${new Date().toISOString()}\n`,
            );
            streamLog.end();
          } catch (err) {
            if (!(err instanceof Error)) throw err;
            // Stream already errored / closed — nothing to recover.
          }
        }
        resolve({
          status: 'error',
          result: null,
          error: `Container spawn error: ${err.message}`,
        });
      });
    });
  } catch (err) {
    // Sync throw before the spawn handlers wired up — handlers can't
    // unregister, so do it here and propagate. #784: re-raise through
    // toContainerAgentError so an infrastructure fault (buildContainerArgs /
    // buildSecretEnvFile, the OneCLI fail-closed throws, mkdirSync(logsDir))
    // reaches runAgent as a typed ContainerAgentError, while a defect
    // propagates untouched.
    unregisterContainer(attributionToken);
    throw toContainerAgentError(err);
  }
}

// Group-visible IPC snapshots moved to ./group-snapshots.ts (#851
// slice 6). Facade re-export keeps the existing import surface.
export {
  writeGroupsSnapshot,
  writeTasksSnapshot,
  type AvailableGroup,
} from './group-snapshots.js';
