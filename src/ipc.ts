import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';

import {
  ASSISTANT_NAME,
  DATA_DIR,
  GROUPS_DIR,
  HOST_PROJECT_ROOT,
  IPC_POLL_INTERVAL,
  ORCHESTRATOR_REPO_URL,
  STORE_DIR,
} from './config.js';
import { SqliteError } from 'better-sqlite3';
import { GrammyError, HttpError } from 'grammy';

import { syncBackupRepo, type SyncResult } from './backup-sync.js';
import { isFsErrorWithCode } from './fs-errors.js';
import { sendPoolMessage } from './channels/telegram.js';
import {
  buildSnitchmdFlags,
  formatSnitchmdHeader,
  parseFetchMarkdownUrl,
  parseSnitchmdStdout,
} from './fetch-markdown-args.js';
import { AvailableGroup, getInstalledTiles } from './container-runner.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';
import { shouldStoreBotMessage, storeMessage } from './db-messages.js';
import { deleteAllSessions } from './db-sessions.js';
import {
  applyTripitSegmentsToTzState,
  getActivePendingRunAtNames,
  getCurrentTz,
  type TripitSegment,
} from './db-tz.js';
import type { ContainerStatus } from './group-queue.js';
import { registerCoreIpcHandlers } from './ipc-handlers/index.js';
import {
  dispatchIpcTask,
  scriptResultPath,
  type IpcTaskPayload,
} from './ipc-registry.js';
import { logger } from './logger.js';
import { stripInternalTags } from './router.js';
import { runSidecar } from './sidecar-runner.js';
import { recomputeLocalSchedules } from './task-scheduler.js';
import { RegisteredGroup } from './types.js';

// Errno codes the IPC poller's best-effort fs ops (readdir, stat, unlink,
// rename over group IPC dirs) may legitimately raise, incl. path-shape
// ENOTDIR/ELOOP/ENAMETOOLONG. Anything else is a real defect and propagates.
const IPC_FS_CODES = [
  'ENOENT',
  'EACCES',
  'EPERM',
  'EISDIR',
  'EBUSY',
  'EROFS',
  'ENOSPC',
  'ENOTDIR',
  'ELOOP',
  'ENAMETOOLONG',
];

/**
 * Recoverable failures when processing one IPC message/task file: a
 * malformed payload (SyntaxError), a persistence failure (SqliteError), a
 * Telegram send/transport failure (GrammyError / HttpError), or an fs errno.
 * Those quarantine the one bad file and let the poller continue. Anything
 * else (a TypeError or other programming defect) is not data-poison and
 * propagates so the bug surfaces.
 */
function isRecoverableIpcError(err: unknown): boolean {
  return (
    err instanceof SyntaxError ||
    // Only constraint-class SQLite errors are per-file data poison (a bad
    // payload violating a NOT NULL / UNIQUE / FK). Infrastructure faults
    // (SQLITE_CORRUPT / BUSY / LOCKED / READONLY / SCHEMA) are NOT per-file
    // recoverable and must propagate.
    (err instanceof SqliteError &&
      typeof err.code === 'string' &&
      err.code.startsWith('SQLITE_CONSTRAINT')) ||
    err instanceof GrammyError ||
    err instanceof HttpError ||
    isFsErrorWithCode(err, IPC_FS_CODES)
  );
}

/**
 * Best-effort quarantine of a bad IPC file into `errors/`. Wrapped so an fs
 * failure during the move can't break the per-file boundary — the file
 * simply stays and is retried next poll; a non-fs defect propagates.
 */
function moveIpcFileToErrors(
  ipcBaseDir: string,
  sourceGroup: string,
  file: string,
  filePath: string,
): void {
  try {
    const errorDir = path.join(ipcBaseDir, 'errors');
    fs.mkdirSync(errorDir, { recursive: true });
    fs.renameSync(filePath, path.join(errorDir, `${sourceGroup}-${file}`));
  } catch (err) {
    if (!isFsErrorWithCode(err, IPC_FS_CODES)) throw err;
    logger.error(
      { err, file, sourceGroup },
      '[ipc] Failed to move bad file to errors/ (will retry next poll)',
    );
  }
}

export interface IpcDeps {
  /**
   * Fired after a successful VISIBLE bot send from the IPC
   * `send_message` / `send_file` handlers (delivery confirmed by a
   * returned message id). `sourceGroupFolder` is the sending group;
   * the consumer decides whether the target chat is that group's own
   * chat before consuming the reply anchor (#722 — the anchor must be
   * released at the visible-send boundary, not only when the SDK
   * result later reaches the output callback).
   */
  onVisibleReply?: (chatJid: string, sourceGroupFolder: string) => void;
  sendReaction?: (
    jid: string,
    messageId: string | undefined,
    emoji: string,
  ) => Promise<void>;
  sendMessage: (
    jid: string,
    text: string,
    replyToMessageId?: string,
  ) => Promise<string | void>;
  pinMessage?: (jid: string, messageId: string) => Promise<void>;
  sendFile?: (
    jid: string,
    filePath: string,
    caption?: string,
    replyToMessageId?: string,
  ) => Promise<string | undefined>;
  registeredGroups: () => Record<string, RegisteredGroup>;
  registerGroup: (jid: string, group: RegisteredGroup) => void;
  /**
   * Inverse of `registerGroup` (#159). Removes the in-memory entry and
   * the DB row in one call. Returns false if the JID was not registered
   * — caller can use that to log a no-op or to surface "nothing to do"
   * to the requester. The caller is responsible for refreshing the
   * `available_groups.json` snapshot afterward (mirrors the
   * registerGroup contract).
   *
   * Out of scope: deleting the on-disk `groups/<folder>/` directory.
   * Operators delete that manually; auto-deletion would silently destroy
   * agent-curated state (CLAUDE.md, MEMORY.md, scheduled-task workspace)
   * on every churn of the registration.
   */
  unregisterGroup: (jid: string) => boolean;
  /** Partial update: flip `containerConfig.trusted` only. Returns false if the JID isn't registered. */
  setGroupTrusted: (jid: string, trusted: boolean) => boolean;
  /**
   * Partial update: change the trigger pattern (and optionally
   * `requiresTrigger`) only. Returns false if (a) the JID isn't
   * registered, or (b) the trigger fails the non-empty/whitespace
   * invariant enforced by `updateGroupTrigger`.
   */
  setGroupTrigger: (
    jid: string,
    trigger: string,
    requiresTrigger?: boolean,
  ) => boolean;
  syncGroups: (force: boolean) => Promise<void>;
  getAvailableGroups: () => AvailableGroup[];
  writeGroupsSnapshot: (
    groupFolder: string,
    isMain: boolean,
    availableGroups: AvailableGroup[],
    registeredJids: Set<string>,
    isTrusted?: boolean,
  ) => void;
  onTasksChanged: () => void;
  nukeSession: (
    groupFolder: string,
    session: 'default' | 'maintenance' | 'all',
    options?: { skipReentry?: boolean },
  ) => void;
  /**
   * Signal every currently-active container across all groups and sessions
   * to wind down (write `_close` sentinels). Called by the `tessl_update`
   * IPC handler after the registry pulls in new tile content so running
   * containers respawn and pick up the fresh tiles on the next inbound
   * message — without this, a long-lived container keeps the old skills/
   * snapshot it copied at spawn time until its 30-min idle timeout.
   *
   * Implemented in `GroupQueue.closeAllActiveContainers()`. The dep is
   * injected (not imported directly) for the same reason `nukeSession`
   * is: ipc.ts mustn't reach into the orchestrator's queue singleton, and
   * tests need to substitute a stub.
   *
   * Returns the number of containers signaled.
   */
  closeAllActiveContainers: () => number;
  /**
   * Read the derived container status for a given (jid, sessionName)
   * slot. Used by `chat_status` to surface running/idle/cooling-down/
   * crashed/not-spawned without exposing the queue's internal state map.
   * Implementations must combine the GroupQueue slot state with the
   * orchestrator-side circuit breaker (per group folder).
   */
  getContainerStatus?: (
    chatJid: string,
    sessionName: 'default' | 'maintenance',
  ) => ContainerStatus;
}

let ipcWatcherRunning = false;

// Host-side allowlist for the five tile-repo names the promote flow is
// wired against. The MCP tools' zod enums (ipc-mcp-stdio.ts::TILE_NAMES)
// mirror this list client-side for a clean schema error at tool-call
// time, but the IPC handler is reachable by any payload dropped into
// the tasks dir — a compromised container could skip the MCP path and
// write `{tileName: "../../etc"}` directly, escaping GROUPS_DIR via
// `path.join` or pointing the bash scripts at an attacker-controlled
// git URL. This set is the actual trust boundary; keeping it in sync
// with the client-side enum is a release-hygiene concern.
const KNOWN_TILE_NAMES: ReadonlySet<string> = new Set([
  'nanoclaw-admin',
  'nanoclaw-core',
  'nanoclaw-untrusted',
  'nanoclaw-trusted',
  'nanoclaw-host',
]);

// Prefix for outbound text emitted by the maintenance-session AyeAye.
// Without the prefix, a scheduled-task reply looks identical to a
// user-facing reply in the chat, which confused Baruch when he
// responded to `[heartbeat from maintenance]` messages as if they were
// live conversation. The prefix is applied BOTH to Telegram-bound text
// AND to the messages.db copy so the full trail shows provenance —
// heartbeat accounting, future message recap, etc.
const MAINTENANCE_MESSAGE_PREFIX = '[M] ';

/**
 * Prepend `[M] ` if the payload came from the maintenance session.
 * Idempotent — if the text already begins with the prefix (double-
 * hop case, agent that hand-typed it, whatever), we don't stack.
 * Exported for the unit test; the production caller is in the same
 * file so the public API is a single entry point.
 *
 * @internal — test-only export, should not be part of the public
 * `.d.ts` surface (we build with `stripInternal: true`).
 */
export function applyMaintenancePrefix(
  text: string,
  sessionName: string | undefined,
): string {
  if (sessionName !== MAINTENANCE_SESSION_NAME) return text;
  if (text.startsWith(MAINTENANCE_MESSAGE_PREFIX)) return text;
  return MAINTENANCE_MESSAGE_PREFIX + text;
}

// The git-tracked, deploy-seeded global persona files `persist_global_file`
// (#393) is allowed to commit. A caller may only name these exact basenames
// — never a path component — so a compromised container can't commit an
// arbitrary tracked file (traversal, secrets, workflow YAML) to the deploy
// source. Keep in lock-step with the `.gitignore` `!groups/global/*`
// allowlist and the zod enum in `ipc-mcp-stdio.ts`'s tool registration.
export const PERSISTABLE_GLOBAL_FILES = [
  'SOUL.md',
  'SOUL-untrusted.md',
] as const;

/**
 * Validate a `persist_global_file` `files` payload against
 * `PERSISTABLE_GLOBAL_FILES` and map it to repo-relative paths under
 * `groups/global/`. An empty or absent payload defaults to every allowlisted
 * file. Returns the offending entry on rejection so the handler can write an
 * actionable error envelope. Defense-in-depth behind the MCP tool's zod enum:
 * a container can write an IPC task file directly, bypassing the tool schema.
 */
export function validateGlobalFilesToPersist(
  files: unknown,
): { ok: true; relPaths: string[] } | { ok: false; invalid: string } {
  const requested: unknown[] =
    Array.isArray(files) && files.length > 0
      ? files
      : [...PERSISTABLE_GLOBAL_FILES];
  const invalid = requested.find(
    (f): boolean =>
      typeof f !== 'string' ||
      !(PERSISTABLE_GLOBAL_FILES as readonly string[]).includes(f),
  );
  if (invalid !== undefined) {
    return {
      ok: false,
      invalid: typeof invalid === 'string' ? invalid : String(invalid),
    };
  }
  // Dedupe while preserving first-seen order so `git add` names each path once.
  const relPaths = [...new Set(requested as string[])].map((f) =>
    path.posix.join('groups', 'global', f),
  );
  return { ok: true, relPaths };
}

/**
 * Redact a GitHub token from text before it reaches a result envelope or a
 * log. git stderr can echo the push URL, which after the `insteadOf` rewrite
 * embeds `x-access-token:<TOKEN>@github.com` — so a raw failure envelope would
 * leak the credential (`coding-policy: no-secrets`). Strips both the exact
 * token (when known) and any `x-access-token:...@` URL credential.
 */
export function redactGitToken(text: string, token?: string): string {
  let out = text;
  if (token) out = out.split(token).join('***');
  return out.replace(/x-access-token:[^@\s]+@/g, 'x-access-token:***@');
}

/**
 * Upper bound on the `persist_tz_segments` payload (#748 review). A real
 * itinerary is a handful of timezone segments; this is generous headroom that
 * still refuses a runaway/accidental payload before it bloats the
 * `tz_state.segments` DB row. Kept in lock-step with the `.max()` on the MCP
 * tool's zod schema in `container/agent-runner/src/ipc-mcp-stdio.ts`.
 */
export const MAX_TZ_SEGMENTS = 200;

/**
 * #748 — coerce the raw `segments` payload of a `persist_tz_segments` IPC call
 * to a `TripitSegment[]`. The value arrives as raw JSON off the IPC wire (the
 * in-container TripIt → Reclaim sync's parsed `result.segments`), so anything
 * that is not an array — `undefined`, `null`, an object, a string — returns an
 * empty array with `wasArray: false` rather than throwing. This is pure
 * normalization; it does NOT decide what to persist. The caller
 * (`classifyTzPersist`) uses `wasArray` to REJECT a non-array (a caller bug must
 * not clear owner `tz_state`) — only a genuine array, including an explicit
 * `[]`, is persisted. Element shapes are not deep-validated here:
 * `applyTripitSegmentsToTzState` / `walkTzSegments` already tolerate partial
 * per-field segment shapes (optional-with-fallback, #229).
 */
export function coerceTzSegments(raw: unknown): {
  segments: TripitSegment[];
  wasArray: boolean;
} {
  if (Array.isArray(raw)) {
    return { segments: raw as TripitSegment[], wasArray: true };
  }
  return { segments: [], wasArray: false };
}

/**
 * #748 — decide what the `persist_tz_segments` IPC handler does with a request,
 * as a pure function so the security-sensitive outcomes are unit-testable
 * without staging `processTaskIpc`:
 *
 *  - `deny`    — the caller is not the main container. The MCP tool is
 *                isMain-gated, but the IPC task-file path is reachable by ANY
 *                container, so this re-check (on the directory-verified
 *                `isMain`) is what actually blocks a non-main agent from
 *                poisoning owner `tz_state`.
 *  - `reject`  — a non-array payload, OR over `MAX_TZ_SEGMENTS`. Both refuse to
 *                write rather than mangle owner tz. A non-array is a caller bug,
 *                not a "no trips" signal: persisting an empty set would clear
 *                `tz_state.segments` and could flip the owner's timezone +
 *                recompute local schedules off a malformed call. The legacy
 *                the removed host-op skipped persistence on malformed stdout for
 *                the same reason; an explicit `[]` is the only way to clear.
 *                Over-cap is refused (not truncated — truncation would silently
 *                drop later trips).
 *  - `persist` — a genuine array within bounds (including an explicit `[]`,
 *                which correctly clears to the no-active-trips state).
 */
export type TzPersistDecision =
  | { action: 'deny'; error: string }
  | { action: 'reject'; error: string }
  | { action: 'persist'; segments: TripitSegment[] };

export function classifyTzPersist(
  isMain: boolean,
  raw: unknown,
): TzPersistDecision {
  if (!isMain) {
    return { action: 'deny', error: 'persist_tz_segments is main-group only' };
  }
  const { segments, wasArray } = coerceTzSegments(raw);
  if (!wasArray) {
    return {
      action: 'reject',
      error: 'segments must be an array (send [] to explicitly clear)',
    };
  }
  if (segments.length > MAX_TZ_SEGMENTS) {
    return {
      action: 'reject',
      error: `too many segments (${segments.length} > ${MAX_TZ_SEGMENTS})`,
    };
  }
  return { action: 'persist', segments };
}

export interface PersistGitResult {
  committed?: boolean;
  stdout?: string;
  error?: string;
  stderr?: string;
  stage?: 'git';
}

// The token-bearing env that lets `git push` authenticate to github.com via
// the `insteadOf` URL rewrite. Only the push step needs it. Empty without a
// token (local-remote pushes and all read-only steps need no auth).
function gitAuthEnv(token?: string): NodeJS.ProcessEnv {
  if (!token) return {};
  return {
    GIT_ASKPASS: 'echo',
    GIT_TERMINAL_PROMPT: '0',
    GITHUB_TOKEN: token,
    GIT_CONFIG_COUNT: '1',
    GIT_CONFIG_KEY_0:
      'url.https://x-access-token:' + token + '@github.com/.insteadOf',
    GIT_CONFIG_VALUE_0: 'https://github.com/',
  };
}

// Run one git invocation in `repoRoot`, resolving with its exit code and
// captured output — never rejecting, so the caller drives control flow off
// `code`. Args are passed directly (no shell), so a caller-supplied commit
// message can't inject anything. `authEnv` adds push-auth for the push step.
function runGit(
  repoRoot: string,
  args: string[],
  authEnv: NodeJS.ProcessEnv = {},
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    execFile(
      'git',
      ['-C', repoRoot, ...args],
      {
        timeout: 60_000,
        maxBuffer: 1024 * 1024,
        env: { ...process.env, ...authEnv },
      },
      (error, stdout, stderr) => {
        // On a non-zero EXIT, execFile sets error.code to the numeric exit
        // code; on a spawn failure (e.g. ENOENT) it's a string — treat that
        // as a generic failure (1).
        const rawCode = (error as { code?: unknown } | null)?.code;
        const code = error ? (typeof rawCode === 'number' ? rawCode : 1) : 0;
        resolve({ code, stdout, stderr });
      },
    );
  });
}

// Commit identity for persona persist commits made inside the dedicated clone
// (#471). A container has no ambient git author; without this `git commit`
// fails with "empty ident name". The push itself carries the operator's
// GITHUB_TOKEN, so this identity only labels the commit.
const PERSONA_GIT_EMAIL = 'nanoclaw-bot@users.noreply.github.com';
const PERSONA_GIT_NAME = 'NanoClaw';

// Serialize persona persists in-process (#471 review). The `persist_global_file`
// handler runs clone/reset + overlay + commit/push as a fire-and-forget task,
// so two closely-timed requests would otherwise race on the shared
// `data/persona-repo` (one `reset --hard`/`clean` while the other stages),
// producing a garbled diff or a spurious failure. Each persist chains behind the
// previous so they run strictly one at a time; `runSerializedPersonaPersist`
// tolerates a rejected predecessor (the chain never stays poisoned).
let personaPersistChain: Promise<void> = Promise.resolve();
export function runSerializedPersonaPersist(task: () => Promise<void>): void {
  // `.then(task, task)` runs the next persist regardless of whether the
  // predecessor resolved or rejected, so a failed run never blocks the queue.
  // The task writes its own result envelope and logs its own failures; the
  // trailing `.catch` only surfaces a truly unexpected task-level crash (never a
  // silent swallow) and resets the chain to resolved for the next persist.
  personaPersistChain = personaPersistChain.then(task, task).catch((err) => {
    logger.error({ err: String(err) }, 'persona persist task crashed');
  });
}

/**
 * Ensure a clean checkout of the orchestrator's deploy-source repo at `repoDir`,
 * tracking `origin/main`, ready for a persona-file commit.
 *
 * Why this exists (#471, regression of #393): the running orchestrator's cwd
 * (`/app`) is the built image directory, NOT a git worktree — `docker-compose.yml`
 * bind-mounts `./groups`, `./data`, etc. but never the repo's `.git`. So
 * `persist_global_file` cannot `git add` in place (it fails with the exact
 * "fatal: not a git repository" this issue reports). Instead it operates on this
 * dedicated clone, applies the runtime persona edits into it, and pushes
 * `HEAD:main` so `deploy.sh`'s `git pull` keeps them.
 *
 * First run clones (shallow, single-branch `main`); later runs reconcile
 * `origin` with `remoteUrl` (so a changed `ORCHESTRATOR_REPO_URL` takes effect),
 * then fetch and hard-reset to `origin/main` so a stranded prior state — a
 * rolled-back commit, a half-applied edit, a diverged local `main` — never
 * blocks a fresh persist. The commit identity is (re)set on every successful
 * run so a clone lacking one can still `git commit`. Never rejects — resolves
 * `{ ok: true }` or an `{ ok: false, ...PersistGitResult }` failure envelope with
 * a distinct `git clone` / `git fetch` / `git reset` stage label, so "no repo"
 * is diagnosable separately from persist's "No changes" no-op (issue ask #2).
 * Any push auth rides on `token` via `gitAuthEnv`.
 */
export async function ensurePersonaRepo(opts: {
  repoDir: string;
  remoteUrl: string;
  token?: string;
}): Promise<{ ok: true } | ({ ok: false } & PersistGitResult)> {
  const { repoDir, remoteUrl, token } = opts;
  // Always disable the interactive credential prompt, even without a token: a
  // missing/misconfigured token must fail fast and deterministically, not hang
  // on a terminal prompt until the 60s `runGit` timeout.
  const auth = { GIT_TERMINAL_PROMPT: '0', ...gitAuthEnv(token) };

  const fail = (
    label: string,
    r: { stdout: string; stderr: string },
  ): { ok: false } & PersistGitResult => ({
    ok: false,
    error: redactGitToken(
      `${label}: ${(r.stderr || r.stdout || 'failed').trim()}`,
      token,
    ),
    stderr: redactGitToken(r.stderr, token).slice(-500),
    stage: 'git',
  });

  const isRepo =
    fs.existsSync(path.join(repoDir, '.git')) &&
    (await runGit(repoDir, ['rev-parse', '--is-inside-work-tree'])).code === 0;

  if (!isRepo) {
    // Clear any partial dir from an interrupted earlier clone — `git clone`
    // refuses a non-empty target — then clone fresh into it.
    fs.rmSync(repoDir, { recursive: true, force: true });
    fs.mkdirSync(path.dirname(repoDir), { recursive: true });
    const clone = await runGit(
      path.dirname(repoDir),
      [
        'clone',
        '--depth',
        '1',
        '--single-branch',
        '--branch',
        'main',
        remoteUrl,
        repoDir,
      ],
      auth,
    );
    if (clone.code !== 0) return fail('git clone', clone);
  } else {
    // Existing clone: reconcile `origin` with the configured `remoteUrl` first
    // — an operator may have changed `ORCHESTRATOR_REPO_URL` (documented as
    // overridable for a fork/alternate remote) since the clone was made, and
    // fetch/push otherwise keep targeting the stale URL captured at clone time,
    // silently persisting persona edits to the wrong repo. Then refresh to
    // origin/main, discarding any local state so a prior run's leftovers can't
    // ride along in the next persist's push. Use a bare `git fetch origin` —
    // NOT `git fetch origin main`, which writes only FETCH_HEAD and leaves
    // `refs/remotes/origin/main` pinned to the original shallow checkout. The
    // clone's own `+refs/heads/main:refs/remotes/origin/main` refspec makes the
    // bare fetch advance the tracking ref, so the following `reset --hard
    // origin/main` lands on the real remote tip; otherwise the next `HEAD:main`
    // push would be a non-fast-forward once main advanced upstream.
    const setUrl = await runGit(repoDir, [
      'remote',
      'set-url',
      'origin',
      remoteUrl,
    ]);
    if (setUrl.code !== 0) return fail('git remote set-url', setUrl);
    const fetch = await runGit(repoDir, ['fetch', 'origin'], auth);
    if (fetch.code !== 0) return fail('git fetch', fetch);
    const reset = await runGit(repoDir, ['reset', '--hard', 'origin/main']);
    if (reset.code !== 0) return fail('git reset', reset);
    const clean = await runGit(repoDir, ['clean', '-fd']);
    if (clean.code !== 0) return fail('git clean', clean);
  }

  // Set the commit identity on EVERY successful run (idempotent), not just the
  // initial clone: a clone made manually or with its local config stripped has
  // no ambient git author, so `git commit` would later fail at the persist
  // boundary with the "empty ident" error this guard exists to prevent. Fail
  // loudly here instead.
  const email = await runGit(repoDir, [
    'config',
    'user.email',
    PERSONA_GIT_EMAIL,
  ]);
  if (email.code !== 0) return fail('git config user.email', email);
  const name = await runGit(repoDir, ['config', 'user.name', PERSONA_GIT_NAME]);
  if (name.code !== 0) return fail('git config user.name', name);
  return { ok: true };
}

/**
 * Commit the allowlisted persona files in `repoRoot` and push `HEAD:main`,
 * resolving with the result envelope (never rejects — the caller always has
 * an envelope to write). Stages ONLY `relPaths` (already allowlist-validated
 * by `validateGlobalFilesToPersist`) so unrelated working-tree changes never
 * ride along. Behavior:
 * - staged change present → commit, then push.
 * - no staged change but `HEAD` is ahead of `origin/main` → a prior run
 *   committed but failed to push; push that pending commit (recovery).
 * - no staged change and nothing pending → `{ committed: false }` no-op.
 * On a push failure AFTER our own commit, the commit is rolled back
 * (`reset --soft`, keeping the edit staged) so a later run can retry — the
 * operation never strands a committed-but-unpushed change, which would be a
 * silent non-retryable state (`coding-policy: error-handling`). Any git
 * failure resolves to a `stage: 'git'` envelope with token-redacted stderr.
 */
export async function persistGlobalFilesToGit(opts: {
  repoRoot: string;
  relPaths: string[];
  message: string;
  token?: string;
}): Promise<PersistGitResult> {
  const { repoRoot, relPaths, message, token } = opts;

  const fail = (
    label: string,
    r: { stdout: string; stderr: string },
  ): PersistGitResult => ({
    error: redactGitToken(
      `${label}: ${(r.stderr || r.stdout || 'failed').trim()}`,
      token,
    ),
    stderr: redactGitToken(r.stderr, token).slice(-500),
    stage: 'git',
  });

  // Stage ONLY the allowlisted paths.
  const add = await runGit(repoRoot, ['add', '--', ...relPaths]);
  if (add.code !== 0) return fail('git add', add);

  // `diff --cached --quiet` exits 1 when those paths have a staged change, 0
  // when they don't, >1 on a real diff error.
  const staged = await runGit(repoRoot, [
    'diff',
    '--cached',
    '--quiet',
    '--',
    ...relPaths,
  ]);
  let committedHere = false;
  if (staged.code === 1) {
    const commit = await runGit(repoRoot, [
      'commit',
      '-m',
      message,
      '--',
      ...relPaths,
    ]);
    if (commit.code !== 0) return fail('git commit', commit);
    committedHere = true;
  } else if (staged.code === 0) {
    // Nothing staged. Recover a commit a prior run made but failed to push
    // (HEAD ahead of origin/main); otherwise it's a genuine no-op.
    const ahead = await runGit(repoRoot, [
      'rev-list',
      '--count',
      'origin/main..HEAD',
    ]);
    const aheadCount = ahead.code === 0 ? parseInt(ahead.stdout.trim(), 10) : 0;
    if (!Number.isFinite(aheadCount) || aheadCount <= 0) {
      return { committed: false, stdout: 'No changes to persist.' };
    }
    // fall through to push the unpushed commit(s)
  } else {
    return fail('git diff', staged);
  }

  // Deterministic push-time allowlist gate for the protected-branch
  // direct-push carve-out (`jbaruch/nanoclaw-host: persona-persist-direct-push`):
  // enumerate every path this push would change on `main` and refuse unless
  // ALL of them are declared persona files. A normal apply commits only
  // allowlisted paths by construction, but the recovery branch pushes a
  // pre-existing HEAD whose contents we didn't author — so verify before
  // touching `main`. An out-of-scope path is refused outright (not branch-PR
  // fallback): nothing but persona files ever direct-pushes.
  const allowedRel = new Set(
    PERSISTABLE_GLOBAL_FILES.map((f) => path.posix.join('groups', 'global', f)),
  );
  const pending = await runGit(repoRoot, [
    'diff',
    '--name-only',
    'origin/main..HEAD',
  ]);
  if (pending.code !== 0) {
    if (committedHere) await runGit(repoRoot, ['reset', '--soft', 'HEAD~1']);
    return fail('git diff (push gate)', pending);
  }
  const offending = pending.stdout
    .split('\n')
    .map((p) => p.trim())
    .filter((p) => p.length > 0 && !allowedRel.has(p));
  if (offending.length > 0) {
    if (committedHere) await runGit(repoRoot, ['reset', '--soft', 'HEAD~1']);
    return {
      error: `refusing to push: ${offending.length} path(s) outside the persona allowlist would change on main (${offending.slice(0, 5).join(', ')})`,
      stage: 'git',
    };
  }

  const push = await runGit(
    repoRoot,
    ['push', 'origin', 'HEAD:main'],
    gitAuthEnv(token),
  );
  if (push.code !== 0) {
    // Roll back the commit WE just made so the edit returns to the working
    // tree (staged) and a later run can retry — never strand it.
    if (committedHere) {
      await runGit(repoRoot, ['reset', '--soft', 'HEAD~1']);
    }
    return fail('git push', push);
  }
  return { committed: true, stdout: 'Committed and pushed.' };
}

/**
 * The `persist_global_file` work item (#471): ensure a clean clone, overlay the
 * approved persona files from the live runtime mirror (`<groupsDir>/global/`)
 * onto it, commit + push HEAD:main, and write a result envelope to `resultPath`.
 * Runs as a fire-and-forget task off the IPC loop, serialized against other
 * persists. ALWAYS writes an envelope: each step failure writes its own, and an
 * outer-boundary catch guarantees an actionable envelope for any unexpected
 * throw — a missing envelope reads as a silent hang to the polling caller.
 * Never rejects for an operational failure; only a failure of the envelope
 * write itself escapes (logged by the serializer). `groupsDir` is injected so
 * tests can point it at a fixture instead of the live `GROUPS_DIR`.
 */
export async function runPersonaPersistTask(opts: {
  personaRepoDir: string;
  groupsDir: string;
  relPaths: string[];
  remoteUrl: string;
  message: string;
  token?: string;
  resultPath: string;
  sourceGroup: string;
}): Promise<void> {
  const {
    personaRepoDir,
    groupsDir,
    relPaths,
    remoteUrl,
    message,
    token,
    resultPath,
    sourceGroup,
  } = opts;
  // Guarded by an outer-boundary catch (see the `catch` below) so this
  // fire-and-forget task always leaves the caller a result envelope. Per-step
  // failures write their own specific envelopes; the catch is the last resort
  // for an unexpected throw.
  try {
    const ready = await ensurePersonaRepo({
      repoDir: personaRepoDir,
      remoteUrl,
      token,
    });
    if (!ready.ok) {
      const { ok: _ok, ...envelope } = ready;
      fs.writeFileSync(resultPath, JSON.stringify(envelope));
      logger.error(
        { sourceGroup, stage: envelope.stage, error: envelope.error },
        'persist_global_file failed',
      );
      return;
    }

    // Overlay each approved persona file from the live runtime mirror
    // (`<groupsDir>/global/<file>`, edited by the agent via its RW bind) onto
    // the freshly-reset clone so the staged diff is exactly the edit.
    try {
      for (const rel of relPaths) {
        const src = path.join(groupsDir, 'global', path.posix.basename(rel));
        const dst = path.join(personaRepoDir, rel);
        fs.mkdirSync(path.dirname(dst), { recursive: true });
        fs.copyFileSync(src, dst);
      }
    } catch (e) {
      // A non-Error throw indicates a bug, not a filesystem failure.
      if (!(e instanceof Error)) throw e;
      const code = (e as NodeJS.ErrnoException).code;
      const envelope: PersistGitResult = {
        error: `persist_global_file: copying persona files into the clone failed${code ? ` (${code})` : ''} — ${e.message}`,
        stage: 'git',
      };
      fs.writeFileSync(resultPath, JSON.stringify(envelope));
      logger.error(
        { sourceGroup, code, error: e.message },
        'persist_global_file failed',
      );
      return;
    }

    const persistResult = await persistGlobalFilesToGit({
      repoRoot: personaRepoDir,
      relPaths,
      message,
      token,
    });
    fs.writeFileSync(resultPath, JSON.stringify(persistResult));
    if (persistResult.error) {
      logger.error(
        { sourceGroup, stage: persistResult.stage, error: persistResult.error },
        'persist_global_file failed',
      );
    } else {
      logger.info(
        { sourceGroup, committed: persistResult.committed },
        'persist_global_file completed',
      );
    }
    // outer-boundary-process-contract (coding-policy: error-handling): the
    // outermost boundary of this fire-and-forget persist task.
    //   - Caller's silent-failure shape: the container-side caller polls
    //     `resultPath` and reads its absence as a hang until its own timeout.
    //   - What the catch emits: an actionable error envelope for THIS request,
    //     so the caller always gets a result. (A failure of the envelope write
    //     itself propagates to the serializer's `.catch`, which logs it — the
    //     rare disk-failure edge case.)
    //   - Why propagation breaks the contract: an escaping throw is absorbed by
    //     the serializer with no envelope, hanging the poll.
    // eslint-disable-next-line no-catch-all/no-catch-all -- outer-boundary-process-contract
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    logger.error(
      { sourceGroup, error: msg },
      'persist_global_file failed (unexpected)',
    );
    fs.writeFileSync(
      resultPath,
      JSON.stringify({
        error: `persist_global_file: unexpected failure — ${msg}`,
        stage: 'git',
      } satisfies PersistGitResult),
    );
  }
}

/**
 * Stage everything in the backup repo, commit with `message`, and push,
 * resolving with the result envelope (never rejects — the IPC handler always
 * has an envelope to write). Git runs via `runGit` argument arrays — no
 * shell — so the agent-supplied commit message can't execute anything on the
 * host (#725; the previous `bash -c` wrapper was injectable through `$()`
 * inside its double-quoted string). Behavior mirrors
 * `persistGlobalFilesToGit`:
 * - staged change present → commit, then push.
 * - nothing staged but HEAD ahead of upstream → push the pending commit
 *   (recovers the stranded state the old shell pipeline could leave when a
 *   push failed after its commit).
 * - nothing staged, nothing pending → `{ committed: false }` no-op.
 * On a push failure after our own commit, roll back (`reset --soft`) so the
 * next run retries the commit instead of reporting "Nothing to commit."
 * Failure envelopes carry token-redacted stderr (`coding-policy: no-secrets`).
 */
export async function backupCommitAndPush(opts: {
  backupDir: string;
  message: string;
  token?: string;
}): Promise<PersistGitResult> {
  const { backupDir, message, token } = opts;

  const fail = (
    label: string,
    r: { stdout: string; stderr: string },
  ): PersistGitResult => ({
    error: redactGitToken(
      `${label}: ${(r.stderr || r.stdout || 'failed').trim()}`,
      token,
    ),
    stderr: redactGitToken(r.stderr, token).slice(-500),
    stage: 'git',
  });

  const add = await runGit(backupDir, ['add', '-A']);
  if (add.code !== 0) return fail('git add', add);

  // `diff --cached --quiet` exits 1 when a staged change exists, 0 when the
  // index matches HEAD, >1 on a real diff error.
  const staged = await runGit(backupDir, ['diff', '--cached', '--quiet']);
  let committedHere = false;
  if (staged.code === 1) {
    const commit = await runGit(backupDir, ['commit', '-m', message]);
    if (commit.code !== 0) return fail('git commit', commit);
    committedHere = true;
  } else if (staged.code === 0) {
    // Nothing staged. Push a commit a prior run made but failed to push
    // (HEAD ahead of upstream); otherwise it's a genuine no-op. A failing
    // `rev-list` (missing/misconfigured upstream, corrupt ref) is a real
    // error — mapping it to "nothing pending" would silently no-op the
    // backup while pending commits exist, and the bare `git push` below
    // needs the same upstream anyway. Surface it.
    const ahead = await runGit(backupDir, [
      'rev-list',
      '--count',
      '@{u}..HEAD',
    ]);
    if (ahead.code !== 0) return fail('git rev-list', ahead);
    const aheadCount = parseInt(ahead.stdout.trim(), 10);
    if (!Number.isFinite(aheadCount) || aheadCount <= 0) {
      return { committed: false, stdout: 'Nothing to commit.' };
    }
    // fall through to push the pending commit(s)
  } else {
    return fail('git diff', staged);
  }

  const push = await runGit(backupDir, ['push'], gitAuthEnv(token));
  if (push.code !== 0) {
    const envelope = fail('git push', push);
    // Roll back the commit WE just made so the next run retries it instead
    // of seeing a clean index and reporting a false no-op. If the rollback
    // itself fails, say so in the envelope — the repo is then in the
    // committed-but-unpushed state, which the ahead-of-upstream recovery
    // above picks up on the next run.
    if (committedHere) {
      const rollback = await runGit(backupDir, ['reset', '--soft', 'HEAD~1']);
      if (rollback.code !== 0) {
        envelope.error = `${envelope.error} (rollback also failed: ${redactGitToken(
          (rollback.stderr || rollback.stdout || 'failed').trim(),
          token,
        )} — a committed-but-unpushed backup commit remains; the next run recovers it via the ahead-of-upstream push)`;
      }
    }
    return envelope;
  }
  return { committed: true, stdout: 'Committed and pushed.' };
}

/**
 * Start the IPC polling loop. Returns a stop handle that halts the
 * loop and cancels the pending poll — production ignores it (the
 * watcher lives for the process), integration tests use it so a
 * finished suite doesn't leave a live timer polling a deleted tempdir.
 */
export function startIpcWatcher(deps: IpcDeps): () => void {
  if (ipcWatcherRunning) {
    logger.debug('IPC watcher already running, skipping duplicate start');
    return () => {};
  }
  ipcWatcherRunning = true;

  const ipcBaseDir = path.join(DATA_DIR, 'ipc');
  fs.mkdirSync(ipcBaseDir, { recursive: true });

  let stopped = false;
  let pollTimer: NodeJS.Timeout | undefined;

  const processIpcFiles = async () => {
    if (stopped) return;
    // Scan all group IPC directories (identity determined by directory)
    let groupFolders: string[];
    try {
      groupFolders = fs.readdirSync(ipcBaseDir).filter((f) => {
        const stat = fs.statSync(path.join(ipcBaseDir, f));
        return stat.isDirectory() && f !== 'errors';
      });
    } catch (err) {
      if (!isFsErrorWithCode(err, IPC_FS_CODES)) throw err;
      logger.error({ err }, 'Error reading IPC base directory');
      if (!stopped) pollTimer = setTimeout(processIpcFiles, IPC_POLL_INTERVAL);
      return;
    }

    const registeredGroups = deps.registeredGroups();

    // Build folder→isMain lookup from registered groups
    const folderIsMain = new Map<string, boolean>();
    for (const group of Object.values(registeredGroups)) {
      if (group.isMain) folderIsMain.set(group.folder, true);
    }

    for (const sourceGroup of groupFolders) {
      const isMain = folderIsMain.get(sourceGroup) === true;
      const messagesDir = path.join(ipcBaseDir, sourceGroup, 'messages');
      const tasksDir = path.join(ipcBaseDir, sourceGroup, 'tasks');

      // Process messages from this group's IPC directory
      try {
        if (fs.existsSync(messagesDir)) {
          const messageFiles = fs
            .readdirSync(messagesDir)
            .filter((f) => f.endsWith('.json'));
          for (const file of messageFiles) {
            const filePath = path.join(messagesDir, file);
            // Hoisted so the catch-block at the bottom can log which
            // message we were processing when things exploded. Without
            // this, a throw after `data = JSON.parse(...)` but before
            // the per-type branches would leave the error-log blind to
            // what content was in flight.
            let data:
              | {
                  type?: string;
                  chatJid?: string;
                  text?: string;
                  sender?: string;
                  replyToMessageId?: string;
                  pin?: boolean;
                  emoji?: string;
                  messageId?: string;
                  filePath?: string;
                  caption?: string;
                  [key: string]: unknown;
                }
              | undefined;
            try {
              const stat = fs.statSync(filePath);
              if (stat.size > 1_048_576) {
                logger.warn(
                  { file, sourceGroup, size: stat.size },
                  'IPC file exceeds 1MB limit, moving to errors',
                );
                const errorDir = path.join(ipcBaseDir, 'errors');
                fs.mkdirSync(errorDir, { recursive: true });
                fs.renameSync(
                  filePath,
                  path.join(errorDir, `${sourceGroup}-${file}`),
                );
                continue;
              }
              data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
              if (!data) {
                // JSON.parse produced null/undefined (e.g. the file
                // contained literal `null`). Skip to the next file.
                logger.warn(
                  { file, sourceGroup },
                  '[ipc] IPC file parsed to null/undefined — skipping',
                );
                fs.unlinkSync(filePath);
                continue;
              }
              if (
                data.type === 'react_to_message' &&
                data.chatJid &&
                data.emoji &&
                deps.sendReaction
              ) {
                const targetGroup = registeredGroups[data.chatJid];
                if (
                  isMain ||
                  (targetGroup && targetGroup.folder === sourceGroup)
                ) {
                  await deps.sendReaction(
                    data.chatJid,
                    data.messageId || undefined,
                    data.emoji,
                  );
                  logger.info(
                    {
                      chatJid: data.chatJid,
                      emoji: data.emoji,
                      sourceGroup,
                    },
                    'IPC reaction sent',
                  );
                } else {
                  logger.warn(
                    { chatJid: data.chatJid, sourceGroup },
                    'Unauthorized IPC reaction attempt blocked',
                  );
                }
              } else if (
                data.type === 'send_file' &&
                data.chatJid &&
                data.filePath &&
                deps.sendFile
              ) {
                const targetGroup = registeredGroups[data.chatJid];
                if (
                  isMain ||
                  (targetGroup && targetGroup.folder === sourceGroup)
                ) {
                  // Translate container path to host path
                  const containerPath: string = data.filePath;
                  let hostPath: string;
                  if (containerPath.startsWith('/workspace/group/')) {
                    hostPath = path.join(
                      GROUPS_DIR,
                      sourceGroup,
                      containerPath.replace('/workspace/group/', ''),
                    );
                  } else if (containerPath.startsWith('/workspace/trusted/')) {
                    hostPath = path.join(
                      process.cwd(),
                      'trusted',
                      containerPath.replace('/workspace/trusted/', ''),
                    );
                  } else {
                    logger.warn(
                      { containerPath, sourceGroup },
                      'send_file: path outside allowed mounts',
                    );
                    fs.unlinkSync(filePath);
                    continue;
                  }

                  if (fs.existsSync(hostPath)) {
                    // Strip <internal>…</internal> blocks from the caption
                    // so agent-written internal reasoning never leaks —
                    // neither to Telegram (display) nor to messages.db
                    // (which feeds heartbeat's answered-check accounting).
                    // Mirrors the message-payload stripping below. If the
                    // caption is fully internal, send the file with no
                    // caption; the file itself is still useful payload.
                    const strippedCaption = data.caption
                      ? stripInternalTags(data.caption)
                      : '';
                    // Tag maintenance-session captions so Baruch can
                    // tell a scheduled-task file-send from a live one.
                    // Skip the prefix entirely when the caption is
                    // empty — `[M] ` alone on a silent file-send is
                    // noise.
                    const cleanCaption = strippedCaption
                      ? applyMaintenancePrefix(
                          strippedCaption,
                          typeof data.sessionName === 'string'
                            ? data.sessionName
                            : undefined,
                        )
                      : '';
                    const sentFileMsgId = await deps.sendFile(
                      data.chatJid,
                      hostPath,
                      cleanCaption || undefined,
                      data.replyToMessageId,
                    );
                    // Store the cleaned caption (if any) so the message
                    // shows up in accounting the same as text messages.
                    // Without this, `send_file` is a bypass: captions
                    // reach Telegram but never hit messages.db, so
                    // heartbeat unanswered-checks think the agent never
                    // responded. Store the cleaned version — storing the
                    // raw caption would let a caption whose visible text
                    // was empty after stripping count as an "answered"
                    // response. Gate on `sentFileMsgId` (#428) — a
                    // failed send must not leave a phantom row that
                    // marks the user as answered when delivery never
                    // landed.
                    const captionDelivered = shouldStoreBotMessage(
                      data.chatJid,
                      sentFileMsgId,
                    );
                    if (cleanCaption && captionDelivered) {
                      storeMessage({
                        id: `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                        chat_jid: data.chatJid,
                        sender: ASSISTANT_NAME,
                        sender_name: ASSISTANT_NAME,
                        content: cleanCaption,
                        timestamp: new Date().toISOString(),
                        is_from_me: true,
                        is_bot_message: true,
                        reply_to_message_id: data.replyToMessageId,
                        // Stamp the Telegram message id so post-hoc
                        // "which bot send corresponds to Telegram
                        // message X" queries match the orchestrator
                        // text-reply path (`src/index.ts:1659`) and
                        // the `send_message` handler. Pre-#428
                        // sendFile returned void so this column was
                        // unavailable; now that we have the id, no
                        // reason not to record it.
                        telegram_message_id: sentFileMsgId,
                      });
                    } else if (cleanCaption && !captionDelivered) {
                      logger.warn(
                        {
                          chatJid: data.chatJid,
                          hostPath,
                          captionLen: cleanCaption.length,
                        },
                        'send_file: skipping caption storeMessage — sendFile returned no message id (delivery failed)',
                      );
                    }
                    // #722: same visible-send anchor release as the
                    // send_message path — a delivered file (with or
                    // without caption) is a visible reply.
                    if (typeof sentFileMsgId === 'string') {
                      deps.onVisibleReply?.(data.chatJid, sourceGroup);
                    }
                    logger.info(
                      { chatJid: data.chatJid, hostPath, sourceGroup },
                      'IPC file sent',
                    );
                  } else {
                    logger.warn(
                      { hostPath, containerPath, sourceGroup },
                      'send_file: file not found on host',
                    );
                  }
                }
              } else if (data.type === 'message' && data.chatJid && data.text) {
                logger.debug(
                  {
                    sourceGroup,
                    chatJid: data.chatJid,
                    rawTextLen: data.text.length,
                    rawPreview: String(data.text).slice(0, 80),
                    hasSender: Boolean(data.sender),
                    senderValue: data.sender,
                    hasReplyTo: Boolean(data.replyToMessageId),
                    hasPin: Boolean(data.pin),
                    ipcFile: file,
                  },
                  '[ipc] Received send_message IPC',
                );
                // Strip <internal> tags via the shared helper so this
                // path can't drift from the send_file caption path
                // above. If nothing remains, skip silently.
                const strippedText = stripInternalTags(data.text);
                if (!strippedText) {
                  logger.debug(
                    { sourceGroup },
                    '[ipc] send_message suppressed (all internal)',
                  );
                  fs.unlinkSync(filePath);
                  continue;
                }
                // Tag maintenance-session text so Baruch can tell a
                // scheduled-task reply from a live conversational one.
                // Applied AFTER internal-tag stripping (no point
                // prefixing text we're about to suppress) and BEFORE
                // both the Telegram send and the messages.db store, so
                // the prefix flows through accounting uniformly.
                const cleanText = applyMaintenancePrefix(
                  strippedText,
                  typeof data.sessionName === 'string'
                    ? data.sessionName
                    : undefined,
                );
                logger.debug(
                  {
                    sourceGroup,
                    chatJid: data.chatJid,
                    cleanLen: cleanText.length,
                    cleanPreview: cleanText.slice(0, 80),
                  },
                  '[ipc] send_message after stripInternalTags + maintenance-prefix',
                );

                // Authorization: verify this group can send to this chatJid
                const targetGroup = registeredGroups[data.chatJid];
                const authOk =
                  isMain ||
                  Boolean(targetGroup && targetGroup.folder === sourceGroup);
                logger.debug(
                  {
                    sourceGroup,
                    chatJid: data.chatJid,
                    isMain,
                    targetGroupFolder: targetGroup?.folder,
                    authOk,
                  },
                  '[ipc] send_message auth check',
                );
                if (authOk) {
                  const usePool = Boolean(
                    data.sender && data.chatJid.startsWith('tg:'),
                  );
                  logger.debug(
                    {
                      sourceGroup,
                      chatJid: data.chatJid,
                      path: usePool ? 'pool' : 'direct',
                      sender: data.sender,
                    },
                    '[ipc] send_message path decision',
                  );
                  // Capture whichever send path's message ID applies. Both
                  // `sendPoolMessage` and `deps.sendMessage` return the
                  // Telegram-native message ID (or undefined if the send
                  // failed or the channel isn't Telegram). Stored on the
                  // messages row so "which bot send produced Telegram ID X"
                  // is queryable without log spelunking.
                  // Normalize immediately: both send paths can return
                  // `string | void | undefined`. Collapsing to the
                  // `string | undefined` domain up front keeps downstream
                  // uses (`pinMessage`, `storeMessage`) type-safe without
                  // truthiness checks that would also drop legitimate
                  // empty-string / '0' IDs if Telegram ever returns them.
                  let sentMsgId: string | undefined;
                  if (usePool) {
                    // `usePool` is only true when `data.sender` is a non-
                    // empty string — TS just can't re-narrow across the
                    // intermediate `Boolean(...)` boundary. The `!` is
                    // safe by the `usePool` definition directly above.
                    const poolResult = await sendPoolMessage(
                      data.chatJid,
                      cleanText,
                      data.sender!,
                      sourceGroup,
                    );
                    sentMsgId =
                      typeof poolResult === 'string' ? poolResult : undefined;
                    logger.debug(
                      {
                        sourceGroup,
                        chatJid: data.chatJid,
                        sentMsgId,
                      },
                      '[ipc] sendPoolMessage returned',
                    );
                  } else {
                    const directResult = await deps.sendMessage(
                      data.chatJid,
                      cleanText,
                      data.replyToMessageId,
                    );
                    sentMsgId =
                      typeof directResult === 'string'
                        ? directResult
                        : undefined;
                    logger.debug(
                      {
                        sourceGroup,
                        chatJid: data.chatJid,
                        sentMsgId,
                      },
                      '[ipc] deps.sendMessage returned',
                    );
                    // Pin the message if requested
                    if (data.pin && sentMsgId && deps.pinMessage) {
                      await deps.pinMessage(data.chatJid, sentMsgId);
                      logger.debug(
                        { sourceGroup, chatJid: data.chatJid, sentMsgId },
                        '[ipc] pinMessage returned',
                      );
                    }
                  }
                  // #722: a confirmed visible reply releases the target
                  // chat's reply anchor at the send boundary — the SDK
                  // result's mark-displayed consumption arrives too late
                  // for pipes landing in the gap. Own-chat gating lives
                  // in the consumer (src/index.ts).
                  if (typeof sentMsgId === 'string') {
                    deps.onVisibleReply?.(data.chatJid, sourceGroup);
                  }
                  // Gate the bot-row write on send success — see the
                  // `shouldStoreBotMessage` helper for the full rationale
                  // (phantom rows on swallowed Telegram sends would
                  // silence the heartbeat / unanswered-cron and feed
                  // cascading hallucinated quote-replies downstream).
                  if (shouldStoreBotMessage(data.chatJid, sentMsgId)) {
                    const botRowId = `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
                    storeMessage({
                      id: botRowId,
                      chat_jid: data.chatJid,
                      sender: data.sender || ASSISTANT_NAME,
                      sender_name: data.sender || ASSISTANT_NAME,
                      content: cleanText,
                      timestamp: new Date().toISOString(),
                      is_from_me: true,
                      is_bot_message: true,
                      reply_to_message_id: data.replyToMessageId,
                      telegram_message_id: sentMsgId,
                    });
                    logger.info(
                      {
                        chatJid: data.chatJid,
                        sourceGroup,
                        botRowId,
                        contentLen: cleanText.length,
                      },
                      '[ipc] send_message complete — DB row written',
                    );
                  } else {
                    logger.error(
                      {
                        chatJid: data.chatJid,
                        sourceGroup,
                        contentLen: cleanText.length,
                      },
                      '[ipc] send_message failed — Telegram returned no message id; skipping DB row to avoid phantom bot reply (would silence heartbeat / unanswered alerts)',
                    );
                  }
                } else {
                  logger.warn(
                    {
                      chatJid: data.chatJid,
                      sourceGroup,
                      targetGroupFolder: targetGroup?.folder,
                    },
                    '[ipc] Unauthorized IPC message attempt blocked',
                  );
                }
              }
              fs.unlinkSync(filePath);
            } catch (err) {
              // Quarantine a bad file (recoverable data/transport failure) and
              // continue; a programming defect propagates.
              if (!isRecoverableIpcError(err)) throw err;
              // Catches anything thrown above (JSON.parse, auth, send, storeMessage).
              // If this fires AFTER the send already landed in Telegram, the
              // user sees a message with no DB row — exactly the ghost-heartbeat
              // symptom. Log the err, the data type, and a preview so the
              // operator can correlate with what appeared in the chat.
              logger.error(
                {
                  file,
                  sourceGroup,
                  err,
                  dataType: (data as { type?: string } | undefined)?.type,
                  dataChatJid: (data as { chatJid?: string } | undefined)
                    ?.chatJid,
                  dataTextPreview:
                    typeof (data as { text?: string } | undefined)?.text ===
                    'string'
                      ? (data as { text: string }).text.slice(0, 200)
                      : undefined,
                },
                '[ipc] Error processing IPC message — message may have been sent to the chat before the throw, in which case no DB row will exist',
              );
              moveIpcFileToErrors(ipcBaseDir, sourceGroup, file, filePath);
            }
          }
        }
      } catch (err) {
        if (!isFsErrorWithCode(err, IPC_FS_CODES)) throw err;
        logger.error(
          { err, sourceGroup },
          'Error reading IPC messages directory',
        );
      }

      // Process tasks from this group's IPC directory
      try {
        if (fs.existsSync(tasksDir)) {
          const taskFiles = fs
            .readdirSync(tasksDir)
            .filter((f) => f.endsWith('.json'));
          for (const file of taskFiles) {
            const filePath = path.join(tasksDir, file);
            try {
              const stat = fs.statSync(filePath);
              if (stat.size > 1_048_576) {
                logger.warn(
                  { file, sourceGroup, size: stat.size },
                  'IPC task file exceeds 1MB limit, moving to errors',
                );
                const errorDir = path.join(ipcBaseDir, 'errors');
                fs.mkdirSync(errorDir, { recursive: true });
                fs.renameSync(
                  filePath,
                  path.join(errorDir, `${sourceGroup}-${file}`),
                );
                continue;
              }
              const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
              // Pass source group identity to processTaskIpc for authorization
              await processTaskIpc(data, sourceGroup, isMain, deps);
              fs.unlinkSync(filePath);
            } catch (err) {
              // Quarantine a bad file (recoverable data/transport failure) and
              // continue; a programming defect propagates.
              if (!isRecoverableIpcError(err)) throw err;
              logger.error(
                { file, sourceGroup, err },
                'Error processing IPC task',
              );
              moveIpcFileToErrors(ipcBaseDir, sourceGroup, file, filePath);
            }
          }
        }
      } catch (err) {
        if (!isFsErrorWithCode(err, IPC_FS_CODES)) throw err;
        logger.error({ err, sourceGroup }, 'Error reading IPC tasks directory');
      }
    }

    if (!stopped) pollTimer = setTimeout(processIpcFiles, IPC_POLL_INTERVAL);
  };

  processIpcFiles();
  logger.info('IPC watcher started (per-group namespaces)');
  return () => {
    stopped = true;
    if (pollTimer) clearTimeout(pollTimer);
    ipcWatcherRunning = false;
  };
}

export async function processTaskIpc(
  data: IpcTaskPayload,
  sourceGroup: string, // Verified identity from IPC directory
  isMain: boolean, // Verified from directory path
  deps: IpcDeps,
): Promise<void> {
  // #845: commands with a registered handler dispatch through the IPC
  // registry; everything still in the legacy switch below falls through
  // until its migration slice lands.
  registerCoreIpcHandlers();
  if (await dispatchIpcTask({ data, sourceGroup, isMain, deps })) return;

  switch (data.type) {
    // --- Named host operations ---

    case 'persist_tz_segments': {
      // #748 — credential-free host ingestion for the in-container TripIt →
      // Reclaim sync. The sync itself runs in the agent container now (creds
      // swapped at the OneCLI gateway), so the host no longer runs the CLI or
      // holds TripIt/Reclaim/Google secrets. What stays host-side is the
      // `tz_state` write: the container hands back the parsed `segments[]` and
      // the host persists them exactly as the removed `sync_tripit` host-op's
      // success path did — same `applyTripitSegmentsToTzState` + `#584`
      // `onTzFlipped`
      // next_run invalidation, so the scheduler-timezone skill, the 30-min
      // heartbeat advisory walker, and the #574 location cascade keep their
      // owner-tz backbone. The deny (non-main) / reject (over-cap) / persist
      // decision — the security-sensitive part — lives in the pure, tested
      // `classifyTzPersist`; this case just carries it out (log + result file +
      // DB write). See `classifyTzPersist` for the isMain-re-check and cap
      // rationale.
      const decision = classifyTzPersist(isMain, data.segments);
      if (decision.action === 'deny') {
        logger.warn(
          { sourceGroup },
          'Unauthorized persist_tz_segments attempt blocked (non-main container)',
        );
        if (data.requestId) {
          fs.writeFileSync(
            scriptResultPath(sourceGroup, data),
            JSON.stringify({ error: decision.error }),
          );
        }
        break;
      }
      if (data.requestId) {
        const resultPath = scriptResultPath(sourceGroup, data);
        if (decision.action === 'reject') {
          // A malformed (non-array) or over-cap payload — refuse rather than
          // clear/mangle owner tz_state. See `classifyTzPersist`.
          logger.warn(
            { sourceGroup, reason: decision.error },
            'persist_tz_segments: refused (not persisted)',
          );
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: decision.error }),
          );
          break;
        }
        // Failures from the persistence helper (SqliteError, programming
        // bugs) propagate per `coding-policy: error-handling`; only the
        // #584 recompute's transient SQLITE_BUSY/LOCKED is swallowed-with-warn
        // inside the writer, exactly as the removed host-op did.
        const { segments } = decision;
        applyTripitSegmentsToTzState({ segments }, new Date(), () => {
          recomputeLocalSchedules(getCurrentTz, new Date());
        });
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            stdout: JSON.stringify({ persisted: segments.length }),
          }),
        );
        logger.info(
          { sourceGroup, segmentCount: segments.length },
          'persist_tz_segments completed',
        );
      }
      break;
    }

    case 'fetch_markdown':
      if (data.requestId) {
        const resultPath = scriptResultPath(sourceGroup, data);

        const parsed = parseFetchMarkdownUrl(data.url);
        if (!parsed.ok) {
          fs.writeFileSync(resultPath, JSON.stringify({ error: parsed.error }));
          break;
        }
        const parsedUrl = parsed.url;

        const flags = buildSnitchmdFlags(data);

        // Cache directory on the HOST filesystem — the snitchmd sibling
        // container is launched via the orchestrator's docker.sock, so
        // -v mount paths must reference the host's view (HOST_PROJECT_ROOT),
        // not the orchestrator's /app/store. Same convention as the agent
        // container mounts in container-runner.ts.
        const hostCacheDir = path.join(
          HOST_PROJECT_ROOT,
          'store',
          'snitchmd-cache',
        );
        // Also mkdir locally so the path is visible to the orchestrator
        // process (e.g. for size-rollup or cleanup tooling). The actual
        // write into the cache is performed by the sibling container as
        // its own user; we just want the directory to exist.
        fs.mkdirSync(path.join(STORE_DIR, 'snitchmd-cache'), {
          recursive: true,
        });

        // snitchmd is an app-level dependency (a renderer + content
        // extractor), not an API contract — we WANT to ride the latest
        // CloakBrowser fingerprint updates as anti-bot detection
        // evolves, and snitchmd's output shape is stable across point
        // releases. Operators who need a reproducible build can pin to
        // a specific tag or `sha256:…` digest via `SNITCHMD_IMAGE`
        // without a code change.
        const snitchmdImage =
          process.env.SNITCHMD_IMAGE || 'syabro/snitchmd:latest';

        logger.info(
          {
            sourceGroup,
            host: parsedUrl.host,
            flags: flags.filter((f) => !f.startsWith('http')),
          },
          'Running fetch_markdown',
        );

        execFile(
          'docker',
          [
            'run',
            '--rm',
            '-v',
            `${hostCacheDir}:/cache`,
            snitchmdImage,
            parsedUrl.toString(),
            ...flags,
          ],
          {
            // First call cold-pulls the image — give it room. Subsequent
            // calls are cached and complete in well under 30s.
            timeout: 240_000,
            // snitchmd's markdown output can run into the megabytes for
            // wiki/long articles; cap at 16 MiB so a runaway page doesn't
            // exhaust orchestrator memory before the agent's --max-chars
            // truncation kicks in.
            maxBuffer: 16 * 1024 * 1024,
          },
          (error, stdout, stderr) => {
            if (error) {
              // execFile attaches `code` (numeric exit code or string like
              // ETIMEDOUT) and `killed` (timeout signal). Surface both so
              // runHostOperation's diagnostic relay can show the agent
              // which failure mode hit (cf. #146 retry context).
              const execErr = error as NodeJS.ErrnoException & {
                code?: number | string;
                killed?: boolean;
              };
              // Per `jbaruch/coding-policy: no-secrets`: Node's `execFile`
              // packs the full command line — including the target URL —
              // into `error.message` as `Command failed: docker run ...
              // <url> --json ...`. A URL with query-string auth (session
              // token, signed-URL signature) would leak into both the
              // structured log AND the result-file the agent reads. Emit
              // a fixed-shape message that names only the host + exit
              // mode; the host attribute itself isn't a secret and the
              // operator can correlate with the structured log.
              const safeError = `fetch_markdown failed for ${parsedUrl.host} (exit_code: ${execErr.code ?? 'unknown'}${execErr.killed ? ', killed' : ''})`;
              // Defense-in-depth: snitchmd's stderr is normally just
              // `snitchmd: title=... quality=... chars=...`, but a
              // Playwright / Chromium fault could echo the input URL.
              // Scrub the exact URL we passed in before writing so a
              // query-string auth secret can't leak via stderr.
              const urlString = parsedUrl.toString();
              const safeStderr = stderr
                .slice(-2000)
                .split(urlString)
                .join('<URL>');
              logger.warn(
                {
                  sourceGroup,
                  host: parsedUrl.host,
                  exitCode: execErr.code,
                  killed: execErr.killed,
                  stderr: safeStderr.slice(-500),
                },
                'fetch_markdown failed',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: safeError,
                  exit_code: execErr.code,
                  killed: execErr.killed,
                  stderr: safeStderr,
                }),
              );
              return;
            }
            // snitchmd --json writes a single JSON object on stdout.
            // Production smoke-test caught CloakBrowser leaking
            // "Downloading newer chromium" log lines onto stdout BEFORE
            // the JSON on the cold-pull path, so we use a tolerant
            // parser that retries with a `{`-to-`}` slice when the
            // strict parse fails. The successful payload's markdown
            // body is passed up as plain text (the <untrusted-input>
            // envelope is applied client-side via the READ_TOOL_PATTERNS
            // row in untrusted-input-wrap.ts — #321).
            const parseResult = parseSnitchmdStdout(stdout);
            if (!parseResult.ok) {
              // Per `jbaruch/coding-policy: no-secrets`: snitchmd's
              // JSON payload always carries `url` and `final_url`
              // fields, and the raw bytes we couldn't parse may still
              // contain those substrings. Scrub the input URL and the
              // host stderr's same vector before persisting the
              // diagnostic so a query-string auth secret doesn't ride
              // the parse-failure path back to the agent.
              const urlString = parsedUrl.toString();
              const safeStderr = stderr
                .slice(-2000)
                .split(urlString)
                .join('<URL>');
              const safeStdout = stdout
                .slice(-2000)
                .split(urlString)
                .join('<URL>');
              logger.warn(
                {
                  sourceGroup,
                  host: parsedUrl.host,
                  stdoutLen: stdout.length,
                },
                'fetch_markdown: stdout did not parse as JSON',
              );
              fs.writeFileSync(
                resultPath,
                JSON.stringify({
                  error: 'fetch_markdown: snitchmd produced non-JSON output',
                  stderr: safeStderr,
                  stdout: safeStdout,
                }),
              );
              return;
            }
            const payload = parseResult.payload;
            const markdown = payload.markdown ?? '';
            const header = formatSnitchmdHeader(payload, parsedUrl.toString());
            logger.info(
              {
                sourceGroup,
                host: parsedUrl.host,
                chars: payload.chars ?? markdown.length,
                quality: payload.quality,
              },
              'fetch_markdown completed',
            );
            fs.writeFileSync(
              resultPath,
              JSON.stringify({
                stdout: header + markdown,
                stderr: stderr.slice(-500) || undefined,
              }),
            );
          },
        );
      }
      break;

    case 'github_backup':
      if (data.requestId) {
        // Authorization: github_backup performs a host-side filesystem
        // sync + `git push` using GITHUB_TOKEN — same privilege class
        // as `audible_backup`, `promote_staging`, all
        // of which gate on `isMain`. Untrusted-tier groups have a
        // separate `#324` token gate at the container layer (the
        // confirmation-tokens hook), but the host-side gate here
        // shrinks the blast radius further: a compromised non-main
        // container can't trigger the backup pipeline by writing an
        // IPC task file directly.
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized github_backup attempt');
          break;
        }

        const backupDir = path.join(GROUPS_DIR, sourceGroup, 'backup-repo');
        const dbPath = path.join(STORE_DIR, 'messages.db');
        const resultPath = scriptResultPath(sourceGroup, data);

        // Sync live group state into backup-repo BEFORE git plumbing.
        // The sync mirrors groups/global/ and every non-hidden
        // groups/<name>/ subdir into backup-repo/{global,groups/<name>}
        // with delete-on-missing semantics, and dumps the SQLite
        // state-table surface into backup-repo/state/<table>.sql.
        // Policy: back up everything under groups/ that the runtime
        // mutates and that ISN'T reproducible from deploy or
        // `tessl install` — see `src/backup-sync.ts` for the denylist
        // (.tessl, .claude, node_modules, dist, logs, tmp,
        // conversations, *.bak-*, etc.).
        // syncBackupRepo also validates groupsRoot / backupDir
        // existence and throws an actionable error if either is
        // missing — the catch block below converts that into a
        // structured `{ error, stage: 'sync' }` envelope.
        let syncSummary: SyncResult;
        try {
          syncSummary = syncBackupRepo({
            groupsRoot: GROUPS_DIR,
            backupDir,
            dbPath,
          });
        } catch (e) {
          // Non-Error throws (TypeScript allows `throw 42`) bubble up
          // — those indicate a bug, not an operational sync failure.
          if (!(e instanceof Error)) throw e;
          logger.error(
            {
              sourceGroup,
              groupsRoot: GROUPS_DIR,
              backupDir,
              dbPath,
              error: e.message,
            },
            'github_backup sync failed',
          );
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: e.message, stage: 'sync' }),
          );
          break;
        }

        const commitMsg =
          data.message || `backup: ${new Date().toISOString().split('T')[0]}`;
        logger.info(
          {
            sourceGroup,
            backupDir,
            commitMsg,
            copied: syncSummary.copied.length,
            removed: syncSummary.removed.length,
            dumped: syncSummary.dumped.length,
            skipped: syncSummary.skipped.length,
          },
          'Running github_backup',
        );

        // Read GitHub token for push auth
        const { readEnvFile: readBackupEnv } = await import('./env.js');
        const backupEnvVars = readBackupEnv(['GITHUB_TOKEN']);
        const ghToken = backupEnvVars.GITHUB_TOKEN;

        // Run the commit+push off the IPC loop (like persist_global_file):
        // the promise resolves with the result envelope; the loop never
        // blocks on git. `backupCommitAndPush` is the testable git boundary
        // — argument-array git invocations, so the agent-supplied commit
        // message can't inject host commands (#725).
        backupCommitAndPush({
          backupDir,
          message: commitMsg,
          token: ghToken,
        }).then((backupResult) => {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ ...backupResult, sync_summary: syncSummary }),
          );
          if (backupResult.error) {
            logger.error(
              {
                sourceGroup,
                stage: backupResult.stage,
                error: backupResult.error,
              },
              'github_backup failed',
            );
          } else {
            logger.info(
              { sourceGroup, committed: backupResult.committed },
              'github_backup completed',
            );
          }
        });
      }
      break;

    case 'persist_global_file':
      if (data.requestId) {
        // Authorization: persist_global_file commits + pushes the
        // orchestrator repo's `main` using GITHUB_TOKEN — same privilege
        // class as `github_backup` / `promote_staging`, gated on `isMain`.
        // A non-main container can't reach the persona source even by
        // writing an IPC task file directly.
        if (!isMain) {
          logger.warn(
            { sourceGroup },
            'Unauthorized persist_global_file attempt',
          );
          break;
        }

        const persistResultPath = scriptResultPath(sourceGroup, data);

        // The container edits `/workspace/global/<file>`, an RW bind onto the
        // host's `groups/global/<file>` (the git-tracked, deploy-seeded
        // source). That edit lands in the working tree but is never
        // committed, so `deploy.sh`'s `git stash; git pull` discards it on
        // the next deploy (jbaruch/nanoclaw-admin#393). This handler commits
        // the working-tree change and pushes `main` so the next `git pull`
        // keeps it. The allowlist gate (`validateGlobalFilesToPersist`)
        // rejects anything but the exact persona basenames, so a compromised
        // container can't commit arbitrary tracked files (path traversal,
        // secrets, workflow YAML).
        const persistValidation = validateGlobalFilesToPersist(data.files);
        if (!persistValidation.ok) {
          logger.warn(
            { sourceGroup, invalidFile: persistValidation.invalid },
            'persist_global_file rejected non-allowlisted file',
          );
          fs.writeFileSync(
            persistResultPath,
            JSON.stringify({
              error: `persist_global_file: ${JSON.stringify(persistValidation.invalid)} is not an allowed global file (allowed: ${PERSISTABLE_GLOBAL_FILES.join(', ')})`,
              stage: 'validate',
            }),
          );
          break;
        }
        const persistRelPaths = persistValidation.relPaths;

        const persistCommitMsg =
          data.message ||
          `soul: persist approved updates ${new Date().toISOString().split('T')[0]}`;

        logger.info(
          {
            sourceGroup,
            relPaths: persistRelPaths,
            commitMsg: persistCommitMsg,
          },
          'Running persist_global_file',
        );

        // Read GitHub token for push auth (same wiring as github_backup).
        const { readEnvFile: readPersistEnv } = await import('./env.js');
        const persistGhToken = readPersistEnv(['GITHUB_TOKEN']).GITHUB_TOKEN;

        // The orchestrator's cwd (`/app`) is the built image dir, not a git
        // worktree (#471), so persist can't commit in place. It operates on a
        // dedicated self-provisioning clone of the deploy-source repo under
        // `data/` (gitignored, mounted, persistent), applies the live runtime
        // edits from `groups/global/` into it, then commits + pushes HEAD:main.
        const personaRepoDir = path.join(DATA_DIR, 'persona-repo');

        // Run clone + copy + commit + push off the IPC loop (like
        // github_backup): the loop never blocks on git. `runPersonaPersistTask`
        // is the testable boundary (ensure-clone → overlay → commit/push,
        // always writing a result envelope). Serialized so two closely-timed
        // persists can't race on the shared clone.
        runSerializedPersonaPersist(() =>
          runPersonaPersistTask({
            personaRepoDir,
            groupsDir: GROUPS_DIR,
            relPaths: persistRelPaths,
            remoteUrl: ORCHESTRATOR_REPO_URL,
            message: persistCommitMsg,
            token: persistGhToken,
            resultPath: persistResultPath,
            sourceGroup,
          }),
        );
      }
      break;

    case 'run_sidecar':
      if (data.requestId) {
        // Authorization: run_sidecar spawns a privileged docker sidecar with
        // host bind-mounts defined in the trusted registry — same privilege
        // class as the former audible_backup case, gated on `isMain`. The
        // plugin supplies only the sidecar NAME and registry-allowlisted
        // flags; the image and mount paths never come from the payload, so a
        // compromised non-main container can't request `-v /:/…` (see
        // src/sidecar-runner.ts).
        if (!isMain) {
          logger.warn(
            { sourceGroup, name: data.name },
            'Unauthorized run_sidecar attempt',
          );
          break;
        }

        const sidecarResultPath = scriptResultPath(sourceGroup, data);

        // IPC payloads are raw JSON, so validate the shape at the boundary and
        // write an actionable error envelope rather than let a malformed
        // request (e.g. `flags: "--dry-run"`) crash runSidecar or silently
        // hang the polling MCP caller.
        if (typeof data.name !== 'string' || data.name.length === 0) {
          fs.writeFileSync(
            sidecarResultPath,
            JSON.stringify({
              error: 'run_sidecar: "name" must be a non-empty string.',
            }),
          );
          break;
        }
        if (
          data.flags !== undefined &&
          !(
            Array.isArray(data.flags) &&
            data.flags.every((f) => typeof f === 'string')
          )
        ) {
          fs.writeFileSync(
            sidecarResultPath,
            JSON.stringify({
              error: 'run_sidecar: "flags" must be an array of strings.',
            }),
          );
          break;
        }

        const sidecarName = data.name;
        const sidecarFlags = data.flags as string[] | undefined;

        // Fire-and-forget like github_backup / persist_global_file: a sidecar
        // run can take up to its registry timeout (600s for audible) and must
        // not block the IPC loop. runSidecar resolves an error envelope for
        // operational failures (unknown sidecar, disallowed flag, non-zero
        // docker exit); the #625 partial-success relay lives inside it.
        runSidecar({ name: sidecarName, flags: sidecarFlags })
          .then((payload) => {
            fs.writeFileSync(sidecarResultPath, JSON.stringify(payload));
          })
          // outer-boundary-process-contract: the MCP caller polls for the
          // result file and reads its absence as a silent hang until a 10-min
          // timeout. This catch emits an `{ error }` envelope for a genuine
          // relay bug (a non-SyntaxError stdout parse rejects runSidecar) so
          // the caller gets an actionable failure instead of hanging; letting
          // it propagate would write no file and break that contract.
          .catch((err: unknown) => {
            const message = err instanceof Error ? err.message : String(err);
            logger.error(
              { sourceGroup, name: sidecarName, error: message },
              'run_sidecar failed unexpectedly',
            );
            fs.writeFileSync(
              sidecarResultPath,
              JSON.stringify({
                error: `run_sidecar failed unexpectedly: ${message}`,
              }),
            );
          });
      }
      break;

    case 'promote_staging':
      if (data.requestId && data.tileName && data.skillName) {
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized promote_staging attempt');
          break;
        }

        const promoteResultPath = scriptResultPath(sourceGroup, data);

        if (!KNOWN_TILE_NAMES.has(data.tileName)) {
          logger.warn(
            { sourceGroup, tileName: data.tileName },
            'promote_staging rejected: tileName not in allowlist',
          );
          fs.writeFileSync(
            promoteResultPath,
            JSON.stringify({
              error: `Invalid tileName "${data.tileName}". Allowed: ${[...KNOWN_TILE_NAMES].join(', ')}.`,
            }),
          );
          break;
        }

        const promoteScript = path.join(
          process.cwd(),
          'scripts',
          'promote-to-tile-repo.sh',
        );

        if (!fs.existsSync(promoteScript)) {
          fs.writeFileSync(
            promoteResultPath,
            JSON.stringify({
              error: 'promote-to-tile-repo.sh not found',
            }),
          );
          break;
        }

        const stagingDir = path.join(
          GROUPS_DIR,
          sourceGroup,
          'staging',
          data.tileName,
        );

        // Read credentials from .env for tile repo push
        const envPath = path.join(process.cwd(), '.env');
        const envContent = fs.existsSync(envPath)
          ? fs.readFileSync(envPath, 'utf-8')
          : '';
        const getEnv = (key: string) =>
          envContent
            .split('\n')
            .find((l) => l.startsWith(`${key}=`))
            ?.split('=')
            .slice(1)
            .join('=') || '';

        logger.info(
          { sourceGroup, tileName: data.tileName, skillName: data.skillName },
          'Running promote_staging',
        );

        execFile(
          'bash',
          [promoteScript, stagingDir, data.tileName, data.skillName],
          {
            // 15 minutes. promote-to-tile-repo.sh runs `tessl skill
            // review --optimize` on each staged skill, and tessl
            // itself tells you that each review "can take up to 1
            // minute." A bulk promote (`skillName=all`) against a tile
            // with 10+ staged skills easily blows past the old
            // 5-minute cap, which observably killed every bulk promote
            // AyeAye tried and returned a mid-run truncated error. 15
            // min fits the typical 10-15 skill worst case with
            // headroom; larger bulk promotes should split the staging
            // directory into smaller batches — every skill is reviewed
            // (no per-skill opt-out).
            timeout: 900_000,
            maxBuffer: 5 * 1024 * 1024,
            env: {
              ...process.env,
              GITHUB_TOKEN: getEnv('GITHUB_TOKEN'),
              TILE_OWNER: getEnv('TILE_OWNER') || 'jbaruch',
              ASSISTANT_NAME: getEnv('ASSISTANT_NAME') || 'AyeAye',
            },
          },
          (error, stdout, stderr) => {
            if (error) {
              logger.error(
                {
                  sourceGroup,
                  error: error.message,
                  stderr: stderr.slice(-500),
                },
                'promote_staging failed',
              );
              fs.writeFileSync(
                promoteResultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
            } else {
              logger.info(
                { sourceGroup },
                'promote_staging opened PR on tile repo',
              );
              fs.writeFileSync(
                promoteResultPath,
                JSON.stringify({ stdout: stdout.trim() }),
              );

              // Post-promote `tessl update` + session clear used to
              // run here on a 5-minute delay, predicated on the old
              // flow that pushed directly to tile main and triggered
              // GHA publish within ~5min. New flow opens a PR and
              // requests Copilot review; publish only happens after
              // the PR is merged (which could be minutes, hours, or
              // never if Copilot/human rejects it). The auto-update
              // would fire against a registry that hasn't changed
              // yet — at best a no-op, at worst tearing down sessions
              // for no reason. The agent now calls the `tessl_update`
              // MCP tool explicitly after the PR merges; a periodic
              // 15-min catch-up in index.ts covers missed invocations.
            }
          },
        );
      }
      break;

    case 'tessl_update':
      if (data.requestId) {
        const tesslResultPath = scriptResultPath(sourceGroup, data);
        if (!isMain) {
          logger.warn({ sourceGroup }, 'Unauthorized tessl_update attempt');
          fs.writeFileSync(
            tesslResultPath,
            JSON.stringify({
              error: 'Only the main group can trigger tessl_update.',
            }),
          );
          break;
        }

        // #496 — gate the on-demand update on the same fresh-lock check
        // the periodic catch-up uses. A scheduled task mid-flight has
        // its skill holding `follow_me_tasks.pending_run_at`; running
        // tessl_update now would force-close the maintenance container
        // and orphan the lock. Defer with a structured response so the
        // calling agent can reason about retrying. Threshold is the same
        // 1h freshness window documented at the constant's definition
        // in `src/index.ts` — duplicated here as a literal because
        // ipc.ts has no shared-config home for it and lifting the
        // constant would force a circular import via index.ts.
        const PENDING_RUN_AT_FRESHNESS_MS = 60 * 60 * 1000;
        let activeLocks: string[];
        try {
          activeLocks = getActivePendingRunAtNames(PENDING_RUN_AT_FRESHNESS_MS);
        } catch (lockErr) {
          if (!(lockErr instanceof Error)) throw lockErr;
          // Don't fail the update on a lock-check error — log and
          // proceed. Failing closed would block legitimate updates
          // forever if the DB or schema regressed.
          logger.warn(
            { sourceGroup, err: lockErr.message },
            'tessl_update: pending_run_at check failed — proceeding without deferral',
          );
          activeLocks = [];
        }
        if (activeLocks.length > 0) {
          logger.info(
            { sourceGroup, activeLocks },
            'tessl_update deferred — scheduled task(s) mid-flight with fresh pending_run_at lock (#496)',
          );
          fs.writeFileSync(
            tesslResultPath,
            JSON.stringify({
              stdout: `Deferred: scheduled task(s) currently mid-flight with fresh pending_run_at lock (${activeLocks.join(', ')}). Retry after the task(s) finish — running tessl_update now would force-close the maintenance container and orphan the lock (#496).`,
              deferred: true,
              activeLocks,
            }),
          );
          break;
        }

        logger.info({ sourceGroup }, 'Running tessl_update');

        execFile(
          'bash',
          [
            '-c',
            'cd /app/tessl-workspace && tessl update --yes --accept-warnings --agent claude-code 2>&1',
          ],
          { timeout: 150_000, maxBuffer: 2 * 1024 * 1024 },
          (error, stdout) => {
            if (error) {
              logger.error(
                {
                  sourceGroup,
                  error: error.message,
                  output: stdout.slice(-500),
                },
                'tessl_update failed',
              );
              fs.writeFileSync(
                tesslResultPath,
                JSON.stringify({
                  error: error.message,
                  stdout: stdout.slice(-2000),
                }),
              );
              return;
            }
            const output = stdout.trim();
            // `tessl update` prints "Updated ..." when a tile actually
            // moved forward. No-op runs don't, and clearing sessions on a
            // no-op would nuke conversation state for nothing — hence
            // the string check instead of an unconditional clear.
            if (/\bUpdated\b/.test(output)) {
              const cleared = deleteAllSessions();
              // Close every currently-running container so it respawns on
              // the next inbound message with the freshly-installed tile
              // content. `deleteAllSessions()` only resets the SDK
              // session-id mapping — without this companion call, a
              // container that's mid-idle-loop right now keeps its old
              // skills/.tessl/ snapshot until its 30-min idle timeout
              // fires (issue #64). The two operations are paired: clear
              // the on-disk session state AND signal the live containers
              // so neither lingers on stale state.
              //
              // Guarded because `closeAllActiveContainers()` rethrows
              // unexpected (non-fs) errors by contract. Without the
              // guard, a programming bug would skip the result-file
              // write below and leave the IPC requester hanging without
              // a structured payload. On failure we still write a
              // partial-success JSON so the caller can distinguish
              // "containers signaled" from "sessions cleared but
              // signaling failed."
              let closed = 0;
              let closeErr: Error | null = null;
              try {
                closed = deps.closeAllActiveContainers();
              } catch (e) {
                // See src/index.ts periodic update for the rationale —
                // outer-boundary guard around an async-callback
                // boundary. Narrow to Error so non-Error throws (a
                // programming bug throwing a non-Error value)
                // propagate per error-handling.md.
                if (!(e instanceof Error)) throw e;
                closeErr = e;
                logger.error(
                  { err: e, sourceGroup, sessionsCleared: cleared },
                  'closeAllActiveContainers threw an unexpected error during tessl_update — sessions still cleared, but live containers will not respawn until idle timeout',
                );
              }
              logger.info(
                {
                  sourceGroup,
                  sessionsCleared: cleared,
                  containersClosed: closed,
                  closeError: closeErr ? String(closeErr) : undefined,
                },
                'tessl_update found new tiles — sessions cleared and running containers signaled to restart',
              );
              fs.writeFileSync(
                tesslResultPath,
                JSON.stringify(
                  closeErr
                    ? {
                        stdout: `${output}\n\nSessions cleared: ${cleared}\nContainers signaled to restart: ${closed}`,
                        warning: `closeAllActiveContainers failed: ${String(closeErr)} — running containers will pick up new tiles on idle timeout (~30 min) instead of immediately`,
                      }
                    : {
                        stdout: `${output}\n\nSessions cleared: ${cleared}\nContainers signaled to restart: ${closed}`,
                      },
                ),
              );
            } else {
              logger.info(
                { sourceGroup },
                'tessl_update completed — no new tiles',
              );
              fs.writeFileSync(
                tesslResultPath,
                JSON.stringify({ stdout: output || '(no output)' }),
              );
            }
          },
        );
      }
      break;

    case 'list_installed_tiles': {
      // Admin tile only. Returns the names of every tile currently
      // installed in the local Tessl registry — the same set
      // `set_additional_tiles` validates against. The agent uses this
      // to surface "what overlay tiles can I set on this chat?" to
      // the operator before proposing a config change. Read-only;
      // never mutates the registry.
      //
      // The result includes BOTH overlay tiles and the trust-tier
      // baseline (`nanoclaw-core`, `nanoclaw-trusted`,
      // `nanoclaw-untrusted`, `nanoclaw-admin`). The agent presenting
      // this list is expected to call out the baseline names as
      // reserved — `selectTiles` dedupes them against the baseline so
      // configuring one as an overlay is a no-op, not an error, but
      // it's still a misuse of the surface. We don't filter here
      // because (a) the baseline set is policy that lives in
      // `selectTiles`, not registry truth, and (b) future trust-tier
      // changes shouldn't silently alter what this tool returns.
      const tilesResultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized list_installed_tiles attempt',
        );
        fs.writeFileSync(
          tilesResultPath,
          JSON.stringify({
            error: 'list_installed_tiles is admin-tile only',
          }),
        );
        break;
      }
      const tiles = getInstalledTiles();
      if (tiles === null) {
        // Registry directory doesn't exist (cold start, never ran
        // `tessl install`). Distinct from "registry exists but empty"
        // so the operator knows whether to run `tessl_update` first.
        logger.warn(
          { sourceGroup },
          'list_installed_tiles: registry directory absent — run tessl_update',
        );
        fs.writeFileSync(
          tilesResultPath,
          JSON.stringify({
            stdout: JSON.stringify({
              tiles: [],
              registryAbsent: true,
            }),
          }),
        );
        break;
      }
      logger.info(
        { sourceGroup, count: tiles.length },
        'list_installed_tiles served via IPC',
      );
      fs.writeFileSync(
        tilesResultPath,
        JSON.stringify({
          stdout: JSON.stringify({
            tiles,
            registryAbsent: false,
          }),
        }),
      );
      break;
    }

    case 'push_staged_to_branch':
      if (
        data.requestId &&
        data.tileName &&
        data.branch &&
        data.commitMessage
      ) {
        const pushResultPath = scriptResultPath(sourceGroup, data);
        if (!isMain) {
          logger.warn(
            { sourceGroup },
            'Unauthorized push_staged_to_branch attempt',
          );
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({
              error: 'Only the main group can push to tile branches.',
            }),
          );
          break;
        }

        if (!KNOWN_TILE_NAMES.has(data.tileName)) {
          logger.warn(
            { sourceGroup, tileName: data.tileName },
            'push_staged_to_branch rejected: tileName not in allowlist',
          );
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({
              error: `Invalid tileName "${data.tileName}". Allowed: ${[...KNOWN_TILE_NAMES].join(', ')}.`,
            }),
          );
          break;
        }

        const pushScript = path.join(
          process.cwd(),
          'scripts',
          'push-staged-to-branch.sh',
        );

        if (!fs.existsSync(pushScript)) {
          fs.writeFileSync(
            pushResultPath,
            JSON.stringify({ error: 'push-staged-to-branch.sh not found' }),
          );
          break;
        }

        const stagingDir = path.join(
          GROUPS_DIR,
          sourceGroup,
          'staging',
          data.tileName,
        );

        // Same .env reader pattern as promote_staging — the orchestrator
        // holds the tile-repo credentials, containers never see them.
        const envPath = path.join(process.cwd(), '.env');
        const envContent = fs.existsSync(envPath)
          ? fs.readFileSync(envPath, 'utf-8')
          : '';
        const getEnv = (key: string) =>
          envContent
            .split('\n')
            .find((l) => l.startsWith(`${key}=`))
            ?.split('=')
            .slice(1)
            .join('=') || '';

        logger.info(
          {
            sourceGroup,
            tileName: data.tileName,
            branch: data.branch,
            skillName: data.skillName || 'all',
          },
          'Running push_staged_to_branch',
        );

        execFile(
          'bash',
          [
            pushScript,
            stagingDir,
            data.tileName,
            data.branch,
            data.commitMessage,
            data.skillName || 'all',
          ],
          {
            // 5 min is enough for a clone-branch + copy + commit +
            // push. No tessl review loop here — fixups don't re-trigger
            // the local optimize pass.
            timeout: 300_000,
            maxBuffer: 2 * 1024 * 1024,
            env: {
              ...process.env,
              GITHUB_TOKEN: getEnv('GITHUB_TOKEN'),
              TILE_OWNER: getEnv('TILE_OWNER') || 'jbaruch',
              ASSISTANT_NAME: getEnv('ASSISTANT_NAME') || 'AyeAye',
            },
          },
          (error, stdout, stderr) => {
            if (error) {
              logger.error(
                {
                  sourceGroup,
                  error: error.message,
                  stderr: stderr.slice(-500),
                },
                'push_staged_to_branch failed',
              );
              fs.writeFileSync(
                pushResultPath,
                JSON.stringify({
                  error: error.message,
                  stderr: stderr.slice(-500),
                }),
              );
              return;
            }
            logger.info(
              {
                sourceGroup,
                tileName: data.tileName,
                branch: data.branch,
              },
              'push_staged_to_branch pushed fixup',
            );
            fs.writeFileSync(
              pushResultPath,
              JSON.stringify({ stdout: stdout.trim() }),
            );
          },
        );
      }
      break;

    default:
      logger.warn({ type: data.type }, 'Unknown IPC task type');
  }
}
