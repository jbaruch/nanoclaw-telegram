import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';

import { logger } from './logger.js';

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
    // (HEAD ahead of origin/main); otherwise it's a genuine no-op. A failing
    // `rev-list` (missing/broken `origin/main` tracking ref, corrupt ref) is a
    // real error — mapping it to "nothing pending" would silently no-op the
    // persist while a committed-but-unpushed persona change waits, and both
    // the push gate and the push below need that same ref anyway. Surface it
    // (#865, mirroring `backupCommitAndPush`).
    const ahead = await runGit(repoRoot, [
      'rev-list',
      '--count',
      'origin/main..HEAD',
    ]);
    if (ahead.code !== 0) return fail('git rev-list', ahead);
    const aheadCount = parseInt(ahead.stdout.trim(), 10);
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
