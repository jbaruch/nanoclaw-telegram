import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect } from 'vitest';

import {
  PERSISTABLE_GLOBAL_FILES,
  backupCommitAndPush,
  ensurePersonaRepo,
  persistGlobalFilesToGit,
  runPersonaPersistTask,
  runSerializedPersonaPersist,
  redactGitToken,
  validateGlobalFilesToPersist,
} from './git-persist.js';

describe('validateGlobalFilesToPersist', () => {
  it('defaults an empty/absent payload to every allowlisted persona file', () => {
    const expected = {
      ok: true,
      relPaths: PERSISTABLE_GLOBAL_FILES.map((f) => `groups/global/${f}`),
    };
    expect(validateGlobalFilesToPersist(undefined)).toEqual(expected);
    expect(validateGlobalFilesToPersist([])).toEqual(expected);
  });

  it('maps an explicit allowlisted subset to repo-relative paths', () => {
    expect(validateGlobalFilesToPersist(['SOUL.md'])).toEqual({
      ok: true,
      relPaths: ['groups/global/SOUL.md'],
    });
    expect(
      validateGlobalFilesToPersist(['SOUL-untrusted.md', 'SOUL.md']),
    ).toEqual({
      ok: true,
      relPaths: ['groups/global/SOUL-untrusted.md', 'groups/global/SOUL.md'],
    });
  });

  it('dedupes repeated entries so git add names each path once', () => {
    expect(validateGlobalFilesToPersist(['SOUL.md', 'SOUL.md'])).toEqual({
      ok: true,
      relPaths: ['groups/global/SOUL.md'],
    });
  });

  it('rejects a path-traversal attempt without committing it', () => {
    expect(validateGlobalFilesToPersist(['../../.env'])).toEqual({
      ok: false,
      invalid: '../../.env',
    });
  });

  it('rejects a tracked-but-not-allowlisted global file', () => {
    // CLAUDE.md is git-tracked under groups/global but is NOT persona content
    // the apply flow may rewrite — the allowlist must keep it out.
    expect(validateGlobalFilesToPersist(['CLAUDE.md'])).toEqual({
      ok: false,
      invalid: 'CLAUDE.md',
    });
  });

  it('rejects a subdirectory escape even under groups/global', () => {
    expect(validateGlobalFilesToPersist(['prompts/main.md'])).toEqual({
      ok: false,
      invalid: 'prompts/main.md',
    });
  });

  it('rejects a non-string entry, surfacing it stringified', () => {
    expect(validateGlobalFilesToPersist([42])).toEqual({
      ok: false,
      invalid: '42',
    });
    expect(validateGlobalFilesToPersist([null])).toEqual({
      ok: false,
      invalid: 'null',
    });
  });

  it('rejects a non-array payload by falling back to the allowlist scan', () => {
    // A scalar payload is not Array.isArray → defaults to the full allowlist,
    // which is valid. A string that looks like a path must NOT slip through as
    // a single char-array; Array.isArray('SOUL.md') is false, so it defaults.
    expect(validateGlobalFilesToPersist('SOUL.md')).toEqual({
      ok: true,
      relPaths: PERSISTABLE_GLOBAL_FILES.map((f) => `groups/global/${f}`),
    });
  });
});

describe('redactGitToken', () => {
  it('replaces every occurrence of a known token with ***', () => {
    const token = 'ghp_secretTOKEN123';
    const text = `fatal: unable to access using ${token}; retried with ${token}`;
    const out = redactGitToken(text, token);
    expect(out).not.toContain(token);
    expect(out).toBe('fatal: unable to access using ***; retried with ***');
  });

  it('redacts an x-access-token URL credential even without a known token', () => {
    const text =
      "remote: Invalid username or password for 'https://x-access-token:ghs_abc123XYZ@github.com/jbaruch/nanoclaw.git/'";
    const out = redactGitToken(text);
    expect(out).not.toContain('ghs_abc123XYZ');
    expect(out).toContain('x-access-token:***@github.com');
  });

  it('leaves token-free text unchanged', () => {
    const text = 'nothing secret here';
    expect(redactGitToken(text, 'ghp_x')).toBe(text);
  });
});

describe('persistGlobalFilesToGit', () => {
  // Real git in a throwaway repo + bare remote — same real-fs/git pattern as
  // backup-sync.test.ts. Deterministic: no network, fixed content.
  function git(cwd: string, args: string[]): string {
    return execFileSync('git', args, {
      cwd,
      encoding: 'utf-8',
      env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
    });
  }

  function setupRepo(): { root: string; remote: string; cleanup: () => void } {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'persist-git-work-'));
    const remote = fs.mkdtempSync(path.join(os.tmpdir(), 'persist-git-bare-'));
    git(remote, ['init', '--bare', '--initial-branch=main']);
    git(root, ['init', '--initial-branch=main']);
    git(root, ['config', 'user.email', 'test@example.com']);
    git(root, ['config', 'user.name', 'Test']);
    git(root, ['remote', 'add', 'origin', remote]);
    fs.mkdirSync(path.join(root, 'groups', 'global'), { recursive: true });
    fs.writeFileSync(
      path.join(root, 'groups', 'global', 'SOUL.md'),
      'baseline\n',
    );
    git(root, ['add', '-A']);
    git(root, ['commit', '-m', 'baseline']);
    git(root, ['push', 'origin', 'HEAD:main']);
    // Establish the origin/main remote-tracking ref the recovery path reads.
    git(root, ['fetch', 'origin']);
    return {
      root,
      remote,
      cleanup: () => {
        fs.rmSync(root, { recursive: true, force: true });
        fs.rmSync(remote, { recursive: true, force: true });
      },
    };
  }

  it('commits the edited file and pushes it to the remote main', async () => {
    const { root, remote, cleanup } = setupRepo();
    try {
      // The skill's Step-3-equivalent working-tree edit.
      fs.writeFileSync(
        path.join(root, 'groups', 'global', 'SOUL.md'),
        'baseline\nan approved change\n',
      );
      const result = await persistGlobalFilesToGit({
        repoRoot: root,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: persist approved updates 2026-06-21',
      });
      expect(result).toMatchObject({ committed: true });
      // The bare remote received the commit with our message.
      const remoteLog = git(remote, ['log', '--oneline', '-1', 'main']);
      expect(remoteLog).toContain('soul: persist approved updates 2026-06-21');
      // The pushed content carries the approved change.
      const pushedBlob = git(remote, ['show', 'main:groups/global/SOUL.md']);
      expect(pushedBlob).toContain('an approved change');
    } finally {
      cleanup();
    }
  });

  it('reports committed:false when the working tree already matches', async () => {
    const { root, cleanup } = setupRepo();
    try {
      // No working-tree edit — the allowlisted path is unchanged.
      const result = await persistGlobalFilesToGit({
        repoRoot: root,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: noop',
      });
      expect(result.committed).toBe(false);
      expect(result.error).toBeUndefined();
    } finally {
      cleanup();
    }
  });

  it('stages only the named paths, never unrelated working-tree changes', async () => {
    const { root, remote, cleanup } = setupRepo();
    try {
      fs.writeFileSync(
        path.join(root, 'groups', 'global', 'SOUL.md'),
        'baseline\npersona edit\n',
      );
      // An unrelated dirty file that must NOT ride along in the commit.
      fs.writeFileSync(path.join(root, 'unrelated.txt'), 'do not commit me\n');
      const result = await persistGlobalFilesToGit({
        repoRoot: root,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: only persona',
      });
      expect(result).toMatchObject({ committed: true });
      const files = git(remote, [
        'show',
        '--name-only',
        '--format=',
        'main',
      ]).trim();
      expect(files).toBe('groups/global/SOUL.md');
    } finally {
      cleanup();
    }
  });

  it('rolls back the local commit on push failure so the change stays retryable', async () => {
    const { root, remote, cleanup } = setupRepo();
    try {
      const headBefore = git(root, ['rev-parse', 'HEAD']).trim();
      // Break the remote so the push fails after the commit lands locally.
      fs.rmSync(remote, { recursive: true, force: true });
      fs.writeFileSync(
        path.join(root, 'groups', 'global', 'SOUL.md'),
        'baseline\nwill fail to push\n',
      );
      const token = 'ghs_supersecretTOKEN';
      const result = await persistGlobalFilesToGit({
        repoRoot: root,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: push fails',
        token,
      });
      expect(result.stage).toBe('git');
      expect(result.error).toBeTruthy();
      // The token must never survive into the envelope (no-secrets).
      expect(JSON.stringify(result)).not.toContain(token);
      // The commit was rolled back — HEAD is back to baseline, no stranded
      // committed-but-unpushed change.
      expect(git(root, ['rev-parse', 'HEAD']).trim()).toBe(headBefore);
      // The approved edit is still in the working tree, so a retry can land it.
      expect(
        fs.readFileSync(
          path.join(root, 'groups', 'global', 'SOUL.md'),
          'utf-8',
        ),
      ).toContain('will fail to push');
    } finally {
      cleanup();
    }
  });

  it('a retry after a push failure lands the change once the remote is back', async () => {
    const { root, remote, cleanup } = setupRepo();
    try {
      fs.writeFileSync(
        path.join(root, 'groups', 'global', 'SOUL.md'),
        'baseline\nflaky push\n',
      );
      // First attempt: remote unreachable → rolled-back failure.
      const remoteBackup = fs.mkdtempSync(
        path.join(os.tmpdir(), 'persist-bak-'),
      );
      fs.cpSync(remote, remoteBackup, { recursive: true });
      fs.rmSync(remote, { recursive: true, force: true });
      const first = await persistGlobalFilesToGit({
        repoRoot: root,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: attempt 1',
      });
      expect(first.stage).toBe('git');
      // Remote recovers; retry must succeed and push the change.
      fs.cpSync(remoteBackup, remote, { recursive: true });
      const second = await persistGlobalFilesToGit({
        repoRoot: root,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: attempt 2',
      });
      expect(second).toMatchObject({ committed: true });
      expect(git(remote, ['show', 'main:groups/global/SOUL.md'])).toContain(
        'flaky push',
      );
      fs.rmSync(remoteBackup, { recursive: true, force: true });
    } finally {
      cleanup();
    }
  });

  it('recovers a previously-stranded unpushed commit on a no-edit run', async () => {
    const { root, remote, cleanup } = setupRepo();
    try {
      // Simulate the pre-fix stranded state: a commit that landed locally but
      // never reached origin/main, with a clean working tree.
      fs.writeFileSync(
        path.join(root, 'groups', 'global', 'SOUL.md'),
        'baseline\nstranded commit\n',
      );
      git(root, ['add', '--', 'groups/global/SOUL.md']);
      git(root, ['commit', '-m', 'soul: stranded']);
      // No working-tree change now; the commit is ahead of origin/main.
      const result = await persistGlobalFilesToGit({
        repoRoot: root,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: recover',
      });
      expect(result).toMatchObject({ committed: true });
      expect(git(remote, ['show', 'main:groups/global/SOUL.md'])).toContain(
        'stranded commit',
      );
    } finally {
      cleanup();
    }
  });

  it('refuses to push a pending commit that touches a non-allowlisted path', async () => {
    const { root, remote, cleanup } = setupRepo();
    try {
      // A stranded local commit that touches an out-of-allowlist file — the
      // push-gate must refuse it rather than land a non-persona change on main.
      fs.writeFileSync(
        path.join(root, 'unrelated.txt'),
        'not persona content\n',
      );
      git(root, ['add', '--', 'unrelated.txt']);
      git(root, ['commit', '-m', 'stray: unrelated change']);
      const result = await persistGlobalFilesToGit({
        repoRoot: root,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: should not push the stray',
      });
      expect(result.stage).toBe('git');
      expect(result.error).toContain('outside the persona allowlist');
      // main on the remote never received the stray file.
      const remoteFiles = git(remote, ['ls-tree', '--name-only', 'main']);
      expect(remoteFiles).not.toContain('unrelated.txt');
    } finally {
      cleanup();
    }
  });
});

describe('ensurePersonaRepo', () => {
  // Real git in throwaway dirs — same deterministic real-fs/git pattern as the
  // persist suite. A bare remote seeded with a `main` that carries the persona
  // file stands in for the deploy-source repo the orchestrator clones (#471).
  function git(cwd: string, args: string[]): string {
    return execFileSync('git', args, {
      cwd,
      encoding: 'utf-8',
      env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
    });
  }

  function setupRemote(): {
    remote: string;
    seed: string;
    workRoot: string;
    cleanup: () => void;
  } {
    const remote = fs.mkdtempSync(path.join(os.tmpdir(), 'persona-bare-'));
    const seed = fs.mkdtempSync(path.join(os.tmpdir(), 'persona-seed-'));
    // A separate scratch dir the clone target lives *inside* — the function
    // creates/removes the target dir itself, so the parent must already exist.
    const workRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'persona-work-'));
    git(remote, ['init', '--bare', '--initial-branch=main']);
    git(seed, ['init', '--initial-branch=main']);
    git(seed, ['config', 'user.email', 'seed@example.com']);
    git(seed, ['config', 'user.name', 'Seed']);
    git(seed, ['remote', 'add', 'origin', remote]);
    fs.mkdirSync(path.join(seed, 'groups', 'global'), { recursive: true });
    fs.writeFileSync(path.join(seed, 'groups', 'global', 'SOUL.md'), 'seed\n');
    git(seed, ['add', '-A']);
    git(seed, ['commit', '-m', 'seed']);
    git(seed, ['push', 'origin', 'HEAD:main']);
    return {
      remote,
      seed,
      workRoot,
      cleanup: () => {
        fs.rmSync(remote, { recursive: true, force: true });
        fs.rmSync(seed, { recursive: true, force: true });
        fs.rmSync(workRoot, { recursive: true, force: true });
      },
    };
  }

  // Advance the remote's `main` by one unrelated commit via the seed clone,
  // returning the new tip SHA — stands in for another PR merging to main
  // between two persist runs.
  function advanceRemote(seed: string): string {
    fs.writeFileSync(path.join(seed, 'README.md'), 'upstream advanced\n');
    git(seed, ['add', '-A']);
    git(seed, ['commit', '-m', 'upstream: unrelated advance']);
    git(seed, ['push', 'origin', 'HEAD:main']);
    return git(seed, ['rev-parse', 'HEAD']).trim();
  }

  // A second, independent bare remote whose `main` carries distinct SOUL
  // content — stands in for an operator repointing ORCHESTRATOR_REPO_URL at a
  // different deploy source (e.g. a fork) between persist runs.
  function seedBareRemote(soulContent: string): {
    remote: string;
    cleanup: () => void;
  } {
    const remote = fs.mkdtempSync(path.join(os.tmpdir(), 'persona-bare2-'));
    const seed = fs.mkdtempSync(path.join(os.tmpdir(), 'persona-seed2-'));
    git(remote, ['init', '--bare', '--initial-branch=main']);
    git(seed, ['init', '--initial-branch=main']);
    git(seed, ['config', 'user.email', 'seed2@example.com']);
    git(seed, ['config', 'user.name', 'Seed2']);
    git(seed, ['remote', 'add', 'origin', remote]);
    fs.mkdirSync(path.join(seed, 'groups', 'global'), { recursive: true });
    fs.writeFileSync(
      path.join(seed, 'groups', 'global', 'SOUL.md'),
      soulContent,
    );
    git(seed, ['add', '-A']);
    git(seed, ['commit', '-m', 'seed2']);
    git(seed, ['push', 'origin', 'HEAD:main']);
    return {
      remote,
      cleanup: () => {
        fs.rmSync(remote, { recursive: true, force: true });
        fs.rmSync(seed, { recursive: true, force: true });
      },
    };
  }

  it('clones a fresh worktree tracking origin/main on first run', async () => {
    const { remote, workRoot, cleanup } = setupRemote();
    try {
      const repoDir = path.join(workRoot, 'persona-repo');
      const result = await ensurePersonaRepo({ repoDir, remoteUrl: remote });
      expect(result).toEqual({ ok: true });
      // It is a real git worktree — the exact thing `process.cwd()` (`/app`)
      // was NOT, which is the whole #471 defect.
      expect(git(repoDir, ['rev-parse', '--is-inside-work-tree']).trim()).toBe(
        'true',
      );
      // Seeded content and an origin/main tracking ref are present.
      expect(
        fs.readFileSync(
          path.join(repoDir, 'groups', 'global', 'SOUL.md'),
          'utf-8',
        ),
      ).toContain('seed');
      expect(git(repoDir, ['rev-parse', 'origin/main']).trim()).toBeTruthy();
    } finally {
      cleanup();
    }
  });

  it('end-to-end: overlay a persona edit, then persist pushes only it to main', async () => {
    const { remote, workRoot, cleanup } = setupRemote();
    try {
      const repoDir = path.join(workRoot, 'persona-repo');
      const ready = await ensurePersonaRepo({ repoDir, remoteUrl: remote });
      expect(ready).toEqual({ ok: true });
      // The handler overlays the live runtime edit onto the clone, then persists.
      fs.writeFileSync(
        path.join(repoDir, 'groups', 'global', 'SOUL.md'),
        'seed\nan approved change\n',
      );
      const result = await persistGlobalFilesToGit({
        repoRoot: repoDir,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: persist approved updates 2026-07-19',
      });
      expect(result).toMatchObject({ committed: true });
      expect(git(remote, ['show', 'main:groups/global/SOUL.md'])).toContain(
        'an approved change',
      );
    } finally {
      cleanup();
    }
  });

  it('refreshes an existing clone to origin/main, discarding local cruft', async () => {
    const { remote, workRoot, cleanup } = setupRemote();
    try {
      const repoDir = path.join(workRoot, 'persona-repo');
      await ensurePersonaRepo({ repoDir, remoteUrl: remote });
      // Simulate a stranded prior run: a stray local commit + dirty tree +
      // an untracked file. A naive in-place persist would carry these along.
      fs.writeFileSync(
        path.join(repoDir, 'groups', 'global', 'SOUL.md'),
        'seed\nhalf-applied\n',
      );
      git(repoDir, ['add', '-A']);
      git(repoDir, ['commit', '-m', 'stray local commit']);
      fs.writeFileSync(path.join(repoDir, 'stray-untracked.txt'), 'cruft\n');
      const strayHead = git(repoDir, ['rev-parse', 'HEAD']).trim();

      const result = await ensurePersonaRepo({ repoDir, remoteUrl: remote });
      expect(result).toEqual({ ok: true });
      // HEAD is back at origin/main, the half-applied edit is gone, and the
      // untracked cruft is cleaned — a fresh persist starts from a clean base.
      expect(git(repoDir, ['rev-parse', 'HEAD']).trim()).toBe(
        git(repoDir, ['rev-parse', 'origin/main']).trim(),
      );
      expect(git(repoDir, ['rev-parse', 'HEAD']).trim()).not.toBe(strayHead);
      expect(
        fs.readFileSync(
          path.join(repoDir, 'groups', 'global', 'SOUL.md'),
          'utf-8',
        ),
      ).not.toContain('half-applied');
      expect(fs.existsSync(path.join(repoDir, 'stray-untracked.txt'))).toBe(
        false,
      );
    } finally {
      cleanup();
    }
  });

  it('advances the tracking ref when the remote moves, so a later persist fast-forwards', async () => {
    // Regression guard for the FETCH_HEAD-only refresh bug (PR #816 review):
    // `git fetch origin main` writes only FETCH_HEAD, leaving
    // refs/remotes/origin/main pinned to the first shallow checkout — so a
    // persist after main advanced would push a non-fast-forward and fail.
    const { remote, seed, workRoot, cleanup } = setupRemote();
    try {
      const repoDir = path.join(workRoot, 'persona-repo');
      await ensurePersonaRepo({ repoDir, remoteUrl: remote });

      // Another PR merges to main between persist runs.
      const advancedTip = advanceRemote(seed);

      const refreshed = await ensurePersonaRepo({ repoDir, remoteUrl: remote });
      expect(refreshed).toEqual({ ok: true });
      // The clone's tracking ref and HEAD both caught up to the new tip — not
      // pinned to the stale original checkout.
      expect(git(repoDir, ['rev-parse', 'origin/main']).trim()).toBe(
        advancedTip,
      );
      expect(git(repoDir, ['rev-parse', 'HEAD']).trim()).toBe(advancedTip);

      // A persist on the refreshed clone now fast-forwards cleanly onto the
      // advanced main rather than being rejected as non-fast-forward.
      fs.writeFileSync(
        path.join(repoDir, 'groups', 'global', 'SOUL.md'),
        'seed\npost-advance edit\n',
      );
      const result = await persistGlobalFilesToGit({
        repoRoot: repoDir,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: after main advanced',
      });
      expect(result).toMatchObject({ committed: true });
      expect(git(remote, ['show', 'main:groups/global/SOUL.md'])).toContain(
        'post-advance edit',
      );
      // The unrelated upstream advance is still present — the persist built on
      // top of it, it didn't clobber it.
      expect(git(remote, ['show', 'main:README.md'])).toContain(
        'upstream advanced',
      );
    } finally {
      cleanup();
    }
  });

  it('reconciles origin to a changed remoteUrl, so a later persist targets the new repo', async () => {
    // Regression guard for the ORCHESTRATOR_REPO_URL drift (PR #816 review):
    // an existing clone made from repo A must switch to repo B when
    // ensurePersonaRepo is called with B's URL — otherwise fetch/push keep
    // targeting A and persona edits land in the wrong repo.
    const { remote: remoteA, workRoot, cleanup } = setupRemote();
    const { remote: remoteB, cleanup: cleanupB } =
      seedBareRemote('from repo B\n');
    try {
      const repoDir = path.join(workRoot, 'persona-repo');
      await ensurePersonaRepo({ repoDir, remoteUrl: remoteA });

      // Operator repoints ORCHESTRATOR_REPO_URL at repo B.
      const result = await ensurePersonaRepo({ repoDir, remoteUrl: remoteB });
      expect(result).toEqual({ ok: true });
      // origin now points at B and the worktree carries B's content, not A's.
      expect(git(repoDir, ['remote', 'get-url', 'origin']).trim()).toBe(
        remoteB,
      );
      expect(
        fs.readFileSync(
          path.join(repoDir, 'groups', 'global', 'SOUL.md'),
          'utf-8',
        ),
      ).toContain('from repo B');

      // A persist now lands in B, and A is left untouched.
      fs.writeFileSync(
        path.join(repoDir, 'groups', 'global', 'SOUL.md'),
        'from repo B\nedit after repoint\n',
      );
      const persisted = await persistGlobalFilesToGit({
        repoRoot: repoDir,
        relPaths: ['groups/global/SOUL.md'],
        message: 'soul: after repoint',
      });
      expect(persisted).toMatchObject({ committed: true });
      expect(git(remoteB, ['show', 'main:groups/global/SOUL.md'])).toContain(
        'edit after repoint',
      );
      expect(
        git(remoteA, ['show', 'main:groups/global/SOUL.md']),
      ).not.toContain('edit after repoint');
    } finally {
      cleanupB();
      cleanup();
    }
  });

  it('returns a token-redacted git-stage envelope when the clone fails', async () => {
    const { workRoot, cleanup } = setupRemote();
    try {
      const repoDir = path.join(workRoot, 'persona-repo');
      const token = 'ghs_supersecretTOKEN';
      const result = await ensurePersonaRepo({
        repoDir,
        remoteUrl: `${workRoot}/does-not-exist.git`,
        token,
      });
      expect(result.ok).toBe(false);
      if (result.ok) throw new Error('expected failure');
      expect(result.stage).toBe('git');
      expect(result.error).toContain('git clone');
      // The token must never survive into the envelope (no-secrets).
      expect(JSON.stringify(result)).not.toContain(token);
      // A failed clone leaves no partial worktree behind.
      expect(fs.existsSync(path.join(repoDir, '.git'))).toBe(false);
    } finally {
      cleanup();
    }
  });
});

describe('runPersonaPersistTask', () => {
  // Real git in throwaway dirs. Proves the task ALWAYS writes a result envelope
  // to `resultPath` — a missing envelope reads as a silent hang to the polling
  // caller (#816 review).
  function git(cwd: string, args: string[]): string {
    return execFileSync('git', args, {
      cwd,
      encoding: 'utf-8',
      env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
    });
  }

  function setup(): {
    remote: string;
    groupsDir: string;
    personaRepoDir: string;
    resultPath: string;
    cleanup: () => void;
  } {
    const remote = fs.mkdtempSync(path.join(os.tmpdir(), 'task-bare-'));
    const seed = fs.mkdtempSync(path.join(os.tmpdir(), 'task-seed-'));
    const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'task-scratch-'));
    git(remote, ['init', '--bare', '--initial-branch=main']);
    git(seed, ['init', '--initial-branch=main']);
    git(seed, ['config', 'user.email', 'seed@example.com']);
    git(seed, ['config', 'user.name', 'Seed']);
    git(seed, ['remote', 'add', 'origin', remote]);
    fs.mkdirSync(path.join(seed, 'groups', 'global'), { recursive: true });
    fs.writeFileSync(path.join(seed, 'groups', 'global', 'SOUL.md'), 'seed\n');
    git(seed, ['add', '-A']);
    git(seed, ['commit', '-m', 'seed']);
    git(seed, ['push', 'origin', 'HEAD:main']);
    // The live runtime mirror the task copies FROM (the agent's edited files).
    const groupsDir = path.join(scratch, 'groups');
    fs.mkdirSync(path.join(groupsDir, 'global'), { recursive: true });
    return {
      remote,
      groupsDir,
      personaRepoDir: path.join(scratch, 'persona-repo'),
      resultPath: path.join(scratch, 'result.json'),
      cleanup: () => {
        fs.rmSync(remote, { recursive: true, force: true });
        fs.rmSync(seed, { recursive: true, force: true });
        fs.rmSync(scratch, { recursive: true, force: true });
      },
    };
  }

  const readEnvelope = (resultPath: string) =>
    JSON.parse(fs.readFileSync(resultPath, 'utf-8'));

  it('writes a committed envelope and pushes the edit on the happy path', async () => {
    const { remote, groupsDir, personaRepoDir, resultPath, cleanup } = setup();
    try {
      fs.writeFileSync(
        path.join(groupsDir, 'global', 'SOUL.md'),
        'seed\nan approved change\n',
      );
      await runPersonaPersistTask({
        personaRepoDir,
        groupsDir,
        relPaths: ['groups/global/SOUL.md'],
        remoteUrl: remote,
        message: 'soul: happy path',
        resultPath,
        sourceGroup: 'tg:1',
      });
      expect(readEnvelope(resultPath)).toMatchObject({ committed: true });
      expect(git(remote, ['show', 'main:groups/global/SOUL.md'])).toContain(
        'an approved change',
      );
    } finally {
      cleanup();
    }
  });

  it('writes an error envelope (never nothing) when the clone step fails', async () => {
    const { groupsDir, personaRepoDir, resultPath, cleanup } = setup();
    try {
      fs.writeFileSync(
        path.join(groupsDir, 'global', 'SOUL.md'),
        'seed\nedit\n',
      );
      await runPersonaPersistTask({
        personaRepoDir,
        groupsDir,
        relPaths: ['groups/global/SOUL.md'],
        remoteUrl: `${personaRepoDir}-does-not-exist.git`,
        message: 'soul: clone fails',
        resultPath,
        sourceGroup: 'tg:1',
      });
      const envelope = readEnvelope(resultPath);
      expect(envelope.stage).toBe('git');
      expect(envelope.error).toContain('git clone');
    } finally {
      cleanup();
    }
  });

  it('writes an error envelope when a source persona file is missing', async () => {
    const { remote, groupsDir, personaRepoDir, resultPath, cleanup } = setup();
    try {
      // groupsDir/global has no SOUL.md — the overlay copy fails.
      await runPersonaPersistTask({
        personaRepoDir,
        groupsDir,
        relPaths: ['groups/global/SOUL.md'],
        remoteUrl: remote,
        message: 'soul: missing source',
        resultPath,
        sourceGroup: 'tg:1',
      });
      const envelope = readEnvelope(resultPath);
      expect(envelope.stage).toBe('git');
      expect(envelope.error).toContain('copying persona files');
    } finally {
      cleanup();
    }
  });
});

describe('runSerializedPersonaPersist', () => {
  // Deferred promise the test resolves by hand to control task timing.
  function deferred(): { promise: Promise<void>; resolve: () => void } {
    let resolve!: () => void;
    const promise = new Promise<void>((r) => {
      resolve = r;
    });
    return { promise, resolve };
  }

  it('runs queued persists strictly one at a time, in order', async () => {
    const order: string[] = [];
    const gateA = deferred();
    const aDone = deferred();
    const bDone = deferred();

    // Task A blocks until the test releases gateA; B must not start meanwhile.
    runSerializedPersonaPersist(async () => {
      order.push('A-start');
      await gateA.promise;
      order.push('A-end');
      aDone.resolve();
    });
    runSerializedPersonaPersist(async () => {
      order.push('B');
      bDone.resolve();
    });

    // Let microtasks flush — B is still queued behind the blocked A.
    await Promise.resolve();
    expect(order).toEqual(['A-start']);

    gateA.resolve();
    await aDone.promise;
    await bDone.promise;
    expect(order).toEqual(['A-start', 'A-end', 'B']);
  });

  it('keeps the queue alive when a task rejects — the next persist still runs', async () => {
    const order: string[] = [];
    const done = deferred();

    // A rejecting task must not poison the chain (a failed persist writes its
    // own envelope; the queue keeps serving later persists).
    runSerializedPersonaPersist(async () => {
      order.push('boom');
      throw new Error('task blew up');
    });
    runSerializedPersonaPersist(async () => {
      order.push('after');
      done.resolve();
    });

    await done.promise;
    expect(order).toEqual(['boom', 'after']);
  });
});

describe('backupCommitAndPush', () => {
  // Same real-git-in-tmpdir pattern as persistGlobalFilesToGit above, but
  // with an upstream-tracking branch since the backup repo pushes its
  // default upstream (`git push` with no refspec).
  function git(cwd: string, args: string[]): string {
    return execFileSync('git', args, {
      cwd,
      encoding: 'utf-8',
      env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
    });
  }

  function setupRepo(): {
    backupDir: string;
    remote: string;
    cleanup: () => void;
  } {
    const backupDir = fs.mkdtempSync(path.join(os.tmpdir(), 'backup-work-'));
    const remote = fs.mkdtempSync(path.join(os.tmpdir(), 'backup-bare-'));
    git(remote, ['init', '--bare', '--initial-branch=main']);
    git(backupDir, ['init', '--initial-branch=main']);
    git(backupDir, ['config', 'user.email', 'test@example.com']);
    git(backupDir, ['config', 'user.name', 'Test']);
    git(backupDir, ['remote', 'add', 'origin', remote]);
    fs.writeFileSync(path.join(backupDir, 'seed.txt'), 'baseline\n');
    git(backupDir, ['add', '-A']);
    git(backupDir, ['commit', '-m', 'baseline']);
    // -u establishes the upstream ref the bare `git push` resolves.
    git(backupDir, ['push', '-u', 'origin', 'main']);
    return {
      backupDir,
      remote,
      cleanup: () => {
        fs.rmSync(backupDir, { recursive: true, force: true });
        fs.rmSync(remote, { recursive: true, force: true });
      },
    };
  }

  it('commits and pushes new backup content', async () => {
    const { backupDir, remote, cleanup } = setupRepo();
    try {
      fs.writeFileSync(path.join(backupDir, 'state.sql'), 'INSERT ...;\n');
      const result = await backupCommitAndPush({
        backupDir,
        message: 'backup: 2026-07-07',
        token: undefined,
      });
      expect(result).toMatchObject({ committed: true });
      expect(git(remote, ['log', '--oneline', '-1', 'main'])).toContain(
        'backup: 2026-07-07',
      );
      expect(git(remote, ['show', 'main:state.sql'])).toContain('INSERT');
    } finally {
      cleanup();
    }
  });

  it('treats an agent-supplied commit message as data, never as shell (#725)', async () => {
    const { backupDir, remote, cleanup } = setupRepo();
    try {
      // The pre-fix `bash -c` pipeline executed `$()` command substitution
      // inside its double-quoted string. If any shell still interprets the
      // message, the marker file appears and the pushed message loses the
      // literal `$()` text.
      const marker = path.join(backupDir, 'pwned');
      const hostileMsg = `backup: $(touch ${marker}) \`touch ${marker}\` "; touch ${marker}; "`;
      fs.writeFileSync(path.join(backupDir, 'data.txt'), 'content\n');
      const result = await backupCommitAndPush({
        backupDir,
        message: hostileMsg,
        token: undefined,
      });
      expect(result).toMatchObject({ committed: true });
      // No command substitution ran on the host.
      expect(fs.existsSync(marker)).toBe(false);
      // The hostile text survived verbatim as an inert commit message.
      const remoteMsg = git(remote, ['log', '-1', '--format=%B', 'main']);
      expect(remoteMsg).toContain('$(touch');
    } finally {
      cleanup();
    }
  });

  it('reports committed:false when there is nothing to back up', async () => {
    const { backupDir, cleanup } = setupRepo();
    try {
      const result = await backupCommitAndPush({
        backupDir,
        message: 'backup: noop',
        token: undefined,
      });
      expect(result.committed).toBe(false);
      expect(result.error).toBeUndefined();
    } finally {
      cleanup();
    }
  });

  it('rolls back its commit on push failure and redacts the token', async () => {
    const { backupDir, remote, cleanup } = setupRepo();
    try {
      const headBefore = git(backupDir, ['rev-parse', 'HEAD']).trim();
      fs.rmSync(remote, { recursive: true, force: true });
      fs.writeFileSync(path.join(backupDir, 'data.txt'), 'will fail\n');
      const token = 'ghs_backupSECRETtoken';
      const result = await backupCommitAndPush({
        backupDir,
        message: 'backup: push fails',
        token,
      });
      expect(result.stage).toBe('git');
      expect(result.error).toBeTruthy();
      expect(JSON.stringify(result)).not.toContain(token);
      // Rolled back — the next run re-stages and retries instead of
      // reporting a false "Nothing to commit."
      expect(git(backupDir, ['rev-parse', 'HEAD']).trim()).toBe(headBefore);
    } finally {
      cleanup();
    }
  });

  it('surfaces a broken upstream as an error instead of a false "Nothing to commit"', async () => {
    // No upstream tracking ref → `rev-list @{u}..HEAD` fails. That must
    // come back as a git-stage error (pending commits could exist and the
    // push needs the same upstream), not as a silent no-op.
    const backupDir = fs.mkdtempSync(path.join(os.tmpdir(), 'backup-noup-'));
    try {
      git(backupDir, ['init', '--initial-branch=main']);
      git(backupDir, ['config', 'user.email', 'test@example.com']);
      git(backupDir, ['config', 'user.name', 'Test']);
      fs.writeFileSync(path.join(backupDir, 'seed.txt'), 'baseline\n');
      git(backupDir, ['add', '-A']);
      git(backupDir, ['commit', '-m', 'baseline']);
      const result = await backupCommitAndPush({
        backupDir,
        message: 'backup: no upstream',
        token: undefined,
      });
      expect(result.stage).toBe('git');
      expect(result.error).toContain('git rev-list');
    } finally {
      fs.rmSync(backupDir, { recursive: true, force: true });
    }
  });

  it('pushes a stranded unpushed commit on a no-change run', async () => {
    const { backupDir, remote, cleanup } = setupRepo();
    try {
      // The pre-fix pipeline could commit and then fail the push, leaving
      // HEAD ahead of upstream with a clean tree.
      fs.writeFileSync(path.join(backupDir, 'data.txt'), 'stranded\n');
      git(backupDir, ['add', '-A']);
      git(backupDir, ['commit', '-m', 'backup: stranded']);
      const result = await backupCommitAndPush({
        backupDir,
        message: 'backup: recover',
        token: undefined,
      });
      expect(result).toMatchObject({ committed: true });
      expect(git(remote, ['show', 'main:data.txt'])).toContain('stranded');
    } finally {
      cleanup();
    }
  });
});

// -----------------------------------------------------------------
// coerceTzSegments — pure normalization of the raw `segments` payload of a
// `persist_tz_segments` IPC call (#748). The value arrives as raw JSON off
// the IPC wire, so the contract under test is: an array passes through
// untouched; anything else returns an empty array with `wasArray: false`
// rather than throwing. Coercion does NOT decide persistence — the caller
// (`classifyTzPersist`) uses `wasArray` to REJECT a non-array, so a caller
// bug never clears owner tz_state; only an explicit `[]` clears.
// -----------------------------------------------------------------
