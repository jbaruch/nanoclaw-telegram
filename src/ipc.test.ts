import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect } from 'vitest';

import {
  DEFAULT_SESSION_NAME,
  // `MAINTENANCE_SESSION_NAME` lives in group-queue.ts — import from there
  // so the test tracks any future rename without silently breaking.
} from './container-runner.js';
import { shouldStoreBotMessage } from './db.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';
import {
  PERSISTABLE_GLOBAL_FILES,
  applyMaintenancePrefix,
  backupCommitAndPush,
  persistGlobalFilesToGit,
  redactGitToken,
  validateGlobalFilesToPersist,
} from './ipc.js';

describe('applyMaintenancePrefix', () => {
  it('prepends [M] for the maintenance session', () => {
    expect(applyMaintenancePrefix('hello', MAINTENANCE_SESSION_NAME)).toBe(
      '[M] hello',
    );
  });

  it('leaves text untouched for the default session', () => {
    expect(applyMaintenancePrefix('hello', DEFAULT_SESSION_NAME)).toBe('hello');
  });

  it('leaves text untouched when sessionName is undefined', () => {
    // A pre-upgrade container that didn't stamp sessionName on the IPC
    // payload should NOT get the maintenance prefix by accident —
    // silent prefixing of user-facing sends is worse than no prefix.
    expect(applyMaintenancePrefix('hello', undefined)).toBe('hello');
  });

  it('is idempotent — does not double-prefix already-prefixed text', () => {
    // Defensive: if an upstream bug or a future re-entry ever feeds
    // already-prefixed text back in, we shouldn't end up with `[M] [M]`.
    expect(applyMaintenancePrefix('[M] hello', MAINTENANCE_SESSION_NAME)).toBe(
      '[M] hello',
    );
  });

  it('ignores non-maintenance sessionName values regardless of prefix state', () => {
    expect(applyMaintenancePrefix('[M] hello', DEFAULT_SESSION_NAME)).toBe(
      '[M] hello',
    );
    expect(applyMaintenancePrefix('[M] hello', 'some-future-session')).toBe(
      '[M] hello',
    );
  });

  it('preserves multi-line text (prefix is line-0 only)', () => {
    expect(
      applyMaintenancePrefix('line1\nline2', MAINTENANCE_SESSION_NAME),
    ).toBe('[M] line1\nline2');
  });

  it('preserves HTML tags in the body (prefix sits outside them)', () => {
    expect(
      applyMaintenancePrefix('<b>bold</b> text', MAINTENANCE_SESSION_NAME),
    ).toBe('[M] <b>bold</b> text');
  });
});

// -----------------------------------------------------------------
// shouldStoreBotMessage — gates `bot-…` row writes on send success.
// See `src/ipc.ts` for the rationale; this is the predicate extracted
// from the IPC `send_message` handler so the gating logic is unit-
// testable without staging the whole watcher (file events, deps mock,
// db init, atomic writes). The full handler exercises the same
// predicate at runtime — testing it in isolation gives the regression
// signal the OpenAI policy reviewer asked for on PR #232.
// -----------------------------------------------------------------

describe('shouldStoreBotMessage', () => {
  it('returns false for Telegram when sentMsgId is undefined', () => {
    // The textbook phantom-row case: Telegram send was swallowed
    // (400 from a bad reply_to, network blip, malformed HTML even
    // after the plain-text fallback, blocked-by-user, rate-limit).
    // A row written here would silence heartbeat alerts on a chat
    // the user actually received nothing in.
    expect(shouldStoreBotMessage('tg:100200300', undefined)).toBe(false);
  });

  it('returns true for Telegram when sentMsgId is a populated string', () => {
    expect(shouldStoreBotMessage('tg:100200300', '12345')).toBe(true);
  });

  it('returns true for Telegram when sentMsgId is an empty string', () => {
    // Forward-compat with the documented `string | undefined` contract:
    // the upstream `sentMsgId` normalization explicitly avoided
    // truthiness checks so future Telegram ids of `''` or `'0'` aren't
    // dropped on the floor. The gate must match that promise.
    expect(shouldStoreBotMessage('tg:100200300', '')).toBe(true);
  });

  it('returns true for Telegram when sentMsgId is the literal string "0"', () => {
    expect(shouldStoreBotMessage('tg:100200300', '0')).toBe(true);
  });

  it('returns true for non-Telegram channels regardless of sentMsgId', () => {
    // WhatsApp / Slack / Discord `Channel.sendMessage` permit returning
    // void on success (see `src/types.ts`); absence of an id is not a
    // failure signal there. Until those channels surface their own
    // success-id contract, gating would punish a passing send.
    expect(shouldStoreBotMessage('120363012345@g.us', undefined)).toBe(true);
    expect(shouldStoreBotMessage('120363012345@g.us', 'wa-id-abc')).toBe(true);
    expect(shouldStoreBotMessage('slack:C12345', undefined)).toBe(true);
  });
});

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
