import { spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

import { describe, it, expect } from 'vitest';

// Contract tests for `git_with_token` in scripts/tile-repo-lib.sh (#728):
// auth must never ride in argv or persist into the clone's .git/config,
// git's exit code must pass through, and stderr must be re-emitted with
// token-bearing URLs redacted. Exercised by shelling into bash and
// sourcing the lib — vitest's `scripts/**/*.test.ts` include runs this
// in the existing CI job with no workflow changes.

const LIB_PATH = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  'tile-repo-lib.sh',
);

const FAKE_TOKEN = 'ghs_fakeTESTtoken1234';

interface BashResult {
  status: number;
  stdout: string;
  stderr: string;
}

// Run a snippet under `set -euo pipefail` with the lib sourced — the
// same shell contract the promote/push scripts give the helper.
// spawnSync (not execFileSync) so stderr is observable on success too.
function runInLib(
  snippet: string,
  extraEnv: Record<string, string> = {},
): BashResult {
  const result = spawnSync(
    'bash',
    ['-c', `set -euo pipefail\nsource "$LIB_PATH"\n${snippet}`],
    {
      encoding: 'utf-8',
      env: {
        ...process.env,
        LIB_PATH,
        GIT_TERMINAL_PROMPT: '0',
        ...extraEnv,
      },
    },
  );
  if (result.error) throw result.error;
  return {
    status: result.status ?? 1,
    stdout: result.stdout,
    stderr: result.stderr,
  };
}

describe('git_with_token', () => {
  it('runs git successfully and keeps the token out of the persisted remote config', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gwt-clone-'));
    try {
      const result = runInLib(`
        git init -q "${dir}/src"
        git -C "${dir}/src" -c user.email=t@e.st -c user.name=T commit -q --allow-empty -m seed
        git_with_token "${FAKE_TOKEN}" clone -q "${dir}/src" "${dir}/dst"
        cat "${dir}/dst/.git/config"
      `);
      expect(result.status).toBe(0);
      // The wrapper's auth config is invocation-scoped: nothing
      // token-bearing may persist into the clone.
      expect(result.stdout).not.toContain(FAKE_TOKEN);
      expect(result.stdout).not.toContain('x-access-token');
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('preserves git exit codes and never lets the token reach stderr on transport failure', () => {
    // Deterministic offline failure: route the github.com request at a
    // closed local port via https_proxy, so git fails without any
    // live-service dependency.
    const result = runInLib(
      `git_with_token "${FAKE_TOKEN}" ls-remote https://github.com/example/nonexistent-repo.git`,
      { https_proxy: 'http://127.0.0.1:1', HTTPS_PROXY: 'http://127.0.0.1:1' },
    );
    expect(result.status).not.toBe(0);
    expect(result.stderr).not.toContain(FAKE_TOKEN);
    // stderr still flows through to the caller (redacted, not dropped).
    expect(result.stderr).toContain('unable to access');
  });

  it('redacts token-bearing URLs that git writes to stderr', () => {
    // Git's own "unable to access" diagnostics print the pre-rewrite
    // URL, so the real leak shape is server-echoed lines (e.g.
    // "remote: Invalid username or password for 'https://x-access-
    // token:<tok>@github.com/...'") — see the redactGitToken fixture in
    // src/ipc.test.ts. Reproduce that shape offline with a git alias
    // that emits a token-bearing URL on stderr through the wrapper.
    const result = runInLib(
      `git_with_token "${FAKE_TOKEN}" -c 'alias.leak=!echo "remote: rejected https://x-access-token:${FAKE_TOKEN}@github.com/x.git" >&2' leak`,
    );
    expect(result.status).toBe(0);
    expect(result.stderr).not.toContain(FAKE_TOKEN);
    expect(result.stderr).toContain('x-access-token:***@github.com/x.git');
  });

  it('redacts token-bearing stdout too (the injected config is printable)', () => {
    // `git config --list` under the wrapper prints the injected
    // insteadOf pair on stdout — and callers' stdout reaches the same
    // IPC/log consumers as stderr, so the wrapper filters both.
    const result = runInLib(`git_with_token "${FAKE_TOKEN}" config --list`);
    expect(result.status).toBe(0);
    expect(result.stdout).toContain('insteadof');
    expect(result.stdout).not.toContain(FAKE_TOKEN);
    expect(result.stdout).toContain('x-access-token:***@');
  });
});
