import { spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

/**
 * Coverage for `tessl_exec_env_flags` in `scripts/deploy.sh` (#887) — the
 * shell bridge that turns `printTesslChildEnv()` output into `docker exec
 * -e` flags. The TypeScript side is covered in `src/tessl-env.test.ts`;
 * this pins the bridge, which is where a silent fallback would hide.
 *
 * `docker` is stubbed by prepending a temp dir to PATH, so nothing here
 * touches a real daemon and the suite runs on any CI runner.
 */

let tmp: string;

/** Write a fake `docker` that emits `stdout`/`stderr` and exits `code`. */
function stubDocker(opts: {
  stdout?: string;
  stderr?: string;
  code?: number;
}): void {
  const bin = path.join(tmp, 'bin');
  fs.mkdirSync(bin, { recursive: true });
  const script = [
    '#!/usr/bin/env bash',
    // `%b` so the \n escapes inside the double-quoted bash string become
    // real newlines — `%s` would emit a literal backslash-n and the helper
    // would see one long line.
    opts.stdout ? `printf '%b' ${JSON.stringify(opts.stdout)}` : ':',
    opts.stderr ? `printf '%b' ${JSON.stringify(opts.stderr)} >&2` : ':',
    `exit ${opts.code ?? 0}`,
    '',
  ].join('\n');
  const p = path.join(bin, 'docker');
  fs.writeFileSync(p, script);
  fs.chmodSync(p, 0o755);
}

/**
 * Source deploy.sh's helper in isolation and print the resulting flags,
 * one per line, so assertions read against the real shell function rather
 * than a reimplementation of it.
 */
function runHelper(): { stdout: string; stderr: string; code: number } {
  const deploySh = path.join(process.cwd(), 'scripts', 'deploy.sh');
  const src = fs.readFileSync(deploySh, 'utf8');
  const start = src.indexOf('tessl_exec_env_flags() {');
  expect(start).toBeGreaterThan(-1);
  const end = src.indexOf('\n}\n', start);
  expect(end).toBeGreaterThan(start);
  const fn = src.slice(start, end + 3);

  const harness = path.join(tmp, 'harness.sh');
  fs.writeFileSync(
    harness,
    [
      '#!/usr/bin/env bash',
      'set -uo pipefail',
      fn,
      'tessl_exec_env_flags',
      'printf "%s\\n" "${TESSL_EXEC_FLAGS[@]:-}"',
      '',
    ].join('\n'),
  );
  fs.chmodSync(harness, 0o755);
  // spawnSync, not execFileSync: the helper exits 0 even when the bridge
  // fails (fail-open by design), so the interesting stderr arrives on the
  // SUCCESS path — execFileSync only surfaces stderr by throwing.
  const r = spawnSync('bash', [harness], {
    encoding: 'utf-8',
    env: {
      ...process.env,
      PATH: `${path.join(tmp, 'bin')}:${process.env.PATH ?? ''}`,
    },
  });
  return {
    stdout: r.stdout ?? '',
    stderr: r.stderr ?? '',
    code: r.status ?? 1,
  };
}

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tessl-flags-test-'));
});

afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

describe('tessl_exec_env_flags (scripts/deploy.sh, #887)', () => {
  it('emits no flags when OneCLI is unconfigured (empty output is normal)', () => {
    stubDocker({ stdout: '', code: 0 });
    const r = runHelper();
    expect(r.stdout.trim()).toBe('');
    expect(r.stderr).not.toMatch(/WARNING/);
  });

  it('wraps each KEY=VALUE line in its own -e flag', () => {
    stubDocker({
      stdout:
        'HTTPS_PROXY=http://gw:1\nNODE_EXTRA_CA_CERTS=/x/ca.pem\nTESSL_TOKEN=onecli-managed\n',
      code: 0,
    });
    const lines = runHelper().stdout.trim().split('\n');
    expect(lines).toEqual([
      '-e',
      'HTTPS_PROXY=http://gw:1',
      '-e',
      'NODE_EXTRA_CA_CERTS=/x/ca.pem',
      '-e',
      'TESSL_TOKEN=onecli-managed',
    ]);
  });

  it('warns on stderr when the bridge fails, instead of falling back silently', () => {
    // The whole point of #887 is that a broken credential path must not
    // masquerade as "nothing to do" — that is the bug being fixed.
    stubDocker({ stderr: 'Error: Cannot find module', code: 1 });
    const r = runHelper();
    expect(r.stderr).toMatch(/WARNING/);
    expect(r.stderr).toMatch(/direct path/);
    expect(r.stdout.trim()).toBe('');
  });

  it('surfaces the failure diagnostic but never the secret-bearing stdout', () => {
    // stdout carries a proxy URL embedding a gateway credential; only
    // stderr is safe to echo.
    stubDocker({
      stdout: 'HTTPS_PROXY=http://x:aoc_SUPERSECRET@gw:10255\n',
      stderr: 'boom: something broke',
      code: 3,
    });
    const r = runHelper();
    expect(r.stderr).toMatch(/exit 3/);
    expect(r.stderr).toMatch(/boom: something broke/);
    expect(r.stderr).not.toMatch(/aoc_SUPERSECRET/);
    expect(r.stdout).not.toMatch(/aoc_SUPERSECRET/);
  });

  it('does not abort the deploy when the bridge fails', () => {
    // Fail-open by design: a gateway problem degrades tessl to the direct
    // path; step 3c is what catches a genuinely stale registry read.
    stubDocker({ stderr: 'nope', code: 1 });
    expect(runHelper().code).toBe(0);
  });
});
