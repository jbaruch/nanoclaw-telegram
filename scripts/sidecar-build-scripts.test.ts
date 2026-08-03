import { spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

/**
 * Coverage for `sidecar_build_scripts` in `scripts/deploy.sh` (#895) — the
 * discovery half of deploy step 2a-bis, which rebuilds host sidecar images
 * so a prune or reprovision can't leave one permanently missing.
 *
 * The function is extracted from the live script rather than reimplemented
 * (same approach as `tessl-exec-env-flags.test.ts`): a copy here would keep
 * passing after the shipped one drifted.
 *
 * Each case builds a fake `container/` tree in a temp dir and runs the real
 * function against it, so the assertions pin behaviour that matters on a
 * host — most importantly that the AGENT build script is never swept in
 * (it takes tag and --no-cache arguments this loop does not pass).
 */

let tmp: string;

/** Lift the shell function out of deploy.sh (or `scriptPath`) and run it in `cwd`. */
function runDiscovery(
  cwd: string,
  scriptPath?: string,
): { stdout: string; stderr: string; status: number } {
  const deploySh =
    scriptPath ?? path.join(process.cwd(), 'scripts', 'deploy.sh');
  const src = fs.readFileSync(deploySh, 'utf8');
  const start = src.indexOf('sidecar_build_scripts() {');
  expect(start).toBeGreaterThan(-1);
  const end = src.indexOf('\n}\n', start);
  expect(end).toBeGreaterThan(start);
  const fn = src.slice(start, end + 3);

  const harness = path.join(tmp, 'harness.sh');
  fs.writeFileSync(
    harness,
    [
      '#!/usr/bin/env bash',
      // -e so a syntax error or failed builtin inside the extracted
      // function aborts instead of falling through to an exit-0 with empty
      // output, which several assertions below would happily accept.
      'set -euo pipefail',
      fn,
      'sidecar_build_scripts',
      '',
    ].join('\n'),
  );
  const r = spawnSync('bash', [harness], { cwd, encoding: 'utf-8' });
  // Assert the harness itself ran. Without this, "the function emitted
  // nothing" and "the function never executed" are the same observation —
  // a regression that broke deploy.sh's parsing would show up as green.
  if (r.error) {
    throw new Error(`harness failed to spawn: ${r.error.message}`);
  }
  return {
    stdout: r.stdout ?? '',
    stderr: r.stderr ?? '',
    // Surfaced, never swallowed: "emitted nothing" and "never ran" are the
    // same stdout, and only the status tells them apart.
    status: r.status ?? -1,
  };
}

/** Create `container/<name>/build.sh`, executable unless told otherwise. */
function addSidecar(root: string, name: string, executable = true): void {
  const dir = path.join(root, 'container', name);
  fs.mkdirSync(dir, { recursive: true });
  const p = path.join(dir, 'build.sh');
  fs.writeFileSync(p, '#!/usr/bin/env bash\necho built\n');
  fs.chmodSync(p, executable ? 0o755 : 0o644);
}

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sidecar-discovery-'));
});

afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

describe('sidecar_build_scripts (scripts/deploy.sh, #895)', () => {
  it('finds every executable container/*/build.sh, sorted', () => {
    const root = fs.mkdtempSync(path.join(tmp, 'repo-'));
    addSidecar(root, 'zebra-sidecar');
    addSidecar(root, 'audible-backup');
    const { stdout, status } = runDiscovery(root);
    expect(status).toBe(0);
    expect(stdout.trim().split('\n')).toEqual([
      'container/audible-backup/build.sh',
      'container/zebra-sidecar/build.sh',
    ]);
  });

  it('never returns the AGENT build script at container/build.sh', () => {
    // The agent image is step 2a's job and takes tag / --no-cache args this
    // loop does not pass. Sweeping it in here would rebuild it a second
    // time, untagged, on every deploy.
    const root = fs.mkdtempSync(path.join(tmp, 'repo-'));
    fs.mkdirSync(path.join(root, 'container'), { recursive: true });
    const agent = path.join(root, 'container', 'build.sh');
    fs.writeFileSync(agent, '#!/usr/bin/env bash\necho agent\n');
    fs.chmodSync(agent, 0o755);
    addSidecar(root, 'audible-backup');
    const { stdout, status } = runDiscovery(root);
    expect(status).toBe(0);
    expect(stdout.trim().split('\n')).toEqual([
      'container/audible-backup/build.sh',
    ]);
  });

  it('emits nothing when no sidecars exist, rather than a literal glob', () => {
    const root = fs.mkdtempSync(path.join(tmp, 'repo-'));
    fs.mkdirSync(path.join(root, 'container', 'skills'), { recursive: true });
    const { stdout, status } = runDiscovery(root);
    expect(status).toBe(0);
    expect(stdout.trim()).toBe('');
  });

  it('FAILS on a non-executable build.sh rather than skipping past it', () => {
    // An unrunnable build script means that sidecar's image silently stops
    // being rebuilt — the #895 failure itself — so discovery exits non-zero
    // and deploy.sh aborts on it.
    const root = fs.mkdtempSync(path.join(tmp, 'repo-'));
    addSidecar(root, 'audible-backup', false);
    const { stdout, stderr, status } = runDiscovery(root);
    expect(status).not.toBe(0);
    expect(stdout.trim()).toBe('');
    expect(stderr).toMatch(/not executable/);
    expect(stderr).toMatch(/chmod \+x/);
  });

  it('still fails when one script is runnable and another is not', () => {
    // The partial case is the dangerous one: emitting the good script with a
    // zero status would let deploy proceed while one image goes unbuilt.
    const root = fs.mkdtempSync(path.join(tmp, 'repo-'));
    addSidecar(root, 'audible-backup');
    addSidecar(root, 'broken-sidecar', false);
    const { stdout, status } = runDiscovery(root);
    expect(status).not.toBe(0);
    expect(stdout).toContain('container/audible-backup/build.sh');
  });

  it('throws instead of reporting empty output when the shell function is broken', () => {
    // The guard the reviewer asked for, proven rather than assumed: before
    // the exit-status assertion, a deploy.sh whose function no longer parsed
    // produced empty stdout, and every "expect empty" case below passed.
    const broken = path.join(tmp, 'broken-deploy.sh');
    fs.writeFileSync(
      broken,
      'sidecar_build_scripts() {\n  this is not valid shell (((\n}\n',
    );
    const root = fs.mkdtempSync(path.join(tmp, 'repo-'));
    addSidecar(root, 'audible-backup');
    expect(runDiscovery(root, broken).status).not.toBe(0);
  });

  it('finds the real repo tree it ships against', () => {
    // Guards the convention itself: if audible-backup's build script moves
    // or loses its +x bit, deploy silently stops rebuilding that image and
    // #895 recurs. Runs against the actual checkout, not a fixture.
    const { stdout, status } = runDiscovery(process.cwd());
    expect(status).toBe(0);
    expect(stdout).toContain('container/audible-backup/build.sh');
  });
});
