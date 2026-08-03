import { spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

/**
 * Coverage for deploy step 3b-quater in `scripts/deploy.sh` — the gate
 * enforcing `nanoclaw-host: os-package-floating` (authority-of-record for
 * `coding-policy: dependency-management` OS-Package Runtime Carve-Out).
 *
 * The carve-out lets the covered images install apt packages without version
 * specifiers, bounded by two conditions: the base image stays pinned, and the
 * package set stays the recorded one. This gate is what makes those bounds
 * real, so every case here is a mutation that must FAIL it — a suite proving
 * only the happy path would pass against a gate that prints nothing ever.
 *
 * The block is extracted from the live script rather than reimplemented, so
 * it cannot drift from what ships.
 */

let tmp: string;

/** The python block deploy.sh feeds on stdin, lifted verbatim. */
function gateSource(): string {
  const deploySh = path.join(process.cwd(), 'scripts', 'deploy.sh');
  const src = fs.readFileSync(deploySh, 'utf8');
  const start = src.indexOf("PY_OS_PKG'\n");
  expect(start).toBeGreaterThan(-1);
  const body = src.slice(start + "PY_OS_PKG'\n".length);
  const end = body.indexOf('\nPY_OS_PKG\n');
  expect(end).toBeGreaterThan(-1);
  return body.slice(0, end);
}

/**
 * Copy the three real Dockerfiles into a temp tree, optionally rewriting one,
 * then run the gate there. Starting from the shipped files means a mutation
 * is the ONLY difference from a passing run.
 */
function runGate(mutate?: (rel: string, text: string) => string): string {
  const covered = [
    'container/Dockerfile',
    'Dockerfile.orchestrator',
    'container/audible-backup/Dockerfile',
  ];
  for (const rel of covered) {
    const src = fs.readFileSync(path.join(process.cwd(), rel), 'utf8');
    const dest = path.join(tmp, rel);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, mutate ? mutate(rel, src) : src);
  }
  const r = spawnSync('python3', ['-c', gateSource()], {
    cwd: tmp,
    encoding: 'utf-8',
  });
  expect(r.error).toBeUndefined();
  expect(r.stderr).toBe('');
  return (r.stdout ?? '').trim();
}

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'os-pkg-gate-'));
});

afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

describe('deploy.sh step 3b-quater — OS-package carve-out bounds', () => {
  it('passes on the shipped Dockerfiles', () => {
    expect(runGate()).toBe('');
  });

  it('fails when a package outside the recorded set appears', () => {
    const out = runGate((rel, text) =>
      rel === 'container/audible-backup/Dockerfile'
        ? text.replace('    ffmpeg \\', '    ffmpeg \\\n    imagemagick \\')
        : text,
    );
    expect(out).toContain('imagemagick');
    expect(out).toMatch(/not in the recorded set/);
  });

  it('fails when a base image drops to :latest', () => {
    const out = runGate((rel, text) =>
      rel === 'container/audible-backup/Dockerfile'
        ? text.replace('FROM python:3.14-slim', 'FROM python:latest')
        : text,
    );
    expect(out).toMatch(/floats on :latest/);
    expect(out).toMatch(/unbounded/);
  });

  it('fails when a base image carries no tag at all', () => {
    const out = runGate((rel, text) =>
      rel === 'container/audible-backup/Dockerfile'
        ? text.replace('FROM python:3.14-slim', 'FROM python')
        : text,
    );
    expect(out).toMatch(/carries no tag/);
  });

  it('accepts a digest-pinned base without a readable tag', () => {
    // The agent and orchestrator images pin by digest; a digest bounds the
    // distro release regardless of the tag beside it.
    const out = runGate((rel, text) =>
      rel === 'Dockerfile.orchestrator'
        ? text.replace(/^FROM \S+/m, 'FROM node@sha256:' + 'a'.repeat(64))
        : text,
    );
    expect(out).toBe('');
  });

  it('fails loudly when a covered Dockerfile is missing', () => {
    fs.mkdirSync(path.join(tmp, 'container'), { recursive: true });
    const r = spawnSync('python3', ['-c', gateSource()], {
      cwd: tmp,
      encoding: 'utf-8',
    });
    expect(r.status).toBe(0);
    // Every covered image reports, rather than an empty pass.
    expect(r.stdout).toMatch(/unreadable/);
    expect(r.stdout).toContain('container/Dockerfile');
    expect(r.stdout).toContain('Dockerfile.orchestrator');
  });

  it("fails when the apt install is moved out of the gate's sight", () => {
    const out = runGate((rel, text) =>
      rel === 'container/audible-backup/Dockerfile'
        ? text.replace('apt-get install', 'apt-get satisfy')
        : text,
    );
    expect(out).toMatch(/no `apt-get install` found/);
  });

  it('sees packages installed in a SECOND apt-get block, not just the first', () => {
    // The bug this caught for real: the orchestrator installs `gh` in a
    // separate apt-get after adding GitHub's repo, which a hand-read of the
    // first block missed. Drop it from the file and the gate must not care;
    // add an unrecorded one there and it must.
    const out = runGate((rel, text) =>
      rel === 'Dockerfile.orchestrator'
        ? text.replace('apt-get install -y gh', 'apt-get install -y gh ripgrep')
        : text,
    );
    expect(out).toContain('ripgrep');
  });
});
