import { spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const REPO = process.cwd();
let tmp: string;
let bin: string;

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'node-bootstrap-'));
  bin = path.join(tmp, 'bin');
  fs.mkdirSync(bin);
  fs.mkdirSync(path.join(tmp, 'setup'));
  fs.copyFileSync(path.join(REPO, 'setup.sh'), path.join(tmp, 'setup.sh'));
  fs.copyFileSync(
    path.join(REPO, 'setup/install-node.sh'),
    path.join(tmp, 'setup/install-node.sh'),
  );
  fs.writeFileSync(path.join(tmp, '.nvmrc'), '24.22.0\n');
});

afterEach(() => fs.rmSync(tmp, { recursive: true, force: true }));

function executable(name: string, contents: string): void {
  fs.writeFileSync(path.join(bin, name), `#!/bin/sh\n${contents}\n`, {
    mode: 0o755,
  });
}

function run(script: string, extra: NodeJS.ProcessEnv = {}) {
  return spawnSync('/bin/bash', [path.join(tmp, script)], {
    cwd: tmp,
    encoding: 'utf8',
    env: {
      ...process.env,
      PATH: `${bin}:${process.env.PATH}`,
      NANOCLAW_NODE_BOOTSTRAP_DIR: path.join(tmp, 'runtime'),
      BOOTSTRAP_FIXTURE: tmp,
      ...extra,
    },
  });
}

describe('the shipped setup runtime gate', () => {
  it.each([
    ['v20.20.1', false],
    ['v22.23.2', false],
    ['v24.22.0', true],
    ['v26.12.0', true],
  ])('checks %s before installing dependencies', (version, supported) => {
    executable('node', `if [ "$1" = --version ]; then echo '${version}'; fi`);
    executable('npm', 'touch "$BOOTSTRAP_FIXTURE/npm-called"');
    const result = run('setup.sh');
    expect(result.status, result.stderr).toBe(supported ? 0 : 2);
    expect(result.stdout).toContain(`NODE_OK: ${supported}`);
    expect(fs.existsSync(path.join(tmp, 'npm-called'))).toBe(supported);
  });

  it('reports a malformed runtime version instead of suppressing the comparison error', () => {
    executable('node', 'echo vbanana');
    const result = run('setup.sh');
    expect(result.status).toBe(2);
    expect(result.stderr).toContain('Invalid Node version');
  });

  it('preserves a failing version command status and diagnostic', () => {
    executable('node', 'echo fixture-failure >&2; exit 7');
    const result = run('setup.sh');
    expect(result.status).toBe(7);
    expect(result.stderr).toContain('fixture-failure');
    expect(result.stderr).toContain('Cannot read the Node version');
  });
});

describe('the pinned runtime installer', () => {
  beforeEach(() => {
    executable(
      'git',
      `
printf '%s\\n' "$*" >>"$BOOTSTRAP_FIXTURE/git-calls"
if [ "$1" = clone ]; then
  for destination do :; done
  mkdir -p "$destination"
  cp "$BOOTSTRAP_FIXTURE/nvm-fixture.sh" "$destination/nvm.sh"
elif [ "$3" = rev-parse ]; then
  echo "\${STUB_NVM_SHA:-a885b885fef16fac4bc544188fb25e9e37ae83e8}"
fi`,
    );
    fs.writeFileSync(
      path.join(tmp, 'nvm-fixture.sh'),
      `
nvm() {
  case "$1" in
    install)
      printf '%s\\n' "$2" >"$BOOTSTRAP_FIXTURE/installed-version"
      cp "$BOOTSTRAP_FIXTURE/bin/node" "$NVM_DIR/node"
      ;;
    which) printf '%s\\n' "$NVM_DIR/node" ;;
  esac
}
`,
    );
    executable('node', 'echo v24.22.0');
  });

  it('can be sourced without installing or changing caller shell state', () => {
    fs.writeFileSync(
      path.join(tmp, 'source-only.sh'),
      `#!/usr/bin/env bash
set +eu
before_flags=$-
before_nvm="\${NVM_DIR-__unset__}"
source "$BOOTSTRAP_FIXTURE/setup/install-node.sh"
[ "$-" = "$before_flags" ] || exit 7
[ "\${NVM_DIR-__unset__}" = "$before_nvm" ] || exit 8
printf 'source-only\\n'
`,
    );
    const result = run('source-only.sh');
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout).toBe('source-only\n');
    expect(fs.existsSync(path.join(tmp, 'git-calls'))).toBe(false);
    expect(fs.existsSync(path.join(tmp, 'installed-version'))).toBe(false);
    expect(fs.existsSync(path.join(tmp, 'runtime'))).toBe(false);
  });

  it('installs the exact .nvmrc release through the pinned upstream installer', () => {
    const result = run('setup/install-node.sh');
    expect(result.status, result.stderr).toBe(0);
    expect(fs.readFileSync(path.join(tmp, 'installed-version'), 'utf8')).toBe(
      '24.22.0\n',
    );
    expect(result.stdout.trim()).toBe(path.join(tmp, 'runtime', 'nvm-v0.40.8'));
    const calls = fs.readFileSync(path.join(tmp, 'git-calls'), 'utf8');
    expect(calls).toContain(
      'clone --depth 1 --branch v0.40.8 https://github.com/nvm-sh/nvm.git',
    );
  });

  it('refuses to source an unexpected installer revision', () => {
    const result = run('setup/install-node.sh', { STUB_NVM_SHA: 'unexpected' });
    expect(result.status).toBe(1);
    expect(result.stderr).toContain('unexpected revision');
    expect(fs.existsSync(path.join(tmp, 'installed-version'))).toBe(false);
  });

  it('rejects a floating runtime pin before contacting the installer', () => {
    fs.writeFileSync(path.join(tmp, '.nvmrc'), '24\n');
    const result = run('setup/install-node.sh');
    expect(result.status).toBe(2);
    expect(result.stderr).toContain('exact Node release');
    expect(fs.existsSync(path.join(tmp, 'git-calls'))).toBe(false);
  });

  it('rejects a binary that differs from the requested release', () => {
    executable('node', 'echo v26.12.0');
    const result = run('setup/install-node.sh');
    expect(result.status).toBe(1);
    expect(result.stderr).toContain('does not match .nvmrc');
  });
});
