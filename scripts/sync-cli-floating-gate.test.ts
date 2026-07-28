import { spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

/**
 * Coverage for deploy step 3b-ter in `scripts/deploy.sh` — the gate enforcing
 * `nanoclaw-host: sync-cli-floating` (authority-of-record for
 * `coding-policy: dependency-management` First-Party Co-Shipped Dependency
 * Carve-Out).
 *
 * The gate's whole job is to fail; a suite that only proved the happy path
 * would pass just as well against a check that prints nothing ever. So every
 * case here is a mutation of the REAL `container/Dockerfile`, and the control
 * asserts silence on the unmutated file.
 *
 * Like `tessl-exec-env-flags.test.ts`, this extracts the live block out of
 * `deploy.sh` rather than reimplementing it — a reimplementation would drift
 * from the shipped gate and pass while production fails.
 */

let tmp: string;

/** The python block deploy.sh feeds on stdin, lifted verbatim. */
function gateSource(): string {
  const deploySh = path.join(process.cwd(), 'scripts', 'deploy.sh');
  const src = fs.readFileSync(deploySh, 'utf8');
  const start = src.indexOf("PY_SYNC_CLI'\n");
  expect(start).toBeGreaterThan(-1);
  const body = src.slice(start + "PY_SYNC_CLI'\n".length);
  const end = body.indexOf('\nPY_SYNC_CLI\n');
  expect(end).toBeGreaterThan(-1);
  return body.slice(0, end);
}

/** Run the gate against a Dockerfile body; returns what it printed. */
function runGate(dockerfile: string): string {
  fs.mkdirSync(path.join(tmp, 'container'), { recursive: true });
  fs.writeFileSync(path.join(tmp, 'container', 'Dockerfile'), dockerfile);
  const r = spawnSync('python3', ['-c', gateSource()], {
    cwd: tmp,
    encoding: 'utf-8',
  });
  expect(r.status).toBe(0);
  expect(r.stderr).toBe('');
  return (r.stdout ?? '').trim();
}

/** The shipped Dockerfile — the fixture every mutation starts from. */
function realDockerfile(): string {
  return fs.readFileSync(
    path.join(process.cwd(), 'container', 'Dockerfile'),
    'utf8',
  );
}

const INSTALL = 'RUN npm install -g jbaruch/reclaim-tripit-timezones-sync';
const ADD_LINE =
  'ADD https://api.github.com/repos/jbaruch/reclaim-tripit-timezones-sync/commits/main';

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'sync-cli-gate-test-'));
});

afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

describe('deploy.sh step 3b-ter — sync CLI floating gate', () => {
  it('passes on the shipped Dockerfile', () => {
    expect(runGate(realDockerfile())).toBe('');
  });

  it('fails when the install carries a tag specifier', () => {
    const mutated = realDockerfile().replace(
      `${INSTALL} `,
      `${INSTALL}#v0.4.0 `,
    );
    expect(mutated).not.toBe(realDockerfile());
    expect(runGate(mutated)).toContain(
      "the install carries the specifier '#v0.4.0'",
    );
  });

  it('fails when the install carries a commit SHA specifier', () => {
    const mutated = realDockerfile().replace(
      `${INSTALL} `,
      `${INSTALL}#dc582da `,
    );
    expect(runGate(mutated)).toContain('the install carries the specifier');
  });

  it('fails when the refetch trigger is missing', () => {
    const mutated = realDockerfile()
      .split('\n')
      .filter((l) => !l.startsWith(ADD_LINE))
      .join('\n');
    expect(runGate(mutated)).toContain(
      'is not an `ADD <commits-url> <dest>` refetch trigger',
    );
  });

  it('fails when the refetch trigger names a different repo', () => {
    const mutated = realDockerfile().replace(
      'repos/jbaruch/reclaim-tripit-timezones-sync/commits',
      'repos/jbaruch/some-other-repo/commits',
    );
    expect(runGate(mutated)).toContain(
      "the ADD refetch trigger points at 'jbaruch/some-other-repo'",
    );
  });

  it('fails when the trigger sits above a DIFFERENT instruction, not the install', () => {
    // The regression the reviewer caught: an ADD present anywhere used to
    // satisfy the gate. BuildKit invalidates downward from the changed
    // instruction, so a trigger separated from the install by another RUN
    // busts the wrong layer and leaves the install cached.
    const lines = realDockerfile().split('\n');
    const addIdx = lines.findIndex((l) => l.startsWith(ADD_LINE));
    expect(addIdx).toBeGreaterThan(-1);
    lines.splice(addIdx + 1, 0, 'RUN echo "unrelated layer"');
    expect(runGate(lines.join('\n'))).toContain(
      'is not an `ADD <commits-url> <dest>` refetch trigger',
    );
  });

  it('fails when the ADD has no destination argument', () => {
    const mutated = realDockerfile().replace(
      /^ADD (https:\S+)\s+\S+$/m,
      'ADD $1',
    );
    expect(mutated).not.toBe(realDockerfile());
    expect(runGate(mutated)).toContain('refetch trigger');
  });

  it('fails when the install line is gone entirely', () => {
    const mutated = realDockerfile()
      .split('\n')
      .filter((l) => !l.startsWith(INSTALL))
      .join('\n');
    expect(runGate(mutated)).toContain('no `RUN npm install -g');
  });

  it('fails loudly when the Dockerfile is unreadable', () => {
    // No container/Dockerfile written at all — a moved target must report,
    // never pass vacuously.
    const r = spawnSync('python3', ['-c', gateSource()], {
      cwd: tmp,
      encoding: 'utf-8',
    });
    expect(r.status).toBe(0);
    expect((r.stdout ?? '').trim()).toContain('unreadable');
  });
});
