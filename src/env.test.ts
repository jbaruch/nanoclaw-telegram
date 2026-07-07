import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, vi, afterEach } from 'vitest';

import { readEnvFile } from './env.js';

// readEnvFile resolves `.env` against process.cwd(). Point cwd at a
// throwaway dir per test via a spy — process.chdir() is unavailable in
// vitest worker threads.
function withCwd(dir: string): void {
  vi.spyOn(process, 'cwd').mockReturnValue(dir);
}

describe('readEnvFile', () => {
  const tmpDirs: string[] = [];

  function makeTmpDir(): string {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'env-test-'));
    tmpDirs.push(dir);
    return dir;
  }

  afterEach(() => {
    vi.restoreAllMocks();
    for (const dir of tmpDirs.splice(0)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('returns requested keys from .env', () => {
    const dir = makeTmpDir();
    fs.writeFileSync(
      path.join(dir, '.env'),
      '# comment\nFOO=bar\nQUOTED="with spaces"\nIGNORED=nope\n',
    );
    withCwd(dir);
    expect(readEnvFile(['FOO', 'QUOTED', 'MISSING'])).toEqual({
      FOO: 'bar',
      QUOTED: 'with spaces',
    });
  });

  it('returns {} when .env does not exist (ENOENT is the expected no-config case)', () => {
    const dir = makeTmpDir();
    withCwd(dir);
    expect(readEnvFile(['FOO'])).toEqual({});
  });

  it('propagates non-ENOENT read failures instead of masking them as "no config"', () => {
    // An unreadable-but-present .env must not silently become {} — the
    // orchestrator would run without its secrets. EISDIR stands in for
    // the unexpected-failure class deterministically (no chmod, so the
    // test also behaves under root, where EACCES wouldn't fire).
    const dir = makeTmpDir();
    fs.mkdirSync(path.join(dir, '.env'));
    withCwd(dir);
    expect(() => readEnvFile(['FOO'])).toThrow();
  });
});
