import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const { execFileMock } = vi.hoisted(() => ({ execFileMock: vi.fn() }));

vi.mock('child_process', () => ({ execFile: execFileMock }));
vi.mock('./logger.js', () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import { runSidecar } from './sidecar-runner.js';

/** Make the mocked execFile invoke its callback with a fixed result. */
function mockExec(
  error: (Error & { code?: number }) | null,
  stdout: string,
  stderr: string,
): void {
  execFileMock.mockImplementation(
    (
      _cmd: string,
      _args: string[],
      _opts: unknown,
      cb: (e: Error | null, out: string, err: string) => void,
    ) => cb(error, stdout, stderr),
  );
}

let tmpDir: string;

/**
 * Write a temp sidecars.json fixture (#850) and point the loader at it.
 * Mirrors the shipped `config/sidecars.example.json` audible-backup
 * entry so the docker-arg assertions below pin the same behavior the
 * pre-#850 in-code registry had.
 */
function writeFixtureConfig(): void {
  const p = path.join(tmpDir, 'sidecars.json');
  fs.writeFileSync(
    p,
    JSON.stringify({
      'audible-backup': {
        image: 'audible-backup:latest',
        mounts: [
          '${HOST_PROJECT_PARENT}/.audible:/root/.audible',
          '/volume1/Google Drive/Audio Books:/library',
        ],
        baseArgs: ['--json'],
        allowedFlags: ['--dry-run'],
        timeoutMs: 600000,
        maxBuffer: 10485760,
      },
    }),
  );
  vi.stubEnv('SIDECARS_CONFIG_PATH', p);
}

beforeEach(() => {
  execFileMock.mockReset();
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sidecar-runner-'));
  vi.stubEnv('HOST_PROJECT_ROOT', '/host/nanoclaw');
  writeFixtureConfig();
});

afterEach(() => {
  vi.unstubAllEnvs();
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('runSidecar', () => {
  it('rejects an unknown sidecar name without invoking docker, naming the config path', async () => {
    const result = await runSidecar({ name: 'not-registered' });
    expect(result.error).toMatch(/Unknown sidecar "not-registered"/);
    expect(result.error).toContain(path.join(tmpDir, 'sidecars.json'));
    expect(execFileMock).not.toHaveBeenCalled();
  });

  it('returns an actionable error envelope when the config is invalid, without invoking docker', async () => {
    fs.writeFileSync(path.join(tmpDir, 'sidecars.json'), 'not json {');
    const result = await runSidecar({ name: 'audible-backup' });
    expect(result.error).toMatch(/Sidecar config invalid/);
    expect(result.error).toMatch(/not valid JSON/);
    expect(execFileMock).not.toHaveBeenCalled();
  });

  it('treats a missing config as an empty registry with a discoverable fix', async () => {
    fs.rmSync(path.join(tmpDir, 'sidecars.json'));
    const result = await runSidecar({ name: 'audible-backup' });
    expect(result.error).toMatch(/Unknown sidecar "audible-backup"/);
    expect(result.error).toMatch(/\(none\)/);
    expect(execFileMock).not.toHaveBeenCalled();
  });

  it('rejects a flag outside the entry allowlist without invoking docker', async () => {
    const result = await runSidecar({
      name: 'audible-backup',
      flags: ['--dry-run', '--evil'],
    });
    expect(result.error).toMatch(/not permitted/);
    expect(result.error).toMatch(/--evil/);
    expect(execFileMock).not.toHaveBeenCalled();
  });

  it('builds docker args from the config registry plus allowlisted flags', async () => {
    mockExec(null, JSON.stringify({ books: [] }), '');
    await runSidecar({ name: 'audible-backup', flags: ['--dry-run'] });
    const [cmd, args] = execFileMock.mock.calls[0] as [string, string[]];
    expect(cmd).toBe('docker');
    expect(args).toEqual([
      'run',
      '--rm',
      '-v',
      // ${HOST_PROJECT_PARENT} expanded from HOST_PROJECT_ROOT at load
      // time — trusted config, never the caller payload.
      '/host/.audible:/root/.audible',
      '-v',
      '/volume1/Google Drive/Audio Books:/library',
      'audible-backup:latest',
      '--json',
      '--dry-run',
    ]);
  });

  it('omits optional flags when none are supplied', async () => {
    mockExec(null, JSON.stringify({ books: [] }), '');
    await runSidecar({ name: 'audible-backup' });
    const [, args] = execFileMock.mock.calls[0] as [string, string[]];
    expect(args[args.length - 1]).toBe('--json');
    expect(args).not.toContain('--dry-run');
  });

  it('relays parsed stdout JSON with a stderr log tail on success', async () => {
    mockExec(null, JSON.stringify({ books: [{ status: 'ok' }] }), 'run log');
    const result = await runSidecar({ name: 'audible-backup' });
    expect(result.books).toEqual([{ status: 'ok' }]);
    expect(result.logs).toBe('run log');
  });

  it('preserves partial success: a non-zero exit keeps books[] and adds exec_error without clobbering the script error', async () => {
    const execErr = Object.assign(new Error('Command failed: exit 1'), {
      code: 1,
    });
    mockExec(
      execErr,
      JSON.stringify({
        books: [{ status: 'ok' }, { status: 'failed' }],
        error: 'inventory missing',
      }),
      'stderr tail',
    );
    const result = await runSidecar({ name: 'audible-backup' });
    expect(result.books).toHaveLength(2);
    // The script's own top-level error wins; the exec error rides alongside.
    expect(result.error).toBe('inventory missing');
    expect(result.exec_error).toMatch(/Command failed/);
    expect(result.logs).toBe('stderr tail');
  });

  it('falls back to a bare error envelope when stdout is not JSON', async () => {
    const execErr = Object.assign(new Error('boom'), { code: 127 });
    mockExec(execErr, 'not json at all', 'err out');
    const result = await runSidecar({ name: 'audible-backup' });
    expect(result.error).toBe('boom');
    expect(result.raw).toBe('not json at all');
    expect(result.logs).toBe('err out');
  });

  it('does not attach fields to a non-object JSON stdout (uses the raw envelope)', async () => {
    mockExec(null, '"just a string"', '');
    const result = await runSidecar({ name: 'audible-backup' });
    expect(result.raw).toBe('"just a string"');
    expect(result.error).toBeUndefined();
  });
});
