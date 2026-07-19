import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

const { execFileMock } = vi.hoisted(() => ({ execFileMock: vi.fn() }));

vi.mock('child_process', () => ({ execFile: execFileMock }));
vi.mock('./logger.js', () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import { runSidecar, getSidecarRegistry } from './sidecar-runner.js';

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

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('getSidecarRegistry', () => {
  beforeEach(() => {
    vi.stubEnv('HOST_PROJECT_ROOT', '/host/nanoclaw');
  });

  it('exposes the audible-backup entry with a trusted image + mounts', () => {
    const reg = getSidecarRegistry();
    const spec = reg['audible-backup'];
    expect(spec.image).toBe('audible-backup:latest');
    // Mount host paths come from trusted config, resolved from
    // HOST_PROJECT_ROOT at call time — never from a caller payload.
    expect(spec.mounts).toEqual([
      '/host/.audible:/root/.audible',
      '/volume1/Google Drive/Audio Books:/library',
    ]);
    expect(spec.baseArgs).toEqual(['--json']);
    expect(spec.allowedFlags).toEqual(['--dry-run']);
  });
});

describe('runSidecar', () => {
  beforeEach(() => {
    execFileMock.mockReset();
    vi.stubEnv('HOST_PROJECT_ROOT', '/host/nanoclaw');
  });

  it('rejects an unknown sidecar name without invoking docker', async () => {
    const result = await runSidecar({ name: 'not-registered' });
    expect(result.error).toMatch(/Unknown sidecar "not-registered"/);
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

  it('builds docker args from the registry plus allowlisted flags', async () => {
    mockExec(null, JSON.stringify({ books: [] }), '');
    await runSidecar({ name: 'audible-backup', flags: ['--dry-run'] });
    const [cmd, args] = execFileMock.mock.calls[0] as [string, string[]];
    expect(cmd).toBe('docker');
    expect(args).toEqual([
      'run',
      '--rm',
      '-v',
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
