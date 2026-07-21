import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import { loadSidecarRegistry, sidecarsConfigPath } from './sidecar-config.js';

let tmpDir: string;

function writeConfig(body: unknown): string {
  const p = path.join(tmpDir, 'sidecars.json');
  fs.writeFileSync(p, typeof body === 'string' ? body : JSON.stringify(body));
  vi.stubEnv('SIDECARS_CONFIG_PATH', p);
  return p;
}

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sidecar-config-'));
});

afterEach(() => {
  vi.unstubAllEnvs();
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('sidecarsConfigPath', () => {
  it('defaults to config/sidecars.json under the working directory (the container-visible path, not HOST_PROJECT_ROOT)', () => {
    // HOST_PROJECT_ROOT points at the HOST filesystem for docker -v
    // composition; the config FILE is read through this process's own
    // fs, where docker-compose mounts ./config at <cwd>/config.
    vi.stubEnv('HOST_PROJECT_ROOT', '/host/nanoclaw');
    expect(sidecarsConfigPath()).toBe(
      path.join(process.cwd(), 'config', 'sidecars.json'),
    );
  });

  it('honors the SIDECARS_CONFIG_PATH override', () => {
    vi.stubEnv('SIDECARS_CONFIG_PATH', '/elsewhere/sidecars.json');
    expect(sidecarsConfigPath()).toBe('/elsewhere/sidecars.json');
  });
});

describe('loadSidecarRegistry', () => {
  it('treats a missing file as an empty registry (platform install without sidecars)', () => {
    vi.stubEnv(
      'SIDECARS_CONFIG_PATH',
      path.join(tmpDir, 'does-not-exist.json'),
    );
    const result = loadSidecarRegistry();
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.registry).toEqual({});
  });

  it('loads a valid entry and applies defaults for omitted fields', () => {
    writeConfig({ minimal: { image: 'minimal:latest' } });
    const result = loadSidecarRegistry();
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.registry.minimal).toEqual({
      image: 'minimal:latest',
      mounts: [],
      baseArgs: [],
      allowedFlags: [],
      timeoutMs: 600_000,
      maxBuffer: 10 * 1024 * 1024,
    });
  });

  it('expands ${VAR} from the environment and the HOST_PROJECT_PARENT builtin in mounts', () => {
    vi.stubEnv('HOST_PROJECT_ROOT', '/host/nanoclaw');
    vi.stubEnv('MEDIA_ROOT', '/volume1/media');
    writeConfig({
      backup: {
        image: 'backup:latest',
        mounts: [
          '${HOST_PROJECT_PARENT}/.audible:/root/.audible',
          '${MEDIA_ROOT}/books:/library',
        ],
      },
    });
    const result = loadSidecarRegistry();
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.registry.backup.mounts).toEqual([
      '/host/.audible:/root/.audible',
      '/volume1/media/books:/library',
    ]);
  });

  it('fails loudly on an unset ${VAR} instead of expanding to empty', () => {
    writeConfig({
      backup: {
        image: 'backup:latest',
        mounts: ['${DEFINITELY_UNSET_VAR_850}/x:/x'],
      },
    });
    const result = loadSidecarRegistry();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error).toMatch(/DEFINITELY_UNSET_VAR_850/);
    expect(result.error).toMatch(/"backup"\.mounts\[0\]/);
  });

  it('fails loudly on invalid JSON with the config path in the message', () => {
    const p = writeConfig('not json {');
    const result = loadSidecarRegistry();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error).toContain(p);
    expect(result.error).toMatch(/not valid JSON/);
  });

  it('rejects a non-object top level', () => {
    writeConfig([{ image: 'x' }]);
    const result = loadSidecarRegistry();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error).toMatch(/must be a JSON object/);
  });

  it('rejects a missing/empty image', () => {
    writeConfig({ broken: { mounts: [] } });
    const result = loadSidecarRegistry();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error).toMatch(/"broken"\.image/);
  });

  it('rejects malformed bind specs with the entry named (empty side, too many segments)', () => {
    for (const mount of [
      '/just-a-path',
      ':/container',
      '/host:',
      '/host:/container:',
      'a:b:c:d',
    ]) {
      writeConfig({ broken: { image: 'x:1', mounts: [mount] } });
      const result = loadSidecarRegistry();
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.error).toMatch(/"broken"\.mounts\[0\]/);
      expect(result.error).toMatch(/hostPath:containerPath/);
    }
  });

  it('accepts a non-empty :options third segment', () => {
    writeConfig({ ok: { image: 'x:1', mounts: ['/host:/container:ro'] } });
    const result = loadSidecarRegistry();
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.registry.ok.mounts).toEqual(['/host:/container:ro']);
  });

  it('rejects an explicit null for a defaultable field (defaults apply only when omitted)', () => {
    writeConfig({ broken: { image: 'x:1', mounts: null } });
    const result = loadSidecarRegistry();
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.error).toMatch(/"broken"\.mounts must be an array/);
  });

  it('rejects wrong-typed fields with entry-specific messages', () => {
    for (const [patch, needle] of [
      [{ mounts: 'not-an-array' }, /"broken"\.mounts must be an array/],
      [{ baseArgs: [42] }, /"broken"\.baseArgs must be an array of strings/],
      [{ allowedFlags: {} }, /"broken"\.allowedFlags must be an array/],
      [{ timeoutMs: -5 }, /"broken"\.timeoutMs must be a positive number/],
      [{ maxBuffer: 'big' }, /"broken"\.maxBuffer must be a positive number/],
    ] as const) {
      writeConfig({ broken: { image: 'x:1', ...patch } });
      const result = loadSidecarRegistry();
      expect(result.ok).toBe(false);
      if (result.ok) return;
      expect(result.error).toMatch(needle);
    }
  });
});
