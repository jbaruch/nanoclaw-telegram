import { describe, it, expect, afterEach, vi } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import * as crypto from 'crypto';

// Mock crypto so tests can pin `randomBytes` to deterministic values
// without depending on probabilistic non-collision (testing-standards:
// "Tests must be deterministic — no self-generated random test data").
// Default behavior is the real implementation; individual tests use
// `mockReturnValueOnce(...)` to control specific calls.
vi.mock('crypto', async () => {
  const actual = await vi.importActual<typeof import('crypto')>('crypto');
  return {
    ...actual,
    randomBytes: vi.fn(actual.randomBytes),
  };
});

import {
  SECRET_CONTAINER_VARS,
  buildSecretEnvFile,
} from './container-runner.js';

// Track files created by tests so afterEach can clean any that escape
// their own cleanup (e.g. test failed before calling cleanup).
const tempFilesToCleanup: string[] = [];

afterEach(() => {
  while (tempFilesToCleanup.length) {
    const p = tempFilesToCleanup.pop();
    if (!p) continue;
    try {
      fs.unlinkSync(p);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
    }
  }
});

describe('SECRET_CONTAINER_VARS', () => {
  it('lists COMPOSIO_API_KEY (the issue-107 leak case — REST gateway, x-api-key, project-scoped ak_* key)', () => {
    expect(SECRET_CONTAINER_VARS.has('COMPOSIO_API_KEY')).toBe(true);
  });

  it('lists COMPOSIO_MCP_URL (headless custom-MCP-server URL, x-api-key — account-identifying server id, same env-file treatment as the API key; replaced the retired ck_* COMPOSIO_MCP_KEY when Connect MCP moved to interactive OAuth)', () => {
    expect(SECRET_CONTAINER_VARS.has('COMPOSIO_MCP_URL')).toBe(true);
  });

  it('no longer lists COMPOSIO_MCP_KEY (Connect MCP gateway moved to interactive AuthKit-JWT OAuth; the headless path uses the ak_* key against a custom MCP server)', () => {
    expect(SECRET_CONTAINER_VARS.has('COMPOSIO_MCP_KEY')).toBe(false);
  });

  it('lists COMPOSIO_USER_ID (account-identifying — same env-file treatment as the API key, jbaruch/nanoclaw#509)', () => {
    expect(SECRET_CONTAINER_VARS.has('COMPOSIO_USER_ID')).toBe(true);
  });

  it('lists GITHUB_TOKEN (`gh` CLI inside main/trusted containers — cost-monitor dashboard skills bypass the Composio MCP tool-schema cache_create tax; same env-file treatment as the Composio keys, OneCLI migration target jbaruch/nanoclaw#564)', () => {
    expect(SECRET_CONTAINER_VARS.has('GITHUB_TOKEN')).toBe(true);
  });

  it('lists BYAIR_MCP_URL (personal MCP link with API key inline — read by jbaruch/nanoclaw-travel precheck for byAir flight-status polling)', () => {
    expect(SECRET_CONTAINER_VARS.has('BYAIR_MCP_URL')).toBe(true);
  });

  it('lists GOOGLE_MAPS_API_KEY (Distance Matrix API key — read by jbaruch/nanoclaw-travel precheck for traffic-aware time-to-leave)', () => {
    expect(SECRET_CONTAINER_VARS.has('GOOGLE_MAPS_API_KEY')).toBe(true);
  });

  it('lists TOMTOM_API_KEY (TomTom Routing/Search key — read by jbaruch/nanoclaw-travel maps_client + drive-planner, calls api.tomtom.com)', () => {
    expect(SECRET_CONTAINER_VARS.has('TOMTOM_API_KEY')).toBe(true);
  });

  it('lists YOUTUBE_API_KEY (YouTube Data API v3 key — read by the admin tile youtube-comment-check skill, native API per jbaruch/nanoclaw-admin#339)', () => {
    expect(SECRET_CONTAINER_VARS.has('YOUTUBE_API_KEY')).toBe(true);
  });

  it('lists SESSIONIZE_SPEAKER_KEY (Sessionize speaker-profile key — read by the conferences tile discover-open-cfps.py for open-CFP discovery, jbaruch/nanoclaw-conferences#9)', () => {
    expect(SECRET_CONTAINER_VARS.has('SESSIONIZE_SPEAKER_KEY')).toBe(true);
  });

  it('lists SESSIONIZE_EVENT_API_KEY (Sessionize event key — read by the conferences tile verify-sessionize.py for per-slug deadline verification)', () => {
    expect(SECRET_CONTAINER_VARS.has('SESSIONIZE_EVENT_API_KEY')).toBe(true);
  });

  it('does NOT include placeholder vars (proxied through OneCLI)', () => {
    expect(SECRET_CONTAINER_VARS.has('ANTHROPIC_API_KEY')).toBe(false);
    expect(SECRET_CONTAINER_VARS.has('CLAUDE_CODE_OAUTH_TOKEN')).toBe(false);
  });

  it('does NOT include non-secret config vars', () => {
    expect(SECRET_CONTAINER_VARS.has('TZ')).toBe(false);
    expect(SECRET_CONTAINER_VARS.has('AGENT_MODEL')).toBe(false);
    expect(SECRET_CONTAINER_VARS.has('NANOCLAW_CHAT_JID')).toBe(false);
  });
});

describe('buildSecretEnvFile', () => {
  it('returns null when there are no secrets to forward', () => {
    expect(buildSecretEnvFile({})).toBeNull();
  });

  it('skips empty-string values (treats as "not set")', () => {
    expect(buildSecretEnvFile({ COMPOSIO_API_KEY: '' })).toBeNull();
  });

  it('writes the env-file with mode 0600 and emits --env-file args', () => {
    const result = buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk-real-secret-1' });
    expect(result).not.toBeNull();
    const filePath = result!.args[1];
    tempFilesToCleanup.push(filePath);

    expect(result!.args[0]).toBe('--env-file');
    expect(filePath).toMatch(
      new RegExp(
        `^${os.tmpdir().replace(/[\\^$*+?.()|[\]{}]/g, '\\$&')}/nanoclaw-env-[0-9a-f]{24}$`,
      ),
    );

    const stat = fs.statSync(filePath);
    // Mask off the file-type bits and assert the permission bits are
    // exactly 0600 — `mode & 0o777 === 0o600` rules out 0644/0666 etc.
    expect(stat.mode & 0o777).toBe(0o600);

    const content = fs.readFileSync(filePath, 'utf8');
    expect(content).toBe('COMPOSIO_API_KEY=sk-real-secret-1\n');
  });

  it('emits one KEY=value line per secret in the env-file', () => {
    const result = buildSecretEnvFile({
      COMPOSIO_API_KEY: 'sk-a',
      OTHER_SECRET: 'sk-b',
    });
    expect(result).not.toBeNull();
    const filePath = result!.args[1];
    tempFilesToCleanup.push(filePath);

    const content = fs.readFileSync(filePath, 'utf8');
    expect(content).toContain('COMPOSIO_API_KEY=sk-a\n');
    expect(content).toContain('OTHER_SECRET=sk-b\n');
  });

  it('emits both COMPOSIO_API_KEY and COMPOSIO_MCP_URL when both are present (REST + headless custom MCP server, both on the ak_* key)', () => {
    // Both forward through the same env-file. The `ak_*` project key
    // (COMPOSIO_API_KEY) authenticates BOTH surfaces via `x-api-key`:
    // REST and the headless custom MCP server. COMPOSIO_MCP_URL carries
    // that server's account-specific `/v3/mcp/<id>/mcp` URL, kept off
    // the docker command line for the same reason COMPOSIO_USER_ID is.
    const result = buildSecretEnvFile({
      COMPOSIO_API_KEY: 'ak-rest-project-key',
      COMPOSIO_MCP_URL:
        'https://backend.composio.dev/v3/mcp/00000000-0000-0000-0000-000000000000/mcp',
    });
    expect(result).not.toBeNull();
    const filePath = result!.args[1];
    tempFilesToCleanup.push(filePath);

    const content = fs.readFileSync(filePath, 'utf8');
    expect(content).toContain('COMPOSIO_API_KEY=ak-rest-project-key\n');
    expect(content).toContain(
      'COMPOSIO_MCP_URL=https://backend.composio.dev/v3/mcp/00000000-0000-0000-0000-000000000000/mcp\n',
    );
  });

  it('refuses values containing newlines or NUL bytes', () => {
    expect(() =>
      buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk\nmalicious=yes' }),
    ).toThrow(/CR\/LF\/NUL/);
    expect(() => buildSecretEnvFile({ K: 'v\rcr' })).toThrow(/CR\/LF\/NUL/);
    expect(() => buildSecretEnvFile({ K: 'v\0nul' })).toThrow(/CR\/LF\/NUL/);
  });

  it('cleanup() removes the env-file', () => {
    const result = buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk-cleanup-test' });
    expect(result).not.toBeNull();
    const filePath = result!.args[1];

    expect(fs.existsSync(filePath)).toBe(true);
    result!.cleanup();
    expect(fs.existsSync(filePath)).toBe(false);
  });

  it('cleanup() is idempotent — safe to call from both close and error handlers', () => {
    const result = buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk-idem' });
    expect(result).not.toBeNull();
    const filePath = result!.args[1];

    result!.cleanup();
    // Second invocation must not throw — the spawn-error and close
    // handlers both call cleanup() unconditionally.
    expect(() => result!.cleanup()).not.toThrow();
    expect(fs.existsSync(filePath)).toBe(false);
  });

  it('uses the random suffix from crypto.randomBytes (deterministic via mock)', () => {
    // Pin randomBytes to a fixed buffer; assert the resulting path
    // contains exactly that hex string. Earlier formulation asserted
    // two random draws never collide — non-deterministic per the
    // testing-standards rule. This formulation verifies the actual
    // mapping (suffix sourced from randomBytes) without coupling to
    // probabilistic behavior.
    const fixedSuffix = Buffer.from('0123456789abcdef01234567', 'hex');
    vi.mocked(crypto.randomBytes).mockReturnValueOnce(
      fixedSuffix as unknown as ReturnType<typeof crypto.randomBytes>,
    );

    const result = buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk-pinned' });
    expect(result).not.toBeNull();
    tempFilesToCleanup.push(result!.args[1]);

    expect(result!.args[1]).toBe(
      path.join(os.tmpdir(), `nanoclaw-env-0123456789abcdef01234567`),
    );
  });

  it('writes the env-file before returning so docker can read it immediately', () => {
    const result = buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk-sync' });
    expect(result).not.toBeNull();
    tempFilesToCleanup.push(result!.args[1]);

    // No await / no setTimeout — file must exist synchronously by the
    // time buildSecretEnvFile returns. This is the contract docker
    // relies on: by the time `--env-file <path>` is in argv, <path>
    // is already populated on disk.
    expect(fs.existsSync(result!.args[1])).toBe(true);
  });
});

describe('buildSecretEnvFile — write-failure cleanup', () => {
  it('removes the tempfile when the write fails (no orphaned partial-secret file)', () => {
    // Pin randomBytes so we know exactly which path will be created.
    const fixedSuffix = Buffer.from('cafebabecafebabecafebabe', 'hex');
    vi.mocked(crypto.randomBytes).mockReturnValueOnce(
      fixedSuffix as unknown as ReturnType<typeof crypto.randomBytes>,
    );
    const expectedPath = path.join(
      os.tmpdir(),
      `nanoclaw-env-cafebabecafebabecafebabe`,
    );

    // Force writeFileSync to throw AFTER the open (so the file is
    // already created on disk). Spy is restored in afterEach guard
    // below — failing this test must not leak a global mock.
    const writeSpy = vi
      .spyOn(fs, 'writeFileSync')
      .mockImplementationOnce(() => {
        throw new Error('simulated EIO');
      });

    expect(() =>
      buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk-write-fail' }),
    ).toThrow(/simulated EIO/);

    // The write failed; the tempfile must NOT remain on disk. Without
    // the cleanup, the file would persist with whatever bytes (if any)
    // had been flushed before the failure — including potentially the
    // full secret if writeFileSync got partway through.
    expect(fs.existsSync(expectedPath)).toBe(false);

    writeSpy.mockRestore();
    // Defensive: if cleanup somehow missed the file, afterEach will
    // catch it.
    tempFilesToCleanup.push(expectedPath);
  });
});

describe('buildSecretEnvFile — symlink-race defense', () => {
  it('throws and leaves a pre-existing path untouched (O_EXCL behavior)', () => {
    // Pin randomBytes to a known suffix, pre-create the exact path
    // buildSecretEnvFile will target, then assert the function's
    // observable result: it throws (refusing to overwrite) AND the
    // pre-existing content is preserved. Tests the module's outcome,
    // not raw fs.openSync behavior.
    const fixedSuffix = Buffer.from('deadbeefdeadbeefdeadbeef', 'hex');
    vi.mocked(crypto.randomBytes).mockReturnValueOnce(
      fixedSuffix as unknown as ReturnType<typeof crypto.randomBytes>,
    );
    const expectedPath = path.join(
      os.tmpdir(),
      `nanoclaw-env-deadbeefdeadbeefdeadbeef`,
    );
    fs.writeFileSync(expectedPath, 'pre-existing-decoy', { mode: 0o644 });
    tempFilesToCleanup.push(expectedPath);

    expect(() =>
      buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk-symlink-race' }),
    ).toThrow(/EEXIST/);

    // The pre-existing file is unmodified — buildSecretEnvFile didn't
    // overwrite or truncate it. This is the symlink-race defense
    // property: a local attacker can't pre-create the path as a
    // symlink to elsewhere and have buildSecretEnvFile silently
    // write the secret through the symlink.
    expect(fs.readFileSync(expectedPath, 'utf8')).toBe('pre-existing-decoy');
  });
});
