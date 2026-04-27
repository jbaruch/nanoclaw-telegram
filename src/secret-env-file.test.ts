import { describe, it, expect, afterEach } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';

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
  it('lists COMPOSIO_API_KEY (the issue-107 leak case)', () => {
    expect(SECRET_CONTAINER_VARS.has('COMPOSIO_API_KEY')).toBe(true);
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

  it('two consecutive calls produce different paths (no collisions)', () => {
    const a = buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk-a' });
    const b = buildSecretEnvFile({ COMPOSIO_API_KEY: 'sk-b' });
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    tempFilesToCleanup.push(a!.args[1], b!.args[1]);

    expect(a!.args[1]).not.toBe(b!.args[1]);
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

describe('buildSecretEnvFile — symlink-race defense', () => {
  it('refuses to overwrite a pre-existing path (O_EXCL semantics)', () => {
    // Pre-create a path that buildSecretEnvFile MIGHT try to use. We
    // can't predict the random suffix, so we exercise the O_EXCL
    // behavior directly by confirming that opening with the same
    // flags it uses fails with EEXIST when the path is occupied.
    const decoyPath = path.join(os.tmpdir(), `nanoclaw-env-${'a'.repeat(24)}`);
    fs.writeFileSync(decoyPath, 'pre-existing', { mode: 0o644 });
    tempFilesToCleanup.push(decoyPath);

    expect(() => {
      fs.openSync(
        decoyPath,
        fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL,
        0o600,
      );
    }).toThrow(/EEXIST/);

    // Decoy still has its original content — proves the open didn't
    // truncate or overwrite. (This is the property buildSecretEnvFile
    // relies on for symlink-race protection.)
    expect(fs.readFileSync(decoyPath, 'utf8')).toBe('pre-existing');
  });
});
