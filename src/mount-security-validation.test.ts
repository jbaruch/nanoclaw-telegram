import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const fixture = vi.hoisted(() => ({ directory: '' }));
vi.mock('./config.js', () => ({
  get MOUNT_ALLOWLIST_PATH() {
    return path.join(fixture.directory, 'allowlist.json');
  },
}));
vi.mock('./logger.js', () => ({
  logger: { warn: vi.fn(), error: vi.fn(), info: vi.fn(), debug: vi.fn() },
}));
beforeEach(() => {
  vi.resetModules();
  fixture.directory = fs.mkdtempSync(
    path.join(os.tmpdir(), 'telegram-mount-types-'),
  );
});
afterEach(() => fs.rmSync(fixture.directory, { recursive: true, force: true }));

it.each([
  null,
  [],
  { path: '/tmp', allowReadWrite: 'false' },
  { path: '/tmp', allowReadWrite: 0 },
  { path: 123, allowReadWrite: false },
  { path: '', allowReadWrite: false },
  { path: '/tmp\0secret', allowReadWrite: false },
  { path: '/tmp', allowReadWrite: true, description: {} },
])('refuses malformed root authority without granting RW: %j', async (root) => {
  fs.writeFileSync(
    path.join(fixture.directory, 'allowlist.json'),
    JSON.stringify({
      allowedRoots: [root],
      blockedPatterns: [],
      nonMainReadOnly: false,
    }),
  );
  const { loadMountAllowlist, validateAdditionalMounts } =
    await import('./mount-security.js');
  expect(loadMountAllowlist()).toBeNull();
  expect(
    validateAdditionalMounts(
      [{ hostPath: fixture.directory, readonly: false }],
      'fixed',
      true,
    ),
  ).toEqual([]);
});
it.each([null, false, {}, '', 'bad\0pattern'])(
  'refuses malformed blocked pattern: %j',
  async (pattern) => {
    fs.writeFileSync(
      path.join(fixture.directory, 'allowlist.json'),
      JSON.stringify({
        allowedRoots: [],
        blockedPatterns: [pattern],
        nonMainReadOnly: true,
      }),
    );
    const { loadMountAllowlist } = await import('./mount-security.js');
    expect(loadMountAllowlist()).toBeNull();
  },
);
it('preserves explicit RO permission and valid RW authority', async () => {
  fs.writeFileSync(
    path.join(fixture.directory, 'allowlist.json'),
    JSON.stringify({
      allowedRoots: [
        {
          path: fs.realpathSync(fixture.directory),
          allowReadWrite: false,
          description: 'fixed',
        },
      ],
      blockedPatterns: ['blocked-fixture'],
      nonMainReadOnly: false,
    }),
  );
  const { loadMountAllowlist, validateAdditionalMounts } =
    await import('./mount-security.js');
  expect(loadMountAllowlist()?.allowedRoots[0].allowReadWrite).toBe(false);
  expect(
    validateAdditionalMounts(
      [{ hostPath: fixture.directory, readonly: false }],
      'fixed',
      true,
    )[0]?.readonly,
  ).toBe(true);
});
