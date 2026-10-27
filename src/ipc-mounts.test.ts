import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
const fixture = vi.hoisted(() => ({ root: '' }));
vi.mock('./config.js', () => ({
  get DATA_DIR() {
    return path.join(fixture.root, 'project', 'data');
  },
  get GROUPS_DIR() {
    return path.join(fixture.root, 'project', 'groups');
  },
}));
import { protectIpcClaimMounts } from './ipc-mounts.js';
let project: string, data: string, claims: string, empty: string;
beforeEach(() => {
  fixture.root = fs.mkdtempSync(path.join(os.tmpdir(), 'telegram-ipc-masks-'));
  project = path.join(fixture.root, 'project');
  data = path.join(project, 'data');
  fs.mkdirSync(path.join(data, 'ipc', 'main'), { recursive: true });
  claims = path.join(data, 'ipc', '.claimed-v1');
  empty = path.join(data, 'ipc', '.claims-empty');
});
afterEach(() => fs.rmSync(fixture.root, { recursive: true, force: true }));
function parent() {
  return {
    hostPath: project,
    containerPath: '/workspace/project',
    readonly: true,
  };
}
it('canonicalizes both RO parent views and appends exact empty masks after every root', () => {
  const alias = path.join(fixture.root, 'alias');
  fs.symlinkSync(project, alias);
  const own = {
    hostPath: path.join(data, 'ipc', 'main'),
    containerPath: '/workspace/ipc',
    readonly: false,
  };
  const mounts = protectIpcClaimMounts([
    parent(),
    {
      hostPath: alias,
      containerPath: '/workspace/extra/alias',
      readonly: true,
    },
    own,
  ]);
  expect(mounts).toHaveLength(5);
  expect(
    mounts
      .slice(0, 3)
      .every((mount) => fs.realpathSync(mount.hostPath) === mount.hostPath),
  ).toBe(true);
  expect(mounts[2]).toEqual({
    ...own,
    hostPath: fs.realpathSync(own.hostPath),
  });
  expect(mounts.slice(3)).toEqual([
    {
      hostPath: fs.realpathSync(empty),
      containerPath: '/workspace/project/data/ipc/.claimed-v1',
      readonly: true,
    },
    {
      hostPath: fs.realpathSync(empty),
      containerPath: '/workspace/extra/alias/data/ipc/.claimed-v1',
      readonly: true,
    },
  ]);
  expect(fs.readdirSync(empty)).toEqual([]);
  expect(fs.statSync(empty).mode & 0o077).toBe(0);
});
it('rejects an approved extra RW ancestor, including a symlink alias, before returning arguments', () => {
  const alias = path.join(fixture.root, 'alias');
  fs.symlinkSync(project, alias);
  expect(() =>
    protectIpcClaimMounts([
      parent(),
      {
        hostPath: alias,
        containerPath: '/workspace/extra/rw',
        readonly: false,
      },
    ]),
  ).toThrow('Read-write parent');
});
it.each(['root', 'descendant', 'mask'] as const)(
  'rejects a direct %s view of reserved storage',
  (kind) => {
    protectIpcClaimMounts([]);
    const file = path.join(claims, 'fixed');
    fs.writeFileSync(file, 'fixed');
    const source = kind === 'root' ? claims : kind === 'mask' ? empty : file;
    expect(() =>
      protectIpcClaimMounts([
        {
          hostPath: source,
          containerPath: '/workspace/extra/direct',
          readonly: true,
        },
      ]),
    ).toThrow('Direct or descendant');
  },
);
it('refuses a nested target that would pierce an otherwise correct empty mask', () => {
  const publicDir = path.join(fixture.root, 'public');
  fs.mkdirSync(publicDir);
  expect(() =>
    protectIpcClaimMounts([
      parent(),
      {
        hostPath: publicDir,
        containerPath: '/workspace/project/data/ipc/.claimed-v1/pierce',
        readonly: true,
      },
    ]),
  ).toThrow('nested mount pierces');
});
it.each(['symlink', 'nonempty'] as const)(
  'preserves unsafe %s mask state without silently repairing it',
  (kind) => {
    protectIpcClaimMounts([]);
    if (kind === 'symlink') {
      fs.rmdirSync(empty);
      const foreign = path.join(fixture.root, 'foreign');
      fs.mkdirSync(foreign);
      fs.symlinkSync(foreign, empty);
    } else fs.writeFileSync(path.join(empty, 'unknown'), 'retained');
    expect(() => protectIpcClaimMounts([parent()])).toThrow(
      /IPC mask directory/,
    );
    if (kind === 'symlink')
      expect(fs.lstatSync(empty).isSymbolicLink()).toBe(true);
    else
      expect(fs.readFileSync(path.join(empty, 'unknown'), 'utf8')).toBe(
        'retained',
      );
  },
);
