import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';

import {
  hashDirectoryTree,
  getPluginRegistryHash,
} from './plugin-content-hash.js';
import { TILE_OWNER } from './config.js';

describe('hashDirectoryTree (#710)', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'plugin-hash-test-'));
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  /** Build a file tree under `root` from a { relPath: content } map. */
  function writeTree(root: string, files: Record<string, string>): void {
    for (const [rel, content] of Object.entries(files)) {
      const abs = path.join(root, rel);
      fs.mkdirSync(path.dirname(abs), { recursive: true });
      fs.writeFileSync(abs, content);
    }
  }

  it('returns null for a directory that does not exist', () => {
    expect(hashDirectoryTree(path.join(tempDir, 'nope'))).toBeNull();
  });

  it('returns identical hashes for identical trees at different roots', () => {
    const tree = {
      'tile-a/skills/foo/SKILL.md': '# Foo\n',
      'tile-a/rules/bar.md': '# Bar\n',
      'tile-b/tile.json': '{"version":"1.0.0"}',
    };
    const rootA = path.join(tempDir, 'a');
    const rootB = path.join(tempDir, 'b');
    writeTree(rootA, tree);
    writeTree(rootB, tree);

    const hashA = hashDirectoryTree(rootA);
    expect(hashA).toMatch(/^[0-9a-f]{64}$/);
    expect(hashDirectoryTree(rootB)).toBe(hashA);
  });

  it('changes when file content changes', () => {
    const root = path.join(tempDir, 'r');
    writeTree(root, { 'skills/foo/SKILL.md': 'old instructions' });
    const before = hashDirectoryTree(root);

    fs.writeFileSync(
      path.join(root, 'skills/foo/SKILL.md'),
      'fixed instructions',
    );
    expect(hashDirectoryTree(root)).not.toBe(before);
  });

  it('changes when a file is renamed, even with identical content', () => {
    const root = path.join(tempDir, 'r');
    writeTree(root, { 'rules/old-name.md': 'same content' });
    const before = hashDirectoryTree(root);

    fs.renameSync(
      path.join(root, 'rules/old-name.md'),
      path.join(root, 'rules/new-name.md'),
    );
    expect(hashDirectoryTree(root)).not.toBe(before);
  });

  it('changes when a file is added or removed', () => {
    const root = path.join(tempDir, 'r');
    writeTree(root, { 'skills/foo/SKILL.md': 'a' });
    const before = hashDirectoryTree(root);

    writeTree(root, { 'skills/foo/scripts/apply.py': 'print(1)' });
    const withScript = hashDirectoryTree(root);
    expect(withScript).not.toBe(before);

    fs.rmSync(path.join(root, 'skills/foo/scripts'), { recursive: true });
    expect(hashDirectoryTree(root)).toBe(before);
  });

  it('is insensitive to file creation order', () => {
    const rootA = path.join(tempDir, 'a');
    writeTree(rootA, { 'x.md': '1' });
    writeTree(rootA, { 'y.md': '2' });

    const rootB = path.join(tempDir, 'b');
    writeTree(rootB, { 'y.md': '2' });
    writeTree(rootB, { 'x.md': '1' });

    expect(hashDirectoryTree(rootA)).toBe(hashDirectoryTree(rootB));
  });

  it('keeps (path, content) pairs unambiguous across file boundaries', () => {
    // Without separators, {a: 'bc'} and {ab: 'c'} could digest
    // identically — the concatenated byte stream is the same.
    const rootA = path.join(tempDir, 'a');
    writeTree(rootA, { a: 'bc' });
    const rootB = path.join(tempDir, 'b');
    writeTree(rootB, { ab: 'c' });

    expect(hashDirectoryTree(rootA)).not.toBe(hashDirectoryTree(rootB));
  });
});

describe('getPluginRegistryHash (#710)', () => {
  let tempDir: string;
  let originalCwd: string;

  beforeEach(() => {
    originalCwd = process.cwd();
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'plugin-registry-test-'));
    process.chdir(tempDir);
  });

  afterEach(() => {
    process.chdir(originalCwd);
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  it('returns null when no tessl workspace exists under cwd', () => {
    expect(getPluginRegistryHash()).toBeNull();
  });

  it('hashes the registry tree resolved via getRegistryTilesDir', () => {
    const registryDir = path.join(
      tempDir,
      'tessl-workspace',
      '.tessl',
      'plugins',
      TILE_OWNER,
    );
    fs.mkdirSync(path.join(registryDir, 'some-tile', 'skills'), {
      recursive: true,
    });
    fs.writeFileSync(
      path.join(registryDir, 'some-tile', 'skills', 'SKILL.md'),
      '# Skill\n',
    );

    expect(getPluginRegistryHash()).toBe(hashDirectoryTree(registryDir));
    expect(getPluginRegistryHash()).toMatch(/^[0-9a-f]{64}$/);
  });
});
