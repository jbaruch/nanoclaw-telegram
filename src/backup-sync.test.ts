import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { syncBackupRepo } from './backup-sync.js';

let tmpRoot: string;
let groupsRoot: string;
let backupDir: string;
let dbPath: string;

beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'backup-sync-'));
  groupsRoot = path.join(tmpRoot, 'groups');
  backupDir = path.join(groupsRoot, 'telegram_test', 'backup-repo');
  dbPath = path.join(tmpRoot, 'store', 'messages.db');
  fs.mkdirSync(path.join(groupsRoot, 'global'), { recursive: true });
  fs.mkdirSync(path.join(groupsRoot, 'telegram_test'), { recursive: true });
  fs.mkdirSync(backupDir, { recursive: true });
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
});

afterEach(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

function makeDbWithOrders(): void {
  const db = new Database(dbPath);
  db.exec(
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, sku TEXT); INSERT INTO orders VALUES (1, 'sku-a');",
  );
  db.close();
}

function writeFile(p: string, content: string): void {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, content);
}

describe('syncBackupRepo', () => {
  it('mirrors groups/global/ → backup-repo/global/', () => {
    writeFile(path.join(groupsRoot, 'global', 'SOUL.md'), 'live soul\n');
    writeFile(
      path.join(groupsRoot, 'global', 'prompts', 'morning.md'),
      'good morning\n',
    );
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.copied).toContain('global/SOUL.md');
    expect(result.copied).toContain('global/prompts/morning.md');
    expect(
      fs.readFileSync(path.join(backupDir, 'global', 'SOUL.md'), 'utf8'),
    ).toBe('live soul\n');
    expect(
      fs.readFileSync(
        path.join(backupDir, 'global', 'prompts', 'morning.md'),
        'utf8',
      ),
    ).toBe('good morning\n');
  });

  it('mirrors every non-hidden group dir → backup-repo/groups/<name>/', () => {
    writeFile(
      path.join(groupsRoot, 'telegram_test', 'MEMORY.md'),
      'test memory\n',
    );
    writeFile(
      path.join(groupsRoot, 'telegram_other', 'cfp-state.json'),
      '{"x":1}\n',
    );
    writeFile(path.join(groupsRoot, 'main', 'ADMIN.md'), 'admin\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.copied).toContain('groups/telegram_test/MEMORY.md');
    expect(result.copied).toContain('groups/telegram_other/cfp-state.json');
    expect(result.copied).toContain('groups/main/ADMIN.md');
  });

  it('overwrites stale copies in the destination', () => {
    writeFile(path.join(groupsRoot, 'global', 'SOUL.md'), 'fresh\n');
    writeFile(path.join(backupDir, 'global', 'SOUL.md'), 'STALE\n');
    makeDbWithOrders();

    syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(
      fs.readFileSync(path.join(backupDir, 'global', 'SOUL.md'), 'utf8'),
    ).toBe('fresh\n');
  });

  it('sweeps orphan files under global/ and groups/', () => {
    // No source files — dest has stale orphans
    writeFile(path.join(backupDir, 'global', 'old.md'), 'orphan\n');
    writeFile(
      path.join(backupDir, 'groups', 'telegram_gone', 'memo.md'),
      'orphan\n',
    );
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.removed).toContain('global/old.md');
    expect(result.removed).toContain('groups/telegram_gone/memo.md');
    expect(fs.existsSync(path.join(backupDir, 'global', 'old.md'))).toBe(false);
    expect(
      fs.existsSync(path.join(backupDir, 'groups', 'telegram_gone', 'memo.md')),
    ).toBe(false);
  });

  it('does not recurse into backup-repo itself (recursion guard)', () => {
    writeFile(path.join(groupsRoot, 'global', 'SOUL.md'), 'soul\n');
    // backupDir lives at groups/telegram_test/backup-repo — the walk
    // of telegram_test must not enter backup-repo or it would mirror
    // itself into itself.
    writeFile(path.join(backupDir, 'preexisting.md'), 'must not be copied\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    // No path under `groups/telegram_test/backup-repo/...` should
    // appear in result.copied.
    expect(result.copied.some((p) => p.includes('backup-repo'))).toBe(false);
    // The preexisting file inside backup-repo IS at the root of the
    // managed prefix (`<root>/preexisting.md`, no prefix match for
    // `global/` or `groups/`) — sweep doesn't touch it because it's
    // outside managed prefixes.
    expect(fs.existsSync(path.join(backupDir, 'preexisting.md'))).toBe(true);
  });

  it('denies tessl/claude dirs at any depth', () => {
    writeFile(
      path.join(groupsRoot, 'global', '.tessl', 'tiles', 'some', 'rule.md'),
      'tessl content\n',
    );
    writeFile(
      path.join(groupsRoot, 'main', '.claude', 'skills', 'foo', 'SKILL.md'),
      'claude content\n',
    );
    writeFile(path.join(groupsRoot, 'main', 'MEMORY.md'), 'real content\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.copied.some((p) => p.includes('.tessl'))).toBe(false);
    expect(result.copied.some((p) => p.includes('.claude'))).toBe(false);
    expect(result.copied).toContain('groups/main/MEMORY.md');
  });

  it('denies regenerable runtime dirs: node_modules, dist, logs, tmp, conversations, .checkpoints', () => {
    for (const denied of [
      'node_modules',
      'dist',
      'logs',
      'tmp',
      'conversations',
      '.checkpoints',
    ]) {
      writeFile(
        path.join(groupsRoot, 'main', denied, 'thing.md'),
        'should not be copied\n',
      );
    }
    writeFile(path.join(groupsRoot, 'main', 'keep.md'), 'keep me\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.copied).toContain('groups/main/keep.md');
    for (const denied of [
      'node_modules',
      'dist',
      'logs',
      'tmp',
      'conversations',
      '.checkpoints',
    ]) {
      expect(result.copied.some((p) => p.includes(denied))).toBe(false);
    }
  });

  it('denies .archive* directories (.archives, .archive-...)', () => {
    writeFile(
      path.join(groupsRoot, 'main', '.archives', 'old.md'),
      'archived\n',
    );
    writeFile(
      path.join(groupsRoot, 'main', '.archive-heartbeat-2026-04-17', 'x.md'),
      'archived\n',
    );
    writeFile(path.join(groupsRoot, 'main', 'live.md'), 'live\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.copied).toContain('groups/main/live.md');
    expect(result.copied.some((p) => p.includes('.archive'))).toBe(false);
  });

  it('denies .bak-YYYY, .migrated-YYYY, .lock, .tmp-*, scripts.new.*, scripts.version.* files', () => {
    writeFile(path.join(groupsRoot, 'main', 'cfp-state.json'), '{}');
    writeFile(
      path.join(groupsRoot, 'main', 'cfp-state.json.bak-2026-04-27'),
      '{}',
    );
    writeFile(
      path.join(groupsRoot, 'main', 'cfp-state.json.bak.2026-05-08-blocklist'),
      '{}',
    );
    writeFile(
      path.join(groupsRoot, 'main', 'calendar-state.json.migrated-2026-05-01'),
      '{}',
    );
    writeFile(
      path.join(groupsRoot, 'main', 'backup-empty-streak.json.lock'),
      '',
    );
    writeFile(path.join(groupsRoot, 'main', '.tmp-something'), 'tmp\n');
    writeFile(
      path.join(groupsRoot, 'main', 'scripts.new.1.1777435107685.66y7r1'),
      'editor temp\n',
    );
    writeFile(
      path.join(groupsRoot, 'main', 'scripts.version.1778524227198.1.yd18gu65'),
      'editor temp\n',
    );
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.copied).toContain('groups/main/cfp-state.json');
    for (const denied of [
      'cfp-state.json.bak-2026-04-27',
      'cfp-state.json.bak.2026-05-08-blocklist',
      'calendar-state.json.migrated-2026-05-01',
      'backup-empty-streak.json.lock',
      '.tmp-something',
      'scripts.new.1.1777435107685.66y7r1',
      'scripts.version.1778524227198.1.yd18gu65',
    ]) {
      expect(result.copied.some((p) => p.endsWith(denied))).toBe(false);
    }
  });

  it('skips hidden top-level group dirs (.archives, .DS_Store)', () => {
    writeFile(path.join(groupsRoot, '.archives', 'MEMORY.md'), 'archived\n');
    writeFile(path.join(groupsRoot, '.DS_Store'), 'fs metadata\n');
    writeFile(path.join(groupsRoot, 'main', 'real.md'), 'real\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.copied.some((p) => p.startsWith('groups/.archives/'))).toBe(
      false,
    );
    expect(result.copied).toContain('groups/main/real.md');
  });

  it('sweeps newly-denied content already in dest (denylist applies to dest scan with denylist DISABLED so orphans get cleaned)', () => {
    // Pre-existing dest file that would now match the denylist: a stale
    // .bak-2026-* that was mirrored before this rule shipped.
    writeFile(
      path.join(backupDir, 'groups', 'main', 'cfp-state.json.bak-2026-04-27'),
      '{}',
    );
    // Source has live content; the bak file is NOT in the source.
    writeFile(path.join(groupsRoot, 'main', 'cfp-state.json'), '{"v":1}');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.removed).toContain(
      'groups/main/cfp-state.json.bak-2026-04-27',
    );
    expect(
      fs.existsSync(
        path.join(backupDir, 'groups', 'main', 'cfp-state.json.bak-2026-04-27'),
      ),
    ).toBe(false);
    expect(result.copied).toContain('groups/main/cfp-state.json');
  });

  it('removes legacy root-level paths on first run, idempotent on second', () => {
    writeFile(path.join(backupDir, 'MEMORY.md'), 'legacy root\n');
    writeFile(path.join(backupDir, 'blog-notes.md'), 'legacy\n');
    writeFile(
      path.join(backupDir, 'memory', 'archive', '2026', 'q1.md'),
      'legacy mirror\n',
    );
    writeFile(path.join(backupDir, 'skills', 'old', 'SKILL.md'), 'legacy\n');
    writeFile(
      path.join(backupDir, 'trusted', 'persona.md'),
      'legacy trusted\n',
    );
    writeFile(path.join(groupsRoot, 'main', 'real.md'), 'real\n');
    makeDbWithOrders();

    const result1 = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result1.removed).toContain('MEMORY.md');
    expect(result1.removed).toContain('blog-notes.md');
    expect(result1.removed).toContain('memory/archive/2026/q1.md');
    expect(result1.removed).toContain('skills/old/SKILL.md');
    expect(result1.removed).toContain('trusted/persona.md');
    expect(fs.existsSync(path.join(backupDir, 'MEMORY.md'))).toBe(false);
    expect(fs.existsSync(path.join(backupDir, 'memory'))).toBe(false);

    // Second run: nothing to remove (idempotent)
    const result2 = syncBackupRepo({ groupsRoot, backupDir, dbPath });
    expect(result2.removed).toEqual([]);
  });

  it('runs dump plan and reports dumped + skipped tables', () => {
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.dumped).toContain('orders');
    expect(result.skipped.length).toBeGreaterThan(0);
    expect(fs.existsSync(path.join(backupDir, 'state', 'orders.sql'))).toBe(
      true,
    );
  });

  it('preserves nested subdirectories under any mirrored tree', () => {
    writeFile(
      path.join(groupsRoot, 'main', 'memory', 'archive', '2026', 'q1.md'),
      'q1\n',
    );
    makeDbWithOrders();

    const result = syncBackupRepo({ groupsRoot, backupDir, dbPath });

    expect(result.copied).toContain('groups/main/memory/archive/2026/q1.md');
    expect(
      fs.readFileSync(
        path.join(
          backupDir,
          'groups',
          'main',
          'memory',
          'archive',
          '2026',
          'q1.md',
        ),
        'utf8',
      ),
    ).toBe('q1\n');
  });

  it('throws actionable error when groups root is missing', () => {
    fs.rmSync(groupsRoot, { recursive: true, force: true });
    expect(() => syncBackupRepo({ groupsRoot, backupDir, dbPath })).toThrow(
      /groups root not found/,
    );
  });

  it('throws actionable error when backup-repo is missing', () => {
    const missingBackup = path.join(groupsRoot, 'telegram_test', 'no-such');
    makeDbWithOrders();
    expect(() =>
      syncBackupRepo({ groupsRoot, backupDir: missingBackup, dbPath }),
    ).toThrow(/backup-repo not found/);
  });
});
