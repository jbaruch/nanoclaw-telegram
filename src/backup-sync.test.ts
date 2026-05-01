import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { syncBackupRepo } from './backup-sync.js';

let tmpRoot: string;
let groupDir: string;
let backupDir: string;
let dbPath: string;

beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'backup-sync-'));
  groupDir = path.join(tmpRoot, 'groups', 'telegram_test');
  backupDir = path.join(groupDir, 'backup-repo');
  dbPath = path.join(tmpRoot, 'store', 'messages.db');
  fs.mkdirSync(groupDir, { recursive: true });
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

describe('syncBackupRepo', () => {
  it('copies MEMORY.md and daily_discoveries.md when present', () => {
    fs.writeFileSync(path.join(groupDir, 'MEMORY.md'), 'live memory\n');
    fs.writeFileSync(path.join(groupDir, 'daily_discoveries.md'), 'today: x\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupDir, backupDir, dbPath });

    expect(result.copied).toContain('MEMORY.md');
    expect(result.copied).toContain('daily_discoveries.md');
    expect(fs.readFileSync(path.join(backupDir, 'MEMORY.md'), 'utf8')).toBe(
      'live memory\n',
    );
    expect(
      fs.readFileSync(path.join(backupDir, 'daily_discoveries.md'), 'utf8'),
    ).toBe('today: x\n');
  });

  it('overwrites stale copies in the destination', () => {
    fs.writeFileSync(path.join(groupDir, 'MEMORY.md'), 'fresh\n');
    fs.writeFileSync(path.join(backupDir, 'MEMORY.md'), 'STALE\n');
    makeDbWithOrders();

    syncBackupRepo({ groupDir, backupDir, dbPath });

    expect(fs.readFileSync(path.join(backupDir, 'MEMORY.md'), 'utf8')).toBe(
      'fresh\n',
    );
  });

  it('skips single files that do not exist in the source — no error, no record', () => {
    // Only daily_discoveries.md present; MEMORY.md missing
    fs.writeFileSync(path.join(groupDir, 'daily_discoveries.md'), 'd\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupDir, backupDir, dbPath });

    expect(result.copied).toContain('daily_discoveries.md');
    expect(result.copied).not.toContain('MEMORY.md');
    expect(fs.existsSync(path.join(backupDir, 'MEMORY.md'))).toBe(false);
  });

  it('mirrors memory/: copies new files, removes orphans in the dest', () => {
    const srcMemory = path.join(groupDir, 'memory');
    const destMemory = path.join(backupDir, 'memory');
    fs.mkdirSync(srcMemory);
    fs.mkdirSync(destMemory);
    fs.writeFileSync(path.join(srcMemory, '2026-04-30.md'), 'today\n');
    fs.writeFileSync(path.join(srcMemory, '2026-05-01.md'), 'tomorrow\n');
    // Orphan in dest — should be removed by the mirror
    fs.writeFileSync(path.join(destMemory, '2026-04-13.md'), 'pre-freeze\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupDir, backupDir, dbPath });

    expect(result.copied).toEqual(
      expect.arrayContaining(['memory/2026-04-30.md', 'memory/2026-05-01.md']),
    );
    expect(result.removed).toEqual(['memory/2026-04-13.md']);
    expect(fs.existsSync(path.join(destMemory, '2026-04-30.md'))).toBe(true);
    expect(fs.existsSync(path.join(destMemory, '2026-04-13.md'))).toBe(false);
  });

  it('preserves nested subdirectories under memory/', () => {
    const srcMemory = path.join(groupDir, 'memory', 'archive', '2026');
    fs.mkdirSync(srcMemory, { recursive: true });
    fs.writeFileSync(path.join(srcMemory, 'q1.md'), 'q1\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupDir, backupDir, dbPath });

    expect(result.copied).toContain('memory/archive/2026/q1.md');
    expect(
      fs.readFileSync(
        path.join(backupDir, 'memory', 'archive', '2026', 'q1.md'),
        'utf8',
      ),
    ).toBe('q1\n');
  });

  it('leaves dest memory/ untouched when source memory/ is missing', () => {
    const destMemory = path.join(backupDir, 'memory');
    fs.mkdirSync(destMemory);
    fs.writeFileSync(path.join(destMemory, 'historical.md'), 'keep\n');
    makeDbWithOrders();

    const result = syncBackupRepo({ groupDir, backupDir, dbPath });

    expect(result.copied.filter((p) => p.startsWith('memory/'))).toEqual([]);
    expect(result.removed).toEqual([]);
    expect(
      fs.readFileSync(path.join(destMemory, 'historical.md'), 'utf8'),
    ).toBe('keep\n');
  });

  it('runs dump plan and reports dumped + skipped tables', () => {
    makeDbWithOrders();

    const result = syncBackupRepo({ groupDir, backupDir, dbPath });

    expect(result.dumped).toContain('orders');
    // Other STATE_TABLES tables that don't exist in this fixture DB
    expect(result.skipped.length).toBeGreaterThan(0);
    expect(fs.existsSync(path.join(backupDir, 'state', 'orders.sql'))).toBe(
      true,
    );
  });

  it('throws actionable error when group dir is missing', () => {
    const missingGroup = path.join(tmpRoot, 'groups', 'nonexistent');
    expect(() =>
      syncBackupRepo({ groupDir: missingGroup, backupDir, dbPath }),
    ).toThrow(/group directory not found/);
  });

  it('throws actionable error when backup-repo is missing', () => {
    const missingBackup = path.join(groupDir, 'backup-repo-missing');
    makeDbWithOrders();
    expect(() =>
      syncBackupRepo({ groupDir, backupDir: missingBackup, dbPath }),
    ).toThrow(/backup-repo not found/);
  });
});
