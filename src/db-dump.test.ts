import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { runDumpPlan, STATE_TABLES } from './db-dump.js';

let tmpRoot: string;
let dbPath: string;
let outDir: string;

beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'dump-state-tables-'));
  dbPath = path.join(tmpRoot, 'messages.db');
  outDir = path.join(tmpRoot, 'state');
});

afterEach(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

interface FixtureTable {
  name: string;
  ddl: string;
  rows: Record<string, string | number | null>[];
}

function makeDb(tables: FixtureTable[]): void {
  const db = new Database(dbPath);
  for (const t of tables) {
    db.exec(t.ddl);
    if (t.rows.length === 0) continue;
    const cols = Object.keys(t.rows[0]);
    const stmt = db.prepare(
      `INSERT INTO ${t.name} (${cols.join(',')}) VALUES (${cols.map(() => '?').join(',')})`,
    );
    for (const row of t.rows) stmt.run(...cols.map((c) => row[c]));
  }
  db.close();
}

describe('runDumpPlan', () => {
  it('dumps tables that exist; skips ones that do not', () => {
    makeDb([
      {
        name: 'orders',
        ddl: 'CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id TEXT, sku TEXT)',
        rows: [
          { id: 1, user_id: 'u1', sku: 'sku-a' },
          { id: 2, user_id: 'u2', sku: 'sku-b' },
        ],
      },
    ]);
    const report = runDumpPlan({
      dbPath,
      outDir,
      tables: ['orders', 'scheduled_reminders'],
    });
    expect(report.dumped).toEqual(['orders']);
    expect(report.skipped).toEqual(['scheduled_reminders']);
    const sql = fs.readFileSync(path.join(outDir, 'orders.sql'), 'utf8');
    expect(sql).toContain('CREATE TABLE orders');
    expect(sql).toContain("INSERT INTO orders VALUES(1,'u1','sku-a')");
    expect(sql).toContain("INSERT INTO orders VALUES(2,'u2','sku-b')");
    expect(fs.existsSync(path.join(outDir, 'scheduled_reminders.sql'))).toBe(false);
  });

  it('round-trips: dumped SQL replays into a fresh DB and produces matching rows', () => {
    makeDb([
      {
        name: 'pending_decisions',
        ddl: 'CREATE TABLE pending_decisions (id TEXT PRIMARY KEY, kind TEXT, payload TEXT)',
        rows: [
          { id: 'd-1', kind: 'cleanup', payload: '{"a":1}' },
          { id: 'd-2', kind: 'undated', payload: '{"b":2}' },
        ],
      },
    ]);
    runDumpPlan({ dbPath, outDir, tables: ['pending_decisions'] });
    const sql = fs.readFileSync(path.join(outDir, 'pending_decisions.sql'), 'utf8');

    const replayPath = path.join(tmpRoot, 'replay.db');
    const replayDb = new Database(replayPath);
    replayDb.exec(sql);
    const rows = replayDb
      .prepare('SELECT id, kind, payload FROM pending_decisions ORDER BY id')
      .all();
    replayDb.close();
    expect(rows).toEqual([
      { id: 'd-1', kind: 'cleanup', payload: '{"a":1}' },
      { id: 'd-2', kind: 'undated', payload: '{"b":2}' },
    ]);
  });

  it('atomic-writes: leaves no .tmp residue on success', () => {
    makeDb([
      {
        name: 'tz_state',
        ddl: 'CREATE TABLE tz_state (id INTEGER PRIMARY KEY, tz TEXT)',
        rows: [{ id: 1, tz: 'UTC' }],
      },
    ]);
    runDumpPlan({ dbPath, outDir, tables: ['tz_state'] });
    const entries = fs.readdirSync(outDir);
    expect(entries).toContain('tz_state.sql');
    expect(entries.some((e) => e.endsWith('.tmp'))).toBe(false);
  });

  it('creates the output directory if missing', () => {
    makeDb([
      {
        name: 'tz_state',
        ddl: 'CREATE TABLE tz_state (id INTEGER PRIMARY KEY)',
        rows: [{ id: 1 }],
      },
    ]);
    const nestedOut = path.join(tmpRoot, 'a', 'b', 'c');
    runDumpPlan({ dbPath, outDir: nestedOut, tables: ['tz_state'] });
    expect(fs.existsSync(path.join(nestedOut, 'tz_state.sql'))).toBe(true);
  });

  it('overwrites a stale dump file with the latest contents', () => {
    makeDb([
      {
        name: 'tz_state',
        ddl: 'CREATE TABLE tz_state (id INTEGER PRIMARY KEY, tz TEXT)',
        rows: [{ id: 1, tz: 'UTC' }],
      },
    ]);
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(path.join(outDir, 'tz_state.sql'), 'STALE CONTENT\n');
    runDumpPlan({ dbPath, outDir, tables: ['tz_state'] });
    const sql = fs.readFileSync(path.join(outDir, 'tz_state.sql'), 'utf8');
    expect(sql).not.toContain('STALE CONTENT');
    expect(sql).toContain('CREATE TABLE tz_state');
  });

  it('throws actionable error when db file is missing', () => {
    expect(() =>
      runDumpPlan({ dbPath: path.join(tmpRoot, 'nonexistent.db'), outDir }),
    ).toThrow(/db not found at/);
  });
});

describe('STATE_TABLES', () => {
  it('includes the post-migration tables called out in #398', () => {
    // Spot-check the irreplaceable surface from the issue body.
    const required = [
      'orders',
      'email_feedback',
      'scheduled_reminders',
      'email_state',
      'email_seen_ids',
      'resumable_cycles',
      'trusted_session_singleton',
      'pending_cleanup_items',
      'pending_decisions',
      'pending_undated_tasks',
      'calendar_snapshots',
      'calendar_events',
      'phase_completions',
      'tz_state',
      'follow_me_tasks',
      'scheduled_tasks',
      'task_run_logs',
    ];
    for (const t of required) expect(STATE_TABLES).toContain(t);
  });

  it('excludes bulk-cache tables that are recoverable from their upstream source', () => {
    // `messages`, `chats`, `reactions` are recoverable from Telegram;
    // `sessions` is an ephemeral SDK cache; `smart_home_events` is
    // recoverable from Hubitat. They're also large enough that
    // buffered `.dump` would OOM and produce poor diffs.
    const excluded = ['messages', 'chats', 'reactions', 'sessions', 'smart_home_events'];
    for (const t of excluded) expect(STATE_TABLES).not.toContain(t);
  });

  it('has no duplicate entries', () => {
    expect(new Set(STATE_TABLES).size).toBe(STATE_TABLES.length);
  });
});

describe('table name validation', () => {
  it('rejects table names containing metacharacters', () => {
    expect(() =>
      runDumpPlan({ dbPath, outDir, tables: ["orders'; DROP TABLE x; --"] }),
    ).toThrow(/invalid table name/);
    expect(() => runDumpPlan({ dbPath, outDir, tables: ['a b'] })).toThrow(/invalid table name/);
    expect(() => runDumpPlan({ dbPath, outDir, tables: ['1leading_digit'] })).toThrow(
      /invalid table name/,
    );
    expect(() => runDumpPlan({ dbPath, outDir, tables: [''] })).toThrow(/invalid table name/);
  });
});
