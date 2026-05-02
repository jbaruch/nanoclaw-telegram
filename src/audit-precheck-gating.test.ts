import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { runAuditSnapshot } from './audit-precheck-gating.js';

let tmpRoot: string;
let dbPath: string;

beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'audit-precheck-'));
  dbPath = path.join(tmpRoot, 'messages.db');
});

afterEach(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

function makeFixtureDb(): void {
  const db = new Database(dbPath);
  db.exec(`
    CREATE TABLE scheduled_tasks (
      id TEXT PRIMARY KEY,
      group_folder TEXT NOT NULL,
      chat_jid TEXT NOT NULL,
      prompt TEXT NOT NULL,
      schedule_type TEXT NOT NULL,
      schedule_value TEXT NOT NULL,
      next_run TEXT,
      last_run TEXT,
      last_result TEXT,
      status TEXT DEFAULT 'active',
      created_at TEXT NOT NULL,
      created_by_role TEXT NOT NULL DEFAULT 'owner',
      script TEXT
    );
    CREATE TABLE task_run_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      task_id TEXT NOT NULL,
      run_at TEXT NOT NULL,
      duration_ms INTEGER NOT NULL,
      status TEXT NOT NULL,
      result TEXT,
      error TEXT
    );
  `);
  db.close();
}

function insertTask(
  id: string,
  group: string,
  // Tests insert 'once' rows too to verify the SQL filter excludes
  // them — string here, not the production union, because exclusion
  // is the property under test.
  schedule_type: string,
  schedule_value: string,
  status: string,
  script: string | null,
): void {
  const db = new Database(dbPath);
  db.prepare(
    `INSERT INTO scheduled_tasks (id, group_folder, chat_jid, prompt,
       schedule_type, schedule_value, status, created_at, script)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    id,
    group,
    'tg:' + group,
    'noop',
    schedule_type,
    schedule_value,
    status,
    '2026-04-01T00:00:00Z',
    script,
  );
  db.close();
}

function insertRun(
  taskId: string,
  runAtIso: string,
  durationMs: number,
  opts: { status?: string; result?: string | null } = {},
): void {
  const db = new Database(dbPath);
  db.prepare(
    `INSERT INTO task_run_logs (task_id, run_at, duration_ms, status, result)
     VALUES (?, ?, ?, ?, ?)`,
  ).run(
    taskId,
    runAtIso,
    durationMs,
    opts.status ?? 'success',
    opts.result === undefined ? null : opts.result,
  );
  db.close();
}

describe('runAuditSnapshot', () => {
  it('selects only active recurring tasks; omits cancelled and once', () => {
    makeFixtureDb();
    insertTask('t-active-cron', 'g1', 'cron', '0 7 * * *', 'active', null);
    insertTask(
      't-active-int',
      'g1',
      'interval',
      '30m',
      'active',
      'precheck.py',
    );
    insertTask('t-cancelled', 'g1', 'cron', '0 8 * * *', 'cancelled', null);
    insertTask('t-once', 'g1', 'once', '2026-05-01T00:00:00Z', 'active', null);

    const snap = runAuditSnapshot({
      dbPath,
      now: new Date('2026-05-01T00:00:00Z'),
    });

    const ids = snap.tasks.map((t) => t.task_id).sort();
    expect(ids).toEqual(['t-active-cron', 't-active-int']);
  });

  it('reports has_precheck and surfaces the script verbatim', () => {
    makeFixtureDb();
    insertTask(
      't-with',
      'g1',
      'cron',
      '*/5 * * * *',
      'active',
      'python3 foo.py',
    );
    insertTask('t-without', 'g1', 'cron', '0 7 * * *', 'active', null);

    const snap = runAuditSnapshot({
      dbPath,
      now: new Date('2026-05-01T00:00:00Z'),
    });

    const withPrecheck = snap.tasks.find((t) => t.task_id === 't-with')!;
    expect(withPrecheck.has_precheck).toBe(true);
    expect(withPrecheck.script).toBe('python3 foo.py');

    const withoutPrecheck = snap.tasks.find((t) => t.task_id === 't-without')!;
    expect(withoutPrecheck.has_precheck).toBe(false);
    expect(withoutPrecheck.script).toBeNull();
  });

  it('aggregates fires + duration stats over the window only', () => {
    makeFixtureDb();
    insertTask('t1', 'g1', 'cron', '0 7 * * *', 'active', null);

    const now = new Date('2026-05-01T00:00:00Z');
    // In window: 3 fires
    insertRun('t1', '2026-04-15T07:00:00Z', 5_000); // gated_likely (< 10s)
    insertRun('t1', '2026-04-20T07:00:00Z', 60_000);
    insertRun('t1', '2026-04-25T07:00:00Z', 120_000);
    // Out of window (older than 90 days): excluded
    insertRun('t1', '2026-01-01T07:00:00Z', 999_000);

    const snap = runAuditSnapshot({ dbPath, windowDays: 90, now });

    const t1 = snap.tasks.find((t) => t.task_id === 't1')!;
    expect(t1.fires).toBe(3);
    expect(t1.gated_likely).toBe(1);
    expect(t1.avg_duration_s).toBeCloseTo((5 + 60 + 120) / 3);
    expect(t1.min_duration_s).toBe(5);
    expect(t1.max_duration_s).toBe(120);
  });

  it('reports null durations + zero fires when the window has no runs', () => {
    makeFixtureDb();
    insertTask('t-quiet', 'g1', 'cron', '0 9 * * 5', 'active', null);

    const snap = runAuditSnapshot({
      dbPath,
      now: new Date('2026-05-01T00:00:00Z'),
    });

    const t = snap.tasks.find((tt) => tt.task_id === 't-quiet')!;
    expect(t.fires).toBe(0);
    expect(t.gated_likely).toBe(0);
    expect(t.avg_duration_s).toBeNull();
    expect(t.min_duration_s).toBeNull();
    expect(t.max_duration_s).toBeNull();
  });

  it('window respects --days override', () => {
    makeFixtureDb();
    insertTask('t1', 'g1', 'cron', '0 7 * * *', 'active', null);

    const now = new Date('2026-05-01T00:00:00Z');
    // In a 30-day window from 2026-05-01: starts 2026-04-01
    insertRun('t1', '2026-04-15T07:00:00Z', 60_000); // in window
    insertRun('t1', '2026-03-01T07:00:00Z', 60_000); // outside 30d, inside 90d

    const snap30 = runAuditSnapshot({ dbPath, windowDays: 30, now });
    const snap90 = runAuditSnapshot({ dbPath, windowDays: 90, now });

    expect(snap30.tasks[0].fires).toBe(1);
    expect(snap90.tasks[0].fires).toBe(2);
    expect(snap30.window_days).toBe(30);
    expect(snap90.window_days).toBe(90);
  });

  it('emits ISO-8601 UTC timestamps with second precision', () => {
    makeFixtureDb();
    const now = new Date('2026-05-01T12:34:56Z');
    const snap = runAuditSnapshot({ dbPath, now });

    expect(snap.snapshot_at).toBe('2026-05-01T12:34:56Z');
    expect(snap.window_end).toBe('2026-05-01T12:34:56Z');
    // 90 days back from 2026-05-01T12:34:56Z = 2026-01-31T12:34:56Z
    expect(snap.window_start).toBe('2026-01-31T12:34:56Z');
  });

  it('throws actionable error when db is missing', () => {
    expect(() =>
      runAuditSnapshot({
        dbPath: path.join(tmpRoot, 'nonexistent.db'),
      }),
    ).toThrow(/db not found at/);
  });

  it('throws on non-positive windowDays', () => {
    makeFixtureDb();
    expect(() => runAuditSnapshot({ dbPath, windowDays: 0 })).toThrow(
      /windowDays must be a positive integer/,
    );
    expect(() => runAuditSnapshot({ dbPath, windowDays: -7 })).toThrow(
      /windowDays must be a positive integer/,
    );
    expect(() => runAuditSnapshot({ dbPath, windowDays: 1.5 })).toThrow(
      /windowDays must be a positive integer/,
    );
  });

  it("counts only `status='success' AND result IS NULL` short runs as gated_likely (#375 review)", () => {
    // Per the audit doc's "How the gate actually works" section, the
    // canonical gate signature is `result=null` + short duration on a
    // success row. Fast errors (status='error') share the short
    // duration but aren't gates; counting them in `gated_likely`
    // would inflate the heuristic.
    makeFixtureDb();
    insertTask('t1', 'g1', 'cron', '0 7 * * *', 'active', 'precheck.py');

    const now = new Date('2026-05-01T00:00:00Z');
    // Genuine gate-out: success + null result + short duration → counts.
    insertRun('t1', '2026-04-15T07:00:00Z', 3_000, {
      status: 'success',
      result: null,
    });
    // Fast error: status='error' → does NOT count (was conflated pre-fix).
    insertRun('t1', '2026-04-16T07:00:00Z', 4_000, {
      status: 'error',
      result: null,
    });
    // Short success WITH result body: agent woke briefly, did work →
    // does NOT count as a gate.
    insertRun('t1', '2026-04-17T07:00:00Z', 5_000, {
      status: 'success',
      result: '{"answered": true}',
    });
    // Long success: full agent run → does NOT count.
    insertRun('t1', '2026-04-18T07:00:00Z', 60_000, {
      status: 'success',
      result: null,
    });

    const snap = runAuditSnapshot({ dbPath, now });

    const t = snap.tasks.find((tt) => tt.task_id === 't1')!;
    expect(t.fires).toBe(4);
    expect(t.gated_likely).toBe(1); // only the genuine gate-out
  });

  it('orders tasks deterministically by group_folder then schedule_type then id', () => {
    makeFixtureDb();
    insertTask('t-z-int', 'group_z', 'interval', '30m', 'active', null);
    insertTask('t-a-cron', 'group_a', 'cron', '0 7 * * *', 'active', null);
    insertTask('t-z-cron', 'group_z', 'cron', '0 7 * * *', 'active', null);

    const snap = runAuditSnapshot({
      dbPath,
      now: new Date('2026-05-01T00:00:00Z'),
    });

    expect(snap.tasks.map((t) => t.task_id)).toEqual([
      't-a-cron',
      't-z-cron',
      't-z-int',
    ]);
  });
});
