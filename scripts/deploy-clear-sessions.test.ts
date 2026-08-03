import { spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

/**
 * Coverage for deploy step 6 ("Clear sessions") in `scripts/deploy.sh`.
 *
 * Two defects observed on 2026-08-03, both pinned here:
 *
 *  1. The bare `sqlite3` CLI defaults to `busy_timeout=0`. The orchestrator
 *     is still writing at this point in the deploy (step 7 is what restarts
 *     it), so a concurrent write aborted the deploy with
 *     `Error: stepping, database is locked (5)`.
 *  2. `SELECT changes()` was asked on a SECOND connection, where it always
 *     reports 0 — the printed count was decorative on every deploy.
 *
 * The command under test is extracted from the shipped script rather than
 * retyped, so a future edit that drops `.timeout` or splits the invocation
 * fails here instead of on the NAS.
 */

let tmp: string;

/** The exact `sqlite3 …` command line deploy.sh step 6 runs. */
function clearSessionsCommand(): string {
  const src = fs.readFileSync(
    path.join(process.cwd(), 'scripts', 'deploy.sh'),
    'utf8',
  );
  const m = src.match(/^CLEARED=\$\((sqlite3 .+)\)$/m);
  expect(m, 'step 6 CLEARED=$(sqlite3 …) assignment not found').toBeTruthy();
  return m![1];
}

/** Run that command against `dbPath`, returning what deploy.sh would print. */
function runClear(dbPath: string): {
  stdout: string;
  stderr: string;
  status: number;
} {
  const cmd = clearSessionsCommand().replace('store/messages.db', dbPath);
  const r = spawnSync('bash', ['-c', cmd], { encoding: 'utf-8' });
  return {
    stdout: (r.stdout ?? '').trim(),
    stderr: (r.stderr ?? '').trim(),
    status: r.status ?? -1,
  };
}

function seedDb(rows: number): string {
  const db = path.join(tmp, 'messages.db');
  const values = Array.from({ length: rows }, (_, i) => `('s${i}')`).join(',');
  const seed = `PRAGMA journal_mode=WAL; CREATE TABLE sessions(id TEXT);${
    rows > 0 ? ` INSERT INTO sessions VALUES ${values};` : ''
  }`;
  const r = spawnSync('sqlite3', [db, seed], { encoding: 'utf-8' });
  expect(r.status).toBe(0);
  return db;
}

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'deploy-clear-'));
});

afterEach(() => {
  fs.rmSync(tmp, { recursive: true, force: true });
});

describe('deploy.sh step 6 — clear sessions', () => {
  it('reports the number of rows it actually deleted', () => {
    const db = seedDb(3);
    const r = runClear(db);
    expect(r.status).toBe(0);
    // The pre-fix shape asked a second connection and always printed 0.
    expect(r.stdout).toBe('3');
  });

  it('reports 0 when there was genuinely nothing to clear', () => {
    const db = seedDb(0);
    const r = runClear(db);
    expect(r.status).toBe(0);
    expect(r.stdout).toBe('0');
  });

  it('emits only the count — no PRAGMA echo to corrupt it', () => {
    // `PRAGMA busy_timeout=5000;` inline would print 5000 alongside the
    // count; the `.timeout` dot-command sets it silently.
    const db = seedDb(2);
    expect(runClear(db).stdout.split('\n')).toHaveLength(1);
  });

  it('carries a busy timeout at least as long as the orchestrator uses', () => {
    // A runtime contention test cannot be both deterministic and meaningful
    // here: guaranteeing the lock is still held when `sqlite3` attempts the
    // write requires holding it for a duration, which is the wall-clock
    // dependence `testing-standards` bans. Releasing the holder on a
    // deterministic signal instead lets it commit before contention occurs —
    // verified: such a test passes against the pre-fix command, so it proves
    // nothing. The contract is asserted directly instead.
    const cmd = clearSessionsCommand();
    const m = cmd.match(/\.timeout (\d+)/);
    expect(
      m,
      'step 6 must set a busy timeout; without it the CLI defaults to 0 and ' +
        'a concurrent orchestrator write aborts the deploy with SQLITE_BUSY',
    ).toBeTruthy();

    // Pin it to the orchestrator's own value so the two writers cannot drift
    // apart silently — deploy must be at least as patient as the app.
    const dbTs = fs.readFileSync(
      path.join(process.cwd(), 'src', 'db.ts'),
      'utf8',
    );
    const appTimeout = dbTs.match(/busy_timeout\s*=\s*(\d+)/);
    expect(appTimeout, 'src/db.ts busy_timeout pragma not found').toBeTruthy();
    expect(Number(m![1])).toBeGreaterThanOrEqual(Number(appTimeout![1]));
  });

  it('runs the delete and the count on ONE connection', () => {
    // `changes()` is per-connection: a second `sqlite3` invocation reports on
    // a connection that modified nothing, which is why every deploy printed
    // `cleared 0 sessions`. One invocation, both statements.
    const cmd = clearSessionsCommand();
    expect(cmd).toMatch(/DELETE FROM sessions;\s*SELECT changes\(\);/);
    expect(cmd.match(/sqlite3/g)).toHaveLength(1);
  });
});
