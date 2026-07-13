import Database from 'better-sqlite3';
import { describe, it, expect } from 'vitest';

import { applyStateMigrations } from '../db.js';

import { STATE_MIGRATIONS } from './index.js';

/**
 * The migration operates on the base-schema `registered_groups` table
 * (created by `createSchema`, not by a state-migration), so the test
 * stands up a minimal table with the one column the migration touches
 * before applying the full registry. Rows are seeded BEFORE
 * `applyStateMigrations` so migration 16's DML processes them exactly as
 * it would an in-place upgrade of a live DB.
 */
function freshDb(): Database.Database {
  const db = new Database(':memory:');
  db.exec(
    `CREATE TABLE registered_groups (
       jid TEXT PRIMARY KEY,
       container_config TEXT
     );`,
  );
  return db;
}

describe('state-016-strip-stage2-enabled', () => {
  it('is the last registered migration and bumps user_version to 16', () => {
    const db = freshDb();
    try {
      applyStateMigrations(db, STATE_MIGRATIONS);
      expect(Number(db.pragma('user_version', { simple: true }))).toBe(16);
    } finally {
      db.close();
    }
  });

  it('removes stage2Enabled while preserving every other config key', () => {
    const db = freshDb();
    try {
      db.prepare(
        'INSERT INTO registered_groups (jid, container_config) VALUES (?, ?)',
      ).run(
        'group-true',
        JSON.stringify({
          stage2Enabled: true,
          trusted: true,
          agentModel: 'sonnet',
          additionalMounts: [{ hostPath: '/x', containerPath: 'x' }],
        }),
      );
      db.prepare(
        'INSERT INTO registered_groups (jid, container_config) VALUES (?, ?)',
      ).run(
        'group-false',
        JSON.stringify({
          stage2Enabled: false,
          maintenanceAgentModel: 'sonnet',
        }),
      );

      applyStateMigrations(db, STATE_MIGRATIONS);

      const trueRow = db
        .prepare('SELECT container_config FROM registered_groups WHERE jid = ?')
        .get('group-true') as { container_config: string };
      const trueCfg = JSON.parse(trueRow.container_config);
      expect(trueCfg).not.toHaveProperty('stage2Enabled');
      expect(trueCfg.trusted).toBe(true);
      expect(trueCfg.agentModel).toBe('sonnet');
      expect(trueCfg.additionalMounts).toEqual([
        { hostPath: '/x', containerPath: 'x' },
      ]);

      const falseRow = db
        .prepare('SELECT container_config FROM registered_groups WHERE jid = ?')
        .get('group-false') as { container_config: string };
      const falseCfg = JSON.parse(falseRow.container_config);
      expect(falseCfg).not.toHaveProperty('stage2Enabled');
      expect(falseCfg.maintenanceAgentModel).toBe('sonnet');
    } finally {
      db.close();
    }
  });

  it('strips stage2Enabled even when its value is JSON null', () => {
    const db = freshDb();
    try {
      // `json_extract(...) IS NOT NULL` would MISS this row (extract
      // yields SQL NULL for a json-null value); `json_type` presence
      // catches it.
      db.prepare(
        'INSERT INTO registered_groups (jid, container_config) VALUES (?, ?)',
      ).run(
        'group-null-value',
        JSON.stringify({ stage2Enabled: null, trusted: true }),
      );

      applyStateMigrations(db, STATE_MIGRATIONS);

      const row = db
        .prepare('SELECT container_config FROM registered_groups WHERE jid = ?')
        .get('group-null-value') as { container_config: string };
      const cfg = JSON.parse(row.container_config);
      expect(cfg).not.toHaveProperty('stage2Enabled');
      expect(cfg.trusted).toBe(true);
    } finally {
      db.close();
    }
  });

  it('leaves clean, NULL, and non-JSON rows byte-identical', () => {
    const db = freshDb();
    try {
      const cleanConfig = JSON.stringify({ trusted: false, timeout: 300000 });
      db.prepare(
        'INSERT INTO registered_groups (jid, container_config) VALUES (?, ?)',
      ).run('group-clean', cleanConfig);
      db.prepare(
        'INSERT INTO registered_groups (jid, container_config) VALUES (?, ?)',
      ).run('group-null', null);
      // A corrupt non-JSON blob must not be rewritten or crash the migration.
      db.prepare(
        'INSERT INTO registered_groups (jid, container_config) VALUES (?, ?)',
      ).run('group-garbage', 'not json {');

      applyStateMigrations(db, STATE_MIGRATIONS);

      const rows = Object.fromEntries(
        (
          db
            .prepare('SELECT jid, container_config FROM registered_groups')
            .all() as Array<{ jid: string; container_config: string | null }>
        ).map((r) => [r.jid, r.container_config]),
      );
      expect(rows['group-clean']).toBe(cleanConfig);
      expect(rows['group-null']).toBeNull();
      expect(rows['group-garbage']).toBe('not json {');
    } finally {
      db.close();
    }
  });
});
