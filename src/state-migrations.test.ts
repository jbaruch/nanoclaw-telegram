import Database from 'better-sqlite3';
import { describe, it, expect } from 'vitest';

import { applyStateMigrations, type StateMigration } from './db.js';

function readUserVersion(database: Database.Database): number {
  return Number(database.pragma('user_version', { simple: true }));
}

function tableExists(database: Database.Database, name: string): boolean {
  const row = database
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(name) as { name: string } | undefined;
  return !!row;
}

describe('applyStateMigrations', () => {
  it('is a no-op when the registry is empty', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, []);
      expect(readUserVersion(database)).toBe(0);
    } finally {
      database.close();
    }
  });

  it('applies a single migration and bumps user_version to its number', () => {
    const database = new Database(':memory:');
    try {
      const migrations: StateMigration[] = [
        {
          version: 1,
          name: 'create widgets',
          sql: 'CREATE TABLE widgets (id TEXT PRIMARY KEY, label TEXT NOT NULL);',
        },
      ];
      applyStateMigrations(database, migrations);
      expect(readUserVersion(database)).toBe(1);
      expect(tableExists(database, 'widgets')).toBe(true);
    } finally {
      database.close();
    }
  });

  it('applies multiple migrations in version order', () => {
    const database = new Database(':memory:');
    try {
      const migrations: StateMigration[] = [
        {
          version: 1,
          name: 'create alpha',
          sql: 'CREATE TABLE alpha (id TEXT PRIMARY KEY);',
        },
        {
          version: 2,
          name: 'create beta',
          sql: 'CREATE TABLE beta (id TEXT PRIMARY KEY, alpha_id TEXT REFERENCES alpha(id));',
        },
        {
          version: 3,
          name: 'add label to alpha',
          sql: 'ALTER TABLE alpha ADD COLUMN label TEXT;',
        },
      ];
      applyStateMigrations(database, migrations);
      expect(readUserVersion(database)).toBe(3);
      expect(tableExists(database, 'alpha')).toBe(true);
      expect(tableExists(database, 'beta')).toBe(true);
      const cols = database.prepare('PRAGMA table_info(alpha)').all() as Array<{
        name: string;
      }>;
      expect(cols.some((c) => c.name === 'label')).toBe(true);
    } finally {
      database.close();
    }
  });

  it('is idempotent when the registry has not changed since the last run', () => {
    const database = new Database(':memory:');
    try {
      const migrations: StateMigration[] = [
        {
          version: 1,
          name: 'create widgets',
          sql: 'CREATE TABLE widgets (id TEXT PRIMARY KEY);',
        },
      ];
      applyStateMigrations(database, migrations);
      expect(readUserVersion(database)).toBe(1);
      // Second pass — would throw "table widgets already exists" if it
      // re-applied the DDL, so a successful second call is itself
      // evidence that the version-gate skipped the migration.
      applyStateMigrations(database, migrations);
      expect(readUserVersion(database)).toBe(1);
    } finally {
      database.close();
    }
  });

  it('applies only the new entries when the registry grows', () => {
    const database = new Database(':memory:');
    try {
      applyStateMigrations(database, [
        {
          version: 1,
          name: 'create alpha',
          sql: 'CREATE TABLE alpha (id TEXT PRIMARY KEY);',
        },
      ]);
      expect(readUserVersion(database)).toBe(1);

      // Same v1, plus a new v2. Re-running v1 against an existing
      // alpha table would throw — passing means only v2 ran.
      applyStateMigrations(database, [
        {
          version: 1,
          name: 'create alpha',
          sql: 'CREATE TABLE alpha (id TEXT PRIMARY KEY);',
        },
        {
          version: 2,
          name: 'create beta',
          sql: 'CREATE TABLE beta (id TEXT PRIMARY KEY);',
        },
      ]);
      expect(readUserVersion(database)).toBe(2);
      expect(tableExists(database, 'beta')).toBe(true);
    } finally {
      database.close();
    }
  });

  it('throws when user_version is higher than the registry knows about', () => {
    const database = new Database(':memory:');
    try {
      database.pragma('user_version = 5');
      expect(() =>
        applyStateMigrations(database, [
          {
            version: 1,
            name: 'create widgets',
            sql: 'CREATE TABLE widgets (id TEXT PRIMARY KEY);',
          },
        ]),
      ).toThrow(/user_version=5.*up to version=1/s);
      // Version unchanged — startup aborts rather than mutating state.
      expect(readUserVersion(database)).toBe(5);
    } finally {
      database.close();
    }
  });

  it('rolls back the version bump when the migration SQL fails', () => {
    const database = new Database(':memory:');
    try {
      const migrations: StateMigration[] = [
        {
          version: 1,
          name: 'create alpha',
          sql: 'CREATE TABLE alpha (id TEXT PRIMARY KEY);',
        },
        {
          version: 2,
          name: 'broken',
          sql: 'CREATE TABLE this is not valid sql;',
        },
      ];
      expect(() => applyStateMigrations(database, migrations)).toThrow();
      // v1 applied, v2's failure rolled back its version bump — next
      // run can retry v2 without re-running v1.
      expect(readUserVersion(database)).toBe(1);
      expect(tableExists(database, 'alpha')).toBe(true);
    } finally {
      database.close();
    }
  });

  it('rejects a registry whose first version is not 1', () => {
    const database = new Database(':memory:');
    try {
      expect(() =>
        applyStateMigrations(database, [
          {
            version: 2,
            name: 'starts at 2',
            sql: 'CREATE TABLE foo (id TEXT);',
          },
        ]),
      ).toThrow(/expected version=1.*got version=2/s);
    } finally {
      database.close();
    }
  });

  it('rejects a registry with a gap in versions', () => {
    const database = new Database(':memory:');
    try {
      expect(() =>
        applyStateMigrations(database, [
          {
            version: 1,
            name: 'first',
            sql: 'CREATE TABLE a (id TEXT);',
          },
          {
            version: 3,
            name: 'skipped 2',
            sql: 'CREATE TABLE c (id TEXT);',
          },
        ]),
      ).toThrow(/expected version=2.*got version=3/s);
    } finally {
      database.close();
    }
  });

  it('rejects a registry with duplicate versions', () => {
    const database = new Database(':memory:');
    try {
      expect(() =>
        applyStateMigrations(database, [
          {
            version: 1,
            name: 'first',
            sql: 'CREATE TABLE a (id TEXT);',
          },
          {
            version: 1,
            name: 'duplicate',
            sql: 'CREATE TABLE b (id TEXT);',
          },
        ]),
      ).toThrow(/expected version=2.*got version=1/s);
    } finally {
      database.close();
    }
  });

  it('rejects non-positive or non-integer versions', () => {
    const database = new Database(':memory:');
    try {
      expect(() =>
        applyStateMigrations(database, [
          {
            version: 0,
            name: 'zero',
            sql: 'CREATE TABLE foo (id TEXT);',
          },
        ]),
      ).toThrow(/version must be a positive integer/);
      expect(() =>
        applyStateMigrations(database, [
          {
            version: 1.5,
            name: 'fractional',
            sql: 'CREATE TABLE foo (id TEXT);',
          },
        ]),
      ).toThrow(/version must be a positive integer/);
    } finally {
      database.close();
    }
  });

  it('rejects empty name or sql', () => {
    const database = new Database(':memory:');
    try {
      expect(() =>
        applyStateMigrations(database, [
          {
            version: 1,
            name: '',
            sql: 'CREATE TABLE foo (id TEXT);',
          },
        ]),
      ).toThrow(/name must be a non-empty string/);
      expect(() =>
        applyStateMigrations(database, [
          {
            version: 1,
            name: 'whitespace-only sql',
            sql: '   \n\t  ',
          },
        ]),
      ).toThrow(/sql must be a non-empty string/);
    } finally {
      database.close();
    }
  });
});
