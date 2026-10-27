import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const paths = vi.hoisted(() => ({ store: '', data: '' }));
vi.mock('./config.js', () => ({
  ASSISTANT_NAME: 'Andy',
  get STORE_DIR() {
    return paths.store;
  },
  get DATA_DIR() {
    return paths.data;
  },
  GROUPS_DIR: '/synthetic/groups',
}));
import { _closeDatabase, initDatabase } from './db.js';

let directory: string;
let opened = false;
beforeEach(() => {
  directory = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-migrations-'));
  paths.store = path.join(directory, 'store');
  paths.data = path.join(directory, 'data');
  fs.mkdirSync(paths.data);
});
afterEach(() => {
  if (opened) _closeDatabase();
  opened = false;
  vi.restoreAllMocks();
  fs.rmSync(directory, { recursive: true });
});
function initialize() {
  opened = true;
  initDatabase();
}
function disk() {
  return new Database(path.join(paths.store, 'messages.db'));
}
function columns(database: Database.Database, table: string): string[] {
  return (database.pragma(`table_info(${table})`) as { name: string }[]).map(
    (row) => row.name,
  );
}

describe('durable schema migration', () => {
  it('repairs a partially old schema, preserves history, and is idempotent after restart', () => {
    initialize();
    _closeDatabase();
    opened = false;
    const old = disk();
    old.exec(`ALTER TABLE scheduled_tasks DROP COLUMN context_mode;
      ALTER TABLE scheduled_tasks DROP COLUMN script;
      ALTER TABLE registered_groups DROP COLUMN is_main;
      ALTER TABLE messages DROP COLUMN is_bot_message;
      ALTER TABLE chats DROP COLUMN is_group;
      INSERT INTO chats(jid,name) VALUES ('tg:123','preserved');
      INSERT INTO messages(id,chat_jid,content) VALUES ('fixed-message','tg:123','Andy: preserved reply');
      INSERT INTO registered_groups(jid,name,folder,trigger_pattern,added_at) VALUES ('tg:123','main','telegram_main','@Andy','2024-01-01T00:00:00Z');`);
    old.close();
    initialize();
    _closeDatabase();
    opened = false;
    initialize();
    const check = disk();
    try {
      expect(columns(check, 'scheduled_tasks')).toEqual(
        expect.arrayContaining(['context_mode', 'script']),
      );
      expect(
        check
          .prepare('SELECT name,is_group FROM chats WHERE jid=?')
          .get('tg:123'),
      ).toEqual({ name: 'preserved', is_group: 0 });
      expect(
        check
          .prepare('SELECT content,is_bot_message FROM messages WHERE id=?')
          .get('fixed-message'),
      ).toEqual({ content: 'Andy: preserved reply', is_bot_message: 1 });
      expect(
        check
          .prepare('SELECT is_main FROM registered_groups WHERE jid=?')
          .get('tg:123'),
      ).toEqual({ is_main: 1 });
    } finally {
      check.close();
    }
  });

  it('propagates failed backfill and rolls back its column, then recovers after the storage fault is fixed', () => {
    initialize();
    _closeDatabase();
    opened = false;
    const old = disk();
    old.exec(`ALTER TABLE registered_groups DROP COLUMN is_main;
      INSERT INTO registered_groups(jid,name,folder,trigger_pattern,added_at) VALUES ('tg:123','main','telegram_main','@Andy','2024-01-01T00:00:00Z');
      CREATE TRIGGER blocked_backfill BEFORE UPDATE ON registered_groups BEGIN SELECT RAISE(ABORT,'fixed migration failure'); END;`);
    old.close();
    expect(initialize).toThrow('fixed migration failure');
    _closeDatabase();
    opened = false;
    const check = disk();
    expect(columns(check, 'registered_groups')).not.toContain('is_main');
    expect(check.prepare('SELECT name FROM registered_groups').all()).toEqual([
      { name: 'main' },
    ]);
    check.exec('DROP TRIGGER blocked_backfill');
    check.close();
    initialize();
    const repaired = disk();
    try {
      expect(
        repaired.prepare('SELECT is_main FROM registered_groups').get(),
      ).toEqual({ is_main: 1 });
    } finally {
      repaired.close();
    }
  });

  it('skips only an invalid legacy group and still migrates the valid group', () => {
    fs.writeFileSync(
      path.join(paths.data, 'registered_groups.json'),
      JSON.stringify({
        'tg:bad': {
          name: 'bad',
          folder: '../escape',
          trigger: '@Andy',
          added_at: '2024-01-01T00:00:00Z',
        },
        'tg:good': {
          name: 'good',
          folder: 'fixed-group',
          trigger: '@Andy',
          added_at: '2024-01-01T00:00:00Z',
        },
      }),
    );
    initialize();
    const check = disk();
    try {
      expect(
        check.prepare('SELECT jid,folder FROM registered_groups').all(),
      ).toEqual([{ jid: 'tg:good', folder: 'fixed-group' }]);
    } finally {
      check.close();
    }
  });

  it('does not hide an unexpected JSON migration implementation failure or rename its source', () => {
    fs.writeFileSync(path.join(paths.data, 'sessions.json'), '{}');
    const defect = new TypeError('fixed migration defect');
    vi.spyOn(fs, 'renameSync').mockImplementation(() => {
      throw defect;
    });
    expect(initialize).toThrow(defect);
    expect(
      fs.readFileSync(path.join(paths.data, 'sessions.json'), 'utf8'),
    ).toBe('{}');
    expect(fs.existsSync(path.join(paths.data, 'sessions.json.migrated'))).toBe(
      false,
    );
  });
});
