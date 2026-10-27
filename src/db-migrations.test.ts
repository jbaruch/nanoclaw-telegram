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
  it('repairs version0 committed columns whose historical backfills never ran', () => {
    initialize();
    _closeDatabase();
    opened = false;
    const old = disk();
    old.exec(`
      INSERT INTO chats(jid,name) VALUES ('tg:123','preserved'),('123@g.us','group');
      INSERT INTO messages(id,chat_jid,content) VALUES ('fixed','tg:123','Andy: saved response');
      INSERT INTO registered_groups(jid,name,folder,trigger_pattern,added_at)
        VALUES ('tg:123','main','telegram_main','@Andy','2024-01-01T00:00:00Z');
    `);
    old.pragma('user_version = 0');
    old.close();
    initialize();
    _closeDatabase();
    opened = false;
    initialize();
    const check = disk();
    try {
      expect(check.pragma('user_version', { simple: true })).toBe(1);
      expect(
        check.prepare('SELECT is_main FROM registered_groups').get(),
      ).toEqual({ is_main: 1 });
      expect(
        check.prepare('SELECT content,is_bot_message FROM messages').get(),
      ).toEqual({ content: 'Andy: saved response', is_bot_message: 1 });
      expect(
        check
          .prepare('SELECT jid,channel,is_group FROM chats ORDER BY jid')
          .all(),
      ).toEqual([
        { jid: '123@g.us', channel: 'whatsapp', is_group: 1 },
        { jid: 'tg:123', channel: 'telegram', is_group: 0 },
      ]);
    } finally {
      check.close();
    }
  });
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
    old.pragma('user_version = 0');
    old.close();
    initialize();
    _closeDatabase();
    opened = false;
    initialize();
    const check = disk();
    try {
      expect(check.pragma('user_version', { simple: true })).toBe(1);
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
    old.pragma('user_version = 0');
    old.close();
    expect(initialize).toThrow('fixed migration failure');
    _closeDatabase();
    opened = false;
    const check = disk();
    expect(columns(check, 'registered_groups')).not.toContain('is_main');
    expect(check.pragma('user_version', { simple: true })).toBe(0);
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

  it('refuses a newer database without rewriting its schema or records', () => {
    initialize();
    _closeDatabase();
    opened = false;
    const newer = disk();
    newer.exec(
      "INSERT INTO router_state(key,value) VALUES ('fixed','preserved')",
    );
    newer.pragma('user_version = 2');
    newer.close();
    const before = fs.readFileSync(path.join(paths.store, 'messages.db'));
    expect(initialize).toThrow('Unsupported database schema version');
    _closeDatabase();
    opened = false;
    expect(fs.readFileSync(path.join(paths.store, 'messages.db'))).toEqual(
      before,
    );
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

  it.each(['router_state', 'sessions', 'registered_groups'] as const)(
    'retains %s legacy bytes and rolls back the whole file on an injected later write failure',
    (name) => {
      initialize();
      _closeDatabase();
      opened = false;
      const source = path.join(paths.data, `${name}.json`);
      const group = (folder: string) => ({
        name: folder,
        folder,
        trigger: '@Andy',
        added_at: '2024-01-01T00:00:00Z',
      });
      const data =
        name === 'router_state'
          ? {
              last_timestamp: 'fixed-timestamp',
              last_agent_timestamp: { 'tg:123': 'fixed-cursor' },
            }
          : name === 'sessions'
            ? { first: 'session-first', second: 'session-second' }
            : { 'tg:first': group('first'), 'tg:second': group('second') };
      const raw = JSON.stringify(data);
      fs.writeFileSync(source, raw);
      const database = disk();
      const key =
        name === 'router_state'
          ? 'key'
          : name === 'sessions'
            ? 'group_folder'
            : 'jid';
      const second =
        name === 'router_state'
          ? 'last_agent_timestamp'
          : name === 'sessions'
            ? 'second'
            : 'tg:second';
      database.exec(
        `CREATE TRIGGER blocked_legacy BEFORE INSERT ON ${name} WHEN NEW.${key}='${second}' BEGIN SELECT RAISE(ABORT,'fixed legacy import failure'); END;`,
      );
      database.close();
      expect(initialize).toThrow('fixed legacy import failure');
      _closeDatabase();
      opened = false;
      const check = disk();
      expect(
        check.prepare(`SELECT COUNT(*) AS count FROM ${name}`).get(),
      ).toEqual({ count: 0 });
      check.exec('DROP TRIGGER blocked_legacy');
      check.close();
      expect(fs.readFileSync(source, 'utf8')).toBe(raw);
      expect(fs.existsSync(`${source}.migrated`)).toBe(false);
      initialize();
      _closeDatabase();
      opened = false;
      initialize();
      const restored = disk();
      expect(
        restored.prepare(`SELECT COUNT(*) AS count FROM ${name}`).get(),
      ).toEqual({ count: 2 });
      restored.close();
      expect(fs.existsSync(source)).toBe(false);
      expect(fs.readFileSync(`${source}.migrated`, 'utf8')).toBe(raw);
    },
  );

  it('keeps a committed file retryable when archival rename fails', () => {
    const source = path.join(paths.data, 'sessions.json'),
      raw = JSON.stringify({ first: 'session-first' });
    fs.writeFileSync(source, raw);
    const defect = new TypeError('fixed archive rename defect');
    const rename = vi.spyOn(fs, 'renameSync').mockImplementationOnce(() => {
      throw defect;
    });
    expect(initialize).toThrow(defect);
    _closeDatabase();
    opened = false;
    expect(fs.readFileSync(source, 'utf8')).toBe(raw);
    const committed = disk();
    expect(
      committed.prepare('SELECT group_folder,session_id FROM sessions').all(),
    ).toEqual([{ group_folder: 'first', session_id: 'session-first' }]);
    committed.close();
    rename.mockRestore();
    initialize();
    expect(fs.readFileSync(`${source}.migrated`, 'utf8')).toBe(raw);
  });

  it('preserves an unsupported legacy shape instead of archiving or partially importing it', () => {
    const source = path.join(paths.data, 'router_state.json'),
      raw = JSON.stringify({ last_timestamp: 'fixed', schema_version: 2 });
    fs.writeFileSync(source, raw);
    expect(initialize).toThrow('Unknown legacy router state fields');
    expect(fs.readFileSync(source, 'utf8')).toBe(raw);
    expect(fs.existsSync(`${source}.migrated`)).toBe(false);
  });

  it('imports a complete typed legacy group configuration and preserves the original archive', () => {
    const source = path.join(paths.data, 'registered_groups.json');
    const config = {
      timeout: 60000,
      additionalMounts: [
        { hostPath: '/synthetic', containerPath: 'fixed', readonly: true },
      ],
    };
    const raw = JSON.stringify({
      'tg:fixed': {
        name: 'fixed',
        folder: 'fixed',
        trigger: '@Andy',
        added_at: '2024-01-01T00:00:00Z',
        requiresTrigger: false,
        isMain: true,
        containerConfig: config,
      },
    });
    fs.writeFileSync(source, raw);
    initialize();
    const check = disk();
    expect(
      check
        .prepare(
          'SELECT container_config,requires_trigger,is_main FROM registered_groups WHERE jid=?',
        )
        .get('tg:fixed'),
    ).toEqual({
      container_config: JSON.stringify(config),
      requires_trigger: 0,
      is_main: 1,
    });
    check.close();
    expect(fs.readFileSync(`${source}.migrated`, 'utf8')).toBe(raw);
  });

  it('preserves malformed nested legacy configuration without importing a permissive alternative', () => {
    const source = path.join(paths.data, 'registered_groups.json');
    const raw = JSON.stringify({
      'tg:fixed': {
        name: 'fixed',
        folder: 'fixed',
        trigger: '@Andy',
        added_at: '2024-01-01T00:00:00Z',
        containerConfig: {
          additionalMounts: [{ hostPath: '/synthetic', readonly: 'false' }],
        },
      },
    });
    fs.writeFileSync(source, raw);
    expect(initialize).toThrow('Legacy mount readonly must be Boolean');
    expect(fs.readFileSync(source, 'utf8')).toBe(raw);
    expect(fs.existsSync(`${source}.migrated`)).toBe(false);
    const check = disk();
    expect(
      check.prepare('SELECT COUNT(*) AS count FROM registered_groups').get(),
    ).toEqual({ count: 0 });
    check.close();
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
