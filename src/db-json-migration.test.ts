import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

type Db = typeof import('./db.js');
type Logger = (typeof import('./logger.js'))['logger'];

const validGroup = {
  name: 'Good group',
  folder: 'good-group',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

const traversalGroup = {
  name: 'Bad group',
  folder: '../../etc',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

let repoRoot: string;
let tempDir: string;
let openDb: Db | null;

function dataFile(name: string): string {
  return path.join(tempDir, 'data', name);
}

function writeDataFile(name: string, content: string): string {
  const file = dataFile(name);
  fs.writeFileSync(file, content);
  return file;
}

// config.js resolves STORE_DIR and DATA_DIR from process.cwd() at module
// load, so every test gets a fresh module graph rooted in its temp dir.
async function freshDb(): Promise<{ db: Db; logger: Logger }> {
  vi.resetModules();
  const db = await import('./db.js');
  const { logger } = await import('./logger.js');
  openDb = db;
  return { db, logger };
}

beforeEach(() => {
  vi.clearAllMocks();
  repoRoot = process.cwd();
  tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-json-migration-'));
  fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
  fs.mkdirSync(path.join(tempDir, 'data'), { recursive: true });
  process.chdir(tempDir);
  openDb = null;
});

afterEach(() => {
  if (openDb) openDb._closeDatabase();
  process.chdir(repoRoot);
  fs.rmSync(tempDir, { recursive: true, force: true });
});

describe('JSON state migration', () => {
  it('migrates registered groups and skips an entry whose folder is invalid', async () => {
    const file = writeDataFile(
      'registered_groups.json',
      JSON.stringify({ 'tg:1': validGroup, 'tg:2': traversalGroup }),
    );
    const { db, logger } = await freshDb();

    expect(() => db.initDatabase()).not.toThrow();

    const groups = db.getAllRegisteredGroups();
    expect(groups['tg:1']).toMatchObject({
      name: 'Good group',
      folder: 'good-group',
    });
    expect(groups['tg:2']).toBeUndefined();
    expect(logger.warn).toHaveBeenCalledWith(
      expect.objectContaining({ jid: 'tg:2', folder: '../../etc' }),
      'Skipping migrated registered group with invalid folder',
    );
    expect(fs.existsSync(file)).toBe(false);
    expect(fs.existsSync(`${file}.migrated`)).toBe(true);
  });

  it('leaves a malformed registered_groups.json in place without aborting startup', async () => {
    const file = writeDataFile('registered_groups.json', 'not json{{{');
    const { db } = await freshDb();

    expect(() => db.initDatabase()).not.toThrow();

    expect(db.getAllRegisteredGroups()).toEqual({});
    expect(fs.existsSync(file)).toBe(true);
    expect(fs.existsSync(`${file}.migrated`)).toBe(false);
  });

  it('migrates router state and renames the source file', async () => {
    const file = writeDataFile(
      'router_state.json',
      JSON.stringify({
        last_timestamp: '2024-02-02T00:00:00.000Z',
        last_agent_timestamp: { 'tg:1': '2024-02-01T00:00:00.000Z' },
      }),
    );
    const { db } = await freshDb();

    db.initDatabase();

    expect(db.getRouterState('last_timestamp')).toBe(
      '2024-02-02T00:00:00.000Z',
    );
    expect(db.getRouterState('last_agent_timestamp')).toBe(
      JSON.stringify({ 'tg:1': '2024-02-01T00:00:00.000Z' }),
    );
    expect(fs.existsSync(`${file}.migrated`)).toBe(true);
  });

  it('initializes an already-migrated store a second time', async () => {
    const first = await freshDb();
    first.db.initDatabase();
    first.db.setRegisteredGroup('tg:9', validGroup);
    first.db._closeDatabase();
    openDb = null;

    const second = await freshDb();

    expect(() => second.db.initDatabase()).not.toThrow();
    expect(second.db.getAllRegisteredGroups()['tg:9']).toMatchObject({
      folder: 'good-group',
    });
  });
});
