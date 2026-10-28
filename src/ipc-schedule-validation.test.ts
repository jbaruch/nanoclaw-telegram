import fs from 'fs';
import os from 'os';
import path from 'path';
import Database from 'better-sqlite3';
import { afterEach, expect, it, vi } from 'vitest';
const fixture = vi.hoisted(() => ({ directory: '' }));
vi.mock('./config.js', () => ({
  get DATA_DIR() {
    return fixture.directory;
  },
  get STORE_DIR() {
    return fixture.directory;
  },
  get GROUPS_DIR() {
    return path.join(fixture.directory, 'groups');
  },
  ASSISTANT_NAME: 'Andy',
  IPC_POLL_INTERVAL: 1000,
  TIMEZONE: 'UTC',
}));
import { startIpcWatcher, type IpcDeps } from './ipc.js';
import {
  _initTestDatabase,
  _closeDatabase,
  createTask,
  getTaskById,
  updateTask,
} from './db.js';
afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  _closeDatabase();
  if (fixture.directory)
    fs.rmSync(fixture.directory, { recursive: true, force: true });
});
it('quarantines an invalid update unchanged, applies a valid once update, and propagates a SQL defect', async () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2024-01-01T00:00:00Z'));
  fixture.directory = fs.mkdtempSync(
    path.join(os.tmpdir(), 'telegram-ipc-schedule-'),
  );
  _initTestDatabase();
  createTask({
    id: 'fixed-task',
    group_folder: 'main',
    chat_jid: 'tg:123',
    prompt: 'fixed',
    schedule_type: 'interval',
    schedule_value: '60000',
    next_run: '2024-01-01T00:01:00.000Z',
    context_mode: 'isolated',
    status: 'active',
    created_at: '2024-01-01T00:00:00Z',
  });
  const tasks = path.join(fixture.directory, 'ipc', 'main', 'tasks'),
    messages = path.join(fixture.directory, 'ipc', 'main', 'messages');
  fs.mkdirSync(tasks, { recursive: true });
  fs.mkdirSync(messages);
  const before = getTaskById('fixed-task');
  const invalid = {
    schema_version: 1,
    type: 'update_task',
    taskId: 'fixed-task',
    schedule_value: '1000junk',
    prompt: 'must not persist',
  };
  fs.writeFileSync(path.join(tasks, 'invalid.json'), JSON.stringify(invalid));
  const deps: IpcDeps = {
    sendMessage: vi.fn(async () => {}),
    registeredGroups: () => ({
      'tg:123': {
        name: 'fixed',
        folder: 'main',
        isMain: true,
        trigger: '@Andy',
        added_at: '2024-01-01T00:00:00Z',
      },
    }),
    registerGroup: vi.fn(),
    syncGroups: vi.fn(async () => {}),
    getAvailableGroups: () => [],
    writeGroupsSnapshot: vi.fn(),
    onTasksChanged: vi.fn(),
  };
  const lifecycle = startIpcWatcher(deps);
  await vi.advanceTimersByTimeAsync(0);
  expect(getTaskById('fixed-task')).toEqual(before);
  expect(
    JSON.parse(
      fs.readFileSync(
        path.join(
          fixture.directory,
          'ipc',
          'errors',
          'main-tasks-0-invalid.json',
        ),
        'utf8',
      ),
    ),
  ).toEqual(invalid);
  fs.writeFileSync(
    path.join(tasks, 'valid.json'),
    JSON.stringify({
      schema_version: 1,
      type: 'update_task',
      taskId: 'fixed-task',
      schedule_type: 'once',
      schedule_value: '2024-02-03T04:05:06Z',
    }),
  );
  await vi.advanceTimersByTimeAsync(1000);
  expect(getTaskById('fixed-task')).toMatchObject({
    schedule_type: 'once',
    schedule_value: '2024-02-03T04:05:06Z',
    next_run: '2024-02-03T04:05:06.000Z',
    prompt: 'fixed',
  });
  expect(deps.onTasksChanged).toHaveBeenCalledTimes(1);
  expect(fs.existsSync(path.join(tasks, 'valid.json'))).toBe(false);
  // Existing owner-written legacy schedule is not reparsed by an unrelated edit.
  updateTask('fixed-task', {
    schedule_value: '2024-02-03T04:05',
    next_run: '2024-02-03T04:05:06.000Z',
  });
  fs.writeFileSync(
    path.join(tasks, 'prompt-only.json'),
    JSON.stringify({
      schema_version: 1,
      type: 'update_task',
      taskId: 'fixed-task',
      prompt: 'updated prompt',
      script: 'printf fixed',
    }),
  );
  await vi.advanceTimersByTimeAsync(1000);
  expect(getTaskById('fixed-task')).toMatchObject({
    prompt: 'updated prompt',
    script: 'printf fixed',
    schedule_value: '2024-02-03T04:05',
    next_run: '2024-02-03T04:05:06.000Z',
  });
  expect(deps.onTasksChanged).toHaveBeenCalledTimes(2);

  const defect = new Database.SqliteError(
    'fixed missing column',
    'SQLITE_ERROR',
  );
  vi.mocked(deps.sendMessage).mockRejectedValueOnce(defect);
  fs.writeFileSync(
    path.join(messages, 'sql-defect.json'),
    JSON.stringify({
      schema_version: 1,
      type: 'message',
      chatJid: 'tg:123',
      text: 'fixed input',
    }),
  );
  const rejected = expect(lifecycle).rejects.toBe(defect);
  await vi.advanceTimersByTimeAsync(1000);
  await rejected;
  expect(
    fs.existsSync(
      path.join(
        fixture.directory,
        'ipc',
        'errors',
        'main-messages-0-sql-defect.json',
      ),
    ),
  ).toBe(true);
});
