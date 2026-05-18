import fs from 'fs';

import {
  describe,
  it,
  expect,
  beforeEach,
  afterEach,
  afterAll,
  vi,
} from 'vitest';

// Isolate filesystem writes to a per-process tempdir so running this test
// file doesn't leave artifacts in the developer's real `data/` tree (or
// collide with a local orchestrator that actually uses `DATA_DIR`).
//
// `vi.mock` is hoisted to the very top of the file, ABOVE regular
// top-level const declarations. To share the tempdir path between the
// mock factory and the rest of the file we compute it inside
// `vi.hoisted`, which runs in the same hoisting pass as the mocks.
const { TEST_DATA_DIR } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const osMod = require('os') as typeof import('os');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pathMod = require('path') as typeof import('path');
  return {
    TEST_DATA_DIR: pathMod.join(
      osMod.tmpdir(),
      `nanoclaw-ipc-auth-test-${process.pid}`,
    ),
  };
});
vi.mock('./config.js', async () => {
  const actual =
    await vi.importActual<typeof import('./config.js')>('./config.js');
  return {
    ...actual,
    DATA_DIR: TEST_DATA_DIR,
  };
});

// `set_additional_tiles` (#305) validates every entry against the local
// Tessl registry by calling `getInstalledTiles()` from container-runner.
// In the test environment there's no real registry on disk, so the
// real implementation would always return `null` and reject every
// write — including the happy-path tests. Stub it here so tests
// control the "installed" set per scenario. Other container-runner
// exports (`DEFAULT_SESSION_NAME`, `resolveAgentModel`, …) are
// preserved so unrelated tests in this file aren't disturbed.
const { mockGetInstalledTiles } = vi.hoisted(() => ({
  mockGetInstalledTiles: vi.fn<() => string[] | null>(() => []),
}));
vi.mock('./container-runner.js', async () => {
  const actual = await vi.importActual<typeof import('./container-runner.js')>(
    './container-runner.js',
  );
  return {
    ...actual,
    getInstalledTiles: mockGetInstalledTiles,
  };
});

import path from 'path';

import {
  _initTestDatabase,
  _seedTzStateForTests,
  createTask,
  deleteRegisteredGroup,
  getAllTasks,
  getRegisteredGroup,
  getTaskById,
  setRegisteredGroup,
  updateGroupTrusted,
  updateGroupTrigger,
} from './db.js';
import { processTaskIpc, IpcDeps } from './ipc.js';
import { RegisteredGroup, TriggerPattern } from './types.js';

// Set up registered groups used across tests
const MAIN_GROUP: RegisteredGroup = {
  name: 'Main',
  folder: 'whatsapp_main',
  trigger: 'always',
  added_at: '2024-01-01T00:00:00.000Z',
  isMain: true,
};

const OTHER_GROUP: RegisteredGroup = {
  name: 'Other',
  folder: 'other-group',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

const THIRD_GROUP: RegisteredGroup = {
  name: 'Third',
  folder: 'third-group',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

let groups: Record<string, RegisteredGroup>;
let deps: IpcDeps;

beforeEach(() => {
  _initTestDatabase();

  groups = {
    'main@g.us': MAIN_GROUP,
    'other@g.us': OTHER_GROUP,
    'third@g.us': THIRD_GROUP,
  };

  // Populate DB as well
  setRegisteredGroup('main@g.us', MAIN_GROUP);
  setRegisteredGroup('other@g.us', OTHER_GROUP);
  setRegisteredGroup('third@g.us', THIRD_GROUP);

  deps = {
    sendMessage: async () => {},
    registeredGroups: () => groups,
    registerGroup: (jid, group) => {
      groups[jid] = group;
      setRegisteredGroup(jid, group);

      // Production registerGroup no longer creates a non-main
      // `heartbeat-<folder>` row — the non-main heartbeat task was
      // retired in #453 along with the `tessl__check-unanswered` skill
      // it drove (`jbaruch/nanoclaw-core#38`). The mock matches: no
      // task creation here for non-main groups regardless of
      // `containerConfig.enableHeartbeat`. Main-group heartbeat is
      // created inline in production but isn't exercised by these IPC
      // auth tests, so it stays out of this mock.
    },
    unregisterGroup: (jid) => {
      // Mirror src/index.ts unregisterGroup: in-memory + DB delete in
      // one call. Returns the DB delete's truthy-changes result so
      // tests can distinguish "actually removed" from "no row matched".
      delete groups[jid];
      return deleteRegisteredGroup(jid);
    },
    setGroupTrusted: (jid, trusted) => {
      const updated = updateGroupTrusted(jid, trusted);
      if (!updated) return false;
      groups[jid] = updated;
      return true;
    },
    setGroupTrigger: (jid, trigger, requiresTrigger) => {
      const updated = updateGroupTrigger(jid, trigger, requiresTrigger);
      if (!updated) return false;
      groups[jid] = updated;
      // Pre-#158, this mock also mirrored a heartbeat-on-flip side
      // effect. Production no longer touches heartbeats from
      // setGroupTrigger — trigger config and heartbeat opt-in are
      // orthogonal — so the mock omits it too.
      return true;
    },
    syncGroups: async () => {},
    getAvailableGroups: () => [],
    writeGroupsSnapshot: () => {},
    onTasksChanged: () => {},
    nukeSession: (
      _folder: string,
      _session: 'default' | 'maintenance' | 'all',
    ) => {},
    closeAllActiveContainers: () => 0,
  };
});

// --- schedule_task authorization ---

describe('schedule_task authorization', () => {
  it('main group can schedule for another group', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'do something',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    // Verify task was created in DB for the other group
    const allTasks = getAllTasks();
    expect(allTasks.length).toBe(1);
    expect(allTasks[0].group_folder).toBe('other-group');
  });

  it('non-main group can schedule for itself', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'self task',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'other@g.us',
      },
      'other-group',
      false,
      deps,
    );

    const allTasks = getAllTasks();
    expect(allTasks.length).toBe(1);
    expect(allTasks[0].group_folder).toBe('other-group');
  });

  it('non-main group cannot schedule for another group', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'unauthorized',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'main@g.us',
      },
      'other-group',
      false,
      deps,
    );

    const allTasks = getAllTasks();
    expect(allTasks.length).toBe(0);
  });

  it('rejects schedule_task for unregistered target JID', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'no target',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'unknown@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const allTasks = getAllTasks();
    expect(allTasks.length).toBe(0);
  });
});

// --- schedule_task provenance (created_by_role) ---

describe('schedule_task provenance', () => {
  it('main group schedule_task writes role=main_agent', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'main-scheduled task',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'main@g.us',
      },
      'whatsapp_main',
      true, // isMain
      deps,
    );
    const tasks = getAllTasks();
    expect(tasks.length).toBe(1);
    expect(tasks[0].created_by_role).toBe('main_agent');
  });

  it('trusted non-main group schedule_task writes role=trusted_agent', async () => {
    // Promote OTHER_GROUP to trusted for this test via a local override
    groups['other@g.us'] = {
      ...OTHER_GROUP,
      containerConfig: { trusted: true },
    };
    setRegisteredGroup('other@g.us', groups['other@g.us']);
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'trusted-scheduled task',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'other@g.us',
      },
      'other-group',
      false, // isMain
      deps,
    );
    const tasks = getAllTasks();
    expect(tasks.length).toBe(1);
    expect(tasks[0].created_by_role).toBe('trusted_agent');
  });

  it('untrusted non-main group schedule_task writes role=untrusted_agent', async () => {
    // OTHER_GROUP has no containerConfig → untrusted by default
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'untrusted-scheduled task',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'other@g.us',
      },
      'other-group',
      false, // isMain
      deps,
    );
    const tasks = getAllTasks();
    expect(tasks.length).toBe(1);
    expect(tasks[0].created_by_role).toBe('untrusted_agent');
  });

  it('payload field cannot spoof the role (security boundary test)', async () => {
    // An untrusted agent MUST NOT be able to claim 'owner' or 'main_agent'
    // by putting it in the IPC payload. The derivation uses the VERIFIED
    // source group's trust tier, not any payload field. Cast-to-Parameters
    // bypasses TS's own protection (which already rejects these fields at
    // compile time) so we can test the runtime behavior on a malicious
    // payload that would arrive as raw JSON from a compromised container.
    const maliciousPayload = {
      type: 'schedule_task',
      prompt: 'spoof attempt',
      schedule_type: 'once',
      schedule_value: '2025-06-01T00:00:00',
      targetJid: 'other@g.us',
      // Intentionally-malicious fields:
      created_by_role: 'owner',
      createdByRole: 'owner',
      role: 'main_agent',
    } as unknown as Parameters<typeof processTaskIpc>[0];
    await processTaskIpc(maliciousPayload, 'other-group', false, deps);
    const tasks = getAllTasks();
    expect(tasks.length).toBe(1);
    expect(tasks[0].created_by_role).toBe('untrusted_agent');
  });
});

// --- schedule_task / update_task prompt coercion (#512) ---
//
// Without coercion at the IPC boundary, a non-string `prompt` (or
// `script`) here lands as a SQLite BLOB via better-sqlite3, and
// `list_tasks` later throws `t.prompt.slice is not a function`. The
// boundary contract: TEXT passes through, the JSON-Buffer shape
// (`{type:'Buffer',data:[...]}`) decodes to UTF-8 so the real prompt
// round-trips, every other non-string shape is REJECTED — better to
// drop the IPC than to persist a garbage row that fires with
// `"[object Object]"` as its prompt.

describe('schedule_task prompt coercion (#512)', () => {
  it('decodes a JSON-Buffer-shaped prompt to its UTF-8 text before persisting', async () => {
    const promptText = 'Skill(skill: "tessl__axis-review")';
    const blobShape = {
      type: 'Buffer',
      data: Array.from(Buffer.from(promptText, 'utf8')),
    };
    await processTaskIpc(
      {
        type: 'schedule_task',
        // Cast through unknown — the TS type says `string`, but at
        // runtime IPC payloads are JSON and a malformed writer can
        // deliver any shape. The Buffer-shape MUST round-trip to its
        // decoded text, not to "[object Object]".
        prompt: blobShape as unknown as string,
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'main@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const tasks = getAllTasks();
    expect(tasks.length).toBe(1);
    expect(typeof tasks[0].prompt).toBe('string');
    expect(tasks[0].prompt).toBe(promptText);
  });

  it('rejects a non-string non-Buffer prompt rather than persisting "[object Object]"', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        // A plain object with no Buffer discriminator — the prior
        // bare-String fallback would have stored "[object Object]"
        // here, which fires with garbage at the next tick. Reject.
        prompt: { not: 'a buffer' } as unknown as string,
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'main@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const tasks = getAllTasks();
    expect(tasks.length).toBe(0);
  });

  it('update_task decodes a JSON-Buffer-shaped prompt to UTF-8', async () => {
    const promptText = 'Skill(skill: "tessl__nightly-housekeeping")';
    createTask({
      id: 'task-update-coerce',
      group_folder: 'whatsapp_main',
      chat_jid: 'main@g.us',
      prompt: 'original',
      schedule_type: 'once',
      schedule_value: '2025-06-01T00:00:00',
      context_mode: 'isolated',
      next_run: '2025-06-01T00:00:00.000Z',
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'main_agent',
    });
    await processTaskIpc(
      {
        type: 'update_task',
        taskId: 'task-update-coerce',
        prompt: {
          type: 'Buffer',
          data: Array.from(Buffer.from(promptText, 'utf8')),
        } as unknown as string,
      },
      'whatsapp_main',
      true,
      deps,
    );
    const updated = getTaskById('task-update-coerce');
    expect(updated).toBeDefined();
    expect(typeof updated!.prompt).toBe('string');
    expect(updated!.prompt).toBe(promptText);
  });

  it('update_task rejects a non-string non-Buffer prompt and leaves the row untouched', async () => {
    createTask({
      id: 'task-update-reject',
      group_folder: 'whatsapp_main',
      chat_jid: 'main@g.us',
      prompt: 'original',
      schedule_type: 'once',
      schedule_value: '2025-06-01T00:00:00',
      context_mode: 'isolated',
      next_run: '2025-06-01T00:00:00.000Z',
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'main_agent',
    });
    await processTaskIpc(
      {
        type: 'update_task',
        taskId: 'task-update-reject',
        prompt: 42 as unknown as string,
      },
      'whatsapp_main',
      true,
      deps,
    );
    const updated = getTaskById('task-update-reject');
    expect(updated).toBeDefined();
    expect(updated!.prompt).toBe('original');
  });
});

// --- pause_task authorization ---

describe('pause_task authorization', () => {
  beforeEach(() => {
    createTask({
      id: 'task-main',
      group_folder: 'whatsapp_main',
      chat_jid: 'main@g.us',
      prompt: 'main task',
      schedule_type: 'once',
      schedule_value: '2025-06-01T00:00:00',
      context_mode: 'isolated',
      next_run: '2025-06-01T00:00:00.000Z',
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
    createTask({
      id: 'task-other',
      group_folder: 'other-group',
      chat_jid: 'other@g.us',
      prompt: 'other task',
      schedule_type: 'once',
      schedule_value: '2025-06-01T00:00:00',
      context_mode: 'isolated',
      next_run: '2025-06-01T00:00:00.000Z',
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
  });

  it('main group can pause any task', async () => {
    await processTaskIpc(
      { type: 'pause_task', taskId: 'task-other' },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-other')!.status).toBe('paused');
  });

  it('non-main group can pause its own task', async () => {
    await processTaskIpc(
      { type: 'pause_task', taskId: 'task-other' },
      'other-group',
      false,
      deps,
    );
    expect(getTaskById('task-other')!.status).toBe('paused');
  });

  it('non-main group cannot pause another groups task', async () => {
    await processTaskIpc(
      { type: 'pause_task', taskId: 'task-main' },
      'other-group',
      false,
      deps,
    );
    expect(getTaskById('task-main')!.status).toBe('active');
  });
});

// --- resume_task authorization ---

describe('resume_task authorization', () => {
  beforeEach(() => {
    createTask({
      id: 'task-paused',
      group_folder: 'other-group',
      chat_jid: 'other@g.us',
      prompt: 'paused task',
      schedule_type: 'once',
      schedule_value: '2025-06-01T00:00:00',
      context_mode: 'isolated',
      next_run: '2025-06-01T00:00:00.000Z',
      status: 'paused',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
  });

  it('main group can resume any task', async () => {
    await processTaskIpc(
      { type: 'resume_task', taskId: 'task-paused' },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-paused')!.status).toBe('active');
  });

  it('non-main group can resume its own task', async () => {
    await processTaskIpc(
      { type: 'resume_task', taskId: 'task-paused' },
      'other-group',
      false,
      deps,
    );
    expect(getTaskById('task-paused')!.status).toBe('active');
  });

  it('non-main group cannot resume another groups task', async () => {
    await processTaskIpc(
      { type: 'resume_task', taskId: 'task-paused' },
      'third-group',
      false,
      deps,
    );
    expect(getTaskById('task-paused')!.status).toBe('paused');
  });
});

// --- cancel_task authorization ---

describe('cancel_task authorization', () => {
  it('main group can cancel any task', async () => {
    createTask({
      id: 'task-to-cancel',
      group_folder: 'other-group',
      chat_jid: 'other@g.us',
      prompt: 'cancel me',
      schedule_type: 'once',
      schedule_value: '2025-06-01T00:00:00',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    await processTaskIpc(
      { type: 'cancel_task', taskId: 'task-to-cancel' },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-to-cancel')).toBeUndefined();
  });

  it('non-main group can cancel its own task', async () => {
    createTask({
      id: 'task-own',
      group_folder: 'other-group',
      chat_jid: 'other@g.us',
      prompt: 'my task',
      schedule_type: 'once',
      schedule_value: '2025-06-01T00:00:00',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    await processTaskIpc(
      { type: 'cancel_task', taskId: 'task-own' },
      'other-group',
      false,
      deps,
    );
    expect(getTaskById('task-own')).toBeUndefined();
  });

  it('non-main group cannot cancel another groups task', async () => {
    createTask({
      id: 'task-foreign',
      group_folder: 'whatsapp_main',
      chat_jid: 'main@g.us',
      prompt: 'not yours',
      schedule_type: 'once',
      schedule_value: '2025-06-01T00:00:00',
      context_mode: 'isolated',
      next_run: null,
      status: 'active',
      created_at: '2024-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    await processTaskIpc(
      { type: 'cancel_task', taskId: 'task-foreign' },
      'other-group',
      false,
      deps,
    );
    expect(getTaskById('task-foreign')).toBeDefined();
  });
});

// --- register_group authorization ---

describe('register_group authorization', () => {
  it('non-main group cannot register a group', async () => {
    await processTaskIpc(
      {
        type: 'register_group',
        jid: 'new@g.us',
        name: 'New Group',
        folder: 'new-group',
        trigger: '@Andy',
      },
      'other-group',
      false,
      deps,
    );

    // registeredGroups should not have changed
    expect(groups['new@g.us']).toBeUndefined();
  });

  it('main group cannot register with unsafe folder path', async () => {
    await processTaskIpc(
      {
        type: 'register_group',
        jid: 'new@g.us',
        name: 'New Group',
        folder: '../../outside',
        trigger: '@Andy',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(groups['new@g.us']).toBeUndefined();
  });
});

// --- refresh_groups authorization ---

describe('refresh_groups authorization', () => {
  it('non-main group cannot trigger refresh', async () => {
    // This should be silently blocked (no crash, no effect)
    await processTaskIpc(
      { type: 'refresh_groups' },
      'other-group',
      false,
      deps,
    );
    // If we got here without error, the auth gate worked
  });
});

// --- IPC message authorization ---
// Tests the authorization pattern from startIpcWatcher (ipc.ts).
// The logic: isMain || (targetGroup && targetGroup.folder === sourceGroup)

describe('IPC message authorization', () => {
  // Replicate the exact check from the IPC watcher
  function isMessageAuthorized(
    sourceGroup: string,
    isMain: boolean,
    targetChatJid: string,
    registeredGroups: Record<string, RegisteredGroup>,
  ): boolean {
    const targetGroup = registeredGroups[targetChatJid];
    return isMain || (!!targetGroup && targetGroup.folder === sourceGroup);
  }

  it('main group can send to any group', () => {
    expect(
      isMessageAuthorized('whatsapp_main', true, 'other@g.us', groups),
    ).toBe(true);
    expect(
      isMessageAuthorized('whatsapp_main', true, 'third@g.us', groups),
    ).toBe(true);
  });

  it('non-main group can send to its own chat', () => {
    expect(
      isMessageAuthorized('other-group', false, 'other@g.us', groups),
    ).toBe(true);
  });

  it('non-main group cannot send to another groups chat', () => {
    expect(isMessageAuthorized('other-group', false, 'main@g.us', groups)).toBe(
      false,
    );
    expect(
      isMessageAuthorized('other-group', false, 'third@g.us', groups),
    ).toBe(false);
  });

  it('non-main group cannot send to unregistered JID', () => {
    expect(
      isMessageAuthorized('other-group', false, 'unknown@g.us', groups),
    ).toBe(false);
  });

  it('main group can send to unregistered JID', () => {
    // Main is always authorized regardless of target
    expect(
      isMessageAuthorized('whatsapp_main', true, 'unknown@g.us', groups),
    ).toBe(true);
  });
});

// --- schedule_task with cron and interval types ---

describe('schedule_task schedule types', () => {
  it('creates task with cron schedule and computes next_run', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'cron task',
        schedule_type: 'cron',
        schedule_value: '0 9 * * *', // every day at 9am
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_type).toBe('cron');
    expect(tasks[0].next_run).toBeTruthy();
    // next_run should be a valid ISO date in the future
    expect(new Date(tasks[0].next_run!).getTime()).toBeGreaterThan(
      Date.now() - 60000,
    );
  });

  it('rejects invalid cron expression', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'bad cron',
        schedule_type: 'cron',
        schedule_value: 'not a cron',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getAllTasks()).toHaveLength(0);
  });

  it('creates task with interval schedule', async () => {
    const before = Date.now();

    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'interval task',
        schedule_type: 'interval',
        schedule_value: '3600000', // 1 hour
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_type).toBe('interval');
    // next_run should be ~1 hour from now
    const nextRun = new Date(tasks[0].next_run!).getTime();
    expect(nextRun).toBeGreaterThanOrEqual(before + 3600000 - 1000);
    expect(nextRun).toBeLessThanOrEqual(Date.now() + 3600000 + 1000);
  });

  it('rejects invalid interval (non-numeric)', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'bad interval',
        schedule_type: 'interval',
        schedule_value: 'abc',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getAllTasks()).toHaveLength(0);
  });

  it('rejects invalid interval (zero)', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'zero interval',
        schedule_type: 'interval',
        schedule_value: '0',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getAllTasks()).toHaveLength(0);
  });

  it('rejects invalid once timestamp', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'bad once',
        schedule_type: 'once',
        schedule_value: 'not-a-date',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getAllTasks()).toHaveLength(0);
  });
});

// --- #102: UTC schedule_value + timezone parameter ---

describe('schedule_task with UTC schedule_value (#102)', () => {
  it('once with Z-suffix is anchored to that exact UTC instant', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'utc once',
        schedule_type: 'once',
        schedule_value: '2030-01-01T12:00:00Z',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    // next_run is normalized to ISO; equality of the underlying instant
    // is what matters — not the literal string.
    expect(new Date(tasks[0].next_run!).toISOString()).toBe(
      '2030-01-01T12:00:00.000Z',
    );
    // schedule_timezone is not used for `once` — left null.
    expect(tasks[0].schedule_timezone).toBeFalsy();
  });

  it('local-time once (no suffix) still works for back-compat', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'local once',
        schedule_type: 'once',
        schedule_value: '2030-01-01T12:00:00',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    // The promised compat behaviour: a no-suffix local string is
    // interpreted in the host's CURRENT tz at schedule time and pinned
    // to that absolute UTC instant. The simplest tz-portable assertion:
    // construct a `Date` from the original local string the same way
    // the host does, then verify next_run matches that exact instant.
    // `toLocaleString` with explicit format options would also work but
    // varies subtly across Node/ICU versions (en-CA punctuation, etc.)
    // and is overkill for what we're really checking.
    const expectedInstant = new Date('2030-01-01T12:00:00').toISOString();
    expect(tasks[0].next_run).toBe(expectedInstant);
  });

  it('cron with explicit timezone persists schedule_timezone', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'utc cron',
        schedule_type: 'cron',
        schedule_value: '0 12 * * *',
        timezone: 'UTC',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_timezone).toBe('UTC');
    expect(tasks[0].next_run).toBeTruthy();
    // next fire is at 12:00 UTC on some date — minute and hour in UTC
    // should be 0 and 12.
    const nextDate = new Date(tasks[0].next_run!);
    expect(nextDate.getUTCMinutes()).toBe(0);
    expect(nextDate.getUTCHours()).toBe(12);
  });

  it('cron with America/Chicago timezone persists schedule_timezone', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'chicago cron',
        schedule_type: 'cron',
        schedule_value: '0 9 * * *',
        timezone: 'America/Chicago',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_timezone).toBe('America/Chicago');
  });

  it('cron without timezone leaves schedule_timezone null (server default)', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'default tz cron',
        schedule_type: 'cron',
        schedule_value: '0 9 * * *',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_timezone).toBeFalsy();
  });

  it('rejects invalid IANA timezone', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'bad tz',
        schedule_type: 'cron',
        schedule_value: '0 9 * * *',
        timezone: 'Not/A/Real/Zone',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getAllTasks()).toHaveLength(0);
  });

  it('schedule_timezone is forced null for once-tasks even if timezone passed', async () => {
    // A timezone value on a non-cron schedule would persist and silently
    // start affecting cron evaluation if the task is later updated to
    // schedule_type: 'cron' without re-passing timezone — Copilot review
    // flagged this as a footgun. Drop it at schedule time.
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'once with stray tz',
        schedule_type: 'once',
        schedule_value: '2030-01-01T12:00:00Z',
        timezone: 'America/Chicago',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_timezone).toBeFalsy();
  });

  it('schedule_timezone is forced null for interval-tasks even if timezone passed', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'interval with stray tz',
        schedule_type: 'interval',
        schedule_value: '3600000',
        timezone: 'Europe/Berlin',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_timezone).toBeFalsy();
  });

  it('update_task recomputes next_run when once-task schedule_value changes', async () => {
    // Seed a once-task with one timestamp, then update it to a later one
    // and verify next_run actually moves. Without the once branch in
    // update_task's recompute, next_run stayed at the original instant
    // and the task fired at the wrong time.
    await processTaskIpc(
      {
        type: 'schedule_task',
        taskId: 'once-update-test',
        prompt: 'once update',
        schedule_type: 'once',
        schedule_value: '2030-01-01T12:00:00Z',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    await processTaskIpc(
      {
        type: 'update_task',
        taskId: 'once-update-test',
        schedule_value: '2030-06-01T18:30:00Z',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const updated = getTaskById('once-update-test');
    expect(updated?.next_run).toBe('2030-06-01T18:30:00.000Z');
  });

  it('update_task rejects invalid once timestamp without breaking existing row', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        taskId: 'once-bad-update',
        prompt: 'once',
        schedule_type: 'once',
        schedule_value: '2030-01-01T12:00:00Z',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    await processTaskIpc(
      {
        type: 'update_task',
        taskId: 'once-bad-update',
        schedule_value: 'not-a-date',
      },
      'whatsapp_main',
      true,
      deps,
    );

    // Original next_run should be intact — invalid update is a no-op.
    expect(getTaskById('once-bad-update')?.next_run).toBe(
      '2030-01-01T12:00:00.000Z',
    );
  });
});

// --- #456: travel-anchored cadence via schedule_timezone='local' ---

describe("schedule_task with schedule_timezone='local' (#456)", () => {
  it("persists 'local' verbatim and resolves next_run against tz_state.current_tz", async () => {
    // Pre-#456 the IPC handler ran every non-null timezone through
    // isValidTimezone(); 'local' threw inside Intl.DateTimeFormat and
    // the handler logged "Invalid IANA timezone for schedule_task" and
    // aborted. Now it's accepted as a literal token, the row stores
    // 'local' (not the resolved zone — the resolution happens at fire
    // time), and the initial next_run matches what cron-parser would
    // produce for the seeded current_tz.
    _seedTzStateForTests({ currentTz: 'America/Chicago' });

    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'travel-anchored cron',
        schedule_type: 'cron',
        schedule_value: '0 7 * * *',
        timezone: 'local',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_timezone).toBe('local');
    // 7am America/Chicago in UTC: standard time → 13:00, DST → 12:00.
    // Either is correct depending on date; assert the UTC hour is
    // one of the two valid values rather than pinning a calendar date.
    const next = new Date(tasks[0].next_run!);
    expect([12, 13]).toContain(next.getUTCHours());
    expect(next.getUTCMinutes()).toBe(0);
  });

  it('falls back to TIMEZONE when tz_state is empty (no row yet)', async () => {
    // First-run shape: agent calls schedule_task with 'local' before
    // task-tz-sync has populated tz_state. The row still persists with
    // schedule_timezone='local' (so it'll start travelling once
    // tz_state lands), but the initial next_run is computed against
    // TIMEZONE — same fallback NULL schedule_timezone uses.
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'local before tz_state',
        schedule_type: 'cron',
        schedule_value: '0 7 * * *',
        timezone: 'local',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_timezone).toBe('local');
    expect(tasks[0].next_run).toBeTruthy();
  });

  it('falls back to TIMEZONE when tz_state.current_tz is corrupt', async () => {
    // Defensive: a task-tz-sync bug or bad import could write garbage
    // into tz_state.current_tz. Pre-validation in resolveCronTz routes
    // that case to TIMEZONE rather than letting cron-parser throw and
    // bricking the schedule call. The row still stores 'local' so the
    // next valid task-tz-sync write recovers the row at fire time.
    _seedTzStateForTests({ currentTz: 'NotAZone' });

    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'local with bad tz_state',
        schedule_type: 'cron',
        schedule_value: '0 7 * * *',
        timezone: 'local',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_timezone).toBe('local');
    expect(tasks[0].next_run).toBeTruthy();
  });

  it("forces schedule_timezone null on once-tasks even with timezone='local'", async () => {
    // 'local' is meaningless for a once-task (the instant is already
    // pinned to a specific UTC moment). Same drop-silently rule that
    // applies to pinned IANA on once/interval — the column is nulled
    // at write time so a later type flip to cron doesn't silently
    // re-activate a stray 'local'.
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'once with stray local',
        schedule_type: 'once',
        schedule_value: '2030-01-01T12:00:00Z',
        timezone: 'local',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks).toHaveLength(1);
    expect(tasks[0].schedule_timezone).toBeFalsy();
  });

  it("update_task can set schedule_timezone to 'local' on an existing cron row", async () => {
    _seedTzStateForTests({ currentTz: 'Europe/Amsterdam' });

    await processTaskIpc(
      {
        type: 'schedule_task',
        taskId: 'tz-local-update',
        prompt: 'pinned then travelling',
        schedule_type: 'cron',
        schedule_value: '0 9 * * *',
        timezone: 'America/New_York',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('tz-local-update')?.schedule_timezone).toBe(
      'America/New_York',
    );

    await processTaskIpc(
      {
        type: 'update_task',
        taskId: 'tz-local-update',
        timezone: 'local',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const updated = getTaskById('tz-local-update');
    expect(updated?.schedule_timezone).toBe('local');
    // next_run recomputed against the seeded current_tz — 9am Amsterdam.
    // Standard time → 08:00 UTC, DST → 07:00 UTC.
    const next = new Date(updated!.next_run!);
    expect([7, 8]).toContain(next.getUTCHours());
  });

  it("update_task can clear schedule_timezone='local' back to null", async () => {
    _seedTzStateForTests({ currentTz: 'America/Chicago' });

    await processTaskIpc(
      {
        type: 'schedule_task',
        taskId: 'tz-local-clear',
        prompt: 'travelling then pinned',
        schedule_type: 'cron',
        schedule_value: '0 7 * * *',
        timezone: 'local',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    await processTaskIpc(
      {
        type: 'update_task',
        taskId: 'tz-local-clear',
        timezone: '',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getTaskById('tz-local-clear')?.schedule_timezone).toBeFalsy();
  });
});

// --- context_mode defaulting ---

describe('schedule_task context_mode', () => {
  it('accepts context_mode=group', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'group context',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        context_mode: 'group',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks[0].context_mode).toBe('group');
  });

  it('accepts context_mode=isolated', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'isolated context',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        context_mode: 'isolated',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks[0].context_mode).toBe('isolated');
  });

  it('defaults invalid context_mode to isolated', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'bad context',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        context_mode: 'bogus' as any,
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks[0].context_mode).toBe('isolated');
  });

  it('defaults missing context_mode to isolated', async () => {
    await processTaskIpc(
      {
        type: 'schedule_task',
        prompt: 'no context mode',
        schedule_type: 'once',
        schedule_value: '2025-06-01T00:00:00',
        targetJid: 'other@g.us',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const tasks = getAllTasks();
    expect(tasks[0].context_mode).toBe('isolated');
  });
});

// --- register_group success path ---

describe('register_group success', () => {
  it('main group can register a new group', async () => {
    await processTaskIpc(
      {
        type: 'register_group',
        jid: 'new@g.us',
        name: 'New Group',
        folder: 'new-group',
        trigger: '@Andy',
      },
      'whatsapp_main',
      true,
      deps,
    );

    // Verify group was registered in DB
    const group = getRegisteredGroup('new@g.us');
    expect(group).toBeDefined();
    expect(group!.name).toBe('New Group');
    expect(group!.folder).toBe('new-group');
    expect(group!.trigger).toBe('@Andy');
  });

  it('register_group rejects request with missing fields', async () => {
    await processTaskIpc(
      {
        type: 'register_group',
        jid: 'partial@g.us',
        name: 'Partial',
        // missing folder and trigger
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('partial@g.us')).toBeUndefined();
  });

  it('register_group does NOT create a heartbeat for a non-main group, regardless of requiresTrigger or enableHeartbeat (#158, #453)', async () => {
    // Pre-#158, a `requiresTrigger !== false` non-main group got an
    // auto-created heartbeat. Pre-#453, an explicit
    // `containerConfig.enableHeartbeat: true` opt-in created a
    // `tessl__check-unanswered`-driven heartbeat. With check-unanswered
    // retired in `jbaruch/nanoclaw-core#38`, both code paths are gone:
    // non-main groups never get a `heartbeat-<folder>` row from
    // registerGroup. The flag stays in the schema for backwards
    // compatibility but is a no-op there.
    await processTaskIpc(
      {
        type: 'register_group',
        jid: 'silent@g.us',
        name: 'Silent',
        folder: 'silent-group',
        trigger: '@Andy',
        requiresTrigger: true,
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('silent@g.us')).toBeDefined();
    expect(getRegisteredGroup('silent@g.us')?.requiresTrigger).toBe(true);
    expect(getTaskById('heartbeat-silent-group')).toBeUndefined();

    // Explicit opt-in via `enableHeartbeat: true` is also a no-op
    // post-#453. Flag value is preserved on the registered group; no
    // heartbeat task row gets created.
    await processTaskIpc(
      {
        type: 'register_group',
        jid: 'beating@g.us',
        name: 'Beating',
        folder: 'beating-group',
        trigger: '@Andy',
        containerConfig: { enableHeartbeat: true },
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('beating@g.us')?.containerConfig).toEqual({
      enableHeartbeat: true,
    });
    expect(getTaskById('heartbeat-beating-group')).toBeUndefined();
  });
});

// --- unregister_group (#159) ---

describe('unregister_group authorization', () => {
  it('non-main group is rejected', async () => {
    await processTaskIpc(
      { type: 'unregister_group', jid: 'other@g.us' },
      'other-group',
      false,
      deps,
    );

    // Group still registered — unauthorized call rejected before any
    // mutation.
    expect(getRegisteredGroup('other@g.us')).toBeDefined();
    expect(groups['other@g.us']).toBeDefined();
  });
});

describe('unregister_group success', () => {
  it('main group can unregister a non-main group from both stores', async () => {
    expect(getRegisteredGroup('other@g.us')).toBeDefined();

    await processTaskIpc(
      { type: 'unregister_group', jid: 'other@g.us' },
      'whatsapp_main',
      true,
      deps,
    );

    // DB row gone
    expect(getRegisteredGroup('other@g.us')).toBeUndefined();
    // In-memory mirror gone — subsequent routing decisions stop
    // treating the JID as registered before any restart.
    expect(groups['other@g.us']).toBeUndefined();
  });

  it('trims whitespace-padded jid before lookup', async () => {
    expect(getRegisteredGroup('other@g.us')).toBeDefined();

    await processTaskIpc(
      { type: 'unregister_group', jid: '  other@g.us  ' },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('other@g.us')).toBeUndefined();
  });

  it('refuses to unregister a main group', async () => {
    setRegisteredGroup('main@g.us', {
      name: 'Main',
      folder: 'main',
      trigger: '@Andy',
      added_at: '2026-01-01',
      isMain: true,
    });
    groups['main@g.us'] = {
      name: 'Main',
      folder: 'main',
      trigger: '@Andy',
      added_at: '2026-01-01',
      isMain: true,
    };

    await processTaskIpc(
      { type: 'unregister_group', jid: 'main@g.us' },
      'whatsapp_main',
      true,
      deps,
    );

    // Main row preserved — losing it mid-runtime would leave the
    // orchestrator without any path to recreate it via IPC.
    expect(getRegisteredGroup('main@g.us')).toBeDefined();
    expect(groups['main@g.us']).toBeDefined();
  });

  it('rejects request with missing jid', async () => {
    await processTaskIpc(
      { type: 'unregister_group' },
      'whatsapp_main',
      true,
      deps,
    );

    // No mutation — sentinel group still registered.
    expect(getRegisteredGroup('other@g.us')).toBeDefined();
  });

  it('is a no-op for an unregistered jid (idempotent)', async () => {
    await processTaskIpc(
      { type: 'unregister_group', jid: 'never-registered@g.us' },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('never-registered@g.us')).toBeUndefined();
    // Sibling registrations untouched.
    expect(getRegisteredGroup('other@g.us')).toBeDefined();
  });

  it('cascade-deletes scheduled tasks tied to the unregistered folder', async () => {
    // Pre-state: a heartbeat-style scheduled task and an unrelated
    // one-off task exist for the group folder. Unregister must clear
    // BOTH so the scheduler doesn't keep firing them every cycle and
    // logging "Group not found for task" noise. A sibling group's
    // task is a control: it must survive untouched.
    //
    // Fixed timestamps per testing-standards (`Provide fixed test
    // data; never have the test generate its own inputs randomly`) —
    // runtime-derived clock values would make the row contents
    // non-deterministic across runs.
    const FIXED_CREATED_AT = '2026-01-01T00:00:00.000Z';
    const FIXED_NEXT_RUN_15MIN = '2026-01-01T00:15:00.000Z';
    const FIXED_NEXT_RUN_ONCE = '2026-12-01T00:00:00.000Z';
    createTask({
      id: 'heartbeat-other-group',
      group_folder: 'other-group',
      chat_jid: 'other@g.us',
      prompt: 'mock-heartbeat-prompt',
      schedule_type: 'cron',
      schedule_value: '*/15 * * * *',
      context_mode: 'isolated',
      next_run: FIXED_NEXT_RUN_15MIN,
      status: 'active',
      created_at: FIXED_CREATED_AT,
      created_by_role: 'owner',
    });
    createTask({
      id: 'oneoff-other-group',
      group_folder: 'other-group',
      chat_jid: 'other@g.us',
      prompt: 'do the thing',
      schedule_type: 'once',
      schedule_value: FIXED_NEXT_RUN_ONCE,
      context_mode: 'group',
      next_run: FIXED_NEXT_RUN_ONCE,
      status: 'active',
      created_at: FIXED_CREATED_AT,
      created_by_role: 'main_agent',
    });
    createTask({
      id: 'sibling-third-group',
      group_folder: 'third-group',
      chat_jid: 'third@g.us',
      prompt: 'unrelated',
      schedule_type: 'once',
      schedule_value: FIXED_NEXT_RUN_ONCE,
      context_mode: 'group',
      next_run: FIXED_NEXT_RUN_ONCE,
      status: 'active',
      created_at: FIXED_CREATED_AT,
      created_by_role: 'main_agent',
    });
    expect(getTaskById('heartbeat-other-group')).toBeDefined();
    expect(getTaskById('oneoff-other-group')).toBeDefined();
    expect(getTaskById('sibling-third-group')).toBeDefined();

    await processTaskIpc(
      { type: 'unregister_group', jid: 'other@g.us' },
      'whatsapp_main',
      true,
      deps,
    );

    // Tasks for the unregistered folder are gone…
    expect(getTaskById('heartbeat-other-group')).toBeUndefined();
    expect(getTaskById('oneoff-other-group')).toBeUndefined();
    // …sibling group's task survives.
    expect(getTaskById('sibling-third-group')).toBeDefined();
    // Registration itself was removed too.
    expect(getRegisteredGroup('other@g.us')).toBeUndefined();
  });
});

// --- set_trusted / set_trigger (#105) ---

describe('set_trusted', () => {
  it('main group can flip trusted on a registered group', async () => {
    await processTaskIpc(
      { type: 'set_trusted', jid: 'other@g.us', trusted: true },
      'whatsapp_main',
      true,
      deps,
    );

    const group = getRegisteredGroup('other@g.us');
    expect(group?.containerConfig?.trusted).toBe(true);
    // Other fields preserved
    expect(group?.trigger).toBe('@Andy');
    expect(group?.folder).toBe('other-group');
  });

  it('main group can flip trusted back to false', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: { trusted: true },
    });
    await processTaskIpc(
      { type: 'set_trusted', jid: 'other@g.us', trusted: false },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('other@g.us')?.containerConfig?.trusted).toBe(
      false,
    );
  });

  it('non-main group cannot flip trusted', async () => {
    await processTaskIpc(
      { type: 'set_trusted', jid: 'other@g.us', trusted: true },
      'other-group',
      false,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.trusted,
    ).toBeUndefined();
  });

  it('set_trusted on unregistered jid is a no-op (no DB row created)', async () => {
    await processTaskIpc(
      { type: 'set_trusted', jid: 'never-registered@g.us', trusted: true },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('never-registered@g.us')).toBeUndefined();
  });

  it('set_trusted preserves additionalMounts and other containerConfig fields', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: {
        trusted: false,
        additionalMounts: [
          { hostPath: '/tmp/extra', containerPath: 'extra', readonly: true },
        ],
      },
    });
    await processTaskIpc(
      { type: 'set_trusted', jid: 'other@g.us', trusted: true },
      'whatsapp_main',
      true,
      deps,
    );

    const group = getRegisteredGroup('other@g.us');
    expect(group?.containerConfig?.trusted).toBe(true);
    expect(group?.containerConfig?.additionalMounts).toEqual([
      { hostPath: '/tmp/extra', containerPath: 'extra', readonly: true },
    ]);
  });

  it('rejects empty/whitespace JID', async () => {
    await processTaskIpc(
      { type: 'set_trusted', jid: '  ', trusted: true },
      'whatsapp_main',
      true,
      deps,
    );

    // No row should be modified — OTHER_GROUP has no containerConfig
    // and an empty-jid call shouldn't have created one.
    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.trusted,
    ).toBeUndefined();
  });

  it('trims surrounding whitespace from the JID before lookup', async () => {
    await processTaskIpc(
      { type: 'set_trusted', jid: '  other@g.us  ', trusted: true },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('other@g.us')?.containerConfig?.trusted).toBe(
      true,
    );
  });
});

describe('set_trigger', () => {
  it('main group can change trigger on a registered group', async () => {
    await processTaskIpc(
      { type: 'set_trigger', jid: 'other@g.us', trigger: '@NewName' },
      'whatsapp_main',
      true,
      deps,
    );

    const group = getRegisteredGroup('other@g.us');
    expect(group?.trigger).toBe('@NewName');
    // Other fields preserved
    expect(group?.folder).toBe('other-group');
  });

  it('main group can update trigger and requiresTrigger together', async () => {
    await processTaskIpc(
      {
        type: 'set_trigger',
        jid: 'other@g.us',
        trigger: '@NewName',
        requiresTrigger: true,
      },
      'whatsapp_main',
      true,
      deps,
    );

    const group = getRegisteredGroup('other@g.us');
    expect(group?.trigger).toBe('@NewName');
    expect(group?.requiresTrigger).toBe(true);
  });

  it('set_trigger leaves requiresTrigger untouched when omitted', async () => {
    setRegisteredGroup('other@g.us', { ...OTHER_GROUP, requiresTrigger: true });
    await processTaskIpc(
      { type: 'set_trigger', jid: 'other@g.us', trigger: '@Andy2' },
      'whatsapp_main',
      true,
      deps,
    );

    const group = getRegisteredGroup('other@g.us');
    expect(group?.trigger).toBe('@Andy2');
    expect(group?.requiresTrigger).toBe(true);
  });

  it('non-main group cannot change trigger', async () => {
    await processTaskIpc(
      { type: 'set_trigger', jid: 'other@g.us', trigger: '@Hijack' },
      'other-group',
      false,
      deps,
    );

    expect(getRegisteredGroup('other@g.us')?.trigger).toBe('@Andy');
  });

  it('set_trigger rejects empty trigger (would silently revert to default)', async () => {
    await processTaskIpc(
      { type: 'set_trigger', jid: 'other@g.us', trigger: '' },
      'whatsapp_main',
      true,
      deps,
    );

    // Trigger should remain @Andy from beforeEach setup.
    expect(getRegisteredGroup('other@g.us')?.trigger).toBe('@Andy');
  });

  it('set_trigger rejects whitespace-only trigger', async () => {
    await processTaskIpc(
      { type: 'set_trigger', jid: 'other@g.us', trigger: '   \t\n  ' },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('other@g.us')?.trigger).toBe('@Andy');
  });

  it('set_trigger trims surrounding whitespace from the stored trigger', async () => {
    await processTaskIpc(
      { type: 'set_trigger', jid: 'other@g.us', trigger: '  @Trimmed  ' },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('other@g.us')?.trigger).toBe('@Trimmed');
  });

  it('set_trigger trims surrounding whitespace from the JID before lookup', async () => {
    // Without trimming, a whitespace-padded JID would never match the
    // registry key and the caller would see a misleading
    // "group not registered" warning.
    await processTaskIpc(
      { type: 'set_trigger', jid: '  other@g.us  ', trigger: '@PaddedJid' },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('other@g.us')?.trigger).toBe('@PaddedJid');
  });

  it('set_trigger on unregistered jid is a no-op', async () => {
    await processTaskIpc(
      {
        type: 'set_trigger',
        jid: 'never-registered@g.us',
        trigger: '@Whatever',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getRegisteredGroup('never-registered@g.us')).toBeUndefined();
  });

  it('set_trigger flipping requiresTrigger false→true does NOT create a heartbeat task', async () => {
    // Heartbeat lifecycle is orthogonal to trigger config (#158) — the
    // only path that creates a non-main heartbeat is registerGroup with
    // `containerConfig.enableHeartbeat`. Flipping the trigger flag must
    // not have a side effect on scheduled tasks.
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      requiresTrigger: false,
    });
    groups['other@g.us'] = { ...OTHER_GROUP, requiresTrigger: false };
    expect(getTaskById('heartbeat-other-group')).toBeUndefined();

    await processTaskIpc(
      {
        type: 'set_trigger',
        jid: 'other@g.us',
        trigger: '@Andy',
        requiresTrigger: true,
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(getTaskById('heartbeat-other-group')).toBeUndefined();
  });

  it('set_trigger flipping requiresTrigger true→false leaves an existing heartbeat in place', async () => {
    // Pre-state: non-main group with an existing heartbeat row (e.g.
    // from a pre-#158 auto-create or an explicit opt-in). The operator
    // is now disabling trigger-required mode. The flip must not
    // delete the row — heartbeat lifecycle is no longer coupled to
    // trigger config (#158), and silently destroying operator state
    // would surprise anyone relying on the row for diagnostics.
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      requiresTrigger: true,
    });
    groups['other@g.us'] = { ...OTHER_GROUP, requiresTrigger: true };
    createTask({
      id: 'heartbeat-other-group',
      group_folder: 'other-group',
      chat_jid: 'other@g.us',
      prompt: 'preexisting-heartbeat',
      schedule_type: 'cron',
      schedule_value: '*/15 * * * *',
      context_mode: 'group',
      next_run: new Date(Date.now() + 15 * 60 * 1000).toISOString(),
      status: 'active',
      created_at: new Date().toISOString(),
      created_by_role: 'owner',
    });
    expect(getTaskById('heartbeat-other-group')).toBeDefined();

    await processTaskIpc(
      {
        type: 'set_trigger',
        jid: 'other@g.us',
        trigger: '@Andy',
        requiresTrigger: false,
      },
      'whatsapp_main',
      true,
      deps,
    );

    // Heartbeat row preserved across the flip.
    const heartbeat = getTaskById('heartbeat-other-group');
    expect(heartbeat).toBeDefined();
    expect(heartbeat?.prompt).toBe('preexisting-heartbeat');
  });
});

// --- set_agent_model (#395) ---
//
// Per-group `containerConfig.agentModel` override. Authorisation mirrors
// schedule_task — main can target any group; non-main can target only
// its own folder. Sibling containerConfig fields must survive untouched
// (regression-bait — set_trusted clobbered them pre-#105).

describe('set_agent_model', () => {
  it('main group can set agentModel on a registered group', async () => {
    await processTaskIpc(
      {
        type: 'set_agent_model',
        groupFolder: 'other-group',
        agentModel: 'sonnet[1m]',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const group = getRegisteredGroup('other@g.us');
    expect(group?.containerConfig?.agentModel).toBe('sonnet[1m]');
    // Other fields preserved.
    expect(group?.trigger).toBe('@Andy');
    expect(group?.folder).toBe('other-group');
  });

  it('non-main group can set agentModel on its own folder', async () => {
    await processTaskIpc(
      {
        type: 'set_agent_model',
        groupFolder: 'other-group',
        agentModel: 'opus',
      },
      'other-group',
      false,
      deps,
    );

    expect(getRegisteredGroup('other@g.us')?.containerConfig?.agentModel).toBe(
      'opus',
    );
  });

  it('non-main group cannot set agentModel on another group', async () => {
    await processTaskIpc(
      {
        type: 'set_agent_model',
        groupFolder: 'third-group',
        agentModel: 'opus',
      },
      'other-group',
      false,
      deps,
    );

    expect(
      getRegisteredGroup('third@g.us')?.containerConfig?.agentModel,
    ).toBeUndefined();
  });

  it('clears agentModel when payload is null', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: { agentModel: 'opus' },
    });
    groups['other@g.us'] = {
      ...OTHER_GROUP,
      containerConfig: { agentModel: 'opus' },
    };

    await processTaskIpc(
      {
        type: 'set_agent_model',
        groupFolder: 'other-group',
        agentModel: null,
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.agentModel,
    ).toBeUndefined();
  });

  it('preserves sibling containerConfig fields on update', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: {
        trusted: true,
        enableHeartbeat: true,
        additionalMounts: [
          { hostPath: '/tmp/extra', containerPath: 'extra', readonly: true },
        ],
      },
    });
    groups['other@g.us'] = getRegisteredGroup('other@g.us')!;

    await processTaskIpc(
      {
        type: 'set_agent_model',
        groupFolder: 'other-group',
        agentModel: 'haiku',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const cfg = getRegisteredGroup('other@g.us')?.containerConfig;
    expect(cfg?.agentModel).toBe('haiku');
    expect(cfg?.trusted).toBe(true);
    expect(cfg?.enableHeartbeat).toBe(true);
    expect(cfg?.additionalMounts).toEqual([
      { hostPath: '/tmp/extra', containerPath: 'extra', readonly: true },
    ]);
  });

  it('rejects missing groupFolder', async () => {
    await processTaskIpc(
      // groupFolder omitted
      { type: 'set_agent_model', agentModel: 'opus' } as Parameters<
        typeof processTaskIpc
      >[0],
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.agentModel,
    ).toBeUndefined();
  });

  it('rejects non-string non-null agentModel (defense vs malformed payload)', async () => {
    await processTaskIpc(
      {
        type: 'set_agent_model',
        groupFolder: 'other-group',
        // 42 is neither a string nor null — must be rejected, not coerced.
        agentModel: 42 as unknown as string,
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.agentModel,
    ).toBeUndefined();
  });

  it('treats empty/whitespace agentModel as a clear', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: { agentModel: 'opus' },
    });
    groups['other@g.us'] = {
      ...OTHER_GROUP,
      containerConfig: { agentModel: 'opus' },
    };

    await processTaskIpc(
      {
        type: 'set_agent_model',
        groupFolder: 'other-group',
        agentModel: '   ',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.agentModel,
    ).toBeUndefined();
  });

  it('set_agent_model on unregistered groupFolder is a no-op', async () => {
    await processTaskIpc(
      {
        type: 'set_agent_model',
        groupFolder: 'never-registered-folder',
        agentModel: 'opus',
      },
      'whatsapp_main',
      true,
      deps,
    );

    // No new registration created.
    const allFolders = Object.values(groups).map((g) => g.folder);
    expect(allFolders).not.toContain('never-registered-folder');
  });
});

// --- set_maintenance_agent_model (#509) ---
//
// Per-session-slot model override that applies only to the maintenance
// container slot. Authorization mirrors set_agent_model (above) — main
// can target any registered group; non-main can target only its own
// folder. Sibling containerConfig fields must survive untouched (same
// regression-bait as set_agent_model). The handler exists so an agent
// or operator can flip the value at runtime without an orchestrator
// restart.

describe('set_maintenance_agent_model', () => {
  it('main group can set maintenanceAgentModel on a registered group', async () => {
    await processTaskIpc(
      {
        type: 'set_maintenance_agent_model',
        groupFolder: 'other-group',
        maintenanceAgentModel: 'sonnet[1m]',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const group = getRegisteredGroup('other@g.us');
    expect(group?.containerConfig?.maintenanceAgentModel).toBe('sonnet[1m]');
    // Other fields preserved.
    expect(group?.trigger).toBe('@Andy');
    expect(group?.folder).toBe('other-group');
  });

  it('non-main group can set maintenanceAgentModel on its own folder', async () => {
    await processTaskIpc(
      {
        type: 'set_maintenance_agent_model',
        groupFolder: 'other-group',
        maintenanceAgentModel: 'opus',
      },
      'other-group',
      false,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.maintenanceAgentModel,
    ).toBe('opus');
  });

  it('non-main group cannot set maintenanceAgentModel on another group', async () => {
    await processTaskIpc(
      {
        type: 'set_maintenance_agent_model',
        groupFolder: 'third-group',
        maintenanceAgentModel: 'opus',
      },
      'other-group',
      false,
      deps,
    );

    expect(
      getRegisteredGroup('third@g.us')?.containerConfig?.maintenanceAgentModel,
    ).toBeUndefined();
  });

  it('clears maintenanceAgentModel when payload is null', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: { maintenanceAgentModel: 'sonnet' },
    });
    groups['other@g.us'] = {
      ...OTHER_GROUP,
      containerConfig: { maintenanceAgentModel: 'sonnet' },
    };

    await processTaskIpc(
      {
        type: 'set_maintenance_agent_model',
        groupFolder: 'other-group',
        maintenanceAgentModel: null,
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.maintenanceAgentModel,
    ).toBeUndefined();
  });

  it('preserves sibling containerConfig fields on update', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: {
        trusted: true,
        enableHeartbeat: true,
        agentModel: 'opus', // user-facing override survives the maintenance update
        additionalMounts: [
          { hostPath: '/tmp/extra', containerPath: 'extra', readonly: true },
        ],
      },
    });
    groups['other@g.us'] = getRegisteredGroup('other@g.us')!;

    await processTaskIpc(
      {
        type: 'set_maintenance_agent_model',
        groupFolder: 'other-group',
        maintenanceAgentModel: 'haiku',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const cfg = getRegisteredGroup('other@g.us')?.containerConfig;
    expect(cfg?.maintenanceAgentModel).toBe('haiku');
    // user-facing per-group override must survive — that's the whole
    // point of the per-session-slot knob (#509).
    expect(cfg?.agentModel).toBe('opus');
    expect(cfg?.trusted).toBe(true);
    expect(cfg?.enableHeartbeat).toBe(true);
    expect(cfg?.additionalMounts).toEqual([
      { hostPath: '/tmp/extra', containerPath: 'extra', readonly: true },
    ]);
  });

  it('rejects missing groupFolder', async () => {
    await processTaskIpc(
      // groupFolder omitted
      {
        type: 'set_maintenance_agent_model',
        maintenanceAgentModel: 'opus',
      } as Parameters<typeof processTaskIpc>[0],
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.maintenanceAgentModel,
    ).toBeUndefined();
  });

  it('rejects non-string non-null maintenanceAgentModel (defense vs malformed payload)', async () => {
    await processTaskIpc(
      {
        type: 'set_maintenance_agent_model',
        groupFolder: 'other-group',
        // 42 is neither a string nor null — must be rejected, not coerced.
        maintenanceAgentModel: 42 as unknown as string,
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.maintenanceAgentModel,
    ).toBeUndefined();
  });

  it('treats empty/whitespace maintenanceAgentModel as a clear', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: { maintenanceAgentModel: 'sonnet' },
    });
    groups['other@g.us'] = {
      ...OTHER_GROUP,
      containerConfig: { maintenanceAgentModel: 'sonnet' },
    };

    await processTaskIpc(
      {
        type: 'set_maintenance_agent_model',
        groupFolder: 'other-group',
        maintenanceAgentModel: '   ',
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.maintenanceAgentModel,
    ).toBeUndefined();
  });

  it('set_maintenance_agent_model on unregistered groupFolder is a no-op', async () => {
    await processTaskIpc(
      {
        type: 'set_maintenance_agent_model',
        groupFolder: 'never-registered-folder',
        maintenanceAgentModel: 'opus',
      },
      'whatsapp_main',
      true,
      deps,
    );

    // No new registration created.
    const allFolders = Object.values(groups).map((g) => g.folder);
    expect(allFolders).not.toContain('never-registered-folder');
  });
});

// --- set_task_agent_model (#509 Phase 3) ---
//
// Per-task AGENT_MODEL override that pins a single scheduled_tasks
// row to a specific model, beating every session-level / group-level
// knob in resolveSessionAgentModel. Authorization mirrors
// set_maintenance_agent_model — main can target any task; non-main
// can target only tasks belonging to its own folder. Persistence
// hits scheduled_tasks.agent_model directly via the setTaskAgentModel
// DB helper (no in-memory registry to refresh — the row is the
// source of truth and the task-scheduler reads it on every fire).

describe('set_task_agent_model', () => {
  function seedTask(id: string, groupFolder: string): void {
    createTask({
      id,
      group_folder: groupFolder,
      chat_jid: groupFolder === 'whatsapp_main' ? 'main@g.us' : 'other@g.us',
      prompt: 'Skill(skill: "tessl__composio-fetch")',
      schedule_type: 'cron',
      schedule_value: '*/30 * * * *',
      context_mode: 'isolated',
      next_run: '2026-05-18T00:30:00.000Z',
      status: 'active',
      created_at: '2026-05-18T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });
  }

  it('main group can set agent_model on any task', async () => {
    seedTask('task-am-other', 'other-group');
    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'task-am-other',
        agentModel: 'haiku',
      },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-am-other')!.agent_model).toBe('haiku');
  });

  it('non-main group can set agent_model on a task in its own folder', async () => {
    seedTask('task-am-own', 'other-group');
    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'task-am-own',
        agentModel: 'sonnet',
      },
      'other-group',
      false,
      deps,
    );
    expect(getTaskById('task-am-own')!.agent_model).toBe('sonnet');
  });

  it('non-main group cannot set agent_model on a task in another folder', async () => {
    seedTask('task-am-cross', 'whatsapp_main');
    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'task-am-cross',
        agentModel: 'haiku',
      },
      'other-group',
      false,
      deps,
    );
    expect(getTaskById('task-am-cross')!.agent_model).toBeNull();
  });

  it('clears agent_model when payload is null', async () => {
    seedTask('task-am-clear', 'other-group');
    // Pre-populate via the same handler so the test exercises the
    // happy path before clearing.
    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'task-am-clear',
        agentModel: 'haiku',
      },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-am-clear')!.agent_model).toBe('haiku');
    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'task-am-clear',
        agentModel: null,
      },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-am-clear')!.agent_model).toBeNull();
  });

  it('treats empty/whitespace agentModel as a clear', async () => {
    seedTask('task-am-ws', 'other-group');
    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'task-am-ws',
        agentModel: 'haiku',
      },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-am-ws')!.agent_model).toBe('haiku');
    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'task-am-ws',
        agentModel: '   ',
      },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-am-ws')!.agent_model).toBeNull();
  });

  it('rejects missing taskId', async () => {
    seedTask('task-am-missing-id', 'other-group');
    await processTaskIpc(
      // taskId omitted
      {
        type: 'set_task_agent_model',
        agentModel: 'haiku',
      } as Parameters<typeof processTaskIpc>[0],
      'whatsapp_main',
      true,
      deps,
    );
    // Row untouched.
    expect(getTaskById('task-am-missing-id')!.agent_model).toBeNull();
  });

  it('rejects non-string non-null agentModel (defense vs malformed payload)', async () => {
    seedTask('task-am-bad-shape', 'other-group');
    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'task-am-bad-shape',
        // 42 is neither a string nor null — must be rejected, not coerced.
        agentModel: 42 as unknown as string,
      },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-am-bad-shape')!.agent_model).toBeNull();
  });

  it('set_task_agent_model on unknown taskId is a no-op (no row materialised)', async () => {
    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'task-am-never-existed',
        agentModel: 'haiku',
      },
      'whatsapp_main',
      true,
      deps,
    );
    expect(getTaskById('task-am-never-existed')).toBeUndefined();
  });

  // PR #587 review feedback: cadence-registry rows are declarative state
  // owned by SKILL.md frontmatter; the rebuild's shape-change UPDATE
  // would silently revert any imperative override on the next tile-
  // touching spawn. Reject those writes loudly at the IPC instead of
  // shipping a "looks fine, isn't" failure mode.
  it('refuses to write to cadence-registry-owned rows (leaves agent_model untouched)', async () => {
    // Seed via the test-only `_execRawForTests` helper so we can pin
    // `source = 'cadence-registry'` — `createTask` always writes the
    // default `source = 'schedule-task'`. The helper talks to the
    // same DB handle the IPC handler reads from, so the source filter
    // sees the seeded row.
    const { _execRawForTests } = await import('./db.js');
    _execRawForTests(
      `INSERT INTO scheduled_tasks (id, group_folder, chat_jid, prompt, schedule_type, schedule_value, status, created_at, created_by_role, source, agent_model)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'cadence-registry', NULL)`,
      [
        'cadence-registry::other-group::tessl__composio-fetch',
        'other-group',
        'other@g.us',
        'Skill(skill: "tessl__composio-fetch")',
        'cron',
        '*/30 * * * *',
        'active',
        '2026-05-18T00:00:00.000Z',
        'owner',
      ],
    );

    await processTaskIpc(
      {
        type: 'set_task_agent_model',
        taskId: 'cadence-registry::other-group::tessl__composio-fetch',
        agentModel: 'haiku',
      },
      'whatsapp_main',
      true,
      deps,
    );

    const task = getTaskById(
      'cadence-registry::other-group::tessl__composio-fetch',
    );
    expect(task).toBeDefined();
    // Refused — column unchanged. Operator must change SKILL.md
    // frontmatter, not the row directly.
    expect(task!.agent_model).toBeNull();
  });
});

// --- promote_learned_trigger (#451 item 1) ---
//
// Flips a learned proposal's `enabled: false → true` so the trigger
// gate starts consuming it. Authorisation mirrors set_agent_model:
// owner-of-bill — main can target any group, non-main can target only
// its own folder. Demotion / re-enable / dashboard / producer-side
// enrichment are #451 items 2/3/4 — separate scopes.

describe('promote_learned_trigger', () => {
  function seedWithLearnedProposal(
    jid: string,
    folder: string,
    overrides: Partial<RegisteredGroup> = {},
  ): RegisteredGroup {
    const group: RegisteredGroup = {
      name: folder === 'whatsapp_main' ? 'Main' : 'Other',
      folder,
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      isMain: folder === 'whatsapp_main',
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            pattern: '@Andy',
            kind: 'keyword',
            source: 'owner-set',
            precision: 1,
            sample_count: 10,
            last_matched_at: null,
            last_updated_at: null,
          },
          {
            pattern: 'help me',
            kind: 'keyword',
            source: 'learned',
            precision: 0.92,
            sample_count: 50,
            last_matched_at: '2026-04-30T12:00:00.000Z',
            last_updated_at: '2026-04-30T12:00:00.000Z',
            pattern_version: 1,
            proposed_at: '2026-04-25T08:00:00.000Z',
            enabled: false,
          },
        ],
      },
      ...overrides,
    };
    setRegisteredGroup(jid, group);
    groups[jid] = group;
    return group;
  }

  it('main group can promote a learned proposal in any group', async () => {
    seedWithLearnedProposal('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const updated = getRegisteredGroup('other@g.us');
    const learned = updated?.triggerPatterns?.patterns.find(
      (p) => p.source === 'learned' && p.pattern === 'help me',
    );
    expect(learned?.enabled).toBe(true);
    // Owner-set sibling pattern preserved verbatim.
    const ownerSet = updated?.triggerPatterns?.patterns.find(
      (p) => p.source === 'owner-set',
    );
    expect(ownerSet?.enabled).toBeUndefined();
  });

  it('non-main group can promote a learned proposal in its own folder', async () => {
    seedWithLearnedProposal('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'other-group',
      false,
      deps,
    );
    const learned = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find(
      (p) => p.source === 'learned' && p.pattern === 'help me',
    );
    expect(learned?.enabled).toBe(true);
  });

  it('non-main group cannot promote a learned proposal in another folder', async () => {
    seedWithLearnedProposal('third@g.us', 'third-group');
    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: 'third-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'other-group',
      false,
      deps,
    );
    // Reject = state unchanged: seed had enabled:false, still enabled:false.
    const learned = getRegisteredGroup(
      'third@g.us',
    )?.triggerPatterns?.patterns.find(
      (p) => p.source === 'learned' && p.pattern === 'help me',
    );
    expect(learned?.enabled).toBe(false);
  });

  it('rejects when the pattern is not in the config', async () => {
    seedWithLearnedProposal('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'nonexistent',
      },
      'whatsapp_main',
      true,
      deps,
    );
    // Existing learned row stays enabled:false (untouched by the
    // unmatched promotion request).
    const learned = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find(
      (p) => p.source === 'learned' && p.pattern === 'help me',
    );
    expect(learned?.enabled).toBe(false);
  });

  it('rejects when the matching pattern is owner-set rather than learned', async () => {
    // Owner-set patterns don't have an `enabled` field — they're
    // active by default. Promoting one would be meaningless. The
    // handler must require `source: 'learned'` on the match.
    seedWithLearnedProposal('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: '@Andy',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const ownerSet = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find((p) => p.source === 'owner-set');
    expect(ownerSet?.enabled).toBeUndefined();
  });

  it('rejects an auto-rolled-back (disabled=true) pattern — operator must re-enable first', async () => {
    seedWithLearnedProposal('other@g.us', 'other-group');
    const seeded = getRegisteredGroup('other@g.us');
    const withDisabled: RegisteredGroup = {
      ...seeded!,
      triggerPatterns: {
        ...seeded!.triggerPatterns!,
        patterns: seeded!.triggerPatterns!.patterns.map((p) =>
          p.source === 'learned' ? { ...p, disabled: true } : p,
        ),
      },
    };
    setRegisteredGroup('other@g.us', withDisabled);
    groups['other@g.us'] = withDisabled;

    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'whatsapp_main',
      true,
      deps,
    );
    // disabled stays true; enabled must NOT have been flipped.
    const learned = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find(
      (p) => p.source === 'learned' && p.pattern === 'help me',
    );
    expect(learned?.disabled).toBe(true);
    expect(learned?.enabled).toBe(false);
  });

  it('is idempotent — already-enabled pattern is a no-op', async () => {
    seedWithLearnedProposal('other@g.us', 'other-group');
    const seeded = getRegisteredGroup('other@g.us');
    const withEnabled: RegisteredGroup = {
      ...seeded!,
      triggerPatterns: {
        ...seeded!.triggerPatterns!,
        patterns: seeded!.triggerPatterns!.patterns.map((p) =>
          p.source === 'learned' ? { ...p, enabled: true } : p,
        ),
      },
    };
    setRegisteredGroup('other@g.us', withEnabled);
    groups['other@g.us'] = withEnabled;

    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const learned = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find(
      (p) => p.source === 'learned' && p.pattern === 'help me',
    );
    expect(learned?.enabled).toBe(true);
  });

  it('rejects whitespace-only pattern (un-trimmed identity must still match real stored patterns)', async () => {
    seedWithLearnedProposal('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: '   ',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const learned = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find(
      (p) => p.source === 'learned' && p.pattern === 'help me',
    );
    expect(learned?.enabled).toBe(false);
  });

  it('rejects when groupFolder is missing or empty', async () => {
    seedWithLearnedProposal('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: '',
        kind: 'keyword',
        pattern: 'help me',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const learned = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find(
      (p) => p.source === 'learned' && p.pattern === 'help me',
    );
    expect(learned?.enabled).toBe(false);
  });

  it('rejects when the group has no learned patterns to promote', async () => {
    // OTHER_GROUP has only a legacy string trigger ('@Andy'); the
    // dual-mode reader auto-derives a single owner-set pattern from
    // it but never a learned row. The handler should reject because
    // the {kind, pattern, source:'learned'} tuple won't match any
    // entry — and crucially must NOT mutate the auto-derived owner-set
    // entry.
    setRegisteredGroup('other@g.us', { ...OTHER_GROUP });
    groups['other@g.us'] = { ...OTHER_GROUP };
    await processTaskIpc(
      {
        type: 'promote_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const cfg = getRegisteredGroup('other@g.us')?.triggerPatterns;
    expect(cfg?.patterns.every((p) => p.source !== 'learned')).toBe(true);
    expect(
      cfg?.patterns.find((p) => p.source === 'owner-set')?.enabled,
    ).toBeUndefined();
  });
});

// --- reenable_learned_trigger (#451 item 2 — re-enable half) ---
//
// Counterpart to the learner's auto-rollback: flips `disabled: true →
// false` on a learned proposal. Same {kind, pattern} identity + same
// owner-of-bill auth as promote/delete. Idempotent on already-active
// rows; refuses to mutate non-learned entries.

describe('reenable_learned_trigger', () => {
  function seedDisabled(jid: string, folder: string): RegisteredGroup {
    const group: RegisteredGroup = {
      name: 'Other',
      folder,
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            pattern: 'help me',
            kind: 'keyword',
            source: 'learned',
            precision: 0.4,
            sample_count: 100,
            last_matched_at: '2026-04-30T12:00:00.000Z',
            last_updated_at: '2026-04-30T12:00:00.000Z',
            enabled: true,
            disabled: true,
          },
        ],
      },
    };
    setRegisteredGroup(jid, group);
    groups[jid] = group;
    return group;
  }

  it('flips disabled true → false on a demoted learned pattern', async () => {
    seedDisabled('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'reenable_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const learned = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find((p) => p.source === 'learned');
    expect(learned?.disabled).toBe(false);
    // enabled stays true (was true before demotion).
    expect(learned?.enabled).toBe(true);
  });

  it('non-main can re-enable in own folder, not in another', async () => {
    seedDisabled('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'reenable_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'other-group',
      false,
      deps,
    );
    expect(
      getRegisteredGroup('other@g.us')?.triggerPatterns?.patterns.find(
        (p) => p.source === 'learned',
      )?.disabled,
    ).toBe(false);

    seedDisabled('third@g.us', 'third-group');
    await processTaskIpc(
      {
        type: 'reenable_learned_trigger',
        groupFolder: 'third-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'other-group',
      false,
      deps,
    );
    expect(
      getRegisteredGroup('third@g.us')?.triggerPatterns?.patterns.find(
        (p) => p.source === 'learned',
      )?.disabled,
    ).toBe(true);
  });

  it('idempotent — already-active pattern is a no-op', async () => {
    const seeded = seedDisabled('other@g.us', 'other-group');
    const withoutDisabled: RegisteredGroup = {
      ...seeded,
      triggerPatterns: {
        ...seeded.triggerPatterns!,
        patterns: seeded.triggerPatterns!.patterns.map((p) => ({
          ...p,
          disabled: false,
        })),
      },
    };
    setRegisteredGroup('other@g.us', withoutDisabled);
    groups['other@g.us'] = withoutDisabled;

    await processTaskIpc(
      {
        type: 'reenable_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const learned = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find((p) => p.source === 'learned');
    expect(learned?.disabled).toBe(false);
  });

  it('rejects when no matching learned pattern', async () => {
    seedDisabled('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'reenable_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'nonexistent',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const learned = getRegisteredGroup(
      'other@g.us',
    )?.triggerPatterns?.patterns.find((p) => p.source === 'learned');
    expect(learned?.disabled).toBe(true);
  });
});

// --- delete_learned_trigger (#451 item 2 — delete half) ---
//
// Permanent removal of a learned proposal. Same {kind, pattern}
// identity + same auth shape. Distinct from re-enable: the learner
// can re-propose the same body later as a fresh row.

describe('delete_learned_trigger', () => {
  function seedTwoLearned(jid: string, folder: string): RegisteredGroup {
    const group: RegisteredGroup = {
      name: 'Other',
      folder,
      trigger: '@Andy',
      added_at: '2024-01-01T00:00:00.000Z',
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            pattern: '@Andy',
            kind: 'keyword',
            source: 'owner-set',
            precision: 1,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          },
          {
            pattern: 'help me',
            kind: 'keyword',
            source: 'learned',
            precision: 0.92,
            sample_count: 50,
            last_matched_at: '2026-04-30T12:00:00.000Z',
            last_updated_at: '2026-04-30T12:00:00.000Z',
            enabled: false,
          },
          {
            pattern: 'thanks',
            kind: 'keyword',
            source: 'learned',
            precision: 0.85,
            sample_count: 30,
            last_matched_at: null,
            last_updated_at: null,
            enabled: false,
          },
        ],
      },
    };
    setRegisteredGroup(jid, group);
    groups[jid] = group;
    return group;
  }

  it('removes the matching learned proposal and preserves siblings', async () => {
    seedTwoLearned('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'delete_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const cfg = getRegisteredGroup('other@g.us')?.triggerPatterns;
    expect(cfg?.patterns.length).toBe(2);
    expect(
      cfg?.patterns.find(
        (p) => p.source === 'learned' && p.pattern === 'help me',
      ),
    ).toBeUndefined();
    // The other learned proposal is preserved.
    expect(
      cfg?.patterns.find(
        (p) => p.source === 'learned' && p.pattern === 'thanks',
      ),
    ).toBeDefined();
    // Owner-set pattern preserved verbatim.
    expect(cfg?.patterns.find((p) => p.source === 'owner-set')?.pattern).toBe(
      '@Andy',
    );
  });

  it('refuses to delete an owner-set pattern with the same body', async () => {
    seedTwoLearned('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'delete_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: '@Andy',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const cfg = getRegisteredGroup('other@g.us')?.triggerPatterns;
    expect(cfg?.patterns.length).toBe(3);
    expect(cfg?.patterns.find((p) => p.source === 'owner-set')?.pattern).toBe(
      '@Andy',
    );
  });

  it('rejects when no matching learned pattern', async () => {
    seedTwoLearned('other@g.us', 'other-group');
    await processTaskIpc(
      {
        type: 'delete_learned_trigger',
        groupFolder: 'other-group',
        kind: 'keyword',
        pattern: 'nonexistent',
      },
      'whatsapp_main',
      true,
      deps,
    );
    expect(
      getRegisteredGroup('other@g.us')?.triggerPatterns?.patterns.length,
    ).toBe(3);
  });

  it('non-main cannot delete in another folder', async () => {
    seedTwoLearned('third@g.us', 'third-group');
    await processTaskIpc(
      {
        type: 'delete_learned_trigger',
        groupFolder: 'third-group',
        kind: 'keyword',
        pattern: 'help me',
      },
      'other-group',
      false,
      deps,
    );
    expect(
      getRegisteredGroup('third@g.us')?.triggerPatterns?.patterns.length,
    ).toBe(3);
  });
});

// --- list_learned_triggers (#451 item 3) ---
//
// Read-side observability surface. Returns the learned-source
// patterns with their full optional-field set so a dashboard or
// status command can render precision / sample_count / proposed_at
// without parsing JSON columns directly. Owner-of-bill auth: non-main
// can only inspect own folder.

describe('list_learned_triggers', () => {
  function seedAcrossGroups(): void {
    const otherGroup: RegisteredGroup = {
      ...OTHER_GROUP,
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            pattern: '@Andy',
            kind: 'keyword',
            source: 'owner-set',
            precision: 1,
            sample_count: 0,
            last_matched_at: null,
            last_updated_at: null,
          },
          {
            pattern: 'help me',
            kind: 'keyword',
            source: 'learned',
            precision: 0.92,
            sample_count: 50,
            last_matched_at: '2026-04-30T12:00:00.000Z',
            last_updated_at: '2026-04-30T12:00:00.000Z',
            pattern_version: 1,
            proposed_at: '2026-04-25T08:00:00.000Z',
            enabled: false,
          },
        ],
      },
    };
    const thirdGroup: RegisteredGroup = {
      ...THIRD_GROUP,
      triggerPatterns: {
        version: 1,
        patterns: [
          {
            pattern: 'thanks',
            kind: 'keyword',
            source: 'learned',
            precision: 0.85,
            sample_count: 30,
            last_matched_at: null,
            last_updated_at: null,
            pattern_version: 2,
            proposed_at: '2026-04-26T09:00:00.000Z',
            enabled: true,
          },
        ],
      },
    };
    setRegisteredGroup('other@g.us', otherGroup);
    setRegisteredGroup('third@g.us', thirdGroup);
    groups['other@g.us'] = otherGroup;
    groups['third@g.us'] = thirdGroup;
  }

  function readResult(sourceGroup: string, requestId: string): unknown {
    const file = path.join(
      TEST_DATA_DIR,
      'ipc',
      sourceGroup,
      'input-default',
      `_script_result_${requestId}.json`,
    );
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  }

  it('main with no filter returns learned patterns from every registered group', async () => {
    seedAcrossGroups();
    await processTaskIpc(
      {
        type: 'list_learned_triggers',
        requestId: 'req-list-all',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const result = readResult('whatsapp_main', 'req-list-all') as {
      stdout: string;
    };
    const parsed = JSON.parse(result.stdout) as {
      groups: Array<{ folder: string; learned: TriggerPattern[] }>;
    };
    const folders = parsed.groups.map((g) => g.folder);
    expect(folders).toContain('other-group');
    expect(folders).toContain('third-group');
    const otherLearned = parsed.groups.find(
      (g) => g.folder === 'other-group',
    )?.learned;
    expect(otherLearned?.length).toBe(1);
    expect(otherLearned?.[0].pattern).toBe('help me');
    expect(otherLearned?.[0].precision).toBe(0.92);
    expect(otherLearned?.[0].proposed_at).toBe('2026-04-25T08:00:00.000Z');
  });

  it('main with groupFolder filter returns only that group', async () => {
    seedAcrossGroups();
    await processTaskIpc(
      {
        type: 'list_learned_triggers',
        groupFolder: 'other-group',
        requestId: 'req-list-other',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const result = readResult('whatsapp_main', 'req-list-other') as {
      stdout: string;
    };
    const parsed = JSON.parse(result.stdout) as {
      groups: Array<{ folder: string }>;
    };
    expect(parsed.groups.map((g) => g.folder)).toEqual(['other-group']);
  });

  it('non-main with no filter is implicitly scoped to own folder', async () => {
    seedAcrossGroups();
    await processTaskIpc(
      {
        type: 'list_learned_triggers',
        requestId: 'req-list-self',
      },
      'other-group',
      false,
      deps,
    );
    const result = readResult('other-group', 'req-list-self') as {
      stdout: string;
    };
    const parsed = JSON.parse(result.stdout) as {
      groups: Array<{ folder: string }>;
    };
    expect(parsed.groups.map((g) => g.folder)).toEqual(['other-group']);
  });

  it('non-main cannot inspect another folder', async () => {
    seedAcrossGroups();
    await processTaskIpc(
      {
        type: 'list_learned_triggers',
        groupFolder: 'third-group',
        requestId: 'req-list-cross',
      },
      'other-group',
      false,
      deps,
    );
    const result = readResult('other-group', 'req-list-cross') as {
      error: string;
    };
    expect(result.error).toContain('cross-folder read denied');
  });

  it('returns an empty learned list for groups with only owner-set patterns', async () => {
    setRegisteredGroup('other@g.us', { ...OTHER_GROUP });
    groups['other@g.us'] = { ...OTHER_GROUP };
    await processTaskIpc(
      {
        type: 'list_learned_triggers',
        groupFolder: 'other-group',
        requestId: 'req-list-empty',
      },
      'whatsapp_main',
      true,
      deps,
    );
    const result = readResult('whatsapp_main', 'req-list-empty') as {
      stdout: string;
    };
    const parsed = JSON.parse(result.stdout) as {
      groups: Array<{ folder: string; learned: TriggerPattern[] }>;
    };
    expect(parsed.groups[0].learned).toEqual([]);
  });
});

// --- set_additional_tiles (#305) ---
//
// Per-chat additive tile overlay. Authorisation: main-only — overlay
// tiles add capabilities, so a non-main agent can't grant itself
// extra skills/rules. Validation is fail-closed against the live
// registry: every entry must resolve to an installed tile or the
// whole write is rejected. Sibling containerConfig fields must
// survive untouched (regression-bait — set_trusted clobbered them
// pre-#105 and #305 must not regress that).

describe('set_additional_tiles', () => {
  beforeEach(() => {
    // Default to a registry that has a couple of overlay tiles
    // installed; individual tests override per-scenario.
    mockGetInstalledTiles.mockReturnValue([
      'nanoclaw-coding',
      'nanoclaw-family',
      'nanoclaw-core',
      'nanoclaw-trusted',
      'nanoclaw-untrusted',
      'nanoclaw-admin',
    ]);
  });

  it('main can set additionalTiles when every entry is installed', async () => {
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: ['nanoclaw-coding', 'nanoclaw-family'],
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toEqual(['nanoclaw-coding', 'nanoclaw-family']);
  });

  it('non-main groups cannot set additionalTiles even on their own folder', async () => {
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: ['nanoclaw-coding'],
      },
      'other-group',
      false,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toBeUndefined();
  });

  it('rejects the whole write if any tile is not in the registry', async () => {
    mockGetInstalledTiles.mockReturnValue(['nanoclaw-coding']);
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: ['nanoclaw-coding', 'nanoclaw-typo'],
      },
      'whatsapp_main',
      true,
      deps,
    );

    // No partial acceptance — `nanoclaw-coding` must NOT have been
    // persisted just because it happened to validate. The whole
    // payload is rejected so the operator notices the typo.
    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toBeUndefined();
  });

  it('rejects when the registry directory does not exist', async () => {
    mockGetInstalledTiles.mockReturnValue(null);
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: ['nanoclaw-coding'],
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toBeUndefined();
  });

  it('clears additionalTiles when payload is null', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: { additionalTiles: ['nanoclaw-coding'] },
    });
    groups['other@g.us'] = getRegisteredGroup('other@g.us')!;

    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: null,
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toBeUndefined();
  });

  it('treats empty array as a clear (drops the field rather than persisting [])', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: { additionalTiles: ['nanoclaw-coding'] },
    });
    groups['other@g.us'] = getRegisteredGroup('other@g.us')!;

    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: [],
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toBeUndefined();
  });

  it('de-duplicates repeated entries at write time', async () => {
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: [
          'nanoclaw-coding',
          'nanoclaw-coding',
          'nanoclaw-family',
        ],
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toEqual(['nanoclaw-coding', 'nanoclaw-family']);
  });

  it('rejects whitespace-only entries (defensive — selectTiles also skips them)', async () => {
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: ['nanoclaw-coding', '   '],
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toBeUndefined();
  });

  it('rejects non-string entries', async () => {
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        // 42 violates the string-only contract.
        additionalTiles: ['nanoclaw-coding', 42] as unknown as string[],
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toBeUndefined();
  });

  it('rejects non-array non-null payload (e.g. a single string)', async () => {
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: 'nanoclaw-coding' as unknown as string[],
      },
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toBeUndefined();
  });

  it('rejects missing groupFolder', async () => {
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        additionalTiles: ['nanoclaw-coding'],
      } as Parameters<typeof processTaskIpc>[0],
      'whatsapp_main',
      true,
      deps,
    );

    expect(
      getRegisteredGroup('other@g.us')?.containerConfig?.additionalTiles,
    ).toBeUndefined();
  });

  it('preserves sibling containerConfig fields on update', async () => {
    setRegisteredGroup('other@g.us', {
      ...OTHER_GROUP,
      containerConfig: {
        trusted: true,
        agentModel: 'opus',
        enableHeartbeat: true,
        additionalMounts: [
          { hostPath: '/tmp/extra', containerPath: 'extra', readonly: true },
        ],
      },
    });
    groups['other@g.us'] = getRegisteredGroup('other@g.us')!;

    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'other-group',
        additionalTiles: ['nanoclaw-coding'],
      },
      'whatsapp_main',
      true,
      deps,
    );

    const cfg = getRegisteredGroup('other@g.us')?.containerConfig;
    expect(cfg?.additionalTiles).toEqual(['nanoclaw-coding']);
    expect(cfg?.trusted).toBe(true);
    expect(cfg?.agentModel).toBe('opus');
    expect(cfg?.enableHeartbeat).toBe(true);
    expect(cfg?.additionalMounts).toEqual([
      { hostPath: '/tmp/extra', containerPath: 'extra', readonly: true },
    ]);
  });

  it('set_additional_tiles on unregistered groupFolder is a no-op', async () => {
    await processTaskIpc(
      {
        type: 'set_additional_tiles',
        groupFolder: 'never-registered-folder',
        additionalTiles: ['nanoclaw-coding'],
      },
      'whatsapp_main',
      true,
      deps,
    );

    const allFolders = Object.values(groups).map((g) => g.folder);
    expect(allFolders).not.toContain('never-registered-folder');
  });
});

// --- tessl_update / push_staged_to_branch authorization ---
//
// These handlers are main-only because they touch the global tile
// registry / open commits against shared tile repos. On unauthorized
// calls, processTaskIpc writes an error response to the requesting
// group's input dir so the container-side MCP caller doesn't just hang
// until its own timeout. The tests below assert both the write and the
// message content — if someone accidentally drops the `!isMain` guard,
// we want vitest to fail, not a runtime CVE.
//
// We run these with a real filesystem write into `TEST_DATA_DIR/ipc/...`
// (the mocked DATA_DIR — see `vi.mock('./config.js', ...)` above)
// because the handler uses `fs.writeFileSync` directly (no mockable
// seam). `afterEach` cleans up the resulting files; `afterAll` wipes
// the whole tempdir as a backstop if a test crashed mid-run.

const UNAUTH_GROUP = 'other-group';
const unauthInputDir = path.join(
  TEST_DATA_DIR,
  'ipc',
  UNAUTH_GROUP,
  'input-default',
);

// Last-resort cleanup after the whole file finishes. Individual afterEach
// calls remove dirs they explicitly created, but a test that crashes
// mid-run could leave the tempdir behind — wiping it on afterAll keeps
// /tmp tidy across repeated test runs.
afterAll(() => {
  if (fs.existsSync(TEST_DATA_DIR)) {
    fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
  }
});
const unauthCreatedDirs: string[] = [];
const unauthCreatedFiles: string[] = [];

function ensureUnauthInputDir(): void {
  // Track what we create so afterEach can clean up without blowing
  // away a pre-existing real orchestrator data dir (unlikely in CI,
  // possible locally).
  let p = unauthInputDir;
  while (!fs.existsSync(p) && p !== path.dirname(p)) {
    unauthCreatedDirs.unshift(p);
    p = path.dirname(p);
  }
  fs.mkdirSync(unauthInputDir, { recursive: true });
}

function resultPathFor(requestId: string): string {
  const p = path.join(unauthInputDir, `_script_result_${requestId}.json`);
  unauthCreatedFiles.push(p);
  return p;
}

// Shared afterEach body for the auth describes below. The logic was
// originally duplicated across blocks; extracting it means the two
// describes can't silently drift (e.g. one forgets the `.reverse()` and
// starts leaving orphan dirs). Kept as a plain function rather than a
// hook so each describe decides when to register it — right now that's
// just `afterEach(cleanupUnauthFixtures)`.
function cleanupUnauthFixtures(): void {
  for (const f of unauthCreatedFiles.splice(0)) {
    if (fs.existsSync(f)) fs.unlinkSync(f);
  }
  // Iterate deepest-first so each rmdirSync sees an empty directory.
  // `ensureUnauthInputDir` unshifts parents onto the array as it walks
  // upward, so the raw array is parent-to-child; reversing puts the
  // leaf directory first, and by the time we reach its parent the
  // leaf is already gone.
  for (const d of unauthCreatedDirs.splice(0).reverse()) {
    if (fs.existsSync(d) && fs.readdirSync(d).length === 0) {
      fs.rmdirSync(d);
    }
  }
}

describe('tessl_update authorization', () => {
  beforeEach(() => {
    ensureUnauthInputDir();
  });

  afterEach(() => {
    cleanupUnauthFixtures();
  });

  it('non-main group is rejected with an error response', async () => {
    const resultPath = resultPathFor('test-tessl-unauth');

    await processTaskIpc(
      { type: 'tessl_update', requestId: 'test-tessl-unauth' },
      UNAUTH_GROUP,
      false,
      deps,
    );

    expect(fs.existsSync(resultPath)).toBe(true);
    const body = JSON.parse(fs.readFileSync(resultPath, 'utf-8'));
    expect(body.error).toMatch(/Only the main group/);
  });

  it('request without requestId writes nothing and returns', async () => {
    await processTaskIpc({ type: 'tessl_update' }, UNAUTH_GROUP, false, deps);
    // No assertion on files — the handler must not spawn execFile
    // or write anything. The test passes if processTaskIpc returns
    // without throwing.
  });
});

// --- list_installed_tiles (#305) ---
//
// Admin-only sync IPC: returns the names of every tile in the local
// Tessl registry so the agent can show the operator valid overlay
// names before proposing a `set_additional_tiles` change. Result-file
// shape mirrors `chat_status` / `tessl_update` — `{stdout: <JSON>}`
// for success, `{error: <message>}` for failure. The test file's
// hoisted `mockGetInstalledTiles` controls what the registry "looks
// like" without needing a real `tessl-workspace/` on disk.

describe('list_installed_tiles', () => {
  beforeEach(() => {
    ensureUnauthInputDir();
    // Ensure the main group's input dir exists too so the success-path
    // test's `fs.writeFileSync(resultPath, ...)` lands somewhere.
    const mainInputDir = path.join(
      TEST_DATA_DIR,
      'ipc',
      'whatsapp_main',
      'input-default',
    );
    fs.mkdirSync(mainInputDir, { recursive: true });
  });

  afterEach(() => {
    cleanupUnauthFixtures();
    // Clean up any result files written under whatsapp_main during
    // success-path tests. Walk the input dir and unlink everything —
    // tessl_update / push_staged_to_branch use the same convention.
    const mainInputDir = path.join(
      TEST_DATA_DIR,
      'ipc',
      'whatsapp_main',
      'input-default',
    );
    if (fs.existsSync(mainInputDir)) {
      for (const f of fs.readdirSync(mainInputDir)) {
        fs.unlinkSync(path.join(mainInputDir, f));
      }
    }
  });

  it('non-main group is rejected with an error response', async () => {
    const resultPath = resultPathFor('list-tiles-unauth');
    await processTaskIpc(
      { type: 'list_installed_tiles', requestId: 'list-tiles-unauth' },
      UNAUTH_GROUP,
      false,
      deps,
    );
    expect(fs.existsSync(resultPath)).toBe(true);
    const body = JSON.parse(fs.readFileSync(resultPath, 'utf-8'));
    expect(body.error).toMatch(/admin-tile only/);
  });

  it('main returns the installed tile list as JSON in stdout', async () => {
    mockGetInstalledTiles.mockReturnValue([
      'nanoclaw-coding',
      'nanoclaw-core',
      'nanoclaw-family',
      'nanoclaw-trusted',
    ]);
    const requestId = 'list-tiles-ok';
    const resultPath = path.join(
      TEST_DATA_DIR,
      'ipc',
      'whatsapp_main',
      'input-default',
      `_script_result_${requestId}.json`,
    );
    await processTaskIpc(
      { type: 'list_installed_tiles', requestId },
      'whatsapp_main',
      true,
      deps,
    );
    expect(fs.existsSync(resultPath)).toBe(true);
    const body = JSON.parse(fs.readFileSync(resultPath, 'utf-8'));
    expect(body.error).toBeUndefined();
    const inner = JSON.parse(body.stdout);
    expect(inner.registryAbsent).toBe(false);
    expect(inner.tiles).toEqual([
      'nanoclaw-coding',
      'nanoclaw-core',
      'nanoclaw-family',
      'nanoclaw-trusted',
    ]);
  });

  it('signals registryAbsent when the registry directory does not exist', async () => {
    mockGetInstalledTiles.mockReturnValue(null);
    const requestId = 'list-tiles-cold';
    const resultPath = path.join(
      TEST_DATA_DIR,
      'ipc',
      'whatsapp_main',
      'input-default',
      `_script_result_${requestId}.json`,
    );
    await processTaskIpc(
      { type: 'list_installed_tiles', requestId },
      'whatsapp_main',
      true,
      deps,
    );
    expect(fs.existsSync(resultPath)).toBe(true);
    const body = JSON.parse(fs.readFileSync(resultPath, 'utf-8'));
    const inner = JSON.parse(body.stdout);
    expect(inner.registryAbsent).toBe(true);
    expect(inner.tiles).toEqual([]);
  });
});

describe('push_staged_to_branch authorization', () => {
  beforeEach(() => {
    ensureUnauthInputDir();
  });

  afterEach(() => {
    cleanupUnauthFixtures();
  });

  it('non-main group is rejected with an error response', async () => {
    const resultPath = resultPathFor('test-push-unauth');

    await processTaskIpc(
      {
        type: 'push_staged_to_branch',
        requestId: 'test-push-unauth',
        tileName: 'nanoclaw-admin',
        branch: 'promote/20260101T000000Z-nanoclaw-admin',
        commitMessage: 'fix: test',
      },
      UNAUTH_GROUP,
      false,
      deps,
    );

    expect(fs.existsSync(resultPath)).toBe(true);
    const body = JSON.parse(fs.readFileSync(resultPath, 'utf-8'));
    expect(body.error).toMatch(/Only the main group/);
  });

  it('request missing required fields is a no-op (no result file)', async () => {
    const resultPath = resultPathFor('test-push-incomplete');

    await processTaskIpc(
      {
        type: 'push_staged_to_branch',
        requestId: 'test-push-incomplete',
        tileName: 'nanoclaw-admin',
        // missing branch, commitMessage — the outer `if` guard drops the request
      },
      'whatsapp_main',
      true,
      deps,
    );

    // No result file: handler validates required fields before acting
    // and the request is silently dropped. (Silent-drop-on-missing-field
    // matches the `promote_staging` pattern already in the code.)
    expect(fs.existsSync(resultPath)).toBe(false);
  });
});
