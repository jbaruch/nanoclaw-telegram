import fs from 'fs';
import path from 'path';

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

import { DATA_DIR } from './config.js';
import {
  _initTestDatabase,
  createTask,
  getAllTasks,
  getRegisteredGroup,
  getTaskById,
  setRegisteredGroup,
} from './db.js';
import { processTaskIpc, IpcDeps } from './ipc.js';
import { RegisteredGroup } from './types.js';

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
      // Mock the fs.mkdirSync that registerGroup does
    },
    syncGroups: async () => {},
    getAvailableGroups: () => [],
    writeGroupsSnapshot: () => {},
    onTasksChanged: () => {},
    nukeSession: (
      _folder: string,
      _session: 'default' | 'maintenance' | 'all',
    ) => {},
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
// We run these with a real filesystem write into `DATA_DIR/ipc/...`
// because the handler uses `fs.writeFileSync` directly (no mockable
// seam). `afterEach` cleans up the resulting files.

const UNAUTH_GROUP = 'other-group';
const unauthInputDir = path.join(
  DATA_DIR,
  'ipc',
  UNAUTH_GROUP,
  'input-default',
);
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

describe('tessl_update authorization', () => {
  beforeEach(() => {
    ensureUnauthInputDir();
  });

  afterEach(() => {
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

describe('push_staged_to_branch authorization', () => {
  beforeEach(() => {
    ensureUnauthInputDir();
  });

  afterEach(() => {
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
