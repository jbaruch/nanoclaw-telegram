import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Mock container-runner BEFORE importing task-scheduler so the scheduler
// picks up the mocked `runContainerAgent`. We can't actually spawn a
// container in a unit test, so this fake calls the streaming callback
// with whatever output the test-under-test wants to simulate.
//
// `vi.hoisted` is required because `vi.mock(...)` itself is hoisted to
// the top of the file — a plain top-level `const` would be accessed
// before initialisation inside the factory.
const { mockRunContainerAgent } = vi.hoisted(() => ({
  mockRunContainerAgent: vi.fn(),
}));
vi.mock('./container-runner.js', () => ({
  runContainerAgent: mockRunContainerAgent,
  writeTasksSnapshot: vi.fn(),
  DEFAULT_SESSION_NAME: 'default',
}));

import {
  _initTestDatabase,
  createTask,
  getAllChats,
  getLastBotMessageTimestamp,
  getSession,
  getTaskById,
  setSession,
  storeChatMetadata,
} from './db.js';
import {
  _resetSchedulerLoopForTests,
  computeNextRun,
  startSchedulerLoop,
} from './task-scheduler.js';
import type { ContainerOutput } from './container-runner.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';

describe('task scheduler', () => {
  beforeEach(() => {
    _initTestDatabase();
    _resetSchedulerLoopForTests();
    mockRunContainerAgent.mockClear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('pauses due tasks with invalid group folders to prevent retry churn', async () => {
    createTask({
      id: 'task-invalid-folder',
      group_folder: '../../outside',
      chat_jid: 'bad@g.us',
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-02-22T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 60_000).toISOString(),
      status: 'active',
      created_at: '2026-02-22T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({}),
      getSessions: () => ({}),
      queue: { enqueueTask } as any,
      onProcess: () => {},
      sendMessage: async () => {},
    });

    await vi.advanceTimersByTimeAsync(10);

    const task = getTaskById('task-invalid-folder');
    expect(task?.status).toBe('paused');
  });

  it('computeNextRun anchors interval tasks to scheduled time to prevent drift', () => {
    const scheduledTime = new Date(Date.now() - 2000).toISOString(); // 2s ago
    const task = {
      id: 'drift-test',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'interval' as const,
      schedule_value: '60000', // 1 minute
      context_mode: 'isolated' as const,
      next_run: scheduledTime,
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    const nextRun = computeNextRun(task);
    expect(nextRun).not.toBeNull();

    // Should be anchored to scheduledTime + 60s, NOT Date.now() + 60s
    const expected = new Date(scheduledTime).getTime() + 60000;
    expect(new Date(nextRun!).getTime()).toBe(expected);
  });

  it('computeNextRun returns null for once-tasks', () => {
    const task = {
      id: 'once-test',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'once' as const,
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated' as const,
      next_run: new Date(Date.now() - 1000).toISOString(),
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    expect(computeNextRun(task)).toBeNull();
  });

  it('maintenance task with context_mode=group uses stored maintenance sessionId and persists newSessionId', async () => {
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };

    // Seed a prior maintenance sessionId in the sessions cache. The
    // scheduler should read this and pass it into runContainerAgent.
    setSession('main', MAINTENANCE_SESSION_NAME, 'prior-maint-session');

    createTask({
      id: 'group-ctx-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        // Simulate a streamed success with a new sessionId — this is what
        // the container-runner reports back after the SDK's query() resolves.
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: 'new-maint-session',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': MAIN_GROUP }),
      getSessions: () => ({ main: { maintenance: 'prior-maint-session' } }),
      queue: { enqueueTask, closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
    });

    await vi.advanceTimersByTimeAsync(10);

    // The stored prior sessionId was passed in as the resume target.
    expect(mockRunContainerAgent).toHaveBeenCalled();
    const containerInput = mockRunContainerAgent.mock.calls[0][1];
    expect(containerInput.sessionId).toBe('prior-maint-session');
    expect(containerInput.sessionName).toBe(MAINTENANCE_SESSION_NAME);

    // The new sessionId from the streaming callback was persisted to the
    // MAINTENANCE slot (not default).
    expect(getSession('main', MAINTENANCE_SESSION_NAME)).toBe(
      'new-maint-session',
    );
    expect(getSession('main', 'default')).toBeUndefined();
  });

  it('maintenance task with context_mode=isolated does NOT persist newSessionId', async () => {
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };

    // No prior sessionId in the cache for isolated tasks.
    createTask({
      id: 'isolated-task',
      group_folder: 'main',
      chat_jid: 'main@g.us',
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'isolated',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
          newSessionId: 'should-not-be-persisted',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ 'main@g.us': MAIN_GROUP }),
      getSessions: () => ({}),
      queue: { enqueueTask, closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
    });

    await vi.advanceTimersByTimeAsync(10);

    // Isolated tasks start fresh — no sessionId passed in.
    const containerInput = mockRunContainerAgent.mock.calls[0][1];
    expect(containerInput.sessionId).toBeUndefined();

    // And the streamed newSessionId was NOT persisted — an isolated task
    // finishing must not contaminate the maintenance slot's chain.
    expect(getSession('main', MAINTENANCE_SESSION_NAME)).toBeUndefined();
  });

  it('streamed scheduled-task result writes a bot row to messages.db', async () => {
    // Regression for jbaruch/nanoclaw#81 root cause: pre-fix, the
    // task-scheduler streaming callback sent to Telegram but never
    // called storeMessage — so every heartbeat/housekeeping cycle
    // reached the user but left no DB row ("ghost heartbeat"). The
    // fix mirrors the ipc.ts send_message pattern: storeMessage is
    // called immediately after sendMessage, with the same row shape.
    // This test fails without the fix.
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const chatJid = 'main@g.us';
    // Seed the chats row so the FK from messages.chat_jid to chats.jid
    // doesn't reject the bot insert. Production has this metadata from
    // the first real user message in the chat; in-test we create it
    // explicitly.
    storeChatMetadata(chatJid, '2026-01-01T00:00:00.000Z', 'Main');

    setSession('main', MAINTENANCE_SESSION_NAME, 'prior-maint-session');

    createTask({
      id: 'store-msg-task',
      group_folder: 'main',
      chat_jid: chatJid,
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const streamedText = 'heartbeat: nothing urgent';
    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: streamedText,
          newSessionId: 'new-maint-session',
        } as ContainerOutput);
        return { status: 'success', result: streamedText };
      },
    );

    const sentTexts: string[] = [];
    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [chatJid]: MAIN_GROUP }),
      getSessions: () => ({ main: { maintenance: 'prior-maint-session' } }),
      queue: { enqueueTask, closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async (_jid: string, text: string) => {
        sentTexts.push(text);
      },
    });

    await vi.advanceTimersByTimeAsync(10);

    // sendMessage was called with the streamed text.
    expect(sentTexts).toEqual([streamedText]);

    // And — the critical assertion — storeMessage was also called, so
    // `messages.db` now has a bot row for this chat. Without the fix
    // the DB would have no row, exposing the "ghost send" bug #81.
    // getLastBotMessageTimestamp returns undefined when no bot row
    // exists for the chat; a string timestamp when one was written.
    const botTs = getLastBotMessageTimestamp(chatJid, 'bot');
    expect(botTs).toBeTruthy();
  });

  it('streamed scheduled-task writes a bot row even when no prior chats row exists', async () => {
    // The FK from `messages.chat_jid` to `chats.jid` means storeMessage
    // throws if no chats row exists for the target chat (scheduled task
    // firing before any inbound message / metadata sync would create
    // one). Verifies the task-scheduler upserts chat metadata before
    // storeMessage so the bot row actually lands in the DB instead of
    // raising a FOREIGN KEY constraint error and recording the run as
    // an error.
    const FRESH_GROUP = {
      name: 'Fresh',
      folder: 'fresh',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const chatJid = 'fresh-no-metadata@g.us';
    // NOTE: NOT calling storeChatMetadata here. The task-scheduler fix
    // must handle the missing-chats-row case on its own AND must write
    // a correctly-shaped chats row (channel='whatsapp', is_group=true
    // for `@g.us` JIDs) so the chat shows up in `getAvailableGroups()`.

    createTask({
      id: 'fresh-chat-task',
      group_folder: 'fresh',
      chat_jid: chatJid,
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    const streamedText = 'first send in a fresh chat';
    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: streamedText,
        } as ContainerOutput);
        return { status: 'success', result: streamedText };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [chatJid]: FRESH_GROUP }),
      getSessions: () => ({}),
      queue: { enqueueTask, closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
    });

    await vi.advanceTimersByTimeAsync(10);

    // Bot row landed despite no pre-existing chats row. If the
    // storeChatMetadata upsert gets dropped, storeMessage will throw
    // FOREIGN KEY constraint failed and this assertion fails.
    expect(getLastBotMessageTimestamp(chatJid, 'bot')).toBeTruthy();

    // And the chats row itself has the right shape — a `@g.us` JID
    // infers `channel: 'whatsapp'`, `is_group: true`, so the chat
    // appears in `getAvailableGroups()` (which filters on
    // `c.is_group`). Missing/NULL here would hide the group from
    // every downstream consumer — the exact behavior Copilot
    // flagged on #83 round 3.
    const chat = getAllChats().find((c) => c.jid === chatJid);
    expect(chat).toBeTruthy();
    expect(chat!.channel).toBe('whatsapp');
    expect(chat!.is_group).toBe(1);
  });

  it('streamed scheduled-task to a Telegram group upserts chats as telegram+group', async () => {
    // Mirror of the `@g.us` test for the Telegram path: negative id
    // after `tg:` indicates a group/channel, positive indicates a
    // private 1:1 — both should infer `channel: 'telegram'`, and
    // only the negative-id case should set `is_group: true`. Guards
    // against channel-prefix abbreviations (`'tg'`) ever landing in
    // the DB.
    const GROUP_REG = {
      name: 'TG Group',
      folder: 'tgg',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const groupJid = 'tg:-1003000000001';
    const dmJid = 'tg:42';

    for (const [jid, taskId] of [
      [groupJid, 'tg-group-task'],
      [dmJid, 'tg-dm-task'],
    ]) {
      createTask({
        id: taskId,
        group_folder: 'tgg',
        chat_jid: jid,
        prompt: 'run',
        schedule_type: 'once',
        schedule_value: '2026-01-01T00:00:00.000Z',
        context_mode: 'group',
        next_run: new Date(Date.now() - 1000).toISOString(),
        status: 'active',
        created_at: '2026-01-01T00:00:00.000Z',
        created_by_role: 'owner' as const,
      });
    }

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [groupJid]: GROUP_REG, [dmJid]: GROUP_REG }),
      getSessions: () => ({}),
      queue: { enqueueTask, closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
    });

    await vi.advanceTimersByTimeAsync(10);

    const group = getAllChats().find((c) => c.jid === groupJid);
    expect(group).toBeTruthy();
    expect(group!.channel).toBe('telegram');
    expect(group!.is_group).toBe(1);

    const dm = getAllChats().find((c) => c.jid === dmJid);
    expect(dm).toBeTruthy();
    expect(dm!.channel).toBe('telegram');
    expect(dm!.is_group).toBe(0);
  });

  it('streamed scheduled-task to a WhatsApp DM (@s.whatsapp.net) upserts chats as whatsapp+dm', async () => {
    // Extension of the TG-group/TG-DM test for WhatsApp's DM JID
    // shape. Matches the db.ts legacy backfill convention
    // (`@s.whatsapp.net` → whatsapp + is_group=0). Without this
    // branch, a first send to a WA DM would land with channel/is_group
    // NULL and the chat would be invisible to getAvailableGroups().
    const GROUP_REG = {
      name: 'WA DM',
      folder: 'wadm',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const chatJid = '15555555555@s.whatsapp.net';

    createTask({
      id: 'wa-dm-task',
      group_folder: 'wadm',
      chat_jid: chatJid,
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: 'ok',
        } as ContainerOutput);
        return { status: 'success', result: 'ok' };
      },
    );

    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [chatJid]: GROUP_REG }),
      getSessions: () => ({}),
      queue: { enqueueTask, closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async () => {},
    });

    await vi.advanceTimersByTimeAsync(10);

    const chat = getAllChats().find((c) => c.jid === chatJid);
    expect(chat).toBeTruthy();
    expect(chat!.channel).toBe('whatsapp');
    expect(chat!.is_group).toBe(0);
  });

  it('streamed scheduled-task with all-internal result does NOT write a bot row', async () => {
    // Sibling regression: if the streamed text is ENTIRELY wrapped in
    // `<internal>…</internal>` tags, the stripped `cleanResult` is
    // empty and the `if (cleanResult)` guard short-circuits both the
    // send AND the store. Verifies the store gate is aligned with the
    // send gate — we don't accidentally write empty-content rows.
    const MAIN_GROUP = {
      name: 'Main',
      folder: 'main',
      trigger: 'always',
      added_at: '2026-01-01T00:00:00.000Z',
      isMain: true,
    };
    const chatJid = 'internal-only@g.us';
    storeChatMetadata(chatJid, '2026-01-01T00:00:00.000Z', 'InternalOnly');

    createTask({
      id: 'internal-only-task',
      group_folder: 'main',
      chat_jid: chatJid,
      prompt: 'run',
      schedule_type: 'once',
      schedule_value: '2026-01-01T00:00:00.000Z',
      context_mode: 'group',
      next_run: new Date(Date.now() - 1000).toISOString(),
      status: 'active',
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    });

    mockRunContainerAgent.mockImplementation(
      async (_group, _input, _onProc, onOutput) => {
        await onOutput({
          status: 'success',
          result: '<internal>debug only — not for user</internal>',
        } as ContainerOutput);
        return { status: 'success', result: '' };
      },
    );

    const sentTexts: string[] = [];
    const enqueueTask = vi.fn(
      (
        _groupJid: string,
        _taskId: string,
        _sessionName: string,
        fn: () => Promise<void>,
      ) => {
        void fn();
      },
    );

    startSchedulerLoop({
      registeredGroups: () => ({ [chatJid]: MAIN_GROUP }),
      getSessions: () => ({}),
      queue: { enqueueTask, closeStdin: vi.fn() } as never,
      onProcess: () => {},
      sendMessage: async (_jid: string, text: string) => {
        sentTexts.push(text);
      },
    });

    await vi.advanceTimersByTimeAsync(10);

    // No send happened (cleanResult was empty after strip).
    expect(sentTexts).toEqual([]);
    // And no bot row was written for this chat — the store gate must
    // be aligned with the send gate, otherwise we'd get empty-content
    // rows polluting heartbeat's answered-check accounting.
    expect(getLastBotMessageTimestamp(chatJid, 'bot')).toBeFalsy();
  });

  it('computeNextRun skips missed intervals without infinite loop', () => {
    // Task was due 10 intervals ago (missed)
    const ms = 60000;
    const missedBy = ms * 10;
    const scheduledTime = new Date(Date.now() - missedBy).toISOString();

    const task = {
      id: 'skip-test',
      group_folder: 'test',
      chat_jid: 'test@g.us',
      prompt: 'test',
      schedule_type: 'interval' as const,
      schedule_value: String(ms),
      context_mode: 'isolated' as const,
      next_run: scheduledTime,
      last_run: null,
      last_result: null,
      status: 'active' as const,
      created_at: '2026-01-01T00:00:00.000Z',
      created_by_role: 'owner' as const,
    };

    const nextRun = computeNextRun(task);
    expect(nextRun).not.toBeNull();
    // Must be in the future
    expect(new Date(nextRun!).getTime()).toBeGreaterThan(Date.now());
    // Must be aligned to the original schedule grid
    const offset =
      (new Date(nextRun!).getTime() - new Date(scheduledTime).getTime()) % ms;
    expect(offset).toBe(0);
  });
});
