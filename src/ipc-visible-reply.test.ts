// Outcome-level coverage for the #722 IPC visible-send anchor release:
// drives the REAL startIpcWatcher loop against a tempdir IPC namespace
// and asserts the `onVisibleReply` hook fires (or doesn't) at the
// send_message / send_file boundary. The pure anchor helpers are
// covered in agent-output-action.test.ts; this file pins the wiring —
// the handlers could otherwise silently stop calling the hook without
// any test noticing (PR #723 review).
//
// `vi.mock` is hoisted above top-level consts; the tempdir is computed
// inside `vi.hoisted` so the config-mock factory and the test body
// share it (same pattern as ipc-auth.test.ts).
const { TEST_DATA_DIR, TEST_GROUPS_DIR } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const osMod = require('os') as typeof import('os');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pathMod = require('path') as typeof import('path');
  const base = pathMod.join(
    osMod.tmpdir(),
    `nanoclaw-ipc-visible-reply-test-${process.pid}`,
  );
  return {
    TEST_DATA_DIR: pathMod.join(base, 'data'),
    TEST_GROUPS_DIR: pathMod.join(base, 'groups'),
  };
});
vi.mock('./config.js', async () => {
  const actual =
    await vi.importActual<typeof import('./config.js')>('./config.js');
  return {
    ...actual,
    DATA_DIR: TEST_DATA_DIR,
    GROUPS_DIR: TEST_GROUPS_DIR,
    // Keep re-polls fast so a test that needs a second pass isn't slow;
    // the stop handle cancels the pending poll at teardown either way.
    IPC_POLL_INTERVAL: 25,
  };
});

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

import { _initTestDatabase, setRegisteredGroup } from './db.js';
import { startIpcWatcher, IpcDeps } from './ipc.js';
import { RegisteredGroup } from './types.js';

const GROUP: RegisteredGroup = {
  name: 'Grp',
  folder: 'grp',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};
const CHAT_JID = 'chat@g.us';

let stopWatcher: (() => void) | undefined;
let onVisibleReply: ReturnType<typeof vi.fn>;

function makeDeps(overrides: Partial<IpcDeps>): IpcDeps {
  return {
    sendMessage: async () => {},
    registeredGroups: () => ({ [CHAT_JID]: GROUP }),
    registerGroup: () => {},
    unregisterGroup: () => false,
    setGroupTrusted: () => false,
    setGroupTrigger: () => false,
    syncGroups: async () => {},
    getAvailableGroups: () => [],
    writeGroupsSnapshot: () => {},
    onTasksChanged: () => {},
    nukeSession: () => {},
    closeAllActiveContainers: () => 0,
    onVisibleReply: onVisibleReply as unknown as IpcDeps['onVisibleReply'],
    ...overrides,
  };
}

function writeIpcMessage(payload: Record<string, unknown>): void {
  const dir = path.join(TEST_DATA_DIR, 'ipc', GROUP.folder, 'messages');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    path.join(
      dir,
      `msg-${Date.now()}-${Math.random().toString(36).slice(2)}.json`,
    ),
    JSON.stringify(payload),
  );
}

beforeEach(() => {
  _initTestDatabase();
  setRegisteredGroup(CHAT_JID, GROUP);
  onVisibleReply = vi.fn();
  fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
  fs.rmSync(TEST_GROUPS_DIR, { recursive: true, force: true });
});

afterEach(() => {
  stopWatcher?.();
  stopWatcher = undefined;
  fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
  fs.rmSync(TEST_GROUPS_DIR, { recursive: true, force: true });
});

describe('IPC visible-send anchor hook (#722)', () => {
  it('fires onVisibleReply after a delivered send_message', async () => {
    // Wire type for the send_message tool is 'message' (see the
    // `data.type === 'message'` guard in ipc.ts).
    writeIpcMessage({ type: 'message', chatJid: CHAT_JID, text: 'hi' });
    const sendMessage = vi.fn(async () => 'tg-1');
    stopWatcher = startIpcWatcher(makeDeps({ sendMessage }));

    await vi.waitFor(() => {
      expect(onVisibleReply).toHaveBeenCalledWith(CHAT_JID, GROUP.folder);
    });
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });

  it('does NOT fire onVisibleReply when delivery fails (no message id)', async () => {
    writeIpcMessage({ type: 'message', chatJid: CHAT_JID, text: 'hi' });
    const sendMessage = vi.fn(async () => undefined);
    stopWatcher = startIpcWatcher(makeDeps({ sendMessage }));

    // Wait until the handler has definitely processed the file (the
    // send mock was invoked), then assert the hook stayed silent — a
    // failed delivery is not a visible reply.
    await vi.waitFor(() => {
      expect(sendMessage).toHaveBeenCalledTimes(1);
    });
    expect(onVisibleReply).not.toHaveBeenCalled();
  });

  it('fires onVisibleReply after a delivered send_file', async () => {
    // send_file translates /workspace/group/<rel> to GROUPS_DIR/<folder>/<rel>
    // and requires the host file to exist.
    const hostFile = path.join(TEST_GROUPS_DIR, GROUP.folder, 'report.txt');
    fs.mkdirSync(path.dirname(hostFile), { recursive: true });
    fs.writeFileSync(hostFile, 'payload');
    writeIpcMessage({
      type: 'send_file',
      chatJid: CHAT_JID,
      filePath: '/workspace/group/report.txt',
    });
    const sendFile = vi.fn(async () => 'tg-2');
    stopWatcher = startIpcWatcher(makeDeps({ sendFile }));

    await vi.waitFor(() => {
      expect(onVisibleReply).toHaveBeenCalledWith(CHAT_JID, GROUP.folder);
    });
    expect(sendFile).toHaveBeenCalledTimes(1);
  });
});
