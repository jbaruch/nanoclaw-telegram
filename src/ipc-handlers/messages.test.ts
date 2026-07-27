import fs from 'fs';
import path from 'path';

import { describe, it, expect, beforeEach, afterAll, vi } from 'vitest';

// Isolate filesystem writes to a per-process tempdir — same shape as
// `ipc-auth.test.ts`. `vi.mock` hoists above top-level consts, so the
// tempdir is computed inside `vi.hoisted` and shared with the factory.
const { TEST_DATA_DIR, TEST_GROUPS_DIR } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const osMod = require('os') as typeof import('os');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pathMod = require('path') as typeof import('path');
  const base = pathMod.join(
    osMod.tmpdir(),
    `nanoclaw-ipc-messages-test-${process.pid}`,
  );
  return {
    TEST_DATA_DIR: pathMod.join(base, 'data'),
    TEST_GROUPS_DIR: pathMod.join(base, 'groups'),
  };
});
vi.mock('../config.js', async () => {
  const actual =
    await vi.importActual<typeof import('../config.js')>('../config.js');
  return { ...actual, DATA_DIR: TEST_DATA_DIR, GROUPS_DIR: TEST_GROUPS_DIR };
});

import {
  _resetIpcMessageRegistryForTests,
  dispatchIpcMessage,
  type IpcMessagePayload,
} from '../ipc-message-registry.js';
import {
  _resetMessageIpcHandlersForTests,
  registerMessageIpcHandlers,
} from './messages.js';
import type { IpcDeps } from '../ipc.js';
import type { RegisteredGroup } from '../types.js';

const SOURCE_GROUP = 'grp';
const CHAT_JID = 'chat@g.us';
const GROUP: RegisteredGroup = {
  name: 'Grp',
  folder: SOURCE_GROUP,
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

let sendFile: ReturnType<typeof vi.fn>;
let groupDir: string;
/** A host file OUTSIDE every allowed mount — the exfiltration target. */
let secretPath: string;

function deps(): IpcDeps {
  return { sendFile } as unknown as IpcDeps;
}

async function dispatch(data: IpcMessagePayload): Promise<void> {
  await dispatchIpcMessage({
    data,
    sourceGroup: SOURCE_GROUP,
    isMain: false,
    registeredGroups: { [CHAT_JID]: GROUP },
    deps: deps(),
    file: 'msg-1.json',
  });
}

beforeEach(() => {
  fs.rmSync(path.dirname(TEST_DATA_DIR), { recursive: true, force: true });
  groupDir = path.join(TEST_GROUPS_DIR, SOURCE_GROUP);
  fs.mkdirSync(groupDir, { recursive: true });
  secretPath = path.join(path.dirname(TEST_DATA_DIR), '.env');
  // Neutral sentinel, deliberately not credential-shaped: the test only
  // needs to prove this file never reaches sendFile, and a committed
  // token-looking fixture trips secret scanners (`coding-policy:
  // no-secrets` bans them even in tests).
  fs.writeFileSync(secretPath, 'EXFIL_MARKER=not-a-secret\n');
  sendFile = vi.fn().mockResolvedValue('sent-1');
  _resetIpcMessageRegistryForTests();
  _resetMessageIpcHandlersForTests();
  registerMessageIpcHandlers();
});

afterAll(() => {
  fs.rmSync(path.dirname(TEST_DATA_DIR), { recursive: true, force: true });
});

describe('send_file mount containment', () => {
  it('sends a file that genuinely lives inside the group mount', async () => {
    // Positive control: without this, every "refused" assertion below
    // would also pass on a handler that refuses unconditionally.
    fs.writeFileSync(path.join(groupDir, 'report.pdf'), 'payload');
    await dispatch({
      type: 'send_file',
      chatJid: CHAT_JID,
      filePath: '/workspace/group/report.pdf',
    });
    expect(sendFile).toHaveBeenCalledOnce();
    expect(sendFile.mock.calls[0][1]).toBe(
      fs.realpathSync(path.join(groupDir, 'report.pdf')),
    );
  });

  it('refuses a ../ traversal that escapes the group mount', async () => {
    // `/workspace/group/../../.env` used to path.join its way onto the
    // host's .env and hand it to the chat.
    await dispatch({
      type: 'send_file',
      chatJid: CHAT_JID,
      filePath: '/workspace/group/../../.env',
    });
    expect(sendFile).not.toHaveBeenCalled();
  });

  it('refuses a traversal buried mid-path', async () => {
    fs.mkdirSync(path.join(groupDir, 'sub'), { recursive: true });
    await dispatch({
      type: 'send_file',
      chatJid: CHAT_JID,
      filePath: '/workspace/group/sub/../../../.env',
    });
    expect(sendFile).not.toHaveBeenCalled();
  });

  it('refuses a sibling group even without leaving GROUPS_DIR', async () => {
    const otherDir = path.join(TEST_GROUPS_DIR, 'other-group');
    fs.mkdirSync(otherDir, { recursive: true });
    fs.writeFileSync(path.join(otherDir, 'private.txt'), 'not yours');
    await dispatch({
      type: 'send_file',
      chatJid: CHAT_JID,
      filePath: '/workspace/group/../other-group/private.txt',
    });
    expect(sendFile).not.toHaveBeenCalled();
  });

  it('refuses a symlink planted inside the group that points outside it', async () => {
    // The group folder is container-writable, so a prefix check that
    // ignores symlinks is the same escape by another route.
    fs.symlinkSync(secretPath, path.join(groupDir, 'innocent.txt'));
    await dispatch({
      type: 'send_file',
      chatJid: CHAT_JID,
      filePath: '/workspace/group/innocent.txt',
    });
    expect(sendFile).not.toHaveBeenCalled();
  });

  it('still refuses a path outside every advertised mount', async () => {
    await dispatch({
      type: 'send_file',
      chatJid: CHAT_JID,
      filePath: '/etc/passwd',
    });
    expect(sendFile).not.toHaveBeenCalled();
  });

  it('reports a missing in-mount file without sending anything', async () => {
    await dispatch({
      type: 'send_file',
      chatJid: CHAT_JID,
      filePath: '/workspace/group/nope.pdf',
    });
    expect(sendFile).not.toHaveBeenCalled();
  });

  it('refuses a cross-group send from an unauthorized sender before any path work', async () => {
    fs.writeFileSync(path.join(groupDir, 'report.pdf'), 'payload');
    await dispatchIpcMessage({
      data: {
        type: 'send_file',
        chatJid: 'someone-else@g.us',
        filePath: '/workspace/group/report.pdf',
      },
      sourceGroup: SOURCE_GROUP,
      isMain: false,
      registeredGroups: { [CHAT_JID]: GROUP },
      deps: deps(),
      file: 'msg-1.json',
    });
    expect(sendFile).not.toHaveBeenCalled();
  });
});
