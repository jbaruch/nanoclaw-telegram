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

// Same TEST_DATA_DIR isolation pattern as chat-status.test.ts:
// processTaskIpc writes `_script_result_<requestId>.json` files via
// scriptResultPath, which uses DATA_DIR. Mocking it to a per-pid
// tempdir keeps these tests off the developer's real ipc tree.
const { TEST_DATA_DIR } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const osMod = require('os') as typeof import('os');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pathMod = require('path') as typeof import('path');
  return {
    TEST_DATA_DIR: pathMod.join(
      osMod.tmpdir(),
      `nanoclaw-send-message-to-chat-test-${process.pid}`,
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

// sendPoolMessage is imported directly into ipc.ts (not via deps) and
// reaches into Telegram bot-pool state we don't want to spin up here.
// Mock it as a vi.fn so the test can assert call shape and program a
// return value (the per-test poolReturn variable below). Returning
// undefined exercises the #232 phantom-row gate path.
const { poolMockFn } = vi.hoisted(() => ({
  poolMockFn: vi.fn(),
}));
vi.mock('./channels/telegram.js', () => ({
  sendPoolMessage: poolMockFn,
}));

import path from 'path';

import {
  _initTestDatabase,
  getLastFromMeMessage,
  storeChatMetadata,
} from './db.js';
import { processTaskIpc, IpcDeps } from './ipc.js';
import { RegisteredGroup } from './types.js';

const MAIN_GROUP: RegisteredGroup = {
  name: 'Main',
  folder: 'whatsapp_main',
  trigger: 'always',
  added_at: '2024-01-01T00:00:00.000Z',
  isMain: true,
};

const TELEGRAM_GROUP: RegisteredGroup = {
  name: 'Family',
  folder: 'family-group',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

const WHATSAPP_GROUP: RegisteredGroup = {
  name: 'Random Chat',
  folder: 'random-group',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

const TWIN_A: RegisteredGroup = {
  name: 'Twin',
  folder: 'twin-a',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};
const TWIN_B: RegisteredGroup = {
  name: 'Twin',
  folder: 'twin-b',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

const TG_JID = 'tg:-1003869886477';
const WA_JID = '120363012345@g.us';
const TWIN_A_JID = 'tg:-1001111';
const TWIN_B_JID = 'tg:-1002222';

let groups: Record<string, RegisteredGroup>;
let sendCalls: Array<{ jid: string; text: string; replyTo?: string }>;
let pinCalls: Array<{ jid: string; messageId: string }>;
let sendReturn: string | undefined;
let sendShouldThrow: Error | null;
let deps: IpcDeps;

const ADMIN_INPUT_DIR = path.join(
  TEST_DATA_DIR,
  'ipc',
  MAIN_GROUP.folder,
  'input-default',
);
const NON_ADMIN_INPUT_DIR = path.join(
  TEST_DATA_DIR,
  'ipc',
  WHATSAPP_GROUP.folder,
  'input-default',
);

function ensureDirs(): void {
  fs.mkdirSync(ADMIN_INPUT_DIR, { recursive: true });
  fs.mkdirSync(NON_ADMIN_INPUT_DIR, { recursive: true });
}

function readResult(sourceFolder: string, requestId: string): unknown {
  const p = path.join(
    TEST_DATA_DIR,
    'ipc',
    sourceFolder,
    'input-default',
    `_script_result_${requestId}.json`,
  );
  return JSON.parse(fs.readFileSync(p, 'utf-8'));
}

beforeEach(() => {
  _initTestDatabase();
  ensureDirs();
  // Foreign-key seed: bot-row writes target storeMessage, which has
  // an FK on chats.jid. Seed every JID we expect a successful send to.
  storeChatMetadata(
    TG_JID,
    '2026-04-25T09:00:00.000Z',
    'Family',
    'telegram',
    true,
  );
  storeChatMetadata(
    WA_JID,
    '2026-04-25T09:00:00.000Z',
    'Random Chat',
    'whatsapp',
    true,
  );

  groups = {
    'main@g.us': MAIN_GROUP,
    [TG_JID]: TELEGRAM_GROUP,
    [WA_JID]: WHATSAPP_GROUP,
    [TWIN_A_JID]: TWIN_A,
    [TWIN_B_JID]: TWIN_B,
  };
  sendCalls = [];
  pinCalls = [];
  sendReturn = 'msg-42';
  sendShouldThrow = null;

  poolMockFn.mockReset();

  deps = {
    sendMessage: async (jid, text, replyToMessageId) => {
      if (sendShouldThrow) throw sendShouldThrow;
      sendCalls.push({ jid, text, replyTo: replyToMessageId });
      return sendReturn;
    },
    pinMessage: async (jid, messageId) => {
      pinCalls.push({ jid, messageId });
    },
    registeredGroups: () => groups,
    registerGroup: () => {},
    unregisterGroup: () => false,
    setGroupTrusted: () => true,
    setGroupTrigger: () => true,
    syncGroups: async () => {},
    getAvailableGroups: () => [],
    writeGroupsSnapshot: () => {},
    onTasksChanged: () => {},
    nukeSession: () => {},
  };
});

afterEach(() => {
  const ipcRoot = path.join(TEST_DATA_DIR, 'ipc');
  if (fs.existsSync(ipcRoot)) {
    fs.rmSync(ipcRoot, { recursive: true, force: true });
  }
});

afterAll(() => {
  if (fs.existsSync(TEST_DATA_DIR)) {
    fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
  }
});

// --- authorization ---

describe('send_message_to_chat authorization', () => {
  it('non-main group is rejected with an error response, no send attempted', async () => {
    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'auth-1',
        chat_id: TG_JID,
        text: 'hi',
      },
      WHATSAPP_GROUP.folder,
      false,
      deps,
    );

    const body = readResult(WHATSAPP_GROUP.folder, 'auth-1') as {
      error?: string;
    };
    expect(body.error).toMatch(/admin-tile only/);
    expect(sendCalls).toHaveLength(0);
  });
});

// --- input validation ---

describe('send_message_to_chat input validation', () => {
  it('rejects when neither chat_id nor chat_name is provided', async () => {
    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'noid',
        text: 'hi',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'noid') as { error?: string };
    expect(body.error).toMatch(/cross-chat/);
    expect(sendCalls).toHaveLength(0);
  });

  it('rejects when both chat_id and chat_name are provided', async () => {
    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'both',
        chat_id: TG_JID,
        chat_name: 'Family',
        text: 'hi',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'both') as { error?: string };
    expect(body.error).toMatch(/not both/);
    expect(sendCalls).toHaveLength(0);
  });

  it('rejects when text is missing or empty', async () => {
    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'no-text',
        chat_id: TG_JID,
        text: '   ',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'no-text') as {
      error?: string;
    };
    expect(body.error).toMatch(/non-empty text/);
    expect(sendCalls).toHaveLength(0);
  });

  it('rejects when text reduces to empty after stripping <internal> tags', async () => {
    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'empty-after-strip',
        chat_id: TG_JID,
        text: '<internal>thinking out loud</internal>',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'empty-after-strip') as {
      error?: string;
    };
    expect(body.error).toMatch(/empty after stripping/);
    expect(sendCalls).toHaveLength(0);
  });

  it('rejects an unregistered chat_id', async () => {
    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'ghost',
        chat_id: 'tg:-9999999',
        text: 'hi',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'ghost') as { error?: string };
    expect(body.error).toMatch(/not registered/);
    expect(sendCalls).toHaveLength(0);
  });

  it('rejects a chat_name that does not match any registered chat', async () => {
    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'no-match',
        chat_name: 'Nonexistent',
        text: 'hi',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'no-match') as {
      error?: string;
    };
    expect(body.error).toMatch(/did not match/);
    expect(sendCalls).toHaveLength(0);
  });

  it('returns ambiguity error with candidate JIDs when chat_name matches multiple chats', async () => {
    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'ambig',
        chat_name: 'Twin',
        text: 'hi',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'ambig') as {
      error?: string;
      candidates?: string[];
    };
    expect(body.error).toMatch(/ambiguous/);
    expect(body.candidates).toEqual(
      expect.arrayContaining([TWIN_A_JID, TWIN_B_JID]),
    );
    expect(sendCalls).toHaveLength(0);
  });
});

// --- happy paths ---

describe('send_message_to_chat dispatch', () => {
  it('chat_id resolves and sends via direct path; bot row written with telegram message id', async () => {
    sendReturn = 'tg-msg-100';

    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'happy-id',
        chat_id: TG_JID,
        text: 'hello family',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    expect(sendCalls).toEqual([
      { jid: TG_JID, text: 'hello family', replyTo: undefined },
    ]);
    expect(poolMockFn).not.toHaveBeenCalled();

    const body = readResult(MAIN_GROUP.folder, 'happy-id') as {
      stdout: string;
    };
    const payload = JSON.parse(body.stdout);
    expect(payload).toMatchObject({
      chat_id: TG_JID,
      chat_name: 'Family',
      sent_message_id: 'tg-msg-100',
      pinned: false,
      status: 'success',
    });

    // Verify a bot row landed in messages.db so the heartbeat /
    // unanswered-cron sees the broadcast.
    const last = getLastFromMeMessage(TG_JID);
    expect(last).not.toBeNull();
    expect(last!.content).toBe('hello family');
  });

  it('chat_name resolves to JID and sends', async () => {
    sendReturn = 'tg-msg-200';

    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'happy-name',
        chat_name: 'Family',
        text: 'meet at 7',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    expect(sendCalls).toEqual([
      { jid: TG_JID, text: 'meet at 7', replyTo: undefined },
    ]);
    const body = readResult(MAIN_GROUP.folder, 'happy-name') as {
      stdout: string;
    };
    const payload = JSON.parse(body.stdout);
    expect(payload.chat_id).toBe(TG_JID);
  });

  it('strips <internal> tags from text before sending and storing', async () => {
    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'strip',
        chat_id: TG_JID,
        text: '<internal>plan: be casual</internal>Hey there!',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    expect(sendCalls).toHaveLength(1);
    expect(sendCalls[0].text).toBe('Hey there!');
    expect(sendCalls[0].text).not.toMatch(/<internal>/);
  });

  it('routes through the bot pool when sender is set on a Telegram chat', async () => {
    poolMockFn.mockResolvedValue('pool-msg-50');

    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'pool',
        chat_id: TG_JID,
        text: 'hi',
        sender: 'Researcher',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    expect(poolMockFn).toHaveBeenCalledTimes(1);
    expect(poolMockFn).toHaveBeenCalledWith(
      TG_JID,
      'hi',
      'Researcher',
      MAIN_GROUP.folder,
    );
    expect(sendCalls).toHaveLength(0);

    const body = readResult(MAIN_GROUP.folder, 'pool') as { stdout: string };
    const payload = JSON.parse(body.stdout);
    expect(payload.sent_message_id).toBe('pool-msg-50');
    // Pool path can't pin — silently dropped, reflected in the response.
    expect(payload.pinned).toBe(false);
  });

  it('does NOT route through the bot pool for non-Telegram chats even when sender is set', async () => {
    sendReturn = 'wa-msg-1';

    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'wa-with-sender',
        chat_id: WA_JID,
        text: 'hi',
        sender: 'Researcher',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    expect(poolMockFn).not.toHaveBeenCalled();
    expect(sendCalls).toHaveLength(1);
    expect(sendCalls[0].jid).toBe(WA_JID);
  });

  it('pins the message on the direct path when pin: true and send returned an id', async () => {
    sendReturn = 'tg-msg-pinned';

    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'pin-1',
        chat_id: TG_JID,
        text: 'announcement',
        pin: true,
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    expect(pinCalls).toEqual([{ jid: TG_JID, messageId: 'tg-msg-pinned' }]);
    const body = readResult(MAIN_GROUP.folder, 'pin-1') as { stdout: string };
    const payload = JSON.parse(body.stdout);
    expect(payload.pinned).toBe(true);
  });

  it('does NOT pin when sender is set (pool path) and reports pinned=false', async () => {
    poolMockFn.mockResolvedValue('pool-msg-99');

    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'pin-pool',
        chat_id: TG_JID,
        text: 'broadcast',
        pin: true,
        sender: 'Bot',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    expect(pinCalls).toHaveLength(0);
    const body = readResult(MAIN_GROUP.folder, 'pin-pool') as {
      stdout: string;
    };
    const payload = JSON.parse(body.stdout);
    expect(payload.pinned).toBe(false);
  });
});

// --- failure surfacing (#232 regression coverage for this entry point) ---

describe('send_message_to_chat failure surfacing', () => {
  it('Telegram returning undefined sentMsgId surfaces error and writes NO bot row in target DB', async () => {
    // sendReturn = undefined → swallowed Telegram send (the textbook
    // phantom-row scenario from #232). For send_message_to_chat the
    // stakes are higher — the target chat isn't being watched here,
    // so a phantom bot- row would silence the heartbeat there.
    sendReturn = undefined;

    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'tg-fail',
        chat_id: TG_JID,
        text: 'hi',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'tg-fail') as {
      error?: string;
      chat_id?: string;
      status?: string;
    };
    expect(body.error).toMatch(/did not return a message id/);
    expect(body.status).toBe('failed');
    expect(body.chat_id).toBe(TG_JID);

    expect(getLastFromMeMessage(TG_JID)).toBeNull();
  });

  it('non-Telegram send returning void still writes the bot row (gate only applies to Telegram)', async () => {
    // The shouldStoreBotMessage gate is Telegram-specific —
    // WhatsApp's Channel.sendMessage contract permits void on
    // success (see src/types.ts), so absence of an id is not a
    // failure signal there.
    sendReturn = undefined;

    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'wa-void',
        chat_id: WA_JID,
        text: 'hi',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'wa-void') as {
      stdout?: string;
      error?: string;
    };
    expect(body.error).toBeUndefined();
    const payload = JSON.parse(body.stdout!);
    expect(payload.status).toBe('success');

    const last = getLastFromMeMessage(WA_JID);
    expect(last).not.toBeNull();
    expect(last!.content).toBe('hi');
  });

  it('exception from sendMessage surfaces as top-level error with chat metadata', async () => {
    sendShouldThrow = new Error('connection reset');

    await processTaskIpc(
      {
        type: 'send_message_to_chat',
        requestId: 'throw',
        chat_id: TG_JID,
        text: 'hi',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );

    const body = readResult(MAIN_GROUP.folder, 'throw') as {
      error?: string;
      chat_id?: string;
      chat_name?: string;
      status?: string;
    };
    // Top-level error — runHostOperation in the agent-runner only
    // surfaces isError: true on result.error, so wrapping the
    // failure in stdout would falsely look like a successful send.
    expect(body.error).toMatch(/connection reset/);
    expect(body.error).toContain(TG_JID);
    expect(body.status).toBe('error');
    expect(body.chat_id).toBe(TG_JID);
    expect(body.chat_name).toBe('Family');

    expect(getLastFromMeMessage(TG_JID)).toBeNull();
  });
});
