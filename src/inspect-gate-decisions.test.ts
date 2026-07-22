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

// Same TEST_DATA_DIR isolation pattern as `chat-status.test.ts`:
// inspect_gate_decisions writes `_script_result_<requestId>.json` files
// via `scriptResultPath` (which uses `DATA_DIR`) and reads
// `data/host-logs/orchestrator.log` via `hostLogsOrchestratorFile`
// (which also uses `DATA_DIR`). One mock pins both into a per-pid
// tempdir so the test never touches a developer's real data dir or
// the orchestrator's live state.
const { TEST_DATA_DIR } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const osMod = require('os') as typeof import('os');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pathMod = require('path') as typeof import('path');
  return {
    TEST_DATA_DIR: pathMod.join(
      osMod.tmpdir(),
      `nanoclaw-inspect-gate-test-${process.pid}`,
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

import path from 'path';

import { _initTestDatabase } from './db.js';
import { processTaskIpc, IpcDeps } from './ipc.js';
import { RegisteredGroup } from './types.js';

const MAIN_GROUP: RegisteredGroup = {
  name: 'Main',
  folder: 'whatsapp_main',
  trigger: 'always',
  added_at: '2024-01-01T00:00:00.000Z',
  isMain: true,
};

const UNTRUSTED_GROUP: RegisteredGroup = {
  name: 'Random',
  folder: 'random-group',
  trigger: '@Andy',
  added_at: '2024-01-01T00:00:00.000Z',
};

const ADMIN_INPUT_DIR = path.join(
  TEST_DATA_DIR,
  'ipc',
  MAIN_GROUP.folder,
  'input-default',
);
const NON_ADMIN_INPUT_DIR = path.join(
  TEST_DATA_DIR,
  'ipc',
  UNTRUSTED_GROUP.folder,
  'input-default',
);
const HOST_LOG_PATH = path.join(TEST_DATA_DIR, 'host-logs', 'orchestrator.log');

let deps: IpcDeps;

function ensureDirs(): void {
  fs.mkdirSync(ADMIN_INPUT_DIR, { recursive: true });
  fs.mkdirSync(NON_ADMIN_INPUT_DIR, { recursive: true });
  fs.mkdirSync(path.dirname(HOST_LOG_PATH), { recursive: true });
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

function writeGateDecisionLogLine(args: {
  timestamp: string;
  chatJid: string;
  messageId: string;
  groupFolder: string;
  finalDecision: 'allow' | 'deny';
  reason: string;
  chain: Array<{
    gate: string;
    decision: 'allow' | 'deny' | 'pass';
    reason: string;
  }>;
}): string {
  return [
    `[${args.timestamp}] INFO (1): gate decision`,
    `    chatJid: ${JSON.stringify(args.chatJid)}`,
    `    messageId: ${JSON.stringify(args.messageId)}`,
    `    groupFolder: ${JSON.stringify(args.groupFolder)}`,
    `    finalDecision: ${JSON.stringify(args.finalDecision)}`,
    `    reason: ${JSON.stringify(args.reason)}`,
    `    chain: ${JSON.stringify(args.chain)}`,
    '',
  ].join('\n');
}

beforeEach(() => {
  _initTestDatabase();
  ensureDirs();
  deps = {
    sendMessage: async () => {},
    registeredGroups: () => ({
      'main@g.us': MAIN_GROUP,
      'random@g.us': UNTRUSTED_GROUP,
    }),
    registerGroup: () => {},
    unregisterGroup: () => false,
    setGroupTrusted: () => true,
    setGroupTrigger: () => true,
    syncGroups: async () => {},
    getAvailableGroups: () => [],
    writeGroupsSnapshot: () => {},
    onTasksChanged: () => {},
    nukeSession: () => {},
    closeAllActiveContainers: () => 0,
    getContainerStatus: () => 'not-spawned',
  };
});

afterEach(() => {
  // Wipe the IPC tree and the host-logs fixture so each test starts
  // fresh — leftover result files or log records would cross-contaminate.
  const ipcRoot = path.join(TEST_DATA_DIR, 'ipc');
  if (fs.existsSync(ipcRoot)) {
    fs.rmSync(ipcRoot, { recursive: true, force: true });
  }
  if (fs.existsSync(HOST_LOG_PATH)) fs.unlinkSync(HOST_LOG_PATH);
});

afterAll(() => {
  if (fs.existsSync(TEST_DATA_DIR)) {
    fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
  }
});

describe('inspect_gate_decisions authorization (#443)', () => {
  it('rejects non-main groups with admin-tile-only error', async () => {
    await processTaskIpc(
      {
        type: 'inspect_gate_decisions',
        requestId: 'unauth-1',
        chat_id: 'tg:-1',
      },
      UNTRUSTED_GROUP.folder,
      false,
      deps,
    );
    const body = readResult(UNTRUSTED_GROUP.folder, 'unauth-1') as {
      error?: string;
    };
    expect(body.error).toMatch(/admin-tile only/);
  });
});

describe('inspect_gate_decisions input validation (#443)', () => {
  it('rejects missing chat_id', async () => {
    await processTaskIpc(
      { type: 'inspect_gate_decisions', requestId: 'missing-cid' },
      MAIN_GROUP.folder,
      true,
      deps,
    );
    const body = readResult(MAIN_GROUP.folder, 'missing-cid') as {
      error?: string;
    };
    expect(body.error).toMatch(/requires chat_id/);
  });

  it('rejects empty / whitespace chat_id', async () => {
    await processTaskIpc(
      {
        type: 'inspect_gate_decisions',
        requestId: 'empty-cid',
        chat_id: '   ',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );
    const body = readResult(MAIN_GROUP.folder, 'empty-cid') as {
      error?: string;
    };
    expect(body.error).toMatch(/requires chat_id/);
  });

  it('rejects non-positive limit', async () => {
    await processTaskIpc(
      {
        type: 'inspect_gate_decisions',
        requestId: 'bad-limit',
        chat_id: 'tg:-1',
        limit: 0,
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );
    const body = readResult(MAIN_GROUP.folder, 'bad-limit') as {
      error?: string;
    };
    expect(body.error).toMatch(/positive integer/);
  });
});

describe('inspect_gate_decisions log search (#443)', () => {
  it('returns empty decisions array when no matching gate-decision records exist', async () => {
    // The logger itself writes to `orchestrator.log` via `host-logs.ts`,
    // so the file may exist as soon as any code path under the test
    // fires a `logger.info` (e.g. the handler's own "served via IPC"
    // line). What matters for this case is the structural one: a chat
    // with no `'gate decision'` records returns `{decisions: []}`
    // rather than failing — "no log file" and "no matching records"
    // are observationally identical for the caller.
    await processTaskIpc(
      {
        type: 'inspect_gate_decisions',
        requestId: 'no-log',
        chat_id: 'tg:-nonexistent',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );
    const body = readResult(MAIN_GROUP.folder, 'no-log') as { stdout: string };
    const payload = JSON.parse(body.stdout) as { decisions: unknown[] };
    expect(payload.decisions).toEqual([]);
  });

  it('returns gate-decision records for a chat, most-recent first', async () => {
    const log = [
      writeGateDecisionLogLine({
        timestamp: '12:00:00.000',
        chatJid: 'tg:-1',
        messageId: 'msg-a',
        groupFolder: 'whatsapp_main',
        finalDecision: 'allow',
        reason: 'short-circuit allow',
        chain: [{ gate: 'trigger', decision: 'allow', reason: 'matched' }],
      }),
      writeGateDecisionLogLine({
        timestamp: '12:00:01.000',
        chatJid: 'tg:-1',
        messageId: 'msg-b',
        groupFolder: 'whatsapp_main',
        finalDecision: 'deny',
        reason: 'gate-b-no: human-to-human',
        chain: [
          { gate: 'trigger', decision: 'allow', reason: 'matched' },
          {
            gate: 'gate-b',
            decision: 'deny',
            reason: 'human-to-human',
          },
        ],
      }),
      // Different chat — must NOT show up in the response.
      writeGateDecisionLogLine({
        timestamp: '12:00:02.000',
        chatJid: 'tg:-2',
        messageId: 'msg-other',
        groupFolder: 'other_group',
        finalDecision: 'allow',
        reason: 'unrelated',
        chain: [{ gate: 'trigger', decision: 'allow', reason: 'matched' }],
      }),
    ].join('');
    fs.writeFileSync(HOST_LOG_PATH, log);

    await processTaskIpc(
      {
        type: 'inspect_gate_decisions',
        requestId: 'search-1',
        chat_id: 'tg:-1',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );
    const body = readResult(MAIN_GROUP.folder, 'search-1') as {
      stdout: string;
    };
    const payload = JSON.parse(body.stdout) as {
      decisions: Array<Record<string, unknown>>;
    };
    expect(payload.decisions).toHaveLength(2);
    expect(payload.decisions[0]).toMatchObject({
      messageId: 'msg-b',
      finalDecision: 'deny',
      reason: 'gate-b-no: human-to-human',
    });
    expect(payload.decisions[0].chain).toEqual([
      { gate: 'trigger', decision: 'allow', reason: 'matched' },
      {
        gate: 'gate-b',
        decision: 'deny',
        reason: 'human-to-human',
      },
    ]);
    expect(payload.decisions[1].messageId).toBe('msg-a');
  });

  it('narrows by message_id when provided', async () => {
    const log = [
      writeGateDecisionLogLine({
        timestamp: '12:00:00.000',
        chatJid: 'tg:-1',
        messageId: 'msg-target',
        groupFolder: 'whatsapp_main',
        finalDecision: 'deny',
        reason: 'gate-b-no',
        chain: [
          { gate: 'trigger', decision: 'allow', reason: 'matched' },
          { gate: 'gate-b', decision: 'deny', reason: 'no-intent' },
        ],
      }),
      writeGateDecisionLogLine({
        timestamp: '12:00:01.000',
        chatJid: 'tg:-1',
        messageId: 'msg-other',
        groupFolder: 'whatsapp_main',
        finalDecision: 'allow',
        reason: 'unrelated',
        chain: [{ gate: 'trigger', decision: 'allow', reason: 'matched' }],
      }),
    ].join('');
    fs.writeFileSync(HOST_LOG_PATH, log);

    await processTaskIpc(
      {
        type: 'inspect_gate_decisions',
        requestId: 'narrow-1',
        chat_id: 'tg:-1',
        message_id: 'msg-target',
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );
    const body = readResult(MAIN_GROUP.folder, 'narrow-1') as {
      stdout: string;
    };
    const payload = JSON.parse(body.stdout) as {
      decisions: Array<Record<string, unknown>>;
    };
    expect(payload.decisions).toHaveLength(1);
    expect(payload.decisions[0].messageId).toBe('msg-target');
  });

  it('honors limit', async () => {
    let log = '';
    for (let i = 0; i < 25; i++) {
      log += writeGateDecisionLogLine({
        timestamp: `12:00:${String(i).padStart(2, '0')}.000`,
        chatJid: 'tg:-1',
        messageId: `msg-${i}`,
        groupFolder: 'whatsapp_main',
        finalDecision: 'allow',
        reason: `r${i}`,
        chain: [{ gate: 'trigger', decision: 'allow', reason: `r${i}` }],
      });
    }
    fs.writeFileSync(HOST_LOG_PATH, log);

    await processTaskIpc(
      {
        type: 'inspect_gate_decisions',
        requestId: 'limit-1',
        chat_id: 'tg:-1',
        limit: 5,
      },
      MAIN_GROUP.folder,
      true,
      deps,
    );
    const body = readResult(MAIN_GROUP.folder, 'limit-1') as {
      stdout: string;
    };
    const payload = JSON.parse(body.stdout) as {
      decisions: Array<Record<string, unknown>>;
    };
    expect(payload.decisions).toHaveLength(5);
    // Most-recent first.
    expect(payload.decisions.map((d) => d.messageId)).toEqual([
      'msg-24',
      'msg-23',
      'msg-22',
      'msg-21',
      'msg-20',
    ]);
  });
});

// PR #445 review feedback (Copilot): the parser contract sits across two
// files — `src/gates/orchestrator.ts:evaluateGateChain` produces the `'gate decision'`
// line, `src/host-log-parser.ts` consumes it. A silent rename of the
// message text or any field name in the producer would let the parser
// keep matching nothing while CI stays green. This producer-side spy
// test pins the wire format so the contract fails loudly at the source
// the moment it drifts.
describe('evaluateGateChain producer log shape (#443 / #445 review)', () => {
  it('emits a "gate decision" INFO line with the canonical field set', async () => {
    // Spy on the logger BEFORE importing the gate framework so the
    // call lands on our mock. `vi.resetModules()` would unmock the
    // logger between tests; we don't need that here — the spy is
    // scoped to this it() and `mockRestore()`d in finally.
    const { logger } = await import('./logger.js');
    const infoSpy = vi.spyOn(logger, 'info');
    const { evaluateGateChain } = await import('./gates/orchestrator.js');
    const { _unregisterGateForTesting, registerGate } =
      await import('./gates/index.js');

    // Register a deterministic test gate that always allows so the
    // chain has exactly one record we can assert against. Cleanup
    // ensures the gate doesn't leak into other tests' registry.
    const TEST_GATE_NAME = 'test-gate-pr445-producer';
    registerGate(TEST_GATE_NAME, () => ({
      decision: 'allow',
      reason: 'always-allow for producer-shape pinning',
    }));

    try {
      const group: RegisteredGroup = {
        ...MAIN_GROUP,
        folder: 'whatsapp_main_producer',
      };
      const message = {
        id: 'producer-msg-1',
        chat_jid: 'tg:-producer',
        sender: 'tg:42',
        sender_name: 'Tester',
        content: 'hello',
        timestamp: '2026-05-02T20:00:00.000Z',
        is_from_me: false,
      };
      await evaluateGateChain(
        group,
        'tg:-producer',
        [message],
        [TEST_GATE_NAME],
      );

      // Filter the spy calls down to the canonical line. Other
      // info-level lines may also fire in this scope; we only assert
      // about the producer contract.
      const matching = infoSpy.mock.calls.filter((call) => {
        return call.length >= 2 && call[1] === 'gate decision';
      });
      expect(matching).toHaveLength(1);
      const fields = matching[0][0] as Record<string, unknown>;
      // Pin every field name the parser depends on. A silent rename
      // (`finalDecision` → `final_decision`, etc.) breaks here.
      expect(fields).toMatchObject({
        chatJid: 'tg:-producer',
        messageId: 'producer-msg-1',
        groupFolder: 'whatsapp_main_producer',
        finalDecision: 'allow',
        reason: 'always-allow for producer-shape pinning',
      });
      expect(Array.isArray(fields.chain)).toBe(true);
      const chain = fields.chain as Array<Record<string, unknown>>;
      expect(chain).toHaveLength(1);
      expect(chain[0]).toMatchObject({
        gate: TEST_GATE_NAME,
        decision: 'allow',
        reason: 'always-allow for producer-shape pinning',
      });
    } finally {
      infoSpy.mockRestore();
      _unregisterGateForTesting(TEST_GATE_NAME);
    }
  });
});
