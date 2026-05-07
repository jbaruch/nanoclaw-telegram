/**
 * Tests for the wiring layer in `trigger-learner-runtime.ts` (#415).
 *
 * Two regression-shaped invariants this file pins down:
 *
 *  1. **Owner-by-JID classification.** The reaction miner must pass
 *     `reactor_jid` (not `reactor_name`) as the JID argument to
 *     `classifySender`, so a reaction whose JID matches the configured
 *     owner handle gets the owner weight (3.0). Previously the miner
 *     passed `reactor_name` for both args, silently degrading owner
 *     reactions to non-owner.
 *
 *  2. **Configured rollback window is honored end-to-end.** The
 *     `TRIGGER_LEARNER_ROLLBACK_WINDOW` env (surfaced as
 *     `cfg.rollbackWindow`) must reach `fetchDecisionHistories`.
 *     Previously the persistence layer hardcoded the default and
 *     larger configured windows had no effect.
 *
 * All tests are deterministic — fixed inputs, no randomness, no
 * shared mutable state across tests.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mocks must be declared BEFORE the imports that pull them in. We
// stub the live DB, the host-log file resolver, and the host-log
// parser — the unit under test should never touch real I/O.
vi.mock('../db.js', () => ({
  getAllRegisteredGroups: vi.fn(),
  getReactionsForMessage: vi.fn(),
  getMessagesSince: vi.fn(),
  getTriggerPatterns: vi.fn(),
  setTriggerPatterns: vi.fn(),
}));

vi.mock('../host-logs.js', () => ({
  hostLogsOrchestratorFile: vi.fn(() => '/tmp/test-host.log'),
}));

vi.mock('../host-log-parser.js', () => ({
  readHostLog: vi.fn(() => []),
  findGateDecisions: vi.fn(() => []),
}));

vi.mock('../logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

vi.mock('../config.js', () => ({
  ASSISTANT_OWNER_HANDLE: undefined,
}));

import {
  getReactionsForMessage,
  getMessagesSince,
  getAllRegisteredGroups,
} from '../db.js';
import { findGateDecisions } from '../host-log-parser.js';
import {
  buildLearnerPersistence,
  classifySender,
  mineHaikuSamples,
  mineSamplesForGroup,
  DEFAULT_LEARNER_ROLLBACK_WINDOW,
} from './trigger-learner-runtime.js';
import { readHostLog } from '../host-log-parser.js';

const GROUP_JID = '120363000000000001@g.us';
const OWNER_HANDLE = 'owner-handle';
const OWNER_JID = `1234567890@s.whatsapp.net:${OWNER_HANDLE}`;

beforeEach(() => {
  vi.mocked(getReactionsForMessage).mockReset();
  vi.mocked(getMessagesSince).mockReset();
  vi.mocked(getAllRegisteredGroups).mockReset();
  vi.mocked(findGateDecisions).mockReset();
  vi.mocked(findGateDecisions).mockReturnValue([]);
});

// ---------------------------------------------------------------------------
// 1. Owner-by-JID classification (the reactor_jid wiring fix)
// ---------------------------------------------------------------------------

describe('classifySender — JID-based owner detection', () => {
  it('returns "owner" when reactorJid contains the owner handle even if reactorName does not match', () => {
    // Owner handle appears only in the JID; reactorName is a freeform
    // display nickname that doesn't match. Without the JID-arg fix
    // this would degrade to "non-owner".
    const tier = classifySender(OWNER_JID, 'Some Nickname', OWNER_HANDLE);
    expect(tier).toBe('owner');
  });

  it('returns "non-owner" when neither JID nor name match owner handle', () => {
    const tier = classifySender(
      'someone-else@s.whatsapp.net',
      'Stranger',
      OWNER_HANDLE,
    );
    expect(tier).toBe('non-owner');
  });

  it('returns "anonymous" when both reactorJid and reactorName are empty', () => {
    const tier = classifySender('', '', OWNER_HANDLE);
    expect(tier).toBe('anonymous');
  });
});

describe('mineSamplesForGroup — owner weight via reactor_jid', () => {
  it('classifies an owner-JID reaction as senderTier="owner" so the owner weight (3.0) applies downstream', () => {
    // Single inbound message with a positive reaction whose
    // reactor_jid matches the owner handle but whose reactor_name is
    // a generic nickname that does NOT match. Pre-fix this
    // misclassified as "non-owner"; post-fix it correctly classifies
    // as "owner".
    vi.mocked(getMessagesSince).mockReturnValue([
      {
        id: 'msg-1',
        chat_jid: GROUP_JID,
        sender: 'sender-1@s.whatsapp.net',
        sender_name: 'Asker',
        content: 'deploy hotfix now',
        timestamp: '2026-05-02T10:00:00Z',
        is_from_me: 0,
        reply_to_message_id: null,
        reply_to_message_content: null,
        reply_to_sender_name: null,
      },
      // Cast to NewMessage row shape — we only care about id+content
      // here; the rest are minimal valid fields.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ] as any);
    vi.mocked(getReactionsForMessage).mockReturnValue([
      {
        reactor_jid: OWNER_JID,
        reactor_name: 'Generic Nickname',
        emoji: '👍',
        timestamp: '2026-05-02T10:01:00Z',
      },
    ]);

    const samples = mineSamplesForGroup(
      GROUP_JID,
      '/tmp/test-host.log',
      OWNER_HANDLE,
    );
    expect(samples).toHaveLength(1);
    expect(samples[0].senderTier).toBe('owner');
    expect(samples[0].intent).toBe('yes');
    expect(samples[0].text).toBe('deploy hotfix now');
  });

  it('classifies a non-owner-JID reaction as senderTier="non-owner" (control case)', () => {
    vi.mocked(getMessagesSince).mockReturnValue([
      {
        id: 'msg-2',
        chat_jid: GROUP_JID,
        sender: 'sender-2@s.whatsapp.net',
        sender_name: 'Asker',
        content: 'ping help me with this',
        timestamp: '2026-05-02T10:00:00Z',
        is_from_me: 0,
        reply_to_message_id: null,
        reply_to_message_content: null,
        reply_to_sender_name: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ] as any);
    vi.mocked(getReactionsForMessage).mockReturnValue([
      {
        reactor_jid: 'somebody-else@s.whatsapp.net',
        reactor_name: 'Random User',
        emoji: '👍',
        timestamp: '2026-05-02T10:01:00Z',
      },
    ]);

    const samples = mineSamplesForGroup(
      GROUP_JID,
      '/tmp/test-host.log',
      OWNER_HANDLE,
    );
    expect(samples).toHaveLength(1);
    expect(samples[0].senderTier).toBe('non-owner');
  });
});

// ---------------------------------------------------------------------------
// 2. Rollback window threading (TRIGGER_LEARNER_ROLLBACK_WINDOW honored)
// ---------------------------------------------------------------------------

describe('buildLearnerPersistence — rollbackWindow threading', () => {
  // The contract: the integer passed to `buildLearnerPersistence`
  // must be used as the trailing decision-history window when
  // `fetchDecisionHistories` runs. Internally `buildDecisionHistories`
  // calls `findGateDecisions({ limit: rollbackWindow * 10 })`, so we
  // can prove the wiring by inspecting the limit.

  beforeEach(() => {
    vi.mocked(getAllRegisteredGroups).mockReturnValue({});
    vi.mocked(getMessagesSince).mockReturnValue([]);
    vi.mocked(getReactionsForMessage).mockReturnValue([]);
  });

  it('forwards a custom rollbackWindow=15 to findGateDecisions (limit = 15 * 10)', () => {
    const persistence = buildLearnerPersistence(15);
    persistence.fetchDecisionHistories(GROUP_JID);
    expect(findGateDecisions).toHaveBeenCalled();
    const callArgs = vi.mocked(findGateDecisions).mock.calls[0];
    expect(callArgs[1].limit).toBe(15 * 10);
  });

  it('forwards a larger rollbackWindow=100 to findGateDecisions (limit = 100 * 10)', () => {
    // Pre-fix bug: the persistence layer hardcoded
    // DEFAULT_LEARNER_ROLLBACK_WINDOW (30) regardless of caller, so
    // a configured window of 100 would still trim at 30. This test
    // proves the fix: changing the input window changes the
    // effective limit downstream.
    const persistence = buildLearnerPersistence(100);
    persistence.fetchDecisionHistories(GROUP_JID);
    const callArgs = vi.mocked(findGateDecisions).mock.calls[0];
    expect(callArgs[1].limit).toBe(100 * 10);
  });

  it('falls back to DEFAULT_LEARNER_ROLLBACK_WINDOW when no argument is provided', () => {
    const persistence = buildLearnerPersistence();
    persistence.fetchDecisionHistories(GROUP_JID);
    const callArgs = vi.mocked(findGateDecisions).mock.calls[0];
    expect(callArgs[1].limit).toBe(DEFAULT_LEARNER_ROLLBACK_WINDOW * 10);
  });
});

// ---------------------------------------------------------------------------
// 3. mineHaikuSamples — #451 item 4 activated the join via messageId+inboundText
// ---------------------------------------------------------------------------

describe('mineHaikuSamples — post-#451-item-4 activated path', () => {
  const GROUP_FOLDER = 'telegram_test';
  const HOST_LOG_PATH = '/tmp/test-host.log';

  beforeEach(() => {
    vi.mocked(readHostLog).mockReset();
  });

  it('returns one sample per high-confidence verdict that carries inboundText', () => {
    vi.mocked(readHostLog).mockReturnValue([
      {
        timestamp: '12:00:00.000',
        level: 'INFO',
        pid: 1,
        msg: 'haiku classifier verdict',
        fields: {
          groupFolder: GROUP_FOLDER,
          messageId: 'msg-1',
          inboundText: 'can you help with X?',
          intent: 'yes',
          confidence: 0.9,
        },
      },
      {
        timestamp: '12:00:01.000',
        level: 'INFO',
        pid: 1,
        msg: 'haiku classifier verdict',
        fields: {
          groupFolder: GROUP_FOLDER,
          messageId: 'msg-2',
          inboundText: 'lol',
          intent: 'no',
          confidence: 0.85,
        },
      },
    ]);
    const samples = mineHaikuSamples(GROUP_FOLDER, HOST_LOG_PATH);
    expect(samples).toHaveLength(2);
    expect(samples[0]).toMatchObject({
      text: 'can you help with X?',
      intent: 'yes',
      source: 'haiku_verdict',
      senderTier: 'anonymous',
      gateResponded: false,
    });
    expect(samples[1].intent).toBe('no');
    expect(samples[1].text).toBe('lol');
  });

  it('skips low-confidence verdicts (ambiguous = noise, not truth)', () => {
    vi.mocked(readHostLog).mockReturnValue([
      {
        timestamp: '12:00:00.000',
        level: 'INFO',
        pid: 1,
        msg: 'haiku classifier verdict',
        fields: {
          groupFolder: GROUP_FOLDER,
          messageId: 'msg-1',
          inboundText: 'maybe?',
          intent: 'yes',
          confidence: 0.55,
        },
      },
    ]);
    const samples = mineHaikuSamples(GROUP_FOLDER, HOST_LOG_PATH);
    expect(samples).toHaveLength(0);
  });

  it('skips verdicts from other groups', () => {
    vi.mocked(readHostLog).mockReturnValue([
      {
        timestamp: '12:00:00.000',
        level: 'INFO',
        pid: 1,
        msg: 'haiku classifier verdict',
        fields: {
          groupFolder: 'telegram_other',
          messageId: 'msg-1',
          inboundText: 'help',
          intent: 'yes',
          confidence: 0.95,
        },
      },
    ]);
    const samples = mineHaikuSamples(GROUP_FOLDER, HOST_LOG_PATH);
    expect(samples).toHaveLength(0);
  });

  it('skips legacy records that pre-date #451 item 4 (no inboundText field)', () => {
    // Pre-#451-item-4 verdict log line had no inboundText. Reading
    // these post-merge yields zero usable samples — the learner
    // tolerates the empty result and falls back to reaction mining.
    vi.mocked(readHostLog).mockReturnValue([
      {
        timestamp: '12:00:00.000',
        level: 'INFO',
        pid: 1,
        msg: 'haiku classifier verdict',
        fields: {
          groupFolder: GROUP_FOLDER,
          intent: 'yes',
          confidence: 0.9,
          // no messageId, no inboundText
        },
      },
    ]);
    const samples = mineHaikuSamples(GROUP_FOLDER, HOST_LOG_PATH);
    expect(samples).toHaveLength(0);
  });

  it('honors a custom minConfidence', () => {
    vi.mocked(readHostLog).mockReturnValue([
      {
        timestamp: '12:00:00.000',
        level: 'INFO',
        pid: 1,
        msg: 'haiku classifier verdict',
        fields: {
          groupFolder: GROUP_FOLDER,
          messageId: 'msg-1',
          inboundText: 'borderline',
          intent: 'yes',
          confidence: 0.65,
        },
      },
    ]);
    expect(mineHaikuSamples(GROUP_FOLDER, HOST_LOG_PATH, 0.7)).toHaveLength(0);
    expect(mineHaikuSamples(GROUP_FOLDER, HOST_LOG_PATH, 0.6)).toHaveLength(1);
  });
});
