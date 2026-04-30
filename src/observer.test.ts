/**
 * Focused tests for `src/observer.ts`.
 *
 * Coverage targets the parser/state-machine surface that's most
 * likely to drift on agent-runner log-format changes (the wire
 * format that links the two halves of the observer pipeline). The
 * actual Telegram send paths are exercised through a stub channel
 * so we can assert on outgoing summary text without standing up the
 * grammy client.
 *
 * Specifically:
 *   - `onAgentLine` parses Query input boundaries (sets state, picks
 *     up `target_message_id`, picks up `scheduled_task`, picks up
 *     `addressed`), per-block thinking / tool_use / tool_result
 *     events, and the final `Query done.` line including its metric
 *     tail.
 *   - `committed` flips on the first non-thinking event so silent
 *     thinking-only turns leave the user's message untouched.
 *   - Watchdog arms for non-scheduled tasks but stays unarmed for
 *     scheduled tasks (cron / heartbeat queries) AND for explicitly
 *     non-addressed queries (#289 — `requires_trigger=false`
 *     bystander chatter that still routes to the agent).
 *   - `chunkText` handles under-size, exact-size, and over-size with
 *     whitespace fallback.
 *   - `addressed=false` on the Query input line short-circuits the
 *     entire reaction ladder; `addressed=true` fires it normally;
 *     `addressed=` omitted falls through to the engagement gate
 *     alone for backward compat.
 *   - `Query done.` cleanup drops `lastReactionEmoji` entries for
 *     the closing query — guards against the unbounded-growth bug
 *     Copilot flagged in PR #240.
 *
 * Module env (`OBSERVER_CHAT_JID`) is read at module load. The
 * test seam `__enableObserverForTests` sets a per-test JID override
 * directly so each test operates on a known JID without relying on
 * the actual env. The stub channel's `sendMessage` captures the
 * dispatched text for assertions.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

const OBSERVER_TEST_JID = 'tg:-100999000111';

// __enableObserverForTests sets the JID override directly, so the
// module-level OBSERVER_CHAT_JID const doesn't need to be seeded
// from the test environment.
import {
  __enableObserverForTests,
  __getObserverInternalsForTests,
  __resetObserverForTests,
  chunkText,
  observerEnabled,
  onAgentLine,
} from './observer.js';
import type { Channel, RegisteredGroup } from './types.js';

interface CapturedMessage {
  jid: string;
  body: string;
  replyTo?: string;
}

interface CapturedReaction {
  jid: string;
  messageId: string;
  emoji: string;
}

function makeStubChannel(): Channel & {
  _sent: CapturedMessage[];
  _reactions: CapturedReaction[];
} {
  const sent: CapturedMessage[] = [];
  const reactions: CapturedReaction[] = [];
  return {
    _sent: sent,
    _reactions: reactions,
    name: 'stub',
    isConnected: () => true,
    ownsJid: (jid: string) =>
      jid === OBSERVER_TEST_JID || jid.startsWith('tg:'),
    disconnect: async () => {},
    sendMessage: async (jid: string, body: string, replyTo?: string) => {
      sent.push({ jid, body, replyTo });
    },
    sendReaction: async (jid: string, messageId: string, emoji: string) => {
      reactions.push({ jid, messageId, emoji });
    },
    isPrivateChat: async () => true,
  } as unknown as Channel & {
    _sent: CapturedMessage[];
    _reactions: CapturedReaction[];
  };
}

function makeGroups(
  jid: string,
  folder: string,
  isMain = true,
): Record<string, RegisteredGroup> {
  return {
    [jid]: {
      folder,
      isMain,
      // Other RegisteredGroup fields aren't read by the code under
      // test; cast through unknown to keep the fixture minimal.
    } as unknown as RegisteredGroup,
  };
}

describe('observer', () => {
  let channel: Channel & {
    _sent: CapturedMessage[];
    _reactions: CapturedReaction[];
  };
  let groups: Record<string, RegisteredGroup>;

  beforeEach(() => {
    __resetObserverForTests();
    channel = makeStubChannel();
    groups = makeGroups('tg:-100123', 'main');
    __enableObserverForTests([channel], () => groups);
  });

  afterEach(() => {
    __resetObserverForTests();
  });

  describe('observerEnabled gate', () => {
    it('returns true after __enableObserverForTests', () => {
      expect(observerEnabled()).toBe(true);
    });

    it('returns false after reset', () => {
      __resetObserverForTests();
      expect(observerEnabled()).toBe(false);
    });

    it('onAgentLine no-ops when observer is disabled', () => {
      __resetObserverForTests();
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=42, scheduled_task=false',
      );
      const internals = __getObserverInternalsForTests();
      expect(internals.states.size).toBe(0);
    });
  });

  describe('Query input parsing', () => {
    it('starts a fresh state and parses target_message_id', () => {
      onAgentLine(
        'main',
        '[agent-runner] Query input: 250 chars, target_message_id=msg_42, scheduled_task=false',
      );
      const internals = __getObserverInternalsForTests();
      const state = internals.states.get('main');
      expect(state).toBeDefined();
      expect(state?.targetMessageId).toBe('msg_42');
      expect(state?.committed).toBe(false);
    });

    it('treats target_message_id=- as no target', () => {
      onAgentLine(
        'main',
        'Query input: 50 chars, target_message_id=-, scheduled_task=false',
      );
      const state = __getObserverInternalsForTests().states.get('main');
      expect(state?.targetMessageId).toBeUndefined();
    });

    it('arms a watchdog for non-scheduled tasks', () => {
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=msg_1, scheduled_task=false',
      );
      expect(__getObserverInternalsForTests().watchdogs.has('main')).toBe(true);
    });

    it('does NOT arm a watchdog for scheduled tasks', () => {
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=-, scheduled_task=true',
      );
      expect(__getObserverInternalsForTests().watchdogs.has('main')).toBe(
        false,
      );
    });

    it('strips the [agent-runner] prefix before matching', () => {
      onAgentLine(
        'main',
        '[agent-runner] Query input: 100 chars, target_message_id=msg_99, scheduled_task=false',
      );
      const state = __getObserverInternalsForTests().states.get('main');
      expect(state?.targetMessageId).toBe('msg_99');
    });
  });

  describe('commitment gate', () => {
    beforeEach(() => {
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=msg_1, scheduled_task=false',
      );
    });

    it('thinking alone does NOT commit the turn', () => {
      onAgentLine('main', '[msg #1] thinking="reasoning about the request"');
      onAgentLine('main', '[msg #1] thinking="checking files now"');
      const state = __getObserverInternalsForTests().states.get('main');
      expect(state?.committed).toBe(false);
      expect(state?.thinkingCount).toBe(2);
    });

    it('tool_use commits the turn', () => {
      onAgentLine('main', '[msg #1] thinking="thinking first"');
      onAgentLine('main', '[msg #1] tool_use=Bash id=abc input={}');
      const state = __getObserverInternalsForTests().states.get('main');
      expect(state?.committed).toBe(true);
      expect(state?.toolCalls).toEqual(['Bash']);
    });

    it('non-internal text commits the turn', () => {
      onAgentLine('main', '[msg #1] text="here is my reply"');
      const state = __getObserverInternalsForTests().states.get('main');
      expect(state?.committed).toBe(true);
    });

    it('text wrapped fully in <internal> does NOT commit', () => {
      onAgentLine(
        'main',
        '[msg #1] text="<internal>silent thinking</internal>"',
      );
      const state = __getObserverInternalsForTests().states.get('main');
      expect(state?.committed).toBe(false);
    });

    it('strips mcp__server__ prefix from tool names', () => {
      onAgentLine(
        'main',
        '[msg #1] tool_use=mcp__nanoclaw__send_message id=xyz input={}',
      );
      const state = __getObserverInternalsForTests().states.get('main');
      expect(state?.toolCalls).toEqual(['send_message']);
    });
  });

  describe('Query done flush', () => {
    beforeEach(() => {
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=msg_1, scheduled_task=false',
      );
      onAgentLine('main', '[msg #1] thinking="planning"');
      onAgentLine('main', '[msg #1] tool_use=Bash id=t1');
      onAgentLine('main', '[msg #1] tool_result id=t1 ok');
    });

    it('parses wall_ms / tokens / cache_hit_rate from the done line', async () => {
      onAgentLine(
        'main',
        'Query done. Messages: 3, results: 1, lastAssistantUuid: abc, closedDuringQuery: false, wall_ms=8400, tokens_in=2500, tokens_out=120, cache_hit_rate=91.0',
      );
      // send() dispatches via a Promise chain; flush microtasks before
      // asserting on the captured outbound list.
      await new Promise((r) => setImmediate(r));
      const sent = channel._sent;
      const summary = sent.find((m) => m.body.startsWith('📊'));
      expect(summary).toBeDefined();
      expect(summary?.body).toContain('⏱ 8.4s');
      expect(summary?.body).toContain('in=2500');
      expect(summary?.body).toContain('out=120');
      expect(summary?.body).toContain('cache=91.0%');
    });

    it('renders cache=n/a% when agent-runner reported n/a', async () => {
      onAgentLine(
        'main',
        'Query done. Messages: 0, results: 1, lastAssistantUuid: none, closedDuringQuery: false, wall_ms=12, tokens_in=0, tokens_out=0, cache_hit_rate=n/a',
      );
      await new Promise((r) => setImmediate(r));
      const summary = channel._sent.find((m) => m.body.startsWith('📊'));
      expect(summary?.body).toContain('cache=n/a%');
    });

    it('clears the per-source state on done', () => {
      onAgentLine(
        'main',
        'Query done. Messages: 3, results: 1, lastAssistantUuid: abc, closedDuringQuery: false, wall_ms=8400, tokens_in=10, tokens_out=5, cache_hit_rate=50.0',
      );
      expect(__getObserverInternalsForTests().states.has('main')).toBe(false);
      expect(__getObserverInternalsForTests().watchdogs.has('main')).toBe(
        false,
      );
    });
  });

  describe('error tool_result alert', () => {
    it('sends a live ❌ alert on tool_result error', async () => {
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=msg_1, scheduled_task=false',
      );
      onAgentLine(
        'main',
        '[msg #1] tool_result id=t1 error preview="429 too many requests"',
      );
      await new Promise((r) => setImmediate(r));
      const errLine = channel._sent.find((m) => m.body.startsWith('❌'));
      expect(errLine).toBeDefined();
      expect(errLine?.body).toContain('error');
    });
  });

  describe('lastReactionEmoji cleanup on Query done', () => {
    it('drops the composite key for the query target on done', () => {
      // Set up: a query for msg_5 that fires a tool_use → observer
      // writes the composite dedupe entry under `${chatJid}:msg_5`.
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=msg_5, scheduled_task=false, addressed=true',
      );
      onAgentLine('main', '[msg #1] tool_use=Bash id=t1');
      // Tool_use fired ⚡ via updateReaction; the dedupe map now
      // holds `tg:-100123:msg_5 → ⚡`.
      expect(
        __getObserverInternalsForTests().lastReactionEmoji.get(
          'tg:-100123:msg_5',
        ),
      ).toBe('⚡');
      onAgentLine(
        'main',
        'Query done. Messages: 1, results: 1, lastAssistantUuid: abc, closedDuringQuery: false, wall_ms=10, tokens_in=5, tokens_out=1, cache_hit_rate=10.0',
      );
      expect(
        __getObserverInternalsForTests().lastReactionEmoji.has(
          'tg:-100123:msg_5',
        ),
      ).toBe(false);
    });
  });

  describe('addressed-ness gate (#289)', () => {
    it('suppresses progress reactions when Query input declares addressed=false', () => {
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=msg_99, scheduled_task=false, addressed=false',
      );
      onAgentLine('main', '[msg #1] tool_use=Bash id=t1');
      onAgentLine(
        'main',
        'Query done. Messages: 1, results: 1, lastAssistantUuid: abc, closedDuringQuery: false, wall_ms=10, tokens_in=5, tokens_out=1, cache_hit_rate=10.0',
      );
      // The user-visible outcome is what matters: zero reactions
      // dispatched on this query, including any watchdog blinks
      // and the final 🤝.
      expect(channel._reactions).toEqual([]);
    });

    it('fires progress reactions when addressed=true', () => {
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=msg_88, scheduled_task=false, addressed=true',
      );
      onAgentLine('main', '[msg #1] tool_use=Bash id=t1');
      // Tool_use commits engagement and fires ⚡ via sendReaction.
      expect(channel._reactions).toEqual([
        { jid: 'tg:-100123', messageId: 'msg_88', emoji: '⚡' },
      ]);
    });

    it('falls through to engagement gate when Query input omits addressed= (legacy)', () => {
      onAgentLine(
        'main',
        'Query input: 100 chars, target_message_id=msg_77, scheduled_task=false',
      );
      onAgentLine('main', '[msg #1] tool_use=Bash id=t1');
      // Backward compat: lines without `addressed=` behave as
      // before — committed → reaction fires. Useful for legacy
      // log lines and any caller that has no addressed-ness signal
      // to emit (the gate prefers a false-positive over silently
      // dropping reactions on legacy paths).
      expect(channel._reactions).toEqual([
        { jid: 'tg:-100123', messageId: 'msg_77', emoji: '⚡' },
      ]);
    });
  });

  describe('unknown lines are no-ops', () => {
    it('ignores agent-runner lines that match no known pattern', () => {
      onAgentLine(
        'main',
        'Query input: 50 chars, target_message_id=msg_1, scheduled_task=false',
      );
      onAgentLine('main', 'random debug log line that the agent emitted');
      const state = __getObserverInternalsForTests().states.get('main');
      expect(state?.thinkingCount).toBe(0);
      expect(state?.toolCalls).toEqual([]);
    });
  });
});

describe('chunkText', () => {
  it('returns the whole text in a single chunk when under size', () => {
    expect(chunkText('hello world', 100)).toEqual(['hello world']);
  });

  it('returns the whole text in a single chunk when exactly at size', () => {
    const t = 'a'.repeat(100);
    expect(chunkText(t, 100)).toEqual([t]);
  });

  it('breaks at the nearest whitespace within the last 200 chars', () => {
    // 300 chars of "a", then a space, then 100 chars of "b". With size
    // 350, the slice would land at index 350 (mid-"b" run); chunkText
    // should rewind to the space at index 300.
    const head = 'a'.repeat(300);
    const tail = 'b'.repeat(100);
    const text = `${head} ${tail}`;
    const chunks = chunkText(text, 350);
    expect(chunks.length).toBeGreaterThan(1);
    // First chunk should be the head, no "b"s pulled in.
    expect(chunks[0]).toBe(head);
    expect(chunks[1]).toBe(tail);
  });

  it('hard-cuts when no whitespace is reachable within slack', () => {
    const text = 'x'.repeat(1000);
    const chunks = chunkText(text, 400);
    // With no whitespace, every chunk fills exactly to size.
    expect(chunks[0].length).toBe(400);
    // Reassembly preserves the full content.
    expect(chunks.join('')).toBe(text);
  });
});
