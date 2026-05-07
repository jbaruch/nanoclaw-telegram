// Stage 2 Haiku classifier tests (#83). Mocks the Anthropic SDK
// client via the `_setAnthropicClientForTesting` seam so no real
// network calls happen.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

let tmpRoot: string;
let groupsDir: string;
let dataDir: string;

vi.mock('../config.js', async () => {
  const actual =
    await vi.importActual<typeof import('../config.js')>('../config.js');
  return {
    ...actual,
    get GROUPS_DIR() {
      return groupsDir;
    },
    get DATA_DIR() {
      return dataDir;
    },
    ASSISTANT_NAME: 'Andy',
  };
});

import Anthropic from '@anthropic-ai/sdk';

import { _initTestDatabase, _writeRawRegisteredGroup } from '../db.js';
import { logger } from '../logger.js';
import {
  haikuClassifierGate,
  _setAnthropicClientForTesting,
} from './haiku-classifier.js';
import type { GateContext } from './index.js';

const TEST_FOLDER = 'telegram_haikutest';
const TEST_JID = 'haikutest@g.us';

function buildCtx(overrides: Partial<GateContext> = {}): GateContext {
  return {
    groupJid: TEST_JID,
    groupFolder: TEST_FOLDER,
    message: {
      text: 'hi',
      messageId: 'msg-test-1',
      senderJid: 's@s.whatsapp.net',
    },
    triggerPatterns: null,
    ...overrides,
  };
}

interface CapturedCall {
  params: Anthropic.MessageCreateParams;
  options?: { signal?: AbortSignal };
}

function buildToolUseResponse(
  intent: 'yes' | 'no',
  reason = 'because',
  confidence = 0.9,
): Anthropic.Message {
  return {
    id: 'msg_123',
    type: 'message',
    role: 'assistant',
    model: 'claude-haiku-4-5-20251001',
    stop_reason: 'tool_use',
    stop_sequence: null,
    content: [
      {
        type: 'tool_use',
        id: 'tu_1',
        name: 'classify_intent',
        input: { intent, confidence, reason },
      },
    ],
    usage: {
      input_tokens: 10,
      output_tokens: 20,
      cache_read_input_tokens: 100,
      cache_creation_input_tokens: 0,
      service_tier: null,
      cache_creation: null,
      server_tool_use: null,
    } as unknown as Anthropic.Usage,
  } as unknown as Anthropic.Message;
}

interface MockClientOptions {
  response?: Anthropic.Message;
  throwError?: Error;
}

function buildMockClient(opts: MockClientOptions = {}): {
  client: Anthropic;
  calls: CapturedCall[];
} {
  const calls: CapturedCall[] = [];
  const create = vi.fn(
    async (
      params: Anthropic.MessageCreateParams,
      options?: { signal?: AbortSignal },
    ): Promise<Anthropic.Message> => {
      calls.push({ params, options });
      if (opts.throwError) throw opts.throwError;
      return opts.response ?? buildToolUseResponse('yes');
    },
  );
  const client = { messages: { create } } as unknown as Anthropic;
  return { client, calls };
}

beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nc-haiku-'));
  groupsDir = path.join(tmpRoot, 'groups');
  dataDir = path.join(tmpRoot, 'data');
  fs.mkdirSync(groupsDir, { recursive: true });
  fs.mkdirSync(dataDir, { recursive: true });
  _initTestDatabase();
  _writeRawRegisteredGroup({
    jid: TEST_JID,
    name: 'Haiku Test Group',
    folder: TEST_FOLDER,
    trigger: '@andy',
    added_at: '2024-01-01T00:00:00Z',
    container_config: JSON.stringify({ stage2Enabled: true }),
  });
  // Seed a CLAUDE.md so the static strategy emits the head section.
  const groupDir = path.join(groupsDir, TEST_FOLDER);
  fs.mkdirSync(groupDir, { recursive: true });
  fs.writeFileSync(
    path.join(groupDir, 'CLAUDE.md'),
    '# Haiku Test\nGroup description for tests.\n',
  );
});

afterEach(() => {
  _setAnthropicClientForTesting(undefined);
  fs.rmSync(tmpRoot, { recursive: true, force: true });
  vi.restoreAllMocks();
});

describe('haikuClassifierGate — verdict mapping', () => {
  it('yes verdict → allow', async () => {
    const { client } = buildMockClient({
      response: buildToolUseResponse('yes', 'directed at assistant'),
    });
    _setAnthropicClientForTesting(client);
    const result = await haikuClassifierGate(buildCtx());
    expect(result.decision).toBe('allow');
    expect(result.reason).toContain('haiku-yes');
    expect(result.reason).toContain('directed at assistant');
  });

  it('no verdict → deny', async () => {
    const { client } = buildMockClient({
      response: buildToolUseResponse('no', 'human-to-human chat'),
    });
    _setAnthropicClientForTesting(client);
    const result = await haikuClassifierGate(buildCtx());
    expect(result.decision).toBe('deny');
    expect(result.reason).toContain('haiku-no');
    expect(result.reason).toContain('human-to-human chat');
  });

  it('emits a usage.jsonl record for every API call (issue #493)', async () => {
    // The classifier runs orchestrator-side and bypasses the credential
    // proxy. Without a direct emit hook its spend would be invisible to
    // anything reading logs/usage.jsonl.
    const usageLogPath = path.join(tmpRoot, 'classifier-usage.jsonl');
    process.env.USAGE_LOG_PATH = usageLogPath;
    try {
      const { _resetUsageLogState } = await import('../usage-log.js');
      _resetUsageLogState();
      const { client } = buildMockClient({
        response: buildToolUseResponse('no', 'peer-to-peer'),
      });
      _setAnthropicClientForTesting(client);
      await haikuClassifierGate(buildCtx());

      // appendUsageRecord is fire-and-forget; poll until the line is
      // visible rather than sleeping a fixed duration (avoids
      // CI-load-induced flakes on slow filesystems).
      const deadline = Date.now() + 2000;
      let content = '';
      while (Date.now() < deadline) {
        if (fs.existsSync(usageLogPath)) {
          content = fs.readFileSync(usageLogPath, 'utf8').trim();
          if (content.length > 0) break;
        }
        await new Promise((r) => setTimeout(r, 10));
      }
      expect(content.length).toBeGreaterThan(0);
      const rec = JSON.parse(content);
      expect(rec).toMatchObject({
        group: TEST_FOLDER,
        tier: 'classifier',
        session: 'gates/haiku-classifier',
        task_id: null,
        message_id: null,
        model: 'claude-haiku-4-5-20251001',
        api_id: 'msg_123',
        in: 10,
        out: 20,
        cache_r: 100,
        cache_c_5m: 0,
        cache_c_1h: 0,
      });
      // 10 in × $1/MTok + 20 out × $5/MTok + 100 cache_r × $0.10/MTok
      // = $0.00001 + $0.0001 + $0.00001 = $0.00012
      // 1 microcent = 10⁻⁸ dollars, so $0.00012 = 12000 microcents
      expect(rec.cost_micro).toBe(12000);
    } finally {
      delete process.env.USAGE_LOG_PATH;
    }
  });
});

describe('haikuClassifierGate — failure modes', () => {
  it('API error → pass with loud ERROR log', async () => {
    const errSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    // Use Anthropic.APIError so the gate's typed-narrow catch
    // recognises this as an expected SDK failure shape (per
    // `coding-policy: error-handling` post-#426 review fix); a
    // generic Error('rate limited') would correctly propagate
    // because it could be a programmer defect instead of an API
    // failure.
    const apiErr = new Anthropic.APIError(
      429,
      { type: 'rate_limit_error', message: 'rate limited' },
      'rate limited',
      new Headers(),
    );
    const { client } = buildMockClient({ throwError: apiErr });
    _setAnthropicClientForTesting(client);
    const result = await haikuClassifierGate(buildCtx());
    expect(result.decision).toBe('pass');
    expect(result.reason).toContain('classifier-failed: api-error');
    const verdictLog = errSpy.mock.calls.find(
      (c) => typeof c[1] === 'string' && c[1] === 'haiku classifier verdict',
    );
    expect(verdictLog).toBeDefined();
  });

  it('unexpected non-API error → propagates (programmer defect, not a fail-open path)', async () => {
    const { client } = buildMockClient({
      throwError: new TypeError('cannot read property of undefined'),
    });
    _setAnthropicClientForTesting(client);
    await expect(haikuClassifierGate(buildCtx())).rejects.toThrow(TypeError);
  });

  it('timeout (abort) → pass with loud ERROR log', async () => {
    const errSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    // Simulate the SDK seeing the abort: throw AbortError synchronously
    // from the mock so the gate's catch detects abort path. (The real
    // SDK rejects with AbortError when the controller fires; we
    // shortcut the 10s timeout for test speed.)
    const abortErr = new Error('Request aborted');
    abortErr.name = 'AbortError';
    const { client } = buildMockClient({ throwError: abortErr });
    _setAnthropicClientForTesting(client);
    const result = await haikuClassifierGate(buildCtx());
    expect(result.decision).toBe('pass');
    expect(result.reason).toContain('classifier-failed: timeout');
    const verdictLog = errSpy.mock.calls.find(
      (c) => typeof c[1] === 'string' && c[1] === 'haiku classifier verdict',
    );
    expect(verdictLog).toBeDefined();
  });

  it('unparseable response (no tool use) → pass with loud ERROR log', async () => {
    const errSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    const malformed: Anthropic.Message = {
      id: 'msg_x',
      type: 'message',
      role: 'assistant',
      model: 'claude-haiku-4-5-20251001',
      stop_reason: 'end_turn',
      stop_sequence: null,
      content: [{ type: 'text', text: 'hello', citations: null }],
      usage: {
        input_tokens: 5,
        output_tokens: 5,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
    } as unknown as Anthropic.Message;
    const { client } = buildMockClient({ response: malformed });
    _setAnthropicClientForTesting(client);
    const result = await haikuClassifierGate(buildCtx());
    expect(result.decision).toBe('pass');
    expect(result.reason).toContain('classifier-failed: unparseable');
    const verdictLog = errSpy.mock.calls.find(
      (c) => typeof c[1] === 'string' && c[1] === 'haiku classifier verdict',
    );
    expect(verdictLog).toBeDefined();
  });

  it('no Anthropic client (no API key) → pass with ERROR log', async () => {
    const errSpy = vi.spyOn(logger, 'error').mockImplementation(() => {});
    _setAnthropicClientForTesting(null);
    const result = await haikuClassifierGate(buildCtx());
    expect(result.decision).toBe('pass');
    expect(result.reason).toContain('classifier-failed: no-client');
    expect(errSpy).toHaveBeenCalled();
  });
});

describe('haikuClassifierGate — prompt assembly', () => {
  it('emits two text system blocks with ephemeral cache_control on both', async () => {
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('yes'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(buildCtx());
    expect(calls).toHaveLength(1);
    const system = calls[0].params.system;
    expect(Array.isArray(system)).toBe(true);
    const blocks = system as Anthropic.TextBlockParam[];
    expect(blocks).toHaveLength(2);
    // Both blocks carry an ephemeral cache_control breakpoint. The
    // prompt happens to be over Haiku 4.5's 4096-token cache floor
    // because it's sized for classifier accuracy (worked examples,
    // multi-language rules, injection defenses); caching is a free
    // happy-accident benefit at this size.
    expect(blocks[0].type).toBe('text');
    expect(blocks[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(blocks[1].type).toBe('text');
    expect(blocks[1].cache_control).toEqual({ type: 'ephemeral' });
  });

  it('keeps cache_control on both blocks even when the volatile suffix is short/empty', async () => {
    // Replace CLAUDE.md with an empty file so the static strategy
    // emits a near-empty suffix. The structural cache_control assertion
    // must hold regardless of suffix content — caching is wired by the
    // call site, not by the suffix length.
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.writeFileSync(path.join(groupDir, 'CLAUDE.md'), '');
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('yes'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(buildCtx());
    const blocks = calls[0].params.system as Anthropic.TextBlockParam[];
    expect(blocks).toHaveLength(2);
    expect(blocks[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(blocks[1].cache_control).toEqual({ type: 'ephemeral' });
  });

  it('frozen prefix retains the worked-example few-shot block (accuracy floor)', async () => {
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('yes'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(buildCtx());
    const blocks = calls[0].params.system as Anthropic.TextBlockParam[];
    const prefix = blocks[0].text;
    // The few-shot examples are the load-bearing part of the prompt
    // for classifier accuracy. This test catches accidental trimming
    // that would silently degrade the labels #82's keyword-learning
    // loop will eventually train on. Counts: 9 yes (Y1-Y9, where Y9
    // is the owner-privileged case), 10 no (N1-N10), 3 tricky (T1-T3)
    // = 22 total. Y9's heading is `Example Y9 (owner privileged).`
    // — the trailing parenthetical means we match the digit boundary
    // with `\d+` (no escaped period) instead of `\d+\.`.
    const yesCount = (prefix.match(/Example Y\d+/g) ?? []).length;
    const noCount = (prefix.match(/Example N\d+\./g) ?? []).length;
    const trickyCount = (prefix.match(/Example T\d+\./g) ?? []).length;
    expect(yesCount).toBe(9);
    expect(noCount).toBe(10);
    expect(trickyCount).toBe(3);
  });

  it('frozen prefix carries the owner-privileged edge-case rule and Y9 example', async () => {
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('yes'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(buildCtx());
    const blocks = calls[0].params.system as Anthropic.TextBlockParam[];
    const prefix = blocks[0].text;
    // Owner-aware classification rests on these two substrings: the
    // edge-case rule that tells Haiku how to react to an `Owner:` line,
    // and the worked example that anchors the rule. Pin presence (not
    // exact text) so spec-compatible rewording stays test-safe.
    expect(prefix).toContain("Sender is the assistant's owner");
    expect(prefix).toContain('Example Y9 (owner privileged)');
  });

  it('volatile suffix uses the configured strategy output', async () => {
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('yes'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(buildCtx());
    const system = calls[0].params.system as Anthropic.TextBlockParam[];
    const suffix = system[1].text;
    // Per static-group-context strategy: group display name + assistant
    // identity + CLAUDE.md head.
    expect(suffix).toContain('Haiku Test Group');
    expect(suffix).toContain('Andy');
    expect(suffix).toContain('# Haiku Test');
    expect(suffix).toContain('Group description for tests.');
  });

  it('forces tool use with classify_intent', async () => {
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('yes'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(buildCtx());
    expect(calls[0].params.tool_choice).toEqual({
      type: 'tool',
      name: 'classify_intent',
    });
    expect(calls[0].params.tools).toHaveLength(1);
    expect((calls[0].params.tools as Anthropic.Tool[])[0].name).toBe(
      'classify_intent',
    );
  });

  it('uses overridden model from containerConfig.stage2ModelId', async () => {
    _writeRawRegisteredGroup({
      jid: TEST_JID,
      name: 'Haiku Test Group',
      folder: TEST_FOLDER,
      trigger: '@andy',
      added_at: '2024-01-01T00:00:00Z',
      container_config: JSON.stringify({
        stage2Enabled: true,
        stage2ModelId: 'claude-haiku-override-id',
      }),
    });
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('yes'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(buildCtx());
    expect(calls[0].params.model).toBe('claude-haiku-override-id');
  });
});

describe('haikuClassifierGate — structured reply rendering (#107)', () => {
  it('emits Sender + Message lines only when replyTo is absent', async () => {
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('yes'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(
      buildCtx({
        message: {
          text: 'hello there',
          messageId: 'msg-test-1',
          senderJid: 's@s.whatsapp.net',
        },
      }),
    );
    const userText = calls[0].params.messages[0].content as string;
    // No reply-context line — only Sender / Message.
    expect(userText).not.toContain('[Replying to');
    expect(userText).toContain('Sender:');
    expect(userText).toContain('Message: hello there');
  });

  it('emits a structured reply line with the assistant marker when replyTo.isAssistant=true', async () => {
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('yes'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(
      buildCtx({
        message: {
          text: 'got it',
          messageId: 'msg-test-1',
          senderJid: 's@s.whatsapp.net',
          replyTo: {
            messageId: 'msg-1',
            senderName: 'Andy',
            isBot: true,
            isAssistant: true,
            contentPreview: 'I deployed the fix in PR #22',
          },
        },
      }),
    );
    const userText = calls[0].params.messages[0].content as string;
    expect(userText).toContain('[Replying to Andy');
    expect(userText).toContain('the assistant');
    expect(userText).toContain('I deployed the fix in PR #22');
    // Body remains clean (no inline prefix bleeding back in).
    expect(userText).toContain('Message: got it');
  });

  it('emits the peer-bot marker when replyTo.isBot=true and isAssistant=false', async () => {
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('no'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(
      buildCtx({
        message: {
          text: 'Yes do it',
          messageId: 'msg-test-1',
          senderJid: 's@s.whatsapp.net',
          replyTo: {
            messageId: '4114',
            senderName: 'PeerBot',
            isBot: true,
            isAssistant: false,
            contentPreview: 'Stage 2 Haiku output',
          },
        },
      }),
    );
    const userText = calls[0].params.messages[0].content as string;
    expect(userText).toContain('[Replying to PeerBot');
    expect(userText).toContain('a peer bot');
    expect(userText).not.toContain('the assistant');
    expect(userText).toContain('Message: Yes do it');
  });

  it('emits no marker for replies to humans', async () => {
    const { client, calls } = buildMockClient({
      response: buildToolUseResponse('no'),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(
      buildCtx({
        message: {
          text: 'see you',
          messageId: 'msg-test-1',
          senderJid: 's@s.whatsapp.net',
          replyTo: {
            messageId: 'msg-h',
            senderName: 'Bob',
            isBot: false,
            isAssistant: false,
            contentPreview: 'meeting tomorrow at 3',
          },
        },
      }),
    );
    const userText = calls[0].params.messages[0].content as string;
    expect(userText).toContain('[Replying to Bob:');
    expect(userText).not.toContain('peer bot');
    expect(userText).not.toContain('the assistant');
  });
});

describe('haikuClassifierGate — observability', () => {
  it('emits one INFO line with verdict, confidence, and usage tokens', async () => {
    const infoSpy = vi.spyOn(logger, 'info').mockImplementation(() => {});
    const { client } = buildMockClient({
      response: buildToolUseResponse('yes', 'plausible request', 0.81),
    });
    _setAnthropicClientForTesting(client);
    await haikuClassifierGate(buildCtx());
    const verdictLines = infoSpy.mock.calls.filter(
      (c) => typeof c[1] === 'string' && c[1] === 'haiku classifier verdict',
    );
    expect(verdictLines).toHaveLength(1);
    const fields = verdictLines[0][0] as Record<string, unknown>;
    expect(fields.intent).toBe('yes');
    expect(fields.confidence).toBe(0.81);
    expect(fields.modelId).toBe('claude-haiku-4-5-20251001');
    expect(fields.inputTokens).toBe(10);
    expect(fields.outputTokens).toBe(20);
    // Cache token fields restored after PR #102 re-enabled cache_control
    // on the system blocks. Operators need these to compute exact per-call
    // cost; the bare inputTokens field shows only the uncached portion.
    expect(fields.cacheReadTokens).toBe(100);
    expect(fields.cacheCreateTokens).toBe(0);
    expect(typeof fields.durationMs).toBe('number');
    expect(fields.groupFolder).toBe(TEST_FOLDER);
    // #451 item 4: messageId + inboundText close the join
    // mineHaikuSamples used to skip — Haiku verdicts can now be
    // attributed to the inbound directly without stitching through
    // a paired `gate decision` record.
    expect(fields.messageId).toBe('msg-test-1');
    expect(fields.inboundText).toBe('hi');
  });
});
