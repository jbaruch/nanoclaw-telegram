import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { promises as fsp } from 'fs';
import { join } from 'path';
import os from 'os';

vi.mock('./logger.js', () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
}));

import {
  PRICING,
  resolvePricing,
  computeCostMicro,
  parseUsageFromBody,
  appendUsageRecord,
  checkSilentZero,
  noteCaptureWrite,
  noteMessagesRequest,
  _resetUsageLogState,
  type ContainerContext,
} from './usage-log.js';

const CTX: ContainerContext = {
  group: 'telegram_main',
  tier: 'main',
  session: 'default',
  task_id: null,
};

describe('resolvePricing', () => {
  it('returns exact match (no fallback flags)', () => {
    const result = resolvePricing('claude-sonnet-4-6');
    expect(result.pricing).toBe(PRICING['claude-sonnet-4-6']);
    expect(result.approximate).toBe(false);
    expect(result.cost_unknown).toBe(false);
  });

  it('strips date suffix when needed', () => {
    const result = resolvePricing('claude-sonnet-4-6-20251015');
    expect(result.pricing).toBe(PRICING['claude-sonnet-4-6']);
    expect(result.approximate).toBe(false);
    expect(result.cost_unknown).toBe(false);
  });

  it('falls back within-family for next-version Opus (#479 sub-#2)', () => {
    // The bug this fix prevents: a future `claude-opus-4-8` would
    // previously be priced as Sonnet (~5× too low for Opus traffic).
    // After the fix it falls back to the latest known Opus entry.
    const result = resolvePricing('claude-opus-4-8');
    expect(result.pricing).toBe(PRICING['claude-opus-4-7']);
    expect(result.approximate).toBe(true);
    expect(result.cost_unknown).toBe(false);
  });

  it('falls back within-family for next-version Sonnet', () => {
    const result = resolvePricing('claude-sonnet-5-0');
    // Either of the two Sonnet entries is acceptable — both have
    // identical pricing. Assert it's a Sonnet entry, approximate.
    const sonnetVariants = [
      PRICING['claude-sonnet-4-6'],
      PRICING['claude-sonnet-4-5'],
    ];
    expect(sonnetVariants).toContain(result.pricing);
    expect(result.approximate).toBe(true);
    expect(result.cost_unknown).toBe(false);
  });

  it('falls back within-family for next-version Haiku', () => {
    const result = resolvePricing('claude-haiku-5-0');
    expect(result.pricing).toBe(PRICING['claude-haiku-4-5']);
    expect(result.approximate).toBe(true);
    expect(result.cost_unknown).toBe(false);
  });

  it('flags cost_unknown for utterly-unknown family', () => {
    const result = resolvePricing('claude-mystery-7');
    // Last-resort Sonnet fallback — but flag it so cost reports
    // surface the gap instead of silently underreporting.
    expect(result.pricing).toBe(PRICING['claude-sonnet-4-6']);
    expect(result.approximate).toBe(true);
    expect(result.cost_unknown).toBe(true);
  });

  it('compares family versions numerically (e.g. -4-10 ranks above -4-9)', () => {
    // Add a fake double-digit-minor entry, resolve a future bump above
    // it, confirm we picked the higher numeric version. Restore PRICING
    // afterward so we don't bleed state into other tests.
    const FAKE = 'claude-opus-4-10';
    const original = PRICING[FAKE];
    PRICING[FAKE] = {
      in_: 7,
      out: 30,
      cache_r: 0.7,
      cache_c_5m: 8,
      cache_c_1h: 12,
    };
    try {
      const result = resolvePricing('claude-opus-4-11');
      // Must pick claude-opus-4-10 (the numerically-latest), not
      // claude-opus-4-7 which would win a string-lex sort
      // ('-4-7' > '-4-10' lexicographically).
      expect(result.pricing).toBe(PRICING[FAKE]);
      expect(result.approximate).toBe(true);
    } finally {
      if (original === undefined) delete PRICING[FAKE];
      else PRICING[FAKE] = original;
    }
  });
});

describe('computeCostMicro', () => {
  it('matches the conversation sample (Sonnet 4.6)', () => {
    // From the issue: in=1, out=52, cache_r=112865, cache_c_5m=585
    // Expected ≈ $0.036 ≈ 3.6M microcents (3,600,000).
    const cost = computeCostMicro(
      { in_: 1, out: 52, cache_r: 112865, cache_c_5m: 585, cache_c_1h: 0 },
      PRICING['claude-sonnet-4-6'],
    );
    // Exact: (1*3 + 52*15 + 112865*0.30 + 585*3.75 + 0) / 1e6 * 1e8
    // = (3 + 780 + 33859.5 + 2193.75) / 1e6 * 1e8
    // = 36836.25 / 1e6 * 1e8
    // = 3683625
    expect(cost).toBe(3683625);
  });

  it('returns 0 for empty token usage', () => {
    expect(
      computeCostMicro(
        { in_: 0, out: 0, cache_r: 0, cache_c_5m: 0, cache_c_1h: 0 },
        PRICING['claude-haiku-4-5'],
      ),
    ).toBe(0);
  });

  it('handles 1h cache writes correctly', () => {
    // 1000 tokens at $6/MTok cache_c_1h on Sonnet = $0.006 = 600,000 microcents
    const cost = computeCostMicro(
      { in_: 0, out: 0, cache_r: 0, cache_c_5m: 0, cache_c_1h: 1000 },
      PRICING['claude-sonnet-4-6'],
    );
    expect(cost).toBe(600000);
  });
});

describe('parseUsageFromBody', () => {
  it('parses non-streaming JSON response', () => {
    const body = JSON.stringify({
      id: 'msg_test',
      model: 'claude-sonnet-4-6',
      usage: {
        input_tokens: 1,
        output_tokens: 52,
        cache_read_input_tokens: 112865,
        cache_creation_input_tokens: 585,
      },
    });
    const rec = parseUsageFromBody(body, CTX, 4218, null);
    expect(rec).not.toBeNull();
    expect(rec!.api_id).toBe('msg_test');
    expect(rec!.model).toBe('claude-sonnet-4-6');
    expect(rec!.in).toBe(1);
    expect(rec!.out).toBe(52);
    expect(rec!.cache_r).toBe(112865);
    expect(rec!.cache_c_5m).toBe(585);
    expect(rec!.cache_c_1h).toBe(0);
    expect(rec!.cost_micro).toBe(3683625);
    expect(rec!.dur_ms).toBe(4218);
    expect(rec!.group).toBe('telegram_main');
    expect(rec!.tier).toBe('main');
    // Exact-match pricing — no fallback flags.
    expect(rec!.cost_unknown).toBeUndefined();
    expect(rec!.cost_approximate).toBeUndefined();
  });

  it('parses streaming SSE with cache_creation breakdown', () => {
    const startEvent = {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        model: 'claude-sonnet-4-6',
        usage: {
          input_tokens: 1,
          output_tokens: 0,
          cache_read_input_tokens: 112865,
          cache_creation_input_tokens: 585,
          cache_creation: {
            ephemeral_5m_input_tokens: 500,
            ephemeral_1h_input_tokens: 85,
          },
        },
      },
    };
    const deltaEvent = {
      type: 'message_delta',
      usage: {
        input_tokens: 1,
        output_tokens: 52,
        cache_read_input_tokens: 112865,
        cache_creation_input_tokens: 585,
        cache_creation: {
          ephemeral_5m_input_tokens: 500,
          ephemeral_1h_input_tokens: 85,
        },
      },
    };
    const sse =
      `event: message_start\ndata: ${JSON.stringify(startEvent)}\n\n` +
      `event: message_delta\ndata: ${JSON.stringify(deltaEvent)}\n\n`;
    const rec = parseUsageFromBody(sse, CTX, 1234, null);
    expect(rec).not.toBeNull();
    expect(rec!.api_id).toBe('msg_stream');
    expect(rec!.in).toBe(1);
    expect(rec!.out).toBe(52);
    expect(rec!.cache_c_5m).toBe(500);
    expect(rec!.cache_c_1h).toBe(85);
  });

  it('returns null when no usage field present', () => {
    expect(parseUsageFromBody('{"error": "x"}', CTX, 0, null)).toBeNull();
    expect(parseUsageFromBody('not json', CTX, 0, null)).toBeNull();
    expect(parseUsageFromBody('', CTX, 0, null)).toBeNull();
  });

  it('uses fallback model when response model is missing', () => {
    const body = JSON.stringify({
      usage: { input_tokens: 1, output_tokens: 1 },
    });
    const rec = parseUsageFromBody(body, CTX, 0, 'claude-haiku-4-5');
    expect(rec!.model).toBe('claude-haiku-4-5');
  });

  it('flags cost_approximate on family-prefix-fallback records (#479 sub-#2)', () => {
    // Future Opus that's not in PRICING yet — record is built at
    // Opus pricing (good enough) and flagged approximate.
    const body = JSON.stringify({
      id: 'msg_future',
      model: 'claude-opus-4-8',
      usage: { input_tokens: 1000, output_tokens: 500 },
    });
    const rec = parseUsageFromBody(body, CTX, 100, null);
    expect(rec).not.toBeNull();
    expect(rec!.model).toBe('claude-opus-4-8');
    expect(rec!.cost_approximate).toBe(true);
    expect(rec!.cost_unknown).toBeUndefined();
    // Cost is computed at Opus 4.7 rates: 1000*5 + 500*25 = 17500
    // → 17500 / 1e6 * 1e8 = 1,750,000 microcents.
    expect(rec!.cost_micro).toBe(1750000);
  });

  it('flags cost_unknown on no-family-match records (#479 sub-#2)', () => {
    const body = JSON.stringify({
      id: 'msg_alien',
      model: 'claude-mystery-7',
      usage: { input_tokens: 1000, output_tokens: 500 },
    });
    const rec = parseUsageFromBody(body, CTX, 100, null);
    expect(rec).not.toBeNull();
    expect(rec!.model).toBe('claude-mystery-7');
    expect(rec!.cost_unknown).toBe(true);
    expect(rec!.cost_approximate).toBe(true);
  });
});

describe('appendUsageRecord', () => {
  beforeEach(() => {
    _resetUsageLogState();
  });

  it('appends a JSON line and creates the directory', async () => {
    const dir = await fsp.mkdtemp(join(os.tmpdir(), 'usage-log-test-'));
    const path = join(dir, 'sub', 'usage.jsonl');
    const rec = {
      ts: '2026-05-03T17:11:41.160Z',
      group: 'telegram_main',
      tier: 'main',
      session: 'default',
      task_id: null,
      model: 'claude-sonnet-4-6',
      api_id: 'msg_xyz',
      in: 1,
      out: 52,
      cache_r: 112865,
      cache_c_5m: 585,
      cache_c_1h: 0,
      cost_micro: 3683625,
      dur_ms: 4218,
    };
    await appendUsageRecord(path, rec);
    await appendUsageRecord(path, rec);
    const content = await fsp.readFile(path, 'utf8');
    const lines = content.trim().split('\n');
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0])).toEqual(rec);
  });

  it('swallows IO errors without throwing', async () => {
    // Path under a file (not a directory) — mkdir will fail.
    const dir = await fsp.mkdtemp(join(os.tmpdir(), 'usage-log-test-'));
    const blocker = join(dir, 'blocker');
    await fsp.writeFile(blocker, 'x');
    const path = join(blocker, 'usage.jsonl'); // blocker is a file
    const rec = {
      ts: 't',
      group: 'g',
      tier: 'main',
      session: 's',
      task_id: null,
      model: 'm',
      api_id: null,
      in: 0,
      out: 0,
      cache_r: 0,
      cache_c_5m: 0,
      cache_c_1h: 0,
      cost_micro: 0,
      dur_ms: 0,
    };
    // Must not throw.
    await expect(appendUsageRecord(path, rec)).resolves.toBeUndefined();
  });
});

describe('checkSilentZero (#479 sub-#3)', () => {
  // Use fake timers so `noteMessagesRequest()` (which calls
  // `Date.now()` internally) and the synthetic `now` we pass into
  // `checkSilentZero()` share one controlled clock — no millisecond
  // drift, no flake on slow CI.
  const T0 = 1_700_000_000_000;

  beforeEach(() => {
    _resetUsageLogState();
    vi.useFakeTimers();
    vi.setSystemTime(T0);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns null when no /v1/messages traffic has been seen', () => {
    expect(checkSilentZero(T0)).toBeNull();
  });

  it('returns null during the warmup grace period', () => {
    noteMessagesRequest();
    // 30 seconds in — still inside the 60s grace window.
    vi.setSystemTime(T0 + 30 * 1000);
    expect(checkSilentZero(T0 + 30 * 1000)).toBeNull();
  });

  it('returns null while captures are flowing', () => {
    noteMessagesRequest();
    noteCaptureWrite();
    // 5 minutes in, but captures > 0 — pipeline is healthy.
    vi.setSystemTime(T0 + 5 * 60 * 1000);
    expect(checkSilentZero(T0 + 5 * 60 * 1000)).toBeNull();
  });

  it('fires when traffic has been seen but zero captures past the grace period', () => {
    noteMessagesRequest();
    // 90 seconds in — past the 60s grace, no captureWrite called.
    vi.setSystemTime(T0 + 90 * 1000);
    const diag = checkSilentZero(T0 + 90 * 1000);
    expect(diag).not.toBeNull();
    expect(diag!.messagesSeen).toBe(1);
    expect(diag!.capturesWritten).toBe(0);
    expect(diag!.ageMs).toBe(90 * 1000);
  });

  it('is edge-triggered on messagesSeen so the same broken state does not re-alert', () => {
    noteMessagesRequest();
    // First check past the grace window — should fire.
    vi.setSystemTime(T0 + 90 * 1000);
    expect(checkSilentZero(T0 + 90 * 1000)).not.toBeNull();
    // Second check with no new requests — must NOT re-fire.
    vi.setSystemTime(T0 + 120 * 1000);
    expect(checkSilentZero(T0 + 120 * 1000)).toBeNull();
    // New request arrives while still broken — re-fires once.
    noteMessagesRequest();
    vi.setSystemTime(T0 + 150 * 1000);
    expect(checkSilentZero(T0 + 150 * 1000)).not.toBeNull();
  });
});
