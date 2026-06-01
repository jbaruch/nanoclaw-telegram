import { describe, it, expect } from 'vitest';

import {
  DEFAULT_SESSION_NAME,
  // `MAINTENANCE_SESSION_NAME` lives in group-queue.ts — import from there
  // so the test tracks any future rename without silently breaking.
} from './container-runner.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';
import {
  applyMaintenancePrefix,
  fetchSessionizeEventsBatch,
  normalizeSessionizeEvent,
  shouldStoreBotMessage,
} from './ipc.js';

describe('applyMaintenancePrefix', () => {
  it('prepends [M] for the maintenance session', () => {
    expect(applyMaintenancePrefix('hello', MAINTENANCE_SESSION_NAME)).toBe(
      '[M] hello',
    );
  });

  it('leaves text untouched for the default session', () => {
    expect(applyMaintenancePrefix('hello', DEFAULT_SESSION_NAME)).toBe('hello');
  });

  it('leaves text untouched when sessionName is undefined', () => {
    // A pre-upgrade container that didn't stamp sessionName on the IPC
    // payload should NOT get the maintenance prefix by accident —
    // silent prefixing of user-facing sends is worse than no prefix.
    expect(applyMaintenancePrefix('hello', undefined)).toBe('hello');
  });

  it('is idempotent — does not double-prefix already-prefixed text', () => {
    // Defensive: if an upstream bug or a future re-entry ever feeds
    // already-prefixed text back in, we shouldn't end up with `[M] [M]`.
    expect(applyMaintenancePrefix('[M] hello', MAINTENANCE_SESSION_NAME)).toBe(
      '[M] hello',
    );
  });

  it('ignores non-maintenance sessionName values regardless of prefix state', () => {
    expect(applyMaintenancePrefix('[M] hello', DEFAULT_SESSION_NAME)).toBe(
      '[M] hello',
    );
    expect(applyMaintenancePrefix('[M] hello', 'some-future-session')).toBe(
      '[M] hello',
    );
  });

  it('preserves multi-line text (prefix is line-0 only)', () => {
    expect(
      applyMaintenancePrefix('line1\nline2', MAINTENANCE_SESSION_NAME),
    ).toBe('[M] line1\nline2');
  });

  it('preserves HTML tags in the body (prefix sits outside them)', () => {
    expect(
      applyMaintenancePrefix('<b>bold</b> text', MAINTENANCE_SESSION_NAME),
    ).toBe('[M] <b>bold</b> text');
  });
});

// -----------------------------------------------------------------
// shouldStoreBotMessage — gates `bot-…` row writes on send success.
// See `src/ipc.ts` for the rationale; this is the predicate extracted
// from the IPC `send_message` handler so the gating logic is unit-
// testable without staging the whole watcher (file events, deps mock,
// db init, atomic writes). The full handler exercises the same
// predicate at runtime — testing it in isolation gives the regression
// signal the OpenAI policy reviewer asked for on PR #232.
// -----------------------------------------------------------------

describe('shouldStoreBotMessage', () => {
  it('returns false for Telegram when sentMsgId is undefined', () => {
    // The textbook phantom-row case: Telegram send was swallowed
    // (400 from a bad reply_to, network blip, malformed HTML even
    // after the plain-text fallback, blocked-by-user, rate-limit).
    // A row written here would silence heartbeat alerts on a chat
    // the user actually received nothing in.
    expect(shouldStoreBotMessage('tg:100200300', undefined)).toBe(false);
  });

  it('returns true for Telegram when sentMsgId is a populated string', () => {
    expect(shouldStoreBotMessage('tg:100200300', '12345')).toBe(true);
  });

  it('returns true for Telegram when sentMsgId is an empty string', () => {
    // Forward-compat with the documented `string | undefined` contract:
    // the upstream `sentMsgId` normalization explicitly avoided
    // truthiness checks so future Telegram ids of `''` or `'0'` aren't
    // dropped on the floor. The gate must match that promise.
    expect(shouldStoreBotMessage('tg:100200300', '')).toBe(true);
  });

  it('returns true for Telegram when sentMsgId is the literal string "0"', () => {
    expect(shouldStoreBotMessage('tg:100200300', '0')).toBe(true);
  });

  it('returns true for non-Telegram channels regardless of sentMsgId', () => {
    // WhatsApp / Slack / Discord `Channel.sendMessage` permit returning
    // void on success (see `src/types.ts`); absence of an id is not a
    // failure signal there. Until those channels surface their own
    // success-id contract, gating would punish a passing send.
    expect(shouldStoreBotMessage('120363012345@g.us', undefined)).toBe(true);
    expect(shouldStoreBotMessage('120363012345@g.us', 'wa-id-abc')).toBe(true);
    expect(shouldStoreBotMessage('slack:C12345', undefined)).toBe(true);
  });
});

// -----------------------------------------------------------------
// normalizeSessionizeEvent — the shared mapper behind both the
// singular `sessionize_get_event` and the batch `sessionize_get_events`
// IPC handlers (#656). Fixed payloads keep these deterministic; the
// far-future / far-past CFP end dates make `cfp_open` independent of the
// wall clock at run time.
// -----------------------------------------------------------------

describe('normalizeSessionizeEvent', () => {
  const fullEvent = {
    name: 'Devoxx Belgium 2026',
    cfpDates: {
      startUtc: '2026-01-01T00:00:00Z',
      endUtc: '2999-12-31T23:59:59Z',
      start: '2026-01-01',
      end: '2999-12-31',
    },
    eventDates: { start: '2026-10-05', end: '2026-10-09' },
    location: { full: 'Antwerp, Belgium', city: 'Antwerp', country: 'Belgium' },
    timezone: { iana: 'Europe/Brussels' },
    expensesCovered: { travel: true, accommodation: true },
    isOnline: false,
    website: 'https://devoxx.be',
    cfpLink: 'https://devoxx.be/cfp',
    organizer: 'Devoxx',
  };

  it('maps nested Sessionize fields onto the flat shape', () => {
    const r = normalizeSessionizeEvent(fullEvent, 'devoxx-be-2026');
    expect(r.name).toBe('Devoxx Belgium 2026');
    expect(r.cfp_start).toBe('2026-01-01T00:00:00Z');
    expect(r.cfp_end).toBe('2999-12-31T23:59:59Z');
    expect(r.cfp_start_local).toBe('2026-01-01');
    expect(r.cfp_end_local).toBe('2999-12-31');
    expect(r.conf_start).toBe('2026-10-05');
    expect(r.conf_end).toBe('2026-10-09');
    expect(r.location).toBe('Antwerp, Belgium');
    expect(r.city).toBe('Antwerp');
    expect(r.country).toBe('Belgium');
    expect(r.timezone).toBe('Europe/Brussels');
    expect(r.is_online).toBe(false);
    expect(r.website).toBe('https://devoxx.be');
    expect(r.expenses_covered).toEqual({ travel: true, accommodation: true });
    expect(r.organizer).toBe('Devoxx');
  });

  it('reports cfp_open=true when the CFP end is in the future', () => {
    expect(normalizeSessionizeEvent(fullEvent, 'devoxx-be-2026').cfp_open).toBe(
      true,
    );
  });

  it('reports cfp_open=false when the CFP end is in the past', () => {
    const past = { cfpDates: { endUtc: '2000-01-01T00:00:00Z' } };
    expect(normalizeSessionizeEvent(past, 'old-conf').cfp_open).toBe(false);
  });

  it('reports cfp_open=false when cfpDates.endUtc is missing', () => {
    expect(normalizeSessionizeEvent({ cfpDates: {} }, 'no-cfp').cfp_open).toBe(
      false,
    );
  });

  it('uses event.cfpLink for cfp_url when present', () => {
    expect(normalizeSessionizeEvent(fullEvent, 'devoxx-be-2026').cfp_url).toBe(
      'https://devoxx.be/cfp',
    );
  });

  it('falls back to a constructed cfp_url from the slug when cfpLink is absent', () => {
    const noLink = { name: 'JFokus' };
    expect(normalizeSessionizeEvent(noLink, 'jfokus-2026').cfp_url).toBe(
      'https://sessionize.com/jfokus-2026/',
    );
  });

  it('does not throw and defaults nested fields for a sparse payload', () => {
    // A partial Sessionize response (no nested groups at all) must yield
    // undefined leaf fields and an empty expenses object — never a throw
    // that would sink the whole batch.
    const r = normalizeSessionizeEvent({}, 'sparse');
    expect(r.location).toBeUndefined();
    expect(r.city).toBeUndefined();
    expect(r.timezone).toBeUndefined();
    expect(r.conf_start).toBeUndefined();
    expect(r.expenses_covered).toEqual({});
    expect(r.cfp_open).toBe(false);
    expect(r.cfp_url).toBe('https://sessionize.com/sparse/');
  });
});

// -----------------------------------------------------------------
// fetchSessionizeEventsBatch — bounded-concurrency fan-out behind
// `sessionize_get_events` (#656). The injected fetcher keeps these
// deterministic (no real HTTP); the assertions cover the contract the
// nightly CFP sync depends on: input order preserved, one bad slug
// isolated rather than sinking the batch, and concurrency capped.
// -----------------------------------------------------------------

describe('fetchSessionizeEventsBatch', () => {
  it('preserves input order across chunk boundaries', async () => {
    const slugs = ['a', 'b', 'c', 'd', 'e'];
    const out = await fetchSessionizeEventsBatch(
      slugs,
      async (slug) => ({ slug }),
      2,
    );
    expect(out.map((r) => r.slug)).toEqual(slugs);
  });

  it('isolates a throwing slug into a {slug, error} entry without sinking the rest', async () => {
    const out = await fetchSessionizeEventsBatch(
      ['ok1', 'bad', 'ok2'],
      async (slug) => {
        if (slug === 'bad') throw new Error('boom');
        return { slug, name: 'fine' };
      },
      2,
    );
    expect(out).toEqual([
      { slug: 'ok1', name: 'fine' },
      { slug: 'bad', error: 'boom' },
      { slug: 'ok2', name: 'fine' },
    ]);
  });

  it('stringifies a non-Error throw into the error field', async () => {
    const out = await fetchSessionizeEventsBatch(
      ['x'],
      async () => {
        throw 'plain-string-failure';
      },
      4,
    );
    expect(out).toEqual([{ slug: 'x', error: 'plain-string-failure' }]);
  });

  it('never runs more than `concurrency` fetchers at once', async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const slugs = Array.from({ length: 7 }, (_, i) => `s${i}`);
    await fetchSessionizeEventsBatch(
      slugs,
      async (slug) => {
        inFlight++;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await Promise.resolve();
        inFlight--;
        return { slug };
      },
      3,
    );
    expect(maxInFlight).toBe(3);
  });

  it('returns an empty array for no slugs', async () => {
    const out = await fetchSessionizeEventsBatch(
      [],
      async (slug) => ({ slug }),
      3,
    );
    expect(out).toEqual([]);
  });

  it('clamps a non-positive concurrency to 1 instead of hanging', async () => {
    // `i += 0` would never advance the loop and hang the host. The clamp
    // degrades a bad value to serial processing rather than looping forever
    // — proven here by the call resolving at all (and processing every slug).
    const out = await fetchSessionizeEventsBatch(
      ['a', 'b', 'c'],
      async (slug) => ({ slug }),
      0,
    );
    expect(out.map((r) => r.slug)).toEqual(['a', 'b', 'c']);
  });
});
