import { describe, it, expect, vi } from 'vitest';

import {
  buildOneCliDenialAlert,
  createTierDenialAlertSender,
  DENIAL_ALERT_COOLDOWN_MS,
  DENIAL_ALERT_STREAK,
  OneCliDenialTracker,
  type AlertChannel,
} from './onecli-denial-alert.js';

// Fixed reference instant. Every `nowMs` below is derived from it so the
// streak and cooldown assertions are the same on any run date
// (`coding-policy: testing-standards` Determinism).
const T0 = Date.parse('2026-07-28T12:00:00Z');

describe('buildOneCliDenialAlert', () => {
  const input = {
    tier: 'untrusted' as const,
    agentIdentifier: 'nanoclaw-untrusted',
    upstreamHost: 'api.anthropic.com',
    status: 401,
    consecutiveDenials: 3,
  };

  it('names the tier, the vault agent, the host, and the status', () => {
    const text = buildOneCliDenialAlert(input);
    expect(text).toContain('untrusted');
    expect(text).toContain('nanoclaw-untrusted');
    expect(text).toContain('api.anthropic.com');
    expect(text).toContain('401');
    expect(text).toContain('3 consecutive');
  });

  it('points at effective-credentials rather than the .env token', () => {
    const text = buildOneCliDenialAlert(input);
    expect(text).toContain('effective-credentials');
    expect(text).not.toContain('.env');
  });

  it('carries no token, proxy URL, or CA', () => {
    // The gateway URL embeds the API key in its userinfo
    // (`http://x:<key>@gw:port`), so a leak here would put a live
    // credential into a chat message (`coding-policy: no-secrets`). The
    // builder takes no such input, and this pins that it never grows one.
    const text = buildOneCliDenialAlert(input);
    expect(text).not.toMatch(/aoc_|sk-ant-|Bearer /);
    expect(text).not.toMatch(/https?:\/\//);
    expect(text).not.toContain('BEGIN CERTIFICATE');
  });

  it('renders as plain text with no channel-specific markup', () => {
    // Crosses whatever channel main is on; Markdown or HTML would show
    // up as literal characters on the others.
    const text = buildOneCliDenialAlert(input);
    expect(text).not.toMatch(/[*_`]|<[a-z]/);
  });
});

describe('OneCliDenialTracker', () => {
  it('stays silent until the streak threshold is reached', () => {
    const tracker = new OneCliDenialTracker();
    for (let i = 1; i < DENIAL_ALERT_STREAK; i++) {
      expect(tracker.recordDenial('untrusted', T0 + i)).toBeNull();
    }
    expect(tracker.recordDenial('untrusted', T0 + DENIAL_ALERT_STREAK)).toBe(
      DENIAL_ALERT_STREAK,
    );
  });

  it('counts each tier separately', () => {
    // A broken untrusted grant must not push `main` toward an alert
    // about a credential that works.
    const tracker = new OneCliDenialTracker();
    for (let i = 0; i < DENIAL_ALERT_STREAK; i++) {
      tracker.recordDenial('untrusted', T0 + i);
    }
    expect(tracker.streakFor('main')).toBe(0);
    expect(tracker.recordDenial('main', T0 + 10)).toBeNull();
  });

  it('resets the streak when a request succeeds', () => {
    // Denials separated by working requests are blips. Without the
    // reset they would accumulate across days into a false alert about
    // a tier that is fine.
    const tracker = new OneCliDenialTracker();
    for (let i = 0; i < DENIAL_ALERT_STREAK - 1; i++) {
      tracker.recordDenial('trusted', T0 + i);
    }
    tracker.recordSuccess('trusted');
    expect(tracker.streakFor('trusted')).toBe(0);
    expect(tracker.recordDenial('trusted', T0 + 100)).toBeNull();
  });

  it('suppresses repeat alerts for one tier inside the cooldown', () => {
    // A missing grant denies every request, so without the cooldown the
    // operator gets one chat message per API call.
    const tracker = new OneCliDenialTracker();
    for (let i = 0; i < DENIAL_ALERT_STREAK - 1; i++) {
      tracker.recordDenial('untrusted', T0 + i);
    }
    expect(tracker.recordDenial('untrusted', T0 + 10)).toBe(
      DENIAL_ALERT_STREAK,
    );
    expect(
      tracker.recordDenial('untrusted', T0 + DENIAL_ALERT_COOLDOWN_MS - 1),
    ).toBeNull();
  });

  it('alerts again once the cooldown has elapsed', () => {
    // All the streak-building calls land on T0 so the alert stamp is
    // T0 exactly; the cooldown is then measured from a known instant
    // rather than from whichever call happened to trip the threshold.
    const tracker = new OneCliDenialTracker();
    for (let i = 0; i < DENIAL_ALERT_STREAK; i++) {
      tracker.recordDenial('untrusted', T0);
    }
    expect(
      tracker.recordDenial('untrusted', T0 + DENIAL_ALERT_COOLDOWN_MS),
    ).toBeGreaterThanOrEqual(DENIAL_ALERT_STREAK);
  });

  it('keeps the cooldown across a recovery inside the window', () => {
    // A tier that recovers and breaks again within the hour is the same
    // incident. The streak resets so the condition has to re-establish,
    // but the cooldown stamp is deliberately not cleared.
    const tracker = new OneCliDenialTracker();
    for (let i = 0; i < DENIAL_ALERT_STREAK; i++) {
      tracker.recordDenial('untrusted', T0 + i);
    }
    tracker.recordSuccess('untrusted');
    for (let i = 0; i < DENIAL_ALERT_STREAK - 1; i++) {
      tracker.recordDenial('untrusted', T0 + 1000 + i);
    }
    expect(tracker.recordDenial('untrusted', T0 + 2000)).toBeNull();
  });

  it('reports a zero streak for a tier with no denials recorded', () => {
    expect(new OneCliDenialTracker().streakFor('untrusted')).toBe(0);
  });
});

describe('createTierDenialAlertSender', () => {
  function makeChannel(
    connected = true,
    send: () => Promise<unknown> = () => Promise.resolve(),
  ): AlertChannel & { sendMessage: ReturnType<typeof vi.fn> } {
    return {
      isConnected: () => connected,
      sendMessage: vi.fn(send),
    } as AlertChannel & { sendMessage: ReturnType<typeof vi.fn> };
  }

  it('sends to the group flagged as main', () => {
    // Not to the affected tier's own chats: the operator reads main,
    // and the tier this is about cannot answer by definition.
    const channel = makeChannel();
    const warn = vi.fn();
    const send = createTierDenialAlertSender({
      registeredGroups: () => ({
        'telegram_other@g.us': { isMain: false },
        'telegram_main@g.us': { isMain: true },
      }),
      findChannel: () => channel,
      warn,
    });

    send('denial text');

    expect(channel.sendMessage).toHaveBeenCalledWith(
      'telegram_main@g.us',
      'denial text',
    );
    expect(warn).not.toHaveBeenCalled();
  });

  it('warns instead of throwing when no group is registered as main', () => {
    const warn = vi.fn();
    const send = createTierDenialAlertSender({
      registeredGroups: () => ({ 'telegram_other@g.us': { isMain: false } }),
      findChannel: () => makeChannel(),
      warn,
    });

    expect(() => send('denial text')).not.toThrow();
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][1])).toContain('no group is registered');
  });

  it('warns when the main group has no channel wired', () => {
    const warn = vi.fn();
    const send = createTierDenialAlertSender({
      registeredGroups: () => ({ 'telegram_main@g.us': { isMain: true } }),
      findChannel: () => null,
      warn,
    });

    send('denial text');

    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][1])).toContain('not connected');
  });

  it('does not send over a disconnected channel', () => {
    const channel = makeChannel(false);
    const warn = vi.fn();
    const send = createTierDenialAlertSender({
      registeredGroups: () => ({ 'telegram_main@g.us': { isMain: true } }),
      findChannel: () => channel,
      warn,
    });

    send('denial text');

    expect(channel.sendMessage).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it('degrades a failed send to a warn rather than an unhandled rejection', async () => {
    // The proxy calls this synchronously from a response handler. A
    // rejection escaping here is something newer Node can terminate the
    // orchestrator over, and the ERROR line is already the durable
    // record — so the send failure is only ever a warn.
    const channel = makeChannel(true, () => Promise.reject(new Error('down')));
    const warn = vi.fn();
    const send = createTierDenialAlertSender({
      registeredGroups: () => ({ 'telegram_main@g.us': { isMain: true } }),
      findChannel: () => channel,
      warn,
    });

    expect(() => send('denial text')).not.toThrow();
    // Let the rejection settle before asserting on the handler.
    await Promise.resolve();
    await Promise.resolve();
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][1])).toContain('send failed');
  });

  it('reads the group registry at send time, not at wiring time', () => {
    // The proxy starts before any container exists, so the callback is
    // built before a main group can be resolved. Capturing the registry
    // eagerly would make every alert miss.
    let groups: Record<string, { isMain?: boolean }> = {};
    const channel = makeChannel();
    const send = createTierDenialAlertSender({
      registeredGroups: () => groups,
      findChannel: () => channel,
      warn: vi.fn(),
    });

    groups = { 'telegram_main@g.us': { isMain: true } };
    send('denial text');

    expect(channel.sendMessage).toHaveBeenCalledWith(
      'telegram_main@g.us',
      'denial text',
    );
  });
});
