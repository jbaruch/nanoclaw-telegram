import { describe, it, expect } from 'vitest';

import {
  DEFAULT_SESSION_NAME,
  // `MAINTENANCE_SESSION_NAME` lives in group-queue.ts — import from there
  // so the test tracks any future rename without silently breaking.
} from './container-runner.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';
import { applyMaintenancePrefix, shouldStoreBotMessage } from './ipc.js';

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
