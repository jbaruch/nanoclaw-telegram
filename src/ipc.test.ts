import { describe, it, expect } from 'vitest';

import { applyMaintenancePrefix } from './ipc.js';

describe('applyMaintenancePrefix', () => {
  it('prepends [M] for the maintenance session', () => {
    expect(applyMaintenancePrefix('hello', 'maintenance')).toBe('[M] hello');
  });

  it('leaves text untouched for the default session', () => {
    expect(applyMaintenancePrefix('hello', 'default')).toBe('hello');
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
    expect(applyMaintenancePrefix('[M] hello', 'maintenance')).toBe(
      '[M] hello',
    );
  });

  it('ignores non-maintenance sessionName values regardless of prefix state', () => {
    expect(applyMaintenancePrefix('[M] hello', 'default')).toBe('[M] hello');
    expect(applyMaintenancePrefix('[M] hello', 'some-future-session')).toBe(
      '[M] hello',
    );
  });

  it('preserves multi-line text (prefix is line-0 only)', () => {
    expect(applyMaintenancePrefix('line1\nline2', 'maintenance')).toBe(
      '[M] line1\nline2',
    );
  });

  it('preserves HTML tags in the body (prefix sits outside them)', () => {
    expect(
      applyMaintenancePrefix('<b>bold</b> text', 'maintenance'),
    ).toBe('[M] <b>bold</b> text');
  });
});
