// Tests for the new-group default `containerConfig.stage2Enabled = true`
// behaviour applied at `registerGroup` time. The pure helper
// `applyNewGroupContainerConfigDefaults` is exported for direct
// testing; the surrounding fs side effects of `registerGroup` are
// not the subject of this test.
import { describe, it, expect, beforeEach } from 'vitest';

import {
  applyNewGroupContainerConfigDefaults,
  hydrateRegisteredGroupTriggerPatterns,
} from './index.js';
import {
  _initTestDatabase,
  getTriggerPatterns,
  setRegisteredGroup,
} from './db.js';
import type { RegisteredGroup } from './types.js';

const BASE: RegisteredGroup = {
  name: 'Reg Test',
  folder: 'telegram_regtest',
  trigger: '@andy',
  added_at: '2024-01-01T00:00:00Z',
  requiresTrigger: false,
};

describe('applyNewGroupContainerConfigDefaults', () => {
  it('NEW group with no explicit stage2Enabled → stage2Enabled = true', () => {
    const out = applyNewGroupContainerConfigDefaults(BASE, true);
    expect(out.containerConfig?.stage2Enabled).toBe(true);
  });

  it('NEW group with explicit stage2Enabled=false → caller-pinned wins', () => {
    const out = applyNewGroupContainerConfigDefaults(
      { ...BASE, containerConfig: { stage2Enabled: false } },
      true,
    );
    expect(out.containerConfig?.stage2Enabled).toBe(false);
  });

  it('NEW group with explicit stage2Enabled=true → preserved', () => {
    const out = applyNewGroupContainerConfigDefaults(
      { ...BASE, containerConfig: { stage2Enabled: true } },
      true,
    );
    expect(out.containerConfig?.stage2Enabled).toBe(true);
  });

  it('NEW group with other containerConfig fields → fields preserved + stage2 added', () => {
    const out = applyNewGroupContainerConfigDefaults(
      { ...BASE, containerConfig: { trusted: true } },
      true,
    );
    expect(out.containerConfig?.stage2Enabled).toBe(true);
    expect(out.containerConfig?.trusted).toBe(true);
  });

  it('EXISTING group with no stage2Enabled → unchanged (no auto-flip)', () => {
    const out = applyNewGroupContainerConfigDefaults(BASE, false);
    expect(out.containerConfig?.stage2Enabled).toBeUndefined();
    // Returned reference is identical (no clone for the no-op path).
    expect(out).toBe(BASE);
  });

  it('EXISTING group with stage2Enabled=false → preserved (no auto-flip)', () => {
    const input: RegisteredGroup = {
      ...BASE,
      containerConfig: { stage2Enabled: false },
    };
    const out = applyNewGroupContainerConfigDefaults(input, false);
    expect(out.containerConfig?.stage2Enabled).toBe(false);
  });
});

// #670 — register_group's IPC payload carries only the `trigger` string,
// never a `triggerPatterns` config. `registerGroup` persists it (the
// serializer synthesizes the keyword column) and must then cache the
// HYDRATED group, not the raw payload — the trigger gate reads
// `triggerPatterns` from the in-memory registry and would otherwise see
// "no patterns configured" and fail-open on every message until reload.
describe('hydrateRegisteredGroupTriggerPatterns (#670)', () => {
  beforeEach(() => {
    _initTestDatabase();
  });

  it('repopulates triggerPatterns from the persisted row when the payload omits it', () => {
    const payload: RegisteredGroup = {
      name: 'WTF',
      folder: 'telegram_wtf',
      trigger: '@AyeAye',
      added_at: '2024-01-01T00:00:00.000Z',
      requiresTrigger: true,
    };
    // The register_group IPC payload never carries a triggerPatterns config.
    expect(payload.triggerPatterns).toBeUndefined();

    // registerGroup persists the payload, then caches the hydrated object.
    setRegisteredGroup('wtf@g.us', payload);
    const cached = hydrateRegisteredGroupTriggerPatterns(payload, 'wtf@g.us');

    expect(cached.triggerPatterns).toBeDefined();
    expect(cached.triggerPatterns!.patterns).toHaveLength(1);
    expect(cached.triggerPatterns!.patterns[0]).toMatchObject({
      pattern: '@AyeAye',
      kind: 'keyword',
      source: 'owner-set',
    });
    // The DB row the hydration reads from agrees.
    expect(getTriggerPatterns('wtf@g.us')!.patterns[0].pattern).toBe('@AyeAye');
    // Every other field of the payload is preserved.
    expect(cached.trigger).toBe('@AyeAye');
    expect(cached.requiresTrigger).toBe(true);
    expect(cached.folder).toBe('telegram_wtf');
  });
});
