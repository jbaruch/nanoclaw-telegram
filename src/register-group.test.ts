// Tests for the new-group default `containerConfig.stage2Enabled = true`
// behaviour applied at `registerGroup` time. The pure helper
// `applyNewGroupContainerConfigDefaults` is exported for direct
// testing; the surrounding fs side effects of `registerGroup` are
// not the subject of this test.
import { describe, it, expect } from 'vitest';

import { applyNewGroupContainerConfigDefaults } from './index.js';
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
