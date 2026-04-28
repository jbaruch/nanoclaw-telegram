import { describe, it, expect } from 'vitest';
import { computeThresholds, classifyUsage } from './threshold.js';

// Threshold formula coverage maps to design doc §5 (`docs/proposals/
// kill-auto-compaction.md`):
//
//   threshold_warn = min(70% of context, context - 200K)
//   threshold_nuke = min(80% of context, context - 100K)
//
// The key acceptance is that the formula crosses over correctly: at
// 1M context (Opus 4.7) the percentage dominates (70% = 700K, which
// is less than 1M - 200K = 800K), so warn = 700K. At 200K context
// (small model), the percentage gives 140K but the headroom floor
// gives 0 (200K - 200K), so warn = 0 — the test confirms the floor
// can clamp the threshold to a value that effectively means "always
// over threshold," which is the right behaviour for a model whose
// context can't fit any reasonable checkpoint+reentry overhead.

describe('computeThresholds — Opus 4.7 1M context', () => {
  const t = computeThresholds(1_000_000);

  it('warn at 70% (700K) — percentage dominates', () => {
    expect(t.warn).toBe(700_000);
  });

  it('nuke at 80% (800K) — percentage dominates', () => {
    expect(t.nuke).toBe(800_000);
  });

  it('echoes the contextWindow input', () => {
    expect(t.contextWindow).toBe(1_000_000);
  });
});

describe('computeThresholds — 500K context (mid-size model)', () => {
  const t = computeThresholds(500_000);

  it('warn = min(350K, 300K) = 300K — headroom floor dominates', () => {
    // 70% of 500K = 350K; 500K - 200K = 300K → floor wins.
    expect(t.warn).toBe(300_000);
  });

  it('nuke = min(400K, 400K) = 400K — both equal, either branch fine', () => {
    // 80% of 500K = 400K; 500K - 100K = 400K → tie.
    expect(t.nuke).toBe(400_000);
  });
});

describe('computeThresholds — 200K context (small model edge case)', () => {
  const t = computeThresholds(200_000);

  it('warn floor clamps to 0 — context too small for warn headroom', () => {
    // 200K - 200K = 0 → floor.
    expect(t.warn).toBe(0);
  });

  it('nuke at 100K (50% of context, headroom floor wins)', () => {
    // 80% = 160K; 200K - 100K = 100K → floor.
    expect(t.nuke).toBe(100_000);
  });
});

describe('classifyUsage', () => {
  const t = computeThresholds(1_000_000); // 700K warn, 800K nuke

  it('below warn returns below_warn', () => {
    expect(classifyUsage(0, t)).toBe('below_warn');
    expect(classifyUsage(699_999, t)).toBe('below_warn');
  });

  it('exactly at warn returns warn (boundary inclusive)', () => {
    expect(classifyUsage(700_000, t)).toBe('warn');
  });

  it('between warn and nuke returns warn', () => {
    expect(classifyUsage(750_000, t)).toBe('warn');
    expect(classifyUsage(799_999, t)).toBe('warn');
  });

  it('exactly at nuke returns nuke (boundary inclusive)', () => {
    expect(classifyUsage(800_000, t)).toBe('nuke');
  });

  it('above nuke returns nuke', () => {
    expect(classifyUsage(900_000, t)).toBe('nuke');
    expect(classifyUsage(1_000_000, t)).toBe('nuke');
  });
});
