import { describe, it, expect } from 'vitest';

import { shouldStampNoModelWork } from './model-work.js';

describe('shouldStampNoModelWork (#901)', () => {
  // The predicate every terminal-marker site shares: stamp noModelWork
  // iff no assistant message ever carried a usage payload, i.e. the SDK
  // returned without the model producing a single turn. This is the
  // subscription-cap abort shape that recorded 92 false successes on
  // 2026-08-01.

  it('stamps when no assistant turn carried usage', () => {
    expect(shouldStampNoModelWork(undefined)).toBe(true);
  });

  it('does NOT stamp when the model produced a turn', () => {
    expect(
      shouldStampNoModelWork({ input_tokens: 1200, output_tokens: 340 }),
    ).toBe(false);
  });

  it('does NOT stamp on a zero-token usage payload', () => {
    // Presence of the payload proves an assistant message arrived, so
    // the model ran even if the counters are zero. Only absence means
    // "no turn happened" — treating zero as no-work would misclassify a
    // genuine empty assistant turn (which still burns tokens) as a cap
    // abort.
    expect(shouldStampNoModelWork({ input_tokens: 0, output_tokens: 0 })).toBe(
      false,
    );
  });
});
