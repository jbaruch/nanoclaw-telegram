import { describe, it, expect } from 'vitest';
import { shouldSynthesizeSilentStop } from './silent-stop-synthesis.js';

describe('shouldSynthesizeSilentStop', () => {
  it('synthesizes when no result event was observed and no error', () => {
    expect(shouldSynthesizeSilentStop(0, false)).toBe(true);
  });

  it('skips synthesis when a result was already emitted (healthy turn)', () => {
    expect(shouldSynthesizeSilentStop(1, false)).toBe(false);
  });

  it('skips synthesis after an error result wrote its own terminal', () => {
    expect(shouldSynthesizeSilentStop(0, true)).toBe(false);
  });

  it('skips synthesis on multi-result runs (agent teams)', () => {
    expect(shouldSynthesizeSilentStop(3, false)).toBe(false);
  });

  it('skips synthesis when both result and error markers fired', () => {
    expect(shouldSynthesizeSilentStop(1, true)).toBe(false);
  });
});
