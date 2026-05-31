import { describe, it, expect } from 'vitest';
import { shouldSuppressEchoedResult } from './empty-turn-echo-suppression.js';

describe('shouldSuppressEchoedResult', () => {
  it('suppresses the echoed prompt when the turn emitted no content', () => {
    expect(
      shouldSuppressEchoedResult(
        false,
        'Human: <system-reminder>\n...envelope...\n<messages>...</messages>',
      ),
    ).toBe(true);
  });

  it('does not suppress real text from a turn that emitted content', () => {
    expect(shouldSuppressEchoedResult(true, 'Here is your answer.')).toBe(
      false,
    );
  });

  it('does not suppress when an empty turn also has empty result text', () => {
    expect(shouldSuppressEchoedResult(false, '')).toBe(false);
    expect(shouldSuppressEchoedResult(false, null)).toBe(false);
  });

  it('does not suppress when a content turn has empty result text', () => {
    expect(shouldSuppressEchoedResult(true, '')).toBe(false);
    expect(shouldSuppressEchoedResult(true, null)).toBe(false);
  });
});
