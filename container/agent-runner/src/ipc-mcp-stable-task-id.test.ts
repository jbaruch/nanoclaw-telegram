import { describe, it, expect } from 'vitest';

import { STABLE_TASK_ID_REGEX } from './stable-task-id.js';

// #440 — Caller-supplied stable IDs for `schedule_task` are pinned to
// a tight format the bridge enforces via Zod. The Zod field uses the
// regex below (re-exported from the bridge module so this test pins
// the same regex, not a re-inlined drift); these cases enumerate the
// branches the bridge has to honour: legitimate intent-named ids
// accepted, hostile shapes (Unicode, control bytes, leading/trailing
// hyphens, length > 64) rejected.
//
// The corresponding host-side path (`src/ipc.ts:998`) already accepted
// `data.taskId || autogen` — so the regex is the only new contract on
// the wire. Anything that passes here goes straight into the
// `scheduled_tasks.id` PK column.

describe('STABLE_TASK_ID_REGEX (#440)', () => {
  describe('accepts intent-named ids', () => {
    const valid = [
      // Sub-skill bootstrap rows from #432.
      'task-subskill-memory-rotation',
      'task-subskill-state-purge',
      // Overlay-plugin shapes from #305.
      'task-overlay-cron-foo',
      'task-overlay-bar-baz',
      // Audit-replay shapes from #375.
      'task-audit-replay-2026-q3',
      // Single-character / minimal valid forms.
      'a',
      '0',
      // Digits-only and mixed.
      '42',
      'task42',
      'task-42-q3',
      // Maximum length (exactly 64 chars).
      'a' + 'a'.repeat(63),
    ];
    for (const id of valid) {
      it(`accepts ${JSON.stringify(id)}`, () => {
        expect(STABLE_TASK_ID_REGEX.test(id)).toBe(true);
      });
    }
  });

  describe('rejects hostile / malformed shapes', () => {
    const invalid: Array<[string, string]> = [
      // Empty string — Zod's `.regex()` runs on every value, even
      // empty; we want a non-empty id since it's a PK.
      ['', 'empty'],
      // Leading hyphen — would clash with shell flags + with `task-`
      // autogen pattern parsing.
      ['-task-foo', 'leading hyphen'],
      // Trailing hyphen — visually ambiguous + cannot tell apart from
      // a truncated id at log boundaries.
      ['task-foo-', 'trailing hyphen'],
      // Whitespace.
      ['task subskill', 'space'],
      ['task\tfoo', 'tab'],
      ['task\nfoo', 'newline'],
      // Uppercase — folds against grep semantics + collides with
      // Linux fs case-sensitivity assumptions in operator scripts.
      ['Task-Foo', 'uppercase'],
      ['TASK', 'all-caps'],
      // Non-ASCII / Unicode.
      ['task-фоо', 'cyrillic'],
      ['task-🔥', 'emoji'],
      // Punctuation other than hyphen.
      ['task_foo', 'underscore'],
      ['task.foo', 'dot'],
      ['task/foo', 'slash'],
      ['task:foo', 'colon'],
      ['task@foo', 'at'],
      // Over-length (65 chars).
      ['a' + 'a'.repeat(64), '65 chars'],
      // Control characters.
      ['task\x00null', 'NUL byte'],
      ['task\x07bell', 'BEL byte'],
    ];
    for (const [id, label] of invalid) {
      it(`rejects ${label} (${JSON.stringify(id)})`, () => {
        expect(STABLE_TASK_ID_REGEX.test(id)).toBe(false);
      });
    }
  });

  it('matches the autogen pattern emitted by the bridge fallback', () => {
    // The fallback path produces ids shaped like `task-<ms>-<rand>`,
    // e.g. `task-1730491234567-a1b2c3`. Even though callers should not
    // re-use this shape (the autogen is the bridge\'s reserved space),
    // the format itself satisfies the regex — guarding against an
    // accidental tightening that breaks the legacy auto-id corpus.
    const autogen = `task-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    expect(STABLE_TASK_ID_REGEX.test(autogen)).toBe(true);
  });
});
