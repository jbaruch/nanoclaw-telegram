import { describe, it, expect } from 'vitest';

import { coerceTaskTextField } from './coerce-task-prompt.js';

// #512 — host-side IPC-boundary coercion for the
// `scheduled_tasks.prompt` / `script` columns. Three terminal
// outcomes:
//   - `string` → passthrough
//   - `{type:'Buffer',data:number[]}` → decoded UTF-8
//   - anything else → `null` (caller rejects the IPC payload)
//
// `null` MUST mean "reject" rather than "store empty string", because
// the alternative — silently storing `"[object Object]"` for a
// malformed object payload — converts a crash into a row that fires
// with garbage at the next scheduled tick. Tests pin both the decode
// path AND the rejection branches.

describe('coerceTaskTextField (#512)', () => {
  it('returns string values unchanged', () => {
    expect(coerceTaskTextField('hello world')).toBe('hello world');
  });

  it('returns empty string when the input is the empty string', () => {
    // Empty string is still a string — passthrough. Caller decides
    // whether to treat it as "clear the field" semantics (the
    // `data.script || null` shape in the schedule_task / update_task
    // handlers).
    expect(coerceTaskTextField('')).toBe('');
  });

  it('decodes the JSON-Buffer shape to UTF-8 text', () => {
    const text = 'Skill(skill: "tessl__axis-review")';
    expect(
      coerceTaskTextField({
        type: 'Buffer',
        data: Array.from(Buffer.from(text, 'utf8')),
      }),
    ).toBe(text);
  });

  it('decodes the JSON-Buffer shape with multi-byte UTF-8 codepoints', () => {
    const text = 'AyeAye 🤖 — daily brief';
    expect(
      coerceTaskTextField({
        type: 'Buffer',
        data: Array.from(Buffer.from(text, 'utf8')),
      }),
    ).toBe(text);
  });

  it('rejects null', () => {
    expect(coerceTaskTextField(null)).toBeNull();
  });

  it('rejects undefined', () => {
    expect(coerceTaskTextField(undefined)).toBeNull();
  });

  it('rejects a plain object with no Buffer discriminator', () => {
    expect(coerceTaskTextField({ foo: 'bar' })).toBeNull();
  });

  it('rejects an object with `type: "Buffer"` but non-array `data`', () => {
    expect(
      coerceTaskTextField({ type: 'Buffer', data: 'not-an-array' }),
    ).toBeNull();
  });

  it('rejects an object with `data: number[]` but a different `type`', () => {
    expect(
      coerceTaskTextField({ type: 'NotABuffer', data: [104, 105] }),
    ).toBeNull();
  });

  it('rejects numbers and booleans', () => {
    expect(coerceTaskTextField(42)).toBeNull();
    expect(coerceTaskTextField(true)).toBeNull();
  });

  it('rejects arrays', () => {
    expect(coerceTaskTextField([1, 2, 3])).toBeNull();
  });
});
