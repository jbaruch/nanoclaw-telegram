import { describe, it, expect } from 'vitest';

import { coerceTaskPrompt, formatTaskRow } from './format-task-row.js';

// #512 — `mcp__nanoclaw__list_tasks` blew up with
// `t.prompt.slice is not a function` because one row in
// `scheduled_tasks` had a BLOB-typed `prompt` column. better-sqlite3
// returns BLOB as `Buffer`, the host JSON.stringifies it as
// `{type:'Buffer',data:[...]}`, and the prior reader assumed `string`.
// These tests pin the formatter's defensive coercion so a single odd
// row can never again poison the whole listing.

describe('coerceTaskPrompt (#512)', () => {
  it('returns string prompts unchanged', () => {
    expect(coerceTaskPrompt('hello world')).toBe('hello world');
  });

  it('returns empty string for null', () => {
    expect(coerceTaskPrompt(null)).toBe('');
  });

  it('returns empty string for undefined', () => {
    expect(coerceTaskPrompt(undefined)).toBe('');
  });

  it('decodes the JSON-Buffer shape that BLOB columns produce', () => {
    const text = 'Skill(skill: "tessl__axis-review")';
    const blobShape = {
      type: 'Buffer',
      data: Array.from(Buffer.from(text, 'utf8')),
    };
    expect(coerceTaskPrompt(blobShape)).toBe(text);
  });

  it('does NOT misinterpret a non-Buffer object that happens to have type/data fields', () => {
    // `type: 'Buffer'` AND `data: number[]` is the discriminator. An
    // object with `type: 'something-else'` falls through to String().
    expect(coerceTaskPrompt({ type: 'NotABuffer', data: [104, 105] })).toBe(
      '[object Object]',
    );
  });

  it('does NOT misinterpret a Buffer-shape with non-array data', () => {
    expect(coerceTaskPrompt({ type: 'Buffer', data: 'not-an-array' })).toBe(
      '[object Object]',
    );
  });

  it('coerces other types via String() rather than throwing', () => {
    expect(coerceTaskPrompt(42)).toBe('42');
    expect(coerceTaskPrompt(true)).toBe('true');
    expect(coerceTaskPrompt({ foo: 'bar' })).toBe('[object Object]');
  });
});

describe('formatTaskRow (#512)', () => {
  it('formats a normal string-prompt row', () => {
    const row = {
      id: 'task-498-axis-review-7d',
      prompt: 'Skill(skill: "tessl__axis-review")',
      schedule_type: 'cron',
      schedule_value: '0 9 * * *',
      status: 'active',
      next_run: '2026-05-12T09:00:00Z',
    };
    expect(formatTaskRow(row)).toBe(
      '- [task-498-axis-review-7d] Skill(skill: "tessl__axis-review")... (cron: 0 9 * * *) - active, next: 2026-05-12T09:00:00Z',
    );
  });

  it('formats a BLOB-prompt row without throwing', () => {
    const text = 'Skill(skill: "tessl__axis-review")';
    const row = {
      id: 'task-498-axis-review-7d',
      prompt: { type: 'Buffer', data: Array.from(Buffer.from(text, 'utf8')) },
      schedule_type: 'cron',
      schedule_value: '0 9 * * *',
      status: 'active',
      next_run: '2026-05-12T09:00:00Z',
    };
    expect(formatTaskRow(row)).toBe(
      '- [task-498-axis-review-7d] Skill(skill: "tessl__axis-review")... (cron: 0 9 * * *) - active, next: 2026-05-12T09:00:00Z',
    );
  });

  it('renders next_run as N/A when null/empty', () => {
    const row = {
      id: 'task-1',
      prompt: 'do thing',
      schedule_type: 'once',
      schedule_value: '2026-05-06T12:00:00Z',
      status: 'paused',
      next_run: null,
    };
    expect(formatTaskRow(row)).toBe(
      '- [task-1] do thing... (once: 2026-05-06T12:00:00Z) - paused, next: N/A',
    );
  });

  it('truncates long prompts at 50 chars (matching the prior format)', () => {
    const longPrompt = 'a'.repeat(120);
    const row = {
      id: 'task-1',
      prompt: longPrompt,
      schedule_type: 'cron',
      schedule_value: '* * * * *',
      status: 'active',
      next_run: '2026-05-06T12:00:00Z',
    };
    const result = formatTaskRow(row);
    expect(result).toContain(`${'a'.repeat(50)}...`);
    expect(result).not.toContain('a'.repeat(51));
  });

  it('survives missing fields without throwing', () => {
    expect(() => formatTaskRow({})).not.toThrow();
  });
});
