import { describe, it, expect } from 'vitest';
import { sanitizeTelegramHtml } from './telegram-sanitize.js';

describe('sanitizeTelegramHtml — Markdown → HTML', () => {
  it('converts **bold** to <b>', () => {
    expect(sanitizeTelegramHtml('say **this** now')).toBe(
      'say <b>this</b> now',
    );
  });

  it('converts __bold__ to <b>', () => {
    expect(sanitizeTelegramHtml('__emphasis__')).toBe('<b>emphasis</b>');
  });

  it('converts *italic* to <i>', () => {
    expect(sanitizeTelegramHtml('feeling *great* today')).toBe(
      'feeling <i>great</i> today',
    );
  });

  it('converts _italic_ to <i>', () => {
    expect(sanitizeTelegramHtml('feeling _great_ today')).toBe(
      'feeling <i>great</i> today',
    );
  });

  it('converts `code` to <code>', () => {
    expect(sanitizeTelegramHtml('run `npm test`')).toBe(
      'run <code>npm test</code>',
    );
  });

  it('converts [text](url) to <a href>', () => {
    expect(sanitizeTelegramHtml('[Docs](https://example.com)')).toBe(
      '<a href="https://example.com">Docs</a>',
    );
  });

  it('converts # headings to <b>', () => {
    expect(sanitizeTelegramHtml('# Top\n## Sub\n### Detail')).toBe(
      '<b>Top</b>\n<b>Sub</b>\n<b>Detail</b>',
    );
  });

  it('converts - and * bullets to •', () => {
    expect(sanitizeTelegramHtml('- one\n* two\n- three')).toBe(
      '\u2022 one\n\u2022 two\n\u2022 three',
    );
  });

  it('handles mixed content in one pass', () => {
    const input = '**Bug fix**: see [ticket](https://jira.example.com/T-1) now';
    expect(sanitizeTelegramHtml(input)).toBe(
      '<b>Bug fix</b>: see <a href="https://jira.example.com/T-1">ticket</a> now',
    );
  });
});

describe('sanitizeTelegramHtml — idempotence (pre-formatted HTML)', () => {
  it('passes well-formed HTML through unchanged', () => {
    const html = '<b>bold</b> and <i>italic</i> with <code>code</code>';
    expect(sanitizeTelegramHtml(html)).toBe(html);
  });

  it('is idempotent — running twice produces the same result', () => {
    const input = '**bold** and *italic* and [link](https://a.com)';
    const once = sanitizeTelegramHtml(input);
    expect(sanitizeTelegramHtml(once)).toBe(once);
  });

  it('preserves <a href> tags even when text contains underscores', () => {
    const input = '<a href="https://a.com/path_with_underscores">click</a>';
    expect(sanitizeTelegramHtml(input)).toBe(input);
  });
});

describe('sanitizeTelegramHtml — protected regions', () => {
  it('does not transform underscores inside URLs', () => {
    const input = 'see https://example.com/path_with_underscores for details';
    expect(sanitizeTelegramHtml(input)).toBe(input);
  });

  it('does not transform underscores inside email addresses', () => {
    const input = 'contact foo_bar@example.com please';
    expect(sanitizeTelegramHtml(input)).toBe(input);
  });

  it('preserves ftp URLs', () => {
    const input = 'archive at ftp://files.example.com/path_name';
    expect(sanitizeTelegramHtml(input)).toBe(input);
  });

  it('inline code blocks with * characters are not mangled', () => {
    expect(sanitizeTelegramHtml('use `a*b*c` as the pattern')).toBe(
      'use <code>a*b*c</code> as the pattern',
    );
  });
});

describe('sanitizeTelegramHtml — edge cases', () => {
  it('empty input returns empty', () => {
    expect(sanitizeTelegramHtml('')).toBe('');
  });

  it('text with no Markdown passes through unchanged', () => {
    expect(sanitizeTelegramHtml('plain text here')).toBe('plain text here');
  });

  it('lone asterisk is not treated as italic', () => {
    expect(sanitizeTelegramHtml('use a * b for multiply')).toBe(
      'use a * b for multiply',
    );
  });

  it('snake_case_identifier is not treated as italic', () => {
    expect(sanitizeTelegramHtml('call my_func_name please')).toBe(
      'call my_func_name please',
    );
  });

  it('handles multi-line mixed input', () => {
    const input = '# Release\n- **feat**: added X\n- _fix_: Y';
    expect(sanitizeTelegramHtml(input)).toBe(
      '<b>Release</b>\n\u2022 <b>feat</b>: added X\n\u2022 <i>fix</i>: Y',
    );
  });
});
