import { describe, it, expect } from 'vitest';
import { formatSource, wrapUntrustedInput } from './untrusted-input-sources.js';

describe('formatSource', () => {
  it('joins prefix and value with a colon', () => {
    expect(formatSource('web', 'https://example.com/page')).toBe(
      'web:https://example.com/page',
    );
    expect(formatSource('tessl', 'search=nanoclaw')).toBe(
      'tessl:search=nanoclaw',
    );
    expect(formatSource('untrusted-container', 'news-group')).toBe(
      'untrusted-container:news-group',
    );
  });

  it("returns the raw value untouched (escaping is the wrapper's job)", () => {
    expect(formatSource('web', 'a"b')).toBe('web:a"b');
  });
});

describe('wrapUntrustedInput', () => {
  it('produces an <untrusted-input> envelope with typed source', () => {
    expect(wrapUntrustedInput('hello', 'web', 'https://example.com')).toBe(
      '<untrusted-input source="web:https://example.com">\nhello\n</untrusted-input>',
    );
  });

  it('escapes embedded double quotes in the source value', () => {
    expect(wrapUntrustedInput('x', 'web', 'a"b')).toBe(
      '<untrusted-input source="web:a&quot;b">\nx\n</untrusted-input>',
    );
  });

  it('escapes ampersands in URL query strings (web:)', () => {
    expect(wrapUntrustedInput('x', 'web', 'https://example.com/?a=1&b=2')).toBe(
      '<untrusted-input source="web:https://example.com/?a=1&amp;b=2">\nx\n</untrusted-input>',
    );
  });

  it('escapes < and > in source values', () => {
    expect(wrapUntrustedInput('x', 'tessl', 'q=<id>')).toBe(
      '<untrusted-input source="tessl:q=&lt;id&gt;">\nx\n</untrusted-input>',
    );
  });

  it('escapes & first so quote escapes are not double-encoded', () => {
    // `&` is replaced before `"` so the `&` inside the resulting `&quot;`
    // entity is NOT itself re-encoded to `&amp;quot;`.
    expect(wrapUntrustedInput('x', 'web', 'a&b')).toBe(
      '<untrusted-input source="web:a&amp;b">\nx\n</untrusted-input>',
    );
    expect(wrapUntrustedInput('x', 'web', 'a"&b')).toBe(
      '<untrusted-input source="web:a&quot;&amp;b">\nx\n</untrusted-input>',
    );
  });

  it('collapses newlines inside the source value to a space', () => {
    expect(wrapUntrustedInput('x', 'file', '/a/b\n/c')).toBe(
      '<untrusted-input source="file:/a/b /c">\nx\n</untrusted-input>',
    );
  });

  it('preserves newlines inside the wrapped content body', () => {
    expect(wrapUntrustedInput('a\nb\nc', 'web', 'u')).toBe(
      '<untrusted-input source="web:u">\na\nb\nc\n</untrusted-input>',
    );
  });

  it('matches the existing #29 prompt-wrap shape after retrofit', () => {
    expect(
      wrapUntrustedInput('the prompt', 'untrusted-container', 'news-group'),
    ).toBe(
      '<untrusted-input source="untrusted-container:news-group">\nthe prompt\n</untrusted-input>',
    );
  });
});
