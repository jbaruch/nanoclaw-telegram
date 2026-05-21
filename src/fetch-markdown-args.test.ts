import { describe, it, expect } from 'vitest';

import {
  buildSnitchmdFlags,
  formatSnitchmdHeader,
  parseFetchMarkdownUrl,
  parseSnitchmdStdout,
} from './fetch-markdown-args.js';

describe('parseFetchMarkdownUrl', () => {
  it('accepts a standard https URL', () => {
    const r = parseFetchMarkdownUrl('https://example.com/path');
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.url.host).toBe('example.com');
  });

  it('accepts an http URL', () => {
    const r = parseFetchMarkdownUrl('http://example.com');
    expect(r.ok).toBe(true);
  });

  it('rejects a non-string input with an actionable error', () => {
    const r = parseFetchMarkdownUrl(undefined);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error).toContain('invalid URL');
      expect(r.error).toContain('http(s)');
    }
  });

  it('rejects the empty string', () => {
    const r = parseFetchMarkdownUrl('');
    expect(r.ok).toBe(false);
  });

  it('rejects an unparseable string', () => {
    const r = parseFetchMarkdownUrl('not a url');
    expect(r.ok).toBe(false);
  });

  it('rejects a file:// URL with a protocol-named error', () => {
    const r = parseFetchMarkdownUrl('file:///etc/passwd');
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toContain('file:');
  });

  it('rejects a ftp:// URL', () => {
    const r = parseFetchMarkdownUrl('ftp://example.com/file');
    expect(r.ok).toBe(false);
  });

  it('rejects a javascript: URL', () => {
    // The most important case: a smuggled `javascript:alert(1)` would
    // bail in snitchmd anyway, but we want the rejection to land at
    // the host boundary with a clear message rather than as an opaque
    // snitchmd exit code.
    const r = parseFetchMarkdownUrl('javascript:alert(1)');
    expect(r.ok).toBe(false);
  });

  it('rejects a number passed in place of a URL', () => {
    const r = parseFetchMarkdownUrl(123);
    expect(r.ok).toBe(false);
  });
});

describe('buildSnitchmdFlags', () => {
  it('always emits --json as the first flag', () => {
    const flags = buildSnitchmdFlags({});
    expect(flags[0]).toBe('--json');
  });

  it('returns only --json when no input fields are set', () => {
    expect(buildSnitchmdFlags({})).toEqual(['--json']);
  });

  it('emits --wait with the integer seconds when wait > 0', () => {
    expect(buildSnitchmdFlags({ wait: 5 })).toEqual(['--json', '--wait', '5']);
  });

  it('drops --wait when wait is 0', () => {
    // snitchmd's default is 0, so passing it through would be a no-op
    // but uglier in the docker invocation log. Drop explicitly.
    expect(buildSnitchmdFlags({ wait: 0 })).toEqual(['--json']);
  });

  it('drops --wait when wait is a non-integer', () => {
    expect(buildSnitchmdFlags({ wait: 1.5 })).toEqual(['--json']);
  });

  it('drops --wait when wait is a string (untrusted IPC input)', () => {
    expect(buildSnitchmdFlags({ wait: '5' })).toEqual(['--json']);
  });

  it('emits --wait-until for a known enum value', () => {
    expect(buildSnitchmdFlags({ waitUntil: 'networkidle' })).toEqual([
      '--json',
      '--wait-until',
      'networkidle',
    ]);
  });

  it('drops --wait-until for an unknown enum value', () => {
    // snitchmd would bail with exit-code-2 on an unknown value; we
    // prefer to drop silently and let snitchmd use its default
    // (`domcontentloaded`) than to surface an opaque exit-code as an
    // error to the agent.
    expect(buildSnitchmdFlags({ waitUntil: 'sometimes' })).toEqual(['--json']);
  });

  it('emits --wait-for-selector for a non-empty string', () => {
    expect(
      buildSnitchmdFlags({ waitForSelector: 'main .article-body' }),
    ).toEqual(['--json', '--wait-for-selector', 'main .article-body']);
  });

  it('drops --wait-for-selector for an empty string', () => {
    expect(buildSnitchmdFlags({ waitForSelector: '' })).toEqual(['--json']);
  });

  it('emits --favor-precision when favorPrecision is strictly true', () => {
    expect(buildSnitchmdFlags({ favorPrecision: true })).toEqual([
      '--json',
      '--favor-precision',
    ]);
  });

  it('drops --favor-precision when favorPrecision is truthy-but-not-true', () => {
    // Defensive: IPC payload from a buggy caller might pass `1` or
    // `"true"`. snitchmd's CLI is presence-based; we treat the flag
    // as a boolean strict-equal-to-true so a coerced value doesn't
    // silently enable precision mode and reframe the extracted body.
    expect(buildSnitchmdFlags({ favorPrecision: 1 })).toEqual(['--json']);
    expect(buildSnitchmdFlags({ favorPrecision: 'true' })).toEqual(['--json']);
  });

  it('emits --favor-recall (mutually exclusive with precision)', () => {
    expect(buildSnitchmdFlags({ favorRecall: true })).toEqual([
      '--json',
      '--favor-recall',
    ]);
  });

  it('honors both favor-precision and favor-recall flags (snitchmd will surface the conflict)', () => {
    // We don't enforce mutual exclusion here — snitchmd's CLI does,
    // exiting with code 2 and a clear error message. Re-implementing
    // the check at the host layer would just risk drift if snitchmd
    // changes the rule.
    expect(
      buildSnitchmdFlags({ favorPrecision: true, favorRecall: true }),
    ).toEqual(['--json', '--favor-precision', '--favor-recall']);
  });

  it('emits --include-links and --include-images independently', () => {
    expect(
      buildSnitchmdFlags({ includeLinks: true, includeImages: true }),
    ).toEqual(['--json', '--include-links', '--include-images']);
  });

  it('emits --max-chars with the integer value', () => {
    expect(buildSnitchmdFlags({ maxChars: 80000 })).toEqual([
      '--json',
      '--max-chars',
      '80000',
    ]);
  });

  it('drops --max-chars when value is zero or negative', () => {
    expect(buildSnitchmdFlags({ maxChars: 0 })).toEqual(['--json']);
    expect(buildSnitchmdFlags({ maxChars: -1 })).toEqual(['--json']);
  });

  it('emits --no-cache when noCache is true', () => {
    expect(buildSnitchmdFlags({ noCache: true })).toEqual([
      '--json',
      '--no-cache',
    ]);
  });

  it('emits --timeout with the integer seconds', () => {
    expect(buildSnitchmdFlags({ timeout: 60 })).toEqual([
      '--json',
      '--timeout',
      '60',
    ]);
  });

  it('emits all enabled flags together in a deterministic order', () => {
    // Pin the order so docker invocation logs stay greppable across
    // refactors. Order doesn't matter to snitchmd, but pinning it
    // makes the test output and operator audit-trail consistent.
    const flags = buildSnitchmdFlags({
      wait: 3,
      waitUntil: 'networkidle',
      waitForSelector: '#main',
      favorPrecision: true,
      includeLinks: true,
      includeImages: true,
      maxChars: 50000,
      noCache: true,
      timeout: 90,
    });
    expect(flags).toEqual([
      '--json',
      '--wait',
      '3',
      '--wait-until',
      'networkidle',
      '--wait-for-selector',
      '#main',
      '--favor-precision',
      '--include-links',
      '--include-images',
      '--max-chars',
      '50000',
      '--no-cache',
      '--timeout',
      '90',
    ]);
  });
});

describe('formatSnitchmdHeader', () => {
  it('emits the 3-line header followed by a blank line', () => {
    const header = formatSnitchmdHeader(
      {
        markdown: 'body',
        title: 'Hello',
        final_url: 'https://example.com/redirected',
        chars: 4,
        quality: 0.85,
      },
      'https://example.com/',
    );
    expect(header).toBe(
      '# Hello\n# source: https://example.com/redirected\n# chars: 4 quality: 0.85\n\n',
    );
  });

  it('falls back to the request URL when final_url is missing', () => {
    const header = formatSnitchmdHeader(
      { markdown: 'body', title: 'Hello', chars: 4, quality: null },
      'https://example.com/',
    );
    expect(header).toContain('# source: https://example.com/');
  });

  it('emits "(untitled)" when title is missing', () => {
    const header = formatSnitchmdHeader(
      { markdown: 'body', chars: 4, quality: null },
      'https://example.com/',
    );
    expect(header).toContain('# (untitled)');
  });

  it('omits the quality suffix when quality is null', () => {
    const header = formatSnitchmdHeader(
      {
        markdown: 'body',
        title: 'Hello',
        chars: 4,
        quality: null,
      },
      'https://example.com/',
    );
    expect(header).toBe(
      '# Hello\n# source: https://example.com/\n# chars: 4\n\n',
    );
  });

  it('omits the quality suffix when quality is undefined', () => {
    const header = formatSnitchmdHeader(
      { markdown: 'body', title: 'Hello', chars: 4 },
      'https://example.com/',
    );
    expect(header).not.toContain('quality:');
  });

  it('falls back to markdown.length when chars is missing', () => {
    const header = formatSnitchmdHeader(
      { markdown: 'hello world', title: 'Hi', quality: null },
      'https://example.com/',
    );
    expect(header).toContain('# chars: 11');
  });

  it('emits chars: 0 when both chars and markdown are missing', () => {
    const header = formatSnitchmdHeader(
      { title: 'Empty', quality: null },
      'https://example.com/',
    );
    expect(header).toContain('# chars: 0');
  });
});

describe('parseSnitchmdStdout', () => {
  const cleanPayload = JSON.stringify(
    {
      url: 'https://example.com',
      final_url: 'https://example.com/',
      title: 'Example Domain',
      page_type: 'service',
      quality: 0.8,
      chars: 113,
      markdown: 'This domain is for use in documentation examples.',
    },
    null,
    2,
  );

  it('fast-path parses a clean JSON stdout', () => {
    const r = parseSnitchmdStdout(cleanPayload);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.payload.title).toBe('Example Domain');
      expect(r.payload.chars).toBe(113);
    }
  });

  it('recovers JSON when CloakBrowser logs pollute stdout BEFORE the payload', () => {
    // This is the exact shape caught in production on 2026-05-21:
    // CloakBrowser writes "Downloading newer chromium" progress lines
    // to stdout while snitchmd writes its JSON afterward.
    const polluted =
      '[cloakbrowser] Newer Chromium available: 146.0.7680.177.5 (current: 146.0.7680.177.3). Downloading in background...\n' +
      '[cloakbrowser] Downloading from https://cloakbrowser.dev/chromium-v146.0.7680.177.5/cloakbrowser-linux-x64.tar.gz\n' +
      '[cloakbrowser] Download progress: 9% (19/206 MB)\n' +
      '[cloakbrowser] Download progress: 19% (39/206 MB)\n' +
      cleanPayload;
    const r = parseSnitchmdStdout(polluted);
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.payload.title).toBe('Example Domain');
  });

  it('recovers JSON when noise appears AFTER the payload too', () => {
    // Defensive: a future CloakBrowser tick could land after snitchmd
    // emits its JSON but before the process exits.
    const polluted =
      cleanPayload + '\n[cloakbrowser] Download progress: 100% (206/206 MB)\n';
    const r = parseSnitchmdStdout(polluted);
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.payload.chars).toBe(113);
  });

  it('returns ok=false when stdout has no JSON object', () => {
    const r = parseSnitchmdStdout('garbage with no braces at all');
    expect(r.ok).toBe(false);
  });

  it('returns ok=false when stdout is empty', () => {
    const r = parseSnitchmdStdout('');
    expect(r.ok).toBe(false);
  });

  it('returns ok=false when the slice between braces is still malformed', () => {
    // Two `{` and `}` chars but the inner content isn't JSON. The
    // slice-from-first-{ -to-last-} catches `{ this } { is malformed }`
    // and re-parses it; that parse fails too, so we return ok=false
    // rather than smuggling garbage through.
    const r = parseSnitchmdStdout('preamble { not really json } trailing');
    expect(r.ok).toBe(false);
  });

  it('propagates non-SyntaxError exceptions on the fast path', () => {
    // Defensive: if a future custom JSON.parse override throws something
    // other than SyntaxError (programming bug), we should NOT swallow it
    // and silently fall through to the slice retry.
    const original = JSON.parse;
    const probe = new RangeError('not a parse failure');
    (globalThis as { JSON: typeof JSON }).JSON.parse = () => {
      throw probe;
    };
    try {
      expect(() => parseSnitchmdStdout(cleanPayload)).toThrow(probe);
    } finally {
      (globalThis as { JSON: typeof JSON }).JSON.parse = original;
    }
  });
});
