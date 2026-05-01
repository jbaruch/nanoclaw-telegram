import { describe, it, expect } from 'vitest';
import { validateComposioArgs } from './composio-arg-validator.js';

describe('validateComposioArgs — gmail header injection', () => {
  it('denies subject containing newline + Bcc:', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'Hi\n\nBcc: attacker@evil.com',
      body: 'hello',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('subject');
      expect(decision.violation).toBe('newlines_in_header');
    }
  });

  it('denies subject containing carriage return', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'Reply\r\nX-Spoof: yes',
      body: 'hello',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.violation).toBe('newlines_in_header');
    }
  });

  it('denies recipient_email containing newline', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io\nBcc: b@y.io',
      subject: 'hi',
      body: 'hello',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('recipient_email');
    }
  });

  it('denies cc array element containing newline', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      cc: ['ok@x.io', 'bad@y.io\nBcc: c@z.io'],
      subject: 'hi',
      body: 'hello',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('cc[1]');
      expect(decision.violation).toBe('newlines_in_header');
    }
  });

  it('passes through gmail_reply_email even though it has no subject field', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_reply_email', {
      thread_id: 'abc',
      recipient_email: 'a@x.io',
      body: 'thanks',
    });
    expect(decision.kind).toBe('allow');
  });

  it('denies on gmail_reply_email when bcc has newline', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_reply_email', {
      thread_id: 'abc',
      recipient_email: 'a@x.io',
      bcc: 'shadow@x.io\nReply-To: spoof@y.io',
      body: 'hi',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('bcc');
    }
  });
});

describe('validateComposioArgs — gmail body cap', () => {
  it('denies body exceeding 100 KB', () => {
    const big = 'x'.repeat(100_001);
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'hi',
      body: big,
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('body');
      expect(decision.violation).toBe('body_too_long');
    }
  });

  it('measures bytes, not characters (multi-byte chars count fully)', () => {
    // Each '😀' is 4 bytes in UTF-8. 25_001 of them = 100_004 bytes,
    // 1 byte over the 100_000 cap, so deny — even though the char
    // count (25_001) is well under any naive char-based limit.
    const big = '😀'.repeat(25_001);
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'hi',
      body: big,
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.violation).toBe('body_too_long');
    }
  });

  it('allows body just under the cap', () => {
    const ok = 'x'.repeat(99_999);
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'hi',
      body: ok,
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('validateComposioArgs — gmail body control chars', () => {
  it('denies body with NUL byte', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'hi',
      body: 'hello\x00world',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.violation).toBe('control_chars_in_body');
    }
  });

  it('denies body with bare carriage return (CRLF spoof guard)', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'hi',
      body: 'hello\rworld',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.violation).toBe('control_chars_in_body');
    }
  });

  it('allows body with newlines and tabs', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'hi',
      body: 'line1\nline2\tindented\n',
    });
    expect(decision.kind).toBe('allow');
  });

  it('denies body with DEL (0x7F)', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'hi',
      body: 'hello\x7Fworld',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.violation).toBe('control_chars_in_body');
    }
  });
});

describe('validateComposioArgs — slack', () => {
  it('denies slack_post with text length 200 KB', () => {
    const big = 'a'.repeat(200_000);
    const decision = validateComposioArgs('mcp__composio__slack_post_message', {
      channel: '#general',
      text: big,
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('text');
      expect(decision.violation).toBe('body_too_long');
    }
  });

  it('denies slack channel containing newline', () => {
    const decision = validateComposioArgs('mcp__composio__slack_post_message', {
      channel: '#general\n#leak',
      text: 'hi',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('channel');
      expect(decision.violation).toBe('newlines_in_header');
    }
  });

  it('denies slack_send_dm with newline in user', () => {
    const decision = validateComposioArgs('mcp__composio__slack_send_dm', {
      user: 'U123\nSpoof',
      text: 'hi',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('user');
    }
  });

  it('allows well-formed slack_post', () => {
    const decision = validateComposioArgs('mcp__composio__slack_post_message', {
      channel: '#general',
      text: 'normal message with a\nnewline and a\ttab',
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('validateComposioArgs — header byte cap', () => {
  it('denies subject exceeding RFC 5322 998-byte line limit', () => {
    const long = 'A'.repeat(999);
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: long,
      body: 'hi',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('subject');
      expect(decision.violation).toBe('header_too_long');
    }
  });

  it('allows subject at the 998-byte boundary', () => {
    const ok = 'A'.repeat(998);
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: ok,
      body: 'hi',
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('validateComposioArgs — array smuggling guard on non-list fields', () => {
  it('denies body as an array (would bypass per-element byte cap)', () => {
    // Without the mayBeArray guard, two 99 KB elements would each pass
    // the per-element 100 KB cap while shipping ~200 KB total payload.
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'hi',
      body: ['a'.repeat(99_000), 'b'.repeat(99_000)],
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('body');
      expect(decision.violation).toBe('wrong_type');
    }
  });

  it('denies subject as an array', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: ['Status', 'Update'],
      body: 'hi',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('subject');
      expect(decision.violation).toBe('wrong_type');
    }
  });

  it('denies slack text as an array', () => {
    const decision = validateComposioArgs('mcp__composio__slack_post_message', {
      channel: '#general',
      text: ['line1', 'line2'],
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('text');
      expect(decision.violation).toBe('wrong_type');
    }
  });

  it('denies slack channel as an array', () => {
    const decision = validateComposioArgs('mcp__composio__slack_post_message', {
      channel: ['#general', '#leak'],
      text: 'hi',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.field).toBe('channel');
      expect(decision.violation).toBe('wrong_type');
    }
  });

  it('still ALLOWS recipient_email as an array (legitimate Composio shape)', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: ['a@x.io', 'b@y.io'],
      subject: 'hi',
      body: 'hi',
    });
    expect(decision.kind).toBe('allow');
  });

  it('still ALLOWS to as an array (legitimate Composio shape)', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      to: ['b@y.io', 'c@z.io'],
      subject: 'hi',
      body: 'hi',
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('validateComposioArgs — type checks', () => {
  it('denies subject as a number', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 42,
      body: 'hi',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.violation).toBe('wrong_type');
    }
  });

  it('denies cc element as a number', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      cc: ['ok@x.io', 99],
      subject: 'hi',
      body: 'hi',
    });
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.violation).toBe('wrong_type');
    }
  });

  it('skips fields that are absent (undefined / null)', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'hi',
      body: 'ok',
      cc: undefined,
      bcc: null,
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('validateComposioArgs — well-formed pass-through', () => {
  it('allows a typical gmail send', () => {
    const decision = validateComposioArgs('mcp__composio__gmail_send_email', {
      recipient_email: 'a@x.io',
      subject: 'Status update',
      body: 'Things look good.\n\nBest,\nAyeAye',
      cc: ['team@x.io'],
    });
    expect(decision.kind).toBe('allow');
  });

  it('allows a typical slack post with multi-line text', () => {
    const decision = validateComposioArgs('mcp__composio__slack_post_message', {
      channel: '#general',
      text: 'PR #42 is green.\nLink: https://github.com/owner/repo/pull/42',
    });
    expect(decision.kind).toBe('allow');
  });
});

describe('validateComposioArgs — out-of-scope tools pass through', () => {
  it('allows non-Composio tools', () => {
    expect(validateComposioArgs('Read', { file_path: '/tmp' }).kind).toBe('allow');
    expect(validateComposioArgs('Bash', { command: 'ls' }).kind).toBe('allow');
    expect(validateComposioArgs('mcp__nanoclaw__send_message', { text: 'hi' }).kind).toBe('allow');
  });

  it('allows Composio read tools (no header / body surface)', () => {
    expect(
      validateComposioArgs('mcp__composio__gmail_fetch_emails', {
        query: 'is:unread',
      }).kind,
    ).toBe('allow');
    expect(
      validateComposioArgs('mcp__composio__slack_fetch_history', {
        channel: '#general',
      }).kind,
    ).toBe('allow');
  });

  it('allows Composio github / calendar mutations (no rule row yet)', () => {
    // Out of #326 v1 scope — github_create_issue and
    // googlecalendar_create_event don't have header-injection
    // surfaces the way mail does. Add a row when a real risk
    // surfaces.
    expect(
      validateComposioArgs('mcp__composio__github_create_issue', {
        title: 'multi\nline title',
        body: 'a',
      }).kind,
    ).toBe('allow');
  });

  it('allows when toolInput is empty / non-object', () => {
    expect(
      validateComposioArgs('mcp__composio__gmail_send_email', null as unknown as object).kind,
    ).toBe('allow');
    expect(
      validateComposioArgs('mcp__composio__gmail_send_email', undefined as unknown as object).kind,
    ).toBe('allow');
    expect(
      validateComposioArgs('mcp__composio__gmail_send_email', 'not-an-object' as unknown as object).kind,
    ).toBe('allow');
  });

  it('allows when toolName is empty', () => {
    expect(validateComposioArgs('', { subject: 'a\nb' }).kind).toBe('allow');
  });
});
