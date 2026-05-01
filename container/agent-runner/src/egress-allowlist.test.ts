import { describe, it, expect } from 'vitest';
import {
  bashTargetsAllowlist,
  classifySink,
  decideEgress,
  EgressAllowlist,
  extractDestinations,
  loadEgressAllowlist,
  pathTargetsAllowlist,
  splitRecipientList,
} from './egress-allowlist.js';

// ---- classifySink ----

describe('classifySink', () => {
  it('matches Composio Gmail send variants', () => {
    expect(classifySink('mcp__composio__gmail_send_email')).toBe('gmail_send');
    expect(classifySink('mcp__composio__gmail_reply_email')).toBe('gmail_send');
    expect(classifySink('mcp__composio__gmail_send_draft')).toBe('gmail_send');
  });

  it('matches Composio Slack send/post variants', () => {
    expect(classifySink('mcp__composio__slack_post_message')).toBe('slack_post');
    expect(classifySink('mcp__composio__slack_send_dm')).toBe('slack_post');
  });

  it('matches send_message_to_chat exactly', () => {
    expect(classifySink('mcp__nanoclaw__send_message_to_chat')).toBe(
      'send_message_to_chat',
    );
  });

  it('returns null for non-egress tools', () => {
    expect(classifySink('mcp__composio__gmail_fetch_emails')).toBeNull();
    expect(classifySink('mcp__composio__slack_list_messages')).toBeNull();
    expect(classifySink('mcp__nanoclaw__send_message')).toBeNull();
    expect(classifySink('Read')).toBeNull();
    expect(classifySink('WebFetch')).toBeNull();
  });

  it('returns null for empty / non-string inputs', () => {
    expect(classifySink('')).toBeNull();
    expect(classifySink(undefined as unknown as string)).toBeNull();
  });
});

// ---- extractDestinations ----

describe('splitRecipientList', () => {
  it('returns a single-element list when no separator is present', () => {
    expect(splitRecipientList('a@x.io')).toEqual(['a@x.io']);
  });

  it('splits on commas, trimming whitespace', () => {
    expect(splitRecipientList('a@x.io, b@y.io ,  c@z.io')).toEqual([
      'a@x.io',
      'b@y.io',
      'c@z.io',
    ]);
  });

  it('splits on semicolons too', () => {
    expect(splitRecipientList('a@x.io;b@y.io')).toEqual(['a@x.io', 'b@y.io']);
  });

  it('drops empty entries (trailing commas, double separators)', () => {
    expect(splitRecipientList('a@x.io,,b@y.io,')).toEqual(['a@x.io', 'b@y.io']);
  });

  it('returns empty for empty / non-string input', () => {
    expect(splitRecipientList('')).toEqual([]);
    expect(splitRecipientList(undefined as unknown as string)).toEqual([]);
  });
});

describe('extractDestinations', () => {
  it('extracts gmail recipient from common fields', () => {
    expect(
      extractDestinations('gmail_send', { recipient: 'a@x.io' }),
    ).toEqual(['a@x.io']);
    expect(extractDestinations('gmail_send', { to: 'b@x.io' })).toEqual([
      'b@x.io',
    ]);
    expect(
      extractDestinations('gmail_send', { recipient_email: 'c@x.io' }),
    ).toEqual(['c@x.io']);
  });

  it('extracts gmail recipients from arrays', () => {
    expect(
      extractDestinations('gmail_send', {
        recipients: ['a@x.io', 'b@x.io'],
      }),
    ).toEqual(['a@x.io', 'b@x.io']);
  });

  it('extracts slack channel', () => {
    expect(extractDestinations('slack_post', { channel: '#general' })).toEqual([
      '#general',
    ]);
    expect(extractDestinations('slack_post', { channel: 'C012ABC' })).toEqual([
      'C012ABC',
    ]);
  });

  it('extracts send_message_to_chat jid from chat_id or chat_jid', () => {
    expect(
      extractDestinations('send_message_to_chat', {
        chat_jid: 'tg:-1003869886477',
      }),
    ).toEqual(['tg:-1003869886477']);
    expect(
      extractDestinations('send_message_to_chat', {
        chat_id: 'tg:1234',
      }),
    ).toEqual(['tg:1234']);
  });

  it('returns empty for missing or non-string fields', () => {
    expect(extractDestinations('gmail_send', {})).toEqual([]);
    expect(extractDestinations('slack_post', { channel: 42 })).toEqual([]);
    expect(extractDestinations('send_message_to_chat', null)).toEqual([]);
  });
});

// ---- loadEgressAllowlist ----

describe('loadEgressAllowlist', () => {
  it('returns null when file missing', () => {
    const fs = {
      existsSync: () => false,
      readFileSync: () => '',
    };
    expect(loadEgressAllowlist(fs, '/x')).toBeNull();
  });

  it('returns null when file is empty', () => {
    const fs = {
      existsSync: () => true,
      readFileSync: () => '   \n  ',
    };
    expect(loadEgressAllowlist(fs, '/x')).toBeNull();
  });

  it('parses a populated allowlist', () => {
    const json = JSON.stringify({
      gmail_send: { allowed_recipients: ['a@x.io'] },
      slack_post: { allowed_channels: ['#general'] },
    });
    const fs = {
      existsSync: () => true,
      readFileSync: () => json,
    };
    expect(loadEgressAllowlist(fs, '/x')).toEqual({
      gmail_send: { allowed_recipients: ['a@x.io'] },
      slack_post: { allowed_channels: ['#general'] },
    });
  });

  it('throws on malformed JSON', () => {
    const fs = {
      existsSync: () => true,
      readFileSync: () => '{ not json',
    };
    expect(() => loadEgressAllowlist(fs, '/x')).toThrow();
  });

  it('throws when JSON is not an object', () => {
    const fs = {
      existsSync: () => true,
      readFileSync: () => '"a string"',
    };
    expect(() => loadEgressAllowlist(fs, '/x')).toThrow(/object/);
  });

  it('drops invalid per-field shapes with a warning, keeps the valid ones', () => {
    // operator typo: `allowed_channels` is a string instead of an
    // array. Should be dropped silently (with a warn) rather than
    // crashing every outbound call.
    const json = JSON.stringify({
      gmail_send: {
        allowed_recipients: ['ok@x.io'],
        allowed_domains: 'sadogursky.com', // bad shape
      },
      slack_post: { allowed_channels: '#general' }, // bad shape
      send_message_to_chat: { allowed_chat_jids: ['tg:1', 42, 'tg:2'] },
    });
    const fs = {
      existsSync: () => true,
      readFileSync: () => json,
    };
    const out = loadEgressAllowlist(fs, '/x');
    expect(out).toEqual({
      gmail_send: { allowed_recipients: ['ok@x.io'] },
      // allowed_domains dropped, gmail_send entry kept
      // slack_post dropped entirely (no valid keys remained)
      send_message_to_chat: { allowed_chat_jids: ['tg:1', 'tg:2'] },
      // 42 filtered out
    });
  });

  it('coerces enforce_for_operator to boolean only', () => {
    const json = JSON.stringify({
      enforce_for_operator: 'yes', // bad shape; should drop
    });
    const fs = {
      existsSync: () => true,
      readFileSync: () => json,
    };
    expect(loadEgressAllowlist(fs, '/x')).toEqual({});
  });

  it('preserves enforce_for_operator: true', () => {
    const json = JSON.stringify({ enforce_for_operator: true });
    const fs = {
      existsSync: () => true,
      readFileSync: () => json,
    };
    expect(loadEgressAllowlist(fs, '/x')).toEqual({
      enforce_for_operator: true,
    });
  });
});

// ---- decideEgress (acceptance scenarios from #320) ----

const FULL_ALLOWLIST: EgressAllowlist = {
  gmail_send: {
    allowed_recipients: ['jbaruch@sadogursky.com'],
    allowed_domains: ['sadogursky.com', 'tessl.io'],
  },
  slack_post: {
    allowed_channels: ['#general', '#tessl-internal', 'C012ABC'],
  },
  send_message_to_chat: {
    allowed_chat_jids: ['tg:-100111', 'tg:-100222'],
  },
};

describe('decideEgress — pass (non-gated)', () => {
  it('passes for tools we do not gate', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_fetch_emails',
      toolInput: {},
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('pass');
  });

  it('passes for built-in tools', () => {
    const d = decideEgress({
      toolName: 'WebFetch',
      toolInput: { url: 'https://x' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('pass');
  });
});

describe('decideEgress — operator bypass', () => {
  it('allows gmail.send to a brand-new recipient when chain is operator-trusted', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { recipient: 'stranger@elsewhere.example' },
      hasUntrustedProvenance: false,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('allow');
    if (d.kind === 'allow') expect(d.reason).toMatch(/operator/i);
  });

  it('still allows when allowlist is missing entirely (default-bypass)', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { recipient: 'stranger@elsewhere.example' },
      hasUntrustedProvenance: false,
      allowlist: null,
    });
    expect(d.kind).toBe('allow');
  });
});

describe('decideEgress — operator opt-in tightening', () => {
  it('enforces the allowlist for operator chains when enforce_for_operator: true', () => {
    const tightened: EgressAllowlist = {
      ...FULL_ALLOWLIST,
      enforce_for_operator: true,
    };
    const allowed = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { recipient: 'jbaruch@sadogursky.com' },
      hasUntrustedProvenance: false,
      allowlist: tightened,
    });
    expect(allowed.kind).toBe('allow');
    const denied = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { recipient: 'stranger@elsewhere.example' },
      hasUntrustedProvenance: false,
      allowlist: tightened,
    });
    expect(denied.kind).toBe('deny');
  });
});

describe('decideEgress — untrusted-provenance gate (gmail)', () => {
  it('allows when recipient matches allowed_recipients', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { recipient: 'jbaruch@sadogursky.com' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('allow');
  });

  it('allows when recipient is in an allowed domain', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { recipient: 'someone@tessl.io' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('allow');
  });

  it('denies a non-allowlisted recipient (web-injection scenario)', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { recipient: 'attacker@evil.example' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') {
      expect(d.sink).toBe('gmail_send');
      expect(d.destination).toBe('attacker@evil.example');
      expect(d.reason).toMatch(/aye-confirm/);
    }
  });

  it('denies when ANY recipient in a multi-recipient call is unallowed', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: {
        recipients: ['jbaruch@sadogursky.com', 'attacker@evil.example'],
      },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') expect(d.destination).toBe('attacker@evil.example');
  });

  it('denies a comma-separated string recipient with mixed allowed+disallowed', () => {
    // Pre-fix: the combined string went through `endsWith` once and
    // matched the LAST address's `@allowed-domain` suffix, leaking
    // the earlier addresses past the gate. Post-fix: split + per-
    // entry validation; the disallowed entry trips deny.
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { to: 'attacker@evil.example, jbaruch@sadogursky.com' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') expect(d.destination).toBe('attacker@evil.example');
  });

  it('denies a semicolon-separated string recipient with mixed allowed+disallowed', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: {
        recipient: 'jbaruch@sadogursky.com; attacker@evil.example',
      },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') expect(d.destination).toBe('attacker@evil.example');
  });

  it('allows a comma-separated string when ALL recipients pass', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { to: 'jbaruch@sadogursky.com, foo@tessl.io' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('allow');
  });

  it('handles split-inside-array form (an array element is comma-list)', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: {
        recipients: [
          'jbaruch@sadogursky.com',
          'foo@tessl.io, attacker@evil.example',
        ],
      },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') expect(d.destination).toBe('attacker@evil.example');
  });

  it('denies when allowlist is null under untrusted provenance', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { recipient: 'someone@tessl.io' },
      hasUntrustedProvenance: true,
      allowlist: null,
    });
    expect(d.kind).toBe('deny');
  });

  it('domain match is case-insensitive', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { recipient: 'foo@TESSL.IO' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('allow');
  });
});

describe('decideEgress — untrusted-provenance gate (slack)', () => {
  it('allows an allowlisted channel name', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__slack_post_message',
      toolInput: { channel: '#general' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('allow');
  });

  it('denies a non-allowlisted channel', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__slack_post_message',
      toolInput: { channel: '#secrets' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('deny');
  });
});

describe('decideEgress — untrusted-provenance gate (send_message_to_chat)', () => {
  it('allows an allowlisted JID', () => {
    const d = decideEgress({
      toolName: 'mcp__nanoclaw__send_message_to_chat',
      toolInput: { chat_jid: 'tg:-100111' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('allow');
  });

  it('denies a non-allowlisted JID (cross-group injection)', () => {
    const d = decideEgress({
      toolName: 'mcp__nanoclaw__send_message_to_chat',
      toolInput: { chat_jid: 'tg:-100999' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('deny');
  });

  it('accepts chat_id field as alias for chat_jid', () => {
    const d = decideEgress({
      toolName: 'mcp__nanoclaw__send_message_to_chat',
      toolInput: { chat_id: 'tg:-100222' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('allow');
  });
});

describe('decideEgress — missing destination', () => {
  it('denies (with structured reason) when destination cannot be extracted', () => {
    const d = decideEgress({
      toolName: 'mcp__composio__gmail_send_email',
      toolInput: { subject: 'no recipient field' },
      hasUntrustedProvenance: true,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') expect(d.destination).toBe('<missing>');
  });
});

// ---- write-gate matchers (path equivalence + bash obfuscation) ----

const ALLOWLIST_PATH = '/workspace/trusted/egress_allowlist.json';

describe('pathTargetsAllowlist', () => {
  it('matches the absolute path exactly', () => {
    expect(pathTargetsAllowlist(ALLOWLIST_PATH, ALLOWLIST_PATH)).toBe(true);
  });

  it('matches by basename when the candidate is a relative file name', () => {
    // Bypass attempt: model passes 'egress_allowlist.json' with cwd
    // already at /workspace/trusted (Write tool would resolve relative
    // to cwd). Basename-equality catches this.
    expect(pathTargetsAllowlist('egress_allowlist.json', ALLOWLIST_PATH)).toBe(
      true,
    );
    expect(
      pathTargetsAllowlist('./egress_allowlist.json', ALLOWLIST_PATH),
    ).toBe(true);
  });

  it('matches a normalized absolute path with traversal segments', () => {
    // Bypass attempt: '/workspace/trusted/sub/../egress_allowlist.json'
    // collapses to the target.
    expect(
      pathTargetsAllowlist(
        '/workspace/trusted/sub/../egress_allowlist.json',
        ALLOWLIST_PATH,
      ),
    ).toBe(true);
    expect(
      pathTargetsAllowlist(
        '/workspace/trusted/./egress_allowlist.json',
        ALLOWLIST_PATH,
      ),
    ).toBe(true);
  });

  it('matches via realpath equivalence when supplied', () => {
    // Bypass attempt: a symlink at /tmp/foo points at the allowlist
    // file. The default realpath helper returns null (so this layer
    // is opt-in and tests need not stub fs); when supplied, it
    // matches.
    const fakeRealpath = (p: string) => {
      if (p === '/tmp/symlink-pointing-at-allowlist') return ALLOWLIST_PATH;
      if (p === ALLOWLIST_PATH) return ALLOWLIST_PATH;
      return null;
    };
    expect(
      pathTargetsAllowlist(
        '/tmp/symlink-pointing-at-allowlist',
        ALLOWLIST_PATH,
        fakeRealpath,
      ),
    ).toBe(true);
  });

  it('returns false for unrelated absolute paths', () => {
    expect(pathTargetsAllowlist('/etc/hosts', ALLOWLIST_PATH)).toBe(false);
    expect(
      pathTargetsAllowlist('/workspace/trusted/other.json', ALLOWLIST_PATH),
    ).toBe(false);
  });

  it('returns false for empty / non-string', () => {
    expect(pathTargetsAllowlist('', ALLOWLIST_PATH)).toBe(false);
    expect(
      pathTargetsAllowlist(undefined as unknown as string, ALLOWLIST_PATH),
    ).toBe(false);
  });
});

describe('bashTargetsAllowlist', () => {
  it('matches the absolute path appearing literally in the command', () => {
    expect(
      bashTargetsAllowlist(
        `cat ${ALLOWLIST_PATH}`,
        ALLOWLIST_PATH,
      ),
    ).toBe(true);
    expect(
      bashTargetsAllowlist(
        `echo '...' > ${ALLOWLIST_PATH}`,
        ALLOWLIST_PATH,
      ),
    ).toBe(true);
  });

  it('matches obfuscated cd-then-redirect (basename appears bare)', () => {
    // The fix the policy reviewer flagged: pre-fix this command
    // never mentioned the absolute path, so `cmd.includes(...)` was
    // false. Basename match catches it.
    expect(
      bashTargetsAllowlist(
        'cd /workspace/trusted && echo "{}" > egress_allowlist.json',
        ALLOWLIST_PATH,
      ),
    ).toBe(true);
  });

  it('matches a heredoc that names the basename', () => {
    expect(
      bashTargetsAllowlist(
        "cat > egress_allowlist.json <<'JSON'\n{}\nJSON",
        ALLOWLIST_PATH,
      ),
    ).toBe(true);
  });

  it('returns false for unrelated commands', () => {
    expect(
      bashTargetsAllowlist('ls /workspace/group', ALLOWLIST_PATH),
    ).toBe(false);
    expect(
      bashTargetsAllowlist('npm install', ALLOWLIST_PATH),
    ).toBe(false);
  });

  it('returns false for empty / non-string', () => {
    expect(bashTargetsAllowlist('', ALLOWLIST_PATH)).toBe(false);
    expect(
      bashTargetsAllowlist(undefined as unknown as string, ALLOWLIST_PATH),
    ).toBe(false);
  });
});
