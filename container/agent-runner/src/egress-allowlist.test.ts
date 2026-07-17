import { describe, it, expect } from 'vitest';
import {
  bashTargetsAllowlist,
  classifySink,
  decideEgress,
  EgressAllowlist,
  extractDestinations,
  loadEgressAllowlist,
  pathTargetsAllowlist,
} from './egress-allowlist.js';

// ---- classifySink ----

describe('classifySink', () => {
  it('matches send_message_to_chat exactly', () => {
    expect(classifySink('mcp__nanoclaw__send_message_to_chat')).toBe(
      'send_message_to_chat',
    );
  });

  it('returns null for non-egress tools', () => {
    expect(classifySink('mcp__tessl__search')).toBeNull();
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

describe('extractDestinations', () => {
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
    expect(extractDestinations('send_message_to_chat', {})).toEqual([]);
    expect(
      extractDestinations('send_message_to_chat', { chat_jid: 42 }),
    ).toEqual([]);
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
      send_message_to_chat: { allowed_chat_jids: ['tg:-100111'] },
    });
    const fs = {
      existsSync: () => true,
      readFileSync: () => json,
    };
    expect(loadEgressAllowlist(fs, '/x')).toEqual({
      send_message_to_chat: { allowed_chat_jids: ['tg:-100111'] },
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
    // operator typo: a non-string entry inside `allowed_chat_jids`.
    // Should be dropped silently (with a warn) rather than crashing
    // every outbound call.
    const json = JSON.stringify({
      send_message_to_chat: { allowed_chat_jids: ['tg:1', 42, 'tg:2'] },
    });
    const fs = {
      existsSync: () => true,
      readFileSync: () => json,
    };
    const out = loadEgressAllowlist(fs, '/x');
    expect(out).toEqual({
      // 42 filtered out
      send_message_to_chat: { allowed_chat_jids: ['tg:1', 'tg:2'] },
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
  send_message_to_chat: {
    allowed_chat_jids: ['tg:-100111', 'tg:-100222'],
  },
};

describe('decideEgress — pass (non-gated)', () => {
  it('passes for tools we do not gate', () => {
    const d = decideEgress({
      toolName: 'mcp__tessl__search',
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
  it('allows a send to a brand-new JID when chain is operator-trusted', () => {
    const d = decideEgress({
      toolName: 'mcp__nanoclaw__send_message_to_chat',
      toolInput: { chat_jid: 'tg:-100999' },
      hasUntrustedProvenance: false,
      allowlist: FULL_ALLOWLIST,
    });
    expect(d.kind).toBe('allow');
    if (d.kind === 'allow') expect(d.reason).toMatch(/operator/i);
  });

  it('still allows when allowlist is missing entirely (default-bypass)', () => {
    const d = decideEgress({
      toolName: 'mcp__nanoclaw__send_message_to_chat',
      toolInput: { chat_jid: 'tg:-100999' },
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
      toolName: 'mcp__nanoclaw__send_message_to_chat',
      toolInput: { chat_jid: 'tg:-100111' },
      hasUntrustedProvenance: false,
      allowlist: tightened,
    });
    expect(allowed.kind).toBe('allow');
    const denied = decideEgress({
      toolName: 'mcp__nanoclaw__send_message_to_chat',
      toolInput: { chat_jid: 'tg:-100999' },
      hasUntrustedProvenance: false,
      allowlist: tightened,
    });
    expect(denied.kind).toBe('deny');
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
      toolName: 'mcp__nanoclaw__send_message_to_chat',
      toolInput: { text: 'no chat_jid field' },
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
    expect(bashTargetsAllowlist(`cat ${ALLOWLIST_PATH}`, ALLOWLIST_PATH)).toBe(
      true,
    );
    expect(
      bashTargetsAllowlist(`echo '...' > ${ALLOWLIST_PATH}`, ALLOWLIST_PATH),
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
    expect(bashTargetsAllowlist('ls /workspace/group', ALLOWLIST_PATH)).toBe(
      false,
    );
    expect(bashTargetsAllowlist('npm install', ALLOWLIST_PATH)).toBe(false);
  });

  it('returns false for empty / non-string', () => {
    expect(bashTargetsAllowlist('', ALLOWLIST_PATH)).toBe(false);
    expect(
      bashTargetsAllowlist(undefined as unknown as string, ALLOWLIST_PATH),
    ).toBe(false);
  });
});
