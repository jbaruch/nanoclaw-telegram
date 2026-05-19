import { describe, it, expect } from 'vitest';
import {
  classifyDestructiveOp,
  ConfirmationToken,
  decideConfirmation,
  findValidToken,
  loadConfirmationTokens,
  markTokenUsed,
  parseDuration,
} from './confirmation-tokens.js';

const NOW = '2026-05-01T12:00:00.000Z';
const FUTURE = '2026-05-01T12:05:00.000Z';
const PAST = '2026-05-01T11:55:00.000Z';

function token(over: Partial<ConfirmationToken> = {}): ConfirmationToken {
  return {
    token: 'abc123',
    scope: 'nuke_session',
    reason: 'stuck container',
    issued_at: NOW,
    expires_at: FUTURE,
    used: false,
    ...over,
  };
}

// ---- classifyDestructiveOp ----

describe('classifyDestructiveOp', () => {
  it('classifies the explicit destructive MCP tools', () => {
    expect(classifyDestructiveOp('mcp__nanoclaw__nuke_session', {})).toEqual({
      scope: 'nuke_session',
      label: expect.any(String),
    });
    expect(classifyDestructiveOp('mcp__nanoclaw__github_backup', {})).toEqual({
      scope: 'github_backup',
      label: expect.any(String),
    });
    expect(classifyDestructiveOp('mcp__nanoclaw__set_trusted', {})).toEqual({
      scope: 'set_trusted',
      label: expect.any(String),
    });
    expect(
      classifyDestructiveOp('mcp__nanoclaw__push_staged_to_branch', {}),
    ).toEqual({ scope: 'push_staged_to_branch', label: expect.any(String) });
    expect(
      classifyDestructiveOp('mcp__nanoclaw__set_agent_model', {}),
    ).toEqual({ scope: 'set_agent_model', label: expect.any(String) });
    expect(
      classifyDestructiveOp(
        'mcp__nanoclaw__set_maintenance_agent_model',
        {},
      ),
    ).toEqual({
      scope: 'set_maintenance_agent_model',
      label: expect.any(String),
    });
    expect(
      classifyDestructiveOp('mcp__nanoclaw__set_task_agent_model', {}),
    ).toEqual({ scope: 'set_task_agent_model', label: expect.any(String) });
  });

  it('gates the three AGENT_MODEL set_* tools through decideConfirmation when chain is untrusted (#595)', () => {
    for (const toolName of [
      'mcp__nanoclaw__set_agent_model',
      'mcp__nanoclaw__set_maintenance_agent_model',
      'mcp__nanoclaw__set_task_agent_model',
    ]) {
      const denied = decideConfirmation({
        toolName,
        toolInput: {},
        hasUntrustedProvenance: true,
        tokens: [],
        nowIso: NOW,
      });
      expect(denied.kind).toBe('deny');
      if (denied.kind === 'deny') {
        expect(denied.scope).toBe(toolName.replace('mcp__nanoclaw__', ''));
      }

      const passedOnTrustedChain = decideConfirmation({
        toolName,
        toolInput: {},
        hasUntrustedProvenance: false,
        tokens: [],
        nowIso: NOW,
      });
      expect(passedOnTrustedChain.kind).toBe('allow');
    }
  });

  it('returns null for non-destructive tools', () => {
    expect(classifyDestructiveOp('mcp__nanoclaw__send_message', {})).toBeNull();
    expect(classifyDestructiveOp('Read', {})).toBeNull();
    expect(classifyDestructiveOp('Bash', {})).toBeNull();
    expect(
      classifyDestructiveOp('mcp__composio__gmail_send_email', {}),
    ).toBeNull();
  });

  it('returns null for empty / non-string tool names', () => {
    expect(classifyDestructiveOp('', {})).toBeNull();
    expect(classifyDestructiveOp(undefined as unknown as string, {})).toBeNull();
  });
});

// ---- findValidToken ----

describe('findValidToken', () => {
  it('returns the first unspent unexpired matching token', () => {
    const t1 = token({ token: 'a' });
    const t2 = token({ token: 'b' });
    const result = findValidToken([t1, t2], 'nuke_session', NOW);
    expect(result.token?.token).toBe('a');
  });

  it('returns null with no_tokens_for_scope when scope absent', () => {
    const result = findValidToken([token()], 'github_backup', NOW);
    expect(result.token).toBeNull();
    expect(result.reason).toBe('no_tokens_for_scope');
  });

  it('skips used tokens', () => {
    const result = findValidToken(
      [token({ used: true })],
      'nuke_session',
      NOW,
    );
    expect(result.token).toBeNull();
    expect(result.reason).toBe('all_tokens_used');
  });

  it('skips expired tokens', () => {
    const result = findValidToken(
      [token({ expires_at: PAST })],
      'nuke_session',
      NOW,
    );
    expect(result.token).toBeNull();
    expect(result.reason).toBe('all_tokens_expired');
  });

  it('finds a valid token among mixed used + expired + valid', () => {
    const valid = token({ token: 'good' });
    const result = findValidToken(
      [
        token({ token: 'used', used: true }),
        token({ token: 'expired', expires_at: PAST }),
        valid,
      ],
      'nuke_session',
      NOW,
    );
    expect(result.token?.token).toBe('good');
  });
});

// ---- markTokenUsed ----

describe('markTokenUsed', () => {
  it('returns a new array with only the matching token marked used', () => {
    const t1 = token({ token: 'a' });
    const t2 = token({ token: 'b' });
    const result = markTokenUsed([t1, t2], t1);
    expect(result[0].used).toBe(true);
    expect(result[1].used).toBe(false);
  });

  it('does not mutate the input', () => {
    const t1 = token({ token: 'a' });
    const input = [t1];
    markTokenUsed(input, t1);
    expect(input[0].used).toBe(false);
  });
});

// ---- loadConfirmationTokens ----

describe('loadConfirmationTokens', () => {
  function mockFs(content?: string) {
    return {
      existsSync: () => content !== undefined,
      readFileSync: () => content ?? '',
      writeFileSync: () => {},
    };
  }

  it('returns empty array when file missing', () => {
    expect(loadConfirmationTokens(mockFs(), '/x')).toEqual([]);
  });

  it('returns empty array when file empty', () => {
    expect(loadConfirmationTokens(mockFs('  \n  '), '/x')).toEqual([]);
  });

  it('parses the wrapped { schema_version, tokens } shape', () => {
    const content = JSON.stringify({
      schema_version: 1,
      tokens: [token()],
    });
    expect(loadConfirmationTokens(mockFs(content), '/x')).toEqual([token()]);
  });

  it('tolerates the bare-array legacy shape', () => {
    const content = JSON.stringify([token()]);
    expect(loadConfirmationTokens(mockFs(content), '/x')).toEqual([token()]);
  });

  it('throws on malformed JSON', () => {
    expect(() => loadConfirmationTokens(mockFs('not json'), '/x')).toThrow();
  });

  it('throws when the wrapped shape lacks tokens', () => {
    const content = JSON.stringify({ schema_version: 1 });
    expect(() => loadConfirmationTokens(mockFs(content), '/x')).toThrow(
      /tokens/,
    );
  });
});

// ---- decideConfirmation (acceptance scenarios) ----

describe('decideConfirmation — acceptance scenarios', () => {
  it('PASSES non-destructive tools with no opinion', () => {
    const d = decideConfirmation({
      toolName: 'mcp__nanoclaw__send_message',
      toolInput: { text: 'hi' },
      hasUntrustedProvenance: true,
      tokens: [],
      nowIso: NOW,
    });
    expect(d.kind).toBe('pass');
  });

  it('ALLOWS operator-originated trusted nuke_session without a token', () => {
    const d = decideConfirmation({
      toolName: 'mcp__nanoclaw__nuke_session',
      toolInput: {},
      hasUntrustedProvenance: false,
      tokens: [],
      nowIso: NOW,
    });
    expect(d.kind).toBe('allow');
    if (d.kind === 'allow') expect(d.reason).toMatch(/operator/i);
  });

  it('DENIES untrusted-provenance nuke_session with no token (web injection scenario)', () => {
    const d = decideConfirmation({
      toolName: 'mcp__nanoclaw__nuke_session',
      toolInput: {},
      hasUntrustedProvenance: true,
      tokens: [],
      nowIso: NOW,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') {
      expect(d.scope).toBe('nuke_session');
      expect(d.reason).toMatch(/aye-confirm/);
      expect(d.reason).toMatch(/--scope nuke_session/);
    }
  });

  it('DENIES untrusted-provenance github_backup absent token (cross-group injection)', () => {
    const d = decideConfirmation({
      toolName: 'mcp__nanoclaw__github_backup',
      toolInput: { force: true },
      hasUntrustedProvenance: true,
      tokens: [token({ scope: 'nuke_session' })], // wrong scope
      nowIso: NOW,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') expect(d.scope).toBe('github_backup');
  });

  it('ALLOWS_WITH_TOKEN when a matching unspent token is present', () => {
    const t = token({ scope: 'nuke_session', token: 'good' });
    const d = decideConfirmation({
      toolName: 'mcp__nanoclaw__nuke_session',
      toolInput: {},
      hasUntrustedProvenance: true,
      tokens: [t],
      nowIso: NOW,
    });
    expect(d.kind).toBe('allow_with_token');
    if (d.kind === 'allow_with_token') {
      expect(d.token.token).toBe('good');
      expect(d.updatedTokens[0].used).toBe(true);
    }
  });

  it('DENIES with reason all_tokens_used when only used tokens match', () => {
    const d = decideConfirmation({
      toolName: 'mcp__nanoclaw__nuke_session',
      toolInput: {},
      hasUntrustedProvenance: true,
      tokens: [token({ used: true })],
      nowIso: NOW,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') expect(d.reason).toMatch(/already been consumed/);
  });

  it('DENIES with reason all_tokens_expired when only expired tokens match', () => {
    const d = decideConfirmation({
      toolName: 'mcp__nanoclaw__nuke_session',
      toolInput: {},
      hasUntrustedProvenance: true,
      tokens: [token({ expires_at: PAST })],
      nowIso: NOW,
    });
    expect(d.kind).toBe('deny');
    if (d.kind === 'deny') expect(d.reason).toMatch(/expired/);
  });

  it('token is single-use — second call with same token denies', () => {
    const t = token({ token: 'one-shot' });
    const first = decideConfirmation({
      toolName: 'mcp__nanoclaw__nuke_session',
      toolInput: {},
      hasUntrustedProvenance: true,
      tokens: [t],
      nowIso: NOW,
    });
    expect(first.kind).toBe('allow_with_token');
    if (first.kind !== 'allow_with_token') return;
    const second = decideConfirmation({
      toolName: 'mcp__nanoclaw__nuke_session',
      toolInput: {},
      hasUntrustedProvenance: true,
      tokens: first.updatedTokens,
      nowIso: NOW,
    });
    expect(second.kind).toBe('deny');
    if (second.kind === 'deny')
      expect(second.reason).toMatch(/already been consumed/);
  });
});

// ---- parseDuration ----

describe('parseDuration', () => {
  it('parses common suffixes', () => {
    expect(parseDuration('30s')).toBe(30_000);
    expect(parseDuration('5m')).toBe(300_000);
    expect(parseDuration('1h')).toBe(3_600_000);
    expect(parseDuration('7d')).toBe(7 * 86_400_000);
  });

  it('tolerates whitespace and case', () => {
    expect(parseDuration('  10M ')).toBe(600_000);
    expect(parseDuration('2H')).toBe(7_200_000);
  });

  it('throws on invalid input', () => {
    expect(() => parseDuration('5')).toThrow(/duration/);
    expect(() => parseDuration('5x')).toThrow(/duration/);
    expect(() => parseDuration('')).toThrow(/duration/);
  });
});
