import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { generateToken, mintToken, parseArgs } from './aye-confirm.js';
import { parseDuration } from '../container/agent-runner/src/confirmation-tokens.js';

// ---- parseArgs ----
//
// `parseArgs` calls `process.exit(2)` via `usage(2)` for argument
// errors. Stub `process.exit` so the test can observe the call without
// killing the test process.

describe('parseArgs', () => {
  let exitMock: ReturnType<typeof vi.spyOn>;
  let stderrMock: ReturnType<typeof vi.spyOn>;
  let stdoutMock: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    exitMock = vi
      .spyOn(process, 'exit')
      .mockImplementation((((code?: number | string | null) => {
        throw new Error(`process.exit(${code ?? 0})`);
      }) as unknown) as typeof process.exit);
    stderrMock = vi
      .spyOn(process.stderr, 'write')
      .mockImplementation(() => true);
    stdoutMock = vi
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);
  });

  afterEach(() => {
    exitMock.mockRestore();
    stderrMock.mockRestore();
    stdoutMock.mockRestore();
  });

  it('parses a fully-specified valid invocation', () => {
    const args = parseArgs([
      '--scope',
      'nuke_session',
      '--reason',
      'stuck container',
      '--ttl',
      '10m',
    ]);
    expect(args).toEqual({
      scope: 'nuke_session',
      reason: 'stuck container',
      ttl: '10m',
      chatJid: undefined,
    });
  });

  it('defaults --ttl to 5m', () => {
    const args = parseArgs(['--scope', 'github_backup', '--reason', 'r']);
    expect(args.ttl).toBe('5m');
  });

  it('captures --chat-jid when supplied', () => {
    const args = parseArgs([
      '--scope',
      'nuke_session',
      '--reason',
      'r',
      '--chat-jid',
      'tg:-100123',
    ]);
    expect(args.chatJid).toBe('tg:-100123');
  });

  it('errors when --scope is missing', () => {
    expect(() => parseArgs(['--reason', 'r'])).toThrow(/process\.exit/);
    const calls = (stderrMock.mock.calls.flat() as string[]).join('');
    expect(calls).toMatch(/--scope is required/);
  });

  it('errors when --reason is missing', () => {
    expect(() => parseArgs(['--scope', 'nuke_session'])).toThrow(
      /process\.exit/,
    );
    const calls = (stderrMock.mock.calls.flat() as string[]).join('');
    expect(calls).toMatch(/--reason is required/);
  });

  it('errors when --scope value is invalid', () => {
    expect(() =>
      parseArgs(['--scope', 'totally-not-real', '--reason', 'r']),
    ).toThrow(/process\.exit/);
    const calls = (stderrMock.mock.calls.flat() as string[]).join('');
    expect(calls).toMatch(/invalid --scope/);
  });

  it('accepts the three new AGENT_MODEL set_* scopes (#595)', () => {
    for (const scope of [
      'set_agent_model',
      'set_maintenance_agent_model',
      'set_task_agent_model',
    ]) {
      const args = parseArgs(['--scope', scope, '--reason', 'cost-tier flip']);
      expect(args.scope).toBe(scope);
    }
  });

  it('errors when an unknown argument is given', () => {
    expect(() => parseArgs(['--bogus'])).toThrow(/process\.exit/);
    const calls = (stderrMock.mock.calls.flat() as string[]).join('');
    expect(calls).toMatch(/unknown argument/);
  });

  it('errors with a clean message when --ttl is the last arg (missing value)', () => {
    expect(() =>
      parseArgs(['--scope', 'nuke_session', '--reason', 'r', '--ttl']),
    ).toThrow(/process\.exit/);
    const calls = (stderrMock.mock.calls.flat() as string[]).join('');
    expect(calls).toMatch(/--ttl requires a value/);
  });

  it('errors with a clean message when --chat-jid value is missing (next is a flag)', () => {
    expect(() =>
      parseArgs([
        '--scope',
        'nuke_session',
        '--reason',
        'r',
        '--chat-jid',
        '--ttl',
        '5m',
      ]),
    ).toThrow(/process\.exit/);
    const calls = (stderrMock.mock.calls.flat() as string[]).join('');
    expect(calls).toMatch(/--chat-jid requires a value/);
  });

  it('errors when --ttl value cannot be parsed (forwarded from parseDuration)', () => {
    expect(() =>
      parseArgs([
        '--scope',
        'nuke_session',
        '--reason',
        'r',
        '--ttl',
        '5x',
      ]),
    ).toThrow(/duration|invalid/i);
  });
});

// ---- mintToken ----

describe('mintToken', () => {
  it('produces a token with all required fields', () => {
    const t = mintToken({
      scope: 'nuke_session',
      reason: 'stuck',
      ttl: '5m',
    });
    expect(t.scope).toBe('nuke_session');
    expect(t.reason).toBe('stuck');
    expect(t.used).toBe(false);
    expect(t.token).toMatch(/^[0-9a-f]{32}$/);
    expect(typeof t.issued_at).toBe('string');
    expect(typeof t.expires_at).toBe('string');
    expect(Date.parse(t.issued_at)).not.toBeNaN();
    expect(Date.parse(t.expires_at)).not.toBeNaN();
  });

  it('sets expires_at to issued_at + ttl', () => {
    const t = mintToken({
      scope: 'github_backup',
      reason: 'release',
      ttl: '5m',
    });
    const issued = Date.parse(t.issued_at);
    const expires = Date.parse(t.expires_at);
    expect(expires - issued).toBe(parseDuration('5m'));
  });

  it('honors a long TTL', () => {
    const t = mintToken({
      scope: 'nuke_session',
      reason: 'r',
      ttl: '7d',
    });
    expect(Date.parse(t.expires_at) - Date.parse(t.issued_at)).toBe(
      parseDuration('7d'),
    );
  });

  it('attaches chat_jid when supplied', () => {
    const t = mintToken({
      scope: 'nuke_session',
      reason: 'r',
      ttl: '5m',
      chatJid: 'tg:-100222',
    });
    expect(t.chat_jid).toBe('tg:-100222');
  });

  it('omits chat_jid when not supplied', () => {
    const t = mintToken({
      scope: 'nuke_session',
      reason: 'r',
      ttl: '5m',
    });
    expect(t).not.toHaveProperty('chat_jid');
  });
});

// ---- generateToken ----
//
// generateToken's contract is `randomBytesFn(16).toString('hex')`.
// We assert the deterministic transform with an injected fake instead
// of asserting probabilistic uniqueness across draws (per
// `testing-standards`: "Tests must be deterministic — no self-
// generated random test data"). The real `crypto.randomBytes` is the
// production default; tests pass a controlled function.

describe('generateToken', () => {
  it('returns the hex of the bytes returned by randomBytesFn', () => {
    const fake = () => Buffer.from('00112233445566778899aabbccddeeff', 'hex');
    expect(generateToken(fake)).toBe('00112233445566778899aabbccddeeff');
  });

  it('always requests 16 bytes (128 bits) of entropy', () => {
    const calls: number[] = [];
    const fake = (n: number) => {
      calls.push(n);
      return Buffer.alloc(n);
    };
    generateToken(fake);
    generateToken(fake);
    expect(calls).toEqual([16, 16]);
  });

  it('hex-encodes the bytes verbatim (32 chars per 16 bytes)', () => {
    const fake = () =>
      Buffer.from([0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11,
                   0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99]);
    expect(generateToken(fake)).toBe('aabbccddeeff00112233445566778899');
  });

  it('produces a different output for a different input buffer', () => {
    let counter = 0;
    const fake = (n: number) => {
      const buf = Buffer.alloc(n);
      buf.writeUInt32BE(counter++, 0);
      return buf;
    };
    expect(generateToken(fake)).not.toBe(generateToken(fake));
  });

  it('falls back to crypto.randomBytes when no fn is supplied (smoke)', () => {
    // Smoke: don't assert ON randomness; just confirm the production
    // path doesn't throw and returns the expected shape (32 hex
    // chars). The deterministic-transform tests above prove behavior.
    const t = generateToken();
    expect(t).toMatch(/^[0-9a-f]{32}$/);
  });
});
