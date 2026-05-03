import { describe, it, expect } from 'vitest';
import { parseScriptOutput } from './script-output-parse.js';

describe('parseScriptOutput', () => {
  it('accepts wake_agent: true with data payload', () => {
    const out = parseScriptOutput(
      '{"wake_agent": true, "data": {"reason": "no_state"}}',
    );
    expect(out).toEqual({
      ok: true,
      result: { wake_agent: true, data: { reason: 'no_state' } },
    });
  });

  it('accepts wake_agent: false', () => {
    const out = parseScriptOutput('{"wake_agent": false, "data": {}}');
    expect(out.ok).toBe(true);
    if (out.ok) expect(out.result.wake_agent).toBe(false);
  });

  it('rejects camelCase wakeAgent — guards against #480 regression', () => {
    const out = parseScriptOutput('{"wakeAgent": true, "data": {}}');
    expect(out).toEqual({
      ok: false,
      reason: 'missing_wake_agent',
      lastLine: '{"wakeAgent": true, "data": {}}',
    });
  });

  it('parses only the last non-empty line of multi-line stdout', () => {
    const out = parseScriptOutput(
      'precheck: starting\nprecheck: 3 reasons accumulated\n{"wake_agent": true}',
    );
    expect(out.ok).toBe(true);
    if (out.ok) expect(out.result.wake_agent).toBe(true);
  });

  it('returns empty on whitespace-only stdout', () => {
    expect(parseScriptOutput('   \n\n  ')).toEqual({
      ok: false,
      reason: 'empty',
    });
  });

  it('returns invalid_json on a non-JSON last line', () => {
    const out = parseScriptOutput('precheck output: nothing to do');
    expect(out.ok).toBe(false);
    if (!out.ok) {
      expect(out.reason).toBe('invalid_json');
      expect(out.lastLine).toBe('precheck output: nothing to do');
    }
  });

  it('returns missing_wake_agent on JSON without the contract field', () => {
    const out = parseScriptOutput('{"foo": "bar"}');
    expect(out.ok).toBe(false);
    if (!out.ok) expect(out.reason).toBe('missing_wake_agent');
  });

  it('returns missing_wake_agent on JSON with non-boolean wake_agent', () => {
    expect(parseScriptOutput('{"wake_agent": "true"}')).toMatchObject({
      ok: false,
      reason: 'missing_wake_agent',
    });
    expect(parseScriptOutput('{"wake_agent": 1}')).toMatchObject({
      ok: false,
      reason: 'missing_wake_agent',
    });
    expect(parseScriptOutput('{"wake_agent": null}')).toMatchObject({
      ok: false,
      reason: 'missing_wake_agent',
    });
  });

  it('returns missing_wake_agent on a top-level non-object JSON value', () => {
    expect(parseScriptOutput('null')).toMatchObject({
      ok: false,
      reason: 'missing_wake_agent',
    });
    expect(parseScriptOutput('true')).toMatchObject({
      ok: false,
      reason: 'missing_wake_agent',
    });
    expect(parseScriptOutput('"wake_agent"')).toMatchObject({
      ok: false,
      reason: 'missing_wake_agent',
    });
  });

  it('accepts payload with data omitted', () => {
    const out = parseScriptOutput('{"wake_agent": true}');
    expect(out.ok).toBe(true);
    if (out.ok) expect(out.result).toEqual({ wake_agent: true });
  });

  it('rejects non-object data per script-delegation rule', () => {
    expect(
      parseScriptOutput('{"wake_agent": true, "data": "ready"}'),
    ).toMatchObject({
      ok: false,
      reason: 'invalid_data_shape',
    });
    expect(parseScriptOutput('{"wake_agent": true, "data": 7}')).toMatchObject({
      ok: false,
      reason: 'invalid_data_shape',
    });
    expect(
      parseScriptOutput('{"wake_agent": true, "data": null}'),
    ).toMatchObject({
      ok: false,
      reason: 'invalid_data_shape',
    });
    expect(
      parseScriptOutput('{"wake_agent": true, "data": [1,2,3]}'),
    ).toMatchObject({
      ok: false,
      reason: 'invalid_data_shape',
    });
  });

  it('accepts empty-object data and nested-object data', () => {
    expect(
      parseScriptOutput('{"wake_agent": false, "data": {}}'),
    ).toMatchObject({
      ok: true,
    });
    expect(
      parseScriptOutput(
        '{"wake_agent": true, "data": {"reasons": ["a", "b"]}}',
      ),
    ).toMatchObject({ ok: true });
  });
});
