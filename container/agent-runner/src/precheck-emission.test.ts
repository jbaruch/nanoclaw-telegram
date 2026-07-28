import { describe, expect, it } from 'vitest';

import {
  buildPrecheckErrorOutput,
  buildPrecheckSkippedOutput,
  formatExecErrorDetail,
} from './precheck-emission.js';

describe('buildPrecheckSkippedOutput', () => {
  it('emits precheck_skipped status with <internal> envelope wrapping the data payload', () => {
    const data = {
      reason: 'within_cadence',
      last_run: '2026-05-20T04:06:58Z',
      age_hours: 47.89,
      cadence_hours: 72.0,
    };

    const out = buildPrecheckSkippedOutput(data);

    expect(out.status).toBe('precheck_skipped');
    expect(out.result).toBe(
      '<internal>precheck-skipped: ' + JSON.stringify(data) + '</internal>',
    );
  });

  it('serialises empty data object verbatim (cadence-elapsed wake_agent=true is OUT of scope; this covers the no-fields wake_agent=false case)', () => {
    const out = buildPrecheckSkippedOutput({});

    expect(out.status).toBe('precheck_skipped');
    expect(out.result).toBe('<internal>precheck-skipped: {}</internal>');
  });

  it('preserves nested data shapes — the watchdog should be able to read whatever the precheck emitted, not a sanitised subset', () => {
    const data = {
      reason: 'within_cadence',
      nested: { extra: ['a', 'b'], flag: true },
    };

    const out = buildPrecheckSkippedOutput(data);

    expect(out.result).toContain('"nested":{"extra":["a","b"],"flag":true}');
  });

  it('never emits status: success — the whole point of the carve-out is to differentiate from a wake-and-empty run', () => {
    const out = buildPrecheckSkippedOutput({ reason: 'within_cadence' });

    expect(out.status).not.toBe('success');
  });

  it('coerces undefined data to {} so the diagnostic never contains the literal "undefined" — `parseScriptOutput` accepts omitted `data` as valid, so `scriptResult.data` may be undefined; JSON.stringify(undefined) would emit the literal string undefined (not valid JSON) and break watchdog parsers', () => {
    const out = buildPrecheckSkippedOutput(undefined);

    expect(out.status).toBe('precheck_skipped');
    expect(out.result).toBe('<internal>precheck-skipped: {}</internal>');
    expect(out.result).not.toContain('undefined');
  });
});

describe('buildPrecheckErrorOutput', () => {
  it('emits error status with diagnostic result naming the specific failure mode', () => {
    const out = buildPrecheckErrorOutput('execfile-error');

    expect(out.status).toBe('error');
    expect(out.result).toBe(
      '<internal>precheck-error: execfile-error</internal>',
    );
    expect(out.error).toBe('precheck script failed: execfile-error');
  });

  it('stamps timedOut only for a declared-budget timeout kill (#890 follow-up)', () => {
    // The host's operator alert keys off this field rather than
    // matching `formatExecErrorDetail`'s prose, so the stamp must
    // survive as structured JSON and must NOT appear on any other
    // failure mode — a crash or bad-JSON precheck stays covered by the
    // heartbeat's task-failure report instead of paging immediately.
    const killed = buildPrecheckErrorOutput(
      'execfile-error',
      'timed out after 30s, signal=SIGTERM',
      true,
    );
    expect(killed.timedOut).toBe(true);

    const crashed = buildPrecheckErrorOutput('execfile-error', 'exit=1', false);
    expect(crashed.timedOut).toBeUndefined();
    expect('timedOut' in crashed).toBe(false);

    const badJson = buildPrecheckErrorOutput('invalid-json');
    expect(badJson.timedOut).toBeUndefined();
    expect('timedOut' in badJson).toBe(false);
  });

  it('covers each PrecheckErrorReason variant so watchdog queries can match on the specific cause', () => {
    const reasons = [
      'execfile-error',
      'empty-output',
      'invalid-json',
      'invalid-data-shape',
      'missing-wake-agent',
    ] as const;

    for (const reason of reasons) {
      const out = buildPrecheckErrorOutput(reason);
      expect(out.status).toBe('error');
      expect(out.result).toBe(`<internal>precheck-error: ${reason}</internal>`);
      expect(out.error).toBe(`precheck script failed: ${reason}`);
    }
  });

  it('result envelope is wrapped in <internal> tags so the orchestrator strips it before chat-echo and only task_run_logs.result sees it', () => {
    const out = buildPrecheckErrorOutput('empty-output');

    expect(out.result.startsWith('<internal>')).toBe(true);
    expect(out.result.endsWith('</internal>')).toBe(true);
  });

  it('never emits status: success — masking a script crash as success was the pre-fix behaviour the watchdog could not act on', () => {
    const out = buildPrecheckErrorOutput('invalid-json');

    expect(out.status).not.toBe('success');
  });

  it('error message is a non-empty string — pre-fix the precheck-error path emitted writeOutput without an error field, so task_run_logs.error was null and the operator had nothing to diagnose with', () => {
    const out = buildPrecheckErrorOutput('invalid-data-shape');

    expect(typeof out.error).toBe('string');
    expect(out.error.length).toBeGreaterThan(0);
  });

  // #812 Bug A: an optional detail suffix disambiguates execfile-error.
  it('appends a detail suffix to both error and result when provided', () => {
    const out = buildPrecheckErrorOutput(
      'execfile-error',
      'timed out after 30s, signal=SIGKILL',
    );

    expect(out.error).toBe(
      'precheck script failed: execfile-error (timed out after 30s, signal=SIGKILL)',
    );
    expect(out.result).toBe(
      '<internal>precheck-error: execfile-error (timed out after 30s, signal=SIGKILL)</internal>',
    );
  });

  it('is byte-identical to the no-detail form when detail is omitted (backward compatible)', () => {
    expect(buildPrecheckErrorOutput('execfile-error')).toEqual(
      buildPrecheckErrorOutput('execfile-error', undefined),
    );
    expect(buildPrecheckErrorOutput('execfile-error', '')).toEqual(
      buildPrecheckErrorOutput('execfile-error'),
    );
  });
});

describe('formatExecErrorDetail', () => {
  // The core #812 Bug A contract: a timeout-kill and a genuine non-zero
  // exit produced the SAME persisted string before this — now they don't.
  it('distinguishes a timeout-kill from a genuine non-zero exit', () => {
    const timeout = formatExecErrorDetail({
      exitCode: null,
      signal: 'SIGKILL',
      timedOut: true,
      timeoutMs: 30_000,
    });
    const nonZeroExit = formatExecErrorDetail({
      exitCode: 1,
      signal: null,
    });

    expect(timeout).toContain('timed out after 30s');
    expect(timeout).toContain('signal=SIGKILL');
    expect(nonZeroExit).toContain('exit=1');
    expect(timeout).not.toBe(nonZeroExit);
  });

  it('reports a spawn failure by its errno and nothing else', () => {
    expect(formatExecErrorDetail({ spawnCode: 'ENOENT' })).toBe(
      'spawn-failed: ENOENT',
    );
  });

  it('never embeds child stderr and stays a controlled, envelope-safe string (no-secrets)', () => {
    const detail = formatExecErrorDetail({
      exitCode: 2,
      signal: 'SIGTERM',
      timedOut: true,
      timeoutMs: 30_000,
    });

    // Only our literals, signal names, and integers — no untrusted text,
    // so it can never leak stderr into the persisted error / operator
    // alert, and no `<`/`>` that could break the `<internal>` envelope.
    expect(detail).toBe('timed out after 30s, signal=SIGTERM, exit=2');
    expect(detail).not.toMatch(/[<>]/);
  });

  it('omits exit= for a clean exit code and falls back when nothing is set', () => {
    expect(formatExecErrorDetail({ exitCode: 0, signal: null })).toBe(
      'unknown-failure',
    );
  });

  it('names a bare signal kill with no exit code', () => {
    expect(formatExecErrorDetail({ exitCode: null, signal: 'SIGSEGV' })).toBe(
      'signal=SIGSEGV',
    );
  });
});
