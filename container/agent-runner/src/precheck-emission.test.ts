import { describe, expect, it } from 'vitest';

import {
  buildPrecheckErrorOutput,
  buildPrecheckSkippedOutput,
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
});

describe('buildPrecheckErrorOutput', () => {
  it('emits error status with diagnostic result and an actionable error message', () => {
    const out = buildPrecheckErrorOutput();

    expect(out.status).toBe('error');
    expect(out.result).toBe(
      '<internal>precheck-error: script crashed / no output / missing wake_agent</internal>',
    );
    expect(out.error).toMatch(/precheck script crashed/);
    expect(out.error).toMatch(/omitted wake_agent/);
  });

  it('result envelope is wrapped in <internal> tags so the orchestrator strips it before chat-echo and only task_run_logs.result sees it', () => {
    const out = buildPrecheckErrorOutput();

    expect(out.result.startsWith('<internal>')).toBe(true);
    expect(out.result.endsWith('</internal>')).toBe(true);
  });

  it('never emits status: success — masking a script crash as success was the pre-fix behaviour the watchdog could not act on', () => {
    const out = buildPrecheckErrorOutput();

    expect(out.status).not.toBe('success');
  });

  it('error message is a string, not undefined — pre-fix the precheck-error path emitted writeOutput without an error field, so task_run_logs.error was null and the operator had nothing to diagnose with', () => {
    const out = buildPrecheckErrorOutput();

    expect(typeof out.error).toBe('string');
    expect(out.error.length).toBeGreaterThan(0);
  });
});
