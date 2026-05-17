import { describe, it, expect } from 'vitest';
import { formatErrorResult } from './format-error-result.js';

describe('formatErrorResult', () => {
  it('emits every SDK classification field as a separate key=value pair', () => {
    const out = formatErrorResult(
      {
        subtype: 'error_during_execution',
        is_error: true,
        terminal_reason: 'tool_use_limit_reached',
        stop_reason: 'tool_use',
      },
      null,
    );
    expect(out).toContain('subtype=error_during_execution');
    expect(out).toContain('is_error=true');
    expect(out).toContain('terminal_reason=tool_use_limit_reached');
    expect(out).toContain('stop_reason=tool_use');
  });

  it("keeps the contradictory subtype='success'+is_error=true shape debuggable (#582)", () => {
    // Reproduces the 2026-05-15 heartbeat false-success that motivated
    // this fix: SDK returns an abort result where subtype is somehow
    // 'success' but is_error is true. The legacy `${subtype}: ${summary}`
    // form collapsed to `"success: completed"` — useless. The new format
    // surfaces is_error + terminal_reason so the actual cause is in the
    // DB error column.
    const out = formatErrorResult(
      {
        subtype: 'success',
        is_error: true,
        terminal_reason: 'tool_use_limit_reached',
        stop_reason: 'tool_use',
      },
      'completed',
    );
    expect(out).toContain('subtype=success');
    expect(out).toContain('is_error=true');
    expect(out).toContain('terminal_reason=tool_use_limit_reached');
    expect(out).not.toBe('success: completed');
  });

  it('marks is_error=false when the SDK omits the flag but subtype indicated error', () => {
    // The runtime path only calls this helper inside the isError branch
    // — so is_error may be undefined even though the function ran (the
    // outer branch test included `subtype !== 'success' && subtype !==
    // 'unknown'`). The output must still distinguish "explicitly true"
    // from "absent" to preserve the diagnostic signal.
    const out = formatErrorResult(
      { subtype: 'permission_denied', stop_reason: 'permissions' },
      null,
    );
    expect(out).toContain('is_error=false');
    expect(out).toContain('subtype=permission_denied');
  });

  it('falls back to subtype as summary when no other source is present', () => {
    const out = formatErrorResult(
      { subtype: 'error_during_execution', is_error: true },
      null,
    );
    expect(out).toContain('terminal_reason=none');
    expect(out).toContain('stop_reason=none');
    expect(out).toContain('summary=error_during_execution');
  });

  it('defaults missing subtype to "unknown"', () => {
    const out = formatErrorResult({ is_error: true }, null);
    expect(out).toContain('subtype=unknown');
  });

  it('caps total length at 500 chars even with pathological summaries', () => {
    const longSummary = 'x'.repeat(10_000);
    const out = formatErrorResult(
      {
        subtype: 'error_during_execution',
        is_error: true,
        errors: [longSummary],
      },
      null,
    );
    expect(out.length).toBeLessThanOrEqual(500);
  });

  it('collapses newlines and runs of whitespace in summary so the IPC marker JSON stays one line', () => {
    const out = formatErrorResult(
      {
        subtype: 'error_during_execution',
        is_error: true,
        errors: ['line1\nline2\n\nline3'],
      },
      null,
    );
    expect(out).not.toContain('\n');
    expect(out).toContain('summary=line1 line2 line3');
  });

  it('coerces non-string terminal_reason instead of throwing (gh-aw review on #583)', () => {
    // `errMsg.terminal_reason` is external SDK runtime data — the
    // `message as { ... }` cast in the call site accepts whatever the
    // SDK sent, including non-string payloads. The pre-fix code called
    // `.replace(...)` on the raw value, which would throw `TypeError`
    // and lose the DB error diagnostic entirely. The defensive
    // `String()` coercion keeps the formatter total.
    const out = formatErrorResult(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      { subtype: 'error', is_error: true, terminal_reason: 42 as any },
      null,
    );
    expect(out).toContain('terminal_reason=42');
    expect(out).toContain('summary=42');
  });

  it('prefers terminal_reason over errors[0] over textResult over subtype for summary', () => {
    expect(
      formatErrorResult(
        {
          subtype: 'sub',
          is_error: true,
          terminal_reason: 'tr',
          errors: ['e0'],
        },
        'tr-text',
      ),
    ).toContain('summary=tr');

    expect(
      formatErrorResult(
        { subtype: 'sub', is_error: true, errors: ['e0'] },
        'tr-text',
      ),
    ).toContain('summary=e0');

    expect(
      formatErrorResult({ subtype: 'sub', is_error: true }, 'tr-text'),
    ).toContain('summary=tr-text');

    expect(
      formatErrorResult({ subtype: 'sub', is_error: true }, null),
    ).toContain('summary=sub');
  });
});
