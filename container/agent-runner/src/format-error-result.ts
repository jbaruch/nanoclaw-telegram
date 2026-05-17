// #582 — structured error formatter for the SDK's result-message
// error path.
//
// When the Anthropic SDK signals an error result (`is_error: true` or a
// non-success/non-unknown `subtype`), the agent-runner has to fold the
// SDK's classification fields into a single string that lands in
// `task_run_logs.error` on the orchestrator side. The legacy form was
// `${subtype}: ${summary}` — which degenerates to useless strings like
// `"success: completed"` when the SDK reports a contradictory
// `subtype: 'success'` with `is_error: true` (a state observed on the
// 2026-05-15 heartbeat false-success fires that motivated #582,
// where the maintenance container's 5-min inactivity cap fired and
// the SDK returned an abort result with `subtype: 'success'`).
//
// The DB error column is the only diagnostic surface that survives a
// container exit — the per-result `log()` line goes to stderr and is
// rotated out of reach quickly. So this helper preserves every SDK
// classification field (subtype, is_error, terminal_reason,
// stop_reason, plus a derived human-readable summary) as
// `key=value | key=value` pairs, capped at 500 chars total to stay
// within the stdout IPC marker JSON's safe size (#149 review found
// excessively-large strings caused buffer issues).
//
// Pulled out as a pure helper so the agent-runner's result-loop logic
// stays testable without spinning up the SDK iterator.

export interface SdkErrorMessageShape {
  subtype?: string;
  errors?: string[];
  terminal_reason?: string;
  is_error?: boolean;
  stop_reason?: string | null;
}

/**
 * Build the `task_run_logs.error` string for an SDK result-message
 * error. `textResult` is the agent's final assistant text (if any) —
 * passed in because the SDK sometimes parks the only readable cause
 * there instead of in `errors[]` / `terminal_reason`.
 *
 * Output shape:
 *   `subtype=<x> | is_error=<true|false> | terminal_reason=<y> | stop_reason=<z> | summary=<short>`
 *
 * Total length is capped at 500 chars. The `summary` field is the
 * most-informative free-form string available; everything else is
 * a short SDK enum.
 */
export function formatErrorResult(
  errMsg: SdkErrorMessageShape,
  textResult: string | null,
): string {
  const subtype = errMsg.subtype || 'unknown';
  const rawSummary =
    errMsg.terminal_reason ||
    (errMsg.errors && errMsg.errors[0]) ||
    textResult ||
    subtype;
  const summary = String(rawSummary).replace(/\s+/g, ' ').slice(0, 300);

  // Defensive `String()` before `.replace()`: `errMsg.terminal_reason` is
  // external SDK runtime data (the `message as { ... }` cast accepts
  // whatever the SDK sent, including non-string payloads). A non-string
  // value would throw `TypeError: ... is not a function` here and we'd
  // lose the DB `error` diagnostic before `writeOutput(...)` runs — the
  // exact failure mode this formatter exists to prevent. Same shape as
  // the `String(rawSummary)` coercion below.
  const terminalReason = String(errMsg.terminal_reason || 'none')
    .replace(/\s+/g, ' ')
    .slice(0, 60);

  const fields = [
    `subtype=${subtype}`,
    `is_error=${errMsg.is_error === true ? 'true' : 'false'}`,
    `terminal_reason=${terminalReason}`,
    `stop_reason=${errMsg.stop_reason || 'none'}`,
    `summary=${summary}`,
  ];
  return fields.join(' | ').slice(0, 500);
}
