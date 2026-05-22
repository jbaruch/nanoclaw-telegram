// #619 — pure isError decision for SDK result messages.
//
// Pulled out of the agent-runner's result loop in `index.ts` so the
// classification can be unit-tested without spinning up the SDK
// iterator, and so the SDK contradiction-shape suppression has a
// single owner across whatever call sites need it later.
//
// The SDK occasionally reports `is_error: true` on what is otherwise a
// clean termination — `subtype: 'success'`, `terminal_reason:
// 'completed'`, `stop_reason: 'stop_sequence'` (the model emitted a
// configured stop sequence and the SDK ended the turn cleanly). Three
// classification signals beat one. Treat that exact shape as success.
// Every other `is_error: true` form — tool_use_limit_reached,
// permission_denied, refusal, max_tokens, an error-shaped subtype —
// stays a genuine failure so the orchestrator's `task_failures`
// surface keeps working.

export interface ResultClassificationShape {
  subtype?: string;
  is_error?: boolean;
  stop_reason?: string | null;
  terminal_reason?: string;
}

/**
 * Returns `true` if the SDK result represents a real failure that
 * should be written to `task_run_logs.status='error'`. The legacy
 * heuristic was `is_error === true || (subtype !== 'success' &&
 * subtype !== 'unknown')`; this preserves it for every shape except
 * the documented SDK contradiction (see `jbaruch/nanoclaw#619`).
 */
export function classifyResultIsError(
  errMsg: ResultClassificationShape,
): boolean {
  const subtype = errMsg.subtype || 'unknown';
  // SDK contradiction-shape suppression (#619): on some normal
  // completions the SDK sets `is_error: true` despite `subtype:
  // 'success'`, `terminal_reason: 'completed'` (or absent), and
  // `stop_reason: 'stop_sequence'`. Three signals beat one — treat as
  // success.
  const isFalsePositiveStopSequence =
    subtype === 'success' &&
    errMsg.is_error === true &&
    errMsg.stop_reason === 'stop_sequence' &&
    (errMsg.terminal_reason === 'completed' || !errMsg.terminal_reason);
  if (isFalsePositiveStopSequence) {
    return false;
  }
  return (
    errMsg.is_error === true || (subtype !== 'success' && subtype !== 'unknown')
  );
}
