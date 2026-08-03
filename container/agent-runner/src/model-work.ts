/**
 * #901 — "the model never ran" detection for terminal success markers.
 *
 * When the deployment's subscription cap is exhausted, the Claude Code
 * CLI ends the turn before issuing a single API request and hands back
 * an ordinary assistant-shaped result whose text is the cap notice
 * (`You've hit your limit · resets 5pm (America/Chicago)`). The SDK sets
 * no `is_error`, no error-shaped `subtype`, and no distinguishing
 * `stop_reason`, so `classifyResultIsError` correctly reads it as a
 * clean termination and the host records `status='success'`.
 *
 * On 2026-08-01 that shape covered a full day: 93 scheduled runs aborted
 * on the cap, 92 recorded `success`, and the credential proxy logged 53
 * API calls for the entire day (all of them after the cap reset at
 * 22:00Z). Three monitoring surfaces reported healthy — the silent-
 * success watchdog counted 0 silent runs across all 28 tasks, because
 * its silence test is `result IS NULL OR result=''` and the cap notice
 * is non-empty prose.
 *
 * The signal this module keys on is structural, not textual: a run in
 * which the model produced no turn at all. Every SDK assistant message
 * carries a `usage` payload (the Messages API echoes input/output token
 * counts on every response), so the runner's `latestUsage` is set the
 * moment the model does any work — including an empty assistant turn,
 * which still burns tokens. `latestUsage` still `undefined` at a
 * terminal marker means no assistant message ever arrived: no tokens, no
 * tool calls, no model output. The run did nothing.
 *
 * Matching the cap notice's wording would be the fragile alternative.
 * The string is provider-owned prose, localized and reworded at will,
 * and `coding-policy: script-delegation`'s regex trap applies — a
 * reworded notice must not silently stop the classification. This
 * mirrors the `timedOut` field's rationale: structured rather than
 * inferred from prose, so both sides own the field.
 *
 * The host (`src/container-runner.ts`) keeps the marker's `success`
 * status so `scheduleClose` still drains the maintenance slot promptly
 * (no #461 wedge), and honours `noModelWork` to resolve the run as
 * `killed` — incomplete / retriable — exactly as it honours `noDelivery`
 * (#689). `killed` is a labelling change, not a retry trigger: the next
 * cron fire is the retry, so a cap outage produces honest rows rather
 * than a redelivery storm against an already-capped API.
 */

/**
 * The per-turn token usage the runner captures from the SDK message
 * stream. Declared structurally rather than imported so this predicate
 * stays dependency-free and unit-testable — `ContainerOutput` is a local
 * interface in `index.ts`, and widening it into a shared module for one
 * field would be unjustified churn. Only presence is read here, so the
 * shape is deliberately minimal.
 */
export interface TurnUsage {
  input_tokens: number;
  output_tokens: number;
}

/**
 * Decide whether a terminal `success` marker should be stamped
 * `noModelWork`. True when no assistant message carried a `usage`
 * payload during the run, i.e. the SDK returned without the model
 * producing a single turn.
 *
 * Centralises the predicate the runner applies at every terminal-marker
 * site (the silent-stop synthesis, the SDK-result success marker, the
 * echo-suppressed empty turn) so the decision is identical across all of
 * them and unit-testable without spinning the SDK iterator.
 *
 * A zero-token `usage` payload is NOT no-model-work: the payload's
 * presence proves an assistant message arrived. Only absence counts.
 */
export function shouldStampNoModelWork(
  latestUsage: TurnUsage | undefined,
): boolean {
  return latestUsage === undefined;
}
