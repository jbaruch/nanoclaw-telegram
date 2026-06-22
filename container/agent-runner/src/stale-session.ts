/**
 * Stale-session detector for the agent-runner's throw-retry path.
 *
 * MUST stay in sync with `src/index.ts:STALE_SESSION_RE` in the
 * orchestrator package. The orchestrator and the agent-runner are
 * separate npm packages with separate tsconfigs, so they cannot share
 * a module — the regex is duplicated here on purpose. Both packages
 * have a unit-test suite that asserts on the same canonical input
 * strings (`stale-session.test.ts` in each); a drift between the two
 * patterns shows up as a test failure on at least one side.
 *
 * Match set rationale: the agent-runner's throw-retry path catches
 * thrown exceptions out of the SDK's `runQuery`. The historical narrow
 * regex (`/session|conversation not found|resume/i`) missed the same
 * SDK error shapes the orchestrator's pre-#144 regex did — most
 * importantly `error_during_execution` and the JSONL-ENOENT shape.
 * Without those tokens, a thrown stale-session error here would NOT
 * trigger the throw-side fresh-session retry, and the run would bubble
 * up as a generic error instead of the recovery the path is meant to
 * enact.
 *
 * Per `jbaruch/nanoclaw#155`, the regex is centralised in this single
 * helper inside the package; the throw-retry call site in `index.ts`
 * imports it rather than embedding its own check.
 */
const STALE_SESSION_RE =
  /no conversation found|ENOENT.*\.jsonl|session.*not found|error_during_execution/i;

export function isStaleSessionError(errorMsg: string | undefined): boolean {
  if (!errorMsg) return false;
  return STALE_SESSION_RE.test(errorMsg);
}

/**
 * Narrow stale-session phrasings for the result-message path (#697).
 *
 * Deliberately OMITS the `error_during_execution` token that
 * `STALE_SESSION_RE` carries. That token exists for the THROWN-error
 * path, where the exception message can be the bare SDK subtype string.
 * On the result-message path the subtype lives in its own `errMsg.subtype`
 * field and the `errors[]` array carries the genuine failure phrasing —
 * so matching `error_during_execution` *inside* `errors[]` would broaden
 * the retry to unrelated `num_turns=0` failures (prompt_too_long,
 * model_error) whose errors happen to mention the subtype. This regex
 * matches only the real stale-session shapes: a missing conversation, a
 * missing transcript JSONL, or an explicit "session … not found".
 */
const STALE_RESUME_ERRORS_RE =
  /no conversation found|ENOENT.*\.jsonl|session.*not found/i;

/**
 * Decide whether an SDK result-message error should defer to a
 * fresh-session retry (#697).
 *
 * The thrown-error branch in `index.ts` already retries a stale session
 * that throws out of `runQuery`. The same failure also arrives as a
 * *result message* — `subtype: 'error_during_execution'`,
 * `num_turns: 0`, `errors: ["No conversation found with session ID …"]`
 * — when an infrequent cadence task resumes a `session_id` whose
 * transcript JSONL was cleaned up between fires (#114 retention). That
 * path was not self-healing: it wrote the error and re-persisted the
 * dead id, wedging the task permanently.
 *
 * Conditions (all required):
 *   - `hadResumeSession` — a resume was actually attempted; on a fresh
 *     session there is nothing to retry, and gating on it prevents the
 *     fresh-session retry from re-triggering itself.
 *   - `numTurns === 0` — the run did zero work. A run that turned before
 *     erroring may have side effects; silently re-running it could
 *     duplicate them.
 *   - the joined SDK `errors` array matches `STALE_RESUME_ERRORS_RE` —
 *     the narrow phrasings only, NOT the generic `error_during_execution`
 *     subtype. Matching the subtype (as `isStaleSessionError` does for
 *     the throw path) would over-broaden the retry to unrelated
 *     num_turns=0 failures (prompt_too_long, model_error) whose `errors[]`
 *     happen to mention it.
 */
export function shouldRetryStaleResume(
  hadResumeSession: boolean,
  numTurns: number | undefined,
  errors: string[] | undefined,
): boolean {
  if (!hadResumeSession) return false;
  if (numTurns !== 0) return false;
  return STALE_RESUME_ERRORS_RE.test((errors ?? []).join(' '));
}
