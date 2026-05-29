// #581 — split the "don't chat-echo" signal from the "don't log" signal
// at the agent-runner's success-result emission point.
//
// History: PR #47 added a suppression branch so that when the agent had
// already used `send_message` / `send_file` successfully, the SDK's
// final closing-thought text would NOT be echoed back as a second user
// reply. The suppression collapsed `result` to `null` and relied on the
// orchestrator's `if (result.result)` gate to skip the chat-echo path.
//
// That collapse was overloaded — the orchestrator and the task-scheduler
// both also branched on `result.result` to populate `task_run_logs.result`
// for observability. Wrapper scheduled-task skills (`nightly-external-sync`,
// `entertainment-sync`) which always finish by calling `send_message`
// therefore landed with `status='success'` + `result=null` in
// `task_run_logs`, breaking forensic greps and silent-success accounting
// (`task_run_logs.status='success'` ≠ task ran).
//
// Fix: split the signals. `result` always carries the SDK's text result
// when present (so observability is preserved); a new `chat_displayed`
// boolean tells downstream consumers (orchestrator, task-scheduler)
// "the agent already wrote to chat — do NOT echo result.text back as a
// second user reply". The orchestrator and task-scheduler each gate
// their chat-echo + storeMessage block on `!chat_displayed`, but keep
// populating `task_run_logs.result` from `result.result` regardless.
//
// Pulled out as a pure helper so the agent-runner's success-branch
// emission decision is unit-testable without spinning the SDK iterator,
// mirroring the shape of `silent-stop-synthesis.ts` and
// `format-error-result.ts`.

export interface SuccessUsageShape {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

export interface SuccessOutputShape {
  status: 'success';
  result: string | null;
  newSessionId?: string;
  usage?: SuccessUsageShape;
  chat_displayed?: boolean;
}

/**
 * Build the writeOutput payload for a successful SDK result event.
 *
 * - `textResult` is the agent's final assistant text (may be empty / null).
 * - `userFacingSendSucceeded` indicates the agent already used
 *   `send_message` / `send_file` successfully this turn.
 *
 * When `userFacingSendSucceeded` is true AND `textResult` is non-empty,
 * the returned payload carries `chat_displayed: true` so the orchestrator
 * + task-scheduler can skip their chat-echo path while STILL recording
 * the result text in `task_run_logs.result`.
 *
 * In every other case (no send tool was used, or no text result), the
 * payload omits `chat_displayed` (effectively false) and the original
 * chat-echo behavior is preserved.
 */
export function buildSuccessOutput(
  textResult: string | null,
  userFacingSendSucceeded: boolean,
  newSessionId: string | undefined,
  usage: SuccessUsageShape | undefined,
): SuccessOutputShape {
  const hasText = !!textResult;
  const chatDisplayed = userFacingSendSucceeded && hasText;
  const payload: SuccessOutputShape = {
    status: 'success',
    result: textResult || null,
    newSessionId,
    usage,
  };
  if (chatDisplayed) {
    payload.chat_displayed = true;
  }
  return payload;
}
