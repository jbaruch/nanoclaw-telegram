// #581 — orchestrator-side streamed-output decision helper.
//
// `processGroupMessages` (`src/index.ts`) receives each non-null SDK
// result event and branches four ways depending on the agent's
// final-text content and the `chat_displayed` flag the agent-runner
// sets when it already delivered a user-facing reply via
// `send_message` / `send_file`. The decision tree is small and pure
// (only depends on the envelope shape + a strip step), but it was
// living inline inside a 1500-line function — impossible to unit-test
// without spinning up the channel, the DB, and the queue.
//
// Pulled out as a pure helper so each decision point is unit-testable
// without the surrounding channel + DB machinery, mirroring the shape
// of `silent-stop-synthesis.ts` and `format-error-result.ts` on the
// agent-runner side.
//
// Decision kinds:
//   - `noop`           — `result.result` is null/undefined. Caller
//                        does nothing for this event (no log, no
//                        idle reset, no send).
//   - `reset-only`     — `result.result` had text, but it was empty
//                        after stripping `<internal>...</internal>`
//                        blocks. Caller logs the raw length + resets
//                        the idle timer (the agent IS doing work,
//                        just not user-visible work) and skips the
//                        send / storeMessage path.
//   - `send`           — non-empty stripped text AND
//                        `chat_displayed` is falsy. Caller invokes
//                        `channel.sendMessage` + `storeMessage` (if
//                        delivery succeeded), consumes
//                        `pendingReplyTo[chatJid]`, and marks
//                        `outputSentToUser`.
//   - `mark-displayed` — non-empty stripped text AND
//                        `chat_displayed` is `true`. The agent
//                        already delivered the reply via
//                        `send_message`; caller logs the
//                        skip-chat-echo line, consumes
//                        `pendingReplyTo[chatJid]`, and marks
//                        `outputSentToUser`. No send, no
//                        storeMessage.

/**
 * Shape of the streamed-output envelope the helper inspects. Mirrors
 * `ContainerOutput` from `container-runner.ts` but typed loosely so
 * the helper stays usable from tests that don't import the full
 * container-runner module.
 */
export interface StreamedOutputShape {
  result?: string | null;
  chat_displayed?: boolean;
}

export type AgentOutputAction =
  | { kind: 'noop' }
  | { kind: 'reset-only'; rawLength: number }
  | { kind: 'send'; textForLog: string; rawLength: number }
  | { kind: 'mark-displayed'; textForLog: string; rawLength: number };

/**
 * Strip `<internal>...</internal>` blocks the agent uses for internal
 * reasoning before chat-echo. Exposed so callers and tests share one
 * implementation and can't drift from the regex inline at the call
 * site in `processGroupMessages`.
 */
export function stripInternalBlocks(raw: string): string {
  return raw.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
}

/**
 * Decide what the orchestrator should do with a single streamed
 * SDK-result event. Pure function — no DB, no channel, no logger.
 *
 * The caller is responsible for the actual side effects (logging,
 * `channel.sendMessage`, `storeMessage`, `pendingReplyTo` consume,
 * `idleTimerControl.reset`). This helper only returns the decision.
 */
export function decideAgentOutputAction(
  streamedOutput: StreamedOutputShape | null | undefined,
): AgentOutputAction {
  if (!streamedOutput || !streamedOutput.result) {
    return { kind: 'noop' };
  }
  const raw =
    typeof streamedOutput.result === 'string'
      ? streamedOutput.result
      : JSON.stringify(streamedOutput.result);
  const text = stripInternalBlocks(raw);
  if (!text) {
    return { kind: 'reset-only', rawLength: raw.length };
  }
  if (streamedOutput.chat_displayed) {
    return { kind: 'mark-displayed', textForLog: text, rawLength: raw.length };
  }
  return { kind: 'send', textForLog: text, rawLength: raw.length };
}
