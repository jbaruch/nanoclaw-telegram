// #651 — suppress the SDK's echoed input prompt on an empty assistant turn.
//
// In streaming-input mode, when the agent produces a turn with no
// user-facing content — no text block and no tool_use, e.g. when
// `rules/default-silence.md` instructs "produce zero output" in a
// passive group — the SDK populates the `result` message's `result`
// field with the rendered input prompt (`Human: <system-reminder>...`
// plus the `<context>` and `<messages>` envelope) instead of empty
// text. `runQuery` forwards `result.result` to the orchestrator, which
// then publishes it as a bot reply — leaking the system-reminder
// envelope and the next inbound user message back into chat.
//
// An empty turn has nothing to say, so any non-empty result text on
// such a turn is the spurious echo and must not be forwarded.
//
// Distinct from `silent-stop-synthesis.ts` (#461): that handles the
// loop draining with NO result event at all; this handles a result
// event that DID arrive carrying the echo.
//
// Pure helper so the decision is unit-testable without the SDK
// iterator, mirroring `silent-stop-synthesis.ts` / `result-suppression.ts`.
export function shouldSuppressEchoedResult(
  assistantEmittedContent: boolean,
  textResult: string | null,
): boolean {
  return !assistantEmittedContent && !!textResult;
}
