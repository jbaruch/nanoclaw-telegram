// #461 — silent-stop synthesis predicate.
//
// runQuery's for-await over the SDK iterator can drain without ever
// yielding a `message.type === 'result'` event (agent exhausts steps,
// SDK ends the stream after a `_close`-driven `stream.end()`, or the
// turn produces no terminal). The host's task-scheduler arms
// `scheduleClose` ONLY on `streamedOutput.status === 'success'`, so a
// missing terminal write left maintenance containers wedged until
// `IDLE_TIMEOUT + 30s` — the slot stayed held, downstream scheduled
// tasks queued up, and tasks got dropped at the dispatch threshold.
//
// The predicate is split out as a pure helper so the for-await loop's
// final state (`resultCount`, `sawErrorResult`) becomes testable
// without spinning up the full SDK iterator.
export function shouldSynthesizeSilentStop(
  resultCount: number,
  sawErrorResult: boolean,
): boolean {
  // Synthesize only when the loop drained cleanly without any
  // result message at all — an error result already wrote a terminal
  // envelope, so re-emitting a synthesized success on top would be
  // wrong. Healthy multi-result runs (resultCount > 0) already wrote
  // their own terminal in the loop body and break'd out.
  return resultCount === 0 && !sawErrorResult;
}
