/**
 * Resilience for the agent-runner's stdio channels.
 *
 * The runner streams result markers on stdout and debug lines on stderr to the
 * orchestrator over OS pipes. If the orchestrator's read end of either pipe
 * closes while the runner is mid-write, Node emits an async `'error'` event
 * (`EPIPE`) on the stream. With no `'error'` listener that event is unhandled
 * → uncaughtException → the process dies with
 * `closeNT (node:internal/streams/destroy)` / exit 1.
 *
 * These handlers make the runner robust to it:
 *   - stdout EPIPE: the output channel is gone, so the run cannot deliver —
 *     exit(1) cleanly (recorded as a failed run, not a silent success).
 *   - stderr EPIPE: only the diagnostic channel is gone; stdout may still be
 *     live, so swallow it and let the run continue.
 * Non-EPIPE stream errors are genuine defects and rethrow so they surface.
 */

export type ExitFn = (code: number) => never;

export function makeStdioErrorHandler(
  fatalOnEpipe: boolean,
  exit: ExitFn,
): (err: NodeJS.ErrnoException) => void {
  return (err) => {
    if (err.code === 'EPIPE') {
      if (fatalOnEpipe) exit(1);
      return;
    }
    throw err;
  };
}

export function installStdioResilience(
  stdout: NodeJS.EventEmitter = process.stdout,
  stderr: NodeJS.EventEmitter = process.stderr,
  exit: ExitFn = process.exit as ExitFn,
): void {
  stdout.on('error', makeStdioErrorHandler(true, exit));
  stderr.on('error', makeStdioErrorHandler(false, exit));
}

/**
 * Catch an EPIPE that escapes the per-stream handlers above.
 *
 * `installStdioResilience` only guards `process.stdout`/`process.stderr`.
 * The runner also holds sockets it does not own — the Agent SDK's HTTP
 * keep-alive connections, MCP-server child pipes — with no `'error'`
 * listener. When the orchestrator tears the container down after the
 * terminal result lands, a write to one of those during the post-query
 * idle wait raises an async EPIPE `'error'` on a `Socket` with no
 * listener → `Unhandled 'error' event` →
 * `emitErrorCloseNT (node:internal/streams/destroy)` → the process dies
 * with a noisy stack and exit 1 (jbaruch/nanoclaw#685). Because the host
 * records a non-zero exit as a failed run regardless of the terminal
 * result it already parsed, a completed maintenance run gets
 * mis-recorded as an error.
 *
 * A process-level `uncaughtException` guard turns that teardown EPIPE
 * into a clean exit:
 *   - terminal result already delivered → exit 0. The crash is
 *     post-delivery teardown noise; the host's #682 `hadTerminalResult`
 *     path then records the run on its merits instead of as an error.
 *   - not yet delivered → exit 1. The output channel is gone before the
 *     run could deliver — a failed run, mirroring the stdout-EPIPE
 *     policy above, never a silent success.
 * A non-EPIPE uncaught exception is a genuine defect and rethrows so it
 * still crashes loudly (matching `makeStdioErrorHandler`).
 *
 * outer-boundary-process-contract (coding-policy: error-handling): this
 * handler runs at the runner's OUTERMOST process boundary — a
 * `process.on('uncaughtException')` filter, not an inner try/catch — so
 * the carve-out audit applies:
 *   - Caller's silent-failure shape: the host
 *     (`src/container-runner.ts` `container.on('close')`) reads a
 *     non-zero container exit as a failed run and records
 *     `status:'error'`, even when a terminal result was already streamed.
 *   - What the handler emits: on EPIPE it calls `exit(0)` when the
 *     terminal result was delivered, else `exit(1)`. It never swallows —
 *     a non-EPIPE error is re-thrown.
 *   - Why propagation breaks the contract: the default (no handler) is a
 *     crash — a noisy unhandled-error stack and exit 1 that the host
 *     mis-records as a failed maintenance run even though the verdict
 *     already landed.
 * The filter is EPIPE-only; signals and deliberate exits aren't
 * delivered as `uncaughtException`, so the process stays killable and
 * every non-EPIPE defect still surfaces.
 */
export function makeUncaughtEpipeHandler(
  hasDeliveredTerminalResult: () => boolean,
  exit: ExitFn,
  reportFatal: (err: unknown) => void = (err) => console.error(err),
): (err: NodeJS.ErrnoException) => void {
  return (err) => {
    // outer-boundary-process-contract — rationale in the function doc above.
    if (err.code === 'EPIPE') {
      exit(hasDeliveredTerminalResult() ? 0 : 1);
      return;
    }
    // Non-EPIPE: a genuine defect. Re-throwing here is a footgun — this
    // runs inside an `uncaughtException` listener, where Node treats a
    // throw as a fatal "error in handler" (exit 7, double-printed
    // stacks). Emit the diagnostic and exit non-zero so the crash stays
    // loud and deterministic — the same end state as Node's default
    // uncaught-exception path.
    reportFatal(err);
    exit(1);
  };
}

export function installUncaughtEpipeGuard(
  hasDeliveredTerminalResult: () => boolean,
  proc: NodeJS.EventEmitter = process,
  exit: ExitFn = process.exit as ExitFn,
): void {
  proc.on(
    'uncaughtException',
    makeUncaughtEpipeHandler(hasDeliveredTerminalResult, exit),
  );
}

/**
 * Per-turn terminal-delivery state for the uncaught-EPIPE guard (#685).
 *
 * The runner is a persistent multi-turn loop. `markDelivered()` fires
 * when a turn writes its terminal result marker; `resetTurn()` clears it
 * at the start of the next turn; `hasDelivered()` is what the guard
 * reads to choose a teardown EPIPE's exit code. Tracking delivery
 * per-turn rather than process-globally keeps a pre-delivery EPIPE on a
 * LATER turn exiting non-zero instead of inheriting an earlier turn's
 * delivered state.
 */
export function createTerminalDeliveryTracker(): {
  markDelivered: () => void;
  resetTurn: () => void;
  hasDelivered: () => boolean;
} {
  let delivered = false;
  return {
    markDelivered: () => {
      delivered = true;
    },
    resetTurn: () => {
      delivered = false;
    },
    hasDelivered: () => delivered,
  };
}
