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
 */
export function makeUncaughtEpipeHandler(
  hasDeliveredTerminalResult: () => boolean,
  exit: ExitFn,
): (err: NodeJS.ErrnoException) => void {
  return (err) => {
    if (err.code === 'EPIPE') {
      exit(hasDeliveredTerminalResult() ? 0 : 1);
      return;
    }
    throw err;
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
