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
