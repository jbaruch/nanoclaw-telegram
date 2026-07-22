// Session-slot naming (#851 slice 5, extracted verbatim from
// src/container-runner.ts so volume-mounts and container-runner can
// both import it without a cycle).

/**
 * Default `sessionName` when callers don't pass one. User-facing paths
 * (inbound IPC messages) resolve here. Scheduled tasks pass `'maintenance'`
 * to get a parallel container slot. See `ContainerInput.sessionName` docs.
 */
export const DEFAULT_SESSION_NAME = 'default';

/**
 * Canonical session name for scheduled work (heartbeat, nightly, weekly,
 * reminders). `src/task-scheduler.ts` is the sole writer of this value;
 * no inbound path ever reaches it. Defined here (not in `group-queue.ts`
 * where it used to live) so the install-loop in `buildVolumeMounts`
 * can reference it directly without creating a
 * `container-runner ↔ group-queue` import cycle (#337 review). The
 * symbol is re-exported from `group-queue.ts` for callers that already
 * import it from there — no other file needs to change.
 */
export const MAINTENANCE_SESSION_NAME = 'maintenance';

/**
 * Per-session subdir name under `<DATA_DIR>/ipc/<folder>/` for the input
 * side of the IPC channel. Each session gets its own subdir so `_close`
 * sentinels and follow-up JSON messages written for one session never
 * leak into the other session's container.
 *
 * Exported so group-queue writes to the same path the container-runner
 * mounted — both must agree on the location. Kept in sync at compile time.
 *
 * Session name is validated here so every caller (orchestrator-trusted
 * and IPC-untrusted alike) gets the same guard. A malicious container
 * that manages to stamp `sessionName: "../default"` onto its IPC request
 * would, without this check, redirect `scriptResultPath` or mount
 * construction into a directory outside the expected `ipc/<group>/`
 * subtree. The allowlist pattern is deliberately narrow — `default`,
 * `maintenance`, and any hypothetical future slot all fit within
 * `[A-Za-z0-9_-]+`.
 */
export const VALID_SESSION_NAME_RE = /^[A-Za-z0-9_-]+$/;
export function sessionInputDirName(sessionName: string): string {
  if (!VALID_SESSION_NAME_RE.test(sessionName)) {
    throw new Error(
      `Invalid session name: ${JSON.stringify(sessionName)} — must match ${VALID_SESSION_NAME_RE}`,
    );
  }
  return `input-${sessionName}`;
}
