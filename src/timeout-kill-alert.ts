/**
 * #890 follow-up — immediate operator alert when a scheduled task is
 * killed on a timeout.
 *
 * Two timeouts can kill a run and both alert here: the precheck's own
 * declared `precheck_timeout_ms` budget, and the host's container kill
 * (`containerConfig.timeout`, falling back to
 * `MAINTENANCE_CONTAINER_TIMEOUT`). Both stamp `ContainerOutput
 * .timedOut`, which is what this alert keys off — never the wording of
 * `task_run_logs.error`, which the three emitters phrase differently
 * and are free to reword.
 *
 * Why this is immediate rather than folded into the heartbeat's
 * existing task-failure report: that check queries `status = 'error'`
 * only (so it never saw a `'killed'` container timeout at all), and it
 * suppresses any failure the task has since recovered from. A skill on
 * a tight cadence recovers within a cycle or two, so its timeout kills
 * were structurally invisible — `flight-assist` fires every 2 minutes
 * and would self-suppress long before the 30-minute heartbeat looked.
 * Relaxing that gate would make the heartbeat noisier for every other
 * failure class, so the timeout kill gets its own path and the
 * heartbeat keeps reporting persistent failures the way it does today.
 *
 * A timeout kill remains recorded in `task_run_logs` exactly as before;
 * this only adds the notification.
 */

/** Inputs the alert text is built from. */
export interface TimeoutKillAlertInput {
  /** Scheduled-task id, e.g. `cadence-registry::…::tessl__flight-assist`. */
  taskId: string;
  /** Skill the task invokes, when the prompt named one. */
  skillName?: string;
  /** Wall-clock duration of the killed run. */
  durationMs: number;
  /** `task_run_logs.status` the run was recorded under. */
  runStatus: string;
  /** The reason persisted to `task_run_logs.error`, if any. */
  error?: string | null;
}

/** Truncate an emitter's error prose so one alert can't flood a chat. */
const MAX_REASON_CHARS = 300;

/**
 * Build the operator-facing alert text for a timeout kill.
 *
 * Pure so the wording is unit-testable without spinning a container or
 * a channel. Deliberately plain text: it crosses whatever channel the
 * group is on, and channel-specific markup would render as literal
 * characters on the others.
 */
export function buildTimeoutKillAlert(input: TimeoutKillAlertInput): string {
  const { taskId, skillName, durationMs, runStatus, error } = input;
  const seconds = (durationMs / 1000).toFixed(1);
  const what = skillName ? `${skillName} (${taskId})` : taskId;
  const lines = [
    `⏱️ Timeout kill: ${what}`,
    `Ran ${seconds}s, recorded as ${runStatus}.`,
  ];
  if (error) {
    const reason =
      error.length > MAX_REASON_CHARS
        ? `${error.slice(0, MAX_REASON_CHARS)}…`
        : error;
    lines.push(reason);
  }
  return lines.join('\n');
}

/**
 * Whether a finished run should raise the alert.
 *
 * `timedOut` alone is the predicate — the emitters already decide what
 * counts. In particular a container that idles out AFTER delivering its
 * result is not stamped, because nothing was killed mid-work and every
 * healthy maintenance run ends that way.
 */
export function shouldAlertTimeoutKill(timedOut: boolean): boolean {
  return timedOut;
}
