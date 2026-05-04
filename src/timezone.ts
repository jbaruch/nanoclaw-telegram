/**
 * Check whether a timezone string is a valid IANA identifier
 * that Intl.DateTimeFormat can use.
 */
export function isValidTimezone(tz: string): boolean {
  try {
    Intl.DateTimeFormat(undefined, { timeZone: tz });
    return true;
  } catch {
    return false;
  }
}

/**
 * Return the given timezone if valid IANA, otherwise fall back to UTC.
 */
export function resolveTimezone(tz: string): string {
  return isValidTimezone(tz) ? tz : 'UTC';
}

/**
 * Convert a UTC ISO timestamp to a localized display string.
 * Uses the Intl API (no external dependencies).
 * Falls back to UTC if the timezone is invalid.
 */
export function formatLocalTime(utcIso: string, timezone: string): string {
  const date = new Date(utcIso);
  return date.toLocaleString('en-US', {
    timeZone: resolveTimezone(timezone),
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

export type ScheduleType = 'cron' | 'interval' | 'once';

export type ScheduleTimezoneOutcome =
  | { action: 'accept'; value: string | null }
  | { action: 'ignore-non-cron' }
  | { action: 'reject-invalid' };

/**
 * Normalize a `schedule_timezone` payload from a `schedule_task` /
 * `update_task` IPC call. Three terminal outcomes:
 *
 *   - `accept`: the value is safe to persist to `scheduled_tasks`. May
 *     be `null` (no per-task tz; fall back to TIMEZONE), an IANA name,
 *     or the literal token `'local'` (#456 — resolved at fire time
 *     against `tz_state.current_tz` by `task-scheduler.ts`'s
 *     `computeNextRunDetailed`, so the row travels with the owner
 *     without `schedule_value` mutation).
 *   - `ignore-non-cron`: the field has no effect for `interval` / `once`
 *     schedule types; drop silently rather than failing the call. The
 *     caller logs the warning to preserve message-level diagnostics.
 *   - `reject-invalid`: a non-empty, non-`'local'` string that
 *     `Intl.DateTimeFormat` does not recognize. Caller logs and aborts
 *     the call so a typo does not silently fall through to TIMEZONE.
 */
export function normalizeScheduleTimezone(
  input: string | null | undefined,
  scheduleType: ScheduleType,
): ScheduleTimezoneOutcome {
  if (input === undefined || input === null || input === '') {
    return { action: 'accept', value: null };
  }
  if (scheduleType !== 'cron') {
    return { action: 'ignore-non-cron' };
  }
  if (input === 'local') {
    return { action: 'accept', value: 'local' };
  }
  if (!isValidTimezone(input)) {
    return { action: 'reject-invalid' };
  }
  return { action: 'accept', value: input };
}
