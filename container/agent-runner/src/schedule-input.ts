import { CronDate, CronExpressionParser } from 'cron-parser';

/** Offset-free wall times use the configured IANA zone. Check nearby native
 * offsets and require exactly one matching instant; DST gaps/folds require an
 * explicit offset instead of silently normalizing or guessing a fold.
 */
function onceInstant(value: string, timezone: string): string {
  if (/(?:Z|[+-]\d{2}:\d{2})$/.test(value))
    return new Date(value).toISOString();
  const wall = new Date(value + 'Z');
  const fields = [
    wall.getUTCFullYear(),
    wall.getUTCMonth(),
    wall.getUTCDate(),
    wall.getUTCHours(),
    wall.getUTCMinutes(),
    wall.getUTCSeconds(),
    wall.getUTCMilliseconds(),
  ];
  const offsets = new Set<number>();
  for (let hours = -48; hours <= 48; hours += 6)
    offsets.add(
      new CronDate(
        new Date(wall.getTime() + hours * 3600000),
        timezone,
      ).getUTCOffset(),
    );
  const matches: number[] = [];
  for (const offset of offsets) {
    const instant = wall.getTime() - offset * 60000;
    const local = new CronDate(new Date(instant), timezone);
    const observed = [
      local.getFullYear(),
      local.getMonth(),
      local.getDate(),
      local.getHours(),
      local.getMinutes(),
      local.getSeconds(),
      local.getMilliseconds(),
    ];
    if (observed.every((field, index) => field === fields[index]))
      matches.push(instant);
  }
  if (matches.length !== 1)
    throw new SyntaxError(
      'Local once time is nonexistent or ambiguous in the configured timezone; provide an explicit offset.',
    );
  return new Date(matches[0]).toISOString();
}

/** Mirrors the host's accepted lexical grammar; host remains mutation authority. */
export function scheduleInputError(
  type: string,
  value: string,
  now = Date.now(),
): string | null {
  if (typeof value !== 'string' || !value.trim())
    return 'Schedule value must be a nonempty string.';
  if (type === 'interval') {
    const ms = Number(value);
    return /^[1-9][0-9]*$/.test(value) &&
      Number.isSafeInteger(ms) &&
      Number.isFinite(new Date(now + ms).getTime())
      ? null
      : 'Interval must be positive integer milliseconds within the supported date range.';
  }
  if (type === 'once') {
    const fields =
      /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})?$/.exec(
        value,
      );
    if (!fields) return 'Once schedule requires an ISO timestamp with seconds.';
    const [year, month, day, hour, minute, second] = fields
      .slice(1)
      .map(Number);
    const calendar = new Date(0);
    calendar.setUTCFullYear(year, month, 0);
    if (
      !(
        month >= 1 &&
        month <= 12 &&
        day >= 1 &&
        day <= calendar.getUTCDate() &&
        hour <= 23 &&
        minute <= 59 &&
        second <= 59 &&
        Number.isFinite(new Date(value).getTime())
      )
    )
      return 'Once schedule must be a valid timestamp.';
    try {
      onceInstant(
        value,
        process.env.TZ ?? Intl.DateTimeFormat().resolvedOptions().timeZone,
      );
      return null;
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      return error.message;
    }
  }
  if (type === 'cron') {
    try {
      return CronExpressionParser.parse(value, { currentDate: now })
        .next()
        .toISOString()
        ? null
        : 'Cron schedule exceeds the supported date range.';
    } catch (error) {
      if (
        !(error instanceof Error) ||
        error.constructor !== Error ||
        !/^(Invalid (cron expression|characters,|range:|step:|list value format|repeat:|explicit day of month definition|last weekday of the month expression:)|Validation error,|Constraint error,|Cron(?:Second|Minute|Hour|DayOfMonth|Month|DayOfWeek) Validation error,|Out of the time span range|Invalid time value)/.test(
          error.message,
        )
      )
        throw error;
      return 'Cron schedule is invalid.';
    }
  }
  return 'Unsupported schedule type.';
}
