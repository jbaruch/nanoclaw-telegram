import { CronExpressionParser } from 'cron-parser';
import { TIMEZONE } from './config.js';
import { isCronInputError } from './operational-errors.js';

/** One runtime-owned schedule grammar for IPC create/update and stored-task reads. */
export type ParsedTaskSchedule =
  | { type: 'cron'; nextRun: string }
  | { type: 'interval'; nextRun: string; intervalMs: number }
  | { type: 'once'; nextRun: string };

export function parseTaskSchedule(
  type: string,
  value: string,
  now = Date.now(),
): ParsedTaskSchedule {
  if (typeof value !== 'string' || !value.trim())
    throw new SyntaxError('Task schedule_value must be a nonempty string.');
  if (type === 'cron') {
    try {
      const nextRun = CronExpressionParser.parse(value, {
        tz: TIMEZONE,
        currentDate: now,
      })
        .next()
        .toISOString();
      if (!nextRun)
        throw new SyntaxError(
          'Task cron schedule exceeds the supported date range.',
        );
      return { type, nextRun };
    } catch (error) {
      if (!isCronInputError(error)) throw error;
      throw new SyntaxError(
        'Task cron schedule is invalid; repair it before submitting.',
        { cause: error },
      );
    }
  }
  if (type === 'interval') {
    const ms = Number(value);
    if (
      !/^[1-9][0-9]*$/.test(value) ||
      !Number.isSafeInteger(ms) ||
      !Number.isFinite(new Date(now + ms).getTime())
    )
      throw new SyntaxError(
        'Task interval must be positive integer milliseconds within the supported date range.',
      );
    return { type, intervalMs: ms, nextRun: new Date(now + ms).toISOString() };
  }
  if (type === 'once') {
    const fields =
      /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})?$/.exec(
        value,
      );
    if (!fields)
      throw new SyntaxError('Task once schedule must be an ISO timestamp.');
    const [year, month, day, hour, minute, second] = fields
      .slice(1)
      .map(Number);
    const calendar = new Date(0);
    calendar.setUTCFullYear(year, month, 0);
    const timestamp = new Date(value);
    if (
      month < 1 ||
      month > 12 ||
      day < 1 ||
      day > calendar.getUTCDate() ||
      hour > 23 ||
      minute > 59 ||
      second > 59 ||
      !Number.isFinite(timestamp.getTime())
    )
      throw new SyntaxError('Task once schedule must be a valid timestamp.');
    return { type, nextRun: timestamp.toISOString() };
  }
  throw new SyntaxError('Unsupported task schedule type.');
}
