import { CronExpressionParser } from 'cron-parser';

import { TIMEZONE } from '../config.js';
import { coerceTaskTextField } from '../coerce-task-prompt.js';
import {
  createTask,
  deleteTask,
  getTaskById,
  updateTask,
} from '../db-tasks.js';
import { getCurrentTz } from '../db-tz.js';
import { registerIpcHandler } from '../ipc-registry.js';
import { logger } from '../logger.js';
import {
  isValidTimezone,
  normalizeScheduleTimezone,
  type ScheduleType,
} from '../timezone.js';

/**
 * Resolve the IANA zone to pass to `cron-parser` when computing an
 * initial `next_run` for a stored task row. Centralised so
 * `schedule_task` and `update_task` produce values consistent with
 * `task-scheduler.ts:computeNextRunDetailed` at fire time.
 *
 * Three inputs:
 *   - `null`         → server-wide TIMEZONE fallback (pre-#102 behaviour
 *                       for rows without a per-task tz).
 *   - IANA name      → pass through unchanged (validated upstream by
 *                       `normalizeScheduleTimezone`).
 *   - `'local'`      → resolve via `getCurrentTz()` — the singleton
 *                       `tz_state.current_tz` written by `task-tz-sync`.
 *                       If the resolver returns null (no tz_state row
 *                       yet, or unfamiliar `schema_version`) OR returns
 *                       a value `Intl.DateTimeFormat` doesn't recognize
 *                       (corrupt write), fall back to TIMEZONE. Same
 *                       fallback the scheduler reaches via its
 *                       catch-and-retry path — narrowed here to a
 *                       sanity check up-front so a corrupt resolver
 *                       result doesn't brick `schedule_task` /
 *                       `update_task` with an "Invalid cron expression"
 *                       error that's actually a tz_state problem.
 */
function resolveCronTz(scheduleTimezone: string | null): string {
  if (scheduleTimezone === 'local') {
    const resolved = getCurrentTz();
    if (resolved && isValidTimezone(resolved)) return resolved;
    logger.warn(
      { resolved },
      'resolveCronTz: tz_state.current_tz unusable for cron-parser — falling back to TIMEZONE',
    );
    return TIMEZONE;
  }
  return scheduleTimezone || TIMEZONE;
}

/**
 * Scheduled-task lifecycle commands (#845 slice 1): schedule / pause /
 * resume / cancel / update. Authorization is finer-grained than the
 * dispatcher's `requiresMain` gate — main can act on any task, a
 * non-main group only on its own — so each handler gates internally.
 */
export function registerTaskIpcHandlers(): void {
  registerIpcHandler('schedule_task', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      if (
        data.prompt &&
        data.schedule_type &&
        data.schedule_value &&
        data.targetJid
      ) {
        // #512 — coerce `prompt` and `script` to text at the IPC
        // boundary. TS declares `string`, but the value is
        // deserialized from JSON: a non-string here would otherwise
        // bind as a BLOB via better-sqlite3 and surface later as
        // `t.prompt.slice is not a function` in `list_tasks`.
        // `coerceTaskTextField` returns:
        //   - the string itself when `data.prompt` is a string,
        //   - decoded UTF-8 when it's the JSON-Buffer shape
        //     (`{type:'Buffer',data:[...]}`),
        //   - `null` for any other shape — signal to reject the
        //     payload rather than persist a garbage row that would
        //     fire with `"[object Object]"` as its prompt.
        const promptStr = coerceTaskTextField(data.prompt);
        if (promptStr === null) {
          logger.warn(
            { sourceGroup, promptType: typeof data.prompt },
            'schedule_task: rejecting non-text prompt payload',
          );
          return;
        }
        let scriptStr: string | null = null;
        if (data.script != null) {
          const coerced = coerceTaskTextField(data.script);
          if (coerced === null) {
            logger.warn(
              { sourceGroup, scriptType: typeof data.script },
              'schedule_task: rejecting non-text script payload',
            );
            return;
          }
          // Treat empty-string script the same as null — same
          // semantics as the prior `data.script || null`.
          scriptStr = coerced || null;
        }
        // Resolve the target group from JID
        const targetJid = data.targetJid as string;
        const targetGroupEntry = registeredGroups[targetJid];

        if (!targetGroupEntry) {
          logger.warn(
            { targetJid },
            'Cannot schedule task: target group not registered',
          );
          return;
        }

        const targetFolder = targetGroupEntry.folder;

        // Authorization: non-main groups can only schedule for themselves
        if (!isMain && targetFolder !== sourceGroup) {
          logger.warn(
            { sourceGroup, targetFolder },
            'Unauthorized schedule_task attempt blocked',
          );
          return;
        }

        const scheduleType = data.schedule_type as 'cron' | 'interval' | 'once';

        // #102: optional timezone parameter. Validated up-front so a
        // typo fails the schedule call rather than silently falling
        // back to server-local at fire time. #456 extends the accepted
        // values with the literal token `'local'` — resolved at fire
        // time against `tz_state.current_tz` by
        // `task-scheduler.ts:computeNextRunDetailed`.
        //
        // Force `null` for non-cron types: the column has no effect on
        // `interval` (always elapsed-ms) or `once` (instant pinned at
        // schedule time). Persisting it for those types would be a
        // footgun if the task were later updated to `cron` without
        // explicitly passing `timezone` — an old, previously-ignored
        // value would silently start affecting cron evaluation.
        const tzOutcome = normalizeScheduleTimezone(
          data.timezone,
          scheduleType,
        );
        if (tzOutcome.action === 'reject-invalid') {
          logger.warn(
            { timezone: data.timezone },
            'Invalid IANA timezone for schedule_task',
          );
          return;
        }
        if (tzOutcome.action === 'ignore-non-cron') {
          logger.warn(
            { timezone: data.timezone, scheduleType },
            'schedule_task: timezone parameter is only meaningful for cron — ignoring',
          );
        }
        const scheduleTimezone: string | null =
          tzOutcome.action === 'accept' ? tzOutcome.value : null;

        let nextRun: string | null = null;
        if (scheduleType === 'cron') {
          try {
            // #456: resolve `'local'` against `tz_state.current_tz` to
            // mirror `computeNextRunDetailed`. The scheduler retries
            // with TIMEZONE on cron-parser failure; we narrow the
            // failure surface here by sanity-checking the resolved
            // value up-front so a corrupt `tz_state.current_tz` (e.g.
            // a future task-tz-sync bug writing garbage) doesn't brick
            // schedule_task — falls through to TIMEZONE just like NULL
            // `schedule_timezone` and like the scheduler's retry path.
            const cronTz = resolveCronTz(scheduleTimezone);
            const interval = CronExpressionParser.parse(data.schedule_value, {
              tz: cronTz,
            });
            nextRun = interval.next().toISOString();
          } catch (err) {
            // Bind + filter rather than catch-all per
            // `jbaruch/coding-policy: error-handling`. CronExpressionParser
            // throws plain Error instances on invalid syntax; anything
            // non-Error here is a bug somewhere else (e.g. a `throw "str"`
            // upstream) and should propagate.
            if (!(err instanceof Error)) throw err;
            logger.warn(
              { err: err.message, scheduleValue: data.schedule_value },
              'Invalid cron expression',
            );
            return;
          }
        } else if (scheduleType === 'interval') {
          const MIN_INTERVAL_MS = 60_000;
          const ms = parseInt(data.schedule_value, 10);
          if (isNaN(ms) || ms < MIN_INTERVAL_MS) {
            logger.warn(
              { scheduleValue: data.schedule_value, minMs: MIN_INTERVAL_MS },
              'Invalid interval: must be at least 60s',
            );
            return;
          }
          nextRun = new Date(Date.now() + ms).toISOString();
        } else if (scheduleType === 'once') {
          const date = new Date(data.schedule_value);
          if (isNaN(date.getTime())) {
            logger.warn(
              { scheduleValue: data.schedule_value },
              'Invalid timestamp',
            );
            return;
          }
          nextRun = date.toISOString();
        }

        const taskId =
          data.taskId ||
          `task-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const contextMode =
          data.context_mode === 'group' || data.context_mode === 'isolated'
            ? data.context_mode
            : 'isolated';
        // Provenance: derived from the VERIFIED source group's trust tier
        // (sourceGroup and isMain are set from the IPC directory path, not
        // from untrusted payload fields). The agent that scheduled the
        // task NEVER gets to claim its own role — this is the security
        // boundary that keeps an untrusted group from self-scheduling a
        // prompt that later fires unwrapped as if it were trusted.
        const sourceGroupEntry = Object.values(registeredGroups).find(
          (g) => g.folder === sourceGroup,
        );
        const createdByRole:
          | 'main_agent'
          | 'trusted_agent'
          | 'untrusted_agent' = isMain
          ? 'main_agent'
          : sourceGroupEntry?.containerConfig?.trusted
            ? 'trusted_agent'
            : 'untrusted_agent';
        // Optional continuation marker (#93/#130). Set by the
        // resumable-cycle helper skill when scheduling the next link of a
        // self-resuming cycle chain; the task-scheduler reads it at fire
        // time and plumbs the matching env vars onto the spawned
        // container. Untyped non-string values are dropped — the field is
        // a free-form opaque slot key (per the proposal: UTC date for
        // nightly/morning-brief, ISO week for weekly), but we never want
        // a stray number / object to land in the DB column.
        let continuationCycleId: string | null = null;
        if (
          typeof data.continuation_cycle_id === 'string' &&
          data.continuation_cycle_id.length > 0
        ) {
          continuationCycleId = data.continuation_cycle_id;
        }
        createTask({
          id: taskId,
          group_folder: targetFolder,
          chat_jid: targetJid,
          prompt: promptStr,
          script: scriptStr,
          schedule_type: scheduleType,
          schedule_value: data.schedule_value,
          schedule_timezone: scheduleTimezone,
          context_mode: contextMode,
          next_run: nextRun,
          status: 'active',
          created_at: new Date().toISOString(),
          created_by_role: createdByRole,
          continuation_cycle_id: continuationCycleId,
        });
        logger.info(
          {
            taskId,
            sourceGroup,
            targetFolder,
            contextMode,
            createdByRole,
            continuationCycleId,
          },
          'Task created via IPC',
        );
        deps.onTasksChanged();
      }
    },
  });

  registerIpcHandler('pause_task', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (task && (isMain || task.group_folder === sourceGroup)) {
          updateTask(data.taskId, { status: 'paused' });
          logger.info(
            { taskId: data.taskId, sourceGroup },
            'Task paused via IPC',
          );
          deps.onTasksChanged();
        } else {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task pause attempt',
          );
        }
      }
    },
  });

  registerIpcHandler('resume_task', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (task && (isMain || task.group_folder === sourceGroup)) {
          updateTask(data.taskId, { status: 'active' });
          logger.info(
            { taskId: data.taskId, sourceGroup },
            'Task resumed via IPC',
          );
          deps.onTasksChanged();
        } else {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task resume attempt',
          );
        }
      }
    },
  });

  registerIpcHandler('cancel_task', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (task && (isMain || task.group_folder === sourceGroup)) {
          deleteTask(data.taskId);
          logger.info(
            { taskId: data.taskId, sourceGroup },
            'Task cancelled via IPC',
          );
          deps.onTasksChanged();
        } else {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task cancel attempt',
          );
        }
      }
    },
  });

  registerIpcHandler('update_task', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (!task) {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Task not found for update',
          );
          return;
        }
        if (!isMain && task.group_folder !== sourceGroup) {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task update attempt',
          );
          return;
        }

        const updates: Parameters<typeof updateTask>[1] = {};
        // #512 — same boundary-coercion contract as schedule_task
        // above. `coerceTaskTextField` decodes the JSON-Buffer shape
        // so an older writer's BLOB-shaped payload still round-trips
        // to its real text; any other non-string shape is rejected
        // (we abort the whole update rather than write a garbage
        // column).
        if (data.prompt !== undefined) {
          const promptCoerced = coerceTaskTextField(data.prompt);
          if (promptCoerced === null) {
            logger.warn(
              {
                taskId: data.taskId,
                sourceGroup,
                promptType: typeof data.prompt,
              },
              'update_task: rejecting non-text prompt payload',
            );
            return;
          }
          updates.prompt = promptCoerced;
        }
        if (data.script !== undefined) {
          if (data.script == null) {
            updates.script = null;
          } else {
            const scriptCoerced = coerceTaskTextField(data.script);
            if (scriptCoerced === null) {
              logger.warn(
                {
                  taskId: data.taskId,
                  sourceGroup,
                  scriptType: typeof data.script,
                },
                'update_task: rejecting non-text script payload',
              );
              return;
            }
            // Empty string is the documented "clear back to no script"
            // signal — preserve that semantics.
            updates.script = scriptCoerced || null;
          }
        }
        if (data.schedule_type !== undefined)
          updates.schedule_type = data.schedule_type as
            | 'cron'
            | 'interval'
            | 'once';
        if (data.schedule_value !== undefined)
          updates.schedule_value = data.schedule_value;

        // #102: optional timezone update. `null`/empty-string clears
        // (back to TIMEZONE default); a non-null IANA string overrides.
        // Only meaningful for cron tasks: if the task IS a cron (or is
        // being changed to cron in this same update), accept and
        // persist; otherwise force the value to null so we don't store
        // a stray timezone that would silently start affecting cron
        // evaluation if the task were later switched to cron without
        // explicitly re-passing it.
        const effectiveScheduleType =
          updates.schedule_type ?? task.schedule_type;

        // If the schedule_type is being changed AWAY from cron AND the
        // existing row had a stored schedule_timezone, drop the stored
        // value too — even if the caller didn't explicitly pass
        // `timezone`. Otherwise a once/interval task can outlive a
        // previous cron incarnation with a stray timezone column that
        // would re-activate if the task were later flipped back to
        // cron without re-stating tz. (Copilot review round 2.)
        if (
          updates.schedule_type !== undefined &&
          updates.schedule_type !== 'cron' &&
          task.schedule_timezone &&
          data.timezone === undefined
        ) {
          updates.schedule_timezone = null;
        }

        if (data.timezone !== undefined) {
          // Same normalizer as schedule_task above — accepts null /
          // empty / IANA / `'local'` (#456); ignores tz on non-cron
          // (caller logs and forces null); rejects unrecognized
          // strings (caller logs and aborts).
          const updateTzOutcome = normalizeScheduleTimezone(
            data.timezone,
            effectiveScheduleType as ScheduleType,
          );
          if (updateTzOutcome.action === 'reject-invalid') {
            logger.warn(
              { taskId: data.taskId, timezone: data.timezone },
              'Invalid IANA timezone in task update',
            );
            return;
          }
          if (updateTzOutcome.action === 'ignore-non-cron') {
            logger.warn(
              {
                taskId: data.taskId,
                timezone: data.timezone,
                effectiveScheduleType,
              },
              'update_task: ignoring timezone — effective schedule_type is not cron',
            );
            updates.schedule_timezone = null;
          } else {
            updates.schedule_timezone = updateTzOutcome.value;
          }
        }

        // Recompute next_run if a recompute-relevant field changed.
        // Use `!== undefined` (not truthiness) for `schedule_value`
        // because an empty string IS a valid input on the wire (the
        // host catches it below as invalid) — truthy-skip would
        // silently leave next_run stale on a malformed update. For
        // `timezone`, only count it as a recompute trigger when the
        // (effective) schedule_type is cron — a timezone-only update
        // on a once/interval task has no effect on next_run.
        const triggerRecompute =
          data.schedule_type !== undefined ||
          data.schedule_value !== undefined ||
          (data.timezone !== undefined && effectiveScheduleType === 'cron');
        if (triggerRecompute) {
          const updatedTask = {
            ...task,
            ...updates,
          };
          if (updatedTask.schedule_type === 'cron') {
            try {
              // #456: same `'local'`-aware resolver as the schedule_task
              // path — sanity-checks `tz_state.current_tz` before
              // passing to cron-parser so a corrupt resolver result
              // falls back to TIMEZONE rather than aborting the update.
              const cronTz = resolveCronTz(
                updatedTask.schedule_timezone ?? null,
              );
              const interval = CronExpressionParser.parse(
                updatedTask.schedule_value,
                { tz: cronTz },
              );
              updates.next_run = interval.next().toISOString();
            } catch (err) {
              // See schedule_task above — same Error-or-rethrow pattern
              // per `jbaruch/coding-policy: error-handling`.
              if (!(err instanceof Error)) throw err;
              logger.warn(
                {
                  err: err.message,
                  taskId: data.taskId,
                  value: updatedTask.schedule_value,
                },
                'Invalid cron in task update',
              );
              return;
            }
          } else if (updatedTask.schedule_type === 'interval') {
            const MIN_INTERVAL_MS = 60_000;
            const ms = parseInt(updatedTask.schedule_value, 10);
            if (!isNaN(ms) && ms >= MIN_INTERVAL_MS) {
              updates.next_run = new Date(Date.now() + ms).toISOString();
            } else if (!isNaN(ms)) {
              logger.warn(
                {
                  taskId: data.taskId,
                  value: updatedTask.schedule_value,
                  minMs: MIN_INTERVAL_MS,
                },
                'Invalid interval in task update: must be at least 60s',
              );
              return;
            }
          } else if (updatedTask.schedule_type === 'once') {
            // #102 follow-up: if a once-task's schedule_value changes
            // (or the type flips to 'once'), recompute next_run from
            // the new timestamp. Without this branch the row would
            // keep its old `next_run` and fire incorrectly.
            const date = new Date(updatedTask.schedule_value);
            if (isNaN(date.getTime())) {
              logger.warn(
                { taskId: data.taskId, value: updatedTask.schedule_value },
                'Invalid once timestamp in task update',
              );
              return;
            }
            updates.next_run = date.toISOString();
          }
        }

        updateTask(data.taskId, updates);
        logger.info(
          { taskId: data.taskId, sourceGroup, updates },
          'Task updated via IPC',
        );
        deps.onTasksChanged();
      }
    },
  });
}
