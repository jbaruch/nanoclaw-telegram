import { ChildProcess } from 'child_process';
import { SqliteError } from 'better-sqlite3';
import { CronExpressionParser } from 'cron-parser';
import fs from 'fs';
import path from 'path';

import {
  ASSISTANT_NAME,
  DATA_DIR,
  MODEL_CONTEXT_WINDOW,
  SCHEDULER_POLL_INTERVAL,
  TIMEZONE,
} from './config.js';
import {
  ContainerOutput,
  runContainerAgent,
  writeTasksSnapshot,
} from './container-runner.js';
import { MAINTENANCE_SESSION_NAME } from './group-queue.js';
import { computeThresholds } from './threshold.js';
import { emitSessionTokens } from './usage-telemetry.js';
import {
  clearTaskSessionId,
  getActiveLocalScheduledTasks,
  getAllTasks,
  getCurrentTz,
  getDormantRecurringTasks,
  getDueTasks,
  resurrectZombieTasks,
  getTaskById,
  logTaskRun,
  pruneCompletedTasks,
  setTaskNextRun,
  setTaskSessionId,
  shouldStoreBotMessage,
  storeChatMetadata,
  storeMessage,
  updateTask,
  updateTaskAfterRun,
} from './db.js';
import { GroupQueue } from './group-queue.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { logger } from './logger.js';
import { evaluateSpawnGate } from './spawn-gates.js';
import { getPluginRegistryHash } from './plugin-content-hash.js';
import {
  pruneSessionArtifacts,
  resolveSessionArtifactRetentionConfig,
} from './session-artifact-retention.js';
import { RegisteredGroup, ScheduledTask } from './types.js';

/**
 * Extract the first `Skill(skill: "...")` invocation name from a
 * scheduled-task prompt for `taskSkill` telemetry tagging (#349). The
 * shape is fully enumerable — it's the literal SDK skill-invocation
 * syntax the orchestrator itself prepends in heartbeat / housekeeping
 * / morning-brief prompts (e.g.
 * `Skill(skill: "tessl__heartbeat")`) — so a regex is appropriate.
 *
 * Returns `undefined` when the prompt has no skill call at all (raw
 * scheduled reminders, ad-hoc one-shots), and the caller falls back
 * to `prompt[:64]` for bucketable identity.
 *
 * The prompt shape is NOT prefix-only — heartbeat prompts wrap the
 * call in a `MANDATORY FIRST ACTION:` directive — so the regex scans
 * the whole string, not just the leading characters.
 */
export function parseTaskSkill(prompt: string): string | undefined {
  const match = prompt.match(/Skill\(\s*skill:\s*["']([^"']+)["']/);
  return match?.[1];
}

/**
 * Compute the next run time for a recurring task, anchored to the
 * task's scheduled time rather than Date.now() to prevent cumulative
 * drift on interval-based tasks.
 *
 * Co-authored-by: @community-pr-601
 */
/**
 * Result type that lets callers know WHY a recurring task got
 * `nextRun: null` so they can apply remediation against the FRESH DB
 * row (avoiding races against concurrent `update_task` IPC).
 *
 * The legacy `string | null` shape is preserved by `computeNextRun`
 * for backwards compat — call `computeNextRunDetailed` to get the
 * structured result.
 */
export type NextRunRemediation =
  | 'pause-broken-cron' // both per-task tz and TIMEZONE retry failed
  | 'clear-bad-timezone'; // per-task tz failed, TIMEZONE retry succeeded

export interface NextRunResult {
  nextRun: string | null;
  remediation?: NextRunRemediation;
}

/**
 * True for a cron-parser parse failure. cron-parser throws a plain base
 * `Error` (constructor `Error`, no dedicated subclass) for an invalid
 * expression. Match that shape positively: a programmer defect is always an
 * `Error` subclass (`TypeError`, `RangeError`, …) whose `constructor` is not
 * `Error`, so it fails this predicate and propagates rather than being
 * mistaken for a bad expression.
 */
function isCronParseFailure(err: unknown): err is Error {
  return err instanceof Error && err.constructor === Error;
}

export function computeNextRunDetailed(
  task: ScheduledTask,
  resolveLocalTz?: () => string | null,
): NextRunResult {
  if (task.schedule_type === 'once') return { nextRun: null };

  const now = Date.now();

  if (task.schedule_type === 'cron') {
    // Per-task `schedule_timezone` (#102) takes precedence over the
    // server-wide TIMEZONE config. NULL/undefined falls back to TIMEZONE
    // — the pre-#102 behavior.
    //
    // The literal token `'local'` (#456) is resolved at fire time
    // against `tz_state.current_tz` via the optional `resolveLocalTz`
    // callback. Rows declared via cadence-registry frontmatter
    // `cadence: "<cron> (TZ=local)"` carry `schedule_timezone='local'`
    // and travel with the owner without row mutation. If no resolver
    // is provided (test harness, pre-#456 callers) or the resolver
    // returns null (tz_state empty / unfamiliar schema_version), fall
    // through to TIMEZONE — the same fallback NULL `schedule_timezone`
    // already uses.
    //
    // Pure function (no DB writes): a previous version called
    // `updateTask` directly here, which raced with concurrent
    // `update_task` IPC — a user fixing a broken tz could have their
    // change clobbered by a still-in-flight scheduler tick that read
    // the old value. Now we just compute and report; the caller is
    // responsible for pausing or clearing the tz against the FRESH
    // DB row.
    let effectiveTz: string;
    if (task.schedule_timezone === 'local') {
      // Wrap the resolver call: a SqliteError from the underlying DB read
      // degrades to the TIMEZONE fallback rather than propagating out of the
      // scheduler tick; any other throw is a defect and propagates. Per
      // `coding-policy: error-handling` § Graceful Fallback.
      let resolved: string | null = null;
      if (resolveLocalTz) {
        try {
          resolved = resolveLocalTz() ?? null;
        } catch (resolverErr) {
          // The resolver reads the local tz from the DB; a SqliteError degrades
          // to the TIMEZONE fallback. Anything else is a defect and propagates.
          if (!(resolverErr instanceof SqliteError)) throw resolverErr;
          logger.warn(
            {
              taskId: task.id,
              err: resolverErr.message,
            },
            'computeNextRun: resolveLocalTz threw — falling back to TIMEZONE',
          );
        }
      }
      effectiveTz = resolved ?? TIMEZONE;
    } else {
      effectiveTz = task.schedule_timezone || TIMEZONE;
    }
    try {
      const interval = CronExpressionParser.parse(task.schedule_value, {
        tz: effectiveTz,
      });
      return { nextRun: interval.next().toISOString() };
    } catch (err) {
      // An invalid cron expression retries with the server TIMEZONE below; a
      // programmer defect propagates.
      if (!isCronParseFailure(err)) throw err;
      logger.warn(
        {
          taskId: task.id,
          scheduleValue: task.schedule_value,
          scheduleTimezone: task.schedule_timezone,
          effectiveTz,
          err: err.message,
        },
        'computeNextRun: cron parse failed — retrying with server TIMEZONE',
      );
      try {
        const interval = CronExpressionParser.parse(task.schedule_value, {
          tz: TIMEZONE,
        });
        // For `schedule_timezone === 'local'`, the bad value is in
        // `tz_state.current_tz` (which `task-tz-sync` writes), not in
        // the row's `schedule_timezone` itself. Emitting
        // `clear-bad-timezone` here would mutate the row and silently
        // strip the `'local'` token — converting a travel-anchored
        // schedule into a server-TZ schedule. Suppress remediation in
        // that case: this tick falls back to TIMEZONE for a single fire,
        // and the next fire retries the resolver (which may have been
        // fixed by an updated `tz_state.current_tz` in the meantime).
        if (task.schedule_timezone === 'local') {
          return { nextRun: interval.next().toISOString() };
        }
        return {
          nextRun: interval.next().toISOString(),
          remediation: 'clear-bad-timezone',
        };
      } catch (retryErr) {
        // Even the TIMEZONE fallback failed to parse → pause the cron; a
        // programmer defect propagates.
        if (!isCronParseFailure(retryErr)) throw retryErr;
        logger.error(
          {
            taskId: task.id,
            scheduleValue: task.schedule_value,
            err: retryErr.message,
          },
          'computeNextRun: cron parse failed even with TIMEZONE fallback',
        );
        return { nextRun: null, remediation: 'pause-broken-cron' };
      }
    }
  }

  if (task.schedule_type === 'interval') {
    const ms = parseInt(task.schedule_value, 10);
    if (!ms || ms <= 0) {
      // Guard against malformed interval that would cause an infinite loop
      logger.warn(
        { taskId: task.id, value: task.schedule_value },
        'Invalid interval value',
      );
      return { nextRun: new Date(now + 60_000).toISOString() };
    }
    // Anchor to the scheduled time, not now, to prevent drift.
    // Skip past any missed intervals so we always land in the future.
    let next = new Date(task.next_run!).getTime() + ms;
    while (next <= now) {
      next += ms;
    }
    return { nextRun: new Date(next).toISOString() };
  }

  return { nextRun: null };
}

/**
 * Backwards-compat shim: `computeNextRun` retains its original
 * `string | null` shape so existing callers that don't care about
 * remediation hints continue to work. Internally delegates to
 * `computeNextRunDetailed` and discards the remediation field —
 * callers that DO need to act on remediation should call the
 * detailed variant directly and apply the remediation against the
 * fresh DB row (re-fetch via `getTaskById`) to avoid clobbering
 * concurrent IPC updates.
 */
export function computeNextRun(
  task: ScheduledTask,
  resolveLocalTz?: () => string | null,
): string | null {
  return computeNextRunDetailed(task, resolveLocalTz).nextRun;
}

/**
 * Apply the remediation hint produced by `computeNextRunDetailed`
 * against the FRESH state of the task (re-fetched from DB). If the
 * task changed since the compute step (e.g. a concurrent
 * `update_task` fixed the cron expression or timezone), we skip the
 * remediation — the caller's fix wins.
 */
export function applyComputeNextRunRemediation(
  taskId: string,
  remediation: NextRunRemediation,
  observedScheduleValue: string,
  observedScheduleTimezone: string | null | undefined,
): void {
  const fresh = getTaskById(taskId);
  if (!fresh) return;
  // If the user updated the task between compute and now, the values
  // we'd be remediating against are no longer the source of the
  // failure. Skip — let the next scheduler tick re-evaluate.
  if (
    fresh.schedule_value !== observedScheduleValue ||
    (fresh.schedule_timezone ?? null) !== (observedScheduleTimezone ?? null)
  ) {
    logger.info(
      { taskId, remediation },
      'applyComputeNextRunRemediation: task changed since compute — skipping',
    );
    return;
  }
  if (remediation === 'pause-broken-cron') {
    updateTask(taskId, { status: 'paused' });
    logger.warn(
      { taskId },
      'Paused task — cron expression unparseable with both per-task tz and server TIMEZONE',
    );
  } else if (remediation === 'clear-bad-timezone') {
    updateTask(taskId, { schedule_timezone: null });
    logger.warn(
      { taskId, droppedTimezone: observedScheduleTimezone },
      'Dropped invalid schedule_timezone — falling back to TIMEZONE going forward',
    );
  }
}

/**
 * #584 — Compute today's UTC midnight in the given IANA timezone. Used
 * by `recomputeLocalSchedules` to gate the "already fired today"
 * decision: a row whose `last_run` falls before today's midnight (in
 * the NEW zone) is overdue and worth a catch-up warn; a row whose
 * `last_run` falls on/after that midnight has already executed for
 * today and is fine.
 *
 * Implementation note: `Intl.DateTimeFormat` with the target tz gives
 * us each calendar field; we then back-construct a UTC instant for
 * 00:00:00 of that calendar date in that zone via two-pass offset
 * resolution. The naive single-pass approach (offset at `now`) is
 * wrong on a DST-transition calendar day — if the offset at midnight
 * differs from the offset at `now` (spring-forward or fall-back falls
 * between them), the computed midnight-UTC instant is off by an hour.
 *
 * Algorithm:
 *   1. Extract the calendar year/month/day in `tz` from `now`.
 *   2. Take a candidate midnight-UTC instant `Date.UTC(year, month-1, day)`
 *      and ask the formatter what wall-clock the target tz shows at
 *      that instant.
 *   3. The residual hours/minutes/seconds (and any day shift across
 *      a tz day boundary) reveal the offset at intended midnight; the
 *      true midnight-UTC is the candidate minus that residual.
 *
 * Returns NaN if the tz string is unparseable — caller treats that as
 * "can't gate, skip the catch-up" rather than throwing.
 */
export function startOfTodayInTz(tz: string, now: Date): number {
  let formatter: Intl.DateTimeFormat;
  try {
    formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: tz,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  } catch (err) {
    // `Intl.DateTimeFormat` throws `RangeError` on an unparseable
    // `timeZone` option (the contract for an unknown / malformed IANA
    // zone string). That's the only expected failure mode here, and
    // the caller treats NaN as "can't gate, skip the catch-up". Any
    // other throw signals a programming bug (e.g. options shape error
    // from a future refactor) and propagates per
    // `coding-policy: error-handling`.
    if (!(err instanceof RangeError)) throw err;
    return NaN;
  }
  const nowParts = formatToPartsAsRecord(formatter, now);
  if (!nowParts) return NaN;
  const year = Number(nowParts.year);
  const month = Number(nowParts.month);
  const day = Number(nowParts.day);
  if (
    !Number.isFinite(year) ||
    !Number.isFinite(month) ||
    !Number.isFinite(day)
  ) {
    return NaN;
  }
  // Pass 2: ask the formatter what wall-clock the target tz reports at
  // the candidate midnight-UTC instant. The residual (wall-clock minus
  // 00:00:00 of the target calendar date) IS the offset at intended
  // midnight — which is what we need, not the offset at `now`.
  const candidateMidnightUtc = Date.UTC(year, month - 1, day);
  const candidateParts = formatToPartsAsRecord(
    formatter,
    new Date(candidateMidnightUtc),
  );
  if (!candidateParts) return NaN;
  const candYear = Number(candidateParts.year);
  const candMonth = Number(candidateParts.month);
  const candDay = Number(candidateParts.day);
  // `hour` from a 24h `Intl.DateTimeFormat` formatter can come back as
  // "24" at midnight in some locales/runtimes; normalise via mod so
  // arithmetic below is sane.
  const candHour = Number(candidateParts.hour) % 24;
  const candMinute = Number(candidateParts.minute);
  const candSecond = Number(candidateParts.second);
  if (
    !Number.isFinite(candYear) ||
    !Number.isFinite(candMonth) ||
    !Number.isFinite(candDay) ||
    !Number.isFinite(candHour) ||
    !Number.isFinite(candMinute) ||
    !Number.isFinite(candSecond)
  ) {
    return NaN;
  }
  // Reconstruct the wall-clock instant the candidate produced in the
  // target zone as a UTC time, then take the residual against the
  // candidate. Uses Date.UTC across the full calendar fields so a tz
  // boundary crossing (candidate maps to the prior/next calendar day in
  // the target zone) is captured by the residual rather than lost.
  const candWallAsUtcMs = Date.UTC(
    candYear,
    candMonth - 1,
    candDay,
    candHour,
    candMinute,
    candSecond,
  );
  const offsetAtMidnightMs = candWallAsUtcMs - candidateMidnightUtc;
  return candidateMidnightUtc - offsetAtMidnightMs;
}

/**
 * Helper: run `formatter.formatToParts(date)` and return a non-literal
 * field lookup. Returns null if the formatter throws (e.g. on a
 * pathological Date).
 */
function formatToPartsAsRecord(
  formatter: Intl.DateTimeFormat,
  date: Date,
): Record<string, string> | null {
  let parts: Intl.DateTimeFormatPart[];
  try {
    parts = formatter.formatToParts(date);
  } catch (err) {
    // `formatToParts` throws `RangeError` for a `Date` value outside
    // the formatter's supported range. Caller treats null as "can't
    // gate, skip the catch-up". Other throws (e.g. internal V8/JS
    // engine errors signalling a programming bug) propagate per
    // `coding-policy: error-handling`.
    if (!(err instanceof RangeError)) throw err;
    return null;
  }
  const byType: Record<string, string> = {};
  for (const p of parts) {
    if (p.type !== 'literal') byType[p.type] = p.value;
  }
  return byType;
}

/**
 * #584 — Recompute `next_run` for every active `schedule_timezone =
 * 'local'` row. Called from the `tz_state.current_tz` write path
 * (`applyTripitSegmentsToTzState`, `runTzHeartbeatAdvisory`) via an
 * `onTzFlipped` callback, so a tz flip mid-slot invalidates cached
 * `next_run` values immediately rather than waiting for each row to
 * elapse against the prior zone.
 *
 * Per-row error handling: `computeNextRunDetailed` is documented as
 * resilient — bad-cron and bad-tz rows return a `remediation` hint
 * (`pause-broken-cron` / `clear-bad-timezone`) without throwing. The
 * recompute invokes `applyComputeNextRunRemediation` for those rows
 * so they're paused (or their bad tz cleared) consistently with every
 * other compute caller; skipping the remediation here would strand
 * the row with `status='active'` + `next_run=null`, invisible to the
 * scheduler's `WHERE next_run <= ?` due-task filter. An unexpected
 * throw from compute signals a programming bug and propagates per
 * `coding-policy: error-handling`. `setTaskNextRun` failures are
 * narrowed to transient SQLite contention (`SQLITE_BUSY` /
 * `SQLITE_LOCKED`); those warn-and-continue because the next
 * scheduler tick (max ~60 s later) retries naturally. Persistent DB
 * faults, FK violations, and other programming bugs propagate so they
 * surface at the outer `tz_state` writer's narrowed catch (and at the
 * scheduler-tick boundary) rather than being hidden behind stale
 * schedule state across the fleet of `'local'`-scheduled rows.
 *
 * Production catch-up: `cron-parser.next()` always returns a future
 * occurrence, so the future `next_run` alone never describes "the row
 * should already have fired today in the new zone". The bug from #584
 * (owner lands in Berlin where 7am already passed → brief should fire
 * NOW, not tomorrow morning) requires asking cron-parser for the
 * PREVIOUS occurrence in the new zone via `.prev()`, comparing it
 * against `last_run` + `now`, and overriding `next_run` to `now` when
 * the prev is missed-and-fireable. The next scheduler tick (max ~60 s
 * later) then picks the row up through the standard due-task gate.
 *
 * The override target is `now` (not the prev's literal past time) so
 * the row evaluates as "due right now" on the next tick, not
 * "overdue by hours" — semantically cleaner for any downstream code
 * that reasons about `now - next_run` as freshness.
 */
export interface RecomputeLocalSchedulesResult {
  recomputed: number;
  caughtUp: number;
}

export interface DecideCatchUpInput {
  scheduleValue: string;
  scheduleTimezone: string;
  lastRun: string | null;
  prevCronTz: string;
  now: Date;
}

export interface DecideCatchUpResult {
  shouldCatchUp: boolean;
  prevOccurrence: Date | null;
}

/**
 * Pure helper for the catch-up decision so `recomputeLocalSchedules`
 * stays DB-free and the prev-occurrence logic is unit-testable in
 * isolation. Returns `shouldCatchUp: true` when the previous cron
 * occurrence in `prevCronTz` falls between `last_run` (exclusive) and
 * `now` (inclusive), within the last 24h.
 *
 *   - `prev <= last_run` → already fired (or fired later) → not catching up
 *   - `prev > now`       → cron-parser disagrees with our slot framing,
 *                          treat defensively as "no prev today" → not catching up
 *   - `now - prev > 24h` → too stale (e.g. row was paused, or no
 *                          cron occurrence fell within the last day);
 *                          let regular cadence pick it up
 *
 * The 24h window is wider than the daily-brief use case strictly
 * needs but bounds the catch-up surface so a weekly cron whose last
 * fire was 5 days ago doesn't get force-fired on a tz flip.
 */
const CATCH_UP_WINDOW_MS = 24 * 60 * 60 * 1000;

export function decideCatchUp(input: DecideCatchUpInput): DecideCatchUpResult {
  // #584 — No catch-all here. By the time `recomputeLocalSchedules`
  // calls this, `computeNextRunDetailed` has already returned a
  // non-null `nextRun` for the same `scheduleValue` (the catch-up
  // gate is only consulted on successful compute paths). Cron-parse
  // failures on a syntactically valid cron string are not expected
  // to vary by `tz` — `CronExpressionParser.parse` only varies its
  // throw shape on the cron string itself, which we've already
  // validated upstream. `.prev()` on a valid daily/weekly cron with
  // a real `currentDate` always has a previous occurrence. Any
  // remaining throw therefore signals a programming bug (corrupted
  // input shape, cron-parser internal invariant violation) and must
  // propagate per `coding-policy: error-handling`. Catching it here
  // would silently anchor rows to the "no catch-up" branch and hide
  // the bug behind stale schedule state.
  const interval = CronExpressionParser.parse(input.scheduleValue, {
    tz: input.prevCronTz,
    currentDate: input.now,
  });
  const prev = interval.prev().toDate();
  const prevMs = prev.getTime();
  const nowMs = input.now.getTime();
  if (prevMs > nowMs) {
    return { shouldCatchUp: false, prevOccurrence: prev };
  }
  if (nowMs - prevMs > CATCH_UP_WINDOW_MS) {
    return { shouldCatchUp: false, prevOccurrence: prev };
  }
  const lastRunMs = input.lastRun ? Date.parse(input.lastRun) : 0;
  // last_run unparseable → treat as never-fired (lastRunMs stays 0)
  const lastForCompare =
    Number.isFinite(lastRunMs) && lastRunMs > 0 ? lastRunMs : 0;
  if (lastForCompare >= prevMs) {
    return { shouldCatchUp: false, prevOccurrence: prev };
  }
  return { shouldCatchUp: true, prevOccurrence: prev };
}

/**
 * #584 — SQLite error codes the `setTaskNextRun` writer treats as
 * recoverable contention. Same set as the `tz_state` writer callbacks
 * in `src/db.ts`; see that constant for the full rationale. Every
 * other error (programming bug, persistent DB failure, malformed
 * schema) propagates per `coding-policy: error-handling`.
 */
const RECOMPUTE_TRANSIENT_SQLITE_CODES: ReadonlySet<string> = new Set([
  'SQLITE_BUSY',
  'SQLITE_LOCKED',
]);

export function recomputeLocalSchedules(
  resolveCurrentTz: () => string | null = getCurrentTz,
  now: Date = new Date(),
  deps: {
    getActiveLocalScheduledTasks?: () => ScheduledTask[];
    setTaskNextRun?: (id: string, nextRun: string | null) => void;
    applyRemediation?: (
      taskId: string,
      remediation: NextRunRemediation,
      observedScheduleValue: string,
      observedScheduleTimezone: string | null | undefined,
    ) => void;
  } = {},
): RecomputeLocalSchedulesResult {
  const readRows =
    deps.getActiveLocalScheduledTasks ?? getActiveLocalScheduledTasks;
  const writeRow = deps.setTaskNextRun ?? setTaskNextRun;
  const applyRemediation =
    deps.applyRemediation ?? applyComputeNextRunRemediation;
  const rows = readRows();
  const tzForGate = resolveCurrentTz();
  // Pre-flight the gate inputs once per recompute pass. NaN from
  // startOfTodayInTz (or null tz) → skip the catch-up branch entirely
  // per the function contract: without a usable "today midnight" we
  // can't reason about the catch-up window correctly, so default to
  // the future-cron path and let the next scheduler tick handle any
  // residual staleness.
  const midnightInNewZoneUtc = tzForGate
    ? startOfTodayInTz(tzForGate, now)
    : NaN;
  const catchUpGateUsable =
    typeof tzForGate === 'string' && Number.isFinite(midnightInNewZoneUtc);
  let recomputed = 0;
  let caughtUp = 0;
  for (const row of rows) {
    // #584 — `computeNextRunDetailed` is documented as resilient:
    // bad-cron rows enter the `paused` status via the sibling
    // `applyComputeRemediation` path (it does NOT throw on documented
    // cases), bad tz falls back to TIMEZONE env var, throwing tz
    // resolvers fall back to TIMEZONE. An unexpected throw here
    // therefore signals a programming bug (corrupted row shape,
    // schema migration mid-flight, undefined behaviour). Per
    // `coding-policy: error-handling`, programming bugs must
    // propagate — they will surface at the outer `tz_state` writer's
    // narrowed catch (which lets non-`SqliteError` propagate) and at
    // the scheduler-tick boundary. Catching them here would hide the
    // bug behind stale schedule state across the entire fleet of
    // `'local'` rows.
    const detailed = computeNextRunDetailed(row, resolveCurrentTz);
    let nextRun = detailed.nextRun;
    // Production catch-up: when the gate is usable, ask cron-parser
    // for the previous occurrence in the new zone and override
    // nextRun to `now` if the prev is missed-and-fireable. Skipped
    // entirely when the gate is unusable (NaN midnight / null tz).
    let catchUpDecided = false;
    // `computeNextRunDetailed` returning a remediation hint means the
    // row's cron / per-task tz was unparseable — invoke
    // `applyComputeNextRunRemediation` so the row is paused (or its
    // bad tz cleared) like every other compute caller does. Skipping
    // this writer would leave the row with `status='active'` +
    // `next_run=null`, invisible to the scheduler's
    // `WHERE next_run <= ?` filter — the standard due-task query
    // strands it permanently rather than surfacing it as paused for
    // the operator. The remediation re-fetches fresh row state so a
    // concurrent `update_task` fix wins. Skip the catch-up branch on
    // this path: `decideCatchUp` would re-throw the same cron-parse
    // error, and the row is being paused anyway so there's no future
    // fire to catch up.
    const remediation = detailed.remediation ?? null;
    if (remediation !== null) {
      applyRemediation(
        row.id,
        remediation,
        row.schedule_value,
        row.schedule_timezone,
      );
      continue;
    }
    if (catchUpGateUsable && row.schedule_type === 'cron') {
      const decision = decideCatchUp({
        scheduleValue: row.schedule_value,
        scheduleTimezone: row.schedule_timezone ?? 'local',
        lastRun: row.last_run,
        prevCronTz: tzForGate as string,
        now,
      });
      if (decision.shouldCatchUp) {
        nextRun = now.toISOString();
        catchUpDecided = true;
      }
    }
    try {
      writeRow(row.id, nextRun);
      recomputed += 1;
    } catch (err) {
      // Narrowed to transient SQLite contention only — SQLITE_BUSY /
      // SQLITE_LOCKED can fire under WAL contention with the
      // orchestrator's other writers and the next scheduler tick
      // will retry the recompute. Every other error (programming bug,
      // FK violation, malformed schema, persistent DB failure)
      // propagates per `coding-policy: error-handling`; hiding those
      // behind a per-row warn would silently anchor rows to stale
      // pre-flip values.
      if (
        !(err instanceof SqliteError) ||
        !RECOMPUTE_TRANSIENT_SQLITE_CODES.has(err.code)
      ) {
        throw err;
      }
      logger.warn(
        {
          taskId: row.id,
          nextRun,
          err: err.message,
          code: err.code,
        },
        'recomputeLocalSchedules: setTaskNextRun transient SQLite contention — next scheduler tick will retry against the stored value',
      );
      continue;
    }
    if (catchUpDecided) {
      caughtUp += 1;
      logger.warn(
        {
          taskId: row.id,
          scheduleValue: row.schedule_value,
          nextRun,
          lastRun: row.last_run,
          currentTz: tzForGate,
        },
        'recomputeLocalSchedules: catch-up — row had a missed occurrence today in new zone; next_run overridden to now (#584). Next scheduler tick will fire it.',
      );
    }
  }
  return { recomputed, caughtUp };
}

/**
 * Default TTL for completed once-tasks. 24h is long enough that a user
 * can still find a recently-completed task in `list_tasks` output, short
 * enough that the table doesn't grow without bound. Cancellations
 * remove rows immediately via deleteTask; this only governs the
 * natural-completion path.
 *
 * The actual TTL passed to `pruneCompletedTasks` comes from
 * `getCompletedTaskTtlMs()`, which honours the
 * `NANOCLAW_COMPLETED_TASK_TTL_MS` env override on every read so tests
 * (and ops at runtime) can flip it without a process restart.
 */
export const COMPLETED_TASK_TTL_MS = 24 * 60 * 60 * 1000;

/**
 * Resolve the active completed-task TTL: the env override
 * `NANOCLAW_COMPLETED_TASK_TTL_MS` if set to a positive integer
 * (milliseconds), otherwise the 24h default. Invalid / non-positive
 * env values fall back to the default and emit a warn log — the env
 * knob is for tuning, not for disabling the prune. Read on each
 * scheduler tick so changing the env between test cases (or via a
 * deploy-time config flip) takes effect without re-importing.
 */
export function getCompletedTaskTtlMs(): number {
  const raw = process.env.NANOCLAW_COMPLETED_TASK_TTL_MS;
  if (!raw) return COMPLETED_TASK_TTL_MS;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    logger.warn(
      { raw },
      'Invalid NANOCLAW_COMPLETED_TASK_TTL_MS; using 24h default',
    );
    return COMPLETED_TASK_TTL_MS;
  }
  return parsed;
}

/**
 * Minimum gap between successive `pruneCompletedTasks` calls. The
 * scheduler loop ticks every `SCHEDULER_POLL_INTERVAL` (seconds-scale)
 * but the prune query only deletes anything once a row has aged past
 * `COMPLETED_TASK_TTL_MS` (default 24h). Running it on every tick is
 * pure overhead — gate it to once per hour. The first tick after
 * process start always runs (see `lastPruneAt = 0` below) so we don't
 * skip the cleanup for an hour after a restart.
 */
export const PRUNE_INTERVAL_MS = 60 * 60 * 1000;

/**
 * Threshold past which an active recurring task is considered dormant
 * and worth a warn-level log. Long enough that the daily heartbeat /
 * morning-brief tasks always exceed any plausible `last_run` jitter,
 * short enough that a genuinely stuck cron is surfaced before the row
 * starts looking like it lives in the database for ornamental reasons.
 * Dormant rows are NOT auto-deleted — see `getDormantRecurringTasks`.
 */
export const DORMANT_CRON_THRESHOLD_MS = 7 * 24 * 60 * 60 * 1000;

/**
 * Per-task cooldown between consecutive dormant warnings. Without this,
 * every prune cycle (`PRUNE_INTERVAL_MS`, currently 1h) re-emits a warn
 * for the same dormant cron — 24 noisy logs/day per stuck task. One per
 * day per dormant task is enough to surface the problem without drowning
 * the log. After a process restart `lastDormantWarnAt` is empty, so the
 * first cycle warns once for every dormant task — that's the desired
 * behaviour: a fresh operator deserves to see the current state.
 */
export const DORMANT_WARN_COOLDOWN_MS = 24 * 60 * 60 * 1000;

/**
 * Tracks the last time we logged a dormant warning per task id. Pruned
 * each cycle to drop ids that no longer exist in `scheduled_tasks` so
 * the map can't grow unbounded across the lifetime of the process.
 */
const lastDormantWarnAt = new Map<string, number>();

export interface SchedulerDependencies {
  registeredGroups: () => Record<string, RegisteredGroup>;
  queue: GroupQueue;
  onProcess: (
    groupJid: string,
    sessionName: string,
    proc: ChildProcess,
    containerName: string,
    groupFolder: string,
  ) => void;
  /**
   * Send a message to the chat. Returns the channel-native message id
   * (Telegram message id) when the send lands, `void` for non-id
   * channels or a swallowed send — mirroring `Channel.sendMessage` so
   * the scheduled-task forward can record `telegram_message_id` and
   * gate the bot-row write on delivery (#681).
   */
  sendMessage: (jid: string, text: string) => Promise<string | void>;
  /**
   * Wipe the on-disk session artifacts (JSONL transcript and the
   * sibling per-session tool-results directory) for a just-finished
   * scheduled-task SDK session. Each scheduled run is a fresh SDK turn
   * (#193) — its sessionId is never persisted to the sessions cache or
   * DB, so `nukeSession` and the time-based `cleanup-sessions.sh`
   * script cannot find it to wipe later. Without this hook, every run
   * leaves orphan files under
   * `data/sessions/<group>/maintenance/.claude/projects/<slug>/`.
   *
   * Invocation contract: the scheduler de-duplicates every `newSessionId`
   * the SDK reports during the run (streaming events plus the terminal
   * `runContainerAgent` return value, since either may carry the id, and
   * the SDK can re-issue the id mid-run) and calls this helper once per
   * unique id from a `finally` block that runs after the post-run DB
   * bookkeeping (`logTaskRun`, `updateTaskAfterRun`). The `finally`
   * placement guarantees the wipe still fires when those DB writes throw
   * — otherwise a transient SQLite error would leave the just-created
   * artifacts orphan-on-disk, defeating #193.
   *
   * Implemented by the orchestrator via `wipeSessionJsonl` (delete-
   * while-open is safe on POSIX, so we don't have to wait for container
   * teardown). The implementation is defensive — ENOENT and other
   * expected fs errors are swallowed internally and reflected in the
   * returned count. Returned count is the total number of filesystem
   * entries removed: up to 2 per slug (1 JSONL + 1 tool-results dir),
   * summed across every project-slug subdirectory walked.
   */
  wipeSessionJsonl: (
    groupFolder: string,
    sessionName: string,
    sessionId: string,
  ) => number;
}

/**
 * Work-evidence post-check for cadence tasks (#720). `evidenceSpec` is
 * the declared contract `<relative-file>#<json-field>` (shape enforced
 * at registration by `validateCadenceDeclaration`); the file is read
 * relative to `groupDir`, parsed as JSON, and the named top-level
 * string field must parse as a date >= `runStartMs` — proving the
 * artifact was actually freshened during the run rather than the
 * agent fabricating a success report. Pure w.r.t. the DB: the caller
 * (`runTask`) maps `{ok: false}` onto `runStatus = 'error'` and clears
 * the pinned session.
 *
 * Only *expected* filesystem failures (ENOENT / EACCES / EISDIR /
 * ENOTDIR) and JSON syntax errors are folded into `{ok: false,
 * reason}`; anything else propagates per `jbaruch/coding-policy:
 * error-handling`. The spec shape is re-guarded here (not just at
 * registration) because `scheduled_tasks.evidence` is a plain DB
 * column a hand-edit can corrupt — a malformed spec fails CLOSED
 * with a remediation hint rather than producing junk path lookups.
 * Every reason carries the operator's next step, since the caller
 * persists it into `task_run_logs.error`.
 */
export function checkTaskEvidence(
  evidenceSpec: string,
  groupDir: string,
  runStartMs: number,
): { ok: true } | { ok: false; reason: string } {
  const hashIdx = evidenceSpec.indexOf('#');
  const relFile = evidenceSpec.slice(0, hashIdx);
  const field = evidenceSpec.slice(hashIdx + 1);
  if (
    hashIdx <= 0 ||
    field === '' ||
    field.includes('#') ||
    relFile.startsWith('/') ||
    relFile.split('/').includes('..')
  ) {
    return {
      ok: false,
      reason: `malformed evidence spec ${JSON.stringify(evidenceSpec)} — expected <relative-file>#<json-field> with a relative file path; fix the 'evidence:' frontmatter in the skill's SKILL.md (or the hand-edited scheduled_tasks.evidence value) and redeploy`,
    };
  }
  const filePath = path.join(groupDir, relFile);
  // Symlink containment (no-secrets): this reader runs HOST-side
  // against a CONTAINER-writable group folder. Lexical validation of
  // the spec can't stop an agent from planting a symlink at the
  // evidence path that targets a host file outside the group tree
  // (e.g. the .env), so resolve both ends through realpath and require
  // the resolved evidence file to stay inside the resolved group
  // folder. On escape, fail closed WITHOUT reading — the reason must
  // never embed external file contents.
  let raw: string;
  try {
    const groupRoot = fs.realpathSync(groupDir);
    const resolved = fs.realpathSync(filePath);
    if (resolved !== groupRoot && !resolved.startsWith(groupRoot + path.sep)) {
      return {
        ok: false,
        reason: `evidence path resolves outside the group folder (symlink?): ${relFile} — the evidence file must live inside the group folder; inspect the group tree for a planted symlink`,
      };
    }
    raw = fs.readFileSync(resolved, 'utf-8');
  } catch (err: unknown) {
    const code = (err as NodeJS.ErrnoException).code;
    if (
      err instanceof Error &&
      (code === 'ENOENT' ||
        code === 'EACCES' ||
        code === 'EISDIR' ||
        code === 'ENOTDIR' ||
        code === 'ELOOP')
    ) {
      return {
        ok: false,
        reason: `evidence file missing/unreadable: ${relFile} (${code}) — verify the skill actually writes this file under the group folder, or correct the 'evidence:' frontmatter in the skill's SKILL.md`,
      };
    }
    throw err;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err: unknown) {
    if (!(err instanceof SyntaxError)) throw err;
    return {
      ok: false,
      reason: `evidence file is not valid JSON: ${relFile} — ${err.message} — inspect the file in the group folder and fix the skill step that writes it`,
    };
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return {
      ok: false,
      reason: `evidence file is not a JSON object: ${relFile} — inspect the file in the group folder and fix the skill step that writes it`,
    };
  }
  const value = (parsed as Record<string, unknown>)[field];
  if (typeof value !== 'string') {
    return {
      ok: false,
      reason: `evidence field missing or not a string: ${field} in ${relFile} (got ${value === undefined ? 'undefined' : typeof value}) — align the 'evidence:' frontmatter field name with what the skill actually writes`,
    };
  }
  // Cap the embedded value so a garbage field can't balloon the
  // persisted task_run_logs.error / log line. Post-containment the
  // value is group-folder data (agent-visible anyway), never external
  // file contents.
  const valuePreview = JSON.stringify(
    value.length > 64 ? `${value.slice(0, 64)}…` : value,
  );
  const parsedMs = Date.parse(value);
  if (Number.isNaN(parsedMs)) {
    return {
      ok: false,
      reason: `evidence field is not a parseable date: ${field}=${valuePreview} in ${relFile} — align the 'evidence:' frontmatter field with a field the skill stamps as an ISO timestamp`,
    };
  }
  if (parsedMs < runStartMs) {
    return {
      ok: false,
      reason: `evidence stale: ${field}=${valuePreview} predates run start ${new Date(runStartMs).toISOString()} — the run did not freshen the artifact; the pinned session is cleared automatically so the next fire retries fresh — if this recurs, inspect the run's container log for skipped pipeline steps`,
    };
  }
  return { ok: true };
}

async function runTask(
  task: ScheduledTask,
  deps: SchedulerDependencies,
): Promise<void> {
  const startTime = Date.now();
  let groupDir: string;
  try {
    groupDir = resolveGroupFolderPath(task.group_folder);
  } catch (err) {
    // resolveGroupFolderPath throws Error on path-validation failure.
    // Anything else is a bug elsewhere; propagate per
    // `jbaruch/coding-policy: error-handling`.
    if (!(err instanceof Error)) throw err;
    const error = err.message;
    // Stop retry churn for malformed legacy rows.
    updateTask(task.id, { status: 'paused' });
    logger.error(
      { taskId: task.id, groupFolder: task.group_folder, error },
      'Task has invalid group folder',
    );
    logTaskRun({
      task_id: task.id,
      run_at: new Date().toISOString(),
      duration_ms: Date.now() - startTime,
      status: 'error',
      result: null,
      error,
    });
    return;
  }
  fs.mkdirSync(groupDir, { recursive: true });

  logger.info(
    { taskId: task.id, group: task.group_folder },
    'Running scheduled task',
  );

  const groups = deps.registeredGroups();
  const group = Object.values(groups).find(
    (g) => g.folder === task.group_folder,
  );

  if (!group) {
    logger.error(
      { taskId: task.id, groupFolder: task.group_folder },
      'Group not found for task',
    );
    logTaskRun({
      task_id: task.id,
      run_at: new Date().toISOString(),
      duration_ms: Date.now() - startTime,
      status: 'error',
      result: null,
      error: `Group not found: ${task.group_folder}`,
    });
    return;
  }

  // #754 pre-spawn eligibility gate. Windowed cadence skills
  // (flight-assist) only do useful work inside a trip window; firing a
  // container every couple of minutes off-window pays a full spawn for a
  // precheck that would just skip. Evaluate a host-side predicate BEFORE
  // any spawn work — out of window, record a distinct
  // `skipped_out_of_window` run and return without spawning. Skills with
  // no registered gate resolve to `null` and spawn unconditionally, as
  // before. `startTime` is this fire's "now"; the recurring row's next
  // fire is still scheduled by the caller's `finally`, so the task keeps
  // ticking — it just doesn't spawn this time.
  const gateVerdict = evaluateSpawnGate(
    parseTaskSkill(task.prompt),
    groupDir,
    new Date(startTime),
  );
  if (gateVerdict && !gateVerdict.eligible) {
    logger.info(
      { taskId: task.id, group: task.group_folder, reason: gateVerdict.reason },
      '[task-scheduler] pre-spawn gate skipped fire — no container spawned (#754)',
    );
    logTaskRun({
      task_id: task.id,
      run_at: new Date().toISOString(),
      duration_ms: Date.now() - startTime,
      status: 'skipped_out_of_window',
      result: null,
      error: null,
    });
    // Advance the recurring schedule exactly as the normal completion
    // path does. This early return bypasses the post-run bookkeeping that
    // is the SINGLE writer of `next_run` (#438) — without re-advancing
    // here, the row stays due, the caller's `finally` clears it from
    // `dispatchedTaskIds`, and the scheduler re-dispatches (and re-logs)
    // the skip on every poll tick instead of at cadence. Re-fetch fresh
    // so a concurrent `update_task` isn't clobbered, mirroring the
    // race guard on the normal path.
    const fresh = getTaskById(task.id) ?? task;
    const computed = computeNextRunDetailed(fresh, getCurrentTz);
    if (computed.remediation) {
      applyComputeNextRunRemediation(
        fresh.id,
        computed.remediation,
        fresh.schedule_value,
        fresh.schedule_timezone,
      );
    }
    updateTaskAfterRun(
      fresh.id,
      computed.nextRun,
      `Skipped: ${gateVerdict.reason}`,
    );
    return;
  }

  // Update tasks snapshot for container to read (filtered by group)
  const isMain = group.isMain === true;
  const tasks = getAllTasks();
  writeTasksSnapshot(
    task.group_folder,
    isMain,
    tasks.map((t) => ({
      id: t.id,
      groupFolder: t.group_folder,
      prompt: t.prompt,
      script: t.script,
      schedule_type: t.schedule_type,
      schedule_value: t.schedule_value,
      status: t.status,
      next_run: t.next_run,
    })),
    !!group.containerConfig?.trusted,
  );

  let result: string | null = null;
  let error: string | null = null;
  // #581 follow-up — captured from the streaming or terminal output
  // so the runStatus mapping after the try / catch / finally can
  // propagate `'precheck_skipped'` into `task_run_logs.status`. The
  // terminal `output` const is scoped inside the try block, so we
  // mirror its `status === 'precheck_skipped'` signal here.
  let precheckSkipped = false;
  // #589 (reopened) — set when the terminal output status is 'killed'
  // (container-runner reaped a maintenance container mid-compose). Drives
  // the runStatus mapping to 'killed' and carries an actionable reason
  // into the row's error column without flipping the status to 'error'.
  let killedMidRun = false;
  let killedReason = '';

  // Per-fire telemetry context (#349). Computed once so every
  // streamed `usage` payload classifies against the same window
  // without recomputing the formula on every turn. `taskSkill` is
  // safe to log because it's derived from the literal SDK skill-
  // invocation syntax (a fixed identifier — `tessl__heartbeat`,
  // `tessl__nightly-housekeeping`, etc.). The raw prompt is NOT
  // logged: scheduled-task prompts include user-authored reminders
  // ("Remind me to call Dr. X about Y results …") that would land
  // verbatim in host-logs — `taskId` plus an optional join against
  // `scheduled_tasks` covers any case where an operator needs to
  // see what a row's prompt actually was, without the leak.
  const thresholds = computeThresholds(MODEL_CONTEXT_WINDOW);
  const taskSkill = parseTaskSkill(task.prompt);

  // Per-task SDK session reuse (#336, evolves #193). Recurring tasks
  // (cron / interval) keep their own `session_id` across fires so the
  // API can cache the per-session message-history prefix even though
  // the prompt-cache TTL (5 min) expires between heartbeat fires (15-
  // 30 min cadence). One-shot tasks (`schedule_type === 'once'`) stay
  // fresh-per-fire — they're out of scope per #336.
  //
  // The #193 cross-task bleed concern doesn't apply: the original
  // bleed came from a single id shared via `sessions[group]
  // [maintenance]` across DIFFERENT tasks (a lunch reminder picking up
  // a heartbeat-loop's terminal message). Persistence here is keyed on
  // `task_id`, so two distinct tasks land in two distinct DB rows
  // hence two distinct SDK sessions — no slot-cache aliasing
  // possible. The `MAINTENANCE_SESSION_NAME` slot still routes maint
  // work into the parallel queue; the SDK session loaded inside that
  // slot is now per-task. `context_mode` stays inert on the schema.
  //
  // Disk hygiene: the SDK writes one JSONL transcript per session id
  // under `data/sessions/<group>/maintenance/.claude/projects/<slug>/`.
  // For reusable tasks the LATEST persisted id is alive (next fire
  // resumes it) — skip its wipe in the finally block. Any other id we
  // saw this fire is either pre-existing-and-rotated or
  // SDK-rotated-mid-run, and its transcript is now orphan — wipe.
  // For once-tasks (not reusable) every observed id is wiped exactly
  // as #193 always did.
  const isReusable = task.schedule_type !== 'once';
  let startingSessionId: string | undefined = isReusable
    ? (task.session_id ?? undefined)
    : undefined;
  const observedSessionIds = new Set<string>();

  // Plugin-hash session invalidation (#710). Skill/rule content is
  // injected into the SDK session only at creation; a resumed session
  // never re-reads the per-spawn snapshot, so a pinned session_id
  // silently outlives every plugin update — the agent keeps following
  // stale instructions plus its own in-context precedent. Compare the
  // registry hash stored when the session was persisted against the
  // current one and rotate to a fresh session on mismatch. Rotation
  // requires a KNOWN current hash: `null` means the registry is absent
  // or vanished mid-walk (`tessl update` swap race), i.e. "content
  // state unknowable this fire" — rotating on it would spuriously
  // burn the session (and, once null is persisted alongside the new
  // id, burn it again next fire), so unknowable resumes as-is and the
  // next fire re-evaluates against a readable registry. A NULL stored
  // hash under a live registry covers pre-#710 rows and rotates them
  // once; registry-less installs stay NULL-vs-null and keep resuming.
  const currentPluginsHash: string | null = isReusable
    ? getPluginRegistryHash()
    : null;
  if (
    startingSessionId &&
    currentPluginsHash !== null &&
    (task.session_plugins_hash ?? null) !== currentPluginsHash
  ) {
    logger.info(
      {
        taskId: task.id,
        staleSessionId: startingSessionId,
        storedPluginsHash: task.session_plugins_hash ?? null,
        currentPluginsHash,
      },
      '[task-scheduler] plugin registry changed since session was pinned — rotating to a fresh SDK session (#710)',
    );
    // Queue the stale transcript for the post-run wipe (it can never
    // be re-resumed) and clear the DB pointer so a crash before the
    // new id persists can't leave a dangling resume target. The clear
    // uses the same narrow SqliteError tolerance as the persist paths
    // below: on a transient DB hiccup the in-memory rotation still
    // holds for this fire, and the next fire re-detects the mismatch
    // and retries.
    observedSessionIds.add(startingSessionId);
    try {
      clearTaskSessionId(task.id);
    } catch (dbErr) {
      if (!(dbErr instanceof SqliteError)) throw dbErr;
      logger.error(
        {
          taskId: task.id,
          staleSessionId: startingSessionId,
          sqliteCode: dbErr.code,
          err: dbErr,
        },
        '[task-scheduler] clearTaskSessionId failed during #710 rotation — continuing with a fresh session this fire; next fire re-detects the mismatch',
      );
    }
    startingSessionId = undefined;
  }
  let persistedSessionId: string | undefined = startingSessionId;

  // After the task produces a result, close the container promptly.
  // Tasks are single-turn — no need to wait IDLE_TIMEOUT (30 min) for the
  // query loop to time out. A short delay handles any final MCP calls.
  // The kill grace after close sentinel is handled by GroupQueue.closeStdin().
  const TASK_CLOSE_DELAY_MS = 10000;
  let closeTimer: ReturnType<typeof setTimeout> | null = null;

  const scheduleClose = () => {
    if (closeTimer) return; // already scheduled
    closeTimer = setTimeout(() => {
      logger.debug({ taskId: task.id }, 'Closing task container after result');
      deps.queue.closeStdin(task.chat_jid, MAINTENANCE_SESSION_NAME);
    }, TASK_CLOSE_DELAY_MS);
  };

  try {
    // Pre-resume artifact pruning (#538) for maintenance-session
    // task fires that resume a persisted session_id. Source PR
    // referenced `sessionId`; this fork's per-task session-reuse
    // path (from #336) names it `startingSessionId` — same value,
    // local rename only.
    if (startingSessionId) {
      const retention = pruneSessionArtifacts({
        dataDir: DATA_DIR,
        groupFolder: task.group_folder,
        sessionName: MAINTENANCE_SESSION_NAME,
        sessionId: startingSessionId,
        config: resolveSessionArtifactRetentionConfig(),
      });
      if (
        retention.imageBlocksReplaced > 0 ||
        retention.toolResultRefsReplaced > 0 ||
        retention.toolResultFilesDeleted > 0
      ) {
        logger.info(
          {
            taskId: task.id,
            groupFolder: task.group_folder,
            sessionId: startingSessionId,
            transcriptPath: retention.transcriptPath,
            imageBlocksReplaced: retention.imageBlocksReplaced,
            toolResultRefsReplaced: retention.toolResultRefsReplaced,
            toolResultFilesDeleted: retention.toolResultFilesDeleted,
          },
          'session_artifact_retention_pruned_task',
        );
      }
    }

    const output = await runContainerAgent(
      group,
      {
        prompt: task.prompt,
        // Per-task session reuse for recurring fires (#336). When the
        // task already has a persisted `session_id`, pass it as
        // `resume:` so the SDK reloads the prior message history (and
        // the API caches the prefix). For once-tasks and for the
        // first fire of a recurring task, this is undefined and the
        // SDK creates a fresh session — the streaming/terminal
        // newSessionId callbacks below persist it for the next fire.
        sessionId: startingSessionId,
        groupFolder: task.group_folder,
        chatJid: task.chat_jid,
        isMain,
        isScheduledTask: true,
        assistantName: ASSISTANT_NAME,
        script: task.script || undefined,
        // Provenance: the role that created this task, so the agent-runner
        // can decide whether to wrap the prompt in <untrusted-input>. Only
        // 'untrusted_agent'-created tasks get wrapped; owner/main/trusted
        // bypass. See ContainerInput.createdByRole docs.
        createdByRole: task.created_by_role,
        // Route every scheduled task into the parallel `maintenance` slot so
        // it runs concurrently with user-facing work. Sole writer of this
        // value — inbound paths route to `'default'` instead.
        sessionName: MAINTENANCE_SESSION_NAME,
        // Continuation marker for self-resuming cycles (#93/#130). NULL on
        // ordinary tasks; set only when the resumable-cycle helper skill
        // scheduled this row as the next link of a chain. Container-runner
        // emits NANOCLAW_CONTINUATION=1 + NANOCLAW_CONTINUATION_CYCLE_ID
        // env vars iff this is non-empty. `?? undefined` normalises the DB
        // SELECT result (NULL for ordinary rows) into the optional
        // ContainerInput field shape — never pass `null` here, since
        // `if (continuationCycleId)` in buildContainerArgs would treat the
        // string `"null"` as truthy if a stringification slipped in.
        continuationCycleId: task.continuation_cycle_id ?? undefined,
        // Per-task AGENT_MODEL override (#509 Phase 3). Read directly
        // from the scheduled_tasks row; `resolveSessionAgentModel`
        // treats undefined / null / empty as "no override" and falls
        // through to the Phase 2 ladder (maintenanceAgentModel → group
        // agentModel → AGENT_MODEL env → DEFAULT_AGENT_MODEL).
        taskAgentModel: task.agent_model ?? undefined,
      },
      (proc, containerName) =>
        deps.onProcess(
          task.chat_jid,
          MAINTENANCE_SESSION_NAME,
          proc,
          containerName,
          task.group_folder,
        ),
      async (streamedOutput: ContainerOutput) => {
        // Per-task session reuse (#336): persist `newSessionId` for
        // recurring tasks. The SDK can re-issue the id mid-run (e.g.
        // when a resumed session rotates to a fresh transcript), and
        // we want last-write-wins semantics — the LATEST id is the
        // one whose JSONL is alive on disk. Once-tasks stay
        // fresh-per-fire (#336 out-of-scope) so we just collect the
        // id for post-run wipe without persisting. Every observed id
        // also goes into `observedSessionIds` so the finally block
        // can wipe rotated/orphan transcripts; the alive id is
        // skipped there.
        if (streamedOutput.newSessionId) {
          observedSessionIds.add(streamedOutput.newSessionId);
          if (
            isReusable &&
            streamedOutput.newSessionId !== persistedSessionId
          ) {
            const newId = streamedOutput.newSessionId;
            // Catch only the recoverable case (SQLite-level failure:
            // busy / disk full / schema mid-migration / FK constraint)
            // and let any other throw propagate per
            // `jbaruch/coding-policy: error-handling`. The narrow
            // catch matters because `runContainerAgent` chains
            // `onOutput` via `.then(...)` with no `.catch(...)`, so a
            // SQLite-class error here would otherwise reject the run
            // promise and wedge the scheduler loop; for a transient
            // DB hiccup the right behaviour is "next fire starts a
            // fresh session" (recoverable). A non-`SqliteError` throw
            // (TypeError, ReferenceError, programming bug) bubbles up
            // to the outer try/catch, which logs `Task failed` and
            // marks the run 'error' — exactly what we want for a real
            // bug. The in-memory `persistedSessionId` advances only
            // after the write succeeds, so the post-run wipe-skip
            // logic can't preserve a transcript whose DB pointer
            // never landed.
            try {
              setTaskSessionId(task.id, newId, currentPluginsHash);
              persistedSessionId = newId;
            } catch (dbErr) {
              if (!(dbErr instanceof SqliteError)) throw dbErr;
              logger.error(
                {
                  taskId: task.id,
                  newSessionId: newId,
                  sqliteCode: dbErr.code,
                  err: dbErr,
                },
                '[task-scheduler] setTaskSessionId failed during streaming — continuing, next fire will start a fresh SDK session (#336)',
              );
            }
          }
        }
        // Kill-auto-compaction telemetry on the scheduled-task path
        // (#349). Same `session_tokens` log key + state classification
        // as the inbound path in `src/index.ts`, plus scheduled-task
        // discriminators (taskId / scheduleType / taskSkill) so cost
        // analyses can bucket by recurring-task identity. Session is
        // intentionally undefined here — #193's contract is that
        // scheduled-task fires never persist a sessionId, so the
        // telemetry line reflects that absence rather than emitting a
        // stale value. The return state is discarded: the inbound-only
        // kill-auto-compaction handshake (`thresholdReached` latch +
        // `nuke_session` call) does NOT apply to maintenance fires —
        // each fire is already a single-turn discardable session by
        // #193, so no nuke-on-cross is needed.
        emitSessionTokens(streamedOutput.usage, {
          group: group.name,
          thresholds,
          extra: {
            taskId: task.id,
            scheduleType: task.schedule_type,
            taskSkill,
          },
        });
        if (streamedOutput.result) {
          // #581 — populate task_run_logs.result UNCONDITIONALLY when
          // the agent emitted text. The chat-echo gate below is the
          // only thing chat_displayed should suppress; observability
          // (forensic greps, silent-success accounting) must stay
          // intact for wrapper skills that always finish via
          // send_message.
          result = streamedOutput.result;
          // Strip <internal> tags — suppress entirely if nothing remains
          const cleanResult = streamedOutput.result
            .replace(/<internal>[\s\S]*?<\/internal>/g, '')
            .trim();
          // #581 — when the agent already used send_message /
          // send_file successfully, the agent-runner sets
          // `chat_displayed: true`. The IPC `send_message` handler
          // already wrote to messages.db, so re-sending here would
          // duplicate the user-visible reply AND double-row the DB.
          // Result text is still captured above for task_run_logs.
          //
          // #681 — skill-invoking scheduled tasks (heartbeat, monitors,
          // morning-brief — anything whose prompt carries a
          // `Skill(skill: "tessl__…")` call, i.e. `taskSkill` is set)
          // must surface ONLY via an explicit send_message (which sets
          // `chat_displayed`). Their leftover terminal text — a bare
          // `Ready.` / `✓`, a `Cycle complete.` ack, an internal status
          // dump, even a hallucinated "host-fix-required" bug report —
          // is internal-by-default and must NOT auto-forward; if the
          // skill had something for the user it would have called
          // send_message. Raw reminders / one-shots have no skill call
          // (`taskSkill === undefined`) and their terminal text IS the
          // intended surface, so they keep forwarding. `result` is
          // still captured above either way for task_run_logs.
          if (
            cleanResult &&
            !streamedOutput.chat_displayed &&
            taskSkill === undefined
          ) {
            const sendResult = await deps.sendMessage(
              task.chat_jid,
              cleanResult,
            );
            // Normalize `string | void` to `string | undefined`; only
            // persist a telegram_message_id when the channel returned
            // one (mirrors the inbound send path in src/index.ts).
            const sentMsgId =
              typeof sendResult === 'string' ? sendResult : undefined;
            // Store the bot send so `messages.db` reflects every
            // delivered send out of this session. Without this,
            // scheduled-task sends
            // (heartbeat, housekeeping, morning-brief, etc.) reach
            // Telegram but leave no DB row — the "ghost heartbeat" /
            // "no trace in messages.db" class of jbaruch/nanoclaw#81.
            // The IPC-path `send_message` handler in src/ipc.ts writes
            // the same shape; this mirrors it so heartbeat's answered-
            // check accounting and forensic greps both see the row.
            //
            // Upsert chat metadata first so the `messages.chat_jid →
            // chats.jid` FK doesn't reject the insert on a chat that
            // has no prior metadata (task fires before any user
            // message, or chat was manually registered without the
            // normal group-sync write-through). Idempotent: existing
            // rows keep their `name` because we pass `name` as
            // undefined and `storeChatMetadata` omits `name` from the
            // UPDATE in that branch (not COALESCE); `channel` and
            // `is_group` are preserved via COALESCE when we pass
            // undefined for them. `last_message_time` advances to the
            // outgoing send's timestamp, same as the IPC path would
            // effectively do by chaining a chat-metadata update.
            //
            // Pass inferred `channel` + `isGroup` so a NEW chat row
            // (first-ever metadata write) has the right shape for
            // `getAvailableGroups()`, which filters on `is_group`.
            // Match the channel-name convention the codebase already
            // uses everywhere else (`'telegram'`, `'whatsapp'`) — NOT
            // the JID prefix abbreviation. JID shapes in this repo:
            //   - `tg:<id>` — Telegram. Negative id = group/channel,
            //     positive = private 1:1.
            //   - `<id>@g.us` — WhatsApp group (no `wa:` prefix).
            //   - `<id>@s.whatsapp.net` — WhatsApp DM.
            // Matches the conventions `db.ts`'s legacy-chat backfill
            // uses (`@g.us` → group, `@s.whatsapp.net` → DM).
            // Anything else: leave both undefined so COALESCE in
            // storeChatMetadata preserves existing values rather than
            // writing NULL or an abbreviated channel string.
            const sendTimestamp = new Date().toISOString();
            let inferredChannel: string | undefined;
            let inferredIsGroup: boolean | undefined;
            if (task.chat_jid.startsWith('tg:')) {
              inferredChannel = 'telegram';
              inferredIsGroup = task.chat_jid.startsWith('tg:-');
            } else if (task.chat_jid.endsWith('@g.us')) {
              inferredChannel = 'whatsapp';
              inferredIsGroup = true;
            } else if (task.chat_jid.endsWith('@s.whatsapp.net')) {
              inferredChannel = 'whatsapp';
              inferredIsGroup = false;
            }
            // #681 — gate the bot-row write on actual delivery, the
            // same `shouldStoreBotMessage` contract the inbound
            // (src/index.ts) and IPC (src/ipc.ts) send paths use: on
            // Telegram a missing message id means the send was
            // swallowed, and a phantom row would make heartbeat's
            // answered-check treat a never-delivered send as a reply.
            // Non-Telegram channels return `void` on success, so the
            // gate is a no-op there. The `telegram_message_id` closes
            // the NULL-tgid observability hole — pre-#681 every
            // scheduled-task send recorded a row with no native id, so
            // a leak could not be traced back to a real Telegram send.
            if (shouldStoreBotMessage(task.chat_jid, sentMsgId)) {
              // Wrap the DB writes so a SQLite error (FK constraint,
              // disk full, schema mid-migration) never rejects the
              // `onOutput` promise. The streaming output chain in
              // `container-runner.ts` awaits this via `.then(...)` with
              // no `.catch(...)`, so a throw here can wedge the run
              // from ever resolving and stall the scheduler loop. The
              // send already succeeded; a missing DB row is recoverable
              // (at worst we'd get a duplicate in `unanswered` on the
              // next cycle) — stalling the scheduler is not.
              try {
                storeChatMetadata(
                  task.chat_jid,
                  sendTimestamp,
                  undefined,
                  inferredChannel,
                  inferredIsGroup,
                );
                storeMessage({
                  id: `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                  chat_jid: task.chat_jid,
                  sender: ASSISTANT_NAME,
                  sender_name: ASSISTANT_NAME,
                  content: cleanResult,
                  timestamp: sendTimestamp,
                  is_from_me: true,
                  is_bot_message: true,
                  telegram_message_id: sentMsgId,
                });
              } catch (dbErr) {
                // Best-effort store of the sent message; a SqliteError is
                // logged and the tick continues. A non-Sqlite defect
                // propagates.
                if (!(dbErr instanceof SqliteError)) throw dbErr;
                logger.error(
                  {
                    taskId: task.id,
                    chatJid: task.chat_jid,
                    err: dbErr,
                    preview: cleanResult.slice(0, 200),
                  },
                  '[task-scheduler] storeChatMetadata/storeMessage failed after send — continuing, send already landed in Telegram',
                );
              }
            } else {
              logger.warn(
                {
                  taskId: task.id,
                  chatJid: task.chat_jid,
                  contentLen: cleanResult.length,
                },
                '[task-scheduler] Skipping bot-message storeMessage — channel returned no message id (delivery failed)',
              );
            }
          }
          // Don't close here — agent may still be polling for host script results.
          // Close only on final 'success' status below.
        }
        if (
          streamedOutput.status === 'success' ||
          streamedOutput.status === 'precheck_skipped'
        ) {
          // No `notifyIdle` here — `notifyIdle` targets the `default` slot
          // only, so calling it from a maintenance-routed task would flip
          // the wrong container's state and could preempt active user work.
          // `scheduleClose` already winds this container down; when runTask
          // finishes, `drainGroup` chains any pending maintenance task.
          // Same wind-down for `'precheck_skipped'` — the agent never woke,
          // there's nothing to poll for, container can close.
          scheduleClose();
        }
        if (streamedOutput.status === 'precheck_skipped') {
          precheckSkipped = true;
        }
        if (streamedOutput.status === 'error') {
          error = streamedOutput.error || 'Unknown error';
        }
      },
    );

    if (closeTimer) clearTimeout(closeTimer);

    // Terminal `output.newSessionId` mirrors the streaming-path
    // persistence (#336): same last-write-wins semantic. Recurring
    // tasks get the id written through to `task.session_id` so the
    // next fire can `resume:`; once-tasks just collect it for the
    // post-run wipe. DB write is wrapped — see the streaming-path
    // comment above for the rationale (a throw here would reach the
    // outer catch and mis-classify a successful run as `'error'`,
    // even though the run itself completed).
    if (output.newSessionId) {
      observedSessionIds.add(output.newSessionId);
      if (isReusable && output.newSessionId !== persistedSessionId) {
        const newId = output.newSessionId;
        // Same narrow catch as the streaming-path above — see that
        // comment for the rationale. Here the propagation target is
        // the outer try/catch (rather than the streaming-chain
        // rejection), so a non-`SqliteError` throw still propagates
        // and gets surfaced as `Task failed`; only recoverable DB
        // hiccups are swallowed.
        try {
          setTaskSessionId(task.id, newId, currentPluginsHash);
          persistedSessionId = newId;
        } catch (dbErr) {
          if (!(dbErr instanceof SqliteError)) throw dbErr;
          logger.error(
            {
              taskId: task.id,
              newSessionId: newId,
              sqliteCode: dbErr.code,
              err: dbErr,
            },
            '[task-scheduler] setTaskSessionId failed at terminal — continuing, next fire will start a fresh SDK session (#336)',
          );
        }
      }
    }

    if (output.status === 'error') {
      error = output.error || 'Unknown error';
    } else if (output.status === 'killed') {
      // #589 (reopened) / #682 — container-runner resolves 'killed' when
      // a maintenance container ends without delivering a terminal
      // result: either reaped by the host inactivity timeout, or exited
      // cleanly (code 0) having streamed only previews. See the
      // ContainerOutput `'killed'` doc note. Flag the run incomplete so
      // the runStatus mapping below records 'killed' (retriable) instead
      // of a misleading 'success'. Carry the reason into the row's error
      // column without flipping the status to 'error' — see the override
      // after the runStatus ternary.
      killedMidRun = true;
      killedReason =
        output.error ||
        'Maintenance container ended without delivering a terminal result — incomplete run, retriable';
    } else {
      if (output.status === 'precheck_skipped') {
        // #581 follow-up — terminal status mirrored to the outer
        // scope so the runStatus mapping after try / catch / finally
        // can propagate it. The streaming callback above sets this
        // too for the streaming path; either signal is sufficient.
        precheckSkipped = true;
      }
      if (output.result) {
        // Result was already forwarded to the user via the streaming callback above
        result = output.result;
      }
    }

    logger.info(
      { taskId: task.id, durationMs: Date.now() - startTime },
      'Task completed',
    );
  } catch (err) {
    if (closeTimer) clearTimeout(closeTimer);
    // Per `jbaruch/coding-policy: error-handling`: non-Error throws
    // indicate bugs upstream and should propagate. The scheduler loop
    // (Step 2 of `loop` below) is the last-resort safety net that
    // catches them, logs, and keeps ticking — so re-throwing here
    // doesn't kill the orchestrator.
    if (!(err instanceof Error)) throw err;
    error = err.message;
    logger.error({ taskId: task.id, error }, 'Task failed');
  }

  const durationMs = Date.now() - startTime;

  // #496 — detect mid-run force-close by `closeAllActiveContainers`
  // (the periodic `tessl_update` catch-up path). The agent-runner
  // watchdog hard-exits 0 after 30s of `_close`, which makes the
  // container's exit code look like a clean success even though the
  // task was killed mid-flight and may have left a `pending_run_at`
  // lock dangling. Reclassify so operator-facing audits (and the
  // reader skills that gate on `task_run_logs.status`) can tell apart
  // bookkeeping-success from semantic-success.
  //
  // `consumeForcedCloseAt` clears the slot's stamp on read so it can't
  // leak into the next run's classification. We only reclassify if the
  // stamp was set DURING this run (`>= startTime`) — a stamp from an
  // earlier run that never got consumed (e.g. the slot wasn't running)
  // shouldn't taint a later, fresh run.
  const forcedCloseAt = deps.queue.consumeForcedCloseAt(
    task.chat_jid,
    MAINTENANCE_SESSION_NAME,
  );
  const wasForcedClosed = forcedCloseAt !== null && forcedCloseAt >= startTime;
  // Map terminal output / streamed status to the persisted
  // `task_run_logs.status`. `'precheck_skipped'` propagates verbatim
  // (#581) so the silent-success watchdog can tell a precheck-gated
  // no-op from a wake-up with empty result. `error` text (set above
  // by the streaming callback when `streamedOutput.status === 'error'`
  // or by the terminal `output.status === 'error'` branch) wins over
  // a precheck-skip signal: any throw or error-stream means the row
  // is `'error'` regardless of what the terminal output said.
  let runStatus: 'success' | 'error' | 'killed' | 'precheck_skipped' = error
    ? 'error'
    : precheckSkipped
      ? 'precheck_skipped'
      : killedMidRun
        ? 'killed'
        : 'success';
  if (killedMidRun && runStatus === 'killed' && !error) {
    // #589 (reopened) / #682 — carry the kill reason into the row's
    // error column (status stays 'killed', not 'error') so audits and
    // the recovery/redelivery path see an actionable, non-success run
    // rather than a misleading bookkeeping success.
    error = killedReason;
    logger.warn(
      { taskId: task.id, durationMs },
      'Task run reclassified as killed — maintenance container ended without delivering a terminal result (#682); recovery/redelivery should treat this as a non-success retriable run',
    );
  }
  if (wasForcedClosed && runStatus === 'success') {
    runStatus = 'killed';
    // Surface the kill in the structured log AND in `task_run_logs.error`
    // so the row carries an actionable explanation rather than a bare
    // `status='killed'` with no context. The owner skill watching
    // `pending_run_at` recovery can read this when reasoning about why
    // its lock was reclaimed.
    error =
      'Container force-closed mid-run by tessl_update / closeAllActiveContainers — agent-runner watchdog hard-exit 0 fires 30s after `_close`, so container exit code is misleadingly 0 (#496)';
    logger.warn(
      { taskId: task.id, forcedCloseAt, startTime, durationMs },
      'Task run reclassified as killed — container was force-closed mid-flight by tessl_update; pending_run_at on follow_me_tasks may be left dangling and will be cleared by the stale-lock TTL on next startup or pre-update check',
    );
  }

  // #720 — work-evidence post-check for cadence tasks. A maintenance
  // agent on a pinned SDK session fabricated an evidence-gated skill's
  // success report (claimed `verification: "live"` without writing the
  // state file), so a self-reported success is no longer sufficient
  // when the task declares an evidence contract: the artifact named by
  // `task.evidence` must have been freshened during the run. Only runs
  // that would otherwise be 'success' are checked — 'error' / 'killed'
  // already record a failure, and a 'precheck_skipped' run never woke
  // the agent, so no evidence is expected.
  if (task.evidence && runStatus === 'success') {
    const evidenceResult = checkTaskEvidence(
      task.evidence,
      groupDir,
      startTime,
    );
    if (!evidenceResult.ok) {
      runStatus = 'error';
      error = `evidence-check: ${evidenceResult.reason}`;
      logger.warn(
        {
          taskId: task.id,
          evidence: task.evidence,
          reason: evidenceResult.reason,
        },
        'Task run reclassified as error — declared work evidence was not freshened during the run (#720)',
      );
      // Clear the pinned session so the next fire starts fresh —
      // breaking the fabrication-precedent loop (#720): a resumed
      // session carries the fabricated report as in-context precedent
      // and keeps fabricating, while a fresh session was
      // experimentally shown to run honestly. Same narrow SqliteError
      // tolerance as the #710 rotation clear above: on a transient DB
      // hiccup the run is still recorded as 'error' and the next
      // fire's evidence check re-detects and retries the clear.
      try {
        clearTaskSessionId(task.id);
        // The DB pointer is gone, so the persisted transcript can
        // never be resumed — drop the in-memory pointer too so the
        // post-run finally wipes it instead of stranding an orphan
        // JSONL on disk (#193 hygiene).
        persistedSessionId = undefined;
      } catch (dbErr) {
        if (!(dbErr instanceof SqliteError)) throw dbErr;
        logger.error(
          {
            taskId: task.id,
            sessionId: persistedSessionId,
            sqliteCode: dbErr.code,
            err: dbErr,
          },
          '[task-scheduler] clearTaskSessionId failed during #720 evidence-check — run still recorded as error; next fire re-detects stale evidence and retries the clear',
        );
      }
    }
  }

  // Post-run bookkeeping is wrapped in try/finally so the disk-hygiene
  // wipe still runs if any DB write throws (transient SQLite, disk
  // full, schema mid-migration). Without the finally a thrown
  // logTaskRun / updateTaskAfterRun would leave the just-created
  // JSONL orphan-on-disk forever — exactly what #193 is preventing.
  try {
    logTaskRun({
      task_id: task.id,
      run_at: new Date().toISOString(),
      duration_ms: durationMs,
      status: runStatus,
      result,
      error,
    });

    // Re-fetch the task to compute next_run against the FRESH schedule
    // fields. The captured `task` is from before dispatch — between
    // there and here a user can have called `update_task` to change
    // `schedule_value`, `schedule_timezone`, or `schedule_type`, and
    // their fix shouldn't be clobbered by a write-back computed from
    // the stale capture (the same race `applyComputeNextRunRemediation`
    // already guards against on the remediation path).
    const fresh = getTaskById(task.id) ?? task;
    const computed = computeNextRunDetailed(fresh, getCurrentTz);
    if (computed.remediation) {
      applyComputeNextRunRemediation(
        fresh.id,
        computed.remediation,
        fresh.schedule_value,
        fresh.schedule_timezone,
      );
    }
    const resultSummary = error
      ? `Error: ${error}`
      : result
        ? result.slice(0, 200)
        : 'Completed';
    updateTaskAfterRun(fresh.id, computed.nextRun, resultSummary);
  } finally {
    // Wipe orphan JSONL transcripts (#193 disk-hygiene + #336 session
    // reuse). For once-tasks every observed id is orphan — wipe all.
    // For recurring tasks the LATEST persisted id is alive on disk so
    // the next fire can resume; wipe everything else (any id the SDK
    // rotated through mid-run plus the pre-existing `task.session_id`
    // if it differs from the final persisted one — that latter case
    // catches rotations where the SDK didn't re-emit the starting id).
    // No try/catch wrapper: `wipeSessionJsonl` already swallows ENOENT
    // and other expected fs errors internally; anything that escapes
    // is a programming bug per `jbaruch/coding-policy: error-handling`,
    // and propagation is caught by the scheduler loop's terminal
    // safety net.
    const idsToWipe = new Set(observedSessionIds);
    if (startingSessionId && startingSessionId !== persistedSessionId) {
      idsToWipe.add(startingSessionId);
    }
    if (isReusable && persistedSessionId) {
      idsToWipe.delete(persistedSessionId);
    }
    for (const sid of idsToWipe) {
      deps.wipeSessionJsonl(task.group_folder, MAINTENANCE_SESSION_NAME, sid);
    }
  }
}

let schedulerRunning = false;
/**
 * Wall-clock timestamp (ms) of the most recent prune sweep. Initialised
 * to 0 so the first scheduler tick after process start always runs
 * cleanup. Updated unconditionally on each gated entry, even if the
 * prune itself touches zero rows — the cost we're throttling is the
 * SELECT, not the DELETE.
 */
let lastPruneAt = 0;

/**
 * In-flight task IDs the scheduler has dispatched but not yet observed
 * completing. The dueTasks loop filters against this so the same task
 * isn't picked up twice while a previous dispatch is still running, and
 * runTask's wrapper deletes the entry once the run resolves (success
 * OR throw) — the cleanup is paired with dispatch, not with DB
 * bookkeeping.
 *
 * Replaces a pre-#438 pre-advance write to `next_run` that double-counted
 * for interval tasks: the post-completion `updateTaskAfterRun` re-fetched
 * the row (now at `N + ms`), `computeNextRunDetailed`'s interval branch
 * anchored on `task.next_run + ms` → `N + 2·ms`, and every fire silently
 * advanced the cadence by `2·ms`. Today the in-memory set is the only
 * gate that prevents the dueTasks loop from re-dispatching during a
 * fire; `next_run` only advances post-completion via runTask, so the
 * compute step always reads the same anchor and drifts by `0` on the
 * happy path.
 *
 * Crash mid-fire: the set is lost with the process. On restart,
 * `getDueTasks` may re-dispatch the in-flight task once. That is
 * strictly less harmful than the silent halving the pre-advance was
 * masking, and matches the once-task crash-safety contract that
 * `resurrectZombieTasks` was already built for.
 */
const dispatchedTaskIds = new Set<string>();

export function startSchedulerLoop(deps: SchedulerDependencies): void {
  if (schedulerRunning) {
    logger.debug('Scheduler loop already running, skipping duplicate start');
    return;
  }

  // Recover zombie once-tasks (#37): rows pre-advanced to
  // `status='completed'` whose dispatch was dropped before
  // `updateTaskAfterRun` ran. Flipping them back to `active` lets the
  // first `getDueTasks()` poll pick them up — late dispatch beats
  // silent loss for once-tasks (reminders, T-30 traffic checks,
  // scheduled briefings). Idempotent across restarts; if dispatch
  // fails again, `pruneCompletedTasks` eventually GCs via age.
  //
  // Runs BEFORE `schedulerRunning` is set so a transient DB error
  // (e.g., SQLite busy) propagates without leaving the module flag
  // stuck at `true`. A retried `startSchedulerLoop` call then gets a
  // clean second attempt rather than no-op'ing on the stale flag.
  const resurrected = resurrectZombieTasks();
  if (resurrected.length > 0) {
    logger.info(
      { count: resurrected.length, ids: resurrected },
      'Resurrected zombie once-tasks at startup',
    );
  }

  schedulerRunning = true;
  logger.info('Scheduler loop started');

  const loop = async () => {
    try {
      // Run prune + dormant-cron sweep at most once per PRUNE_INTERVAL_MS.
      // The first tick after process start always passes this gate
      // (lastPruneAt initialised to 0), so a restart immediately runs
      // cleanup rather than waiting an hour. `lastPruneAt` is updated
      // AFTER the housekeeping calls succeed — if pruneCompletedTasks
      // or getDormantRecurringTasks throws, the next 60s tick retries
      // rather than gating the whole housekeeping cycle for an hour
      // on a transient DB error.
      const nowMs = Date.now();
      if (nowMs - lastPruneAt >= PRUNE_INTERVAL_MS) {
        const pruned = pruneCompletedTasks(getCompletedTaskTtlMs());
        if (pruned > 0) {
          logger.info({ count: pruned }, 'Pruned completed once-tasks');
        }
        // Dormant-cron visibility: log but never delete. A genuinely
        // stuck cron task points at a dispatch problem (next_run not
        // advancing, queue wedged) — surfacing it as a warn lets a
        // human decide; auto-deleting would silently lose the schedule.
        // Each task is warned at most once per DORMANT_WARN_COOLDOWN_MS
        // so a long-stuck cron doesn't spam the log on every cycle.
        const dormant = getDormantRecurringTasks(DORMANT_CRON_THRESHOLD_MS);
        const dormantIds = new Set<string>();
        for (const task of dormant) {
          dormantIds.add(task.id);
          const lastWarnedAt = lastDormantWarnAt.get(task.id) ?? 0;
          if (nowMs - lastWarnedAt < DORMANT_WARN_COOLDOWN_MS) {
            continue;
          }
          lastDormantWarnAt.set(task.id, nowMs);
          logger.warn(
            {
              taskId: task.id,
              groupFolder: task.group_folder,
              scheduleType: task.schedule_type,
              scheduleValue: task.schedule_value,
              lastRun: task.last_run,
              nextRun: task.next_run,
            },
            'Dormant recurring task — last_run older than threshold',
          );
        }
        // Drop bookkeeping for tasks that are no longer dormant (or no
        // longer exist) so the map can't grow without bound.
        for (const id of lastDormantWarnAt.keys()) {
          if (!dormantIds.has(id)) {
            lastDormantWarnAt.delete(id);
          }
        }
        lastPruneAt = nowMs;
      }

      const dueTasks = getDueTasks();
      if (dueTasks.length > 0) {
        logger.info({ count: dueTasks.length }, 'Found due tasks');
      }

      for (const task of dueTasks) {
        // Re-check task status in case it was paused/cancelled
        const currentTask = getTaskById(task.id);
        if (!currentTask || currentTask.status !== 'active') {
          continue;
        }

        // Skip tasks already dispatched and not yet completed. Without
        // this gate the dueTasks loop would re-pick interval/cron rows
        // every tick while runTask is still running, since `next_run`
        // no longer pre-advances post-#438. See `dispatchedTaskIds`
        // module-level docstring for the full rationale.
        if (dispatchedTaskIds.has(currentTask.id)) {
          continue;
        }

        // Compute remediation hints (broken cron/tz) eagerly so the
        // apply-against-fresh-row helper can pause/clear before
        // dispatch. Distinct from advancing `next_run` — the
        // remediation path treats `nextRun: null` as "this row is
        // structurally unable to schedule itself", whereas the
        // pre-advance write was a separate (and bug-prone) attempt at
        // crash-safety. Removing the pre-advance leaves remediation
        // intact.
        const computed = computeNextRunDetailed(currentTask, getCurrentTz);
        if (computed.remediation) {
          // Apply remediation against the FRESH DB row — if a user
          // raced an `update_task` IPC between the read above and
          // here that fixed the broken cron/tz, the helper detects
          // the mismatch and skips, letting the user's fix stand.
          applyComputeNextRunRemediation(
            currentTask.id,
            computed.remediation,
            currentTask.schedule_value,
            currentTask.schedule_timezone,
          );
          // Short-circuit on pause-broken-cron: the helper just
          // flipped the row to status='paused' (intent: stop running
          // a structurally broken schedule), but `currentTask` in
          // memory still says 'active' because we read it before the
          // remediation. Without this skip we'd dispatch the very
          // task we just paused — Copilot review on PR #446.
          if (computed.remediation === 'pause-broken-cron') {
            continue;
          }
        }
        if (computed.nextRun === null && currentTask.schedule_type === 'once') {
          // Genuine once-task completion — pre-mark as completed so
          // `getDueTasks` doesn't return it again before
          // `updateTaskAfterRun` runs. Load-bearing for
          // `resurrectZombieTasks` (#37): rows pre-marked completed
          // whose dispatch was dropped get flipped back to active at
          // startup. Recurring tasks (interval/cron) deliberately do
          // NOT pre-advance `next_run` here — `dispatchedTaskIds`
          // covers the duplicate-dispatch concern in-memory, and the
          // post-completion `updateTaskAfterRun` is the single writer
          // that advances the column (#438).
          updateTask(currentTask.id, { status: 'completed' });
        }
        // else: cron/interval with nextRun=null means
        // `computeNextRunDetailed` returned a `pause-broken-cron`
        // remediation that the apply step above already handled.
        // Do NOT flip to completed — that would lose the paused
        // state set by the remediation. See #102 round-4 review.

        dispatchedTaskIds.add(currentTask.id);
        try {
          deps.queue.enqueueTask(
            currentTask.chat_jid,
            currentTask.id,
            MAINTENANCE_SESSION_NAME,
            () => {
              // Pair the dispatched-set cleanup with the dispatch itself
              // so it runs regardless of how runTask resolves (success,
              // throw, early-return on invalid group folder). Putting
              // the cleanup here rather than inside runTask's existing
              // `finally` keeps the runTask body unaware of the
              // bookkeeping the loop owns.
              return runTask(currentTask, deps).finally(() => {
                dispatchedTaskIds.delete(currentTask.id);
              });
            },
          );
        } catch (err) {
          // enqueueTask threw synchronously before the runTask wrapper
          // got invoked → the wrapper's `.finally` never runs → the
          // dispatched-set cleanup never happens. Without this catch
          // the row would be wedged in `dispatchedTaskIds` forever and
          // the dueTasks loop would skip it on every subsequent tick.
          // Clear the bookkeeping and re-throw so the outer terminal
          // catch logs the underlying enqueue failure, per
          // `jbaruch/coding-policy: error-handling` (graceful recovery
          // from infrastructure faults — don't fail into a stuck
          // state). Verified by the dispatched-leak regression test
          // in `task-scheduler.test.ts`.
          dispatchedTaskIds.delete(currentTask.id);
          throw err;
        }
      }
      // outer-boundary-process-contract (coding-policy: error-handling): the
      // scheduler's top-level tick boundary — a self-rescheduling setTimeout
      // loop.
      //   - Caller's silent-failure shape: an uncaught throw escapes the tick
      //     callback as an unhandled rejection, killing the loop so NO further
      //     scheduled task ever fires.
      //   - What the catch emits: an error log; the loop reschedules the next
      //     tick and keeps running.
      //   - Why propagation breaks the contract: one task's bug would take down
      //     the whole orchestrator scheduler.
      // eslint-disable-next-line no-catch-all/no-catch-all -- outer-boundary-process-contract
    } catch (err) {
      logger.error({ err }, 'Scheduler loop caught Error');
    }

    setTimeout(loop, SCHEDULER_POLL_INTERVAL);
  };

  loop();
}

/** @internal - for tests only. */
export function _resetSchedulerLoopForTests(): void {
  schedulerRunning = false;
  lastPruneAt = 0;
  lastDormantWarnAt.clear();
  dispatchedTaskIds.clear();
}
