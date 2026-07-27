import fs from 'fs';

import {
  applyTripitSegmentsToTzState,
  getCurrentTz,
  type TripitSegment,
} from '../db-tz.js';
import { registerIpcHandler, scriptResultPath } from '../ipc-registry.js';
import { logger } from '../logger.js';
import { recomputeLocalSchedules } from '../task-scheduler.js';

/**
 * Upper bound on the `persist_tz_segments` payload (#748 review). A real
 * itinerary is a handful of timezone segments; this is generous headroom that
 * still refuses a runaway/accidental payload before it bloats the
 * `tz_state.segments` DB row. Kept in lock-step with the `.max()` on the MCP
 * tool's zod schema in `container/agent-runner/src/ipc-mcp-stdio.ts`.
 */
export const MAX_TZ_SEGMENTS = 200;

/**
 * #748 — coerce the raw `segments` payload of a `persist_tz_segments` IPC call
 * to a `TripitSegment[]`. The value arrives as raw JSON off the IPC wire (the
 * in-container TripIt → Reclaim sync's parsed `result.segments`), so anything
 * that is not an array — `undefined`, `null`, an object, a string — returns an
 * empty array with `wasArray: false` rather than throwing. This is pure
 * normalization; it does NOT decide what to persist. The caller
 * (`classifyTzPersist`) uses `wasArray` to REJECT a non-array (a caller bug must
 * not clear owner `tz_state`) — only a genuine array, including an explicit
 * `[]`, is persisted. Element shapes are not deep-validated here:
 * `applyTripitSegmentsToTzState` / `walkTzSegments` already tolerate partial
 * per-field segment shapes (optional-with-fallback, #229).
 */
export function coerceTzSegments(raw: unknown): {
  segments: TripitSegment[];
  wasArray: boolean;
} {
  if (Array.isArray(raw)) {
    return { segments: raw as TripitSegment[], wasArray: true };
  }
  return { segments: [], wasArray: false };
}

/**
 * #748 — decide what the `persist_tz_segments` IPC handler does with a request,
 * as a pure function so the security-sensitive outcomes are unit-testable
 * without staging `processTaskIpc`:
 *
 *  - `deny`    — the caller is not the main container. The MCP tool is
 *                isMain-gated, but the IPC task-file path is reachable by ANY
 *                container, so this re-check (on the directory-verified
 *                `isMain`) is what actually blocks a non-main agent from
 *                poisoning owner `tz_state`.
 *  - `reject`  — a non-array payload, OR over `MAX_TZ_SEGMENTS`. Both refuse to
 *                write rather than mangle owner tz. A non-array is a caller bug,
 *                not a "no trips" signal: persisting an empty set would clear
 *                `tz_state.segments` and could flip the owner's timezone +
 *                recompute local schedules off a malformed call. The legacy
 *                the removed host-op skipped persistence on malformed stdout for
 *                the same reason; an explicit `[]` is the only way to clear.
 *                Over-cap is refused (not truncated — truncation would silently
 *                drop later trips).
 *  - `persist` — a genuine array within bounds (including an explicit `[]`,
 *                which correctly clears to the no-active-trips state).
 */
export type TzPersistDecision =
  | { action: 'deny'; error: string }
  | { action: 'reject'; error: string }
  | { action: 'persist'; segments: TripitSegment[] };

export function classifyTzPersist(
  isMain: boolean,
  raw: unknown,
): TzPersistDecision {
  if (!isMain) {
    return { action: 'deny', error: 'persist_tz_segments is main-group only' };
  }
  const { segments, wasArray } = coerceTzSegments(raw);
  if (!wasArray) {
    return {
      action: 'reject',
      error: 'segments must be an array (send [] to explicitly clear)',
    };
  }
  if (segments.length > MAX_TZ_SEGMENTS) {
    return {
      action: 'reject',
      error: `too many segments (${segments.length} > ${MAX_TZ_SEGMENTS})`,
    };
  }
  return { action: 'persist', segments };
}

/**
 * Owner-timezone persistence (#879, split out of `ops.ts`): the
 * `persist_tz_segments` command the in-container TripIt sync calls after
 * resolving the owner's itinerary. The pure decision helpers above are
 * exported so the security-sensitive outcomes stay unit-testable without
 * staging `processTaskIpc`.
 */
export function registerOpsTzIpcHandlers(): void {
  registerIpcHandler('persist_tz_segments', {
    handler: ({ data, sourceGroup, isMain }) => {
      // #748 — credential-free host ingestion for the in-container TripIt →
      // Reclaim sync. The sync itself runs in the agent container now (creds
      // swapped at the OneCLI gateway), so the host no longer runs the CLI or
      // holds TripIt/Reclaim/Google secrets. What stays host-side is the
      // `tz_state` write: the container hands back the parsed `segments[]` and
      // the host persists them exactly as the removed `sync_tripit` host-op's
      // success path did — same `applyTripitSegmentsToTzState` + `#584`
      // `onTzFlipped`
      // next_run invalidation, so the scheduler-timezone skill, the 30-min
      // heartbeat advisory walker, and the #574 location cascade keep their
      // owner-tz backbone. The deny (non-main) / reject (over-cap) / persist
      // decision — the security-sensitive part — lives in the pure, tested
      // `classifyTzPersist`; this case just carries it out (log + result file +
      // DB write). See `classifyTzPersist` for the isMain-re-check and cap
      // rationale.
      const decision = classifyTzPersist(isMain, data.segments);
      if (decision.action === 'deny') {
        logger.warn(
          { sourceGroup },
          'Unauthorized persist_tz_segments attempt blocked (non-main container)',
        );
        if (data.requestId) {
          fs.writeFileSync(
            scriptResultPath(sourceGroup, data),
            JSON.stringify({ error: decision.error }),
          );
        }
        return;
      }
      if (data.requestId) {
        const resultPath = scriptResultPath(sourceGroup, data);
        if (decision.action === 'reject') {
          // A malformed (non-array) or over-cap payload — refuse rather than
          // clear/mangle owner tz_state. See `classifyTzPersist`.
          logger.warn(
            { sourceGroup, reason: decision.error },
            'persist_tz_segments: refused (not persisted)',
          );
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: decision.error }),
          );
          return;
        }
        // Failures from the persistence helper (SqliteError, programming
        // bugs) propagate per `coding-policy: error-handling`; only the
        // #584 recompute's transient SQLITE_BUSY/LOCKED is swallowed-with-warn
        // inside the writer, exactly as the removed host-op did.
        const { segments } = decision;
        applyTripitSegmentsToTzState({ segments }, new Date(), () => {
          recomputeLocalSchedules(getCurrentTz, new Date());
        });
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            stdout: JSON.stringify({ persisted: segments.length }),
          }),
        );
        logger.info(
          { sourceGroup, segmentCount: segments.length },
          'persist_tz_segments completed',
        );
      }
    },
  });
}
