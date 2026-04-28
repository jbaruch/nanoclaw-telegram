/**
 * Graceful-shutdown handoff marker (#213).
 *
 * Background: every `./scripts/deploy.sh` cascades 137 across every
 * active agent container. The killer is `cleanupOrphans()` at the new
 * orchestrator's startup — it lists every `nanoclaw-*` container and
 * stops them, on the assumption that surviving containers must be
 * leftovers from a crashed prior run. That assumption is wrong when
 * the prior orchestrator exited gracefully (deploy SIGTERM): the
 * agent containers are still doing useful work, and killing them
 * loses any in-flight conversation turn or scheduled-task progress.
 *
 * The marker fixes that. On a graceful shutdown the orchestrator
 * writes the names of every active agent container it knows about
 * (from `GroupQueue` state) plus a wall-clock timestamp. On the next
 * startup, `cleanupOrphans()` consults the marker:
 *
 *   - Marker present AND `now - shutdown_at < HANDOFF_TTL_MS` →
 *     skip cleanup for the listed names. They're intentional handoffs
 *     from a graceful shutdown; let them finish their queries and
 *     exit naturally via `--rm`.
 *   - Marker absent OR stale → behave as before (full cleanup).
 *     A SIGKILL'd / crashed orchestrator never wrote a marker, so
 *     this is the safe default for genuine orphans.
 *
 * The marker is read-and-deleted in one atomic step on startup —
 * each marker corresponds to exactly one shutdown→startup handoff,
 * so leaving it around would risk a future startup mistakenly
 * adopting stale names.
 *
 * Schema_version per `jbaruch/coding-policy: stateful-artifacts`:
 * any shape change bumps the field, and the reader treats unknown
 * versions as "no usable prior state" (full cleanup), which is the
 * fail-closed direction for a recovery surface.
 */
import fs from 'fs';
import path from 'path';

import { DATA_DIR } from './config.js';
import { logger } from './logger.js';

/**
 * How long after a graceful shutdown the marker is still trusted.
 * Five minutes is generous for a deploy turnaround (image rebuild +
 * container recreate) yet short enough that a hung-then-restarted
 * orchestrator from hours ago can't accidentally adopt long-since-
 * dead container names.
 */
export const HANDOFF_TTL_MS = 5 * 60 * 1000;

/**
 * Spawn-collision detection window (#213 Phase A observability).
 *
 * After a successful adoption, the orchestrator's GroupQueue starts
 * empty — it has no in-memory record that the adopted containers
 * exist. If a new message arrives for one of those groups during
 * the handoff window, the orchestrator will spawn a fresh container
 * racing against the still-finishing adopted one (Phase B would fix
 * this by registering adopted containers in GroupQueue + polling
 * `docker ps` for natural exit; Phase A defers that work behind a
 * detector).
 *
 * `markHandoffActive` is called once after a successful marker
 * consumption to extend the detection window forward by HANDOFF_TTL_MS.
 * Outside that window, `isHandoffActive` returns false and the
 * spawn-side check skips the `docker ps` call — there are no
 * adopted containers to collide with, so the check would be pure
 * overhead. Inside the window, the spawn path consults `docker ps`
 * and logs a WARN if a same-prefix container is already running.
 *
 * The window is deliberately a single timestamp rather than a set
 * of adopted names because the prefix-based collision check
 * (`nanoclaw-<groupFolder>[-<sessionName>]-...`) doesn't need the
 * full names — it only needs to know whether handoff-era containers
 * could plausibly still be alive.
 */
let handoffWindowEndMs: number | null = null;

export function markHandoffActive(): void {
  handoffWindowEndMs = Date.now() + HANDOFF_TTL_MS;
}

export function isHandoffActive(): boolean {
  return handoffWindowEndMs !== null && Date.now() < handoffWindowEndMs;
}

/** @internal — for tests only. Resets the in-process window state. */
export function _resetHandoffWindowForTests(): void {
  handoffWindowEndMs = null;
}

const SCHEMA_VERSION = 1;

export interface HandoffContainer {
  name: string;
  groupJid: string;
  sessionName: string;
  groupFolder: string | null;
}

export interface HandoffMarker {
  schema_version: number;
  shutdown_at: string;
  containers: HandoffContainer[];
}

const MARKER_PATH = path.join(DATA_DIR, 'handoff.json');

/**
 * Write the marker atomically: temp file in the same directory,
 * fsync, rename. Same pattern as `task-tz-state.json` writes per
 * `rules/follow-me-two-phase-lock.md`. Crash mid-write leaves either
 * the previous marker (or no marker if first time) intact, never a
 * truncated tail.
 */
export function writeHandoffMarker(containers: HandoffContainer[]): void {
  fs.mkdirSync(path.dirname(MARKER_PATH), { recursive: true });
  const marker: HandoffMarker = {
    schema_version: SCHEMA_VERSION,
    shutdown_at: new Date().toISOString(),
    containers,
  };
  const tmp = `${MARKER_PATH}.${process.pid}.tmp`;
  const fd = fs.openSync(tmp, 'w');
  try {
    fs.writeFileSync(fd, JSON.stringify(marker, null, 2) + '\n');
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  fs.renameSync(tmp, MARKER_PATH);
}

/**
 * Read the marker once and delete it. Returns `null` if absent,
 * stale, malformed, or schema-version-incompatible — every "not
 * trustworthy" branch falls through to the same null result so the
 * caller has a single failure mode (no usable prior state → full
 * cleanup).
 *
 * Deletion is part of read because the marker represents a single
 * shutdown→startup handoff. Re-using it on a later startup would
 * incorrectly adopt names that have long since been replaced.
 */
export function readAndConsumeHandoffMarker(): HandoffMarker | null {
  let raw: string;
  try {
    raw = fs.readFileSync(MARKER_PATH, 'utf-8');
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null;
    logger.warn({ err, path: MARKER_PATH }, 'handoff: failed to read marker');
    return null;
  }
  // Always remove the file before any other work, so a parse failure
  // doesn't leave a poisoned marker that the next startup also fails
  // on. The data we already read is in `raw` and survives the unlink.
  try {
    fs.unlinkSync(MARKER_PATH);
  } catch (err) {
    logger.warn({ err, path: MARKER_PATH }, 'handoff: failed to delete marker');
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    logger.warn(
      { err },
      'handoff: marker is not valid JSON, treating as crash recovery',
    );
    return null;
  }
  if (
    !parsed ||
    typeof parsed !== 'object' ||
    (parsed as { schema_version?: number }).schema_version !== SCHEMA_VERSION
  ) {
    logger.warn(
      {
        schemaVersion: (parsed as { schema_version?: number })?.schema_version,
      },
      'handoff: marker schema_version mismatch, treating as crash recovery',
    );
    return null;
  }
  const marker = parsed as HandoffMarker;
  const ageMs = Date.now() - new Date(marker.shutdown_at).getTime();
  if (!Number.isFinite(ageMs) || ageMs < 0 || ageMs > HANDOFF_TTL_MS) {
    logger.info(
      { ageMs, ttlMs: HANDOFF_TTL_MS },
      'handoff: marker is stale, treating as crash recovery',
    );
    return null;
  }
  if (!Array.isArray(marker.containers)) {
    logger.warn(
      {},
      'handoff: marker containers is not an array, treating as crash recovery',
    );
    return null;
  }
  // Per-entry shape validation. The shape check at the array level
  // is necessary but not sufficient — a JSON-valid `[null]` or
  // `[{}]` would otherwise crash callers like
  // `handoff.containers.map(c => c.name)` at startup, dropping the
  // orchestrator into a fail-OPEN state instead of the documented
  // fail-closed-to-crash-recovery contract.
  const validContainers: HandoffContainer[] = [];
  for (const entry of marker.containers) {
    if (
      entry &&
      typeof entry === 'object' &&
      typeof (entry as HandoffContainer).name === 'string' &&
      (entry as HandoffContainer).name.length > 0 &&
      typeof (entry as HandoffContainer).groupJid === 'string' &&
      typeof (entry as HandoffContainer).sessionName === 'string'
    ) {
      const e = entry as HandoffContainer;
      validContainers.push({
        name: e.name,
        groupJid: e.groupJid,
        sessionName: e.sessionName,
        groupFolder: typeof e.groupFolder === 'string' ? e.groupFolder : null,
      });
    } else {
      logger.warn(
        { entry },
        'handoff: dropping malformed container entry from marker',
      );
    }
  }
  return { ...marker, containers: validContainers };
}
