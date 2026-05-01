/**
 * #323 — Rate limits on `Task` (sub-agent spawn) and
 * `mcp__nanoclaw__schedule_task`.
 *
 * Provenance-conditional per #318's design principle, mirroring
 * #322's ACL pattern: a `PreToolUse` hook walks back through the
 * transcript collecting untrusted-provenance markers (#321/#322
 * Encoding A + B), classifies the chain into a row of the cap
 * matrix, and either denies the call when an in-window counter
 * exceeds that row's per-hour cap, allows audit-only when the row
 * is `auditOnly: true` (operator-trusted), or simply records the
 * event when within budget.
 *
 * State lives at `/workspace/state/rate-limit-counters.json` —
 * per-group (since `/workspace/state/` is the group-scoped state
 * mount), so an injection in one chat can't poison the counters
 * for another. The file is small (just two timestamp arrays),
 * loaded on each hook fire; when the hook allows and records an
 * event, the updated counters are written atomically (temp-file +
 * rename). Pruning is rolling-window: timestamps older than 1 hour
 * are filtered during evaluation, and persisted pruning happens
 * when the counters are saved, so the file size remains bounded by
 * the highest cap value (operator-trusted row: 50 spawns/hr).
 *
 * Operator opt-in tightening: an optional file at
 * `/workspace/trusted/rate-limit-overrides.json` overrides any row's
 * caps. Loaded once per hook fire (cheap — file is small or absent).
 * The file is in `/workspace/trusted/` — host-only writable, same
 * posture as #320's allowlist and #324's token store, so an
 * injection in trusted/main can't silently widen the gate.
 *
 * Out of scope for v1 (deferred to follow-ups):
 *   - Recursion-depth tracking on Agent spawn (requires depth
 *     propagation through the SDK to sub-agents — a chain crossing
 *     three sub-agents currently appears to each as its own
 *     "first" call).
 *   - Global ceilings across all groups (host-side coordination —
 *     each container only sees its own group's state).
 *   - Per-chat active-task and cron-recurring caps (require a DB
 *     read on every hook fire; v1 enforces per-hour rate caps,
 *     which already gate the fork-bomb case).
 */

import { AclPrefix } from './capability-acl.js';

export type ProvenanceClass =
  | 'operator-trusted'
  | 'operator-untrusted'
  | 'untrusted-source'
  | 'cross-group'
  | 'mixed';

export interface CapMatrix {
  agentSpawnsPerHour: number;
  scheduleTaskPerHour: number;
  /**
   * If true, the hook records the event and emits a heartbeat
   * warning to host logs but never denies. The trusted-operator row
   * sets this so a multi-agent workflow Baruch issues directly
   * never trips. The cap value is still meaningful — it's the
   * threshold at which the warning fires, so backstop monitoring
   * still surfaces unusual operator behavior.
   */
  auditOnly: boolean;
}

/**
 * The cap matrix from #323's issue body. Hard-coded here as the
 * default; the override file at `/workspace/trusted/rate-limit-
 * overrides.json` can lower (or raise) any row.
 *
 * Operator-trusted ceilings are generous (50/30 per hour) and
 * audit-only — the row exists so a heartbeat warning surfaces
 * unusual cadence, not to gate operator workflow. Untrusted rows
 * are tight (2/1 per hour) — the injection-driven case is rare in
 * normal operation, so a low cap is unlikely to false-positive on
 * legitimate non-owner activity.
 */
export const DEFAULT_CAP_MATRIX: Record<ProvenanceClass, CapMatrix> = {
  'operator-trusted': {
    agentSpawnsPerHour: 50,
    scheduleTaskPerHour: 30,
    auditOnly: true,
  },
  'operator-untrusted': {
    agentSpawnsPerHour: 5,
    scheduleTaskPerHour: 3,
    auditOnly: false,
  },
  'untrusted-source': {
    agentSpawnsPerHour: 2,
    scheduleTaskPerHour: 1,
    auditOnly: false,
  },
  'cross-group': {
    agentSpawnsPerHour: 2,
    scheduleTaskPerHour: 1,
    auditOnly: false,
  },
  'mixed': {
    agentSpawnsPerHour: 2,
    scheduleTaskPerHour: 1,
    auditOnly: false,
  },
};

/**
 * Classify a walk-back result into a provenance row.
 *
 * Empty prefix set means no untrusted-provenance marker in the
 * boundary span — operator-originated. The container's trust tier
 * picks between the trusted (audit-only) and untrusted (gated) row.
 *
 * Non-empty: any prefix other than `cross-group` is treated as
 * untrusted-source class (web, gmail, calendar, slack, github,
 * file, agent-browser, tessl, untrusted-container, plus the
 * `__unknown__` sentinel from #322's classifySourcePrefix that
 * fails closed on unfamiliar prefixes).
 *
 * If the chain has both `cross-group` and an untrusted-source
 * prefix, it's classified as `mixed` — same caps as
 * untrusted-source, but a distinct label so logs and deny
 * messages can pinpoint why the gate fired.
 */
export function classifyProvenance(
  prefixes: ReadonlySet<AclPrefix>,
  isTrustedContainer: boolean,
): ProvenanceClass {
  if (prefixes.size === 0) {
    return isTrustedContainer ? 'operator-trusted' : 'operator-untrusted';
  }
  const hasCrossGroup = prefixes.has('cross-group');
  let hasUntrustedSource = false;
  for (const p of prefixes) {
    if (p !== 'cross-group') {
      hasUntrustedSource = true;
      break;
    }
  }
  if (hasCrossGroup && hasUntrustedSource) return 'mixed';
  if (hasCrossGroup) return 'cross-group';
  return 'untrusted-source';
}

/**
 * Stateful counter store — one per group, persisted at
 * `/workspace/state/rate-limit-counters.json`. Stored as raw unix-
 * second timestamps so a rolling window prune is just a filter; no
 * separate "current bucket" book-keeping needed.
 *
 * Per `rules/stateful-artifacts.md`: schema_version starts at 1; a
 * shape change bumps the version and the owner skill (this module)
 * migrates on read.
 */
export interface RateLimitCounters {
  schema_version: 1;
  agent_spawns: number[];
  schedule_task_calls: number[];
}

export const COUNTERS_FILENAME = 'rate-limit-counters.json';

export function emptyCounters(): RateLimitCounters {
  return {
    schema_version: 1,
    agent_spawns: [],
    schedule_task_calls: [],
  };
}

/**
 * Validate a counters payload from disk. Returns either the
 * normalized record or `null` when the shape doesn't conform. The
 * caller treats `null` as "no usable prior state" and starts fresh
 * — a corrupt counter file MUST NOT crash the hook (denying every
 * spawn is worse than losing the counter; the cap is per-hour
 * anyway so the new window starts immediately).
 *
 * Strict on types — silently coercing a non-array `agent_spawns`
 * could let a malformed file (or an injection-driven write that
 * snuck past #324's gate) authorize a spawn that should deny.
 */
export function parseCounters(raw: unknown): RateLimitCounters | null {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  if (r.schema_version !== 1) return null;
  if (!Array.isArray(r.agent_spawns)) return null;
  if (!Array.isArray(r.schedule_task_calls)) return null;
  for (const t of r.agent_spawns) {
    if (typeof t !== 'number' || !Number.isFinite(t)) return null;
  }
  for (const t of r.schedule_task_calls) {
    if (typeof t !== 'number' || !Number.isFinite(t)) return null;
  }
  return {
    schema_version: 1,
    agent_spawns: r.agent_spawns as number[],
    schedule_task_calls: r.schedule_task_calls as number[],
  };
}

const ONE_HOUR_SECONDS = 3600;

/**
 * Drop entries older than `cutoff` (exclusive). Returns a NEW array
 * — the input is not mutated.
 */
export function pruneTimestamps(timestamps: ReadonlyArray<number>, cutoff: number): number[] {
  return timestamps.filter((t) => t >= cutoff);
}

export type RateKind = 'agent_spawn' | 'schedule_task';

export type RateDecision =
  | {
      kind: 'allow';
      provenance: ProvenanceClass;
      auditOnlyExceeded?: boolean;
    }
  | {
      kind: 'deny';
      provenance: ProvenanceClass;
      cap: number;
      observed: number;
      reason: string;
    };

/**
 * Override file shape (loaded from
 * `/workspace/trusted/rate-limit-overrides.json`). Every key is
 * optional; missing keys keep the default. Numeric values clamp to
 * non-negative integers — a negative override would otherwise
 * silently disable the gate.
 *
 * `auditOnly` cannot be flipped to `true` on non-trusted rows via
 * this file (would defeat the gate). It can be flipped to `false`
 * on the trusted-operator row to make the operator opt in to
 * enforcement.
 */
export interface RateLimitOverrides {
  schema_version?: 1;
  rows?: Partial<Record<ProvenanceClass, Partial<CapMatrix>>>;
}

export function applyOverrides(
  defaults: Record<ProvenanceClass, CapMatrix>,
  overrides: RateLimitOverrides | null,
): Record<ProvenanceClass, CapMatrix> {
  if (!overrides || !overrides.rows) return defaults;
  const out: Record<ProvenanceClass, CapMatrix> = { ...defaults };
  for (const row of Object.keys(out) as ProvenanceClass[]) {
    const o = overrides.rows[row];
    if (!o) continue;
    const base = out[row];
    out[row] = {
      agentSpawnsPerHour: clampNonNegInt(
        o.agentSpawnsPerHour,
        base.agentSpawnsPerHour,
      ),
      scheduleTaskPerHour: clampNonNegInt(
        o.scheduleTaskPerHour,
        base.scheduleTaskPerHour,
      ),
      auditOnly: resolveAuditOnly(row, o.auditOnly, base.auditOnly),
    };
  }
  return out;
}

function clampNonNegInt(candidate: unknown, fallback: number): number {
  if (typeof candidate !== 'number') return fallback;
  if (!Number.isFinite(candidate)) return fallback;
  if (candidate < 0) return fallback;
  return Math.floor(candidate);
}

function resolveAuditOnly(
  row: ProvenanceClass,
  candidate: unknown,
  fallback: boolean,
): boolean {
  if (typeof candidate !== 'boolean') return fallback;
  // Non-trusted rows can't be flipped TO audit-only via override —
  // that would defeat the gate. Trusted rows can be flipped FROM
  // audit-only to enforced (operator opting in to tightening).
  if (row !== 'operator-trusted' && candidate === true) return fallback;
  return candidate;
}

export function parseOverrides(raw: unknown): RateLimitOverrides | null {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  // schema_version is optional; if present must be 1
  if (r.schema_version !== undefined && r.schema_version !== 1) return null;
  if (r.rows === undefined) return { schema_version: 1 };
  if (!r.rows || typeof r.rows !== 'object') return null;
  const rows = r.rows as Record<string, unknown>;
  const out: Partial<Record<ProvenanceClass, Partial<CapMatrix>>> = {};
  const validKeys: ReadonlyArray<ProvenanceClass> = [
    'operator-trusted',
    'operator-untrusted',
    'untrusted-source',
    'cross-group',
    'mixed',
  ];
  for (const key of validKeys) {
    const v = rows[key];
    if (!v || typeof v !== 'object') continue;
    const r2 = v as Record<string, unknown>;
    out[key] = {
      ...(typeof r2.agentSpawnsPerHour === 'number'
        ? { agentSpawnsPerHour: r2.agentSpawnsPerHour }
        : {}),
      ...(typeof r2.scheduleTaskPerHour === 'number'
        ? { scheduleTaskPerHour: r2.scheduleTaskPerHour }
        : {}),
      ...(typeof r2.auditOnly === 'boolean'
        ? { auditOnly: r2.auditOnly }
        : {}),
    };
  }
  return { schema_version: 1, rows: out };
}

/**
 * Core decision function — pure given the inputs. The hook wires
 * up disk I/O around it (load counters → decide → record on allow
 * → save). Tested independently from the hook so the matrix +
 * provenance + window logic doesn't need an SDK or filesystem.
 */
export function decideRate(
  rateKind: RateKind,
  prefixes: ReadonlySet<AclPrefix>,
  isTrustedContainer: boolean,
  counters: RateLimitCounters,
  matrix: Record<ProvenanceClass, CapMatrix>,
  nowSec: number,
): RateDecision {
  const provenance = classifyProvenance(prefixes, isTrustedContainer);
  const caps = matrix[provenance];
  const cutoff = nowSec - ONE_HOUR_SECONDS;
  const stream =
    rateKind === 'agent_spawn'
      ? counters.agent_spawns
      : counters.schedule_task_calls;
  const inWindow = pruneTimestamps(stream, cutoff).length;
  const cap =
    rateKind === 'agent_spawn'
      ? caps.agentSpawnsPerHour
      : caps.scheduleTaskPerHour;

  if (inWindow >= cap) {
    if (caps.auditOnly) {
      return {
        kind: 'allow',
        provenance,
        auditOnlyExceeded: true,
      };
    }
    const prefixList = [...prefixes].join(', ') || `(empty — ${provenance})`;
    return {
      kind: 'deny',
      provenance,
      cap,
      observed: inWindow + 1,
      reason:
        `rate_limit: ${rateKind} count would be ${inWindow + 1}/${cap}/hr ` +
        `under provenance row '${provenance}'. ` +
        `Walk-back saw [${prefixList}]. ` +
        `Wait for the rolling window to drain or — for an operator-` +
        `originated workflow — issue the request directly without ` +
        `external content in the same chain.`,
    };
  }

  return { kind: 'allow', provenance };
}

/**
 * Append a new event timestamp to the appropriate stream and prune
 * the rolling window in the same pass. Returns a NEW counters
 * record — the input is not mutated.
 */
export function recordEvent(
  rateKind: RateKind,
  counters: RateLimitCounters,
  nowSec: number,
): RateLimitCounters {
  const cutoff = nowSec - ONE_HOUR_SECONDS;
  const prunedSpawns = pruneTimestamps(counters.agent_spawns, cutoff);
  const prunedTasks = pruneTimestamps(counters.schedule_task_calls, cutoff);
  return {
    schema_version: 1,
    agent_spawns:
      rateKind === 'agent_spawn' ? [...prunedSpawns, nowSec] : prunedSpawns,
    schedule_task_calls:
      rateKind === 'schedule_task' ? [...prunedTasks, nowSec] : prunedTasks,
  };
}

/**
 * Map a tool name to its rate kind. Returns `null` for tools that
 * the rate-limiter doesn't track — the hook treats `null` as
 * pass-through.
 */
export function classifyTool(toolName: string): RateKind | null {
  if (toolName === 'Task') return 'agent_spawn';
  if (toolName === 'mcp__nanoclaw__schedule_task') return 'schedule_task';
  return null;
}
