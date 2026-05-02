/**
 * Host-side Stage 1 gate framework (#80).
 *
 * Gates evaluate inbound messages BEFORE the orchestrator spawns a
 * container so no-op spawns (zero SDK queries, zero work) cost zero
 * Docker startup overhead. Gate FUNCTIONS should be pure, stateless,
 * and side-effect-free (they may read DB but never write); the
 * surrounding framework (`runGateChain`) does emit observability log
 * records, so the chain itself is not strictly pure.
 *
 * The combinator is AND-only for v1: any `deny` short-circuits, `pass`
 * means "no opinion — ask the next gate", and a chain that resolves
 * with at least one `allow` (and zero `deny`s) results in `allow`. A
 * chain where every gate `pass`-es resolves to `allow` (fail-open).
 *
 * Future extension hooks (NOT implemented in v1):
 *   - `combinator: 'and' | 'or'` field on the chain config
 *   - Stage 2 LLM classifier gate (#83)
 *   - Self-improvement loop reading log records (#82)
 */
import { logger } from '../logger.js';
import type { TriggerPatternConfig } from '../types.js';

export type GateDecisionKind = 'allow' | 'deny' | 'pass';

export interface GateDecision {
  decision: GateDecisionKind;
  reason: string;
}

/**
 * Read-only context handed to each gate. Kept intentionally small —
 * gates that need additional inputs should grow this type explicitly,
 * never reach into module-level state.
 */
export interface GateContext {
  groupJid: string;
  groupFolder: string;
  message: {
    text: string;
    senderJid: string;
    replyToMessageId?: string;
    /**
     * Channel-native mention list (e.g. Telegram entities). Optional —
     * channels that don't surface a structured mention array may leave
     * this undefined and rely on text-substring matching.
     */
    mentions?: string[];
    /**
     * True if the message originated from the bot itself. Gates that
     * care about loop-suppression check this; the trigger gate does
     * not, since `is_from_me` traffic is always allowed by the
     * pre-existing call-site sender check (we only run gates on
     * not-from-me messages today, but pass it through for forward
     * compat).
     */
    isFromMe?: boolean;
  };
  /**
   * Group's full trigger pattern set, as parsed from
   * `registered_groups.trigger_pattern` (#81 / PR #84). The field is
   * required; pass `null` for groups that pre-date the migration
   * backfill or whose row is shape-invalid — the trigger gate treats
   * `null` as "no patterns configured" and returns `pass` rather
   * than failing closed.
   */
  triggerPatterns: TriggerPatternConfig | null;
}

export type GateFn = (ctx: GateContext) => GateDecision;

export interface GateRunRecord {
  gateName: string;
  decision: GateDecisionKind;
  reason: string;
  durationMs: number;
  /** Set when the gate threw — its decision is downgraded to `pass`. */
  error?: { name: string; message: string };
}

export interface GateChainResult {
  finalDecision: 'allow' | 'deny';
  reason: string;
  chain: GateRunRecord[];
  totalDurationMs: number;
}

/**
 * Per-gate latency budget. Slow gates are warned, never killed —
 * killing a gate mid-flight would force a `pass` and hide a real
 * regression. The point of the warn is for the operator to notice
 * before users do.
 */
export const GATE_WARN_DURATION_MS = 50;

const registry: Record<string, GateFn> = {};

export function registerGate(name: string, fn: GateFn): void {
  if (Object.prototype.hasOwnProperty.call(registry, name)) {
    throw new Error(`Gate "${name}" already registered`);
  }
  registry[name] = fn;
}

export function getRegisteredGate(name: string): GateFn | undefined {
  return registry[name];
}

export function listRegisteredGates(): string[] {
  return Object.keys(registry).sort();
}

/**
 * Test-only: drop a gate from the registry. Production code must not
 * call this — gates are registered once at module load and never
 * unregistered.
 */
export function _unregisterGateForTesting(name: string): void {
  delete registry[name];
}

function nowMs(): number {
  return Number(process.hrtime.bigint() / 1_000_000n);
}

/**
 * Walk gates in the order specified by `gateNames` (the per-group
 * config order, NOT the global registration order), applying AND-only
 * combinator semantics. See module docstring for the truth table.
 *
 * Throws are converted to `pass` records so a buggy gate can't
 * black-hole legitimate traffic. Errors are logged with full context
 * so the operator can fix the gate without first having to find the
 * black hole.
 */
export function runGateChain(
  gateNames: string[],
  ctx: GateContext,
): GateChainResult {
  const records: GateRunRecord[] = [];
  const chainStart = nowMs();
  let finalDecision: 'allow' | 'deny' = 'allow';
  let reason = 'no gates configured';
  let sawAllow = false;

  // TODO(future): combinator: 'and' | 'or'. v1 is AND-only — see
  // module docstring. Adding OR means: short-circuit on first allow,
  // and "all pass" stays fail-open here too. Don't add until #82
  // produces a real use case.

  for (const gateName of gateNames) {
    const fn = registry[gateName];
    const start = nowMs();
    if (!fn) {
      const durationMs = nowMs() - start;
      logger.error(
        { groupFolder: ctx.groupFolder, gateName },
        'gate not registered — treating as pass',
      );
      records.push({
        gateName,
        decision: 'pass',
        reason: 'gate not registered',
        durationMs,
        error: { name: 'GateNotRegistered', message: gateName },
      });
      continue;
    }
    let decision: GateDecision;
    let errorRecord: GateRunRecord['error'] | undefined;
    try {
      decision = fn(ctx);
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err));
      errorRecord = { name: e.name, message: e.message };
      decision = { decision: 'pass', reason: `gate threw: ${e.name}` };
      logger.error(
        {
          groupFolder: ctx.groupFolder,
          gateName,
          err: e.message,
          stack: e.stack,
        },
        'gate threw — downgraded to pass',
      );
    }
    const durationMs = nowMs() - start;
    if (durationMs > GATE_WARN_DURATION_MS) {
      logger.warn(
        { groupFolder: ctx.groupFolder, gateName, durationMs },
        'gate exceeded warn duration',
      );
    }
    // Per-gate trace at debug level only — host-side gates fire on
    // every inbound message-poll, so info-level here would dwarf the
    // rest of the orchestrator log. The chain-level summary below
    // keeps an info-tier signal so operators can still see a single
    // line per spawn-decision.
    logger.debug(
      {
        groupFolder: ctx.groupFolder,
        gateName,
        decision: decision.decision,
        reason: decision.reason,
        durationMs,
      },
      'gate evaluated',
    );
    records.push({
      gateName,
      decision: decision.decision,
      reason: decision.reason,
      durationMs,
      error: errorRecord,
    });

    if (decision.decision === 'deny') {
      finalDecision = 'deny';
      reason = decision.reason;
      break;
    }
    if (decision.decision === 'allow') {
      sawAllow = true;
      reason = decision.reason;
    }
    // 'pass' → continue
  }

  if (finalDecision !== 'deny' && !sawAllow) {
    // All gates passed (no opinion). Fail-open.
    finalDecision = 'allow';
    if (gateNames.length > 0) {
      reason = 'all gates passed (no opinion) — fail-open';
    }
  }

  const totalDurationMs = nowMs() - chainStart;
  logger.debug(
    {
      groupFolder: ctx.groupFolder,
      finalDecision,
      reason,
      chain: records.map((r) => ({
        gateName: r.gateName,
        decision: r.decision,
        durationMs: r.durationMs,
      })),
      totalDurationMs,
    },
    'gate chain complete',
  );

  return { finalDecision, reason, chain: records, totalDurationMs };
}

// Built-in gate registration. Side-effect import — the registry is
// populated once at module load and stays constant for the process.
import { triggerGate } from './trigger.js';
registerGate('trigger', triggerGate);
