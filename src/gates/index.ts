/**
 * Host-side Stage 1/2 gate framework (#80, #83, #97).
 *
 * Gates evaluate inbound messages BEFORE the orchestrator spawns a
 * container so no-op spawns (zero SDK queries, zero work) cost zero
 * Docker startup overhead. Gate FUNCTIONS should be pure, stateless,
 * and side-effect-free (they may read DB but never write); the
 * surrounding framework (`runGateChain`) does emit observability log
 * records, so the chain itself is not strictly pure.
 *
 * Combinator semantics — "last-gate-wins" (#97):
 *   - Any gate can decisively `allow`. The first `allow` short-circuits
 *     the chain and becomes the final verdict; later gates are NOT run.
 *     This is the cost-saving path: a deterministic Stage 1 match
 *     prevents a wasted Stage 2 LLM call.
 *   - Only the LAST gate's `deny` is decisive. An intermediate gate's
 *     `deny` is advisory — the chain falls through to the next gate so
 *     a downstream classifier (e.g. Stage 2 Haiku) still gets the
 *     chance to allow grey-zone messages that earlier deterministic
 *     gates couldn't match.
 *   - `pass` always falls through to the next gate (no opinion).
 *   - Chain end with no decisive `allow` and no last-gate `deny` →
 *     fail-open `allow`.
 *
 * Truth table for the canonical `[trigger, haiku-classifier]` chain:
 *
 *   trigger | haiku    | final | rationale
 *   --------|----------|-------|---------------------------------------
 *   allow   | (skip)   | allow | Stage 1 short-circuit, no Haiku spent
 *   pass    | allow    | allow | Stage 2 caught grey-zone yes
 *   pass    | deny     | deny  | Stage 2 said no, last-gate decisive
 *   deny    | allow    | allow | Stage 1 advisory deny ignored
 *   deny    | deny     | deny  | Both agree on no (last-gate decisive)
 *   deny    | pass     | allow | pass + chain-end fail-open
 *
 * Historical note: prior to #97 the combinator was AND-only — `deny`
 * short-circuited from any position and `allow` did not short-circuit.
 * That produced two real-world failures: Stage 1 allows paid for an
 * unnecessary Haiku call, and Stage 1 denies bypassed the Stage 2
 * safety net for exactly the messages it was designed to catch.
 *
 * Future extension hooks (NOT implemented yet):
 *   - Stage 3 / additional gate types — append to the chain; the
 *     last-gate-wins rule keeps composing cleanly.
 *   - Self-improvement loop reading log records (#82).
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
    /**
     * The user's actual message body, with NO inline `[Replying to ...]`
     * quote prefix. The Telegram channel bakes a quote prefix into the
     * stored `content` for the agent prompt path; gates must NOT see it
     * because keyword/mention/synthetic-identity matchers would
     * false-positive on tokens inside the quoted preview (#107). Reply
     * context is exposed via `replyTo` instead.
     */
    text: string;
    senderJid: string;
    /**
     * @deprecated Use `replyTo` instead. Kept for backward compat with
     * the trigger gate's `kind: 'reply'` matcher in the v1 wiring; new
     * code should consult `replyTo.isAssistant`.
     */
    replyToMessageId?: string;
    /**
     * Structured reply metadata for the message being replied to, when
     * the inbound message is a Telegram-style reply. `undefined` when
     * the message is not a reply.
     *
     * Why structured (not just a prefix string): Stage 1's matchers
     * need to see the user's CLEAN body (so `LoMBot` inside a quote
     * preview doesn't false-positive a synthetic identity match), but
     * Stage 2's classifier still needs the reply context as a positive
     * signal. Splitting into `text` + `replyTo` gives both paths what
     * they need.
     */
    replyTo?: {
      messageId: string;
      senderName: string;
      /** True iff the reply target was a bot (any bot, including the assistant). */
      isBot: boolean;
      /** True iff the reply target was sent by THIS bot (the assistant). */
      isAssistant: boolean;
      /** First ~200 chars of the reply target's content. */
      contentPreview: string;
    };
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

/**
 * Gates may be sync or async. Stage 1 deterministic gates (e.g.
 * `trigger`) stay sync — return-type widening is a superset, so
 * existing sync functions still satisfy the type. Stage 2 gates that
 * call out to an LLM (e.g. `haiku-classifier` from #83) return a
 * Promise.
 */
export type GateFn = (ctx: GateContext) => GateDecision | Promise<GateDecision>;

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
 * config order, NOT the global registration order), applying
 * last-gate-wins combinator semantics (#97). See module docstring
 * for the truth table.
 *
 * Short-circuit rules:
 *   - First `allow` → final `allow`, return immediately (later gates
 *     are NOT invoked).
 *   - `deny` from the last gate → final `deny`.
 *   - `deny` from a non-last gate → advisory, fall through.
 *   - `pass` → fall through.
 *   - Chain end with no decisive verdict → fail-open `allow`.
 *
 * Throws are converted to `pass` records so a buggy gate can't
 * black-hole legitimate traffic. Errors are logged with full context
 * so the operator can fix the gate without first having to find the
 * black hole.
 */
export async function runGateChain(
  gateNames: string[],
  ctx: GateContext,
): Promise<GateChainResult> {
  const records: GateRunRecord[] = [];
  const chainStart = nowMs();
  const lastIdx = gateNames.length - 1;

  const finalize = (
    finalDecision: 'allow' | 'deny',
    reason: string,
  ): GateChainResult => {
    const totalDurationMs = nowMs() - chainStart;
    // Hot-path observability at debug level — host-side gates fire on
    // every inbound poll; info-tier per-call would dwarf the
    // orchestrator log.
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
  };

  for (let i = 0; i < gateNames.length; i++) {
    const gateName = gateNames[i];
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
      // Await covers both sync and async return types. Sync gates
      // resolve synchronously through the microtask queue.
      decision = await fn(ctx);
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

    if (decision.decision === 'allow') {
      // Allow always short-circuits — first decisive allow wins.
      // Remaining gates are NOT invoked (cost saving for Stage 2).
      return finalize('allow', decision.reason);
    }
    if (decision.decision === 'deny' && i === lastIdx) {
      // Only the last gate's deny is decisive. Intermediate-gate
      // denies are advisory and fall through to the next gate so a
      // downstream classifier still gets a chance to allow.
      return finalize('deny', decision.reason);
    }
    // pass OR (non-last-gate deny) → continue to the next gate.
  }

  // Chain end with no decisive allow and no last-gate deny → fail-open.
  const reason =
    gateNames.length === 0
      ? 'no gates configured'
      : 'fail-open: no gate produced a decisive verdict';
  return finalize('allow', reason);
}

// Built-in gate registration. Side-effect import — the registry is
// populated once at module load and stays constant for the process.
//
// Ordering note: the `trigger` gate is deterministic and zero-cost;
// the `haiku-classifier` gate (#83) makes an Anthropic API call.
// When `stage2Enabled` is true on a group, `haiku-classifier` is
// appended LAST to the resolved chain in `resolveGatesForGroup`
// (`src/index.ts`) for two reasons under last-gate-wins (#97):
//   1. A trigger `allow` short-circuits BEFORE the API call, so the
//      Anthropic API is only hit for messages Stage 1 couldn't
//      decisively allow.
//   2. The classifier's `deny` becomes the decisive last-gate verdict
//      so it can adjudicate grey-zone messages where Stage 1 said
//      `pass` (or even an advisory `deny`).
// Don't reorder this without re-reading that comment.
import { triggerGate } from './trigger.js';
registerGate('trigger', triggerGate);

import { haikuClassifierGate } from './haiku-classifier.js';
registerGate('haiku-classifier', haikuClassifierGate);
