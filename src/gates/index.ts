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
 *   trigger | haiku      | final | rationale
 *   --------|------------|-------|-------------------------------------
 *   allow   | (skip)     | allow | Stage 1 short-circuit, no Haiku spent
 *   pass    | allow      | allow | Stage 2 caught grey-zone yes
 *   pass    | deny       | deny  | Stage 2 said no, last-gate decisive
 *   deny    | allow      | allow | Stage 1 advisory deny ignored
 *   deny    | deny       | deny  | Both agree on no (last-gate decisive)
 *   deny    | pass(ok)   | allow | healthy pass + chain-end fail-open
 *   deny    | pass(fail) | deny  | classifier FAILED — upstream advisory
 *           |            |       | deny preserved, not nullified (#671)
 *   pass    | pass(fail) | allow | classifier FAILED but no upstream deny
 *           |            |       | to preserve → fail-open
 *
 * The `pass(fail)` rows close the #671 money-bleed: the Stage 2 Haiku
 * classifier returns `pass` ONLY when it could not run (api-error /
 * timeout / unparseable / no-client), tagged `failed: true`. Under the
 * plain last-gate-wins rule a failed `pass` let a deterministic trigger
 * `deny` fall through to fail-open `allow`, so every transient
 * classifier outage silently degraded every non-strict Stage 2 group to
 * allow-all (one container spawn per message). A FAILED downstream gate
 * must never be MORE permissive than a healthy "no": when a gate that
 * runs after an advisory `deny` fails, the chain preserves that `deny`
 * instead of fail-opening. A HEALTHY classifier never returns `pass`, so
 * the `deny | pass(ok) | allow` row is unreachable for this chain today
 * and stays as documented.
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
  /**
   * Set by a gate that returns `pass` because it could NOT form an
   * opinion (the Stage 2 classifier's API call errored, timed out, or
   * returned an unparseable verdict), as distinct from a healthy "no
   * opinion" pass. `runGateChain` uses this so a FAILED downstream gate
   * can't nullify an upstream advisory `deny` — a transient classifier
   * outage must never be MORE permissive than a healthy "no" (#671). A
   * healthy pass leaves this unset.
   */
  failed?: boolean;
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
     * Channel-side message id for the inbound (#451 item 4). Required
     * so the producer-side enrichment of `haiku classifier verdict`
     * can stamp `messageId` directly on the verdict log line —
     * `mineHaikuSamples` no longer needs to stitch via a paired
     * `gate decision` record. Plumbed from `NewMessage.id` at the
     * `buildGateContext` call site in the orchestrator.
     */
    messageId: string;
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

/**
 * Gates may throw this for an explicitly recoverable operational failure that
 * should degrade to a failed `pass` instead of propagating. Any other throw is
 * treated as unexpected and is re-thrown per `coding-policy: error-handling`.
 */
export class RecoverableGateError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    Object.setPrototypeOf(this, new.target.prototype);
    this.name = 'RecoverableGateError';
  }
}

export interface GateRunRecord {
  gateName: string;
  decision: GateDecisionKind;
  reason: string;
  durationMs: number;
  /** Set when the gate threw — its decision is downgraded to `pass`. */
  error?: { name: string; message: string };
  /**
   * True when the gate could not form an opinion — when `error` is set
   * (it threw, or was unregistered) or it returned `pass` with
   * `failed: true`. This is the
   * internal signal `runGateChain` uses to decide whether to preserve an
   * upstream advisory deny at chain end (#671). The operator-facing
   * explanation of a preserved deny is the chain's final `reason`
   * (`upstream deny preserved (downstream gate failed): ...`), which the
   * `gate decision` log line emits; this per-gate marker is not part of
   * that log's chain shape.
   */
  failed?: boolean;
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
 *   - `deny` from a non-last gate → advisory, fall through (but
 *     remembered — see the fail-preservation rule below).
 *   - `pass` → fall through.
 *   - Chain end with no decisive verdict → fail-open `allow`, UNLESS a
 *     gate that ran after the most recent advisory `deny` FAILED (threw,
 *     or returned `pass` with `failed: true`). In that case the advisory
 *     `deny` is preserved as the final verdict (#671): a failed gate
 *     can't be the reason a deterministic `deny` gets nullified into an
 *     allow-all.
 *
 * A gate that throws an explicit {@link RecoverableGateError} is
 * converted to a FAILED `pass` record so a transient operational blip
 * can't black-hole legitimate traffic; the throw is logged with full
 * context and the record is marked `failed` so it participates in the
 * deny-preservation rule above. Any other throw is NOT swallowed — it
 * propagates out of the chain per `coding-policy: error-handling` (let
 * unexpected exceptions propagate) so a code bug or unclassified error
 * surfaces loudly instead of silently degrading the chain to fail-open.
 * The caller's per-group retry/backoff (`group-queue.ts` `runForGroup`)
 * contains the throw to one group (eventually the circuit breaker), not
 * the orchestrator.
 */
export async function runGateChain(
  gateNames: string[],
  ctx: GateContext,
): Promise<GateChainResult> {
  const records: GateRunRecord[] = [];
  const chainStart = nowMs();
  const lastIdx = gateNames.length - 1;

  // Deny-preservation state (#671). `advisoryDenyReason` holds the
  // reason from the most recent non-last `deny` (an advisory deny that
  // fell through). `safetyNetFailed` tracks whether any gate that ran
  // AFTER that advisory deny could not form an opinion (threw, or
  // returned `pass` with `failed: true`). When both hold at chain end,
  // the advisory deny is preserved instead of fail-opening — a failed
  // downstream gate must not be more permissive than a healthy "no".
  let advisoryDenyReason: string | null = null;
  let safetyNetFailed = false;

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
    let decision: GateDecision;
    let errorRecord: GateRunRecord['error'] | undefined;
    if (!fn) {
      // An unregistered gate can't form an opinion. Treat it as a
      // FAILED pass (not a healthy one) so it flows through the same
      // failure accounting below and participates in deny-preservation
      // (#671) — otherwise an unregistered LAST gate after an advisory
      // deny would silently fall through to fail-open allow.
      errorRecord = { name: 'GateNotRegistered', message: gateName };
      decision = { decision: 'pass', reason: 'gate not registered' };
      logger.error(
        { groupFolder: ctx.groupFolder, gateName },
        'gate not registered — treating as pass',
      );
    } else {
      try {
        // Await covers both sync and async return types. Sync gates
        // resolve synchronously through the microtask queue.
        decision = await fn(ctx);
      } catch (err) {
        if (!(err instanceof RecoverableGateError)) {
          // Per `coding-policy: error-handling`, swallow ONLY the
          // explicitly recoverable gate failure shape. Any other
          // exception is unexpected/unclassified and must propagate so
          // it surfaces loudly instead of degrading the chain. Log the
          // gate context first: the caller's catch (`group-queue.ts`
          // `runForGroup`) logs only `{ groupJid, err }`, so without this
          // the gateName/groupFolder needed for triage would be lost.
          const ue = err instanceof Error ? err : new Error(String(err));
          logger.error(
            {
              groupFolder: ctx.groupFolder,
              gateName,
              err: ue.message,
              stack: ue.stack,
            },
            'gate threw an unexpected error — propagating',
          );
          throw err;
        }
        const e = err;
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
    // A gate "failed" — could not form an opinion — when it threw / was
    // unregistered (errorRecord set) OR returned `pass` with
    // `failed: true`. The `decision === 'pass'` guard keeps a future gate
    // from marking an `allow`/`deny` as failed and skewing the rule (#671).
    const gateFailed =
      errorRecord !== undefined ||
      (decision.decision === 'pass' && decision.failed === true);
    records.push({
      gateName,
      decision: decision.decision,
      reason: decision.reason,
      durationMs,
      error: errorRecord,
      failed: gateFailed || undefined,
    });

    if (decision.decision === 'allow') {
      // Allow always short-circuits — first decisive allow wins.
      // Remaining gates are NOT invoked (cost saving for Stage 2).
      return finalize('allow', decision.reason);
    }
    if (decision.decision === 'deny') {
      if (i === lastIdx) {
        // Only the last gate's deny is decisive.
        return finalize('deny', decision.reason);
      }
      // Intermediate-gate deny is advisory and falls through so a
      // downstream classifier still gets a chance to allow. Remember it
      // and reset the failure watch: only failures from gates that run
      // AFTER this deny can justify preserving it.
      advisoryDenyReason = decision.reason;
      safetyNetFailed = false;
      continue;
    }
    // pass → fall through. If this pass was a FAILURE and we're holding
    // an advisory deny, the safety net that justified treating that deny
    // as advisory just collapsed.
    if (gateFailed && advisoryDenyReason !== null) {
      safetyNetFailed = true;
    }
  }

  // Chain end. A downstream gate failed after an advisory deny → preserve
  // the deny rather than fail-opening (#671).
  if (advisoryDenyReason !== null && safetyNetFailed) {
    return finalize(
      'deny',
      `upstream deny preserved (downstream gate failed): ${advisoryDenyReason}`,
    );
  }
  // Otherwise no decisive allow and no last-gate deny → fail-open.
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
