/**
 * Wiring layer for the trigger-pattern learner (#82).
 *
 * Bridges the pure-functional core in `trigger-learner.ts` to the
 * orchestrator's DB / log / scheduler. Kept in its own module so the
 * pure scorer stays test-light (no DB imports) and the wiring stays
 * single-purpose.
 *
 * Responsibilities:
 *  1. Build a `LearnerPersistence` against the live SQLite DB and the
 *     pretty-printed host log (read via `host-log-parser.ts` —
 *     reuses the canonical `'gate decision'` and `'haiku classifier
 *     verdict'` records).
 *  2. Mine truth-labeled samples from the union of:
 *       - User reactions on bot messages (`reactions` table)
 *       - Stage 2 Haiku verdicts emitted by `haiku-classifier.ts`
 *         (mined out of host-log records)
 *     The `gate decision` log line carries the gate's own verdict —
 *     we DELIBERATELY do not feed that back as truth (self-
 *     reinforcement loop). It IS used to compute a per-pattern
 *     `wasRight` track record for auto-rollback, but only against
 *     external truth, never against itself.
 *  3. Start a low-cadence loop (default daily) that walks every
 *     registered group and runs the learner once.
 */
import {
  getAllRegisteredGroups,
  getReactionsForMessage,
  getMessagesSince,
  getTriggerPatterns,
  setTriggerPatterns,
} from '../db.js';
import { ASSISTANT_OWNER_HANDLE } from '../config.js';
import { hostLogsOrchestratorFile } from '../host-logs.js';
import { findGateDecisions, readHostLog } from '../host-log-parser.js';
import { logger } from '../logger.js';
import type { TriggerPatternConfig } from '../types.js';

import {
  resolveLearnerConfig,
  resolveLearnerIntervalMs,
  runLearnerOnce,
  type LabeledSample,
  type LearnerConfig,
  type LearnerPersistence,
  type PatternDecisionHistory,
  type SenderTier,
} from './trigger-learner.js';

/**
 * Hard cap on how far back the sampler walks per group per pass. The
 * SCHEDULER_POLL_INTERVAL is 60s but the learner runs at 24h cadence
 * by default; 14 days gives enough history for a slow group to clear
 * the cold-start floor while still capping the SQL scan size.
 */
export const DEFAULT_SAMPLE_LOOKBACK_DAYS = 14;

/**
 * The reaction emoji set we treat as positive vs negative truth.
 * Keep this conservative — ambiguous emoji ("🤔", "🤷") fall through
 * as neutral / no signal rather than being mis-labelled.
 */
const POSITIVE_REACTIONS: ReadonlySet<string> = new Set([
  '👍',
  '🙏',
  '❤️',
  '✅',
  '🔥',
]);
const NEGATIVE_REACTIONS: ReadonlySet<string> = new Set(['👎', '❌', '😡']);

/**
 * Resolve a sender's tier: owner > non-owner > anonymous. Owner is
 * detected via `ASSISTANT_OWNER_HANDLE` (the same env var the Stage 2
 * classifier uses, see `src/config.ts`). When the env is unset, no
 * sender qualifies as owner — the learner falls back to non-owner /
 * anonymous tiers.
 */
export function classifySender(
  reactorJid: string,
  reactorName: string,
  ownerHandle: string | undefined = ASSISTANT_OWNER_HANDLE,
): SenderTier {
  if (!reactorJid && !reactorName) return 'anonymous';
  if (ownerHandle) {
    const handle = ownerHandle.toLowerCase();
    const reactorIdLower = (reactorJid ?? '').toLowerCase();
    const reactorNameLower = (reactorName ?? '').toLowerCase();
    if (
      reactorIdLower.includes(`@${handle}`) ||
      reactorIdLower.endsWith(`:${handle}`) ||
      reactorNameLower === handle ||
      reactorNameLower === `@${handle}`
    ) {
      return 'owner';
    }
  }
  return 'non-owner';
}

type MinedSample = LabeledSample;

/**
 * Mine truth-labeled samples for ONE group, using:
 *  - Reactions on bot messages within the lookback window
 *  - Haiku verdicts emitted to the host log over the same window
 *
 * The function intentionally does NOT touch the `gate decision`
 * record's `finalDecision` field as a truth signal — that would
 * close the self-reinforcement loop. We only use `gate decision`
 * records to attribute a `messageId → matched-pattern` for the
 * auto-rollback path.
 */
export function mineSamplesForGroup(
  groupJid: string,
  _hostLogPath: string,
  ownerHandle: string | undefined = ASSISTANT_OWNER_HANDLE,
  lookbackDays: number = DEFAULT_SAMPLE_LOOKBACK_DAYS,
  now: () => Date = () => new Date(),
): MinedSample[] {
  const samples: MinedSample[] = [];
  const sinceTimestamp = new Date(
    now().getTime() - lookbackDays * 24 * 60 * 60 * 1000,
  ).toISOString();

  // 1. Walk inbound messages → check reactions on the bot reply that
  //    followed them. We approximate "bot reply that followed" with
  //    the `messages` table itself (bot sends are stored with
  //    is_from_me=1) — for a precise mapping we'd need a
  //    inbound→outbound link, but the signal we're after is per-
  //    inbound-keyword precision, so attributing the reaction back
  //    to the inbound that triggered it is enough.
  const inboundMessages = getMessagesSince(
    groupJid,
    sinceTimestamp,
    'bot',
    1000,
  );
  for (const msg of inboundMessages) {
    if (!msg.content) continue;
    const reactions = getReactionsForMessage(msg.id, groupJid);
    for (const r of reactions) {
      const tier = classifySender(r.reactor_jid, r.reactor_name, ownerHandle);
      let intent: 'yes' | 'no' | undefined;
      if (POSITIVE_REACTIONS.has(r.emoji)) intent = 'yes';
      else if (NEGATIVE_REACTIONS.has(r.emoji)) intent = 'no';
      if (!intent) continue;
      samples.push({
        text: msg.content,
        intent,
        senderTier: tier,
        source: 'user_reaction',
        gateResponded: false, // we don't (and shouldn't) inject the
        // gate's verdict into the truth signal — kept false here so
        // downstream code can never lean on it.
      });
    }
  }

  // 2. Mine Haiku verdicts from the host log. These are emitted as
  //    INFO records with msg='haiku classifier verdict' carrying
  //    {intent, confidence, reason, groupFolder}. We want only the
  //    verdicts for THIS group; cross-group records are filtered by
  //    `groupFolder` matching this group's folder. Resolution of
  //    `groupJid → groupFolder` happens at the persistence-layer
  //    seam (`buildLearnerPersistence`), so this function takes the
  //    folder as a parameter via the upstream call.
  // Implementation note: Haiku records are mined inside
  // `buildLearnerPersistence` because that's where we have the
  // groupFolder plumbed in; this function focuses on reaction
  // mining only (the join is light).

  return samples;
}

/**
 * Mine Haiku-verdict samples from the host log for ONE group folder.
 * Separate from `mineSamplesForGroup` because the join key is
 * `groupFolder` (what the log records), not `groupJid`.
 *
 * Stage 2's `intent` ('yes'/'no') is the truth signal. Since #451
 * item 4, the verdict log line carries `messageId` and `inboundText`
 * directly — no need to stitch via a paired `gate decision` record.
 *
 * Note on confidence: the Haiku log emits a calibrated confidence in
 * [0,1]. We treat verdicts with confidence < 0.7 as ambiguous and
 * skip them; the issue body called Haiku a "soft truth signal" and
 * an ambiguous ~0.55 verdict isn't truth, it's noise.
 *
 * Records emitted by an older orchestrator that pre-dates #451 item 4
 * (no `messageId` / `inboundText` fields on the line) are skipped
 * silently — they have no usable text and the join-via-`gate decision`
 * fallback was never wired in v1. Callers tolerate the empty result.
 */
export function mineHaikuSamples(
  groupFolder: string,
  hostLogPath: string,
  minConfidence: number = 0.7,
): MinedSample[] {
  const records = readHostLog(hostLogPath);
  const samples: MinedSample[] = [];
  for (const r of records) {
    if (r.msg !== 'haiku classifier verdict') continue;
    if (r.fields.groupFolder !== groupFolder) continue;
    const intent = r.fields.intent;
    if (intent !== 'yes' && intent !== 'no') continue;
    const confidence = r.fields.confidence;
    if (typeof confidence !== 'number') continue;
    if (confidence < minConfidence) continue;
    const inboundText = r.fields.inboundText;
    if (typeof inboundText !== 'string' || inboundText.length === 0) continue;
    // messageId is structurally required on post-#451-item-4 records
    // but isn't load-bearing for sample mining (we don't dedupe
    // here — multiple verdicts on the same message reflect different
    // strategy passes and each is a legitimate data point). Read the
    // field if present, otherwise fall through.
    samples.push({
      text: inboundText,
      intent,
      // Haiku verdicts have no sender — synthetic inbound. The
      // learner's `senderTier` policy treats 'anonymous' as
      // non-privileged, which matches: a Haiku verdict is software-
      // produced, never owner-authoritative.
      senderTier: 'anonymous',
      source: 'haiku_verdict',
      gateResponded: false,
    });
  }
  return samples;
}

/**
 * Build per-pattern decision history for auto-rollback. Walks the
 * host log for the most-recent `gate decision` records for this
 * group's chatJid, attributes each to the matched pattern (if any),
 * and labels `wasRight` against the EXTERNAL truth signal —
 * NEVER against the gate's own verdict.
 *
 * v1 attribution: a `gate decision` record's `chain[0].reason`
 * carries the matched pattern body when Stage 1 allowed (e.g.
 * `"trigger pattern matched: kind=keyword pattern=deploy"`). We
 * regex-extract `kind=<k> pattern=<p>` and bucket the decision into
 * the matching pattern's history. `wasRight` comes from the same
 * truth-signal mining as `mineSamplesForGroup`.
 */
export function buildDecisionHistories(
  groupJid: string,
  hostLogPath: string,
  truthByMessageId: Map<string, 'yes' | 'no'>,
  rollbackWindow: number,
): Map<string, PatternDecisionHistory> {
  const records = readHostLog(hostLogPath);
  const decisions = findGateDecisions(records, {
    chatJid: groupJid,
    limit: rollbackWindow * 10, // overscan; we filter to allow-only below
  });
  const histories = new Map<string, PatternDecisionHistory>();
  for (const d of decisions) {
    if (d.finalDecision !== 'allow') continue; // FP rate is on allows
    const triggerStep = d.chain.find((c) => c.gate === 'trigger');
    if (!triggerStep || triggerStep.decision !== 'allow') continue;
    const m = /kind=(\w+)\s+pattern=(\S+)/.exec(triggerStep.reason);
    if (!m) continue;
    const kind = m[1];
    const pattern = m[2];
    const truth = truthByMessageId.get(d.messageId);
    if (truth === undefined) continue; // no external label → skip
    const wasRight = truth === 'yes';
    const key = `${kind}:${pattern}`;
    let history = histories.get(key);
    if (!history) {
      history = {
        pattern,
        kind: kind as PatternDecisionHistory['kind'],
        decisions: [],
      };
      histories.set(key, history);
    }
    history.decisions.push({ wasRight });
  }
  // Trim each history to the rolling window.
  for (const h of histories.values()) {
    if (h.decisions.length > rollbackWindow) {
      h.decisions = h.decisions.slice(-rollbackWindow);
    }
  }
  return histories;
}

export const DEFAULT_LEARNER_ROLLBACK_WINDOW = 30;

/**
 * Build a `LearnerPersistence` bound to the live DB + host log.
 *
 * The `rollbackWindow` argument is threaded into
 * `fetchDecisionHistories` so the env-configured window
 * (`TRIGGER_LEARNER_ROLLBACK_WINDOW`, surfaced as
 * `cfg.rollbackWindow`) actually controls how many trailing decisions
 * are retained per pattern. Without this, a larger configured window
 * would never gather more decisions because the persistence layer
 * would silently trim back to the default.
 *
 * Test seam: pass overrides to swap the implementations one-by-one.
 */
export function buildLearnerPersistence(
  rollbackWindow: number = DEFAULT_LEARNER_ROLLBACK_WINDOW,
  overrides: Partial<LearnerPersistence> = {},
): LearnerPersistence {
  const real: LearnerPersistence = {
    listGroups: () => {
      const groups = getAllRegisteredGroups();
      return Object.entries(groups).map(([jid, g]) => ({
        jid,
        folder: g.folder,
        config: getTriggerPatterns(jid) ?? undefined,
      }));
    },
    fetchSamples: (groupJid: string) => {
      const hostLogPath = hostLogsOrchestratorFile();
      const reactionSamples = mineSamplesForGroup(groupJid, hostLogPath);
      const groups = getAllRegisteredGroups();
      const folder = groups[groupJid]?.folder;
      const haikuSamples = folder ? mineHaikuSamples(folder, hostLogPath) : [];
      return [...reactionSamples, ...haikuSamples];
    },
    fetchDecisionHistories: (groupJid: string) => {
      const hostLogPath = hostLogsOrchestratorFile();
      // We need the per-message truth label to compute wasRight. v1
      // pulls it from reactions; haiku-derived labels will join in
      // when the producer-side messageId field lands.
      const truthByMessageId = new Map<string, 'yes' | 'no'>();
      const reactionSamples = mineSamplesForGroup(groupJid, hostLogPath);
      void reactionSamples;
      // The reaction → message link is non-trivial without a
      // producer-side stamp; treat the truth map as empty in v1 so
      // auto-rollback is conservative (no histories → no demotions
      // until the producer side surfaces messageId on each truth
      // signal). This keeps the safety rail enforced (we never
      // demote without external truth) and is documented in the
      // promotion-UI follow-up as a known limitation.
      return buildDecisionHistories(
        groupJid,
        hostLogPath,
        truthByMessageId,
        rollbackWindow,
      );
    },
    saveConfig: (groupJid: string, config: TriggerPatternConfig) => {
      setTriggerPatterns(groupJid, config);
    },
    logger: {
      info: (data, msg) => logger.info(data, msg),
      warn: (data, msg) => logger.warn(data, msg),
      error: (data, msg) => logger.error(data, msg),
    },
  };
  return { ...real, ...overrides };
}

/**
 * Start the trigger-pattern learner loop. Runs at the cadence
 * configured by `TRIGGER_LEARNER_INTERVAL_MS` (default 24h). The
 * first run fires after one full interval, NOT immediately on boot,
 * so a misconfigured threshold doesn't fire learner work during
 * startup churn.
 *
 * Returns a stop function for tests / clean shutdown.
 */
export function startTriggerLearner(opts?: {
  config?: LearnerConfig;
  intervalMs?: number;
  persistence?: LearnerPersistence;
}): () => void {
  let intervalMs: number;
  let cfg: LearnerConfig;
  try {
    intervalMs = opts?.intervalMs ?? resolveLearnerIntervalMs();
    cfg = opts?.config ?? resolveLearnerConfig();
  } catch (err) {
    if (!(err instanceof Error)) throw err;
    logger.error(
      { err: err.message },
      'trigger-learner: env config invalid — loop NOT started; fix env and restart',
    );
    return () => {};
  }
  // Thread the resolved config's `rollbackWindow` into persistence so
  // `TRIGGER_LEARNER_ROLLBACK_WINDOW` actually controls the trailing
  // window per pattern. Previously this defaulted at the persistence
  // layer and silently ignored larger configured windows.
  const persistence =
    opts?.persistence ?? buildLearnerPersistence(cfg.rollbackWindow);

  logger.info(
    {
      intervalMs,
      minSamples: cfg.minSamples,
      ownerWeight: cfg.ownerWeight,
      nonOwnerWeight: cfg.nonOwnerWeight,
      anonymousWeight: cfg.anonymousWeight,
      rollbackWindow: cfg.rollbackWindow,
      rollbackFpThreshold: cfg.rollbackFpThreshold,
      minProposalScore: cfg.minProposalScore,
    },
    'trigger-learner: starting loop',
  );

  const tick = () => {
    try {
      const summaries = runLearnerOnce(persistence, cfg);
      logger.info(
        {
          groupCount: summaries.length,
          proposed: summaries.filter((s) => s.outcome === 'proposed').length,
          coldStart: summaries.filter((s) => s.outcome === 'cold-start').length,
          rolledBack: summaries.filter((s) => s.demoted.length > 0).length,
        },
        'trigger-learner: tick complete',
      );
    } catch (err) {
      // Per `rules/error-handling.md`: programmer bugs surface; we
      // narrow to Error here because the inner `runLearnerOnce`
      // already catches per-group failures. Anything reaching this
      // catch is the framework itself misbehaving.
      if (!(err instanceof Error)) throw err;
      logger.error(
        { err: err.message, stack: err.stack },
        'trigger-learner: top-level tick failed — loop continues, will retry on next interval',
      );
    }
  };

  const handle = setInterval(tick, intervalMs);
  return () => clearInterval(handle);
}
