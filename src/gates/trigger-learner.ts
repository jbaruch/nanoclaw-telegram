/**
 * Self-improvement loop on Stage 1 trigger patterns (#82).
 *
 * Periodically samples observed message↔response correlations and
 * proposes Stage 1 trigger-pattern adjustments per group. PROPOSALS
 * ONLY — they land in the `registered_groups.trigger_pattern` JSON
 * column with `source: 'learned'` and `enabled: false` until the
 * owner promotes them via the existing admin path. The trigger gate
 * already skips `enabled: false` rows (see `src/gates/trigger.ts`),
 * so a proposal in flight has zero matcher effect.
 *
 * ============================================================
 * NON-NEGOTIABLE SAFETY RAILS (from ligolnik/nanoclaw-public#82)
 * ============================================================
 *
 * 1. **External corrective signal only.** Truth comes from the
 *    union of: user reactions on bot messages and owner explicit
 *    corrections. The agent's own Stage 1 verdict is NEVER used as
 *    truth — that closes a self-reinforcement loop where the gate
 *    learns to confirm whatever it already does.
 *
 * 2. **Cold-start floor.** Until at least N truth-labeled samples
 *    accumulate per group (default 50, configurable via
 *    `TRIGGER_LEARNER_MIN_SAMPLES` env), the loop emits NO
 *    proposals — the matcher falls through to the universal /
 *    owner-set patterns. Below the floor, scores are too noisy
 *    to act on.
 *
 * 3. **Sender-tier weighting.** Owner reactions / corrections
 *    carry more weight than non-owner traffic. Default weights:
 *    owner=3.0, non-owner=1.0, anonymous=0.5. Tunable via
 *    `TRIGGER_LEARNER_OWNER_WEIGHT` etc.
 *
 * 4. **Auto-rollback on precision drop.** A learned pattern whose
 *    FP rate over the last K (default 30) decisions exceeds X%
 *    (default 30%) is demoted via `disabled: true`. The row is
 *    KEPT (not deleted) — the owner can inspect why and re-enable
 *    after fixing the root cause.
 *
 * 5. **Pattern versioning.** Every learned pattern carries a
 *    `pattern_version` integer and a `proposed_at` timestamp.
 *    When the loop supersedes a prior proposal, the previous
 *    record is preserved under `prior_versions[]` (capped at one
 *    level deep) so the owner can revert manually.
 *
 * ============================================================
 * SCOPE OF THIS MODULE
 * ============================================================
 *
 * IN:
 *   - Pure functions for scoring / proposing / demoting (testable)
 *   - A persistence wrapper (`runLearnerForGroup`) that reads the
 *     truth signal, computes proposals, and writes them back via
 *     `setTriggerPatterns`
 *   - Scheduler hook (`startTriggerLearner`) that wires the loop
 *     into the existing scheduler at a configurable cadence
 *
 * OUT (deferred to follow-up issues):
 *   - The promotion UI / admin path that flips `enabled: false` →
 *     `enabled: true`. The hooks are present in the schema; the
 *     UX is not built here.
 *   - Per-group dashboard surfacing precision / lift / proposal
 *     queue. See follow-up.
 *   - Truth-signal mining for owner explicit corrections from
 *     freeform group chat (e.g. "you should have replied to that").
 *     The `OwnerCorrection` shape is defined here for the writer
 *     contract; the reader/parser is its own scoped feature.
 */
import type {
  TriggerPattern,
  TriggerPatternConfig,
  TriggerPatternKind,
} from '../types.js';

/**
 * Default cold-start floor: until N truth-labeled samples accumulate
 * per group, the learner emits no proposals. 50 is large enough that
 * a handful of accidental signals can't bootstrap a bad pattern, small
 * enough that a normal active group reaches the floor in days.
 */
export const DEFAULT_MIN_SAMPLES = 50;

/** Default sender-tier weights. Match the issue spec verbatim. */
export const DEFAULT_OWNER_WEIGHT = 3.0;
export const DEFAULT_NON_OWNER_WEIGHT = 1.0;
export const DEFAULT_ANONYMOUS_WEIGHT = 0.5;

/**
 * Default rolling window for FP-rate auto-rollback. K=30 trailing
 * decisions per pattern is short enough that a precision regression
 * surfaces within one or two days of normal traffic, long enough that
 * a single noisy hour can't trip the rollback on a low-traffic group.
 */
export const DEFAULT_ROLLBACK_WINDOW = 30;

/**
 * Default false-positive-rate threshold for auto-rollback. >30% FP
 * over the rolling window demotes the pattern. Strict enough to catch
 * obviously-mismatched proposals, lenient enough that mid-quality
 * proposals stay live until the owner reviews them.
 */
export const DEFAULT_ROLLBACK_FP_THRESHOLD = 0.3;

/**
 * Minimum positive weighted score before a candidate is promoted to
 * a proposal. Scores below this are observed silently — the learner
 * collects evidence but doesn't write a row. This is the load-bearing
 * filter that prevents the proposal queue from filling up with weak
 * signal.
 */
export const DEFAULT_MIN_PROPOSAL_SCORE = 2.0;

/**
 * Default cadence for the scheduler hook. Daily is slow enough that
 * a handful of bad samples don't immediately produce a bad proposal,
 * fast enough that operators see new proposals within a working day.
 */
export const DEFAULT_LEARNER_INTERVAL_MS = 24 * 60 * 60 * 1000;

/**
 * Sender tier for weighting truth signals. The owner's voice is
 * authoritative — they can definitively say "you should have replied"
 * or "stop replying to this". Non-owner = a normal participant.
 * Anonymous = a sender we couldn't classify (no senderJid, e.g. legacy
 * imported message).
 */
export type SenderTier = 'owner' | 'non-owner' | 'anonymous';

/**
 * Truth-signal source. Each carries a different default weight and
 * different staleness semantics; the learner combines them but
 * NEVER uses Stage 1's own verdict — that's the self-reinforcement
 * loop the safety rails forbid.
 */
export type TruthSource =
  | 'user_reaction' // 👍 / 👎 / similar emoji on the bot's response
  | 'owner_correction'; // freeform owner text correcting the gate

/**
 * One labeled sample. Each represents an inbound message + the
 * external truth signal that says whether the gate's decision
 * (responded vs. stayed silent) was right or wrong.
 *
 * Fields are intentionally minimal — the scorer only needs the
 * features it uses (text, sender tier, intent, source). Adding more
 * features (timestamps, reply context) would couple this to the
 * mining layer; keep it small.
 */
export interface LabeledSample {
  /** The inbound message text — used for keyword candidate extraction. */
  text: string;
  /** Sender tier — drives the per-sample weight. */
  senderTier: SenderTier;
  /**
   * The external truth label: `yes` = the assistant SHOULD have
   * engaged on this message; `no` = the assistant should NOT have.
   * The label is the union of all truth sources in scope; the source
   * itself is preserved in `source` so the scorer can apply
   * source-specific weights if needed.
   */
  intent: 'yes' | 'no';
  /** Which signal produced the label. */
  source: TruthSource;
  /**
   * Whether the gate actually responded to this message. Used to
   * compute precision (TP / (TP + FP)) when evaluating an existing
   * pattern's track record — NOT used as truth itself.
   */
  gateResponded: boolean;
}

/**
 * Config knobs the scheduler hook reads from env. All optional;
 * the defaults above apply when unset.
 */
export interface LearnerConfig {
  minSamples: number;
  ownerWeight: number;
  nonOwnerWeight: number;
  anonymousWeight: number;
  rollbackWindow: number;
  rollbackFpThreshold: number;
  minProposalScore: number;
}

/**
 * Resolve the active learner config from env, falling back to
 * documented defaults. Pure function — no side effects, no DB.
 *
 * Per `rules/error-handling.md`: invalid env values surface a
 * descriptive `Error` rather than silently falling back, so an
 * operator typo is visible at scheduler-tick time, not after a
 * day of weirdness.
 */
export function resolveLearnerConfig(
  env: NodeJS.ProcessEnv = process.env,
): LearnerConfig {
  function parseFloatEnv(name: string, defaultValue: number): number {
    const raw = env[name];
    if (raw === undefined || raw === '') return defaultValue;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0) {
      throw new Error(
        `[trigger-learner] env ${name}="${raw}" is not a non-negative finite number; ` +
          `expected a number ≥ 0 or unset for default ${defaultValue}`,
      );
    }
    return parsed;
  }
  function parseIntEnv(name: string, defaultValue: number): number {
    const raw = env[name];
    if (raw === undefined || raw === '') return defaultValue;
    if (!/^\d+$/.test(raw)) {
      throw new Error(
        `[trigger-learner] env ${name}="${raw}" is not a non-negative integer; ` +
          `expected an integer ≥ 0 or unset for default ${defaultValue}`,
      );
    }
    return parseInt(raw, 10);
  }
  return {
    minSamples: parseIntEnv('TRIGGER_LEARNER_MIN_SAMPLES', DEFAULT_MIN_SAMPLES),
    ownerWeight: parseFloatEnv(
      'TRIGGER_LEARNER_OWNER_WEIGHT',
      DEFAULT_OWNER_WEIGHT,
    ),
    nonOwnerWeight: parseFloatEnv(
      'TRIGGER_LEARNER_NON_OWNER_WEIGHT',
      DEFAULT_NON_OWNER_WEIGHT,
    ),
    anonymousWeight: parseFloatEnv(
      'TRIGGER_LEARNER_ANONYMOUS_WEIGHT',
      DEFAULT_ANONYMOUS_WEIGHT,
    ),
    rollbackWindow: parseIntEnv(
      'TRIGGER_LEARNER_ROLLBACK_WINDOW',
      DEFAULT_ROLLBACK_WINDOW,
    ),
    rollbackFpThreshold: parseFloatEnv(
      'TRIGGER_LEARNER_ROLLBACK_FP_THRESHOLD',
      DEFAULT_ROLLBACK_FP_THRESHOLD,
    ),
    minProposalScore: parseFloatEnv(
      'TRIGGER_LEARNER_MIN_PROPOSAL_SCORE',
      DEFAULT_MIN_PROPOSAL_SCORE,
    ),
  };
}

/**
 * Resolve a sender's tier weight from the config. Pure function;
 * used both at scoring time and exposed for tests.
 */
export function weightForTier(tier: SenderTier, cfg: LearnerConfig): number {
  switch (tier) {
    case 'owner':
      return cfg.ownerWeight;
    case 'non-owner':
      return cfg.nonOwnerWeight;
    case 'anonymous':
      return cfg.anonymousWeight;
  }
}

// ----- Keyword candidate extraction ------------------------------------

/**
 * Stop words that would generate noisy proposals if scored as
 * keyword candidates. Conservative list — covering the most common
 * function words across English/Russian.
 *
 * Not exhaustive: the goal is to drop obvious non-candidates, not
 * to be linguistically precise. The min-score gate downstream is
 * what actually filters weak candidates; this just keeps the
 * candidate pool small.
 */
const STOP_WORDS: ReadonlySet<string> = new Set([
  'the',
  'a',
  'an',
  'is',
  'are',
  'was',
  'were',
  'and',
  'or',
  'but',
  'to',
  'of',
  'in',
  'on',
  'at',
  'for',
  'with',
  'by',
  'as',
  'it',
  'its',
  'this',
  'that',
  'these',
  'those',
  'i',
  'you',
  'he',
  'she',
  'we',
  'they',
  'me',
  'us',
  'them',
  'my',
  'your',
  'his',
  'her',
  'our',
  'their',
  'be',
  'been',
  'being',
  'do',
  'does',
  'did',
  'have',
  'has',
  'had',
  'will',
  'would',
  'can',
  'could',
  'should',
  'may',
  'might',
  'so',
  'no',
  'not',
  'yes',
  'ok',
  'okay',
  'и',
  'в',
  'на',
  'я',
  'не',
  'что',
  'это',
  'как',
  'но',
  'а',
  'к',
  'у',
  'по',
  'из',
  'за',
  'от',
  'для',
  'же',
  'ты',
  'мы',
  'они',
  'он',
  'она',
  'оно',
  'да',
  'нет',
]);

/**
 * Tokenise a message body for keyword-candidate extraction. Splits
 * on Unicode word boundaries, lowercases, drops stop-words and
 * tokens shorter than 3 characters.
 *
 * Why a regex here is OK (vs. the regex-trap rule): we are NOT
 * parsing meaning — we're extracting word-shaped candidates from a
 * fully-enumerable token boundary. Meaning attribution happens via
 * the truth signal upstream; this function just lists candidate
 * surfaces.
 */
export function extractKeywordCandidates(text: string): string[] {
  const matches = text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [];
  const out: string[] = [];
  const seen = new Set<string>();
  for (const tok of matches) {
    if (tok.length < 3) continue;
    if (STOP_WORDS.has(tok)) continue;
    if (seen.has(tok)) continue;
    seen.add(tok);
    out.push(tok);
  }
  return out;
}

// ----- Scoring ----------------------------------------------------------

/**
 * Per-keyword tally accumulated while walking the labeled samples.
 * Kept as a plain shape for testability.
 */
export interface KeywordTally {
  keyword: string;
  /** Sum of weights where intent='yes' AND keyword in text. */
  positiveWeight: number;
  /** Sum of weights where intent='no' AND keyword in text. */
  negativeWeight: number;
  /** How many samples contributed to either side. */
  sampleCount: number;
}

/**
 * Score keyword candidates across labeled samples. Pure function:
 * fixed input → fixed output. The scorer attributes each labeled
 * sample's weighted intent to every keyword it contains.
 *
 * Returns a map keyed by lowercase keyword. The downstream proposal
 * step ranks by `positiveWeight - negativeWeight`.
 */
export function scoreKeywordCandidates(
  samples: LabeledSample[],
  cfg: LearnerConfig,
): Map<string, KeywordTally> {
  const tallies = new Map<string, KeywordTally>();
  for (const s of samples) {
    const w = weightForTier(s.senderTier, cfg);
    if (w === 0) continue;
    const candidates = extractKeywordCandidates(s.text);
    for (const k of candidates) {
      let t = tallies.get(k);
      if (!t) {
        t = {
          keyword: k,
          positiveWeight: 0,
          negativeWeight: 0,
          sampleCount: 0,
        };
        tallies.set(k, t);
      }
      if (s.intent === 'yes') {
        t.positiveWeight += w;
      } else {
        t.negativeWeight += w;
      }
      t.sampleCount += 1;
    }
  }
  return tallies;
}

/**
 * Result shape returned by `proposeFromSamples`. Distinguishes the
 * three terminal outcomes the caller cares about so the persistence
 * layer can branch cleanly.
 */
export type ProposalResult =
  | { kind: 'cold-start'; samplesSeen: number; minSamples: number }
  | { kind: 'no-proposals'; samplesSeen: number }
  | {
      kind: 'proposed';
      samplesSeen: number;
      proposals: TriggerPattern[];
    };

/**
 * Build proposed `TriggerPattern[]` from labeled samples. Pure
 * function; takes existing patterns into account so the caller can
 * deduplicate / supersede.
 *
 * Cold-start branch: if `samples.length < cfg.minSamples`, returns
 * `{kind: 'cold-start', ...}` and the caller skips proposal-write
 * entirely. The trigger gate then keeps using the universal /
 * owner-set patterns unchanged.
 *
 * The sample-count check is on TRUTH-LABELED samples (i.e. the
 * input array), not raw inbound messages — a low-traffic group
 * with 200 raw messages but 5 reactions still falls below the
 * floor. This is the correct shape: the scorer can't learn
 * anything from unlabeled messages.
 *
 * Proposals start `enabled: false` (pure proposal — owner promotes
 * via the existing admin path) per the `enabled` field contract in
 * `trigger-learner-schema.md`. Each proposal carries
 * `pattern_version: 1`, `proposed_at: now`, and `source: 'learned'`.
 */
export function proposeFromSamples(
  samples: LabeledSample[],
  cfg: LearnerConfig,
  now: () => string = () => new Date().toISOString(),
): ProposalResult {
  if (samples.length < cfg.minSamples) {
    return {
      kind: 'cold-start',
      samplesSeen: samples.length,
      minSamples: cfg.minSamples,
    };
  }
  const tallies = scoreKeywordCandidates(samples, cfg);
  const proposals: TriggerPattern[] = [];
  // Sort tallies for determinism: highest score first, ties broken
  // alphabetically. Tests assert on exact proposal order; without
  // a sort the Map iteration order would be insertion-driven and
  // tied scores would flap.
  const entries = Array.from(tallies.values()).sort((a, b) => {
    const scoreA = a.positiveWeight - a.negativeWeight;
    const scoreB = b.positiveWeight - b.negativeWeight;
    if (scoreA !== scoreB) return scoreB - scoreA;
    return a.keyword.localeCompare(b.keyword);
  });
  const proposedAt = now();
  for (const t of entries) {
    const score = t.positiveWeight - t.negativeWeight;
    if (score < cfg.minProposalScore) continue;
    // precision = positives / (positives + negatives) restricted to
    // the truth-labeled support that contributed to this keyword.
    // Defensive divisor: positiveWeight is already > 0 because score
    // > minProposalScore > 0 implies positiveWeight > 0, so the
    // denominator can't be zero.
    const denom = t.positiveWeight + t.negativeWeight;
    const precision = denom > 0 ? t.positiveWeight / denom : 0;
    proposals.push({
      pattern: t.keyword,
      kind: 'keyword' satisfies TriggerPatternKind,
      source: 'learned',
      precision,
      sample_count: t.sampleCount,
      last_matched_at: null,
      last_updated_at: proposedAt,
      pattern_version: 1,
      proposed_at: proposedAt,
      enabled: false,
      disabled: false,
    });
  }
  if (proposals.length === 0) {
    return { kind: 'no-proposals', samplesSeen: samples.length };
  }
  return {
    kind: 'proposed',
    samplesSeen: samples.length,
    proposals,
  };
}

// ----- Auto-rollback ----------------------------------------------------

/**
 * Per-pattern rolling decision history. The caller (persistence
 * layer) is responsible for assembling these by walking the host log
 * via `host-log-parser.ts`'s `findGateDecisions`.
 *
 * `decisions` is a list of `{matched, wasRight}` pairs over the K
 * most-recent gate decisions where THIS pattern was the one that
 * matched. `matched: false` entries are not stored — they're irrelevant
 * for FP-rate computation.
 */
export interface PatternDecisionHistory {
  pattern: string;
  kind: TriggerPatternKind;
  /** Most-recent K decisions where this pattern matched. */
  decisions: Array<{ wasRight: boolean }>;
}

/**
 * Compute the FP rate for one pattern over its rolling window.
 * Returns `null` when there's not enough history to evaluate (less
 * than half the window of decisions); the caller treats `null` as
 * "no opinion" and leaves the pattern alone.
 *
 * Pure function — no env, no DB.
 */
export function computeFalsePositiveRate(
  history: PatternDecisionHistory,
  cfg: LearnerConfig,
): number | null {
  const window = history.decisions.slice(-cfg.rollbackWindow);
  // Require at least half the window's worth of decisions before
  // demoting. Fewer samples → too noisy; the learner waits.
  const minDecisionsToEvaluate = Math.max(
    1,
    Math.floor(cfg.rollbackWindow / 2),
  );
  if (window.length < minDecisionsToEvaluate) return null;
  const wrong = window.filter((d) => !d.wasRight).length;
  return wrong / window.length;
}

/**
 * Apply auto-rollback to learned patterns: any learned pattern whose
 * FP rate over the rolling window exceeds `cfg.rollbackFpThreshold`
 * gets `disabled: true`. Owner-set / universal patterns are NEVER
 * touched.
 *
 * Returns a NEW `TriggerPattern[]` (never mutates input). Callers
 * persist via `setTriggerPatterns`.
 *
 * Pure function — fixed input → fixed output.
 */
export function applyAutoRollback(
  patterns: TriggerPattern[],
  histories: Map<string, PatternDecisionHistory>,
  cfg: LearnerConfig,
  now: () => string = () => new Date().toISOString(),
): { patterns: TriggerPattern[]; demoted: string[] } {
  const demoted: string[] = [];
  const updatedAt = now();
  const out = patterns.map((p) => {
    if (p.source !== 'learned') return p;
    if (p.disabled === true) return p; // already demoted
    const histKey = `${p.kind}:${p.pattern}`;
    const history = histories.get(histKey);
    if (!history) return p;
    const fpRate = computeFalsePositiveRate(history, cfg);
    if (fpRate === null) return p;
    if (fpRate <= cfg.rollbackFpThreshold) return p;
    demoted.push(p.pattern);
    return {
      ...p,
      disabled: true,
      last_updated_at: updatedAt,
    };
  });
  return { patterns: out, demoted };
}

// ----- Versioning / merging --------------------------------------------

/**
 * Merge proposals into the existing pattern set with versioning.
 *
 * Rules:
 *  - A proposal whose `{kind, pattern}` matches an existing learned
 *    row supersedes it: increment `pattern_version`, snapshot the
 *    PRIOR record into `prior_versions[]` (one level deep — older
 *    history is dropped to avoid unbounded column growth).
 *  - A proposal whose `{kind, pattern}` matches an OWNER-set or
 *    UNIVERSAL row is dropped. Owner-set takes precedence; the
 *    learner does not overwrite operator intent.
 *  - A proposal with no match is appended.
 *  - Existing learned rows that are NOT in the new proposal set are
 *    preserved verbatim (the loop didn't see fresh evidence; that's
 *    silence, not a vote against).
 *
 * Pure function — fixed input → fixed output.
 */
export function mergeProposals(
  existing: TriggerPattern[],
  proposals: TriggerPattern[],
): TriggerPattern[] {
  // Index existing rows for O(1) lookup. The key is the matcher's
  // identity tuple `{kind, pattern}` — two patterns that differ only
  // in observability fields are still the same logical row.
  const existingByKey = new Map<string, TriggerPattern>();
  for (const p of existing) {
    existingByKey.set(`${p.kind}:${p.pattern}`, p);
  }

  const out: TriggerPattern[] = [];
  const consumed = new Set<string>();

  for (const proposal of proposals) {
    const key = `${proposal.kind}:${proposal.pattern}`;
    const prior = existingByKey.get(key);
    if (!prior) {
      out.push(proposal);
      continue;
    }
    consumed.add(key);
    if (prior.source === 'owner-set' || prior.source === 'universal') {
      // Owner intent wins. Drop the proposal silently — the owner
      // already authored an authoritative pattern with this body.
      out.push(prior);
      continue;
    }
    // Same body, prior was learned — supersede. Increment lineage
    // version, snapshot prior under prior_versions (one deep), and
    // preserve the prior `enabled` flag so a previously-promoted
    // pattern stays promoted across refreshes (the learner never
    // re-flips `enabled` once the owner has set it).
    const nextVersion = (prior.pattern_version ?? 1) + 1;
    out.push({
      ...proposal,
      pattern_version: nextVersion,
      enabled: prior.enabled,
      prior_versions: [stripPriorVersions(prior)],
    });
  }

  // Carry forward any existing rows that weren't superseded by a
  // proposal in this pass.
  for (const p of existing) {
    const key = `${p.kind}:${p.pattern}`;
    if (consumed.has(key)) continue;
    out.push(p);
  }
  return out;
}

/**
 * Snapshot a record for embedding under `prior_versions`. We drop
 * the inner `prior_versions` to keep snapshots one-deep — without
 * this trim, every supersede would compound history exponentially.
 */
function stripPriorVersions(p: TriggerPattern): TriggerPattern {
  if (!p.prior_versions) return p;
  return { ...p, prior_versions: undefined };
}

// ----- Persistence wrapper ---------------------------------------------

/**
 * Hook contract the scheduler dependency layer satisfies. Lets tests
 * inject a fake DB and a fake "fetch labeled samples" without pulling
 * in `src/db.ts` directly.
 */
export interface LearnerPersistence {
  /** All registered groups (jid → group). */
  listGroups: () => Array<{
    jid: string;
    folder: string;
    config: TriggerPatternConfig | null | undefined;
  }>;
  /** Fetch the labeled samples for one group since the last run. */
  fetchSamples: (groupJid: string) => LabeledSample[];
  /** Fetch per-pattern decision history for auto-rollback. */
  fetchDecisionHistories: (
    groupJid: string,
  ) => Map<string, PatternDecisionHistory>;
  /** Persist updated config back. */
  saveConfig: (groupJid: string, config: TriggerPatternConfig) => void;
  /** Logger sink. */
  logger: {
    info: (data: Record<string, unknown>, msg: string) => void;
    warn: (data: Record<string, unknown>, msg: string) => void;
    error: (data: Record<string, unknown>, msg: string) => void;
  };
}

/**
 * One pass of the learner, restricted to a single group. Wraps
 * `proposeFromSamples` + `applyAutoRollback` + `mergeProposals` and
 * persists. Returns a structured summary so the caller (scheduler
 * loop) can log a single line per group.
 *
 * Errors are NARROW per `rules/error-handling.md`: only persistence
 * (DB) failures are caught and re-emitted as actionable warn logs.
 * Programming bugs (TypeError, etc.) propagate to the scheduler's
 * terminal safety net.
 */
export interface GroupRunSummary {
  groupJid: string;
  groupFolder: string;
  outcome:
    | 'cold-start'
    | 'no-proposals'
    | 'proposed'
    | 'rollback-only'
    | 'no-config';
  samplesSeen: number;
  newProposals: number;
  demoted: string[];
}

export function runLearnerForGroup(
  groupJid: string,
  groupFolder: string,
  persistence: LearnerPersistence,
  cfg: LearnerConfig,
  now: () => string = () => new Date().toISOString(),
): GroupRunSummary {
  const groupRecord = persistence.listGroups().find((g) => g.jid === groupJid);
  const config = groupRecord?.config ?? null;
  if (!config) {
    persistence.logger.warn(
      {
        groupJid,
        groupFolder,
        reason: 'no trigger pattern config persisted for group',
      },
      'trigger-learner: skipping group with no config',
    );
    return {
      groupJid,
      groupFolder,
      outcome: 'no-config',
      samplesSeen: 0,
      newProposals: 0,
      demoted: [],
    };
  }

  const samples = persistence.fetchSamples(groupJid);
  const histories = persistence.fetchDecisionHistories(groupJid);

  // Auto-rollback ALWAYS runs (even below cold-start floor) — it
  // only acts on EXISTING learned patterns, and a learned pattern
  // that's already in the DB has already passed the floor at some
  // point. Skipping rollback below the floor would let a bad
  // pattern keep firing forever after the group's truth signal
  // dries up.
  const rolled = applyAutoRollback(config.patterns, histories, cfg, now);

  const result = proposeFromSamples(samples, cfg, now);
  if (result.kind === 'cold-start') {
    if (rolled.demoted.length > 0) {
      // Persist the rollback even on cold-start.
      const updated: TriggerPatternConfig = {
        version: 1,
        patterns: rolled.patterns,
      };
      try {
        persistence.saveConfig(groupJid, updated);
      } catch (err) {
        if (!(err instanceof Error)) throw err;
        persistence.logger.error(
          {
            groupJid,
            groupFolder,
            err: err.message,
            demoted: rolled.demoted,
            samplesSeen: result.samplesSeen,
            minSamples: result.minSamples,
          },
          'trigger-learner: saveConfig failed during cold-start rollback — proposals not persisted; run will retry on next tick',
        );
        return {
          groupJid,
          groupFolder,
          outcome: 'cold-start',
          samplesSeen: result.samplesSeen,
          newProposals: 0,
          demoted: [],
        };
      }
    }
    persistence.logger.info(
      {
        groupJid,
        groupFolder,
        samplesSeen: result.samplesSeen,
        minSamples: result.minSamples,
        demoted: rolled.demoted,
      },
      'trigger-learner: cold-start floor not reached — no proposals',
    );
    return {
      groupJid,
      groupFolder,
      outcome: 'cold-start',
      samplesSeen: result.samplesSeen,
      newProposals: 0,
      demoted: rolled.demoted,
    };
  }

  if (result.kind === 'no-proposals') {
    if (rolled.demoted.length === 0) {
      persistence.logger.info(
        { groupJid, groupFolder, samplesSeen: result.samplesSeen },
        'trigger-learner: no proposals scored above threshold and no rollback needed',
      );
      return {
        groupJid,
        groupFolder,
        outcome: 'no-proposals',
        samplesSeen: result.samplesSeen,
        newProposals: 0,
        demoted: [],
      };
    }
    // Rollback-only path — persist demotions even with no fresh
    // proposals.
    const updated: TriggerPatternConfig = {
      version: 1,
      patterns: rolled.patterns,
    };
    try {
      persistence.saveConfig(groupJid, updated);
    } catch (err) {
      if (!(err instanceof Error)) throw err;
      persistence.logger.error(
        {
          groupJid,
          groupFolder,
          err: err.message,
          demoted: rolled.demoted,
          samplesSeen: result.samplesSeen,
        },
        'trigger-learner: saveConfig failed during rollback-only path — proposals not persisted; run will retry on next tick',
      );
      return {
        groupJid,
        groupFolder,
        outcome: 'rollback-only',
        samplesSeen: result.samplesSeen,
        newProposals: 0,
        demoted: [],
      };
    }
    persistence.logger.info(
      {
        groupJid,
        groupFolder,
        samplesSeen: result.samplesSeen,
        demoted: rolled.demoted,
      },
      'trigger-learner: applied rollback only (no fresh proposals)',
    );
    return {
      groupJid,
      groupFolder,
      outcome: 'rollback-only',
      samplesSeen: result.samplesSeen,
      newProposals: 0,
      demoted: rolled.demoted,
    };
  }

  // proposed: merge proposals into the rollback-applied set, persist.
  const merged = mergeProposals(rolled.patterns, result.proposals);
  const updated: TriggerPatternConfig = { version: 1, patterns: merged };
  try {
    persistence.saveConfig(groupJid, updated);
  } catch (err) {
    if (!(err instanceof Error)) throw err;
    persistence.logger.error(
      {
        groupJid,
        groupFolder,
        err: err.message,
        proposalCount: result.proposals.length,
        demoted: rolled.demoted,
      },
      'trigger-learner: saveConfig failed while writing proposals — run will retry on next tick',
    );
    return {
      groupJid,
      groupFolder,
      outcome: 'proposed',
      samplesSeen: result.samplesSeen,
      newProposals: 0,
      demoted: [],
    };
  }
  persistence.logger.info(
    {
      groupJid,
      groupFolder,
      samplesSeen: result.samplesSeen,
      newProposals: result.proposals.length,
      demoted: rolled.demoted,
    },
    'trigger-learner: proposals written',
  );
  return {
    groupJid,
    groupFolder,
    outcome: 'proposed',
    samplesSeen: result.samplesSeen,
    newProposals: result.proposals.length,
    demoted: rolled.demoted,
  };
}

// ----- Scheduler hook ---------------------------------------------------

/**
 * Resolve the learner cadence from env. Same parsing contract as
 * `resolveLearnerConfig` — invalid values throw a descriptive Error
 * at startup rather than silently falling back.
 */
export function resolveLearnerIntervalMs(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const raw = env.TRIGGER_LEARNER_INTERVAL_MS;
  if (raw === undefined || raw === '') return DEFAULT_LEARNER_INTERVAL_MS;
  if (!/^\d+$/.test(raw)) {
    throw new Error(
      `[trigger-learner] env TRIGGER_LEARNER_INTERVAL_MS="${raw}" is not a non-negative integer of ms; ` +
        `expected an integer ≥ 0 or unset for default ${DEFAULT_LEARNER_INTERVAL_MS} (24h)`,
    );
  }
  const parsed = parseInt(raw, 10);
  if (parsed === 0) {
    throw new Error(
      `[trigger-learner] env TRIGGER_LEARNER_INTERVAL_MS=0 disables the loop entirely; ` +
        `unset the variable to use the default ${DEFAULT_LEARNER_INTERVAL_MS}, or set a positive integer`,
    );
  }
  return parsed;
}

/**
 * Run one learner pass across every registered group. Used by the
 * scheduler hook AND directly by tests / CLI tooling.
 *
 * Per-group failures don't abort the pass — each group is in its own
 * try/catch (narrow to Error per the error-handling rule).
 */
export function runLearnerOnce(
  persistence: LearnerPersistence,
  cfg: LearnerConfig,
  now: () => string = () => new Date().toISOString(),
): GroupRunSummary[] {
  const summaries: GroupRunSummary[] = [];
  for (const g of persistence.listGroups()) {
    try {
      summaries.push(
        runLearnerForGroup(g.jid, g.folder, persistence, cfg, now),
      );
    } catch (err) {
      if (!(err instanceof Error)) throw err;
      persistence.logger.error(
        { groupJid: g.jid, groupFolder: g.folder, err: err.message },
        'trigger-learner: group run threw — skipping group, remaining groups continue',
      );
      summaries.push({
        groupJid: g.jid,
        groupFolder: g.folder,
        outcome: 'no-config',
        samplesSeen: 0,
        newProposals: 0,
        demoted: [],
      });
    }
  }
  return summaries;
}
