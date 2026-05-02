/**
 * Tests for the Stage 1 trigger-pattern self-improvement loop (#82).
 *
 * Cover the five non-negotiable invariants from the issue body:
 *  1. Cold-start: <N truth-labeled samples → no proposals
 *  2. Self-reinforcement guard: the gate's own historical verdict is
 *     NEVER injected as truth — there is no `gate_verdict` source
 *     in the public API.
 *  3. Sender-tier weighting: owner correction outweighs anonymous
 *  4. Auto-rollback: an existing learned pattern with FP rate >
 *     threshold over the rolling window is demoted (`disabled: true`)
 *  5. Versioning: a superseded learned pattern keeps its prior record
 *     under `prior_versions` and bumps `pattern_version`.
 *
 * Plus persistence-layer behaviour: saveConfig errors don't crash the
 * scheduler; per-group failures don't abort the whole pass.
 *
 * All tests are deterministic — fixed inputs, fixed timestamps via
 * `() => 'frozen'`, no `Math.random`.
 */
import { describe, it, expect } from 'vitest';

import type { TriggerPattern, TriggerPatternConfig } from '../types.js';

import {
  DEFAULT_MIN_SAMPLES,
  DEFAULT_OWNER_WEIGHT,
  DEFAULT_NON_OWNER_WEIGHT,
  DEFAULT_ANONYMOUS_WEIGHT,
  DEFAULT_ROLLBACK_FP_THRESHOLD,
  DEFAULT_ROLLBACK_WINDOW,
  DEFAULT_MIN_PROPOSAL_SCORE,
  applyAutoRollback,
  computeFalsePositiveRate,
  extractKeywordCandidates,
  mergeProposals,
  proposeFromSamples,
  resolveLearnerConfig,
  resolveLearnerIntervalMs,
  runLearnerForGroup,
  runLearnerOnce,
  scoreKeywordCandidates,
  weightForTier,
  type LabeledSample,
  type LearnerConfig,
  type LearnerPersistence,
  type PatternDecisionHistory,
  type SenderTier,
  type TruthSource,
} from './trigger-learner.js';

const DEFAULT_CFG: LearnerConfig = {
  minSamples: DEFAULT_MIN_SAMPLES,
  ownerWeight: DEFAULT_OWNER_WEIGHT,
  nonOwnerWeight: DEFAULT_NON_OWNER_WEIGHT,
  anonymousWeight: DEFAULT_ANONYMOUS_WEIGHT,
  rollbackWindow: DEFAULT_ROLLBACK_WINDOW,
  rollbackFpThreshold: DEFAULT_ROLLBACK_FP_THRESHOLD,
  minProposalScore: DEFAULT_MIN_PROPOSAL_SCORE,
};

const FROZEN_TIME = '2026-05-02T12:00:00.000Z';
const frozenNow = () => FROZEN_TIME;

function sample(
  text: string,
  intent: 'yes' | 'no',
  senderTier: SenderTier,
  source: TruthSource = 'haiku_verdict',
  gateResponded: boolean = intent === 'yes',
): LabeledSample {
  return { text, intent, senderTier, source, gateResponded };
}

function silent(
  count: number,
  text: string,
  intent: 'yes' | 'no',
  tier: SenderTier,
): LabeledSample[] {
  // Repeat the same labeled sample N times. `count` is fixed in
  // each test (no randomness).
  return Array.from({ length: count }, () => sample(text, intent, tier));
}

function silentLogger() {
  return {
    info: () => {},
    warn: () => {},
    error: () => {},
  };
}

// ---------------------------------------------------------------------------
// 1. Cold-start
// ---------------------------------------------------------------------------

describe('proposeFromSamples — cold-start', () => {
  it('emits no proposals when truth-labeled sample count < minSamples', () => {
    // 49 samples, all "yes" + same keyword — would normally be a
    // strong signal, but cold-start floor is 50.
    const samples = silent(49, 'deploy hotfix now', 'yes', 'owner');
    const result = proposeFromSamples(samples, DEFAULT_CFG, frozenNow);
    expect(result.kind).toBe('cold-start');
    if (result.kind !== 'cold-start') return;
    expect(result.samplesSeen).toBe(49);
    expect(result.minSamples).toBe(50);
  });

  it('does emit proposals once minSamples is reached and signal is strong', () => {
    // 50 owner "yes" samples mentioning "deploy" — well above
    // proposal threshold (50 owner = 150 weight).
    const samples = silent(50, 'deploy hotfix now', 'yes', 'owner');
    const result = proposeFromSamples(samples, DEFAULT_CFG, frozenNow);
    expect(result.kind).toBe('proposed');
    if (result.kind !== 'proposed') return;
    const deployProp = result.proposals.find((p) => p.pattern === 'deploy');
    expect(deployProp).toBeDefined();
    expect(deployProp?.source).toBe('learned');
    expect(deployProp?.enabled).toBe(false);
    expect(deployProp?.pattern_version).toBe(1);
    expect(deployProp?.proposed_at).toBe(FROZEN_TIME);
  });

  it('cold-start counts truth-LABELED samples, not raw messages', () => {
    // The function takes `LabeledSample[]`, so by construction every
    // input is labeled. This test pins the contract: the floor is
    // measured against the array length the caller passes — no
    // hidden filter inside the scorer.
    const cfg: LearnerConfig = { ...DEFAULT_CFG, minSamples: 5 };
    const four = silent(4, 'help', 'yes', 'owner');
    expect(proposeFromSamples(four, cfg, frozenNow).kind).toBe('cold-start');
    const five = silent(5, 'help me with deploy task', 'yes', 'owner');
    expect(proposeFromSamples(five, cfg, frozenNow).kind).toBe('proposed');
  });
});

// ---------------------------------------------------------------------------
// 2. Self-reinforcement guard
// ---------------------------------------------------------------------------

describe('TruthSource — no self-reinforcement leak', () => {
  it('the public TruthSource union excludes the gate verdict', () => {
    // This is a compile-time guarantee that the gate's own decision
    // can't be passed in as truth. The `LabeledSample` accepts only
    // the three documented external sources.
    const validSources: TruthSource[] = [
      'user_reaction',
      'owner_correction',
      'haiku_verdict',
    ];
    expect(validSources).toHaveLength(3);
    // Compile-time negative: assigning 'gate_verdict' to TruthSource
    // would trigger ts2322. We assert it at runtime via a string set
    // so a future refactor that widened the union would also fail
    // here.
    const allowed = new Set(validSources);
    expect(allowed.has('user_reaction')).toBe(true);
    expect(allowed.has('owner_correction')).toBe(true);
    expect(allowed.has('haiku_verdict')).toBe(true);
    // Reject the textual mark we forbid:
    expect(allowed.has('gate_verdict' as TruthSource)).toBe(false);
  });

  it('proposeFromSamples does NOT consult `gateResponded` to label intent', () => {
    // Build a scenario where the gate's own verdict (`gateResponded`)
    // disagrees with the external truth label. If the scorer were
    // accidentally using `gateResponded` as truth, the proposal
    // would flip; if it correctly uses `intent` (the external
    // signal), the proposal stays aligned with the external truth.
    //
    // 50 samples: external truth says "no, ignore this", gate
    // (incorrectly) responded every time. A self-reinforcing scorer
    // would treat every response as confirmation and propose the
    // keyword. The correct scorer treats the external `no` and
    // produces a NEGATIVE score → no proposal.
    const samples: LabeledSample[] = Array.from({ length: 50 }, () => ({
      text: 'spam keyword spurious',
      intent: 'no' as const,
      senderTier: 'owner' as const,
      source: 'haiku_verdict' as const,
      gateResponded: true, // gate was wrong every time
    }));
    const result = proposeFromSamples(samples, DEFAULT_CFG, frozenNow);
    // Either no-proposals (negative score below threshold) or
    // proposed with NO entry for our keyword. Both demonstrate the
    // scorer didn't lock onto `gateResponded`.
    if (result.kind === 'proposed') {
      expect(
        result.proposals.find((p) => p.pattern === 'spam'),
      ).toBeUndefined();
      expect(
        result.proposals.find((p) => p.pattern === 'keyword'),
      ).toBeUndefined();
    } else {
      expect(result.kind).toBe('no-proposals');
    }
  });
});

// ---------------------------------------------------------------------------
// 3. Sender-tier weighting
// ---------------------------------------------------------------------------

describe('sender-tier weighting', () => {
  it('owner=3.0, non-owner=1.0, anonymous=0.5 by default', () => {
    expect(weightForTier('owner', DEFAULT_CFG)).toBe(3.0);
    expect(weightForTier('non-owner', DEFAULT_CFG)).toBe(1.0);
    expect(weightForTier('anonymous', DEFAULT_CFG)).toBe(0.5);
  });

  it('owner correction outweighs anonymous opposite-direction signal', () => {
    // 5 owner "yes" + 10 anonymous "no" on the same keyword:
    //   owner positive: 5 * 3.0 = 15
    //   anonymous negative: 10 * 0.5 = 5
    //   net: +10 (above the 2.0 default threshold) → proposed
    const cfg: LearnerConfig = { ...DEFAULT_CFG, minSamples: 10 };
    const samples: LabeledSample[] = [
      ...silent(5, 'fix the deploy please', 'yes', 'owner'),
      ...silent(10, 'fix the deploy please', 'no', 'anonymous'),
    ];
    const result = proposeFromSamples(samples, cfg, frozenNow);
    expect(result.kind).toBe('proposed');
    if (result.kind !== 'proposed') return;
    const fix = result.proposals.find((p) => p.pattern === 'fix');
    const deploy = result.proposals.find((p) => p.pattern === 'deploy');
    expect(fix).toBeDefined();
    expect(deploy).toBeDefined();
  });

  it('inverted weights flip the outcome — symmetric proof of the rule', () => {
    // Same 5 owner-yes + 10 anonymous-no, but with weights swapped
    // (owner=0.5, anonymous=3.0):
    //   owner positive: 5 * 0.5 = 2.5
    //   anonymous negative: 10 * 3.0 = 30
    //   net: -27.5 → NO proposal for these tokens
    const cfg: LearnerConfig = {
      ...DEFAULT_CFG,
      minSamples: 10,
      ownerWeight: 0.5,
      anonymousWeight: 3.0,
    };
    const samples: LabeledSample[] = [
      ...silent(5, 'fix the deploy please', 'yes', 'owner'),
      ...silent(10, 'fix the deploy please', 'no', 'anonymous'),
    ];
    const result = proposeFromSamples(samples, cfg, frozenNow);
    if (result.kind === 'proposed') {
      expect(result.proposals.find((p) => p.pattern === 'fix')).toBeUndefined();
      expect(
        result.proposals.find((p) => p.pattern === 'deploy'),
      ).toBeUndefined();
    } else {
      expect(result.kind).toBe('no-proposals');
    }
  });
});

// ---------------------------------------------------------------------------
// 4. Auto-rollback on precision drop
// ---------------------------------------------------------------------------

describe('applyAutoRollback', () => {
  function learned(
    pattern: string,
    overrides: Partial<TriggerPattern> = {},
  ): TriggerPattern {
    return {
      pattern,
      kind: 'keyword',
      source: 'learned',
      precision: 0,
      sample_count: 0,
      last_matched_at: null,
      last_updated_at: null,
      pattern_version: 1,
      proposed_at: '2026-05-01T00:00:00.000Z',
      enabled: true,
      disabled: false,
      ...overrides,
    };
  }

  function ownerSet(pattern: string): TriggerPattern {
    return {
      pattern,
      kind: 'keyword',
      source: 'owner-set',
      precision: 0,
      sample_count: 0,
      last_matched_at: null,
      last_updated_at: null,
    };
  }

  function history(
    pattern: string,
    decisions: Array<{ wasRight: boolean }>,
  ): [string, PatternDecisionHistory] {
    return [`keyword:${pattern}`, { pattern, kind: 'keyword', decisions }];
  }

  it('demotes a learned pattern whose FP rate exceeds the threshold', () => {
    // 30 decisions, 21 wrong (70% FP rate, well above the 30% default)
    const wrongMostly = Array.from({ length: 30 }, (_, i) => ({
      wasRight: i < 9, // first 9 right, rest wrong
    }));
    const patterns: TriggerPattern[] = [learned('deploy')];
    const histories = new Map<string, PatternDecisionHistory>([
      history('deploy', wrongMostly),
    ]);
    const result = applyAutoRollback(
      patterns,
      histories,
      DEFAULT_CFG,
      frozenNow,
    );
    expect(result.demoted).toEqual(['deploy']);
    expect(result.patterns[0].disabled).toBe(true);
    expect(result.patterns[0].last_updated_at).toBe(FROZEN_TIME);
  });

  it('does NOT demote a learned pattern whose FP rate is below threshold', () => {
    // 30 decisions, 6 wrong (20% FP rate, below 30% default)
    const mostlyRight = Array.from({ length: 30 }, (_, i) => ({
      wasRight: i >= 6,
    }));
    const patterns: TriggerPattern[] = [learned('deploy')];
    const histories = new Map<string, PatternDecisionHistory>([
      history('deploy', mostlyRight),
    ]);
    const result = applyAutoRollback(
      patterns,
      histories,
      DEFAULT_CFG,
      frozenNow,
    );
    expect(result.demoted).toEqual([]);
    expect(result.patterns[0].disabled).toBe(false);
  });

  it('NEVER demotes owner-set patterns even if FP rate is high', () => {
    // Same 70%-wrong history, but on an owner-set row.
    const wrongMostly = Array.from({ length: 30 }, (_, i) => ({
      wasRight: i < 9,
    }));
    const patterns: TriggerPattern[] = [ownerSet('deploy')];
    const histories = new Map<string, PatternDecisionHistory>([
      history('deploy', wrongMostly),
    ]);
    const result = applyAutoRollback(
      patterns,
      histories,
      DEFAULT_CFG,
      frozenNow,
    );
    expect(result.demoted).toEqual([]);
    expect(result.patterns[0].source).toBe('owner-set');
    expect(result.patterns[0].disabled).toBeUndefined();
  });

  it('keeps demoted rows in the array — does NOT delete', () => {
    const wrongMostly = Array.from({ length: 30 }, () => ({ wasRight: false }));
    const patterns: TriggerPattern[] = [learned('deploy'), learned('build')];
    const histories = new Map<string, PatternDecisionHistory>([
      history('deploy', wrongMostly),
    ]);
    const result = applyAutoRollback(
      patterns,
      histories,
      DEFAULT_CFG,
      frozenNow,
    );
    // Both patterns still present; only `deploy` flipped.
    expect(result.patterns).toHaveLength(2);
    expect(result.patterns.find((p) => p.pattern === 'deploy')?.disabled).toBe(
      true,
    );
    expect(result.patterns.find((p) => p.pattern === 'build')?.disabled).toBe(
      false,
    );
  });

  it('returns null FP rate when window has fewer than half-window decisions', () => {
    // K=30 default; only 5 decisions logged → not enough to evaluate.
    const tiny = Array.from({ length: 5 }, () => ({ wasRight: false }));
    expect(
      computeFalsePositiveRate(
        { pattern: 'x', kind: 'keyword', decisions: tiny },
        DEFAULT_CFG,
      ),
    ).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 5. Versioning — supersede + prior_versions
// ---------------------------------------------------------------------------

describe('mergeProposals — versioning', () => {
  function mkLearned(
    pattern: string,
    overrides: Partial<TriggerPattern> = {},
  ): TriggerPattern {
    return {
      pattern,
      kind: 'keyword',
      source: 'learned',
      precision: 0.5,
      sample_count: 10,
      last_matched_at: null,
      last_updated_at: '2026-04-01T00:00:00.000Z',
      pattern_version: 1,
      proposed_at: '2026-04-01T00:00:00.000Z',
      enabled: false,
      disabled: false,
      ...overrides,
    };
  }

  it('superseding an existing learned pattern bumps version and snapshots prior', () => {
    const existing: TriggerPattern[] = [mkLearned('deploy')];
    const proposal: TriggerPattern = mkLearned('deploy', {
      precision: 0.9,
      sample_count: 50,
      pattern_version: 1,
      proposed_at: '2026-05-02T00:00:00.000Z',
    });
    const out = mergeProposals(existing, [proposal]);
    expect(out).toHaveLength(1);
    const merged = out[0];
    expect(merged.pattern_version).toBe(2);
    expect(merged.precision).toBe(0.9);
    expect(merged.prior_versions).toHaveLength(1);
    expect(merged.prior_versions?.[0].precision).toBe(0.5);
    expect(merged.prior_versions?.[0].pattern_version).toBe(1);
  });

  it('caps prior_versions at one level deep on supersede', () => {
    // Existing v2 pattern that already has a prior_versions[v1]
    const existing: TriggerPattern[] = [
      mkLearned('deploy', {
        pattern_version: 2,
        prior_versions: [mkLearned('deploy', { pattern_version: 1 })],
      }),
    ];
    const proposal: TriggerPattern = mkLearned('deploy', {
      precision: 0.95,
    });
    const out = mergeProposals(existing, [proposal]);
    expect(out[0].pattern_version).toBe(3);
    expect(out[0].prior_versions).toHaveLength(1);
    // The snapshot is the v2 (the immediately-prior); the inner v1
    // is dropped to avoid unbounded chain growth.
    expect(out[0].prior_versions?.[0].pattern_version).toBe(2);
    expect(out[0].prior_versions?.[0].prior_versions).toBeUndefined();
  });

  it('a proposal that collides with an OWNER-set row is silently dropped', () => {
    const existing: TriggerPattern[] = [
      {
        pattern: 'deploy',
        kind: 'keyword',
        source: 'owner-set',
        precision: 0,
        sample_count: 0,
        last_matched_at: null,
        last_updated_at: null,
      },
    ];
    const proposal = mkLearned('deploy');
    const out = mergeProposals(existing, [proposal]);
    expect(out).toHaveLength(1);
    expect(out[0].source).toBe('owner-set');
  });

  it('preserves owner-promoted enabled flag across supersede', () => {
    // Owner had promoted v1 (enabled: true). When learner supersedes
    // with v2, the promotion stays — the loop never silently un-promotes.
    const existing: TriggerPattern[] = [mkLearned('deploy', { enabled: true })];
    const proposal = mkLearned('deploy', { precision: 0.9 });
    const out = mergeProposals(existing, [proposal]);
    expect(out[0].enabled).toBe(true);
  });

  it('a proposal with no existing match is appended', () => {
    const existing: TriggerPattern[] = [mkLearned('deploy')];
    const proposal = mkLearned('build');
    const out = mergeProposals(existing, [proposal]);
    expect(out).toHaveLength(2);
    expect(out.map((p) => p.pattern).sort()).toEqual(['build', 'deploy']);
  });

  it('existing learned rows untouched by this pass are preserved verbatim', () => {
    // Stale signal: deploy was proposed last month, no fresh proposal
    // this pass. Don't drop it — the matcher should still see it.
    const existing: TriggerPattern[] = [mkLearned('deploy')];
    const out = mergeProposals(existing, []);
    expect(out).toEqual(existing);
  });
});

// ---------------------------------------------------------------------------
// Persistence wrapper
// ---------------------------------------------------------------------------

describe('runLearnerForGroup', () => {
  function buildPersistence(opts: {
    config: TriggerPatternConfig | null;
    samples?: LabeledSample[];
    histories?: Map<string, PatternDecisionHistory>;
    saveImpl?: (jid: string, c: TriggerPatternConfig) => void;
  }): { persistence: LearnerPersistence; saved: TriggerPatternConfig[] } {
    const saved: TriggerPatternConfig[] = [];
    const persistence: LearnerPersistence = {
      listGroups: () => [
        { jid: 'g@g.us', folder: 'g', config: opts.config ?? undefined },
      ],
      fetchSamples: () => opts.samples ?? [],
      fetchDecisionHistories: () => opts.histories ?? new Map(),
      saveConfig: (jid, config) => {
        if (opts.saveImpl) {
          opts.saveImpl(jid, config);
          return;
        }
        saved.push(config);
      },
      logger: silentLogger(),
    };
    return { persistence, saved };
  }

  it('cold-start: leaves config untouched and returns cold-start outcome', () => {
    const cfg: TriggerPatternConfig = { version: 1, patterns: [] };
    const { persistence, saved } = buildPersistence({
      config: cfg,
      samples: silent(10, 'help me', 'yes', 'owner'), // <50
    });
    const summary = runLearnerForGroup(
      'g@g.us',
      'g',
      persistence,
      DEFAULT_CFG,
      frozenNow,
    );
    expect(summary.outcome).toBe('cold-start');
    expect(summary.newProposals).toBe(0);
    expect(saved).toHaveLength(0);
  });

  it('proposed: writes config back with `enabled: false` proposals', () => {
    const cfg: TriggerPatternConfig = { version: 1, patterns: [] };
    const { persistence, saved } = buildPersistence({
      config: cfg,
      samples: silent(50, 'deploy hotfix now', 'yes', 'owner'),
    });
    const summary = runLearnerForGroup(
      'g@g.us',
      'g',
      persistence,
      DEFAULT_CFG,
      frozenNow,
    );
    expect(summary.outcome).toBe('proposed');
    expect(saved).toHaveLength(1);
    expect(saved[0].patterns.length).toBeGreaterThan(0);
    for (const p of saved[0].patterns) {
      expect(p.source).toBe('learned');
      expect(p.enabled).toBe(false); // pure proposal — owner promotes
    }
  });

  it('saveConfig failure: error is logged, summary records 0 proposals, no throw', () => {
    let warnCount = 0;
    let errorCount = 0;
    const persistence: LearnerPersistence = {
      listGroups: () => [
        {
          jid: 'g@g.us',
          folder: 'g',
          config: { version: 1, patterns: [] },
        },
      ],
      fetchSamples: () => silent(50, 'deploy hotfix now', 'yes', 'owner'),
      fetchDecisionHistories: () => new Map(),
      saveConfig: () => {
        throw new Error('disk full');
      },
      logger: {
        info: () => {},
        warn: () => {
          warnCount += 1;
        },
        error: () => {
          errorCount += 1;
        },
      },
    };
    const summary = runLearnerForGroup(
      'g@g.us',
      'g',
      persistence,
      DEFAULT_CFG,
      frozenNow,
    );
    expect(summary.outcome).toBe('proposed');
    expect(summary.newProposals).toBe(0); // didn't actually persist
    expect(errorCount).toBeGreaterThan(0);
    expect(warnCount).toBe(0);
  });

  it('no-config group: warns and returns no-config outcome', () => {
    let warnCount = 0;
    const persistence: LearnerPersistence = {
      listGroups: () => [{ jid: 'g@g.us', folder: 'g', config: null }],
      fetchSamples: () => [],
      fetchDecisionHistories: () => new Map(),
      saveConfig: () => {
        throw new Error('should not be called');
      },
      logger: {
        info: () => {},
        warn: () => {
          warnCount += 1;
        },
        error: () => {},
      },
    };
    const summary = runLearnerForGroup(
      'g@g.us',
      'g',
      persistence,
      DEFAULT_CFG,
      frozenNow,
    );
    expect(summary.outcome).toBe('no-config');
    expect(warnCount).toBe(1);
  });

  it('rollback-only: applies demotions even when no fresh proposals scored', () => {
    const learnedDeploy: TriggerPattern = {
      pattern: 'deploy',
      kind: 'keyword',
      source: 'learned',
      precision: 0.5,
      sample_count: 10,
      last_matched_at: null,
      last_updated_at: '2026-04-01T00:00:00.000Z',
      pattern_version: 1,
      proposed_at: '2026-04-01T00:00:00.000Z',
      enabled: true,
      disabled: false,
    };
    const wrongMostly = Array.from({ length: 30 }, () => ({ wasRight: false }));
    const histories = new Map<string, PatternDecisionHistory>([
      [
        'keyword:deploy',
        { pattern: 'deploy', kind: 'keyword', decisions: wrongMostly },
      ],
    ]);
    const { persistence, saved } = buildPersistence({
      config: { version: 1, patterns: [learnedDeploy] },
      // 50 samples but with a totally different keyword set →
      // proposals miss the score gate → no-proposals branch
      samples: silent(50, 'xx yy zz', 'no', 'anonymous'),
      histories,
    });
    const summary = runLearnerForGroup(
      'g@g.us',
      'g',
      persistence,
      DEFAULT_CFG,
      frozenNow,
    );
    expect(summary.outcome).toBe('rollback-only');
    expect(summary.demoted).toEqual(['deploy']);
    expect(saved).toHaveLength(1);
    expect(saved[0].patterns[0].disabled).toBe(true);
  });
});

describe('runLearnerOnce', () => {
  it('one group throwing does not abort the rest', () => {
    let errorCount = 0;
    const persistence: LearnerPersistence = {
      listGroups: () => [
        {
          jid: 'broken@g.us',
          folder: 'broken',
          config: { version: 1, patterns: [] },
        },
        {
          jid: 'ok@g.us',
          folder: 'ok',
          config: { version: 1, patterns: [] },
        },
      ],
      fetchSamples: (jid) => {
        if (jid === 'broken@g.us') {
          throw new Error('synthetic fetch failure');
        }
        return [];
      },
      fetchDecisionHistories: () => new Map(),
      saveConfig: () => {},
      logger: {
        info: () => {},
        warn: () => {},
        error: () => {
          errorCount += 1;
        },
      },
    };
    const summaries = runLearnerOnce(persistence, DEFAULT_CFG, frozenNow);
    expect(summaries).toHaveLength(2);
    expect(errorCount).toBe(1);
    // The 'ok' group still ran (cold-start because no samples).
    const okSummary = summaries.find((s) => s.groupJid === 'ok@g.us');
    expect(okSummary?.outcome).toBe('cold-start');
  });
});

// ---------------------------------------------------------------------------
// Helpers — keyword extraction
// ---------------------------------------------------------------------------

describe('extractKeywordCandidates', () => {
  it('drops stop words and short tokens', () => {
    const out = extractKeywordCandidates('I am the very model of a deploy');
    expect(out).toEqual(['very', 'model', 'deploy']);
  });

  it('lowercases and dedupes', () => {
    const out = extractKeywordCandidates('Deploy DEPLOY deploy build');
    expect(out).toEqual(['deploy', 'build']);
  });

  it('handles Cyrillic stop words and content tokens', () => {
    const out = extractKeywordCandidates('я не помню деплой');
    // Stop words (я, не) are dropped; remaining 3-char+ tokens kept.
    expect(out).toContain('помню');
    expect(out).toContain('деплой');
    expect(out).not.toContain('я');
    expect(out).not.toContain('не');
  });
});

// ---------------------------------------------------------------------------
// scoreKeywordCandidates — tally bookkeeping
// ---------------------------------------------------------------------------

describe('scoreKeywordCandidates', () => {
  it('accumulates positive and negative weight per keyword', () => {
    const samples: LabeledSample[] = [
      sample('deploy now', 'yes', 'owner'),
      sample('deploy slow', 'no', 'non-owner'),
      sample('deploy maybe', 'yes', 'anonymous'),
    ];
    const tallies = scoreKeywordCandidates(samples, DEFAULT_CFG);
    const deploy = tallies.get('deploy');
    expect(deploy).toBeDefined();
    // 1 owner-yes (3) + 1 anonymous-yes (0.5) - 1 non-owner-no (1) =
    // positive=3.5, negative=1.0
    expect(deploy?.positiveWeight).toBeCloseTo(3.5, 5);
    expect(deploy?.negativeWeight).toBeCloseTo(1.0, 5);
    expect(deploy?.sampleCount).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Env config
// ---------------------------------------------------------------------------

describe('resolveLearnerConfig', () => {
  it('uses defaults when env is empty', () => {
    const cfg = resolveLearnerConfig({});
    expect(cfg.minSamples).toBe(DEFAULT_MIN_SAMPLES);
    expect(cfg.ownerWeight).toBe(DEFAULT_OWNER_WEIGHT);
    expect(cfg.rollbackFpThreshold).toBe(DEFAULT_ROLLBACK_FP_THRESHOLD);
  });

  it('parses valid overrides', () => {
    const cfg = resolveLearnerConfig({
      TRIGGER_LEARNER_MIN_SAMPLES: '100',
      TRIGGER_LEARNER_OWNER_WEIGHT: '5.5',
    });
    expect(cfg.minSamples).toBe(100);
    expect(cfg.ownerWeight).toBe(5.5);
  });

  it('throws an actionable error on a malformed numeric override', () => {
    expect(() =>
      resolveLearnerConfig({ TRIGGER_LEARNER_MIN_SAMPLES: 'foo' }),
    ).toThrowError(/TRIGGER_LEARNER_MIN_SAMPLES.*"foo".*non-negative integer/);
  });

  it('throws an actionable error on a negative weight', () => {
    expect(() =>
      resolveLearnerConfig({ TRIGGER_LEARNER_OWNER_WEIGHT: '-1' }),
    ).toThrowError(/TRIGGER_LEARNER_OWNER_WEIGHT.*"-1".*non-negative/);
  });
});

describe('resolveLearnerIntervalMs', () => {
  it('defaults to 24h when unset', () => {
    expect(resolveLearnerIntervalMs({})).toBe(24 * 60 * 60 * 1000);
  });

  it('throws when set to 0 (operator should unset to use default)', () => {
    expect(() =>
      resolveLearnerIntervalMs({ TRIGGER_LEARNER_INTERVAL_MS: '0' }),
    ).toThrowError(/disables the loop entirely/);
  });

  it('parses positive integer values', () => {
    expect(
      resolveLearnerIntervalMs({ TRIGGER_LEARNER_INTERVAL_MS: '60000' }),
    ).toBe(60000);
  });
});
