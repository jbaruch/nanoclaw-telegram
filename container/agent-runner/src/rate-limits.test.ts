import { describe, it, expect } from 'vitest';
import { AclPrefix } from './capability-acl.js';
import {
  DEFAULT_CAP_MATRIX,
  applyOverrides,
  classifyProvenance,
  classifyTool,
  decideRate,
  emptyCounters,
  parseCounters,
  parseOverrides,
  pruneTimestamps,
  recordEvent,
} from './rate-limits.js';

const FIXED_NOW = 1_700_000_000; // synthetic unix-second timestamp

function p(prefixes: AclPrefix[]): Set<AclPrefix> {
  return new Set(prefixes);
}

describe('classifyProvenance', () => {
  it('empty + trusted → operator-trusted', () => {
    expect(classifyProvenance(p([]), true)).toBe('operator-trusted');
  });

  it('empty + untrusted → operator-untrusted', () => {
    expect(classifyProvenance(p([]), false)).toBe('operator-untrusted');
  });

  it('cross-group only → cross-group (regardless of trust tier)', () => {
    expect(classifyProvenance(p(['cross-group']), true)).toBe('cross-group');
    expect(classifyProvenance(p(['cross-group']), false)).toBe('cross-group');
  });

  it('web → untrusted-source', () => {
    expect(classifyProvenance(p(['web']), true)).toBe('untrusted-source');
  });

  it('gmail → untrusted-source', () => {
    expect(classifyProvenance(p(['gmail']), true)).toBe('untrusted-source');
  });

  it('multiple untrusted-source prefixes → untrusted-source', () => {
    expect(classifyProvenance(p(['web', 'gmail', 'tessl']), true)).toBe(
      'untrusted-source',
    );
  });

  it('cross-group AND web → mixed', () => {
    expect(classifyProvenance(p(['cross-group', 'web']), true)).toBe('mixed');
  });

  it('cross-group AND multiple untrusted-source → mixed', () => {
    expect(
      classifyProvenance(p(['cross-group', 'web', 'gmail']), false),
    ).toBe('mixed');
  });

  it('unknown prefix sentinel → untrusted-source (fail closed)', () => {
    expect(classifyProvenance(p(['__unknown__' as AclPrefix]), true)).toBe(
      'untrusted-source',
    );
  });

  it('cross-group + unknown sentinel → mixed', () => {
    expect(
      classifyProvenance(
        p(['cross-group', '__unknown__' as AclPrefix]),
        true,
      ),
    ).toBe('mixed');
  });
});

describe('classifyTool', () => {
  it('maps Task → agent_spawn', () => {
    expect(classifyTool('Task')).toBe('agent_spawn');
  });

  it('maps mcp__nanoclaw__schedule_task → schedule_task', () => {
    expect(classifyTool('mcp__nanoclaw__schedule_task')).toBe('schedule_task');
  });

  it('returns null for unrelated tools', () => {
    expect(classifyTool('Read')).toBeNull();
    expect(classifyTool('mcp__nanoclaw__send_message')).toBeNull();
    expect(classifyTool('TaskOutput')).toBeNull();
    expect(classifyTool('TaskStop')).toBeNull();
    expect(classifyTool('mcp__composio__gmail_send_email')).toBeNull();
  });
});

describe('pruneTimestamps', () => {
  it('removes entries older than the cutoff', () => {
    const ts = [100, 200, 300, 400, 500];
    expect(pruneTimestamps(ts, 300)).toEqual([300, 400, 500]);
  });

  it('returns an empty array when all entries are stale', () => {
    expect(pruneTimestamps([100, 200], 1000)).toEqual([]);
  });

  it('returns the input unchanged when all entries are within window', () => {
    expect(pruneTimestamps([100, 200], 50)).toEqual([100, 200]);
  });

  it('does not mutate the input', () => {
    const ts = [100, 200, 300];
    const out = pruneTimestamps(ts, 200);
    expect(ts).toEqual([100, 200, 300]);
    expect(out).toEqual([200, 300]);
  });
});

describe('recordEvent', () => {
  it('appends an agent_spawn timestamp + prunes stale entries', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: [FIXED_NOW - 7200, FIXED_NOW - 100],
      schedule_task_calls: [],
    };
    const next = recordEvent('agent_spawn', counters, FIXED_NOW);
    expect(next.agent_spawns).toEqual([FIXED_NOW - 100, FIXED_NOW]);
    expect(next.schedule_task_calls).toEqual([]);
  });

  it('appends a schedule_task timestamp + prunes the OTHER stream too', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: [FIXED_NOW - 7200, FIXED_NOW - 50],
      schedule_task_calls: [FIXED_NOW - 200],
    };
    const next = recordEvent('schedule_task', counters, FIXED_NOW);
    // The stream we DIDN'T append to still gets pruned to keep the
    // file bounded.
    expect(next.agent_spawns).toEqual([FIXED_NOW - 50]);
    expect(next.schedule_task_calls).toEqual([FIXED_NOW - 200, FIXED_NOW]);
  });

  it('does not mutate the input counters', () => {
    const counters = emptyCounters();
    const next = recordEvent('agent_spawn', counters, FIXED_NOW);
    expect(counters.agent_spawns).toEqual([]);
    expect(next.agent_spawns).toEqual([FIXED_NOW]);
  });
});

describe('decideRate — operator-trusted (audit only)', () => {
  it('allows when within budget', () => {
    const decision = decideRate(
      'agent_spawn',
      p([]),
      true, // trusted container
      emptyCounters(),
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('allow');
    if (decision.kind === 'allow') {
      expect(decision.provenance).toBe('operator-trusted');
      expect(decision.auditOnlyExceeded).toBeUndefined();
    }
  });

  it('allows EVEN when over the cap (audit-only) — operator workflow not gated', () => {
    // Cap is 50 agent spawns/hr for operator-trusted. Saturate it.
    const counters = {
      schema_version: 1 as const,
      agent_spawns: Array.from(
        { length: 60 },
        (_, i) => FIXED_NOW - 1000 + i,
      ),
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p([]),
      true,
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('allow');
    if (decision.kind === 'allow') {
      expect(decision.provenance).toBe('operator-trusted');
      expect(decision.auditOnlyExceeded).toBe(true);
    }
  });
});

describe('decideRate — operator-untrusted (gated)', () => {
  it('allows the 5th agent spawn within budget', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: Array.from(
        { length: 4 },
        (_, i) => FIXED_NOW - 100 + i,
      ),
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p([]),
      false, // untrusted container
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('allow');
  });

  it('denies the 6th agent spawn (cap is 5/hr)', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: Array.from(
        { length: 5 },
        (_, i) => FIXED_NOW - 100 + i,
      ),
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p([]),
      false,
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.provenance).toBe('operator-untrusted');
      expect(decision.cap).toBe(5);
      expect(decision.observed).toBe(6);
    }
  });
});

describe('decideRate — untrusted-source (web injection scenario)', () => {
  it('denies the 3rd agent spawn under web provenance (cap is 2/hr)', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: [FIXED_NOW - 200, FIXED_NOW - 100],
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p(['web']),
      true, // even in trusted container, web provenance wins
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.provenance).toBe('untrusted-source');
      expect(decision.cap).toBe(2);
    }
  });

  it('allows the 1st agent spawn under web provenance', () => {
    const decision = decideRate(
      'agent_spawn',
      p(['web']),
      false,
      emptyCounters(),
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('allow');
    if (decision.kind === 'allow') {
      expect(decision.provenance).toBe('untrusted-source');
    }
  });

  it('denies the 2nd schedule_task under untrusted-source (cap is 1/hr)', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: [],
      schedule_task_calls: [FIXED_NOW - 100],
    };
    const decision = decideRate(
      'schedule_task',
      p(['gmail']),
      true,
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.provenance).toBe('untrusted-source');
      expect(decision.cap).toBe(1);
    }
  });
});

describe('decideRate — cross-group (non-owner)', () => {
  it('denies the 2nd schedule_task under cross-group provenance', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: [],
      schedule_task_calls: [FIXED_NOW - 30],
    };
    const decision = decideRate(
      'schedule_task',
      p(['cross-group']),
      false,
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.provenance).toBe('cross-group');
      expect(decision.cap).toBe(1);
    }
  });
});

describe('decideRate — mixed chain', () => {
  it('classifies cross-group + web as mixed and applies the mixed row', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: [FIXED_NOW - 200, FIXED_NOW - 100],
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p(['cross-group', 'web']),
      true,
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.provenance).toBe('mixed');
      expect(decision.cap).toBe(2);
    }
  });
});

describe('decideRate — rolling window pruning', () => {
  it('expired timestamps do not count toward the cap', () => {
    const counters = {
      schema_version: 1 as const,
      // Five spawns more than 1 hour old + zero recent.
      agent_spawns: Array.from(
        { length: 5 },
        (_, i) => FIXED_NOW - 7200 + i,
      ),
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p([]),
      false,
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    // Operator-untrusted cap is 5/hr; the 5 stale entries are
    // pruned, so the next call is the 1st in window.
    expect(decision.kind).toBe('allow');
  });

  it('boundary: a timestamp exactly 3600s old is still in window', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: Array.from(
        { length: 5 },
        () => FIXED_NOW - 3600,
      ),
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p([]),
      false,
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('deny');
  });
});

describe('parseCounters', () => {
  it('accepts a well-formed payload', () => {
    expect(
      parseCounters({
        schema_version: 1,
        agent_spawns: [100, 200],
        schedule_task_calls: [],
      }),
    ).toEqual({
      schema_version: 1,
      agent_spawns: [100, 200],
      schedule_task_calls: [],
    });
  });

  it('returns null for null / non-object', () => {
    expect(parseCounters(null)).toBeNull();
    expect(parseCounters('string')).toBeNull();
    expect(parseCounters(42)).toBeNull();
  });

  it('returns null for missing schema_version', () => {
    expect(
      parseCounters({ agent_spawns: [], schedule_task_calls: [] }),
    ).toBeNull();
  });

  it('returns null for unknown schema_version', () => {
    expect(
      parseCounters({
        schema_version: 99,
        agent_spawns: [],
        schedule_task_calls: [],
      }),
    ).toBeNull();
  });

  it('returns null when arrays missing', () => {
    expect(parseCounters({ schema_version: 1 })).toBeNull();
    expect(
      parseCounters({ schema_version: 1, agent_spawns: [] }),
    ).toBeNull();
  });

  it('returns null when array contains non-number entries', () => {
    expect(
      parseCounters({
        schema_version: 1,
        agent_spawns: ['not a number'],
        schedule_task_calls: [],
      }),
    ).toBeNull();
    expect(
      parseCounters({
        schema_version: 1,
        agent_spawns: [],
        schedule_task_calls: [Infinity],
      }),
    ).toBeNull();
  });
});

describe('parseOverrides + applyOverrides', () => {
  it('parses a minimal override and applies it', () => {
    const parsed = parseOverrides({
      schema_version: 1,
      rows: { 'operator-trusted': { agentSpawnsPerHour: 5, auditOnly: false } },
    });
    expect(parsed).not.toBeNull();
    const matrix = applyOverrides(DEFAULT_CAP_MATRIX, parsed);
    expect(matrix['operator-trusted'].agentSpawnsPerHour).toBe(5);
    expect(matrix['operator-trusted'].auditOnly).toBe(false);
    // Unchanged rows pass through.
    expect(matrix['untrusted-source']).toEqual(
      DEFAULT_CAP_MATRIX['untrusted-source'],
    );
  });

  it('operator opt-in tightening: 6th spawn denied when trusted row is lowered to 5', () => {
    const matrix = applyOverrides(
      DEFAULT_CAP_MATRIX,
      parseOverrides({
        rows: {
          'operator-trusted': { agentSpawnsPerHour: 5, auditOnly: false },
        },
      }),
    );
    const counters = {
      schema_version: 1 as const,
      agent_spawns: Array.from({ length: 5 }, (_, i) => FIXED_NOW - 100 + i),
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p([]),
      true, // trusted container, operator-originated
      counters,
      matrix,
      FIXED_NOW,
    );
    expect(decision.kind).toBe('deny');
    if (decision.kind === 'deny') {
      expect(decision.provenance).toBe('operator-trusted');
      expect(decision.cap).toBe(5);
    }
  });

  it('rejects an attempt to flip non-trusted rows to auditOnly', () => {
    const matrix = applyOverrides(
      DEFAULT_CAP_MATRIX,
      parseOverrides({
        rows: {
          'untrusted-source': { auditOnly: true },
        },
      }),
    );
    // Untrusted-source row stays gated regardless of the override.
    expect(matrix['untrusted-source'].auditOnly).toBe(false);
  });

  it('clamps negative values to the default (a negative override does NOT silently disable the gate)', () => {
    const matrix = applyOverrides(
      DEFAULT_CAP_MATRIX,
      parseOverrides({
        rows: {
          'untrusted-source': { agentSpawnsPerHour: -1 },
        },
      }),
    );
    expect(matrix['untrusted-source'].agentSpawnsPerHour).toBe(
      DEFAULT_CAP_MATRIX['untrusted-source'].agentSpawnsPerHour,
    );
  });

  it('returns null for non-object input', () => {
    expect(parseOverrides(null)).toBeNull();
    expect(parseOverrides('not a json')).toBeNull();
  });

  it('rejects unknown schema_version', () => {
    expect(parseOverrides({ schema_version: 99, rows: {} })).toBeNull();
  });

  it('ignores unknown row keys (forward-compat)', () => {
    const parsed = parseOverrides({
      rows: {
        'operator-trusted': { agentSpawnsPerHour: 99 },
        'future-row': { agentSpawnsPerHour: 1 },
      },
    });
    expect(parsed).not.toBeNull();
    const matrix = applyOverrides(DEFAULT_CAP_MATRIX, parsed);
    expect(matrix['operator-trusted'].agentSpawnsPerHour).toBe(99);
  });
});

describe('decideRate reason text', () => {
  it('names the matched provenance row in the deny reason', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: [FIXED_NOW - 200, FIXED_NOW - 100],
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p(['web']),
      true,
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    if (decision.kind !== 'deny') throw new Error('expected deny');
    expect(decision.reason).toContain('untrusted-source');
    expect(decision.reason).toContain('agent_spawn');
    expect(decision.reason).toContain('3/2');
  });

  it('names the walk-back prefixes in the deny reason', () => {
    const counters = {
      schema_version: 1 as const,
      agent_spawns: [FIXED_NOW - 200, FIXED_NOW - 100],
      schedule_task_calls: [],
    };
    const decision = decideRate(
      'agent_spawn',
      p(['web', 'gmail']),
      true,
      counters,
      DEFAULT_CAP_MATRIX,
      FIXED_NOW,
    );
    if (decision.kind !== 'deny') throw new Error('expected deny');
    expect(decision.reason).toContain('web');
    expect(decision.reason).toContain('gmail');
  });
});
