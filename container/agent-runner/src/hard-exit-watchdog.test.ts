import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import {
  decideHardExitWatchdog,
  DRAIN_TIMEOUT_MS_MAX,
  HARD_EXIT_IDLE_BUDGET_MS,
  parseSkillNameFromPrompt,
  parseDrainTimeoutMsFromFrontmatter,
  resolveDrainTimeoutMs,
  shouldArmHardExitWatchdog,
} from './hard-exit-watchdog.js';

describe('decideHardExitWatchdog (#545 / #589)', () => {
  // The pure decision function the post-close hard-exit watchdog
  // consults on every fire. Tests cover the three action shapes (exit
  // / rearm-after-idle / rearm-due-to-pending-tool-call) plus the
  // boundary semantics that protect the wall-clock bound from
  // infinite re-arming on a chatty agent.

  it('exits when no SDK activity within the idle budget (stuck SDK iterator)', () => {
    const now = 1_000_000;
    const lastActivity = now - HARD_EXIT_IDLE_BUDGET_MS;
    const decision = decideHardExitWatchdog(now, lastActivity);
    expect(decision).toEqual({
      action: 'exit',
      idleMs: HARD_EXIT_IDLE_BUDGET_MS,
    });
  });

  it('exits when idle window exceeds the budget by any margin', () => {
    // The watchdog timer can fire slightly late (event-loop
    // contention, GC pause). idleMs > budget must still exit, not
    // re-arm at a negative interval.
    const now = 1_000_000;
    const lastActivity = now - (HARD_EXIT_IDLE_BUDGET_MS + 5_000);
    const decision = decideHardExitWatchdog(now, lastActivity);
    expect(decision.action).toBe('exit');
    expect(decision.action === 'exit' && decision.idleMs).toBe(
      HARD_EXIT_IDLE_BUDGET_MS + 5_000,
    );
  });

  it('re-arms with remaining budget when activity is recent (working agent)', () => {
    // Morning-brief shape: agent composed an assistant turn 5s ago,
    // close was detected earlier. The watchdog re-arms for the
    // remaining budget (budget - 5s) after the most recent event.
    const now = 1_000_000;
    const lastActivity = now - 5_000;
    const decision = decideHardExitWatchdog(now, lastActivity);
    expect(decision).toEqual({
      action: 'rearm',
      rearmInMs: HARD_EXIT_IDLE_BUDGET_MS - 5_000,
      idleMs: 5_000,
    });
  });

  it('re-arms with full budget when activity is now (zero idle)', () => {
    // Edge case: timer fires at exactly the same instant as a fresh
    // SDK event lands. idleMs == 0; re-arm for the full budget.
    const now = 1_000_000;
    const decision = decideHardExitWatchdog(now, now);
    expect(decision).toEqual({
      action: 'rearm',
      rearmInMs: HARD_EXIT_IDLE_BUDGET_MS,
      idleMs: 0,
    });
  });

  it('respects an explicit budget override (testability)', () => {
    const now = 1_000_000;
    const decisionExit = decideHardExitWatchdog(now, now - 1_500, 1_000);
    expect(decisionExit.action).toBe('exit');

    const decisionRearm = decideHardExitWatchdog(now, now - 500, 1_000);
    expect(decisionRearm).toEqual({
      action: 'rearm',
      rearmInMs: 500,
      idleMs: 500,
    });
  });

  it('treats the boundary `idleMs == budget` as exit when no tools are in flight', () => {
    // Without the >= boundary, the rearm branch could schedule a
    // 0ms setTimeout that fires immediately and re-enters the same
    // decision — a tight CPU-burning loop instead of a clean kill.
    const now = 1_000_000;
    const decision = decideHardExitWatchdog(
      now,
      now - HARD_EXIT_IDLE_BUDGET_MS,
      HARD_EXIT_IDLE_BUDGET_MS,
    );
    expect(decision.action).toBe('exit');
  });

  it('default budget is 90s (#589) — large enough for heavy-maintenance compose phases', () => {
    // Morning-brief composes a Telegram-HTML brief in one assistant
    // turn after a 33KB data fetch; thinking doesn't stream as a
    // separate SDK event so the activity-aware re-arm only sees one
    // block of silence. 30s tripped it consistently; 90s leaves
    // headroom while the SIGKILL deploy-kill cascade (#249/#250)
    // still bounds genuinely-stuck cases at the host hard timeout.
    expect(HARD_EXIT_IDLE_BUDGET_MS).toBe(90_000);
  });

  describe('in-flight tool_use re-arm (#589 option 3)', () => {
    // Between `tool_use` emission and the matching `tool_result`, the
    // agent is provably alive. The watchdog must re-arm
    // unconditionally so a Bash that legitimately runs 60s+ doesn't
    // trip on the artificially-flat idle window between those two
    // SDK events.

    it('re-arms with full budget when a tool call is in flight, even past the idle budget', () => {
      const now = 1_000_000;
      const decision = decideHardExitWatchdog(
        now,
        now - HARD_EXIT_IDLE_BUDGET_MS * 2,
        HARD_EXIT_IDLE_BUDGET_MS,
        1,
      );
      expect(decision).toEqual({
        action: 'rearm',
        rearmInMs: HARD_EXIT_IDLE_BUDGET_MS,
        idleMs: HARD_EXIT_IDLE_BUDGET_MS * 2,
      });
    });

    it('counts > 1 in-flight tool calls the same as 1 (any pending tool re-arms)', () => {
      const now = 1_000_000;
      const decision = decideHardExitWatchdog(
        now,
        now - HARD_EXIT_IDLE_BUDGET_MS,
        HARD_EXIT_IDLE_BUDGET_MS,
        3,
      );
      expect(decision.action).toBe('rearm');
    });

    it('exits normally when pending count is 0 (default arg)', () => {
      const now = 1_000_000;
      const decision = decideHardExitWatchdog(
        now,
        now - HARD_EXIT_IDLE_BUDGET_MS,
        HARD_EXIT_IDLE_BUDGET_MS,
        0,
      );
      expect(decision.action).toBe('exit');
    });
  });
});

describe('parseSkillNameFromPrompt (#589)', () => {
  it('extracts the first Skill(skill: "...") invocation', () => {
    expect(
      parseSkillNameFromPrompt('Skill(skill: "tessl__morning-brief")'),
    ).toBe('tessl__morning-brief');
  });

  it('accepts single quotes', () => {
    expect(parseSkillNameFromPrompt("Skill(skill: 'tessl__heartbeat')")).toBe(
      'tessl__heartbeat',
    );
  });

  it('matches inside a longer prompt (not just prefix)', () => {
    const prompt =
      'MANDATORY FIRST ACTION: invoke Skill(skill: "tessl__soul-searching") and proceed.';
    expect(parseSkillNameFromPrompt(prompt)).toBe('tessl__soul-searching');
  });

  it('returns undefined when no Skill() call is present', () => {
    expect(
      parseSkillNameFromPrompt('please summarize this thread'),
    ).toBeUndefined();
  });

  it('returns the first invocation when multiple appear', () => {
    expect(
      parseSkillNameFromPrompt(
        'Skill(skill: "tessl__a")\nthen Skill(skill: "tessl__b")',
      ),
    ).toBe('tessl__a');
  });
});

describe('parseDrainTimeoutMsFromFrontmatter (#589)', () => {
  it('extracts drain_timeout_ms when present and positive', () => {
    const content = `---
name: morning-brief
drain_timeout_ms: 180000
---

# Morning brief
`;
    expect(parseDrainTimeoutMsFromFrontmatter(content)).toBe(180000);
  });

  it('accepts quoted integer values', () => {
    const content = `---
drain_timeout_ms: "120000"
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(content)).toBe(120000);
  });

  it('returns undefined when the key is absent', () => {
    const content = `---
name: heartbeat
cadence: "*/2 * * * *"
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(content)).toBeUndefined();
  });

  it('returns undefined for non-integer values (regex, prose)', () => {
    const content = `---
drain_timeout_ms: forever
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(content)).toBeUndefined();
  });

  it('returns undefined for zero or negative integers', () => {
    const zero = `---
drain_timeout_ms: 0
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(zero)).toBeUndefined();
  });

  it('returns undefined when there is no frontmatter block', () => {
    expect(
      parseDrainTimeoutMsFromFrontmatter('# Just a header\n'),
    ).toBeUndefined();
  });

  it('returns undefined when the frontmatter block is unterminated', () => {
    const content = `---
drain_timeout_ms: 60000
`;
    expect(parseDrainTimeoutMsFromFrontmatter(content)).toBeUndefined();
  });

  it('ignores comments and blank lines inside the block', () => {
    const content = `---
# top comment

drain_timeout_ms: 45000
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(content)).toBe(45000);
  });

  it('strips an unquoted inline `# ...` comment before validating', () => {
    // Tile authors annotate non-obvious values with trailing
    // comments; matches the host-side cadence-registry's frontmatter
    // semantics so `drain_timeout_ms: 180000 # 3 minutes` doesn't
    // silently fall back to the default.
    const content = `---
drain_timeout_ms: 180000 # 3 minutes
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(content)).toBe(180000);
  });

  it('keeps a `#` inside a quoted value (no peek inside quotes)', () => {
    // Unlikely shape for drain_timeout_ms specifically, but the
    // parser's contract with quoted values is "don't peek inside" —
    // a malformed quoted-with-hash value should fail the digit
    // regex, not get truncated to a passing integer.
    const content = `---
drain_timeout_ms: "180000 # 3 minutes"
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(content)).toBeUndefined();
  });

  it('rejects values above DRAIN_TIMEOUT_MS_MAX (Int32-cliff guard)', () => {
    // Node clamps setTimeout delays > 2147483647 (Int32 max) to 1ms,
    // which would make the watchdog re-arm in a tight loop. The
    // 10-minute cap is well below the cliff and also catches typos
    // like a unit-confusion `min` vs `ms` (e.g. 180 mistakenly
    // written as 180_000_000).
    const big = `---
drain_timeout_ms: 99999999999
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(big)).toBeUndefined();

    const overOneMs = `---
drain_timeout_ms: ${DRAIN_TIMEOUT_MS_MAX + 1}
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(overOneMs)).toBeUndefined();

    const atMax = `---
drain_timeout_ms: ${DRAIN_TIMEOUT_MS_MAX}
---
`;
    expect(parseDrainTimeoutMsFromFrontmatter(atMax)).toBe(
      DRAIN_TIMEOUT_MS_MAX,
    );
  });
});

describe('resolveDrainTimeoutMs (#589)', () => {
  // Integration-shaped tests over the real filesystem — small SKILL.md
  // fixtures under a tmpdir mirror the container's skills mount
  // layout (`<skillsDir>/<skillName>/SKILL.md`). The cleanup hook
  // removes the tmpdir between tests so fixtures don't leak.

  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hard-exit-watchdog-'));
  });

  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  function installSkill(name: string, frontmatterBody: string): void {
    const dir = path.join(tmp, name);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(
      path.join(dir, 'SKILL.md'),
      `---\n${frontmatterBody}\n---\n\n# ${name}\n`,
    );
  }

  it('returns the skill override when frontmatter declares drain_timeout_ms', () => {
    installSkill('tessl__morning-brief', 'drain_timeout_ms: 180000');
    const prompt = 'Skill(skill: "tessl__morning-brief")';
    expect(resolveDrainTimeoutMs(prompt, tmp, 90_000)).toBe(180000);
  });

  it('falls back to default when the prompt has no Skill() call', () => {
    expect(resolveDrainTimeoutMs('hello', tmp, 90_000)).toBe(90_000);
  });

  it('falls back to default when the skill is not installed', () => {
    const prompt = 'Skill(skill: "tessl__missing")';
    expect(resolveDrainTimeoutMs(prompt, tmp, 90_000)).toBe(90_000);
  });

  it('falls back to default when the SKILL.md has no drain_timeout_ms field', () => {
    installSkill('tessl__heartbeat', 'cadence: "*/2 * * * *"');
    const prompt = 'Skill(skill: "tessl__heartbeat")';
    expect(resolveDrainTimeoutMs(prompt, tmp, 90_000)).toBe(90_000);
  });

  it('falls back to default when the override is malformed', () => {
    installSkill('tessl__broken', 'drain_timeout_ms: forever');
    const prompt = 'Skill(skill: "tessl__broken")';
    expect(resolveDrainTimeoutMs(prompt, tmp, 90_000)).toBe(90_000);
  });

  it('falls back to default when skillsDir does not exist (no skills mounted)', () => {
    const prompt = 'Skill(skill: "tessl__morning-brief")';
    expect(
      resolveDrainTimeoutMs(prompt, path.join(tmp, 'missing'), 90_000),
    ).toBe(90_000);
  });

  it('rejects path-traversal in the skill name and falls back to default', () => {
    // A prompt like `Skill(skill: "../../etc/passwd")` parses to that
    // exact string. Without the safe-name regex, `path.join(skillsDir,
    // '..', '..', 'etc', 'passwd', 'SKILL.md')` could walk out of
    // the mount entirely. Sanity-check: create a real `SKILL.md`
    // outside the skills root with a valid override; the resolver
    // must NOT see it.
    fs.writeFileSync(
      path.join(tmp, 'attacker-SKILL.md'),
      `---\ndrain_timeout_ms: 9999\n---\n`,
    );
    const skillsRoot = path.join(tmp, 'skills');
    fs.mkdirSync(skillsRoot, { recursive: true });
    // Naming the file `attacker-SKILL.md` directly (not under a
    // dir/`SKILL.md`) makes the regex assertion the load-bearing
    // gate even if path.join collapses `..` differently across
    // platforms — the regex blocks the lookup before the join.
    const prompt = 'Skill(skill: "../attacker")';
    expect(resolveDrainTimeoutMs(prompt, skillsRoot, 90_000)).toBe(90_000);
  });

  it('rejects skill names with slashes', () => {
    const prompt = 'Skill(skill: "foo/bar")';
    expect(resolveDrainTimeoutMs(prompt, tmp, 90_000)).toBe(90_000);
  });

  it('rejects skill names starting with a dot or hyphen', () => {
    expect(resolveDrainTimeoutMs('Skill(skill: ".hidden")', tmp, 90_000)).toBe(
      90_000,
    );
    expect(resolveDrainTimeoutMs('Skill(skill: "-flag")', tmp, 90_000)).toBe(
      90_000,
    );
  });

  it('accepts the canonical `tessl__<name>` shape with hyphens and underscores', () => {
    // Regression guard against a future regex tightening that
    // accidentally rejects the shape the installer actually uses.
    installSkill('tessl__morning-brief', 'drain_timeout_ms: 180000');
    const prompt = 'Skill(skill: "tessl__morning-brief")';
    expect(resolveDrainTimeoutMs(prompt, tmp, 90_000)).toBe(180_000);
  });

  it('propagates unexpected fs errors instead of silently disabling the override', () => {
    // Per coding-policy: error-handling, the resolver narrows the
    // swallowed errors to ENOENT/ENOTDIR — a missing SKILL.md means
    // "use the default". A permission error (EACCES) or other
    // non-miss should surface so the operator can diagnose, rather
    // than silently falling back. We assert by directing the
    // resolver at a directory that isn't readable by the current
    // user; on systems where chmod 0 still lets root read (CI),
    // skip the assertion rather than fail.
    if (process.getuid && process.getuid() === 0) return;
    const restrictedRoot = path.join(tmp, 'restricted');
    fs.mkdirSync(restrictedRoot, { recursive: true });
    const skillDir = path.join(restrictedRoot, 'tessl__locked');
    fs.mkdirSync(skillDir);
    fs.writeFileSync(
      path.join(skillDir, 'SKILL.md'),
      `---\ndrain_timeout_ms: 120000\n---\n`,
    );
    fs.chmodSync(skillDir, 0o000);
    try {
      const prompt = 'Skill(skill: "tessl__locked")';
      expect(() =>
        resolveDrainTimeoutMs(prompt, restrictedRoot, 90_000),
      ).toThrow();
    } finally {
      fs.chmodSync(skillDir, 0o700);
    }
  });
});

describe('shouldArmHardExitWatchdog (#589 reopened)', () => {
  // Maintenance one-shot spawns defer to the host-side
  // MAINTENANCE_CONTAINER_TIMEOUT inactivity bound, so the in-container
  // post-close watchdog is disabled for them. The compose turn that
  // stalled on an LLM / proxy blip emits no SDK event for the whole
  // stall and would otherwise trip the in-container budget before
  // `send_message` fires.
  it('does NOT arm for maintenance sessions (host inactivity timeout is the single bound)', () => {
    expect(shouldArmHardExitWatchdog(true)).toBe(false);
  });

  it('arms for interactive / default sessions (post-close idle is a real stuck-iterator signal there)', () => {
    expect(shouldArmHardExitWatchdog(false)).toBe(true);
  });
});
