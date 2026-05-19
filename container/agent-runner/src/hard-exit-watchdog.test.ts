import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import {
  decideHardExitWatchdog,
  HARD_EXIT_IDLE_BUDGET_MS,
  parseSkillNameFromPrompt,
  parseDrainTimeoutMsFromFrontmatter,
  resolveDrainTimeoutMs,
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
    expect(parseSkillNameFromPrompt('Skill(skill: "tessl__morning-brief")')).toBe(
      'tessl__morning-brief',
    );
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
    expect(parseSkillNameFromPrompt('please summarize this thread')).toBeUndefined();
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
    expect(parseDrainTimeoutMsFromFrontmatter('# Just a header\n')).toBeUndefined();
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
});
