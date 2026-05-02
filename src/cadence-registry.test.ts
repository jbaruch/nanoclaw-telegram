import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import Database from 'better-sqlite3';

import {
  parseSkillFrontmatter,
  splitCadenceTz,
  validateCadenceDeclaration,
  walkInstalledSkills,
  rebuildCadenceRegistry,
  defaultComputeNextRun,
} from './cadence-registry.js';

// Pinned `next_run` so per-test assertions don't drift with wall clock.
// Tests that exercise `defaultComputeNextRun` directly use a separate
// describe block.
const PINNED_NEXT_RUN = '2026-05-03T07:00:00.000Z';
const PINNED_NOW = new Date('2026-05-02T18:00:00.000Z');

function makeDb(): Database.Database {
  const db = new Database(':memory:');
  // Mirror the production scheduled_tasks shape: every column the
  // cadence-registry's INSERT touches must exist or the prepare() will
  // throw at test time exactly the way it would in production. Comments
  // intentionally pruned vs the production createSchema; this is just
  // the columns the registry writes plus the indexes the dispatcher
  // relies on.
  db.exec(`
    CREATE TABLE scheduled_tasks (
      id TEXT PRIMARY KEY,
      group_folder TEXT NOT NULL,
      chat_jid TEXT NOT NULL,
      prompt TEXT NOT NULL,
      script TEXT,
      schedule_type TEXT NOT NULL,
      schedule_value TEXT NOT NULL,
      schedule_timezone TEXT,
      context_mode TEXT,
      next_run TEXT,
      last_run TEXT,
      last_result TEXT,
      status TEXT DEFAULT 'active',
      created_at TEXT NOT NULL,
      created_by_role TEXT NOT NULL DEFAULT 'owner',
      continuation_cycle_id TEXT,
      session_id TEXT,
      source TEXT NOT NULL DEFAULT 'schedule-task'
    );
  `);
  return db;
}

function writeSkill(
  skillsDir: string,
  skillName: string,
  frontmatter: string,
  body: string = '# stub body',
): void {
  const dir = path.join(skillsDir, skillName);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(
    path.join(dir, 'SKILL.md'),
    `---\n${frontmatter}\n---\n\n${body}\n`,
  );
}

let tmpRoot: string;
beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'cadence-registry-test-'));
});
afterEach(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

describe('parseSkillFrontmatter', () => {
  it('parses bare scalar key:value pairs', () => {
    const fm = parseSkillFrontmatter(
      '---\nname: heartbeat\ncadence: "*/30 * * * *"\npriority: 0\n---\n# body\n',
    );
    expect(fm).toEqual({
      name: 'heartbeat',
      cadence: '*/30 * * * *',
      priority: 0,
    });
  });

  it('coerces signed integers but leaves cron strings alone', () => {
    const fm = parseSkillFrontmatter(
      '---\npriority: -100\ncadence: "0 7 * * *"\n---\n',
    );
    expect(fm.priority).toBe(-100);
    expect(fm.cadence).toBe('0 7 * * *');
  });

  it('strips both single and double quotes', () => {
    const fm = parseSkillFrontmatter(
      `---\na: "double"\nb: 'single'\nc: bare\n---\n`,
    );
    expect(fm).toEqual({ a: 'double', b: 'single', c: 'bare' });
  });

  it('returns {} for content without leading frontmatter block', () => {
    expect(parseSkillFrontmatter('# Skill\n\nNo frontmatter here.\n')).toEqual(
      {},
    );
  });

  it('returns {} for unterminated frontmatter', () => {
    expect(
      parseSkillFrontmatter('---\nname: orphan\n# never closed\n# body\n'),
    ).toEqual({});
  });

  it('skips comment-only and blank lines inside frontmatter', () => {
    const fm = parseSkillFrontmatter(
      '---\n# this is a comment\n\nname: lone\n# trailing comment\n---\n',
    );
    expect(fm).toEqual({ name: 'lone' });
  });

  it('handles CRLF line endings', () => {
    const fm = parseSkillFrontmatter(
      '---\r\nname: crlf\r\ncadence: "0 0 * * *"\r\n---\r\n# body\r\n',
    );
    expect(fm).toEqual({ name: 'crlf', cadence: '0 0 * * *' });
  });

  it('strips a UTF-8 BOM before the leading marker', () => {
    const fm = parseSkillFrontmatter('﻿---\nname: bommed\n---\n# body\n');
    expect(fm).toEqual({ name: 'bommed' });
  });

  it('does not coerce decimals or strings that merely contain digits', () => {
    const fm = parseSkillFrontmatter(
      '---\na: 3.14\nb: "0 7 * * *"\nc: 100abc\n---\n',
    );
    expect(fm.a).toBe('3.14');
    expect(fm.b).toBe('0 7 * * *');
    expect(fm.c).toBe('100abc');
  });

  it('strips inline `# comment` from bare scalars', () => {
    const fm = parseSkillFrontmatter(
      '---\npriority: 0 # ordering hint\nname: hb # the heartbeat\n---\n',
    );
    expect(fm.priority).toBe(0);
    expect(fm.name).toBe('hb');
  });

  it('does not strip a `#` inside a quoted scalar value', () => {
    const fm = parseSkillFrontmatter('---\nfragment: "section#foo"\n---\n');
    expect(fm.fragment).toBe('section#foo');
  });

  it('keeps a `#` adjacent to value (no leading whitespace) as part of the value', () => {
    const fm = parseSkillFrontmatter('---\ntag: v1#beta\n---\n');
    expect(fm.tag).toBe('v1#beta');
  });
});

describe('splitCadenceTz', () => {
  it('returns null tz for an unqualified cron', () => {
    expect(splitCadenceTz('*/30 * * * *')).toEqual({
      cron: '*/30 * * * *',
      tz: null,
    });
  });

  it('extracts the literal "local" qualifier', () => {
    expect(splitCadenceTz('0 7 * * * (TZ=local)')).toEqual({
      cron: '0 7 * * *',
      tz: 'local',
    });
  });

  it('extracts a pinned IANA name', () => {
    expect(splitCadenceTz('0 7 * * * (TZ=America/Los_Angeles)')).toEqual({
      cron: '0 7 * * *',
      tz: 'America/Los_Angeles',
    });
  });
});

describe('validateCadenceDeclaration', () => {
  it('returns null declaration when frontmatter has no cadence:', () => {
    const r = validateCadenceDeclaration({ name: 'x' }, 'x');
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.declaration).toBeNull();
  });

  it('extracts a valid cadence and default priority 0', () => {
    const r = validateCadenceDeclaration(
      { cadence: '*/30 * * * *' },
      'heartbeat',
    );
    expect(r.ok).toBe(true);
    if (r.ok && r.declaration)
      expect(r.declaration).toEqual({
        cadence: '*/30 * * * *',
        priority: 0,
      });
  });

  it('preserves an explicit priority', () => {
    const r = validateCadenceDeclaration(
      { cadence: '0 7 * * *', priority: -100 },
      'lock-acquire',
    );
    expect(r.ok).toBe(true);
    if (r.ok && r.declaration) expect(r.declaration.priority).toBe(-100);
  });

  it('rejects malformed cron with a skill-tagged error', () => {
    const r = validateCadenceDeclaration(
      { cadence: 'not-a-cron' },
      'busted-skill',
    );
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.errors).toHaveLength(1);
      expect(r.errors[0]).toContain('busted-skill');
      expect(r.errors[0]).toContain('not a valid cron');
    }
  });

  it('rejects non-integer priority', () => {
    const r = validateCadenceDeclaration(
      { cadence: '*/30 * * * *', priority: 'high' },
      'bad-priority',
    );
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.errors[0]).toContain("'priority:' must be an integer");
  });

  it('rejects a cadence with an empty TZ qualifier', () => {
    const r = validateCadenceDeclaration(
      { cadence: '0 7 * * * (TZ=)' },
      'empty-tz',
    );
    expect(r.ok).toBe(false);
  });

  it('accepts (TZ=local) and pinned IANA without parsing the bare cron in the wrong zone', () => {
    expect(
      validateCadenceDeclaration({ cadence: '0 7 * * * (TZ=local)' }, 'morning')
        .ok,
    ).toBe(true);
    expect(
      validateCadenceDeclaration(
        { cadence: '0 7 * * * (TZ=America/Los_Angeles)' },
        'morning',
      ).ok,
    ).toBe(true);
  });
});

describe('walkInstalledSkills', () => {
  it('returns [] for a missing directory rather than throwing', () => {
    expect(walkInstalledSkills(path.join(tmpRoot, 'absent'))).toEqual([]);
  });

  it('returns [] for an existing-but-empty skills dir', () => {
    const skills = path.join(tmpRoot, 'skills');
    fs.mkdirSync(skills);
    expect(walkInstalledSkills(skills)).toEqual([]);
  });

  it('skips subdirectories without a SKILL.md', () => {
    const skills = path.join(tmpRoot, 'skills');
    fs.mkdirSync(path.join(skills, 'orphan-dir'), { recursive: true });
    writeSkill(skills, 'real-skill', 'name: real-skill');
    const out = walkInstalledSkills(skills);
    expect(out.map((s) => s.name)).toEqual(['real-skill']);
  });

  it('returns one record per SKILL.md with parsed frontmatter', () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'a', 'name: a\ncadence: "*/30 * * * *"');
    writeSkill(skills, 'b', 'name: b');
    const out = walkInstalledSkills(skills);
    const byName = Object.fromEntries(out.map((s) => [s.name, s]));
    expect(byName.a.frontmatter.cadence).toBe('*/30 * * * *');
    expect(byName.b.frontmatter.cadence).toBeUndefined();
  });
});

describe('rebuildCadenceRegistry', () => {
  it('inserts one row per declared cadence and 0 for skills without one', () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__heartbeat', 'cadence: "*/30 * * * *"');
    writeSkill(
      skills,
      'tessl__morning-brief',
      'cadence: "0 7 * * * (TZ=local)"',
    );
    writeSkill(skills, 'tessl__manage-tasks', 'name: manage-tasks'); // no cadence
    const db = makeDb();
    const r = rebuildCadenceRegistry({
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner',
      skillsDir: skills,
      computeNextRun: () => PINNED_NEXT_RUN,
      now: () => PINNED_NOW,
    });
    expect(r.inserted).toBe(2);
    expect(r.deleted).toBe(0);
    expect(r.walked).toBe(3);
    expect(r.errors).toEqual([]);
    const rows = db
      .prepare(
        "SELECT id, prompt, schedule_value, schedule_timezone, source FROM scheduled_tasks WHERE source = 'cadence-registry' AND group_folder = ? ORDER BY id",
      )
      .all('g1');
    expect(rows).toEqual([
      {
        id: 'cadence-registry::g1::tessl__heartbeat',
        prompt: 'Skill(skill: "tessl__heartbeat")',
        schedule_value: '*/30 * * * *',
        schedule_timezone: null,
        source: 'cadence-registry',
      },
      {
        id: 'cadence-registry::g1::tessl__morning-brief',
        prompt: 'Skill(skill: "tessl__morning-brief")',
        schedule_value: '0 7 * * *',
        schedule_timezone: 'local',
        source: 'cadence-registry',
      },
    ]);
  });

  it('preserves untouched rows on a second rebuild — same shape ⇒ no INSERT/UPDATE', () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__heartbeat', 'cadence: "*/30 * * * *"');
    const db = makeDb();
    const deps = {
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner' as const,
      skillsDir: skills,
      computeNextRun: () => PINNED_NEXT_RUN,
      now: () => PINNED_NOW,
    };
    const r1 = rebuildCadenceRegistry(deps);
    expect(r1).toMatchObject({
      inserted: 1,
      updated: 0,
      preserved: 0,
      deleted: 0,
    });
    const r2 = rebuildCadenceRegistry(deps);
    expect(r2).toMatchObject({
      inserted: 0,
      updated: 0,
      preserved: 1,
      deleted: 0,
    });
    const rows = db
      .prepare(
        "SELECT id FROM scheduled_tasks WHERE source = 'cadence-registry' AND group_folder = ?",
      )
      .all('g1');
    expect(rows).toEqual([{ id: 'cadence-registry::g1::tessl__heartbeat' }]);
  });

  it('preserves next_run / session_id / last_run / last_result on shape-unchanged rebuild', () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__heartbeat', 'cadence: "*/30 * * * *"');
    const db = makeDb();
    const deps = {
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner' as const,
      skillsDir: skills,
      computeNextRun: () => PINNED_NEXT_RUN,
      now: () => PINNED_NOW,
    };
    rebuildCadenceRegistry(deps);
    // Simulate the scheduler having advanced state on this row in
    // between two spawns: a fire happened, session_id was set on
    // first fire (per #336), last_run + last_result were stamped,
    // and next_run was advanced past the spawn boundary.
    db.prepare(
      `UPDATE scheduled_tasks
         SET next_run = ?, session_id = ?, last_run = ?, last_result = ?
       WHERE id = ?`,
    ).run(
      '2026-05-04T07:00:00.000Z',
      'sdk-session-abc123',
      '2026-05-03T07:00:00.000Z',
      'success',
      'cadence-registry::g1::tessl__heartbeat',
    );
    rebuildCadenceRegistry(deps);
    const row = db
      .prepare(
        `SELECT next_run, session_id, last_run, last_result FROM scheduled_tasks WHERE id = ?`,
      )
      .get('cadence-registry::g1::tessl__heartbeat');
    expect(row).toEqual({
      next_run: '2026-05-04T07:00:00.000Z',
      session_id: 'sdk-session-abc123',
      last_run: '2026-05-03T07:00:00.000Z',
      last_result: 'success',
    });
  });

  it('UPDATEs and recomputes next_run when the cadence string changes (preserves session_id / last_run / last_result)', () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__heartbeat', 'cadence: "*/30 * * * *"');
    const db = makeDb();
    const FIRST_NEXT = '2026-05-03T07:00:00.000Z';
    const SECOND_NEXT = '2026-05-03T07:15:00.000Z';
    const deps1 = {
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner' as const,
      skillsDir: skills,
      computeNextRun: () => FIRST_NEXT,
      now: () => PINNED_NOW,
    };
    rebuildCadenceRegistry(deps1);
    db.prepare(
      `UPDATE scheduled_tasks
         SET session_id = ?, last_run = ?, last_result = ?
       WHERE id = ?`,
    ).run(
      'sdk-session-abc123',
      '2026-05-02T07:00:00.000Z',
      'success',
      'cadence-registry::g1::tessl__heartbeat',
    );
    // Simulate a tile author tightening the cadence between spawns.
    fs.rmSync(path.join(skills, 'tessl__heartbeat'), { recursive: true });
    writeSkill(skills, 'tessl__heartbeat', 'cadence: "*/15 * * * *"');
    const r = rebuildCadenceRegistry({
      ...deps1,
      computeNextRun: () => SECOND_NEXT,
    });
    expect(r).toMatchObject({ inserted: 0, updated: 1, preserved: 0 });
    const row = db
      .prepare(
        `SELECT schedule_value, next_run, session_id, last_run, last_result FROM scheduled_tasks WHERE id = ?`,
      )
      .get('cadence-registry::g1::tessl__heartbeat');
    expect(row).toEqual({
      schedule_value: '*/15 * * * *',
      next_run: SECOND_NEXT, // recomputed because cadence changed
      session_id: 'sdk-session-abc123', // history preserved
      last_run: '2026-05-02T07:00:00.000Z',
      last_result: 'success',
    });
  });

  it("does not abort the rebuild when one skill's computeNextRun throws", () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__good', 'cadence: "*/30 * * * *"');
    writeSkill(skills, 'tessl__bad-tz', 'cadence: "0 7 * * * (TZ=Not/A/Real)"');
    const db = makeDb();
    const r = rebuildCadenceRegistry({
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner',
      skillsDir: skills,
      // Throw only for the bad-tz skill's lookup.
      computeNextRun: (_cron, tz) => {
        if (tz === 'Not/A/Real') throw new Error('Invalid IANA zone');
        return PINNED_NEXT_RUN;
      },
      now: () => PINNED_NOW,
    });
    expect(r.inserted).toBe(1);
    expect(r.errors).toHaveLength(1);
    expect(r.errors[0]).toContain('tessl__bad-tz');
    expect(r.errors[0]).toContain('computeNextRun failed');
    const ids = db
      .prepare(
        "SELECT id FROM scheduled_tasks WHERE source = 'cadence-registry'",
      )
      .all() as Array<{ id: string }>;
    expect(ids.map((r) => r.id)).toEqual(['cadence-registry::g1::tessl__good']);
  });

  it("does not touch source='schedule-task' rows in the same group", () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__heartbeat', 'cadence: "*/30 * * * *"');
    const db = makeDb();
    db.prepare(
      `INSERT INTO scheduled_tasks (id, group_folder, chat_jid, prompt, schedule_type, schedule_value, status, created_at, created_by_role, source)
       VALUES (?, ?, ?, ?, 'cron', ?, 'active', ?, 'owner', 'schedule-task')`,
    ).run(
      'owner-monitor-1',
      'g1',
      'g1@chat',
      'remind me to drink water',
      '0 14 * * *',
      PINNED_NOW.toISOString(),
    );
    rebuildCadenceRegistry({
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner',
      skillsDir: skills,
      computeNextRun: () => PINNED_NEXT_RUN,
      now: () => PINNED_NOW,
    });
    const surviving = db
      .prepare(
        "SELECT id FROM scheduled_tasks WHERE source = 'schedule-task' AND group_folder = ?",
      )
      .all('g1');
    expect(surviving).toEqual([{ id: 'owner-monitor-1' }]);
  });

  it('does not touch cadence-registry rows in OTHER groups', () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__heartbeat', 'cadence: "*/30 * * * *"');
    const db = makeDb();
    // Pre-existing cadence-registry row from another group's prior spawn.
    db.prepare(
      `INSERT INTO scheduled_tasks (id, group_folder, chat_jid, prompt, schedule_type, schedule_value, status, created_at, created_by_role, source)
       VALUES (?, ?, ?, ?, 'cron', ?, 'active', ?, 'owner', 'cadence-registry')`,
    ).run(
      'cadence-registry::g2::tessl__heartbeat',
      'g2',
      'g2@chat',
      'Skill(skill: "tessl__heartbeat")',
      '*/30 * * * *',
      PINNED_NOW.toISOString(),
    );
    rebuildCadenceRegistry({
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner',
      skillsDir: skills,
      computeNextRun: () => PINNED_NEXT_RUN,
      now: () => PINNED_NOW,
    });
    const g2Rows = db
      .prepare(
        "SELECT id FROM scheduled_tasks WHERE source = 'cadence-registry' AND group_folder = ?",
      )
      .all('g2');
    expect(g2Rows).toEqual([{ id: 'cadence-registry::g2::tessl__heartbeat' }]);
  });

  it('skips skills with malformed declarations and reports them', () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__good', 'cadence: "*/30 * * * *"');
    writeSkill(skills, 'tessl__bad', 'cadence: "not-a-cron"');
    const db = makeDb();
    const r = rebuildCadenceRegistry({
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner',
      skillsDir: skills,
      computeNextRun: () => PINNED_NEXT_RUN,
      now: () => PINNED_NOW,
    });
    expect(r.inserted).toBe(1);
    expect(r.errors).toHaveLength(1);
    expect(r.errors[0]).toContain('tessl__bad');
    const ids = db
      .prepare(
        "SELECT id FROM scheduled_tasks WHERE source = 'cadence-registry'",
      )
      .all() as Array<{ id: string }>;
    expect(ids.map((r) => r.id)).toEqual(['cadence-registry::g1::tessl__good']);
  });

  it('reduces to a no-op DELETE+0-INSERT when no skill declares a cadence', () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__plain', 'name: plain'); // no cadence:
    const db = makeDb();
    const r = rebuildCadenceRegistry({
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner',
      skillsDir: skills,
      computeNextRun: () => PINNED_NEXT_RUN,
      now: () => PINNED_NOW,
    });
    expect(r.inserted).toBe(0);
    expect(r.deleted).toBe(0);
    expect(r.errors).toEqual([]);
  });

  it('removes a previously-registered skill on a subsequent rebuild', () => {
    const skills = path.join(tmpRoot, 'skills');
    writeSkill(skills, 'tessl__heartbeat', 'cadence: "*/30 * * * *"');
    writeSkill(skills, 'tessl__nightly', 'cadence: "0 2 * * *"');
    const db = makeDb();
    const baseDeps = {
      db,
      groupFolder: 'g1',
      chatJid: 'g1@chat',
      createdByRole: 'owner' as const,
      skillsDir: skills,
      computeNextRun: () => PINNED_NEXT_RUN,
      now: () => PINNED_NOW,
    };
    rebuildCadenceRegistry(baseDeps);
    expect(
      db
        .prepare(
          "SELECT COUNT(*) as n FROM scheduled_tasks WHERE source = 'cadence-registry'",
        )
        .get(),
    ).toEqual({ n: 2 });
    // Simulate uninstalling `nightly` between spawns.
    fs.rmSync(path.join(skills, 'tessl__nightly'), { recursive: true });
    rebuildCadenceRegistry(baseDeps);
    const surviving = db
      .prepare(
        "SELECT id FROM scheduled_tasks WHERE source = 'cadence-registry'",
      )
      .all() as Array<{ id: string }>;
    expect(surviving.map((r) => r.id)).toEqual([
      'cadence-registry::g1::tessl__heartbeat',
    ]);
  });
});

describe('defaultComputeNextRun', () => {
  it('produces an ISO timestamp for a bare cron', () => {
    const next = defaultComputeNextRun('*/30 * * * *', null);
    expect(next).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  });

  it('honours a pinned IANA TZ', () => {
    // We only check the call doesn't throw on a valid IANA name. The
    // exact timestamp depends on the test wall clock; the important
    // contract is that cron-parser accepts the tz option.
    const next = defaultComputeNextRun('0 7 * * *', 'America/Los_Angeles');
    expect(next).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  });

  it('treats tz=local as UTC at insert-time (scheduler reapplies on fire)', () => {
    const a = defaultComputeNextRun('0 7 * * *', null);
    const b = defaultComputeNextRun('0 7 * * *', 'local');
    expect(a).toBe(b);
  });
});
