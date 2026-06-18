import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import {
  parseRequiresDeliveryFromFrontmatter,
  resolveRequiresDelivery,
} from './delivery-requirement.js';

describe('parseRequiresDeliveryFromFrontmatter (#689)', () => {
  // The pure frontmatter parser the runner consults to decide whether a
  // skill's run that delivered nothing should be flagged `noDelivery`.
  // Covers the true-shapes, the false/absent shapes, and the malformed
  // edge cases that must NOT accidentally read as `true`.

  it('returns true for bare `requires_delivery: true`', () => {
    const content =
      '---\nname: morning-brief\nrequires_delivery: true\n---\n\n# body\n';
    expect(parseRequiresDeliveryFromFrontmatter(content)).toBe(true);
  });

  it('accepts a quoted true value', () => {
    const dq = '---\nrequires_delivery: "true"\n---\n';
    const sq = "---\nrequires_delivery: 'true'\n---\n";
    expect(parseRequiresDeliveryFromFrontmatter(dq)).toBe(true);
    expect(parseRequiresDeliveryFromFrontmatter(sq)).toBe(true);
  });

  it('accepts case-insensitive True / TRUE (YAML booleans)', () => {
    expect(
      parseRequiresDeliveryFromFrontmatter(
        '---\nrequires_delivery: True\n---\n',
      ),
    ).toBe(true);
    expect(
      parseRequiresDeliveryFromFrontmatter(
        '---\nrequires_delivery: TRUE\n---\n',
      ),
    ).toBe(true);
  });

  it('strips an unquoted inline comment before validating', () => {
    const content =
      '---\nrequires_delivery: true # always pins the brief to chat\n---\n';
    expect(parseRequiresDeliveryFromFrontmatter(content)).toBe(true);
  });

  it('returns false for an explicit false value', () => {
    expect(
      parseRequiresDeliveryFromFrontmatter(
        '---\nrequires_delivery: false\n---\n',
      ),
    ).toBe(false);
  });

  it('returns false when the key is absent', () => {
    const content = '---\nname: heartbeat\ncadence: "*/2 * * * *"\n---\n';
    expect(parseRequiresDeliveryFromFrontmatter(content)).toBe(false);
  });

  it('returns false for a non-boolean scalar', () => {
    expect(
      parseRequiresDeliveryFromFrontmatter(
        '---\nrequires_delivery: yes\n---\n',
      ),
    ).toBe(false);
    expect(
      parseRequiresDeliveryFromFrontmatter('---\nrequires_delivery: 1\n---\n'),
    ).toBe(false);
  });

  it('returns false when there is no frontmatter block', () => {
    expect(parseRequiresDeliveryFromFrontmatter('# Just a header\n')).toBe(
      false,
    );
  });

  it('returns false when the frontmatter block is unterminated', () => {
    const content = '---\nrequires_delivery: true\n\n# body with no close\n';
    expect(parseRequiresDeliveryFromFrontmatter(content)).toBe(false);
  });

  it('ignores comments and blank lines inside the block', () => {
    const content =
      '---\n# leading comment\n\nname: morning-brief\nrequires_delivery: true\n---\n';
    expect(parseRequiresDeliveryFromFrontmatter(content)).toBe(true);
  });

  it('tolerates a leading BOM', () => {
    const content = '﻿---\nrequires_delivery: true\n---\n';
    expect(parseRequiresDeliveryFromFrontmatter(content)).toBe(true);
  });

  it('keeps a `#` inside a quoted value (no peek inside quotes)', () => {
    // `"true # x"` is not the boolean `true` — a quoted value with an
    // embedded hash must not be split at the hash and read as `true`.
    const content = '---\nrequires_delivery: "true # x"\n---\n';
    expect(parseRequiresDeliveryFromFrontmatter(content)).toBe(false);
  });
});

describe('resolveRequiresDelivery (#689)', () => {
  // Integration-shaped tests over the real filesystem — small SKILL.md
  // fixtures under a tmpdir mirror the container's skills mount layout
  // (`<skillsDir>/<skillName>/SKILL.md`). Mirrors resolveDrainTimeoutMs.

  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'delivery-requirement-'));
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

  it('returns true when the invoked skill declares requires_delivery: true', () => {
    installSkill(
      'tessl__morning-brief',
      'name: morning-brief\nrequires_delivery: true',
    );
    const prompt = 'Skill(skill: "tessl__morning-brief")';
    expect(resolveRequiresDelivery(prompt, tmp)).toBe(true);
  });

  it('returns false when the prompt has no Skill() call', () => {
    expect(resolveRequiresDelivery('good morning', tmp)).toBe(false);
  });

  it('returns false when the skill is not installed', () => {
    expect(resolveRequiresDelivery('Skill(skill: "tessl__missing")', tmp)).toBe(
      false,
    );
  });

  it('returns false when the SKILL.md has no requires_delivery field', () => {
    installSkill('tessl__heartbeat', 'cadence: "*/2 * * * *"');
    expect(
      resolveRequiresDelivery('Skill(skill: "tessl__heartbeat")', tmp),
    ).toBe(false);
  });

  it('returns false when requires_delivery is explicitly false', () => {
    installSkill('tessl__nightly', 'requires_delivery: false');
    expect(resolveRequiresDelivery('Skill(skill: "tessl__nightly")', tmp)).toBe(
      false,
    );
  });

  it('returns false when skillsDir does not exist (no skills mounted)', () => {
    const prompt = 'Skill(skill: "tessl__morning-brief")';
    expect(resolveRequiresDelivery(prompt, path.join(tmp, 'missing'))).toBe(
      false,
    );
  });

  it('rejects path-traversal in the skill name and falls back to false', () => {
    // A prompt like `Skill(skill: "../../etc/passwd")` parses to that
    // exact string; the safe-name regex must reject it before any read.
    fs.writeFileSync(
      path.join(tmp, 'attacker-SKILL.md'),
      `---\nrequires_delivery: true\n---\n`,
    );
    const prompt = 'Skill(skill: "../attacker")';
    expect(resolveRequiresDelivery(prompt, tmp)).toBe(false);
  });
});
