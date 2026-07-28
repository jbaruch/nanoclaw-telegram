import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import {
  parsePrecheckTimeoutMsFromFrontmatter,
  resolvePrecheckTimeoutMs,
} from './precheck-timeout.js';

describe('parsePrecheckTimeoutMsFromFrontmatter (#890)', () => {
  // The pure frontmatter parser. Every rejection returns `undefined`,
  // which the resolver passes through as "no declared budget" — so a
  // malformed declaration degrades to the container-bounded run, never
  // to a broken precheck.

  it('reads a bare positive integer', () => {
    const content =
      '---\nname: drive-engine\nprecheck_timeout_ms: 90000\n---\n';
    expect(parsePrecheckTimeoutMsFromFrontmatter(content)).toBe(90_000);
  });

  it('reads quoted values (single and double)', () => {
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        '---\nprecheck_timeout_ms: "90000"\n---\n',
      ),
    ).toBe(90_000);
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        "---\nprecheck_timeout_ms: '90000'\n---\n",
      ),
    ).toBe(90_000);
  });

  it('strips an inline comment from an unquoted value', () => {
    // The shape a skill author naturally writes when annotating a
    // non-obvious budget. Without comment-stripping this fails the
    // digit check and silently reads as no declaration at all.
    const content =
      '---\nprecheck_timeout_ms: 90000 # cold reconcile sweep\n---\n';
    expect(parsePrecheckTimeoutMsFromFrontmatter(content)).toBe(90_000);
  });

  it('honours a large declaration — no ceiling is imposed here', () => {
    // Two timeouts bound a precheck: this declaration and the host's
    // container kill. Re-asserting a ceiling in the runner would be a
    // third, and would be wrong for any group that raised
    // `containerConfig.timeout`.
    const content = '---\nprecheck_timeout_ms: 600000\n---\n';
    expect(parsePrecheckTimeoutMsFromFrontmatter(content)).toBe(600_000);
  });

  it('rejects a value above Node’s max timer delay', () => {
    // Node substitutes 1ms for a delay above Int32 max instead of
    // waiting longer, so honouring the literal number would invert the
    // widest declaration into an immediate kill. Resolving to "no
    // declaration" lands it on the container bound instead.
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        '---\nprecheck_timeout_ms: 2147483648\n---\n',
      ),
    ).toBeUndefined();
    // The boundary itself is representable and still honoured.
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        '---\nprecheck_timeout_ms: 2147483647\n---\n',
      ),
    ).toBe(2_147_483_647);
  });

  it('rejects zero and negative values', () => {
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        '---\nprecheck_timeout_ms: 0\n---\n',
      ),
    ).toBeUndefined();
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        '---\nprecheck_timeout_ms: -5000\n---\n',
      ),
    ).toBeUndefined();
  });

  it('rejects non-integer scalars', () => {
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        '---\nprecheck_timeout_ms: 90s\n---\n',
      ),
    ).toBeUndefined();
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        '---\nprecheck_timeout_ms: 90.5\n---\n',
      ),
    ).toBeUndefined();
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        '---\nprecheck_timeout_ms: \n---\n',
      ),
    ).toBeUndefined();
  });

  it('returns undefined when the key is absent', () => {
    expect(
      parsePrecheckTimeoutMsFromFrontmatter('---\ncadence: "0 4 * * *"\n---\n'),
    ).toBeUndefined();
  });

  it('returns undefined when there is no frontmatter block', () => {
    expect(
      parsePrecheckTimeoutMsFromFrontmatter('# drive-engine\n\nbody\n'),
    ).toBeUndefined();
  });

  it('returns undefined for an unterminated frontmatter block', () => {
    expect(
      parsePrecheckTimeoutMsFromFrontmatter(
        '---\nprecheck_timeout_ms: 90000\n\n# body with no close\n',
      ),
    ).toBeUndefined();
  });

  it('tolerates a leading BOM', () => {
    const content = '﻿---\nprecheck_timeout_ms: 90000\n---\n';
    expect(parsePrecheckTimeoutMsFromFrontmatter(content)).toBe(90_000);
  });

  it('tolerates CRLF line endings', () => {
    const content = '---\r\nprecheck_timeout_ms: 90000\r\n---\r\n';
    expect(parsePrecheckTimeoutMsFromFrontmatter(content)).toBe(90_000);
  });

  it('ignores a commented-out declaration', () => {
    const content = '---\n# precheck_timeout_ms: 90000\n---\n';
    expect(parsePrecheckTimeoutMsFromFrontmatter(content)).toBeUndefined();
  });
});

describe('resolvePrecheckTimeoutMs (#890)', () => {
  // Integration-shaped over the real filesystem — small SKILL.md
  // fixtures under a tmpdir mirror the container's skills mount layout
  // (`<skillsDir>/<skillName>/SKILL.md`).

  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'precheck-timeout-'));
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

  it('returns the declared budget for the invoked skill', () => {
    installSkill(
      'tessl__drive-engine',
      'name: drive-engine\nprecheck_timeout_ms: 90000',
    );
    expect(
      resolvePrecheckTimeoutMs('Skill(skill: "tessl__drive-engine")', tmp),
    ).toBe(90_000);
  });

  it('resolves the cadence-registry prompt shape verbatim', () => {
    // `src/cadence-registry.ts` builds every cadence task's prompt as
    // exactly this string — the association #890 flagged as its open
    // implementation question. Pin it: if the registry's prompt shape
    // drifts, this fails rather than silently dropping every skill's
    // declared budget.
    const skillName = 'tessl__drive-engine';
    installSkill(skillName, 'precheck_timeout_ms: 90000');
    expect(resolvePrecheckTimeoutMs(`Skill(skill: "${skillName}")`, tmp)).toBe(
      90_000,
    );
  });

  it('returns undefined when the skill declares nothing', () => {
    // No declared budget → runScript arms no timer → the container
    // kill is the only bound.
    installSkill('tessl__heartbeat', 'cadence: "*/2 * * * *"');
    expect(
      resolvePrecheckTimeoutMs('Skill(skill: "tessl__heartbeat")', tmp),
    ).toBeUndefined();
  });

  it('returns undefined when the prompt carries no Skill() invocation', () => {
    // An ad-hoc scheduled task whose script isn't skill-owned.
    expect(
      resolvePrecheckTimeoutMs('run the nightly sweep', tmp),
    ).toBeUndefined();
  });

  it('returns undefined when the skill is not installed', () => {
    expect(
      resolvePrecheckTimeoutMs('Skill(skill: "tessl__missing")', tmp),
    ).toBeUndefined();
  });

  it('refuses a path-traversal skill name instead of reading outside the mount', () => {
    // The safe-name regex is the defence: a prompt that smuggles `..`
    // must never walk the resolver into a SKILL.md outside `skillsDir`.
    fs.writeFileSync(
      path.join(tmp, 'SKILL.md'),
      '---\nprecheck_timeout_ms: 600000\n---\n',
    );
    const mount = path.join(tmp, 'victim');
    fs.mkdirSync(mount, { recursive: true });
    expect(resolvePrecheckTimeoutMs('Skill(skill: "../")', mount)).toBe(
      undefined,
    );
  });
});
