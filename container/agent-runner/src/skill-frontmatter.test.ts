import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import {
  readFrontmatterScalar,
  readSkillMdForPrompt,
  parseSkillNameFromPrompt,
  SAFE_SKILL_NAME_RE,
} from './skill-frontmatter.js';

describe('readFrontmatterScalar (#890)', () => {
  // The single frontmatter walk all three per-skill overrides share.
  // It returns the RAW scalar — range and type validation belong to
  // each override, so this only covers extraction and the shapes that
  // must yield nothing.

  it('reads a bare scalar', () => {
    expect(readFrontmatterScalar('---\nkey: value\n---\n', 'key')).toBe(
      'value',
    );
  });

  it('strips surrounding double and single quotes', () => {
    expect(readFrontmatterScalar('---\nkey: "value"\n---\n', 'key')).toBe(
      'value',
    );
    expect(readFrontmatterScalar("---\nkey: 'value'\n---\n", 'key')).toBe(
      'value',
    );
  });

  it('strips an inline comment from an unquoted value', () => {
    expect(readFrontmatterScalar('---\nkey: value # why\n---\n', 'key')).toBe(
      'value',
    );
  });

  it('leaves a # inside a quoted value alone', () => {
    expect(
      readFrontmatterScalar('---\nkey: "value # kept"\n---\n', 'key'),
    ).toBe('value # kept');
  });

  it('does not treat a hash with no leading space as a comment', () => {
    expect(readFrontmatterScalar('---\nkey: a#b\n---\n', 'key')).toBe('a#b');
  });

  it('returns the requested key, not a neighbouring one', () => {
    const content = '---\nother: nope\nkey: yes\nlater: nope\n---\n';
    expect(readFrontmatterScalar(content, 'key')).toBe('yes');
  });

  it('returns undefined for an absent key', () => {
    expect(
      readFrontmatterScalar('---\nother: v\n---\n', 'key'),
    ).toBeUndefined();
  });

  it('returns undefined without a leading frontmatter block', () => {
    expect(readFrontmatterScalar('# title\n\nkey: value\n', 'key')).toBe(
      undefined,
    );
  });

  it('returns undefined for an unterminated block', () => {
    expect(
      readFrontmatterScalar('---\nkey: value\n\n# body\n', 'key'),
    ).toBeUndefined();
  });

  it('skips commented-out declarations', () => {
    expect(
      readFrontmatterScalar('---\n# key: value\n---\n', 'key'),
    ).toBeUndefined();
  });

  it('tolerates a leading BOM and CRLF endings', () => {
    expect(readFrontmatterScalar('﻿---\nkey: value\n---\n', 'key')).toBe(
      'value',
    );
    expect(readFrontmatterScalar('---\r\nkey: value\r\n---\r\n', 'key')).toBe(
      'value',
    );
  });
});

describe('parseSkillNameFromPrompt / SAFE_SKILL_NAME_RE (#890 move)', () => {
  // Moved verbatim from hard-exit-watchdog.ts; re-exported there for
  // existing importers. Covered here at their new home.

  it('extracts the skill name from both quote styles', () => {
    expect(parseSkillNameFromPrompt('Skill(skill: "tessl__brief")')).toBe(
      'tessl__brief',
    );
    expect(parseSkillNameFromPrompt("Skill(skill: 'tessl__brief')")).toBe(
      'tessl__brief',
    );
  });

  it('returns undefined with no invocation', () => {
    expect(parseSkillNameFromPrompt('good morning')).toBeUndefined();
  });

  it('accepts tile-namespaced names and rejects traversal', () => {
    expect(SAFE_SKILL_NAME_RE.test('tessl__drive-engine')).toBe(true);
    expect(SAFE_SKILL_NAME_RE.test('../etc')).toBe(false);
    expect(SAFE_SKILL_NAME_RE.test('a/b')).toBe(false);
    expect(SAFE_SKILL_NAME_RE.test('.hidden')).toBe(false);
    expect(SAFE_SKILL_NAME_RE.test('')).toBe(false);
  });
});

describe('readSkillMdForPrompt (#890)', () => {
  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'skill-frontmatter-'));
  });

  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it('reads the invoked skill SKILL.md', () => {
    const dir = path.join(tmp, 'tessl__brief');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'SKILL.md'), '---\nkey: v\n---\n');
    expect(readSkillMdForPrompt('Skill(skill: "tessl__brief")', tmp)).toContain(
      'key: v',
    );
  });

  it('returns undefined with no invocation, unsafe name, or missing skill', () => {
    expect(readSkillMdForPrompt('no skill here', tmp)).toBeUndefined();
    expect(readSkillMdForPrompt('Skill(skill: "../escape")', tmp)).toBe(
      undefined,
    );
    expect(readSkillMdForPrompt('Skill(skill: "tessl__absent")', tmp)).toBe(
      undefined,
    );
  });

  it('returns undefined when the skill dir exists but has no SKILL.md', () => {
    // ENOENT on the file, not the dir — must fall back, not throw.
    fs.mkdirSync(path.join(tmp, 'tessl__empty'), { recursive: true });
    expect(readSkillMdForPrompt('Skill(skill: "tessl__empty")', tmp)).toBe(
      undefined,
    );
  });

  it('returns undefined when the skill path is a file, not a dir (ENOTDIR)', () => {
    fs.writeFileSync(path.join(tmp, 'tessl__file'), 'not a directory');
    expect(readSkillMdForPrompt('Skill(skill: "tessl__file")', tmp)).toBe(
      undefined,
    );
  });

  it('refuses a symlinked skill dir pointing outside the mount', () => {
    // A lexical `path.resolve` containment check passes here — the
    // joined path still starts with the mount — while `readFileSync`
    // follows the link and reads the outside file. Only realpath
    // containment refuses it.
    const outside = path.join(tmp, 'outside');
    fs.mkdirSync(outside, { recursive: true });
    fs.writeFileSync(
      path.join(outside, 'SKILL.md'),
      '---\nprecheck_timeout_ms: 999\n---\n',
    );
    const mount = path.join(tmp, 'mount');
    fs.mkdirSync(mount, { recursive: true });
    fs.symlinkSync(outside, path.join(mount, 'tessl__evil'), 'dir');
    expect(readSkillMdForPrompt('Skill(skill: "tessl__evil")', mount)).toBe(
      undefined,
    );
  });

  it('refuses a symlinked SKILL.md pointing outside the mount', () => {
    const outside = path.join(tmp, 'outside2');
    fs.mkdirSync(outside, { recursive: true });
    const secret = path.join(outside, 'SKILL.md');
    fs.writeFileSync(secret, '---\nprecheck_timeout_ms: 999\n---\n');
    const mount = path.join(tmp, 'mount2');
    const skillDir = path.join(mount, 'tessl__evil2');
    fs.mkdirSync(skillDir, { recursive: true });
    fs.symlinkSync(secret, path.join(skillDir, 'SKILL.md'));
    expect(readSkillMdForPrompt('Skill(skill: "tessl__evil2")', mount)).toBe(
      undefined,
    );
  });

  it('still reads a normal skill when the mount root itself is a symlink', () => {
    // The root is realpathed too, so a symlinked mount (a legitimate
    // deployment shape) resolves to a matching prefix rather than
    // failing every lookup.
    const real = path.join(tmp, 'real-skills');
    const dir = path.join(real, 'tessl__ok');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'SKILL.md'), '---\nkey: v\n---\n');
    const linkedMount = path.join(tmp, 'linked-skills');
    fs.symlinkSync(real, linkedMount, 'dir');
    expect(
      readSkillMdForPrompt('Skill(skill: "tessl__ok")', linkedMount),
    ).toContain('key: v');
  });
});
