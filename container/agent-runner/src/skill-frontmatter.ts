/**
 * Shared SKILL.md frontmatter access for the runner's per-skill
 * overrides (#890).
 *
 * Three overrides now resolve the same way — `drain_timeout_ms`
 * (#589, `hard-exit-watchdog.ts`), `requires_delivery` (#689,
 * `delivery-requirement.ts`), and `precheck_timeout_ms` (#890,
 * `precheck-timeout.ts`). Each was written as its own copy of the
 * same two steps: walk the leading YAML block for one scalar, and
 * resolve the invoked skill's SKILL.md off the prompt. This module
 * owns both steps once so a fourth override is a validator, not a
 * fourth transcription of the walk.
 *
 * The supported frontmatter shape stays deliberately narrow — top-level
 * scalar `key: value` lines, optional surrounding quotes, an inline
 * `# ...` comment stripped from unquoted values. Lists, nested
 * mappings, and multi-line strings are not supported: every override
 * is a scalar, so pulling js-yaml just to read one would be
 * unjustified weight. Same shape and same justification as the
 * host-side `parseSkillFrontmatter` in `src/cadence-registry.ts`.
 */

import * as fs from 'fs';
import * as path from 'path';

/**
 * Extract the first `Skill(skill: "...")` invocation name from a
 * prompt. Mirrors the host-side `parseTaskSkill` in
 * `src/task-scheduler.ts` so per-skill overrides resolve the same
 * way in both layers. The shape is fully enumerable — the literal
 * SDK skill-invocation syntax the orchestrator prepends — so a regex
 * is appropriate per `coding-policy: script-delegation`.
 */
export function parseSkillNameFromPrompt(prompt: string): string | undefined {
  const match = prompt.match(/Skill\(\s*skill:\s*["']([^"']+)["']/);
  return match?.[1];
}

/**
 * Skill names are directory names under the container's skills
 * mount. Allow ASCII letters, digits, `_`, `-`, and the `__`
 * namespace separator the tile installer uses (`tessl__<name>`).
 * Rejecting anything else — slashes, dots, leading hyphens, NUL,
 * empty — defends against a prompt that smuggles `..` or an
 * absolute path into a `Skill(skill: "...")` invocation and tries
 * to walk the override resolver into reading a SKILL.md outside the
 * mount. The character class is deliberately tighter than
 * "anything `path.join` would accept" so an attacker can't slip a
 * dotted segment past the regex.
 */
export const SAFE_SKILL_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

/**
 * Read the raw scalar for `key` from a SKILL.md's leading YAML
 * frontmatter. Returns the value with surrounding quotes and any
 * inline `# ...` comment stripped, or `undefined` when the content
 * has no leading `---` … `---` block, the block is unterminated, or
 * the key is absent.
 *
 * Returns the raw string rather than a typed value — each override
 * owns its own validation (a positive integer, a capped positive
 * integer, a strict boolean), and a shared coercion would have to
 * guess which.
 */
export function readFrontmatterScalar(
  content: string,
  key: string,
): string | undefined {
  // Trim a single leading BOM (U+FEFF, written as the Unicode escape
  // rather than a literal so eslint's no-irregular-whitespace rule
  // doesn't trip on the source character) defensively — some editors
  // emit it on save. Matches the host-side `parseSkillFrontmatter`.
  const stripped = content.replace(/^\uFEFF/, '');
  if (!stripped.startsWith('---\n') && !stripped.startsWith('---\r\n')) {
    return undefined;
  }
  const afterOpen = stripped.replace(/^---\r?\n/, '');
  const closeIdx = afterOpen.search(/^---\s*$/m);
  if (closeIdx < 0) return undefined;
  const body = afterOpen.slice(0, closeIdx);
  for (const rawLine of body.split(/\r?\n/)) {
    const line = rawLine.replace(/\s+$/, '');
    if (!line.trim() || line.trim().startsWith('#')) continue;
    const colonIdx = line.indexOf(':');
    if (colonIdx <= 0) continue;
    if (line.slice(0, colonIdx).trim() !== key) continue;
    let value = line.slice(colonIdx + 1).trim();
    // Strip an inline `# ...` comment from an unquoted value before
    // the caller validates it. Without this, `drain_timeout_ms:
    // 180000 # 3 minutes` would fail a digit check and silently fall
    // back to the default — exactly the shape tile authors naturally
    // write when annotating a non-obvious value. Quoted values are
    // left alone so a `#` inside quotes survives.
    if (!value.startsWith('"') && !value.startsWith("'")) {
      const inlineCommentIdx = value.search(/\s+#/);
      if (inlineCommentIdx >= 0) {
        value = value.slice(0, inlineCommentIdx).trimEnd();
      }
    }
    if (
      value.length >= 2 &&
      ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'")))
    ) {
      value = value.slice(1, -1);
    }
    return value;
  }
  return undefined;
}

/**
 * Read the SKILL.md of the skill invoked by `prompt`, from under
 * `skillsDir`. Returns `undefined` on any miss — no `Skill(skill:
 * "...")` invocation in the prompt, a skill name that fails
 * `SAFE_SKILL_NAME_RE`, a resolved path that escapes `skillsDir`, or
 * a skill that isn't installed.
 *
 * Filesystem misses (`ENOENT` / `ENOTDIR`) resolve to `undefined`
 * silently — a missing SKILL.md means "no declaration, use the
 * default", not "abort the run". Other I/O errors (permission, a
 * malformed path that survives the regex) are unexpected and
 * propagate so the operator can diagnose; the runner's existing
 * error-handling surface writes them to the SDK result diagnostic
 * channel rather than silently disabling the override.
 */
export function readSkillMdForPrompt(
  prompt: string,
  skillsDir: string,
): string | undefined {
  const skillName = parseSkillNameFromPrompt(prompt);
  if (!skillName) return undefined;
  if (!SAFE_SKILL_NAME_RE.test(skillName)) return undefined;
  const skillPath = path.join(skillsDir, skillName, 'SKILL.md');
  // Defence-in-depth: even with the safe-name regex above, verify the
  // path actually read is still under `skillsDir`.
  //
  // Resolved with `realpathSync`, not `path.resolve`: the latter is
  // purely lexical, so a symlinked skill directory or SKILL.md passes
  // a `startsWith` check while `readFileSync` follows the link and
  // reads outside the mount — the containment this comment claims
  // would not have held. Matches the `send_file` handler, which
  // realpaths for the same reason (a symlink planted in a
  // container-writable folder is refused, not followed).
  let resolvedRoot: string;
  let resolvedSkill: string;
  try {
    resolvedRoot = fs.realpathSync(skillsDir) + path.sep;
    resolvedSkill = fs.realpathSync(skillPath);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT' || code === 'ENOTDIR') return undefined;
    throw err;
  }
  if (!resolvedSkill.startsWith(resolvedRoot)) return undefined;
  try {
    // Read the realpath, so the containment check above and the read
    // resolve to the same file.
    return fs.readFileSync(resolvedSkill, 'utf-8');
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT' || code === 'ENOTDIR') return undefined;
    throw err;
  }
}
