/**
 * Cadence registry — Phase 2a of #305.
 *
 * Reads `cadence:` / `priority:` declarations from per-skill SKILL.md
 * frontmatter at container spawn and idempotently materialises them as
 * `scheduled_tasks` rows with `source = 'cadence-registry'`. The
 * existing task-scheduler picks them up and fires them at the declared
 * cron time exactly like rows produced by the `schedule-task` IPC.
 *
 * Phase 2a scope (this file):
 *   - Frontmatter parser, dependency-free, supports the scalar shape
 *     SKILL.mds actually use (top-level `key: value` only).
 *   - Cadence-declaration validator: cron string parseable by
 *     cron-parser (TZ qualifier `(TZ=local|<IANA>)` stripped before
 *     validation; applied at fire time, not here), `priority:`
 *     integer if present.
 *   - Idempotent rebuild: walks `<skillsDir>/<skill>/SKILL.md`,
 *     validates declarations, DELETEs rows where
 *     `source = 'cadence-registry' AND group_folder = ?`, then
 *     INSERTs one row per valid declaration. Owner-scheduled tasks
 *     (`source = 'schedule-task'`) are untouched across respawns.
 *
 * Out of scope for Phase 2a (deferred to Phase 2c):
 *   - `contributes_to:` / `section_order:` / `depends_on:` aggregator
 *     pattern. The frontmatter parser ignores those fields if present;
 *     the validator does not enforce mixed-axis rejection until 2c.
 *
 * Coexistence with the existing `schedule-task` IPC path:
 *   - Today, admin's `heartbeat` / `morning-brief` / `nightly-
 *     housekeeping` / `weekly-housekeeping` shells self-register via
 *     the `schedule-task` IPC (rows land with `source =
 *     'schedule-task'`). Until those shells migrate to declared
 *     frontmatter (Phase 2b, separate PR on `nanoclaw-admin`), the
 *     cadence-registry stays empty for them — no double-fires.
 */
import * as fs from 'fs';
import * as path from 'path';
import { CronExpressionParser } from 'cron-parser';
import type Database from 'better-sqlite3';
import { logger } from './logger.js';

/**
 * Frontmatter fields recognised by the cadence registry. Only the two
 * named here drive Phase 2a behaviour; everything else in the
 * frontmatter (`name`, `description`, `user-invocable`, ...) is left
 * untouched.
 */
export interface CadenceDeclaration {
  /**
   * Cron expression, optionally suffixed with a TZ qualifier:
   *   `0 7 * * *`              — UTC by default
   *   `0 7 * * * (TZ=local)`   — owner's resident IANA zone (resolved
   *                              from `tz_state.current_tz` at fire time)
   *   `0 7 * * * (TZ=America/Los_Angeles)` — pinned IANA zone
   */
  cadence: string;
  /**
   * Ordering hint within a single fire. Default 0. Lock-acquire skills
   * declare -100, lock-confirm +100 (see follow-me-two-phase-lock.md).
   * Phase 2a stores priority but does NOT yet expose ordering to the
   * dispatcher — the existing task-scheduler fires per-task, not in
   * batches. Phase 2c will use this when batching contributors for an
   * aggregator's fire.
   */
  priority: number;
}

interface ParsedSkill {
  name: string;
  skillPath: string;
  frontmatter: Record<string, string | number>;
}

/**
 * Parse a SKILL.md's leading YAML frontmatter into a flat key→value
 * map. The supported shape is intentionally narrow: top-level scalar
 * `key: value` lines, optionally surrounded by single or double
 * quotes; bare integers are coerced to JS numbers. Lists, nested
 * mappings, and multi-line strings are not supported — the cadence-
 * registry's two fields are both scalars, so pulling js-yaml just to
 * read them would be unjustified weight.
 *
 * Returns `{}` for content without a leading `---` … `---` block, for
 * a malformed block, or for a block with no recognisable scalars. The
 * registry treats an empty result as "skill not registered" — the
 * skill is silently skipped, never errored.
 */
export function parseSkillFrontmatter(
  content: string,
): Record<string, string | number> {
  // Frontmatter must start at the very beginning of the file. Trim a
  // single leading BOM (U+FEFF, written as the Unicode escape rather
  // than a literal so eslint's no-irregular-whitespace rule doesn't
  // trip on the source character) but otherwise reject content that
  // doesn't start with `---\n`.
  const stripped = content.replace(/^\uFEFF/, '');
  if (!stripped.startsWith('---\n') && !stripped.startsWith('---\r\n')) {
    return {};
  }
  const afterOpen = stripped.replace(/^---\r?\n/, '');
  const closeIdx = afterOpen.search(/^---\s*$/m);
  if (closeIdx < 0) return {};
  const body = afterOpen.slice(0, closeIdx);
  const out: Record<string, string | number> = {};
  for (const rawLine of body.split(/\r?\n/)) {
    const line = rawLine.replace(/\s+$/, '');
    if (!line.trim() || line.trim().startsWith('#')) continue;
    const colonIdx = line.indexOf(':');
    if (colonIdx <= 0) continue;
    const key = line.slice(0, colonIdx).trim();
    let value = line.slice(colonIdx + 1).trim();
    if (!key) continue;
    // Strip an inline `# ...` comment from a bare (unquoted) value
    // before quote-stripping, matching standard YAML semantics. Skip
    // this for quoted values so a `#` inside a quoted cron is left
    // alone (cron itself doesn't use `#`, but the rule is "don't peek
    // inside quotes" — keeps the parser predictable). The leading
    // whitespace before `#` is required so a literal `key:#hash` (no
    // space) stays as the literal value rather than being dropped.
    if (!value.startsWith('"') && !value.startsWith("'")) {
      const inlineCommentIdx = value.search(/\s+#/);
      if (inlineCommentIdx >= 0) {
        value = value.slice(0, inlineCommentIdx).trimEnd();
      }
    }
    // Strip a single layer of surrounding quotes. We don't try to be
    // YAML-perfect — escape sequences inside quoted strings aren't
    // interpreted, which is fine for cron expressions.
    if (
      value.length >= 2 &&
      ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'")))
    ) {
      value = value.slice(1, -1);
    }
    // Numeric coercion for bare integers (priority: 0 / -100 / +100).
    if (/^[+-]?\d+$/.test(value)) {
      out[key] = Number.parseInt(value, 10);
    } else {
      out[key] = value;
    }
  }
  return out;
}

/**
 * Walk an installed skills tree and return one record per SKILL.md
 * found. Returns `[]` if the directory doesn't exist (cold start, no
 * tiles mounted yet). Subdirectories without a SKILL.md are silently
 * skipped — they're either non-skill artefacts or partially-installed
 * tiles, neither of which this layer should error on.
 */
export function walkInstalledSkills(skillsDir: string): ParsedSkill[] {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(skillsDir, { withFileTypes: true });
  } catch (err: unknown) {
    if (
      err instanceof Error &&
      (err as NodeJS.ErrnoException).code === 'ENOENT'
    ) {
      return [];
    }
    throw err;
  }
  const out: ParsedSkill[] = [];
  for (const dirent of entries) {
    if (!dirent.isDirectory()) continue;
    const skillName = dirent.name;
    const skillPath = path.join(skillsDir, skillName, 'SKILL.md');
    let content: string;
    try {
      content = fs.readFileSync(skillPath, 'utf-8');
    } catch (err: unknown) {
      if (
        err instanceof Error &&
        (err as NodeJS.ErrnoException).code === 'ENOENT'
      ) {
        continue; // directory without a SKILL.md isn't a skill
      }
      throw err;
    }
    out.push({
      name: skillName,
      skillPath,
      frontmatter: parseSkillFrontmatter(content),
    });
  }
  return out;
}

export interface CadenceValidationOk {
  ok: true;
  declaration: CadenceDeclaration | null; // null = no declaration in frontmatter
}
export interface CadenceValidationErr {
  ok: false;
  errors: string[];
}
export type CadenceValidationResult =
  | CadenceValidationOk
  | CadenceValidationErr;

/**
 * Strip an optional `(TZ=local|<IANA>)` suffix from a cadence string.
 * Returns `{ cron, tz }` where `tz` is `null` for an unqualified
 * expression, the literal string `'local'` for `(TZ=local)`, or an
 * IANA zone name otherwise. The scheduler resolves `'local'` against
 * `tz_state.current_tz` at fire time; pinned IANA names pass through
 * to the cron parser's `tz` option directly.
 */
export function splitCadenceTz(cadence: string): {
  cron: string;
  tz: string | null;
} {
  const m = cadence.match(/^(.*?)\s*\(TZ=([^)]+)\)\s*$/);
  if (!m) return { cron: cadence.trim(), tz: null };
  return { cron: m[1].trim(), tz: m[2].trim() };
}

/**
 * Validate a SKILL.md frontmatter against the Phase 2a cadence-registry
 * contract. Returns:
 *   { ok: true, declaration: null }       — no `cadence:` field; skill
 *                                           simply doesn't register.
 *   { ok: true, declaration: {...} }      — valid declaration extracted.
 *   { ok: false, errors: [...] }          — declaration present but
 *                                           malformed; skill skipped,
 *                                           errors logged at WARN.
 */
export function validateCadenceDeclaration(
  frontmatter: Record<string, string | number>,
  skillName: string,
): CadenceValidationResult {
  const rawCadence = frontmatter.cadence;
  if (rawCadence === undefined) {
    return { ok: true, declaration: null };
  }
  const errors: string[] = [];
  if (typeof rawCadence !== 'string' || rawCadence.trim() === '') {
    errors.push(
      `${skillName}: 'cadence:' must be a non-empty string, got ${JSON.stringify(rawCadence)}`,
    );
  }
  if (typeof rawCadence === 'string' && rawCadence.trim() !== '') {
    const { cron, tz } = splitCadenceTz(rawCadence);
    try {
      // The TZ qualifier is just a label here — cron-parser doesn't
      // accept arbitrary label suffixes, only `tz` as a parser option.
      // For validation we only need to confirm the bare cron parses;
      // the tz string itself is plumbed onto schedule_timezone where
      // the scheduler validates it on fire.
      CronExpressionParser.parse(cron);
    } catch (err: unknown) {
      const reason = err instanceof Error ? err.message : String(err);
      errors.push(
        `${skillName}: 'cadence:' is not a valid cron expression — ${reason}`,
      );
    }
    if (tz !== null && tz !== 'local') {
      // Pinned IANA name. We don't have a host-side IANA validator
      // beyond cron-parser's own (which only checks at parse-with-tz
      // time). Defer strict IANA validation to fire time; here we
      // just confirm the qualifier isn't empty.
      if (tz === '') {
        errors.push(`${skillName}: '(TZ=)' qualifier is empty`);
      }
    }
  }
  let priority = 0;
  const rawPriority = frontmatter.priority;
  if (rawPriority !== undefined) {
    if (typeof rawPriority !== 'number' || !Number.isInteger(rawPriority)) {
      errors.push(
        `${skillName}: 'priority:' must be an integer, got ${JSON.stringify(rawPriority)}`,
      );
    } else {
      priority = rawPriority;
    }
  }
  if (errors.length > 0) {
    return { ok: false, errors };
  }
  return {
    ok: true,
    declaration: {
      cadence: typeof rawCadence === 'string' ? rawCadence : '',
      priority,
    },
  };
}

export interface CadenceRegistryRebuildResult {
  /** Cadence-registry rows DELETEd because the skill was uninstalled or dropped its cadence:. */
  deleted: number;
  /** Cadence-registry rows newly INSERTed (skill declared a cadence and didn't have a prior row). */
  inserted: number;
  /** Cadence-registry rows UPDATEd because the cadence string or prompt changed. `next_run` recomputes; `session_id` / `last_run` / `last_result` are preserved. */
  updated: number;
  /** Cadence-registry rows left alone because the declaration is identical to the prior spawn — `next_run` and per-fire state survive intact. */
  preserved: number;
  /** Number of skills walked (whether or not they declared a cadence). */
  walked: number;
  /** Skills with malformed declarations or computeNextRun failures; logged at WARN, skipped from rebuild. */
  errors: string[];
}

export interface CadenceRegistryDeps {
  db: Database.Database;
  groupFolder: string;
  chatJid: string;
  /**
   * Trust-boundary provenance for the agent-runner's untrusted-input
   * wrapping. Pass through whatever the spawn context already
   * computed — the cadence-registry doesn't infer trust on its own.
   */
  createdByRole: 'owner' | 'main_agent' | 'trusted_agent' | 'untrusted_agent';
  /** Host-side path to the per-group skills mount (e.g. `<sessionsDir>/skills`). */
  skillsDir: string;
  /**
   * Returns the next-fire timestamp (UTC ISO string) for a bare cron
   * expression with an optional IANA TZ. Injected so unit tests can
   * pin `next_run` without monkey-patching cron-parser globally.
   */
  computeNextRun: (cron: string, tz: string | null) => string;
  /** Source of `created_at` timestamps. Injected for testability. */
  now: () => Date;
}

/**
 * Idempotent rebuild of the cadence-registry rows for a single group.
 * Safe to call on every container spawn — a second invocation with
 * identical SKILL.md frontmatter produces identical rows (modulo
 * `next_run` and `created_at`, which advance with wall-clock time).
 *
 * Transactional: the DELETE+INSERT runs inside a single
 * `BEGIN IMMEDIATE` so a concurrent reader (the task-scheduler firing
 * a tick) never observes a torn state with the old rows gone but the
 * new rows not yet inserted.
 */
export function rebuildCadenceRegistry(
  deps: CadenceRegistryDeps,
): CadenceRegistryRebuildResult {
  const skills = walkInstalledSkills(deps.skillsDir);
  const errors: string[] = [];
  const valid: Array<{ skillName: string; declaration: CadenceDeclaration }> =
    [];
  for (const skill of skills) {
    const result = validateCadenceDeclaration(skill.frontmatter, skill.name);
    if (!result.ok) {
      errors.push(...result.errors);
      logger.warn(
        {
          skill: skill.name,
          skillPath: skill.skillPath,
          errors: result.errors,
        },
        'cadence-registry: invalid declaration, skipping',
      );
      continue;
    }
    if (result.declaration !== null) {
      valid.push({ skillName: skill.name, declaration: result.declaration });
    }
  }

  // Pre-compute desired-state rows, including next_run. Each row's
  // computeNextRun is wrapped individually so a single bad pinned-IANA
  // zone (`(TZ=Not/A/Real/Zone)`) reports against just that skill
  // instead of throwing inside the rebuild transaction and aborting
  // every cadence in the group. Errors collected here join the
  // validator's errors at return time so callers see one combined
  // surface.
  interface DesiredRow {
    taskId: string;
    skillName: string;
    prompt: string;
    cron: string;
    tz: string | null;
    nextRun: string;
  }
  const desired: DesiredRow[] = [];
  for (const { skillName, declaration } of valid) {
    const taskId = `cadence-registry::${deps.groupFolder}::${skillName}`;
    // skillName is the directory name as it appears under skillsDst.
    // Tile skills land as `tessl__<name>` (the form `Skill(skill:
    // ...)` expects); built-in and staging skills land as `<name>`.
    // We pass the directory name through unchanged, so an agent's
    // `Skill(skill: "<this string>")` invocation matches the
    // installed surface form regardless of source.
    const prompt = `Skill(skill: "${skillName}")`;
    const { cron, tz } = splitCadenceTz(declaration.cadence);
    let nextRun: string;
    try {
      nextRun = deps.computeNextRun(cron, tz);
    } catch (err: unknown) {
      const reason = err instanceof Error ? err.message : String(err);
      const msg = `${skillName}: computeNextRun failed for cadence ${JSON.stringify(declaration.cadence)} — ${reason}`;
      errors.push(msg);
      logger.warn(
        { skill: skillName, cadence: declaration.cadence, err },
        'cadence-registry: computeNextRun failed, skipping',
      );
      continue;
    }
    desired.push({ taskId, skillName, prompt, cron, tz, nextRun });
  }

  const tx = deps.db.transaction(() => {
    // Existing cadence-registry rows for this group, by id. We use
    // these to (a) preserve `next_run`, `session_id`, `last_run`,
    // `last_result` on rows whose declared shape hasn't changed (so a
    // spawn happening between two cron fires can't push the due-time
    // forward and miss the firing window, and `session_id` reuse for
    // #336 cache-stability survives respawns), and (b) compute the
    // orphan set (rows whose skill was uninstalled or stopped
    // declaring a cadence).
    interface ExistingRow {
      id: string;
      schedule_value: string;
      schedule_timezone: string | null;
      prompt: string;
      next_run: string | null;
    }
    const existing = new Map<string, ExistingRow>();
    const existingRows = deps.db
      .prepare(
        `SELECT id, schedule_value, schedule_timezone, prompt, next_run
         FROM scheduled_tasks
         WHERE source = 'cadence-registry' AND group_folder = ?`,
      )
      .all(deps.groupFolder) as ExistingRow[];
    for (const row of existingRows) existing.set(row.id, row);

    const insertStmt = deps.db.prepare(`
      INSERT INTO scheduled_tasks (
        id, group_folder, chat_jid, prompt, script,
        schedule_type, schedule_value, schedule_timezone,
        context_mode, next_run, status, created_at,
        created_by_role, continuation_cycle_id, source
      ) VALUES (?, ?, ?, ?, NULL, 'cron', ?, ?, 'isolated', ?, 'active', ?, ?, NULL, 'cadence-registry')
    `);
    // Shape-change UPDATE: preserve session_id / last_run / last_result
    // (they're per-fire history, not declaration), recompute next_run
    // because the cron expression changed, refresh chat_jid /
    // created_by_role / created_at to match the current spawn.
    const updateChangedStmt = deps.db.prepare(`
      UPDATE scheduled_tasks
         SET prompt = ?, schedule_value = ?, schedule_timezone = ?,
             next_run = ?, chat_jid = ?, created_by_role = ?,
             created_at = ?
       WHERE id = ?
    `);
    const createdAt = deps.now().toISOString();
    let inserted = 0;
    let updated = 0;
    let preserved = 0;
    for (const row of desired) {
      const prior = existing.get(row.taskId);
      if (!prior) {
        // New skill registration — INSERT.
        insertStmt.run(
          row.taskId,
          deps.groupFolder,
          deps.chatJid,
          row.prompt,
          row.cron,
          row.tz,
          row.nextRun,
          createdAt,
          deps.createdByRole,
        );
        inserted++;
        continue;
      }
      const shapeChanged =
        prior.schedule_value !== row.cron ||
        prior.schedule_timezone !== row.tz ||
        prior.prompt !== row.prompt;
      if (shapeChanged) {
        // Cadence declaration moved — recompute next_run, refresh the
        // declarative columns, but leave session_id / last_run /
        // last_result intact (they're row-history, not row-shape).
        updateChangedStmt.run(
          row.prompt,
          row.cron,
          row.tz,
          row.nextRun,
          deps.chatJid,
          deps.createdByRole,
          createdAt,
          row.taskId,
        );
        updated++;
      } else {
        // Identical declaration — leave the row alone entirely. This
        // is the path that matters for #305: an idempotent rebuild on
        // every spawn can't disturb in-flight `next_run`, in-progress
        // session reuse, or last-fire status.
        preserved++;
      }
    }

    // Orphan DELETE: any cadence-registry row for this group whose id
    // isn't in the desired set means the skill was uninstalled or
    // dropped its `cadence:` declaration since the last spawn.
    const desiredIds = new Set(desired.map((d) => d.taskId));
    let deleted = 0;
    for (const [id] of existing) {
      if (desiredIds.has(id)) continue;
      deps.db
        .prepare(
          `DELETE FROM scheduled_tasks WHERE id = ? AND source = 'cadence-registry'`,
        )
        .run(id);
      deleted++;
    }
    return { deleted, inserted, updated, preserved };
  });
  const { deleted, inserted, updated, preserved } = tx();
  return {
    deleted,
    inserted,
    updated,
    preserved,
    walked: skills.length,
    errors,
  };
}

/**
 * Default `computeNextRun` implementation backed by `cron-parser`.
 * For `tz === 'local'` we don't have access to `tz_state.current_tz`
 * here — the orchestrator's task-scheduler reapplies TZ on fire, so
 * passing UTC as the initial cursor is correct: the next fire will
 * recompute with the live current_tz, and any drift between the two
 * is at most one tick window. For pinned IANA names we honour the
 * declared zone immediately.
 */
export function defaultComputeNextRun(cron: string, tz: string | null): string {
  const opts = tz && tz !== 'local' ? { tz } : {};
  const next = CronExpressionParser.parse(cron, opts).next();
  return next.toDate().toISOString();
}
