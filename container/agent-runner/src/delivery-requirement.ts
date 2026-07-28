/**
 * #689 — per-skill "requires delivery" declaration.
 *
 * A maintenance skill that ALWAYS delivers user-facing content
 * (morning-brief composes the daily brief and pins it to chat on every
 * run) declares `requires_delivery: true` in its SKILL.md frontmatter.
 * When such a run's `runQuery` drains WITHOUT delivering anything — the
 * final compose-and-send turn was cut, or the SDK loop ended before
 * `send_message` fired — the agent-runner stamps the terminal marker it
 * emits with `noDelivery: true`.
 *
 * The host (`src/container-runner.ts`) keeps that marker's `success`
 * status so `scheduleClose` still drains the maintenance slot promptly
 * (no #461 wedge), but honours `noDelivery` to resolve the run as
 * `killed` — incomplete / retriable — so the recovery path redelivers
 * instead of recording a silent success. That silent success is the
 * #589/#682 lineage #689 reopened: a brief that never reached chat was
 * recorded `status=success` because a synthesized terminal marker
 * flipped the host's `hadTerminalResult`, defeating the #682
 * killed-classification.
 *
 * Skills WITHOUT the declaration are unaffected — `noDelivery` is never
 * stamped, so a legitimately-silent maintenance task (heartbeat, a
 * precheck-gated cleanup that sends nothing) keeps its `success`. The
 * declaration is the only thing that distinguishes "drained without
 * delivering, and that's a failure" from "drained without delivering,
 * and that's fine"; the runner can't infer a skill's delivery profile
 * any other way.
 *
 * Resolution shares `readSkillMdForPrompt` with the runner's other
 * per-skill overrides (`skill-frontmatter.ts`) — same safe-name guard,
 * same ENOENT/ENOTDIR fallback. The frontmatter scalar is fully
 * enumerable, so a parser is appropriate per `coding-policy:
 * script-delegation`.
 */

import {
  readFrontmatterScalar,
  readSkillMdForPrompt,
} from './skill-frontmatter.js';

/**
 * Parse the `requires_delivery` boolean from a SKILL.md's leading YAML
 * frontmatter. Returns `true` only when the key is present AND its value
 * is the YAML boolean `true` (bare or quoted, case-insensitive). Any
 * other shape — key absent, no frontmatter, `false`, a non-boolean
 * scalar, an unterminated block — returns `false`. Deliberately narrow,
 * mirroring `parseDrainTimeoutMsFromFrontmatter`: pulling js-yaml just
 * for one scalar would be unjustified weight.
 */
export function parseRequiresDeliveryFromFrontmatter(content: string): boolean {
  const value = readFrontmatterScalar(content, 'requires_delivery');
  if (value === undefined) return false;
  return value.trim().toLowerCase() === 'true';
}

/**
 * Decide whether a terminal `success` marker should be stamped
 * `noDelivery`. True only when the invoked skill requires delivery AND
 * the run delivered no user-facing content. Centralises the predicate
 * the runner applies at every terminal-marker site (the silent-stop
 * synthesis, the SDK-result success marker, the post-query
 * session-update) so the no-delivery decision is identical across all
 * of them and unit-testable without spinning the SDK iterator.
 */
export function shouldStampNoDelivery(
  requiresDelivery: boolean,
  deliveredUserFacingContent: boolean,
): boolean {
  return requiresDelivery && !deliveredUserFacingContent;
}

/**
 * Resolve whether the skill invoked by `prompt` requires delivery.
 * Reads the prompt's `Skill(skill: "...")` invocation, looks up the
 * named skill's SKILL.md under `skillsDir`, and returns its
 * `requires_delivery` frontmatter flag. Falls back to `false` on any
 * miss — no skill invocation, unsafe skill name, skill not installed,
 * missing or malformed frontmatter.
 *
 * Filesystem misses (`ENOENT` / `ENOTDIR`) fall back to `false` — a
 * missing SKILL.md means "no declaration", not "abort the run". Other
 * I/O errors (permission, etc.) are unexpected and propagate, matching
 * `resolveDrainTimeoutMs`.
 */
export function resolveRequiresDelivery(
  prompt: string,
  skillsDir: string,
): boolean {
  const content = readSkillMdForPrompt(prompt, skillsDir);
  if (content === undefined) return false;
  return parseRequiresDeliveryFromFrontmatter(content);
}
