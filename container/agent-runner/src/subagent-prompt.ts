/**
 * Pure helpers for assembling the subagent prompt. Subagents don't
 * inherit `settingSources` from the parent — they only see what's in
 * their explicit `prompt` + `skills` array — so the agent-runner has
 * to enumerate every rule/behavior file by hand and read it in.
 *
 * The host CLAUDE.md is a thin trust-tier pointer post-#153 that
 * `@import`s its real targets; this loader does NOT resolve `@import`,
 * so the targets must be enumerated directly. Branching on `isMain`
 * is required because main's thin CLAUDE.md @-imports a different
 * rules path (project-root `.tessl/RULES.md`) and `groups/main/ADMIN.md`
 * for the admin runbook — these are not in the standard chain and
 * would silently disappear from main subagents without an explicit
 * branch (caught in PR #164 review).
 */

/**
 * Decide whether to attach the general-purpose subagent definitions for
 * this spawn. The definitions ship a large cached prompt — every
 * installed skill's name plus the full rule/behavior chain — as agent
 * context on session creation, so it is re-created on every cold cache.
 *
 * Maintenance-session spawns (cadence tasks: heartbeat, composio-fetch,
 * morning-brief, nightly-*) provably never call `Task`/`TeamCreate` —
 * 0 of 7869 production maintenance runs over the audit window spawned a
 * subagent — yet they fire on short cadences (e.g. every 30 min) that
 * always outlast the prompt-cache TTL, so each wake pays full
 * cache-creation for definitions it never uses. Skipping the surface for
 * maintenance removes the dominant cold-cache cost of those wakes.
 *
 * Default: include for default/user-facing sessions, skip for
 * maintenance. The `MAINTENANCE_LOAD_SUBAGENTS=1` escape hatch forces
 * inclusion if a future maintenance task genuinely needs to fan out to a
 * subagent.
 */
export function shouldIncludeSubagentDefinitions(
  isMaintenanceSession: boolean,
  env: Record<string, string | undefined> = {},
): boolean {
  if (!isMaintenanceSession) return true;
  return env.MAINTENANCE_LOAD_SUBAGENTS === '1';
}

export interface SubagentRuleFilePathsInput {
  isMain: boolean;
  soulMdPath: string;
  formattingMdPath: string;
}

export function buildSubagentRuleFilePaths(
  input: SubagentRuleFilePathsInput,
): string[] {
  const files = [
    input.soulMdPath,
    input.formattingMdPath,
    '/workspace/group/MEMORY.md',
    '/workspace/group/.tessl/RULES.md',
  ];
  if (input.isMain) {
    files.push(
      '/workspace/project/.tessl/RULES.md',
      '/workspace/project/groups/main/ADMIN.md',
    );
  }
  return files;
}
