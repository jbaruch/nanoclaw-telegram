/**
 * Mutating-tool taxonomy for the kill-auto-compaction `## Facts` writer.
 *
 * Design doc §3 (`docs/proposals/kill-auto-compaction.md`): the
 * checkpoint's "do NOT re-execute" list captures every tool call that
 * MUTATES state. Reads (Grep, Glob, Read, read-only Bash patterns) get
 * no entry — the agent re-reading post-reentry is fine and even
 * desirable.
 *
 * False positives here are noise; false negatives are the JCON failure
 * mode the epic exists to prevent. Bias toward over-classifying as
 * mutating.
 */

/**
 * Tool names that always mutate state. Anything in this set lands in
 * the `## Facts` "do NOT re-execute" list.
 *
 * Sourced from design doc §3. Add to this set when a new MCP tool that
 * mutates state ships — the alternative is the JCON-class incident
 * where reentry re-fires a state-mutating call because the parser
 * didn't recognise the tool name. There's a CI-check follow-up
 * (#124) that will eventually require every tool definition to declare
 * a `mutates` boolean explicitly; until then this hard-coded set is
 * the source of truth.
 */
export const ALWAYS_MUTATING_TOOLS: ReadonlySet<string> = new Set([
  // Core file/code mutation
  'Write',
  'Edit',
  'MultiEdit',
  'NotebookEdit',
  // Skill invocation — most skills mutate something (memory, state,
  // PRs, messages); the few that are pure-read still get listed
  // because re-firing is the wrong default.
  'Skill',
  // NanoClaw MCP tools that touch user-visible state
  'mcp__nanoclaw__send_message',
  'mcp__nanoclaw__send_file',
  'mcp__nanoclaw__react_to_message',
  'mcp__nanoclaw__pin_message',
  'mcp__nanoclaw__schedule_task',
  'mcp__nanoclaw__update_task',
  'mcp__nanoclaw__register_group',
  'mcp__nanoclaw__set_trusted',
  'mcp__nanoclaw__set_trigger',
  // SDK task primitives that spawn sub-agents (re-firing would
  // re-execute the sub-agent's effects)
  'Task',
  'TeamCreate',
  'TeamDelete',
]);

/**
 * Tool names that NEVER mutate state. Reads, queries, searches.
 *
 * Listed explicitly so the `classifyTool` function can short-circuit
 * to "not mutating" for the read-only side of the SDK's built-ins.
 * Anything not in either set is treated as mutating (safe default).
 */
export const ALWAYS_READ_TOOLS: ReadonlySet<string> = new Set([
  'Read',
  'Grep',
  'Glob',
  'WebFetch',
  'WebSearch',
  'TaskOutput',
  'TodoWrite', // arguably mutates the todo list, but the todo list is
  // ephemeral per-session state that doesn't survive nuke anyway, so
  // re-firing is harmless and listing it under "do NOT re-execute"
  // would just clutter the checkpoint.
  'ToolSearch',
]);

/**
 * Bash subcommand prefixes that are always read-only. Anything else
 * starting with `Bash:` is treated as mutating.
 *
 * Order doesn't matter here; matching is by `argv[0]` of the bash
 * command line. `git` is special-cased because the read-only git
 * subcommands (`status`, `log`, `diff`, `show`) are far more common
 * than the mutating ones in agent transcripts, but `git push`,
 * `git commit`, `git rebase`, etc. must classify as mutating.
 *
 * Extend per pattern observed in production. The cost of a false
 * negative (mutating bash command misclassified as read) is the JCON
 * failure mode; the cost of a false positive (read bash command
 * landing in the checkpoint) is a longer "do NOT re-execute" list
 * that the agent ignores anyway.
 */
export const BASH_READ_ONLY_COMMANDS: ReadonlySet<string> = new Set([
  'cat',
  'ls',
  'grep',
  'find',
  'head',
  'tail',
  'wc',
  'sort',
  'uniq',
  'pwd',
  'echo',
  'file',
  'stat',
  'date',
]);

/**
 * Read-only `git` subcommands. Used in tandem with BASH_READ_ONLY_COMMANDS:
 * if argv[0] is `git`, argv[1] must be in this set to classify as read.
 * Anything else (`git push`, `git commit`, `git rebase`, etc.) is
 * mutating.
 */
export const GIT_READ_ONLY_SUBCOMMANDS: ReadonlySet<string> = new Set([
  'status',
  'log',
  'diff',
  'show',
  'rev-parse',
  'config', // `git config --get` is read; `git config --set` writes,
  // but we err on the side of listing config touches in the
  // checkpoint because re-running a `--set` after reentry would
  // double-apply the config change.
  // Actually, treat `git config` as mutating to be safe — comment
  // above explains why. Removed from this set.
]);
// `git config` removed per the comment above — bias toward mutating
// classification.
(GIT_READ_ONLY_SUBCOMMANDS as Set<string>).delete('config');

/**
 * Classify a tool invocation as mutating or read-only.
 *
 * `toolName` is the SDK's tool name (e.g. `Write`, `Bash`,
 * `mcp__nanoclaw__send_message`). For Bash, `bashCommand` is the
 * literal command string the SDK emitted; the classifier extracts
 * argv[0] and (for `git`) argv[1] to decide.
 *
 * Returns `true` if mutating (belongs in the `## Facts` "do NOT
 * re-execute" list), `false` if read-only.
 *
 * Default for unknown tool names: `true` (mutating). Better to
 * over-list than to miss a state-mutating tool the parser doesn't
 * recognise.
 */
export function classifyTool(toolName: string, bashCommand?: string): boolean {
  if (ALWAYS_MUTATING_TOOLS.has(toolName)) return true;
  if (ALWAYS_READ_TOOLS.has(toolName)) return false;

  if (toolName === 'Bash') {
    if (!bashCommand) return true; // missing command — can't verify, classify mutating
    const argv = bashCommand.trim().split(/\s+/);
    const cmd = argv[0];
    if (!cmd) return true;
    if (cmd === 'git') {
      const sub = argv[1];
      return !sub || !GIT_READ_ONLY_SUBCOMMANDS.has(sub);
    }
    return !BASH_READ_ONLY_COMMANDS.has(cmd);
  }

  // Unknown tool — classify as mutating (safe default)
  return true;
}
