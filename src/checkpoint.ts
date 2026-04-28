import fs from 'fs';
import path from 'path';

import { classifyTool } from './mutating-tools.js';
import { parseSessionTranscript } from './jsonl-parser.js';
import { Thresholds } from './threshold.js';
import { logger } from './logger.js';

/**
 * Checkpoint writer for kill-auto-compaction (design doc §1, §2, §6,
 * `docs/proposals/kill-auto-compaction.md`).
 *
 * Writes the orchestrator-built `## Facts` section to
 * `<groupDir>/.checkpoints/default.md`. Rotates the prior file to
 * `previous.md` first (atomic mv, overwriting any existing
 * previous.md) so the live file is never read mid-rewrite by a
 * concurrent reentry skill.
 *
 * `previous.md` is forensics-only — reentry never reads it. Falling
 * back to it on a corrupt `default.md` would mean loading state from
 * two sessions ago, which is structurally worse than no list because
 * the "do NOT re-execute" entries miss every mutating call from the
 * just-nuked session.
 *
 * The agent-authored `## Reasoning` section (design doc §1) is NOT
 * written here — it's appended by the agent itself in response to the
 * Phase 4 system-reminder. This writer only owns the deterministic
 * half.
 */

export interface CheckpointInputs {
  /** Host-side absolute path to the group folder (the per-group
   *  writable mount). The checkpoints dir is `<groupDir>/.checkpoints`. */
  groupDir: string;
  /** Absolute path to the session's JSONL transcript. */
  jsonlPath: string;
  /** SDK session id at threshold-cross time. */
  sessionId: string;
  /** Threshold values that triggered this write — copied into the
   *  Facts section so the operator can correlate "checkpoint fired
   *  at usage X" to "thresholds were Y/Z". */
  thresholds: Thresholds;
  /** Most-recent assistant input_tokens count at threshold-cross. */
  usedTokens: number;
  /** Group name for log lines. */
  groupName: string;
  /** Pending Telegram/etc. message IDs awaiting reply. Optional —
   *  when the caller can't easily produce this list, an empty array
   *  is fine; the Facts section just renders a "none" entry. */
  pendingReplyIds?: string[];
}

const CHECKPOINTS_SUBDIR = '.checkpoints';
const LIVE_FILENAME = 'default.md';
const PREVIOUS_FILENAME = 'previous.md';

export function checkpointPaths(groupDir: string): {
  dir: string;
  live: string;
  previous: string;
} {
  const dir = path.join(groupDir, CHECKPOINTS_SUBDIR);
  return {
    dir,
    live: path.join(dir, LIVE_FILENAME),
    previous: path.join(dir, PREVIOUS_FILENAME),
  };
}

/**
 * Render the ## Facts section from the current session state.
 *
 * Pure function — no fs I/O, no side effects. Exposed separately
 * from `writeCheckpoint` so tests can assert on the rendered output
 * without staging a temp dir.
 */
export function renderFacts(
  inputs: CheckpointInputs,
  mutatingInvocations: Array<{
    name: string;
    inputSummary: string;
    completedAt: string | null;
    isError: boolean | null;
  }>,
): string {
  const lines: string[] = [];
  lines.push('# Session Checkpoint');
  lines.push('');
  lines.push(
    '_Written by the orchestrator at threshold-cross. The agent-authored ' +
      '`## Reasoning` section follows below; if absent, the agent did not ' +
      'reach the threshold-reminder before nuke (the documented degraded mode)._',
  );
  lines.push('');
  lines.push('## Facts');
  lines.push('');
  lines.push(`- **Group:** ${inputs.groupName}`);
  lines.push(`- **Session id:** \`${inputs.sessionId}\``);
  lines.push(
    `- **Tokens used at trigger:** ${inputs.usedTokens.toLocaleString()} ` +
      `/ ${inputs.thresholds.contextWindow.toLocaleString()} ` +
      `(warn ${inputs.thresholds.warn.toLocaleString()}, nuke ${inputs.thresholds.nuke.toLocaleString()})`,
  );
  lines.push(`- **Trigger time:** ${new Date().toISOString()}`);
  lines.push('');

  lines.push('### Pending replies');
  if (!inputs.pendingReplyIds || inputs.pendingReplyIds.length === 0) {
    lines.push('- _none_');
  } else {
    for (const id of inputs.pendingReplyIds) {
      lines.push(`- \`${id}\``);
    }
  }
  lines.push('');

  lines.push('### Do NOT re-execute');
  lines.push(
    '_Mutating tool calls observed in the just-nuked session. ' +
      'Reentry must NOT re-fire any of these._',
  );
  lines.push('');
  if (mutatingInvocations.length === 0) {
    lines.push('- _no mutating calls observed_');
  } else {
    for (const inv of mutatingInvocations) {
      const completed = inv.completedAt ?? '_in-flight at trigger_';
      const status =
        inv.isError === true ? ' (errored)' : inv.isError === null ? '' : '';
      lines.push(
        `- \`${inv.name}\` — ${inv.inputSummary} — completed ${completed}${status}`,
      );
    }
  }
  lines.push('');

  return lines.join('\n');
}

/**
 * Summarise a tool invocation's input for the Facts list. Picks the
 * most-load-bearing key per tool family rather than dumping the
 * entire input object (which can be megabytes for Write tool calls).
 */
export function summariseInput(
  toolName: string,
  input: Record<string, unknown>,
): string {
  if (toolName === 'Write' || toolName === 'Edit' || toolName === 'MultiEdit') {
    const file = input.file_path ?? input.path ?? '?';
    return `\`${file}\``;
  }
  if (toolName === 'Bash') {
    const cmd = String(input.command ?? '').slice(0, 120);
    return `\`${cmd}${cmd.length === 120 ? '…' : ''}\``;
  }
  if (toolName === 'Skill') {
    return `skill=\`${input.skill ?? '?'}\``;
  }
  if (toolName.startsWith('mcp__nanoclaw__send_message')) {
    const reply = input.reply_to ? ` reply_to=\`${input.reply_to}\`` : '';
    const len = String(input.message ?? '').length;
    return `${len} chars${reply}`;
  }
  if (toolName.startsWith('mcp__nanoclaw__react_to_message')) {
    return `emoji=\`${input.emoji ?? '?'}\` msg=\`${input.message_id ?? '?'}\``;
  }
  if (toolName.startsWith('mcp__nanoclaw__schedule_task')) {
    return `prompt=${String(input.prompt ?? '').slice(0, 80)}…`;
  }
  // Fallback: show key names only (the agent probably doesn't need
  // the values to recognise what it did, and values can be huge).
  const keys = Object.keys(input);
  return `keys=[${keys.join(', ')}]`;
}

/**
 * Atomic-rotate `default.md` → `previous.md`, then write the new
 * `default.md` with the rendered Facts section. Idempotent at the
 * directory-creation level; the rotation step tolerates a missing
 * prior file (first-ever checkpoint write).
 */
export async function writeCheckpoint(inputs: CheckpointInputs): Promise<void> {
  const { dir, live, previous } = checkpointPaths(inputs.groupDir);

  fs.mkdirSync(dir, { recursive: true });

  // Rotate prior live → previous (forensics-only). Tolerate ENOENT
  // for the first-ever write.
  if (fs.existsSync(live)) {
    fs.renameSync(live, previous);
  }

  const invocations = await parseSessionTranscript(inputs.jsonlPath);
  const mutating = invocations
    .filter((inv) =>
      classifyTool(
        inv.name,
        inv.name === 'Bash' ? String(inv.input.command ?? '') : undefined,
      ),
    )
    .map((inv) => ({
      name: inv.name,
      inputSummary: summariseInput(inv.name, inv.input),
      completedAt: inv.completedAt,
      isError: inv.isError,
    }));

  const facts = renderFacts(inputs, mutating);
  fs.writeFileSync(live, facts);

  logger.info(
    {
      group: inputs.groupName,
      session: inputs.sessionId,
      checkpoint_path: live,
      mutating_count: mutating.length,
      total_invocations: invocations.length,
      used_tokens: inputs.usedTokens,
      context_window: inputs.thresholds.contextWindow,
    },
    'checkpoint_written',
  );
}

/**
 * Delete the per-group checkpoint pair (`default.md` + `previous.md`)
 * under `<groupDir>/.checkpoints/`. Returns the number of files that
 * were actually unlinked (0 = both already absent, 1 = one of the
 * pair was missing, 2 = both were present and deleted).
 *
 * Used by `nukeSession({ skipReentry: true })` (#127) when the
 * operator wants the next spawn to start without ANY reentry context.
 * Default `nukeSession` preserves the checkpoint — only this opt-in
 * path clears it.
 *
 * Idempotent: ENOENT on either file is the expected case for groups
 * that never crossed the threshold (no checkpoint ever written) or
 * for the first-ever-write group (no `previous.md`). Other fs errors
 * are logged-and-swallowed so a single bad checkpoint file doesn't
 * block the rest of the nuke.
 */
export function clearCheckpoints(groupDir: string): number {
  const { live, previous } = checkpointPaths(groupDir);
  let removed = 0;
  for (const file of [live, previous]) {
    try {
      fs.unlinkSync(file);
      removed++;
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') continue;
      logger.warn(
        {
          file,
          err: err instanceof Error ? err.message : String(err),
        },
        'clearCheckpoints: failed to unlink checkpoint file',
      );
    }
  }
  return removed;
}
