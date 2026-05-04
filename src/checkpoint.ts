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
  /** Per-turn context size at threshold-cross — sum of
   *  `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`
   *  from the SDK `usage` payload. The bare `input_tokens` field is
   *  only the delta and underreports cache-heavy turns by 2–3 orders
   *  of magnitude (see #498). */
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
 * (EACCES, EPERM, EROFS, EBUSY, EIO, …) are logged and then re-thrown
 * to the caller per `jbaruch/coding-policy: error-handling` — claiming
 * cleanup succeeded when the file is still on disk would be a lie. The
 * caller (`nukeSession`) wraps this in a try/catch and continues the
 * rest of the nuke, so the operator sees an error log but the main
 * session wipe (DB rows + JSONL) is still applied.
 *
 * **Security**: `<groupDir>/.checkpoints/` lives inside a writable
 * container mount, so a compromised container could try to plant a
 * symlink to redirect the unlink at host files. Defense in depth
 * mirrors `wipeSessionJsonl` (#100):
 *
 *   1. lstat the `.checkpoints/` dir; refuse to traverse if it's a
 *      symlink, regardless of where it points.
 *   2. For each leaf, lstat first to learn the entry type. If it's
 *      a symlink, `fs.unlinkSync` removes the LINK entry only — the
 *      target is preserved. If it's a regular file, realpath both
 *      the dir and the file, assert the file's real path is inside
 *      the dir's real path, then unlink.
 *
 * The realpath check guards against an ancestor-symlink swap of the
 * `.checkpoints/` dir between the outer lstat and the leaf unlink
 * (TOCTOU). The symlink-branch keeps the "operator can opt to clear
 * a poisoned checkpoint" promise honest even if the file was
 * replaced with a link.
 */
export function clearCheckpoints(groupDir: string): number {
  const { dir, live, previous } = checkpointPaths(groupDir);

  // Realpath the parent groupDir up front. We need it to assert that
  // the resolved `.checkpoints/` real path lands at exactly the
  // expected child of groupDir — not somewhere else through a
  // TOCTOU symlink swap between the lstat below and the realpath
  // call further down.
  // Error-handling discipline (per `jbaruch/coding-policy: error-handling`):
  // ENOENT is the ONE recoverable code — it means the path the caller
  // asked us to wipe doesn't exist, which is what a successful clear
  // leaves anyway. Every other fs errno (EACCES, EPERM, EROFS, EBUSY,
  // EIO, …) is unexpected: we DID find the file but couldn't remove
  // it, so claiming the cleanup succeeded would be a lie. Log
  // diagnostic context, then re-throw so the caller (nukeSession) and
  // the IPC dispatch wrapper see the failure. Same for non-Error
  // throws — those indicate upstream bugs.
  //
  // Trade-off: a partial wipe (one file gone, the other throws) is
  // acceptable. Disk is consistent — the file we couldn't unlink is
  // still there for the operator to inspect and clean up manually.
  let realGroupDir: string;
  try {
    realGroupDir = fs.realpathSync(groupDir);
  } catch (err) {
    if (!(err instanceof Error)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0; // group folder doesn't exist → nothing to clear
    logger.warn(
      { groupDir, err },
      'clearCheckpoints: realpath failed on groupDir',
    );
    throw err;
  }
  const expectedRealDir = path.join(realGroupDir, CHECKPOINTS_SUBDIR);

  let dirLstat: fs.Stats;
  try {
    dirLstat = fs.lstatSync(dir);
  } catch (err) {
    if (!(err instanceof Error)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0; // .checkpoints/ never created
    logger.warn(
      { dir, err },
      'clearCheckpoints: lstat failed on .checkpoints/',
    );
    throw err;
  }
  if (dirLstat.isSymbolicLink()) {
    logger.error(
      { dir },
      'clearCheckpoints: refusing to traverse — .checkpoints/ itself is a symlink (possible escape attempt)',
    );
    return 0;
  }
  if (!dirLstat.isDirectory()) return 0;

  let realDir: string;
  try {
    realDir = fs.realpathSync(dir);
  } catch (err) {
    if (!(err instanceof Error)) throw err;
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT') return 0;
    logger.warn(
      { dir, err },
      'clearCheckpoints: realpath failed on .checkpoints/',
    );
    throw err;
  }
  // TOCTOU defense: between the `lstatSync(dir)` above and this
  // realpath, a compromised container could have swapped
  // `.checkpoints/` for a symlink. The lstat-not-symlink branch
  // would have passed (it ran on the original inode), but realpath
  // now resolves through the new symlink. Assert the resolved real
  // path equals the expected child of groupDir; refuse otherwise.
  if (realDir !== expectedRealDir) {
    logger.error(
      { dir, realDir, expectedRealDir },
      'clearCheckpoints: refusing — .checkpoints/ realpath escapes groupDir (TOCTOU?)',
    );
    return 0;
  }

  let removed = 0;
  for (const file of [live, previous]) {
    let entryStat: fs.Stats;
    try {
      entryStat = fs.lstatSync(file);
    } catch (err) {
      if (!(err instanceof Error)) throw err;
      const code = (err as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') continue;
      logger.warn({ file, err }, 'clearCheckpoints: lstat failed');
      throw err;
    }

    if (entryStat.isSymbolicLink()) {
      // Unlink the link entry only; target is preserved.
      try {
        fs.unlinkSync(file);
        removed++;
        logger.info(
          { file },
          'clearCheckpoints: unlinked symlinked checkpoint file (target preserved)',
        );
      } catch (err) {
        if (!(err instanceof Error)) throw err;
        const code = (err as NodeJS.ErrnoException).code;
        if (code === 'ENOENT') continue;
        logger.warn(
          { file, err },
          'clearCheckpoints: unlink-of-symlink failed',
        );
        throw err;
      }
      continue;
    }

    // Defensive type check: only proceed with realpath+unlink if
    // the leaf is a regular file. The SDK only writes regular files
    // here, but a corrupt/weird filesystem state (entry is a
    // directory, FIFO, socket, block device) would make unlinkSync
    // throw EISDIR/EPERM/etc. With the throw-on-non-ENOENT
    // discipline, that throw would abort the loop and skip the
    // sibling file. Logging-and-skipping malformed entries keeps
    // the helper resilient and matches the docstring's
    // "regular file: …" branch.
    if (!entryStat.isFile()) {
      logger.warn(
        { file },
        'clearCheckpoints: skipping non-file, non-symlink checkpoint entry',
      );
      continue;
    }

    // Regular file: realpath containment check before unlink.
    let realFile: string;
    try {
      realFile = fs.realpathSync(file);
    } catch (err) {
      if (!(err instanceof Error)) throw err;
      const code = (err as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') continue;
      logger.warn({ file, err }, 'clearCheckpoints: realpath failed');
      throw err;
    }
    if (!realFile.startsWith(realDir + path.sep)) {
      logger.warn(
        { file, realDir, realFile },
        'clearCheckpoints: refusing to unlink — realpath escapes .checkpoints/',
      );
      continue;
    }
    try {
      fs.unlinkSync(file);
      removed++;
    } catch (err) {
      if (!(err instanceof Error)) throw err;
      const code = (err as NodeJS.ErrnoException).code;
      if (code === 'ENOENT') continue;
      logger.warn({ file, err }, 'clearCheckpoints: unlink failed');
      throw err;
    }
  }
  return removed;
}
