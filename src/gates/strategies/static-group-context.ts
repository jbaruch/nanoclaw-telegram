/**
 * Default Stage 2 context strategy: derive the volatile suffix from
 * stable, host-side data (group display name, group CLAUDE.md head,
 * assistant identity).
 *
 * Failure mode: a missing CLAUDE.md is treated as "no group context
 * yet" — the strategy still emits a minimal suffix with the group
 * name and assistant identity, never throws.
 */
import fs from 'fs';
import path from 'path';

import type { ContextStrategy } from '../context-strategy.js';
import type { GateContext } from '../index.js';
import { resolveGroupFolderPath } from '../../group-folder.js';
import {
  ASSISTANT_NAME,
  ASSISTANT_OWNER_HANDLE,
  ASSISTANT_OWNER_NAME,
  ASSISTANT_USERNAMES,
} from '../../config.js';
import { getRegisteredGroup } from '../../db.js';
import { logger } from '../../logger.js';

const CLAUDE_MD_HEAD_LINES = 200;

interface ClaudeMdHead {
  body: string;
}

function readClaudeMdHead(folder: string): ClaudeMdHead | null {
  const groupDir = resolveGroupFolderPath(folder);
  const claudeMdPath = path.join(groupDir, 'CLAUDE.md');
  let raw: string;
  try {
    raw = fs.readFileSync(claudeMdPath, 'utf8');
  } catch (err) {
    const e = err as NodeJS.ErrnoException;
    if (e.code === 'ENOENT') return null;
    logger.warn(
      { folder, err: e.message },
      'static-group-context: read CLAUDE.md failed',
    );
    return null;
  }
  const lines = raw.split('\n').slice(0, CLAUDE_MD_HEAD_LINES);
  return { body: lines.join('\n') };
}

interface ResolvedGroupInfo {
  groupName: string;
  claudeMd: ClaudeMdHead | null;
}

function resolveGroupInfo(ctx: GateContext): ResolvedGroupInfo {
  // Look up display name via DB. The GateContext only carries jid +
  // folder; the registered-groups table holds the human-friendly name.
  const row = getRegisteredGroup(ctx.groupJid);
  const groupName = row?.name ?? ctx.groupFolder;
  const claudeMd = readClaudeMdHead(ctx.groupFolder);
  return { groupName, claudeMd };
}

export const staticGroupContextStrategy: ContextStrategy = {
  name: 'static-group-context',

  async buildContext(ctx: GateContext): Promise<string> {
    const info = resolveGroupInfo(ctx);
    // Render every configured handle so the classifier knows the bot
    // can be addressed under any of them (#464). Single-handle deploys
    // see the original `Telegram @-handle: @x` line; multi-handle
    // deploys see a plural label listing every alias. The label
    // change is what the classifier prompt's identity-match rule
    // keys off — pluralizing only when there's more than one handle
    // keeps single-handle prompts byte-stable.
    const handleLine =
      ASSISTANT_USERNAMES.length === 1
        ? `  Telegram @-handle: @${ASSISTANT_USERNAMES[0]}`
        : `  Telegram @-handles (aliases — any of these refers to the assistant): ${ASSISTANT_USERNAMES.map((u) => `@${u}`).join(', ')}`;
    const lines = [
      `Group: ${info.groupName}`,
      `Group folder: ${ctx.groupFolder}`,
      `Assistant identity:`,
      `  Display name: ${ASSISTANT_NAME}`,
      handleLine,
    ];
    // Owner line is opt-in: the classifier's owner-aware rule keys off
    // the `Owner: <Name> (@<handle>)` substring. When either var is
    // unset we suppress the line entirely so installs that don't
    // configure owner identity see zero behaviour change.
    if (ASSISTANT_OWNER_NAME && ASSISTANT_OWNER_HANDLE) {
      lines.push(
        `  Owner: ${ASSISTANT_OWNER_NAME} (@${ASSISTANT_OWNER_HANDLE})`,
      );
    }
    if (info.claudeMd) {
      lines.push('', 'Group CLAUDE.md (head):', info.claudeMd.body);
    } else {
      lines.push('', 'Group CLAUDE.md: not present');
    }
    return lines.join('\n');
  },
};
