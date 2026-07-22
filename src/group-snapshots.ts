// Group-visible IPC snapshots (#851 slice 6, extracted verbatim from
// src/container-runner.ts).
//
// The tasks/groups JSON files the orchestrator writes into a group's
// IPC dir for the container to read, filtered by trust tier: main
// sees everything, trusted sees its own tasks, untrusted gets no IPC
// root files at all.
import fs from 'fs';
import path from 'path';

import { CR_FS_CODES, isFsErrorWithCode } from './fs-errors.js';
import { resolveGroupIpcPath } from './group-folder.js';

export function writeTasksSnapshot(
  groupFolder: string,
  isMain: boolean,
  tasks: Array<{
    id: string;
    groupFolder: string;
    prompt: string;
    script?: string | null;
    schedule_type: string;
    schedule_value: string;
    status: string;
    next_run: string | null;
  }>,
  isTrusted?: boolean,
): void {
  // Untrusted containers don't get IPC root files — tasks/ not mounted
  if (!isMain && !isTrusted) return;

  // Write filtered tasks to the group's IPC directory
  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

  // Main sees all tasks, others only see their own
  const filteredTasks = isMain
    ? tasks
    : tasks.filter((t) => t.groupFolder === groupFolder);

  const tasksFile = path.join(groupIpcDir, 'current_tasks.json');
  fs.writeFileSync(tasksFile, JSON.stringify(filteredTasks, null, 2));
}

export interface AvailableGroup {
  jid: string;
  name: string;
  lastActivity: string;
  isRegistered: boolean;
  containerConfig?: import('./types.js').RegisteredGroup['containerConfig'];
  requiresTrigger?: boolean;
}

/**
 * Write available groups snapshot for the container to read.
 * Only main group can see all available groups (for activation).
 * Non-main groups only see their own registration status.
 */
export function writeGroupsSnapshot(
  groupFolder: string,
  isMain: boolean,
  groups: AvailableGroup[],
  _registeredJids: Set<string>,
  isTrusted?: boolean,
): void {
  // Untrusted containers don't get IPC root files — available_groups not mounted
  if (!isMain && !isTrusted) return;

  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

  // Main sees all groups; others see nothing (they can't activate groups)
  const visibleGroups = isMain ? groups : [];

  const groupsFile = path.join(groupIpcDir, 'available_groups.json');

  // Preserve JID-keyed entries that agents may have written
  let existing: Record<string, unknown> = {};
  if (fs.existsSync(groupsFile)) {
    try {
      existing = JSON.parse(fs.readFileSync(groupsFile, 'utf-8'));
    } catch (err) {
      // Corrupt JSON (SyntaxError) or an fs read error resets to empty; a
      // non-fs, non-parse defect propagates.
      if (!(err instanceof SyntaxError) && !isFsErrorWithCode(err, CR_FS_CODES))
        throw err;
      existing = {};
    }
  }

  fs.writeFileSync(
    groupsFile,
    JSON.stringify(
      {
        ...existing,
        groups: visibleGroups,
        lastSync: new Date().toISOString(),
      },
      null,
      2,
    ),
  );
}
