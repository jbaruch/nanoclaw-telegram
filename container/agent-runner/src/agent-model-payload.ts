/**
 * Pure helpers for the `set_*_agent_model` MCP bridge tools. Extracted so
 * the input-normalisation and IPC-payload shape are testable in isolation
 * — the MCP server registration in `ipc-mcp-stdio.ts` is a thin adapter
 * over these.
 *
 * Normalisation rule: the host-side IPC handlers in `src/ipc.ts` trim
 * incoming `agentModel` strings and treat an empty-after-trim value as
 * "clear the override" (i.e. delete the field / write null). Mirroring
 * that here means the bridge's "requested → X" response text matches what
 * the host actually applies, instead of claiming we pinned a model to a
 * whitespace string and the host quietly clearing it.
 */

export function normalizeAgentModelInput(input: string | null): string | null {
  if (input === null) return null;
  const trimmed = input.trim();
  return trimmed.length === 0 ? null : trimmed;
}

export interface SetAgentModelArgs {
  groupFolder: string;
  agentModel: string | null;
}

export interface SetMaintenanceAgentModelArgs {
  groupFolder: string;
  maintenanceAgentModel: string | null;
}

export interface SetTaskAgentModelArgs {
  task_id: string;
  agentModel: string | null;
}

export function buildSetAgentModelPayload(
  args: SetAgentModelArgs,
  now: Date,
): {
  type: 'set_agent_model';
  groupFolder: string;
  agentModel: string | null;
  timestamp: string;
} {
  return {
    type: 'set_agent_model',
    groupFolder: args.groupFolder,
    agentModel: normalizeAgentModelInput(args.agentModel),
    timestamp: now.toISOString(),
  };
}

export function buildSetMaintenanceAgentModelPayload(
  args: SetMaintenanceAgentModelArgs,
  now: Date,
): {
  type: 'set_maintenance_agent_model';
  groupFolder: string;
  maintenanceAgentModel: string | null;
  timestamp: string;
} {
  return {
    type: 'set_maintenance_agent_model',
    groupFolder: args.groupFolder,
    maintenanceAgentModel: normalizeAgentModelInput(args.maintenanceAgentModel),
    timestamp: now.toISOString(),
  };
}

export function buildSetTaskAgentModelPayload(
  args: SetTaskAgentModelArgs,
  now: Date,
): {
  type: 'set_task_agent_model';
  taskId: string;
  agentModel: string | null;
  timestamp: string;
} {
  return {
    type: 'set_task_agent_model',
    // MCP-facing param is `task_id` (matches pause_task / resume_task /
    // cancel_task / update_task siblings); IPC-facing field is `taskId`
    // (matches the host-side handler shape in `src/ipc.ts`). The
    // snake↔camel mapping happens once, here.
    taskId: args.task_id,
    agentModel: normalizeAgentModelInput(args.agentModel),
    timestamp: now.toISOString(),
  };
}

export function describeAgentModelChange(
  normalised: string | null,
  clearedFallbackDescription: string,
): string {
  return normalised === null ? clearedFallbackDescription : `"${normalised}"`;
}
