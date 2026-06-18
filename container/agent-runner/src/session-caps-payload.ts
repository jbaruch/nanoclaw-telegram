/**
 * Pure helpers for the `set_session_caps` MCP bridge tool (#561).
 * Extracted so the empty-update guard, IPC-payload shape, and response
 * text are testable in isolation — the MCP server registration in
 * `ipc-mcp-stdio.ts` is a thin adapter over these. Mirrors the
 * `agent-model-payload.ts` pattern.
 *
 * Contract mirror: the host-side handler in `src/ipc.ts` treats an absent
 * cap field as "leave unchanged", a `null` as "clear back to the global
 * default", and rejects a request carrying neither cap. Building the
 * payload with only the explicitly-provided fields here keeps the bridge's
 * "requested → X" response text matching what the host actually applies.
 */

export interface SetSessionCapsArgs {
  groupFolder: string;
  sessionTurnCap?: number | null;
  sessionTokenCap?: number | null;
}

export interface SetSessionCapsPayload {
  type: 'set_session_caps';
  groupFolder: string;
  sessionTurnCap?: number | null;
  sessionTokenCap?: number | null;
  timestamp: string;
}

/**
 * True when neither cap was provided — a no-op the host rejects. The
 * bridge guards on this BEFORE writing an IPC file so the tool returns an
 * actionable error instead of a success-looking empty change set.
 */
export function isEmptySessionCapsUpdate(args: SetSessionCapsArgs): boolean {
  return (
    args.sessionTurnCap === undefined && args.sessionTokenCap === undefined
  );
}

/**
 * Build the IPC payload carrying only the explicitly-provided caps. Caller
 * must reject empty updates first via `isEmptySessionCapsUpdate`.
 */
export function buildSetSessionCapsPayload(
  args: SetSessionCapsArgs,
  now: Date,
): SetSessionCapsPayload {
  const payload: SetSessionCapsPayload = {
    type: 'set_session_caps',
    groupFolder: args.groupFolder,
    timestamp: now.toISOString(),
  };
  if (args.sessionTurnCap !== undefined) {
    payload.sessionTurnCap = args.sessionTurnCap;
  }
  if (args.sessionTokenCap !== undefined) {
    payload.sessionTokenCap = args.sessionTokenCap;
  }
  return payload;
}

/**
 * One-line human description of the requested change for the tool's
 * response text. Post-guard the payload always carries at least one cap;
 * the empty fallback is defensive.
 */
export function describeSessionCapsChange(
  payload: SetSessionCapsPayload,
): string {
  const parts: string[] = [];
  const describe = (v: number | null | undefined, name: string): void => {
    if (v === undefined) return;
    parts.push(
      v === null ? `${name} cleared (use global default)` : `${name} → ${v}`,
    );
  };
  describe(payload.sessionTurnCap, 'turn cap');
  describe(payload.sessionTokenCap, 'token cap');
  return parts.length > 0 ? parts.join(', ') : '(no change)';
}
