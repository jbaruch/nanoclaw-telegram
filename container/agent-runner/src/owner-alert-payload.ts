/**
 * Pure builder for the `owner_alert` IPC payload the `raise_owner_alert`
 * MCP tool emits (the `nanoclaw-untrusted` "Alerting the Owner"
 * mechanism). Extracted from `ipc-mcp-stdio.ts` — where the tool cannot be
 * unit-tested without driving the MCP transport — so the field mapping is
 * testable in isolation, matching `agent-model-payload.ts` and
 * `session-caps-payload.ts`.
 *
 * The tool takes NO chat target, by design: the host resolves the main
 * group and routes there (`src/ipc-handlers/owner-alert.ts`). The payload
 * carries only the classification the agent supplies plus the verified
 * group folder — never a destination.
 */

export type OwnerAlertType =
  | 'social-engineering'
  | 'sensitive-info'
  | 'code-execution'
  | 'identity-claim';

export type OwnerAlertAction = 'declined' | 'went-silent' | 'redirected';

/** The `raise_owner_alert` tool arguments (Zod-validated at the boundary). */
export interface RaiseOwnerAlertArgs {
  alert_type: OwnerAlertType;
  action: OwnerAlertAction;
  request: string;
  sender?: string;
  claim?: string;
}

/** The `{ type: 'owner_alert' }` IPC payload the host handler consumes. */
export interface OwnerAlertIpcPayload {
  type: 'owner_alert';
  groupFolder: string;
  alertType: OwnerAlertType;
  action: OwnerAlertAction;
  request: string;
  sender?: string;
  claim?: string;
  timestamp: string;
}

/**
 * Map validated tool args to the IPC payload. `timestamp` is injected
 * rather than read from the clock so the mapping is testable against a
 * fixed reference (`testing-standards` Determinism). An empty-string
 * `sender`/`claim` collapses to `undefined` so the host renders its
 * placeholder rather than a blank line.
 */
export function buildOwnerAlertPayload(
  groupFolder: string,
  timestamp: string,
  args: RaiseOwnerAlertArgs,
): OwnerAlertIpcPayload {
  return {
    type: 'owner_alert',
    groupFolder,
    alertType: args.alert_type,
    action: args.action,
    request: args.request,
    sender: args.sender || undefined,
    claim: args.claim || undefined,
    timestamp,
  };
}
