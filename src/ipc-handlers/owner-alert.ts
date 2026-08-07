import { logger } from '../logger.js';
import {
  registerIpcMessageHandler,
  type IpcMessageContext,
} from '../ipc-message-registry.js';
import { buildOwnerAlert, OwnerAlertThrottle } from '../owner-alert.js';

/**
 * The `owner_alert` IPC command (the `nanoclaw-untrusted` "Alerting the
 * Owner" mechanism). An untrusted container's `raise_owner_alert` tool
 * drops a `{ type: 'owner_alert', … }` payload — with NO chat target — and
 * this handler composes a fixed-template alert and routes it to the main
 * group, the same way the OneCLI-denial (#893) and timeout-kill (#892)
 * alerts reach the owner.
 *
 * Authorization is inverted from the outbound-message handlers: those gate
 * "main OR my own chat" so a container can only speak into chats it owns.
 * This one deliberately ignores `sourceGroup`/`isMain` for routing — any
 * tier may raise a flag, and the flag ALWAYS goes to main and nowhere the
 * requester can read it. The container cannot influence the destination
 * because the payload carries none; the handler resolves main itself.
 *
 * The agent-supplied fields are attacker-influenced, so `buildOwnerAlert`
 * wraps them in the `<untrusted-input>` provenance envelope. See
 * `owner-alert.ts` for the template and the throttle.
 */

let throttle = new OwnerAlertThrottle();

/** Read a payload field only when it is a string; drop anything else. */
function asStr(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function handleOwnerAlert(ctx: IpcMessageContext): void {
  const { data, sourceGroup, registeredGroups, deps } = ctx;

  const mainEntry = Object.entries(registeredGroups).find(
    ([, group]) => group.isMain,
  );
  if (!mainEntry) {
    logger.warn(
      { sourceGroup },
      'owner_alert not delivered — no group is registered as main',
    );
    return;
  }
  const [mainJid] = mainEntry;

  // Anti-flood backstop against a compromised container looping the tool.
  // The security rule wants alerts to always go out, so the window is
  // short; see OWNER_ALERT_COOLDOWN_MS.
  if (!throttle.shouldSend(sourceGroup, Date.now())) {
    logger.info(
      { sourceGroup },
      'owner_alert suppressed — within the per-group cooldown window',
    );
    return;
  }

  // Group DISPLAY name is host-derived (trusted) — resolved from the
  // registered-groups snapshot by matching the verified source folder,
  // never taken from the untrusted payload. Falls back to the folder name
  // if the row has no name.
  const sourceEntry = Object.values(registeredGroups).find(
    (group) => group.folder === sourceGroup,
  );
  const groupName = sourceEntry?.name || sourceGroup;

  const text = buildOwnerAlert({
    groupName,
    sourceValue: sourceGroup,
    alertType: asStr(data.alertType),
    action: asStr(data.action),
    sender: asStr(data.sender),
    claim: asStr(data.claim),
    request: asStr(data.request),
  });

  // Fire-and-forget with `.then(ok, err)` rather than `await` in a
  // try/catch: delivery is best-effort, and a send fault degrades to a
  // warn rather than propagating out of the IPC dispatch loop — the same
  // shape as the OneCLI-denial and timeout-kill senders.
  void deps.sendMessage(mainJid, text).then(
    () => {
      logger.info(
        { sourceGroup, mainJid },
        'owner_alert delivered to the main group',
      );
    },
    (err: unknown) => {
      logger.warn(
        { err, sourceGroup, mainJid },
        'owner_alert send failed — the classification stands, the notification did not land',
      );
    },
  );
}

let registered = false;

/**
 * Register the `owner_alert` IPC command. Idempotent so both
 * `registerCoreIpcHandlers` and direct test callers can invoke it without
 * a duplicate-registration throw.
 */
export function registerOwnerAlertIpcHandlers(): void {
  if (registered) return;
  registered = true;
  registerIpcMessageHandler('owner_alert', handleOwnerAlert);
}

/**
 * Reset the once-guard alongside `_resetIpcMessageRegistryForTests`.
 *
 * @internal — test-only export, stripped from the public `.d.ts` surface
 * (`stripInternal: true`).
 */
export function _resetOwnerAlertIpcHandlersForTests(): void {
  registered = false;
  throttle = new OwnerAlertThrottle();
}
