import { logger } from './logger.js';
import type { IpcDeps } from './ipc.js';
import type { RegisteredGroup } from './types.js';

/**
 * Message-file payload shape shared by every IPC message handler (#878).
 * Deserialized from raw JSON dropped into a group's `messages/` IPC dir —
 * every field is untrusted container input; handlers validate before use.
 *
 * Sibling of `IpcTaskPayload` in `ipc-registry.ts`, kept separate because
 * the two surfaces carry different fields and different authorization
 * shapes: a task payload can be admin-gated wholesale (`requiresMain`),
 * while every message command uses the same finer-grained "main OR the
 * target chat is my own group" rule, evaluated inside the handler.
 */
export interface IpcMessagePayload {
  type?: string;
  chatJid?: string;
  text?: string;
  sender?: string;
  replyToMessageId?: string;
  pin?: boolean;
  emoji?: string;
  messageId?: string;
  filePath?: string;
  caption?: string;
  [key: string]: unknown;
}

/**
 * Everything a message handler gets to work with. `sourceGroup` and
 * `isMain` are VERIFIED identity — derived from the IPC directory the
 * message file arrived in, never from payload fields. That derivation is
 * the security boundary; handlers must base authorization on these, not
 * on anything inside `data`.
 *
 * `registeredGroups` is the snapshot the poller already read for this
 * pass, passed down rather than re-read per handler so every message in
 * one poll sees a consistent view (and so a handler can't be surprised
 * by a registration that landed mid-loop).
 *
 * `file` is the IPC filename, carried for log correlation only.
 */
export interface IpcMessageContext {
  data: IpcMessagePayload;
  /** Verified identity from the IPC directory path. */
  sourceGroup: string;
  /** Verified from the directory path, not the payload. */
  isMain: boolean;
  registeredGroups: Record<string, RegisteredGroup>;
  deps: IpcDeps;
  /** IPC filename, for log correlation. */
  file: string;
}

export type IpcMessageHandler = (
  ctx: IpcMessageContext,
) => Promise<void> | void;

const handlers = new Map<string, IpcMessageHandler>();

/**
 * Register a host IPC message command (#878). New outbound-message
 * capabilities plug in here instead of extending the `else if` chain in
 * `startIpcWatcher`. Duplicate names are a wiring bug (two modules
 * claiming one command) and fail loudly.
 */
export function registerIpcMessageHandler(
  name: string,
  handler: IpcMessageHandler,
): void {
  if (handlers.has(name)) {
    throw new Error(`IPC message handler already registered: ${name}`);
  }
  handlers.set(name, handler);
}

export function hasIpcMessageHandler(name: string): boolean {
  return handlers.has(name);
}

/**
 * Wipe the handler registry between tests. The registry is module-global
 * shared state; `testing-standards` requires tests to clean it up so
 * order never matters. Callers that also exercised
 * `registerCoreIpcHandlers` must reset its once-guard too — see
 * `_resetCoreIpcHandlersForTests` in `ipc-handlers/index.ts`.
 *
 * @internal — test-only export, stripped from the public `.d.ts`
 * surface (`stripInternal: true`).
 */
export function _resetIpcMessageRegistryForTests(): void {
  handlers.clear();
}

/**
 * Dispatch one message payload to its registered handler. Returns false
 * when no handler is registered for `data.type`.
 *
 * The caller deletes the IPC file either way — an unroutable message
 * payload is consumed silently, exactly as the pre-#878 `else if` chain
 * did when no branch matched (there was never a trailing `else`). A
 * handler that declines the payload on its own guards (missing field, an
 * optional channel dep the deployment doesn't provide) returns without
 * acting, which is the same outcome as the old unmatched-branch fall-
 * through.
 */
export async function dispatchIpcMessage(
  ctx: IpcMessageContext,
): Promise<boolean> {
  const handler = ctx.data.type ? handlers.get(ctx.data.type) : undefined;
  if (!handler) {
    logger.debug(
      { sourceGroup: ctx.sourceGroup, type: ctx.data.type, file: ctx.file },
      '[ipc] no message handler registered for payload type — discarding',
    );
    return false;
  }
  await handler(ctx);
  return true;
}
