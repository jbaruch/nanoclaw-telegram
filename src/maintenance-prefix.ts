import { MAINTENANCE_SESSION_NAME } from './group-queue.js';

// Prefix for outbound text emitted by the maintenance-session AyeAye.
// Without the prefix, a scheduled-task reply looks identical to a
// user-facing reply in the chat, which confused Baruch when he
// responded to `[heartbeat from maintenance]` messages as if they were
// live conversation. The prefix is applied BOTH to Telegram-bound text
// AND to the messages.db copy so the full trail shows provenance —
// heartbeat accounting, future message recap, etc.
const MAINTENANCE_MESSAGE_PREFIX = '[M] ';

/**
 * Prepend `[M] ` if the payload came from the maintenance session.
 * Idempotent — if the text already begins with the prefix (double-
 * hop case, agent that hand-typed it, whatever), we don't stack.
 * Exported for the unit test; the production caller is in the same
 * file so the public API is a single entry point.
 *
 * @internal — test-only export, should not be part of the public
 * `.d.ts` surface (we build with `stripInternal: true`).
 */
export function applyMaintenancePrefix(
  text: string,
  sessionName: string | undefined,
): string {
  if (sessionName !== MAINTENANCE_SESSION_NAME) return text;
  if (text.startsWith(MAINTENANCE_MESSAGE_PREFIX)) return text;
  return MAINTENANCE_MESSAGE_PREFIX + text;
}
