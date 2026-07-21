import { GroupQueue } from './group-queue.js';
import { Channel } from './types.js';

/**
 * Shared orchestrator runtime singletons, extracted from src/index.ts
 * (#749). Created once and shared by the composition root (index.ts) and
 * the message pipeline (message-pipeline.ts). All are const — never
 * reassigned, only mutated in place — so importers get one live shared
 * reference.
 */
// Per-chat reply-to tracking: updated when follow-up messages are piped,
// consumed by the output callback to quote-reply the latest message.
export const pendingReplyTo: Record<string, string | undefined> = {};

export const channels: Channel[] = [];

export const queue = new GroupQueue();

// Per-folder timestamp of the most recent `nukeSession` call. Used to
// gate the post-spawn `setSession` writes against a race where a nuke
// fires while a container is still being awaited: the dying container
// emits a final SDK result containing the same `newSessionId` it was
// processing, and the completion handler would otherwise resurrect
// that row in the DB right after nuke deleted it. Resurrected row
// points at the JSONL file that nuke just wiped — every subsequent
// spawn reads the resurrected sessionId, the SDK can't load the
// transcript, and the chat wedges permanently. See #144 bug 1.
//
// Compare against the spawn's start timestamp captured BEFORE
// `runContainerAgent` is invoked: if `nukeTimestamps[folder] >=
// spawnStart`, the nuke landed after the spawn began (or
// concurrently), so any session-id write coming back from this
// container is stale and must be dropped.
export const nukeTimestamps: Record<string, number> = {};
