/**
 * In-memory token → container-context registry shared between the
 * orchestrator (writer) and the credential proxy (reader).
 *
 * Each agent-container spawn registers a unique random token mapped to
 * `{group, tier, session, task_id}`. The orchestrator embeds the token
 * in the container's `ANTHROPIC_BASE_URL` as a path prefix:
 *
 *   ANTHROPIC_BASE_URL=http://gw:3001/c/<token>
 *
 * The Claude SDK joins endpoint paths to this base, so requests arrive
 * at the proxy as `/c/<token>/v1/messages`. The proxy strips the
 * prefix, looks up the context, and forwards the un-prefixed path
 * upstream to api.anthropic.com.
 *
 * Tokens live only for the duration of the spawn and are unregistered
 * when the container exits. The registry is process-local — no disk,
 * no network, no cross-process coordination needed.
 */
import { randomBytes } from 'crypto';

import type { ContainerContext } from './usage-log.js';

const registry = new Map<string, ContainerContext>();

/**
 * Generate a fresh URL-safe token. 16 bytes of randomness encoded as
 * base64url (~22 chars) — collision probability is negligible for the
 * registry's lifetime.
 */
export function generateToken(): string {
  return randomBytes(16).toString('base64url');
}

/** Register a context under a freshly-generated token. Returns the token. */
export function registerContainer(ctx: ContainerContext): string {
  const token = generateToken();
  registry.set(token, ctx);
  return token;
}

/** Look up the context for a token, or null if unknown / expired. */
export function lookupContainer(token: string): ContainerContext | null {
  return registry.get(token) ?? null;
}

/** Unregister a token. Idempotent. */
export function unregisterContainer(token: string): void {
  registry.delete(token);
}

/** Test-only: clear the entire registry. */
export function _resetRegistry(): void {
  registry.clear();
}

/** Test-only: registry size. */
export function _registrySize(): number {
  return registry.size;
}
