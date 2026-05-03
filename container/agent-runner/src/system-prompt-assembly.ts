/**
 * System-prompt assembly for the agent SDK `query()` call.
 *
 * Splits the appended system-prompt content into two layers so the
 * Anthropic prompt cache (5-min TTL, API-key-scoped) can land the
 * frozen prefix on every subsequent message in the same group/session
 * window:
 *
 *   FROZEN (cacheable, byte-identical across messages within a single
 *   `runQuery` invocation; see "Cache window" below for what holds
 *   across runQuery boundaries):
 *     - identity preamble (resolved once per runQuery from
 *       ASSISTANT_NAME / ASSISTANT_USERNAME env vars set by the
 *       orchestrator at container spawn)
 *     - SOUL.md content (trust-tier resolved by the orchestrator's
 *       mount layer; see note on mount mode below)
 *     - FORMATTING.md content
 *
 *   VOLATILE (not part of the cache prefix; may change per message):
 *     - reserved for future use; today the SDK preset emits its own
 *       dynamic sections (cwd, auto-memory, git status) which we strip
 *       via `excludeDynamicSections: true` so they re-inject as a
 *       synthetic `isMeta:true` user message instead of breaking the
 *       system-prompt cache.
 *
 * Out of scope for this module. Group folder `CLAUDE.md` / `MEMORY.md`
 * and `additionalDirectories` CLAUDE.md files are loaded by the SDK
 * via `settingSources` / `additionalDirectories` — separate pipelines
 * from the explicit `append` string this builder controls. Whether
 * those inputs sit before or after the SDK's internal cache breakpoint
 * depends on the agent-SDK's own behavior, which this PR does not
 * promise to know or rely on. They are intentionally NOT in the
 * `append` we control, but documenting them as guaranteed-volatile
 * would overstate what this builder gates. The frozen builder gates
 * only the explicit `append` string; cache misses driven by other SDK
 * inputs are out of its remit.
 *
 * The classification is deterministic — no judgement at runtime. A
 * value is FROZEN when it cannot change WITHIN a single `runQuery`
 * invocation (because `runQuery` reads each frozen input once at the
 * top of the call and reuses the captured value for every turn that
 * runQuery emits). A value is VOLATILE when it changes per turn
 * (timestamp, the user's current message, per-turn dynamic sections).
 * "Within a single runQuery" — not "within a container's lifetime" —
 * is the precise scope: the cache lives across runQuery invocations
 * within a 5-min window, and SOUL.md / FORMATTING.md edits between
 * runQuery boundaries SHIFT the prefix on the next runQuery (see
 * "Cache window scope" below). That is the desired behavior; it lets
 * operator-edited steering propagate without restarting the container.
 * What we MUST avoid is volatile content leaking into the frozen
 * append within ONE runQuery — that would invalidate the cache on
 * every turn even when nothing actually changed at the source.
 *
 * Cache window scope. The mount mode for `/workspace/global` differs
 * by tier: untrusted/trusted containers mount it RO, but `main`
 * containers mount it RW (see `src/container-runner.ts` —
 * `readonly: !isMain`). What makes the frozen append byte-identical
 * is that `runQuery` reads SOUL.md / FORMATTING.md exactly once at
 * the top of the call and reuses the captured strings for every turn
 * inside that runQuery; the file-on-disk could change mid-session
 * without affecting an in-flight runQuery. ACROSS runQuery
 * invocations within the same long-lived agent-runner process (e.g., a
 * fresh inbound spawning a new runQuery in a container that's been
 * up for an hour), if SOUL.md or FORMATTING.md content has changed,
 * the prefix shifts by design — operator-edited authoritative
 * steering MUST take effect on the next message and the cache miss is
 * the correct behavior.
 *
 * `ASSISTANT_NAME` / `ASSISTANT_USERNAME` are the orchestrator-injected
 * spawn-time env vars (`-e ASSISTANT_NAME=...` from
 * `src/container-runner.ts`). `resolveIdentityPreamble` re-reads them
 * each runQuery, but the values are pinned at container spawn — they
 * cannot change while the agent-runner process is alive. An operator
 * who edits these values in the host config has to bounce the
 * container (or wait for it to recycle) for the new identity to take
 * effect; while the container is up, the identity preamble is stable
 * and the cache prefix doesn't shift on identity drift.
 *
 * Why this lives in its own module: the `index.ts` runtime imports
 * `@anthropic-ai/claude-agent-sdk`, which isn't installed in the root
 * CI step's `npm ci`. Pure functions go here so the unit tests can
 * import them directly without dragging the SDK in. Same pattern as
 * `./identity-preamble.ts`.
 */

/**
 * Inputs to the frozen system-prompt builder. Each field maps 1:1 to
 * a frozen content source. Missing fields are tolerated (some
 * deployments don't ship all three) — the builder skips them rather
 * than emitting a half-formed prompt.
 */
export interface FrozenSystemPromptInputs {
  /**
   * Authoritative identity preamble rendered from ASSISTANT_NAME /
   * ASSISTANT_USERNAME (see `./identity-preamble.ts`). Undefined when
   * either env var is missing — the runtime caller logs a skip notice.
   */
  identityPreamble: string | undefined;
  /**
   * Contents of `/workspace/global/SOUL.md` (trust-tier resolved by
   * mount), captured once per `runQuery` invocation. The mount is RO
   * on untrusted/trusted tiers and RW on main; the cache invariant is
   * upheld by reading once per runQuery, not by mount mode (see module
   * docstring "Cache window scope"). Undefined when the file is absent.
   */
  soulMd: string | undefined;
  /**
   * Contents of `/workspace/global/FORMATTING.md`, captured once per
   * `runQuery` invocation under the same one-read-per-runQuery rule
   * as `soulMd`. Undefined when absent.
   */
  formattingMd: string | undefined;
}

/**
 * Build the frozen append segment from its content inputs.
 *
 * Returns a single concatenated string with `\n\n---\n\n` separators
 * between non-empty parts (matches the pre-refactor format), or
 * `undefined` when every input is missing. Order is fixed: identity
 * preamble first (so it sits at the top of the appended system prompt
 * and reads as authoritative), then SOUL, then FORMATTING.
 *
 * Determinism guarantee: for the same inputs, the output is
 * byte-identical. The runtime caller must not include any value that
 * shifts between turns WITHIN a single `runQuery` invocation — see
 * the module docstring's frozen-vs-volatile split for the precise
 * scope (changes BETWEEN runQuery invocations are allowed and
 * intentional; the cache miss on the next runQuery is the right
 * behavior because operator edits to authoritative steering must take
 * effect on the next message).
 */
export function buildFrozenSystemPromptAppend(
  inputs: FrozenSystemPromptInputs,
): string | undefined {
  const parts: string[] = [];
  if (inputs.identityPreamble) {
    parts.push(inputs.identityPreamble);
  }
  if (inputs.soulMd) {
    parts.push(inputs.soulMd);
  }
  if (inputs.formattingMd) {
    parts.push(inputs.formattingMd);
  }
  return parts.length > 0 ? parts.join('\n\n---\n\n') : undefined;
}
