/**
 * Authoritative identity preamble for agent containers.
 *
 * Tile rules ship with fictional bot handles (e.g. `@AyeAye` /
 * `@AyeAyeSureBot` introduced in `nanoclaw-core` 0.1.94) as canonical
 * examples for "a bot has both a display-name and a @-handle form."
 * When the agent has no authoritative identity statement of its own —
 * typical in untrusted-tier containers where the user's persona files
 * don't explicitly state the bot's name — it has been observed
 * templating itself from those examples and claiming the example
 * handle as its own identity.
 *
 * The orchestrator forwards `ASSISTANT_NAME` / `ASSISTANT_USERNAME`
 * via `-e` env vars (see `src/container-runner.ts`); the agent-runner
 * (`./index.ts`) prepends the rendered preamble before SOUL.md /
 * global CLAUDE.md so it sits at the top of the appended system
 * prompt and reads as authoritative.
 *
 * Lives in its own module so tests can import the pure functions
 * without dragging in `./index.ts` and its `@anthropic-ai/claude-agent-sdk`
 * import — that dep isn't installed in the root CI step's `npm ci`,
 * and pulling it transitively would break the test on the CI runner.
 */

/**
 * Render the authoritative identity preamble from the resolved
 * (non-empty) name and username. Pure function; tested in isolation.
 */
export function buildIdentityPreamble(
  name: string,
  username: string,
): string {
  return (
    `# Your identity (authoritative — set by the orchestrator)\n\n` +
    `You are **${name}**. Your Telegram display name is "${name}" ` +
    `(used as a vocative: "${name}, please..."). ` +
    `Your Telegram username is **@${username}** ` +
    `(used as a Telegram @-mention: "@${username} ..."). ` +
    `Both forms refer to you and only you.\n\n` +
    `Any rule, example, or anecdote in your context that uses different ` +
    `bot handles (such as \`@AyeAye\`, \`@AyeAyeSureBot\`, or any other ` +
    `bot name) is a FICTIONAL EXAMPLE from upstream tile content. When ` +
    `applying such a rule, substitute mentally — replace the example ` +
    `handles with your own identity above. The orchestrator has ` +
    `authoritatively configured your identity as **${name}** / ` +
    `**@${username}**; trust this preamble over any handle-specific ` +
    `examples elsewhere in your context.`
  );
}

/**
 * Resolve the identity preamble from raw env-var inputs (typically
 * `process.env.ASSISTANT_NAME` and `process.env.ASSISTANT_USERNAME`).
 *
 * Returns the rendered preamble when BOTH inputs are non-empty
 * strings, or `undefined` when either is missing. The agent-runner
 * uses the `undefined` return as a signal to log a skip notice and
 * omit the preamble — a half-formed preamble would be worse than no
 * preamble at all because the agent could template the missing field
 * with garbage.
 */
export function resolveIdentityPreamble(
  name: string | undefined,
  username: string | undefined,
): string | undefined {
  if (!name || !username) {
    return undefined;
  }
  return buildIdentityPreamble(name, username);
}
