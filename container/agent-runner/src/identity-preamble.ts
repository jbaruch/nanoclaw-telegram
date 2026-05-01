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

// Cap on rendered identity values. The orchestrator sets these via
// `-e ASSISTANT_NAME` / `-e ASSISTANT_USERNAME`, so they're not
// attacker-controlled in practice; the cap exists to keep an operator
// typo (a multi-paragraph value pasted into `.env` by mistake) from
// flooding the system prompt.
const IDENTITY_VALUE_MAX_LEN = 256;

/**
 * Sanitize a single identity field before interpolating it into the
 * preamble template. Collapses `\r` / `\n` to spaces (raw newlines
 * would otherwise create stray system-prompt lines / headings inside
 * a markdown-shaped block), trims whitespace, and length-caps.
 * Returns the empty string for whitespace-only / `undefined` input so
 * the caller's "missing → skip" check catches it.
 */
function sanitizeIdentityValue(raw: string | undefined): string {
  if (!raw) return '';
  const collapsed = raw.replace(/[\r\n]+/g, ' ').trim();
  if (collapsed.length === 0) return '';
  return collapsed.slice(0, IDENTITY_VALUE_MAX_LEN);
}

/**
 * Render the authoritative identity preamble from the resolved
 * (non-empty) name and username. Pure function; tested in isolation.
 *
 * The wording is intentionally channel-neutral — the agent container
 * is reused across Telegram, WhatsApp, Slack, Discord, and Gmail, so
 * "display name" / "@-handle" frame the two forms without binding
 * the preamble to any particular channel's vocabulary.
 */
export function buildIdentityPreamble(
  name: string,
  username: string,
): string {
  return (
    `# Your identity (authoritative — set by the orchestrator)\n\n` +
    `You are **${name}**. Your display name is "${name}" ` +
    `(used as a vocative: "${name}, please..."). ` +
    `Your @-handle is **@${username}** ` +
    `(used as an @-mention: "@${username} ..."). ` +
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
 * Returns the rendered preamble when BOTH inputs sanitize to a
 * non-empty string, or `undefined` when either is missing or
 * whitespace-only after normalization. The agent-runner uses the
 * `undefined` return as a signal to log a skip notice and omit the
 * preamble — a half-formed preamble would be worse than no preamble
 * at all because the agent could template the missing field with
 * garbage.
 *
 * Normalization (`sanitizeIdentityValue`) collapses raw `\r` / `\n`
 * to spaces and length-caps each field, so an operator typo that
 * pastes multi-line content into `.env` can't break the preamble's
 * markdown structure or smuggle stray heading lines into the
 * top-of-context system prompt.
 */
export function resolveIdentityPreamble(
  name: string | undefined,
  username: string | undefined,
): string | undefined {
  const cleanName = sanitizeIdentityValue(name);
  const cleanUsername = sanitizeIdentityValue(username);
  if (!cleanName || !cleanUsername) {
    return undefined;
  }
  return buildIdentityPreamble(cleanName, cleanUsername);
}
