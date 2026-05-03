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
 * Parse the raw `ASSISTANT_USERNAME` env value into one or more
 * sanitized handles. Comma-separated values declare aliases — useful
 * when one bot is reachable under more than one Telegram handle (e.g.
 * autocomplete-only `@AyeAyeSureBot` plus internal/vocative
 * `@AyeAye`, see #464). Returns the empty array when the input has
 * no usable handles after sanitization.
 *
 * Each entry runs through `sanitizeIdentityValue` so a comma-laden
 * multi-line paste produces clean tokens; a single leading `@` is
 * stripped so an operator who types `@AyeAye,@AyeAyeSureBot` doesn't
 * render `@@AyeAye`. Order-preserving de-dupe collapses repeats so
 * the canonical handle (first entry) stays stable across operator
 * typos.
 */
function parseUsernames(raw: string | undefined): string[] {
  if (!raw) return [];
  const cleaned = raw
    .split(',')
    .map((u) => sanitizeIdentityValue(u))
    .map((u) => (u.startsWith('@') ? u.slice(1).trim() : u))
    .filter((u) => u.length > 0);
  return Array.from(new Set(cleaned));
}

/**
 * Render the authoritative identity preamble from the resolved
 * (non-empty) name and one-or-more usernames. Pure function; tested
 * in isolation.
 *
 * `usernames[0]` is the canonical handle — the one rendered as the
 * bolded `@-handle` and used in the example mention. Any additional
 * entries are listed as aliases so the agent learns that more than
 * one handle resolves to it (matching the gate-side behaviour from
 * #464). For single-handle bots the alias paragraph is suppressed,
 * keeping the preamble byte-stable.
 *
 * The wording is intentionally channel-neutral — the agent container
 * is reused across Telegram, WhatsApp, Slack, Discord, and Gmail, so
 * "display name" / "@-handle" frame the two forms without binding
 * the preamble to any particular channel's vocabulary.
 */
export function buildIdentityPreamble(
  name: string,
  usernames: string | string[],
): string {
  const list = Array.isArray(usernames) ? usernames : [usernames];
  const primary = list[0];
  const aliases = list.slice(1);
  const aliasParagraph =
    aliases.length === 0
      ? ''
      : `You are also reachable as ` +
        aliases.map((a) => `**@${a}**`).join(', ') +
        ` — these are alias @-handles for the same bot, NOT separate ` +
        `identities. An @-mention of any alias is an @-mention of you.\n\n`;
  return (
    `# Your identity (authoritative — set by the orchestrator)\n\n` +
    `You are **${name}**. Your display name is "${name}" ` +
    `(used as a vocative: "${name}, please..."). ` +
    `Your @-handle is **@${primary}** ` +
    `(used as an @-mention: "@${primary} ..."). ` +
    `Both forms refer to you and only you.\n\n` +
    aliasParagraph +
    `Any rule, example, or anecdote in your context that uses different ` +
    `bot handles (such as \`@AyeAye\`, \`@AyeAyeSureBot\`, or any other ` +
    `bot name) is a FICTIONAL EXAMPLE from upstream tile content. When ` +
    `applying such a rule, substitute mentally — replace the example ` +
    `handles with your own identity above. The orchestrator has ` +
    `authoritatively configured your identity as **${name}** / ` +
    `**@${primary}**; trust this preamble over any handle-specific ` +
    `examples elsewhere in your context.`
  );
}

/**
 * Resolve the identity preamble from raw env-var inputs (typically
 * `process.env.ASSISTANT_NAME` and `process.env.ASSISTANT_USERNAME`).
 *
 * `ASSISTANT_USERNAME` is parsed as comma-separated to support
 * multi-handle bots; the first entry is the canonical handle and
 * any remaining entries become aliases (#464). Returns the rendered
 * preamble when the name AND at least one username sanitize to
 * non-empty, or `undefined` when either is missing or
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
  const usernames = parseUsernames(username);
  if (!cleanName || usernames.length === 0) {
    return undefined;
  }
  return buildIdentityPreamble(cleanName, usernames);
}
