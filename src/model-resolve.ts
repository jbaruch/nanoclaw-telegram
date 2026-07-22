// Agent-model resolution (#851 slice 1, extracted verbatim from
// src/container-runner.ts).
//
// The model ladder for every container spawn: the global default and
// its env override, the per-trust-tier base (#613), the per-group
// override (#395), and the per-session/per-task resolution (#509).
// Pure functions + constants only — the env-derived AGENT_MODEL /
// AGENT_EFFORT snapshots stay in container-runner.ts with the spawn
// code that forwards them.
import { logger } from './logger.js';

/**
 * Model the agent-runner passes to the SDK's `query()` call. Forwarded as
 * the `AGENT_MODEL` env var on container spawn and read by
 * `container/agent-runner/src/index.ts`. Bumping this constant is the single
 * source of truth for the container agent's model — no rebuild of the
 * agent-runner image needed.
 *
 * Format: SDK model alias (`opus`, `sonnet[1m]`) or full model ID
 * (`claude-opus-4-7[1m]`). See
 * `container/agent-runner/node_modules/@anthropic-ai/claude-agent-sdk/sdk.d.ts`
 * for the `model` field on `Options`.
 *
 * NOTE: changing the model family may require matching changes in
 * agent-runner's `query()` call. The current default (Opus 4.8) and its
 * predecessor 4.7 both take `thinking: { type: 'adaptive', display:
 * 'summarized' }` (manual `type: 'enabled'` is rejected; `display` would
 * otherwise default to `'omitted'` and silently empty out thinking
 * content) and run on the env-driven `xhigh` effort, not `effort: 'max'`.
 * The runner is set up for these expectations — re-verify them before a
 * cross-family bump (Sonnet/Haiku already degrade `xhigh` → `high`
 * gracefully in the SDK).
 */
// Operators can override at deploy time without editing source — handy for
// running a fork on a cheaper model (Sonnet) without forking just to change
// this one constant. If unset, the default below is what the upstream
// runner is tuned for. AGENT_EFFORT (alongside this) is already env-
// overridable in the agent-runner via VALID_AGENT_EFFORTS.
//
// The `[1m]` suffix on the default model selects the 1M-token extended-
// context tier. Long conversations and large per-turn payloads (transcript
// archives, multi-message digests) rely on it; if you override AGENT_MODEL
// to a different model that supports extended context, include `[1m]` to
// match. Models without the suffix run the standard context window and
// will surface as truncation / earlier compaction in long sessions.
//
// Light validation: trim whitespace (so `AGENT_MODEL="  "` falls back to
// the default rather than passing two spaces to the SDK) and warn on
// values that don't look like a Claude model ID. We don't enumerate a
// whitelist because the SDK accepts both aliases (`opus`, `sonnet[1m]`)
// and full IDs (`claude-opus-4-7[1m]`), the set churns with each model
// release, and a missed model would block legit upgrades. The warn
// surfaces typos at startup instead of at first `query()` call deep in
// runtime — a typo like `claud-opus-4-7` is operator-error territory but
// cheap to flag.
/**
 * @internal Exported so tests can assert against the same literal the
 *   helper returns, instead of duplicating the string in two places where
 *   a default-model bump could silently drift.
 */
export const DEFAULT_AGENT_MODEL = 'claude-opus-4-8[1m]';
const KNOWN_MODEL_PREFIX_RE = /^(claude|opus|sonnet|haiku)/i;
export function resolveAgentModel(raw: string | undefined): string {
  const trimmed = raw?.trim();
  if (!trimmed) return DEFAULT_AGENT_MODEL;
  if (!KNOWN_MODEL_PREFIX_RE.test(trimmed)) {
    logger.warn(
      { agentModel: trimmed, fallback: DEFAULT_AGENT_MODEL },
      'AGENT_MODEL does not look like a Claude model ID — will pass to SDK as-is, but check for a typo. Expected forms: full ID like "claude-opus-4-7[1m]" or alias like "opus" / "sonnet[1m]".',
    );
  }
  return trimmed;
}

/**
 * Per-trust-tier base model (#613 Stage 1 — Claude tier-down). The base
 * model for EVERY container spawn — inbound chat AND scheduled/maintenance
 * runs alike — keyed to the spawn's trust tier instead of every tier
 * defaulting to Opus:
 *
 *   - main      → the global default (`AGENT_MODEL`) — quality floor kept
 *   - trusted   → Sonnet 4.6 (near-Opus reasoning, materially cheaper)
 *   - untrusted → Haiku 4.5 (low-stakes; keeps the Claude prompt-cache
 *                 discount on hostile content)
 *
 * Same-family Claude SKUs, so the agent-runner's `query()` config (tuned
 * for Opus, with `xhigh` effort gracefully falling back on Sonnet/Haiku)
 * needs no change. No `[1m]` suffix on the cheaper tiers — the extended-
 * context tier is reserved for main's long sessions.
 *
 * This is the BASE for a spawn: per-group / maintenance / per-task
 * overrides (`resolveSessionAgentModel`) still win on top. Scheduled-task
 * model selection (trivial → Haiku, substantive → Sonnet) is set per-task
 * via each skill's `agentModel:` frontmatter, not here.
 *
 * Exported so tests pin the tier→model mapping independently.
 */
export const TRUSTED_TIER_MODEL = 'claude-sonnet-4-6';
export const UNTRUSTED_TIER_MODEL = 'claude-haiku-4-5-20251001';
export function resolveTierBaseModel(
  isMain: boolean,
  trusted: boolean,
  globalDefault: string,
): string {
  if (isMain) return globalDefault;
  if (trusted) return TRUSTED_TIER_MODEL;
  return UNTRUSTED_TIER_MODEL;
}

/**
 * Resolve a per-group `containerConfig.agentModel` override to the value
 * actually forwarded to the spawned container (#395). Stricter than the
 * global `resolveAgentModel`:
 *
 *   - empty / whitespace-only / undefined / null → fall back to `fallback`
 *     (the global AGENT_MODEL). Same intent as the global helper, so an
 *     operator that clears the per-group field via `set_agent_model`
 *     `null` reverts the group to the global default rather than
 *     surfacing an empty string downstream.
 *
 *   - unknown-prefix value (`'foobar'`, `'claud-opus-4-7'`) → ALSO fall
 *     back to `fallback`, with a warn. The global helper passes
 *     unknown-prefix values through to surface typos at startup; per-
 *     group overrides are set at runtime via IPC by an agent (or
 *     operator) and there's no operator-driven startup audit log to
 *     catch a typo before it kills the next spawn for that group.
 *     Failing closed to the global default keeps the group running
 *     while the warn flags the bad value.
 *
 * Returns the trimmed override on a known prefix, or `fallback` in all
 * other cases. Exported so tests can pin the four branches independently
 * of the global `resolveAgentModel` contract.
 */
export function resolvePerGroupAgentModel(
  raw: string | undefined | null,
  fallback: string,
): string {
  const trimmed = typeof raw === 'string' ? raw.trim() : '';
  if (!trimmed) return fallback;
  if (!KNOWN_MODEL_PREFIX_RE.test(trimmed)) {
    // Phase 3 (#509) callers pass `fallback` = the session-level value
    // (e.g. `maintenanceAgentModel` resolved against the user-facing
    // value), not the global default. The warn message intentionally
    // says "the caller-supplied fallback" rather than "global default"
    // — debugging "I set task X to haik and got Sonnet" is harder when
    // the log claims the value was sent to the global default while
    // the actual fallback was the maintenance override. The `fallback`
    // field in the log payload carries the actual value the caller
    // will route to.
    logger.warn(
      { agentModel: trimmed, fallback },
      'Per-group AGENT_MODEL override does not look like a Claude model ID — falling back to the caller-supplied fallback (see `fallback` field). Expected forms: full ID like "claude-opus-4-7[1m]" or alias like "opus" / "sonnet[1m]".',
    );
    return fallback;
  }
  return trimmed;
}

/**
 * Resolve the effective AGENT_MODEL for a single spawn given the
 * session slot (#509). Per-session-slot model tier with optional
 * per-task override (Phase 3) — currently only the maintenance slot
 * has its own session-level override; user-facing `'default'` (and
 * any future named slot) falls through to `agentModel` → AGENT_MODEL
 * → DEFAULT_AGENT_MODEL.
 *
 * Resolution order (highest precedence first):
 *   1. `taskAgentModel` (Phase 3) — per-row override from the
 *      `scheduled_tasks.agent_model` column. Beats every other knob;
 *      fires for any spawn that carries a task-level value (in
 *      practice always a maintenance-session scheduled-task fire).
 *      Validated through `resolvePerGroupAgentModel` against the
 *      maintenance/user-facing-resolved value as the fallback so a
 *      typo doesn't silently jump past the operator's session-level
 *      override.
 *   2. `containerConfig.maintenanceAgentModel` (Phase 2) — applies
 *      only to maintenance spawns. Validated against the
 *      user-facing-resolved value as the fallback.
 *   3. `containerConfig.agentModel` (#395) — per-group override.
 *      Validated against `globalDefault`.
 *   4. `globalDefault` — the orchestrator's `AGENT_MODEL` env.
 *
 * Non-maintenance spawns skip step 2 but otherwise follow the same
 * ladder.
 *
 * Returns `{ effective, source }` so the per-spawn audit log line
 * (#418) can attribute the value to the right config layer without
 * the caller re-deriving the comparison. `task_override` is the
 * new Phase 3 source-tag; the prior three values are unchanged.
 */
export function resolveSessionAgentModel(
  containerConfig:
    | { agentModel?: string; maintenanceAgentModel?: string }
    | undefined,
  isMaintenance: boolean,
  globalDefault: string,
  taskAgentModel?: string | null,
): {
  effective: string;
  source:
    | 'global_default'
    | 'group_override'
    | 'maintenance_override'
    | 'task_override';
} {
  const userFacingRaw = containerConfig?.agentModel;
  const userFacingResolved = userFacingRaw
    ? resolvePerGroupAgentModel(userFacingRaw, globalDefault)
    : globalDefault;
  const userFacingSource: 'group_override' | 'global_default' =
    userFacingResolved !== globalDefault ? 'group_override' : 'global_default';

  // Compute the session-level value first (step 2 if maintenance, else
  // step 3) so the per-task override has a coherent fallback when its
  // own raw value fails resolvePerGroupAgentModel's prefix check.
  let sessionLevelResolved = userFacingResolved;
  let sessionLevelSource:
    | 'group_override'
    | 'global_default'
    | 'maintenance_override' = userFacingSource;
  if (isMaintenance) {
    const maintenanceRaw = containerConfig?.maintenanceAgentModel;
    if (maintenanceRaw && maintenanceRaw.trim()) {
      const maintenanceResolved = resolvePerGroupAgentModel(
        maintenanceRaw,
        userFacingResolved,
      );
      if (maintenanceResolved !== userFacingResolved) {
        sessionLevelResolved = maintenanceResolved;
        sessionLevelSource = 'maintenance_override';
      }
      // else: unknown-prefix fall-through warned by resolvePerGroupAgentModel,
      // OR maintenance value deliberately matches user-facing — either way
      // there's no effective maintenance-specific routing, so leave the
      // session-level pair at the user-facing values.
    }
  }

  // Step 1 — per-task override beats everything else. A typo / unknown
  // prefix falls back through resolvePerGroupAgentModel to the
  // session-level value, NOT all the way to globalDefault — so an
  // operator who already set a deliberate maintenance override sees a
  // bad per-task value land on maintenance, not on the global default.
  if (taskAgentModel && taskAgentModel.trim()) {
    const taskResolved = resolvePerGroupAgentModel(
      taskAgentModel,
      sessionLevelResolved,
    );
    if (taskResolved !== sessionLevelResolved) {
      return { effective: taskResolved, source: 'task_override' };
    }
    // Unknown-prefix or deliberately-matches-session — no effective
    // task-specific routing happening; emit the session-level pair.
  }

  return { effective: sessionLevelResolved, source: sessionLevelSource };
}
