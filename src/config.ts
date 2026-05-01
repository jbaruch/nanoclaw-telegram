import os from 'os';
import path from 'path';

import { readEnvFile } from './env.js';
import { isValidTimezone } from './timezone.js';

// Read config values from .env (falls back to process.env).
// Secrets (API keys, tokens) are NOT read here — they are loaded only
// by the credential proxy (credential-proxy.ts), never exposed to containers.
const envConfig = readEnvFile([
  'ASSISTANT_NAME',
  'ASSISTANT_USERNAME',
  'ASSISTANT_HAS_OWN_NUMBER',
  'TZ',
  'TELEGRAM_BOT_POOL',
  'TILE_OWNER',
  'HUBITAT_HUB_IP',
  'HUBITAT_APP_ID',
  'HUBITAT_EVENT_RETENTION_DAYS',
  'HUBITAT_ALERT_SENSITIVITY',
  'MAINTENANCE_RULE_BLOCKLIST',
  'MAINTENANCE_SKILL_BLOCKLIST',
]);

export const ASSISTANT_NAME =
  process.env.ASSISTANT_NAME || envConfig.ASSISTANT_NAME || 'Andy';
// Telegram @-handle without leading `@`. Lowercased ASSISTANT_NAME is
// the safe default — most deployments mirror the display name in the
// handle. Override via ASSISTANT_USERNAME when the bot's handle differs
// (e.g. ASSISTANT_NAME=AyeAye + ASSISTANT_USERNAME=AyeAyeSureBot for the
// dual-handle pattern). Read by container-runner.ts and forwarded into
// every agent container so agent-runner can prepend the authoritative
// identity preamble (#407 / ligolnik/nanoclaw-public#90).
export const ASSISTANT_USERNAME =
  process.env.ASSISTANT_USERNAME ||
  envConfig.ASSISTANT_USERNAME ||
  ASSISTANT_NAME.toLowerCase();
export const ASSISTANT_HAS_OWN_NUMBER =
  (process.env.ASSISTANT_HAS_OWN_NUMBER ||
    envConfig.ASSISTANT_HAS_OWN_NUMBER) === 'true';
export const TELEGRAM_BOT_POOL = (
  process.env.TELEGRAM_BOT_POOL ||
  envConfig.TELEGRAM_BOT_POOL ||
  ''
)
  .split(',')
  .map((t) => t.trim())
  .filter(Boolean);
export const POLL_INTERVAL = 2000;
export const SCHEDULER_POLL_INTERVAL = 60000;

// Absolute paths needed for container mounts
const PROJECT_ROOT = process.cwd();
const HOME_DIR = process.env.HOME || os.homedir();

// Docker-out-of-Docker: when the orchestrator runs inside a container,
// mount paths (-v) must reference the HOST filesystem, not the container's.
// Set HOST_PROJECT_ROOT in docker-compose.yml to the repo path on the host.
// When running directly on the host (e.g., Mac), this defaults to cwd().
export const HOST_PROJECT_ROOT = process.env.HOST_PROJECT_ROOT || PROJECT_ROOT;

// In DooD, process.getuid() returns the orchestrator container's uid (1000).
// HOST_UID/HOST_GID env vars override this with the actual host user's uid/gid.
//
// Validation: a set-but-malformed value (`HOST_UID=foo`, `-1`, `1.5`,
// `123abc`, or empty string) resolves to `undefined` here so
// downstream chown sites fall through to their default branch
// (Mac-host posture / chown to uid 1000) instead of forwarding the
// malformed value into `fs.chownSync` — `NaN` throws there, `-1`
// casts to uid 4294967295 and silently mis-owns. A stderr line
// surfaces the operator typo at startup; without it, the misconfig
// looks identical to "not running in DooD" and the original
// permission issue (#258) is invisible.
//
// Stderr is used directly rather than `logger` because this runs at
// module-evaluation time, before the orchestrator has wired any
// logger sinks. Stderr is always available and doesn't pull config
// into a tighter coupling with logger initialization order.
//
/**
 * @internal exported ONLY for `config.test.ts`. Re-importing the
 * module via `vi.resetModules()` to flip env values per case would
 * re-execute logger.ts each pass and leak
 * `process.on('uncaughtException')` / `unhandledRejection` handlers
 * (Node defaults to a max of 10 before warning). Direct invocation
 * with mutated `process.env` gives equivalent coverage of the
 * validation contract without the leak. `stripInternal: true` keeps
 * this out of generated `.d.ts` so the public API stays minimal.
 */
export function parseHostId(name: 'HOST_UID' | 'HOST_GID'): number | undefined {
  const raw = process.env[name];
  if (raw === undefined) return undefined;
  // Strict digits-only match: `parseInt` would silently accept partial
  // parses (`"123abc"` → 123, `"1.5"` → 1) and `!raw` would treat an
  // explicit empty string as "unset" — both shapes are operator typos
  // we want to surface, not absorb.
  if (!/^\d+$/.test(raw)) {
    process.stderr.write(
      `[config] ${name}="${raw}" is not a non-negative integer — ignoring; chowns to host user will fall back to default uid/gid.\n`,
    );
    return undefined;
  }
  return parseInt(raw, 10);
}
export const HOST_UID = parseHostId('HOST_UID');
export const HOST_GID = parseHostId('HOST_GID');

// Mount security: allowlist stored OUTSIDE project root, never mounted into containers
export const MOUNT_ALLOWLIST_PATH =
  process.env.MOUNT_ALLOWLIST_PATH ||
  path.join(HOME_DIR, '.config', 'nanoclaw', 'mount-allowlist.json');

// Local paths for filesystem operations (mkdirSync, existsSync, etc.)
export const STORE_DIR = path.resolve(PROJECT_ROOT, 'store');
export const GROUPS_DIR = path.resolve(PROJECT_ROOT, 'groups');
export const DATA_DIR = path.resolve(PROJECT_ROOT, 'data');

export const CONTAINER_IMAGE =
  process.env.CONTAINER_IMAGE || 'nanoclaw-agent:latest';
export const CONTAINER_TIMEOUT = parseInt(
  process.env.CONTAINER_TIMEOUT || '1800000',
  10,
);
export const CONTAINER_MAX_OUTPUT_SIZE = parseInt(
  process.env.CONTAINER_MAX_OUTPUT_SIZE || '10485760',
  10,
); // 10MB default
export const CREDENTIAL_PROXY_PORT = parseInt(
  process.env.CREDENTIAL_PROXY_PORT || '3001',
  10,
);
export const ONECLI_URL = process.env.ONECLI_URL || envConfig.ONECLI_URL;
export const MAX_MESSAGES_PER_PROMPT = Math.max(
  1,
  parseInt(process.env.MAX_MESSAGES_PER_PROMPT || '10', 10) || 10,
);
export const IPC_POLL_INTERVAL = 1000;
export const IDLE_TIMEOUT = parseInt(process.env.IDLE_TIMEOUT || '1800000', 10); // 30min default — how long to keep container alive after last result

// Kill-auto-compaction master flag (issue #104, design at
// docs/proposals/kill-auto-compaction.md). When OFF (default), the
// orchestrator collects token-usage telemetry and writes ## Facts
// checkpoints at threshold-cross — both side-effect-free observability —
// but does NOT pass DISABLE_COMPACT=1 to containers and does NOT fire
// the threshold-nuke handshake. Auto-compaction stays on; nothing
// destructive activates. When ON, DISABLE_COMPACT=1 is set on container
// spawn AND the threshold-cross handshake (system-reminder + grace
// timer + nuke_session + session-reentry) replaces auto-compaction
// end-to-end. The two halves are gated by the same flag because they
// must ship together — disabling compaction without the replacement
// leaves long sessions with no overflow safety net.
export const ENABLE_THRESHOLD_NUKE = process.env.ENABLE_THRESHOLD_NUKE === '1';

// Model context window in tokens. Mirrors the SDK env
// `CLAUDE_CODE_MAX_CONTEXT_WINDOW` we set per-session in
// container-runner.ts settings.json (currently 1,000,000 for Opus 4.7).
// The threshold formula reads this; if it ever drifts from the SDK env,
// the threshold will fire at the wrong percentage. Override via
// MODEL_CONTEXT_WINDOW for tests / smaller-context model bumps.
export const MODEL_CONTEXT_WINDOW = parseInt(
  process.env.MODEL_CONTEXT_WINDOW || '1000000',
  10,
);

// SDK auto-compact working window in tokens (issue #252). Forwarded to
// the agent-runner as `CLAUDE_CODE_AUTO_COMPACT_WINDOW` so the SDK's
// `Jn()` resolver clamps `min(model_default, this)` and uses it as the
// working window for both auto-compaction (when ENABLE_THRESHOLD_NUKE=0)
// and the blocking-limit check (always — `DISABLE_COMPACT=1` only
// suppresses `isAboveAutoCompactThreshold`, not `isAtBlockingLimit`).
//
// Default 800,000 leaves headroom on the 1M Opus window and lets the
// observe-only telemetry from #104 see realistic warn (700k) crossings
// before the SDK compacts. The previous upstream hardcode of 165,000
// (qwibitai/nanoclaw `f77f9ce`) capped real-world heartbeat cycles at
// ~16% of the paid-for context window and suppressed every #104
// threshold telemetry signal — see #252.
//
// IMPORTANT: when ENABLE_THRESHOLD_NUKE=1, the orchestrator does NOT
// forward this value (see container-runner.ts). Letting the SDK fall
// back to the model default keeps the blocking-limit (~window − 30k)
// well above the orchestrator's 800k nuke threshold; clamping it would
// drop blocking-limit below the nuke and wedge sends mid-handshake.
//
// Validation: a non-numeric / non-positive value would forward as
// `NaN`, which the SDK's `Lp()` validator rejects and silently falls
// back to model default — so the blast radius is limited, but a
// stderr warning surfaces operator typos at startup rather than at
// first `query()` deep in runtime. Same shape as `resolveAgentModel`
// in container-runner.ts (logger.warn there; stderr here because
// config.ts is below logger.ts in the import graph and a logger
// import would close a circular dep through host-logs.ts).
const DEFAULT_AGENT_AUTO_COMPACT_WINDOW = 800_000;
function resolveAgentAutoCompactWindow(): number {
  const raw = process.env.AGENT_AUTO_COMPACT_WINDOW;
  if (!raw) return DEFAULT_AGENT_AUTO_COMPACT_WINDOW;
  const parsed = parseInt(raw, 10);
  if (Number.isFinite(parsed) && Number.isInteger(parsed) && parsed > 0) {
    return parsed;
  }
  process.stderr.write(
    `[config] AGENT_AUTO_COMPACT_WINDOW="${raw}" is not a positive integer — falling back to default ${DEFAULT_AGENT_AUTO_COMPACT_WINDOW}.\n`,
  );
  return DEFAULT_AGENT_AUTO_COMPACT_WINDOW;
}
export const AGENT_AUTO_COMPACT_WINDOW = resolveAgentAutoCompactWindow();

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function buildTriggerPattern(trigger: string): RegExp {
  return new RegExp(`(?:^|\\s)${escapeRegex(trigger.trim())}\\b`, 'i');
}

export const DEFAULT_TRIGGER = `@${ASSISTANT_NAME}`;

export function getTriggerPattern(trigger?: string): RegExp {
  const normalizedTrigger = trigger?.trim();
  return buildTriggerPattern(normalizedTrigger || DEFAULT_TRIGGER);
}

export const TRIGGER_PATTERN = buildTriggerPattern(DEFAULT_TRIGGER);

// --- Hubitat Smart Home ---
export const HUBITAT_HUB_IP =
  process.env.HUBITAT_HUB_IP || envConfig.HUBITAT_HUB_IP || '';
export const HUBITAT_APP_ID =
  process.env.HUBITAT_APP_ID || envConfig.HUBITAT_APP_ID || '';
export const HUBITAT_EVENT_RETENTION_DAYS = parseInt(
  process.env.HUBITAT_EVENT_RETENTION_DAYS ||
    envConfig.HUBITAT_EVENT_RETENTION_DAYS ||
    '90',
  10,
);
export const HUBITAT_ALERT_SENSITIVITY = (process.env
  .HUBITAT_ALERT_SENSITIVITY ||
  envConfig.HUBITAT_ALERT_SENSITIVITY ||
  'low') as 'low' | 'medium' | 'high';

// Tile owner namespace for tessl registry (e.g., "jbaruch" → "jbaruch/nanoclaw-core")
export const TILE_OWNER =
  process.env.TILE_OWNER || envConfig.TILE_OWNER || 'nanoclaw';

// Timezone for scheduled tasks, message formatting, etc.
// Validates each candidate is a real IANA identifier before accepting.
function resolveConfigTimezone(): string {
  const candidates = [
    process.env.TZ,
    envConfig.TZ,
    Intl.DateTimeFormat().resolvedOptions().timeZone,
  ];
  for (const tz of candidates) {
    if (tz && isValidTimezone(tz)) return tz;
  }
  return 'UTC';
}
export const TIMEZONE = resolveConfigTimezone();

// --- Maintenance-class spawn blocklists (#337) ---
//
// At spawn time the orchestrator copies every installed tile's rules and
// skills into the container's `.tessl/` and `skills/` dirs (see
// `src/container-runner.ts` install loop). For non-conversational task
// classes the bulk of that content is dead weight that still pays full
// `cache_create` cost on the first turn of every fresh maintenance
// session. These blocklists let the orchestrator skip irrelevant items
// when the spawn's `sessionName === 'maintenance'`. Empty / unset =
// no filter (the regression-safe default).
//
// Format: comma-separated names. Whitespace and empty entries trimmed.
//   MAINTENANCE_RULE_BLOCKLIST  — rule filenames as they appear in
//     `tiles/<owner>/<tile>/rules/`, e.g. "skill-authoring.md,plugin-evals.md".
//   MAINTENANCE_SKILL_BLOCKLIST — skill directory names as they appear
//     in `tiles/<owner>/<tile>/skills/` and `container/skills/`, e.g.
//     "agent-browser,channel-formatting". The `tessl__` prefix added at
//     copy-into-container time is NOT part of the blocklist key — list
//     the bare directory name.
//
// Inbound user messages route to `'default'` and bypass the filter
// entirely; only the maintenance slot (heartbeat, nightly, weekly,
// reminders) sees a slimmed prompt.
function parseBlocklist(raw: string | undefined): Set<string> {
  if (!raw) return new Set();
  return new Set(
    raw
      .split(',')
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0),
  );
}

export const MAINTENANCE_RULE_BLOCKLIST = parseBlocklist(
  process.env.MAINTENANCE_RULE_BLOCKLIST ||
    envConfig.MAINTENANCE_RULE_BLOCKLIST,
);

export const MAINTENANCE_SKILL_BLOCKLIST = parseBlocklist(
  process.env.MAINTENANCE_SKILL_BLOCKLIST ||
    envConfig.MAINTENANCE_SKILL_BLOCKLIST,
);
