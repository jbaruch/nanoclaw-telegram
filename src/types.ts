export interface AdditionalMount {
  hostPath: string; // Absolute path on host (supports ~ for home)
  containerPath?: string; // Optional — defaults to basename of hostPath. Mounted at /workspace/extra/{value}
  readonly?: boolean; // Default: true for safety
}

/**
 * Mount Allowlist - Security configuration for additional mounts
 * This file should be stored at ~/.config/nanoclaw/mount-allowlist.json
 * and is NOT mounted into any container, making it tamper-proof from agents.
 */
export interface MountAllowlist {
  // Directories that can be mounted into containers
  allowedRoots: AllowedRoot[];
  // Glob patterns for paths that should never be mounted (e.g., ".ssh", ".gnupg")
  blockedPatterns: string[];
  // If true, non-main groups can only mount read-only regardless of config
  nonMainReadOnly: boolean;
}

export interface AllowedRoot {
  // Absolute path or ~ for home (e.g., "~/projects", "/var/repos")
  path: string;
  // Whether read-write mounts are allowed under this root
  allowReadWrite: boolean;
  // Optional description for documentation
  description?: string;
}

export interface ContainerConfig {
  additionalMounts?: AdditionalMount[];
  timeout?: number; // Default: 300000 (5 minutes)
  trusted?: boolean; // Trusted groups get limited credentials (e.g. voice transcription)
  /**
   * Opt this non-main group into the 15-min unanswered-message heartbeat.
   * Default: undefined / false — no heartbeat. The main group always gets
   * a heartbeat regardless of this flag (handled separately in
   * `registerGroup`). Made explicit by #158 to kill the historical
   * "trigger-required → auto-heartbeat" coupling.
   */
  enableHeartbeat?: boolean;
  /**
   * Per-group AGENT_MODEL override (#395). When set, the orchestrator
   * forwards this value as the `AGENT_MODEL` env var on container spawn
   * for this group only, in place of the global default. Accepts the
   * same forms as the global env (`opus`, `sonnet[1m]`,
   * `claude-opus-4-7[1m]`, etc.). Validated through
   * `resolvePerGroupAgentModel`: an unknown-prefix value falls back to
   * the global default rather than passing through, so a fat-fingered
   * per-group override can't silently route to a non-existent model and
   * crash every spawn for that group.
   *
   * Undefined / null / empty / whitespace-only → use the global
   * AGENT_MODEL (preserves pre-#395 behaviour for groups that don't opt
   * in). Cleared via the `set_agent_model` IPC with `agentModel: null`.
   */
  agentModel?: string;
  /**
   * Per-session-slot model override for the maintenance session
   * (`jbaruch/nanoclaw#509`). When set AND the spawn's `sessionName`
   * is `'maintenance'`, this value replaces the resolved model for
   * that single spawn — leaving the user-facing `'default'` session
   * (and any future named slots) on the existing `agentModel` →
   * `AGENT_MODEL` → `DEFAULT_AGENT_MODEL` ladder.
   *
   * Same value forms accepted as `agentModel`; same validation
   * (`resolvePerGroupAgentModel`) — an unknown-prefix value falls
   * back to whatever `agentModel` would have resolved to (NOT the
   * global default), so the user-facing slot's intentional override
   * isn't silently bypassed by a fat-fingered maintenance value.
   *
   * Use case: scheduled maintenance work (heartbeat, nightly-*,
   * memory-rotation, etc.) does triage / orchestration that Sonnet
   * handles equivalently to Opus, while the user-facing session
   * stays on Opus for full reasoning headroom. Per `#509`'s
   * `logs/usage.jsonl` analysis, ~99.8% of the maintenance container's
   * spend was on `claude-opus-4-7` despite the maintenance work
   * profile being lighter than user-facing chat.
   *
   * Undefined / null / empty / whitespace-only → maintenance falls
   * back to `agentModel` (or its fallbacks). Cleared via
   * `set_maintenance_agent_model` IPC with `maintenanceAgentModel: null`.
   */
  maintenanceAgentModel?: string;
  /**
   * Per-group opt-in for the per-tier custom system prompt (#113).
   * When `true`, the agent-runner is launched with `USE_CUSTOM_PROMPT=1`
   * and reads `/workspace/global/prompts/{main,trusted,untrusted}.md`
   * instead of the SDK's `claude_code` preset. When `false`, force OFF
   * even if the global `USE_CUSTOM_PROMPT_FOR_MAIN` env is set.
   *
   * Default (undefined): defer to the global env. The global env only
   * enables custom prompts for the main-tier container; trusted/untrusted
   * stay on the preset until per-group opt-in.
   *
   * Reversible: unsetting reverts to the preset path on the next spawn.
   */
  useCustomPrompt?: boolean;
  /**
   * Per-group session turn-cap override (#561). When set to a positive
   * integer, this group's session-length turn cap replaces the global
   * `SESSION_TURN_CAP` for reset decisions (`shouldMarkForReset`). The
   * global cap is one knob and must cover the busiest group; this lets a
   * quiet group run a tighter cap — freeing context (and dropping
   * maintenance spend) sooner — without guillotining a group that
   * legitimately runs long.
   *
   * The `set_session_caps` IPC write path enforces a positive **integer**
   * (turns are discrete); the read-time resolver (`resolveSessionCaps`)
   * is the more lenient validation boundary for a hand-edited
   * `container_config` row — it accepts any positive finite value and
   * inherits the global on undefined / null / non-finite / non-positive /
   * non-number. A fat-fingered override can't silently disable or zero a
   * group's cap — it falls back, mirroring `resolvePerGroupAgentModel`.
   *
   * Independent of the global enable state: a positive override caps this
   * group even when the global cap is disabled (global `<= 0`) — the
   * override means "this group wants a cap", not "tighten the global".
   * Disabling a cap stays a global-only operation.
   *
   * Cleared via the `set_session_caps` IPC with `sessionTurnCap: null`.
   */
  sessionTurnCap?: number;
  /**
   * Per-group session token-cap override (#561). Same resolution and
   * fallback semantics as `sessionTurnCap`, applied to the cumulative
   * `total_input_tokens` cap (global `SESSION_TOKEN_CAP`). Cleared via
   * the `set_session_caps` IPC with `sessionTokenCap: null`.
   */
  sessionTokenCap?: number;
  /**
   * Host-side Stage 1 gate chain (#80). Names of gates from
   * `src/gates/index.ts` registry, evaluated in `gateNames` order
   * (the per-group config order, not registration order). Combinator
   * is last-gate-wins with allow short-circuit (#99): an `allow`
   * exits the chain; intermediate `deny` is advisory unless it's the
   * final verdict; all-`pass` falls open to allow. Empty/undefined
   * preserves pre-#80 behaviour (no gating beyond the legacy
   * `requiresTrigger` boolean — see backwards-compat shim in
   * `src/index.ts`).
   */
  gates?: string[];
  /**
   * Legacy Stage 2 classifier opt-in. The Stage 2 Haiku classifier was
   * removed, so this flag no longer selects any gate — it is retained
   * for config-shape compatibility with stored rows (new groups still
   * default it to `true` via `applyNewGroupContainerConfigDefaults`) and
   * is inert until a follow-up prunes it.
   */
  stage2Enabled?: boolean;
  /**
   * Legacy Stage 2 classifier model override. Inert since the Stage 2
   * Haiku classifier was removed; retained for config-shape
   * compatibility until a follow-up prunes it.
   */
  stage2ModelId?: string;
  /**
   * Legacy Stage 2 classifier context-strategy selector. Inert since
   * the Stage 2 Haiku classifier was removed; retained for config-shape
   * compatibility until a follow-up prunes it.
   */
  stage2ContextStrategy?: string;
  /**
   * Per-chat additive tile overlay (#305). Tile names from the local
   * registry under `tessl-workspace/.tessl/tiles/<TILE_OWNER>/` that
   * load IN ADDITION TO the trust-tier baseline (`selectTiles`),
   * never as a replacement. Empty / undefined → tier-tiles only.
   *
   * Validation is fail-closed at both ends:
   *  - Write-time (`set_additional_tiles` IPC): every entry must
   *    resolve to an installed tile or the write is rejected with a
   *    diagnostic log line.
   *  - Spawn-time (`startContainerSession`): if any entry fails to
   *    resolve at spawn (e.g. a registry rebuild dropped it after the
   *    write), the container refuses to spawn rather than silently
   *    losing capabilities.
   *
   * Order is preserved; duplicates against the baseline are dropped so
   * `selectTiles` returns a stable de-duplicated install order
   * (baseline first, additions appended).
   */
  additionalTiles?: string[];
}

/**
 * Provenance of a single trigger pattern. Used by the self-improvement loop
 * (#82) to decide whether a pattern is owner-locked, learned by the agent,
 * or part of a universal default set.
 *
 * - `owner-set`: explicitly configured by the group owner (e.g. via
 *   `setup/register.ts --trigger`). Never overwritten by automated updates.
 * - `learned`: added by the self-improvement loop based on observed
 *   precision. Subject to demotion / removal by the same loop.
 * - `universal`: shipped defaults that apply to every group of a given
 *   shape (e.g. the always-on `@<assistant>` keyword).
 */
export type TriggerPatternSource = 'owner-set' | 'learned' | 'universal';

/**
 * Kind of pattern. Loose taxonomy that matches how the orchestrator
 * actually decides to wake up:
 *
 * - `keyword`: literal substring matched at word boundary
 *   (current `buildTriggerPattern` behaviour).
 * - `mention`: explicit @-mention or channel-native ping.
 * - `reply`: triggered because the message replies to one of ours.
 * - `regex`: free-form regex pattern (advanced).
 * - `sender_tier`: triggered by sender belonging to a privileged tier
 *   (reserved for #82; not consumed by the matcher today).
 */
export type TriggerPatternKind =
  | 'keyword'
  | 'mention'
  | 'reply'
  | 'regex'
  | 'sender_tier';

/**
 * One trigger pattern entry. The observability fields (`precision`,
 * `sample_count`, `last_matched_at`, `last_updated_at`) are written
 * through helpers; #81 itself only stores them — the self-improvement
 * loop in #82 is what populates them.
 */
export interface TriggerPattern {
  /** Pattern body. Interpretation depends on `kind`. */
  pattern: string;
  kind: TriggerPatternKind;
  source: TriggerPatternSource;
  /**
   * Running precision in [0, 1]. Initialised to 0 on insert. Re-computed
   * by the self-improvement loop from `sample_count` and observed true
   * positives. Default 0 means "no signal yet" — readers must treat
   * absent / 0 as "no opinion", not "definitely bad".
   */
  precision: number;
  /** Total number of triggers attributed to this pattern. */
  sample_count: number;
  /** ISO timestamp of the most recent match, or null if never matched. */
  last_matched_at: string | null;
  /** ISO timestamp of the most recent metric update, or null. */
  last_updated_at: string | null;
  /**
   * Self-improvement loop fields (#82). All optional so legacy rows
   * without them remain valid `TriggerPattern` instances. Owner/universal
   * patterns may also leave them unset — they only matter for `source:
   * 'learned'` rows the loop manages. See `src/gates/trigger-learner-schema.md`
   * for the full writer/reader contract.
   */
  /**
   * Pattern lineage version. Starts at 1 on first proposal, increments
   * each time the learner supersedes the same logical pattern (e.g. body
   * stays the same, precision metrics refresh). The owner can revert by
   * picking an older `prior_versions` entry. Optional / unset for
   * non-learned rows.
   */
  pattern_version?: number;
  /** ISO timestamp the learner first proposed this pattern. */
  proposed_at?: string;
  /**
   * `false` (default / unset) means active per the gate matcher; `true`
   * means demoted by auto-rollback (precision dropped below threshold).
   * Demoted rows stay in the array — the owner may re-enable manually
   * after inspecting why the loop demoted them.
   */
  disabled?: boolean;
  /**
   * Owner-controlled gate: when `true`, learned proposals start active
   * and the matcher consumes them on the next gate run. When `false` /
   * unset, learned proposals are inert (pure proposals; the owner
   * promotes via the existing admin path). The trigger gate skips any
   * pattern with `enabled: false`.
   */
  enabled?: boolean;
  /**
   * Snapshot of the immediately-prior version of THIS pattern, kept so
   * the owner can revert without losing the metrics. Capped at one
   * level deep (older history is dropped — keeping a full chain
   * unbounded would let the column grow without limit). Recursive type
   * is fine because the inner record is always the prior state, never
   * a forward-pointer.
   */
  prior_versions?: TriggerPattern[];
}

/**
 * Schema-versioned wrapper for the trigger pattern set on a registered
 * group. Stored as JSON in `registered_groups.trigger_pattern`. Reads
 * are dual-mode (legacy string also accepted) until the backfill
 * migration has run on every install — see `parseTriggerPatternColumn`
 * in db.ts.
 */
export interface TriggerPatternConfig {
  version: 1;
  patterns: TriggerPattern[];
}

export interface RegisteredGroup {
  name: string;
  folder: string;
  /**
   * Derived legacy trigger string. Drives `getTriggerPattern(group.trigger)`
   * for the existing matching code paths (orchestrator, telegram channel,
   * session-commands). Derivation order (see `deriveTriggerString` in
   * `db.ts`): first `keyword`-kind entry verbatim → first `mention`-kind
   * entry with `@` re-prepended → `null`.
   *
   * `null` means "no group-specific trigger string" — call sites coerce
   * to `undefined` and let `getTriggerPattern` fall back to the global
   * default `@<assistant>` regex. The `null` sentinel replaces the
   * earlier ambiguous empty-string return for #82-style configs that
   * contain only non-keyword/non-mention patterns (e.g. regex,
   * sender_tier). See review of #84, comment-id 4360940433.
   */
  trigger: string | null;
  added_at: string;
  containerConfig?: ContainerConfig;
  requiresTrigger?: boolean; // Default: true for groups, false for solo chats
  isMain?: boolean; // True for the main control group (no trigger, elevated privileges)
  /**
   * Full pattern set with provenance + observability. Optional during
   * the dual-mode transition: legacy string-only rows surface here as
   * an auto-derived single-element config (`{kind: "keyword", source:
   * "owner-set"}`) to keep readers monomorphic.
   */
  triggerPatterns?: TriggerPatternConfig;
}

export interface NewMessage {
  id: string;
  chat_jid: string;
  sender: string;
  sender_name: string;
  content: string;
  timestamp: string;
  is_from_me?: boolean;
  is_bot_message?: boolean;
  thread_id?: string;
  reply_to_message_id?: string;
  reply_to_message_content?: string;
  reply_to_sender_name?: string;
  // Telegram-native message ID, populated for BOTH directions on
  // Telegram (#691) so reply_to_message_id has a single column to join
  // against. Outbound bot sends need it because the `id` column holds
  // our synthetic `bot-<ts>-<rand>`; inbound rows also stamp it (via the
  // channel's `deliverInbound`) even though `id` already carries the
  // Telegram ID, so a direct SQL consumer never has to branch on
  // direction. NULL for non-Telegram channels (their platform ID lives
  // in `id`). Queryable for "what's at Telegram ID X in chat Y?".
  //
  // Optional + NULL-able: writers may omit (column still defaults to
  // NULL via `?? null` in storeMessage), and DB getters surface the
  // persisted NULL as `null` IF their SELECT list includes the
  // column. The three runtime states:
  //   - `undefined` — writer didn't provide a value (normalized to
  //     NULL by storeMessage before persisting), OR reader loaded
  //     from a query whose explicit SELECT list doesn't include
  //     this column (e.g. `getNewMessages` / `getMessagesSince` in
  //     src/db.ts — they project a fixed subset of fields).
  //   - `null` — column was selected and the row's stored value is
  //     SQL NULL (non-Telegram row, or a legacy Telegram row not yet
  //     reached by the #691 backfill).
  //   - `string` — the Telegram message ID.
  // Call sites: writers with a known id pass a string; writers
  // without it omit; readers may see undefined / null / string
  // depending on the SELECT they went through.
  telegram_message_id?: string | null;
}

/**
 * Provenance of a scheduled_tasks row. Drives the agent-runner's decision
 * to wrap the prompt in `<untrusted-input>` at fire time.
 * - 'owner':           host code or Baruch's direct tooling — trusted
 * - 'main_agent':      main group's agent scheduled it — trusted
 * - 'trusted_agent':   trusted non-main group's agent — trusted
 * - 'untrusted_agent': untrusted group's agent — NOT trusted, wrap applies
 * The untrusted_agent case is the reason this field exists: without it,
 * an untrusted agent could self-schedule a malicious prompt that later
 * fires unwrapped and bypasses the trust boundary.
 */
export type CreatedByRole =
  | 'owner'
  | 'main_agent'
  | 'trusted_agent'
  | 'untrusted_agent';

export interface ScheduledTask {
  id: string;
  group_folder: string;
  chat_jid: string;
  prompt: string;
  script?: string | null;
  schedule_type: 'cron' | 'interval' | 'once';
  schedule_value: string;
  /**
   * IANA timezone for evaluating `cron` expressions (e.g. "UTC",
   * "America/Chicago"). Null/undefined = use the server's `TIMEZONE`
   * config at fire time, preserving pre-#102 behavior. Has no effect
   * on `interval` (always elapsed-ms) or `once` — for `once`, any
   * offset-suffixed ISO-8601 (`Z`, `+HH:MM`, `-HH:MM`) is treated as
   * an absolute instant; bare strings without a suffix are
   * interpreted in server-local time at schedule/update time and
   * pinned to the resulting UTC moment in `next_run`.
   */
  schedule_timezone?: string | null;
  context_mode: 'group' | 'isolated';
  next_run: string | null;
  last_run: string | null;
  last_result: string | null;
  status: 'active' | 'paused' | 'completed';
  created_at: string;
  created_by_role: CreatedByRole;
  /**
   * Continuation marker for self-resuming cycles (#93/#130). NULL/undefined
   * for ordinary one-shot scheduled tasks. When set by the resumable-cycle
   * helper skill (in the `nanoclaw-admin` tile), the task-scheduler
   * surfaces the value to the spawned container as
   * `NANOCLAW_CONTINUATION=1` plus
   * `NANOCLAW_CONTINUATION_CYCLE_ID=<value>`. The calling skill (nightly /
   * weekly / morning-brief) checks the env var alongside a prompt-prefix
   * marker; both must agree to take the lock-skip continuation branch,
   * otherwise the run is treated as a fresh user invocation. A scheduler
   * that sets the env but mangles the prompt (or vice versa) therefore
   * fails closed instead of silently bypassing the two-phase lock.
   */
  continuation_cycle_id?: string | null;
  /**
   * Per-task SDK session id for #336. NULL/undefined for tasks that
   * have never fired, for once-tasks (out of scope), and for recurring
   * tasks immediately after a `nukeSession` clear. Populated by
   * `runTask` on first fire of a recurring task and reused as
   * `resume:` on subsequent fires so the API caches the per-session
   * message-history prefix across the (otherwise expiring) prompt-
   * cache window. The #193 cross-task bleed concern doesn't apply —
   * persistence is keyed on `task_id`, so different tasks have
   * different rows hence different sessions hence no bleed.
   */
  session_id?: string | null;
  /**
   * Plugin-registry content hash paired with `session_id` (#710).
   * Written by `setTaskSessionId` in the same UPDATE, cleared by every
   * `session_id` clear path. At fire time the scheduler compares it
   * against the current registry hash and rotates to a fresh SDK
   * session on mismatch — a resumed session never re-reads skill/rule
   * content, so this is the only surface that lets a plugin fix reach
   * a pinned cadence session. NULL = hash unknown (registry absent or
   * vanished mid-walk during a registry swap at persist time, or the
   * id predates #710).
   */
  session_plugins_hash?: string | null;
  /**
   * Per-task AGENT_MODEL override for #509 Phase 3. NULL/undefined =
   * no override; fall through to the Phase 2 ladder
   * (`maintenanceAgentModel` → group `agentModel` → `AGENT_MODEL` env →
   * `DEFAULT_AGENT_MODEL`). When set AND the resolved value differs
   * from the session-level fallback, the spawn routes through it and
   * the audit log emits `source: 'task_override'`. When the value is
   * unknown-prefix (typo) or deliberately matches the session-level
   * fallback, `resolveSessionAgentModel` returns the session-level
   * source instead — the column is set but had no effective routing
   * impact, so the audit log doesn't lie about what changed the spawn.
   * Accepts the same shape the existing knobs accept (full ID like
   * `'claude-haiku-4-5-20251001'` or alias like `'haiku'` /
   * `'sonnet[1m]'`); unknown-prefix values fall back to the session-
   * level value (NOT the global default) via `resolvePerGroupAgentModel`.
   * Populated by cadence-registry's `agentModel:` frontmatter
   * (declarative — the rebuild is authoritative for `source =
   * 'cadence-registry'` rows; the IPC handler refuses writes to those
   * rows so a runtime override can't silently revert on the next
   * tile-touching spawn) or via the `set_task_agent_model` IPC
   * (imperative — only valid for `source = 'schedule-task'` rows).
   */
  agent_model?: string | null;
}

export interface TaskRunLog {
  task_id: string;
  run_at: string;
  duration_ms: number;
  // 'killed' (#496) marks a run that was force-terminated by the host
  // (e.g. `tessl_update`'s `_close` sentinel + agent-runner watchdog
  // fired, exit code 0 from the watchdog, but the task didn't actually
  // produce a complete result). Distinct from 'error' (the run threw)
  // and 'success' (the run finished cleanly) so operator-facing audits
  // can tell apart bookkeeping-success from semantic-success — exit
  // code 0 alone is no longer a guarantee the work landed.
  //
  // 'precheck_skipped' (#581) marks a fire whose precheck script
  // returned `wake_agent: false` — the agent never woke, the wrapper
  // never ran. Distinct from 'success' (the agent ran cleanly with no
  // output) so silent-success watchdogs querying `task_run_logs` can
  // tell a precheck-gated no-op (healthy quiet) apart from a wake-up
  // that left an empty result column (the original #581 silent-success
  // bug shape). A precheck script that crashes / emits non-JSON / omits
  // `wake_agent` is `'error'`, not `'precheck_skipped'`.
  status: 'success' | 'error' | 'killed' | 'precheck_skipped';
  result: string | null;
  error: string | null;
}

// --- Channel abstraction ---

export interface Channel {
  name: string;
  connect(): Promise<void>;
  sendMessage(
    jid: string,
    text: string,
    replyToMessageId?: string,
  ): Promise<string | void>;
  isConnected(): boolean;
  ownsJid(jid: string): boolean;
  disconnect(): Promise<void>;
  // Optional: typing indicator. Channels that support it implement it.
  setTyping?(jid: string, isTyping: boolean): Promise<void>;
  // Optional: sync group/chat names from the platform.
  syncGroups?(force: boolean): Promise<void>;
  // Optional: send an emoji reaction to a message.
  sendReaction?(jid: string, messageId: string, emoji: string): Promise<void>;
  // Optional: report whether a JID points at a 1:1 / DM chat (true) vs.
  // a multi-participant group / channel (false). Used by the observer
  // module to verify chat privacy when possible before mirroring
  // reasoning. A `false` result currently triggers a loud warning at
  // startup but still enables the observer (some operators run a
  // single-user private group as the observer chat); refusal is
  // reserved for cases where no channel owns the JID or chat-type
  // verification throws. Channels that can't determine this may leave
  // the method unimplemented — observer treats missing support as a
  // refusal, since "unknown" is the unsafe default for a leak surface.
  isPrivateChat?(jid: string): Promise<boolean>;
  // Optional: react to the most recent message in a chat.
  reactToLatestMessage?(jid: string, emoji: string): Promise<void>;
  // Optional: pin a message in the chat.
  pinMessage?(jid: string, messageId: string): Promise<void>;
  // Optional: send a file to the chat. Returns the channel-specific
  // message id when the send succeeds, undefined otherwise. Callers
  // gate post-send persistence (e.g. `storeMessage` for caption
  // accounting) on `result !== undefined` (NOT a truthy check —
  // empty-string and `'0'` ids are forward-compat valid per the
  // `shouldStoreBotMessage` contract) so a failed send doesn't
  // leave a phantom "answered" record in `messages.db` (#428).
  sendFile?(
    jid: string,
    filePath: string,
    caption?: string,
    replyToMessageId?: string,
  ): Promise<string | undefined>;
}

// Callback type that channels use to deliver inbound messages
export type OnInboundMessage = (chatJid: string, message: NewMessage) => void;

// Location capture (#574 Phase 3). Channels emit one event per
// location share or live-location update; the orchestrator persists
// to the `locations` table. Decoupled from `OnInboundMessage` because
// `edited_message:location` updates from Telegram's live-location
// stream are not conversational "messages" — they shouldn't pollute
// chat history, but they ARE the freshest signal of where the owner
// is for the host-side TZ resolver (replaces the TripIt-segment
// walker fallback path; Phase 2).
//
// `source` discriminates the four channel-level shapes:
//   - 'static' — one-time location pin (no live_period)
//   - 'venue'  — Telegram `message:venue` with `location` nested
//   - 'live_initial' — first event of a live share (`message:location`
//                       with `live_period > 0`)
//   - 'live_update'  — `edited_message:location` updates against an
//                       active live share; same `message_id` as
//                       the corresponding `live_initial` row
export type LocationSource =
  | 'static'
  | 'venue'
  | 'live_initial'
  | 'live_update';

export interface LocationRecord {
  chat_jid: string;
  sender: string;
  message_id: string;
  latitude: number;
  longitude: number;
  accuracy_m?: number | null;
  source: LocationSource;
  // ISO-8601 UTC, NOT a wall-clock at receive time. The source field
  // determines which Telegram timestamp this maps to:
  //   - 'static' / 'venue' / 'live_initial' → `message.date`
  //     (original send time of the location/venue/share message)
  //   - 'live_update' → `editedMessage.edit_date`
  //     (the movement-tick time; `editedMessage.date` is the original
  //     share time the Bot API echoes on every edit, and using it
  //     would freeze recorded_at and defeat the Phase 2 freshness gate)
  recorded_at: string;
  live_period?: number | null; // seconds; populated only when source ∈ {live_initial, live_update}
}

export type OnLocation = (record: LocationRecord) => void;

// Callback for chat metadata discovery.
// name is optional — channels that deliver names inline (Telegram) pass it here;
// channels that sync names separately (via syncGroups) omit it.
export type OnChatMetadata = (
  chatJid: string,
  timestamp: string,
  name?: string,
  channel?: string,
  isGroup?: boolean,
) => void;
