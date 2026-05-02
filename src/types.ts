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
   * Stage 2 Haiku classifier (#83). Opt-out: when undefined, the
   * orchestrator defaults new groups to `true` via
   * `applyNewGroupContainerConfigDefaults` so #82-style grey-zone
   * messages route through the classifier; existing groups keep
   * whatever was previously persisted. Set explicitly to `false`
   * to disable the classifier on a specific group. When effectively
   * true, `haiku-classifier` is appended to the resolved gate chain
   * so deterministic gates short-circuit before any API call. Skipped
   * for `requires_trigger=true` groups (deterministic chain only —
   * see #98).
   */
  stage2Enabled?: boolean;
  /**
   * Override the Haiku classifier model. Defaults to
   * `claude-haiku-4-5-20251001` when unset. Pin to a dated snapshot
   * if you need cache-stability across model rolls.
   */
  stage2ModelId?: string;
  /**
   * Selects a registered `ContextStrategy` (see
   * `src/gates/context-strategy.ts`) for building the volatile suffix
   * of the classifier prompt. Defaults to `static-group-context` when
   * unset. Unknown values fall back to the default with an ERROR log.
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
  // Channel-native message ID returned by the platform on send. Only
  // populated for outbound bot messages on Telegram — the `id` column
  // for bot sends is our synthetic `bot-<ts>-<rand>` so there's no other
  // place to pin the Telegram numeric ID. Inbound user messages already
  // store the platform ID as `id` itself and leave this null. Queryable
  // for debugging "what did the bot actually post at Telegram ID X?".
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
  //     SQL NULL.
  //   - `string` — recorded bot-send id.
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
}

export interface TaskRunLog {
  task_id: string;
  run_at: string;
  duration_ms: number;
  status: 'success' | 'error';
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
  // Optional: send a file to the chat.
  sendFile?(
    jid: string,
    filePath: string,
    caption?: string,
    replyToMessageId?: string,
  ): Promise<void>;
}

// Callback type that channels use to deliver inbound messages
export type OnInboundMessage = (chatJid: string, message: NewMessage) => void;

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
