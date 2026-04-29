# Changelog

All notable changes to NanoClaw will be documented in this file.

For detailed release notes, see the [full changelog on the documentation site](https://docs.nanoclaw.dev/changelog).

## [Unreleased]

- `scripts/deploy.sh` step 5 now emits a window-pair marker per force-kill burst to `data/host-logs/deploy-kills.log` (#249). Captures `DEPLOY_KILL_START` before sending `_close`, appends `<start_iso>\t<end_iso>` (UTC, ms ISO 8601) after `xargs docker kill` returns, and only writes the row when force-kills actually happened so graceful-only deploys leave the file untouched. `host-logs` is already mounted into agent containers read-only at `/workspace/host-logs/`, so the heartbeat skill in `jbaruch/nanoclaw-admin` reads the same file and drops 137 `task_run_logs` rows whose `run_at` falls inside any window. Pre-fix every deploy that hit a stuck-agent holdout produced false-positive task-failure alerts in `system_health.issues`; deploys now emit zero spurious 137 alerts while genuine OOM 137s (outside every window) keep flowing through. The original "host supervisor SIGTERMing the orchestrator" framing in #213 was the wrong premise — the 137 source is deploy.sh step 5 itself, doing exactly what it advertises (forced cleanup of holdouts after the 30 s grace); docker-events evidence is in the comment thread on #249.
- New `send_message_to_chat` MCP tool — main-group only host tool that posts a plain text message into another registered chat without spawning a container there (#234, #245). Replaces the `schedule_task + once: now+5s + target_group_jid` kludge for one-shot cross-chat broadcasts. Resolves `chat_id` OR `chat_name` (mutually exclusive) against `registered_groups`; ambiguous names return candidate JIDs so the agent can disambiguate. Supports `pin: true` (direct path only — silently ignored on the bot-pool path because pool sends do not expose a pin hook) and `sender: "<name>"` (Telegram bot pool, named identity). `<internal>` tags are stripped from `text` for parity with the regular `send_message` flow. The #232 phantom-row guard carries through: a swallowed Telegram send (no `sentMsgId` returned) skips the `bot-…` row write into the target chat's `messages.db` AND surfaces a top-level `error` so the calling agent sees `isError: true` rather than a misleading "successful" response — critical for a tool whose entire purpose is sending into chats the agent isn't watching, where a phantom row would silence that chat's heartbeat / unanswered-message detection. Sender is trimmed at both the MCP boundary and the host so a whitespace-only value collapses to `undefined` instead of binding a pool bot to a blank identity. The DB row records `ASSISTANT_NAME` whenever the actual outbound went via the direct path (sender only takes effect on the pool path), so heartbeat / unanswered-cron / future audits never see a persona that didn't reach the wire. `reply_to` is deliberately out of scope (foreign chat message ids aren't reachable from main; Telegram message ids are per-chat so guessing collides). File send is a separate, not-yet-built tool. Companion admin-tile skill (`broadcast-to-chat`) ships in `jbaruch/nanoclaw-admin#93` and depends on this PR landing first.
- Installed tile content (`/home/node/.claude/skills/` and `/home/node/.claude/.tessl/`) is now mounted read-only into agent containers (#247). Pre-fix, an agent could `Write`/`Edit` over its own installed `SKILL.md` / `RULES.md` / per-tile rules from inside the container; the changes didn't survive container restart (the host-side cpSync at the top of every spawn rebuilt them) but were live for the current container's lifetime — sometimes minutes-to-hours of monkey-patched behaviour before reset. Two readonly bind-mounts now layer on top of the existing writable `/home/node/.claude` mount so the kernel rejects any write to those subdirs with `EROFS`. The parent stays writable for SDK transcript writes (`projects/<slug>/`), debug logs, todos, telemetry, session-env, and the auto-memory overlay. `tessl update` is unaffected — it runs in the orchestrator container against `/app/tessl-workspace/.tessl/tiles/`, a completely different filesystem path the agent never sees. The per-spawn cpSync into `<groupSessionsDir>` still runs host-side before the agent container starts, so the readonly overlay is not in effect during the copy. Modifications to installed tile content must go through staging → `promote_to_tile_repo` → publish → `tessl update`.
- `AGENT_MODEL` is now overridable via the orchestrator's environment (#225, cherry-picks public #56). Default unchanged at `claude-opus-4-7[1m]` so existing deployments see no behavior change; operators can set `AGENT_MODEL=opus` / `AGENT_MODEL=claude-sonnet-4-6[1m]` / etc. at deploy time without editing source. Mirrors the existing `AGENT_EFFORT` env override on the agent-runner. Resolution lives in a new `resolveAgentModel()` helper in `src/container-runner.ts`: trims whitespace (so `AGENT_MODEL="  "` falls back to default rather than passing two spaces to the SDK), and warns at startup via `logger.warn` when the value doesn't begin with `claude|opus|sonnet|haiku` — surfaces typos like `claud-opus-4-7` at boot instead of as an opaque error deep in the first `query()` call. The prefix check is loose-by-design: the SDK accepts both aliases (`opus`, `sonnet[1m]`) and full IDs, the set churns with each model release, and a missed-name whitelist would block legit upgrades. `DEFAULT_AGENT_MODEL` is exported `@internal` so the unit tests can pin against the same literal the helper returns instead of duplicating the string.
- `scripts/promote-to-tile-repo.sh` now prepends an `**Author-Model:** $AUTHOR_MODEL` line to the PR body it opens against tile repos (#217). Without the declaration the gh-aw policy review pair (`review-anthropic.lock.yml` / `review-openai.lock.yml`) self-skipped promote PRs because `jbaruch/coding-policy:author-model-declaration` requires the signal — observed on tile PRs #78/#79 which never received cross-family review. `AUTHOR_MODEL` defaults to `claude-opus-4-7` and is overridable via the env var the IPC handler already passes through to the spawned bash (`...process.env`), so a non-Claude promoter can override without code changes.
- New canonical writable per-group state mount at `/workspace/state/` — sourced from `data/state/<folder>/` on the host, mounted into every container regardless of trust tier (#99 Cat 4). Solves the silent-EACCES failure mode that motivated the original audit: skills writing to `/workspace/group/` for cross-run state worked on trusted/main but silently broke on untrusted (read-only mount), and the only existing workaround was `tessl__check-unanswered`'s ad-hoc routing through `/home/node/.claude/nanoclaw-state/`. Per-group scoping (not per-session) so a scheduled task and a user-facing turn in the same group can read each other's state. Mount is always writable — `data/state/<folder>/` is `mkdir -p`'d at spawn and `chown`ed to the container user. Tier-uniform availability is the contract: `container-runner.test.ts` asserts the mount is present without `:ro` for main, trusted, and untrusted, and that the host path is `<DATA_DIR>/state/<folder>`. `docs/SPEC.md` adds a workspace-layout subsection documenting the choice between `/workspace/group/` (operator-readable, trust-tier-conditional readonly) and `/workspace/state/` (agent's persisted state, always writable). Tile-side migrations of `tessl__check-unanswered` and `tessl__brief-cleanup` ship as separate issues on `nanoclaw-core` and `nanoclaw-admin`.
- `nuke_session` MCP tool gains a `skipReentry` argument (#127, follow-up to #104). When `skipReentry: true`, the orchestrator deletes the per-group checkpoint pair (`<groupDir>/.checkpoints/default.md` + `previous.md`) after the standard wipe steps so the next container spawn has no Facts/Reasoning to load via the reentry skill. Default behaviour (false / omitted) preserves the checkpoint, since the standard nuke is "fresh session, but reentry continues to work". `skipReentry` is for the case where the checkpoint itself is suspect (poisoned plan, stale do-not-re-execute list); without it the operator's only workaround was a manual `rm` from the host. Implemented via a new `clearCheckpoints(groupDir)` helper exported from `src/checkpoint.ts` (idempotent: ENOENT on either file returns the count of files actually unlinked). Strict boolean check on the IPC payload — anything non-true falls back to preserving the checkpoint, so a malformed payload can't accidentally erase reentry state.
- Kill-auto-compaction Phases 1–4 land behind `ENABLE_THRESHOLD_NUKE` (#125, #122, #123, #128, tracks #104; design at `docs/proposals/kill-auto-compaction.md`). Token-usage telemetry is **always-on**: every per-turn `usage` payload from the SDK's `stream-json` output (`input_tokens` / `output_tokens` / cache fields) is logged via `logger.info({…}, 'session_tokens')`, with WARN at the warn threshold and ERROR at the nuke threshold. Threshold formula is `min(70% of context, context-200K)` warn / `min(80%, context-100K)` nuke; for Opus 4.7 1M that's 700K warn / 800K nuke. The `## Facts` checkpoint writer fires at threshold-cross (always-on, validates the file format on real transcripts before the destructive flip relies on it), rotating `default.md → previous.md` first then writing the orchestrator-deterministic Facts section to `<group>/.checkpoints/default.md` — mutating-tool taxonomy filter (Write, Edit, MultiEdit, NotebookEdit, Skill, every `mcp__nanoclaw__*` mutator, Task, TeamCreate/Delete) plus a Bash read-only argv allowlist (cat, ls, grep, find, head, tail, wc, sort, uniq, pwd, echo, file, stat, date, plus `git status/log/diff/show/rev-parse`) keeps the "do NOT re-execute" list focused on real state changes. The agent-runner's existing SessionStart auto-context hook now ALSO loads `/workspace/group/.checkpoints/default.md` as a CHECKPOINT section (last, closest to the prompt for max salience) so the next session after a nuke picks up the prior facts deterministically. A user-invocable `/session-reentry` skill ships as the manual fallback. **Behaviorally inert by default** — `ENABLE_THRESHOLD_NUKE=0` keeps SDK auto-compaction on and the orchestrator's `nuke_session` call gated off; the orchestrator only logs `threshold_nuke_inert (ENABLE_THRESHOLD_NUKE=0)` at warn-level when the threshold crosses. When the operator flips `ENABLE_THRESHOLD_NUKE=1`, `DISABLE_COMPACT=1` is set on every container spawn AND the threshold-cross handler calls `nuke_session` for the default slot after the current turn completes — the next inbound spawns a fresh container which auto-loads the checkpoint via the SessionStart hook. The flag-off period is the data-collection window: ssh nas + grep `'session_tokens|threshold_nuke'` on host-logs to curve real-session usage distributions before flipping.
- `wipeSessionJsonl` now also removes the sibling per-session tool-results directory at `<slug>/<sessionId>/` alongside the JSONL transcript (#202, follow-up to #193). The SDK writes tool-call result snapshots (image attachments, search outputs) into that directory; pre-fix it was orphaned by every scheduled-task run and by every `nukeSession` call (since the helper is shared). Implemented via a new `removeToolResultsDirInSlug` helper that mirrors `unlinkJsonlInSlug`'s lstat → branch on type → realpath-containment discipline: a symlink at the dir path is unlinked as a link (target preserved), a regular file is left alone (logs warn — not the SDK's shape), and a real directory is removed via `fs.rmSync({ recursive: true })` with `ENOENT` handled explicitly in the catch so the returned delete count stays accurate (a concurrent cleanup that vanishes the path between lstat and rm returns 0, not 1). Node's `rmSync` does not follow symlinks during recursion, so a compromised container that scattered host-pointing symlinks inside its own tool-results dir cannot redirect the wipe outward. The `wipeSessionJsonl` return count is now up to 2 per slug (1 transcript + 1 tool-results dir) rather than 0–1.
- Added `unregister_group` MCP tool — main-group only inverse of `register_group` (#159). Removes the `registered_groups` row, cascade-deletes scheduled_tasks tied to the unregistered folder, and refreshes `available_groups.json` in one call; refuses to touch the main-group registration; leaves the on-disk `groups/<folder>/` directory intact (operators delete that manually). Companion one-shot cleanup at `initDatabase()` drops the dormant `tg:1698969` / `telegram_main` row that lingered because there was no inverse path until now. Companion drift detector logs (does not auto-delete) any `registered_groups` row whose JID has no matching `chats` row at startup, so future drift surfaces immediately.
- [BREAKING] Removed the auto-create heartbeat rule for non-main groups (#158). Heartbeat is now opt-in via `containerConfig.enableHeartbeat` instead of being implicit on `requiresTrigger !== false` (a flag no group ever actually had set). Existing heartbeat rows are preserved (the startup prompt-migration still rewrites legacy prompts on any non-main row that exists), and the main-group heartbeat is unchanged. `setGroupTrigger` no longer touches heartbeat lifecycle — trigger config and heartbeat opt-in are now orthogonal. The `register_group` MCP tool exposes the new `enableHeartbeat` parameter so opt-in flows through the same channel as other registration config.
- [BREAKING] Removed the `MAX_CONCURRENT_CONTAINERS` global concurrency cap (#157). With ~10 registered groups × 2 slots, the theoretical ceiling is ~20 and the cap was more likely to delay legitimate work (e.g. a heartbeat firing alongside an inbound user message on a different group) than to prevent runaway spawn. Hardware and Docker remain the only limits worth honouring; per-group rate limits are out of scope. The `MAX_CONCURRENT_CONTAINERS` env var is now ignored.
- Pin `thinking.display: 'summarized'` on the agent-runner's `query()` call (#163). Opus 4.7 silently flipped the default to `'omitted'`, which would surface thinking blocks as empty content with an opaque encrypted signature — invisible today but a hard prerequisite for the upcoming lifecycle reaction state machine (#162) that triggers on the first thinking block with content.
- Scheduled tasks no longer resume the SDK maintenance session across discrete invocations (#193). The maintenance slot's `sessionId` was shared by every `context_mode: 'group'` task on a folder, so a prior turn's terminal message could bleed into the next run's stream — observed as a lunch reminder's `last_result` opening with heartbeat-loop language from a 6-day-old turn. Each scheduled run now starts a fresh SDK turn: no `resume: sessionId` is passed and no `newSessionId` is persisted on completion. To prevent orphan transcripts from accumulating under `data/sessions/<group>/maintenance/.claude/projects/<slug>/` (the sessionId is no longer persisted, so neither `nukeSession` nor the time-based `cleanup-sessions.sh` script can find it later), the scheduler wipes each run's JSONL via `wipeSessionJsonl` from the post-run `finally` path, after the run bookkeeping (`logTaskRun` and the `updateTaskAfterRun` attempt) has been attempted. The `MAINTENANCE_SESSION_NAME` slot still carries the per-session `.claude/` mount and parallel queue routing, so user-facing default-slot work is unaffected. The `context_mode` column is retained on the schema but no longer gates SDK resume.
- Hooks epic — `bash-safety-net` PreToolUse hook denies known-destructive Bash commands deterministically (#143). Catalogue covers `rm -rf` on root / mount-root paths (combined and split flags, end-of-options marker, trailing-slash and dot-segment variants), force-push to `main`/`master` (`--force` flag and `+refspec` syntax), `mkfs.*`, raw-disk `dd`, raw block-device redirects, `chmod -R 777`, `chown -R` on mount roots, and the canonical fork bomb. Anchored to command-start positions so prose mentions of the same tokens (e.g. `echo mkfs.ext4 docs.md`) are not flagged.
- Hooks epic — `reply-threading-enforcement` PreToolUse hook denies a standalone `mcp__nanoclaw__send_message` (no `reply_to`) when the latest user inbound is unanswered (#137). Carve-outs: `pin: true` (status updates), `sender` set (multi-bot persona), maintenance / scheduled-task session, and any `reply_to` (which marks the inbound addressed and unlocks subsequent standalones in the same turn). Single-turn enforcement only — cross-turn de-dup needs a `messages.db` query and is queued as a follow-up.
- Hooks epic — `lazy-verification-detector` Stop hook blocks end-of-turn messages that surface banned verification excuses ("site is JS-rendered", "page is thin", "can't access this", etc.) without enumerating real attempts (#135). On match, the SDK re-runs the turn with a reminder injected via `systemMessage`. Genuine-failure carve-out: messages with ≥2 `Tried X — got Y` enumerations pass through. The `nanoclaw-core/rules/no-lazy-verification.md` prose rule is fully replaced by this runtime check and is deleted in a paired tile-cleanup PR.
- Hooks epic — `composio-fidelity` PostToolUse hook flags fabricated-ID signatures in MCP tool returns (sequential `prefix_NN` ≥5 dense, `pr_notif` compound shape ≥3, `promo_NNN` ≥3) (#140). Detections append to `/workspace/host-logs/fidelity-alerts.log` and a `systemMessage` warning is injected so the agent treats the result as untrusted instead of silently quoting fabricated IDs. Tool result is NOT silently rewritten — masking the failure mode would be worse than surfacing it.
- Hooks epic — `no-markdown-in-send-message` PreToolUse hook auto-rewrites the four common Markdown leaks (`**bold**`, `[label](url)`, `` `code` ``, `- bullet` lines) to HTML before `mcp__nanoclaw__send_message` and `mcp__nanoclaw__send_file` reach the IPC layer (#138). Code-block regions (` ``` ` fences, `<pre>`, `<code>`) are passed through bytewise so the agent can quote raw Markdown samples without the hook mangling them. Link labels and bold inner content are HTML-entity-escaped to prevent stray-tag smuggling.
- Hooks epic — `path-hygiene-cadence` PreToolUse hook suppresses duplicate path-hygiene reports within a 4-hour window (#139). Signature is `<keyword>:<lc-path>` for catalogued keywords (`path-hygiene`, `orphaned`, `misplaced`, `staging-drift`); cadence persists across container restarts via the per-group daily-log mtime. Carve-outs: `pin: true` and any `reply_to` (responding to an explicit user ask). Deny reason renders the actual configured window so a custom `windowMs` doesn't lie.
- Hooks epic — `react-first` UserPromptSubmit hook synthesises an acknowledgement reaction (👀) before the model spends any tokens on a new inbound (#136). Skips on sub-agent turns, scheduled tasks, prompts wrapped as `[SCHEDULED TASK]`, and containers without a named user-facing assistant. The agent can still react with a more specific emoji later in the turn — Telegram replaces the bot's reaction on each new call, so this is a floor, not a ceiling. The "React with an emoji to acknowledge" line in `nanoclaw-core/rules/default-silence.md` is removed in a paired tile-cleanup PR.
- Hooks epic — `session-start-auto-context` SessionStart hook auto-injects MEMORY.md, RUNBOOK.md, and the most-recent daily log into the session before the first turn fires (#141). Pure file-IO, no LLM round-trip; per-file byte-cap truncation; `Date.parse`-validated daily-log filename pick. Fires only on `source === 'startup'` and skips containers without a named user-facing assistant. Replaces the implicit "read at session start" path on the `tessl__trusted-memory` skill (the skill keeps explicit memory writes).
- Hooks epic — `stop-hook-end-of-turn-audit` Stop hook (#142). Observability only — never blocks the turn. Tracks per-turn react/reply state via paired UserPromptSubmit + PreToolUse callbacks; on Stop, if neither a `react_to_message` to the triggering inbound nor a `send_message` with `reply_to === triggeringInboundId` landed, appends a JSONL entry to `/workspace/host-logs/silent-turns.log` with chat JID, message id, session id, and timing. Skips on sub-agent / maintenance-session / scheduled-task turns and on turns with no triggering inbound (orchestrator-typed text, slash commands).

## [1.2.54] - 2026-04-26

- [BREAKING] Per-group `CLAUDE.md` is now a thin trust-tier pointer (TRUSTED/UNTRUSTED marker + `@import` to SOUL.md / FORMATTING.md / MEMORY.md / RULES.md) mounted readonly from `groups/global/` at every container spawn (#153). Trust flips are reflected on the next message with no reconciliation step. Identity and behavior live in `groups/global/SOUL.md`; channel formatting in `groups/global/FORMATTING.md`; per-group writable memory in each group's new `MEMORY.md`; main-only operational content in `groups/main/ADMIN.md`. Existing installs run `tsx scripts/migrate-thin-claude-md.ts --apply` on the host to delete vanilla per-group `CLAUDE.md` copies and seed `MEMORY.md`; customized files are flagged for manual reconciliation.

## [1.2.53] - 2026-04-26

- Poison defense: `TaskOutput(block!=false)` is denied at the PreToolUse hook to stop the SDK from leaking raw sub-agent JSONL on timeout (#116). MCP tool results are scrubbed of Cf-class invisible-Unicode characters and capped at `TOOL_RESULT_MAX_BYTES` bytes (default 64 KiB) before reaching the model (#117).

## [1.2.36] - 2026-03-26

- [BREAKING] Replaced pino logger with built-in logger. WhatsApp users must re-merge the WhatsApp fork to pick up the Baileys logger compatibility fix: `git fetch whatsapp main && git merge whatsapp/main`. If the `whatsapp` remote is not configured: `git remote add whatsapp https://github.com/qwibitai/nanoclaw-whatsapp.git`.

## [1.2.35] - 2026-03-26

- [BREAKING] OneCLI Agent Vault replaces the built-in credential proxy. Check your runtime: `grep CONTAINER_RUNTIME_BIN src/container-runtime.ts` — if it shows `'container'` you are on Apple Container, if `'docker'` you are on Docker. Docker users: run `/init-onecli` to install OneCLI and migrate `.env` credentials to the vault. Apple Container users: re-merge the skill branch (`git fetch upstream skill/apple-container && git merge upstream/skill/apple-container`) then run `/convert-to-apple-container` and follow all instructions (configures credential proxy networking) — do NOT run `/init-onecli`, it requires Docker.

## [1.2.21] - 2026-03-22

- Added opt-in diagnostics via PostHog with explicit user consent (Yes / No / Never ask again)

## [1.2.20] - 2026-03-21

- Added ESLint configuration with error-handling rules

## [1.2.19] - 2026-03-19

- Reduced `docker stop` timeout for faster container restarts (`-t 1` flag)

## [1.2.18] - 2026-03-19

- User prompt content no longer logged on container errors — only input metadata
- Added Japanese README translation

## [1.2.17] - 2026-03-18

- Added `/capabilities` and `/status` container-agent skills

## [1.2.16] - 2026-03-18

- Tasks snapshot now refreshes immediately after IPC task mutations

## [1.2.15] - 2026-03-16

- Fixed remote-control prompt auto-accept to prevent immediate exit
- Added `KillMode=process` so remote-control survives service restarts

## [1.2.14] - 2026-03-14

- Added `/remote-control` command for host-level Claude Code access from within containers

## [1.2.13] - 2026-03-14

**Breaking:** Skills are now git branches, channels are separate fork repos.

- Skills live as `skill/*` git branches merged via `git merge`
- Added Docker Sandboxes support
- Fixed setup registration to use correct CLI commands

## [1.2.12] - 2026-03-08

- Added `/compact` skill for manual context compaction
- Enhanced container environment isolation via credential proxy

## [1.2.11] - 2026-03-08

- Added PDF reader, image vision, and WhatsApp reactions skills
- Fixed task container to close promptly when agent uses IPC-only messaging

## [1.2.10] - 2026-03-06

- Added `LIMIT` to unbounded message history queries for better performance

## [1.2.9] - 2026-03-06

- Agent prompts now include timezone context for accurate time references

## [1.2.8] - 2026-03-06

- Fixed misleading `send_message` tool description for scheduled tasks

## [1.2.7] - 2026-03-06

- Added `/add-ollama` skill for local model inference
- Added `update_task` tool and return task ID from `schedule_task`

## [1.2.6] - 2026-03-04

- Updated `claude-agent-sdk` to 0.2.68

## [1.2.5] - 2026-03-04

- CI formatting fix

## [1.2.4] - 2026-03-04

- Fixed `_chatJid` rename to `chatJid` in `onMessage` callback

## [1.2.3] - 2026-03-04

- Added sender allowlist for per-chat access control

## [1.2.2] - 2026-03-04

- Added `/use-local-whisper` skill for local voice transcription
- Atomic task claims prevent scheduled tasks from executing twice

## [1.2.1] - 2026-03-02

- Version bump (no functional changes)

## [1.2.0] - 2026-03-02

**Breaking:** WhatsApp removed from core, now a skill. Run `/add-whatsapp` to re-add.

- Channel registry: channels self-register at startup via `registerChannel()` factory pattern
- `isMain` flag replaces folder-name-based main group detection
- `ENABLED_CHANNELS` removed — channels detected by credential presence
- Prevent scheduled tasks from executing twice when container runtime exceeds poll interval

## [1.1.6] - 2026-03-01

- Added CJK font support for Chromium screenshots

## [1.1.5] - 2026-03-01

- Fixed wrapped WhatsApp message normalization

## [1.1.4] - 2026-03-01

- Added third-party model support
- Added `/update-nanoclaw` skill for syncing with upstream

## [1.1.3] - 2026-02-25

- Added `/add-slack` skill
- Restructured Gmail skill for new architecture

## [1.1.2] - 2026-02-24

- Improved error handling for WhatsApp Web version fetch

## [1.1.1] - 2026-02-24

- Added Qodo skills and codebase intelligence
- Fixed WhatsApp 405 connection failures

## [1.1.0] - 2026-02-23

- Added `/update` skill to pull upstream changes from within Claude Code
- Enhanced container environment isolation via credential proxy
