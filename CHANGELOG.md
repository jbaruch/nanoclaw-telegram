# Changelog

All notable changes to NanoClaw will be documented in this file.

For detailed release notes, see the [full changelog on the documentation site](https://docs.nanoclaw.dev/changelog).

### Runtime diagnostics

- Refresh the customizable per-group runner from every canonical source byte through a host-only versioned digest marker, preserving custom-only extras and same-base edits while retaining interrupted refresh debt. Treat any lone channel entry as work so non-request evidence reaches private custody.

- Capture observed IPC entries individually so late JSON published through a retained old channel directory stays pending for the next scan, while only observed non-request leaves enter durable quarantine. Refuse no-op task updates before publication, preserving explicit empty-script removal.

- Quarantine known invalid or access-refused captured JSON entries before dispatch without following aliases or deleting unknown contents. Preserve storage/rename/fsync uncertainty and all pending/in-flight conflicts as holds, while valid same-batch and other-source IPC can progress.

- Flush agent input-refusal and host outer diagnostics before forced exit, align MCP required nonempty strings with the host schema, and describe scheduled activation data as an optional object. Reject empty host follow-up text before publication.

- Retain captured non-request directory entries in durable private quarantine without following or deleting their contents, so ordinary completed IPC batches release capacity; failed quarantine durability remains held across restart.

- Version retained host follow-up input, preserve incompatible input before SDK admission, require object activation data from scheduled prechecks, flush outer failure diagnostics before native exit, and document the Remote Control starting gate lifecycle.

- Quarantine only native task-create primary-key conflicts from untrusted request IDs, preserve retained task definitions, and continue independent IPC work while leaving other SQLite defects visible to the host owner.

- Version durable MCP message/task requests independently of custody metadata and quarantine incompatible bytes before effects. Own both agent pollers through awaited promises, abort only the affected native query on input-lifecycle failure, and preserve collected input across expected per-file I/O failures. Pin CI action commits with automated renewal and gate macOS tests on successful diagnostics.

- Retain already-consumed agent input across a later expected filesystem failure, return collected progress to the active turn, and preserve the failed input file for repair/retry.

- Version host-owned task snapshots as schema 1, validate both agent readers, and keep the live database as mutation authority. Refuse incompatible/legacy/malformed snapshots without rewriting them; publish fresh metadata by verified private atomic replacement. Extend existing formatting and lint gates to nested agent sources, and route unexpected agent defects through the genuine outer framed process boundary.

- Prepare Remote Control's existing verified runtime-owned data-directory write bits through a held descriptor before launch; refuse aliases/rebinding and preserve all payload and held session bytes.

- Create private quarantine directories explicitly under permissive umasks and prepare only verified same-writer legacy structural write bits without changing evidence. Run configured root lint before CI tests, renew the existing nested npm lock weekly, and cover native Remote Control identity/lifecycle on macOS as well as Linux.

- Reserve the host quarantine `errors` namespace case-insensitively and mask its evidence from agent parent views. Preserve refused registrations and existing quarantine bytes. Keep expected Remote Control state holds local to that optional feature during startup; unrelated channels, IPC and scheduling continue without adopting or signaling an unproved process.

- Preserve original IPC filename order through private custody and restart. Propagate failed close controls to the actual runner/scheduler cleanup owner. Keep prompt-only edits to legacy schedules independent of schedule validation, and clear positively retired preview identity before fallback delivery.
- Bind Remote Control state schema 2 to native process identity, bound and fatally decode capability output, and preserve unproved legacy state unchanged. Linux uses native boot/start/executable/command observations; Darwin requires its source-generated nonce in native command metadata. Real Claude nonce preservation remains a separate native acceptance requirement.

- Flush both IPC publication directories before acknowledging a queued request, keep captured batch/request parent names durable before source removal, and retain the sole claim when first-quarantine parent durability fails.

- Preserve explicit existing database main/bot flags during legacy migration. Run stored legacy one-shots at their already captured UTC instant without rewriting schedule history; keep new schedule input strict. Notify idle queues only after terminal output. Durably record Remote Control sessions, remove private transient child output after capture, and retain stopping sessions until native process absence is verified. Prepare only verified runtime-owned legacy IPC parent modes and avoid idle claim-journal churn through a bounded non-authoritative directory hint.

- Keep marker text inside agent results intact, refuse invalid UTF-8 IPC bytes without delivery, and preserve atomic message publication while the host captures channel directories. Create IPC parents with explicit safe modes under permissive umasks. Retire startup process groups before releasing Remote Control retry gates; keep the gate when absence is unverified. CI installs the existing locked agent protocol dependencies before running actual subprocess tests.

- Keep native delivery refusals visible through Telegram sends and draft finalization; mark delivery only after native acknowledgment. Route agent callback failures through owned cleanup and the caller Promise, and surface delayed background-loop failures through the main process owner.
- Validate container output markers, keep IPC and scheduler polls scheduled after failures, and finish failed-task logs/schedule advancement before propagating unexpected errors.
- Preserve undelivered message cursors and cleanup on unexpected errors; quarantine malformed IPC before processing and quarantine unexpected per-file failures before propagation. Version the runtime-owned database migration and remote-control state, migrate accepted database records and hold unproved legacy process identities, refuse newer versions, and route delayed remote-control polling failures through the caller Promise with owned-process cleanup.
- Type Telegram handlers and test doubles with the installed SDK contracts. Handle expected filesystem, database, command and provider failures explicitly; let unexpected implementation errors propagate. Keep legacy schema migration transactional and retryable after a failed backfill, and report only successfully stopped orphan containers.

### Runtime compatibility

- Require Node.js 24 or newer for the SQLite 13 upgrade. CI and the exact installation release come from `.nvmrc`; the bootstrap rejects older or malformed runtime versions before installing dependencies. `setup/install-node.sh` installs that pinned release through a separately pinned nvm checkout without changing shell profiles, and returns a JSON `bin_dir` for subsequent setup commands. Downloads are validated in a temporary sibling before installation; incomplete managed caches are preserved and replaced, so retrying an interrupted clone needs no manual cleanup. Sourcing it exposes its entrypoint without installing or changing shell state. Bootstrap regressions run with their own formatting, lint and TypeScript gates in CI.

## [1.2.35] - 2026-03-26

- [BREAKING] OneCLI Agent Vault replaces the built-in credential proxy. Existing `.env` credentials must be migrated to the vault. Run `/init-onecli` to install OneCLI and migrate credentials.

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
