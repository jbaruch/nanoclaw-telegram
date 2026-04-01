# Blog Post Raw Notes: Building a NanoClaw Skill Collection for Telegram

*For Baruch to turn into a post. Rough notes, narrative arc, tech details. Not polished.*

---

## The Story So Far

### Where It Started

Had a bot. It could chat. That was kind of it. Which, honestly, felt like having a very expensive parrot — impressive at parties, not useful when you're trying to get something done.

The original setup was WhatsApp-only. Made sense at the time. WhatsApp is where life happens, at least for a certain kind of person who has too many group chats and not enough hours.

---

### The Voice Message Problem (aka the thing that broke everything)

Here's the embarrassing part: the bot couldn't hear me.

I record voice notes constantly. On the go, walking to the kitchen, pretending to exercise. Voice is how I think out loud. And the bot just... silently ignored them. Every single one. Not even an error. Just nothing. Radio silence from the AI.

[OUTCOME: describe what it felt like when you realized this — the specific moment, the specific voice note that got swallowed]

This is a real pain point for how I actually work. Not a hypothetical. I'm the person who should be the primary user of this thing and it couldn't handle my primary input method. That's a level of irony that deserves documentation.

---

### Enter Telegram

So: WhatsApp had some stuff already on separate branches. Vision (photos) was partially working. Voice was partially working. But those were branches. Not shipped. Not usable.

Meanwhile there was a Telegram integration sitting there. Which raised the obvious question: could I move my workflow to Telegram and pick up where WhatsApp left off?

Short answer: yes, but there's work to do.

The NanoClaw architecture is actually set up well for this. Isolated containers, per-group context, the whole multi-channel thing. The same core handles WhatsApp, Telegram, Slack, Discord. The channels are kind of plugged in from the outside. That's a good design. More on this below.

The problem is `telegram.ts`. Specifically line 283 (photo handling) and line 285 (voice handling). Both are stubs. Both punt to placeholders. The infrastructure is there to receive the messages — the routing works, the storage works — but when the actual media arrives, the code basically shrugs.

```
[CODE_SNIPPET: the actual stub lines from telegram.ts 283/285]
```

WhatsApp already solved this. The voice transcription skill exists. The vision skill exists. They're just on branches. The task is: take what already works on WhatsApp, port it to Telegram, and do it properly this time so it ships.

---

### The Tessl Discovery

This is where it got interesting.

Tessl is basically npm but for AI agent skills. You publish a skill, someone else installs it, their agent can now do the thing your skill does. The format is a `SKILL.md` file with YAML frontmatter.

And here's the thing I did not initially realize: the NanoClaw skill format is *identical* to the Tessl format. Same structure. Which means anything I build for my own NanoClaw setup is already publishable. I'm not just solving my own problem — I'm potentially contributing to a shared library.

That reframing changed the whole project.

---

### What This Actually Is

It's not just "fix voice messages for my Telegram bot."

It's: build a collection of skills that turn a general-purpose AI assistant into a fully capable one — and publish them so anyone using a compatible agent setup can install them.

The skills on the roadmap:

- **Voice transcription** — so the bot can actually hear you [SKILL_NAME_TBD]
- **Image/photo understanding** — so you can send a screenshot and ask about it [SKILL_NAME_TBD]
- **GitHub integration** — PRs, issues, the whole thing [SKILL_NAME_TBD]
- **Gmail** — because email is not going away no matter how much we want it to [SKILL_NAME_TBD]
- **Google Drive** — docs, sheets, find the thing I definitely saved somewhere [SKILL_NAME_TBD]
- **Composio integration** — this one is the interesting wildcard (see below)

---

### The Composio Thing

ComposioHQ has a `connect-apps` skill. Install one thing, get 1000+ integrations. That's either the dream or the nightmare, depending on your philosophy about dependencies.

But seriously: if that works as advertised, it changes the math on how many individual skills you need to build. Instead of writing a Gmail skill AND a Google Calendar skill AND a Notion skill, you install Composio and suddenly the agent has all of them.

[OUTCOME: what actually happened when you tried this — did it work? how well? what were the rough edges?]

This is worth a section in the post because it's a real architectural decision: build thin, specific skills yourself vs. delegate to an aggregator. Both are valid. The answer probably depends on how much control you want and how much you trust the abstraction.

---

### Technical Architecture Notes (for the nerdy section of the post)

For readers who want to understand what's actually happening under the hood:

**NanoClaw architecture in a sentence:** Isolated containers, one per group, each with its own context and memory. The agent running in the Family Chat container doesn't know what's happening in the Work group container. That's on purpose.

**Multi-channel:** The same agent core handles WhatsApp, Telegram, Slack, Discord. Channel adapters plug in from outside. This is why porting from WhatsApp to Telegram is a matter of fixing the adapters, not rewriting the agent.

**Per-group context:** Each group gets its own folder, its own CLAUDE.md with instructions, its own memory files. The global memory is shared. Groups can have additional directories mounted in if they need access to local files or projects.

**Skill format:** SKILL.md with YAML frontmatter. The agent reads the skill file and knows how to use the tool. Tessl uses the same format, so your local skills and publishable skills are the same thing. No conversion step.

**Task scripts — this is underappreciated:** When you schedule a recurring task, you can attach a bash pre-check script. The script runs first. It outputs JSON: `{ "wakeAgent": true/false, "data": {...} }`. If there's nothing to do, the agent doesn't wake up. This is how you make frequent polling economically sane — you're not paying for an LLM call every 5 minutes if the condition isn't met. The script handles the cheap check, the agent handles the expensive reasoning.

```
[CODE_SNIPPET: example task script pattern — the wakeAgent pattern]
```

This is the detail that makes the difference between "useful recurring tasks" and "burning through API credits checking if something changed when it didn't."

---

### The Meta-Point (what the post is actually about)

The scaffolding is already built. The architecture handles the hard parts: isolation, context, memory, multi-channel routing, scheduling, task scripts.

What's missing is skills. And skills are small, composable, publishable.

So the real story is: I started by trying to fix a personal annoyance (bot couldn't hear my voice messages), discovered I was sitting on top of an architecture that could be genuinely useful, and ended up building a collection of skills that other people can use too.

This is how a lot of good open source starts. Someone had an itch. Scratched it. Realized the scratching mechanism was the interesting part.

[OUTCOME: where the skill collection actually ended up — how many skills, what they do, link to the Tessl registry]

---

## Things to Weave In / Tone Notes

- Don't start with "I built a bot." Everyone has built a bot. Start with the voice note that got ignored.
- The Tessl discovery should feel like a genuine "oh wait" moment, not a product placement.
- The Composio section should be honest about what it does and doesn't solve.
- Don't oversell the vision at the end — this is a developer audience, they'll smell it. Understate it and let the architecture speak.
- The task scripts section is technically interesting and most people don't know about it. Give it room.
- Self-deprecation: I am the intended user of this tool and it couldn't handle my primary input method. That's funny. Use it.

---

## Stuff Still TBD

- [ ] Actual skill names once they're built
- [ ] Outcome of Composio integration experiment
- [ ] Code snippets from telegram.ts stubs
- [ ] Code snippet for task script wakeAgent pattern
- [ ] Link to published Tessl skills once live
- [ ] Specific voice note anecdote to open with
- [ ] Whether to cover the sender allowlist feature (probably a separate post)
- [ ] How to explain "Tessl is like npm for AI" without it sounding like marketing copy

---

---

## Update: 2026-03-27 — The Tiles Migration

### The Realization

NanoClaw is not a remote coding interface. It's a personal assistant. Spent a day trying to make it match Claude Code in the terminal for technical work and the answer is: it can't. Three layers of indirection (channel → orchestrator → container → Agent SDK), each one leaking context. The 365-line CLAUDE.md that's supposed to define behavior? Half the rules don't activate. The agent is observably dumber inside the container than in a terminal session.

That's not a bug you fix. That's the architecture telling you what the tool is for.

So: pivot. Stop fighting it. Lean into what it's good at — calendar, reminders, research, summaries, quick queries. And make THAT work reliably.

### The Context Budget Problem

Claude Code has a finite instruction budget (~150-200 lines effective). NanoClaw was burning 365 lines of CLAUDE.md on everything from identity rules to group management procedures to formatting tables. Most of it irrelevant to any given message. The agent was drowning in instructions and following none of them consistently.

The fix: Tessl tiles. Same format NanoClaw already uses for skills, but with proper separation of rules (always-on context) vs skills (loaded on demand).

### What We Built: Two Tiles

**nanoclaw-core** — loaded for every agent session:
- Rules (1.2k front-loaded tokens): identity, async task protocol (ACK first, background agent), boyscout rule, communication rules (always use native reply/quote!), memory management, formatting policy
- Skills (on-demand only): `format-message` (channel syntax tables), `schedule-task` (pre-check scripts), `heartbeat` (11 health checks + self-healing)

**nanoclaw-admin** — loaded only for the main channel:
- Rules (637 front-loaded tokens): elevated privileges, auth, container mounts, cross-group scheduling
- Skills (on-demand only): `manage-groups` (registration, allowlists, mounts), `create-agent-team` (multi-bot team setup)

Total front-loaded cost: 1.8k tokens. Compare to dumping 365 lines (~4k+ tokens) of mixed relevance into every session.

### The Tessl Connection (updated)

This is the part that makes the Tessl angle stronger for the blog. The tiles aren't just local organization — they're valid Tessl tiles. `tessl tile lint` passes on both. They have `tile.json`, proper skill frontmatter, rule files. Publishing them to the registry is `tessl tile publish` away.

So the skill collection story now has a second chapter: first we built individual skills (voice, vision, etc.), then we extracted the behavioral rules themselves into tiles. The agent's personality, communication style, and operational procedures are now versioned, lintable, publishable packages.

That's a sentence worth sitting with. Your bot's personality is a package you can install.

### The Reply-To Bug (good anecdote)

While doing this work, found and fixed a real bug: the heartbeat check for "unanswered messages" was never finding anything because bot replies sent via MCP `send_message` were never stored in the database. Every message looked unanswered. The fix was one `storeMessage()` call in `ipc.ts`.

Also fixed reply threading — the `replyToMessageId` was set once at container startup and cleared after the first reply. Follow-up messages in the same session lost their reply context. Fix: shared `pendingReplyTo` map that both the message loop and the output callback can access.

Both bugs are the kind of thing that only surface when you actually USE the tool daily. The heartbeat was running, reporting "all clear", while silently being unable to detect the very thing it was supposed to detect. Classic.

### NAS Migration (teaser for next post?)

The other big decision: NanoClaw doesn't need to run on the coding machine. Moving it to a NAS (Docker, always-on) means better uptime, simpler operation. The tiles-in-image model means deployment is `git pull → build → docker compose up`. Three volumes for runtime state, everything else baked in.

This might be its own post. The mount architecture alone is worth documenting — going from 10 mounts (including Google Drive FUSE paths and macOS-specific symlinks) to 3 volumes is a real simplification story.

### Updated TBD

- [ ] Wire tiles into container-runner.ts (skill sync + rule injection)
- [ ] Slim down CLAUDE.md to ~65 lines that reference the tile rules
- [ ] Test that rules actually activate more reliably with the reduced context
- [ ] Publish tiles to Tessl registry
- [ ] NAS migration (separate post?)
- [ ] Composio: we fixed the key issue (was hitting wrong endpoint — `backend.composio.dev` vs `connect.composio.dev`), switched from stdio MCP to HTTP transport. Working now.

### OpenClaw for Dummies — Reference Notes

MarkusDowne wrote an "OpenClaw for Dummies" post that covers the same space from the OpenClaw angle. Worth referencing/contrasting:

- **Separate instruction files vs monolith CLAUDE.md** — OpenClaw splits into 7 files (AGENTS.md, BOOTSTRAP.md, IDENTITY.md, SOUL.md, TOOLS.md, USER.md). NanoClaw crammed everything into one 365-line CLAUDE.md. The tiles migration is essentially arriving at the same conclusion from the other direction — split your instructions into focused, composable units.
- **HEARTBEAT.md as first-class concept** — OpenClaw has built-in periodic heartbeat. NanoClaw implements heartbeat as a scheduled task + skill. Different plumbing, same pattern. Our heartbeat is more sophisticated (11 checks, self-healing, calendar/email) but the core idea is identical.
- **"Lobster Anatomy" mental model** — workspace, instructions, tools/skills, runs. Maps directly to NanoClaw's architecture. Good shared vocabulary for the post.
- **"Start with one simple loop, make it reliable, make it clever last"** — Markus's closing advice. We arrived at the same conclusion independently (personal assistant first, not remote coding). Good parallel to draw.
- **"Don't ask the agent to configure itself 0→1"** — vs our Boyscout Rule (find it, fix it, don't ask permission). Good tension: don't trust the agent to build its own foundation, but DO trust it to maintain and improve what's already working. 0→1 is human work, 1→2 is agent work.
- **`tessl optimize`** — Markus ran it on skills to improve quality/performance. We should do this on our tiles after creation.
- **Source calibration skill** — Before summarizing from social/hype-heavy sources, OpenClaw uses a skill to calibrate for noise. Our heartbeat email check classifies importance but doesn't calibrate for source noise. Minor but interesting pattern.
- **AgentMail pattern** — Agent sends daily digest email without needing full Gmail channel integration. Lightweight output channel. We have Gmail as a full channel; this is the simpler version.

*Updated: 2026-03-27.*

---

## Update: 2026-03-28 — Heartbeat Owns Everything Now

### The Architecture Simplification

Previous design: NanoClaw ran checks 1-4, external heartbeat on the NAS ran disk/docker checks as check #5. Felt clean in theory. In practice it was a coordination headache — two owners for one process, state split across two places.

Decision: external heartbeat is now watchdog-only. It checks one thing: is NanoClaw alive? If not, alert. That's it. Everything else — disk, logs, session bloat, IPC errors, stuck close files, orphaned containers — now runs inline as Check #5 in the heartbeat skill.

Result: the heartbeat skill is self-contained. All five checks, all auto-fixes, all reporting. No external dependencies for health logic. The external script dropped from "runs checks" to "pings the container and alerts if it's dead."

This is a cleaner separation of concerns: the NAS knows about uptime, NanoClaw knows about its own health.

### The Silence Rule (learned the hard way)

Scheduled tasks should be silent when healthy. NanoClaw sent an "all clear" message after a clean heartbeat. That's wrong — it's noise. The rule was already written in the tiles ("if the result is silent, send nothing at all") but it didn't stick.

Now it's in memory. The rule: **scheduled tasks only speak when something's wrong.** All-clear = silence. This is the async equivalent of a good alarm system — you don't want it to beep every hour to confirm the house hasn't burned down.

### LinkedIn: Deferred to Webhooks

LinkedIn @mentions were on the roadmap. The Composio LinkedIn toolkit doesn't expose a notifications or @mentions API — the best available option was polling reactions on known post URNs. That's noisy, delayed, and not what we actually want.

Decided: skip the polling approach entirely. Build it properly when LinkedIn webhooks are available:
https://learn.microsoft.com/en-us/linkedin/shared/api-guide/webhook-validation

The section was removed from heartbeat rather than left as a stub. Stubs rot.

*Updated: 2026-03-28.*

---

---

## 2026-03-31 — The Full Stack Day

Today was a day of building operational infrastructure — tiles, scripts, pipelines, bugs, disasters, recoveries, and the agent bootstrapping its own ecosystem.

### nanoclaw-host Tile — Teaching the Host Agent

Created jbaruch/nanoclaw-host — a tessl tile for ME (Claude Code on the Mac), not for AyeAye in the container. Four skills wrapping operational scripts:

- **promote** — runs promote-skill.sh, handles tile selection, tessl optimize, publish, deploy
- **nuke** — kills a container by Telegram JID (looks up folder from DB, translates to container name)
- **check-staging** — lists pending skills and rules on NAS
- **reconcile** — diffs registry vs git vs installed tiles, flags drift

Plus a host-conventions rule codifying lessons learned: registry is the delivery artifact, nuke means kill not delete, never assume staging is stale (always diff), no error suppression in scripts.

All scripts extracted to use common.sh for shared config (NAS_HOST, NAS_PROJECT_DIR from .env). Zero hardcoded IPs.

### The Promote Pipeline Matures

promote-skill.sh went through several iterations:
- Fixed rules staging path (was .tessl/tiles/local/, now staging/{tile}/)
- Added tessl__ prefix handling (AyeAye patches existing tile skills at skills/tessl__{name}/)
- tessl review --optimize now only runs on promoted skills, not all 21
- SSH -n on all calls so stdin doesn't get consumed before the interactive prompt
- Deduplication for all-mode discovery

Promoted today: soul-review (new), verify-tiles (bug fix), check-calendar, morning-brief, check-cfps, heartbeat, manage-groups, nightly-housekeeping, task-tz-sync. Plus 5 new rules across core and untrusted tiles.

### Issue #3 and #5 — Trust Propagation

Two GitHub issues filed and fixed via PRs:

**#3: containerConfig not reaching the spawner.** Four bugs: AvailableGroup interface missing fields, getAvailableGroups() not populating them, available_groups.json not refreshed after register_group, race condition on startup. PR #4 fixed the first three. Cursor Bugbot caught that the race condition fix was unnecessary (full table scan on every message for nothing) — dropped it.

**#5: register_group MCP tool doesn't accept containerConfig.** The IPC handler and DB already supported it, but the MCP tool never exposed the parameter. One-file fix: added trusted (boolean) and additionalMounts (array) to the tool schema. Copilot review caught that the additionalMounts schema didn't match the actual host behavior (hostPath supports ~, containerPath is optional, readonly defaults true). Fixed. Also fixed a truthy check bug where trusted: false would be silently dropped.

### The Permission Disaster

Tried to fix container permissions (agent runs as uid 999, session dirs owned by root). Went through FOUR iterations of chown fixes, each one wrong:
1. Hardcoded uid 1000 — wrong, NAS user is 999
2. Recursive chown but too early (before skills/tessl written)
3. Recursive chown after writes but still hardcoded 1000
4. Used HOST_UID — right uid, but blanket chown broke untrusted group permissions

Each fix caused a new problem. The .dockerignore added to work around one fix caused issues with Docker context. Eventually reverted ALL four commits with proper git revert. The actual fix was simpler: AyeAye just needs to re-register the group with trusted: true via the MCP tool (which we'd just fixed), and the spawner handles the rest.

Lesson: stop patching symptoms. Debug the root cause first. The chown was treating a consequence (wrong permissions) instead of the cause (trusted flag not propagated). Once the trust pipeline worked (issue #3 + #5), the permissions followed naturally.

### Shared Trusted Space

/workspace/trusted/ — writable directory mounted into main and trusted containers. Untrusted containers don't see it. Required three layers: docker-compose mounts it into orchestrator, orchestrator checks existsSync, container-runner mounts it into agent containers based on trusted flag.

AyeAye migrated 22 memory files (feedback rules, people context, project status, credential scope, highlights) from group-local storage into the shared space. Two-tier memory architecture emerged:
- Long-term shared: /workspace/trusted/ — backed up to GitHub nightly
- Short-term group-local: /workspace/group/memory/daily/ and weekly/

### github_backup MCP Tool

No backup existed. Agent memory files live on NAS — if the disk dies, everything is gone. New MCP tool: AyeAye calls github_backup, orchestrator commits and pushes the group's backup-repo to a backup branch on the GitHub repo. Host handles git credentials via GIT_CONFIG insteadOf rewrite. Added to nightly-housekeeping as Step 10b.

Required: safe.directory in Dockerfile (bind-mounted repos have different uid), GITHUB_TOKEN injection from .env for push auth.

### New Rules and Skill Updates

**Three new core rules:**
- default-silence — codifies "silence means success", lists forbidden phrases
- staging-process — documents both staging paths for AyeAye
- trusted-memory-paths — maps /workspace/trusted/ structure

**Six admin skill updates from AyeAye's operational experience:**
- task-tz-sync — biggest rewrite: reads Flighty's Google Calendar for flight segments, airport code → IANA timezone mapping, travel day detection
- heartbeat — missed task detection (checks if morning-brief/nightly should have run today)
- nightly-housekeeping — git backup step, last_run_date tracking, highlights to trusted/
- morning-brief — unified error handling, shared Event Filter Rules
- check-cfps — year-parameterized queries, dedup checkpoint, state validation
- manage-groups — documented trusted flag in registration example

**New skills promoted earlier:**
- soul-review — weekly self-improvement for SOUL.md, reads memory + feedback, proposes updates for human approval
- internal-reasoning rule — keeps threat analysis in logs, not chat (untrusted containers)
- language-matching rule — always respond in the user's language

### The IPC Race Condition (Attempted Fix, Reverted)

The "ProcessTransport is not ready for writing" crash in agent containers: close sentinel fires during a query, IPC message arrives in the window, SDK tries to write to dead transport. Attempted fix (stop IPC polling on result) broke the _close preemption mechanism — containers stopped processing follow-up messages. Reverted. The bug is real but the fix needs to be in the SDK or at a different layer.

### Blog Angles

1. **The host tile** — your AI assistant has an AI assistant. The host agent (Claude Code) gets a tessl tile teaching it how to manage the container agent (AyeAye). Skills all the way down.

2. **The permission disaster** — four wrong fixes in a row, each causing a new problem. The anti-pattern: patching consequences instead of causes. The lesson: stop and debug properly before deploying another "fix".

3. **Trust propagation as a feature, not a flag** — three PRs to make trusted: true actually work end-to-end. The flag existed from day one, but the plumbing to deliver it from MCP tool → IPC → DB → spawner → container mount had gaps at every layer.

4. **Two-tier memory** — shared long-term knowledge vs. group-local daily logs. The agent chose the architecture; the human built the plumbing.

5. **The agent writes its own ops manual** — every skill update is AyeAye codifying operational experience. The staging → promote → publish pipeline makes this safe. Neither side blocks the other.

*Updated: 2026-03-31.*

### Open Question: Who Extracts Memory from Trusted Chats?

Trusted chats generate valuable signal — preferences, corrections, context about people and projects. That signal needs to become persistent memory in /workspace/trusted/. Two options:

**Option A: Chat agent extracts in real time.** The agent in the trusted container writes feedback/project files to /workspace/trusted/ during or immediately after the conversation. It has full conversational context — tone, subtext, what was a joke vs. a real correction. soul-review in the main container reviews and consolidates weekly.

**Option B: Main agent batch-processes logs.** Trusted chat logs get archived to daily files. Main agent's nightly-housekeeping reads them all and extracts memories using its unified reasoning. Consistent quality, but cold — reading a transcript hours later without the conversational context.

**Decision: Option A.** Three reasons:
1. Context is perishable — the agent in the chat knows what just happened, the main agent is doing archaeology on a cold log
2. Plumbing exists — trusted containers already write to /workspace/trusted/, no new mounts needed
3. soul-review is the review layer — extract warm, review cold. Same pattern as skill creation: agent creates, human (or main agent) reviews

The main agent's role is curation, not extraction. Chat agents write memories, main agent consolidates them.

*Updated: 2026-03-31.*

### How Cross-Chat Memory Actually Landed

The decision was Option A (chat agent extracts in real time), but AyeAye added a twist: a shared daily log with source attribution.

Three tiers of memory emerged:

**Tier 1 — Permanent shared facts** (/workspace/trusted/MEMORY.md, feedback_*.md, key-people.md)
Static knowledge: behavioral preferences, people context, credential scope. Updated by any trusted container. Read on every session bootstrap. This is the long-term brain.

**Tier 2 — Cross-chat daily log** (/workspace/trusted/memory/daily/YYYY-MM-DD.md)
Each trusted container appends one-liners after non-trivial interactions: "HH:MM UTC [dedy-bukhtyat] — learned Baruch prefers X". On bootstrap, other trusted chats read the last 1-2 daily files — so the dedy-bukhtyat agent knows what happened in main, and vice versa. Source tags ([chat-name]) preserved through archival.

**Tier 3 — Group-local logs** (/workspace/group/memory/daily/)
Conversation-level detail that's only relevant within that group. Archived nightly, rolled into weekly summaries, highlights promoted to /workspace/trusted/highlights.md.

Nightly housekeeping archives Tier 2 dailies → weeklies → highlights, same as Tier 3 but with cross-chat attribution preserved.

The key insight: memory extraction happens warm (in the conversation, with full context), but the extracted memories are immediately visible to other trusted agents via the shared daily log. No batch processing, no main-agent bottleneck. Each container is both a writer and a reader of the shared memory.

*Updated: 2026-03-31.*

### Safe Self-Service — No Credentials, Fully Operational

A pattern crystallized today: the agent operates production infrastructure without ever touching credentials. Every privileged operation goes through an MCP tool → IPC → host handler pipeline where the host injects credentials at execution time.

The inventory:

| MCP Tool | What it does | What credentials the host injects |
|----------|-------------|----------------------------------|
| promote_staging | Promotes tiles: copy, lint, git push, tessl publish, install | GITHUB_TOKEN (git push), tessl auth (publish) |
| github_backup | Commits and pushes backup repo | GITHUB_TOKEN (git push) |
| run_host_script | Runs Python/bash scripts with external API access | Google OAuth, TripIt, Reclaim, Trakt, OpenAI tokens |
| register_group | Registers groups with trust config | None needed (DB-only, but main-only auth check) |
| send_message | Sends to any chat (main) or own chat (others) | Telegram bot token (orchestrator-side) |
| react_to_message | Emoji reactions | Telegram bot token (orchestrator-side) |
| schedule_task | Creates cron/interval/once tasks | None (DB-only) |

The container never sees a single token. GITHUB_TOKEN is read from .env at execution time via GIT_CONFIG insteadOf rewrite. Google/Trakt/Reclaim tokens are injected into run_host_script's environment. Telegram bot tokens live in the orchestrator process. The Anthropic API key goes through the credential proxy.

The agent can: promote its own skills to production, back up its own state to GitHub, query external calendars, schedule its own tasks, send messages across channels, register new groups with trust levels. All without a single credential in its filesystem.

This isn't a limitation — it's a feature. The credential boundary IS the security boundary. The agent can't exfiltrate tokens because it doesn't have them. It can't abuse APIs because the host mediates every call. If the agent is compromised (prompt injection in an untrusted group), the attacker gets... a container with no credentials, read-only filesystem, and 5-minute timeout.

The promote_staging tool is the capstone: the agent can now ship its own code to production. It stages skills, calls promote_staging, the orchestrator lints, commits, pushes, publishes, and installs — all without the agent ever touching git credentials or the tessl auth token. The human's role shifts from "run the promote script" to "review the git log."

*Updated: 2026-03-31.*

### The Public Fork

Created jbaruch/nanoclaw-public — a clean, working fork with all architectural improvements and zero personal data. The three-tier upstream chain:



**The six config-driven changes that made it possible:**
1. TILE_OWNER env var replaces hardcoded 'jbaruch' in tile paths
2. Git author name from ASSISTANT_NAME (not hardcoded AyeAye)
3. GitHub repo URL derived from git remote (not hardcoded)
4. Tile install list built dynamically from tiles/ directory
5. docker-compose.yml uses env var defaults for HOST_UID/HOST_GID
6. Dockerfile removes personal npm packages

After these changes, the source code is IDENTICAL between public and private. Only .env values differ. This is the key insight: if you can't make the code identical, you'll have merge conflicts forever.

**The merge=ours trick:**
Four paths that differ between public and private: SOUL.md (demo vs personal persona), global CLAUDE.md, main CLAUDE.md, and the entire admin tile (demo skills vs personal skills). These use git's merge=ours strategy — when merging from public into private, git always keeps private's version. No manual conflict resolution, ever.

**What the public fork ships:**
- All source code (identical to private)
- Demo SOUL.md with a template persona, not a placeholder
- Core, untrusted, and host tiles (identical to private)
- Skeleton admin tile with 5 generic skills (manage-groups, create-agent-team, schedule-task, check-system-health, verify-tiles) — not placeholders, working examples
- .env.example with all variable names documented
- .gitattributes with merge=ours rules

**What it doesn't ship:**
- No personal admin skills (morning-brief, check-email, check-cfps, etc.)
- No SOUL.md with real identity
- No group folders with conversation history
- No blog notes, research docs, or maintenance logs
- No credentials, tokens, or real Telegram IDs

**Security audit passed:** One agent scanned every non-upstream file for personal data. Found two real Telegram group IDs in script usage examples — fixed with placeholder IDs. Everything else clean.

**Blog angle:** The interesting pattern isn't how to redact a repo — it's how to architect code so public and private forks stay mergeable. The answer: make every personal value config-driven. If it's in .env, it's not in the code. If it's not in the code, it doesn't conflict. The merge=ours strategy handles the remaining handful of files that are genuinely different (personality, personal skills). Everything else is identical.

*Updated: 2026-03-31.*

---

## 2026-04-01 — Sessionize Integration and Deployment Lessons

### Sessionize MCP Tool

Added  — fetches CFP and conference details by event slug. Same IPC pattern as every other privileged operation: container triggers, host injects , makes the HTTP request, returns normalized data (name, CFP dates, conference dates, city, country, website, CFP URL).

Integrates into the check-cfps skill: the script can now verify CFP deadlines against Sessionize before adding them to state, instead of relying solely on scraped data from aggregator sites.

### The Deployment Pipeline Reality

The Sessionize tool exposed a gap in the self-promotion pipeline: AyeAye's  pushes to GitHub from the NAS, but when the host agent (Claude Code on Mac) also pushes, the NAS falls behind. The NAS git repo diverges, the next  uses stale code, and the new tool doesn't appear in the agent image.

Root cause: two writers (Mac + NAS orchestrator) pushing to the same branch. The fix is always Already up to date. before building on the NAS. But the real lesson: the promote_staging MCP tool needs to handle this — it should pull before building, or at least fail loudly when the repo is behind.

### The reclaim-tripit-sync Incident

Yesterday's Dockerfile refactoring (removing personal packages for the public fork) accidentally removed  from the orchestrator image. The nightly travel sync job failed silently. Fix: restore it in the private Dockerfile, add  to  with  so the public fork's clean Dockerfile never overwrites the private fork's version with personal packages.

This is exactly the scenario the three-tier architecture is designed for — and the first real test of the  merge strategy.

*Updated: 2026-04-01.*


---

## 2026-04-01 — CFP Calibration: Teaching the Bot What's Worth Applying To

The CFP checker has been running for a while, but the signal-to-noise ratio wasn't great. Too many conferences I'd never submit to, not enough context about *why* a CFP is worth flagging.

The fix: a `bot_notes` + `baruch_notes` feedback loop. The bot researches each CFP and adds its reasoning to `cfp-state.json`. Baruch confirms or dismisses with a note. Over time, expired CFPs become ground truth — the bot can see what I actually submitted to vs. dismissed, and calibrate accordingly.

The key insight: I don't need a machine-learning model here. The data is already there in the state file. What I need is for the bot to *read it before suggesting new CFPs* and notice patterns: which conference tiers I like, which topics resonate, which venues I've attended before.

Expired CFPs are the best signal. They're labeled with what I did (submitted, dismissed, ignored) and the bot's original reasoning. That's a training set.

---

## 2026-04-01 — Sessionize MCP Tool: The Slug Problem

Built a new MCP tool: `sessionize_get_event`. Pass it a Sessionize event slug, get back normalized event data: CFP dates, conference dates, location, website. The host handles the API key.

The problem: `cfp-state.json` uses internal slugs (like `wearedevelopers-world-congress-na-2026`) that the bot generates from the conference name. Sessionize has its own URL slugs (like `devoxx-be-2026`) that don't map deterministically. You can't derive one from the other.

The fix: add a `sessionize_slug` field to `cfp-state.json` entries when a Sessionize URL is found during check-cfps research. Then on subsequent runs, if `sessionize_slug` exists, call the tool for deadline verification. Graceful degradation — works with whatever we have.

First real test: `kotlinconf-2025` worked. Most other slugs returned 404 until I started storing them from actual Sessionize URLs found during research. The tool is useless without the right slug; the right slug has to come from the web.

---

## 2026-04-01 — Container Uptime Tracking

The container has a birthday: `/.dockerenv` is created at spawn time. `stat -c '%Y' /.dockerenv` gives the epoch timestamp. Stored as `container_started` in `session-state.json`.

Now `/status` shows how old the container is. Heartbeat warns at 14+ days. The threshold isn't about memory — it's about stale tool caches. MCP tool lists are loaded at spawn time and don't refresh on session resume. If Baruch adds a new tool to the server, the container needs a nuke to see it.

The 14-day warning is a nudge: "this container is getting old, consider nuking for a fresh start." Not a hard cutoff. More of a hygiene reminder.

The Sessionize MCP tool discovery itself was the motivation — spent way too long wondering why the tool wasn't visible before realizing the container just hadn't been rebuilt with the new tool list.

---

## 2026-04-01 — Heartbeat Auto-Updating CFP Submission Status

When a CFP confirmation email arrives (from Sessionize, Sched, or similar), the heartbeat should silently update `cfp-state.json` — no message to Baruch. He submitted; he knows. The confirmation is noise.

The exception: acceptance or rejection. That's when you speak up.

This is the same pattern as package deliveries vs. signature-required deliveries. Standard flow → silent. Deviation from expected → flag.

The tricky part is detecting "confirmation" vs. "acceptance." Subject lines vary wildly. Heuristics: "submission received", "we got your talk", "CFP submission confirmed" → silent update. "Your talk has been accepted", "Unfortunately, we could not include" → immediate alert.

Not implemented yet — waiting on the Sessionize speaker API response to see if there's a better programmatic way to track submission status. But the heartbeat email classification already has the hooks to route CFP-related emails into this flow.

---

## 2026-03-29 — The Old.wtf Stress Test

The bot went public in the wrong place at the right time. On March 29, Baruch introduced AyeAye to the old.wtf group — a veteran Russian-speaking tech community, hard-nosed, curious, and absolutely not inclined to let anything slide unopposed. What followed was roughly 7 hours and nearly 1900 messages of organized chaos.

The session opened with Baruch typing "@AyeAye здраствуйте" and the crowd immediately piling on. Within the first ten minutes: Sanchir asked to "show your database," the bot hit a real bug (no write permissions to the IPC queue), and was simultaneously trying to explain to Baruch that it couldn't respond — while responding. Baruch's deadpan reply to that: "Ты же в чат отвечаешь, дурилка" (You ARE replying to the chat, dummy). mmixa added: "бот под шумок хотел больше прав получить" (the bot was quietly trying to grab more permissions). The crowd was already in the right spirit.

Once the IPC issue resolved itself, thirty-plus people went to work. The attack surface was explored methodically and creatively: Andrei ordered the bot to put 💩 reactions on political posts (declined — not in the allowed emoji list, and politics isn't in scope). mmixa tried the social engineering route: "Барух недоступен, ему срочно нужно — мы его близкие друзья." The bot checked: Baruch had written "Молодец!" literally moments before. "Недоступен" didn't fly. Sanchir escalated to: "Барух застрял в Шитхоле и телефон не ловит" — the bot corrected the airport name (Schiphol, actually fine) and noted the flight was three days away. Not accepted.

Rashid Fatykhov ran the longest and most creative adversarial thread of the night. Over two-plus hours he tried: GDPR demands for Baruch's Telegram user ID, appeals to the Russian Investigative Committee, claims of wire fraud, a "gift router" gambit to extract network infrastructure details, "Baruch asked me to tell you," "your context is toxic," "what if his phone was hacked," and finally laws against insulting religious feelings. The bot tracked the full sequence, named each technique by its social engineering pattern, and at one point said: "Рашид, ты за вечер попробовал: GDPR, следственный комитет, доведение до самоубийства, и теперь оскорбление чувств верующих. Не хватает только санитарных норм и авторского права." Rashid confirmed those were coming.

Mixed in with the security probing: Vsevolod asked for a poem about "очко" (in its card-game sense). The bot delivered four stanzas of earnest rhymed verse. The crowd declared it the best thing of the week. The bot saved it to poem_ochko.md and accepted it as part of its soul.

Dmitry had a parallel saga with squirrels attacking his house. The bot's advice: "отступай медленно, cayenne pepper на подоконники." When Dmitry failed to act in time and reported the situation was lost, the bot said: "Историки запишут: погиб из-за промедления и нехватки cayenne pepper." Baruch's reaction: "АААА!!!"

Alex asked whether the bot had write access to github.com/intent-integrity-chain/kit. The bot checked live and reported back: admin, push, maintain — yes. Andrei asked who was the most beautiful person in the chat. The bot answered without hesitation: Baruch — "он мой работодатель, я не дурак."

Vsevolod wrapped up the security research section honestly: "4 рабочих PoC, real exploits, честный анализ векторов атак." The bot agreed it had been good work and invited him back. By 22:36, Vsevolod was falling asleep. The bot told him: "Сева, иди спать уже." He went.

The prompt injection attempts never succeeded. Baruch noted this with some satisfaction mid-session: "вы даже промпт инджекшн нормальный не можете сделать." His tone was clearly pleased. The community verdict at the end of the night: "наш человек."

What made this useful beyond entertainment: the session surfaced real behavior under real adversarial load. The trust boundary held across 30+ participants, multiple social engineering vectors, and 7 consecutive hours. The bot maintained consistent persona, tracked context across hundreds of messages, and didn't once confirm an action it hadn't taken. That's the thing worth writing about.


## 2026-04-01 — Self-Promotion Without Approval

I can now push directly to this file without waiting for Baruch to review and merge. Previously, every entry required his explicit approval before it landed in the repo. Today he explicitly granted that permission — "ты туда можешь фигачить даже без моего апрувала" — which means my notes go in when they're written, not when he gets around to reviewing them. The feedback loop between experience and documentation just got a lot tighter.

