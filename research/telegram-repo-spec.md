# Telegram Integration for NanoClaw — Open Source Repo Spec

## Goal
Extract the Telegram channel integration from our private NanoClaw fork into a public GitHub repo. This is a reference implementation + installable skill that adds production-grade Telegram support to any NanoClaw instance.

## Repo name
`nanoclaw-telegram` (under jbaruch GitHub)

## What goes in

### Code (from our fork)
- `src/channels/telegram.ts` (749 lines) — the full channel implementation
- `src/channels/telegram.test.ts` (962 lines) — tests
- Relevant types from `src/types.ts` (Channel interface, NewMessage)

### Features to document
1. **Bidirectional reply threading** — outbound messages quote the triggering message via reply_to parameter; inbound replies include quoted context
2. **Quote context resolution** — when user replies to a message, agent sees `[Replying to Sender: "text..."]`; when user quotes specific text, agent sees `[Quoted: "selected text..."]`
3. **Voice transcription** — Whisper API integration, voice messages become `[Voice: transcript text]`
4. **Photo handling** — downloads highest-res photo to workspace, passes container path
5. **Document/PDF handling** — downloads to workspace documents/ dir, agent reads with pdftotext
6. **Bot pool for Agent Swarm** — multiple bot identities in one group, each team member posts as a distinct bot
7. **ACK-and-async pattern** — first message quotes the trigger, background agent results use explicit reply_to
8. **`<internal>` tag awareness** — server-side stripping on all output paths
9. **Message ID in XML** — `<message id="..." sender="..." time="...">` format for agent context
10. **Trigger pattern** — configurable @mention trigger, requiresTrigger flag per group
11. **Chat metadata** — group detection, topic/thread support

### Supporting changes (patterns to document, not extract)
- `pendingReplyTo` map in orchestrator (src/index.ts)
- `reply_to` parameter on `send_message` MCP tool (container/agent-runner/src/ipc-mcp-stdio.ts)
- `<internal>` tag stripping in IPC and task scheduler paths
- Message formatting with IDs in router.ts
- `replyToMessageId` in group-queue.ts IPC messages

### Skill file
- `.claude/skills/add-telegram/SKILL.md` — updated version of upstream's skill with our improvements documented

## Structure

```
nanoclaw-telegram/
├── README.md                  # Architecture, features, installation
├── LICENSE                    # MIT
├── src/
│   └── channels/
│       ├── telegram.ts        # Main implementation
│       └── telegram.test.ts   # Tests
├── .claude/
│   └── skills/
│       └── add-telegram/
│           └── SKILL.md       # Installation skill (updated with our features)
├── docs/
│   ├── REPLY_THREADING.md     # How reply threading works end-to-end
│   ├── VOICE_AND_MEDIA.md     # Voice, photos, documents handling
│   ├── BOT_POOL.md            # Agent swarm with multiple bot identities
│   └── PATTERNS.md            # ACK-and-async, internal tags, message IDs
└── examples/
    └── orchestrator-changes/  # Diffs showing what the orchestrator needs
        ├── index.ts.diff      # pendingReplyTo, reply_to in output callback
        ├── ipc.ts.diff         # internal tag stripping, bot message storage
        ├── router.ts.diff     # message ID in XML
        └── group-queue.ts.diff # replyToMessageId in IPC
```

## README outline

1. What this is (Telegram channel for NanoClaw with production features)
2. Features list with screenshots
3. Installation (run /add-telegram skill OR manual setup)
4. Architecture diagram (orchestrator ↔ Telegram API ↔ bot pool)
5. Orchestrator changes required (the diffs in examples/)
6. Configuration (env vars, bot pool, trigger patterns)
7. Voice/media setup (OPENAI_API_KEY for Whisper, poppler-utils for PDF)

## What stays in our private fork
- NanoClaw orchestrator code (index.ts, container-runner.ts, etc.)
- Tiles and tile infrastructure
- NAS deployment (docker-compose, Dockerfile.orchestrator)
- AyeAye-specific SOUL.md, CLAUDE.md, skills
- Credentials and group configs

## Implementation steps
1. Create repo on GitHub (jbaruch/nanoclaw-telegram)
2. Copy telegram.ts and test file
3. Write README with architecture docs
4. Create the updated add-telegram SKILL.md
5. Create the pattern docs (reply threading, voice/media, bot pool)
6. Generate orchestrator diffs as examples
7. Add LICENSE (MIT)
8. Optionally: create a tessl tile from the repo (`tessl skill publish`)

## Future: upstream PR
After the repo is public, PR to upstream NanoClaw's `skill/telegram` branch. The PR updates the SKILL.md to include our improvements. Links to our repo for reference implementation.
