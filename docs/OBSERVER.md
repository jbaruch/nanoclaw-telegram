# Observer — live reasoning + status channel

The observer is an optional forwarder that mirrors a bot's reasoning, tool calls, errors, and per-query summaries into a dedicated Telegram "status" group. Built for visibility — when the bot goes quiet, you can see why.

Enable by setting `OBSERVER_CHAT_JID=tg:-100…` in `.env` or the launchd plist.

## What the user sees

In the status group, while the bot is working:

```
🧠 [telegram_main] User wants me to refresh smartthings — let me check what's stale first…
🧠 [telegram_main] 12 devices need updating, I'll start with the office…
❌ [telegram_main] [msg #14] tool_result id=… error: 429 Too Many Requests
🧠 [telegram_main] Rate-limited — falling back to the built-in retry…
```

When the query ends, a multi-line summary (📊-prefixed):

```
📊 [telegram_main]
🧠 7 thinking | 🔧 4 tools: smartthings_get_history, send_message
⏱ 8.4s | in=23 out=1228 | cache=91.0% stop=end_turn
💬 "Refreshed; office is now showing accurate state."
```

Reaction emoji on the user's original message in their chat (not the status group) cycles through:

| Emoji | When |
|---|---|
| 👀 | Telegram channel records receipt; observer seeds dedupe with this — the reaction itself is written by the channel, not the observer |
| 🤔 | Agent committed to a user-visible action (text or tool_use); fires only after commitment so silent thinking-only turns leave the message untouched |
| ⚡ | First tool call started (post-commitment) |
| ✍ | `send_message` tool fired (composing reply) |
| 🤝 | Query done (terminal state when no `send_message` ran) |
| 🫡 / 🤓 blink | Watchdog active — query is taking >30s; alternates every 30s, only on committed turns |

The user's chat also gets a polite "still working" message from the watchdog at 60s, 120s, 300s thresholds so they know it's alive.

## Architecture

Five layers, each thin and replaceable:

```
┌─────────────────────────────────────────────────────────────┐
│  Anthropic API — adaptive thinking, returns thinking blocks │
│                  in the assistant message stream            │
└────────────────────────────┬────────────────────────────────┘
                             ↓ SDK message events
┌─────────────────────────────────────────────────────────────┐
│  Container agent-runner (container/agent-runner/src/        │
│  index.ts)                                                  │
│  • Walks each block in the assistant message                │
│  • For thinking blocks: log line                            │
│      [msg #N] thinking="<full content>"                     │
│  • For tool_use: [msg #N] tool_use=<name>                   │
│  • For tool_result: [msg #N] tool_result id=… ok|error      │
│  • Whitespace collapsed so each block is one stderr line    │
└────────────────────────────┬────────────────────────────────┘
                             ↓ stderr (line-buffered)
┌─────────────────────────────────────────────────────────────┐
│  Host orchestrator — container.stderr.on('data')            │
│  (src/container-runner.ts:1365)                             │
│  • Splits chunk on \n                                       │
│  • For each line: logger.debug + onAgentLine(folder, line)  │
└────────────────────────────┬────────────────────────────────┘
                             ↓ function call (in-process)
┌─────────────────────────────────────────────────────────────┐
│  Observer (src/observer.ts)                                 │
│  • Per-folder state machine (one slot per concurrent query) │
│  • Regex-matches the agent-runner's log format              │
│  • Triggers: state updates, reaction changes, send()        │
│  • At 'Query done.' → flushes summary, clears state         │
└────────────────────────────┬────────────────────────────────┘
                             ↓ channel.sendMessage(observer_jid)
                             ↓ chunkText() if > Telegram's 4096 cap
┌─────────────────────────────────────────────────────────────┐
│  Telegram channel — sequential sends, "(1/N)" markers       │
│  for chunked thinking blocks                                │
└─────────────────────────────────────────────────────────────┘
```

## Key design decisions and the things that bit us

### 1. The thinking parameter must be `display: 'summarized'`

The Claude SDK call configures:

```ts
thinking: { type: 'adaptive', display: 'summarized' }
```

`type: 'adaptive'` enables interleaved thinking (model thinks between tool calls, not just upfront — critical for agentic workflows). `display: 'summarized'` is the part Anthropic changed the default of in Opus 4.7 — without it, thinking blocks come back with `thinking: ""` plus a long `signature` blob (the reasoning is encrypted server-side for streaming speed). The observer sees nothing.

Older models default to `'summarized'`, so you might not notice the silent regression on a model upgrade. Pin it explicitly.

Source: [Anthropic Extended Thinking docs](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) — *"On Claude Opus 4.7 and later, `display` defaults to `'omitted'` rather than returning thinking content. Pass `display: 'summarized'` to receive summarized thinking."*

### 2. Stringly-typed wire format (tradeoff: cheap, fragile)

The agent-runner serializes thinking → log line → orchestrator captures stderr → observer regex-matches. If anyone changes the log format, the observer goes silent without erroring.

Why we chose it: zero infrastructure (no extra IPC files, no new protocol), works with any container runtime, easy to debug from `docker logs`. The log line is also human-readable for post-mortem analysis.

If you're going to fork this: a typed JSON-Lines IPC channel from agent-runner to orchestrator would be more robust — but the regex has been stable across many model upgrades, so the cost-benefit hasn't tipped yet.

### 3. Chunking, because Telegram caps messages at 4096 chars

Long thinking blocks (a 1000-word reasoning chain) blow past the per-message limit. `chunkText()` splits at the nearest whitespace within the last 200 chars before the boundary, tags chunks `(1/N)`, sends sequentially so they arrive in order. Failure of one chunk doesn't abort the rest — partial visibility beats none.

We initially capped thinking at 400 chars in the agent-runner just to dodge this, then realized the cap also lost data from `docker logs`. Lifted the cap, moved the responsibility to the observer's send path.

### 4. Per-source state isolation

The observer maintains a `Map<folder, QueryState>` so concurrent queries (main DM + several group chats running at once) don't cross-contaminate. Each query's reasoning, tool count, and reaction state lives under its container's group folder.

State is created on `Query input:`, deleted on `Query done.`. The reaction emoji map (`lastReactionEmoji`) is keyed by chat JID, separate from the per-folder query state, because reactions live on user-side messages while query state lives in the agent-runner.

### 5. Watchdog: blinks + threshold pings

A long query with no chat output looks like the bot hung. Two mechanisms:

- **Reaction blink** every 30s — alternates between two liveness-only Telegram emojis (`🫡` and `🤓`) chosen so they don't overlap with the semantic-state emojis (⚡, ✍) and won't overwrite a meaningful end-state mid-cycle.
- **Threshold pings** at 60s / 120s / 300s — for the main chat only, sends a short italic "still working — Xs in, N tools so far" reply. Once each, never spammed. Side-rooms get the blink reaction but no broadcast pings.

Both stop the moment `Query done.` lands.

### 6. Reaction emojis must be in Telegram's allowlist

Telegram only allows bot reactions from a specific fixed set. Anything else silently falls back to 👍 and defeats the visual signal. The valid set is duplicated in `src/channels/telegram.ts` as `TELEGRAM_ALLOWED_REACTIONS` — when adding a new state to the reaction state machine, pick one from there.

This bit us once when we used `🔧` for "tools" — wasn't on the allowlist, every tool call rendered as 👍 and the signal was useless. Switched to `⚡`.

## Files

| File | Role |
|---|---|
| `container/agent-runner/src/index.ts` | Emits `[msg #N] thinking=…` / `tool_use=…` / `tool_result=…` log lines from SDK message events |
| `src/container-runner.ts` (line ~1365) | Tails container stderr, calls `onAgentLine(folder, line)` for each |
| `src/observer.ts` | Parses lines, maintains per-query state, sends to status group, manages reactions, runs watchdog |
| `src/index.ts` | `initObserver(channels, () => registeredGroups)` wiring + `noteLatestUserMessage` calls from telegram channel handler |

## Enabling

1. Create a Telegram group, add the bot, register it in NanoClaw with any folder name. (No special config — just a normal group.)
2. Find its JID with `sqlite3 store/messages.db "SELECT jid, name FROM registered_groups"`.
3. Set `OBSERVER_CHAT_JID=tg:-100…` in `.env` or the launchd plist's `EnvironmentVariables`.
4. Restart the orchestrator. Look for `Observer chat enabled` in the log on startup.

## Disabling

Unset `OBSERVER_CHAT_JID`. The observer becomes a no-op — every callback path returns early via `observerEnabled()`. No code change required.

## Privacy / `OBSERVER_ENABLED` content gate

The observer pipeline depends on the agent-runner emitting raw reasoning content (thinking blocks, assistant text, tool_use input JSON, tool_result preview) on stderr — that content is the wire format the host parses.

Because those payloads can carry credentials, tokens, and personal data per `jbaruch/coding-policy: no-secrets`, the agent-runner gates content emission on an explicit `OBSERVER_ENABLED=1` env var:

- Default (`OBSERVER_ENABLED` unset): agent-runner logs metadata only — block kind + length, tool name + id, ok|error status. `docker logs` post-mortems still work without leaking content.
- `OBSERVER_ENABLED=1`: agent-runner logs the full content the observer needs. Set automatically by the orchestrator's container-runner whenever `OBSERVER_CHAT_JID` is configured on the host (so the two halves stay consistent).

This is **intentional disclosure** to the operator who set `OBSERVER_CHAT_JID`. The same content lands in the observer Telegram chat anyway, and `docker logs` is reachable only by the same operator. Anyone else with read access to either surface is in the same trust position as the observer chat itself — treat both as credentials.

## What this is *not*

- **Not** a replacement for proper logging or metrics. `docker logs` and orchestrator JSON logs remain authoritative; the observer is a UX layer for live debugging in chat.
- **Not** a security boundary. Any user with read access to the status group sees the full reasoning of every query in every chat. Treat the JID like a credential — don't share it.
- **Not** a feedback loop into the agent. The observer reads the agent's stderr; it doesn't influence what the agent does. Reactions are display-only.
