# NanoClaw Operations Runbook

## Architecture

- **NAS** (192.168.10.32): orchestrator container + agent containers via Docker socket
- **Mac**: dev machine only — edit code, push to git
- **Repo**: `~/nanoclaw` on NAS, `/Users/jbaruch/Projects/nanoclaw` on Mac

## Common Operations

### Redeploy after code changes

```bash
# On Mac: push changes
git push origin main

# On NAS:
ssh 192.168.10.32
cd ~/nanoclaw
git pull
docker compose up -d --build    # rebuilds orchestrator image
```

Agent containers use the latest image automatically on next spawn. No rebuild needed unless the agent Dockerfile changed.

### Rebuild agent image (after Dockerfile or tile changes)

```bash
ssh 192.168.10.32
cd ~/nanoclaw/container
./build.sh                      # requires tessl credentials at ~/.tessl/
cd .. && docker compose restart  # restart orchestrator to use new image
```

### Install a new tessl tile

1. Add tile name to `container/Dockerfile` in the `tessl install` line
2. If the tile needs env vars, add them to `FORWARDED_ENV_VARS` in `src/container-runner.ts`
3. Add env var values to `.env` on the NAS
4. Push, pull on NAS, rebuild agent image, restart orchestrator:
   ```bash
   ssh 192.168.10.32
   cd ~/nanoclaw && git pull
   cd container && ./build.sh
   cd .. && docker compose restart
   ```

**Known issues with tessl tiles in Docker:**
- Tessl creates skills as **symlinks** to vendored tiles in `.tessl/tiles/`. The entrypoint uses `cp -rL` to dereference them when copying to the bind-mounted `.claude/skills/`.
- Tessl sets tile directories to **700 permissions**. The Dockerfile runs `chmod -R a+rX /opt/tessl-staging` to make them readable by any UID (needed because agent containers run as HOST_UID, not the image's node user).
- If a tile appears missing after rebuild, check for **stale broken symlinks** in `data/sessions/telegram_swarm/<slot>/.claude/skills/` (where `<slot>` is `default` or `maintenance`). Delete them: `find data/sessions/telegram_swarm/*/.claude/skills/ -xtype l -delete` (`-xtype l` matches broken symlinks only, leaving valid ones intact).
- Tessl requires **BuildKit** for secret mounts. buildx is installed on the NAS at `~/.docker/cli-plugins/docker-buildx`.
- Tessl credentials at `~/.tessl/api-credentials.json` are mounted as a Docker secret during build — never baked into the image layer.

### Add env vars for agent containers

1. Add the var to `FORWARDED_ENV_VARS` in `src/container-runner.ts`
2. Add the value to `~/nanoclaw/.env` on the NAS
3. Push, pull, rebuild orchestrator: `docker compose up -d --build`

### Create a new skill (AyeAye)

AyeAye writes to `/workspace/group/skills/{name}/SKILL.md`. Available on next container spawn.

### Promote AyeAye skill to tile

```bash
# On Mac:
# First, copy the skill from NAS to local
ssh 192.168.10.32 "cat ~/nanoclaw/groups/telegram_swarm/skills/SKILL_NAME/SKILL.md" > /tmp/skill.md
# Review it, then:
./scripts/promote-skill.sh SKILL_NAME
```

### Clear session (force fresh conversation)

```bash
ssh 192.168.10.32
sqlite3 ~/nanoclaw/store/messages.db "UPDATE sessions SET session_id = NULL WHERE group_folder = 'telegram_swarm'"
cd ~/nanoclaw && docker compose restart
```

### Nuke session (kill container + start fresh)

To kill the running agent container and force a completely fresh session:

```bash
# Kill running agent containers for the swarm group
ssh 192.168.10.32 "docker ps --filter name=nanoclaw-telegram-swarm -q | xargs -r docker kill"

# Clear the stored session ID so next spawn doesn't resume
ssh 192.168.10.32 "sqlite3 ~/nanoclaw/store/messages.db \"UPDATE sessions SET session_id = NULL WHERE group_folder = 'telegram_swarm'\""

# Restart orchestrator
ssh 192.168.10.32 "cd ~/nanoclaw && docker compose restart"
```

Next message to AyeAye starts a completely new session — no prior context, fresh RULES.md, fresh SOUL.md.

### View logs

```bash
# Orchestrator logs
ssh 192.168.10.32 "docker compose -f ~/nanoclaw/docker-compose.yml logs --tail 50"

# Agent container logs (per-group)
ssh 192.168.10.32 "ls ~/nanoclaw/groups/telegram_swarm/logs/"
ssh 192.168.10.32 "cat ~/nanoclaw/groups/telegram_swarm/logs/container-*.log | tail -50"

# NanoClaw application log
ssh 192.168.10.32 "tail -50 ~/nanoclaw/logs/nanoclaw.log"
```

### External heartbeat

Runs on NAS host via cron every 15 min. Set up:

```bash
# Requires sudo on the Ugreen NAS:
echo '*/15 * * * * /home/jbaruch/nanoclaw/scripts/heartbeat-external.sh >> /home/jbaruch/nanoclaw/logs/heartbeat.log 2>&1
0 * * * * /home/jbaruch/nanoclaw/scripts/logrotate.sh /home/jbaruch/nanoclaw/logs' | sudo crontab -u jbaruch -

# Verify:
sudo crontab -u jbaruch -l
```

**Note:** `crontab -e` doesn't work over SSH (terminal type issue). Use the pipe method above.

Config: `~/nanoclaw/scripts/heartbeat-external.conf` (not in git — contains bot token)

### Update scheduled tasks

Tasks are in SQLite. To update a task prompt:

```bash
ssh 192.168.10.32
sqlite3 ~/nanoclaw/store/messages.db "SELECT id, substr(prompt,1,60) FROM scheduled_tasks WHERE status='active'"
sqlite3 ~/nanoclaw/store/messages.db "UPDATE scheduled_tasks SET prompt='new prompt' WHERE id='task-xxx'"
```

## File Locations (NAS)

| Path | Content | Persists |
|------|---------|----------|
| `~/nanoclaw/` | Git repo (code, tiles, Dockerfiles) | Git |
| `~/nanoclaw/.env` | All credentials | Manual |
| `~/nanoclaw/store/messages.db` | Messages, tasks, sessions | Volume |
| `~/nanoclaw/data/` | Sessions, IPC, nanoclaw.db | Volume |
| `~/nanoclaw/groups/telegram_swarm/` | Group memory, state, conversations, skills | Volume |
| `~/nanoclaw/groups/global/` | SOUL.md, global CLAUDE.md, HEARTBEAT.md | Volume |
| `~/nanoclaw/logs/` | Application and heartbeat logs | Volume |
| `~/.tessl/api-credentials.json` | Tessl auth (for agent image build) | Manual |

## Credentials (.env)

| Variable | Source | Used by |
|----------|--------|---------|
| `ANTHROPIC_API_KEY` | console.anthropic.com | Credential proxy → agent containers |
| `TELEGRAM_BOT_TOKEN` | @BotFather | Orchestrator (main bot) |
| `TELEGRAM_BOT_POOL` | @BotFather (6 bots) | Orchestrator (agent swarm) |
| `OPENAI_API_KEY` | platform.openai.com | Voice transcription (Whisper) |
| `COMPOSIO_API_KEY` | app.composio.dev | Agent containers, main/trusted only. Project-scoped `ak_*` key; sent as `x-api-key` to BOTH Composio surfaces — REST (`composio-fetch` precheck) and the headless custom MCP server (`mcp__composio__*`: Gmail, Calendar, Tasks). |
| `COMPOSIO_MCP_URL` | app.composio.dev (custom MCP server) | Agent containers, main/trusted only. URL of the headless custom MCP server (`backend.composio.dev/v3/mcp/<id>/mcp`); the agent runner appends `?user_id=$COMPOSIO_USER_ID` and authenticates with `x-api-key`. Replaced the retired `COMPOSIO_MCP_KEY` after Composio's consumer "Connect" gateway moved to interactive OAuth. Account-identifying server id → env-file 0600 like the key. |
| `COMPOSIO_USER_ID` | app.composio.dev (connected-accounts list) | Agent containers, main/trusted only (binds Composio REST + MCP calls to the user's connected accounts; account-identifying, treated like the API key) |
| `GITHUB_TOKEN` | github.com/settings/tokens | Host scripts (git push via IPC) with the real `.env` value; **and** the `gh` CLI in main/trusted containers via OneCLI placeholder + gateway swap (`ONECLI_MANAGED_VARS`) — cost-monitor dashboard skills run `gh issue edit/comment` |
| `BYAIR_MCP_URL` | byairapp.com/mcp (Pro) | Agent containers, main/trusted only. byAir flight-status polling (`jbaruch/nanoclaw-travel` flight-assist precheck); API key inline in the URL → forwarded real (env-file 0600; query-embedded, injection-gap) |
| `GOOGLE_MAPS_API_KEY` | console.cloud.google.com | Agent containers, main/trusted only. Distance Matrix traffic-aware time-to-leave (same tile). OneCLI-managed (placeholder + swap) |
| `TOMTOM_API_KEY` | developer.tomtom.com | Agent containers, main/trusted only. TomTom geocode + `calculateRoute` (`api.tomtom.com`) — routing backup behind Google Maps + the `drive-planner` skill (same tile). OneCLI-managed (placeholder + swap) |
| `YOUTUBE_API_KEY` | console.cloud.google.com | Agent containers, main/trusted only. Native YouTube Data API for the admin tile's `youtube-comment-check` skill. OneCLI-managed (placeholder + swap) |
| `TRIPIT_ICAL_URL` | TripIt settings | Host scripts only (tripit-reclaim sync) |
| `RECLAIM_API_TOKEN` | reclaim.ai settings | Host scripts only (tripit-reclaim sync) |
| `GOOGLE_CLIENT_ID` | GCP console | Host scripts only (Calendar OOO blocks) |
| `GOOGLE_CLIENT_SECRET` | GCP console | Host scripts only (Calendar OOO blocks) |
| `GOOGLE_REFRESH_TOKEN` | OAuth flow | Host scripts only (Calendar OOO blocks) |

Forwarded-into-container credentials live in `src/container-runner.ts` (`CONTAINER_VARS` is the full forwarded list; `SECRET_CONTAINER_VARS` is the subset routed through a mode-0600 env-file rather than `-e` so it stays off `docker ps`; `ONECLI_MANAGED_VARS` is the subset whose real value is NOT forwarded).

**OneCLI-managed (`ONECLI_MANAGED_VARS`)** — with the agent proxy live (`ONECLI_AGENT_PROXY=1`), the container receives an `onecli-managed` placeholder and OneCLI's MITM gateway injects the real vaulted value on the outbound request (real key never in the agent environ; falls back to real-value forwarding when the proxy is off):

- `GOOGLE_MAPS_API_KEY` — Distance Matrix (`maps.googleapis.com`, param `key`).
- `TOMTOM_API_KEY` — TomTom geocode + `calculateRoute` (`api.tomtom.com`, param `key`; routing backup + `drive-planner`).
- `YOUTUBE_API_KEY` — native YouTube Data API (`www.googleapis.com` path `/youtube/*`, param `key`; admin tile's `youtube-comment-check`).
- `GITHUB_TOKEN` — the `gh` CLI inside main/trusted containers (`api.github.com`, `Authorization: Bearer`). The real value stays in host `.env` for the host-side `github_backup` `git push`; only the container is placeholdered.

**Still forwarded as real** (env-file 0600) — OneCLI can't vault these yet or they're retiring:

- The three Composio values — `COMPOSIO_API_KEY` (`x-api-key` for both Composio REST and the headless custom MCP server), `COMPOSIO_MCP_URL` (the account-specific `/v3/mcp/<id>/mcp` URL), and `COMPOSIO_USER_ID` (account-identifying). Retiring with #639.
- `BYAIR_MCP_URL` — byAir flight-status polling; token inline in the query string (OneCLI query-param injection not yet available).
- `SESSIONIZE_*` — CFP discovery/verification; key embedded in the URL path (OneCLI path-segment injection not yet available).

All forward to main/trusted tiers only, **never** untrusted. Everything else stays host-side and is reached through host scripts invoked via IPC. `docs/SECURITY.md` §4 is the authoritative per-tier view. Finishing the OneCLI migration (#564) — retiring Composio and growing OneCLI path/query injection — is the remaining work.

## Agent Container Capabilities

Installed in the agent image (`container/Dockerfile`):
- **Claude Code** + Agent SDK
- **Chromium** (agent-browser for web automation)
- **poppler-utils** (`pdftotext` for PDF text extraction)
- **Whisper** (voice transcription via OpenAI API, runs in orchestrator)
- **Tessl** (tile skills, library docs MCP)
- **Composio** (Google Calendar, Gmail, etc. via HTTP MCP)

Media handling in orchestrator (`telegram.ts`):
- **Photos**: downloaded to `/workspace/group/images/`, path passed to agent
- **Documents/PDFs**: downloaded to `/workspace/group/documents/`, agent reads with `pdftotext` or `Read` tool
- **Voice**: transcribed by Whisper, text passed to agent

## Docker Images

| Image | Purpose | Built by |
|-------|---------|----------|
| `nanoclaw-nanoclaw` | Orchestrator (Node.js + Docker CLI) | `docker compose build` |
| `nanoclaw-agent:latest` | Agent container (Claude Code + tools) | `container/build.sh` |

## Troubleshooting

### Agent not responding
1. Check orchestrator: `docker compose logs --tail 20`
2. Check if container spawned: `docker ps -a --filter name=nanoclaw-`
3. Check group logs: `ls groups/telegram_swarm/logs/`

### Skills not loading
1. Check `.claude/skills/` (replace `<slot>` with `default` for user-facing or `maintenance` for scheduled): `ls data/sessions/telegram_swarm/<slot>/.claude/skills/`
2. Check RULES.md: `cat data/sessions/telegram_swarm/<slot>/.claude/RULES.md | head -20`
3. Rebuild agent image if tiles changed: `cd container && ./build.sh`

### Credential proxy not working
1. Check port 3001: `docker compose logs | grep proxy`
2. Verify ANTHROPIC_API_KEY in .env
3. Agent containers reach proxy via `host.docker.internal:3001`

### Session stale / wrong behavior
1. Clear session: see "Clear session" above
2. Container will start fresh on next message

## Upstream Merge Conflicts

When running `/update-nanoclaw`, these files will always conflict because we've diverged from upstream:

| File | Our change | Resolution |
|------|-----------|------------|
| `package.json` | Removed @onecli-sh/sdk, added grammy/openai | Keep our removals + upstream version bumps/dep changes |
| `package-lock.json` | Different dep tree | Accept upstream, run `npm install` to regenerate |
| `src/config.ts` | Added HOST_PROJECT_ROOT, HOST_UID/GID, CREDENTIAL_PROXY_PORT, removed ONECLI_URL | Keep our additions, accept upstream additions (e.g., MAX_MESSAGES_PER_PROMPT) |
| `src/container-runner.test.ts` | Added HOST_PROJECT_ROOT/UID/GID mocks, credential proxy mock | Keep our mocks, accept upstream mock changes |
| `src/index.ts` | Credential proxy startup, pendingReplyTo, removed OneCLI | Keep our proxy code, accept upstream features |

**Quick resolution recipe:**

For each conflicted file, the rule is: **keep our additions (credential proxy, HOST_*, pendingReplyTo), accept upstream additions (new features, version bumps, dep removals), drop anything that references OneCLI from either side.**

After resolving: `npm run build && npm test` — both must pass before committing.
