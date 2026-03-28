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
- If a tile appears missing after rebuild, check for **stale broken symlinks** in `data/sessions/telegram_swarm/.claude/skills/`. Delete them: `find data/sessions/telegram_swarm/.claude/skills/ -type l -delete`
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
ssh 192.168.10.32
crontab -e
# Add:
*/15 * * * * /home/jbaruch/nanoclaw/scripts/heartbeat-external.sh >> /home/jbaruch/nanoclaw/logs/heartbeat.log 2>&1
0 * * * * /home/jbaruch/nanoclaw/scripts/logrotate.sh /home/jbaruch/nanoclaw/logs
```

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
| `COMPOSIO_API_KEY` | app.composio.dev | Agent containers (Google Calendar, Gmail) |
| `GITHUB_TOKEN` | github.com/settings/tokens | Agent containers (git push) |
| `TRIPIT_ICAL_URL` | TripIt settings | Agent containers (tripit-reclaim sync) |
| `RECLAIM_API_TOKEN` | reclaim.ai settings | Agent containers (tripit-reclaim sync) |
| `GOOGLE_CLIENT_ID` | GCP console | Agent containers (Calendar OOO blocks) |
| `GOOGLE_CLIENT_SECRET` | GCP console | Agent containers (Calendar OOO blocks) |
| `GOOGLE_REFRESH_TOKEN` | OAuth flow | Agent containers (Calendar OOO blocks) |

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
1. Check `.claude/skills/`: `ls data/sessions/telegram_swarm/.claude/skills/`
2. Check RULES.md: `cat data/sessions/telegram_swarm/.claude/RULES.md | head -20`
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
