---
name: NanoClaw architecture post-migration
description: Current architecture after NAS migration, tile system, and skill unification (2026-03-28)
type: project
---

## Runtime
- **NAS** (192.168.10.32, Debian 12, Docker 26.1.0, uid 999): orchestrator container + agent containers via Docker socket (DooD)
- **Mac**: dev only — edit code, push to git, run promote-skill.sh
- Deploy: `git push` → `ssh NAS "cd ~/nanoclaw && git pull && docker compose up -d --build"`
- Agent image rebuild needed for Dockerfile/tile changes: `cd container && ./build.sh`

## Key env vars
- `HOST_PROJECT_ROOT=/home/jbaruch/nanoclaw` — DooD path translation
- `HOST_UID=999`, `HOST_GID=10` — NAS user mapping
- `CREDENTIAL_PROXY_HOST=0.0.0.0` — proxy binds all interfaces for DooD

## Credential proxy
- Replaced OneCLI. Built-in HTTP proxy (src/credential-proxy.ts, 125 lines)
- Reads ANTHROPIC_API_KEY from .env, injects into container HTTPS traffic
- Containers use placeholder keys, proxy swaps at request time
- Port 3001, containers reach via host.docker.internal

## Skill delivery (unified)
- All skills come from Docker image: `tessl install` (tiles) + COPY (built-in container skills)
- Entrypoint copies to bind-mounted .claude/skills/ (with -rL to dereference symlinks)
- container-runner.ts only syncs AyeAye-created group skills (staging area)
- Rules still aggregated by container-runner.ts from local tiles/ dir → RULES.md → systemPrompt.append

## Tiles (private, jbaruch workspace)
- nanoclaw-core: 10 skills + rules (heartbeat, check-*, morning-brief, format-message, schedule-task, nightly-housekeeping, check-cfps)
- nanoclaw-admin: 2 skills + rules (manage-groups, create-agent-team)
- reclaim-tripit-sync: 2 skills (sync-tripit, onboard)
- Tessl auth via secret-mounted api-credentials.json at Docker build time
- chmod -R a+rX on staging dir (tessl sets 700 permissions)

## Mounts (4 total)
1. `/workspace/group` — group folder (r/w)
2. `/workspace/global` — SOUL.md, global CLAUDE.md (r/w for main, r/o for others)
3. `/workspace/store` — messages.db (r/o)
4. `/home/node/.claude` — sessions (r/w)
5. `/workspace/ipc` — IPC messages/tasks (r/w)

## External heartbeat
- Liveness-only: is NanoClaw container running?
- Sends to 1:1 Telegram chat (chat ID 1698969), not swarm group
- Cron on NAS host every 15 min (sudo crontab -u jbaruch)

## <internal> tag filtering
- Three output paths all strip server-side: streaming output, IPC send_message, scheduled task output
- If entire message is internal → suppressed

## Reply threading
- Message IDs in <message id="..."> XML tags
- send_message has optional reply_to parameter
- ACK consumes pendingReplyTo, background agent uses explicit reply_to from prompt
- Telegram reply/quote context resolved: DB lookup + fallback to Telegram API text

## Promote script
- `./scripts/promote-skill.sh skill-name` — pulls from NAS, optimizes, lints, commits, pushes, publishes, deploys, deletes staging
