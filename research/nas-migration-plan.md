# NAS Migration Plan
_2026-03-27_

## Context

NanoClaw is a personal assistant, not a remote coding interface. Moving it from the Mac coding machine to a NAS (192.168.10.32, Docker, SSH) for better uptime and simpler operation. No project source access, no Google Drive mounts needed.

## Architecture

Orchestrator + agent containers all run on NAS via Docker Compose. Mac is dev-only. Deploy model: `git push` → build on NAS → `docker compose up -d`.

## What Goes Into the Agent Image (Static, Changes on Deploy)

- `groups/global/CLAUDE.md` — shared persona/rules
- `groups/global/SOUL.md` — identity
- `groups/telegram_swarm/CLAUDE.md` — main channel rules (template, not runtime state)
- `container/skills/` — all container skills
- `.claude/settings.json` — model config, env vars
- `~/.tessl/` — tessl credentials
- `container/agent-runner/` — agent-runner source (no more per-group copy/cache)
- Blog-writer-persona and any other static config

## What Stays as Volumes (Runtime State, Persists Across Deploys)

| Volume | Container Path | Purpose |
|--------|---------------|---------|
| `group-data` | `/workspace/group` | Conversations, memory files, heartbeat state, calendar state |
| `ipc` | `/workspace/ipc` | Orchestrator ↔ agent message bridge |
| `sessions` | `/home/node/.claude` (subset) | Session resumption across container runs |

Three volumes. Everything else is in the image.

## What Gets Dropped (vs Current Mac Setup)

| Current Mount | Why Dropped |
|---------------|-------------|
| `/workspace/project` (NanoClaw source) | Not coding on NanoClaw from inside the agent |
| `.env` shadow mount | Goes away with project mount |
| `/workspace/extra/host-claude` (host ~/.claude) | Persona is in `groups/global/SOUL.md` already |
| `/workspace/extra/blogs,presentations,travel` (Google Drive) | Not doing content creation on NAS |
| `~/.tessl` host mount | Baked into image instead |

## Problems That Go Away

- Blog-writer-persona symlink getting masked by bind mount → baked into image
- Stale agent-runner-src cache → baked into image, no per-group copy
- Skills not syncing → baked into image
- `process.cwd()` path resolution for mounts → simpler, fewer mounts to resolve

## Key Technical Changes

### 1. HOST_PROJECT_ROOT

When orchestrator runs inside a container, `process.cwd()` is `/app` but agent container `-v` paths must reference NAS host filesystem. Add `HOST_PROJECT_ROOT` env var for mount path translation.

Affects: `container-runner.ts` `buildVolumeMounts()`, `config.ts`

### 2. Credential Proxy

OneCLI won't be on the NAS. Merge `skill/native-credential-proxy` branch — reads API key from `.env`, injects into container HTTPS traffic. No external service needed.

### 3. Dockerfile Changes

- Remove macOS Google Drive symlink block (lines 56-60)
- Bake skills into `/home/node/.claude/skills/`
- Bake SOUL.md and global CLAUDE.md into `/workspace/global/`
- Bake tessl credentials into `/home/node/.tessl/`
- Bake settings.json into `/home/node/.claude/`

### 4. New Dockerfile.orchestrator

NanoClaw Node.js process running in a container with Docker CLI for spawning agent siblings:

```dockerfile
FROM node:22-slim
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY dist/ ./dist/
COPY container/ ./container/
COPY groups/ ./groups/
COPY data/env/ ./data/env/
ENTRYPOINT ["node", "dist/index.js"]
```

### 5. container-runner.ts Simplification

`buildVolumeMounts()` drops from ~185 lines to roughly:

```
mounts = [
  { hostPath: HOST_GROUP_DIR,    containerPath: '/workspace/group',  readonly: false },
  { hostPath: HOST_IPC_DIR,      containerPath: '/workspace/ipc',    readonly: false },
  { hostPath: HOST_SESSIONS_DIR, containerPath: '/home/node/.claude', readonly: false },
]
// No project mount, no Google Drive, no host-claude, no tessl mount,
// no agent-runner-src copy, no skill sync, no persona symlink
```

Global CLAUDE.md, SOUL.md, skills, settings, tessl — all in image already.

### 6. Docker Compose

```yaml
services:
  nanoclaw:
    build:
      context: .
      dockerfile: Dockerfile.orchestrator
    restart: unless-stopped
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - nanoclaw-data:/app/data
      - nanoclaw-store:/app/store
      - nanoclaw-groups:/app/groups
    environment:
      - HOST_PROJECT_ROOT=/opt/nanoclaw
      - TZ=America/Chicago
    env_file: .env

volumes:
  nanoclaw-data:
  nanoclaw-store:
  nanoclaw-groups:
```

## Migration Steps

### Phase 1: Code Changes (on Mac)

1. Merge `skill/native-credential-proxy` branch
2. Add `HOST_PROJECT_ROOT` support to `config.ts` and `container-runner.ts`
3. Simplify `buildVolumeMounts()` — drop project, Google Drive, host-claude, tessl mounts
4. Bake skills, settings, persona into agent Dockerfile
5. Remove macOS-specific symlink block from agent Dockerfile
6. Create `Dockerfile.orchestrator`
7. Create `docker-compose.yml`
8. Test locally with `HOST_PROJECT_ROOT=$(pwd)`

### Phase 2: NAS Setup

9. SSH into NAS, create `/opt/nanoclaw/`
10. Clone repo (or scp), build both images
11. Copy runtime state from Mac:
    ```bash
    rsync -avz data/ 192.168.10.32:/opt/nanoclaw/data/
    rsync -avz store/ 192.168.10.32:/opt/nanoclaw/store/
    rsync -avz groups/ 192.168.10.32:/opt/nanoclaw/groups/
    scp .env 192.168.10.32:/opt/nanoclaw/.env
    ```

### Phase 3: Cutover

12. Stop NanoClaw on Mac: `launchctl bootout gui/$(id -u)/com.nanoclaw`
13. Final rsync of runtime state
14. `docker compose up -d` on NAS
15. Send test message in Telegram
16. Verify session resumption, heartbeat, scheduled tasks

## Dev Workflow After Migration

```
# On Mac (dev)
vim groups/global/SOUL.md   # edit persona
git commit && git push

# On NAS (deploy)
git pull && docker compose build && docker compose up -d
```

Or automate: NAS watches repo, rebuilds on push.

## Open Questions

1. **Session persistence granularity** — do we mount all of `/home/node/.claude` or just the session subdirectory? Full mount masks baked-in skills/settings. Could use entrypoint to copy defaults then mount over.
2. **NAS Docker socket permissions** — orchestrator container needs access to `/var/run/docker.sock`. May need `--group-add` or socket permission changes.
3. **UID mapping** — Mac UID 501 vs NAS UID. Data directories need `chown` on initial copy.
4. **Webhook vs polling for auto-deploy** — GitHub webhook to NAS, or cron `git pull`?
5. **Tessl credential refresh** — if baked into image, tokens expire. May still need a mount or refresh mechanism.
