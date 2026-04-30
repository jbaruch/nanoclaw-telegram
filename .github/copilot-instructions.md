# NanoClaw — Copilot Cloud Agent Instructions

## What This Repository Is

NanoClaw is a personal Claude assistant that runs agents securely in isolated Linux containers. It is a **single Node.js process** (~50 TypeScript source files) that bridges messaging channels (WhatsApp, Telegram, Slack, Discord, Gmail) to Claude agents. Simplicity and security are the core values — the entire codebase is intentionally small enough to be read and understood in one sitting.

---

## Repository Layout (critical files first)

```
src/
  index.ts              # Main orchestrator: message loop, agent invocation, state
  config.ts             # All env-var-driven config (ASSISTANT_NAME, paths, timeouts)
  db.ts                 # SQLite operations (messages, groups, sessions, tasks)
  container-runner.ts   # Spawns agent containers with correct mounts per trust tier
  container-runtime.ts  # Detects Docker vs Apple Container; builds mount args
  ipc.ts                # File-based IPC watcher — handles agent → host commands
  router.ts             # Outbound message formatting and delivery
  group-queue.ts        # Per-group queue with global concurrency limit
  task-scheduler.ts     # Runs scheduled tasks (cron / interval / once)
  channels/
    registry.ts         # Channel self-registration at startup
    telegram.ts         # Telegram channel implementation (grammy)
  types.ts              # All shared TypeScript interfaces
  credential-proxy.ts   # Local HTTP proxy — injects real API keys at request time

container/
  Dockerfile            # Agent container image
  agent-runner/         # Node.js process that runs INSIDE the container
  skills/               # Container-side skills (browser, status, formatting)
  build.sh              # Build the container image

.claude/
  skills/               # Host-side Claude Code skills (SKILL.md files)

groups/                 # Per-group working directories (CLAUDE.md memory, files)
data/                   # SQLite DB, sessions, IPC directories
docs/                   # Architecture docs, security model, spec, requirements
```

---

## Architecture in One Paragraph

Channels deliver inbound messages into SQLite. The orchestrator polls SQLite every 2 s, matches messages against registered group triggers, and spawns a Docker/Apple Container with the correct mounts for that group's trust tier. The agent runs Claude Code (Claude Agent SDK) inside the container, writes its response to an IPC directory, and exits. The IPC watcher picks up the response and routes it back through the originating channel. A separate scheduler loop fires time-based tasks.

```
Channel → SQLite → poll loop → Container (Claude Agent SDK) → IPC → Channel
```

---

## Three-Tier Trust Model

Every group has a trust level that controls container mounts, credentials, and which "tiles" (behavior rules) are loaded:

| Tier | Flag | Container restrictions | Tiles |
|------|------|------------------------|-------|
| **Main** | `isMain: true` | None — full DB, all mounts, all credentials | core + trusted + admin |
| **Trusted** | `containerConfig.trusted: true` | Read-only DB, limited credentials | core + trusted |
| **Untrusted** | (default) | Read-only group folder, filtered DB (own chat only), no credentials, 512 MB/1 CPU/5 min | core + untrusted |

Untrusted containers **never** see raw API keys. All credentials flow through `credential-proxy.ts` (a local HTTP proxy on port 3001 by default).

---

## Skill System (four types — know these before touching `.claude/skills/`)

NanoClaw uses Claude Code skills as its extension mechanism. There are **four distinct types**:

### 1. Feature skills (branch-based)
Add channels/integrations. Code lives on a `skill/*` git branch; `.claude/skills/<name>/SKILL.md` contains instructions. Example: `/add-telegram` merges the `skill/telegram` branch.

### 2. Utility skills (code + SKILL.md)
Standalone tools with supporting files in `.claude/skills/<name>/`. No branch merge needed. Example: `/claw` (CLI).

### 3. Operational skills (instruction-only)
Pure workflow guides — no code changes, no branch merges. Always on `main`. Examples: `/setup`, `/debug`, `/customize`, `/update-nanoclaw`.

### 4. Container skills
Live under `container/skills/` and are loaded by Claude Code *inside* the container at runtime. They are not invoked by the user on the host. Examples: `agent-browser`, `status`, `slack-formatting`.

**SKILL.md rules:** frontmatter with `name` + `description` required; keep under 500 lines; no inline code (use separate files); use `${CLAUDE_SKILL_DIR}` to reference skill-local files.

---

## Development Commands

```bash
npm run dev          # tsx src/index.ts — run orchestrator with hot reload
npm run build        # tsc — compile to dist/
npm run typecheck    # tsc --noEmit — type-check without emitting
npm run lint         # eslint src/
npm run lint:fix     # eslint src/ --fix
npm run test         # vitest run — run all tests
npm run format       # prettier --write
./container/build.sh # Rebuild agent container image
```

Tests live in `src/**/*.test.ts` (also `setup/`, `container/agent-runner/src/`, `scripts/`). Run `npm test` after any source change.

---

## Key Conventions

### TypeScript
- ES module project (`"type": "module"` in package.json); import paths must use `.js` extensions even when the actual source file is `.ts` — e.g., `import { foo } from './bar.js'` where `bar.ts` is the real file. This is required by NodeNext module resolution.
- Strict mode. No `any` (warning). No catch-all catches (custom ESLint rule `no-catch-all/no-catch-all`).
- All types in `src/types.ts`; extend there rather than inventing local interfaces.

### Configuration
All runtime config is in `src/config.ts`. It reads only **non-secret** values from `.env` via `readEnvFile` (assistant name, paths, timeouts, feature flags). API keys and auth tokens are **not** read here — they are loaded exclusively by `credential-proxy.ts` and injected at request time, so they are never exposed inside containers.

### Database
`src/db.ts` wraps `better-sqlite3` (synchronous). All schema migrations are in `src/state-migrations/` and applied via `applyStateMigrations()` using SQLite `PRAGMA user_version`. Add a new numbered migration file; register it in `src/state-migrations/index.ts`.

### IPC
Agents communicate with the host by writing JSON files to their IPC directory (`data/ipc/<group>/`). The host watches these files. Do not bypass IPC for agent→host communication.

### Logging
Use `logger` from `src/logger.ts` (never `console.log` in production code).

### Error Handling
Errors must name the caught variable (ESLint `@typescript-eslint/no-unused-vars`). Avoid bare `catch` clauses.

---

## What Changes Are Accepted

| Accepted in source | Should be a skill instead |
|--------------------|--------------------------|
| Bug fixes | New channels (WhatsApp, Signal, etc.) |
| Security fixes | New integrations |
| Simplifications / code reduction | OS compatibility |
| Clear improvements (90%+ of users benefit) | Enhancements for a subset of users |

**Do not add features to core.** If it's not a fix or a clear simplification, it belongs in a skill branch.

---

## Pull Request Checklist

Before opening a PR, confirm:
1. `npm run build` passes (no TypeScript errors).
2. `npm run lint` passes.
3. `npm run test` passes.
4. For skills: tested on a fresh clone end-to-end.
5. PR is scoped to one thing (one fix, one skill, one simplification).
6. Check for existing PRs/issues: `gh pr list --repo qwibitai/nanoclaw --search "<topic>"`.
7. Fill in the PR template (`.github/PULL_REQUEST_TEMPLATE.md`).

---

## Credential & Secret Policy

- **Never commit secrets** to the repository.
- `.env.example` documents the expected shape; `.env` is gitignored.
- Secrets at runtime flow through OneCLI Agent Vault (`onecli --help`) or the local credential proxy (`credential-proxy.ts`).
- Container environment variables that hold real secrets use `--env-file` (not `-e KEY=value`) to avoid leaking them via `ps -ef`. See `SECRET_CONTAINER_VARS` in `container-runner.ts`.

---

## Common Pitfalls

| Problem | Solution |
|---------|----------|
| Container image stale after source changes | Run `./container/build.sh`; if still stale, prune buildkit cache first (`docker buildx prune`) |
| Import resolves to wrong file | Use `.js` extension on all `src/` imports (NodeNext module resolution) |
| WhatsApp not connecting after upgrade | WhatsApp is a separate skill — run `/add-whatsapp` |
| Tests fail due to SQLite path | Tests run in-process; ensure `DATA_DIR` / `STORE_DIR` are not hardcoded |
| Merge conflicts after `skill/*` branch merge | Skill branches diverge from main intentionally; resolve against `main` HEAD |

---

## Service Management

```bash
# macOS (launchd)
launchctl load ~/Library/LaunchAgents/com.nanoclaw.plist
launchctl unload ~/Library/LaunchAgents/com.nanoclaw.plist
launchctl kickstart -k gui/$(id -u)/com.nanoclaw   # restart

# Linux (systemd)
systemctl --user start nanoclaw
systemctl --user stop nanoclaw
systemctl --user restart nanoclaw
```

---

## Further Reading (in-repo docs)

| Document | When to read it |
|----------|-----------------|
| `docs/REQUIREMENTS.md` | Design decisions and philosophy from the project creator |
| `docs/SPEC.md` | Full architecture, message flow, MCP tools, deployment |
| `docs/SECURITY.md` | Security boundaries per trust tier, memory system, deployment |
| `docs/skills-as-branches.md` | How feature skills work as git branches |
| `CONTRIBUTING.md` | Skill taxonomy, SKILL.md format rules, PR requirements |
| `src/state-migrations/README.md` | Convention for adding DB schema migrations |
