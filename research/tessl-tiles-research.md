# Tessl Tiles Integration Research
_Generated 2026-03-27 — complements skills-migration-research.md_

---

## How Tessl Actually Works (Verified)

### Installation Flow

`tessl install workspace/tile` does three things:

1. **Vendoring** — Downloads tile contents to `.tessl/tiles/workspace/tile-name/`
2. **Skill wiring** — Copies skills into `.claude/skills/tessl__skillname/` (with supporting files like scripts)
3. **Rule wiring** — Creates `.tessl/RULES.md` with `@` imports pointing to each tile's rule files, then wires `AGENTS.md → @.tessl/RULES.md` and `CLAUDE.md → @AGENTS.md`

### File Structure After Install

```
project/
├── CLAUDE.md                          # Appends: @AGENTS.md
├── AGENTS.md                          # tessl-managed: @.tessl/RULES.md
├── tessl.json                         # Manifest: dependencies + versions
├── .tessl/
│   ├── RULES.md                       # Aggregates all rule @-imports
│   └── tiles/
│       └── workspace/tile-name/
│           ├── tile.json              # Metadata
│           ├── rules/rule-name.md     # Rule content (markdown)
│           ├── skills/skill-name/
│           │   └── SKILL.md           # Standard Claude Code skill
│           └── README.md
├── .claude/
│   └── skills/
│       └── tessl__skillname/          # Copied from tile, Claude Code discovers these
│           ├── SKILL.md
│           └── scripts/               # Supporting files come along
└── .mcp.json                          # Adds tessl MCP server (search, install, query_library_docs)
```

### Key Insight: Rules vs Skills

- **Rules** = always-on context injected via CLAUDE.md → AGENTS.md → .tessl/RULES.md chain. Loaded every conversation.
- **Skills** = on-demand, triggered by description match. Only loaded when invoked.
- **Docs** = queryable via `mcp__tessl__query_library_docs` MCP tool. Not loaded into context at all unless queried.

### tile.json Format

```json
{
  "name": "workspace/tile-name",
  "version": "0.1.0",
  "summary": "What this tile does",
  "private": true,
  "rules": {
    "rule-id": { "rules": "rules/rule-name.md" }
  },
  "skills": {
    "skill-id": { "path": "skills/skill-name/SKILL.md" }
  }
}
```

---

## The Problem: NanoClaw Container Context

NanoClaw agents run inside containers. The standard `tessl install` flow assumes a host-side project directory. Inside a container:

1. `/home/node/.claude/` is bind-mounted from host session dir (per-group)
2. `/workspace/group/` is the working directory (bind-mounted, per-group)
3. `/workspace/project/` is the NanoClaw project root (read-only, main only)
4. No `tessl.json` exists anywhere currently
5. Tessl CLI IS installed globally in the image (`npm install -g tessl`)
6. Tessl credentials ARE mounted at `/home/node/.tessl` (read-only from host `~/.tessl/`)

### What Works Today

- `tessl mcp start` — runs as MCP server, gives agents `search`, `install`, `query_library_docs` tools
- The tessl MCP server is already wired in agent-runner when `~/.tessl/api-credentials.json` exists
- Agents CAN use `mcp__tessl__install` at runtime to install tiles into their workspace

### What Doesn't Work

- No tiles are pre-installed — every new container session starts clean
- Rules from tiles need the CLAUDE.md → AGENTS.md → .tessl/RULES.md chain, which doesn't exist in the container's workspace
- Runtime `tessl install` adds latency to every session and wastes tokens on "let me install my dependencies"

---

## Architecture Options

### Option A: Build-Time `tessl install` in Dockerfile

```dockerfile
# After npm install -g tessl and WORKDIR /workspace/group
RUN tessl init && tessl install jbaruch/nanoclaw-core --yes
```

**Problem:** `tessl install` needs authentication (`~/.tessl/api-credentials.json`). Build-time doesn't have access to host credentials. Would need `--mount=type=secret` or a build arg with a token.

**Problem 2:** `/home/node/.claude/` gets bind-mounted at runtime, masking anything created at build time (same symlink problem we just fixed).

**Verdict:** Doesn't work cleanly for skills (bind mount masks them). Could work for rules if they go in `/workspace/group/`.

### Option B: Host-Side Pre-Install, Mount Into Container

Run `tessl install` on the host into a shared directory. Mount that directory read-only into containers.

```
Host: data/tessl-tiles/                    # tessl install target
Container: /workspace/tessl/ (read-only)   # Mounted at runtime
```

Then in the group's CLAUDE.md or settings, reference the tiles:
```
@/workspace/tessl/.tessl/RULES.md
```

And sync skills from `.claude/skills/` into each group's session dir (like we already do with `container/skills/`).

**Verdict:** Works, but adds another mount and another sync step. Duplicates the existing skill-sync pattern.

### Option C: Tessl Tiles as Source of Truth, Synced by container-runner.ts

This is the cleanest path. The idea:

1. **Create tiles** in a local directory (e.g., `tiles/` in the NanoClaw project)
2. **Run `tessl install`** on the host to install from local path or registry
3. **container-runner.ts** already syncs `container/skills/` into `.claude/skills/` per group — extend this to also sync tile skills
4. **Rules** get written into each group's CLAUDE.md or into the session dir

**Flow:**
```
tiles/nanoclaw-core/          # Local tile source
  ├── tile.json
  ├── rules/
  │   ├── communication.md    # Always-on rules (async ACK, send_message, etc.)
  │   ├── memory.md           # Memory management rules
  │   └── formatting.md       # Channel-aware formatting rules
  └── skills/
      ├── format-message/SKILL.md
      ├── create-agent-team/SKILL.md
      ├── manage-groups/SKILL.md
      └── schedule-task/SKILL.md

container-runner.ts:
  1. Reads tile.json from tiles/nanoclaw-core/
  2. Copies skills/ into group's .claude/skills/
  3. Copies rules/ content and wires into group's .claude/ context
```

**Verdict:** Best option. Uses Tessl's tile structure as the organizational format, but the actual delivery to containers stays in NanoClaw's existing sync mechanism. Can ALSO publish to registry for reuse.

### Option D: Runtime MCP-Based Install

Let the agent install tiles on first run via `mcp__tessl__install`.

**Verdict:** Too slow, wastes tokens, non-deterministic. Only useful for on-demand library docs, not core rules/skills.

---

## Recommended Architecture: Option C + Registry Publishing

### Tile Layout

Create multiple tiles for modularity:

```
tiles/
├── nanoclaw-core/                    # Always installed in every container
│   ├── tile.json
│   ├── rules/
│   │   ├── async-tasks.md            # ACK first, background agent
│   │   ├── communication.md          # send_message protocol, <internal> tags
│   │   └── memory.md                 # conversations/ folder, file structure
│   └── skills/
│       ├── format-message/SKILL.md   # Channel formatting lookup
│       ├── schedule-task/SKILL.md    # Task scheduling with pre-flight scripts
│       └── heartbeat/SKILL.md        # Periodic health check and self-healing
│
├── nanoclaw-admin/                   # Only installed for main channel
│   ├── tile.json
│   ├── rules/
│   │   └── admin-context.md          # Elevated privileges, credential management
│   └── skills/
│       ├── manage-groups/SKILL.md    # Group registration, config, allowlists
│       └── create-agent-team/SKILL.md # Multi-agent team coordination
│
└── nanoclaw-browser/                 # Existing agent-browser, repackaged
    ├── tile.json
    └── skills/
        └── agent-browser/SKILL.md
```

### container-runner.ts Changes

```typescript
// After existing skill sync block:
// Sync tile skills into group's .claude/skills/
const tilesDirs = [
  path.join(projectRoot, 'tiles', 'nanoclaw-core'),
  ...(isMain ? [path.join(projectRoot, 'tiles', 'nanoclaw-admin')] : []),
  path.join(projectRoot, 'tiles', 'nanoclaw-browser'),
];

for (const tileDir of tilesDirs) {
  const tileSkillsDir = path.join(tileDir, 'skills');
  if (!fs.existsSync(tileSkillsDir)) continue;
  for (const skillDir of fs.readdirSync(tileSkillsDir)) {
    const src = path.join(tileSkillsDir, skillDir);
    if (!fs.statSync(src).isDirectory()) continue;
    const dst = path.join(skillsDst, skillDir);
    fs.cpSync(src, dst, { recursive: true });
  }
}

// Aggregate tile rules into a RULES.md in the session dir
const rulesContent: string[] = ['# Agent Rules\n'];
for (const tileDir of tilesDirs) {
  const tileRulesDir = path.join(tileDir, 'rules');
  if (!fs.existsSync(tileRulesDir)) continue;
  for (const ruleFile of fs.readdirSync(tileRulesDir)) {
    if (!ruleFile.endsWith('.md')) continue;
    rulesContent.push(fs.readFileSync(path.join(tileRulesDir, ruleFile), 'utf8'));
    rulesContent.push('\n---\n');
  }
}
if (rulesContent.length > 1) {
  fs.writeFileSync(path.join(groupSessionsDir, 'RULES.md'), rulesContent.join('\n'));
  // Wire into group CLAUDE.md via @-import would need separate handling
}
```

### Rules Injection

Rules need to be in CLAUDE.md context. Two approaches:

**Approach 1: Generate AGENTS.md in group dir**
Write `@/home/node/.claude/RULES.md` into the group's CLAUDE.md. Since `settingSources: ['project', 'user']` is configured, rules from `~/.claude/` should be discoverable.

**Approach 2: Inline rules into group CLAUDE.md**
container-runner.ts already generates `settings.json` per group. It could also append rule content to the group's CLAUDE.md. But this modifies the user's file — not ideal.

**Approach 3: Use systemPrompt.append**
The agent-runner already reads `globalClaudeMd` and passes it as `systemPrompt.append`. Could aggregate tile rules into this same path.

Best bet: **Approach 3** — aggregate tile rules and pass via `systemPrompt.append` in the agent-runner. Zero file mutations needed.

---

## Container Image Build Considerations

### What Needs `tessl` at Build Time

Nothing, if we use Option C. The tile files live in the NanoClaw repo under `tiles/` and get synced by container-runner.ts at runtime (same pattern as `container/skills/`).

### What Needs `tessl` at Runtime

- **MCP server** (`tessl mcp start`) — already configured in agent-runner for library doc queries
- **On-demand install** (`mcp__tessl__install`) — agents can install additional tiles if needed (e.g., library docs for a specific npm package)

### Current Dockerfile Already Has

```dockerfile
RUN npm install -g agent-browser @anthropic-ai/claude-code tessl @composio/mcp
```

This is sufficient. No `tessl install` needed at build time.

---

## Migration Plan

### Phase 1: Create Tile Structure (No Behavior Change)

1. Create `tiles/nanoclaw-core/` with tile.json
2. Extract rules from `groups/main/CLAUDE.md` and `groups/global/CLAUDE.md` into `tiles/nanoclaw-core/rules/`
3. Move container skills that are really behavioral into tile skills
4. Create `tiles/nanoclaw-admin/` with main-channel-only rules and skills
5. Validate with `tessl tile lint`

### Phase 2: Wire Into Container Runner

1. Extend skill sync in container-runner.ts to include tile skills
2. Aggregate tile rules and inject via systemPrompt.append in agent-runner
3. Slim down `groups/main/CLAUDE.md` and `groups/global/CLAUDE.md` to minimal stubs
4. Test that rules activate correctly

### Phase 3: Publish to Registry (Optional)

1. `tessl tile publish` for nanoclaw-core (public or private)
2. Other NanoClaw users (or other projects) can `tessl install jbaruch/nanoclaw-core`
3. Version management via `tessl tile publish --bump`

### Phase 4: Tessl-Managed Dependencies

1. Run `tessl init` in the NanoClaw project
2. Add `tessl install --project-dependencies` to discover useful library doc tiles
3. The agent gets access to docs for installed npm packages via MCP

---

## Open Questions

1. **Rule loading in container**: Does `systemPrompt.append` have a length limit? The whole point of this migration is reducing context — need to verify rules passed this way count differently than CLAUDE.md content.

2. **Tile versioning in monorepo**: If tiles live in the NanoClaw repo under `tiles/`, they share the repo's version. Publishing to registry gives them independent versions. Worth the overhead?

3. **Per-group tile selection**: Currently all groups get the same skills. Tiles make it natural to give different groups different tile sets (main gets admin tile, others don't). container-runner.ts already has the `isMain` flag — extend to per-group tile config?

4. **Tessl MCP auth in container**: The read-only mount of `~/.tessl/` gives the container agent search and install capabilities. But installed tiles go to `/workspace/group/.tessl/` — which is writable but ephemeral per session. Need to decide if runtime-installed tiles should persist across sessions.

5. **Interaction with `@OPERATIONS.md`**: The existing research proposes extracting operational docs into OPERATIONS.md with an `@` import. Tessl's rule chain (CLAUDE.md → AGENTS.md → RULES.md) is a parallel `@` chain. Need to make sure these don't conflict or create circular imports.
