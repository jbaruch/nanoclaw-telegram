# Skills Migration & Container Build-Time Skills Research
_Generated 2026-03-27 from multi-agent research session_

---

## Part 1: Migrating CLAUDE.md Rules to Skills

### The Core Problem

Current `/workspace/group/CLAUDE.md` is 365 lines. Claude Code's instruction budget is ~150-200 total, with ~50 already consumed by Claude Code's own system prompt. Every line in CLAUDE.md is expensive. The file is failing to activate rules reliably because it's too long.

**Target:** ~65 lines of universal rules + `@OPERATIONS.md` import. Everything else becomes skills.

---

### Section-by-Section Disposition

| Section | Lines | → | Rationale |
|---|---|---|---|
| Identity / SOUL.md read | 1–5 | **Stay as rule** | Must fire unconditionally |
| Async Tasks (ACK + background) | 7–17 | **Stay as rule** | Timing-critical, always applies |
| Skills — Always Use Real Skill | 19–21 | **Stay as rule** | Meta-rule about skill invocation |
| Бойскаут Rule | 23–25 | **Stay as rule** | Personality/behavior, not a procedure |
| What You Can Do | 27–35 | **DELETE** | Claude already knows this; pure token waste |
| Communication (send_message, internal tags) | 37–57 | **Stay as rule** | Core output protocol, always applies |
| Memory | 59–66 | **Stay as rule** | 8 lines, always-relevant |
| Message Formatting (channel syntax tables) | 68–94 | **New skill: `format-message`** | 27-line lookup table, only needed when composing output |
| Agent Teams | 96–130 | **New skill: `create-agent-team`** | 35-line procedure, invoked on demand |
| Admin Context | 133–135 | **Stay as rule (2 lines)** | Single fact |
| Authentication | 137–139 | **Operational doc** | Reference info, not behavior |
| Container Mounts | 141–154 | **Operational doc** | Static reference table |
| Managing Groups (all subsections) | 157–310 | **New skill: `manage-groups`** | 154 lines only needed when managing groups |
| Global Memory | 314–316 | **Stay as rule (3 lines)** | Always-relevant pointer |
| Scheduling for Other Groups | 320–325 | **New skill: `schedule-task`** | On-demand procedure |
| Task Scripts | 329–365 | **New skill: `schedule-task`** (merged) | Tightly coupled with scheduling |

**Result:** 365 lines → ~65 lines of rules. 5 new skills created.

---

### New Skills to Create

#### 1. `format-message`
```yaml
name: format-message
description: Look up the correct message formatting syntax for the current channel.
  Use before sending any formatted response — covers WhatsApp, Telegram, Slack,
  and Discord syntax rules. Triggers on "how do I format", "what syntax", or
  whenever you need to check bold/italic/link/bullet rules for the target channel.
```
**Contains:** Current lines 68–94 (formatting tables) verbatim.
**Note:** Different from existing `channel-formatting` skill which installs a code formatter — this is Claude's runtime output behavior.

#### 2. `create-agent-team`
```yaml
name: create-agent-team
description: Create a multi-agent team where each member posts to the group as a
  distinct identity. Use when the user asks for a panel, debate, research team,
  or any task involving multiple named agents working in parallel. Covers team
  setup, sender identity, message brevity rules, and lead-agent coordination.
```
**Contains:** Current lines 96–130 including example prompt template and lead-agent rules.

#### 3. `manage-groups`
```yaml
name: manage-groups
description: Add, remove, list, or configure NanoClaw groups and channels. Use when
  the user asks to register a new WhatsApp/Telegram/Slack group, remove a group,
  list registered groups, configure sender allowlists, or add directory mounts
  to a group container.
```
**Contains:** Lines 157–310 (all group management subsections).

#### 4. `schedule-task`
```yaml
name: schedule-task
description: Schedule a task for another group or create a conditional task with a
  pre-check script. Use when scheduling for a non-current group (requires
  target_group_jid), or when the task should only wake the agent if a condition
  is met (new PRs, website changes, API status checks).
```
**Contains:** Lines 320–325 (cross-group scheduling) + 329–365 (task scripts), merged.

#### 5. `heartbeat`
```yaml
name: heartbeat
description: Periodic health check and self-healing. Run system checks (stuck tasks,
  IPC errors, DB size, logs, unanswered messages, calendar changes, email),
  auto-fix what's possible, report only failures. Use when running as a scheduled
  heartbeat task.
```
**Contains:** Full `groups/global/HEARTBEAT.md` (268 lines) — checks, alert rules, self-healing actions, output format.

---

### Existing Skills to Fix

| Skill | Problem | Fix |
|---|---|---|
| `channel-formatting` | Confusable with new `format-message` | Rename to `install-channel-formatting` |
| `customize` | "Add new capabilities" is too vague | Add: "modify router, change trigger word, edit CLAUDE.md, add integration" |
| `debug` | "Things aren't working" is weak signal | Add: "container fails to start, 401 errors, agent not responding, missing tool" |
| `add-parallel` | **No frontmatter description at all** | Add YAML frontmatter with description |

---

### Operational Doc: `OPERATIONS.md`

Create `/workspace/group/OPERATIONS.md` containing:
- Authentication / credential info
- Container Mounts table
- Registered groups schema
- Key path reference

Import in CLAUDE.md via `@OPERATIONS.md` (confirmed syntax — bare `@` + relative path, no quotes).

---

### No Hooks Needed

Checked all current rules against hook suitability. Verdict: **0 current rules are good hook candidates.** Hooks are for deterministic guardrails (block specific tool calls, enforce file patterns). Current rules are behavioral/advisory. `settings.json` stays `{}` for now.

---

### Target CLAUDE.md Skeleton (~65 lines)

```
# AyeAye

[Identity + SOUL.md read — 2 lines]

## Async Tasks
[ACK first, background agent, no text output — 6 lines]

## Skills
[Always use the real skill — 2 lines]

## Бойскаут Rule
[Find it, fix it, report it — 3 lines]

## Communication
[send_message, <internal> tags, sub-agent behavior — 10 lines]

## Memory
[conversations/ folder, file creation, 500-line split — 5 lines]

## Message Formatting
[Check channel prefix, use correct syntax, invoke /format-message if unsure — 2 lines]

## Admin Context
[Main channel, elevated privileges — 2 lines]

## Global Memory
[global/CLAUDE.md pointer, when to update globally — 3 lines]

@OPERATIONS.md
```

---

## Part 2: Baking Skills into the Container at Build Time

### Current State (broken)

- Skills discovered at runtime from `~/.claude/skills/` and `.claude/skills/`
- `settingSources: ['user', 'project']` is already configured in `agent-runner/src/index.ts` ✓
- **But:** NO skills are copied into the image at build time
- `container/skills/` has 4 skills (agent-browser, capabilities, slack-formatting, status) but Dockerfile doesn't COPY them
- 30+ skills in `.claude/skills/` are also not baked in
- Only existing image manipulation: one symlink for `blog-writer-persona` (which is also broken on rebuild)

### Target State

All behavioral skills present at `/home/node/.claude/skills/` when container starts. No runtime mounting needed for skills.

### How It Works

Skills directory structure in the image:
```
/home/node/.claude/
├── skills/
│   ├── format-message/
│   │   └── SKILL.md
│   ├── create-agent-team/
│   │   └── SKILL.md
│   ├── manage-groups/
│   │   └── SKILL.md
│   ├── schedule-task/
│   │   └── SKILL.md
│   ├── agent-browser/
│   │   └── SKILL.md
│   ├── capabilities/
│   │   └── SKILL.md
│   └── ... (all others)
└── blog-writer-persona -> /workspace/extra/blogs/persona  (existing symlink)
```

### Dockerfile Changes Required

Add after line 52 (after workspace dirs, before symlinks section), while still running as root:

```dockerfile
# Copy all behavioral skills into the node user's home
RUN mkdir -p /home/node/.claude/skills
COPY container/skills/ /home/node/.claude/skills/
RUN chown -R node:node /home/node/.claude
```

If skills should come from `.claude/skills/` (the full 30+ skill set from the project tree), change source path:

```dockerfile
COPY .claude/skills/ /home/node/.claude/skills/
RUN chown -R node:node /home/node/.claude
```

Or fetch from git at build time (for skills in a separate repo):

```dockerfile
RUN git clone --depth 1 <repo-url> /tmp/skills && \
    cp -r /tmp/skills/skills/* /home/node/.claude/skills/ && \
    rm -rf /tmp/skills && \
    chown -R node:node /home/node/.claude/skills
```

### Key Decision: Which Skills to Bake In?

Two options:

**Option A: Only `container/skills/` (4 skills)**
- Pro: Minimal, fast build, explicit about what's bundled
- Con: The 30+ behavioral skills in `.claude/skills/` aren't available without mounting

**Option B: All of `.claude/skills/` (30+ skills)**
- Pro: Fully self-contained, no runtime mounting needed
- Con: Larger image, all skills are "locked" to build time (updates require rebuild)

**Recommendation: Option B** for behavioral skills (the ones that define AyeAye's behavior), since they change with configuration not with code. Plus: after the CLAUDE.md → skills migration, the 4 new behavioral skills (format-message, create-agent-team, manage-groups, schedule-task) need to be in the image.

### MCP Servers

Already handled — line 34 in Dockerfile: `RUN npm install -g agent-browser @anthropic-ai/claude-code tessl @composio/mcp`. Nothing extra needed here.

### Fix the blog-writer-persona Symlink

While fixing the Dockerfile, also add the missing symlink (currently missing, breaking blog-writer skill on every rebuild):

```dockerfile
RUN ln -s /workspace/extra/blogs/persona /home/node/.claude/blog-writer-persona
```

---

## Summary: What Needs to Be Done

### Immediate (Desktop Claude tasks)

1. **Create 5 new skills** in `container/skills/`:
   - `format-message/SKILL.md`
   - `create-agent-team/SKILL.md`
   - `manage-groups/SKILL.md`
   - `schedule-task/SKILL.md`
   - `heartbeat/SKILL.md`

2. **Rewrite `/workspace/group/CLAUDE.md`** to ~65 lines using the skeleton above

3. **Create `/workspace/group/OPERATIONS.md`** with auth/mounts/schema reference content

4. **Update Dockerfile** (`container/Dockerfile`):
   - Add `COPY .claude/skills/ /home/node/.claude/skills/`
   - Add `RUN chown -R node:node /home/node/.claude`
   - Add missing `blog-writer-persona` symlink

5. **Rename `channel-formatting` → `install-channel-formatting`** to prevent confusion

6. **Fix `add-parallel`** — add YAML frontmatter with description

7. **Improve descriptions** on `customize` and `debug` skills

### After Implementation

- Rebuild container: `docker build ...`
- Verify skills load: send a message that should trigger each new skill
- Test that CLAUDE.md activation improves (fewer ignored rules, faster responses)
