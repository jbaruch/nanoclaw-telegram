---
name: promote-tiles
description: "Promotes staged skills and rules to tiles, then schedules nuke (20 min) and verify (21 min) so verify runs in a fresh container after tessl review+optimize completes. Use when promoting skills or rules via /promote-tiles."
---

# Promote Tiles

## Step 1: Check what's staged

```bash
ls /workspace/group/skills/ 2>/dev/null
find /workspace/group/staging -type f -name "*.md" 2>/dev/null
```

For each staged skill in `/workspace/group/skills/`, determine which tile it belongs to:
- Requires Composio/Google APIs/external credentials, or is main-channel-only → **nanoclaw-admin**
- Needs no external APIs, useful to all containers → **nanoclaw-core**

When in doubt → **nanoclaw-admin**. See the skill-tile-placement rule.

## Step 2: Promote staged content

Call `mcp__nanoclaw__promote_staging` for each tile that has staged content:
- `mcp__nanoclaw__promote_staging(tileName: "nanoclaw-admin")` — if admin skills or rules are staged
- `mcp__nanoclaw__promote_staging(tileName: "nanoclaw-core")` — if core skills or rules are staged

If a specific skill was requested, pass `skillName` as well.

## Step 3: Send promotion result

Send a message via `mcp__nanoclaw__send_message` with:
- What was promoted (skill names, tile names, new tile versions)
- Note that nuke fires in 20 min, verify in 21 min

## Step 4: Schedule nuke in 20 minutes

Compute `now + 20 minutes` as local time (NO Z suffix). Schedule:

```
mcp__nanoclaw__schedule_task(
  prompt: "Nuke this session to restart with fresh tiles: call mcp__nanoclaw__nuke_session()",
  schedule_type: "once",
  schedule_value: "<now+20min, format YYYY-MM-DDTHH:MM:SS, NO Z suffix>"
)
```

The 20-minute delay lets tessl's review+optimize pipeline finish before the container restarts.

## Step 5: Schedule verify-tiles in 21 minutes

Compute `now + 21 minutes` as local time (NO Z suffix). Schedule:

```
mcp__nanoclaw__schedule_task(
  prompt: "Run verify-tiles to confirm tile installation: Skill(skill: 'tessl__verify-tiles')",
  schedule_type: "once",
  schedule_value: "<now+21min, format YYYY-MM-DDTHH:MM:SS, NO Z suffix>"
)
```

The 1-minute gap after the nuke ensures the fresh container is ready before verify runs.
