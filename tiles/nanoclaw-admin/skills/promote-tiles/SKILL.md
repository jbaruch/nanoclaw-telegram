---
name: promote-tiles
description: "Promotes staged skills and rules to tiles, then pushes to tile GitHub repos where GHA handles review, lint, and publish. Schedules nuke (20 min) and verify (21 min). Use when the user wants to promote, deploy, or push staged skills or rules."
---

**Every step below is mandatory. Execute them in order. Do not skip, reorder, or abbreviate any step.**

## Step 1: Check what's staged

```bash
ls /workspace/group/skills/ 2>/dev/null
find /workspace/group/staging -type f -name "*.md" 2>/dev/null
```

If nothing is staged, stop and report. Do not proceed.

## Step 2: Determine tile placement

**For EACH staged item**, apply the `skill-tile-placement` rule. This rule is defined in your loaded RULES.md — read it there directly (do NOT call Skill() for it).

For each item, state:
- The skill/rule name
- Which tile it goes to
- Why (one of: "external API", "host script", "trusted memory", "security rule", "pure logic/universal")

**VALIDATION:** Before assigning core or untrusted, consult the red flags list in the `skill-tile-placement` rule in RULES.md. If ANY red flag matches → admin, not core.

## Step 3: Promote staged content

Call `mcp__nanoclaw__promote_staging` for each tile that has staged content:
- `mcp__nanoclaw__promote_staging(tileName: "nanoclaw-admin")` — most skills go here
- `mcp__nanoclaw__promote_staging(tileName: "nanoclaw-trusted")` — shared operational only
- `mcp__nanoclaw__promote_staging(tileName: "nanoclaw-core")` — rare! Pure logic only

The promote script commits to the nanoclaw repo AND pushes to the tile's GitHub repo (jbaruch/nanoclaw-{tile}). GitHub Actions handles skill review (85% threshold), lint, and tessl publish automatically.

Validate the result. If any call fails, stop and report.

## Step 4: Send promotion result

Send via `mcp__nanoclaw__send_message`:
- What was promoted (skill names, target tiles, reasoning)
- Note that GHA is running review + publish
- Note that nuke fires in 20 min, verify in 21 min

## Step 5: Schedule nuke in 20 minutes

```
mcp__nanoclaw__schedule_task(
  prompt: "Nuke this session to restart with fresh plugins: call mcp__nanoclaw__nuke_session()",
  schedule_type: "once",
  schedule_value: "<now+20min, format YYYY-MM-DDTHH:MM:SS, NO Z suffix>"
)
```

## Step 6: Schedule verify-tiles in 21 minutes

```
mcp__nanoclaw__schedule_task(
  prompt: "Run verify-tiles to confirm tile installation: Skill(skill: 'tessl__verify-tiles')",
  schedule_type: "once",
  schedule_value: "<now+21min, format YYYY-MM-DDTHH:MM:SS, NO Z suffix>"
)
```
