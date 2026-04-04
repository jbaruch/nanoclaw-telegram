---
name: promote-tiles
description: "Promotes staged skills and rules to tiles, then schedules nuke (20 min) and verify (21 min) so verify runs in a fresh container after tessl review+optimize completes. Use when the user wants to promote, deploy, or push staged skills or rules to tiles, or invokes /promote-tiles."
---

**Every step below is mandatory. Execute them in order. Do not skip, reorder, or abbreviate any step.**

## Step 1: Check what's staged

```bash
ls /workspace/group/skills/ 2>/dev/null
find /workspace/group/staging -type f -name "*.md" 2>/dev/null
```

If nothing is staged, stop and report. Do not proceed.

## Step 2: Determine tile placement

**For EACH staged item**, read the `skill-tile-placement` rule and find the skill in the concrete placement table.

**HARD VALIDATION — do this for every item before promoting:**

1. Look up the skill name in the placement table
2. If found → use the listed tile. No exceptions.
3. If NOT found → apply the decision rules (credentials → admin, /workspace/trusted/ → trusted, pure logic → core, security → untrusted)
4. If unsure → admin

**NEVER promote to core or untrusted without verifying the skill is explicitly listed for that tile in the placement table.** Core and untrusted are security boundaries. Getting this wrong exposes admin skills to untrusted containers.

List each item and its target tile. Confirm the assignments before proceeding.

## Step 3: Promote staged content

Call `mcp__nanoclaw__promote_staging` for each tile that has staged content:
- `mcp__nanoclaw__promote_staging(tileName: "nanoclaw-admin")` — admin items
- `mcp__nanoclaw__promote_staging(tileName: "nanoclaw-trusted")` — trusted items
- `mcp__nanoclaw__promote_staging(tileName: "nanoclaw-core")` — core items (rare!)

Validate the result. If any call fails, stop and report.

## Step 4: Send promotion result

Send via `mcp__nanoclaw__send_message`:
- What was promoted (skill names, target tiles)
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
