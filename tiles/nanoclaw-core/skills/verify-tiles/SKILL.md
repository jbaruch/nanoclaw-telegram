---
name: verify-tiles
description: Promotes staged skills and rules to tiles, then verifies installation. Use after staging new skills or rules, when skills seem outdated, or to run the full promote+verify cycle.
---

# Promote & Verify Tiles

Runs the host-side promote script via MCP, then verifies tile installation locally.

## Step 1: Determine what's staged

```bash
ls /workspace/group/skills/ 2>/dev/null
find /workspace/group/staging -type f -name "*.md" 2>/dev/null
```

If both are empty — skip to Step 4 (verify only).

## Step 2: Call mcp__nanoclaw__promote_staging

For each tile that has staged content, call `mcp__nanoclaw__promote_staging`:

- Skills in `/workspace/group/skills/tessl__*` → determine tile from the skill name (most go to `nanoclaw-core`; admin-only skills go to `nanoclaw-admin`)
- Rules in `/workspace/group/staging/nanoclaw-core/` → `promote_staging(tileName: "nanoclaw-core", skillName: "--rules-only")`
- Rules in `/workspace/group/staging/nanoclaw-admin/` → `promote_staging(tileName: "nanoclaw-admin", skillName: "--rules-only")`
- To promote all staging for a tile at once: `promote_staging(tileName: "nanoclaw-core")` or `promote_staging(tileName: "nanoclaw-admin")`

If `promote_staging` returns an error — report it to Baruch and stop.

## Step 3: Compare each staging skill against its tile version

For each skill in `/workspace/group/skills/`, check if the tile version exists:

```bash
for skill in $(ls /workspace/group/skills/ 2>/dev/null); do
  tile_name="${skill#tessl__}"
  if [ -d "/home/node/.claude/skills/tessl__${tile_name}" ]; then
    echo "PROMOTED: $skill"
  else
    echo "STAGING ONLY: $skill"
  fi
done
```

For each PROMOTED skill, semantically compare staging vs tile (diff or read both). If identical → remove staging copy. If mismatch → keep staging, report discrepancy.

```bash
rm -rf /workspace/group/skills/<skill>
```

## Step 4: Clean up promoted staging rules

```bash
find /workspace/group/staging -type f -name "*.md" 2>/dev/null
```

Remove each file, then clean up empty dirs:

```bash
while IFS= read -r file; do
  rm "$file" && echo "Removed: $file"
done < <(find /workspace/group/staging -type f -name "*.md" 2>/dev/null)
find /workspace/group/staging -type d -empty -delete 2>/dev/null
```

## Step 5: Report

```
Promote & verify:
• Promoted: N skills, M rules
• Removed N stale staging skill copies (list names)
• Kept M staging-only skills (list names)
• Kept K staging skills due to MISMATCH (list names + discrepancies)
• Removed J staging rule files (list paths)
• Total tile skills: X installed
```

If staging was already empty — report that cleanly.
