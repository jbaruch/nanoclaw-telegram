---
name: verify-tiles
description: Verifies that tiles are correctly installed after promotion. Compares installed skill files against git/staging state, reports any mismatches, cleans up stale staging copies. Runs in a fresh container after promote-tiles nukes the old one. Use after promotion or when skill versions seem wrong.
---

# Verify Tile Installation

## Step 1: List installed skills in both tiles

```bash
ls /home/node/.claude/.tessl/tiles/jbaruch/nanoclaw-admin/skills/
ls /home/node/.claude/.tessl/tiles/jbaruch/nanoclaw-core/skills/
```

## Step 2: List staging copies

```bash
ls /workspace/group/skills/ 2>/dev/null
ls /workspace/group/staging/ 2>/dev/null
```

## Step 3: Compare staging skills against installed tile versions

For each skill in `/workspace/group/skills/`, check if a `tessl__` version exists in `.claude/skills/`:

```bash
for skill in $(ls /workspace/group/skills/); do
  tile_name="${skill#tessl__}"
  if [ -d "/home/node/.claude/skills/tessl__${tile_name}" ]; then
    echo "PROMOTED: $skill (tile version exists)"
  else
    echo "STAGING ONLY: $skill (no tile version — keep)"
  fi
done
```

## Step 4: Semantic comparison and cleanup

For each skill marked PROMOTED — read both versions and reason about whether the tile faithfully implements the staging version.

1. Read the staging skill: `/workspace/group/skills/<skill>/SKILL.md`
2. Read the tile skill (strip `tessl__` prefix): `/home/node/.claude/skills/tessl__<name>/SKILL.md`
3. Semantically compare — small rewording is fine; missing steps, removed rules, or altered logic is a **MISMATCH**.

If **MATCH**:
- Delete the staging copy: `rm -rf /workspace/group/skills/<skill>`
- Report: "Removed stale staging copy: <skill> (content verified)"

If **MISMATCH**:
- Keep the staging copy — do not delete.
- Report the discrepancy clearly (which sections or rules differ).

**Do NOT remove** skills marked STAGING ONLY — those are works in progress.

## Step 5: Check and clean staging rules directories

```bash
find /workspace/group/staging -type f -name "*.md" 2>/dev/null
```

Remove stale rule files (promotion already happened):

```bash
while IFS= read -r file; do
  rm "$file" && echo "Removed: $file"
done < <(find /workspace/group/staging -type f -name "*.md" 2>/dev/null)

find /workspace/group/staging -type d -empty -delete 2>/dev/null
```

## Step 6: Report

Send report via `mcp__nanoclaw__send_message`:

```
Tile verification:
• Removed N stale staging skill copies (list names)
• Kept M staging-only skills (list names)
• Kept K staging skills due to MISMATCH (list names + discrepancies)
• Removed J staging rule files (list paths)
• Total installed: X admin skills, Y core skills
```

If staging was already empty — report that cleanly.
