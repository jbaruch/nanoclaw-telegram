---
name: verify-tiles
description: Verify that tile skills from the registry are installed correctly and clean up stale staging copies that would override them. Run after Baruch promotes skills, or when skills seem outdated.
---

# Verify Tile Installation

Check that skills from the tessl registry are correctly installed and not being overridden by stale staging copies.

## Step 1: List staging skills

```bash
ls /workspace/group/skills/ 2>/dev/null
```

If empty — all clean, nothing to do.

## Step 2: Compare each staging skill against its tile version

For each skill in staging, check if a `tessl__` version exists in `.claude/skills/`:

```bash
for skill in $(ls /workspace/group/skills/); do
  if [ -d "/home/node/.claude/skills/tessl__${skill}" ]; then
    echo "PROMOTED: $skill (tile version exists, staging overrides it)"
  else
    echo "STAGING ONLY: $skill (no tile version — keep)"
  fi
done
```

## Step 3: Remove promoted staging copies

For skills marked PROMOTED — the staging copy is stale and overriding the optimized tile version. Delete them:

For each PROMOTED skill, verify the tile version has the expected content (check it has YAML frontmatter and reasonable length), then remove the staging copy.

**Do NOT remove** skills marked STAGING ONLY — those are works in progress that haven't been promoted yet.

## Step 4: Report

Report what was cleaned and what was kept. Format:

```
Tile verification:
• Removed N stale staging copies (list names)
• Kept M staging-only skills (list names)
• Total tile skills: X installed
```
