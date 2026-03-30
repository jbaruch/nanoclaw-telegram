---
name: verify-tiles
description: Checks that tile skills from the tessl registry are correctly installed by comparing local skill files against registry originals, and removes stale staging copies that would override them. Use when skills seem outdated, a skill is not updating, you're seeing the wrong version of a skill, there are skill override issues, or after Baruch promotes skills or rules.
---

# Verify Tile Installation

Check that skills and rules from the tessl registry are correctly installed and not being overridden by stale staging copies.

## Step 1: List staging skills

```bash
ls /workspace/group/skills/ 2>/dev/null
```

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

## Step 3: Remove promoted staging skill copies

For skills marked PROMOTED — the staging copy is stale and overriding the optimized tile version. Verify the tile version has expected content, then delete the staging copy:

```bash
for skill in <PROMOTED_SKILLS>; do
  tile="/home/node/.claude/skills/tessl__${skill}"
  # Verify tile has YAML frontmatter and reasonable length (>10 lines)
  if head -1 "$tile/skill.md" 2>/dev/null | grep -q '^---' && [ "$(wc -l < "$tile/skill.md")" -gt 10 ]; then
    rm -rf "/workspace/group/skills/${skill}"
    echo "Removed stale staging copy: $skill"
  else
    echo "SKIPPED: $skill — tile version failed validation"
  fi
done
```

**Do NOT remove** skills marked STAGING ONLY — those are works in progress that haven't been promoted yet.

## Step 4: Check staging rules directories

Rules live in `/workspace/group/staging/` under tile subdirectories. When verify-tiles runs, promotion is assumed to have already happened — staging rule files are stale and should be removed.

```bash
find /workspace/group/staging -type f -name "*.md" 2>/dev/null
```

Remove each found file, then clean up any empty directories:

```bash
while IFS= read -r file; do
  rm "$file" && echo "Removed: $file"
done < <(find /workspace/group/staging -type f -name "*.md" 2>/dev/null)

find /workspace/group/staging -type d -empty -delete 2>/dev/null
```

## Step 5: Report

Report everything that was cleaned. Format:

```
Tile verification:
• Removed N stale staging skill copies (list names)
• Kept M staging-only skills (list names)
• Removed K staging rule files (list paths)
• Total tile skills: X installed
```

If staging was already empty on all counts — report that cleanly.
