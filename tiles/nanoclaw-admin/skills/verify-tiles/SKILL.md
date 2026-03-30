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

## Step 3: Semantic comparison and cleanup of promoted staging skills

For skills marked PROMOTED — read both the staging and tile versions in full, then reason about whether the tile faithfully implements the staging version.

For each PROMOTED skill:

1. Read `/workspace/group/skills/<skill>/SKILL.md` and `/home/node/.claude/skills/tessl__<skill>/SKILL.md`.

2. Semantically compare using this checklist — a single **No** = **MISMATCH**:
   - [ ] All major sections present?
   - [ ] All key rules and steps preserved (rewording/reformatting OK)?
   - [ ] No logic altered, removed, or omitted?

3. **MATCH** → delete staging copy and report:
   ```bash
   rm -rf /workspace/group/skills/<skill>
   # "Removed stale staging copy: <skill> (content verified)"
   ```

4. **MISMATCH** → keep staging copy as-is and report which sections or rules differ, so Baruch can investigate and re-promote.

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
• Kept K staging skills due to MISMATCH (list names + discrepancies)
• Removed J staging rule files (list paths)
• Total tile skills: X installed
```

If staging was already empty on all counts — report that cleanly.
