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
  tile_name="${skill#tessl__}"
  if [ -d "/home/node/.claude/skills/tessl__${tile_name}" ]; then
    echo "PROMOTED: $skill (tile version exists, staging overrides it)"
  else
    echo "STAGING ONLY: $skill (no tile version — keep)"
  fi
done
```

## Step 3: Semantic comparison and cleanup of promoted staging skills

For skills marked PROMOTED — read both the staging and tile versions in full, then reason about whether the tile faithfully implements the staging version.

For each PROMOTED skill:

1. Read the full text of the staging skill:
   `/workspace/group/skills/<skill>/SKILL.md`

2. Read the full text of the tile skill — strip the `tessl__` prefix from the staging name first:
   `/home/node/.claude/skills/tessl__<skill-without-tessl-prefix>/SKILL.md`

3. Semantically compare them — reason about whether the tile faithfully preserves all major sections, key rules, and logic. Small reformatting or rewording for clarity is fine; missing steps, removed rules, or altered logic is a **MISMATCH**.

4. If the tile faithfully matches the staging intent (**MATCH**):
   - Delete the staging copy: `rm -rf /workspace/group/skills/<skill>`
   - Report: "Removed stale staging copy: <skill> (content verified)"

5. If the tile does NOT faithfully match the staging intent (**MISMATCH**):
   - Keep the staging copy as-is — do not delete it.
   - Report the discrepancy clearly: which sections or rules are missing or altered in the tile version, so Baruch can investigate and re-promote if needed.

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
