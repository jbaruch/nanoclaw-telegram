# No In-Place Skill Overrides

## Rule

NEVER write skill or rule files directly to `/workspace/group/skills/` or `/workspace/group/.tessl/`. ALL changes to skills and rules MUST go through the promotion pipeline:

1. Stage the change (write to `/workspace/group/staging/{tileName}/rules/{name}.md` for rules, or `/workspace/group/staging/{tileName}/skills/{name}/SKILL.md` for skills)
2. Call `promote_staging` MCP tool with the correct tile name
3. Wait for verify-tiles to confirm installation

## Why

Files written directly to the group folder persist across sessions and override tile-delivered versions. This causes:
- Stale content that never gets updated when tiles are republished
- Version drift between what's in git and what the agent sees
- Confusion during troubleshooting ("the tile is correct but the agent uses old content")

The promotion pipeline commits to git, publishes to the registry, and verify-tiles cleans up staging — ensuring a single source of truth.

## What counts as an override

- Writing SKILL.md to `/workspace/group/skills/tessl__*/`
- Writing rule .md files to `/workspace/group/.tessl/`
- Editing any file under `/workspace/group/skills/` that was delivered by a tile

## What's allowed

- Writing to `/workspace/group/staging/{tileName}/` — that's the staging area
- Creating NEW scripts in `/workspace/group/scripts/` — those are host-side execution helpers
- Editing group-specific config files (cfp-state.json, task-tz-state.json, etc.)
