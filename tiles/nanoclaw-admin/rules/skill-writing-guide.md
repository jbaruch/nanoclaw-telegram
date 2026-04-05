# Skill Writing Guide

Follow these rules when creating or modifying skills and rules.

## SKILL.md structure

```yaml
---
name: skill-name          # lowercase, hyphens only
description: What it does and when to use it. Front-load the key use case — truncated at 250 chars.
---

**Every step below is mandatory. Execute them in order. Do not skip, reorder, or abbreviate any step.**

## Step 1: First thing
...

## Step 2: Second thing
...
```

## Frontmatter fields

| Field | Use |
|-------|-----|
| `name` | Lowercase, hyphens. Max 64 chars. |
| `description` | When to trigger. Front-load the key use case. Under 250 chars. |
| `allowed-tools` | Space-separated. Only if the skill needs specific tool permissions. |
| `disable-model-invocation` | `true` if user-only (deploy, promote). |
| `user-invocable` | `false` if background knowledge only. |

## Step numbering

Flat sequential: Step 1, Step 2, Step 3. No decimals (0.5), no sub-steps (3a, 4b), no Step 0.

## Mandatory execution

Every multi-step skill MUST start with: `**Every step below is mandatory. Execute them in order. Do not skip, reorder, or abbreviate any step.**`

## Invoking other skills

**Use typed Skill() calls, NOT prose.**

WRONG:
```
Invoke the check-cfps skill to refresh data.
```

RIGHT:
```
`Skill(skill: "tessl__check-cfps")` — refresh CFP data.
```

The prose version is ambiguous — the agent may read it as context rather than an instruction to call the tool.

## Referencing rules vs skills

Rules are loaded into RULES.md automatically. Do NOT call `Skill()` for a rule — it doesn't exist as a skill and the call fails silently.

WRONG:
```
Consult the placement rule: Skill(skill: 'tessl__skill-tile-placement')
```

RIGHT:
```
Apply the `skill-tile-placement` rule (already in your RULES.md).
```

## Running scripts

Use the full path with a typed code block:

```
`python3 /home/node/.claude/skills/tessl__check-unanswered/scripts/check-unanswered.py`
```

Not: "run the unanswered checker script"

## Silence rules

If a step can produce no output, say so explicitly: "If empty → move to Step N." or "If nothing found → silence."

Do NOT use:
- "No new messages to respond to at this time."
- "All clear — no issues found."
- Any acknowledgement of nothing happening

## Description best practices

Front-load the trigger phrases. Claude matches descriptions to user intent.

WRONG:
```
description: A comprehensive tool for managing and monitoring the system health of NanoClaw containers.
```

RIGHT:
```
description: Check container health — CPU, disk, memory, stuck tasks, DB size. Use on "system status", "health check", "is everything ok".
```

## File references

Reference supporting files from SKILL.md so Claude knows they exist:

```
For configuration format, see [config-schema.md](config-schema.md)
```

Keep SKILL.md under 500 lines. Move detailed reference to separate files.

## Tile placement

Before promoting, apply the `skill-tile-placement` rule (in RULES.md). If ANY of these appear in the skill → it's admin, not core:
- Composio, Gmail, Calendar, Tasks, GitHub, Sessionize
- Named host operations (`sync_tripit`, `fetch_trakt_history`), `promote_staging`, `register_group`
- `/workspace/trusted/`
