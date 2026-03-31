# Skill Tile Placement Rule

When promoting a skill, always choose the correct tile based on who needs the skill:

## nanoclaw-admin
Skills that require **elevated privileges** or are only meaningful in the main channel:
- `verify-tiles` — tile management, requires promote_staging MCP tool
- `manage-groups` — group registration and configuration
- `morning-brief` — personal daily briefing (main channel only)
- `nightly-housekeeping` — maintenance tasks requiring host scripts
- `soul-review` — personal profile review
- Any skill that calls `run_host_script`, `promote_staging`, or manages NanoClaw infrastructure

## nanoclaw-core
Skills needed by **all containers** (main, trusted, untrusted):
- `heartbeat` — universal health check
- `check-email`, `check-calendar`, `check-unanswered` — universal monitoring
- `format-message` — formatting reference for all groups
- `check-unanswered` — message auditing
- Any skill that a non-main group container might legitimately invoke

## Rule of thumb
Ask: "Does an untrusted group container need this?" → core.
Ask: "Does this require admin/host access or is it main-channel-specific?" → admin.

**Never put admin-only skills in nanoclaw-core.** They'll be available to untrusted containers.
