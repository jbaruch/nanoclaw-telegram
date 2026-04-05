# Skill Tile Placement

When promoting a skill or rule, choose the correct tile. Getting this wrong breaks the security model — admin skills in core means untrusted containers get admin capabilities.

## The four tiles

| Tile | Who gets it | Purpose |
|------|------------|---------|
| **nanoclaw-core** | ALL containers (including untrusted) | Universal behavior: formatting, silence, language, context recovery. No external APIs, no credentials. |
| **nanoclaw-trusted** | Trusted + main | Shared operational: system health checks, memory management. No personal APIs (Gmail, Calendar). |
| **nanoclaw-admin** | Main only | Everything personal: email, calendar, CFPs, host scripts, group management, task scheduling, Composio integrations. |
| **nanoclaw-untrusted** | Untrusted only | Security restrictions: credential protection, code execution refusal, bad actor handling. |

## Decision criteria — apply in order

1. **Does it call Composio, Gmail, Calendar, Tasks, GitHub, Sessionize, or any external API?** → admin
2. **Does it call named host operations (sync_tripit, fetch_trakt_history) or manage infrastructure (promote, verify, groups)?** → admin
3. **Does it read/write `/workspace/trusted/` or manage shared memory?** → trusted
4. **Is it a security restriction for public groups?** → untrusted
5. **Is it pure logic with no credentials that ALL containers need?** → core

## Examples

| Skill | Tile | Reasoning |
|-------|------|-----------|
| check-email | admin | Gmail via Composio |
| morning-brief | admin | Calendar + Tasks via Composio |
| check-cfps | admin | Sessionize API + host scripts |
| heartbeat | admin | Calls Composio skills (email, calendar) |
| soul-searching | admin | Writes to /workspace/trusted/ |
| check-system-health | trusted | No external APIs, operational |
| check-unanswered | core | No external APIs, all containers need it |
| status | core | Basic container info |
| default-silence | core | Universal behavior rule |
| bad-actor-disengage | untrusted | Security rule for public groups |

## Red flags — if you see any of these, it's NOT core

- `Composio`, `GMAIL`, `GOOGLECALENDAR`, `GOOGLETASKS` anywhere in the skill
- Named host operations (`sync_tripit`, `fetch_trakt_history`), `promote_staging`, `register_group`
- `/workspace/trusted/`
- Any API key or credential reference
- `schedule_task` with complex scheduling logic
- Skills that only make sense for Baruch personally (books, shows, orders, travel)

## When in doubt → admin

Putting something in admin that belongs in core wastes a few tokens. Putting something in core that belongs in admin **gives untrusted containers admin capabilities**. Always err toward admin.
