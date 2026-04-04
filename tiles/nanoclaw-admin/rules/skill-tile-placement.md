# Skill Tile Placement — Hard Rules

**STOP. Before promoting anything, read this entire table. If a skill or rule is listed below, put it in the listed tile. No exceptions. No reasoning your way around it.**

## Concrete placement table

| Skill/Rule | Tile | Why |
|-----------|------|-----|
| heartbeat | admin | Composio, host scripts |
| morning-brief | admin | Composio (Calendar, Tasks) |
| nightly-housekeeping | admin | Composio, host scripts |
| check-email | admin | Composio (Gmail) |
| check-calendar | admin | Composio (Google Calendar) |
| check-cfps | admin | Sessionize API, host scripts |
| check-orders | admin | Composio |
| check-travel-bookings | admin | Composio |
| check-watchlist | admin | Composio |
| soul-searching | admin | Reads /workspace/trusted/, writes SOUL.md |
| promote-tiles | admin | Infrastructure management |
| verify-tiles | admin | Infrastructure management |
| manage-groups | admin | Group registration |
| schedule-task | admin | Task management |
| create-agent-team | admin | Agent teams |
| recommend-books | admin | Personal |
| recommend-shows | admin | Personal |
| brief-cleanup | admin | Email classification feedback |
| max-effort | admin | Extended tool reference (Composio examples) |
| no-unverified-claims | admin | Extended verification (Composio examples) |
| trakt-watch-history | admin | External API |
| task-tz-sync | admin | Host scripts |
| scheduler-timezone | admin | Task management |
| check-system-health | trusted | No external APIs, shared operational |
| check-unanswered | core | No external APIs, all containers need it |
| trusted-memory | trusted | Reads /workspace/trusted/ |
| status | core | Basic container health |
| whoami | untrusted | Identity disclosure for untrusted |

## What NEVER goes in core

- Anything that calls Composio, Gmail, Calendar, Tasks, GitHub
- Anything that calls `run_host_script`
- Anything that references `/workspace/trusted/`
- Anything that manages infrastructure (promote, verify, groups, tasks)
- Any skill that exists in the admin table above

## What NEVER goes in untrusted

- Any operational skill (heartbeat, morning-brief, check-*)
- Any skill that writes files (brief-cleanup, soul-searching)
- Any skill from admin or trusted — untrusted gets core + untrusted-security only

## Decision process

1. **Is the skill in the table above?** Use the listed tile. Done.
2. **New skill not in the table?** Apply these rules in order:
   - Needs external credentials → **admin**
   - Needs `/workspace/trusted/` → **trusted**
   - Pure logic, no credentials, useful for all containers → **core**
   - Security restriction → **untrusted**
3. **Still unsure?** → **admin**. Wrong tile = security model broken.
