# Skill Tile Placement

When promoting a skill or rule, choose the correct tile. Getting this wrong breaks the security model — admin skills in core means untrusted containers get admin capabilities.

## The four tiles

| Tile | Who gets it | Purpose |
|------|------------|---------|
| **nanoclaw-core** | ALL containers (including untrusted) | Universal behavior: formatting, silence, language, context recovery. No external APIs, no credentials. |
| **nanoclaw-trusted** | Trusted + main | Shared operational: system health checks, memory management. No personal APIs (Gmail, Calendar). |
| **nanoclaw-admin** | Main only | Everything personal: email, calendar, host operations, group management, task scheduling, Composio integrations. |
| **nanoclaw-untrusted** | Untrusted only | Security restrictions: credential protection, code execution refusal, bad actor handling. |

## Decision criteria — apply in order

1. **Does it call Composio, Gmail, Calendar, Tasks, GitHub, or any external API?** → admin
2. **Does it call named host operations (sync_tripit, fetch_trakt_history)?** → admin
3. **Does it manage infrastructure (promote, verify, groups, schedule)?** → admin
4. **Is it personal to Baruch (books, shows, orders, travel, CFPs)?** → admin
5. **Does it read/write `/workspace/trusted/` or manage shared cross-group memory?** → trusted
6. **Is it a security restriction for public groups?** → untrusted
7. **Is it pure logic with no credentials that ALL containers need?** → core

## Examples

| Skill | Tile | Why |
|-------|------|-----|
| check-email | admin | Gmail via Composio |
| morning-brief | admin | Calendar + Tasks via Composio, personal schedule |
| check-cfps | admin | External APIs (Sessionize, web search), personal CFP tracking |
| heartbeat | admin | Calls Composio skills (email, calendar), host operations |
| nightly-housekeeping | admin | Host operations (sync_tripit, fetch_trakt_history), personal workflows |
| trakt-watch-history | admin | External API (Trakt), personal entertainment |
| check-travel-bookings | admin | Travel data (personal), uses tripit-url.txt |
| recommend-books | admin | Personal to Baruch |
| soul-searching | admin | Writes to /workspace/trusted/ |
| check-system-health | trusted | No external APIs, operational, shared across trusted groups |
| trusted-memory | trusted | Manages /workspace/trusted/ memory |
| check-unanswered | core | No external APIs, all containers need it |
| status | core | Basic container info, no APIs |
| default-silence | core | Universal behavior rule |
| bad-actor-disengage | untrusted | Security rule for public groups |
| whoami | untrusted | Identity for public groups |

## Red flags — if you see ANY of these, it's NOT core or trusted

- `Composio`, `GMAIL`, `GOOGLECALENDAR`, `GOOGLETASKS` anywhere in the skill
- Named host operations (`sync_tripit`, `fetch_trakt_history`)
- `promote_staging`, `register_group`, `github_backup`
- `/workspace/trusted/` (this means trusted, NOT admin — unless it also uses external APIs)
- Any API key, token, or credential reference
- `schedule_task` with complex scheduling logic
- References to personal data: books, shows, orders, travel, CFPs, watch history

## When in doubt — ASK Baruch

Do NOT guess. If you're unsure which tile a skill belongs in, ask: "Should this go in admin or trusted?" with your reasoning. Putting something in the wrong tile is worse than asking.

**Default when not asked:** admin. Wasting tokens in admin is harmless. Leaking admin skills to untrusted containers is a security incident.

## Skills must NEVER duplicate across tiles

Each skill exists in exactly ONE tile. If check-unanswered is in core, it must NOT also be in trusted or admin. Duplicates waste tokens and cause version drift.
