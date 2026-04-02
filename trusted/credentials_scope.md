# Credentials Scope

## What I actually have
- `COMPOSIO_API_KEY` — for Google Calendar, Gmail, Google Tasks, Google Drive, etc. via Composio
- `ANTHROPIC_API_KEY` — placeholder, proxied by host (no real key in container)

## What I do NOT have
- ❌ Reclaim.ai API — no token
- ❌ TripIt API — no token
- ❌ Google OAuth directly — only via Composio
- ❌ Any other personal API keys

## Rule
Never claim access to Reclaim, TripIt, or raw Google OAuth in any chat (main or otherwise).
For anything requiring these: use `mcp__nanoclaw__run_host_script(script: "...")`.
Host has the credentials; I do not.

This was learned after incorrectly claiming Reclaim access in old.wtf group (2026-03-29).
