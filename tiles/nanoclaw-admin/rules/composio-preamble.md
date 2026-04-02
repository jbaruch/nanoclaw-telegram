# Composio Tool Discovery

Before using any external API via Composio, discover the tool first using `COMPOSIO_SEARCH_TOOLS`. Cache the resolved tool name for the duration of the skill.

## Common tools

| Need | Search query | Expected tool |
|------|-------------|---------------|
| Calendar events | `"googlecalendar events list all calendars"` | GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS |
| Gmail fetch | `"gmail fetch emails"` | GMAIL_FETCH_EMAILS |
| Google Tasks list | `"googletasks list"` | GOOGLETASKS_LIST_TASKS |
| Google Tasks get | `"googletasks get"` | GOOGLETASKS_GET_TASKS |
| Google Tasks patch | `"googletasks patch task"` | GOOGLETASKS_PATCH_TASK |
| Google Tasks update | `"googletasks update task"` | GOOGLETASKS_UPDATE_TASK |

## Pattern

```
1. COMPOSIO_SEARCH_TOOLS(query="<search string>")
2. Use the returned tool name for all subsequent calls
3. If tool not found: note the failure, skip the step, continue
```

Do NOT hardcode tool names — they may change. Always discover first.
