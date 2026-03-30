---
name: check-calendar
description: Detect calendar changes and reschedule reminders. Compares current Google Calendar events against saved state, cancels stale reminders, creates new ones for moved/added events. Use as part of heartbeat or standalone. Triggers on "check calendar", "calendar changes", "reschedule reminders".
---

# Check Calendar Changes

## Precondition

Read `/workspace/group/nanoclaw-state.json`. If it exists and `date` matches today, proceed. If the state file doesn't exist or is from a previous day, skip — morning brief handles initial scheduling at 8am.

### Expected nanoclaw-state.json calendar section

```json
{
  "date": "2024-06-10",
  "events": [
    {
      "event_id": "abc123xyz",
      "title": "Team Standup",
      "start": "2024-06-10T09:00:00",
      "end": "2024-06-10T09:30:00",
      "all_day": false,
      "reminder_task_id": "task_7f3a9b"
    },
    {
      "event_id": "def456uvw",
      "title": "Lunch with Sarah",
      "start": "2024-06-10T12:00:00",
      "end": "2024-06-10T13:00:00",
      "all_day": false,
      "reminder_task_id": null
    }
  ]
}
```

- `date`: ISO date string for today (`YYYY-MM-DD`).
- `event_id`: Google Calendar event ID (used as the match key during comparison).
- `reminder_task_id`: The task ID returned when the reminder was scheduled, or `null` if no reminder was created (e.g. all-day or Travel events).

## Fetch current events

Use `COMPOSIO_SEARCH_TOOLS` to find `GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS`, then fetch today's events:
- time_min/time_max = today in America/Chicago
- single_events = true
- order_by = startTime

## Compare

Compare the fetched events to the state file's `events` list (match by event_id). Check for:
- New events added
- Existing events changed time or title
- Events removed

**If nothing changed:** return nothing (wrap output in `<internal>`).

## Reschedule

If calendar changed:

1. **Cancel stale reminders:** For each event in state with a `reminder_task_id`, call `mcp__nanoclaw__cancel_task`:
   ```
   mcp__nanoclaw__cancel_task(task_id="task_7f3a9b")
   ```

2. **Create new reminders:** For each timed event (not all-day, not Travel) starting more than 20 min from now, schedule a new `once` task 15 min before start (local time, no Z suffix):
   ```
   mcp__nanoclaw__create_task(
     scheduled_time="2024-06-10T08:45:00",   # local time, no Z suffix
     recurrence="once",
     message="Reminder: Team Standup in 15 minutes"
   )
   ```
   Capture the returned `task_id` and store it as `reminder_task_id` for that event.

3. **Update state:** Write the new event list (with updated `reminder_task_id` values) and today's date back to `/workspace/group/nanoclaw-state.json`.

## Output

Return the changes found and actions taken:
```
Calendar changed: [Event X] moved to 3pm -> reminder rescheduled
```

Or empty if no changes.
