---
name: check-calendar
description: Detect calendar changes and reschedule reminders. Compares current Google Calendar events against saved state, cancels stale reminders, creates new ones for moved/added events. Also verifies UTC-based scheduled reminders for timezone correctness and fixes drift. Use as part of heartbeat or standalone. Triggers on "check calendar", "calendar changes", "reschedule reminders".
---

# Check Calendar Changes

## Related Skills

- **Morning Brief** — handles initial scheduling at 8am and writes the first `calendar-state.json` and `scheduled-reminders.json` for the day. This skill takes over from there, detecting intra-day changes.

## Precondition

Read `/workspace/group/calendar-state.json`. If it exists and `date` matches today, proceed. If the state file doesn't exist or is from a previous day, skip — morning brief handles initial scheduling at 8am.

### Expected calendar-state.json calendar section

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
- Events removed or declined (responseStatus changed to "declined")

**If nothing changed:** proceed to the [Verify UTC Reminders](#verify-utc-reminders) step below, then return (wrap output in `<internal>` if no changes were found and no reminders were fixed).

## Reschedule

If calendar changed:

1. **Cancel stale reminders:** For each event in state with a `reminder_task_id`, call `mcp__nanoclaw__cancel_task`:
   ```
   mcp__nanoclaw__cancel_task(task_id="task_7f3a9b")
   ```
   Also remove the corresponding entry from `/workspace/group/scheduled-reminders.json` by `event_id`.

2. **Create new reminders:** For each timed event (not all-day, not Travel, not "Home", not week-number events) starting more than 20 min from now — **and where `jbaruch@sadogursky.com` (`self=true`) does NOT have `responseStatus="declined"`** — schedule a new `once` task 15 min before start (local time, no Z suffix):
   ```
   mcp__nanoclaw__schedule_task(
     schedule_type="once",
     schedule_value="2024-06-10T08:45:00",   # local time, no Z suffix
     prompt="Reminder: Team Standup in 15 minutes"
   )
   ```
   Capture the returned `task_id` and store it as `reminder_task_id` for that event.

   After scheduling, append an entry to `/workspace/group/scheduled-reminders.json`:
   ```json
   {
     "event_id": "<google_calendar_event_id>",
     "title": "<event title>",
     "utc_time": "<event start in UTC, e.g. 2026-04-01T14:30:00Z>",
     "reminder_offset_min": 15,
     "task_id": "<returned task_id>"
   }
   ```
   Convert the event's local start time to UTC before storing.

3. **Update state:** Write the new event list (with updated `reminder_task_id` values) and today's date back to `/workspace/group/calendar-state.json`.

---

## Verify UTC Reminders

This step runs after the calendar compare (whether or not changes were detected). It audits all scheduled reminders for timezone drift.

### 1. Read scheduled-reminders.json

Read `/workspace/group/scheduled-reminders.json`. If the file doesn't exist or `reminders` is empty, skip this section entirely.

### 2. Determine current timezone

Check for `<context timezone>` in the prompt. If not present, read `/workspace/group/nanoclaw-state.json` and use the `current_tz` field. Fall back to `America/Chicago` if neither is available.

### 3. List all scheduled tasks

Call `mcp__nanoclaw__list_tasks` to get the current list of scheduled tasks.

### 4. For each reminder in scheduled-reminders.json

Compute the **expected fire time**:
```
expected_fire_utc = utc_time - reminder_offset_min (in minutes)
```

Find the matching task by `task_id` in the list_tasks output. Extract the task's current scheduled fire time (convert to UTC if needed).

**Check for mismatch:** If the task is not found OR the difference between expected and actual fire time is greater than 2 minutes:

1. If old task exists — call `mcp__nanoclaw__cancel_task(task_id=old_task_id)`.
2. Compute the correct local fire time: convert `expected_fire_utc` to the current timezone.
3. Schedule a new task:
   ```
   mcp__nanoclaw__schedule_task(
     schedule_type="once",
     schedule_value="<correct local time, no Z suffix>",
     prompt="Reminder: <title> in 15 minutes"
   )
   ```
4. Update the reminder entry in `scheduled-reminders.json` with the new `task_id`.

### 5. Write updated scheduled-reminders.json

After processing all reminders, write the updated `scheduled-reminders.json` back to disk (only if any task_ids changed).

### 6. Report

If any reminders were fixed, include a line in the output:
```
Reminder drift fixed: [Event Title] rescheduled (was <old_time>, now <new_time> local)
```

If nothing needed fixing, wrap in `<internal>`.

---

## Output

Return the changes found and actions taken:
```
Calendar changed: [Event X] moved to 3pm -> reminder rescheduled
Reminder drift fixed: [Event Y] rescheduled (was 14:30, now 09:30 local)
```

Or empty (wrapped in `<internal>`) if no changes and no drift found.
