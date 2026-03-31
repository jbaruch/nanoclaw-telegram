---
name: check-calendar
description: Detect calendar changes and reschedule reminders. Compares current Google Calendar events against saved state, cancels stale reminders, creates new ones for moved/added events. Use as part of heartbeat or standalone. Triggers on "check calendar", "calendar changes", "reschedule reminders".
---

# Check Calendar Changes

## Related Skills

- **Morning Brief** — handles initial scheduling at 8am and writes the first `calendar-state.json` for the day. This skill takes over from there, detecting intra-day changes.

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

**If nothing changed:** proceed to the UTC Reminder Verification step below (return nothing if that also finds nothing).

## Reschedule

If calendar changed:

1. **Cancel stale reminders:** For each event in state with a `reminder_task_id`, call `mcp__nanoclaw__cancel_task`:
   ```
   mcp__nanoclaw__cancel_task(task_id="task_7f3a9b")
   ```

2. **Create new reminders:** For each timed event (not all-day, not Travel, not "Home", not week-number events) starting more than 20 min from now — **and where `jbaruch@sadogursky.com` (`self=true`) does NOT have `responseStatus="declined"`** — schedule a new `once` task 15 min before start (local time, no Z suffix):
   ```
   mcp__nanoclaw__create_task(
     scheduled_time="2024-06-10T08:45:00",   # local time, no Z suffix
     recurrence="once",
     message="Reminder: Team Standup in 15 minutes"
   )
   ```
   Capture the returned `task_id` and store it as `reminder_task_id` for that event.

3. **Update state:** Write the new event list (with updated `reminder_task_id` values) and today's date back to `/workspace/group/calendar-state.json`.

## UTC Reminder Verification

Run this step every time (even when calendar has no changes). It ensures reminders are correct for the current timezone — handles travel timezone shifts.

1. Read `/workspace/group/scheduled-reminders.json`. If it doesn't exist or is empty, skip.

2. Determine current timezone:
   - First: look for `<context timezone="...">` tag in the current prompt
   - Fallback: read `current_tz` from `/workspace/group/nanoclaw-state.json`
   - Default: `America/Chicago`

3. Call `mcp__nanoclaw__list_tasks` to get all active scheduled tasks.

4. For each reminder in `scheduled-reminders.json`:
   - Compute `expected_fire_utc = utc_time - reminder_offset_min` (subtract offset in minutes from the UTC event time)
   - Convert `expected_fire_utc` to current local timezone → `expected_fire_local` (no Z suffix)
   - Find the task by `task_id` in the list_tasks output
   - If task not found OR `|task.fire_time - expected_fire_local| > 2 minutes`:
     a. Cancel the old task: `mcp__nanoclaw__cancel_task(task_id=old_task_id)`
     b. Schedule a new task at `expected_fire_local` (local time, no Z suffix)
     c. Update `task_id` in `scheduled-reminders.json` with the new task ID
     d. Save updated `scheduled-reminders.json`

5. If any reminders were rescheduled, wrap a brief note in `<internal>` tags (no user-visible output unless something went wrong).

## Output

Return the changes found and actions taken:
```
Calendar changed: [Event X] moved to 3pm -> reminder rescheduled
```

Or empty if no changes.
