---
name: check-calendar
description: Detect calendar changes and reschedule reminders. Compares current Google Calendar events against saved state, cancels stale reminders, creates new ones for moved/added events. Use as part of heartbeat or standalone. Triggers on "check calendar", "calendar changes", "reschedule reminders".
---

# Check Calendar Changes

## Related Skills

- **Morning Brief** — handles initial scheduling at 8am and writes the first `calendar-state.json` for the day. This skill takes over from there, detecting intra-day changes.

## Precondition

Read `/workspace/group/calendar-state.json`. If it exists and `date` matches today, proceed. If the state file doesn't exist or is from a previous day, skip the Compare/Reschedule steps — but **still run the UTC Reminder Verification and Declined Event Sweep steps below**, fetching today's events fresh if needed.

### Expected calendar-state.json calendar section

```json
{
  "date": "2024-06-10",
  "events": [
    {
      "event_id": "abc123xyz",
      "title": "Team Standup",
      "start": "2024-06-10T09:00:00Z",
      "end": "2024-06-10T09:30:00Z",
      "all_day": false,
      "reminder_task_id": "task_7f3a9b"
    }
  ]
}
```

`start` is always stored as **UTC with Z suffix**. `reminder_task_id`: task ID from when the reminder was scheduled, or `null` if none was created (e.g. all-day or Travel events).

## Fetch current events

Discover calendar tool per `composio-preamble` rule, then fetch today's events:
- time_min/time_max = today in `current_tz` (from `task-tz-state.json`)
- single_events = true
- order_by = startTime

Keep the fetched events list in memory — it's used by both the Compare step and the Declined Event Sweep below.

## Compare

Compare fetched events to the state file's `events` list (match by `event_id`). Check for:
- New events added
- Existing events with changed time or title
- Events removed or declined (responseStatus changed to "declined")

**If nothing changed:** proceed to UTC Reminder Verification (return nothing if that also finds nothing).

## Reschedule

If calendar changed:

1. **Cancel stale reminders:** For each event in state with a `reminder_task_id`, call `mcp__nanoclaw__cancel_task`:
   ```
   mcp__nanoclaw__cancel_task(task_id="task_7f3a9b")
   ```

2. **Create new reminders:** For each timed event (not all-day, not Travel, not "Home", not week-number events) starting more than 20 min from now — **applying the `event-filter-rules` (including declined event check)**:

   **TIMEZONE CONVERSION — do this exactly:**
   ```python
   from zoneinfo import ZoneInfo
   from datetime import datetime, timedelta

   # event_start_utc comes from the API response start.dateTime
   # The API returns local time with offset, e.g. "2026-04-03T16:00:00-05:00"
   event_dt = datetime.fromisoformat(event_start_str)  # timezone-aware
   event_utc = event_dt.astimezone(ZoneInfo("UTC"))
   reminder_utc = event_utc - timedelta(minutes=15)

   # Convert to current local timezone for the scheduler
   current_tz = ZoneInfo(current_tz_str)  # e.g., "America/Chicago"
   reminder_local = reminder_utc.astimezone(current_tz)
   schedule_value = reminder_local.strftime("%Y-%m-%dT%H:%M:%S")  # NO Z suffix
   ```

   Example: event at 4:00 PM CDT ("2026-04-03T16:00:00-05:00" = 21:00 UTC) → reminder_utc = 20:45 UTC → CDT = 3:45 PM → `schedule_value = "2026-04-03T15:45:00"`

   Schedule:
   ```
   mcp__nanoclaw__schedule_task(
     prompt="Send reminder: ...",
     schedule_type="once",
     schedule_value="2026-04-03T15:45:00"  # local time, NO Z suffix
   )
   ```
   Capture the returned `task_id` and store it as `reminder_task_id` for that event.

3. **Update state:** Write the new event list (with updated `reminder_task_id` values) and today's date back to `/workspace/group/calendar-state.json`. Store `start` as UTC with Z suffix.

## Declined Event Sweep

Run this step every time (even when calendar has no changes). It catches reminders scheduled by morning-brief for events that were later declined.

1. Read `/workspace/group/scheduled-reminders.json`. If it doesn't exist or is empty, skip.

2. Build a lookup map of `event_id → responseStatus` from the fetched events (fetched above). If events weren't fetched (precondition failed), fetch today's events now.

3. For each reminder entry in `scheduled-reminders.json` that has an `event_id`:
   - Look up that `event_id` in the fetched events
   - If the attendee with `self=true` has `responseStatus="declined"` **OR** the event no longer appears in today's calendar:
     a. Cancel the task: `mcp__nanoclaw__cancel_task(task_id=<reminder's task_id>)`
     b. Remove this entry from `scheduled-reminders.json`
     c. Save the updated file

4. Wrap any cancellations in `<internal>` tags — no user-visible output.

## UTC Reminder Verification

Run every time (even when calendar has no changes) to ensure reminders are correct for the current timezone — handles travel timezone shifts.

1. Read `/workspace/group/scheduled-reminders.json`. If missing or empty, skip.

2. Determine current timezone (in priority order):
   - `<context timezone="...">` tag in the current prompt
   - `current_tz` from `/workspace/group/task-tz-state.json`
   - `home_tz` from `task-tz-state.json` or TZ env var

3. Call `mcp__nanoclaw__list_tasks` to get all active scheduled tasks.

4. For each reminder in `scheduled-reminders.json`:
   - Compute `reminder_utc = utc_time - reminder_offset_min` (subtract minutes from the stored UTC time)
   - Convert `reminder_utc` to current local timezone → `expected_fire_local` (no Z suffix)
   - Find the task by `task_id` in list_tasks output
   - If task not found OR `|task.fire_time_utc - reminder_utc| > 2 minutes`:
     a. Cancel: `mcp__nanoclaw__cancel_task(task_id=old_task_id)`
     b. Schedule new task at `expected_fire_local` (no Z suffix)
     c. Update `task_id` in `scheduled-reminders.json` and save

5. If any reminders were rescheduled, wrap a brief note in `<internal>` tags.

## Output

Return the changes found and actions taken:
```
Calendar changed: [Event X] moved to 3pm -> reminder rescheduled
```

Or empty if no changes.
