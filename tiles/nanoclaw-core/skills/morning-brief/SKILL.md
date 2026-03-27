---
name: morning-brief
description: "Morning briefing — fetches today's Google Calendar events and Tasks, sends a Telegram-formatted summary via mcp__nanoclaw__send_message, schedules 15-minute reminders for upcoming events, and saves state to disk. Runs daily at 8am as a scheduled task. Use when: \"morning brief\", \"daily briefing\", \"what's on today\", \"today's schedule\"."
---

# Morning Brief

**Overview — six steps run in order:**
1. Fetch today's Google Calendar events
2. Fetch Google Tasks (due today + overdue)
3. Check closing CFPs (`/check-cfps`)
4. Send a formatted Telegram briefing (including CFP deadlines if any)
5. Schedule 15-minute reminders for timed events
6. Save state to `/workspace/group/calendar-state.json`

---

## Step 1: Fetch today's calendar

Use `COMPOSIO_SEARCH_TOOLS` to find `GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS`, then call it:
- time_min: today at 00:00:00 America/Chicago (March-November = CDT = -05:00, November-March = CST = -06:00)
- time_max: today at 23:59:59 America/Chicago
- single_events: true, order_by: startTime

**Error handling:** If the calendar fetch fails or returns an error, send a degraded briefing via `mcp__nanoclaw__send_message` noting that calendar data is unavailable, then continue to Step 2.

## Step 2: Fetch Google Tasks

Use `COMPOSIO_SEARCH_TOOLS` to find Google Tasks tools (search "googletasks"). Fetch all task lists, then for each list fetch tasks that are:
- Due today (due date = today)
- Overdue (due date before today, status != completed)

If no Google Tasks connection, skip silently. If the fetch errors unexpectedly, skip silently and note the omission in the briefing footer.

## Step 3: Check closing CFPs

Invoke `/check-cfps` to find relevant CFPs closing within 7 days. Collect the results for inclusion in the briefing.

If the check fails or returns nothing, skip the CFP section silently.

## Step 4: Send morning brief

Format in Telegram style (`*bold*` single asterisks, `•` bullets, no markdown headings):

```
*Доброе утро! [weekday], [date]*

*📅 Сегодня:*
• 09:00 — Team Standup
• 12:00 — Lunch with Sarah
• 15:00 — PR Review

*✅ Задачи:*
• ⚠️ Overdue: Fix CI pipeline (due Mar 25)
• Write blog post draft

*📢 CFPs closing this week:*
• Devoxx Belgium — closes in 2 days
• AI Dev Summit — closes in 5 days

_N событий, M задач, K CFPs_
```

Rules:
- Skip: Travel events, all-day "Home" entries, week number entries
- Flag tight connections or conflicts between events
- If no tasks: skip the tasks section entirely
- Send via `mcp__nanoclaw__send_message`

**Error handling:** If `mcp__nanoclaw__send_message` fails, log the error and retry once. If it fails again, abort the remaining steps and surface the error.

## Step 5: Schedule reminders for today's events

For each timed event (not all-day, not travel) that starts MORE than 20 minutes from now:
- Calculate reminder time = event start minus 15 minutes (America/Chicago)
- Format as local ISO: `2026-03-27T08:45:00` (NO Z suffix, no timezone offset)
- Schedule a `once` task:
  - schedule_type: `once`
  - schedule_value: the local time string
  - context_mode: `isolated`
  - prompt: `Send a reminder to Baruch: *[event title]* starts in 15 minutes at [time]. Use mcp__nanoclaw__send_message to deliver it.`

**Error handling:** If `schedule_task` fails for an individual event, record `reminder_task_id: null` for that event in state and continue scheduling the remaining events. Do not abort the whole step.

## Step 6: Save state

Write to `/workspace/group/calendar-state.json`:

```json
{
  "date": "YYYY-MM-DD",
  "fetched_at": "ISO timestamp",
  "events": [
    {
      "event_id": "...",
      "title": "...",
      "start": "ISO",
      "end": "ISO",
      "all_day": false,
      "reminder_task_id": "task-xxx or null"
    }
  ]
}
```

Only timed, non-travel events. `reminder_task_id` = task ID from schedule_task, or `null` if event is too soon to remind (< 20 min away) or if scheduling failed.

After writing, read the file back and confirm it parses as valid JSON containing the expected number of events. If the file is missing, empty, or unparseable, retry the write once. If it fails again, send a warning message via `mcp__nanoclaw__send_message`.

This state file is read by `/check-calendar` (heartbeat sub-check) to detect mid-day calendar changes and reschedule reminders.
