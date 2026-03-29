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
6. Save state to `/workspace/group/nanoclaw-state.json`

See **Error Policy** at the bottom for how failures at each step are handled.

---

## Step 1: Fetch today's calendar

Call `GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS` (discover via `COMPOSIO_SEARCH_TOOLS` if needed):

```
GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS(
    time_min="<today>T00:00:00-05:00",   # today 00:00 America/Chicago
    time_max="<today>T23:59:59-05:00",   # today 23:59 America/Chicago
    single_events=true,
    order_by="startTime"
  )
```

Timezone offset: CDT (Mar–Nov) = `-05:00`; CST (Nov–Mar) = `-06:00`.

## Step 2: Fetch Google Tasks

Use `COMPOSIO_SEARCH_TOOLS(query="googletasks")` to find Google Tasks tools. Fetch all task lists, then for each list fetch tasks that are:
- Due today (due date = today)
- Overdue (due date before today, status != completed)

## Step 3: Check closing CFPs

Invoke `/check-cfps` to find relevant CFPs closing within 7 days. Collect the results for inclusion in the briefing.

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

## Step 5: Schedule reminders for today's events

For each timed event (not all-day, not travel) that starts MORE than 20 minutes from now:
- Calculate reminder time = event start minus 15 minutes (America/Chicago)
- Format as local ISO: `<today>T08:45:00` (NO Z suffix, no timezone offset)
- Schedule a `once` task:
  - schedule_type: `once`
  - schedule_value: the local time string
  - context_mode: `isolated`
  - prompt: `Send a reminder to Baruch: *[event title]* starts in 15 minutes at [time]. Use mcp__nanoclaw__send_message to deliver it.`

## Step 6: Save state

Write to `/workspace/group/nanoclaw-state.json`:

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

Only timed, non-travel events. `reminder_task_id` = task ID from schedule_task, or `null` if event is too soon (< 20 min) or scheduling failed.

After writing, read the file back and confirm it parses as valid JSON containing the expected number of events.

This state file is read by `/check-calendar` (heartbeat sub-check) to detect mid-day calendar changes and reschedule reminders.

---

## Error Policy

| Step | Failure | Action |
|------|---------|--------|
| 1 — Calendar fetch | Error or empty response | Send degraded briefing noting calendar unavailable; continue to Step 2 |
| 2 — Tasks fetch | No connection | Skip silently |
| 2 — Tasks fetch | Unexpected error | Skip silently; note omission in briefing footer |
| 3 — CFPs check | Fails or returns nothing | Skip CFP section silently |
| 4 — Send message | `mcp__nanoclaw__send_message` fails | Retry once; if still failing, abort remaining steps and surface the error |
| 5 — Schedule reminder | `schedule_task` fails for one event | Set `reminder_task_id: null` for that event; continue scheduling the rest |
| 6 — Write state | File missing, empty, or unparseable after write | Retry write once; if still failing, send a warning via `mcp__nanoclaw__send_message` |
