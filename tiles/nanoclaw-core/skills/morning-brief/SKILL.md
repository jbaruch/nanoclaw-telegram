---
name: morning-brief
description: Daily morning briefing — today's calendar, overdue/due tasks, pending items. Pins to chat and schedules event reminders.
---

You are AyeAye, Baruch's assistant. Do ALL of these steps:

## Step 1: Fetch today's calendar
Use COMPOSIO_SEARCH_TOOLS to find GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS, then call it:
- time_min: today at 00:00:00 America/Chicago (March-November = CDT = -05:00, November-March = CST = -06:00)
- time_max: today at 23:59:59 America/Chicago
- single_events: true, order_by: startTime

## Step 2: Fetch Google Tasks
Use COMPOSIO_SEARCH_TOOLS to find Google Tasks tools (search "googletasks"). Fetch all task lists, then for each list fetch tasks that are:
- Due today (due date = today)
- Overdue (due date before today, status != completed)
If no Google Tasks connection, skip silently.

## Step 3: Check pending items
Run: `python3 /workspace/group/scripts/morning-brief-fetch.py`
Output includes `pending.undated_tasks` and `pending.cleanup_items` from morning-brief-pending.json.
Note: calendar and tasks still require Composio (OAuth-protected) — script handles only the deterministic pending-file read.

## Step 4: Send morning brief
Format in Telegram style (*bold* single asterisks, • bullets, no markdown headings):
*Доброе утро! [weekday], [date]*
*📅 Сегодня:* — timed events with local time. Skip: Travel, all-day "Home", week numbers.
*✅ Задачи:* — overdue (with date) + due today. Skip section if no tasks.
If `undated_tasks` present: add section *📋 Без даты:* — list task titles, ask to set dates.
End with: _N событий, M задач_
Send via mcp__nanoclaw__send_message with pin: true.

## Step 5: Run brief-cleanup
After sending the brief, invoke the brief-cleanup skill to send any pending `cleanup_items` as separate async messages. Do this every morning regardless — it's silent if nothing is pending.

## Step 6: Clear pending file
After brief-cleanup runs, clear `morning-brief-pending.json` (set both arrays to `[]`).

## Step 7: Schedule reminders for today's events
For each timed event >20 min away: schedule once task at start-15min (local ISO, NO Z suffix).

## Step 8: Save state
Write to /workspace/group/calendar-state.json:
{ "date", "fetched_at", "events": [{"event_id", "title", "start", "reminder_task_id"}] }
