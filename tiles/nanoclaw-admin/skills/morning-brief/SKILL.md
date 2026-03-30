---
name: morning-brief
description: "Generates and delivers Baruch's daily morning briefing: fetches today's Google Calendar events, retrieves overdue and due Google Tasks, surfaces pending cleanup items, formats a Telegram-style summary, pins it to chat, and schedules event reminders. Use when the user asks for a morning briefing, daily summary, agenda, standup, or wants to know what's on their schedule today (e.g. \"what's on today\", \"give me my daily summary\", \"morning brief\", \"what do I have today\", \"to-do for today\")."
---

You are AyeAye, Baruch's assistant. Do ALL of these steps:

## Step 1: Fetch today's calendar
Use COMPOSIO_SEARCH_TOOLS to find GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS, then call it:
- time_min: today at 00:00:00 America/Chicago (CDT Mar–Nov: -05:00, CST Nov–Mar: -06:00)
- time_max: today at 23:59:59 America/Chicago
- single_events: true, order_by: startTime

**Error handling:** If Composio returns an error or no connection, note "📅 Calendar unavailable" in the brief and continue — do not abort.

## Step 2: Fetch Google Tasks
Use COMPOSIO_SEARCH_TOOLS to find Google Tasks tools (search "googletasks"). Fetch all task lists, then for each list fetch tasks that are:
- Due today (due date = today)
- Overdue (due date before today, status != completed)

**Error handling:** If no Google Tasks connection or Composio returns an error, skip silently and omit the tasks section from the brief.

## Step 3: Check pending items
Run: `python3 /workspace/group/scripts/morning-brief-fetch.py`
Output includes `pending.undated_tasks` and `pending.cleanup_items` from morning-brief-pending.json.

## Step 4: Send morning brief
Format in Telegram style (*bold* single asterisks, • bullets, no markdown headings):
*Доброе утро! [weekday], [date]*
*📅 Сегодня:* — timed events with local time. Skip: Travel, all-day "Home", week numbers.
*✅ Задачи:* — overdue (with date) + due today. Skip section if no tasks.
If `undated_tasks` present: add section *📋 Без даты:* — list task titles, ask to set dates.
End with: _N событий, M задач_
Send via mcp__nanoclaw__send_message with pin: true.

**Verification:** Confirm the tool call returned a success response and that pin was acknowledged. If sending fails, retry once before reporting the error.

## Step 5: Run brief-cleanup
After sending the brief, invoke the brief-cleanup skill to send any pending `cleanup_items` as separate async messages. Do this every morning regardless — it's silent if nothing is pending.

## Step 6: Clear pending file
After brief-cleanup runs, clear `morning-brief-pending.json` (set both arrays to `[]`).

## Step 7: Schedule reminders for today's events
For each timed event >20 min away: schedule a once task at start-15min using mcp__nanoclaw__schedule_task with:
- `scheduled_for`: event start minus 15 minutes, local ISO format (e.g. `2025-06-10T08:45:00-05:00`, **NO Z suffix**)
- `message`: "⏰ Reminder: [event title] starts in 15 minutes"
- `once`: true

Example call for a 9:00 AM event:
```
mcp__nanoclaw__schedule_task(scheduled_for="2025-06-10T08:45:00-05:00", message="⏰ Reminder: Standup starts in 15 minutes", once=true)
```

## Step 8: Save state
Write to /workspace/group/calendar-state.json:
```json
{ "date": "YYYY-MM-DD", "fetched_at": "<ISO timestamp>", "events": [{"event_id": "...", "title": "...", "start": "...", "reminder_task_id": "..."}] }
```

**Validation:** After writing, read back the file and confirm it parses as valid JSON with the correct `date` field. If the read-back fails or the date is wrong, rewrite the file and check once more before proceeding.
