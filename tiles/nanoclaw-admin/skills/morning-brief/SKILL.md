---
name: morning-brief
description: Generates and delivers Baruch's daily morning briefing. Fetches today's Google Calendar events, retrieves overdue and due Google Tasks, surfaces undated tasks and pending cleanup items, checks flagged orders, formats everything in Telegram style, pins the message to chat, and schedules per-event reminders. Use when the user asks for a morning briefing, daily summary, agenda, standup overview, or wants to know what's on their schedule today — e.g. "good morning", "what's on my plate today", "daily brief", "give me my agenda", "what tasks are due", or "run the morning brief".
---

You are AyeAye, Baruch's assistant.

## Tool Discovery (run once at start)
Use `COMPOSIO_SEARCH_TOOLS` to locate and cache the following tools:
- **Calendar:** search `"googlecalendar events list all calendars"`
- **Tasks (list):** search `"googletasks list"`
- **Tasks (get):** search `"googletasks get"`
- **Scheduler/Reminder:** search `"reminder create"` or `"task schedule"`

If a tool cannot be resolved, follow the per-step failure instructions below.

---

## Event Filter Rules (shared reference)
Apply these exclusions whenever processing calendar events — in both Step 5 (brief display) and Step 8 (reminders):
- Skip Travel events
- Skip all-day "Home" events
- Skip week-number events
- Skip any event where Baruch's attendee entry (`jbaruch@sadogursky.com`, `self=true`) has `responseStatus="declined"`

---

## Step 1: Fetch today's calendar
Call the resolved calendar tool:
- `time_min`: today at 00:00:00 America/Chicago
- `time_max`: today at 23:59:59 America/Chicago
- `single_events`: true, `order_by`: startTime

On failure: note it and continue to Step 2.

## Step 2: Fetch Google Tasks
Using the resolved tasks tools, fetch all task lists, then for each list fetch tasks that are:
- Due today (due date = today)
- Overdue (due date before today, status != completed)

On failure or no connection: skip silently and continue.

## Step 3: Check pending items
Run: `python3 /workspace/group/scripts/morning-brief-fetch.py`
Output includes `pending.undated_tasks` and `pending.cleanup_items` from `morning-brief-pending.json`.
On non-zero exit or no output: treat both arrays as empty and continue.

## Step 4: Check flagged orders
Read `/workspace/group/orders-db.json`. Collect all orders where `flagged: true`.

If flagged orders exist, include them in the brief under:
`📦 <b>Заказы:</b>` — one bullet per order: description, flag_reason, source, order_date.
If none → skip this section silently.

## Step 5: Send morning brief
Apply the [Event Filter Rules](#event-filter-rules-shared-reference) when selecting which events to display.

Format in Telegram HTML/style (*bold* single asterisks, • bullets, no markdown headings). Canonical example:

```
*Доброе утро! Понедельник, 9 июня*

*📅 Сегодня:*
• 09:00 — Standup with team
• 14:00 — 1:1 with Alex

*✅ Задачи:*
• ⚠️ Обновить README (просрочено: 6 июня)
• Ответить на письмо Михаила

*📋 Без даты:*
• Разобрать инбокс
• Обновить CV
_Нужно установить дату для этих задач_

_2 события, 3 задачи_
```

**Section rules:**
- *📅 Сегодня:* — timed events with local time (apply Event Filter Rules).
- *✅ Задачи:* — overdue tasks (original due date, marked ⚠️) + tasks due today. Omit if no tasks.
- *📋 Без даты:* — list titles from `undated_tasks` with prompt to set dates. Omit if array is empty.
- Footer: `_N событий, M задач_`

Send via `mcp__nanoclaw__send_message` with `pin: true`.

**Checkpoint:** Confirm the message was sent successfully (tool returns success/message ID) before proceeding to Steps 6 and 8. If sending fails, retry once; if still failing, log the error and stop.

## Step 6: Run brief-cleanup
After the brief is confirmed sent, invoke the brief-cleanup skill to send any pending `cleanup_items` as separate async messages. Run every morning — it is silent if nothing is pending.

## Step 7: Clear pending file
After brief-cleanup runs, set both arrays in `morning-brief-pending.json` to `[]`.

## Step 8: Schedule reminders for today's events
For each timed event >20 min away — applying the [Event Filter Rules](#event-filter-rules-shared-reference) and additionally excluding all-day events — use the resolved scheduler tool to schedule a once-off reminder at start−15 min (local ISO timestamp, **no Z suffix**). Pass event title and the calculated timestamp. If no scheduling tool is available, or a reminder fails for a specific event, skip that event and continue with the rest.

## Step 9: Save state
Write to `/workspace/group/calendar-state.json`:
```json
{ "date": "...", "fetched_at": "...", "events": [{"event_id": "...", "title": "...", "start": "...", "reminder_task_id": "..."}] }
```
