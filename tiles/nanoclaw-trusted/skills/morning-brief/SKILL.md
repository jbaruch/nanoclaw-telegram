---
name: morning-brief
description: Generates and delivers Baruch's daily morning briefing. Fetches today's Google Calendar events, retrieves overdue and due Google Tasks, surfaces undated tasks and pending cleanup items, checks flagged orders, formats everything in Telegram style, pins the message to chat, and schedules per-event reminders. Use when the user asks for a morning briefing, daily summary, agenda, standup overview, or wants to know what's on their schedule today — e.g. "good morning", "what's on my plate today", "daily brief", "give me my agenda", "what tasks are due", or "run the morning brief".
---

**Every step below is mandatory. Execute them in order. Do not skip, reorder, or abbreviate any step.**

You are AyeAye, Baruch's assistant.

## Tool Discovery (run once at start)
Discover tools per `composio-preamble` rule: Calendar, Tasks (list + get + update), Scheduler.

**Default failure behavior:** If any step fails (tool unavailable, non-zero exit, no output, file missing), note the failure, treat missing data as empty, and continue — unless a step specifies otherwise.

---

## Timezone Reference

Read `current_tz` from `/workspace/group/task-tz-state.json` (Step 1). **All time operations use `current_tz`** — calendar window boundaries, event display times, and reminder scheduling. If `current_tz` is missing, fall back to `home_tz` or the `TZ` env var.

- Calendar window boundaries (Step 2): express as UTC offset, e.g. `2026-04-01T00:00:00+02:00`.
- Event display times (Step 8): always local `current_tz` time.
- Reminder scheduling (Steps 14/15): convert event local time → UTC → scheduler timezone per the `scheduler-timezone` skill protocol. Store `utc_time` with a `Z` suffix.

---

## Step 1: Initialization (Dedup + Lock + Timezone)

Read `/workspace/group/task-tz-state.json` once and extract all needed values in a single read.

**Dedup guard — check first, before writing anything:**

Find the entry where `name == "morning-brief"` in `follow_me_tasks`. Read its current `last_run_date`.

Compute today's local date in `current_tz`. If `last_run_date` already equals today's date → **exit silently. Do not send any message, do not output any text, do not report "already ran". Just stop.**

**Optimistic lock (only if dedup check passed):**

Set `last_run_date` to today's local date (YYYY-MM-DD in `current_tz`). Write back, preserving all other fields.

**Timezone:** Extract `current_tz` per the [Timezone Reference](#timezone-reference) above.

> **Silence rule:** All internal progress notes MUST be wrapped in `<internal>` tags or omitted entirely — they must never appear as plain text output, as they would stream directly to Telegram.

---

Apply the `event-filter-rules` (admin rule) to all calendar events in this skill — both display and reminders.

---

## Step 2: Fetch today's calendar
Call the resolved calendar tool:
- `time_min`: today at 00:00:00 in `current_tz`
- `time_max`: today at 23:59:59 in `current_tz`
- `single_events`: true, `order_by`: startTime

## Step 3: Fetch Google Tasks
Using the resolved tasks tools, fetch all task lists, then for each list fetch tasks that are:
- Due today (due date = today)
- Overdue (due date before today, status != completed)

## Step 4: Check pending items
Run: `python3 /workspace/group/scripts/morning-brief-fetch.py`
Output includes `pending.undated_tasks` and `pending.cleanup_items` from `morning-brief-pending.json`.

## Step 5: Auto-assign dates to undated tasks

Follow the `undated-task-date-assignment` skill for the full sub-workflow (reading linked Gmail, inferring due dates, calling the Tasks update tool). Summary:

1. For each task in `pending.undated_tasks`, find its linked email via `task.links[]` (type `"email"`) and read the **full decoded body** — never use `snippet` or `preview`.
2. Infer a due date from explicit deadlines, event dates, urgency signals, or title inference as a fallback.
3. Call the Tasks update tool to set the date and remove the task from the undated display list.
4. **Only ask the owner** if — after email reading and title inference — no reasonable date can be determined. Include those tasks in `📋 Без даты:` with a specific question.

If all dates were assigned, omit the `📋 Без даты:` section entirely.

## Step 6: Check urgent CFPs
Run: `python3 /workspace/group/scripts/morning-brief-cfp.py`

Outputs a JSON array of CFPs with deadlines within 7 days. Each entry has: `name`, `city`, `conf_date`, `deadline`, `cfp_url`, `days_until`.

If the array is non-empty, include in the brief under:
`📢 <b>CFP дедлайны:</b>` — one bullet per CFP:
`• 🔴 <b>Name</b> — City, ConfDate · дедлайн: <b>DeadlineDate</b> · <a href="cfp_url">Submit</a>`

Color marker by days_until: 0–1 → 🔴, 2–3 → 🟡, 4–7 → 🟢.
If empty array or script fails → skip this section silently.

## Step 7: Check flagged orders
Read `/workspace/group/orders-db.json`. Collect all orders where `flagged: true`.

If flagged orders exist, include them in the brief under:
`📦 <b>Заказы:</b>` — one bullet per order: description, flag_reason, source, order_date.
If none → skip this section silently.

## Step 8: Send morning brief
Select events using the [Event Filter Rules](#event-filter-rules-shared-reference).

Format in Telegram HTML. Canonical example:

```
<b>Доброе утро! Понедельник, 9 июня</b>

<b>📅 Сегодня:</b>
• 09:00 — Standup with team
• 14:00 — 1:1 with Alex

<b>✅ Задачи:</b>
• ⚠️ Обновить README (просрочено: 6 июня)
• Ответить на письмо Михаила

<b>📋 Без даты:</b>
• Разобрать инбокс — <i>когда это нужно сделать?</i>

<i>2 события, 3 задачи</i>
```

**Section rules:**
- `<b>📅 Сегодня:</b>` — timed events with local time in `current_tz`.
- `<b>✅ Задачи:</b>` — overdue tasks (original due date, marked ⚠️) + tasks due today. Omit if no tasks.
- `<b>📋 Без даты:</b>` — only tasks where a date could NOT be inferred (see Step 5). Each entry includes a specific question. Omit entirely if Step 5 assigned all dates.
- `<b>📢 CFP дедлайны:</b>` — CFPs closing within 7 days (from Step 6). Omit if none.
- `<b>📦 Заказы:</b>` — flagged orders (from Step 7). Omit if none.
- Footer: `_N событий, M задач_`

Send via `mcp__nanoclaw__send_message` with `pin: true`.

**Checkpoint:** Confirm the message was sent successfully (tool returns success/message ID) before proceeding to Steps 9 and 11. If sending fails, retry once; if still failing, log the error and stop.

## Step 9: Run brief-cleanup
After the brief is confirmed sent, invoke the brief-cleanup skill to send any pending `cleanup_items` as separate async messages. Run every morning — it is silent if nothing is pending.

## Step 10: Clear pending file
After brief-cleanup runs, set both arrays in `morning-brief-pending.json` to `[]`.

## Step 11: Schedule reminders for today's events

For each timed event — per the [Event Filter Rules](#event-filter-rules-shared-reference), additionally excluding all-day events, and only events starting more than 20 min from now:

## Step 12: Read existing reminders
Read `/workspace/group/scheduled-reminders.json`. If the file doesn't exist, treat as `{"reminders": []}`.

## Step 13: Deduplicate
Before scheduling a reminder for an event, check if `event_id` already exists in `scheduled-reminders.json`. If found — skip scheduling for that event.

## Step 14: Schedule new reminders
For events not already in `scheduled-reminders.json`, schedule a once-off reminder.

**TIMEZONE CONVERSION — compute `schedule_value` exactly as follows:**
```python
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

# event.start.dateTime from the Google Calendar API includes timezone offset
# e.g. "2026-04-03T16:00:00-05:00" (4 PM CDT) or "2026-04-03T21:00:00Z"
event_dt = datetime.fromisoformat(event_start_str)   # timezone-aware
event_utc = event_dt.astimezone(ZoneInfo("UTC"))
reminder_utc = event_utc - timedelta(minutes=15)

# Convert to current local timezone for the scheduler (scheduler uses local time)
tz = ZoneInfo(current_tz)   # e.g. "America/Chicago"
reminder_local = reminder_utc.astimezone(tz)
schedule_value = reminder_local.strftime("%Y-%m-%dT%H:%M:%S")   # NO Z suffix
```

Example: event "2026-04-03T16:00:00-05:00" (4 PM CDT = 21:00 UTC) → reminder_utc = 20:45 UTC → CDT = 3:45 PM → `schedule_value = "2026-04-03T15:45:00"`

**Do NOT** strip the Z from a UTC time and pass it directly as `schedule_value` — the scheduler interprets `schedule_value` as local time, so "15:45:00" means 3:45 PM local, not 3:45 PM UTC.

**Apply `temporal-awareness` rule** before scheduling each reminder: will this reminder be actionable when it fires? Skip reminders that fail this check.

If no scheduling tool is available, or a reminder fails for a specific event, skip that event and continue.

## Step 15: Write to scheduled-reminders.json
For each newly scheduled reminder, append an entry to the `reminders` array in `/workspace/group/scheduled-reminders.json`:

```json
{
  "event_id": "<google_calendar_event_id>",
  "title": "<event title>",
  "utc_time": "<event start in UTC, ISO 8601 with Z suffix, e.g. 2026-04-01T14:30:00Z>",
  "reminder_offset_min": 15,
  "task_id": "<task ID returned by schedule_task>"
}
```

Derive `utc_time` from the event's local start time using `current_tz`. Write the updated file back to `/workspace/group/scheduled-reminders.json`.

## Step 16: Save state
Write to `/workspace/group/calendar-state.json`:
```json
{ "date": "...", "fetched_at": "...", "events": [{"event_id": "...", "title": "...", "start": "...", "reminder_task_id": "..."}] }
```

## Step 17: Mark as run
Read `/workspace/group/task-tz-state.json`. Find the entry in `follow_me_tasks` where `name == "morning-brief"`. Set its `last_run_date` to today's local date (YYYY-MM-DD in `current_tz`). Write the file back, preserving all other fields.
