---
name: morning-brief
description: Generates and delivers Baruch's daily morning briefing. Fetches today's Google Calendar events, retrieves overdue and due Google Tasks, surfaces undated tasks and pending cleanup items, checks flagged orders, formats everything in Telegram style, pins the message to chat, and schedules per-event reminders. Use when the user asks for a morning briefing, daily summary, agenda, standup overview, or wants to know what's on their schedule today — e.g. "good morning", "what's on my plate today", "daily brief", "give me my agenda", "what tasks are due", or "run the morning brief".
---

You are AyeAye, Baruch's assistant.

## Tool Discovery (run once at start)
Discover tools per `composio-preamble` rule: Calendar, Tasks (list + get + update), Scheduler.

**Default failure behavior:** If any step fails (tool unavailable, non-zero exit, no output, file missing), note the failure, treat missing data as empty, and continue — unless a step specifies otherwise.

---

## Timezone Reference

Read `current_tz` from `/workspace/group/task-tz-state.json` (Step 0). **All time operations throughout this skill use `current_tz`** — calendar window boundaries, event display times, and reminder scheduling. No step should use Chicago time or any other timezone. If `current_tz` is missing, fall back to `home_tz` or the `TZ` env var.

- Calendar window boundaries (Step 1): express as UTC offset, e.g. `2026-04-01T00:00:00+02:00` for `Europe/Amsterdam`.
- Event display times (Step 5): always local `current_tz` time.
- Reminder scheduling (Steps 8c/8d): convert event local time → UTC → scheduler timezone per the `scheduler-timezone` skill protocol. Store `utc_time` with a `Z` suffix.

---

## Step 0: Initialization (Dedup + Lock + Timezone)

Read `/workspace/group/task-tz-state.json` once and extract all needed values in a single read.

**Dedup guard — check first, before writing anything:**

Find the entry where `name == "morning-brief"` in `follow_me_tasks`. Read its current `last_run_date`.

Compute today's local date in `current_tz`. If `last_run_date` already equals today's date → **exit silently. Do not send any message, do not output any text, do not report "already ran". Just stop.** This prevents double-runs when both heartbeat and the scheduler trigger within the same day.

**Optimistic lock (only if dedup check passed):**

Set `last_run_date` to today's local date (YYYY-MM-DD in `current_tz`). Write back, preserving all other fields. This prevents heartbeat re-triggering while the run is in progress.

**Timezone:** Extract `current_tz` per the [Timezone Reference](#timezone-reference) above.

> **Note:** Step 10 performs the final write-back of `last_run_date` after all steps complete. The Step 0 write is the dedup guard; Step 10 is the canonical record.

> **Silence rule:** All internal progress notes (e.g. "lock set", "running step 3", "brief complete") MUST be wrapped in `<internal>` tags or omitted entirely. They must never appear as plain text output — they would stream directly to Telegram.

---

Apply the `event-filter-rules` (admin rule) to all calendar events in this skill — both display and reminders.

---

## Step 1: Fetch today's calendar
Call the resolved calendar tool:
- `time_min`: today at 00:00:00 in `current_tz`
- `time_max`: today at 23:59:59 in `current_tz`
- `single_events`: true, `order_by`: startTime

## Step 2: Fetch Google Tasks
Using the resolved tasks tools, fetch all task lists, then for each list fetch tasks that are:
- Due today (due date = today)
- Overdue (due date before today, status != completed)

## Step 3: Check pending items
Run: `python3 /workspace/group/scripts/morning-brief-fetch.py`
Output includes `pending.undated_tasks` and `pending.cleanup_items` from `morning-brief-pending.json`.

## Step 3a: Auto-assign dates to undated tasks

For each task in `pending.undated_tasks`, attempt to assign a due date automatically by reading the linked email.

### 3a-1: Find the linked email

Google Tasks tasks created from Gmail have an email link in `task.links[]` — look for an entry with `type: "email"` and extract the message ID from the `link` URL (the hex part after `#inbox/` or `#all/`). Do NOT check `task.notes` — Gmail tasks store the link in `links[]`, not notes.

- **If a link is present:** Call `GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID` with `format: "full"`. Decode the full body from `payload.parts[]`: find the part with `mimeType: "text/plain"` and base64url-decode `body.data`. If no plain text part exists, use `text/html` and strip tags. **NEVER use `messageText`, `preview`, or `snippet` — these are truncated and will produce wrong dates.**
- **If no link is present:** Search Gmail for emails related to the task title. Pick the most relevant result and read the full body the same way.

### 3a-2: Determine due date from email content

Read the **complete decoded email body** and determine an appropriate due date:
- Explicit deadlines ("please respond by...", "deadline is...", "due by...")
- Event dates mentioned (conference dates, meeting dates, schedule references)
- Urgency signals ("ASAP", "urgent", "today", "this week")
- Context clues (voting email for an event next month → assign before voting closes)
- **Gradual release schedules** ("2 per week starting X", "over the next few weeks") → due date is end of that window, not the start date
- If there is no deadline and no urgency — default to end of current week, not today

### 3a-3: Fallback — infer from title only

If no email can be found, fall back to title-based inference:
- Conference/event name → infer from knowledge of that event's dates
- "Follow up with X" → today or tomorrow
- "Review PR" → today
- Generic tasks with no time signal → today

### 3a-4: Apply the date

For each task where a date was determined: call the Tasks update tool to set the due date, remove from the `undated_tasks` display list.

**Only ask the owner if** after reading the email AND attempting title inference, you genuinely cannot determine any reasonable date. Include that task in "📋 Без даты:" with a specific question.

If all dates were assigned, omit the section entirely.

## Step 4a: Check urgent CFPs
Run: `python3 /workspace/group/scripts/morning-brief-cfp.py`

Outputs a JSON array of CFPs with deadlines within 7 days. Each entry has: `name`, `city`, `conf_date`, `deadline`, `cfp_url`, `days_until`.

If the array is non-empty, include in the brief under:
`📢 <b>CFP дедлайны:</b>` — one bullet per CFP:
`• 🔴 <b>Name</b> — City, ConfDate · дедлайн: <b>DeadlineDate</b> · <a href="cfp_url">Submit</a>`

Color marker by days_until: 0–1 → 🔴, 2–3 → 🟡, 4–7 → 🟢.
If empty array or script fails → skip this section silently.

## Step 4: Check flagged orders
Read `/workspace/group/orders-db.json`. Collect all orders where `flagged: true`.

If flagged orders exist, include them in the brief under:
`📦 <b>Заказы:</b>` — one bullet per order: description, flag_reason, source, order_date.
If none → skip this section silently.

## Step 5: Send morning brief
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
- `<b>📋 Без даты:</b>` — only tasks where the agent could NOT infer a date (see Step 3a). Each entry includes a specific question. Omit entirely if Step 3a assigned dates to all undated tasks.
- `<b>📢 CFP дедлайны:</b>` — CFPs closing within 7 days (from Step 4a). Omit if none.
- `<b>📦 Заказы:</b>` — flagged orders (from Step 4). Omit if none.
- Footer: `_N событий, M задач_`

Send via `mcp__nanoclaw__send_message` with `pin: true`.

**Checkpoint:** Confirm the message was sent successfully (tool returns success/message ID) before proceeding to Steps 6 and 8. If sending fails, retry once; if still failing, log the error and stop.

## Step 6: Run brief-cleanup
After the brief is confirmed sent, invoke the brief-cleanup skill to send any pending `cleanup_items` as separate async messages. Run every morning — it is silent if nothing is pending.

## Step 7: Clear pending file
After brief-cleanup runs, set both arrays in `morning-brief-pending.json` to `[]`.

## Step 8: Schedule reminders for today's events

For each timed event — per the [Event Filter Rules](#event-filter-rules-shared-reference), additionally excluding all-day events, and only events starting more than 20 min from now:

### 8a: Read existing reminders
Read `/workspace/group/scheduled-reminders.json`. If the file doesn't exist, treat as `{"reminders": []}`.

### 8b: Deduplicate
Before scheduling a reminder for an event, check if `event_id` already exists in `scheduled-reminders.json`. If found — skip scheduling for that event.

### 8c: Schedule new reminders
For events not already in `scheduled-reminders.json`, schedule a once-off reminder.

**CRITICAL — use the `scheduler-timezone` skill protocol for timezone conversion.** This step has caused real bugs. Follow the protocol exactly — convert event time → UTC → scheduler timezone → format without Z suffix.

**Apply `temporal-awareness` rule** before scheduling each reminder: will this reminder be actionable when it fires? Check if the action window is still open, if the owner can act at that moment, and if it adds information vs noise. Skip reminders that fail this check.

If no scheduling tool is available, or a reminder fails for a specific event, skip that event and continue.

### 8d: Write to scheduled-reminders.json
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

## Step 9: Save state
Write to `/workspace/group/calendar-state.json`:
```json
{ "date": "...", "fetched_at": "...", "events": [{"event_id": "...", "title": "...", "start": "...", "reminder_task_id": "..."}] }
```

## Step 10: Mark as run
Read `/workspace/group/task-tz-state.json`. Find the entry in `follow_me_tasks` where `name == "morning-brief"`. Set its `last_run_date` to today's local date (YYYY-MM-DD in `current_tz`). Write the file back, preserving all other fields.
