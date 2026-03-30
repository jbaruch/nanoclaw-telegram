---
name: task-tz-sync
description: Detects Baruch's current timezone from travel schedule and reschedules follow-me tasks (morning-brief, nightly-housekeeping) to fire at the same local time in the new timezone. Use when Baruch is traveling, a timezone change is detected, or schedule adjustment is needed due to a new location. Invoke from heartbeat or nightly-housekeeping.
---

**Note: This skill should be invoked from heartbeat (Step 0.5, before Step 1) or from nightly-housekeeping as an additional step.**

## Step 1: Determine current timezone

Read `/workspace/group/travel-schedule.json`.

Find the active trip segment:
- Look for entries with `"type": "Trip"` where today's date (YYYY-MM-DD) falls within `start` (inclusive) and `end` (inclusive).
- If multiple Trip entries overlap today, prefer the one whose `start` is closest to today.
- Map the trip's `location` field to an IANA timezone:

| Location keywords | IANA timezone |
|---|---|
| Amsterdam / Netherlands | `Europe/Amsterdam` |
| Austin TX / Texas / US Central | `America/Chicago` |
| Nashville / BNA | `America/Chicago` |
| London / United Kingdom | `Europe/London` |
| Edinburgh / Scotland | `Europe/London` |
| Cologne / Germany / Frankfurt / Munich | `Europe/Berlin` |
| Krakow / Poland | `Europe/Warsaw` |
| Coimbra / Portugal / Lisbon | `Europe/Lisbon` |
| Copenhagen / Denmark | `Europe/Copenhagen` |
| Stockholm / Sweden | `Europe/Stockholm` |
| Tel Aviv / Israel | `Asia/Jerusalem` |
| Chania / Greece | `Europe/Athens` |
| New York / JFK / LGA | `America/New_York` |
| Atlanta / ATL | `America/New_York` |
| Any other US city | infer from location name |

- If no active Trip entry found → use `America/Chicago` (Baruch's home timezone).

Store this as `new_tz`.

## Step 2: Compare with stored timezone

Read `/workspace/group/task-tz-state.json`.

If `current_tz` equals `new_tz` → no change. Exit completely silently (wrap output in `<internal>`).

If `current_tz` differs from `new_tz` → proceed to Step 3.

## Step 3: Reschedule follow-me tasks

For each entry in `follow_me_tasks` array from `task-tz-state.json`:

You have `local_hour` and `local_minute` — the desired local time in any timezone.

Calculate the UTC equivalent in the new timezone using standard UTC offset rules, accounting for DST. US and EU DST transitions occur on different dates — use the specific timezone's actual rules:

| Timezone | Standard offset | DST offset | DST start | DST end |
|---|---|---|---|---|
| `America/Chicago` | UTC−6 (CST) | UTC−5 (CDT) | 2nd Sun Mar | 1st Sun Nov |
| `America/New_York` | UTC−5 (EST) | UTC−4 (EDT) | 2nd Sun Mar | 1st Sun Nov |
| `Europe/London` | UTC+0 (GMT) | UTC+1 (BST) | Last Sun Mar | Last Sun Oct |
| `Europe/Amsterdam` | UTC+1 (CET) | UTC+2 (CEST) | Last Sun Mar | Last Sun Oct |
| `Europe/Berlin` / `Europe/Warsaw` | UTC+1 (CET) | UTC+2 (CEST) | Last Sun Mar | Last Sun Oct |
| `Europe/Lisbon` | UTC+0 (WET) | UTC+1 (WEST) | Last Sun Mar | Last Sun Oct |
| `Europe/Copenhagen` / `Europe/Stockholm` | UTC+1 (CET) | UTC+2 (CEST) | Last Sun Mar | Last Sun Oct |
| `Europe/Athens` | UTC+2 (EET) | UTC+3 (EEST) | Last Sun Mar | Last Sun Oct |
| `Asia/Jerusalem` | UTC+2 (IST) | UTC+3 (IDT) | — | — |

Example calculations:
- morning-brief at 7am, moving to `Europe/Amsterdam` (UTC+2 in summer): 7am − 2h = 5am UTC → cron `0 5 * * *`
- nightly-housekeeping at 3am, `Europe/Amsterdam` (UTC+2): 3am − 2h = 1am UTC → cron `0 1 * * *`
- If UTC hour goes negative (e.g. 3am CDT = UTC−5): 3 + 5 = 8am UTC → cron `0 8 * * *`

Call `mcp__nanoclaw__update_task` with:
- `task_id`: the task's ID
- `schedule_type`: "cron"
- `schedule_value`: the new cron string (e.g. `"0 5 * * *"`)

## Step 4: Update state file

Write `/workspace/group/task-tz-state.json` with:
- `current_tz`: set to `new_tz`
- `last_checked`: current UTC timestamp in ISO 8601 format
- `follow_me_tasks`: keep the same array (do not modify task IDs or local times)

## Step 5: Report change

Send a proactive message to Baruch via `mcp__nanoclaw__send_message` (no `reply_to` — this is a proactive notification):

Format (Telegram HTML):
```
📍 Timezone changed: <b>OLD_TZ → NEW_TZ</b>

Rescheduled follow-me tasks:
• <code>morning-brief</code>: 7am local = HH:MM UTC → cron <code>0 HH * * *</code>
• <code>nightly-housekeeping</code>: 3am local = HH:MM UTC → cron <code>0 HH * * *</code>
```

Replace OLD_TZ, NEW_TZ, and HH:MM with actual values. Use plain timezone names (e.g. "America/Chicago", "Europe/Amsterdam").

## Step 6: New task classification (advisory)

When creating a new recurring scheduled task that represents a personal rhythm (morning routines, nightly maintenance, daily check-ins), add it to the `follow_me_tasks` array in `/workspace/group/task-tz-state.json` with:
```json
{
  "task_id": "task-XXXX",
  "name": "task-name",
  "local_hour": H,
  "local_minute": M
}
```

Tasks tied to external deadlines or other people's timezones should NOT be added to `follow_me_tasks`.
