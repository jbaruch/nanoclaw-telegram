---
name: task-tz-sync
description: Detects Baruch's current timezone from travel schedule and reschedules follow-me tasks (morning-brief, nightly-housekeeping) to fire at the same local time in the new timezone. Use when Baruch is traveling, a timezone change is detected, or schedule adjustment is needed due to a new location. Invoke from heartbeat or nightly-housekeeping.
---

**Note: This skill should be invoked from heartbeat (Step 0.5, before Step 1) or from nightly-housekeeping as an additional step.**

## Step 1: Determine current timezone

### Primary: Flighty Google Calendar

Query Google Calendar using `GOOGLECALENDAR_EVENTS_LIST` with:
- `calendar_id`: `c_d158e24aa6c8dcd13b4657344fa2246cacd9d9a248aa7979fa6c1858070dc18d@group.calendar.google.com`
- `time_min`: start of today in UTC (e.g. `2025-06-10T00:00:00Z`)
- `time_max`: end of today in UTC (e.g. `2025-06-10T23:59:59Z`)
- `single_events`: true

Flighty events are flight segments with titles like `"DL73 AMS → ATL"` and have precise `dateTime` start/end with timezone info.

**Extract airport codes** from each event title (format: `FLIGHT# AAA → BBB`). Parse departure airport (AAA) and arrival airport (BBB).

**Airport → IANA timezone mapping:**

| Airport code(s) | IANA timezone |
|---|---|
| AMS | `Europe/Amsterdam` |
| ATL, DTW, BOS, JFK, LGA, EWR | `America/New_York` |
| BNA, AUS, MSP | `America/Chicago` |
| LHR, LGW, STN, EDI | `Europe/London` |
| FRA, MUC | `Europe/Berlin` |
| CDG | `Europe/Paris` |
| KRK | `Europe/Warsaw` |
| CPH | `Europe/Copenhagen` |
| ARN | `Europe/Stockholm` |
| OPO, LIS | `Europe/Lisbon` |
| YYZ | `America/Toronto` |
| TLV | `Asia/Jerusalem` |
| ATH, CHQ | `Europe/Athens` |

**Algorithm for determining timezone from Flighty events** (get current UTC time, sort events by start time):

1. **In-flight:** start ≤ now ≤ end → use **arrival airport's** timezone.
2. **Pre-departure:** now < first flight's start → use **departure airport's** timezone of the first flight.
3. **Post-arrival:** now > last flight's end → use **arrival airport's** timezone of the last flight.
4. **Layover:** now is between a completed flight's end and the next flight's start → use **arrival airport's** timezone of the completed flight.

If the event's `end.timeZone` field is provided, prefer it over the mapping table for arrival timezone.

### Fallback: travel-schedule.json

If no Flighty events are found for today, fall back to `/workspace/group/travel-schedule.json`:

- Look for entries with `"type": "Trip"` where today (YYYY-MM-DD) falls within `start`–`end` (inclusive).
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

Store the determined timezone as `new_tz`.

## Step 2: Compare with stored timezone

Read `/workspace/group/task-tz-state.json`.

- `current_tz` equals `new_tz` → no change. Exit completely silently (wrap output in `<internal>`).
- `current_tz` differs from `new_tz` → proceed to Step 3.

## Step 3: Reschedule follow-me tasks

For each entry in `follow_me_tasks` from `task-tz-state.json`, use `local_hour` and `local_minute` to calculate the correct cron value.

**CRITICAL: The task scheduler uses America/Chicago local time.** The cron hour must be expressed in America/Chicago — NOT UTC.

Two-step conversion:
1. Convert local time in `new_tz` → UTC: `utc_hour = local_hour − tz_offset`
2. Convert UTC → America/Chicago: `chi_hour = utc_hour + chi_offset` where `chi_offset = −5` (CDT, Mar–Nov) or `−6` (CST, Nov–Mar). Handle wraparound: if `chi_hour < 0`, add 24.

Example calculations (April = CDT = UTC−5):
- 7am `Europe/Amsterdam` (UTC+2): 7 − 2 = 5am UTC → 5 − 5 = **0 (midnight CDT)** → `0 0 * * *`
- 3am `Europe/Amsterdam` (UTC+2): 3 − 2 = 1am UTC → 1 − 5 = −4 → +24 = **20 (8pm CDT)** → `0 20 * * *`
- 7am `America/Chicago` CDT (UTC−5): already local → **7am CDT** → `0 7 * * *`
- 3am `America/Chicago` CDT (UTC−5): already local → **3am CDT** → `0 3 * * *`

Call `mcp__nanoclaw__update_task` with:
- `task_id`: the task's ID
- `schedule_type`: `"cron"`
- `schedule_value`: the cron string using America/Chicago hour (e.g. `"0 0 * * *"`)

## Step 4: Update state file

Write `/workspace/group/task-tz-state.json` with:
- `current_tz`: set to `new_tz`
- `last_checked`: current UTC timestamp (ISO 8601)
- `follow_me_tasks`: keep the same array unchanged (do not modify task IDs, local times, or `last_run_date` fields)

## Step 5: Report change

Send a proactive message via `mcp__nanoclaw__send_message` (no `reply_to`):

```
📍 Timezone changed: <b>OLD_TZ → NEW_TZ</b>

Rescheduled follow-me tasks:
• <code>morning-brief</code>: 7am local = HH:MM UTC = HH:MM CDT → cron <code>0 HH * * *</code>
• <code>nightly-housekeeping</code>: 3am local = HH:MM UTC = HH:MM CDT → cron <code>0 HH * * *</code>
```

Replace OLD_TZ, NEW_TZ, and HH:MM with actual values. Use plain timezone names (e.g. `America/Chicago`, `Europe/Amsterdam`).

## Step 6: New task classification (advisory)

When creating a new recurring scheduled task representing a personal rhythm (morning routines, nightly maintenance, daily check-ins), add it to `follow_me_tasks` in `/workspace/group/task-tz-state.json`:
```json
{
  "task_id": "task-XXXX",
  "name": "task-name",
  "local_hour": H,
  "local_minute": M
}
```

Tasks tied to external deadlines or other people's timezones should **not** be added to `follow_me_tasks`.
