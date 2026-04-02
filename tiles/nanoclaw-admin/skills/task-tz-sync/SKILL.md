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

**Algorithm for determining timezone from Flighty events** — get current UTC time, sort events by start time, then apply the first matching rule:

| Condition | Use timezone of… |
|---|---|
| start ≤ now ≤ end (in-flight) | Arrival airport of current flight |
| now < first flight's start (pre-departure) | Departure airport of first flight |
| now > last flight's end (post-arrival) | Arrival airport of last flight |
| now between completed flight's end and next flight's start (layover) | Arrival airport of completed flight |

If the event's `end.timeZone` field is provided, prefer it over the mapping table for the arrival timezone.

### Fallback: travel-schedule.json

If no Flighty events are found for today, fall back to `/workspace/group/travel-schedule.json`:

- Look for entries with `"type": "Trip"` where today (YYYY-MM-DD) falls within `start`–`end` (inclusive).
- If multiple Trip entries overlap today, prefer the one whose `start` is closest to today.
- Map the trip's `location` field to an IANA timezone using the reference table at `/workspace/group/tz-mappings.md`.
- If no active Trip entry found → use `home_tz` from `task-tz-state.json` (the owner's home timezone).

Store the determined timezone as `new_tz`.

### Airport/Location → IANA Timezone Mapping

See `/workspace/group/tz-mappings.md` for the full canonical reference. Key entries for quick lookup:

| Airport code(s) | Location keywords | IANA timezone |
|---|---|---|
| AMS | Amsterdam / Netherlands | `Europe/Amsterdam` |
| ATL, DTW, BOS, JFK, LGA, EWR | New York / JFK / LGA / Atlanta / ATL | `America/New_York` |
| BNA, AUS, MSP | Austin TX / Texas / US Central / Nashville / BNA | `America/Chicago` |
| LHR, LGW, STN, EDI | London / United Kingdom / Edinburgh / Scotland | `Europe/London` |
| FRA, MUC | Cologne / Germany / Frankfurt / Munich | `Europe/Berlin` |
| CDG | Paris | `Europe/Paris` |
| KRK | Krakow / Poland | `Europe/Warsaw` |
| CPH | Copenhagen / Denmark | `Europe/Copenhagen` |
| ARN | Stockholm / Sweden | `Europe/Stockholm` |
| OPO, LIS | Coimbra / Portugal / Lisbon | `Europe/Lisbon` |
| YYZ | Toronto / Canada | `America/Toronto` |
| TLV | Tel Aviv / Israel | `Asia/Jerusalem` |
| ATH, CHQ | Chania / Greece | `Europe/Athens` |
| — | Any other US city | infer from location name |

## Step 2: Compare with stored timezone

Read `/workspace/group/task-tz-state.json`.

- `current_tz` equals `new_tz` → no change. Exit completely silently (wrap output in `<internal>`).
- `current_tz` differs from `new_tz` → proceed to Step 3.

## Step 3: Reschedule follow-me tasks

For each entry in `follow_me_tasks` from `task-tz-state.json`, use `local_hour` and `local_minute` to calculate the correct cron value.

**CRITICAL: Use the `scheduler-timezone` skill protocol.** The cron hour must be expressed in the scheduler timezone (from `scheduler_tz` in `task-tz-state.json`) — NOT UTC, NOT the travel timezone.

Two-step conversion:
1. Convert local time in `new_tz` → UTC
2. Convert UTC → scheduler timezone. Handle hour wraparound (if negative, add 24).

Use Python `zoneinfo` — never compute offsets by hand.

Call `mcp__nanoclaw__update_task` with:
- `task_id`: the task's ID
- `schedule_type`: `"cron"`
- `schedule_value`: the cron string using scheduler timezone hour

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

---

**Advisory — adding new follow-me tasks:** When creating a new recurring scheduled task representing a personal rhythm (morning routines, nightly maintenance, daily check-ins), add it to `follow_me_tasks` in `/workspace/group/task-tz-state.json`:
```json
{
  "task_id": "task-XXXX",
  "name": "task-name",
  "local_hour": H,
  "local_minute": M
}
```
Tasks tied to external deadlines or other people's timezones should **not** be added to `follow_me_tasks`.
