---
name: scheduler-timezone
description: Timezone conversion protocol for the NanoClaw scheduler. Converts event times from any time zone (including UTC or local time) to the server's local timezone, adjusts schedule_value fields to match, and handles cross-timezone reminder scheduling with configurable offsets. Use when scheduling reminders or tasks from events in other timezones, when converting time between UTC and local time, or when a different timezone, time zone abbreviation, or local time needs to be mapped to the NanoClaw scheduler's schedule_value format.
---

# Scheduler Timezone Conversion

The NanoClaw task scheduler interprets `schedule_value` in the **server timezone** (the TZ env var on the orchestrator — read from `/workspace/group/task-tz-state.json` field `scheduler_tz`, or default to the `.env` TZ value).

When scheduling from events in other timezones, you MUST convert to the scheduler timezone.

## Configuration

| Setting | Where | Purpose |
|---------|-------|---------|
| Scheduler timezone | `task-tz-state.json` → `scheduler_tz`, or `.env` TZ | What the scheduler interprets `schedule_value` as — **use for `schedule_value`** |
| Current timezone | `task-tz-state.json` → `current_tz` | Where the owner currently is (set by [task-tz-sync](#)) — **use for displaying times and computing event windows** |
| Home timezone | `task-tz-state.json` → `home_tz` | Default when not traveling |

> `current_tz` and `scheduler_tz` are usually different when traveling. If `task-tz-state.json` is missing, default both to the TZ env var. If the file exists but contains an invalid or unrecognised timezone string, raise an error immediately — do not silently fall back, as a wrong timezone produces a quietly broken schedule.

## The protocol

1. Get the event time (may be local timezone or UTC)
2. Convert to UTC
3. Apply any offset (e.g., subtract 15 minutes for a reminder)
4. Convert UTC → scheduler timezone
5. Format as `"YYYY-MM-DDTHH:MM:SS"` — **no Z suffix**, **no timezone offset**
6. Validate: confirm the scheduled time is in the future and falls within a reasonable window of the event time

## Code template

```python
from datetime import datetime, timezone, timedelta
import zoneinfo, json, os, sys

# Read scheduler timezone from state, with full error handling
def load_scheduler_tz():
    try:
        state = json.load(open('/workspace/group/task-tz-state.json'))
        tz_name = state.get('scheduler_tz')
        if not tz_name:
            raise ValueError("scheduler_tz field missing or empty")
        return zoneinfo.ZoneInfo(tz_name)   # raises ZoneInfoNotFoundError if invalid
    except FileNotFoundError:
        # State file absent — fall back to TZ env var
        tz_name = os.environ.get('TZ', 'UTC')
        print(f"Warning: task-tz-state.json not found, defaulting to TZ={tz_name}", file=sys.stderr)
        return zoneinfo.ZoneInfo(tz_name)
    except (json.JSONDecodeError, KeyError, zoneinfo.ZoneInfoNotFoundError) as e:
        # Malformed file or unrecognised timezone string — abort rather than silently misfire
        raise RuntimeError(f"Cannot determine scheduler timezone: {e}") from e

sched_tz = load_scheduler_tz()

# 1-2. Parse event time to UTC
event_utc = datetime(2026, 4, 1, 7, 50, tzinfo=timezone.utc)

# 3. Apply offset (e.g., -15 min for reminder)
target_utc = event_utc - timedelta(minutes=15)

# 4-5. Convert to scheduler tz, format without Z
target_local = target_utc.astimezone(sched_tz)
schedule_value = target_local.strftime("%Y-%m-%dT%H:%M:%S")

# 6. Validate: scheduled time must be in the future and close to the event
# Without conversion, passing UTC directly to a mismatched-tz scheduler causes the reminder to fire hours late.
now_utc = datetime.now(timezone.utc)
assert target_utc > now_utc, f"Scheduled time {schedule_value} is in the past"
delta = abs((target_utc - event_utc).total_seconds())
assert delta <= 86400, f"Scheduled time is {delta}s away from event — check offset logic"

print(schedule_value)
```

Always use Python `zoneinfo` for conversion. Never compute offsets by hand.
