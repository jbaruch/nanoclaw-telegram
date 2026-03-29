---
name: nightly-housekeeping
description: Nightly maintenance tasks — run TripIt→Reclaim sync, refresh TripIt travel schedule, and clean up stale state. Use when running nightly maintenance, scheduled tasks, overnight jobs, or cron jobs; or when the user mentions TripIt-Reclaim synchronization, calendar sync, or refreshing the travel schedule.
---

# Nightly Housekeeping

Run silently at 3am. No output unless something fails.

## Step 1: TripIt→Reclaim sync

Invoke the `sync-tripit` skill to sync timezone changes from TripIt to Reclaim:

```
Skill(skill: "sync-tripit")
```

- If result is `noChanges: true` and no errors: stay silent
- If changes detected: include summary in output (new timezones, OOO blocks)
- If skill fails: alert with error message

## Step 2: Refresh TripIt travel schedule

Write the script below to `/workspace/group/scripts/refresh-travel-schedule.py`, then execute it:

```bash
python /workspace/group/scripts/refresh-travel-schedule.py
```

**Script contents** (`/workspace/group/scripts/refresh-travel-schedule.py`):

```python
import urllib.request, json, re, time
from datetime import datetime, timezone

url = open('/workspace/group/tripit-url.txt').read().strip()

# Fetch with one retry on transient failure
ics = None
last_error = None
for attempt in range(2):
    try:
        ics = urllib.request.urlopen(url).read().decode('utf-8')
        break
    except Exception as e:
        last_error = e
        if attempt == 0:
            time.sleep(5)

if ics is None:
    raise RuntimeError(f'ICS fetch failed after 2 attempts: {last_error}')

now = datetime.now(timezone.utc)
events = []

for component in ics.split('BEGIN:VEVENT')[1:]:
    def get(field):
        m = re.search(rf'{field}[^:]*:(.+)', component)
        return m.group(1).strip() if m else ''

    def parse_dt(s):
        try:
            return datetime.strptime(s.split('T')[0], '%Y%m%d').replace(tzinfo=timezone.utc)
        except:
            return None

    start = parse_dt(get('DTSTART'))
    end = parse_dt(get('DTEND'))
    if not start or not end or end < now:
        continue

    events.append({
        'summary': get('SUMMARY'),
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'location': get('LOCATION')
    })

events.sort(key=lambda e: e['start'])

output_path = '/workspace/group/travel-schedule.json'
with open(output_path, 'w') as f:
    json.dump(events, f, indent=2)

# Validate written file
with open(output_path) as f:
    verified = json.load(f)
assert isinstance(verified, list), 'travel-schedule.json is not a JSON array'
assert all('summary' in e and 'start' in e and 'end' in e for e in verified), \
    'travel-schedule.json contains events missing required fields'

print(f'TripIt: {len(events)} upcoming events written and verified')
```

**Alert if:** fetch fails after retry (likely URL expired), produces 0 events, or post-write validation fails.

## Output

Silent on success. Alert only on failure or when sync reports changes:

```
*Nightly housekeeping* ⚠️

• TripIt sync: [changes or error]
• TripIt refresh failed: [error]
```
