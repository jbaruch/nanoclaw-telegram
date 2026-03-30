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

# Unfold ICS line continuations (RFC 5545: CRLF + whitespace = continuation)
ics = re.sub(r'\r?\n[ \t]', '', ics)

now = datetime.now(timezone.utc)
events = []

for component in ics.split('BEGIN:VEVENT')[1:]:
    def get(field):
        m = re.search(rf'^{field}[^:\r\n]*:(.+)', component, re.MULTILINE)
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

    uid = get('UID')
    description = get('DESCRIPTION')

    # Determine type from DESCRIPTION [Type] marker or UID
    event_type = 'Unknown'
    if 'item-' not in uid:
        event_type = 'Trip'
    else:
        # Parse [Type] from DESCRIPTION field (appears as e.g. "[Flight] ATL to SJO")
        type_match = re.search(r'\[([^\]]+)\]', description)
        if type_match:
            raw_type = type_match.group(1)
            known_types = ['Flight', 'Lodging', 'Car Rental', 'Rail', 'Ferry', 'Restaurant', 'Cruise']
            event_type = raw_type if raw_type in known_types else raw_type

    events.append({
        'summary': get('SUMMARY'),
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'location': get('LOCATION'),
        'type': event_type,
        'uid': uid
    })

events.sort(key=lambda e: e['start'])

output_path = '/workspace/group/travel-schedule.json'
with open(output_path, 'w') as f:
    json.dump(events, f, indent=2)

# Validate written file
with open(output_path) as f:
    verified = json.load(f)
assert isinstance(verified, list), 'travel-schedule.json is not a JSON array'
assert all('summary' in e and 'start' in e and 'end' in e and 'type' in e for e in verified), \
    'travel-schedule.json contains events missing required fields'

# Print type breakdown for visibility
from collections import Counter
type_counts = Counter(e['type'] for e in verified)
print(f'TripIt: {len(events)} upcoming events written and verified')
print(f'Type breakdown: {dict(type_counts)}')
print()
print('Sample events with types:')
for e in verified[:15]:
    print(f'  [{e["type"]}] {e["summary"]} ({e["start"]})')
