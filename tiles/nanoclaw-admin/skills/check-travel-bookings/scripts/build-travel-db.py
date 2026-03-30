#!/usr/bin/env python3
"""
Build a per-trip, per-day travel database from travel-schedule.json.

Reads the flat event list produced by refresh-travel-schedule.py and
organises it into trips with all their items indexed by start date.
ALL item types are stored; alert logic lives in the consumers.

Output: /workspace/group/travel-db.json
Schema:
  {
    "generated_at": "...",
    "trips": {
      "<slug>": {
        "summary": "...",
        "start":   "YYYY-MM-DD",
        "end":     "YYYY-MM-DD",
        "days": {
          "YYYY-MM-DD": [
            {"type": "Flight|Lodging|Rail|Car Rental|...",
             "summary": "...",
             "start": "YYYY-MM-DD",
             "end":   "YYYY-MM-DD",
             "uid":   "..."}
          ]
        }
      }
    }
  }
"""

import json
import re
import sys
from datetime import date, datetime

SCHEDULE_PATH = '/workspace/group/travel-schedule.json'
DB_PATH = '/workspace/group/travel-db.json'


def make_slug(summary: str, start_str: str) -> str:
    start = date.fromisoformat(start_str)
    clean = re.sub(r'\s+\d{4}$', '', summary.strip())
    slug_base = re.sub(r'[^a-z0-9]+', '-', clean.lower()).strip('-')
    return f"{slug_base}-{start.year}-{start.month:02d}"


def main():
    try:
        with open(SCHEDULE_PATH) as f:
            events = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {SCHEDULE_PATH} not found — run refresh-travel-schedule.py first",
              file=sys.stderr)
        sys.exit(1)

    today = date.today()

    trips_raw = [e for e in events if 'item-' not in e.get('uid', '')]
    items_raw = [e for e in events if 'item-' in e.get('uid', '')]

    db_trips = {}
    for trip in trips_raw:
        trip_end = date.fromisoformat(trip['end'])
        if trip_end < today:
            continue

        trip_start = date.fromisoformat(trip['start'])
        slug = make_slug(trip['summary'], trip['start'])

        # Items that overlap with this trip's date range
        days: dict[str, list] = {}
        for item in items_raw:
            item_start = date.fromisoformat(item['start'])
            item_end = date.fromisoformat(item['end'])
            # Overlap check: item starts before trip ends AND item ends on/after trip starts
            if item_start <= trip_end and item_end >= trip_start:
                day_key = item['start']
                days.setdefault(day_key, []).append({
                    'type':    item['type'],
                    'summary': item['summary'],
                    'start':   item['start'],
                    'end':     item['end'],
                    'uid':     item['uid'],
                })

        # Sort each day's events by type for readability
        TYPE_ORDER = {'Flight': 0, 'Rail': 1, 'Lodging': 2, 'Car Rental': 3}
        for day_events in days.values():
            day_events.sort(key=lambda e: (TYPE_ORDER.get(e['type'], 9), e['summary']))

        db_trips[slug] = {
            'summary': trip['summary'],
            'start':   trip['start'],
            'end':     trip['end'],
            'days':    dict(sorted(days.items())),   # sorted by date
        }

    db = {
        'generated_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'trips': db_trips,
    }

    with open(DB_PATH, 'w') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    total_items = sum(
        len(evts) for t in db_trips.values() for evts in t['days'].values()
    )
    print(f"travel-db.json: {len(db_trips)} trips, {total_items} item-events written")
    for slug, t in sorted(db_trips.items(), key=lambda x: x[1]['start']):
        item_count = sum(len(v) for v in t['days'].values())
        type_counts: dict[str, int] = {}
        for evts in t['days'].values():
            for ev in evts:
                type_counts[ev['type']] = type_counts.get(ev['type'], 0) + 1
        breakdown = ', '.join(f"{v}×{k}" for k, v in sorted(type_counts.items()))
        print(f"  {t['start']}–{t['end']}  {t['summary']}")
        print(f"    {breakdown or '(no items)'}")


if __name__ == '__main__':
    main()
