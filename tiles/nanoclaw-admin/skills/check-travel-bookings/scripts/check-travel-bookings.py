#!/usr/bin/env python3
"""
Travel booking gap checker — reads from travel-db.json.

travel-db.json is built nightly by build-travel-db.py (via nightly-housekeeping).
Falls back to fetching TripIt ICS directly if the DB is stale/missing.

Alerts only on Flight + Lodging gaps; all item types are in the DB for future use.
"""

import json
import os
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta


# ---------------------------------------------------------------------------
# ICS parsing (fallback path only)
# ---------------------------------------------------------------------------

def fetch_ics(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode('utf-8', errors='replace')


def unfold_ics(ics_text: str) -> str:
    return re.sub(r'\r?\n[ \t]', '', ics_text)


def parse_ics_date(val: str) -> date:
    val = val.split(';')[-1].split(':')[-1].strip()
    val = val[:8]
    return datetime.strptime(val, '%Y%m%d').date()


def parse_vevents(ics_text: str) -> list[dict]:
    events = []
    blocks = re.split(r'BEGIN:VEVENT', ics_text)
    for block in blocks[1:]:
        end_idx = block.find('END:VEVENT')
        if end_idx != -1:
            block = block[:end_idx]
        event = {}
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            colon_idx = line.find(':')
            if colon_idx == -1:
                continue
            prop_full = line[:colon_idx]
            value = line[colon_idx + 1:]
            prop_name = prop_full.split(';')[0].upper()
            if prop_name == 'UID':
                event['uid'] = value
            elif prop_name == 'SUMMARY':
                event['summary'] = value
            elif prop_name == 'DTSTART':
                try:
                    event['dtstart'] = parse_ics_date(prop_full + ':' + value)
                except Exception:
                    pass
            elif prop_name == 'DTEND':
                try:
                    event['dtend'] = parse_ics_date(prop_full + ':' + value)
                except Exception:
                    pass
            elif prop_name == 'DESCRIPTION':
                event['description'] = value
        if 'uid' in event and 'dtstart' in event:
            events.append(event)
    return events


def extract_item_type(description: str) -> str:
    if not description:
        return 'Unknown'
    m = re.search(r'\[([^\]]+)\]', description)
    return m.group(1) if m else 'Unknown'


# ---------------------------------------------------------------------------
# Core logic (shared between DB and ICS paths)
# ---------------------------------------------------------------------------

def make_slug(summary: str, start: date) -> str:
    clean = re.sub(r'\s+\d{4}$', '', summary.strip())
    slug_base = re.sub(r'[^a-z0-9]+', '-', clean.lower()).strip('-')
    return f"{slug_base}-{start.year}-{start.month:02d}"


def build_lodging_ranges(lodging_items: list[dict]) -> list[tuple]:
    """
    Pair 'Check-in: Hotel' and 'Check-out: Hotel' events by hotel name.
    Returns list of (checkin_date, checkout_date) tuples.
    """
    checkins: dict[str, date] = {}
    checkouts: dict[str, date] = {}
    for item in lodging_items:
        summary = item.get('summary', '')
        dtstart = item.get('dtstart')
        if dtstart is None:
            continue
        if summary.startswith('Check-in:'):
            hotel = summary[len('Check-in:'):].strip()
            checkins[hotel] = dtstart
        elif summary.startswith('Check-out:'):
            hotel = summary[len('Check-out:'):].strip()
            checkouts[hotel] = dtstart
    ranges = []
    for hotel, ci in checkins.items():
        co = checkouts.get(hotel)
        if co and co > ci:
            ranges.append((ci, co))
        else:
            ranges.append((ci, ci + timedelta(days=1)))
    return ranges


def classify_trip(items: list[dict], trip_start: date, trip_end: date) -> dict:
    """Return classification flags and per-night gap list for a trip."""
    if not items:
        return {
            'is_empty': True, 'has_transport': False,
            'has_lodging': False, 'uncovered_nights': [],
        }

    types = [i.get('item_type', 'Unknown') for i in items]
    has_flight = 'Flight' in types
    has_rail   = 'Rail' in types
    has_lodging = 'Lodging' in types
    has_transport = has_flight or has_rail

    lodging_items = [i for i in items if i.get('item_type') == 'Lodging']
    lodging_ranges = build_lodging_ranges(lodging_items)
    uncovered_nights = []

    if has_transport:
        # Only count transport dates strictly within [trip_start, trip_end).
        # This prevents the next trip's outbound flight (included via the date-
        # overlap query) from making tail-end home-nights look like gaps.
        trip_transport_dates: set[date] = set()
        for item in items:
            if item.get('item_type') in ('Flight', 'Rail'):
                for d in [item.get('dtstart'), item.get('dtend')]:
                    if d and trip_start <= d < trip_end:
                        trip_transport_dates.add(d)

        night = trip_start
        while night < trip_end:
            covered = any(ci <= night < co for ci, co in lodging_ranges)
            is_travel_night = night in trip_transport_dates
            # No future transport = traveller is home; don't flag tail nights.
            has_future_transport = any(d > night for d in trip_transport_dates)
            if not covered and not is_travel_night and has_future_transport:
                uncovered_nights.append(night.isoformat())
            night += timedelta(days=1)

    return {
        'is_empty': False,
        'has_transport': has_transport,
        'has_lodging': has_lodging,
        'has_flight': has_flight,
        'has_rail': has_rail,
        'uncovered_nights': uncovered_nights,
    }


# ---------------------------------------------------------------------------
# Data loading: DB (preferred) vs ICS (fallback)
# ---------------------------------------------------------------------------

def load_trips_from_db(db_path: str) -> list[dict] | None:
    """
    Load trips from travel-db.json.
    Returns list of dicts with keys: summary, start (date), end (date), items.
    items is a list of dicts with: item_type, summary, dtstart (date), dtend (date).
    Returns None if DB is missing or stale (>25h old).
    """
    try:
        with open(db_path) as f:
            db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    # Check freshness — DB should have been rebuilt within last 25 hours
    generated_at_str = db.get('generated_at', '')
    if generated_at_str:
        try:
            generated_at = datetime.strptime(generated_at_str, '%Y-%m-%dT%H:%M:%SZ')
            age_hours = (datetime.utcnow() - generated_at).total_seconds() / 3600
            if age_hours > 25:
                return None  # stale, fall back to live ICS
        except ValueError:
            pass

    trips = []
    for slug, t in db.get('trips', {}).items():
        try:
            trip_start = date.fromisoformat(t['start'])
            trip_end   = date.fromisoformat(t['end'])
        except (KeyError, ValueError):
            continue

        # Flatten days → items list, mapping DB field names to what classify_trip expects
        items = []
        for day_events in t.get('days', {}).values():
            for ev in day_events:
                try:
                    items.append({
                        'item_type': ev['type'],
                        'summary':   ev['summary'],
                        'dtstart':   date.fromisoformat(ev['start']),
                        'dtend':     date.fromisoformat(ev['end']),
                        'uid':       ev.get('uid', ''),
                    })
                except (KeyError, ValueError):
                    continue

        trips.append({
            'summary':    t['summary'],
            'start':      trip_start,
            'end':        trip_end,
            'items':      items,
            'slug':       slug,
        })

    return trips


def load_trips_from_ics() -> list[dict]:
    """Fallback: fetch TripIt ICS and parse trips + items on the fly."""
    url = os.environ.get('TRIPIT_ICAL_URL', '').strip()
    if not url:
        url_file = '/workspace/group/tripit-url.txt'
        if os.path.exists(url_file):
            with open(url_file) as f:
                url = f.read().strip()
    if not url:
        print(json.dumps({'error': 'TRIPIT_ICAL_URL not set and tripit-url.txt not found'}))
        sys.exit(1)

    raw = fetch_ics(url)
    ics = unfold_ics(raw)
    all_events = parse_vevents(ics)

    today = date.today()
    trip_events = []
    item_events = []
    for ev in all_events:
        uid = ev.get('uid', '')
        if 'item-' in uid:
            ev['item_type'] = extract_item_type(ev.get('description', ''))
            item_events.append(ev)
        else:
            trip_events.append(ev)

    trips = []
    for trip in trip_events:
        trip_start = trip.get('dtstart')
        trip_end   = trip.get('dtend', trip_start)
        if trip_start is None:
            continue
        if trip_end and trip_end < today:
            continue

        trip_items = []
        for item in item_events:
            item_start = item.get('dtstart')
            item_end   = item.get('dtend', item_start)
            if item_start is None:
                continue
            item_end_eff  = item_end if item_end else item_start
            trip_end_eff  = trip_end if trip_end else trip_start
            if item_start <= trip_end_eff and item_end_eff >= trip_start:
                trip_items.append(item)

        summary = trip.get('summary', 'Unknown Trip')
        trips.append({
            'summary': summary,
            'start':   trip_start,
            'end':     trip_end,
            'items':   trip_items,
            'slug':    make_slug(summary, trip_start),
        })

    return trips


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    today = date.today()
    db_path = '/workspace/group/travel-db.json'

    trips = load_trips_from_db(db_path)
    source = 'db'
    if trips is None:
        trips = load_trips_from_ics()
        source = 'ics'

    # Load snooze state
    state_path = '/workspace/group/travel-booking-state.json'
    try:
        with open(state_path) as f:
            snooze_state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        snooze_state = {}

    gaps = []
    complete_trips = 0

    for trip in trips:
        trip_start = trip['start']
        trip_end   = trip['end']
        summary    = trip['summary']
        slug       = trip['slug']
        items      = trip['items']

        # Skip past trips
        if trip_end < today:
            continue

        classification = classify_trip(items, trip_start, trip_end)

        issue = None
        uncovered = classification.get('uncovered_nights', [])
        if classification['is_empty']:
            issue = 'ничего не забукано'
        elif classification['has_transport'] and not classification['has_lodging']:
            issue = 'рейсы есть, отеля нет'
        elif classification['has_transport'] and uncovered:
            issue = f'нет отеля на {len(uncovered)} ноч.: {uncovered[0]}…{uncovered[-1]}'

        if issue is None:
            complete_trips += 1
            continue

        # Check snooze
        snooze_entry = snooze_state.get(slug, {})
        snooze_until_str = snooze_entry.get('snooze_until', '')
        if snooze_until_str:
            try:
                if date.fromisoformat(snooze_until_str) >= today:
                    complete_trips += 1
                    continue
            except ValueError:
                pass

        gaps.append({
            'trip':             summary,
            'start':            trip_start.isoformat(),
            'end':              trip_end.isoformat(),
            'issue':            issue,
            'slug':             slug,
            'uncovered_nights': uncovered if uncovered else [],
        })

    output = {
        'gaps':          gaps,
        'checked_at':    datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'total_trips':   len(trips),
        'complete_trips': complete_trips,
        'source':        source,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
