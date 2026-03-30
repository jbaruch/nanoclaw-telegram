#!/usr/bin/env python3
"""
CFP fetch-and-filter pipeline for the check-cfps skill.

Fetches structured CFP data from primary sources, applies hard deterministic filters
(virtual/online, excluded locations, travel conflicts, cfp-state), and outputs
a filtered + sorted JSON list for the skill to reason about relevance and format.

NOT done here (left to AI reasoning in the skill):
  - Conference topic relevance (Web3/blockchain, .NET/PHP/Ruby, etc.)
  - Email actionability

Output JSON:
  {
    "cfps": [
      {
        "name":     "Conference Name",
        "city":     "City, Country",
        "conf_date": "YYYY-MM-DD",  // earliest conf date
        "cfp_url":  "https://...",
        "deadline": "YYYY-MM-DD",
        "days_left": 14,
        "slug":     "conference-name-2026",
        "source":   "developers.events" | "javaconferences.org"
      },
      ...
    ],
    "warnings": ["Source A unreachable", ...],
    "checked_at": "2026-03-29T05:00:00Z"
  }

Exit code 0 always.
"""

import json
import re
import urllib.request
import urllib.error
from datetime import date, datetime, timezone
from pathlib import Path


STATE_PATH    = Path('/workspace/group/cfp-state.json')
TRAVEL_PATH   = Path('/workspace/group/travel-schedule.json')

# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def make_slug(name: str) -> str:
    """Normalize conference name to a slug including the year."""
    lower = name.lower().strip()
    # Extract trailing year if present
    year_match = re.search(r'\b(20\d\d)\b', lower)
    year = year_match.group(1) if year_match else str(date.today().year)
    # Remove trailing year from base (will re-append)
    base = re.sub(r'\s*20\d\d\s*$', '', lower).strip()
    slug_base = re.sub(r'[^a-z0-9]+', '-', base).strip('-')
    return f"{slug_base}-{year}"


# ---------------------------------------------------------------------------
# Source A: developers.events
# ---------------------------------------------------------------------------

def fetch_developers_events(warnings: list) -> list:
    url = 'https://developers.events/all-cfps.json'
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8', errors='replace'))
    except Exception as e:
        warnings.append(f'Source A (developers.events) unreachable: {e}')
        return []

    if not isinstance(data, list):
        warnings.append('Source A: unexpected format (not a list)')
        return []

    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    today  = date.today()
    results = []

    for entry in data:
        try:
            until_ms = entry.get('untilDate', 0)
            if not until_ms or until_ms <= now_ms:
                continue

            deadline = date.fromtimestamp(until_ms / 1000)

            conf = entry.get('conf', {})
            name = conf.get('name', '').strip()
            if not name:
                continue

            cfp_url = entry.get('link', '') or conf.get('hyperlink', '')
            location = conf.get('location', '')

            # Conference date: first date in conf.date array (ms timestamps)
            conf_dates = conf.get('date', [])
            conf_date = None
            if conf_dates:
                try:
                    conf_date = date.fromtimestamp(min(conf_dates) / 1000).isoformat()
                except Exception:
                    pass

            results.append({
                'name':      name,
                'city':      location,
                'conf_date': conf_date or '',
                'cfp_url':   cfp_url,
                'deadline':  deadline.isoformat(),
                'source':    'developers.events',
            })
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Source B: javaconferences.org
# ---------------------------------------------------------------------------

def fetch_javaconferences(warnings: list) -> list:
    url = 'https://javaconferences.org/conferences.json'
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8', errors='replace'))
    except Exception as e:
        warnings.append(f'Source B (javaconferences.org) unreachable: {e}')
        return []

    if not isinstance(data, list):
        warnings.append('Source B: unexpected format (not a list)')
        return []

    today   = date.today()
    results = []

    for entry in data:
        try:
            cfp_link = entry.get('cfpLink', '').strip()
            if not cfp_link:
                continue

            cfp_end_str = entry.get('cfpEndDate', '')
            if not cfp_end_str:
                continue
            # cfpEndDate may be full ISO datetime or date-only
            deadline = date.fromisoformat(cfp_end_str[:10])
            if deadline < today:
                continue

            name = entry.get('name', '').strip()
            if not name:
                continue

            location = entry.get('locationName', '')
            conf_date_str = entry.get('date', '')
            conf_date = conf_date_str[:10] if conf_date_str else ''

            results.append({
                'name':      name,
                'city':      location,
                'conf_date': conf_date,
                'cfp_url':   cfp_link,
                'deadline':  deadline.isoformat(),
                'source':    'javaconferences.org',
            })
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

VIRTUAL_KEYWORDS = {'online', 'virtual', 'remote', 'hybrid'}

EXCLUDED_LOCATIONS = {
    'nigeria', 'kenya', 'south africa', 'ghana', 'ethiopia',
    'tanzania', 'uganda', 'rwanda',
}


def is_virtual(cfp: dict) -> bool:
    city = cfp.get('city', '').lower()
    name = cfp.get('name', '').lower()
    for kw in VIRTUAL_KEYWORDS:
        if kw in city or kw in name:
            return True
    return not city.strip()  # no location listed


def is_excluded_location(cfp: dict) -> bool:
    city = cfp.get('city', '').lower()
    for loc in EXCLUDED_LOCATIONS:
        if loc in city:
            return True
    return False


def load_travel_schedule(warnings: list) -> list:
    if not TRAVEL_PATH.exists():
        warnings.append('travel-schedule.json not found — skipping travel conflict check')
        return []
    try:
        with open(TRAVEL_PATH) as f:
            events = json.load(f)
        # Only trip events (no 'item-' in uid)
        trips = []
        for ev in events:
            if 'item-' not in ev.get('uid', '') and ev.get('start') and ev.get('end'):
                try:
                    trips.append({
                        'start': date.fromisoformat(ev['start']),
                        'end':   date.fromisoformat(ev['end']),
                    })
                except ValueError:
                    pass
        return trips
    except Exception as e:
        warnings.append(f'Failed to load travel-schedule.json: {e}')
        return []


def has_travel_conflict(cfp: dict, trips: list) -> bool:
    conf_date_str = cfp.get('conf_date', '')
    if not conf_date_str:
        return False
    try:
        # Treat conf_date as a single-day event for conflict check
        conf_start = date.fromisoformat(conf_date_str)
        # Assume 4-day conference if no end info
        from datetime import timedelta
        conf_end = conf_start + timedelta(days=4)
    except ValueError:
        return False
    for trip in trips:
        if conf_start <= trip['end'] and conf_end >= trip['start']:
            return True
    return False


def load_cfp_state(warnings: list) -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception as e:
        warnings.append(f'Failed to load cfp-state.json: {e}')
        return {}


def apply_state_filter(cfp: dict, state: dict, today: date) -> bool:
    """Return True if this CFP should be shown (not filtered out by state)."""
    # Check blocked_prefixes — conference name patterns filtered indefinitely
    name_lower = cfp.get('name', '').lower()
    for prefix in state.get('_blocked_prefixes', []):
        if name_lower.startswith(prefix.lower()):
            return False

    slug = cfp.get('slug', '')
    entry = state.get(slug, {})
    status = entry.get('status', '')

    if status in ('sent', 'dismissed'):
        return False

    if status == 'remind':
        try:
            deadline = date.fromisoformat(cfp['deadline'])
            remind_days = entry.get('remind_before_days', 7)
            days_left = (deadline - today).days
            return days_left <= remind_days
        except (ValueError, KeyError):
            return True

    return True  # no state → show


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(cfps: list) -> list:
    seen = {}
    result = []
    for cfp in cfps:
        key = cfp['name'].lower().strip()
        if key not in seen:
            seen[key] = True
            result.append(cfp)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warnings = []
    today = date.today()

    # Fetch
    source_a = fetch_developers_events(warnings)
    source_b = fetch_javaconferences(warnings)
    all_cfps = source_a + source_b

    if not all_cfps:
        warnings.append('Both primary sources returned empty — web search fallback needed')

    # Load supporting data
    trips = load_travel_schedule(warnings)
    state = load_cfp_state(warnings)

    # Enrich with slug and days_left
    for cfp in all_cfps:
        cfp['slug'] = make_slug(cfp['name'])
        try:
            deadline = date.fromisoformat(cfp['deadline'])
            cfp['days_left'] = (deadline - today).days
        except ValueError:
            cfp['days_left'] = 9999

    # Filter — hard rules only; relevance judgment left to AI
    filtered = []
    for cfp in all_cfps:
        if is_virtual(cfp):
            continue
        if is_excluded_location(cfp):
            continue
        if cfp.get('days_left', 0) < 0:
            continue
        if has_travel_conflict(cfp, trips):
            continue
        if not apply_state_filter(cfp, state, today):
            continue
        filtered.append(cfp)

    # Deduplicate and sort by deadline
    filtered = deduplicate(filtered)
    filtered.sort(key=lambda c: c.get('deadline', '9999-99-99'))

    output = {
        'cfps':       filtered,
        'warnings':   warnings,
        'checked_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
