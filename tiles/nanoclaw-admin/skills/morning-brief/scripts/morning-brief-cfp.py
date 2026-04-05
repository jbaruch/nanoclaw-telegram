#!/usr/bin/env python3
"""
Urgent CFP filter for morning brief.

Reads cfp-state.json (maintained by the check-cfps skill during nightly
housekeeping) and filters to open CFPs with deadlines within 7 days.

Output: JSON array of {name, city, conf_date, deadline, cfp_url, days_until}.
Empty array if no urgent CFPs or state file missing. Exit code 0 always.
"""

import json
import sys
from datetime import date
from pathlib import Path

STATE_PATH = Path('/workspace/group/cfp-state.json')

if not STATE_PATH.exists():
    print(json.dumps([]))
    sys.exit(0)

try:
    state = json.loads(STATE_PATH.read_text())
except Exception:
    print(json.dumps([]))
    sys.exit(0)

today = date.today()
urgent = []

for slug, entry in state.items():
    # Skip metadata keys (e.g. _blocked_prefixes)
    if slug.startswith('_'):
        continue
    if entry.get('status') != 'open':
        continue

    deadline_str = entry.get('deadline', '')
    try:
        deadline = date.fromisoformat(deadline_str)
    except (ValueError, TypeError):
        continue

    days_until = (deadline - today).days
    if days_until < 0 or days_until > 7:
        continue

    urgent.append({
        'name': entry.get('name', slug),
        'city': entry.get('city', ''),
        'conf_date': entry.get('conf_date', ''),
        'deadline': deadline_str,
        'cfp_url': entry.get('cfp_url', ''),
        'days_until': days_until,
    })

urgent.sort(key=lambda c: c['days_until'])
print(json.dumps(urgent, indent=2))
