#!/usr/bin/env python3
"""
Urgent CFP filter for morning brief.

Reads cfp-state.json (maintained by the check-cfps skill during nightly
housekeeping) and filters to open CFPs with deadlines within 7 days.

Additionally includes "sticky" CFPs: those previously shown in a brief
(shown_in_brief=true) whose deadline hasn't passed yet. This prevents
good CFPs from disappearing between briefs due to non-deterministic filtering.

When called with --mark-shown, updates cfp-state.json to set shown_in_brief
on all CFPs that were output in the current run.

Output: JSON array of {name, city, conf_date, deadline, cfp_url, days_until, previously_shown}.
Empty array if no urgent CFPs or state file missing. Exit code 0 always.
"""

import json
import sys
from datetime import date
from pathlib import Path

STATE_PATH = Path('/workspace/group/cfp-state.json')
MARK_SHOWN = '--mark-shown' in sys.argv

if not STATE_PATH.exists():
    print(json.dumps([]))
    sys.exit(0)

try:
    state = json.loads(STATE_PATH.read_text())
except Exception:
    print(json.dumps([]))
    sys.exit(0)

today = date.today()
results = []
shown_slugs = []

for slug, entry in state.items():
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

    # Skip if deadline already passed
    if days_until < 0:
        continue

    # Include if: within 7-day window OR previously shown in a brief
    in_window = days_until <= 7
    previously_shown = bool(entry.get('shown_in_brief'))

    if not in_window and not previously_shown:
        continue

    shown_slugs.append(slug)
    results.append({
        'slug': slug,
        'name': entry.get('name', slug),
        'city': entry.get('city', ''),
        'conf_date': entry.get('conf_date', ''),
        'deadline': deadline_str,
        'cfp_url': entry.get('cfp_url', ''),
        'days_until': days_until,
        'previously_shown': previously_shown,
    })

results.sort(key=lambda c: c['days_until'])

# If --mark-shown, update cfp-state.json with shown_in_brief for all output CFPs
if MARK_SHOWN and shown_slugs:
    for slug in shown_slugs:
        if slug in state and isinstance(state[slug], dict):
            state[slug]['shown_in_brief'] = True
            state[slug]['last_shown_date'] = today.isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))

print(json.dumps(results, indent=2))
