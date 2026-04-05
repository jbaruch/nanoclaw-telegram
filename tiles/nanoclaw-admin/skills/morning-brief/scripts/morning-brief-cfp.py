#!/usr/bin/env python3
"""
Urgent CFP filter for morning brief.

Runs check-cfps-fetch.py and filters to CFPs with deadlines within 7 days.
Outputs a JSON array with simplified fields for the brief.

Exit code 0 always.
"""

import json
import subprocess
import sys
from pathlib import Path

FETCH_SCRIPT = Path('/home/node/.claude/skills/tessl__check-cfps/scripts/check-cfps-fetch.py')

try:
    result = subprocess.run(
        [sys.executable, str(FETCH_SCRIPT)],
        capture_output=True, text=True, timeout=60,
    )
    data = json.loads(result.stdout)
except Exception as e:
    print(json.dumps([]))
    sys.exit(0)

urgent = []
for cfp in data.get('cfps', []):
    days = cfp.get('days_left')
    if days is not None and days <= 7:
        urgent.append({
            'name': cfp.get('name', ''),
            'city': cfp.get('city', ''),
            'conf_date': cfp.get('conf_date', ''),
            'deadline': cfp.get('deadline', ''),
            'cfp_url': cfp.get('cfp_url', ''),
            'days_until': days,
        })

urgent.sort(key=lambda c: c['days_until'])
print(json.dumps(urgent, indent=2))
