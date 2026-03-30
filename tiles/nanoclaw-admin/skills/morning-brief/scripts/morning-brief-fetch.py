#!/usr/bin/env python3
"""
Fetch today's Google Tasks for morning brief.

Outputs JSON:
{
  "overdue": [{"id", "title", "tasklist", "due"}],
  "due_today": [{"id", "title", "tasklist", "due"}],
  "fetched_at": "ISO timestamp"
}

Calendar fetching is left to the skill (requires OAuth flow via Composio).
Exit code 0 always.
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path

PENDING_PATH = Path('/workspace/group/morning-brief-pending.json')


def load_pending():
    if not PENDING_PATH.exists():
        return {"undated_tasks": [], "cleanup_items": []}
    try:
        with open(PENDING_PATH) as f:
            data = json.load(f)
        data.setdefault("undated_tasks", [])
        data.setdefault("cleanup_items", [])
        return data
    except Exception:
        return {"undated_tasks": [], "cleanup_items": []}


def main():
    today = date.today().isoformat()
    pending = load_pending()

    output = {
        "today": today,
        "pending": pending,
        "fetched_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "note": "Tasks due today/overdue must be fetched via Composio (Google Tasks API). Use GOOGLETASKS_LIST_TASKS with dueMax=today."
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
