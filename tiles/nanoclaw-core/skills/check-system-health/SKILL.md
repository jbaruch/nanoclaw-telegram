---
name: check-system-health
description: Check NanoClaw system health — stuck tasks, DB size, task run failures. Uses /workspace/store/messages.db directly. Use as part of heartbeat or standalone. Triggers on "system health", "check tasks", "check database".
---

# Check System Health

**Invoked from:** heartbeat (Step 5). Also available standalone.

DB is at `/workspace/store/messages.db`. Run each check below.

## 1. Stuck scheduled tasks

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/workspace/store/messages.db')
rows = conn.execute(\"SELECT id, substr(prompt, 1, 50), next_run FROM scheduled_tasks WHERE status='active' AND next_run <= datetime('now', '-5 minutes')\").fetchall()
for r in rows: print(r)
print(f'stuck={len(rows)}')
conn.close()
"
```

**If stuck > 0:** Report the stuck task IDs and prompts. The DB is read-only from the container — auto-fix is not possible. The orchestrator's scheduler will retry on the next poll cycle. If tasks remain stuck, flag for the owner to investigate.

## 2. Database size

```bash
python3 -c "
import sqlite3, os
conn = sqlite3.connect('/workspace/store/messages.db')
msg_count = conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
log_count = conn.execute('SELECT COUNT(*) FROM task_run_logs').fetchone()[0]
conn.close()
size_mb = os.path.getsize('/workspace/store/messages.db') / 1048576
print(f'messages={msg_count} task_run_logs={log_count} size={size_mb:.1f}MB')
"
```

**Alert if:** messages > 100k rows, task_run_logs > 10k rows, or DB > 500MB.

## 3. Recent task failures

Task failure checks are handled by `heartbeat-checks.py` (`check_task_failures` function), which queries `task_run_logs` and respects the dismiss file at `/workspace/group/system-health-dismissed.json`.

**Alert if:** failures > 0 and not dismissed. Report task IDs and error summaries.

**Note:** The correct column name is `run_at` (not `timestamp`) in `task_run_logs`.

## 4. Dismiss mechanism

Persistent dismissals are stored in `/workspace/group/system-health-dismissed.json`:

```json
{
  "dismissed": {
    "task_failure:<task_id>": {
      "reason": "why dismissed",
      "dismissed_at": "2026-04-02T16:00:00Z",
      "expires_at": null
    }
  }
}
```

- **Fingerprint format:** `task_failure:<task_id>` (e.g., `task_failure:task-1774576028296-wfve4q`)
- **`expires_at`: null** = permanent dismiss (never re-reports)
- **`expires_at`: ISO timestamp** = snooze until that time (e.g., `"2026-04-03T16:00:00Z"` = 24h snooze)

**To dismiss an issue:** write its fingerprint into `system-health-dismissed.json`. The check will skip it on all future runs (until expiry if set).

**To re-enable:** remove the entry from `system-health-dismissed.json` or set `expires_at` to a past timestamp.

## Output

Return issues found or empty if all clear.
