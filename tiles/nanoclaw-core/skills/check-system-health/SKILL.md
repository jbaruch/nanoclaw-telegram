---
name: check-system-health
description: Check NanoClaw system health — stuck tasks, DB size, task run failures. Uses /workspace/store/messages.db directly. Use as part of heartbeat or standalone. Triggers on "system health", "check tasks", "check database".
---

# Check System Health

**Invoked from:** heartbeat (Step 5). Also available standalone.

DB is at `/workspace/store/messages.db`. Run each check below.

> **Error handling (all checks):** If a `sqlite3.OperationalError` occurs (table missing, DB locked, etc.), report the error message and skip remaining checks on the affected table. Do not fail silently.

## 0. Pre-flight: verify DB is accessible

Before running checks, confirm the DB file exists and is readable:

```bash
python3 -c "
import os, sys
db = '/workspace/store/messages.db'
if not os.path.exists(db):
    print('ERROR: DB file not found:', db)
    sys.exit(1)
if not os.access(db, os.R_OK):
    print('ERROR: DB file not readable:', db)
    sys.exit(1)
print('DB accessible, size:', os.path.getsize(db), 'bytes')
"
```

**If the file is missing or unreadable:** report the error via `mcp__nanoclaw__send_message` and stop. Do not proceed with further checks.

## 1. Stuck scheduled tasks

```bash
python3 -c "
import sqlite3, sys
db = '/workspace/store/messages.db'
try:
    conn = sqlite3.connect(db, timeout=5)
    rows = conn.execute(\"SELECT id, substr(prompt, 1, 50), next_run FROM scheduled_tasks WHERE status='active' AND next_run <= datetime('now', '-5 minutes')\").fetchall()
    for r in rows: print(r)
    print(f'stuck={len(rows)}')
    conn.close()
except sqlite3.OperationalError as e:
    print('ERROR:', e)
    sys.exit(1)
"
```

**If stuck > 0:** Report the stuck task IDs and prompts. The DB is read-only from the container — auto-fix is not possible. The orchestrator's scheduler will retry on the next poll cycle. If tasks remain stuck, flag for the owner to investigate.

## 2. Database size

```bash
python3 -c "
import sqlite3, os, sys
db = '/workspace/store/messages.db'
try:
    conn = sqlite3.connect(db, timeout=5)
    msg_count = conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
    log_count = conn.execute('SELECT COUNT(*) FROM task_run_logs').fetchone()[0]
    conn.close()
    size_mb = os.path.getsize(db) / 1048576
    print(f'messages={msg_count} task_run_logs={log_count} size={size_mb:.1f}MB')
except sqlite3.OperationalError as e:
    print('ERROR:', e)
    sys.exit(1)
"
```

**Alert if:** messages > 100k rows, task_run_logs > 10k rows, or DB > 500MB.

## 3. Recent task failures

Task failure checks are handled by `heartbeat-checks.py` (`check_task_failures` function at `/workspace/group/heartbeat-checks.py`), which queries `task_run_logs` (column: `run_at`) and respects the dismiss file at `/workspace/group/system-health-dismissed.json`.

**Alert if:** failures > 0 and not dismissed. Report task IDs and error summaries.

**If `heartbeat-checks.py` is not found or raises an error:** report the error directly and skip this check rather than failing silently.

## 4. Dismiss mechanism

Persistent dismissals are stored in `/workspace/group/system-health-dismissed.json`. Each entry uses the fingerprint `task_failure:<task_id>` as the key:

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

- **`expires_at`: null** = permanent dismiss; **ISO timestamp** = snooze until that time.
- **To dismiss:** write the fingerprint into the file. The check skips it on all future runs (until expiry).
- **To re-enable:** remove the entry or set `expires_at` to a past timestamp.

## Output

**If issues found:** report them via `mcp__nanoclaw__send_message`.

**If no issues: output nothing. Complete silence. Never output "all clear", "no issues found", "everything looks good", or any confirmation that checks passed. Silence IS the success signal.**
