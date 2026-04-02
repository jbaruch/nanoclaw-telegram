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

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/workspace/store/messages.db')
rows = conn.execute(\"SELECT task_id, substr(error, 1, 80), timestamp FROM task_run_logs WHERE status='error' AND timestamp >= datetime('now', '-24 hours') ORDER BY timestamp DESC LIMIT 5\").fetchall()
for r in rows: print(r)
print(f'failures={len(rows)}')
conn.close()
"
```

**Alert if:** failures > 0. Report task IDs and error summaries.

## Output

Return issues found or empty if all clear.
