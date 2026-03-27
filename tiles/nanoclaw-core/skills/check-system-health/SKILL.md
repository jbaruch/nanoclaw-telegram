---
name: check-system-health
description: Check NanoClaw infrastructure health — stuck tasks, IPC errors, database size, log growth, session bloat, retry exhaustion, stuck close files, OneCLI status. Auto-fixes what's possible, reports the rest. Use as part of heartbeat or standalone diagnostics. Triggers on "system health", "check infrastructure", "check logs", "check database".
---

# Check System Health

Run each check below. Auto-fix what you can, report what needs human attention.

## 1. Stuck scheduled tasks

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/workspace/project/store/messages.db')
rows = conn.execute(\"SELECT id, prompt, next_run FROM scheduled_tasks WHERE status='active' AND next_run <= datetime('now', '-5 minutes')\").fetchall()
for r in rows: print(r)
conn.close()
"
```

**Alert if:** any rows returned — these tasks should have fired but didn't.

**Auto-fix:** Reset `next_run` to now + 1 minute so they retry:
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/workspace/project/store/messages.db')
conn.execute(\"UPDATE scheduled_tasks SET next_run = datetime('now', '+1 minute') WHERE status='active' AND next_run <= datetime('now', '-5 minutes')\")
conn.commit()
print(f'Reset {conn.total_changes} stuck tasks')
conn.close()
"
```

**Verify:** Re-run the initial query — confirm no rows are returned (or count has decreased to zero).

## 2. Failed IPC files

```bash
ls -la /workspace/ipc/errors/ 2>/dev/null | tail -20
du -sh /workspace/ipc/errors/ 2>/dev/null
```

**Alert if:** error directory exists and is non-empty. Include the filenames.

**Auto-fix:** Delete IPC error files older than 7 days. Report count deleted.

**Verify:** Re-run `du -sh /workspace/ipc/errors/` — confirm size has reduced accordingly.

## 3. Database size

```bash
python3 -c "
import sqlite3, os
conn = sqlite3.connect('/workspace/project/store/messages.db')
msg_count = conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
log_count = conn.execute('SELECT COUNT(*) FROM task_run_logs').fetchone()[0]
conn.close()
size_mb = os.path.getsize('/workspace/project/store/messages.db') / 1048576
print(f'messages={msg_count} task_run_logs={log_count} size={size_mb:.1f}MB')
"
```

**Alert if:** messages > 100k rows, task_run_logs > 10k rows, or DB file > 500MB.

**Report only:** suggest archiving old messages.

## 4. Log growth

```bash
du -sh /workspace/project/logs/
find /workspace/project/logs/ -type f -size +50M 2>/dev/null
```

**Alert if:** total logs > 100MB, or any single file > 50MB.

**Auto-fix:** Truncate files > 50MB to last 10k lines. Report old size -> new size.

**Verify:** Re-run `du -sh /workspace/project/logs/` and `find /workspace/project/logs/ -type f -size +50M` — confirm total is reduced and no files remain above 50MB.

## 5. Session memory bloat

```bash
du -sh /workspace/project/data/sessions/*/ 2>/dev/null
du -sh /workspace/project/data/sessions/ 2>/dev/null
```

**Alert if:** total sessions > 500MB, or any single group > 100MB.

**Auto-fix:** For each group session directory, find and delete Claude Code session files (`.jsonl` in `projects/`) older than 7 days. These are conversation transcripts that grow without bound. Keep the latest 5 per group. Report bytes freed.

```bash
python3 -c "
import os, glob, time

sessions_base = '/workspace/project/data/sessions'
cutoff = time.time() - 7 * 86400
freed = 0
for group_dir in glob.glob(f'{sessions_base}/*/.claude/projects/*'):
    jsonls = sorted(glob.glob(f'{group_dir}/*.jsonl'), key=os.path.getmtime)
    if len(jsonls) <= 5:
        continue
    for f in jsonls[:-5]:
        if os.path.getmtime(f) < cutoff:
            size = os.path.getsize(f)
            os.unlink(f)
            freed += size
            # Also remove subagent dirs for this session
            session_dir = f.rsplit('.', 1)[0]
            if os.path.isdir(session_dir):
                import shutil
                freed += sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(session_dir) for fn in fns)
                shutil.rmtree(session_dir)
if freed > 0:
    print(f'Freed {freed // 1048576}MB from old sessions')
else:
    print('OK')
"
```

**Verify:** Re-run `du -sh` on session directories to confirm reduction.

## 6. Retry exhaustion (dropped messages)

```bash
python3 -c "
import subprocess, os, json

log_file = '/workspace/project/logs/nanoclaw.log'
state_file = '/workspace/group/nanoclaw-state.json'

result = subprocess.run(['grep', '-c', 'Max retries exceeded', log_file],
    capture_output=True, text=True)
current = int(result.stdout.strip()) if result.returncode == 0 else 0

state = {}
try:
    state = json.load(open(state_file))
except:
    pass
prev = state.get('retry_exhaustion_count', current)

state['retry_exhaustion_count'] = current
json.dump(state, open(state_file, 'w'))

new_drops = current - prev
if new_drops > 0:
    print(f'NEW_DROPS={new_drops}')
else:
    print('OK')
"
```

**Alert if:** output contains `NEW_DROPS=`. Only new drops since last run.

**Report only:** grep the log for the specific group/JID that was dropped.

## 7. IPC close files stuck

```bash
find /workspace/project/data/ipc/*/input -name '_close' -mmin +30 2>/dev/null
```

**Alert if:** any files found — close signals that were never processed.

**Auto-fix:** Delete the stuck `_close` file. Report which group.

**Verify:** Re-run the `find` command — confirm no `_close` files older than 30 minutes remain.

## 8. OneCLI health

```bash
curl -sf http://host.docker.internal:10254/api/health 2>/dev/null || echo "ONECLI_DOWN"
```

**Alert if:** returns ONECLI_DOWN or non-200. Credentials won't work without it.

**Report only:** suggest `docker compose -p onecli up -d`.

## Output

Return a list of issues found (with auto-fix results) or empty if all clear. The heartbeat orchestrator will format and send the final message.

## Post-Fix Validation

After applying any auto-fixes, re-run the checks for each affected section to confirm resolution. Report the before/after state for each fix applied. If a re-run still shows issues (e.g., new stuck tasks appeared, log files grew back), flag for human attention rather than looping indefinitely.
