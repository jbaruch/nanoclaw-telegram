# Heartbeat

You are running as a periodic health check. Run every check below, collect results, and ONLY message the user if something is wrong. Silent when healthy.

## Checks

Run these commands and evaluate the results:

### 1. Stuck scheduled tasks

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

### 2. Failed IPC files

```bash
ls -la /workspace/ipc/errors/ 2>/dev/null | tail -20
du -sh /workspace/ipc/errors/ 2>/dev/null
```

**Alert if:** error directory exists and is non-empty. Include the filenames — they indicate what failed.

### 3. Database size

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

### 4. Log growth

```bash
du -sh /workspace/project/logs/
find /workspace/project/logs/ -type f -size +50M 2>/dev/null
```

**Alert if:** total logs > 100MB, or any single file > 50MB.

### 5. Session memory bloat

```bash
du -sh /workspace/project/data/sessions/*/ 2>/dev/null
du -sh /workspace/project/data/sessions/ 2>/dev/null
```

**Alert if:** total sessions > 500MB, or any single group > 100MB.

### 6. Retry exhaustion (dropped messages)

```bash
python3 -c "
import subprocess, os, json

log_file = '/workspace/project/logs/nanoclaw.log'
state_file = '/workspace/group/heartbeat-state.json'

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

**Alert if:** output contains `NEW_DROPS=`. Only new drops since last heartbeat run — not cumulative total.

### 7. IPC close files stuck

```bash
find /workspace/project/data/ipc/*/input -name '_close' -mmin +30 2>/dev/null
```

**Alert if:** any files found — these are stdin close signals that were never processed.

### 8. OneCLI health

```bash
curl -sf http://host.docker.internal:10254/api/health 2>/dev/null || echo "ONECLI_DOWN"
```

**Alert if:** returns ONECLI_DOWN or non-200. Credentials won't work without it.

### 9. Unanswered messages

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/workspace/project/store/messages.db')
rows = conn.execute('''
  SELECT m.id, m.chat_jid, m.sender_name, substr(m.content, 1, 80), m.timestamp
  FROM messages m
  WHERE m.is_from_me = 0
    AND m.is_bot_message = 0
    AND m.timestamp >= datetime('now', '-15 minutes')
    AND m.timestamp <= datetime('now', '-5 minutes')
    AND NOT EXISTS (
      SELECT 1 FROM messages r
      WHERE r.chat_jid = m.chat_jid
        AND r.timestamp > m.timestamp
        AND r.is_bot_message = 1
    )
  ORDER BY m.timestamp ASC
''').fetchall()
for r in rows: print(r)
print(f'total={len(rows)}')
conn.close()
"
```

**Alert if:** total > 0. These are messages from Baruch that received no bot reply within 5-15 minutes. Include the sender name, chat, and truncated content in the alert.

### 10. Calendar changes (reschedule reminders if needed)

Read `/workspace/group/calendar-state.json`. If it exists and `date` matches today:

Use `COMPOSIO_SEARCH_TOOLS` to find `GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS`, then fetch today's events (time_min/time_max = today in America/Chicago, single_events=true, order_by=startTime).

Compare the fetched events to the state file's `events` list (match by event_id). Check for:
- New events added
- Existing events changed time or title
- Events removed

**If nothing changed:** do nothing (wrap output in `<internal>`).

**If calendar changed:**
1. Cancel all existing reminder tasks from state: for each event with a `reminder_task_id`, call `mcp__nanoclaw__cancel_task`.
2. Reschedule reminders: for each timed event (not all-day, not 🚌 Travel) starting more than 20 min from now, schedule a new `once` task 15 min before start (local time, no Z suffix).
3. Update `/workspace/group/calendar-state.json` with the new event list and new task IDs.
4. Alert with the changes.

**If state file doesn't exist or is from a previous day:** skip — morning brief handles initial scheduling at 8am.

**Alert format when changed:**
```
• Calendar changed: [Event X] moved to 3pm → reminder rescheduled
```

### 11. Important email check

Use `COMPOSIO_SEARCH_TOOLS` to find `GMAIL_FETCH_EMAILS`, then fetch recent emails:
- max_results: 20
- label_ids: ["INBOX"]
- Do NOT include spam/trash

Read `/workspace/group/heartbeat-state.json` to get `last_email_checked` (a messageId string). Only process emails NEWER than that ID (higher messageId = newer in Gmail). If no state, process the latest 5 only.

After fetching, update `last_email_checked` in the state file with the newest messageId seen.

**Classify each new email as important if ANY of these apply:**
- Sender is a real person (not noreply@, not alerts@, not newsletters, not @*.sendgrid.net, not @*.mailchimp.com, not automated bulk senders)
- Subject looks like a direct reply or question (Re:, Fwd: from a human, contains "?", action words like "review", "approve", "can you", "please")
- Sender domain matches known work contacts (jfrog.com, colleagues, conference organizers, etc.)
- Email is not in CATEGORY_PROMOTIONS or CATEGORY_UPDATES label

**For each important email, include in alert:**
- Sender name + email
- Subject
- First 100 chars of body

**If no important emails:** wrap in `<internal>`, say nothing.

**Alert format:**
```
• 📬 New email from [Name]: "[Subject]" — [preview...]
```

## Output format

If ALL checks pass: wrap your entire output in `<internal>` tags. Say nothing to the user.

```
<internal>Heartbeat: all clear.</internal>
```

If ANY check fails, send a message with only the failures:

```
*Heartbeat* 🫀

• Stuck tasks: task-abc123 overdue by 12 minutes
• IPC errors: 5 stuck files in error directory
• Logs: nanoclaw.log at 127MB, needs rotation
```

Keep it short. No preamble. No "I ran a health check and found..." — just the problems.

## Self-healing rules

When a check fires, don't just report — diagnose, fix if possible, report the outcome.

### Auto-fix (do it, report what you did):

- **IPC error files > 7 days old**: delete them, report count deleted
- **Log files > 50MB**: truncate to last 10k lines, report old size → new size
- **IPC close files stuck > 30 min**: delete the `_close` file, report which group
- **Stuck scheduled tasks**: check if the task's container is still running. If not, reset `next_run` to now + 1 minute so it retries:
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

### Report only (fix needs human):

- **DB size > 500MB**: report size, suggest archiving old messages
- **Session memory > 500MB**: report which groups are bloated, suggest pruning
- **OneCLI down**: report HTTP status, suggest `docker compose -p onecli up -d`
- **Unanswered messages**: For each unanswered message, investigate logs for why (container crash? timeout? error?), then evaluate the message content and decide:

  1. **Expired / no longer actionable** — the request was time-sensitive and the window has passed (e.g. "what's my next meeting?" and the meeting already ended; "remind me in 5 minutes" from 20 min ago). Send a brief note: _"Пропустил: [summary of what was asked]. Уже не актуально — [what the answer would have been]."_

  2. **Still actionable** — the task has no hard deadline or the deadline hasn't passed (research, write an email, find something, general questions). Just go do it: process the message as if it just arrived and send the response normally. Do NOT report it as an issue.

  3. **Unclear** — can't determine if expired or not. Report it: include sender, chat, message content, log diagnosis, and note the ambiguity.
- **Retry exhaustion (dropped messages)**: report count and grep the log for the specific group/JID that was dropped

### Output format for fixes

When you auto-fix something, report it as:

```
*Heartbeat* 🫀

• Stuck tasks: 2 tasks overdue → reset next_run (heartbeat, task-abc123)
• Logs: nanoclaw.log was 67MB → truncated to 1.2MB
• IPC errors: deleted 3 files older than 7 days
```

When you can only diagnose:

```
*Heartbeat* 🫀

• OneCLI down (HTTP 000) — credentials won't work until restarted
• Unanswered: 1 message in telegram_main from 12 min ago. Log shows container exited code 137 (OOM). Next message should work.
```
