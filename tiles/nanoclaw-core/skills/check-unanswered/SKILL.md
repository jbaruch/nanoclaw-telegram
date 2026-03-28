---
name: check-unanswered
description: Find and triage unanswered messages. Queries messages.db directly, then triages each — expired requests get acknowledged, actionable ones get processed, unclear ones get reported. Use as part of heartbeat or standalone. Triggers on "check unanswered", "missed messages", "unreplied messages".
---

# Check Unanswered Messages

## Detection

```bash
python3 -c "
import sqlite3, sys, os
db_path = '/workspace/store/messages.db'
if not os.path.exists(db_path):
    print('ERROR: messages.db not found at ' + db_path)
    sys.exit(1)
try:
    conn = sqlite3.connect(db_path)
    rows = conn.execute('''
      SELECT m.id, m.chat_jid, m.sender_name, substr(m.content, 1, 80), m.timestamp
      FROM messages m
      WHERE m.is_from_me = 0
        AND m.is_bot_message = 0
        AND m.timestamp <= datetime('now', '-5 minutes')
        AND m.timestamp >= datetime('now', '-24 hours')
        AND NOT EXISTS (
          SELECT 1 FROM messages r
          WHERE r.chat_jid = m.chat_jid
            AND r.timestamp > m.timestamp
            AND r.is_bot_message = 1
        )
      ORDER BY m.timestamp DESC
      LIMIT 10
    ''').fetchall()
    for r in rows: print(r)
    print(f'total={len(rows)}')
    conn.close()
except sqlite3.Error as e:
    print('ERROR: DB query failed: ' + str(e))
    sys.exit(1)
"
```

**If the script exits with an error:** report the failure and stop.

**If total = 0:** all clear, return nothing.

## Triage

For each unanswered message, evaluate the content and decide:

### 1. Expired / no longer actionable

The request was time-sensitive and the window has passed (e.g., "what's my next meeting?" and the meeting already ended; "remind me in 5 minutes" from 20 min ago).

**Action:** Send a brief note acknowledging you missed it and what the answer would have been.

### 2. Still actionable

The task has no hard deadline or the deadline hasn't passed (research, write an email, find something, general questions).

**Action:** Just go do it — process the message as if it just arrived and send the response normally. Do NOT report it as an issue.

### 3. Unclear

Can't determine if expired or not.

**Action:** Report it — include sender, chat, message content, and note the ambiguity.

## Output

Return the list of triage actions taken, or empty if nothing to do.
