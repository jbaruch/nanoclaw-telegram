---
name: check-unanswered
description: Find messages that received no bot reply within 5-15 minutes and triage them. Expired requests get a brief acknowledgement, actionable ones get processed immediately, unclear ones get reported. Use as part of heartbeat or standalone. Triggers on "check unanswered", "missed messages", "unreplied messages".
---

# Check Unanswered Messages

## Detection

```bash
python3 -c "
import sqlite3, sys, os
db_path = '/workspace/project/store/messages.db'
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
except sqlite3.Error as e:
    print('ERROR: DB query failed: ' + str(e))
    sys.exit(1)
"
```

**If the script exits with an error:** report the failure (missing DB, schema mismatch, etc.) and stop — do not proceed with triage.

**If total = 0:** all clear, return nothing.

## Triage

For each unanswered message, investigate logs for why it was missed (container crash? timeout? error?), then evaluate the message content and decide:

### 1. Expired / no longer actionable

The request was time-sensitive and the window has passed (e.g., "what's my next meeting?" and the meeting already ended; "remind me in 5 minutes" from 20 min ago).

**Action:** Send a brief note acknowledging you missed it and what the answer would have been. After sending, verify the acknowledgement appears in the message store (check that a new `is_bot_message = 1` row exists for that `chat_jid` with a timestamp after the original message).

### 2. Still actionable

The task has no hard deadline or the deadline hasn't passed (research, write an email, find something, general questions).

**Action:** Just go do it — process the message as if it just arrived and send the response normally. After sending, confirm the reply is recorded in the message store before marking this item resolved. Do NOT report it as an issue.

### 3. Unclear

Can't determine if expired or not.

**Action:** Report it — include sender, chat, message content, log diagnosis, and note the ambiguity.

## Output

Return the list of issues found (with triage actions taken) or empty if all clear.
