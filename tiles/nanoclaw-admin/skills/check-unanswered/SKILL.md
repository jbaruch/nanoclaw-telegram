---
name: check-unanswered
description: Scans a Telegram-backed SQLite message store for user messages that have received neither a bot text reply nor a bot reaction, returning a list of unanswered threads with sender, content, and timestamp. Use when the user asks to find unanswered messages, check for pending replies, audit response coverage, review missed messages, or identify unresponded threads in the message history.
---

# Check Unanswered Messages

Find user messages since last check that have no text reply AND no bot reaction.

## Logic

A message is considered answered if:
1. There is a text reply from the bot (`is_from_me=1`) in the same chat with a later timestamp
2. OR there is a reaction from the bot in the `reactions` table (`reactor_jid = 'bot@telegram'` on that message)

## Code

```python
import sqlite3, json, os

STATE_FILE = '/workspace/group/nanoclaw-state.json'
DB = '/workspace/store/messages.db'

# Load state file, defaulting gracefully if missing
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        state = json.load(f)
else:
    state = {}
last_id = state.get('unanswered_last_checked_id', 0)

try:
    conn = sqlite3.connect(DB, timeout=5)
    rows = conn.execute('''
      SELECT m.id, m.sender_name, m.content, m.timestamp
      FROM messages m
      WHERE CAST(m.id AS INTEGER) > ?
        AND m.is_from_me = 0
        AND NOT EXISTS (
          SELECT 1 FROM messages r
          WHERE r.chat_jid = m.chat_jid
            AND r.is_from_me = 1
            AND r.timestamp > m.timestamp
        )
        AND NOT EXISTS (
          SELECT 1 FROM reactions rx
          WHERE rx.message_id = m.id
            AND rx.reactor_jid = 'bot@telegram'
        )
      ORDER BY CAST(m.id AS INTEGER) ASC
    ''', (last_id,)).fetchall()

    max_id = conn.execute(
      'SELECT MAX(CAST(id AS INTEGER)) FROM messages WHERE CAST(id AS INTEGER) > ? AND is_from_me = 0',
      (last_id,)
    ).fetchone()[0]
finally:
    conn.close()

new_cursor = max_id or last_id
state['unanswered_last_checked_id'] = new_cursor
# Remove legacy reacted_to if present
state.pop('reacted_to', None)

with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

# Validate the cursor was persisted correctly
with open(STATE_FILE) as f:
    saved = json.load(f)
assert saved.get('unanswered_last_checked_id') == new_cursor, "State file cursor mismatch after write"

# unanswered = list of (id, sender_name, content, timestamp) needing a response
```

## Output

Return `unanswered` list to caller. If empty, return nothing.
