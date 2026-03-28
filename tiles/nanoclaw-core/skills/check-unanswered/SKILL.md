---
name: check-unanswered
description: Check for user messages that haven't been answered (text reply OR reaction from bot)
---

# Check Unanswered Messages

Find user messages since last check that have no text reply AND no bot reaction.

## Logic

A message is considered answered if:
1. There is a text reply from the bot (`is_from_me=1`) in the same chat with a later timestamp
2. OR there is a reaction from the bot in the `reactions` table (`reactor_jid = 'bot@telegram'` on that message)

## Code

```python
import sqlite3, json

STATE_FILE = '/workspace/group/nanoclaw-state.json'
DB = '/workspace/store/messages.db'

state = json.load(open(STATE_FILE))
last_id = state.get('unanswered_last_checked_id', 0)

conn = sqlite3.connect(DB)
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
conn.close()

new_cursor = max_id or last_id
state['unanswered_last_checked_id'] = new_cursor
# Remove legacy reacted_to if present
state.pop('reacted_to', None)
with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

# unanswered = list of (id, sender_name, content, timestamp) needing a response
```

## Output

Return `unanswered` list to caller. If empty, return nothing.
