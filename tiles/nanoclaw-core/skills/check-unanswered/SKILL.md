---
name: check-unanswered
description: Scans a Telegram-backed SQLite message store for user messages that have received no bot text reply, returning a list of unanswered threads with sender, content, and timestamp. Use when the user asks to find unanswered messages, check for pending replies, audit response coverage, review missed messages, or identify unresponded threads in the message history.
---

# Check Unanswered Messages

Find user messages since last check that have no text reply from the bot.

## Code

```python
import sqlite3, json, os, sys

STATE_FILE = '/workspace/group/nanoclaw-state.json'
DB = '/workspace/store/messages.db'

# Load state file, defaulting gracefully if missing or corrupted
try:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
    else:
        state = {}
except (json.JSONDecodeError, OSError):
    state = {}

try:
    conn = sqlite3.connect(DB, timeout=5)
except sqlite3.OperationalError as e:
    raise RuntimeError(f"Could not open message DB: {e}")

try:
    has_reactions = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reactions'"
    ).fetchone()
    if not has_reactions:
        print("WARNING: reactions table not found — proceeding with text-reply-only detection", file=sys.stderr)

    # Detect current chat from the most recent bot message, then scope all queries to it
    row = conn.execute(
        "SELECT chat_jid FROM messages WHERE is_from_me=1 ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()

    if not row:
        unanswered = []
    else:
        current_chat = row[0]

        cursors = state.get('unanswered_cursors', {})
        last_ts = cursors.get(current_chat, '1970-01-01T00:00:00.000Z')

        try:
            # A message is answered if any later is_from_me=1 message exists in the same chat
            rows = conn.execute('''
              SELECT m.id, m.sender_name, m.content, m.timestamp
              FROM messages m
              WHERE m.chat_jid = ?
                AND m.timestamp > ?
                AND m.is_from_me = 0
                AND NOT EXISTS (
                  SELECT 1 FROM messages r
                  WHERE r.chat_jid = m.chat_jid
                    AND r.is_from_me = 1
                    AND r.timestamp > m.timestamp
                )
              ORDER BY m.timestamp ASC
            ''', (current_chat, last_ts)).fetchall()
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Query failed — possible schema mismatch: {e}")

        # Advance cursor to most recent user message in this chat
        max_ts = conn.execute(
            "SELECT MAX(timestamp) FROM messages WHERE chat_jid = ? AND is_from_me = 0",
            (current_chat,)
        ).fetchone()[0]

        cursors[current_chat] = max_ts or last_ts
        state['unanswered_cursors'] = cursors
        # Remove legacy global cursor if present
        state.pop('unanswered_last_checked_id', None)

        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except OSError as e:
            raise RuntimeError(f"Failed to persist state file: {e}")

        unanswered = rows

finally:
    conn.close()
```

## Output

Return `unanswered` list to caller. **If empty: return nothing. Send NO message, output NO text, produce NO acknowledgement. "No new messages to respond to at this time." is a forbidden phrase — never output it. Silence is the correct and complete response when there is nothing to report.**

Each entry in `unanswered` is a tuple of `(id, sender_name, content, timestamp)`, where `id` is the message row ID, `sender_name` is the display name of the user, `content` is the raw message text, and `timestamp` is an ISO-8601 string.

## Validation

After retrieving results, confirm no returned message has a later bot reply in the same chat by running:

```sql
SELECT m.id, m.timestamp, r.timestamp AS bot_reply_ts
FROM messages m
JOIN messages r ON r.chat_jid = m.chat_jid
                AND r.is_from_me = 1
                AND r.timestamp > m.timestamp
WHERE m.id IN (<comma-separated ids from unanswered>)
LIMIT 10;
```

This query should return zero rows. Any match indicates the main query missed a reply and the result set should be discarded for investigation.
