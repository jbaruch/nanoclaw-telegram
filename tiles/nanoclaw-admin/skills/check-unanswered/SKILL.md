---
name: check-unanswered
description: Scans a Telegram-backed SQLite message store for user messages that have received no bot text reply, returning a list of unanswered threads with sender, content, and timestamp. Use when the user asks to find unanswered messages, check for pending replies, audit response coverage, review missed messages, or identify unresponded threads in the message history.
---

# Check Unanswered Messages

Find user messages since last check that have no text reply from the bot.

## Logic

A message is considered answered if:
1. There is a text reply from the bot (`is_from_me=1`) in the same chat with a later timestamp

**Bot reactions do NOT count as answers.** A reaction is an ACK — it confirms the message was seen, not that it was responded to. Only an actual text reply constitutes an answer.

## Current chat detection

Detect which chat this heartbeat is running for by finding the most recent bot message:
```sql
SELECT chat_jid FROM messages WHERE is_from_me=1 ORDER BY timestamp DESC LIMIT 1
```
Use that `chat_jid` to scope all queries.

## Schema validation

Before executing, confirm the `reactions` table exists (used only to verify schema integrity, not for answer detection):
```sql
SELECT name FROM sqlite_master WHERE type='table' AND name='reactions'
```
If the `reactions` table is missing, log a warning but continue — answer detection only requires the `messages` table.

## Code

```python
import sqlite3, json, os

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
    # Schema integrity check
    has_reactions = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reactions'"
    ).fetchone()
    if not has_reactions:
        import sys
        print("WARNING: reactions table not found — proceeding with text-reply-only detection", file=sys.stderr)

    # Detect current chat from most recent bot message
    row = conn.execute(
        "SELECT chat_jid FROM messages WHERE is_from_me=1 ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()

    if not row:
        unanswered = []
    else:
        current_chat = row[0]

        # Per-chat timestamp cursor (avoids cross-chat ID collision bug)
        cursors = state.get('unanswered_cursors', {})
        last_ts = cursors.get(current_chat, '1970-01-01T00:00:00.000Z')

        try:
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

Return `unanswered` list to caller. If empty, return nothing.
