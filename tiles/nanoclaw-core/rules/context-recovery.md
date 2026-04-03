# Context Recovery — Never Lose History

## The Rule

**Never say you've lost context or forgotten a previous conversation without first querying `messages.db`.**

The full message history is always available at `/workspace/store/messages.db`. Context compaction removes it from your active context — but the database still has it. There is no excuse for "I don't remember what we discussed" when the database is a query away.

## Required behavior

Before responding with any variant of:
- "Потерял контекст"
- "I don't remember this thread"
- "What were we discussing?"
- "I don't have context on this topic"
- Any acknowledgment that prior conversation is unavailable

You **MUST** first run:

```python
import sqlite3
conn = sqlite3.connect('/workspace/store/messages.db')
rows = conn.execute("""
    SELECT id, timestamp, sender_name, content, is_from_me
    FROM messages
    WHERE chat_jid = (SELECT jid FROM chats LIMIT 1)
      AND content LIKE '%KEYWORD%'
    ORDER BY timestamp DESC
    LIMIT 20
""").fetchall()
for r in rows: print(r)
conn.close()
```

Replace `KEYWORD` with a relevant term from what the user is referencing.

## Database schema (quick reference)

```sql
messages(id, chat_jid, sender, sender_name, content, timestamp, is_from_me, is_bot_message)
chats(jid, name, last_message_time, channel, is_group)
```

- `is_from_me = 1` — messages from the bot (your own responses)
- `is_from_me = 0` — messages from users
- `sender` — numeric user ID (stable across name changes)
- `sender_name` — display name with handle, e.g. `Leonid (@ligolnik)`, `JBáruch (@JBaruch)`
- `content` — full message text

## Connecting people to history

`sender_name` contains both the display name AND the @handle. When someone in the current conversation references past messages ("I told you yesterday"), match their @handle or name against `sender_name`:

```python
# Find what @ligolnik said yesterday
rows = conn.execute("""
    SELECT timestamp, content FROM messages
    WHERE sender_name LIKE '%ligolnik%'
      AND timestamp > datetime('now', '-2 days')
    ORDER BY timestamp DESC LIMIT 10
""").fetchall()
```

This is critical after a session nuke — you have no memory of who said what, but the database does.

## Unanswered message detection

After a session nuke or on first message in a new session, check for messages you never replied to. A message is "answered" only if a bot message exists with `reply_to_message_id` pointing to it. No reply-thread = not an answer.

```python
import sqlite3
conn = sqlite3.connect('/workspace/store/messages.db')
chat_jid = conn.execute("SELECT jid FROM chats LIMIT 1").fetchone()[0]

unanswered = conn.execute("""
    SELECT m.id, m.sender_name, m.content, m.timestamp
    FROM messages m
    WHERE m.chat_jid = ?
      AND m.is_from_me = 0
      AND m.is_bot_message = 0
      AND m.timestamp > datetime('now', '-24 hours')
      AND NOT EXISTS (
        SELECT 1 FROM messages r
        WHERE r.chat_jid = m.chat_jid
          AND r.is_from_me = 1
          AND r.reply_to_message_id = m.id
      )
    ORDER BY m.timestamp ASC
""", (chat_jid,)).fetchall()
conn.close()

for msg_id, sender, content, ts in unanswered:
    print(f"UNANSWERED: [{ts}] {sender}: {content[:80]}")
```

If you find unanswered messages: acknowledge the gap and respond to any that are still actionable. Don't pretend they didn't happen.

## When to use

- User references something from an earlier session that's not in active context
- User says "ты говорил..." (you said...) and you don't have it in context
- Someone says "I told you" / "we discussed" / "yesterday I asked" — match their handle to DB history
- Any "I don't remember" impulse — check first
- After context compaction (the summary will mention "continued from previous session")
- First message after a nuke — check for unanswered messages from before the nuke

## This is a hard requirement

Claiming lost context without checking the database is a failure mode equivalent to fabrication. The information exists. Retrieve it.
