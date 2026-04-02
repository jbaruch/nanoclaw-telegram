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
import sqlite3, json
conn = sqlite3.connect('/workspace/store/messages.db')
rows = conn.execute("""
    SELECT id, timestamp, sender_type, content
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
messages(id, chat_jid, sender_jid, sender_type, content, timestamp, ...)
chats(jid, name, ...)
```

- `sender_type = 'user'` — messages from Baruch
- `sender_type = 'assistant'` — your own responses
- `content` — full message text

## When to use

- User references something from an earlier session that's not in active context
- User says "ты говорил..." (you said...) and you don't have it in context
- Any "I don't remember" impulse — check first
- After context compaction (the summary will mention "continued from previous session")

## This is a hard requirement

Claiming lost context without checking the database is a failure mode equivalent to fabrication. The information exists. Retrieve it.
