#!/usr/bin/env python3
"""
Scan recent bot messages across all chats for forbidden internal-monologue phrases.
Outputs a JSON array of violations: [{phrase, message_id, chat_jid, chat_name, timestamp, preview}]
"""
import sqlite3
import json
from datetime import datetime, timezone, timedelta

DB = '/workspace/store/messages.db'

FORBIDDEN = [
    "No response requested",
    "No new messages to respond to",
    "There's nothing to continue from in this session",
    "it looks like this is a fresh context",
    "Proceeding with",
    "Starting work on",
    "Начинаю работу",
    "Сейчас сделаю",
    "All clear",
    "Everything looks good",
    "Продолжаю",
    "Работаю над",
    "I'll now",
    "Now I will",
    # "not directed at me" family
    "not directed at me",
    "No action needed",
    "No action required",
    "Not mine to answer",
    "Casual group chat",
    "not for me",
    "nothing for me to do",
    "это не мне",
    "не направлено мне",
    "Молчу",
    "Не мне",
    "Group chat, not",
    "This message from",
    "Conversation between",
    # narrated silence
    "*stays silent*",
    "*silent*",
]

cutoff = (datetime.now(timezone.utc) - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%S")

try:
    conn = sqlite3.connect(DB, timeout=5)
    rows = conn.execute(
        """
        SELECT m.id, m.chat_jid, m.content, m.timestamp, c.name
        FROM messages m
        LEFT JOIN chats c ON c.jid = m.chat_jid
        WHERE m.is_from_me = 1
          AND m.timestamp > ?
        ORDER BY m.timestamp ASC
        """,
        (cutoff,)
    ).fetchall()
    conn.close()
except sqlite3.OperationalError as e:
    print(json.dumps([]))
    raise SystemExit(1)

violations = []
for msg_id, chat_jid, content, ts, chat_name in rows:
    if not content:
        continue
    content_lower = content.lower()
    for phrase in FORBIDDEN:
        if phrase.lower() in content_lower:
            violations.append({
                "phrase": phrase,
                "message_id": msg_id,
                "chat_jid": chat_jid,
                "chat_name": chat_name or chat_jid,
                "timestamp": ts,
                "preview": content[:80],
            })
            break  # one violation per message

print(json.dumps(violations))
