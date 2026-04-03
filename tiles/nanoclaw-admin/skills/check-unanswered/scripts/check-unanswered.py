#!/usr/bin/env python3
"""
Deterministic unanswered message detector.

Finds user messages that never got a bot reply within a reasonable window.
Unlike the naive "any later bot message = answered" approach, this uses
gap detection: a message is unanswered if no bot reply appeared within
REPLY_WINDOW_MINUTES after it, AND no bot message explicitly quotes it.

Outputs JSON to stdout. Empty list = nothing to report.
"""

import sqlite3
import json
import sys
import os
from datetime import datetime, timedelta, timezone

DB = os.environ.get('NANOCLAW_DB', '/workspace/store/messages.db')
CHAT_JID = os.environ.get('NANOCLAW_CHAT_JID', '')
LOOKBACK_HOURS = int(os.environ.get('LOOKBACK_HOURS', '24'))
REPLY_WINDOW_MINUTES = int(os.environ.get('REPLY_WINDOW_MINUTES', '15'))

if not CHAT_JID:
    # Fall back to detecting current chat from most recent bot message
    try:
        conn = sqlite3.connect(DB, timeout=5)
        row = conn.execute(
            "SELECT chat_jid FROM messages WHERE is_from_me=1 ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            CHAT_JID = row[0]
        else:
            print(json.dumps({"unanswered": [], "error": "no bot messages found"}))
            sys.exit(0)
    except Exception as e:
        print(json.dumps({"unanswered": [], "error": str(e)}))
        sys.exit(0)

try:
    conn = sqlite3.connect(DB, timeout=5)
except sqlite3.OperationalError as e:
    print(json.dumps({"unanswered": [], "error": f"DB open failed: {e}"}))
    sys.exit(0)

cutoff = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime('%Y-%m-%dT%H:%M:%S')

# Get all user messages in the lookback window
user_msgs = conn.execute("""
    SELECT id, sender_name, content, timestamp
    FROM messages
    WHERE chat_jid = ?
      AND is_from_me = 0
      AND is_bot_message = 0
      AND timestamp > ?
    ORDER BY timestamp ASC
""", (CHAT_JID, cutoff)).fetchall()

# Get all bot messages in the lookback window (plus some buffer for replies to late messages)
bot_msgs = conn.execute("""
    SELECT id, content, timestamp
    FROM messages
    WHERE chat_jid = ?
      AND is_from_me = 1
      AND timestamp > ?
    ORDER BY timestamp ASC
""", (CHAT_JID, cutoff)).fetchall()

conn.close()

# Build a set of bot message timestamps for window checking
bot_timestamps = []
for _, _, ts in bot_msgs:
    try:
        bot_timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
    except (ValueError, AttributeError):
        pass

# Also check if bot messages quote/reply to specific user messages
# Bot replies often contain "[Replying to ..." which references the user message
bot_contents = [content or '' for _, content, _ in bot_msgs]

unanswered = []
for msg_id, sender, content, ts in user_msgs:
    try:
        msg_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        continue

    window_end = msg_time + timedelta(minutes=REPLY_WINDOW_MINUTES)

    # Check if any bot message falls within the reply window after this message
    has_reply_in_window = any(
        msg_time < bt <= window_end
        for bt in bot_timestamps
    )

    if has_reply_in_window:
        continue

    # Check if any later bot message explicitly references this message's content
    # (crude but catches quote-replies)
    content_snippet = (content or '')[:40]
    has_quote_reply = any(
        content_snippet and content_snippet in bc
        for bc in bot_contents
    )

    if has_quote_reply:
        continue

    unanswered.append({
        "id": msg_id,
        "sender_name": sender,
        "content": content,
        "timestamp": ts
    })

print(json.dumps({
    "unanswered": unanswered,
    "chat_jid": CHAT_JID,
    "lookback_hours": LOOKBACK_HOURS,
    "reply_window_minutes": REPLY_WINDOW_MINUTES,
    "checked_at": datetime.now(timezone.utc).isoformat()
}))
