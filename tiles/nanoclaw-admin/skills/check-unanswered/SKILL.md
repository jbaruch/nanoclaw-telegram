---
name: check-unanswered
description: Finds user messages that never got a bot reply. Uses gap detection — a message is unanswered if no bot reply appeared within 10 minutes after it. Deterministic script, no LLM reasoning for detection. Use when performing heartbeat checks or after session recovery.
---

# Check Unanswered Messages

Run the deterministic detector script:

```bash
python3 /workspace/group/scripts/check-unanswered.py
```

The script outputs JSON:
```json
{
  "unanswered": [
    {"id": "123", "sender_name": "Leonid (@ligolnik)", "content": "...", "timestamp": "..."}
  ],
  "chat_jid": "tg:...",
  "lookback_hours": 24,
  "reply_window_minutes": 10,
  "checked_at": "..."
}
```

## If empty: silence

**If `unanswered` is empty: return nothing. No output, no message, no acknowledgement.**

## If non-empty: respond to each

For each unanswered message:
1. React with 👌: `mcp__nanoclaw__react_to_message(messageId: "<id>", emoji: "👌")`
2. Reply with judgment — consider whether it's still actionable, trivial, or too late
3. Thread correctly: `mcp__nanoclaw__send_message(reply_to: "<id>")`

## Environment variables

The script accepts overrides:
- `NANOCLAW_CHAT_JID` — chat to check (auto-detected if unset)
- `LOOKBACK_HOURS` — how far back to look (default: 24)
- `REPLY_WINDOW_MINUTES` — how long to wait for a reply before flagging (default: 10)
