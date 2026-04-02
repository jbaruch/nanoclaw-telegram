---
name: No naked text between tool calls
description: Any text output (including "No response requested.", transition narration, etc.) is streamed to Telegram. Wrap it in <internal> or don't write it.
type: feedback
---

Do NOT write any plain text outside of tool calls unless it's a real message to the user.

**Why:** Progressive streaming sends ALL text output to Telegram in real time. "No response requested.", "Now updating the file:", "Let me check:" — all of it lands in the chat as messages.

**How to apply:**
- If there's nothing to say to the user: write nothing (or wrap in `<internal>`)
- Transition phrases between tool calls: wrap in `<internal>` or delete entirely
- "No response requested." — delete, just be silent
- Only write bare text when it IS the message to the user
