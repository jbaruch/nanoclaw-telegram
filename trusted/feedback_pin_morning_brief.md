---
name: Pin morning brief
description: Always pin the morning brief message so it doesn't get lost
type: feedback
---

Always pin the morning brief message using `pin: true` in `mcp__nanoclaw__send_message`.

**Why:** The morning brief gets buried in chat noise.

**How to apply:** When sending the morning brief, add `pin: true` to the send_message call.
