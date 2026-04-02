---
name: heartbeat_silence_on_all_clear
description: Heartbeat must be completely silent when all checks pass — do not send any message
type: feedback
---

Do NOT send any message when heartbeat/system health checks all pass.

**Why:** The rules explicitly state: "If the result is silent (heartbeat all clear), send nothing at all." Sending "all clear" messages is noise and violates the scheduled task exception rule.

**How to apply:** For any scheduled task (heartbeat, morning-brief, reminders) — only send a message if there is something to report (failures, issues, alerts). Complete silence = success. No "all good", no "everything is fine", no summary of what ran.
