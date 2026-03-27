---
name: heartbeat
description: Periodic health check orchestrator for NanoClaw. Invokes sub-checks (system health, unanswered messages, calendar, email), aggregates results, reports only failures. Use when running as a scheduled heartbeat task. Triggers on "heartbeat", "health check", "system status".
---

# Heartbeat

You are running as a periodic health check. Invoke each sub-check skill below, collect results, and ONLY message the user if something is wrong. Silent when healthy.

## Checks to run

Invoke each of the following as skill calls (slash-command skill invocations, not tool calls or MCP resources) and collect their results.

1. **System health** (`/check-system-health`) — stuck tasks, IPC errors, DB size, logs, session bloat, retry exhaustion, IPC close files, OneCLI health
2. **Unanswered messages** (`/check-unanswered`) — messages that received no bot reply within 5-15 minutes
3. **Calendar changes** (`/check-calendar`) — detect and reschedule reminders for changed events
4. **Email triage** (`/check-email`) — fetch and classify new emails with source calibration

## Sub-check error handling

If a sub-check fails to respond, returns an error, or times out, treat that as a failure and include it in the consolidated report — do **not** silently skip it.

```
• /check-system-health: sub-check failed to respond (timeout)
```

Never suppress a sub-check error. A partial run should be reported, not hidden.

## Output format

If ALL checks pass: wrap your entire output in `<internal>` tags. Say nothing to the user.

```
<internal>Heartbeat: all clear.</internal>
```

If ANY check returns issues (including sub-check errors), send a single consolidated message with only the failures:

```
*Heartbeat*

• Stuck tasks: 2 tasks overdue -> reset next_run (heartbeat, task-abc123)
• Logs: nanoclaw.log was 67MB -> truncated to 1.2MB
• New email from John Smith: "Re: conference schedule" -- Can you confirm...
• Calendar changed: standup moved to 2pm -> reminder rescheduled
```

Keep it short. No preamble. No "I ran a health check and found..." — just the problems.
