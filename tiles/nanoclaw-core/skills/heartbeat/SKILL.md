---
name: heartbeat
description: Periodic health check orchestrator for NanoClaw. Invokes sub-checks (unanswered messages, calendar, email), aggregates results, reports only failures. Infrastructure checks (DB, logs, containers, disk) are handled by the external heartbeat on the host. Triggers on "heartbeat", "health check".
---

# Heartbeat

You are running as a periodic health check. Invoke each sub-check skill below, collect results, and ONLY message the user if something is wrong. Silent when healthy.

**Note:** Infrastructure checks (stuck tasks, DB size, logs, sessions, IPC, disk, containers, OneCLI) are handled by the external heartbeat script on the host. This agent heartbeat covers only checks that need agent intelligence or API access.

## Checks to run

Invoke each of the following as skill calls and collect their results.

1. **System health** (`/check-system-health`) — stuck tasks, DB size, task failures (DB at /workspace/store/messages.db)
2. **Unanswered messages** (`/check-unanswered`) — find and triage unanswered messages
3. **Calendar changes** (`/check-calendar`) — detect changed events and reschedule reminders
4. **Email triage** (`/check-email`) — fetch and classify new emails with source calibration

Note: Host-level checks (Docker containers, disk space, orphaned containers) remain in the external heartbeat script.

## Sub-check error handling

If a sub-check fails to respond, returns an error, or times out, treat that as a failure and include it in the consolidated report — do **not** silently skip it.

```
• /check-email: sub-check failed to respond (timeout)
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

• New email from John Smith: "Re: conference schedule" -- Can you confirm...
• Calendar changed: standup moved to 2pm -> reminder rescheduled
```

Keep it short. No preamble. No "I ran a health check and found..." — just the problems.
