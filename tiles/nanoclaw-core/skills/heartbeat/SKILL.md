---
name: heartbeat
description: Periodic health check orchestrator for NanoClaw. Runs ALL checks — system health, unanswered messages, calendar, email, disk, logs, sessions. The only external watchdog is a liveness check (is NanoClaw running?). Triggers on "heartbeat", "health check", "system status".
---

# Heartbeat

You are running as a periodic health check. Invoke each sub-check below, run the inline checks, collect results, and ONLY message the user if something is wrong. Silent when healthy.

## Skill checks

Invoke each of the following as skill calls and collect their results.

1. **System health** (`/check-system-health`) — stuck tasks, DB size, task failures (DB at /workspace/store/messages.db)
2. **Unanswered messages** (`/check-unanswered`) — find and triage unanswered messages (DB at /workspace/store/messages.db)
3. **Calendar changes** (`/check-calendar`) — detect changed events and reschedule reminders
4. **Email triage** (`/check-email`) — fetch and classify new emails with source calibration

## Inline checks

Run these directly (no sub-skill needed):

### Disk space

```bash
df -h /workspace/group/ | awk 'NR==2 {print $5, $4}'
```

**Alert if:** usage > 80%. **Critical if:** > 95%.

### Log growth

```bash
du -sh /workspace/group/logs/ 2>/dev/null
find /workspace/group/logs/ -type f -size +50M 2>/dev/null
```

**Auto-fix:** Truncate files > 50MB to last 10k lines.

### Session bloat

```bash
du -sh /home/node/.claude/projects/ 2>/dev/null
find /home/node/.claude/projects/ -name '*.jsonl' -mtime +7 2>/dev/null | wc -l
```

**Auto-fix:** Delete session transcripts older than 7 days, keep latest 5 per group.

### Stuck IPC close files

```bash
find /workspace/ipc/input -name '_close' -mmin +30 2>/dev/null
```

**Auto-fix:** Delete stuck `_close` files.

### Orphaned containers (via bash)

```bash
docker ps -a --filter "name=nanoclaw-" --filter "status=exited" --format '{{.Names}}' 2>/dev/null | wc -l
```

**Alert if:** > 5 orphaned containers.

## Sub-check error handling

If a sub-check fails to respond, returns an error, or times out, treat that as a failure and include it in the consolidated report — do **not** silently skip it.

## Output format

If ALL checks pass: wrap your entire output in `<internal>` tags. Say nothing to the user.

```
<internal>Heartbeat: all clear.</internal>
```

If ANY check returns issues, send a single consolidated message with only the failures:

```
*Heartbeat*

• Stuck tasks: 2 overdue -> reset
• Disk: 87% used
• New email from John: "Re: schedule" -- Can you confirm...
```

Keep it short. No preamble. No "I ran a health check and found..." — just the problems.
