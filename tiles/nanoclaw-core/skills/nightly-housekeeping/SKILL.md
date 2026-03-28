---
name: nightly-housekeeping
description: Nightly maintenance automation skill that syncs TripIt calendar data to Reclaim, detects overlapping trips, and applies timezone corrections. Reports only failures or notable findings via Telegram; silence means success. Use when scheduled nightly maintenance is triggered, when the user requests a manual nightly run, or when overnight batch jobs, cron tasks, or a maintenance window should be executed.
---

# Nightly Housekeeping

Runs at 3am every day. Execute all tasks below in order. Report only failures or notable findings via mcp__nanoclaw__send_message — silence is success.

## Tasks

### 1. TripIt → Reclaim Sync

Invoke: `Skill(skill: "tessl__sync-tripit")`

Report if:
- Timezone changes were applied (summarize what changed)
- Overlapping trips detected (flag with trip names and dates)
- Fatal error (report with context)

Silence if no changes.

**Error recovery:** If the skill invocation fails or times out, retry once. If it fails again, report the error via mcp__nanoclaw__send_message with the task name and any available error context, then continue to the next task.

---

## Output rules

- Use mcp__nanoclaw__send_message only when there's something worth reporting
- Start each alert with the task name: e.g. "*TripIt sync:* 2 timezone changes applied"
- All-clear = no message at all
- Telegram formatting: *bold* (single asterisks), • bullets
