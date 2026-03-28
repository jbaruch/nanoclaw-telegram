---
name: nightly-housekeeping
description: Nightly maintenance tasks. Runs at 3am daily. Executes async background jobs that shouldn't run during the day — syncs, cleanups, and batch operations. Reports only failures or notable findings.
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

---

## Output rules

- Use mcp__nanoclaw__send_message only when there's something worth reporting
- Start each alert with the task name: e.g. "*TripIt sync:* 2 timezone changes applied"
- All-clear = no message at all
- Telegram formatting: *bold* (single asterisks), • bullets
