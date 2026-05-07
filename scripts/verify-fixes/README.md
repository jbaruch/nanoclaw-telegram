# Verification scripts for #493 / #497 / #451

Host-side regression watchdogs for the three fixes shipped in
PRs #518 / #519 / #520. They run on the NAS via cron, read host
paths directly, and post Telegram alerts (via the same bot used by
`heartbeat-external.sh`) when a regression is detected.

| Script | Cadence | Detects |
|---|---|---|
| `verify-493-classifier-emit.sh` | hourly | non-main inbound traffic but zero `tier:'classifier'` records in `usage.jsonl` |
| `verify-497-shutdown-checkpoint.sh` | every 10 min | recent `deploy-kills.log` entry without a matching `**Trigger:** shutdown` checkpoint file |
| `verify-451-list-learned-triggers.sh` | daily | `list_learned_triggers` IPC silently regressed (no response, malformed JSON, missing `groups` array) |

## Install

Add to NAS crontab (`crontab -e`) under the `jbaruch` user:

```cron
# Verify the three fixes shipped 2026-05-07
5 * * * * /home/jbaruch/nanoclaw/scripts/verify-fixes/verify-493-classifier-emit.sh >> /home/jbaruch/nanoclaw/logs/verify-fixes.log 2>&1
*/10 * * * * /home/jbaruch/nanoclaw/scripts/verify-fixes/verify-497-shutdown-checkpoint.sh >> /home/jbaruch/nanoclaw/logs/verify-fixes.log 2>&1
30 9 * * * /home/jbaruch/nanoclaw/scripts/verify-fixes/verify-451-list-learned-triggers.sh >> /home/jbaruch/nanoclaw/logs/verify-fixes.log 2>&1
```

## Why host cron, not AyeAye scheduled tasks

The simpler shape was a `mcp__nanoclaw__schedule_task` payload that ran inside an agent container, but `logs/usage.jsonl` is not bind-mounted into agent containers (only `data/host-logs/` is). Adding the mount is a separate change touching `src/container-runner.ts`; running host-side avoids that scope creep and matches the precedent set by `scripts/heartbeat-external.sh`.

## Removing the watchdogs

These scripts exist to catch regressions during the post-deploy observation window. After a few clean weeks, retire them per the same pattern as #492 (the usage-log structural-diag retirement). Track the retirement in a follow-up issue rather than letting them rot indefinitely.
