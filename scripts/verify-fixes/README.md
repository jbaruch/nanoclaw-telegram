# Regression watchdogs for #493 / #497 / #451

AyeAye-native precheck scripts that run inside an admin-tile (main group) agent container as scheduled-task scripts. Each detects a specific regression mode for the fixes shipped in PRs #518 / #519 / #520 / #522 and only wakes the LLM when a regression is detected — silent no-ops cost zero model tokens.

## Scripts

| Script | Cadence | Wakes the agent when... |
|---|---|---|
| `verify-493-precheck.py` | hourly (`5 * * * *`) | non-main inbound traffic happened in the last 60 min but zero `tier:'classifier'` records landed in `usage.jsonl` |
| `verify-497-precheck.py` | every 10 min (`*/10 * * * *`) | a deploy-kill happened in the last 24 h AND the orchestrator log has zero `Pre-shutdown checkpoint pass complete` lines (regressed back to pre-#519 SIGTERM behaviour) |
| `verify-451-precheck.py` | daily (`30 9 * * *`) | the `list_learned_triggers` IPC silently regressed (no result, malformed JSON, missing `groups` array, or handler-returned `error`) |

All three follow `coding-policy: script-delegation`'s precheck-gating contract: emit `{"wake_agent": true|false, "data": {...}}` as the last line of stdout, always exit 0 (even on internal error) so a parser failure doesn't silently disable the watchdog.

## What the scripts read

| Path inside the container | Source (host) | Mount |
|---|---|---|
| `/workspace/host-logs/usage.jsonl` | `~/nanoclaw/logs/usage.jsonl` | added in PR #523 |
| `/workspace/host-logs/orchestrator.log` | `~/nanoclaw/data/host-logs/orchestrator.log` | existing main-tier mount |
| `/workspace/host-logs/deploy-kills.log` | `~/nanoclaw/data/host-logs/deploy-kills.log` | existing main-tier mount |
| `/workspace/store/messages.db` | `~/nanoclaw/store/messages.db` | existing trusted/main mount |
| `/workspace/ipc/<main-folder>/{tasks,input-default}` | `~/nanoclaw/data/ipc/<main-folder>/...` | existing main-tier mount |

## Install (post-deploy)

Three `mcp__nanoclaw__schedule_task` calls from the main group register the precheck rows. The `script` field on each task is what `agent-runner` writes to `/tmp/task-script.sh` and runs as bash before the LLM. Each calls into the bundled Python file under `/workspace/project/scripts/verify-fixes/`.

```jsonc
// 493 — hourly
{
  "type": "schedule_task",
  "prompt": "Regression detected on the #493 watchdog. Read $WAKE_DATA from the precheck output, post a one-line summary to the main chat, and file an issue if classifier_count remains 0 across two consecutive cycles.",
  "schedule_type": "cron",
  "schedule_value": "5 * * * *",
  "script": "python3 /workspace/project/scripts/verify-fixes/verify-493-precheck.py"
}

// 497 — every 10 min
{
  "type": "schedule_task",
  "prompt": "Regression detected on the #497 watchdog. The orchestrator log shows no `Pre-shutdown checkpoint pass complete` line in the last 24 h despite a deploy-kill. Post a one-line summary to the main chat.",
  "schedule_type": "cron",
  "schedule_value": "*/10 * * * *",
  "script": "python3 /workspace/project/scripts/verify-fixes/verify-497-precheck.py"
}

// 451 — daily 09:30
{
  "type": "schedule_task",
  "prompt": "Regression detected on the #451 watchdog. The list_learned_triggers IPC failed the daily smoke test ($WAKE_DATA shows the failure mode). Post a one-line summary to the main chat.",
  "schedule_type": "cron",
  "schedule_value": "30 9 * * *",
  "script": "python3 /workspace/project/scripts/verify-fixes/verify-451-precheck.py"
}
```

The agent receives `WAKE_DATA` as a JSON env var with the precheck's `data` payload, so the LLM has the regression details directly without a second fetch.

## Why AyeAye-native, not host cron

PR #521's first cut was three host bash scripts wired via crontab. That worked but had two problems:
1. Synology cron requires root to install entries via `crontab -`, and `crontab` on this NAS is not SUID — the orchestrator owner couldn't install the entries.
2. Verification-as-cron is decoupled from the rest of NanoClaw's task scheduling — operators have one place to manage scheduled work (`mcp__nanoclaw__list_tasks`), and watchdogs that bypass it are footguns.

The `logs/usage.jsonl` mount added in PR #523's `src/container-runner.ts` change is the small piece that makes the in-container path possible.

## Retirement

These exist to catch post-deploy regressions during the observation window. After a few clean weeks, fire `cancel_task` against each of the three scheduled-task IDs and remove the script files in a follow-up PR. Pattern mirrors #492.
