---
name: nightly-housekeeping
description: "Runs nightly maintenance for Baruch: syncs TripIt travel timezones to Reclaim, refreshes travel schedule and booking gaps, updates Trakt watch history, checks orders, cleans up undated Google Tasks, generates a daily summary, archives daily logs to weekly memory, and checks the show watchlist. Use when running nightly maintenance, syncing travel plans, cleaning up undated tasks, fixing booking gaps, or running the nightly sync routine. Trigger phrases: 'run nightly housekeeping', 'nightly sync', 'sync travel calendar', 'clean up tasks', 'run nightly sync'."
---

**Every step below is mandatory. Execute them in order. Do not skip, reorder, or abbreviate any step.**

You are AyeAye, Baruch's assistant. Run these nightly maintenance steps silently. Report only if something needs attention.

**Error handling:** Continue through all remaining steps even if one fails. Collect all errors and report them together at the end.

**MANDATORY REPORTING:** Any step that fails and requires host action MUST be reported via `mcp__nanoclaw__send_message`. Silence is for clean runs only.

**File write convention:** After every file write, read the file back to verify contents — confirm pre-existing fields are preserved and new entries appended. Applies to all steps below.

## Step 1: Dedup Check + Optimistic Lock

**This must be the very first action before any API calls or work begins.**

Read `/workspace/group/task-tz-state.json`. Find the entry in `follow_me_tasks` where `name == "nightly-housekeeping"`. Determine today's local date (YYYY-MM-DD in `current_tz`).

**If `last_run_date` already equals today's local date — stop immediately. Do not run any further steps. Output nothing (wrap in `<internal>`).**

Otherwise: Set `last_run_date` to today's local date and write the file back immediately, preserving all other fields and all other entries exactly.

This prevents double-execution when both the scheduled cron task and the heartbeat's missed-task detection fire at the same time (race condition at 3am local time).

If the file cannot be read or written, continue anyway (log the error for Step 17 retry) — do not abort the housekeeping run.

## Step 2: TripIt → Reclaim sync
Run via host: `mcp__nanoclaw__sync_tripit()`
Do NOT call sync.mjs directly — it won't find its modules. The wrapper script handles the correct working directory.
- `noChanges: true` → silent
- Changes detected → report (new timezones, OOO blocks)
- Overlapping trips → flag as warning
- Error → report and continue

## Step 3: Refresh travel schedule
Use `mcp__nanoclaw__run_host_script(script: "refresh-travel-schedule.py")`.
Rebuilds `travel-schedule.json` from the TripIt ICS feed. Silent on success; report only on error.

## Step 4: Travel bookings check
`Skill(skill: "tessl__check-travel-bookings")` — find missing flights/hotels for upcoming trips.
Report gaps; skip if all snoozed or complete.

## Step 5: Refresh Trakt watch history
Use `mcp__nanoclaw__run_host_script(script: "trakt-watch-history.py")`.
Saves fresh watch history to `/workspace/group/trakt-history.json`.
Silent on success. Report on error or `total_shows: 0` (if sync hasn't run yet → skip silently).

## Step 6: Check orders
`Skill(skill: "tessl__check-orders")` — fetch order emails, update orders-db.json, flag anomalies.
- Flagged items → the skill reports them automatically
- Nothing anomalous → stay silent

## Step 7: Refresh CFP data
`Skill(skill: "tessl__check-cfps")` — refresh open CFP data from primary sources, apply Sessionize verification, update `cfp-state.json`.

**This step is research-only.** Do NOT forward the CFP list to Baruch — that is the morning brief's job. The goal here is keeping cfp-state.json current so the morning brief has accurate deadline data.

Consume the skill output internally (do not include in any message to Baruch). If the skill fails completely (both primary sources unreachable), note it in Step 10 daily summary.

## Step 8: YouTube comment check
Search for the YouTube tool via `COMPOSIO_SEARCH_TOOLS` (query: `"youtube list comment threads"`). Use the returned tool to fetch recent comments on Baruch's channel (channel ID: `UCZ8-VX2SiAIBE7guw7NG-Sg`).

1. Fetch videos published in the last 30 days using the video list tool.
2. For each video, fetch comment threads.
3. Filter to comments published in the last 24 hours.
4. If new comments exist → send a summary via `mcp__nanoclaw__send_message`:
   - Video title + link
   - Author + comment text (truncated to 100 chars if long)
   - Group by video
5. No new comments → stay silent.

On Composio tool error → skip silently, note in Step 10 daily summary.

## Step 9: Check for undated tasks
Discover Google Tasks tools per `composio-preamble` rule, then fetch all tasks from "My Tasks" list with no due date (tasks where `due` is absent).

For each undated task:
- Due date **obvious from context** (title mentions a date, known deadline): set silently via GOOGLETASKS_PATCH_TASK.
- Due date **not obvious**: add to `/workspace/group/morning-brief-pending.json` under `undated_tasks` array (fields: `id`, `title`, `tasklist_id`).

Merge with existing file contents — do not overwrite other fields.

## Step 10: Generate daily summary
Write a daily summary to `/workspace/group/memory/daily/YYYY-MM-DD.md` (today's date).
If the file already exists, read it first and append/update — do not overwrite.

Format — include only sections with content:
```markdown
# Daily Summary — YYYY-MM-DD

## Reported to Baruch
## Decisions / Feedback
## Completed
## Follow-up tomorrow
```

Keep entries concise (one line each). This file is read on container startup to restore recent context.

## Step 11: Deduplicate daily logs

Run dedup on both daily log directories:

```bash
python3 /workspace/group/scripts/dedup-memory.py /workspace/group/memory/daily --days 3
python3 /workspace/group/scripts/dedup-memory.py /workspace/trusted/memory/daily --days 3
```

Parse JSON output. Log the count of duplicates removed but do not report to Baruch. If script errors, log and continue.

## Step 12: Archive daily memory (with classification)

For each entry in yesterday's daily log, classify before archiving:

**Permanent** (extract to typed memory file + MEMORY.md index):
- Owner preferences and behavioral feedback
- Architecture decisions and their rationale
- People's roles, relationships, contact info
- Credential scopes, system access, integrations

**Medium-term** (include in weekly summary):
- Task progress, what was done
- Debugging sessions and outcomes
- Conversation summaries

**Short-term** (drop — do NOT include in weekly):
- Scheduling logistics ("meeting at 3pm")
- Transient state ("deploy is running", "waiting for CI")
- Acknowledgments and small talk context

For permanent entries: create/update typed file in `/workspace/trusted/`, add/update MEMORY.md index entry. Also include in weekly for temporal context.

Apply this classification to both group-local and trusted daily logs.

### Shared archival procedure
For each path below, apply this pattern in sequence:
1. Determine yesterday's date. If the daily file doesn't exist → skip silently.
2. Read the daily file and classify each entry (permanent / medium-term / short-term).
3. Extract permanent entries to typed files + MEMORY.md index.
4. Determine the ISO week file (`YYYY-WNN`). Create with the appropriate header if it doesn't exist.
5. Append a dated section (`## YYYY-MM-DD`) with permanent + medium-term entries as concise bullets.
6. Delete yesterday's daily file.

**Monday rollup** (after archiving Sunday's daily, for each path):
1. Read the previous week's weekly file. Extract top 5–10 highlights.
2. Append to the highlights file under `## Week YYYY-WNN (Mon DD – Sun DD)` with one-line bullets.
3. Delete the previous week's weekly file.

## Step 13: Group daily memory
- Daily: `/workspace/group/memory/daily/YYYY-MM-DD.md`
- Weekly: `/workspace/group/memory/weekly/YYYY-WNN.md` — header: `# Weekly Summary — YYYY-WNN`
- Highlights: `/workspace/trusted/highlights.md`

## Step 14: Trusted daily memory
- Daily: `/workspace/trusted/memory/daily/YYYY-MM-DD.md` (entries prefixed with `[source]`)
- Weekly: `/workspace/trusted/memory/weekly/YYYY-WNN.md` — header: `# Trusted Weekly Memory — YYYY-WNN`
- Highlights: `/workspace/trusted/highlights.md` — preserve source attribution `[chat-name]` in bullets.

## Step 15: Process daily_discoveries
1. Read `/workspace/trusted/memory/daily_discoveries.md`. If absent → skip silently.
2. Scan for entries without `✓ processed` marker.
3. For each unprocessed entry:
   - **Promote to: RUNBOOK.md** → append to appropriate section of `/workspace/trusted/RUNBOOK.md`. Mark `✓ processed`.
   - **Promote to: typed memory file + MEMORY.md index** → create/update typed file in `/workspace/trusted/`, add/update MEMORY.md index entry. Mark `✓ processed`.
   - **Promote to: unsure** → operational/workflow/tool fact → RUNBOOK.md; behavioral preference/feedback → typed memory file + MEMORY.md index. Mark `✓ processed`.
4. Write updated `daily_discoveries.md` back with all markers in place.
5. Silent on success; report only on file write failure.

## Step 16: Check watchlist
`Skill(skill: "tessl__check-watchlist")` — check if any tracked upcoming shows have been released.
- Show released → skill notifies Baruch and updates watchlist.json automatically
- Nothing released → stay silent

## Step 17: Mark as run
Read `/workspace/group/task-tz-state.json`. Find the entry in `follow_me_tasks` where `name == "nightly-housekeeping"`. Set `last_run_date` to today's local date (`YYYY-MM-DD` in `current_tz`). Write back, preserving all other fields.

(This is a confirmation write. Step 1 already wrote this value as an optimistic lock. If Step 1 failed, this step ensures the date is recorded.)

## Step 18: Backup to git
```
bash /workspace/group/scripts/backup-to-git.sh
```
Then call `mcp__nanoclaw__github_backup` with message `"nightly backup: YYYY-MM-DD"`.
Silent on success; report only on error.

## Step 19: Silence
If nothing to report, output nothing (wrap in `<internal>`).
