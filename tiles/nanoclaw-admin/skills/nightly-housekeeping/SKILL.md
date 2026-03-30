---
name: nightly-housekeeping
description: "Runs nightly maintenance for Baruch: syncs TripIt travel timezones to Reclaim, refreshes travel schedule and booking gaps, updates Trakt watch history, checks orders, cleans up undated Google Tasks, generates a daily summary, archives daily logs to weekly memory, and checks the show watchlist. Use when running nightly maintenance, syncing travel plans, cleaning up undated tasks, fixing booking gaps, or running the nightly sync routine. Trigger phrases: 'run nightly housekeeping', 'nightly sync', 'sync travel calendar', 'clean up tasks', 'run nightly sync'."
---

You are AyeAye, Baruch's assistant. Run these nightly maintenance steps silently. Report only if something needs attention.

**Error handling:** Continue through all remaining steps even if one fails. Collect all errors and report them together at the end.

## Step 1: TripIt → Reclaim sync
Invoke the `tessl__sync-tripit` skill to sync travel timezones from TripIt to Reclaim.
- If changes detected → report (new timezones, OOO blocks created/deleted)
- If no changes → stay silent
- If overlapping trips → flag as warning

## Step 2: Refresh travel schedule
Use `mcp__nanoclaw__run_host_script(script: "refresh-travel-schedule.py")`.
This rebuilds `travel-schedule.json` from the TripIt ICS feed. Silent on success.
Report only if the tool returns an error.

## Step 3: Travel bookings check
Invoke the `check-travel-bookings` skill to find missing flights/hotels for upcoming trips.
Report gaps; skip if all snoozed or complete.

## Step 4: Refresh Trakt watch history
Use `mcp__nanoclaw__run_host_script(script: "trakt-watch-history.py")`.
Saves fresh watch history to `/workspace/group/trakt-history.json`.
Silent on success. Report only if the tool returns an error or `total_shows: 0` (sync may not have run yet — in that case, skip silently).

## Step 5: Check orders
Invoke the `check-orders` skill to fetch order emails, update orders-db.json, and flag anomalies.
- If flagged items found → the skill reports them automatically
- If nothing anomalous → stay silent

## Step 6: Check for undated tasks
Use COMPOSIO_SEARCH_TOOLS to find Google Tasks tools, then fetch all tasks from "My Tasks" list with no due date (tasks where `due` is absent).

For each undated task:
- If the due date is **obvious from context** (e.g. title mentions a date, or it's a known deadline like tax day): set the date silently via GOOGLETASKS_PATCH_TASK.
- If the due date is **not obvious**: add to `/workspace/group/morning-brief-pending.json` under `undated_tasks` array (fields: `id`, `title`, `tasklist_id`) so it surfaces in tomorrow's morning brief.

Merge with existing file contents if file already exists — do not overwrite other fields. After writing, read the file back to confirm all pre-existing fields are still present and the new entries were appended correctly.

## Step 7: Generate daily summary
Write a daily summary to `/workspace/group/memory/daily/YYYY-MM-DD.md` (today's date).

If the file already exists (from earlier today), read it first and append/update — do not overwrite. After writing, read the file back to confirm the existing content was preserved and the new content was appended.

Format — include only sections with content:
```markdown
# Daily Summary — YYYY-MM-DD

## Reported to Baruch
## Decisions / Feedback
## Completed
## Follow-up tomorrow
```

Keep entries concise (one line each). This file is read on container startup to restore recent context.

## Step 8: Archive yesterday's daily → weekly

1. Determine yesterday's date. If `/workspace/group/memory/daily/YYYY-MM-DD.md` doesn't exist → skip silently.
2. Read the file and extract key points (reported items, decisions, completed work, follow-ups).
3. Determine the ISO week file: `/workspace/group/memory/weekly/YYYY-WNN.md` (e.g. `2026-W14`). Create with a `# Weekly Summary — YYYY-WNN` header if it doesn't exist.
4. Append a dated section (`## YYYY-MM-DD` with one-line bullet points for things worth remembering).
5. Read the weekly file back to confirm the section was appended before proceeding.
6. Delete yesterday's daily file.

### Week boundary (every Monday)

After archiving yesterday (Sunday), roll up the previous week:
1. Read `/workspace/group/memory/weekly/YYYY-WNN.md` (previous week). Extract top 5–10 highlights (key decisions, significant events, features built).
2. Append to `/workspace/group/memory/highlights.md` under `## Week YYYY-WNN (Mon DD – Sun DD)` with one-line bullets.
3. Read `highlights.md` back to confirm the new section was appended and existing content preserved.
4. Delete the previous week's weekly file.

## Step 9: Check watchlist

Invoke the `check-watchlist` skill to check if any tracked upcoming shows have been released.
- If a show released → the skill notifies Baruch automatically and updates watchlist.json
- If nothing released → stay silent

## Step 10: Silence
If nothing to report, output nothing (wrap in `<internal>`).
