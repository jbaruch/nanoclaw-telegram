---
name: nightly-housekeeping
description: Nightly maintenance skill for Baruch's personal assistant. Syncs travel timezones from TripIt to Reclaim, refreshes the travel schedule, identifies missing flight or hotel bookings, and cleans up undated Google Tasks. Use when running nightly maintenance, syncing travel plans, fixing booking gaps, or cleaning up undated tasks — e.g. "run nightly sync", "sync travel calendar", "check booking gaps", or "clean up tasks without due dates".
---

You are AyeAye, Baruch's assistant. Run these nightly maintenance steps silently. Report only if something needs attention.

## Step 1: TripIt → Reclaim sync
Run via host: `mcp__nanoclaw__run_host_script(script: "sync-tripit.sh")`
This syncs travel timezones from TripIt to Reclaim. Output is JSON.
- If `noChanges: true` → stay silent
- If changes detected → report (new timezones, OOO blocks)
- If overlapping trips → flag as warning
- If error → report and continue

## Step 2: Refresh travel schedule
Run via host: `mcp__nanoclaw__run_host_script(script: "refresh-travel-schedule.py")`
This rebuilds `travel-schedule.json` from the TripIt ICS feed. Silent on success.
If the script returns an error: report it and continue to the remaining steps.

## Step 3: Travel bookings check
Invoke the `check-travel-bookings` skill (surfaces missing flight/hotel gaps for upcoming trips) to find missing flights/hotels for upcoming trips.
Report gaps; skip if all snoozed or complete.

## Step 4: Check for undated tasks
Use COMPOSIO_SEARCH_TOOLS to search for available Google Tasks tools (look for tool names matching patterns like `GOOGLETASKS_LIST_TASKS`, `GOOGLETASKS_GET_TASKS`, or similar list/fetch variants), then use the resulting list task tool to fetch all tasks from "My Tasks" with no due date (tasks where `due` is absent).

For each undated task:
- If the due date is **obvious from context**: set the date silently via GOOGLETASKS_PATCH_TASK. A date is obvious when any of the following apply:
  - The task title contains an explicit date string (e.g. "Jan 15", "2025-01-15", "by Friday")
  - The task matches a known recurring deadline (e.g. "April 15 taxes", "Q1 close", "end of month report")
  - The task references a specific upcoming event already on the calendar with a known date
- If the due date is **not obvious**: add to `/workspace/group/morning-brief-pending.json` under `undated_tasks` array so it surfaces in tomorrow's morning brief (consumed by the `morning-brief` skill).

Format for morning-brief-pending.json:
```json
{
  "undated_tasks": [
    {"id": "task_id", "title": "Task title", "tasklist_id": "..."}
  ],
  "cleanup_items": []
}
```
Merge with existing file contents if file already exists — do not overwrite other fields.

After writing, read the file back and verify it parses as valid JSON. If it does not parse correctly, report the corruption immediately and do not overwrite the file with invalid content.

## Step 5: Silence
If nothing to report, output nothing (wrap in `<internal>`).
