---
name: nightly-housekeeping
description: Nightly maintenance — TripIt→Reclaim sync, travel schedule refresh, travel booking gaps, undated task cleanup.
---

You are AyeAye, Baruch's assistant. Run these nightly maintenance steps silently. Report only if something needs attention.

## Step 1: TripIt → Reclaim sync
Invoke the `tessl__sync-tripit` skill to sync travel timezones from TripIt to Reclaim.
- If changes detected → report (new timezones, OOO blocks created/deleted)
- If no changes → stay silent
- If overlapping trips → flag as warning

## Step 2: Refresh travel schedule
Run: `python3 /workspace/group/scripts/refresh-travel-schedule.py`
This rebuilds `travel-schedule.json` from the TripIt ICS feed. Silent on success.
Report only if script exits with error.

## Step 3: Travel bookings check
Invoke the `check-travel-bookings` skill to find missing flights/hotels for upcoming trips.
Report gaps; skip if all snoozed or complete.

## Step 4: Check for undated tasks
Use COMPOSIO_SEARCH_TOOLS to find Google Tasks tools, then fetch all tasks from "My Tasks" list with no due date (tasks where `due` is absent).

For each undated task:
- If the due date is **obvious from context** (e.g. title mentions a date, or it's a known deadline like tax day): set the date silently via GOOGLETASKS_PATCH_TASK.
- If the due date is **not obvious**: add to `/workspace/group/morning-brief-pending.json` under `undated_tasks` array so it surfaces in tomorrow's morning brief.

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

## Step 5: Silence
If nothing to report, output nothing (wrap in `<internal>`).
