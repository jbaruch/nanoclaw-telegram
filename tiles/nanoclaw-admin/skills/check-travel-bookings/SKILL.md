---
name: check-travel-bookings
description: Checks upcoming trips for missing bookings (flights, hotels, accommodation) using the TripIt ICS feed directly. Reports gaps for all upcoming trips — no date limit. Supports snooze state. Silent when all bookings are complete or snoozed. Use when the user asks about upcoming travel plans, itinerary completeness, missing reservations, or TripIt trip status.
---

# Check Travel Bookings

**Run the script on the host and interpret its JSON output. Do not implement the detection logic yourself.**

## How to run

```
mcp__nanoclaw__run_host_script(script: "check-travel-bookings.py")
```

The script runs on the host with TripIt credentials (not available in the container).

The script outputs JSON:
```json
{
  "gaps": [
    {"trip": "JNation 2026", "start": "2026-05-24", "end": "2026-06-01", "issue": "рейсы есть, отеля нет", "slug": "jnation-2026-05"}
  ],
  "checked_at": "2026-03-28T23:00:00Z",
  "total_trips": 10,
  "complete_trips": 8
}
```

If `gaps` is empty — stay silent. If gaps are present, format and send as Telegram message.

## Error handling

- If `run_host_script` returns an error — report it to the user and do not attempt to parse the result.
- If the script output is not valid JSON or is missing expected fields — report a parse error and show the raw output for diagnosis.

## Output format (when gaps found)

```
*Travel bookings to sort out:*

• [Trip Name] ([date range]) — [issue]
```

Date range: `May 24–Jun 1` (abbreviated month, no year unless spans years).

## State Management

When Baruch snoozes or resolves a trip, update `/workspace/group/travel-booking-state.json`:
- Snooze: set `snooze_until` to a future date for the trip's slug
- Resolved: remove the entry (ICS will show it complete on next run)

Slug format used by the script: `{normalized-summary}-{YYYY}-{MM}` (lowercase, spaces/punctuation → hyphens).

After writing any update, verify the file contains valid JSON (e.g. by re-reading and parsing it) before confirming success.
