---
name: check-travel-bookings
description: Checks upcoming trips for missing bookings (flights, hotels) using the TripIt ICS feed directly. Reports gaps for all upcoming trips — no date limit. Supports snooze state. Silent when all bookings are complete or snoozed.
---

# Check Travel Bookings

**Run the script at `/workspace/group/scripts/check-travel-bookings.py` and interpret its JSON output. Do not implement the detection logic yourself.**

## How to run

```bash
TRIPIT_ICAL_URL="$(cat /workspace/group/tripit-url.txt)" python3 /workspace/group/scripts/check-travel-bookings.py
```

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


