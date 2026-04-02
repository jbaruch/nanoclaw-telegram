# Calendar Event Filter Rules

Apply these exclusions whenever processing calendar events — for display, reminders, or scheduling.

## Always skip

- **Travel events** — events with "Travel" in the title or category
- **All-day "Home" events** — placeholder events, not real appointments
- **Week-number events** — informational, not actionable
- **Declined events** — where the owner's attendee entry (`self=true`) has `responseStatus="declined"`

## Owner identification

The owner's calendar identity is stored in `/workspace/trusted/MEMORY.md` or `/workspace/trusted/key-people.md`. Look for the primary Google Calendar email. Use it for the `self=true` attendee check.

Do NOT hardcode email addresses in skills.

## Applying filters

Every skill that processes calendar events (morning-brief, check-calendar, heartbeat) must apply ALL of the above before displaying or scheduling reminders. If a filter is missing from a skill, it's a bug.
