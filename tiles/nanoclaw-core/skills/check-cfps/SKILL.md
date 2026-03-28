---
name: check-cfps
description: Find open CFPs relevant to Baruch (Java/AI/developer conferences) using two structured data sources + web search, filtered by travel conflicts and no online conferences
---

# Check CFPs

Fetches open CFPs from two authoritative sources plus web search. Filters out online conferences and travel conflicts.

## Step 1 — Fetch primary sources (in parallel)

**Source A:** `https://developers.events/all-cfps.json`
- Fetch the full JSON array
- Each entry has: `link` (CFP URL), `until` (deadline string), `untilDate` (ms timestamp), `conf.name`, `conf.date` (array of ms timestamps), `conf.hyperlink`, `conf.location`
- Keep only entries where `untilDate` > now (CFP still open)

**Source B:** `https://javaconferences.org/conferences.json`
- Fetch the JSON array (current + next year Java conferences)
- Each entry has: `name`, `link` (website), `locationName`, `hybrid`, `date`, `cfpLink`, `cfpEndDate`
- Keep only entries where `cfpLink` is non-empty and `cfpEndDate` > today

## Step 2 — Web search for gaps

Run these searches to catch AI/developer conferences not in the above sources:

1. `AI developer conference CFP open 2026 "call for speakers" deadline`
2. `developer conference CFP 2026 autumn fall open submissions`

Add any new CFPs found that aren't already in the combined list.

## Step 3 — Load travel schedule

Read `/workspace/group/travel-schedule.json`. Array of trips with `start` and `end` (YYYY-MM-DD).

A conference has a **travel conflict** if its dates overlap any trip: `conf_start <= trip_end AND conf_end >= trip_start`.

## Step 4 — Filter

Remove entries where:
- **Online/virtual**: location contains "online", "virtual", "remote", or no city listed
- **Travel conflict**: conference dates overlap a committed trip (per Step 3)
- **CFP closed**: deadline already passed today
- **Not relevant** (per `/workspace/group/cfp-relevance-config.md`): Web3/blockchain/crypto, pure .NET/PHP/Ruby/iOS-only, academic-only, meetups < 1 day, no CFP link
- **Location exclusions** (per cfp-relevance-config.md): Nigeria, Kenya, South Africa, Ghana, Ethiopia, Tanzania, Uganda, Rwanda

Deduplicate by conference name (case-insensitive).

## Step 5 — Sort and format

Sort by CFP deadline ascending (soonest first).

Group into urgency tiers based on days until CFP deadline:
- 🔴 **≤3 days** — act today
- 🟡 **4–7 days**
- 🟢 **8–31 days**
- ⬜ **>31 days**

Format each entry as:
```
• *[Conference Name]* — [City, Country], [Conference Date]
  CFP closes [deadline] ([N days])
  Submit: [URL]
```

If no open CFPs found: return nothing (output nothing / wrap in `<internal>`).

## Output

Return the formatted, grouped list to the caller.
