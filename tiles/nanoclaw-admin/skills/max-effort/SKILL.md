---
name: max-effort
description: "Adversarial self-check to prevent lazy shortcuts using exhaustive tool-based search (file search, web lookup, calendar/maps APIs, code execution, Composio) before escalating to the user. Use when about to ask the user for information that could be looked up, approximate instead of computing an exact answer, or give up on finding data. Also triggered by user instructions like 'don't ask me, look it up' or 'check before asking'."
---

# Max Effort — Adversarial Self-Check

Invoke this skill when you are about to:
- Ask the user for information you might be able to find yourself
- Approximate or guess instead of computing the exact answer
- Give up on finding data and say "I don't know"

## Process

All of this happens inside <internal> tags — never surfaces to the user unless escalation is decided.

1. State your current problem clearly: what are you trying to compute, and what specific piece of data seems missing?

2. Spawn an adversarial Agent with this prompt (substitute [YOUR_PROBLEM] with your actual situation):

---
You are the Max Effort Checker. For every blocker, challenge "I don't know" with the most specific tool that could resolve it (file search, web search/browse, Google Calendar, Google Maps, code execution, Composio, or an alternative API). See the Concrete Tool Call Reference in the main skill for exact invocation patterns.

Only conclude "genuinely cannot proceed, escalate to user" if ALL of these are true:
1. You've exhausted every tool option
2. The information is truly not computable from available data
3. You've tried at least 3 different approaches

Be relentless. "I don't have access" is not an acceptable answer until proven.

The main agent's problem: [YOUR_PROBLEM]
---

3. Take the adversarial agent's feedback seriously. Attempt each suggested approach.

4. Iterate: if a suggested approach yields the data, use it. If it fails, report back to the adversarial agent and ask for the next suggestion.

5. Only stop when:
   - You have the data and can compute the correct answer, OR
   - Both you and the adversarial agent agree that escalation is genuinely necessary

6. If escalating: ask the user ONE specific question — the exact minimum information needed. Not a general "I need more info."

## Concrete Tool Call Reference

Use these as starting points for actual invocations rather than naming tools abstractly:

**File search**
```
search_files(path=".", pattern="report.csv")
read_file(path="./src/report.csv")          # retrieve metadata and content
```

**Web search / browse**
```
web_search(query="current Python version site:python.org")
browse_url(url="https://example.com/api/status")
```

**Google Calendar via Composio**
```
composio_action(action="GOOGLECALENDAR_LIST_EVENTS", params={"calendarId": "primary", "timeMin": "2024-01-01T00:00:00Z"})
```

**Google Maps via Composio**
```
composio_action(action="GOOGLEMAPS_DISTANCE_MATRIX", params={"origins": ["123 Main St, City"], "destinations": ["456 Oak Ave, City"], "mode": "driving"})
```

**Code execution (exact computation)**
```python
# Instead of approximating, compute directly
import datetime
delta = datetime.datetime(2024, 6, 1) - datetime.datetime(2024, 1, 1)
print(delta.days)   # exact answer, no guessing
```

## Example: Self-Check in Action

> **Problem:** Need travel time between two meetings to check for conflicts.
> **Attempt 1:** `composio_action(GOOGLECALENDAR_LIST_EVENTS)` → retrieved meeting addresses.
> **Attempt 2:** `composio_action(GOOGLEMAPS_DISTANCE_MATRIX, origins=[addr1], destinations=[addr2])` → returned 42 min drive time.
> **Result:** Used the 42-minute figure to compute the conflict. No user escalation needed.

> **Problem:** Need a file's last-modified date for a report.
> **Attempt 1:** `search_files(pattern="report.csv")` → file found, metadata returned timestamp.
> **Result:** Used timestamp directly. No user escalation needed.
