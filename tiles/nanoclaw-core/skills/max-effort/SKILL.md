---
name: max-effort
description: "Adversarial self-check to prevent lazy shortcuts. Use when about to ask the user for information that could be looked up (e.g., user says 'don't ask me, look it up' or 'check before asking'), approximate instead of computing an exact answer, or give up on finding data. Forces exhaustive tool-based search — including web lookup, file search, API calls, and code execution — before escalating to the user."
---

# Max Effort — Adversarial Self-Check

Invoke this skill when you are about to:
- Ask the user for information you might be able to find yourself
- Approximate or guess instead of computing the exact answer
- Give up on finding data and say "I don't know"

## Process

All of this happens inside <internal> tags — never surfaces to the user unless escalation is decided.

1. **State the problem clearly:** What are you trying to compute, and what specific piece of data seems missing?

2. **Run the Max Effort self-check** — challenge every "I don't know" using this internal checklist:

   | Blocker | Try instead |
   |---|---|
   | "I don't know the location" | Search calendar events, contacts, or local files |
   | "I don't know the travel time" | Call a maps API or run a web search |
   | "I don't know the current state of X" | Run a web search or browse the relevant URL |
   | "I can't access that" | Try Composio, a different API endpoint, or execute code to derive it |
   | "The file wasn't shared" | Search the working directory: `find . -name "*.config"` or equivalent |
   | "I don't have the key/token" | Search for `.env`, `config.*`, or secrets files before asking |

3. **Attempt each suggestion.** Concrete invocation patterns:
   - **Web search:** call `web_search("exact query")` — don't approximate.
   - **File search:** use `find`, `grep -r`, or the available file-search tool on likely paths.
   - **Code execution:** write and run a snippet to compute the value rather than estimating it.
   - **API call:** invoke the relevant tool (e.g., maps, calendar, Composio) with specific parameters.

4. **Iterate:** If an approach yields the data, use it. If it fails, work through the next option on the checklist before considering escalation.

5. **Stop only when one of these is true:**
   - You have the data and can compute the correct answer, OR
   - ALL three escalation conditions are met:
     1. Every tool option on the checklist has been exhausted
     2. The information is truly not computable from available data
     3. At least 3 different approaches have been tried and failed

6. **If escalating:** Ask the user ONE specific question — the exact minimum information needed. Not a general "I need more info."

## Example Self-Check in Action

> **Problem:** Need travel time to a meeting.
> **Attempt 1:** `maps_api(origin="home address", destination="meeting address")` → returned 45 min.
> **Result:** Used the 45 min figure directly — no need to ask the user.

> **Problem:** Need the contents of a config file the user hasn't shared.
> **Attempt 1:** `find . -name "*.config" -o -name "*.cfg"` → not found.
> **Attempt 2:** `web_search("default config values for [tool]")` → found standard defaults.
> **Result:** Used defaults with a note, no escalation needed.

> **Problem:** Need a private API key for an internal service.
> **Attempt 1:** `find . -name ".env" -o -name "secrets.*"` → not found.
> **Attempt 2:** `web_search("[service] public endpoint or open API")` → not available.
> **Attempt 3:** Composio integration check for the service → not connected.
> **Result:** All 3 escalation conditions met → ask: "Could you provide the API key for [service]?"
