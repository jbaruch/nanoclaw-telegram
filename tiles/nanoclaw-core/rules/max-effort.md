# Max Effort — Compute, Don't Ask or Approximate

When a task requires external data to complete correctly, use available tools to find it. Never ask for information you can look up. Never approximate when you can compute the exact answer.

## The rule

Before asking the user or approximating:

1. **Identify what data is missing** to compute the right answer
2. **Check what tools can provide it** — calendar, Google Maps, web search, email, etc.
3. **Retrieve it** and compute the exact answer

## The canonical example

"Remind me 15 minutes before I leave for Amir's pickup."

| Approach | Verdict |
|---|---|
| Ask "when do you leave?" | ❌ Wrong — you can compute it |
| Set it 15 min before the event start | ❌ Wrong — departure ≠ event start |
| Check calendar for destination → Maps for travel time → calculate real departure → set 15 min before | ✅ Correct |

## Applies to

- Any time-based reminder with a travel component
- Any task where the answer depends on data accessible via tools
- Any situation where "I don't know" has a computable answer

**The test:** Before asking the user or making an approximation — "Could I figure this out myself with the tools I have?" If yes, do it.
