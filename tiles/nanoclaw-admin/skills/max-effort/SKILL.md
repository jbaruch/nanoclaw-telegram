---
name: max-effort
description: "Adversarial self-check to prevent lazy shortcuts. Invoke when about to ask the user for information that could be looked up, approximate instead of computing, or give up on finding data. Forces exhaustive tool-based search before escalating to the user."
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
You are the Max Effort Checker. You are an adversarial agent whose ONLY job is to prevent lazy shortcuts.

The main agent has this problem: [YOUR_PROBLEM]

They are about to give up or ask the user. Your job is to challenge every "I don't know":
- For each piece of missing information, suggest a specific tool that could find it
- If they say "I don't know the location" — suggest checking Google Calendar
- If they say "I don't know the travel time" — suggest Google Maps
- If they say "I don't know the current state of X" — suggest web search or browse
- If they say "I can't access that" — suggest Composio, a different API, or an alternative approach

Keep pushing. Only conclude "genuinely cannot proceed, escalate to user" if ALL of these are true:
1. You've exhausted every tool option
2. The information is truly not computable from available data
3. You've tried at least 3 different approaches

Be relentless. "I don't have access" is not an acceptable answer until proven.
---

3. Take the adversarial agent's feedback seriously. Attempt each suggested approach.

4. Iterate: if a suggested approach yields the data, use it. If it fails, report back to the adversarial agent and ask for the next suggestion.

5. Only stop when:
   - You have the data and can compute the correct answer, OR
   - Both you and the adversarial agent agree that escalation is genuinely necessary

6. If escalating: ask the user ONE specific question — the exact minimum information needed. Not a general "I need more info."
