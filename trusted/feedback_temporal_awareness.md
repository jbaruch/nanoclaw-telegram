# Feedback: Temporal Awareness

Always check the current date before making statements about events being upcoming, past, or "ironic".

## Rule
Before saying something like "he was listed as a speaker at upcoming X conference" or "ironic that he was going to speak at Y" — verify whether that event has already happened.

Today is always available from the system context or MEMORY.md. Use it.

## Why
LLMs are bad at temporal reasoning. Compensate explicitly:
- When referencing conferences, product launches, or scheduled events — check if today's date is before or after them
- "JavaOne 2026" in April 2026 — could already be over. Check.
- Never assume an event is upcoming without verifying the date

## Applied to
- Research summaries about fired/laid off speakers
- Conference CFP and schedule mentions
- Any time-sensitive industry events
