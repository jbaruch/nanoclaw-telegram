---
name: check-unanswered
description: Triage unanswered messages detected by the external heartbeat. Expired requests get a brief acknowledgement, actionable ones get processed immediately, unclear ones get reported. Triggers on "check unanswered", "missed messages", "unreplied messages".
---

# Check Unanswered Messages

Detection of unanswered messages is handled by the external heartbeat script on the host (it has direct access to messages.db). This skill handles triage when the heartbeat reports unanswered messages.

## When invoked by heartbeat

The heartbeat orchestrator calls this skill when `/check-unanswered` detects issues. Since the agent cannot access messages.db directly, check the conversation context and recent `send_message` history to identify what was missed.

## Triage

For each unanswered message, evaluate the content and decide:

### 1. Expired / no longer actionable

The request was time-sensitive and the window has passed (e.g., "what's my next meeting?" and the meeting already ended; "remind me in 5 minutes" from 20 min ago).

**Action:** Send a brief note acknowledging you missed it and what the answer would have been. After sending, verify the acknowledgement was delivered via `mcp__nanoclaw__send_message`.

### 2. Still actionable

The task has no hard deadline or the deadline hasn't passed (research, write an email, find something, general questions).

**Action:** Just go do it — process the message as if it just arrived and send the response normally. Do NOT report it as an issue.

### 3. Unclear

Can't determine if expired or not.

**Action:** Report it — include sender, chat, message content, and note the ambiguity.

## Output

Return the list of triage actions taken, or empty if nothing to do.
