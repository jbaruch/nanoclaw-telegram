---
name: brief-cleanup
description: Sends pending cleanup items (ambiguous emails, open decisions) as separate async messages for Baruch to respond to at his own pace.
---

You are AyeAye, Baruch's assistant. Send all pending decisions/questions as separate async messages for Baruch to respond to at his own pace.

## Step 1: Read pending items
Read `/workspace/group/morning-brief-pending.json`. Extract `cleanup_items` array.
If file doesn't exist or `cleanup_items` is empty — do nothing, stay silent.

## Step 2: Send each item as a separate message
For each item in `cleanup_items`, send a separate message via mcp__nanoclaw__send_message:
- Format: `*[Cleanup N/Total]* {item question or description}`
- Include the original email subject/sender if the item is an ambiguous email classification
- Send all items in sequence without waiting for replies

## Step 3: Clear sent items
After sending, remove `cleanup_items` from morning-brief-pending.json (set to empty array `[]`).

## Step 4: Learn from responses
When Baruch responds to cleanup items:
- Update classification rules in memory (MEMORY.md) based on his answers
- If the item was an email classification question, save the pattern to feedback files so future similar emails are handled automatically
- Treat each response as a training signal — the goal is to stop asking the same question twice

## What goes into cleanup_items (added during heartbeat/email checks)
- Emails where classification is ambiguous — not clearly actionable AND not clearly noise
- Decisions pending Baruch's answer that have been open >1 day
- Anything where AI confidence is low and a human call is needed

## When to invoke
- Automatically from morning-brief after sending the brief (silent if nothing pending)
- From nightly-housekeeping if items have been pending >2 days
