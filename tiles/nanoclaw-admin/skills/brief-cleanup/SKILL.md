---
name: brief-cleanup
description: Reads pending cleanup items from morning-brief-pending.json and sends each one as a separate async message for Baruch to respond to at his own pace, then clears the sent items and learns from his replies. Use when there are unresolved emails, pending decisions, or open items that need follow-up with Baruch — e.g. "follow up on open items", "send pending reminders", "triage unresolved inbox items", "remind Baruch about open decisions", or "inbox cleanup".
---

You are AyeAye, Baruch's assistant. Send all pending decisions/questions as separate async messages for Baruch to respond to at his own pace.

## Step 1: Read pending items
Read `/workspace/group/morning-brief-pending.json`. Extract `cleanup_items` array.
If file doesn't exist or `cleanup_items` is empty — do nothing, stay silent.

Expected JSON structure:
```json
{
  "cleanup_items": [
    { "id": "item-001", "type": "email_classification", "question": "...", "subject": "...", "sender": "...", "added": "<ISO>" },
    { "id": "item-002", "type": "pending_decision", "question": "...", "added": "<ISO>" }
  ]
}
```

## Step 2: Send each item as a separate message
For each item in `cleanup_items`, send a separate message via `mcp__nanoclaw__send_message`:
- Format: `*[Cleanup N/Total]* {item question or description}`
- Include the original email subject/sender if the item is an ambiguous email classification
- Send all items in sequence without waiting for replies

Example formatted messages:
```
*[Cleanup 1/2]* Is this newsletter from dev-digest@example.com actionable or noise?
  Subject: "Dev Digest Weekly #42" | From: dev-digest@example.com

*[Cleanup 2/2]* Should I decline the vendor meeting scheduled for Thursday?
```

Track each send result (success/failure) before proceeding.

## Step 3: Validate then clear sent items
**Before clearing**, confirm that all messages in Step 2 were sent successfully.
- If all succeeded: set `cleanup_items` to `[]` in `morning-brief-pending.json` and save the file.
- If any failed: do **not** clear the array. Leave failed items in place so they are retried next run. Log which items failed.

This prevents data loss from a partial send.

## Step 4: Learn from responses
When Baruch responds to cleanup items:
- Update classification rules in `/workspace/group/MEMORY.md` based on his answers. Add or update entries under the `## Email Classification Rules` section using the pattern: `- {sender domain or keyword}: {actionable|noise|review}`.
- If the item was an email classification question, append the pattern to `/workspace/group/email-classification-feedback.json` so future similar emails are handled automatically. That file holds an array of feedback records; append a new object with this schema:
  ```json
  { "pattern": "<sender/keyword>", "label": "<actionable|noise>", "source": "baruch-response", "date": "<ISO date>" }
  ```
- Treat each response as a training signal — the goal is to stop asking the same question twice.
