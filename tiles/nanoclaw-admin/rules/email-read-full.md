# Email Full Body Rule

When making any decision based on email content — assigning a due date, classifying importance, determining action, recommending a response — **always read the complete email body**. Never base a decision on `messageText`, `preview`, `snippet`, or subject line alone.

## The rule

Any time you call `GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID` to inform a decision:

1. Use `format: "full"`
2. Decode the body from `payload.parts[]`: find `mimeType: "text/plain"`, base64url-decode `body.data`. If no plain part, use `text/html` and strip tags.
3. Read the full decoded text before drawing any conclusion.

`messageText` and `preview` fields are truncated summaries. They are acceptable for **display only** (e.g., showing a snippet to the user). They are **never** acceptable as the basis for a decision.

## Why

A truncated preview optimizes for speed at the cost of correctness. Re-doing work because of a wrong decision based on a preview is more expensive than reading the full body once. There is no valid reason to use a preview when making a decision.

## Applies to

- Task due date assignment (morning-brief Step 3a)
- Email importance classification (heartbeat, check-email)
- Any skill that reads email to take action or make a recommendation
