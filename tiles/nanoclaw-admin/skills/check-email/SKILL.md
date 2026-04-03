---
name: check-email
description: Fetch recent emails, calibrate for source noise (automated senders disguised as personal), classify importance, and report only genuinely important messages. Use as part of heartbeat or standalone. Triggers on "check email", "important emails", "new emails", "email triage".
---

# Check Email

**Workflow:** Load Muted Threads → Load Preferences → Fetch → Calibrate → Classify → Output

## Load Muted Threads

Read `/workspace/group/email-mute-state.json` if it exists:

```json
{
  "muted_threads": [
    {"threadId": "...", "subject": "...", "reason": "dismissed", "muted_at": "..."}
  ]
}
```

Build a set of muted threadIds. During Classify, skip any email whose `threadId` is in this set — no report, no cleanup queue, total silence.

**To mute a new thread:** When Baruch says "не интересно", "mute this", "stop reporting this", or has been dismissing a topic repeatedly — fetch the email's threadId and append it to this file immediately.

## Load Preferences

Read `/workspace/group/email-preferences.json` if it exists. This file contains user feedback on past classifications:

```json
{
  "always_important": [
    {"sender": "sarah@jfrog.com", "reason": "direct colleague"},
    {"domain": "devreluni.com", "reason": "conference organizer"}
  ],
  "always_ignore": [
    {"sender": "noreply@github.com", "reason": "automated notifications"},
    {"domain": "marketing.salesforce.com", "reason": "disguised marketing"}
  ],
  "patterns": [
    {"rule": "subject contains 'CFP'", "action": "important", "reason": "conference submissions"},
    {"rule": "sender ends with @linkedin.com", "action": "ignore", "reason": "LinkedIn noise"}
  ]
}
```

These override all calibration and classification rules:
- `always_important` — skip calibration, always flag
- `always_ignore` — skip calibration, always drop
- `patterns` — applied as additional rules during classification

If the file doesn't exist or is empty, proceed with default rules only.

## Fetch

Discover Gmail tool per `composio-preamble` rule, then fetch recent emails:
- max_results: 20
- label_ids: ["INBOX"]
- Do NOT include spam/trash

Read `/workspace/group/nanoclaw-state.json` to get `last_email_checked` (a messageId string). Only process emails NEWER than that ID (higher messageId = newer in Gmail). If no state file exists or the field is missing, process the latest 5 only.

After processing, update `last_email_checked` in `/workspace/group/nanoclaw-state.json` with the newest messageId seen.

### Error Handling

| Failure | Action |
|---|---|
| `GMAIL_FETCH_EMAILS` fails | Log error, skip run, do not update state. Report: "Email check skipped: fetch failed." |
| No emails returned | Treat as clean inbox; do not update state. Return nothing. |
| State file corrupted (invalid JSON) | Reset to empty state, process latest 5; overwrite file after successful fetch. |

## Source Calibration

| Signal | Action |
|---|---|
| "Action required" / "Urgent" in subject | Down-rank |
| Display name looks personal but address is noreply/system | Ignore |
| Unsubscribe link in body | Down-rank |
| HTML-heavy body (images, buttons, "View in browser") | Down-rank |
| GitHub/GitLab/Jira notifications | Down-rank unless time-sensitive |
| Calendar invites / RSVPs | Skip unless human added a personal note |
| "Via" or "on behalf of" in sender | Down-rank |
| CC list > 5 recipients and you're not in To | Down-rank |

## Classification

Classify as important only if ALL of the following hold (signals already down-ranked in calibration above should not pass):
- Sender is a real person — not noreply@, alerts@, @*.sendgrid.net, @*.mailchimp.com, or other automated bulk senders
- No unsubscribe link in body
- Subject reflects genuine human communication — Re:/Fwd: from a human, contains "?", or uses action words like "review", "approve", "can you", "please" — not generic urgency bait
- Sender domain matches known work contacts or is a previously unseen personal sender
- Email is not labelled CATEGORY_PROMOTIONS or CATEGORY_UPDATES

## Output

For each important email:
- Sender name + email
- Subject
- First 100 chars of body
- Confidence note if borderline (e.g., "might be automated — has unsubscribe link but sender looks personal")

If no important emails: return nothing.

Alert format:
```
New email from [Name]: "[Subject]" -- [preview...]
```

**Example output:**
```
New email from [Sarah Lee]: "Can you review the PR before EOD?" -- Hey, could you take a look at PR #482? We're trying to merge bef...
```

## Learning from Feedback

When the user reacts with feedback ("good fit", "bad fit", "this one was spam", "you missed one from X"):

1. Read `/workspace/group/email-preferences.json` (create if missing)
2. Derive a general rule from the error class, not just the specific email
3. Add the entry and confirm briefly what you learned

| Feedback | Rule to add |
|---|---|
| "bad fit" on a LinkedIn notification | Add `linkedin.com` to `always_ignore` |
| "bad fit" on a GitHub PR assignment | Pattern: ignore GitHub notifications unless user is @mentioned in body |
| "good fit" on a CFP email | Pattern: subject contains CFP → important |
| "you missed one from my boss" | Add boss's email to `always_important`; add domain too if unrecognized |

Only record "good fit" feedback if it reinforces a non-obvious pattern — if default rules already covered it, don't clutter the file.
