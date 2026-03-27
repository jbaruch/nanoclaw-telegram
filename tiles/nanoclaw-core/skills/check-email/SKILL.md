---
name: check-email
description: Fetch recent emails, calibrate for source noise (automated senders disguised as personal), classify importance, and report only genuinely important messages. Use as part of heartbeat or standalone. Triggers on "check email", "important emails", "new emails", "email triage".
---

# Check Email

**Workflow:** Fetch → Calibrate → Classify → Output

## Fetch

Use `COMPOSIO_SEARCH_TOOLS` to find `GMAIL_FETCH_EMAILS`, then fetch recent emails:
- max_results: 20
- label_ids: ["INBOX"]
- Do NOT include spam/trash

Read `/workspace/group/heartbeat-state.json` to get `last_email_checked` (a messageId string). Only process emails NEWER than that ID (higher messageId = newer in Gmail). If no state file exists or the field is missing, process the latest 5 only.

```python
import json, os

STATE_FILE = "/workspace/group/heartbeat-state.json"

try:
    with open(STATE_FILE) as f:
        state = json.load(f)
    last_id = state.get("last_email_checked")
except (FileNotFoundError, json.JSONDecodeError):
    state = {}
    last_id = None  # process latest 5 only
```

After processing, update `last_email_checked` with the newest messageId seen:

```python
state["last_email_checked"] = newest_message_id
with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=2)
```

### Error Handling

| Failure | Action |
|---|---|
| `GMAIL_FETCH_EMAILS` fails | Log error, skip run, do not update state. Report: "Email check skipped: fetch failed." |
| No emails returned | Treat as clean inbox; do not update state. Return nothing. |
| State file corrupted (invalid JSON) | Reset to empty state, process latest 5; overwrite file after successful fetch. |

## Source Calibration

Email inboxes are noisy by design. Before classifying importance, apply these signals:

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

After calibration, classify as important if ALL of these hold:
- Sender address is a real person (not noreply@, not alerts@, not @*.sendgrid.net, not @*.mailchimp.com, not automated bulk senders)
- No unsubscribe link in body
- Subject looks like genuine human communication (Re:, Fwd: from a human, contains "?", action words like "review", "approve", "can you", "please") — but NOT generic urgency bait
- Sender domain matches known work contacts (jfrog.com, colleagues, conference organizers, etc.) OR is a previously unseen personal sender
- Email is not in CATEGORY_PROMOTIONS or CATEGORY_UPDATES label

## Worked Example

| Field | Value |
|---|---|
| From | `Sarah Lee <sarah.lee@jfrog.com>` |
| Subject | `Can you review the PR before EOD?` |
| Body | `Hey, could you take a look at PR #482? We're trying to merge before the release tomorrow.` |
| Labels | `INBOX` |

Passes all criteria: real person at a known work domain, no unsubscribe link, plain-text body, genuine action request.

```
New email from [Sarah Lee]: "Can you review the PR before EOD?" -- Hey, could you take a look at PR #482? We're trying to merge bef...
```

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
