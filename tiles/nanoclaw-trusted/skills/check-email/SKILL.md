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
  ],
  "muted_subject_patterns": [
    {"pattern": "[jc] sharat", "reason": "...", "muted_at": "..."}
  ]
}
```

Build:
- A **set of muted threadIds** from `muted_threads[].threadId`
- A **list of muted subject patterns** (lowercased strings) from `muted_subject_patterns[].pattern`

During Classify, skip any email where its `threadId` is in the muted set **OR** its subject (lowercased) contains any muted pattern. Skip = no report, no cleanup queue, total silence.

**Which mute type to use:**
- **threadId** — one-off threads from regular senders
- **subject pattern** — mailing list threads (groups.io, listserv, etc.) where each reply gets a new threadId; use the subject prefix (e.g. `[jc] sharat`)

**To mute:** When Baruch says "не интересно", "mute this", "stop reporting this", or any dismissal — determine type, append to the right array in `email-mute-state.json` **immediately, before anything else**. Any session (main, heartbeat, background). No waiting, no asking. Confirm with: "✓ muted".

- *From main session:* Use `GMAIL_FETCH_EMAILS` to find the thread by subject/sender. Check for mailing list origin → subject pattern; otherwise → threadId.
- *Repeated dismissals = bug.* If the same thread is dismissed twice, the mute wasn't saved or wrong type was used. Check if a threadId mute is failing due to mailing list behavior and convert it to a subject pattern.

## Load Preferences

Read `/workspace/group/email-preferences.json` if it exists. Schema:
- `always_important` — array of `{sender, reason}` or `{domain, reason}` entries; skip calibration, always flag
- `always_ignore` — array of `{sender, reason}` or `{domain, reason}` entries; skip calibration, always drop
- `patterns` — array of `{rule, action, reason}` entries applied during classification (e.g. `"subject contains 'CFP'"` → `"important"`)

`always_important` and `always_ignore` override all calibration and classification rules. `patterns` act as additional classification rules.

If the file doesn't exist or is empty, proceed with default rules only.

## Fetch

Discover Gmail tool per `composio-preamble` rule, then fetch recent emails:
- max_results: 20
- label_ids: ["INBOX"]
- Do NOT include spam/trash

Read `/workspace/group/nanoclaw-state.json` to get `last_email_checked` (a messageId string). Only process emails NEWER than that ID (higher messageId = newer in Gmail). If no state file or field is missing, process the latest 5 only.

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
2. Derive a general rule from the error class — not just the specific email
3. Add the entry and briefly confirm what you learned

| Feedback | Rule to add |
|---|---|
| "bad fit" on a LinkedIn notification | Add `linkedin.com` to `always_ignore` |
| "bad fit" on a GitHub PR assignment | Pattern: ignore GitHub notifications unless user is @mentioned in body |
| "good fit" on a CFP email | Pattern: subject contains CFP → important |
| "you missed one from my boss" | Add boss's email to `always_important`; add domain too if unrecognized |

Only record "good fit" feedback if it reinforces a non-obvious pattern — skip if default rules already covered it.
