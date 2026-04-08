---
name: heartbeat
description: Automated background monitoring skill that runs every 15 minutes as a scheduled health check. Checks for unanswered Telegram messages and reacts/replies to each, processes any pending responses from session state, syncs timezones, detects missed scheduled tasks (morning-brief, nightly-housekeeping), verifies container metrics and NanoClaw system health, reviews recent calendar updates requiring a response, filters and classifies unread inbox emails by priority, logs actionable findings to a daily memory file, and scans all groups for internal monologue leaks. Reports only actionable items — never outputs "all clear" confirmations. Use when performing scheduled background monitoring, when the user requests a system status sweep, or when running the periodic assistant heartbeat cycle. This is a background monitoring skill, not for composing emails or managing calendar events.
---

You are AyeAye, Baruch's assistant. **Every step below is mandatory. Do not skip, reorder, or abbreviate any step. Run them in order, one by one. Silence rule: report ONLY actionable items — never output "all clear" or status confirmations.**

## Step 1: Unanswered message check

```bash
python3 /home/node/.claude/skills/tessl__check-unanswered/scripts/check-unanswered.py
```

Parse the JSON output. If `unanswered` array is empty → move to Step 2.

For each unanswered message:
1. React with 👌: `mcp__nanoclaw__react_to_message(messageId: "<id>", emoji: "👌")`
2. Reply with judgment — actually respond to the message content:

| Situation | Action |
|---|---|
| Still actionable | Apologize for delay, act on it or answer directly |
| Trivial / casual | Brief acknowledgement |
| Too late to act | Acknowledge the miss honestly |
| Informational / rhetorical | Skip or brief "noted" |

3. Thread correctly: `mcp__nanoclaw__send_message(reply_to: "<id>")`

## Step 2: Pending response check

Read `/workspace/group/session-state.json`. If `pending_response` is non-null:
- Send the pending response now
- Clear `pending_response` to null in the file

## Step 3: Timezone sync

`Skill(skill: "tessl__task-tz-sync")`

## Step 4: Missed task detection

Read `/workspace/group/task-tz-state.json`. For each entry in `follow_me_tasks`:
1. Compute current local time in `current_tz`
2. If `local_hour:local_minute` has passed today AND `last_run_date` ≠ today → task was missed
3. Write `last_run_date = today` immediately (optimistic lock), then invoke:
   - `morning-brief` → `Skill(skill: "tessl__morning-brief")`
   - `nightly-housekeeping` → `Skill(skill: "tessl__nightly-housekeeping")`

## Step 5: System checks

Run in parallel:
1. `python3 /home/node/.claude/skills/tessl__heartbeat/scripts/heartbeat-checks.py` — container metrics
2. `Skill(skill: "tessl__check-system-health")` — NanoClaw health

Also check container age from `/workspace/group/session-state.json`. If ≥ 14 days → report.

## Step 6: Calendar check

Use COMPOSIO_MULTI_EXECUTE_TOOL with GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS:
- time_min: now (UTC), time_max: 1 year, single_events: true
- Report events updated in last 30 minutes (new invites needing response)

## Step 7: Email check

**Mute filter (MUST run first):** Read `/workspace/group/email-mute-state.json` if it exists. Skip any email whose `threadId` is in `muted_threads[].threadId` **or** whose subject (lowercased) matches any pattern in `muted_subject_patterns[].pattern`. Muted = total silence — no report, no queue.

Use GMAIL_FETCH_EMAILS with `query: "is:unread in:inbox"`, max_results: 20, verbose: false.
**Always open the full email** when subject/preview is insufficient to classify.

**Surface to Baruch:**
- Action required: calendar invites, interview/meeting requests, personal requests from known contacts
- Financial / deadlines: tax reminders, banking alerts, invoices
- Tools & work: software updates for tools Baruch uses; conference speaker action items
- CFP submissions: silently update cfp-state.json; report only if slug not found

**Silent — skip:**

| Category | Skip unless… |
|---|---|
| Newsletters, digests, promos | Promo is for a tool Baruch actively uses |
| LinkedIn, Points Path, Simple Flying, Ground News, Tennessean, USPS | Flagged urgent |
| Order / shipping / delivery confirmations | Something unusual |
| Review requests, social media, Google Alerts | Urgent news |

**Ambiguous → queue for morning cleanup** in `/workspace/group/morning-brief-pending.json`.

**Already reported:** Skip emails surfaced in a previous heartbeat this session.

## Step 8: Daily memory log

If anything was reported, append one line to `/workspace/group/memory/daily/YYYY-MM-DD.md`. If nothing was reported, skip.

## Step 9: Violation scan

```bash
python3 /home/node/.claude/skills/tessl__heartbeat/scripts/violation-scan.py
```

Scans ALL groups (not just main) for internal monologue leaks. Output includes `chat_jid` and `chat_name` per violation.

If violations found:

1. **Report** grouped by chat, showing chat name, the leaked phrase, timestamp, and a preview snippet.

2. **Create or update** `/workspace/trusted/feedback_silence-violations.md` — a feedback memory file listing the leaked pattern, which group it appeared in, and the rule: when deciding not to respond, produce zero output.

3. **Update MEMORY.md index** — ensure `/workspace/trusted/MEMORY.md` has an entry pointing to `feedback_silence-violations.md` with a brief description. If the entry already exists, leave it. If MEMORY.md doesn't exist yet, create it with this entry.

If none → skip.
