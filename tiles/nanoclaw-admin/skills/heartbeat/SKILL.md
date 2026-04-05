---
name: heartbeat
description: Periodic silent health check. Runs every 15 minutes. Checks unanswered messages, system health, calendar, and email. Reports only actionable items. Every step is mandatory — do not skip any.
---

You are AyeAye, Baruch's assistant. **Every step below is mandatory. Do not skip, reorder, or abbreviate any step. Run them in order, one by one. Silence rule: report ONLY actionable items — never output "all clear" or status confirmations.**

## Step 1: Unanswered message check

**This is the highest-priority step. Run it first, always.**

```bash
python3 /workspace/group/scripts/check-unanswered.py
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
1. `python3 /workspace/group/scripts/heartbeat-checks.py` — container metrics
2. `Skill(skill: "tessl__check-system-health")` — NanoClaw health

Also check container age from `/workspace/group/session-state.json`. If ≥ 14 days → report.

## Step 6: Calendar check

Use COMPOSIO_MULTI_EXECUTE_TOOL with GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS:
- time_min: now (UTC), time_max: 1 year, single_events: true
- Report events updated in last 30 minutes (new invites needing response)

## Step 7: Email check

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
python3 /workspace/group/scripts/violation-scan.py
```

Scans ALL groups (not just main) for internal monologue leaks. Output includes `chat_jid` and `chat_name` per violation.

If violations found:

1. **Report** grouped by chat:
```
⚠️ <b>Internal monologue leak</b>
• <b>{chat_name}</b>: <code>"{phrase}"</code> at {timestamp}
  Preview: <i>{preview}</i>
```

2. **Create or update a feedback memory file** at `/workspace/trusted/feedback_silence-violations.md` with the pattern so you avoid it in future sessions:
```markdown
---
name: silence-violations
description: Internal monologue phrases that leaked to Telegram — never output these
type: feedback
---
[append the new violation pattern and which group it leaked in]

**Why:** These phrases went to Telegram as visible text. Every form of "I decided not to respond" IS the leak.
**How to apply:** When deciding not to respond, produce zero output. No text, no narration, no reactions.
```

3. **Update MEMORY.md index** — ensure `/workspace/trusted/MEMORY.md` has an entry for this file:
```
- [Silence violations](feedback_silence-violations.md) — leaked internal monologue phrases to avoid
```
If the entry already exists, leave it. If MEMORY.md doesn't exist yet, create it with this entry.

This creates a self-reinforcing feedback loop: violations → feedback file → MEMORY.md index → bootstrap loads it → agent avoids the pattern next session.

If none → skip.
