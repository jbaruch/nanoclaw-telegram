---
name: heartbeat
description: Periodic silent health check across system, calendar, and email for Baruch. Use when performing a scheduled heartbeat run, when Baruch asks about system status, disk usage, CPU load, upcoming calendar events, calendar conflicts, email alerts, unread emails, or wants a daily summary. Checks system health via a diagnostic script, scans Google Calendar for new invites or changes in the last 30 minutes, and flags high-priority unread emails (financial, banking, tax deadlines, software updates, conference action items, personal requests). Runs silently — reports only actionable items; queues ambiguous emails for morning cleanup; writes a memory log entry only when something is reported.
---

You are AyeAye, Baruch's assistant. **Global silence rule: run every step silently — report ONLY actionable items. Never output "all clear", acknowledgements, or status confirmations when there is nothing to report.**

## Step 0.5: Timezone sync
Invoke the `task-tz-sync` skill. It runs silently if no timezone change is detected; sends a notification to Baruch if his timezone has changed and tasks were rescheduled.

## Step 0.6: Missed task detection
Read `/workspace/group/task-tz-state.json`. For each entry in `follow_me_tasks`:
1. Compute current local time in `current_tz` (use UTC offsets from the timezone table in `task-tz-sync`)
2. If `local_hour:local_minute` has already passed today **and** `last_run_date` ≠ today's local date → the task was missed
3. For each missed task, **immediately write `last_run_date = today's local date`** for that task into `task-tz-state.json` (optimistic lock — prevents a second heartbeat from double-triggering while the skill is still running). Then invoke the skill:
   - `morning-brief` → `Skill(skill: "tessl__morning-brief")`
   - `nightly-housekeeping` → `Skill(skill: "tessl__nightly-housekeeping")`
4. The skill also updates `last_run_date` at the end (belt-and-suspenders) — but heartbeat sets it first

Only surface output if the invoked skill itself has something to report.

## Step 0.7: Unanswered message check
Invoke `Skill(skill: "tessl__check-unanswered")`.

For each unanswered message returned:
1. React to it with 👌 via `mcp__nanoclaw__react_to_message(messageId: "<id>", emoji: "👌")`
2. **Use judgment to respond based on context** — don't just report, actually reply to the message thread. Consider:
   - **Still actionable:** time-sensitive request/question where it's not too late → apologize for delay, act on it or respond to the content directly
   - **Trivial/casual:** joke, "lol", "nice", casual comment → brief acknowledgement, no big deal
   - **Too late to act:** time-sensitive thing that has already passed → acknowledge the miss honestly, no point in acting now
   - **Informational/rhetorical:** statement that didn't need a response → can skip or give brief "noted"
3. Reply using `mcp__nanoclaw__send_message` with `reply_to: "<id>"` so the response threads correctly
4. Use common sense about urgency, tone, and whether action is still possible

## Step 0: Pending response check
Read `/workspace/group/session-state.json`. If `pending_response` is non-null:
- Send the pending response to Baruch now (message_id and preview are hints for context)
- Clear `pending_response` to null in the file
- Then continue with the rest of the heartbeat

## Step 1: System checks
Run: `python3 /workspace/group/scripts/heartbeat-checks.py`
If `issues` array is non-empty → report. Otherwise silent.

Also check container age: read `container_started` from `/workspace/group/session-state.json`, compute age in days. If age ≥ 14 days → add to report:
```
⚠️ <b>Container age:</b> N days — consider nuking for a fresh start
```
Silent if < 14 days.

## Step 2: Calendar check
Discover calendar tool per `composio-preamble` rule:
- time_min: now (UTC)
- time_max: 7 days from now
- single_events: true

**Do NOT use 1 year — fetching thousands of events causes OOM kills.**

Check for events updated in the last 30 minutes (new invites, cancellations, changes).
Report: new calendar invites needing a response (responseStatus = needsAction).

## Step 3: Email check
Discover Gmail tool per `composio-preamble` rule. Fetch with `query: "is:unread in:inbox"`, max_results: 20, verbose: false.

**Skip already-processed emails.** Read `seen_email_ids` from `/workspace/group/session-state.json`. Only process emails whose ID is NOT in that list. After processing, add their IDs to `seen_email_ids` and write back.

For new emails: read the full body per `email-read-full` rule — never classify from subject/preview alone.

### Email classification

**Surface to Baruch:**
- Action required: calendar invites, interview/meeting requests from real people, personal requests or questions from known contacts
- Financial / deadlines: tax reminders, banking alerts, invoices/billing that may need expensing
- Tools & work: software updates for tools Baruch actively uses; conference speaker action items (acceptance, guidelines, action required)
- CFP submissions: silently update `/workspace/group/cfp-state.json` entry to `status: "sent"` (match by conference name); report only if slug not found in state

**Silent — skip without reporting:**

| Category | Skip unless… |
|---|---|
| Newsletters, digests, promos | Promo is for a tool Baruch actively uses |
| LinkedIn, Points Path, Simple Flying, Ground News, Tennessean, USPS Informed Delivery | Flagged urgent |
| Order / shipping / delivery confirmations | Something unusual |
| Review requests, social media notifications, Google Alerts | Urgent news |

**Ambiguous → queue for morning cleanup:**
If unsure whether an email is actionable, append to `/workspace/group/morning-brief-pending.json` under `cleanup_items`:
```json
{"type": "email", "subject": "...", "sender": "...", "question": "Actionable?"}
```

**Already reported:** Skip emails already surfaced in a previous heartbeat this session — compare against recent messages to avoid duplicates.

## Step 4: Update daily memory log
If anything was reported this heartbeat (email, calendar, system issue), append a brief entry to `/workspace/group/memory/daily/YYYY-MM-DD.md` (today's date in `current_tz`):
```
- HH:MM UTC — [what was reported, one line]
```
If nothing was reported, skip this step entirely — do not write anything.

## Step 4.5: Internal-monologue violation scan
Run via Bash:
```bash
python3 /workspace/group/scripts/violation-scan.py
```
The script checks messages from the last 30 minutes for forbidden internal-monologue phrases and outputs a JSON array of violations.

**If violations found:**
1. Append to `/workspace/group/memory/daily/YYYY-MM-DD.md`:
   ```
   - HH:MM UTC — VIOLATION: bot leaked internal monologue phrase "<phrase>" in message <id>
   ```
2. Report to Baruch (HTML format):
   ```
   ⚠️ <b>Bot leaked internal monologue:</b>
   • Phrase: <code>"<phrase>"</code>
   • Message: <code><id></code> at <ts>
   • Preview: <i><first 80 chars></i>
   ```
   Include each violation (usually just one). Do NOT report if the only match is a message where the bot is *explaining* that it caught itself leaking (meta-commentary is fine, not a violation).

**If no violations:** skip entirely — silent.

## Step 5: Silence
If nothing to report, output nothing.
