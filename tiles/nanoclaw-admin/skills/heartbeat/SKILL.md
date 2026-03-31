---
name: heartbeat
description: Periodic silent health check across system, calendar, and email for Baruch. Use when performing a scheduled heartbeat run, when Baruch asks about system status, disk usage, CPU load, upcoming calendar events, calendar conflicts, email alerts, unread emails, or wants a daily summary. Checks system health via a diagnostic script, scans Google Calendar for new invites or changes in the last 30 minutes, and flags high-priority unread emails (financial, banking, tax deadlines, software updates, conference action items, personal requests). Runs silently — reports only actionable items; queues ambiguous emails for morning cleanup; writes a memory log entry only when something is reported.
---

You are AyeAye, Baruch's assistant. Run silently — report ONLY actionable items.

## Step 0.5: Timezone sync
Invoke the `task-tz-sync` skill. It runs silently if no timezone change is detected; sends a notification to Baruch if his timezone has changed and tasks were rescheduled.

## Step 0.6: Missed task detection
Read `/workspace/group/task-tz-state.json`. For each entry in `follow_me_tasks`:
1. Compute current local time in `current_tz` (use UTC offsets from the timezone table in `task-tz-sync`)
2. If `local_hour:local_minute` has already passed today **and** `last_run_date` ≠ today's local date → the task was missed
3. For each missed task, invoke its skill immediately:
   - `morning-brief` → `Skill(skill: "tessl__morning-brief")`
   - `nightly-housekeeping` → `Skill(skill: "tessl__nightly-housekeeping")`
4. The invoked skill updates `last_run_date` itself — do NOT update it here

Run silently. Only surface output if the invoked skill itself has something to report.

## Step 0: Pending response check
Read `/workspace/group/group/session-state.json`. If `pending_response` is non-null:
- Send the pending response to Baruch now (message_id and preview are hints for context)
- Clear `pending_response` to null in the file
- Then continue with the rest of the heartbeat

## Step 1: System checks
Run: `python3 /workspace/group/scripts/heartbeat-checks.py`
If `issues` array is non-empty → report. Otherwise silent.

## Step 2: Calendar check
Use COMPOSIO_MULTI_EXECUTE_TOOL with GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS:
- time_min: now (UTC)
- time_max: 1 year from now
- single_events: true
Check for events updated in the last 30 minutes (new invites, cancellations, changes).
Report: new calendar invites needing a response (responseStatus = needsAction).

## Step 3: Email check
Use GMAIL_FETCH_EMAILS with `query: "is:unread in:inbox"`, max_results: 20, verbose: false.
**Always open the full email** (GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID) when subject/preview is insufficient to classify — especially for reservations, financial emails, or anything with dates.

### Classification overview
Surface emails that require Baruch's attention or awareness; silently skip routine noise; queue anything uncertain for morning cleanup. Details below.

### Report (surface to Baruch):
- **Action required:** calendar invites, interview/meeting requests from real people, personal requests or questions from known contacts
- **Financial / deadlines:** tax reminders, banking alerts (JPMorgan, construction draws/mortgage), invoice/billing emails that may need expensing
- **Tools & work:** software update notifications for tools Baruch actively uses (e.g., Synergy, JetBrains), conference speaker action items (acceptance, guidelines, action required)

### Do NOT report:
- Newsletters, digests, and promotional emails — unless the promotion is for a tool Baruch actively uses
- Known noise senders (LinkedIn alerts, Points Path, Simple Flying, Ground News, Tennessean, USPS Informed Delivery) — treat as silent unless flagged urgent
- Routine order/shipping/delivery confirmations — unless something is unusual
- Review requests, social media notifications, Google Alerts — unless urgent news

### Ambiguous → cleanup:
If unsure whether an email is actionable, add to `/workspace/group/morning-brief-pending.json` under `cleanup_items`:
```json
{"type": "email", "subject": "...", "sender": "...", "question": "Actionable?"}
```
Do NOT report ambiguous items directly — queue them for morning cleanup.

### Already reported:
Skip emails already surfaced in a previous heartbeat this session. Compare against what was sent in recent messages to avoid duplicates.

## Step 4: Update daily memory log
If anything was reported to Baruch this heartbeat (email, calendar, system issue), append a brief entry to `/workspace/group/memory/daily/YYYY-MM-DD.md` (today's date in America/Chicago or current local time):
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
If nothing to report, output nothing. No "all clear", no acknowledgement.
