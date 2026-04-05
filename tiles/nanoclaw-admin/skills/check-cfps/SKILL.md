---
name: check-cfps
description: Finds open CFPs relevant to Baruch across Java/AI/developer conferences. Extends the tessl tile version with persistent CFP state (sent/dismissed/remind) from cfp-state.json. Use when Baruch asks about upcoming conferences, call for papers, speaking opportunities, CFP deadlines, or where to submit a talk proposal.
---

**Every step below is mandatory. Execute them in order. Do not skip, reorder, or abbreviate any step.**

# Check CFPs (with State Management)

Fetches and filters open CFPs, applies AI-based relevance reasoning, and maintains persistent state across sessions.

## Step 1 — Sessionize speaker API candidates

Call the Sessionize speaker API to collect open CFPs from conferences that have already invited Baruch or match his speaker profile:

```
mcp__nanoclaw__sessionize_open_cfps(filter: {isOnline: false, isUserGroup: false})
```

For each event returned:
1. Extract the slug from `cfpLink` — it's the last path segment of `https://sessionize.com/{slug}`
2. Check if the slug already exists in `/workspace/group/cfp-state.json`
   - If it exists (any status) → skip (already known)
   - If it doesn't exist → add to the candidate pool for this run with fields: `name`, `city` (from `location`), `conf_date` (from `eventDates`), `deadline` (from `cfpDates.endUtc[:10]`), `cfp_url` (cfpLink), `slug`, `source: "sessionize-speaker-api"`

**Do not write anything to state here.** Do not surface to Baruch. These candidates flow into Steps 2–6 exactly like any other source.

If 0 new candidates: continue to Step 2 with an empty Sessionize pool.

## Step 2 — Run fetch-and-filter script

Execute the deterministic pipeline (fetches sources, applies hard filters, checks state).
Script does NOT filter by topic relevance — that's your job in Step 5.

```bash
python3 /home/node/.claude/skills/tessl__check-cfps/scripts/check-cfps-fetch.py
```

Parse the JSON output:
- `cfps` — filtered, sorted list of open CFPs with fields: `name`, `city`, `conf_date`, `cfp_url`, `deadline`, `days_left`, `slug`, `source`
- `warnings` — data source failures or skipped checks to surface in output
- `checked_at` — timestamp

**Merge** the Sessionize candidates from Step 1 into this list (deduplicate by slug).

**Alert if:** script fails to run (report error and abort).
**Note:** warnings about unreachable sources should be mentioned briefly at the top of output.

## Step 3 — Web search for gaps

Run these searches to catch AI/developer conferences not in the primary sources:

1. `AI developer conference CFP open 2026 "call for speakers" deadline`
2. `developer conference CFP 2026 autumn fall open submissions`

Add new CFPs found that aren't already in the list (deduplicate by conference name).
Apply hard filters (no online/virtual, no excluded locations). Do NOT apply relevance filtering yet — that's Step 5.

## Step 4 — Enrich with Sessionize event details

Fetch Sessionize details for every CFP that has a slug. Names can be deceiving — the AI needs the full description and metadata to make good calls.

For each CFP with a slug, call Sessionize:

```
mcp__nanoclaw__sessionize_get_event(slug: "{slug}")
```

Run calls in parallel where possible. Do not call for slugs that are clearly not Sessionize event IDs.

If the call succeeds, enrich the CFP entry:
- `cfp_open: false` → remove from list immediately (CFP closed)
- `is_online: true` → remove from list immediately (online-only)
- Update `deadline` with `cfp_end_local[:10]` (authoritative deadline, more accurate than scraped sources)
- Attach `description` and `expenses_covered` to the entry
- Note `expenses_covered` for state entries (add to `bot_notes`)

If the call returns an error or 404 → skip silently (not all conferences are on Sessionize). **Never block on Sessionize failures** — continue with original data.

## Step 5 — Relevance filter (AI reasoning)

All data is gathered. Now judge. For each CFP you have the conference name, city, Sessionize description (when available), and web search context — use ALL of it.

**The core question for every conference:** "Could Baruch realistically submit a talk about Java/JVM/Kotlin/Spring, developer tools/DevRel, or AI-for-developers here, and would it land with the audience?"

Apply the full YES/NO criteria from `/workspace/group/RELEVANCE-CRITERIA.md`. Use reasoning — not keyword matching, not a default fallback. Arrive at a confident YES or NO.

**When Sessionize description is available:** Use it as ground truth. A conference called "BC TechDays" could be British Columbia (developer conf) or Business Central/Dynamics 365 (ERP). If the description mentions Dynamics 365, ERP, business processes, NAV, or supply chain → NO, regardless of how the name sounds. See `/workspace/group/RELEVANCE-CRITERIA.md` for NO categories.

**When no description is available and the name is ambiguous:** Do a targeted web search: `"[Conference Name]" topics speakers audience 2026` before making the call.

**Reasoning for ambiguous AI conferences:** Is the speaker lineup typically ML engineers and data scientists (Python/PyTorch/TensorFlow), or software developers building on top of AI APIs? Former → skip. Latter → keep.

**Sessionize-sourced candidates:** These are already pre-filtered to Baruch's speaker profile. Still apply relevance reasoning, but lean YES when the conference topic is ambiguous.

**No fallback default.** Think it through and make a call. Both false positives (irrelevant confs Baruch has to dismiss) and false negatives (missing good confs) are bad — use judgment to avoid both.

## Step 6 — Sort and format

The script already returns results sorted by deadline. Merge in web search additions (also sorted).

Group into urgency tiers:
- 🔴 **≤3 days** — act today
- 🟡 **4–7 days**
- 🟢 **8–31 days**
- ⬜ **>31 days**

Format each entry as:
```
• *[Conference Name]* — [City, Country], [Conference Date]
  CFP closes [deadline] ([N days])
  Submit: [URL]
```

If no open CFPs after filtering: return nothing (wrap in `<internal>`).

## Output

Return the formatted, grouped list. Include a brief note at the top if any data sources were unavailable or conflict filtering was skipped.

## State Management

The script handles state filtering automatically. When Baruch gives feedback about a conference, update `/workspace/group/cfp-state.json` directly.

**Slug format:** `{conference-name-slug}-{year}` — lowercase, spaces/punctuation → hyphens, strip leading/trailing hyphens.

### Writing discovered CFPs to state

After Step 5, **write every relevant CFP** (kept after relevance filter) to `cfp-state.json`. This is what feeds the morning brief CFP section — without it, CFPs never appear in the brief.

Rules:
- If the slug already has a user action (`dismissed`/`sent`/`remind`) → preserve it, do NOT overwrite
- If the slug doesn't exist yet → write a new `"open"` entry with full data including a `bot_notes` field explaining why you included it

### Calibration notes

When Baruch dismisses a CFP, record his reason in `baruch_notes`. When a deadline expires while `status` is still `"open"`, ask if he submitted — record the outcome in `baruch_notes` to calibrate future filtering. These notes are never shown to Baruch unless he asks.

### User feedback actions

| User input | Action |
|-----------|--------|
| "отправил на [конф]" / "submitted to [conf]" | `status: sent`, update `updated` to today |
| "не интересно [конф]" / "skip [conf]" | `status: dismissed` |
| "напомни за [N] дней до дедлайна [конф]" | `status: remind`, `remind_before_days: N` |
| "напомни о [конф] через неделю" | `status: remind`, `remind_before_days: 7` |
| "покажи снова [конф]" | remove entry from cfp-state.json |

### State format

```json
{
  "all-things-open-2026": {
    "status": "open",
    "name": "All Things Open 2026",
    "city": "Raleigh, NC, USA",
    "conf_date": "Oct 18–20",
    "deadline": "2026-03-31",
    "cfp_url": "https://allthingsopen.org/call-for-papers",
    "updated": "2026-03-31",
    "bot_notes": "General open-source dev conf with broad audience; typically has Java/JVM content"
  },
  "voxxed-lu-2026": {
    "status": "sent",
    "name": "VoxxedDays Luxembourg 2026",
    "city": "Luxembourg",
    "conf_date": "Jun 20",
    "deadline": "2026-04-15",
    "cfp_url": "https://...",
    "updated": "2026-03-28"
  }
}
```

### Slug examples

| Conference name | Slug |
|----------------|------|
| VoxxedDays Luxembourg 2026 | `voxxed-days-luxembourg-2026` |
| Devoxx Belgium 2026 | `devoxx-belgium-2026` |
