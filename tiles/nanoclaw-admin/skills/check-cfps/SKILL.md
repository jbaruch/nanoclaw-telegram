---
name: check-cfps
description: Finds open CFPs relevant to Baruch across Java/AI/developer conferences. Extends the tessl tile version with persistent CFP state (sent/dismissed/remind) from cfp-state.json. Use when Baruch asks about upcoming conferences, call for papers, speaking opportunities, CFP deadlines, or where to submit a talk proposal.
---

# Check CFPs (with State Management)

Fetches and filters open CFPs, applies AI-based relevance reasoning, and maintains persistent state across sessions.

## Step 1 — Run fetch-and-filter script

Execute the deterministic pipeline (fetches sources, applies hard filters, checks state).
Script does NOT filter by topic relevance — that's your job in Step 1b.

```bash
python3 /workspace/group/scripts/check-cfps-fetch.py
```

Parse the JSON output:
- `cfps` — filtered, sorted list of open CFPs with fields: `name`, `city`, `conf_date`, `cfp_url`, `deadline`, `days_left`, `slug`, `source`
- `warnings` — data source failures or skipped checks to surface in output
- `checked_at` — timestamp

**Alert if:** script fails to run (report error and abort).
**Note:** warnings about unreachable sources should be mentioned briefly at the top of output.

## Step 1b — Relevance filter (AI reasoning)

For each CFP in the script output, reason about whether it's relevant to Baruch:

**The core question for every conference:** "Could Baruch realistically submit a talk about Java/JVM/Kotlin/Spring, developer tools/DevRel, or AI-for-developers here, and would it land with the audience?"

Apply the full YES/NO criteria from `/workspace/group/RELEVANCE-CRITERIA.md`. Use reasoning — not keyword matching, not a default fallback. Arrive at a confident YES or NO.

**Reasoning for ambiguous AI conferences:** Ask yourself — is the speaker lineup typically ML engineers and data scientists (Python/PyTorch/TensorFlow), or software developers building on top of AI APIs? If it's the former → skip. If developers building AI-powered apps → keep.

**No fallback default.** Think it through and make a call. Both false positives (irrelevant confs Baruch has to dismiss) and false negatives (missing good confs) are bad — use judgment to avoid both.

## Step 1c — Sessionize deadline verification

For each CFP that survived Step 1b, attempt to verify/enrich via Sessionize using the slug:

```
mcp__nanoclaw__sessionize_get_event(slug: "{slug}")
```

If the call succeeds:
- `cfp_open: false` → remove from list (CFP closed per Sessionize)
- `is_online: true` → remove from list (online-only)
- Update `deadline` with `cfp_end_local[:10]` (authoritative deadline from Sessionize, more accurate than scraped sources)
- Note `expenses_covered` for new state entries (add to `bot_notes`)

If the call returns an error or 404 → skip silently (not all conferences are on Sessionize). **Never block on Sessionize failures** — continue with original data.

Run calls in parallel where possible (one per CFP with a slug). Do not call for slugs that are clearly not Sessionize event IDs.

## Step 2 — Web search for gaps

Run these searches to catch AI/developer conferences not in the primary sources:

1. `AI developer conference CFP open 2026 "call for speakers" deadline`
2. `developer conference CFP 2026 autumn fall open submissions`

Add new CFPs found that aren't already in the list (deduplicate by conference name).
Apply hard filters (no online/virtual, no excluded locations) then the same relevance reasoning as Step 1b.

## Step 3 — Sort and format

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

After Steps 1b and 2, **write every relevant CFP** (kept after relevance filter) to `cfp-state.json`. This is what feeds the morning brief CFP section — without it, CFPs never appear in the brief.

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
