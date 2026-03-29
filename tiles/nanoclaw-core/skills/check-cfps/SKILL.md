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

| Decision | Criteria |
|----------|----------|
| **Keep** | Java, JVM, Kotlin, Spring; Devoxx/Voxxed/JBCNConf family; developer tools/DX, devrel; general developer conferences with known Java or AI tracks (QCon, KubeCon, FOSDEM, NDC, GOTO); applied AI for software developers — LLM/GenAI/agents in dev context, AI-assisted development, AI infrastructure for engineers |
| **Skip** | Pure Web3/blockchain/crypto/NFT; non-JVM single-language conferences (unless Java or AI track confirmed); pure functional programming conferences (Lambda World, etc.); platform engineering/DevOps/SRE/cloud infrastructure (DevOpsDays, Fast Flow); academic-only research; meetups (<1 day); data science/MLOps/analytics events (primary audience is data engineers, not software developers) |

Use your judgment — "AI for developers" is in; "data2day", "DataEngConf", "MLOps Summit" style events are out. When unsure about a borderline AI conference, include and let Baruch decide.

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

| User input | Action |
|-----------|--------|
| "отправил на [конф]" / "submitted to [conf]" | `status: sent`, update `updated` to today |
| "не интересно [конф]" / "skip [conf]" | `status: dismissed` |
| "напомни за [N] дней до дедлайна [конф]" | `status: remind`, `remind_before_days: N` |
| "напомни о [конф] через неделю" | `status: remind`, `remind_before_days: 7` |
| "покажи снова [конф]" | remove entry from cfp-state.json |

State format:
```json
{
  "voxxed-lu-2026": { "status": "sent", "updated": "2026-03-28" },
  "javazone-2026": { "status": "dismissed", "updated": "2026-03-29" },
  "devoxx-be-2026": { "status": "remind", "remind_before_days": 7, "updated": "2026-03-28" }
}
```

### Slug examples

| Conference name | Slug |
|----------------|------|
| VoxxedDays Luxembourg 2026 | `voxxed-days-luxembourg-2026` |
| Devoxx Belgium 2026 | `devoxx-belgium-2026` |
