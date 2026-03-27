---
name: check-cfps
description: Find conference CFPs closing within the next 7 days for AI and Java topics. Uses developers.events JSON API and javaconferences.org. Filters for relevance, excludes Africa. Use as part of morning-brief or standalone. Triggers on "check cfps", "closing cfps", "conference deadlines", "upcoming cfp deadlines".
---

# Check CFPs

Find conference Call for Papers closing within 7 days that are relevant to Baruch (AI, agents, agentic coding, Java).

## Step 1: Fetch from developers.events

```bash
curl -sf https://developers.events/all-cfps.json -o /tmp/all-cfps.json
```

Validate that the file is non-empty and a valid JSON array; skip this source and continue with Step 2 if either check fails.

Parse and filter using `jq`:

```bash
NOW=$(date +%s%3N)
PLUS7=$(( NOW + 7 * 86400000 ))

jq --argjson now "$NOW" --argjson plus7 "$PLUS7" '
  [.[] |
    select(
      .untilDate >= $now and
      .untilDate <= $plus7 and
      (.conf.status // "open") == "open" and
      # exclude Africa
      (.conf.location // "" | test("Nigeria|Kenya|South Africa|Ghana|Ethiopia|Tanzania|Uganda|Rwanda"; "i") | not)
    ) |
    {
      name: .conf.name,
      location: .conf.location,
      conf_date: .conf.date,
      deadline_ms: .untilDate,
      cfp_link: .link,
      conf_url: .conf.hyperlink
    }
  ] | sort_by(.deadline_ms)
' /tmp/all-cfps.json > /tmp/cfps-filtered.json
```

Validate that `/tmp/cfps-filtered.json` is a valid JSON array before proceeding; log an error and skip if not.

Then apply relevance filtering (see Relevance Rules below) to the entries in `/tmp/cfps-filtered.json`.

## Step 2: Fetch from javaconferences.org

Use `WebFetch` to get `https://javaconferences.org/`. The page renders an HTML table with columns: **Conference**, **Location**, **Date**, **CFP**, **CFP Deadline**.

Parse with these selectors:
- Table rows: `table tbody tr`
- CFP link: `td:nth-child(4) a[href]`
- CFP deadline text: `td:nth-child(5)` (format varies: `YYYY-MM-DD` or `Month DD, YYYY`)

Keep only rows where:
1. The CFP link cell is non-empty
2. The parsed CFP deadline falls within the next 7 days

**Deduplicate:** If a conference appears in both sources (fuzzy name match, e.g. lowercase + strip punctuation), keep the entry with the more specific deadline date.

## Relevance Rules

Keyword lists and category definitions are maintained in `/workspace/group/cfp-relevance-config.md` — update that file to change keywords without modifying this skill. Apply the rules defined there to decide whether each conference is relevant, included, or skipped.

## Step 3: Format output

For each relevant CFP, include:
- Conference name
- Location
- Conference date(s)
- CFP deadline (how many days left)
- CFP submission link

Sort by deadline (closest first).

**Output format (Telegram):**

```
*📢 CFPs closing this week:*

• *Devoxx Belgium* — Antwerp, Oct 6-10
  CFP closes in 2 days (Mar 29)
  Submit: https://sessionize.com/devoxx-be

• *AI Dev Summit* — San Francisco, Jun 15-16
  CFP closes in 5 days (Apr 1)
  Submit: https://sessionize.com/ai-dev-summit
```

If no relevant CFPs found: return nothing (empty output, no message).

## Step 4: Save state

Write the list of found CFPs to `/workspace/group/cfp-state.json`:

```json
{
  "checked_at": "ISO timestamp",
  "cfps": [
    {
      "name": "Devoxx Belgium",
      "deadline": "2026-03-29",
      "link": "https://sessionize.com/devoxx-be",
      "notified": true
    }
  ]
}
```

On subsequent runs, don't re-notify about CFPs already in the state file with `notified: true`. Only notify about new CFPs or CFPs whose deadline moved closer (< 2 days left and not yet reminded at that threshold).
