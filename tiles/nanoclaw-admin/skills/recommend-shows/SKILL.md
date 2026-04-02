---
name: recommend-shows
description: Analyzes Baruch's viewing history and explicit ratings across netflix-history.csv, imdb-ratings.csv, and trakt-history.json to identify preferred genres, classify completed and abandoned shows, and rank unwatched titles by predicted interest. Generates targeted TV show recommendations with quality thresholds, searches for new releases, and tracks upcoming shows in watchlist.json. Use when Baruch asks for show recommendations, "что посмотреть", "что смотреть", or similar requests for what to watch next.
---

# TV Show Recommendation Skill

## Data Sources

- `/workspace/group/netflix-history.csv` — Netflix viewing history (Title, Date). Episode-level, every play event.
- `/workspace/group/imdb-ratings.csv` — IMDB ratings (if available). Explicit ratings = strong signal.
- `/workspace/group/trakt-history.json` — Trakt watch history. **Potentially stale** — see Step 1a.
- `/workspace/group/watchlist.json` — Upcoming tracked shows. Check before web research (Step 4a).

## Step 1: Load and parse viewing history

Netflix CSV format: `"Show Name: Season X: Episode Name", "date"`

Parse each row:
- Split on `: ` — first part is show name, second is season, third is episode
- Single-part titles = movies
- Group all plays by show name

Filter out kids' content: animated children's shows, preschool series, toy-brand cartoons, and similar family/kids programming.

### Step 1a: Source priority

**Trakt is the primary source** — live and synced across all platforms (Netflix, Apple TV+, Max, Prime, Disney+). The CSVs are static exports that go stale.

| Trakt says watched | CSV says watched | → Decision |
|---|---|---|
| ✓ | any | Watched (primary signal) |
| ✗ | ✓ | Watched (Trakt sync may be incomplete) |
| ✗ | ✗ | Candidate for recommendation |

If Baruch reports "уже видел" for something recommended, Trakt sync hadn't finished — note it and pivot immediately.

## Step 2: Classify shows

Derive classifications dynamically from data files:

**Completed (strong positive signal):** 8+ play events across multiple episodes/seasons
**Abandoned (negative signal):** 1–3 plays with no return after the first session

**IMDB ratings** (imdb-ratings.csv, ~160 entries): explicit taste signal.
- Fields: Const, Your Rating (1-10), Title, Title Type, Year, Genres
- High-rated genres = confirmed loves; low-rated = genres that don't click

## Step 3: Taste profile

Derive entirely from data files using Step 2 classifications and IMDB ratings:

- **Top genres** — most frequent in high-rated and high-completion shows
- **Avoided genres** — frequent in abandoned or low-rated shows
- **Style signals** — e.g. grounded characters, non-English originals, procedural craft, slow-burn tension

## Step 4a: Check watchlist for tracked shows

Before web research, read `/workspace/group/watchlist.json`.
- `notified: false` → search: has it been released?
  - Released → include as top recommendation, note it was on the watchlist
  - Not yet → mention briefly as "coming soon" at end

## Step 4: Check new releases (web search required)

Training cutoff is stale. Always search before recommending:
- `best crime thriller drama series 2025 2026`
- `new season [show he likes] 2025 2026`

Cross-reference against all three data files — if present in any, he's seen it. Flag shows that started but aren't in history as "new to you."

### Quality filter (mandatory)

Only recommend shows meeting **all** thresholds:
- **IMDB ≥ 7.5**
- **Rotten Tomatoes ≥ 75%**

Search for ratings before recommending if unavailable. Skip borderline shows (e.g. IMDB 7.3) — there are plenty of high-quality options. Always include ratings in the recommendation.

## Step 5: Generate recommendations

**Prioritize:**
1. New seasons of shows he completed (if not yet in history)
2. Shows similar to his top-completed (same genre/vibe/style)
3. International/non-English content (proven appetite)

**Avoid:**
- Pure fantasy without grounded characters
- Romance-first shows
- Previously abandoned shows
- Anime-style (even if live action)
- Shows below quality threshold

**Targeted pitch format:**
```
*[Show Name]* ([Year], [Seasons]) — IMDB X.X | RT XX%
[1-2 sentence targeted pitch: connect to something he already loves]
[Where to watch] | [Status: ongoing/finished]
```

Max 3–5 recommendations.

## Step 6: Save announced/upcoming shows to watchlist

After generating recommendations, for any shows discovered that are announced but not yet released (or a new season just announced) and match Baruch's taste profile, add to `/workspace/group/watchlist.json` under the `tracking` array if not already present:

```json
{
  "title": "Show Name",
  "platform": "Platform",
  "expected": "YYYY or YYYY-QN",
  "reason": "Why it matches Baruch's taste",
  "added": "YYYY-MM-DD",
  "notified": false
}
```

Read existing watchlist.json first, merge, and write back. Do not add duplicates.

Reply in Russian if Baruch asked in Russian.
