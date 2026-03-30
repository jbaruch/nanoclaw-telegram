---
name: recommend-shows
description: Analyzes Baruch's viewing history and explicit ratings across netflix-history.csv, imdb-ratings.csv, and trakt-history.json to identify preferred genres, classify completed and abandoned shows, and rank unwatched titles by predicted interest. Generates targeted TV show recommendations with quality thresholds, searches for new releases, and tracks upcoming shows in watchlist.json. Use when Baruch asks for show recommendations, "что посмотреть", "что смотреть", or similar requests for what to watch next.
---

# TV Show Recommendation Skill

## Data Sources

- `/workspace/group/netflix-history.csv` — Netflix viewing history (Title, Date). Episode-level, every play event.
- `/workspace/group/imdb-ratings.csv` — IMDB ratings (if available). Explicit ratings = strong signal.
- `/workspace/group/trakt-history.json` — Trakt watch history. **Potentially stale** — see Step 1a note.
- `/workspace/group/watchlist.json` — Upcoming tracked shows. Check before web research (Step 4a).

## Step 1: Load and parse viewing history

Netflix CSV format: `"Show Name: Season X: Episode Name", "date"`

Parse each row:
- Split on `: ` — first part is show name, second is season, third is episode
- Single-part titles = movies
- Group all plays by show name

Filter out kids' content: animated children's shows, preschool series, toy-brand cartoons, and similar family/kids programming.

### Step 1a: Source priority

**Trakt is the primary source** — it's live and synced across all platforms (Netflix, Apple TV+, Max, Prime, Disney+). Always prefer trakt-history.json.

The CSVs are static exports that go stale:
- `netflix-history.csv` — covers Netflix only, will grow stale over time
- `imdb-ratings.csv` — explicit ratings, useful for taste signals but not for watch status

Cross-reference logic:
1. **Trakt says watched** → treat as watched (primary signal)
2. **CSV says watched, Trakt missing** → treat as watched (Trakt sync may be incomplete / still catching up)
3. **Neither source has it** → recommend it

If Baruch reports "уже видел" for something recommended, it means Trakt sync hadn't finished yet — note it but don't downgrade Trakt's reliability long-term.

## Step 2: Classify shows

Derive classifications dynamically from the data files using these thresholds:

**Completed (strong positive signal):** 8+ play events across multiple episodes/seasons
- Apply this threshold to grouped plays from netflix-history.csv and trakt-history.json
- Completed shows are a strong positive signal for genre and style preferences

**Abandoned (negative signal):** 1-3 plays with no return after the first session
- Apply this threshold to grouped plays — short engagement with no continuation
- Notable abandonment patterns: fantasy/anime-style (dropped after 1ep), Korean romance, pure action without substance

**IMDB ratings** (imdb-ratings.csv, ~160 entries): explicit taste signal.
- Fields: Const, Your Rating (1-10), Title, Title Type, Year, Genres
- Use ratings as a direct preference signal — high-rated genres = confirmed loves, low-rated = genres that don't click
- Low-rated examples: Citadel (3), Silo (4), Cross (5) — sci-fi spectacle and Amazon action without substance

## Step 3: Taste profile

Derive the taste profile entirely from the data files. Use completion counts from Step 2 and IMDB ratings from Step 2 to identify:

- **Top genres** — genres appearing most in high-rated and high-completion shows
- **Avoided genres** — genres appearing in abandoned or low-rated shows
- **Style signals** — e.g. preference for grounded characters over spectacle, non-English originals, procedural craft, slow-burn tension

Do not hardcode assumptions — let the data speak. The classification thresholds in Step 2 are sufficient to reconstruct preferences from scratch each run.

**Patterns that typically emerge as negatives (validate against data):**
- Pure fantasy without grounded characters (abandoned after 1ep pattern)
- Pure romance
- Sci-fi spectacle without substance (low IMDB ratings pattern)
- Action-first shows with weak writing

## Step 4a: Check watchlist for tracked shows

Before doing web research, read `/workspace/group/watchlist.json`.
- For any tracked show where `notified: false`, do a quick search: has it been released?
- If released: include it as a top recommendation and note it was on the watchlist
- If not yet released: mention it briefly as "coming soon" at the end of recommendations

## Step 4: Check new releases (web search required)

Training cutoff is stale. Always search for new seasons/shows before recommending:
- Search: `best crime thriller drama series 2025 2026`
- Search: `new season [show he likes] 2025 2026`
- Cross-reference: is it already in trakt-history.json, netflix-history.csv, or imdb-ratings.csv? If yes in any, he's seen it.
- Flag shows that started but he hasn't watched yet as "new to you"

### Quality filter (mandatory)

Only recommend shows that meet ALL of these thresholds:
- **IMDB ≥ 7.5**
- **Rotten Tomatoes ≥ 75%**

If ratings are unavailable for a show, search for them before recommending. Do not recommend shows below threshold. If a show is borderline (e.g. IMDB 7.3), skip it — there are plenty of high-quality options. Mention ratings in the recommendation.

## Step 5: Generate recommendations

**Prioritize:**
1. New seasons of shows he completed (if not yet in history)
2. Shows similar to his top-completed (same genre/vibe/style)
3. International/non-English content (proven appetite)

**Avoid:**
- Pure fantasy without grounded characters
- Romance-first shows
- Shows he already abandoned
- Anime-style even if live action
- Shows below quality threshold (IMDB < 7.5 or RT < 75%)

**Targeted pitch format** — same as books: *why this, for Baruch specifically*:
```
*[Show Name]* ([Year], [Seasons]) — IMDB X.X | RT XX%
[1-2 sentence targeted pitch: connect to something he already loves]
[Where to watch] | [Status: ongoing/finished]
```

Max 3-5 recommendations. If he says "уже видел" → pivot immediately and note it as a Trakt sync gap.

## Step 6: Save announced/upcoming shows to watchlist

After generating recommendations, check if any shows discovered during research are:
- Announced but not yet released (or just announced for a new season)
- Match Baruch's taste profile strongly

For each such show, add it to `/workspace/group/watchlist.json` under the `tracking` array if not already present:
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

Read the existing watchlist.json first, merge, and write back. Do not add duplicates.

Reply in Russian if Baruch asked in Russian.
